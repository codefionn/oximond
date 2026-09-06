//! `StringByte`: the byte of a bound string a number names.
//!
//! A key is checked as an integer. A float key is truncated, converted back
//! and compared with itself, which only a whole number in the range of an
//! integer passes, and then one unsigned compare of the zero-based index
//! with the byte count stands for both bounds. That is eight instructions
//! where the float form (two range compares, a sum and a difference with
//! 2^52, the conversion) had twelve.
//!
//! Reads of one string come in groups: `string.byte(s, i, i + 3)`, the bytes
//! of a `string.sub` the trace never made, a byte read twice. Their keys are
//! one number plus small constants. The first read of a group checks that
//! number (the root) and keeps it as an integer; a later one adds its
//! constant to the integer and makes the one compare, or none when the
//! constants checked before it lie on both sides of its own. A byte read
//! twice is loaded once.
//!
//! Checking the root in place of the key is sound: a root that is whole and
//! inside the string plus a small constant is an exact sum, so the index is
//! the key's. It can leave where the key itself would have passed (a root
//! just below a whole number whose sum rounds up to one); the exit runs the
//! call in the interpreter, as every exit of a `StringByte` does.

#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;

/// The largest constant a key may be away from its root.
const FAR: i64 = 1 << 20;

/// A number the straight-line code has as an integer, and what the checks
/// made on it prove about the bytes of one string.
#[derive(Clone, Copy)]
pub(super) struct WholeFact {
    /// The value index of the number (the root).
    root: u32,
    /// It as an integer.
    whole: Val,
    /// The binding of the string, or `NONE` when no read was checked yet.
    binding: u32,
    /// `root + low` and `root + high` are positions inside the string, and
    /// so is every `root + c` between them.
    low: i64,
    high: i64,
}

/// A byte the straight-line code loaded: the binding, the root, the constant
/// added to it, and the value the read made.
pub(super) type ByteFact = (u32, u32, i64, Val);

impl Lowerer<'_, '_> {
    /// `key` as a root plus a constant: the sums with a small whole constant
    /// that made it are followed back.
    fn root_of(&self, key: ValueId) -> (ValueId, i64) {
        let mut root = key;
        let mut added = 0_i64;
        for _ in 0..4 {
            let Some(TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left,
                right,
            }) = defining_op(self.ir, root)
            else {
                break;
            };
            let constant = |value: ValueId| match self.s.konst[value.index()] {
                #[allow(clippy::cast_possible_truncation, reason = "checked to be small")]
                Some(TraceScalar::Float(number))
                    if number.fract() == 0.0 && number.abs() <= FAR as f64 =>
                {
                    Some(number as i64)
                }
                Some(TraceScalar::Integer(number)) if number.abs() <= FAR => Some(number),
                _ => None,
            };
            let (next, by) = match (constant(*left), constant(*right)) {
                (None, Some(by)) => (*left, by),
                (Some(by), None) => (*right, by),
                _ => break,
            };
            if self.ty(next) != self.ty(root) || (added + by).abs() > FAR {
                break;
            }
            root = next;
            added += by;
        }
        (root, added)
    }

    /// `StringByte` of the string with the words `words`, bound as
    /// `binding`: the byte `key` names, as a number of type `ty`. Leaves
    /// through `guard` unless the key is a whole number inside the string
    /// (see the module comment for a key that is a sum).
    pub(super) fn string_byte(
        &mut self,
        binding: u32,
        words: Words,
        key: ValueId,
        guard: GuardId,
        ty: TraceType,
    ) -> Res<Val> {
        let key_ty = self.ty(key);
        if self.s.konst[key.index()].is_some()
            || !matches!(key_ty, TraceType::Float | TraceType::Integer)
        {
            let off = self.index_within(words, key, guard, 0)?;
            let byte = self.load8(words.base, off, 0);
            return Ok(self.byte_as(byte, ty));
        }
        let (root, added) = self.root_of(key);
        let root_index = root.index() as u32;
        // A side may be attached to the guard: then the check is made, so
        // that the side has its way in.
        let attached = self.members[self.member]
            .attached
            .iter()
            .any(|(attached, _)| *attached == guard.index());
        if !attached
            && let Some(&(.., byte)) = self
                .s
                .bytes
                .iter()
                .find(|(held, of, at, _)| *held == binding && *of == root_index && *at == added)
        {
            return Ok(byte);
        }
        let known = self
            .s
            .wholes
            .iter()
            .position(|fact| fact.root == root_index);
        let at = if let Some(at) = known {
            at
        } else {
            let number = self.get(root)?;
            let whole = if key_ty == TraceType::Float {
                // Whole and in the range of an integer: the conversion of
                // anything else (a fraction, a NaN, a number past 2^63)
                // does not convert back to it.
                let whole = self.un(Op::F2I, Ty::I64, number);
                let back = self.un(Op::I2F, Ty::F64, whole);
                self.guard_cc(Cc::Eq, back, number, guard)?;
                whole
            } else {
                number
            };
            self.s.wholes.push(WholeFact {
                root: root_index,
                whole,
                binding: NONE,
                low: 0,
                high: 0,
            });
            self.s.wholes.len() - 1
        };
        let fact = self.s.wholes[at];
        let proven = fact.binding == binding && fact.low <= added && added <= fact.high;
        if !proven || attached {
            // The zero-based index, unsigned, against the byte count: a
            // position below 1 wraps to a number no string is as long as.
            let index = if added == 1 {
                fact.whole
            } else {
                let less = self.iconst(added - 1);
                self.f.bin(Op::Add, fact.whole, less)
            };
            self.guard_cc(Cc::Ult, index, words.len, guard)?;
            let held = &mut self.s.wholes[at];
            if held.binding == binding {
                held.low = held.low.min(added);
                held.high = held.high.max(added);
            } else {
                // The checks of another string are forgotten: one fact per
                // root is enough for the groups a trace has.
                (held.binding, held.low, held.high) = (binding, added, added);
            }
        }
        let byte = self.load8(words.base, Off::V(fact.whole), added - 1);
        let byte = self.byte_as(byte, ty);
        self.s.bytes.push((binding, root_index, added, byte));
        Ok(byte)
    }

    /// Ends the facts about bytes: the code that follows is not below the
    /// checks that made them, or names other values.
    pub(super) fn forget_bytes(&mut self) {
        self.s.wholes.clear();
        self.s.bytes.clear();
    }

    /// A loaded byte as a number of type `ty`.
    fn byte_as(&mut self, byte: Val, ty: TraceType) -> Val {
        if ty == TraceType::Float {
            self.un(Op::I2F, Ty::F64, byte)
        } else {
            byte
        }
    }
}
