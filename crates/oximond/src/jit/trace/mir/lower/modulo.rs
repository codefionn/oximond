//! The float remainder of the Lua 5.3 and later profiles (`ModuloWith` of
//! `Fmod53` and `Fmod54`) and `NumberModulo`, which picks the integer or the
//! float remainder of two numbers at run time.
//!
//! On a CPU with a fused multiply-add the remainder is the form Cranelift
//! emits there (`float_fmod_fma` in `emit.rs`), for every operand pair, not
//! only integral ones: `q = trunc(a / b)` and `r = fma(-q, b, a)` are the exact
//! remainder whenever the quotient is below 2^53 and `r` is zero or has the
//! sign of `a`, and the Lua sign fix then needs two selects. A zero or
//! infinite divisor, a NaN, an infinite dividend, a quotient of 2^53 or more
//! and a remainder of the wrong sign (the rounded quotient was one too many)
//! fail the two tests and take the compat function, which is what
//! `emit.rs` calls. The straight-line path is about thirty instructions with
//! two compares and no call.
//!
//! The call is out of line (`CallIf`: its stub saves the registers around it,
//! so the loop's values stay where they are). The fast result is stored in the
//! control block's `scratch` word first and the call stores its own over it, so
//! the result is what that word holds after the two conditional calls, with
//! no select. Both tests may fail, and then the call is made twice, which
//! gives the same value.
//!
//! Without FMA the remainder is Cranelift's integral form (`float_fmod`): two
//! operands that are integral, below 2^53 in magnitude, with a divisor that is
//! not zero, take an exact integer remainder with the floor sign fix, in a
//! skip, and every other pair takes the compat function. One call, on one
//! flag that folds the five tests.

#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;

use crate::jit_ir::trace::range::FLOAT_EXACT_LIMIT;

#[cfg(test)]
thread_local! {
    /// Calls of the compat remainders from the code of this thread's tests.
    pub(in super::super) static HELPER_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The compat remainders, answering with the bits of the float so the
/// conditional call can store them in the control block.
extern "C" fn trace_fmod53_bits(left: f64, right: f64) -> i64 {
    #[cfg(test)]
    HELPER_CALLS.with(|calls| calls.set(calls.get() + 1));
    trace_mod_fmod53(left, right).to_bits().cast_signed()
}

extern "C" fn trace_fmod54_bits(left: f64, right: f64) -> i64 {
    #[cfg(test)]
    HELPER_CALLS.with(|calls| calls.set(calls.get() + 1));
    trace_mod_fmod54(left, right).to_bits().cast_signed()
}

impl Lowerer<'_, '_> {
    /// `left % right` of Lua 5.3 and later for floats (`fifty_four` for the
    /// Lua 5.4 form of the sign fix, the 5.3 form otherwise).
    pub(super) fn float_fmod(&mut self, left: Val, right: Val, fifty_four: bool) -> Val {
        let helper = if fifty_four {
            trace_fmod54_bits as *const () as usize
        } else {
            trace_fmod53_bits as *const () as usize
        };
        if !self.fused {
            return self.float_fmod_integral(left, right, helper);
        }
        let (zero, limit) = (self.f.fconst(0.0), self.f.fconst(FLOAT_EXACT_LIMIT));
        let quotient = self.f.bin(Op::FDiv, left, right);
        let abs_quotient = self.un(Op::FAbs, Ty::F64, quotient);
        let truncated = self.un(Op::FTrunc, Ty::F64, quotient);
        // `left - truncated * right`, the product not rounded.
        let exact = self.f.fnma(truncated, right, left);
        // `m == exact` unless `exact` is non-zero with a sign other than
        // `left`'s, or NaN.
        let m = self.f.bin(Op::FCopySign, exact, left);
        let shifted = self.f.bin(Op::FAdd, m, right);
        let fast = if fifty_four {
            // `m` is non-zero with a sign other than `right`'s: positive with
            // a negative divisor, or negative with a positive one. So `m`
            // with its sign turned under a negative divisor is below zero,
            // which a zero of either sign is not. A zero divisor and a NaN
            // take the helper below, whatever this picks.
            let turned = self.un(Op::FNeg, Ty::F64, m);
            let against = self.f.select_any(Cc::Lt, right, zero, turned, m);
            self.f.select_any(Cc::Lt, against, zero, shifted, m)
        } else {
            let product = self.f.bin(Op::FMul, m, right);
            self.f.select_any(Cc::Lt, product, zero, shifted, m)
        };
        let word = mem::offset_of!(TraceControl, scratch);
        self.store_at(self.control, Off::K(0), offset(word), fast);
        let result_at = Some(u32::try_from(word).expect("small"));
        // The quotient is below 2^53 (ordered: a NaN fails), and `m` is
        // `exact` (equal: a NaN fails).
        self.f.call_if_storing(
            Cc::Nlt,
            (abs_quotient, limit),
            helper,
            false,
            &[left, right],
            result_at,
        );
        self.f
            .call_if_storing(Cc::Ne, (m, exact), helper, false, &[left, right], result_at);
        self.load_at(Ty::F64, self.control, Off::K(0), offset(word))
    }

    /// The remainder where there is no fused multiply-add, as Cranelift's
    /// `float_fmod` has it. The exact path is in a skip, since `SRem` traps on
    /// a zero divisor, and the call is out of line on the flag that the
    /// operands are not what the exact path takes.
    fn float_fmod_integral(&mut self, left: Val, right: Val, helper: usize) -> Val {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let limit = self.f.fconst(FLOAT_EXACT_LIMIT);
        let abs_left = self.un(Op::FAbs, Ty::F64, left);
        let abs_right = self.un(Op::FAbs, Ty::F64, right);
        // The conversions saturate or answer the indefinite integer outside
        // the range, which the `abs < 2^53` tests then refuse (a NaN fails
        // them), so they matter only for operands that are exact.
        let a = self.un(Op::F2I, Ty::I64, left);
        let b = self.un(Op::F2I, Ty::I64, right);
        let a_back = self.un(Op::I2F, Ty::F64, a);
        let b_back = self.un(Op::I2F, Ty::F64, b);
        let ok = self.f.select_any(Cc::Lt, abs_left, limit, one, zero);
        let ok = self.f.select_any(Cc::Lt, abs_right, limit, ok, zero);
        let ok = self.f.select_any(Cc::Eq, a_back, left, ok, zero);
        let ok = self.f.select_any(Cc::Eq, b_back, right, ok, zero);
        let ok = self.f.select_any(Cc::Ne, b, zero, ok, zero);
        let open = self.f.begin_if(Cc::Ne, ok, zero);
        // `|b| < 2^53`, so the division neither traps nor overflows.
        let remainder = self.f.bin(Op::SRem, a, b);
        let signs = self.f.bin(Op::Xor, remainder, b);
        let differ = self.f.select_any(Cc::Lt, signs, zero, b, zero);
        let correction = self.f.select_any(Cc::Ne, remainder, zero, differ, zero);
        let floored = self.f.bin(Op::Add, remainder, correction);
        let magnitude = self.un(Op::I2F, Ty::F64, floored);
        // A zero result has the sign of the dividend, as `fmod` gives.
        let signed_zero = self.f.bin(Op::FCopySign, magnitude, left);
        let exact = self
            .f
            .select_any(Cc::Eq, floored, zero, signed_zero, magnitude);
        let fast = self.f.end_if(open, &[(exact, left)]);
        let word = mem::offset_of!(TraceControl, scratch);
        self.store_at(self.control, Off::K(0), offset(word), fast);
        let result_at = Some(u32::try_from(word).expect("small"));
        self.f
            .call_if_storing(Cc::Eq, (ok, zero), helper, false, &[left, right], result_at);
        self.load_at(Ty::F64, self.control, Off::K(0), offset(word))
    }

    /// [`TraceOp::NumberModulo`]: the integer floor modulo of two integers
    /// held as floats when `integer` is set, the float modulo `float`
    /// otherwise, each in a skip that runs for its operands only.
    pub(super) fn number_modulo(
        &mut self,
        float: ArithmeticOp,
        (left, right): (ValueId, ValueId),
        integer: ValueId,
    ) -> Res<Val> {
        let (l, r, flag) = (self.get(left)?, self.get(right)?, self.get(integer)?);
        let zero = self.iconst(0);
        let open = self.f.begin_if(Cc::Ne, flag, zero);
        // Both are integers within 2^53 in magnitude, so the conversions are
        // exact. The divisor is not zero (the trace checked), and the dividend
        // is far from `MIN`, so the division needs neither special case. The
        // floor form adds the divisor to a remainder of the other sign.
        let left_integer = self.un(Op::F2I, Ty::I64, l);
        let right_integer = self.un(Op::F2I, Ty::I64, r);
        let remainder = self.f.bin(Op::SRem, left_integer, right_integer);
        let signs = self.f.bin(Op::Xor, remainder, right_integer);
        let differ = self.f.select_any(Cc::Lt, signs, zero, right_integer, zero);
        let correction = self.f.select_any(Cc::Ne, remainder, zero, differ, zero);
        let floored = self.f.bin(Op::Add, remainder, correction);
        let whole = self.un(Op::I2F, Ty::F64, floored);
        let value = self.f.end_if(open, &[(whole, l)]);
        let open = self.f.begin_if(Cc::Eq, flag, zero);
        let by_float = self.float_arithmetic(float, l, r, right)?;
        Ok(self.f.end_if(open, &[(by_float, value)]))
    }
}
