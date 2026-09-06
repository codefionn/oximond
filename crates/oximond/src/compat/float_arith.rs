//! Float `%` and `^`, spelled the way each stock Lua computes them.
//!
//! Every consumer of float arithmetic (the interpreter, constant folding,
//! trace semantics, native helpers, string arithmetic) goes through these
//! functions, so they cannot disagree.
//!
//! Modulo, from the C sources:
//!
//! * Lua 5.1, 5.2 and `LuaJIT` 2.1: `a - floor(a / b) * b`, unfused (multiply,
//!   round, then subtract). The distro `lua5.1` the difftest corpus is
//!   checked against is built with FP contraction (x86-64-v3), which fuses
//!   the two. That is a build artefact, not the language, so oximond does
//!   not follow it (see `float-modulo-lua51.known-failure`).
//! * Lua 5.3: `m = fmod(a, b); if (m * b < 0) m += b`.
//! * Lua 5.4 and 5.5: `m = fmod(a, b); if ((m > 0) ? b < 0 : (m < 0 && b > 0)) m += b`
//!   (the sign of a non-zero remainder differs from the divisor's).
//!
//! Power is C `pow`, except that Lua 5.4 and 5.5 compute `a ^ 2` as `a * a`
//! (`luai_numpow`), which rounds differently from `pow` for some `a`.

#![allow(clippy::float_cmp, reason = "IEEE comparison is the semantics")]

use crate::compat::CompatibilityProfile;

/// `f64::floor` without the libm call baseline x86-64 emits for it.
///
/// For magnitudes below 2^51, adding and subtracting 1.5 * 2^52 rounds to the
/// nearest integer in two additions, with no float-to-integer conversion. A
/// step down fixes the rounding when it went up. Larger values are already
/// integral or close to it, and NaN fails the range check, so all of them take
/// the exact library path.
#[inline(always)]
#[allow(clippy::float_cmp)]
fn fast_floor(value: f64) -> f64 {
    const ROUNDER: f64 = 6_755_399_441_055_744.0;
    const LIMIT: f64 = 2_251_799_813_685_248.0;
    if !(-LIMIT..LIMIT).contains(&value) {
        return value.floor();
    }
    let rounded = (value + ROUNDER) - ROUNDER;
    let floored = if rounded > value {
        rounded - 1.0
    } else {
        rounded
    };
    // Keep the sign of a negative zero quotient, as `floor(-0.0)` does.
    if floored == 0.0 { value * 0.0 } else { floored }
}

/// How a profile computes float modulo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatModulo {
    /// `a - floor(a / b) * b`, rounding the product (Lua 5.1, 5.2, `LuaJIT`).
    Floor,
    /// `fmod` and the `m * b < 0` fix (Lua 5.3).
    Fmod53,
    /// `fmod` and the sign fix of Lua 5.4 and 5.5.
    Fmod54,
}

/// Whether the CPU has the instructions of
/// [`FloatModulo::apply_fmod_hardware`]: [`FMOD_HARDWARE_YES`] once
/// [`FloatModulo::apply_fmod_undetected`] found them.
#[cfg(target_arch = "x86_64")]
static FMOD_HARDWARE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
const FMOD_HARDWARE_YES: u8 = 1;

impl FloatModulo {
    /// `left % right` on floats.
    #[must_use]
    #[allow(clippy::inline_always, reason = "the interpreter's hot arm")]
    #[inline(always)]
    pub fn apply(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Floor => Self::floor_apply(left, right),
            // Out of line: the `fmod` forms are bigger, and keeping them
            // out of the arm keeps a Lua 5.1 `%` as small as it was.
            Self::Fmod53 | Self::Fmod54 => self.apply_fmod(left, right),
        }
    }

    /// `a - floor(a / b) * b`, the [`FloatModulo::Floor`] form, with no
    /// dispatch on the profile. The interpreter's hot arm calls this when
    /// the profile, checked once outside the dispatch loop, is `Floor`.
    #[must_use]
    #[allow(clippy::inline_always, reason = "the interpreter's hot arm")]
    #[inline(always)]
    pub fn floor_apply(left: f64, right: f64) -> f64 {
        left - fast_floor(left / right) * right
    }

    /// Whether [`Self::floor_apply_hardware`] may run. On x86-64 it needs
    /// `roundsd` of SSE4.1, which floors a double in one instruction. The
    /// baseline x86-64 build cannot assume it, so the interpreter asks once
    /// per run and keeps the answer. A CPU without it takes the out-of-line
    /// [`Self::apply_fmod`], which answers `Floor` with [`Self::floor_apply`].
    /// Other targets have no requirement.
    #[must_use]
    #[inline]
    pub fn has_hardware_floor() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // The detection result is cached in an atomic by std.
            std::arch::is_x86_feature_detected!("sse4.1")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            true
        }
    }

    /// [`Self::floor_apply`] with the quotient floored by one `roundsd`
    /// (round toward negative infinity, precision exception suppressed)
    /// where `floor_apply` takes a range check, two additions, a correction
    /// and a test for zero. `roundsd` is IEEE `floor` for every input, a
    /// negative zero, an infinity and a NaN included, so the result is the
    /// same bit for bit. The instruction is written out because the build
    /// does not enable SSE4.1, and a function compiled with the feature
    /// would not inline into the dispatch loop.
    ///
    /// # Safety
    ///
    /// The CPU has SSE4.1: [`Self::has_hardware_floor`] returned `true`.
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    #[allow(clippy::inline_always, reason = "the interpreter's hot arm")]
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn floor_apply_hardware(left: f64, right: f64) -> f64 {
        let mut quotient = left / right;
        // SAFETY: the contract. The instruction reads and writes only the
        // one register.
        unsafe {
            std::arch::asm!(
                "roundsd {q}, {q}, 9",
                q = inout(xmm_reg) quotient,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        left - quotient * right
    }

    /// [`Self::floor_apply`]: this target has no hardware form to choose.
    ///
    /// # Safety
    ///
    /// None is needed here; the signature matches the x86-64 form.
    #[cfg(not(target_arch = "x86_64"))]
    #[must_use]
    #[allow(clippy::inline_always, reason = "the interpreter's hot arm")]
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn floor_apply_hardware(left: f64, right: f64) -> f64 {
        Self::floor_apply(left, right)
    }

    /// The `fmod` forms ([`FloatModulo::Fmod53`], [`FloatModulo::Fmod54`]),
    /// out of line. `Floor` is answered with its own formula, so a caller
    /// that passes it still gets the profile's result.
    ///
    /// A CPU with `fma` and `roundsd` takes the remainder from one fused
    /// multiply-add ([`Self::apply_fmod_hardware`]). The baseline x86-64
    /// build has neither instruction, so the form is chosen at run time.
    #[must_use]
    #[inline(never)]
    pub fn apply_fmod(self, left: f64, right: f64) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            // One byte answers for both features. Asking std for each cost
            // two loads and three tests, and its slow path gave this
            // function a prologue that every call paid.
            if FMOD_HARDWARE.load(std::sync::atomic::Ordering::Relaxed) == FMOD_HARDWARE_YES {
                // SAFETY: the byte is `FMOD_HARDWARE_YES` only after
                // `apply_fmod_undetected` found `fma` and `sse4.1`.
                #[allow(unsafe_code)]
                return unsafe { self.apply_fmod_hardware(left, right) };
            }
            self.apply_fmod_undetected(left, right)
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.apply_fmod_portable(left, right)
    }

    /// [`Self::apply_fmod`] before the CPU was asked, or on a CPU without
    /// the hardware form: asks once, remembers a yes, and answers.
    #[cfg(target_arch = "x86_64")]
    #[cold]
    #[inline(never)]
    fn apply_fmod_undetected(self, left: f64, right: f64) -> f64 {
        if std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            FMOD_HARDWARE.store(FMOD_HARDWARE_YES, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: the CPU supports `fma` and `sse4.1`, just checked.
            #[allow(unsafe_code)]
            return unsafe { self.apply_fmod_hardware(left, right) };
        }
        self.apply_fmod_portable(left, right)
    }

    /// [`Self::apply_fmod`] compiled with the `fma` and `sse4.1`
    /// instructions: the remainder [`fma_fmod_checked`] proves, then the
    /// profile's sign fix, in one function with no call. That is `fmod` bit
    /// for bit, so the result is the portable form's for every operand
    /// pair, integral ones included: their remainder is exact either way,
    /// a zero keeps the dividend's sign and both sign fixes leave a zero
    /// alone. Operands the one `fma` does not prove (a huge quotient, a
    /// zero divisor, an infinity, a NaN) take the portable form.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma,sse4.1")]
    #[allow(unsafe_code)]
    unsafe fn apply_fmod_hardware(self, left: f64, right: f64) -> f64 {
        if self == Self::Floor {
            return Self::floor_apply(left, right);
        }
        let Some(m) = fma_fmod_checked_body(left, right) else {
            return self.apply_fmod_portable(left, right);
        };
        let fix = if self == Self::Fmod53 {
            m * right < 0.0
        } else {
            (m > 0.0 && right < 0.0) || (m < 0.0 && right > 0.0)
        };
        if fix { m + right } else { m }
    }

    /// [`Self::apply_fmod`] without the hardware form: integral operands
    /// take an integer remainder, and the rest go through [`fmod`].
    #[must_use]
    #[inline(never)]
    fn apply_fmod_portable(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Fmod53 => {
                integral_remainder(left, right).unwrap_or_else(|| fmod53_general(left, right))
            }
            Self::Fmod54 => {
                integral_remainder(left, right).unwrap_or_else(|| fmod54_general(left, right))
            }
            Self::Floor => Self::floor_apply(left, right),
        }
    }
}

/// Both operands are integral with magnitude below 2^53, and the divisor is
/// not zero: the floored remainder, computed exactly on integers. This is
/// what `fmod` plus either sign fix gives for such operands: a non-zero
/// remainder takes the divisor's sign, and a zero one keeps the sign of the
/// dividend (so `-6.0 % 3.0` is `-0.0`). The native trace code does the same
/// test and arithmetic inline.
#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn integral_remainder(left: f64, right: f64) -> Option<f64> {
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    if !(left.abs() < LIMIT && right.abs() < LIMIT) {
        return None;
    }
    let (a, b) = (left as i64, right as i64);
    if a as f64 != left || b as f64 != right || b == 0 {
        return None;
    }
    let mut m = a % b;
    if m != 0 && (m ^ b) < 0 {
        m += b;
    }
    Some(if m == 0 {
        0.0_f64.copysign(left)
    } else {
        m as f64
    })
}

/// `left % right` on floats, bit for bit: the `fma` form when the CPU has
/// one, else Rust's `%`. The baseline x86-64 build has no `fma` instruction,
/// and `%` is a software loop of 200 or more instructions per operation, so
/// the form is chosen at run time. [`fma_fmod_checked`] declines operands
/// whose quotient is too large, infinite, NaN or zero-divided, and `%` answers
/// those.
#[allow(clippy::inline_always, reason = "the interpreter's modulo path")]
#[inline(always)]
fn fmod(left: f64, right: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        // The detection result is cached in an atomic by std.
        if std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            // SAFETY: the CPU supports `fma` and `sse4.1`, just checked.
            #[allow(unsafe_code)]
            if let Some(m) = unsafe { fma_fmod_hardware(left, right) } {
                return m;
            }
        }
    }
    left % right
}

/// [`fma_fmod_checked`] compiled with the `fma` and `sse4.1` instructions, so
/// `mul_add` and `trunc` are one instruction each instead of libm calls.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,sse4.1")]
#[allow(unsafe_code)]
unsafe fn fma_fmod_hardware(left: f64, right: f64) -> Option<f64> {
    fma_fmod_checked_body(left, right)
}

/// Lua 5.3's `%` on floats for any operands.
#[inline(never)]
fn fmod53_general(left: f64, right: f64) -> f64 {
    let m = fmod(left, right);
    if m * right < 0.0 { m + right } else { m }
}

/// Lua 5.4's and 5.5's `%` on floats for any operands.
#[inline(never)]
fn fmod54_general(left: f64, right: f64) -> f64 {
    let m = fmod(left, right);
    if (m > 0.0 && right < 0.0) || (m < 0.0 && right > 0.0) {
        m + right
    } else {
        m
    }
}

/// The helper native code calls for the operands the inline integral path
/// does not take, for [`FloatModulo::Fmod53`].
#[must_use]
pub fn fmod53_helper(left: f64, right: f64) -> f64 {
    fmod53_general(left, right)
}

/// As [`fmod53_helper`], for [`FloatModulo::Fmod54`].
#[must_use]
pub fn fmod54_helper(left: f64, right: f64) -> f64 {
    fmod54_general(left, right)
}

/// `fmod(left, right)` with the sign of `left` on a zero, from one
/// single-rounding `fma`, for finite `right` and `|left / right| < 2^53`.
/// `None` for any other operands.
///
/// `q = trunc(left / right)` is within one of the true quotient. The exact
/// remainder `left - q * right` is representable once `q` is right, and
/// otherwise rounds to a value whose magnitude is at least `|right|` (`q` too
/// small) or whose sign differs from `left`'s (`q` too large), so one
/// correction and a second `fma` give `fmod` bit for bit. Native trace code
/// emits this inline when the target has a one-instruction `fma`, and the
/// interpreter runs it through `fmod` when the CPU reports `fma`. The
/// baseline x86-64 build has no `fma` instruction, so compiled without the
/// target feature `mul_add` would be a libm call, no cheaper than `fmod`.
/// This function is the executable specification the tests check against
/// libm.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn fma_fmod(left: f64, right: f64) -> Option<f64> {
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    let quotient = left / right;
    if !(quotient.abs() < LIMIT && right.abs() < f64::INFINITY) {
        return None;
    }
    let q = quotient as i64 as f64;
    let first = (-q).mul_add(right, left);
    let away = 1.0_f64.copysign(left) * 1.0_f64.copysign(right);
    let step = if first.abs() >= right.abs() {
        away
    } else if first != 0.0 && first.is_sign_negative() != left.is_sign_negative() {
        -away
    } else {
        0.0
    };
    let exact = (-(q + step)).mul_add(right, left);
    Some(exact.copysign(left))
}

/// `fmod(left, right)` with the sign of `left` on a zero, when one `fma`
/// proves it, else `None`. This is what native trace code emits inline on a
/// target with a one-instruction `fma`, and the executable specification the
/// tests check against libm.
///
/// With `q = trunc(left / right)` and `r = fma(-q, right, left)`, `r` is the
/// exact remainder when `|left / right| < 2^53`, `|r| < |right|` and `r` is
/// zero or has the sign of `left`. A `q` that is one too small leaves
/// `|r| >= |right|`. One too large leaves a non-zero `r` of the wrong sign.
/// Either way the rounded `r` still fails the test, and so do NaN, an
/// infinite dividend or divisor and a zero divisor.
#[must_use]
pub fn fma_fmod_checked(left: f64, right: f64) -> Option<f64> {
    fma_fmod_checked_body(left, right)
}

/// The body of [`fma_fmod_checked`], inlined into the `fma`-enabled copy so
/// its `mul_add` is one instruction there.
#[inline(always)]
fn fma_fmod_checked_body(left: f64, right: f64) -> Option<f64> {
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    let quotient = left / right;
    let q = quotient.trunc();
    let r = (-q).mul_add(right, left);
    let m = r.copysign(left);
    if quotient.abs() < LIMIT && r.abs() < right.abs() && m == r {
        Some(m)
    } else {
        None
    }
}

/// A profile's float `%` and `^`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatArith {
    /// The modulo formula.
    pub modulo: FloatModulo,
    /// Whether `a ^ 2` is `a * a` instead of `pow`.
    pub square_shortcut: bool,
}

impl FloatArith {
    /// `left % right`.
    #[must_use]
    #[allow(clippy::inline_always, reason = "the interpreter's hot arm")]
    #[inline(always)]
    pub fn modulo(self, left: f64, right: f64) -> f64 {
        self.modulo.apply(left, right)
    }

    /// `left ^ right`.
    #[must_use]
    #[inline]
    pub fn power(self, left: f64, right: f64) -> f64 {
        if self.square_shortcut && right == 2.0 {
            left * left
        } else {
            left.powf(right)
        }
    }
}

impl CompatibilityProfile {
    /// How this profile computes float `%` and `^`.
    #[must_use]
    pub const fn float_arith(self) -> FloatArith {
        match self {
            Self::Lua51 | Self::Lua52 | Self::LuaJit21 => FloatArith {
                modulo: FloatModulo::Floor,
                square_shortcut: false,
            },
            Self::Lua53 => FloatArith {
                modulo: FloatModulo::Fmod53,
                square_shortcut: false,
            },
            Self::Lua54 | Self::Lua55 => FloatArith {
                modulo: FloatModulo::Fmod54,
                square_shortcut: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f64) -> u64 {
        value.to_bits()
    }

    #[test]
    fn fast_floor_matches_floor_bit_for_bit() {
        let limit = 2_251_799_813_685_248.0_f64;
        let mut cases = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            9_007_199_254_740_992.0,
            9_223_372_036_854_775_808.0,
            -9_223_372_036_854_775_808.0,
        ];
        for edge in [
            limit,
            4_503_599_627_370_496.0,
            0.5,
            1.0,
            9_007_199_254_740_992.0,
        ] {
            for sign in [1.0, -1.0] {
                let edge = edge * sign;
                cases.extend([edge, edge.next_up(), edge.next_down()]);
            }
        }
        for value in cases {
            assert_eq!(bits(fast_floor(value)), bits(value.floor()), "{value:e}");
        }
        // SplitMix64 with a fixed seed.
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for index in 0..10_500_000_u32 {
            let raw = next();
            let value = match index % 5 {
                // Any bit pattern: huge, tiny, NaN, and infinities.
                0 => f64::from_bits(raw),
                // Unit-scale fractions of both signs.
                1 => (raw as f64 / u64::MAX as f64 - 0.5) * 20.0,
                // Integral values up to 2^52.
                2 => ((raw >> 12) as i64 - (1 << 51)) as f64,
                // Half-integral values.
                3 => ((raw >> 12) as i64 - (1 << 51)) as f64 + 0.5,
                // Magnitudes around the fast-path limit.
                _ => (raw as f64 / u64::MAX as f64 - 0.5) * 2.0 * limit * 1.1,
            };
            assert_eq!(bits(fast_floor(value)), bits(value.floor()), "{value:e}");
        }
    }

    /// Native trace code tests only the range of the quotient and the sign
    /// of the remainder, not `|r| < |right|`: `trunc` of the rounded quotient
    /// is the true quotient or one more (a rounded quotient is never below an
    /// integer the true one reached), and one more leaves a non-zero `r` of
    /// the wrong sign. This checks that whenever those two tests pass, the
    /// remainder is `fmod`'s, bit for bit, on pairs that make the quotient
    /// land next to an integer.
    #[test]
    fn range_and_sign_alone_prove_the_fma_remainder() {
        let mut state = 0xD1B5_4A32_D192_ED03_u64;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let unit = |raw: u64| raw as f64 / u64::MAX as f64;
        let mut accepted = 0_u32;
        for index in 0..6_000_000_u32 {
            let divisor = match index % 4 {
                0 => f64::from_bits(next()),
                1 => (unit(next()) - 0.5) * 200.0,
                2 => ((next() >> 11) as i64 - (1 << 52)) as f64 * 0.125,
                _ => (unit(next()) - 0.5) * 1e-3,
            };
            let multiple = ((next() >> 20) as i64 - (1 << 43)) as f64;
            let dividend = match index % 5 {
                0 => f64::from_bits(next()),
                1 => multiple * divisor,
                2 => (multiple * divisor).next_up(),
                3 => (multiple * divisor).next_down(),
                _ => (unit(next()) - 0.5) * 1e6,
            };
            let quotient = dividend / divisor;
            let rest = (-quotient.trunc()).mul_add(divisor, dividend);
            let signed = rest.copysign(dividend);
            if quotient.abs() < 9_007_199_254_740_992.0 && signed == rest {
                accepted += 1;
                assert!(
                    rest.abs() < divisor.abs(),
                    "{dividend:e} % {divisor:e}: {rest:e}"
                );
                let expected = dividend % divisor;
                assert_eq!(bits(signed), bits(expected), "{dividend:e} % {divisor:e}");
            }
        }
        assert!(accepted > 1_000_000, "{accepted}");
    }

    #[test]
    fn modulo_matches_the_stock_results() {
        // Values printed by the stock interpreters (lua5.1, lua5.4, luajit).
        let unfused = FloatModulo::Floor;
        let fmod = FloatModulo::Fmod54;
        assert_eq!(unfused.apply(-583.0, 3.001), 2.194_999_999_999_936_3);
        assert_eq!(fmod.apply(-583.0, 3.001), 2.194_999_999_999_978_5);
        assert_eq!(unfused.apply(1e10, 4.0 / -2.25), 0.0);
        assert_eq!(fmod.apply(1e10, 4.0 / -2.25), -1.777_777_222_666_265_4);
    }

    #[test]
    fn fmod_modulo_edge_cases() {
        let m = FloatModulo::Fmod54;
        assert_eq!(m.apply(5.0, f64::INFINITY), 5.0);
        assert_eq!(m.apply(-5.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(m.apply(5.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(m.apply(-5.0, f64::NEG_INFINITY), -5.0);
        assert!(m.apply(f64::INFINITY, 3.0).is_nan());
        assert!(m.apply(3.0, 0.0).is_nan());
        assert_eq!(bits(m.apply(-0.0, 3.0)), bits(-0.0));
        assert_eq!(bits(m.apply(0.0, -3.0)), bits(0.0));
        assert_eq!(bits(m.apply(-0.0, -3.0)), bits(-0.0));
        assert_eq!(bits(m.apply(-6.0, 3.0)), bits(-0.0));
        assert_eq!(bits(m.apply(-0.0, f64::NEG_INFINITY)), bits(-0.0));
    }

    /// The integral fast path agrees with the general formula.
    #[test]
    fn integral_fast_path_matches_the_general_formula() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.0,
            -3.0,
            5.0,
            -5.0,
            6.0,
            -6.0,
            7.0,
            -7.0,
            1e15,
            -1e15,
            9_007_199_254_740_991.0,
            -9_007_199_254_740_991.0,
            9_007_199_254_740_992.0,
            -9_007_199_254_740_992.0,
            1e300,
            0.5,
            -2.5,
            3.001,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &a in &values {
            for &b in &values {
                let fast54 = FloatModulo::Fmod54.apply(a, b);
                let slow54 = fmod54_general(a, b);
                let fast53 = FloatModulo::Fmod53.apply(a, b);
                let slow53 = fmod53_general(a, b);
                assert!(
                    fast54.to_bits() == slow54.to_bits() || (fast54.is_nan() && slow54.is_nan()),
                    "5.4 {a} % {b}: {fast54} vs {slow54}"
                );
                assert!(
                    fast53.to_bits() == slow53.to_bits() || (fast53.is_nan() && slow53.is_nan()),
                    "5.3 {a} % {b}: {fast53} vs {slow53}"
                );
            }
        }
    }

    /// Deterministic xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// A float of mixed magnitude: random bits, a bounded exponent,
        /// integers, or a value near a power of two.
        fn float(&mut self) -> f64 {
            let bits = self.next();
            match bits & 7 {
                0 => f64::from_bits(self.next()),
                1 => ((self.next() >> 11) as i64 - (1 << 52)) as f64,
                2 => (self.next() % 2001) as f64 - 1000.0,
                3 => {
                    let near = 9_007_199_254_740_992.0_f64;
                    let delta = (self.next() % 9) as f64 - 4.0;
                    (near + delta).copysign(f64::from_bits(self.next()))
                }
                4 => f64::from_bits(self.next() & !(0x7ff << 52)) * 1e-300,
                _ => {
                    let mantissa = self.next() & ((1 << 52) - 1);
                    let exponent = 1023 - 60 + (self.next() % 120);
                    let sign = self.next() & (1 << 63);
                    f64::from_bits(sign | (exponent << 52) | mantissa)
                }
            }
        }
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    fn check_pair(a: f64, b: f64) {
        // libm `fmod` is Rust's `%` on floats.
        let expected = a % b;
        if let Some(got) = fma_fmod(a, b) {
            assert!(
                expected == 0.0 && got == 0.0 && got.is_sign_negative() == a.is_sign_negative()
                    || same(got, expected),
                "fma_fmod({a:e}, {b:e}) = {got:e}, fmod = {expected:e}"
            );
        }
        if let Some(got) = fma_fmod_checked(a, b) {
            assert!(
                expected == 0.0 && got == 0.0 && got.is_sign_negative() == a.is_sign_negative()
                    || same(got, expected),
                "fma_fmod_checked({a:e}, {b:e}) = {got:e}, fmod = {expected:e}"
            );
        }
        let m = fmod54_general(a, b);
        let m53 = fmod53_general(a, b);
        assert!(
            same(FloatModulo::Fmod54.apply(a, b), m),
            "5.4 {a:e} % {b:e}"
        );
        assert!(
            same(FloatModulo::Fmod53.apply(a, b), m53),
            "5.3 {a:e} % {b:e}"
        );
    }

    fn random_pairs(count: u64) {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..count {
            let (a, b) = (rng.float(), rng.float());
            check_pair(a, b);
            // A divisor of nearby magnitude exercises the correction step.
            let scaled = a / (1.0 + (rng.next() % 1000) as f64) * f64::from_bits(rng.next() | 1);
            if scaled.is_finite() {
                check_pair(a, scaled);
            }
        }
    }

    /// The `%` forms against Rust's `%`, written out here so the check does
    /// not go through the code under test, including the edge values.
    #[test]
    fn modulo_forms_match_the_reference_formulas() {
        let reference53 = |a: f64, b: f64| {
            let m = a % b;
            if m * b < 0.0 { m + b } else { m }
        };
        let reference54 = |a: f64, b: f64| {
            let m = a % b;
            if (m > 0.0 && b < 0.0) || (m < 0.0 && b > 0.0) {
                m + b
            } else {
                m
            }
        };
        let edges = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -2.5,
            7.5,
            -7.5,
            1e300,
            -1e300,
            5e-324,
            9_007_199_254_740_993.0,
            -9_223_372_036_854_775_808.0,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut pairs: Vec<(f64, f64)> = edges
            .iter()
            .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
            .collect();
        pairs.extend((0..20_000).map(|_| (rng.float(), rng.float())));
        for (a, b) in pairs {
            assert!(
                same(FloatModulo::Fmod53.apply_fmod(a, b), reference53(a, b)),
                "5.3 {a:e} % {b:e}"
            );
            assert!(
                same(FloatModulo::Fmod54.apply_fmod(a, b), reference54(a, b)),
                "5.4 {a:e} % {b:e}"
            );
        }
    }

    /// The inline path must take the common operands, or native code pays the
    /// helper call: integral and fractional pairs of both signs.
    #[test]
    fn fma_fmod_checked_takes_ordinary_operands() {
        let divisors = [7.5, -7.5, 0.25, -0.25, 1e3, -3.0, 3.0];
        let mut taken = 0;
        let mut total = 0;
        for i in 0..2000_i32 {
            let x = f64::from(i % 1013 - 506) * 0.75;
            for d in divisors {
                total += 1;
                taken += usize::from(fma_fmod_checked(x, d).is_some());
            }
        }
        // A one-off quotient is rare, never common.
        assert!(taken * 100 >= total * 98, "{taken} of {total}");
    }

    #[test]
    fn fma_fmod_matches_libm_on_random_pairs() {
        random_pairs(300_000);
    }

    /// Several million pairs:
    /// `cargo test -p oximond --release fma_fmod_matches_libm_on_millions_of_pairs -- --ignored`.
    #[test]
    #[ignore = "millions of pairs; run in release mode"]
    fn fma_fmod_matches_libm_on_millions_of_pairs() {
        random_pairs(5_000_000);
    }

    #[test]
    fn fma_fmod_matches_libm_on_edge_values() {
        let min_subnormal = f64::from_bits(1);
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.1,
            0.1,
            3.0,
            -3.0,
            1e-310,
            -1e-310,
            min_subnormal,
            -min_subnormal,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            f64::EPSILON,
            9_007_199_254_740_991.0,
            9_007_199_254_740_992.0,
            9_007_199_254_740_993.0,
            -9_007_199_254_740_991.0,
            4_503_599_627_370_497.5,
            1e22,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &a in &values {
            for &b in &values {
                check_pair(a, b);
            }
        }
    }

    /// The dispatch-free entry points the interpreter's hot arm calls agree
    /// with `apply` for every form.
    #[test]
    fn direct_entry_points_match_apply() {
        let mut rng = Rng(7);
        for _ in 0..50_000 {
            let (a, b) = (rng.float(), rng.float());
            assert!(same(
                FloatModulo::floor_apply(a, b),
                FloatModulo::Floor.apply(a, b)
            ));
            for form in [FloatModulo::Fmod53, FloatModulo::Fmod54] {
                assert!(same(form.apply_fmod(a, b), form.apply(a, b)));
            }
        }
    }

    /// The run-time-selected `fmod` forms give the portable forms bit for
    /// bit, on integral and fractional operands and on the operands the
    /// hardware form hands back.
    #[test]
    fn fmod_forms_match_their_portable_forms() {
        let special = [
            0.0,
            -0.0,
            0.25,
            -0.25,
            1.0,
            -1.0,
            3.0,
            -3.0,
            6.0,
            -6.0,
            7.5,
            -7.5,
            1e3,
            -506.0 * 0.75,
            5e-324,
            f64::MIN_POSITIVE,
            9_007_199_254_740_991.0,
            9_007_199_254_740_992.0,
            -9_007_199_254_740_993.0,
            1e22,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for form in [FloatModulo::Fmod53, FloatModulo::Fmod54, FloatModulo::Floor] {
            let check = |a: f64, b: f64| {
                let (got, expected) = (form.apply_fmod(a, b), form.apply_fmod_portable(a, b));
                assert!(
                    same(got, expected),
                    "{form:?}: {a:e} % {b:e} = {got:e}, portable {expected:e}"
                );
            };
            for &a in &special {
                for &b in &special {
                    check(a, b);
                }
            }
            let mut rng = Rng(13);
            for _ in 0..100_000 {
                check(rng.float(), rng.float());
            }
            for i in -1500..1500 {
                for d in [7.5, -7.5, 0.25, -0.25, 1e3, -3.0, 3.0, 7.0, -7.0] {
                    check(f64::from(i) * 0.75, d);
                    check(f64::from(i), d);
                }
            }
        }
    }

    /// The one-instruction floor gives the floor formula bit for bit: random
    /// operands, quotients on both sides of the software floor's range
    /// limit, and the zeros, infinities and NaN that the software form
    /// handles on its own paths.
    #[test]
    #[allow(unsafe_code)]
    fn hardware_floor_matches_the_floor_formula() {
        if !FloatModulo::has_hardware_floor() {
            return;
        }
        let check = |a: f64, b: f64| {
            // SAFETY: the CPU has the instruction, checked above.
            let got = unsafe { FloatModulo::floor_apply_hardware(a, b) };
            let expected = FloatModulo::floor_apply(a, b);
            assert!(
                same(got, expected),
                "{a:e} % {b:e}: hardware {got:e}, formula {expected:e}"
            );
        };
        let special = [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            3.0,
            -3.0,
            7.5,
            -7.5,
            97.0,
            -97.0,
            1e-320,
            -1e-320,
            f64::MIN_POSITIVE,
            2_251_799_813_685_247.5,
            2_251_799_813_685_248.0,
            -2_251_799_813_685_248.0,
            -2_251_799_813_685_248.5,
            4_503_599_627_370_496.0,
            9_007_199_254_740_993.0,
            1e22,
            -1e22,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &a in &special {
            for &b in &special {
                check(a, b);
            }
        }
        let mut rng = Rng(11);
        for _ in 0..200_000 {
            check(rng.float(), rng.float());
        }
        for i in -2000..2000 {
            for d in [1.0, -1.0, 7.0, -7.0, 0.75, -0.75, 1013.0] {
                check(f64::from(i) * 0.25, d);
            }
        }
    }

    #[test]
    fn lua51_and_lua52_modulo_is_the_unfused_floor_formula() {
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua52] {
            let arith = profile.float_arith();
            assert_eq!(arith.modulo, FloatModulo::Floor);
            assert_eq!(arith.modulo(-583.0, 3.001), 2.194_999_999_999_936_3);
            assert_eq!(arith.modulo(1e10, 4.0 / -2.25), 0.0);
        }
    }

    #[test]
    fn square_shortcut_rounds_like_a_product() {
        let x = 483.0_f64.powf(3.0);
        let stock = FloatArith {
            modulo: FloatModulo::Floor,
            square_shortcut: false,
        };
        // The C library's pow: glibc rounds this tie of
        // 12_696_463_968_316_569 up to ..570, the Windows C runtime to ..568.
        assert_eq!(stock.power(x, 2.0), x.powf(2.0));
        assert_eq!(
            CompatibilityProfile::Lua54.float_arith().power(x, 2.0),
            12_696_463_968_316_568.0
        );
        assert_eq!(
            CompatibilityProfile::Lua55.float_arith().power(x, 3.0),
            x.powf(3.0)
        );
    }
}
