//! Comparison of generated observations. Exact comparison is the default.

/// Default tolerance for explicitly enabled approximate comparisons.
pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-12;
/// Default relative tolerance for explicitly enabled approximate comparisons.
pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Comparison {
    #[default]
    Exact,
    Approximate {
        absolute: f64,
        relative: f64,
    },
}

impl Comparison {
    /// Enable approximate comparison with the standard tolerances.
    #[must_use]
    pub const fn approximate() -> Self {
        Self::Approximate {
            absolute: DEFAULT_ABSOLUTE_TOLERANCE,
            relative: DEFAULT_RELATIVE_TOLERANCE,
        }
    }

    /// Reject negative and nonfinite tolerances, including direct enum construction.
    ///
    /// # Errors
    /// Returns an error if either tolerance is negative, infinite, or NaN.
    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Exact => Ok(()),
            Self::Approximate { absolute, relative }
                if absolute.is_finite()
                    && relative.is_finite()
                    && absolute >= 0.0
                    && relative >= 0.0 =>
            {
                Ok(())
            }
            Self::Approximate { .. } => Err("float tolerances must be finite and nonnegative"),
        }
    }
}

/// Compare stdout while preserving line boundaries and the final newline.
/// Invalid tolerances never enable approximate equality.
#[must_use]
pub fn stdout_equal(expected: &[u8], actual: &[u8], comparison: Comparison) -> bool {
    if expected == actual {
        return true;
    }
    let mut expected = expected.split(|&b| b == b'\n');
    let mut actual = actual.split(|&b| b == b'\n');
    loop {
        match (expected.next(), actual.next()) {
            (Some(a), Some(b)) if line_equal(a, b, comparison) => {}
            (None, None) => return true,
            _ => return false,
        }
    }
}

/// Compare one generated `tag#count value ...` observation.
/// Headers, whitespace, quoted strings, integer tags, and unknown formats stay exact.
#[must_use]
pub fn line_equal(expected: &[u8], actual: &[u8], comparison: Comparison) -> bool {
    if expected == actual {
        return true;
    }
    let Comparison::Approximate { absolute, relative } = comparison else {
        return false;
    };
    if comparison.validate().is_err() {
        return false;
    }
    let (Some((ah, av)), Some((bh, bv))) = (observation(expected), observation(actual)) else {
        return false;
    };
    ah == bh
        && av.len() == bv.len()
        && av
            .iter()
            .zip(&bv)
            .all(|(a, b)| a == b || numeric_equal(a, b, absolute, relative))
}

fn observation(line: &[u8]) -> Option<(&[u8], Vec<&[u8]>)> {
    let space = line.iter().position(|&b| b == b' ')?;
    let header = &line[..space];
    let hash = header.iter().rposition(|&b| b == b'#')?;
    if hash == 0 || header[..hash].iter().any(u8::is_ascii_whitespace) {
        return None;
    }
    let count = &header[hash + 1..];
    if count.is_empty() || !count.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let count: usize = std::str::from_utf8(count).ok()?.parse().ok()?;
    let mut values = Vec::new();
    let mut pos = space + 1;
    while pos < line.len() {
        let start = pos;
        if line[pos] == b'"' {
            pos += 1;
            loop {
                match *line.get(pos)? {
                    b'\\' => {
                        pos += 1;
                        line.get(pos)?;
                        pos += 1;
                    }
                    b'"' => {
                        pos += 1;
                        break;
                    }
                    b'\n' | b'\r' => return None,
                    _ => pos += 1,
                }
            }
        } else {
            while pos < line.len() && line[pos] != b' ' {
                if line[pos].is_ascii_whitespace() || line[pos] == b'"' {
                    return None;
                }
                pos += 1;
            }
        }
        if pos == start {
            return None;
        }
        values.push(&line[start..pos]);
        if pos < line.len() {
            pos += 1;
            // The serializer uses exactly one separating space, with no trailing space.
            if pos == line.len() || line[pos] == b' ' {
                return None;
            }
        }
    }
    (values.len() == count).then_some((header, values))
}

fn finite_number(token: &[u8]) -> Option<f64> {
    if !matches!(token.first(), Some(b'f' | b'n')) {
        return None;
    }
    let number = &token[1..];
    let mut pos = usize::from(matches!(number.first(), Some(b'+' | b'-')));
    let mut digits = 0;
    while number.get(pos).is_some_and(u8::is_ascii_digit) {
        digits += 1;
        pos += 1;
    }
    if number.get(pos) == Some(&b'.') {
        pos += 1;
        while number.get(pos).is_some_and(u8::is_ascii_digit) {
            digits += 1;
            pos += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(number.get(pos), Some(b'e' | b'E')) {
        pos += 1;
        if matches!(number.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let start = pos;
        while number.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        if pos == start {
            return None;
        }
    }
    if pos != number.len() {
        return None;
    }
    let value: f64 = std::str::from_utf8(number).ok()?.parse().ok()?;
    value.is_finite().then_some(value)
}

fn numeric_equal(a: &[u8], b: &[u8], absolute: f64, relative: f64) -> bool {
    if a.first() != b.first() {
        return false;
    }
    let (Some(a), Some(b)) = (finite_number(a), finite_number(b)) else {
        return false;
    };
    if a == 0.0 && b == 0.0 {
        return a.is_sign_negative() == b.is_sign_negative();
    }
    let difference = (a - b).abs();
    if difference <= absolute {
        return true;
    }
    if relative == 0.0 {
        return false;
    }
    // Divide first, so opposite large values and large tolerances cannot overflow.
    let scale = a.abs().max(b.abs());
    let scaled_difference = if a.is_sign_negative() == b.is_sign_negative() {
        difference / scale
    } else {
        a.abs() / scale + b.abs() / scale
    };
    scaled_difference <= relative
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equal(a: &str, b: &str) -> bool {
        stdout_equal(a.as_bytes(), b.as_bytes(), Comparison::approximate())
    }

    #[test]
    fn opt_in_and_tolerances() {
        assert!(!stdout_equal(
            b"x#1 f1",
            b"x#1 f1.0000000000001",
            Comparison::default()
        ));
        assert!(equal("x#1 f1", "x#1 f1.0000000000001"));
        assert!(equal("x#1 n0", "x#1 n0.0000000000001"));
        assert!(!equal("x#1 f1", "x#1 f1.00001"));
        for tolerance in [-1.0, f64::INFINITY, f64::NAN] {
            let policy = Comparison::Approximate {
                absolute: tolerance,
                relative: 0.0,
            };
            assert!(policy.validate().is_err());
            assert!(!line_equal(b"x#1 f1", b"x#1 f1.1", policy));
        }
    }

    #[test]
    fn large_values_and_exponents() {
        assert!(equal("x#1 f1e308", "x#1 f1.0000000000001e308"));
        assert!(equal("x#1 f1e+0", "x#1 f1.0E0"));
        assert!(!equal("x#1 f1e308", "x#1 f-1e308"));
        assert!(line_equal(
            b"x#1 f1e308",
            b"x#1 f-1e308",
            Comparison::Approximate {
                absolute: 0.0,
                relative: 2.0
            }
        ));
        assert!(!line_equal(
            b"x#1 f1e308",
            b"x#1 f-1e308",
            Comparison::Approximate {
                absolute: f64::MAX,
                relative: 1.5
            }
        ));
    }

    #[test]
    fn zero_tolerances_require_equal_float_values() {
        let policy = Comparison::Approximate {
            absolute: 0.0,
            relative: 0.0,
        };
        assert!(line_equal(b"x#1 n1e0", b"x#1 n1.0", policy));
        assert!(!line_equal(b"x#1 n1", b"x#1 n1.0000000000000002", policy));
        assert!(!line_equal(b"x#1 f0", b"x#1 f5e-324", policy));
        assert!(!line_equal(b"x#1 f-0", b"x#1 f0", policy));
    }

    #[test]
    fn structure_and_integers_stay_exact() {
        for (a, b) in [
            ("x#1 i9007199254740992", "x#1 i9007199254740993"),
            ("x#1 f1", "x#1 n1"),
            ("x#1 f1", "y#1 f1.0"),
            ("x#1 f1", "x#2 f1.0"),
            ("x#1 f1\n", "x#1 f1.0"),
            ("x#1 f1", "x#1  f1.0"),
            ("ordinary f1", "ordinary f1.0"),
            ("x#1 af1", "x#1 af1.0"),
        ] {
            assert!(!equal(a, b), "{a:?} versus {b:?}");
        }
    }

    #[test]
    fn strings_and_escapes_stay_exact() {
        assert!(!equal("x#1 \"f1\"", "x#1 \"f1.0\""));
        assert!(equal("x#2 \"a\\\" f1\" f2", "x#2 \"a\\\" f1\" f2.0"));
        assert!(!equal("x#2 \"a\\\" f1\" f2", "x#2 \"a\\\" f1.0\" f2"));
        assert!(!equal("x#1 \"f1", "x#1 \"f1.0"));
    }

    #[test]
    fn exceptional_and_malformed_numbers_stay_exact() {
        for (a, b) in [
            ("fnan", "fNaN"),
            ("finf", "f+inf"),
            ("finf", "f-inf"),
            ("f0", "f-0"),
            ("n0.0", "n-0e0"),
            ("f1e", "f1e0"),
            ("f1e+", "f1e0"),
            ("f1x", "f1.0x"),
            ("f1e999", "f2e999"),
        ] {
            assert!(
                !equal(&format!("x#1 {a}"), &format!("x#1 {b}")),
                "{a} versus {b}"
            );
        }
        assert!(equal("x#1 fnan", "x#1 fnan"));
        assert!(equal("x#1 f-0", "x#1 f-0.0"));
    }
}
