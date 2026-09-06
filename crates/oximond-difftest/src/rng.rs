//! The generator's only source of randomness.
//!
//! `SplitMix64` (Steele, Lea and Flood 2014) with fixed constants. Every choice
//! goes through [`Rng::below`], which maps 64 random bits onto `0..n` by the
//! high half of a 128-bit product, so a seed produces the same program on
//! every platform and Rust version.

/// `SplitMix64`'s state increment.
const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// Seeded `SplitMix64` stream.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A stream for `seed`. The same seed always yields the same stream.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Output number `position` (from 0) of the stream for `seed`,
    /// without producing the ones before it.
    #[must_use]
    pub const fn nth(seed: u64, position: u64) -> u64 {
        Self::new(seed.wrapping_add(position.wrapping_mul(GAMMA))).next_u64()
    }

    /// The next 64 bits.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `0..n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below needs a positive bound");
        let wide = u128::from(self.next_u64()) * u128::from(n);
        u64::try_from(wide >> 64).expect("the high half of a u64 product fits u64")
    }

    /// A value in `0..n` as `usize`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn index(&mut self, n: usize) -> usize {
        usize::try_from(self.below(n as u64)).expect("an index below a usize fits usize")
    }

    /// A value in `low..=high`.
    ///
    /// # Panics
    ///
    /// Panics if `low > high`.
    pub fn range(&mut self, low: i64, high: i64) -> i64 {
        assert!(low <= high, "Rng::range needs low <= high");
        let span = high.abs_diff(low) + 1;
        low.wrapping_add_unsigned(self.below(span))
    }

    /// True with probability `numerator / denominator`.
    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.below(denominator) < numerator
    }

    /// One element of a non-empty slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.index(items.len())]
    }

    /// The index of a weighted choice. Zero weights are never picked.
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u64 = weights.iter().map(|&weight| u64::from(weight)).sum();
        let mut roll = self.below(total);
        for (index, &weight) in weights.iter().enumerate() {
            if roll < u64::from(weight) {
                return index;
            }
            roll -= u64::from(weight);
        }
        unreachable!("a roll below the total lands on some weight")
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn the_stream_is_pinned() {
        // SplitMix64's published first output for seed 0. If this changes,
        // every saved seed in a report names a different program.
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn nth_jumps_to_the_same_output_as_stepping() {
        let mut rng = Rng::new(42);
        for position in 0..100 {
            assert_eq!(Rng::nth(42, position), rng.next_u64());
        }
    }

    #[test]
    fn bounded_choices_stay_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            assert!(rng.below(3) < 3);
            let value = rng.range(-5, 5);
            assert!((-5..=5).contains(&value));
        }
        assert_eq!(rng.range(i64::MIN, i64::MIN), i64::MIN);
    }
}
