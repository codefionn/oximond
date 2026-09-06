//! SplitMix64, the same algorithm as `oximond-difftest`'s `rng.rs`, copied
//! because this crate must not depend on the oximond crates.

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

pub struct Rng {
    state: u64,
}

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Output number `position` of the stream for `seed`.
    #[must_use]
    pub const fn nth(seed: u64, position: u64) -> u64 {
        Self::new(seed.wrapping_add(position.wrapping_mul(GAMMA))).next_u64()
    }

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
        assert!(n > 0);
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}
