//! Wall-clock and pseudo-random number state.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::runtime::Runtime;

impl Runtime {
    /// Whether `math.random` draws from `LuaJIT`'s generator.
    #[must_use]
    pub const fn random_is_luajit(&self) -> bool {
        self.random_luajit
    }

    /// Whether `math.random` draws from the xoshiro generator of Lua 5.4.
    #[must_use]
    pub const fn random_is_modern(&self) -> bool {
        self.random_modern
    }

    /// Selects `LuaJIT`'s combined Tausworthe generator.
    pub fn set_random_luajit(&mut self, enabled: bool) {
        self.random_luajit = enabled;
    }

    /// Seeds `LuaJIT`'s generator from a floating-point seed.
    pub fn random_seed_luajit(&mut self, mut seed: f64) {
        let mut shifts = 0x1109_0601_u32;
        for state in &mut self.random_luajit_state {
            let minimum = 1_u64 << (shifts & 255);
            shifts >>= 8;
            seed = seed * std::f64::consts::PI + std::f64::consts::E;
            let mut bits = seed.to_bits();
            if bits < minimum {
                bits += minimum;
            }
            *state = bits;
        }
        for _ in 0..10 {
            self.random_bits_luajit();
        }
    }

    /// `LuaJIT`'s `randomseed()` chooses a new seed when called without arguments.
    pub fn random_seed_luajit_auto(&mut self) {
        #[cfg(unix)]
        {
            use std::io::Read;
            let mut bytes = [0_u8; 32];
            if std::fs::File::open("/dev/urandom")
                .and_then(|mut file| file.read_exact(&mut bytes))
                .is_ok()
            {
                for (state, chunk) in self
                    .random_luajit_state
                    .iter_mut()
                    .zip(bytes.as_chunks::<8>().0)
                {
                    *state = u64::from_ne_bytes(*chunk);
                }
                for (state, shift) in self.random_luajit_state.iter_mut().zip([1, 6, 9, 17]) {
                    let minimum = 1_u64 << shift;
                    if *state < minimum {
                        *state += minimum;
                    }
                }
                self.random_bits_luajit();
                return;
            }
        }
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |time| time.as_secs_f64());
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        let seed = 0.0;
        self.random_seed_luajit(seed);
    }

    /// Selects Lua 5.4's xoshiro256** generator for `math.random`.
    pub fn set_random_modern(&mut self, enabled: bool) {
        self.random_modern = enabled;
        if enabled {
            // The library starts with an automatically chosen seed.
            let _ = self.random_seed_modern_auto();
        }
    }

    /// Seeds the Lua 5.4 generator with the two integer seeds it returns.
    pub fn random_seed_modern(&mut self, first: i64, second: i64) {
        self.random_modern_state = [first.cast_unsigned(), 0xff, second.cast_unsigned(), 0];
        for _ in 0..16 {
            self.random_bits_modern();
        }
    }

    /// Chooses and returns a fresh pair of seeds for Lua 5.4 `randomseed()`.
    #[must_use]
    pub fn random_seed_modern_auto(&mut self) -> (i64, i64) {
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        let first = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |time| time.as_secs())
            .cast_signed();
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        let first = 0;
        let second = u64::try_from(std::ptr::from_ref(self).addr())
            .unwrap_or(0)
            .cast_signed();
        self.random_seed_modern(first, second);
        (first, second)
    }

    /// Seeds the runtime-owned deterministic random-number generator.
    ///
    /// # Errors
    ///
    /// Returns an error unless `seed` is finite.
    pub fn random_seed(&mut self, seed: f64) -> RuntimeResult<()> {
        if !seed.is_finite() {
            return Err(RuntimeError::message("random seed must be finite"));
        }
        self.random_state = seed.to_bits() ^ 0xd1b5_4a32_d192_ed03;
        if self.random_state == 0 {
            self.random_state = 1;
        }
        Ok(())
    }

    /// Returns the next deterministic pseudo-random 64-bit value.
    #[must_use]
    pub fn random_bits(&mut self) -> u64 {
        if self.random_luajit {
            return self.random_bits_luajit();
        }
        if self.random_modern {
            return self.random_bits_modern();
        }
        let mut state = self.random_state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.random_state = state;
        state
    }

    fn random_bits_luajit(&mut self) -> u64 {
        let mut result = 0;
        for (state, (k, q, shift)) in self.random_luajit_state.iter_mut().zip([
            (63, 31, 18),
            (58, 19, 28),
            (55, 24, 7),
            (47, 21, 8),
        ]) {
            let old = *state;
            let next =
                (((old << q) ^ old) >> (k - shift)) ^ ((old & (u64::MAX << (64 - k))) << shift);
            *state = next;
            result ^= next;
        }
        result
    }

    fn random_bits_modern(&mut self) -> u64 {
        let [mut a, mut b, mut c, mut d] = self.random_modern_state;
        let result = b.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let shifted = b.wrapping_shl(17);
        c ^= a;
        d ^= b;
        b ^= c;
        a ^= d;
        c ^= shifted;
        d = d.rotate_left(45);
        self.random_modern_state = [a, b, c, d];
        result
    }

    /// Returns the next deterministic pseudo-random unit interval value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Lua 5.4 uses 53 random bits, all exactly representable.
    pub fn random_unit(&mut self) -> f64 {
        if self.random_luajit {
            let bits = (self.random_bits_luajit() & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
            return f64::from_bits(bits) - 1.0;
        }
        if self.random_modern {
            return (self.random_bits_modern() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
        }
        let bytes = self.random_bits().to_be_bytes();
        let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        f64::from(high) / 4_294_967_296.0
    }

    /// Returns process CPU seconds on Linux. Other targets retain a monotonic
    /// elapsed-time fallback because the standard library has no CPU clock.
    #[must_use]
    pub fn process_cpu_seconds(&self) -> f64 {
        #[cfg(target_os = "linux")]
        if let Some(seconds) = linux_process_cpu_seconds() {
            return seconds;
        }
        self.monotonic_seconds()
    }

    /// Returns elapsed wall-clock seconds since runtime creation.
    /// Bare WebAssembly has no built-in clock and returns NaN.
    #[must_use]
    pub fn monotonic_seconds(&self) -> f64 {
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        {
            self.started_at.elapsed().as_secs_f64()
        }
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        {
            f64::NAN
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_process_cpu_seconds() -> Option<f64> {
    use std::sync::OnceLock;

    static TICKS_PER_SECOND: OnceLock<Option<f64>> = OnceLock::new();
    let ticks_per_second = (*TICKS_PER_SECOND.get_or_init(|| {
        let output = std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .ok()?;
        std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|rate| *rate > 0.0)
    }))?;
    // The comm field is parenthesized and can itself contain spaces.
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields = stat
        .rsplit_once(") ")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    let user: f64 = fields.get(11)?.parse().ok()?;
    let system: f64 = fields.get(12)?.parse().ok()?;
    Some((user + system) / ticks_per_second)
}
