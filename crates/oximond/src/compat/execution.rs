//! Execution engine selection for Oximond hosts.
//!
//! Execution mode is independent of Lua compatibility. A host can request the
//! same compatibility profile from the bytecode interpreter or the tracing
//! engine. [`TracingOptLevel`] sets how much compile time the tracing engine
//! may spend on a trace, and [`TracingOptFlags`] names the optimizations a
//! level turns on.

use core::fmt;
use core::str::FromStr;

/// The engine policy used to execute Lua bytecode.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionMode {
    /// Run every chunk in the portable Rust bytecode interpreter.
    #[default]
    Interpreter,
    /// Start in the interpreter and let the trace engine compile hot loops.
    /// Unsupported operations and guard exits resume in the interpreter.
    Tracing,
}

impl ExecutionMode {
    /// Every canonical mode, in CLI display order.
    pub const ALL: [Self; 2] = [Self::Interpreter, Self::Tracing];

    /// The canonical selector accepted by [`FromStr`].
    #[must_use]
    pub const fn selector(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Tracing => "tracing",
        }
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.selector())
    }
}

/// An invalid execution mode selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseExecutionModeError {
    value: String,
}

impl ParseExecutionModeError {
    /// The selector that could not be parsed.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseExecutionModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown execution mode `{}`", self.value)
    }
}

impl std::error::Error for ParseExecutionModeError {}

impl FromStr for ExecutionMode {
    type Err = ParseExecutionModeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "interpreter" | "interp" | "bytecode" => Ok(Self::Interpreter),
            "tracing" => Ok(Self::Tracing),
            _ => Err(ParseExecutionModeError {
                value: value.to_owned(),
            }),
        }
    }
}

/// How much compile time the tracing engine may spend to make a trace faster.
///
/// Level 0 holds every trace optimization Oximond has today: the baseline and
/// optimized tiers of the trace optimizer and Cranelift. Levels 1 and 2 are
/// for advanced optimizations of frequently run code, which trade longer
/// compile time for less run time. Both currently add
/// [`TracingOptFlags::STRING_CONSTANTS`] and nothing else. The interpreter
/// ignores the level.
///
/// Each level stands for a set of [`TracingOptFlags`], see [`Self::flags`].
/// A host can start from a level's flags and switch single optimizations on
/// or off.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TracingOptLevel {
    /// The current trace optimizations.
    #[default]
    O0,
    /// Level 0 plus advanced optimizations of hot traces.
    O1,
    /// Level 1 plus the most expensive optimizations of hot traces.
    O2,
}

impl TracingOptLevel {
    /// Every level, lowest first.
    pub const ALL: [Self; 3] = [Self::O0, Self::O1, Self::O2];

    /// The canonical selector accepted by [`FromStr`].
    #[must_use]
    pub const fn selector(self) -> &'static str {
        match self {
            Self::O0 => "0",
            Self::O1 => "1",
            Self::O2 => "2",
        }
    }

    /// The optimizations this level turns on.
    #[must_use]
    pub const fn flags(self) -> TracingOptFlags {
        match self {
            Self::O0 => TracingOptFlags::LEVEL_0,
            Self::O1 | Self::O2 => TracingOptFlags::LEVEL_1,
        }
    }

    /// The level as a number.
    #[must_use]
    pub const fn number(self) -> u8 {
        match self {
            Self::O0 => 0,
            Self::O1 => 1,
            Self::O2 => 2,
        }
    }
}

/// A set of trace optimizations, one bit each.
///
/// A flag allows an optimization; the trace's compile tier still decides
/// whether it runs. Code that has only just become hot compiles at the
/// baseline tier, which skips the expensive passes whatever the flags say.
/// The passes every tier needs (copy elimination, redundant guard
/// elimination, dead code elimination) have no flag.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TracingOptFlags(u64);

impl TracingOptFlags {
    /// No optional optimization.
    pub const NONE: Self = Self(0);
    /// Move branch guards on loop-invariant conditions into the preheader.
    pub const HOIST_GUARDS: Self = Self(1 << 0);
    /// Forward stores to loads, merge equal loads, and hoist invariant table
    /// reads.
    pub const MEMORY: Self = Self(1 << 1);
    /// Check the key range of a counted loop's array accesses once where the
    /// loop starts, and run a version of the loop without the per-access
    /// bounds checks when it holds.
    pub const RANGE_PROOFS: Self = Self(1 << 2);
    /// Prepare the constant patterns of `string.find`, `string.match`,
    /// `string.gmatch` and `string.gsub` and the constant format strings of
    /// `string.format` once when the trace compiles.
    pub const STRING_CONSTANTS: Self = Self(1 << 3);
    /// The optimizations of [`TracingOptLevel::O0`].
    pub const LEVEL_0: Self = Self(Self::HOIST_GUARDS.0 | Self::MEMORY.0 | Self::RANGE_PROOFS.0);
    /// The optimizations of [`TracingOptLevel::O1`] and [`TracingOptLevel::O2`].
    pub const LEVEL_1: Self = Self(Self::LEVEL_0.0 | Self::STRING_CONSTANTS.0);

    /// The flags with the given bits.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The bits of the set.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether every flag of `other` is in the set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The set with the flags of `other` added.
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The set with the flags of `other` removed.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl Default for TracingOptFlags {
    /// The flags of the default level.
    fn default() -> Self {
        TracingOptLevel::default().flags()
    }
}

impl fmt::Debug for TracingOptFlags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "TracingOptFlags({:#x})", self.0)
    }
}

impl fmt::Display for TracingOptLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.selector())
    }
}

/// An invalid tracing optimization level selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseTracingOptLevelError {
    value: String,
}

impl ParseTracingOptLevelError {
    /// The selector that could not be parsed.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseTracingOptLevelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown tracing optimization level `{}`",
            self.value
        )
    }
}

impl std::error::Error for ParseTracingOptLevelError {}

impl FromStr for TracingOptLevel {
    type Err = ParseTracingOptLevelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "0" | "o0" => Ok(Self::O0),
            "1" | "o1" => Ok(Self::O1),
            "2" | "o2" => Ok(Self::O2),
            _ => Err(ParseTracingOptLevelError {
                value: value.to_owned(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionMode, TracingOptFlags, TracingOptLevel};

    #[test]
    fn opt_flags_switch_single_optimizations() {
        let flags = TracingOptLevel::O0.flags();
        assert_eq!(flags, TracingOptFlags::default());
        assert!(flags.contains(TracingOptFlags::HOIST_GUARDS));
        let without = flags.without(TracingOptFlags::MEMORY);
        assert!(!without.contains(TracingOptFlags::MEMORY));
        assert!(without.contains(TracingOptFlags::HOIST_GUARDS));
        assert_eq!(without.with(TracingOptFlags::MEMORY), flags);
        assert_eq!(TracingOptFlags::from_bits(flags.bits()), flags);
    }

    #[test]
    fn level_one_adds_string_constants() {
        assert!(
            !TracingOptLevel::O0
                .flags()
                .contains(TracingOptFlags::STRING_CONSTANTS)
        );
        for level in [TracingOptLevel::O1, TracingOptLevel::O2] {
            assert_eq!(level.flags(), TracingOptFlags::LEVEL_1);
            assert!(level.flags().contains(TracingOptFlags::LEVEL_0));
            assert!(level.flags().contains(TracingOptFlags::STRING_CONSTANTS));
        }
        let alone = TracingOptFlags::from_bits(0x8);
        assert!(alone.contains(TracingOptFlags::STRING_CONSTANTS));
        assert!(!alone.contains(TracingOptFlags::MEMORY));
    }

    #[test]
    fn opt_level_selectors_round_trip() {
        for level in TracingOptLevel::ALL {
            assert_eq!(level.selector().parse(), Ok(level));
            assert_eq!(level.to_string(), level.selector());
        }
        assert_eq!("O2".parse(), Ok(TracingOptLevel::O2));
        assert_eq!(TracingOptLevel::default(), TracingOptLevel::O0);
    }

    #[test]
    fn invalid_opt_levels_preserve_the_input() {
        let error = "3".parse::<TracingOptLevel>().unwrap_err();
        assert_eq!(error.value(), "3");
        assert_eq!(error.to_string(), "unknown tracing optimization level `3`");
    }

    #[test]
    fn canonical_selectors_round_trip() {
        for mode in ExecutionMode::ALL {
            assert_eq!(mode.selector().parse(), Ok(mode));
            assert_eq!(mode.to_string(), mode.selector());
        }
    }

    #[test]
    fn aliases_are_case_insensitive() {
        assert_eq!("BYTECODE".parse(), Ok(ExecutionMode::Interpreter));
        assert_eq!("Tracing".parse(), Ok(ExecutionMode::Tracing));
    }

    #[test]
    fn invalid_selectors_preserve_the_input() {
        let error = "baseline-jit".parse::<ExecutionMode>().unwrap_err();
        assert_eq!(error.value(), "baseline-jit");
        assert_eq!(error.to_string(), "unknown execution mode `baseline-jit`");
    }

    #[test]
    fn the_removed_strict_jit_mode_is_rejected() {
        assert!("jit".parse::<ExecutionMode>().is_err());
        assert!("native".parse::<ExecutionMode>().is_err());
    }

    #[test]
    fn the_old_tiered_selectors_are_rejected() {
        for selector in ["tiered", "mixed", "hybrid", "auto"] {
            assert!(selector.parse::<ExecutionMode>().is_err(), "{selector}");
        }
    }
}
