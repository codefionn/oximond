#![deny(unsafe_code)]

//! Runtime compatibility identities and the semantic groups selected by them.
//!
//! This crate models a selected Lua target. It does not make a parser or VM
//! support a feature merely because that feature exists in the target release.

use core::fmt;
use core::str::FromStr;

mod execution;
mod float_arith;

pub use execution::{
    ExecutionMode, ParseExecutionModeError, ParseTracingOptLevelError, TracingOptFlags,
    TracingOptLevel,
};
pub use float_arith::{
    FloatArith, FloatModulo, fma_fmod, fma_fmod_checked, fmod53_helper, fmod54_helper,
};

/// A canonical Oximond compatibility profile.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompatibilityProfile {
    /// Stock Lua 5.1.
    Lua51,
    /// Stock Lua 5.2.
    Lua52,
    /// Stock Lua 5.3.
    Lua53,
    /// Stock Lua 5.4.
    Lua54,
    /// Stock Lua 5.5.1.
    Lua55,
    /// `LuaJIT` 2.1, with Lua 5.1 language identity.
    #[default]
    LuaJit21,
}

impl CompatibilityProfile {
    /// Every canonical profile, in CLI display order.
    pub const ALL: [Self; 6] = [
        Self::Lua51,
        Self::Lua52,
        Self::Lua53,
        Self::Lua54,
        Self::Lua55,
        Self::LuaJit21,
    ];

    /// The canonical selector accepted by the CLI and [`FromStr`].
    #[must_use]
    pub const fn selector(self) -> &'static str {
        match self {
            Self::Lua51 => "lua51",
            Self::Lua52 => "lua52",
            Self::Lua53 => "lua53",
            Self::Lua54 => "lua54",
            Self::Lua55 => "lua55",
            Self::LuaJit21 => "luajit21",
        }
    }

    /// Typed identity visible to hosts and the standard library.
    #[must_use]
    pub const fn identity(self) -> RuntimeIdentity {
        RuntimeIdentity::new(self)
    }

    /// Semantic target groups associated with this profile.
    #[must_use]
    pub const fn capabilities(self) -> Capabilities {
        Capabilities::for_profile(self)
    }

    /// Whether this profile follows Lua 5.1 semantics (stock 5.1 or `LuaJIT` 2.1).
    #[must_use]
    pub const fn is_lua51_family(self) -> bool {
        matches!(self, Self::Lua51 | Self::LuaJit21)
    }

    /// Whether this profile is Lua 5.3, 5.4, or 5.5 (integer subtype, UTF-8 escapes).
    #[must_use]
    pub const fn is_lua53_or_later(self) -> bool {
        matches!(self, Self::Lua53 | Self::Lua54 | Self::Lua55)
    }
}

impl fmt::Display for CompatibilityProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.selector())
    }
}

/// An invalid compatibility selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseCompatibilityProfileError {
    value: String,
}

impl ParseCompatibilityProfileError {
    /// The selector that could not be parsed.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseCompatibilityProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown compatibility profile `{}`", self.value)
    }
}

impl std::error::Error for ParseCompatibilityProfileError {}

impl FromStr for CompatibilityProfile {
    type Err = ParseCompatibilityProfileError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.to_ascii_lowercase();
        match normalized.as_str() {
            "lua51" | "lua5.1" | "lua-5.1" => Ok(Self::Lua51),
            "lua52" | "lua5.2" | "lua-5.2" => Ok(Self::Lua52),
            "lua53" | "lua5.3" | "lua-5.3" => Ok(Self::Lua53),
            "lua54" | "lua5.4" | "lua-5.4" => Ok(Self::Lua54),
            "lua55" | "lua5.5" | "lua-5.5" | "lua5.5.1" | "lua-5.5.1" => Ok(Self::Lua55),
            "luajit21" | "luajit2.1" | "luajit-2.1" => Ok(Self::LuaJit21),
            _ => Err(ParseCompatibilityProfileError {
                value: value.to_owned(),
            }),
        }
    }
}

/// The language version named by a profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LuaVersion {
    Lua51,
    Lua52,
    Lua53,
    Lua54,
    Lua55,
}

impl LuaVersion {
    /// The stock value of Lua's `_VERSION` global.
    #[must_use]
    pub const fn version_string(self) -> &'static str {
        match self {
            Self::Lua51 => "Lua 5.1",
            Self::Lua52 => "Lua 5.2",
            Self::Lua53 => "Lua 5.3",
            Self::Lua54 => "Lua 5.4",
            Self::Lua55 => "Lua 5.5",
        }
    }
}

/// The implementation family selected for execution.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeImplementation {
    Lua,
    LuaJit21,
}

/// Typed runtime identity for a selected profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeIdentity {
    profile: CompatibilityProfile,
    language: LuaVersion,
    implementation: RuntimeImplementation,
}

impl RuntimeIdentity {
    const fn new(profile: CompatibilityProfile) -> Self {
        match profile {
            CompatibilityProfile::Lua51 => Self {
                profile,
                language: LuaVersion::Lua51,
                implementation: RuntimeImplementation::Lua,
            },
            CompatibilityProfile::Lua52 => Self {
                profile,
                language: LuaVersion::Lua52,
                implementation: RuntimeImplementation::Lua,
            },
            CompatibilityProfile::Lua53 => Self {
                profile,
                language: LuaVersion::Lua53,
                implementation: RuntimeImplementation::Lua,
            },
            CompatibilityProfile::Lua54 => Self {
                profile,
                language: LuaVersion::Lua54,
                implementation: RuntimeImplementation::Lua,
            },
            CompatibilityProfile::Lua55 => Self {
                profile,
                language: LuaVersion::Lua55,
                implementation: RuntimeImplementation::Lua,
            },
            CompatibilityProfile::LuaJit21 => Self {
                profile,
                language: LuaVersion::Lua51,
                implementation: RuntimeImplementation::LuaJit21,
            },
        }
    }

    #[must_use]
    pub const fn profile(self) -> CompatibilityProfile {
        self.profile
    }

    #[must_use]
    pub const fn language(self) -> LuaVersion {
        self.language
    }

    #[must_use]
    pub const fn implementation(self) -> RuntimeImplementation {
        self.implementation
    }

    /// The value installed in `_VERSION` by a stock runtime.
    #[must_use]
    pub const fn version_string(self) -> &'static str {
        self.language.version_string()
    }
}

/// Parser-relevant language choices. These describe the selected target,
/// rather than claiming that Oximond has implemented every listed grammar form.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ParserCapabilities {
    pub language: LuaVersion,
    /// Accepts `;` as an empty statement, including at the start of a block.
    pub empty_statements: bool,
    /// Rejects a parenthesized call whose `(` starts on a later source line.
    pub reject_ambiguous_call_newline: bool,
    pub goto_labels: bool,
    pub bitwise_operators: bool,
    pub integer_division: bool,
    pub local_attributes: bool,
}

/// The target's numeric representation model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NumberModel {
    Float64,
    IntegerAndFloat64,
}

/// Library API families introduced by a language generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct LibraryCapabilities {
    pub package_searchers: bool,
    pub utf8_library: bool,
    pub bit32_library: bool,
    pub table_move: bool,
    pub coroutine_close: bool,
}

/// Garbage collector controls available to the target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GarbageCollectionCapabilities {
    pub generational_mode: bool,
    pub incremental_parameters: bool,
}

/// Coroutine behavior exposed by the target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CoroutineCapabilities {
    pub yieldable_pcall: bool,
    pub close: bool,
}

/// Debug-library API families.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DebugCapabilities {
    pub getuservalue: bool,
    pub upvalue_id: bool,
    pub multiple_uservalues: bool,
}

/// Binary chunk compatibility properties.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BytecodeCapabilities {
    pub stock_format: LuaVersion,
    pub integer_constants: bool,
    pub luajit_format: bool,
}

/// Command-line conventions offered by a target runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CliCapabilities {
    pub environment_initialization: bool,
    pub execute_option: bool,
    pub require_option: bool,
}

/// All compatibility groups computed from a selected profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Capabilities {
    pub parser: ParserCapabilities,
    pub numbers: NumberModel,
    pub libraries: LibraryCapabilities,
    pub gc: GarbageCollectionCapabilities,
    pub coroutines: CoroutineCapabilities,
    pub debug: DebugCapabilities,
    pub bytecode: BytecodeCapabilities,
    pub cli: CliCapabilities,
}

impl Capabilities {
    const fn for_profile(profile: CompatibilityProfile) -> Self {
        let identity = profile.identity();
        let language = identity.language();
        let at_least_lua52 = !matches!(language, LuaVersion::Lua51);
        let at_least_lua53 = matches!(
            language,
            LuaVersion::Lua53 | LuaVersion::Lua54 | LuaVersion::Lua55
        );
        let at_least_lua54 = matches!(language, LuaVersion::Lua54 | LuaVersion::Lua55);
        Self {
            parser: ParserCapabilities {
                language,
                empty_statements: !matches!(profile, CompatibilityProfile::Lua51),
                reject_ambiguous_call_newline: matches!(language, LuaVersion::Lua51),
                goto_labels: at_least_lua52,
                bitwise_operators: at_least_lua53,
                integer_division: at_least_lua53,
                local_attributes: at_least_lua54,
            },
            numbers: if at_least_lua53 {
                NumberModel::IntegerAndFloat64
            } else {
                NumberModel::Float64
            },
            libraries: LibraryCapabilities {
                package_searchers: at_least_lua52,
                utf8_library: at_least_lua53,
                bit32_library: matches!(language, LuaVersion::Lua52),
                table_move: at_least_lua53,
                coroutine_close: at_least_lua54,
            },
            gc: GarbageCollectionCapabilities {
                generational_mode: at_least_lua54,
                incremental_parameters: at_least_lua52,
            },
            coroutines: CoroutineCapabilities {
                yieldable_pcall: at_least_lua52,
                close: at_least_lua54,
            },
            debug: DebugCapabilities {
                getuservalue: at_least_lua52,
                upvalue_id: at_least_lua52,
                multiple_uservalues: at_least_lua54,
            },
            bytecode: BytecodeCapabilities {
                stock_format: language,
                integer_constants: at_least_lua53,
                luajit_format: matches!(identity.implementation(), RuntimeImplementation::LuaJit21),
            },
            cli: CliCapabilities {
                environment_initialization: at_least_lua52,
                execute_option: true,
                require_option: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompatibilityProfile, LuaVersion, NumberModel, RuntimeImplementation};

    #[test]
    fn selectors_accept_canonical_and_documented_aliases() {
        assert_eq!("lua51".parse(), Ok(CompatibilityProfile::Lua51));
        assert_eq!("lua5.2".parse(), Ok(CompatibilityProfile::Lua52));
        assert_eq!("lua-5.3".parse(), Ok(CompatibilityProfile::Lua53));
        assert_eq!("LUA5.4".parse(), Ok(CompatibilityProfile::Lua54));
        assert_eq!("lua-5.5.1".parse(), Ok(CompatibilityProfile::Lua55));
        assert_eq!("luajit-2.1".parse(), Ok(CompatibilityProfile::LuaJit21));
        assert!("lua56".parse::<CompatibilityProfile>().is_err());
    }

    #[test]
    fn luajit_uses_lua_51_language_identity() {
        let identity = CompatibilityProfile::LuaJit21.identity();
        assert_eq!(identity.language(), LuaVersion::Lua51);
        assert_eq!(identity.implementation(), RuntimeImplementation::LuaJit21);
        assert_eq!(identity.version_string(), "Lua 5.1");
    }

    #[test]
    fn capabilities_follow_the_selected_generation() {
        let lua51 = CompatibilityProfile::Lua51.capabilities();
        assert!(!lua51.parser.empty_statements);
        assert!(lua51.parser.reject_ambiguous_call_newline);

        let lua55 = CompatibilityProfile::Lua55.capabilities();
        assert!(lua55.parser.empty_statements);
        assert!(!lua55.parser.reject_ambiguous_call_newline);
        assert!(lua55.parser.goto_labels);
        assert!(lua55.parser.local_attributes);
        assert_eq!(lua55.numbers, NumberModel::IntegerAndFloat64);
        assert!(lua55.coroutines.close);

        let luajit = CompatibilityProfile::LuaJit21.capabilities();
        assert!(luajit.parser.empty_statements);
        assert!(luajit.parser.reject_ambiguous_call_newline);
        assert!(!luajit.parser.goto_labels);
        assert_eq!(luajit.numbers, NumberModel::Float64);
        assert!(luajit.bytecode.luajit_format);
    }
}
