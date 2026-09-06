//! Compatibility profiles and the stock interpreter that judges each one.

use std::fmt;

/// A Lua dialect the generator writes for and oximond runs as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Profile {
    Lua51,
    Lua52,
    Lua53,
    Lua54,
    Lua55,
    LuaJit21,
}

impl Profile {
    pub const ALL: [Self; 6] = [
        Self::Lua51,
        Self::Lua52,
        Self::Lua53,
        Self::Lua54,
        Self::Lua55,
        Self::LuaJit21,
    ];

    /// The `--compat` value oximond accepts.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lua51 => "lua51",
            Self::Lua52 => "lua52",
            Self::Lua53 => "lua53",
            Self::Lua54 => "lua54",
            Self::Lua55 => "lua55",
            Self::LuaJit21 => "luajit21",
        }
    }

    /// Parses a `--compat` value.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| profile.name() == text)
    }

    /// The stock interpreter command that matches this profile, looked up on
    /// `PATH`. The runner checks its `-v` banner against
    /// [`Profile::oracle_banner`] and records a case as inconclusive when it
    /// is missing or wrong: agreement between oximond's own modes never
    /// counts as a pass.
    #[must_use]
    pub const fn default_oracle(self) -> &'static str {
        match self {
            Self::Lua51 => "lua5.1",
            Self::Lua52 => "lua5.2",
            Self::Lua53 => "lua5.3",
            Self::Lua54 => "lua5.4",
            Self::Lua55 => "lua5.5",
            Self::LuaJit21 => "luajit",
        }
    }

    /// The version banner prefix the oracle must print for `-v`, so a
    /// `lua5.4` that is really 5.3 is rejected instead of silently used.
    #[must_use]
    pub const fn oracle_banner(self) -> &'static str {
        match self {
            Self::Lua51 => "Lua 5.1",
            Self::Lua52 => "Lua 5.2",
            Self::Lua53 => "Lua 5.3",
            Self::Lua54 => "Lua 5.4",
            Self::Lua55 => "Lua 5.5",
            Self::LuaJit21 => "LuaJIT 2.1",
        }
    }

    /// Language features the generator may use for this profile.
    #[must_use]
    pub const fn features(self) -> Dialect {
        let modern = !matches!(self, Self::Lua51 | Self::LuaJit21);
        let integers = matches!(self, Self::Lua53 | Self::Lua54 | Self::Lua55);
        let jit = matches!(self, Self::LuaJit21);
        Dialect {
            lua55: matches!(self, Self::Lua55),
            integers,
            goto: modern || jit,
            hex_escapes: modern || jit,
            unicode_escapes: integers || jit,
            env_local: modern,
            setfenv: !modern,
            len_metamethod: modern,
            yield_across_pcall: modern || jit,
            table_unpack: modern,
            rep_separator: modern || jit,
            table_pack: modern,
            table_move: integers,
            utf8: integers,
            utf8_lax: matches!(self, Self::Lua54 | Self::Lua55),
            close_and_const: matches!(self, Self::Lua54 | Self::Lua55),
            bit_library: jit,
            bit32_library: matches!(self, Self::Lua52),
            rawlen: modern,
            xpcall_arguments: modern || jit,
            mixed_eq_metamethod: integers,
            integer_for_edges: matches!(self, Self::Lua54 | Self::Lua55),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What one profile's language and library offer the generator.
#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one flag per language feature"
)]
pub struct Dialect {
    /// Lexical globals, named varargs and `table.create`.
    pub lua55: bool,
    /// Integer subtype, `//`, bitwise operators, `math.type`.
    pub integers: bool,
    pub goto: bool,
    /// `\xXX` and `\z` string escapes.
    pub hex_escapes: bool,
    /// `\u{XXX}` string escapes.
    pub unicode_escapes: bool,
    /// `local _ENV = ...`.
    pub env_local: bool,
    /// `setfenv` and `getfenv`.
    pub setfenv: bool,
    /// `__len` on tables.
    pub len_metamethod: bool,
    pub yield_across_pcall: bool,
    /// `table.unpack` rather than `unpack`.
    pub table_unpack: bool,
    /// `string.rep(s, n, sep)`.
    pub rep_separator: bool,
    pub table_pack: bool,
    pub table_move: bool,
    pub utf8: bool,
    /// UTF-8 decoding accepts a lax argument and supports extended codes.
    pub utf8_lax: bool,
    /// `<close>` and `<const>` locals.
    pub close_and_const: bool,
    /// `LuaJIT`'s `bit` module.
    pub bit_library: bool,
    /// Lua 5.2's `bit32` module.
    pub bit32_library: bool,
    pub rawlen: bool,
    /// `xpcall(f, handler, ...)` passes the extra arguments to `f`.
    pub xpcall_arguments: bool,
    /// `__eq` runs for two tables whose metatables differ (5.3 and later).
    pub mixed_eq_metamethod: bool,
    /// Integer loops at `math.maxinteger` stop without overflowing (5.4).
    pub integer_for_edges: bool,
}
