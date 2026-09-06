//! Byte-level lexer producing tokens, including numbers, strings, and long brackets.

use crate::string::LuaString;
use crate::syntax::{
    Boundary, IncrementalLexer, LexPoll, Lexer, NumberMode, PendingBoundary, Span, SyntaxError,
    Token, TokenBoundary, TokenKind, TokenState, TriviaBoundary, TriviaState,
};

impl<'a> Lexer<'a> {
    #[allow(dead_code)]
    pub(crate) const fn new(source: &'a [u8]) -> Self {
        Self {
            source,
            position: 0,
            lua52_escapes: true,
            unicode_escapes: true,
            integer_literals: false,
            cint64_literals: false,
        }
    }
    #[allow(clippy::fn_params_excessive_bools)] // One flag per profile feature.
    pub(crate) const fn with_escapes(
        source: &'a [u8],
        lua52_escapes: bool,
        unicode_escapes: bool,
        integer_literals: bool,
        cint64_literals: bool,
    ) -> Self {
        Self {
            source,
            position: 0,
            lua52_escapes,
            unicode_escapes,
            integer_literals,
            cint64_literals,
        }
    }
    #[allow(dead_code)]
    pub(crate) fn lex(mut self) -> Result<Vec<Token>, SyntaxError> {
        let mut out = Vec::new();
        if self.source.starts_with(b"#!") {
            // Lua's standalone interpreter ignores the first physical line
            // when it starts with `#!`. Leave its line ending for
            // `skip_trivia`, which already accepts every CR/LF variant.
            while !matches!(self.byte(), None | Some(b'\n' | b'\r')) {
                self.position += 1;
            }
        }
        loop {
            self.skip_trivia()?;
            let start = self.position;
            let Some(_) = self.byte() else {
                out.push(Token {
                    kind: TokenKind::Eof,
                    span: Span { start, end: start },
                });
                return Ok(out);
            };
            let kind = self.token(start)?;
            out.push(Token {
                kind,
                span: Span {
                    start,
                    end: self.position,
                },
            });
        }
    }
    #[allow(clippy::too_many_lines)]
    pub(crate) fn token(&mut self, start: usize) -> Result<TokenKind, SyntaxError> {
        Ok(match self.byte().expect("lexer checked source position") {
            b'=' => {
                self.position += 1;
                if self.take(b'=') {
                    TokenKind::Equal
                } else {
                    TokenKind::Assign
                }
            }
            b'+' => {
                self.position += 1;
                TokenKind::Plus
            }
            b'-' => {
                self.position += 1;
                TokenKind::Minus
            }
            b'*' => {
                self.position += 1;
                TokenKind::Star
            }
            b'/' => {
                self.position += 1;
                if self.take(b'/') {
                    TokenKind::FloorDivide
                } else {
                    TokenKind::Slash
                }
            }
            b'%' => {
                self.position += 1;
                TokenKind::Percent
            }
            b'^' => {
                self.position += 1;
                TokenKind::Caret
            }
            b'~' => {
                self.position += 1;
                if self.take(b'=') {
                    TokenKind::NotEqual
                } else {
                    TokenKind::BitXor
                }
            }
            b'<' => {
                self.position += 1;
                if self.take(b'<') {
                    TokenKind::ShiftLeft
                } else if self.take(b'=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::LessThan
                }
            }
            b'>' => {
                self.position += 1;
                if self.take(b'>') {
                    TokenKind::ShiftRight
                } else if self.take(b'=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::GreaterThan
                }
            }
            b'#' => {
                self.position += 1;
                TokenKind::Hash
            }
            b'&' => {
                self.position += 1;
                TokenKind::BitAnd
            }
            b'|' => {
                self.position += 1;
                TokenKind::BitOr
            }
            b'.' if matches!(self.source.get(self.position + 1), Some(b'0'..=b'9')) => {
                self.number(start)?
            }
            b'.' => {
                self.position += 1;
                if self.take(b'.') {
                    if self.take(b'.') {
                        TokenKind::Vararg
                    } else {
                        TokenKind::Concat
                    }
                } else {
                    TokenKind::Dot
                }
            }
            b':' => {
                self.position += 1;
                if self.take(b':') {
                    TokenKind::DoubleColon
                } else {
                    TokenKind::Colon
                }
            }
            b',' => {
                self.position += 1;
                TokenKind::Comma
            }
            b';' => {
                self.position += 1;
                TokenKind::Semi
            }
            b'(' => {
                self.position += 1;
                TokenKind::LParen
            }
            b')' => {
                self.position += 1;
                TokenKind::RParen
            }
            b'[' => {
                if let Some((level, content_start)) = self.long_bracket_opening() {
                    self.long_string(start, level, content_start)?
                } else {
                    self.position += 1;
                    TokenKind::LBracket
                }
            }
            b']' => {
                self.position += 1;
                TokenKind::RBracket
            }
            b'{' => {
                self.position += 1;
                TokenKind::LBrace
            }
            b'}' => {
                self.position += 1;
                TokenKind::RBrace
            }
            b'0'..=b'9' => self.number(start)?,
            quote @ (b'\'' | b'"') => self.string(quote, start)?,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.identifier(start),
            _ => {
                return Err(Self::error(
                    Span {
                        start,
                        end: start + 1,
                    },
                    "unexpected character",
                ));
            }
        })
    }
    pub(crate) fn skip_trivia(&mut self) -> Result<(), SyntaxError> {
        loop {
            let rest = &self.source[self.position..];
            self.position += rest
                .iter()
                .position(|byte| !matches!(*byte, b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c'))
                .unwrap_or(rest.len());
            if self.source.get(self.position..self.position + 2) != Some(b"--") {
                return Ok(());
            }
            let comment_start = self.position;
            self.position += 2;
            if let Some((level, content_start)) = self.long_bracket_opening() {
                self.long_bracket(
                    comment_start,
                    level,
                    content_start,
                    "unterminated long comment",
                )?;
            } else {
                let rest = &self.source[self.position..];
                self.position += rest
                    .iter()
                    .position(|byte| matches!(*byte, b'\n' | b'\r'))
                    .unwrap_or(rest.len());
            }
        }
    }
    #[allow(clippy::too_many_lines)]
    pub(crate) fn number(&mut self, start: usize) -> Result<TokenKind, SyntaxError> {
        if self
            .source
            .get(start..start + 2)
            .is_some_and(|prefix| matches!(prefix, [b'0', b'x' | b'X']))
        {
            self.position += 2;
            let digits_start = self.position;
            while self.byte().and_then(Self::hex_digit).is_some() {
                self.position += 1;
            }
            let integer_digits = self.position - digits_start;
            let mut integer_literal = true;
            if self.byte() == Some(b'.') && self.source.get(self.position + 1) != Some(&b'.') {
                integer_literal = false;
                self.position += 1;
                while self.byte().and_then(Self::hex_digit).is_some() {
                    self.position += 1;
                }
            }
            let mantissa_end = self.position;
            let mut exponent = 0_i64;
            if matches!(self.byte(), Some(b'p' | b'P')) {
                integer_literal = false;
                let exponent_start = self.position;
                self.position += 1;
                let negative = if self.byte() == Some(b'-') {
                    self.position += 1;
                    true
                } else {
                    if self.byte() == Some(b'+') {
                        self.position += 1;
                    }
                    false
                };
                let exponent_digits = self.position;
                while let Some(byte @ b'0'..=b'9') = self.byte() {
                    self.position += 1;
                    exponent = exponent
                        .saturating_mul(10)
                        .saturating_add(i64::from(byte - b'0'));
                }
                if self.position == exponent_digits {
                    return Err(Self::error(
                        Span {
                            start: exponent_start,
                            end: self.position,
                        },
                        "invalid hexadecimal exponent",
                    ));
                }
                if negative {
                    exponent = exponent.saturating_neg();
                }
            }
            if let Some(token) =
                self.cint64_literal(start, digits_start..mantissa_end, 16, integer_literal)
            {
                return token;
            }
            if mantissa_end == digits_start
                || (mantissa_end == digits_start + 1
                    && self.source.get(digits_start) == Some(&b'.'))
                || matches!(self.byte(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
            {
                return Err(Self::error(
                    Span {
                        start,
                        end: self.position,
                    },
                    "invalid hexadecimal number",
                ));
            }
            if self.integer_literals && integer_literal {
                let mut value = 0_u64;
                for byte in &self.source[digits_start..mantissa_end] {
                    let digit = Self::hex_digit(*byte).expect("validated hexadecimal digit");
                    value = value.wrapping_mul(16).wrapping_add(u64::from(digit));
                }
                return Ok(TokenKind::Integer(value.cast_signed()));
            }
            return Ok(TokenKind::Number(Self::hex_number(
                &self.source[digits_start..mantissa_end],
                integer_digits,
                exponent,
            )));
        }
        if self.byte() == Some(b'.') {
            self.position += 1;
        }
        while matches!(self.byte(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        if self.byte() == Some(b'.') && self.source.get(self.position + 1) != Some(&b'.') {
            self.position += 1;
            while matches!(self.byte(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        if matches!(self.byte(), Some(b'e' | b'E')) {
            let exponent = self.position;
            self.position += 1;
            if matches!(self.byte(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            let digits = self.position;
            while matches!(self.byte(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            if self.position == digits {
                return Err(Self::error(
                    Span {
                        start: exponent,
                        end: self.position,
                    },
                    "invalid number exponent",
                ));
            }
        }
        let integer_literal = self.source[start..self.position]
            .iter()
            .all(u8::is_ascii_digit);
        if let Some(token) = self.cint64_literal(start, start..self.position, 10, integer_literal) {
            return token;
        }
        let source =
            core::str::from_utf8(&self.source[start..self.position]).expect("ASCII number");
        if self.integer_literals && source.bytes().all(|byte| byte.is_ascii_digit()) {
            if let Ok(value) = source.parse::<i64>() {
                return Ok(TokenKind::Integer(value));
            }
            if source.parse::<u64>() == Ok(1_u64 << 63) {
                return Ok(TokenKind::IntegerBoundary);
            }
        }
        source.parse().map(TokenKind::Number).map_err(|_| {
            Self::error(
                Span {
                    start,
                    end: self.position,
                },
                "invalid number",
            )
        })
    }

    /// Lexes `LuaJIT`'s 64-bit integer suffix (`LL`, `ULL` or `LLU` in any
    /// case) after the number digits in `digits`. `None` when the profile
    /// has no such literals or no suffix follows. The suffix only follows
    /// plain integers; a value beyond 64 bits is malformed.
    fn cint64_literal(
        &mut self,
        start: usize,
        digits: core::ops::Range<usize>,
        radix: u32,
        integer_literal: bool,
    ) -> Option<Result<TokenKind, SyntaxError>> {
        if !self.cint64_literals {
            return None;
        }
        let suffix_start = self.position;
        let mut suffix_end = suffix_start;
        while matches!(
            self.source.get(suffix_end),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            suffix_end += 1;
        }
        let suffix = self.source[suffix_start..suffix_end].to_ascii_lowercase();
        let unsigned = match suffix.as_slice() {
            b"ll" => Some(false),
            b"ull" | b"llu" => Some(true),
            // The hexadecimal scanner reports other letters itself.
            _ if suffix.is_empty() || radix == 16 => return None,
            _ => None,
        };
        // LuaJIT rejects any other letters glued to a number.
        self.position = suffix_end;
        let malformed = Self::error(
            Span {
                start,
                end: self.position,
            },
            "malformed number",
        );
        let Some(unsigned) = unsigned else {
            return Some(Err(malformed));
        };
        if !integer_literal || digits.is_empty() {
            return Some(Err(malformed));
        }
        let mut bits: u64 = 0;
        for &byte in &self.source[digits] {
            let digit = char::from(byte)
                .to_digit(radix)
                .expect("validated number digit");
            let Some(next) = bits
                .checked_mul(u64::from(radix))
                .and_then(|bits| bits.checked_add(u64::from(digit)))
            else {
                return Some(Err(malformed));
            };
            bits = next;
        }
        Some(Ok(if unsigned {
            TokenKind::UInt64(bits)
        } else {
            TokenKind::Int64(bits.cast_signed())
        }))
    }

    /// Converts a hexadecimal mantissa and binary exponent without delegating
    /// to the platform C library. The retained prefix has enough bits to round
    /// a binary64 significand; the rest only contributes a sticky bit.
    #[allow(clippy::cast_precision_loss)] // Rounded to at most 53 significant bits above.
    pub(crate) fn hex_number(mantissa: &[u8], integer_digits: usize, exponent: i64) -> f64 {
        let digits: Vec<u8> = mantissa
            .iter()
            .filter_map(|byte| Self::hex_digit(*byte))
            .collect();
        let Some(first) = digits.iter().position(|digit| *digit != 0) else {
            return 0.0;
        };

        let mut prefix = 0_u64;
        let mut count = 0_usize;
        for digit in &digits[first..] {
            prefix = (prefix << 4) | u64::from(*digit);
            count += 1;
            if 64 - prefix.leading_zeros() >= 54 {
                break;
            }
        }
        let prefix_bits = 64 - prefix.leading_zeros();
        let shift = prefix_bits.saturating_sub(53);
        let mut significand = prefix >> shift;
        if shift != 0 {
            let guard = (prefix >> (shift - 1)) & 1 != 0;
            let sticky = shift > 1 && prefix & ((1_u64 << (shift - 1)) - 1) != 0
                || digits[first + count..].iter().any(|digit| *digit != 0);
            if guard && (sticky || significand & 1 != 0) {
                significand += 1;
            }
        }

        let hex_places = i64::try_from(integer_digits)
            .unwrap_or(i64::MAX)
            .saturating_sub(i64::try_from(first + count).unwrap_or(i64::MAX));
        let binary_shift = exponent
            .saturating_add(hex_places.saturating_mul(4))
            .saturating_add(i64::from(shift));
        Self::scale_binary(significand as f64, binary_shift)
    }

    /// Scales a finite positive value by an integral binary exponent. Chunked
    /// exact powers avoid platform math routines and give f64's normal
    /// overflow and underflow results, including subnormals.
    pub(crate) fn scale_binary(mut value: f64, mut exponent: i64) -> f64 {
        while exponent > 0 {
            let step = exponent.min(1023);
            value *= f64::from_bits(u64::try_from(step + 1023).expect("valid exponent") << 52);
            if value.is_infinite() {
                return f64::INFINITY;
            }
            exponent -= step;
        }
        while exponent < 0 {
            let step = exponent.saturating_neg().min(1022);
            value *= f64::from_bits(u64::try_from(1023 - step).expect("valid exponent") << 52);
            if value == 0.0 {
                return 0.0;
            }
            exponent += step;
        }
        value
    }

    pub(crate) const fn hex_digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    pub(crate) fn identifier(&mut self, start: usize) -> TokenKind {
        self.position += 1;
        self.position += self.source[self.position..]
            .iter()
            .position(|byte| !matches!(*byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'))
            .unwrap_or(self.source.len() - self.position);
        let text = &self.source[start..self.position];
        match text {
            b"and" => TokenKind::And,
            b"break" => TokenKind::Break,
            b"do" => TokenKind::Do,
            b"else" => TokenKind::Else,
            b"elseif" => TokenKind::ElseIf,
            b"end" => TokenKind::End,
            b"function" => TokenKind::Function,
            b"for" => TokenKind::For,
            b"if" => TokenKind::If,
            b"in" => TokenKind::In,
            b"then" => TokenKind::Then,
            b"until" => TokenKind::Until,
            b"while" => TokenKind::While,
            b"local" => TokenKind::Local,
            b"global" => TokenKind::Global,
            b"goto" => TokenKind::Goto,
            b"not" => TokenKind::Not,
            b"or" => TokenKind::Or,
            b"repeat" => TokenKind::Repeat,
            b"nil" => TokenKind::Nil,
            b"true" => TokenKind::True,
            b"false" => TokenKind::False,
            b"return" => TokenKind::Return,
            _ => TokenKind::Identifier(
                core::str::from_utf8(text)
                    .expect("ASCII identifier")
                    .to_owned(),
            ),
        }
    }
    #[allow(clippy::too_many_lines)] // Escape handling is kept together so lexer position updates stay auditable.
    pub(crate) fn string(&mut self, quote: u8, start: usize) -> Result<TokenKind, SyntaxError> {
        self.position += 1;
        // Most literals have no escapes: take their bytes straight from the
        // source. Anything else restarts on the escape-aware path below.
        let body = &self.source[self.position..];
        if let Some(length) = body
            .iter()
            .position(|byte| matches!(*byte, b'\\' | b'\n' | b'\r') || *byte == quote)
            && body[length] == quote
        {
            self.position += length + 1;
            return Ok(TokenKind::String(LuaString::from(&body[..length])));
        }
        let mut bytes = Vec::new();
        loop {
            let Some(byte) = self.byte() else {
                return Err(Self::error(
                    Span {
                        start,
                        end: self.position,
                    },
                    "unterminated string",
                ));
            };
            self.position += 1;
            if byte == quote {
                break;
            }
            if byte == b'\n' || byte == b'\r' {
                return Err(Self::error(
                    Span {
                        start,
                        end: self.position,
                    },
                    "newline in string",
                ));
            }
            if byte != b'\\' {
                bytes.push(byte);
                continue;
            }
            let Some(escaped) = self.byte() else {
                return Err(Self::error(
                    Span {
                        start,
                        end: self.position,
                    },
                    "unterminated string escape",
                ));
            };
            self.position += 1;
            match escaped {
                b'a' => bytes.push(7),
                b'b' => bytes.push(8),
                b'f' => bytes.push(12),
                b'n' => bytes.push(b'\n'),
                b'\n' => {
                    if self.byte() == Some(b'\r') {
                        self.position += 1;
                    }
                    bytes.push(b'\n');
                }
                b'r' => bytes.push(b'\r'),
                b't' => bytes.push(b'\t'),
                b'v' => bytes.push(11),
                b'\\' | b'\'' | b'"' => bytes.push(escaped),
                b'z' => {
                    if !self.lua52_escapes {
                        return Err(Self::error(
                            Span {
                                start: self.position - 2,
                                end: self.position,
                            },
                            "'\\z' escape requires Lua 5.2 or later",
                        ));
                    }
                    while matches!(
                        self.byte(),
                        Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c')
                    ) {
                        self.position += 1;
                    }
                }
                b'x' => {
                    if !self.lua52_escapes {
                        return Err(Self::error(
                            Span {
                                start: self.position - 2,
                                end: self.position,
                            },
                            "hexadecimal escapes require Lua 5.2 or later",
                        ));
                    }
                    let escape_start = self.position.saturating_sub(2);
                    let Some(high) = self.byte().and_then(Self::hex_digit) else {
                        return Err(Self::error(
                            Span {
                                start: escape_start,
                                end: self.position,
                            },
                            "hexadecimal digit expected",
                        ));
                    };
                    self.position += 1;
                    let Some(low) = self.byte().and_then(Self::hex_digit) else {
                        return Err(Self::error(
                            Span {
                                start: escape_start,
                                end: self.position,
                            },
                            "hexadecimal digit expected",
                        ));
                    };
                    self.position += 1;
                    bytes.push(high * 16 + low);
                }
                b'u' => {
                    if !self.unicode_escapes {
                        return Err(Self::error(
                            Span {
                                start: self.position - 2,
                                end: self.position,
                            },
                            "Unicode escapes require Lua 5.3 or later",
                        ));
                    }
                    let escape_start = self.position.saturating_sub(2);
                    if !self.take(b'{') {
                        return Err(Self::error(
                            Span {
                                start: escape_start,
                                end: self.position,
                            },
                            "expected '{' after '\\u'",
                        ));
                    }
                    let digits_start = self.position;
                    let mut value = 0_u32;
                    while let Some(digit) = self.byte().and_then(Self::hex_digit) {
                        self.position += 1;
                        value = value
                            .checked_mul(16)
                            .and_then(|value| value.checked_add(u32::from(digit)))
                            .ok_or_else(|| {
                                Self::error(
                                    Span {
                                        start: escape_start,
                                        end: self.position,
                                    },
                                    "invalid Unicode escape",
                                )
                            })?;
                    }
                    if self.position == digits_start || !self.take(b'}') {
                        return Err(Self::error(
                            Span {
                                start: escape_start,
                                end: self.position,
                            },
                            "invalid Unicode escape",
                        ));
                    }
                    if value > 0x7fff_ffff {
                        return Err(Self::error(
                            Span {
                                start: escape_start,
                                end: self.position,
                            },
                            "invalid Unicode escape",
                        ));
                    }
                    // Lua's historical UTF-8 form keeps surrogate values and
                    // accepts all 31-bit code values in up to six bytes.
                    let count = if value <= 0x7f {
                        1
                    } else if value <= 0x7ff {
                        2
                    } else if value <= 0xffff {
                        3
                    } else if value <= 0x1f_ffff {
                        4
                    } else if value <= 0x3ff_ffff {
                        5
                    } else {
                        6
                    };
                    let mut encoded = [0_u8; 6];
                    let mut remaining = value;
                    for index in (1..count).rev() {
                        encoded[index] =
                            0x80 | u8::try_from(remaining & 0x3f).expect("six-bit continuation");
                        remaining >>= 6;
                    }
                    encoded[0] = if count == 1 {
                        u8::try_from(remaining).expect("single-byte code value")
                    } else {
                        (0xff_u8 << (8 - count))
                            | u8::try_from(remaining).expect("leading code bits")
                    };
                    bytes.extend_from_slice(&encoded[..count]);
                }
                b'\r' => {
                    if matches!(self.byte(), Some(b'\n')) {
                        self.position += 1;
                    }
                    bytes.push(b'\n');
                }
                b'0'..=b'9' => {
                    let mut value = u16::from(escaped - b'0');
                    for _ in 0..2 {
                        if let Some(digit @ b'0'..=b'9') = self.byte() {
                            self.position += 1;
                            value = value * 10 + u16::from(digit - b'0');
                        } else {
                            break;
                        }
                    }
                    if value > u16::from(u8::MAX) {
                        return Err(Self::error(
                            Span {
                                start,
                                end: self.position,
                            },
                            "string escape is outside byte range",
                        ));
                    }
                    bytes.push(u8::try_from(value).expect("checked byte"));
                }
                _ => {
                    return Err(Self::error(
                        Span {
                            start: self.position - 2,
                            end: self.position,
                        },
                        "invalid string escape",
                    ));
                }
            }
        }
        Ok(TokenKind::String(LuaString::from(bytes)))
    }
    /// Returns the delimiter level and the first byte after a long-bracket
    /// opener at the current position, without consuming it.
    pub(crate) fn long_bracket_opening(&self) -> Option<(usize, usize)> {
        if self.byte() != Some(b'[') {
            return None;
        }
        let mut cursor = self.position + 1;
        while self.source.get(cursor) == Some(&b'=') {
            cursor += 1;
        }
        if self.source.get(cursor) != Some(&b'[') {
            return None;
        }
        Some((cursor - self.position - 1, cursor + 1))
    }
    pub(crate) fn long_string(
        &mut self,
        start: usize,
        level: usize,
        content_start: usize,
    ) -> Result<TokenKind, SyntaxError> {
        let bytes = self.long_bracket(start, level, content_start, "unterminated long string")?;
        Ok(TokenKind::String(LuaString::from(bytes)))
    }
    pub(crate) fn long_bracket(
        &mut self,
        start: usize,
        level: usize,
        mut content_start: usize,
        unterminated_message: &str,
    ) -> Result<Vec<u8>, SyntaxError> {
        // Lua discards one physical newline immediately after the opener.
        // The remaining bytes are literal: long brackets have no escapes.
        if matches!(self.source.get(content_start), Some(b'\n' | b'\r')) {
            let first = self.source[content_start];
            content_start += 1;
            if matches!(
                (first, self.source.get(content_start)),
                (b'\n', Some(b'\r')) | (b'\r', Some(b'\n'))
            ) {
                content_start += 1;
            }
        }
        let mut cursor = content_start;
        while cursor < self.source.len() {
            if self.source[cursor] == b']'
                && self
                    .source
                    .get(cursor + 1..cursor + 1 + level)
                    .is_some_and(|equals| equals.iter().all(|byte| *byte == b'='))
                && self.source.get(cursor + 1 + level) == Some(&b']')
            {
                self.position = cursor + level + 2;
                return Ok(Self::normalize_long_bracket_content(
                    &self.source[content_start..cursor],
                ));
            }
            cursor += 1;
        }
        Err(Self::error(
            Span {
                start,
                end: self.position.max(self.source.len()),
            },
            unterminated_message,
        ))
    }
    pub(crate) fn normalize_long_bracket_content(content: &[u8]) -> Vec<u8> {
        let mut normalized = Vec::with_capacity(content.len());
        let mut index = 0;
        while let Some(byte) = content.get(index) {
            if matches!(*byte, b'\n' | b'\r') {
                normalized.push(b'\n');
                let next = content.get(index + 1);
                index += usize::from(matches!(
                    (*byte, next),
                    (b'\n', Some(b'\r')) | (b'\r', Some(b'\n'))
                ));
            } else {
                normalized.push(*byte);
            }
            index += 1;
        }
        normalized
    }
    pub(crate) fn byte(&self) -> Option<u8> {
        self.source.get(self.position).copied()
    }
    pub(crate) fn take(&mut self, expected: u8) -> bool {
        if self.byte() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
    pub(crate) fn error(span: Span, message: &str) -> SyntaxError {
        SyntaxError::new(span, message)
    }
}

impl TriviaBoundary {
    pub(crate) const fn new() -> Self {
        Self {
            cursor: 0,
            state: TriviaState::Space,
        }
    }

    pub(crate) fn advance(&mut self, bytes: &[u8]) -> Boundary {
        loop {
            match self.state {
                TriviaState::Space => {
                    while matches!(
                        bytes.get(self.cursor),
                        Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c')
                    ) {
                        self.cursor += 1;
                    }
                    match &bytes[self.cursor..] {
                        [] | [b'-'] | [b'-', b'-'] => return Boundary::NeedInput,
                        [b'-', b'-', b'[', rest @ ..] => {
                            self.cursor = bytes.len() - rest.len();
                            self.state = TriviaState::LongOpening { equals: 0 };
                        }
                        [b'-', b'-', ..] => {
                            self.cursor += 2;
                            self.state = TriviaState::LineComment;
                        }
                        _ => return Boundary::Complete(self.cursor),
                    }
                }
                TriviaState::LineComment => {
                    while let Some(byte) = bytes.get(self.cursor) {
                        if matches!(*byte, b'\n' | b'\r') {
                            self.cursor += 1;
                            self.state = TriviaState::Space;
                            break;
                        }
                        self.cursor += 1;
                    }
                    if !matches!(self.state, TriviaState::Space) {
                        return Boundary::NeedInput;
                    }
                }
                TriviaState::LongOpening { mut equals } => {
                    while let Some(byte) = bytes.get(self.cursor) {
                        if *byte == b'=' {
                            equals += 1;
                            self.cursor += 1;
                            continue;
                        }
                        if *byte == b'[' {
                            self.cursor += 1;
                            self.state = TriviaState::LongComment {
                                level: equals,
                                matched: 0,
                            };
                        } else {
                            self.state = TriviaState::LineComment;
                        }
                        break;
                    }
                    if matches!(self.state, TriviaState::LongOpening { .. }) {
                        self.state = TriviaState::LongOpening { equals };
                        return Boundary::NeedInput;
                    }
                }
                TriviaState::LongComment { level, mut matched } => {
                    while let Some(byte) = bytes.get(self.cursor) {
                        matched = long_close_step(*byte, level, matched);
                        self.cursor += 1;
                        if matched == level + 2 {
                            self.state = TriviaState::Space;
                            break;
                        }
                    }
                    if !matches!(self.state, TriviaState::Space) {
                        self.state = TriviaState::LongComment { level, matched };
                        return Boundary::NeedInput;
                    }
                }
            }
        }
    }
}

impl TokenBoundary {
    pub(crate) fn new(bytes: &[u8], start: usize) -> Self {
        let state = match bytes[start] {
            b'=' | b'~' | b'<' | b'>' | b'-' | b'*' | b'+' | b'/' | b'%' | b'^' | b'#' | b':'
            | b',' | b';' | b'(' | b')' | b']' | b'{' | b'}' => TokenState::Fixed {
                required: start + 2,
            },
            b'.' => TokenState::Dot,
            b'[' => TokenState::LongOpening {
                cursor: start + 1,
                equals: 0,
            },
            quote @ (b'\'' | b'"') => TokenState::ShortString {
                cursor: start + 1,
                quote,
                escaped: false,
                skip_whitespace: false,
                escaped_newline: None,
            },
            b'0'..=b'9' => TokenState::Number {
                cursor: start + 1,
                mode: NumberMode::Decimal,
                exponent_sign_allowed: false,
            },
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => TokenState::Identifier { cursor: start + 1 },
            _ => TokenState::Complete,
        };
        Self { start, state }
    }

    #[allow(clippy::too_many_lines)] // Each token boundary preserves a different lexer lookahead rule.
    pub(crate) fn advance(&mut self, bytes: &[u8]) -> Boundary {
        loop {
            match self.state {
                TokenState::Complete => return Boundary::Complete(self.start),
                TokenState::Fixed { required } => {
                    return if bytes.len() >= required {
                        Boundary::Complete(self.start)
                    } else {
                        Boundary::NeedInput
                    };
                }
                TokenState::Identifier { mut cursor } => {
                    while matches!(
                        bytes.get(cursor),
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
                    ) {
                        cursor += 1;
                    }
                    self.state = TokenState::Identifier { cursor };
                    return if cursor < bytes.len() {
                        Boundary::Complete(self.start)
                    } else {
                        Boundary::NeedInput
                    };
                }
                TokenState::Number {
                    mut cursor,
                    mut mode,
                    mut exponent_sign_allowed,
                } => {
                    while let Some(byte) = bytes.get(cursor) {
                        if matches!(mode, NumberMode::Decimal)
                            && cursor == self.start + 1
                            && self.start + 1 < bytes.len()
                            && bytes[self.start] == b'0'
                            && matches!(*byte, b'x' | b'X')
                        {
                            mode = NumberMode::Hexadecimal;
                            cursor += 1;
                            continue;
                        }
                        if *byte == b'.' {
                            match bytes.get(cursor + 1) {
                                Some(b'.') => break,
                                None => {
                                    self.state = TokenState::Number {
                                        cursor,
                                        mode,
                                        exponent_sign_allowed,
                                    };
                                    return Boundary::NeedInput;
                                }
                                Some(_) => {}
                            }
                        }
                        if matches!(*byte, b'+' | b'-') && !exponent_sign_allowed {
                            break;
                        }
                        if !matches!(*byte, b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'.' | b'+' | b'-')
                        {
                            break;
                        }
                        exponent_sign_allowed = match mode {
                            NumberMode::Decimal => matches!(*byte, b'e' | b'E'),
                            NumberMode::Hexadecimal => matches!(*byte, b'p' | b'P'),
                        };
                        cursor += 1;
                    }
                    self.state = TokenState::Number {
                        cursor,
                        mode,
                        exponent_sign_allowed,
                    };
                    return if cursor < bytes.len() {
                        Boundary::Complete(self.start)
                    } else {
                        Boundary::NeedInput
                    };
                }
                TokenState::Dot => {
                    return match bytes.get(self.start + 1) {
                        None => Boundary::NeedInput,
                        Some(b'0'..=b'9') => {
                            self.state = TokenState::Number {
                                cursor: self.start + 2,
                                mode: NumberMode::Decimal,
                                exponent_sign_allowed: false,
                            };
                            continue;
                        }
                        Some(b'.') if bytes.len() < self.start + 3 => Boundary::NeedInput,
                        Some(_) => Boundary::Complete(self.start),
                    };
                }
                TokenState::ShortString {
                    mut cursor,
                    quote,
                    mut escaped,
                    mut skip_whitespace,
                    mut escaped_newline,
                } => {
                    while let Some(byte) = bytes.get(cursor) {
                        if let Some(first) = escaped_newline {
                            escaped_newline = None;
                            if matches!((first, *byte), (b'\n', b'\r') | (b'\r', b'\n')) {
                                cursor += 1;
                                continue;
                            }
                            continue;
                        }
                        if skip_whitespace {
                            if matches!(*byte, b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c') {
                                cursor += 1;
                                continue;
                            }
                            skip_whitespace = false;
                            continue;
                        }
                        cursor += 1;
                        if escaped {
                            escaped = false;
                            if *byte == b'z' {
                                skip_whitespace = true;
                            } else if matches!(*byte, b'\n' | b'\r') {
                                escaped_newline = Some(*byte);
                            }
                        } else if *byte == b'\\' {
                            escaped = true;
                        } else if *byte == quote || matches!(*byte, b'\n' | b'\r') {
                            return Boundary::Complete(self.start);
                        }
                    }
                    self.state = TokenState::ShortString {
                        cursor,
                        quote,
                        escaped,
                        skip_whitespace,
                        escaped_newline,
                    };
                    return Boundary::NeedInput;
                }
                TokenState::LongOpening {
                    mut cursor,
                    mut equals,
                } => {
                    while let Some(byte) = bytes.get(cursor) {
                        if *byte == b'=' {
                            equals += 1;
                            cursor += 1;
                            continue;
                        }
                        if *byte != b'[' {
                            return Boundary::Complete(self.start);
                        }
                        self.state = TokenState::LongString {
                            cursor: cursor + 1,
                            level: equals,
                            matched: 0,
                        };
                        break;
                    }
                    if matches!(self.state, TokenState::LongOpening { .. }) {
                        self.state = TokenState::LongOpening { cursor, equals };
                        return Boundary::NeedInput;
                    }
                }
                TokenState::LongString {
                    mut cursor,
                    level,
                    mut matched,
                } => {
                    while let Some(byte) = bytes.get(cursor) {
                        matched = long_close_step(*byte, level, matched);
                        cursor += 1;
                        if matched == level + 2 {
                            return Boundary::Complete(self.start);
                        }
                    }
                    self.state = TokenState::LongString {
                        cursor,
                        level,
                        matched,
                    };
                    return Boundary::NeedInput;
                }
            }
        }
    }
}

pub(crate) fn long_close_step(byte: u8, level: usize, matched: usize) -> usize {
    if matched == 0 {
        return usize::from(byte == b']');
    }
    if matched <= level {
        if byte == b'=' {
            return matched + 1;
        }
    } else if byte == b']' {
        return level + 2;
    }
    usize::from(byte == b']')
}

impl IncrementalLexer {
    #[allow(dead_code)]
    pub(crate) const fn new() -> Self {
        Self::with_escapes(true, true, false, false)
    }

    #[allow(clippy::fn_params_excessive_bools)] // One flag per profile feature.
    pub(crate) const fn with_escapes(
        lua52_escapes: bool,
        unicode_escapes: bool,
        integer_literals: bool,
        cint64_literals: bool,
    ) -> Self {
        Self {
            bytes: Vec::new(),
            cursor: 0,
            pending_boundary: None,
            shebang_scan: 2,
            offset: 0,
            ended: false,
            emitted_eof: false,
            lua52_escapes,
            unicode_escapes,
            integer_literals,
            cint64_literals,
        }
    }

    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        assert!(!self.ended, "cannot feed an ended lexer");
        self.compact();
        self.bytes.extend_from_slice(bytes);
    }

    pub(crate) fn end(&mut self) {
        self.ended = true;
    }

    pub(crate) fn poll(&mut self) -> Result<LexPoll, SyntaxError> {
        if self.emitted_eof {
            return Ok(LexPoll::Eof);
        }
        {
            let bytes = &self.bytes[self.cursor..];
            if self.offset == 0 && bytes.first() == Some(&b'#') {
                if bytes.get(1).is_none() && !self.ended {
                    return Ok(LexPoll::NeedInput);
                }
                if bytes.get(1) == Some(&b'!') {
                    if let Some(end) = bytes[self.shebang_scan..]
                        .iter()
                        .position(|byte| matches!(*byte, b'\n' | b'\r'))
                        .map(|end| self.shebang_scan + end)
                    {
                        self.cursor += end;
                        self.offset += end;
                        self.shebang_scan = 2;
                    } else if self.ended {
                        self.offset += bytes.len();
                        self.cursor = self.bytes.len();
                        self.emitted_eof = true;
                        return Ok(LexPoll::Eof);
                    } else {
                        self.shebang_scan = bytes.len();
                        return Ok(LexPoll::NeedInput);
                    }
                }
            }
        }
        if !self.ended
            && self.pending_boundary.is_none()
            && let Some(token) = self.speculate()
        {
            return Ok(LexPoll::Token(token));
        }
        let token_start = if self.ended {
            self.pending_boundary = None;
            None
        } else {
            match self.boundary() {
                Boundary::NeedInput => return Ok(LexPoll::NeedInput),
                Boundary::Complete(start) => Some(start),
            }
        };
        let bytes = &self.bytes[self.cursor..];
        let mut lexer = Lexer::with_escapes(
            bytes,
            self.lua52_escapes,
            self.unicode_escapes,
            self.integer_literals,
            self.cint64_literals,
        );
        if let Some(start) = token_start {
            lexer.position = start;
        } else {
            lexer
                .skip_trivia()
                .map_err(|error| error.shifted(self.offset))?;
        }
        if lexer.byte().is_none() {
            if self.ended {
                self.offset += lexer.position;
                self.cursor = self.bytes.len();
                self.emitted_eof = true;
                return Ok(LexPoll::Eof);
            }
            return Ok(LexPoll::NeedInput);
        }
        let start = lexer.position;
        let kind = lexer
            .token(start)
            .map_err(|error| error.shifted(self.offset))?;
        let consumed = lexer.position;
        let token = Token {
            kind,
            span: Span {
                start: self.offset + start,
                end: self.offset + consumed,
            },
        };
        self.cursor += consumed;
        self.offset += consumed;
        Ok(LexPoll::Token(token))
    }

    /// Lexes the next token in one pass when more input cannot change it.
    ///
    /// The one-shot lexer reads at most one byte past a token, except for a
    /// `[` followed by a run of `=`. A token that ends before the buffered
    /// input does, and is not such a run reaching the end, is therefore
    /// final. Anything else, including errors that a later fragment could
    /// turn into valid input, returns `None` and goes through the resumable
    /// boundary scan, which keeps fragmented input linear.
    fn speculate(&mut self) -> Option<Token> {
        let bytes = &self.bytes[self.cursor..];
        let mut lexer = Lexer::with_escapes(
            bytes,
            self.lua52_escapes,
            self.unicode_escapes,
            self.integer_literals,
            self.cint64_literals,
        );
        lexer.skip_trivia().ok()?;
        let start = lexer.position;
        lexer.byte()?;
        let kind = lexer.token(start).ok()?;
        let consumed = lexer.position;
        if consumed >= bytes.len()
            || (matches!(kind, TokenKind::LBracket)
                && bytes[start + 1..].iter().all(|byte| *byte == b'='))
        {
            return None;
        }
        let token = Token {
            kind,
            span: Span {
                start: self.offset + start,
                end: self.offset + consumed,
            },
        };
        self.cursor += consumed;
        self.offset += consumed;
        Some(token)
    }

    pub(crate) fn boundary(&mut self) -> Boundary {
        let bytes = &self.bytes[self.cursor..];
        loop {
            let pending = self
                .pending_boundary
                .take()
                .unwrap_or_else(|| PendingBoundary::Trivia(TriviaBoundary::new()));
            match pending {
                PendingBoundary::Trivia(mut trivia) => match trivia.advance(bytes) {
                    Boundary::NeedInput => {
                        self.pending_boundary = Some(PendingBoundary::Trivia(trivia));
                        return Boundary::NeedInput;
                    }
                    Boundary::Complete(start) => {
                        self.pending_boundary =
                            Some(PendingBoundary::Token(TokenBoundary::new(bytes, start)));
                    }
                },
                PendingBoundary::Token(mut token) => match token.advance(bytes) {
                    Boundary::NeedInput => {
                        self.pending_boundary = Some(PendingBoundary::Token(token));
                        return Boundary::NeedInput;
                    }
                    Boundary::Complete(start) => return Boundary::Complete(start),
                },
            }
        }
    }

    pub(crate) fn compact(&mut self) {
        // Keep a small consumed prefix until the next reader fragment. This
        // avoids copying for each token while bounding retained source to the
        // current incomplete suffix plus a small amortization window.
        if self.cursor != 0 && (self.cursor >= 8 * 1024 || self.cursor * 2 >= self.bytes.len()) {
            self.bytes.copy_within(self.cursor.., 0);
            self.bytes.truncate(self.bytes.len() - self.cursor);
            self.cursor = 0;
        }
    }

    #[cfg(test)]
    pub(crate) fn unconsumed(&self) -> &[u8] {
        &self.bytes[self.cursor..]
    }

    #[cfg(test)]
    pub(crate) fn pending_boundary_progress(&self) -> Option<usize> {
        match self.pending_boundary.as_ref()? {
            PendingBoundary::Trivia(TriviaBoundary { cursor, .. }) => Some(*cursor),
            PendingBoundary::Token(TokenBoundary { state, .. }) => match state {
                TokenState::Identifier { cursor }
                | TokenState::Number { cursor, .. }
                | TokenState::ShortString { cursor, .. }
                | TokenState::LongOpening { cursor, .. }
                | TokenState::LongString { cursor, .. } => Some(*cursor),
                TokenState::Fixed { .. } | TokenState::Dot | TokenState::Complete => None,
            },
        }
    }
}
