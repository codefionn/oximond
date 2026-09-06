//! Recursive-descent parser for a source that is complete up front.
//!
//! The streaming parser feeds every token through resumable state machines,
//! which costs several calls and moves per token. When the whole chunk is in
//! memory, this parser reads it directly instead. It builds the same tree as
//! the streaming parser for every chunk it accepts.
//!
//! It never reports errors. On any syntax error, and on rare constructs it
//! leaves to the streaming parser, it gives up and the caller parses the
//! source again with [`oximond::syntax::StreamingParser`]. That parser stays the single
//! source of diagnostics, and this one only has to agree with it on valid
//! programs.

use crate::compat::{CompatibilityProfile, ParserCapabilities};
use crate::string::LuaString;
use crate::syntax::{
    Block, Chunk, ElseIf, Expr, FunctionName, Lexer, LocalAttribute, Name, ParserConfig,
    PrattMachine, PrattOperator, Span, Statement, StreamingParser, SyntaxDepth, TableField, Token,
    TokenKind, make_unary_expression,
};
use std::sync::Arc;

/// The parser gave up. The streaming parser decides what is wrong.
struct GiveUp;

type Parse<T> = Result<T, GiveUp>;

/// Recursion this parser allows before it leaves a chunk to the streaming
/// parser, which keeps deeply nested input on the heap. Real code rarely
/// nests this deep, and it keeps debug builds within small thread stacks.
const MAX_NESTING: usize = 40;

/// Parses a complete chunk, or returns `None` when the streaming parser must
/// handle it.
pub(crate) fn parse_complete(source: &[u8], config: ParserConfig) -> Option<Chunk> {
    parse_complete_with_depth(source, config).map(|(chunk, _)| chunk)
}

/// Like [`parse_complete`], and also returns the tree's `nesting_depth`, which
/// the parser tracks as it builds nodes. The compiler sizes its stack from it
/// and so avoids a separate walk over the whole tree.
pub(crate) fn parse_complete_with_depth(
    source: &[u8],
    config: ParserConfig,
) -> Option<(Chunk, usize)> {
    let mut parser = FastParser::new(source, config);
    let statements = parser.chunk().ok()?;
    let chunk = Chunk {
        statements,
        span: Span {
            start: 0,
            end: source.len(),
        },
        line_starts: line_starts(source),
    };
    Some((chunk, parser.deepest))
}

/// The parser's tokens, read one at a time under the same rules, for the
/// compiler's direct path for data chunks (`compiler::data`). Like the
/// parser it never reports errors: `None` means give up.
pub(crate) struct Tokens<'a> {
    parser: FastParser<'a>,
}

impl<'a> Tokens<'a> {
    /// The tokens of `source`, positioned on the first one.
    pub(crate) fn new(source: &'a [u8], config: ParserConfig) -> Option<Self> {
        let mut parser = FastParser::new(source, config);
        parser.advance().ok()?;
        Some(Self { parser })
    }

    /// The current token's kind.
    pub(crate) const fn kind(&self) -> &TokenKind {
        &self.parser.token.kind
    }

    /// The current token's span.
    pub(crate) const fn span(&self) -> Span {
        self.parser.token.span
    }

    /// Moves to the next token.
    pub(crate) fn advance(&mut self) -> Option<()> {
        self.parser.advance().ok()
    }

    /// The kind of the token after the current one.
    pub(crate) fn peek(&mut self) -> Option<&TokenKind> {
        self.parser.peek().ok().map(|token| &token.kind)
    }

    /// The current string token's value; the token is about to be replaced.
    pub(crate) fn take_string(&mut self) -> LuaString {
        self.parser.take_string()
    }

    /// The text of the identifier at `span`.
    pub(crate) fn text(&mut self, span: Span) -> Option<Arc<str>> {
        self.parser.text(span).ok()
    }
}

/// Which tokens end a block.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Stop {
    Eof,
    End,
    Until,
    /// `end`, `elseif` or `else`.
    IfArm,
}

#[allow(clippy::struct_excessive_bools)] // Profile switches checked per token.
struct FastParser<'a> {
    source: &'a [u8],
    lexer: Lexer<'a>,
    /// The current token.
    token: Token,
    /// A raw token read ahead of `token`.
    peeked: Option<Token>,
    capabilities: ParserCapabilities,
    luajit: bool,
    lua55: bool,
    goto_keyword: bool,
    depth: SyntaxDepth,
    previous_end: Option<usize>,
    previous_can_end_statement: bool,
    /// The current token is a `(` that Lua 5.1 rejects as the start of a
    /// call, because a line break precedes it.
    call_on_new_line: bool,
    /// Whether the innermost function accepts `...`.
    vararg: bool,
    loop_depth: usize,
    legacy_break_terminates: bool,
    nesting: usize,
    values: Vec<Expr>,
    /// The height of each entry of `values`.
    heights: Vec<usize>,
    operators: Vec<PrattOperator>,
    /// One string per distinct short string literal and field name of the
    /// chunk, so a key used a hundred times is built once.
    strings: hashbrown::HashTable<LuaString>,
    /// One shared text per distinct name of the chunk.
    names: hashbrown::HashTable<Arc<str>>,
    /// The height of the expression or block parsed last. A node is one more
    /// than its tallest child, which is how `Chunk::nesting_depth` counts.
    height: usize,
    /// The tallest node completed since the enclosing statement began.
    deepest: usize,
}

impl<'a> FastParser<'a> {
    fn new(source: &'a [u8], config: ParserConfig) -> Self {
        let compatibility = config.compatibility_profile();
        let luajit = compatibility == CompatibilityProfile::LuaJit21;
        Self {
            source,
            lexer: Lexer::with_escapes(
                source,
                !compatibility.is_lua51_family() || luajit,
                compatibility.is_lua53_or_later() || luajit,
                compatibility.is_lua53_or_later(),
                luajit,
            ),
            token: Token {
                kind: TokenKind::Eof,
                span: Span::default(),
            },
            peeked: None,
            capabilities: compatibility.capabilities().parser,
            luajit,
            lua55: compatibility == CompatibilityProfile::Lua55,
            goto_keyword: matches!(
                compatibility,
                CompatibilityProfile::Lua52
                    | CompatibilityProfile::Lua53
                    | CompatibilityProfile::Lua54
                    | CompatibilityProfile::Lua55
                    | CompatibilityProfile::LuaJit21
            ),
            depth: SyntaxDepth::default(),
            previous_end: None,
            previous_can_end_statement: false,
            call_on_new_line: false,
            vararg: true,
            loop_depth: 0,
            legacy_break_terminates: compatibility.is_lua51_family(),
            nesting: 0,
            values: Vec::new(),
            heights: Vec::new(),
            operators: Vec::new(),
            // Sized from Luanti's sources, which hold a distinct name every
            // 50 bytes and a distinct string or field name every 130, so the
            // tables seldom grow.
            strings: hashbrown::HashTable::with_capacity(source.len() / 128),
            names: hashbrown::HashTable::with_capacity(source.len() / 64),
            height: 0,
            deepest: 0,
        }
    }

    /// The chunk's string with `bytes`, built on first use.
    fn string(&mut self, bytes: &[u8]) -> LuaString {
        let hash = table_hash(bytes);
        if let Some(string) = self.strings.find(hash, |string| string.as_bytes() == bytes) {
            return string.clone();
        }
        let string = LuaString::new(bytes);
        self.strings
            .insert_unique(hash, string.clone(), |string| table_hash(string.as_bytes()));
        string
    }

    // ---- Tokens ----

    /// Lexes the next token exactly as [`Lexer`] would. Common tokens are
    /// scanned here; everything else goes to the full lexer. Identifiers
    /// carry an empty string: their text is read from the source when a
    /// name is built, so a field name never allocates twice.
    #[allow(clippy::too_many_lines)] // One dispatch on the first byte.
    fn lex(&mut self) -> Parse<Token> {
        if let Some(token) = self.peeked.take() {
            return Ok(token);
        }
        let source = self.source;
        let mut position = self.lexer.position;
        if position == 0 && source.starts_with(b"#!") {
            // The first line of a script may be a `#!` line.
            position = find_line_break(source, 0).unwrap_or(source.len());
        }
        loop {
            match source.get(position) {
                Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c') => position += 1,
                Some(b'-') if source.get(position + 1) == Some(&b'-') => {
                    if source.get(position + 2) == Some(&b'[') {
                        // Possibly a long comment.
                        self.lexer.position = position;
                        self.lexer.skip_trivia().map_err(|_| GiveUp)?;
                        position = self.lexer.position;
                    } else {
                        position = find_line_break(source, position + 2).unwrap_or(source.len());
                    }
                }
                _ => break,
            }
        }
        let start = position;
        let Some(&first) = source.get(start) else {
            self.lexer.position = start;
            let end = source.len();
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span { start: end, end },
            });
        };
        let next = source.get(start + 1).copied();
        let (kind, length) = match first {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let length = 1 + source[start + 1..]
                    .iter()
                    .position(
                        |byte| !matches!(*byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'),
                    )
                    .unwrap_or(source.len() - start - 1);
                (keyword(&source[start..start + length]), length)
            }
            b'(' => (TokenKind::LParen, 1),
            b')' => (TokenKind::RParen, 1),
            b'{' => (TokenKind::LBrace, 1),
            b'}' => (TokenKind::RBrace, 1),
            b']' => (TokenKind::RBracket, 1),
            b',' => (TokenKind::Comma, 1),
            b';' => (TokenKind::Semi, 1),
            b'+' => (TokenKind::Plus, 1),
            b'-' => (TokenKind::Minus, 1),
            b'*' => (TokenKind::Star, 1),
            b'%' => (TokenKind::Percent, 1),
            b'^' => (TokenKind::Caret, 1),
            b'#' => (TokenKind::Hash, 1),
            b'&' => (TokenKind::BitAnd, 1),
            b'|' => (TokenKind::BitOr, 1),
            b'=' if next == Some(b'=') => (TokenKind::Equal, 2),
            b'=' => (TokenKind::Assign, 1),
            b'~' if next == Some(b'=') => (TokenKind::NotEqual, 2),
            b'~' => (TokenKind::BitXor, 1),
            b'<' if next == Some(b'<') => (TokenKind::ShiftLeft, 2),
            b'<' if next == Some(b'=') => (TokenKind::LessEqual, 2),
            b'<' => (TokenKind::LessThan, 1),
            b'>' if next == Some(b'>') => (TokenKind::ShiftRight, 2),
            b'>' if next == Some(b'=') => (TokenKind::GreaterEqual, 2),
            b'>' => (TokenKind::GreaterThan, 1),
            b'/' if next == Some(b'/') => (TokenKind::FloorDivide, 2),
            b'/' => (TokenKind::Slash, 1),
            b'[' if !matches!(next, Some(b'[' | b'=')) => (TokenKind::LBracket, 1),
            b'.' if next == Some(b'.') && source.get(start + 2) == Some(&b'.') => {
                (TokenKind::Vararg, 3)
            }
            b'.' if next == Some(b'.') => (TokenKind::Concat, 2),
            b'.' if !next.is_some_and(|byte| byte.is_ascii_digit()) => (TokenKind::Dot, 1),
            b':' if next == Some(b':') => (TokenKind::DoubleColon, 2),
            b':' => (TokenKind::Colon, 1),
            b'0'..=b'9' => match self.decimal_integer(start) {
                Some(token) => token,
                None => self.lex_slow(start)?,
            },
            quote @ (b'"' | b'\'') => {
                let body = &source[start + 1..];
                match body
                    .iter()
                    .position(|byte| matches!(*byte, b'\\' | b'\n' | b'\r') || *byte == quote)
                {
                    Some(length) if body[length] == quote => {
                        (TokenKind::String(self.string(&body[..length])), length + 2)
                    }
                    _ => self.lex_slow(start)?,
                }
            }
            _ => self.lex_slow(start)?,
        };
        let end = start + length;
        self.lexer.position = end;
        Ok(Token {
            kind,
            span: Span { start, end },
        })
    }

    /// Lexes a token with the full lexer.
    fn lex_slow(&mut self, start: usize) -> Parse<(TokenKind, usize)> {
        self.lexer.position = start;
        let kind = self.lexer.token(start).map_err(|_| GiveUp)?;
        Ok((kind, self.lexer.position - start))
    }

    /// A short decimal integer with nothing number-like after it, which
    /// every profile reads without rounding.
    fn decimal_integer(&self, start: usize) -> Option<(TokenKind, usize)> {
        let digits = self.source[start..]
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .unwrap_or(self.source.len() - start);
        if digits > 15
            || self.source.get(start + digits).is_some_and(
                |byte| matches!(*byte, b'.' | b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'0'..=b'9'),
            )
        {
            return None;
        }
        let value = self.source[start..start + digits]
            .iter()
            .fold(0_i64, |value, digit| value * 10 + i64::from(digit - b'0'));
        let kind = if self.lexer.integer_literals {
            TokenKind::Integer(value)
        } else {
            #[allow(clippy::cast_precision_loss)] // Fifteen digits are exact.
            TokenKind::Number(value as f64)
        };
        Some((kind, digits))
    }

    /// Moves the current string token's value out, without touching its
    /// reference count. The token is about to be replaced.
    fn take_string(&mut self) -> LuaString {
        match core::mem::replace(&mut self.token.kind, TokenKind::Nil) {
            TokenKind::String(value) => value,
            _ => unreachable!("the current token is a string"),
        }
    }

    /// The text of the identifier at `span`, one shared string per distinct
    /// name of the chunk.
    fn text(&mut self, span: Span) -> Parse<Arc<str>> {
        let bytes = &self.source[span.start..span.end];
        let hash = table_hash(bytes);
        if let Some(name) = self.names.find(hash, |name| name.as_bytes() == bytes) {
            return Ok(name.clone());
        }
        let name: Arc<str> = core::str::from_utf8(bytes).map_err(|_| GiveUp)?.into();
        self.names
            .insert_unique(hash, name.clone(), |name| table_hash(name.as_bytes()));
        Ok(name)
    }

    /// The raw token after the current one.
    fn peek(&mut self) -> Parse<&Token> {
        if self.peeked.is_none() {
            self.peeked = Some(self.lex()?);
        }
        Ok(self.peeked.as_ref().expect("token was just read"))
    }

    /// Moves to the next token and applies the streaming parser's
    /// token-level rules to it.
    fn advance(&mut self) -> Parse<()> {
        let mut token = self.lex()?;
        if matches!(token.kind, TokenKind::Eof) {
            self.token = token;
            return Ok(());
        }
        let keyword_as_name = match token.kind {
            // LuaJIT reads `goto` as a statement only before a name.
            TokenKind::Goto if self.luajit => {
                !matches!(self.peek()?.kind, TokenKind::Identifier(_))
            }
            TokenKind::Goto => !self.goto_keyword,
            TokenKind::Global => !self.lua55,
            _ => false,
        };
        if keyword_as_name {
            token.kind = TokenKind::Identifier(String::new());
        }
        self.depth.check(&token, &[]).map_err(|_| GiveUp)?;
        let capabilities = self.capabilities;
        let rejected = match token.kind {
            TokenKind::FloorDivide => !capabilities.integer_division,
            TokenKind::BitAnd
            | TokenKind::BitOr
            | TokenKind::BitXor
            | TokenKind::ShiftLeft
            | TokenKind::ShiftRight => !capabilities.bitwise_operators,
            // The streaming parser checks labels against its root state
            // only. They are rare enough to leave to it.
            TokenKind::DoubleColon => !self.goto_keyword,
            TokenKind::Semi => !capabilities.empty_statements && !self.previous_can_end_statement,
            // Lua 5.5 declarations are left to the streaming parser.
            TokenKind::Global => true,
            _ => false,
        };
        if rejected {
            return Err(GiveUp);
        }
        self.call_on_new_line = matches!(token.kind, TokenKind::LParen)
            && capabilities.reject_ambiguous_call_newline
            && self.line_break_before(&token);
        self.previous_can_end_statement = StreamingParser::token_can_end_statement(&token.kind);
        self.previous_end = Some(token.span.end);
        self.token = token;
        Ok(())
    }

    fn line_break_before(&self, token: &Token) -> bool {
        self.previous_end.is_some_and(|end| {
            self.source[end..token.span.start]
                .iter()
                .any(|byte| matches!(*byte, b'\n' | b'\r'))
        })
    }

    fn is(&self, kind: &TokenKind) -> bool {
        core::mem::discriminant(&self.token.kind) == core::mem::discriminant(kind)
    }

    /// Consumes the current token if it has `kind`'s variant.
    fn expect(&mut self, kind: &TokenKind) -> Parse<Span> {
        if !self.is(kind) {
            return Err(GiveUp);
        }
        let span = self.token.span;
        self.advance()?;
        Ok(span)
    }

    fn name(&mut self) -> Parse<Name> {
        if !matches!(self.token.kind, TokenKind::Identifier(_)) {
            return Err(GiveUp);
        }
        let span = self.token.span;
        let name = Name {
            text: self.text(span)?,
            span,
        };
        self.advance()?;
        Ok(name)
    }

    fn enter(&mut self) -> Parse<()> {
        self.nesting += 1;
        if self.nesting > MAX_NESTING {
            return Err(GiveUp);
        }
        Ok(())
    }

    const fn leave(&mut self) {
        self.nesting -= 1;
    }

    // ---- Statements ----

    fn chunk(&mut self) -> Parse<Vec<Statement>> {
        self.advance()?;
        let (statements, _) = self.block(Stop::Eof)?;
        Ok(statements)
    }

    /// Parses statements up to and including the block's terminator, whose
    /// span is returned.
    fn block(&mut self, stop: Stop) -> Parse<(Vec<Statement>, Token)> {
        self.enter()?;
        let outer = core::mem::take(&mut self.deepest);
        let mut statements = Vec::new();
        let mut last_statement = false;
        loop {
            let accepted = match self.token.kind {
                TokenKind::Eof => stop == Stop::Eof,
                TokenKind::End => matches!(stop, Stop::End | Stop::IfArm),
                TokenKind::Until => stop == Stop::Until,
                TokenKind::Else | TokenKind::ElseIf => stop == Stop::IfArm,
                _ => {
                    if last_statement {
                        if self.luajit && matches!(self.token.kind, TokenKind::Semi) {
                            self.advance()?;
                            continue;
                        }
                        return Err(GiveUp);
                    }
                    self.statement(&mut statements)?;
                    last_statement = matches!(statements.last(), Some(Statement::Return { .. }))
                        || (self.legacy_break_terminates
                            && matches!(statements.last(), Some(Statement::Break { .. })));
                    if last_statement && matches!(self.token.kind, TokenKind::Semi) {
                        self.advance()?;
                    }
                    continue;
                }
            };
            if !accepted {
                return Err(GiveUp);
            }
            let terminator = self.token.clone();
            if stop != Stop::Eof {
                self.advance()?;
            }
            self.leave();
            // A block is not a node: its statements count as children of its owner.
            self.height = self.deepest;
            self.deepest = self.deepest.max(outer);
            return Ok((statements, terminator));
        }
    }

    fn statement(&mut self, statements: &mut Vec<Statement>) -> Parse<()> {
        let before = statements.len();
        let outer = core::mem::take(&mut self.deepest);
        self.statement_kind(statements)?;
        let height = if statements.len() > before {
            self.deepest + 1
        } else {
            0
        };
        self.deepest = outer.max(height);
        Ok(())
    }

    #[allow(clippy::too_many_lines)] // One arm per statement keyword.
    fn statement_kind(&mut self, statements: &mut Vec<Statement>) -> Parse<()> {
        let start = self.token.span;
        match self.token.kind {
            TokenKind::Semi => self.advance()?,
            TokenKind::Do => {
                self.advance()?;
                let (body, end) = self.block(Stop::End)?;
                statements.push(Statement::Do(Block {
                    statements: body,
                    span: start.join(end.span),
                }));
            }
            TokenKind::While => {
                self.advance()?;
                let condition = self.expression()?;
                self.expect(&TokenKind::Do)?;
                let body = self.loop_body(start)?;
                statements.push(Statement::While {
                    condition,
                    span: start.join(body.span),
                    body,
                });
            }
            TokenKind::Repeat => {
                self.advance()?;
                self.loop_depth += 1;
                let (body, _) = self.block(Stop::Until)?;
                self.loop_depth -= 1;
                let condition = self.expression()?;
                let span = start.join(condition.span());
                statements.push(Statement::Repeat {
                    span,
                    body: Block {
                        statements: body,
                        span,
                    },
                    condition,
                });
            }
            TokenKind::If => self.if_statement(statements)?,
            TokenKind::For => self.for_statement(statements)?,
            TokenKind::Function => {
                self.advance()?;
                let mut segments = vec![self.name()?];
                let mut method = false;
                loop {
                    if self.is(&TokenKind::Dot) {
                        self.advance()?;
                        segments.push(self.name()?);
                    } else if self.is(&TokenKind::Colon) {
                        self.advance()?;
                        segments.push(self.name()?);
                        method = true;
                        break;
                    } else {
                        break;
                    }
                }
                let function = self.function_body()?;
                let span = start.join(function.end);
                statements.push(Statement::Function {
                    name: FunctionName {
                        segments,
                        method,
                        span,
                    },
                    parameters: function.parameters,
                    vararg: function.vararg,
                    vararg_name: None,
                    body: function.body,
                    span,
                });
            }
            TokenKind::Local => {
                self.advance()?;
                if self.is(&TokenKind::Function) {
                    self.advance()?;
                    let name = self.name()?;
                    let function = self.function_body()?;
                    statements.push(Statement::LocalFunction {
                        name,
                        parameters: function.parameters,
                        vararg: function.vararg,
                        vararg_name: None,
                        body: function.body,
                        span: start.join(function.end),
                    });
                } else {
                    self.local(start, statements)?;
                }
            }
            TokenKind::Return => {
                self.advance()?;
                if matches!(
                    self.token.kind,
                    TokenKind::Semi
                        | TokenKind::Eof
                        | TokenKind::End
                        | TokenKind::Else
                        | TokenKind::ElseIf
                        | TokenKind::Until
                ) {
                    statements.push(Statement::Return {
                        values: Vec::new(),
                        span: start,
                    });
                } else {
                    let values = self.expression_list()?;
                    let span = start.join(values.last().expect("one value").span());
                    statements.push(Statement::Return { values, span });
                }
            }
            TokenKind::Break => {
                if self.loop_depth == 0 {
                    return Err(GiveUp);
                }
                self.advance()?;
                statements.push(Statement::Break { span: start });
            }
            TokenKind::Goto => {
                self.advance()?;
                let name = self.name()?;
                statements.push(Statement::Goto {
                    span: start.join(name.span),
                    name,
                });
            }
            TokenKind::DoubleColon => {
                self.advance()?;
                let name = self.name()?;
                let end = self.expect(&TokenKind::DoubleColon)?;
                statements.push(Statement::Label {
                    span: start.join(end),
                    name,
                });
            }
            _ => self.expression_statement(statements)?,
        }
        Ok(())
    }

    fn loop_body(&mut self, start: Span) -> Parse<Block> {
        self.loop_depth += 1;
        let (statements, end) = self.block(Stop::End)?;
        self.loop_depth -= 1;
        Ok(Block {
            statements,
            span: start.join(end.span),
        })
    }

    fn if_statement(&mut self, statements: &mut Vec<Statement>) -> Parse<()> {
        let start = self.token.span;
        self.advance()?;
        let condition = self.expression()?;
        self.expect(&TokenKind::Then)?;
        let (body, mut terminator) = self.block(Stop::IfArm)?;
        let mut else_ifs = Vec::new();
        let mut else_body = None;
        loop {
            match terminator.kind {
                TokenKind::ElseIf => {
                    let arm_start = terminator.span;
                    let arm_condition = self.expression()?;
                    self.expect(&TokenKind::Then)?;
                    let (arm_body, next) = self.block(Stop::IfArm)?;
                    let span = arm_start.join(arm_condition.span());
                    else_ifs.push(ElseIf {
                        condition: arm_condition,
                        body: Block {
                            statements: arm_body,
                            span,
                        },
                        span,
                    });
                    terminator = next;
                }
                TokenKind::Else => {
                    let else_start = terminator.span;
                    let (else_statements, end) = self.block(Stop::End)?;
                    else_body = Some(Block {
                        statements: else_statements,
                        span: else_start,
                    });
                    terminator = end;
                }
                _ => break,
            }
        }
        let span = start.join(terminator.span);
        statements.push(Statement::If {
            condition,
            body: Block {
                statements: body,
                span,
            },
            else_ifs,
            else_body,
            span,
        });
        Ok(())
    }

    fn for_statement(&mut self, statements: &mut Vec<Statement>) -> Parse<()> {
        let start = self.token.span;
        self.advance()?;
        let name = self.name()?;
        if self.is(&TokenKind::Assign) {
            self.advance()?;
            let initial = self.expression()?;
            self.expect(&TokenKind::Comma)?;
            let limit = self.expression()?;
            let step = if self.is(&TokenKind::Comma) {
                self.advance()?;
                Some(self.expression()?)
            } else {
                None
            };
            self.expect(&TokenKind::Do)?;
            let body = self.loop_body(start)?;
            statements.push(Statement::NumericFor {
                name,
                start: Box::new(initial),
                limit: Box::new(limit),
                step: step.map(Box::new),
                span: start.join(body.span),
                body,
            });
            return Ok(());
        }
        let mut names = vec![name];
        while self.is(&TokenKind::Comma) {
            self.advance()?;
            names.push(self.name()?);
        }
        self.expect(&TokenKind::In)?;
        let expressions = self.expression_list()?;
        self.expect(&TokenKind::Do)?;
        let body = self.loop_body(start)?;
        statements.push(Statement::GenericFor {
            names,
            expressions,
            span: start.join(body.span),
            body,
        });
        Ok(())
    }

    fn local(&mut self, start: Span, statements: &mut Vec<Statement>) -> Parse<()> {
        let mut names = Vec::new();
        let mut attributes = Vec::new();
        loop {
            if names.len() >= 200 {
                return Err(GiveUp);
            }
            names.push(self.name()?);
            attributes.push(LocalAttribute::None);
            if self.is(&TokenKind::LessThan) {
                // Older profiles reject attributes depending on where they
                // appear; the streaming parser decides.
                if !self.capabilities.local_attributes {
                    return Err(GiveUp);
                }
                self.advance()?;
                let span = self.token.span;
                let attribute = match (&self.token.kind, &self.source[span.start..span.end]) {
                    (TokenKind::Identifier(_), b"const") => LocalAttribute::Const,
                    (TokenKind::Identifier(_), b"close") => LocalAttribute::Close,
                    _ => return Err(GiveUp),
                };
                self.advance()?;
                self.expect(&TokenKind::GreaterThan)?;
                *attributes.last_mut().expect("attribute follows a name") = attribute;
            }
            if !self.is(&TokenKind::Comma) {
                break;
            }
            self.advance()?;
        }
        let (values, span) = if self.is(&TokenKind::Assign) {
            self.advance()?;
            let values = self.expression_list()?;
            let span = start.join(values.last().expect("one value").span());
            (values, span)
        } else {
            let span = start.join(names.last().expect("one name").span);
            (Vec::new(), span)
        };
        statements.push(Statement::Local {
            names,
            attributes,
            values,
            span,
        });
        Ok(())
    }

    fn expression_statement(&mut self, statements: &mut Vec<Statement>) -> Parse<()> {
        let first = self.expression()?;
        if !matches!(self.token.kind, TokenKind::Assign | TokenKind::Comma) {
            if !matches!(first, Expr::Call { .. } | Expr::MethodCall { .. }) {
                return Err(GiveUp);
            }
            statements.push(Statement::Call(first));
            return Ok(());
        }
        let start = first.span();
        let mut targets = vec![first];
        while self.is(&TokenKind::Comma) {
            self.advance()?;
            targets.push(self.expression()?);
        }
        self.expect(&TokenKind::Assign)?;
        let values = self.expression_list()?;
        if !targets.iter().all(Expr::is_assignment_target) {
            return Err(GiveUp);
        }
        let span = start.join(values.last().expect("one value").span());
        statements.push(Statement::Assign {
            targets,
            values,
            span,
        });
        Ok(())
    }

    fn expression_list(&mut self) -> Parse<Vec<Expr>> {
        let mut values = vec![self.expression()?];
        let mut tallest = self.height;
        while self.is(&TokenKind::Comma) {
            self.advance()?;
            values.push(self.expression()?);
            tallest = tallest.max(self.height);
        }
        self.height = tallest;
        Ok(values)
    }

    /// Parses `(parameters) body end` from the current `(`.
    fn function_body(&mut self) -> Parse<ParsedBody> {
        let open = self.expect(&TokenKind::LParen)?;
        let mut parameters = Vec::new();
        let mut vararg = false;
        if !self.is(&TokenKind::RParen) {
            loop {
                if self.is(&TokenKind::Vararg) {
                    self.advance()?;
                    vararg = true;
                    break;
                }
                parameters.push(self.name()?);
                if !self.is(&TokenKind::Comma) {
                    break;
                }
                self.advance()?;
            }
        }
        // A named vararg (`...name`) is left to the streaming parser.
        self.expect(&TokenKind::RParen)?;
        let saved = (self.vararg, self.loop_depth);
        self.vararg = vararg;
        self.loop_depth = 0;
        let (statements, end) = self.block(Stop::End)?;
        (self.vararg, self.loop_depth) = saved;
        Ok(ParsedBody {
            parameters,
            vararg,
            body: Block {
                statements,
                span: open.join(end.span),
            },
            end: end.span,
        })
    }

    // ---- Expressions ----

    /// Parses one expression and stops at the first token that cannot
    /// continue it. Operators reduce exactly as in [`PrattMachine`].
    fn expression(&mut self) -> Parse<Expr> {
        self.enter()?;
        let values_base = self.values.len();
        let operators_base = self.operators.len();
        loop {
            loop {
                let op = match self.token.kind {
                    TokenKind::Minus => crate::syntax::UnaryOp::Negate,
                    TokenKind::Not => crate::syntax::UnaryOp::Not,
                    TokenKind::Hash => crate::syntax::UnaryOp::Length,
                    TokenKind::BitXor => crate::syntax::UnaryOp::BitNot,
                    _ => break,
                };
                self.operators.push(PrattOperator::Unary {
                    op,
                    span: self.token.span,
                });
                self.advance()?;
            }
            let operand = self.suffixed()?;
            let Some((op, precedence, right_associative)) =
                PrattMachine::binary_operator(&self.token.kind)
            else {
                if self.operators.len() == operators_base && self.values.len() == values_base {
                    // A lone operand, the most common expression.
                    self.leave();
                    self.deepest = self.deepest.max(self.height);
                    return Ok(operand);
                }
                self.values.push(operand);
                self.heights.push(self.height);
                break;
            };
            self.values.push(operand);
            self.heights.push(self.height);
            while self.should_reduce(operators_base, precedence, right_associative) {
                self.reduce(values_base)?;
            }
            self.operators.push(PrattOperator::Binary {
                op,
                precedence,
                span: self.token.span,
            });
            self.advance()?;
        }
        while self.operators.len() > operators_base {
            self.reduce(values_base)?;
        }
        if self.values.len() != values_base + 1 {
            return Err(GiveUp);
        }
        self.leave();
        self.height = self.heights.pop().expect("one height");
        self.deepest = self.deepest.max(self.height);
        Ok(self.values.pop().expect("one value"))
    }

    fn should_reduce(&self, operators_base: usize, incoming: u8, right_associative: bool) -> bool {
        if self.operators.len() <= operators_base {
            return false;
        }
        match self.operators.last() {
            Some(PrattOperator::Unary { .. }) => 7 >= incoming,
            Some(PrattOperator::Binary { precedence, .. }) => {
                *precedence > incoming || (*precedence == incoming && !right_associative)
            }
            None => false,
        }
    }

    fn reduce(&mut self, values_base: usize) -> Parse<()> {
        let operator = self.operators.pop().expect("an operator to reduce");
        let mut pop = || {
            if self.values.len() > values_base {
                self.values.pop().ok_or(GiveUp)
            } else {
                Err(GiveUp)
            }
        };
        let height = match operator {
            PrattOperator::Unary { .. } => 1 + self.heights.pop().ok_or(GiveUp)?,
            PrattOperator::Binary { .. } => {
                let right = self.heights.pop().ok_or(GiveUp)?;
                1 + right.max(self.heights.pop().ok_or(GiveUp)?)
            }
        };
        let value = match operator {
            PrattOperator::Unary { op, span } => {
                let expression = pop()?;
                let span = span.join(expression.span());
                make_unary_expression(op, expression, span)
            }
            PrattOperator::Binary {
                op,
                span: operator_span,
                ..
            } => {
                let right = pop()?;
                let left = pop()?;
                Expr::Binary {
                    op,
                    operator_span,
                    span: left.span().join(right.span()),
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
        };
        self.values.push(value);
        self.heights.push(height);
        Ok(())
    }

    /// A primary expression and its suffixes.
    fn suffixed(&mut self) -> Parse<Expr> {
        let mut value = self.primary()?;
        let mut height = self.height;
        while PrattMachine::is_prefix_expression(&value) {
            value = match self.token.kind {
                TokenKind::Dot => {
                    self.advance()?;
                    if !matches!(self.token.kind, TokenKind::Identifier(_)) {
                        return Err(GiveUp);
                    }
                    let field = self.token.span;
                    let key =
                        Expr::String(self.string(&self.source[field.start..field.end]), field);
                    let span = value.span().join(self.token.span);
                    self.advance()?;
                    height += 1;
                    Expr::Index {
                        span,
                        table: Box::new(value),
                        key: Box::new(key),
                    }
                }
                TokenKind::LBracket => {
                    self.advance()?;
                    let key = self.expression()?;
                    let close = self.expect(&TokenKind::RBracket)?;
                    height = 1 + height.max(self.height);
                    Expr::Index {
                        span: value.span().join(close),
                        table: Box::new(value),
                        key: Box::new(key),
                    }
                }
                TokenKind::Colon => {
                    self.advance()?;
                    let method = self.name()?;
                    let argument_span = self.token.span;
                    let (arguments, end) = self.call_arguments()?;
                    height = 1 + height.max(self.height);
                    Expr::MethodCall {
                        argument_span,
                        span: value.span().join(end),
                        receiver: Box::new(value),
                        method: Box::new(method),
                        arguments,
                    }
                }
                TokenKind::LParen | TokenKind::LBrace | TokenKind::String(_) => {
                    let (arguments, end) = self.call_arguments()?;
                    height = 1 + height.max(self.height);
                    Expr::Call {
                        span: value.span().join(end),
                        callee: Box::new(value),
                        arguments,
                    }
                }
                _ => break,
            };
        }
        self.height = height;
        Ok(value)
    }

    /// Call arguments and the span that ends the call.
    fn call_arguments(&mut self) -> Parse<(Vec<Expr>, Span)> {
        if self.call_on_new_line {
            return Err(GiveUp);
        }
        match &mut self.token.kind {
            TokenKind::String(_) => {
                let span = self.token.span;
                let argument = Expr::String(self.take_string(), span);
                self.advance()?;
                self.height = 1;
                Ok((vec![argument], span))
            }
            TokenKind::LBrace => {
                let table = self.table()?;
                let span = table.span();
                Ok((vec![table], span))
            }
            TokenKind::LParen => {
                self.advance()?;
                let mut arguments = Vec::new();
                self.height = 0;
                if !self.is(&TokenKind::RParen) {
                    arguments = self.expression_list()?;
                }
                let end = self.expect(&TokenKind::RParen)?;
                Ok((arguments, end))
            }
            _ => Err(GiveUp),
        }
    }

    fn primary(&mut self) -> Parse<Expr> {
        let span = self.token.span;
        let value = match &mut self.token.kind {
            TokenKind::Identifier(_) => Expr::Name(Name {
                text: self.text(span)?,
                span,
            }),
            TokenKind::Nil => Expr::Nil(span),
            TokenKind::True => Expr::Boolean(true, span),
            TokenKind::False => Expr::Boolean(false, span),
            TokenKind::Integer(value) => Expr::Integer(*value, span),
            TokenKind::IntegerBoundary => Expr::IntegerBoundary(span),
            TokenKind::Number(value) => Expr::Number(*value, span),
            TokenKind::Int64(value) => Expr::Int64(*value, span),
            TokenKind::UInt64(value) => Expr::UInt64(*value, span),
            TokenKind::String(_) => Expr::String(self.take_string(), span),
            TokenKind::Vararg if self.vararg => Expr::Vararg(span),
            TokenKind::LParen => {
                self.advance()?;
                let expression = self.expression()?;
                let close = self.expect(&TokenKind::RParen)?;
                self.height += 1;
                return Ok(Expr::Parenthesized {
                    span: span.join(close),
                    expression: Box::new(expression),
                });
            }
            TokenKind::LBrace => return self.table(),
            TokenKind::Function => {
                self.advance()?;
                let function = self.function_body()?;
                self.height += 1;
                return Ok(Expr::Function {
                    parameters: function.parameters,
                    vararg: function.vararg,
                    vararg_name: None,
                    span: span.join(function.end),
                    body: Box::new(function.body),
                });
            }
            _ => return Err(GiveUp),
        };
        self.advance()?;
        self.height = 1;
        Ok(value)
    }

    fn table(&mut self) -> Parse<Expr> {
        let open = self.expect(&TokenKind::LBrace)?;
        self.enter()?;
        let mut fields = Vec::new();
        let mut tallest = 0;
        while !self.is(&TokenKind::RBrace) {
            let named = matches!(self.token.kind, TokenKind::Identifier(_))
                && matches!(self.peek()?.kind, TokenKind::Assign);
            let field = match self.token.kind {
                TokenKind::Identifier(_) if named => {
                    let name = self.name()?;
                    self.advance()?;
                    let value = self.expression()?;
                    tallest = tallest.max(self.height);
                    TableField::Name { name, value }
                }
                TokenKind::LBracket => {
                    self.advance()?;
                    let key = self.expression()?;
                    tallest = tallest.max(self.height);
                    self.expect(&TokenKind::RBracket)?;
                    self.expect(&TokenKind::Assign)?;
                    let value = self.expression()?;
                    tallest = tallest.max(self.height);
                    TableField::Key { key, value }
                }
                _ => {
                    let value = self.expression()?;
                    tallest = tallest.max(self.height);
                    TableField::List(value)
                }
            };
            fields.push(field);
            if matches!(self.token.kind, TokenKind::Comma | TokenKind::Semi) {
                self.advance()?;
            } else if !self.is(&TokenKind::RBrace) {
                return Err(GiveUp);
            }
        }
        let close = self.expect(&TokenKind::RBrace)?;
        self.leave();
        self.height = tallest + 1;
        Ok(Expr::Table {
            fields,
            span: open.join(close),
        })
    }
}

struct ParsedBody {
    parameters: Vec<Name>,
    vararg: bool,
    body: Block,
    end: Span,
}

/// The keyword spelled by `text`, or an identifier whose text the parser
/// reads from the source.
fn keyword(text: &[u8]) -> TokenKind {
    // Dispatch on the length and first byte, so an identifier is compared
    // with at most two keywords.
    let candidate = match (text.len(), text[0]) {
        (2, b'd') => (&b"do"[..], TokenKind::Do),
        (2, b'i') if text[1] == b'f' => (&b"if"[..], TokenKind::If),
        (2, b'i') => (&b"in"[..], TokenKind::In),
        (2, b'o') => (&b"or"[..], TokenKind::Or),
        (3, b'a') => (&b"and"[..], TokenKind::And),
        (3, b'e') => (&b"end"[..], TokenKind::End),
        (3, b'f') => (&b"for"[..], TokenKind::For),
        (3, b'n') if text[1] == b'i' => (&b"nil"[..], TokenKind::Nil),
        (3, b'n') => (&b"not"[..], TokenKind::Not),
        (4, b'e') => (&b"else"[..], TokenKind::Else),
        (4, b'g') => (&b"goto"[..], TokenKind::Goto),
        (4, b't') if text[1] == b'h' => (&b"then"[..], TokenKind::Then),
        (4, b't') => (&b"true"[..], TokenKind::True),
        (5, b'b') => (&b"break"[..], TokenKind::Break),
        (5, b'f') => (&b"false"[..], TokenKind::False),
        (5, b'l') => (&b"local"[..], TokenKind::Local),
        (5, b'u') => (&b"until"[..], TokenKind::Until),
        (5, b'w') => (&b"while"[..], TokenKind::While),
        (6, b'e') => (&b"elseif"[..], TokenKind::ElseIf),
        (6, b'g') => (&b"global"[..], TokenKind::Global),
        (6, b'r') if text[1] == b'e' && text[2] == b'p' => (&b"repeat"[..], TokenKind::Repeat),
        (6, b'r') => (&b"return"[..], TokenKind::Return),
        (8, b'f') => (&b"function"[..], TokenKind::Function),
        _ => return TokenKind::Identifier(String::new()),
    };
    if candidate.0 == text {
        candidate.1
    } else {
        TokenKind::Identifier(String::new())
    }
}

/// Where each line of `source` starts, as the streaming parser records them:
/// `\r\n` and `\n\r` end one line, any other `\r` or `\n` ends one each.
pub(crate) fn line_starts(source: &[u8]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(source.len() / 32 + 1);
    starts.push(0);
    let mut offset = 0;
    while let Some(found) = find_line_break(source, offset) {
        let pair = source
            .get(found + 1)
            .is_some_and(|next| *next != source[found] && matches!(*next, b'\r' | b'\n'));
        offset = found + 1 + usize::from(pair);
        starts.push(offset);
    }
    starts
}

/// The hash of the parser's own name and string tables. It mixes whole
/// words, which beats the byte-wise Lua string hash on short names.
fn table_hash(bytes: &[u8]) -> u64 {
    use core::hash::BuildHasher;
    rustc_hash::FxBuildHasher.hash_one(bytes)
}

/// The index of the first `\n` or `\r` at or after `from`.
pub(crate) fn find_line_break(bytes: &[u8], from: usize) -> Option<usize> {
    let rest = bytes.get(from..)?;
    memchr::memchr2(b'\n', b'\r', rest).map(|offset| from + offset)
}
