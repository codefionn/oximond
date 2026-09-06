#![forbid(unsafe_code)]

use crate::compat::{CompatibilityProfile, ParserCapabilities};
use crate::string::LuaString;
use core::fmt;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
impl Span {
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        Self {
            start: self.start,
            end: other.end,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub statements: Vec<Statement>,
    pub span: Span,
    /// Byte offsets of physical source-line starts. The first line starts at
    /// zero. Parsers that do not retain source bytes may leave this empty.
    pub line_starts: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub span: Span,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ElseIf {
    pub condition: Expr,
    pub body: Block,
    pub span: Span,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Do(Block),
    Local {
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        values: Vec<Expr>,
        span: Span,
    },
    /// A Lua 5.5 lexical global declaration.
    Global {
        mode: GlobalDeclarationMode,
        /// `global <const> *` makes declarations admitted by this scope read-only.
        all_const: bool,
        names: Vec<Name>,
        const_names: Vec<bool>,
        values: Vec<Expr>,
        span: Span,
    },
    Assign {
        targets: Vec<Expr>,
        values: Vec<Expr>,
        span: Span,
    },
    Function {
        name: FunctionName,
        parameters: Vec<Name>,
        vararg: bool,
        /// Lua 5.5 may bind the vararg table to this read-only local.
        vararg_name: Option<Name>,
        body: Block,
        span: Span,
    },
    LocalFunction {
        name: Name,
        parameters: Vec<Name>,
        vararg: bool,
        /// Lua 5.5 may bind the vararg table to this read-only local.
        vararg_name: Option<Name>,
        body: Block,
        span: Span,
    },
    Return {
        values: Vec<Expr>,
        span: Span,
    },
    If {
        condition: Expr,
        body: Block,
        else_ifs: Vec<ElseIf>,
        else_body: Option<Block>,
        span: Span,
    },
    While {
        condition: Expr,
        body: Block,
        span: Span,
    },
    Repeat {
        body: Block,
        condition: Expr,
        span: Span,
    },
    Break {
        span: Span,
    },
    /// A Lua 5.2+ jump to a label in this function.
    Goto {
        name: Name,
        span: Span,
    },
    /// A Lua 5.2+ label, written as `::name::`.
    Label {
        name: Name,
        span: Span,
    },
    NumericFor {
        name: Name,
        start: Box<Expr>,
        limit: Box<Expr>,
        step: Option<Box<Expr>>,
        body: Block,
        span: Span,
    },
    GenericFor {
        names: Vec<Name>,
        expressions: Vec<Expr>,
        body: Block,
        span: Span,
    },
    Call(Expr),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Name {
    pub text: std::sync::Arc<str>,
    pub span: Span,
}
/// A Lua 5.4 local-variable attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalAttribute {
    None,
    Const,
    Close,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalDeclarationMode {
    Names,
    All,
    None,
}
/// The target of a function declaration, such as `a.b.c:f`.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionName {
    pub segments: Vec<Name>,
    pub method: bool,
    pub span: Span,
}
impl FunctionName {
    /// Returns the final field in this declaration path.
    ///
    /// # Panics
    ///
    /// Panics if constructed manually without any path segments. The parser
    /// always creates at least one segment.
    #[must_use]
    pub fn terminal(&self) -> &Name {
        self.segments
            .last()
            .expect("parsed function names have a segment")
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Name(Name),
    Nil(Span),
    Boolean(bool, Span),
    Integer(i64, Span),
    /// The decimal magnitude `2^63`, retained so unary minus can form `i64::MIN`.
    IntegerBoundary(Span),
    Number(f64, Span),
    /// A `LuaJIT` `LL` literal.
    Int64(i64, Span),
    /// A `LuaJIT` `ULL` literal.
    UInt64(u64, Span),
    String(LuaString, Span),
    Vararg(Span),
    Table {
        fields: Vec<TableField>,
        span: Span,
    },
    Function {
        parameters: Vec<Name>,
        vararg: bool,
        /// Lua 5.5 may bind the vararg table to this read-only local.
        vararg_name: Option<Box<Name>>,
        body: Box<Block>,
        span: Span,
    },
    Index {
        table: Box<Expr>,
        key: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        operator_span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        expression: Box<Expr>,
        span: Span,
    },
    Call {
        callee: Box<Expr>,
        arguments: Vec<Expr>,
        span: Span,
    },
    MethodCall {
        argument_span: Span,
        receiver: Box<Expr>,
        method: Box<Name>,
        arguments: Vec<Expr>,
        span: Span,
    },
    Parenthesized {
        expression: Box<Expr>,
        span: Span,
    },
}
impl Expr {
    #[must_use]
    pub const fn span(&self) -> Span {
        match self {
            Self::Name(name) => name.span,
            Self::Nil(span)
            | Self::Boolean(_, span)
            | Self::Integer(_, span)
            | Self::IntegerBoundary(span)
            | Self::Number(_, span)
            | Self::Int64(_, span)
            | Self::UInt64(_, span)
            | Self::String(_, span)
            | Self::Vararg(span)
            | Self::Index { span, .. }
            | Self::Binary { span, .. }
            | Self::Unary { span, .. }
            | Self::Call { span, .. }
            | Self::MethodCall { span, .. }
            | Self::Parenthesized { span, .. }
            | Self::Function { span, .. }
            | Self::Table { span, .. } => *span,
        }
    }
    #[must_use]
    pub const fn is_assignment_target(&self) -> bool {
        matches!(self, Self::Name(_) | Self::Index { .. })
    }
}

fn make_unary_expression(op: UnaryOp, expression: Expr, span: Span) -> Expr {
    if op == UnaryOp::Negate && matches!(expression, Expr::IntegerBoundary(_)) {
        Expr::Integer(i64::MIN, span)
    } else {
        Expr::Unary {
            op,
            expression: Box::new(expression),
            span,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum TableField {
    List(Expr),
    Name { name: Name, value: Expr },
    Key { key: Expr, value: Expr },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOp {
    Or,
    And,
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Modulo,
    Power,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    Concat,
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryOp {
    Negate,
    Not,
    Length,
    BitNot,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxError {
    pub span: Span,
    pub message: String,
    near_span: Span,
    function_span: Option<Span>,
    diagnostic: Option<LuaString>,
}
impl SyntaxError {
    fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            near_span: span,
            function_span: None,
            diagnostic: None,
        }
    }

    fn with_near_span(mut self, near_span: Span) -> Self {
        self.near_span = near_span;
        self
    }

    fn with_function_span(mut self, function_span: Span) -> Self {
        self.function_span = Some(function_span);
        self
    }

    fn shifted(mut self, offset: usize) -> Self {
        self.span.start = self.span.start.saturating_add(offset);
        self.span.end = self.span.end.saturating_add(offset);
        self.near_span.start = self.near_span.start.saturating_add(offset);
        self.near_span.end = self.near_span.end.saturating_add(offset);
        if let Some(function_span) = &mut self.function_span {
            function_span.start = function_span.start.saturating_add(offset);
            function_span.end = function_span.end.saturating_add(offset);
        }
        self
    }

    /// Attaches the chunk name and source text used by Lua-style diagnostics.
    #[must_use]
    pub fn with_source(mut self, source_name: &LuaString, source: &[u8]) -> Self {
        self.diagnostic = Some(lua_syntax_diagnostic(
            source_name,
            source,
            &self.message,
            self.near_span,
            self.function_span,
        ));
        self
    }

    /// Returns the byte-preserving diagnostic passed back by `load`.
    #[must_use]
    pub fn diagnostic(&self) -> LuaString {
        self.diagnostic.clone().unwrap_or_else(|| {
            LuaString::from(format!("{} at byte {}", self.message, self.span.start))
        })
    }
}
impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.diagnostic().fmt(f)
    }
}
impl std::error::Error for SyntaxError {}

fn lua_syntax_diagnostic(
    source_name: &LuaString,
    source: &[u8],
    message: &str,
    near_span: Span,
    function_span: Option<Span>,
) -> LuaString {
    let token_start = source
        .get(near_span.start..near_span.end.min(source.len()))
        .and_then(|candidate| {
            candidate
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .map(|offset| near_span.start + offset)
        })
        .unwrap_or(near_span.start.min(source.len()));
    let line = source_line(source, token_start);
    let near = source_token(source, token_start, near_span.end);
    let mut diagnostic = lua_chunk_id(source_name);
    diagnostic.extend_from_slice(format!(":{line}: {message}").as_bytes());
    if let Some(function_span) = function_span {
        let function_line = source_line(source, function_span.start);
        diagnostic.extend_from_slice(format!(" in function at line {function_line}").as_bytes());
    }
    diagnostic.extend_from_slice(b" near '");
    diagnostic.extend_from_slice(&near);
    diagnostic.push(b'\'');
    LuaString::from(diagnostic)
}

/// Formats a source-positioned diagnostic for semantic errors found after parsing.
#[must_use]
pub fn source_diagnostic(
    source_name: &LuaString,
    source: &[u8],
    message: &str,
    span: Span,
) -> LuaString {
    let mut diagnostic = lua_chunk_id(source_name);
    let line = source_line(source, span.start);
    diagnostic.extend_from_slice(format!(":{line}: {message}").as_bytes());
    LuaString::from(diagnostic)
}

fn source_line(source: &[u8], offset: usize) -> usize {
    let mut line = 1usize;
    let mut cursor = 0usize;
    let end = offset.min(source.len());
    while cursor < end {
        if matches!(source[cursor], b'\n' | b'\r') {
            let first = source[cursor];
            cursor += 1;
            if cursor < end && source[cursor] != first && matches!(source[cursor], b'\n' | b'\r') {
                cursor += 1;
            }
            line += 1;
        } else {
            cursor += 1;
        }
    }
    line
}

fn source_token(source: &[u8], start: usize, span_end: usize) -> Vec<u8> {
    if start >= source.len() {
        return b"<eof>".to_vec();
    }
    let upper = span_end.max(start + 1).min(source.len());
    let candidate = &source[start..upper];
    if candidate.len() == 1 {
        return candidate.to_vec();
    }
    let end = match candidate[0] {
        b'a'..=b'z' | b'A'..=b'Z' | b'_' => candidate
            .iter()
            .position(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
            .unwrap_or(candidate.len()),
        b'0'..=b'9' | b'.' => candidate
            .iter()
            .position(|byte| byte.is_ascii_whitespace() || matches!(*byte, b',' | b';'))
            .unwrap_or(candidate.len()),
        b'\'' | b'"' => candidate
            .iter()
            .skip(1)
            .position(|byte| *byte == candidate[0])
            .map_or(candidate.len(), |offset| offset + 2),
        _ => candidate.len(),
    };
    candidate[..end.max(1)].to_vec()
}

fn lua_chunk_id(source_name: &LuaString) -> Vec<u8> {
    const ID_SIZE: usize = 60;
    const PREFIX: &[u8] = b"[string \"";
    const SUFFIX: &[u8] = b"\"]";
    let bytes = source_name.as_bytes();
    if let Some(b'=' | b'@') = bytes.first() {
        return bytes[1..].to_vec();
    }
    let available = ID_SIZE.saturating_sub(PREFIX.len() + SUFFIX.len() + 1);
    let first_line = bytes
        .iter()
        .position(|byte| matches!(*byte, b'\n' | b'\r'))
        .unwrap_or(bytes.len());
    let truncated = first_line < bytes.len() || first_line > available;
    let shown = if truncated {
        first_line.min(available.saturating_sub(3))
    } else {
        first_line
    };
    let mut id = Vec::with_capacity(ID_SIZE);
    id.extend_from_slice(PREFIX);
    id.extend_from_slice(&bytes[..shown]);
    if truncated {
        id.extend_from_slice(b"...");
    }
    id.extend_from_slice(SUFFIX);
    id
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TokenKind {
    And,
    Break,
    Do,
    Else,
    ElseIf,
    End,
    Function,
    Return,
    For,
    If,
    In,
    Then,
    Until,
    While,
    Local,
    Global,
    Goto,
    Not,
    Or,
    Repeat,
    Nil,
    True,
    False,
    Identifier(String),
    Integer(i64),
    IntegerBoundary,
    Number(f64),
    /// A `LuaJIT` `LL` literal: a signed 64-bit integer cdata.
    Int64(i64),
    /// A `LuaJIT` `ULL` literal: an unsigned 64-bit integer cdata.
    UInt64(u64),
    String(LuaString),
    Vararg,
    Assign,
    Equal,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    FloorDivide,
    Caret,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Hash,
    Dot,
    Concat,
    Colon,
    DoubleColon,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Semi,
    Eof,
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Token {
    kind: TokenKind,
    span: Span,
}

/// Parse a source chunk.
///
/// # Errors
///
/// Returns a span-aware error when the source is outside this slice's grammar.
pub fn parse(source: &str) -> Result<Chunk, SyntaxError> {
    parse_bytes(source.as_bytes())
}

/// Parser settings selected at the host boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParserConfig {
    compatibility: CompatibilityProfile,
}

impl ParserConfig {
    /// Creates parser settings for one compatibility profile.
    #[must_use]
    pub const fn new(compatibility: CompatibilityProfile) -> Self {
        Self { compatibility }
    }

    /// Returns the selected compatibility profile.
    #[must_use]
    pub const fn compatibility_profile(self) -> CompatibilityProfile {
        self.compatibility
    }
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self::new(CompatibilityProfile::default())
    }
}

/// Parses source text using an explicit compatibility configuration.
///
/// # Errors
///
/// Returns a span-aware error when the source is invalid for the selected
/// profile.
pub fn parse_with_config(source: &str, config: ParserConfig) -> Result<Chunk, SyntaxError> {
    parse_bytes_with_config(source.as_bytes(), config)
}

/// Parse a byte-oriented Lua source chunk.
///
/// Lua source may contain arbitrary bytes inside string literals. Syntax
/// outside those literal forms remains ASCII, so invalid bytes there return a
/// normal span-aware error rather than requiring UTF-8 source text.
///
/// # Errors
///
/// Returns a span-aware error when the source is outside this slice's grammar.
pub fn parse_bytes(source: &[u8]) -> Result<Chunk, SyntaxError> {
    parse_bytes_with_config(source, ParserConfig::default())
}

/// Parses byte-oriented source using an explicit compatibility configuration.
///
/// # Errors
///
/// Returns a span-aware error when the source is invalid for the selected
/// profile.
pub fn parse_bytes_with_config(source: &[u8], config: ParserConfig) -> Result<Chunk, SyntaxError> {
    if let Some(chunk) = fast::parse_complete(source, config) {
        return Ok(chunk);
    }
    let mut parser = StreamingParser::with_config(config);
    match parser.push(source) {
        SourceParsePoll::NeedInput => {}
        SourceParsePoll::Parsed(chunk) => return Ok(chunk),
        SourceParsePoll::Failed(error) => return Err(error),
    }
    parser.end_owned()
}

/// Parses like [`parse_bytes_with_config`] and also returns the tree's
/// `nesting_depth` when the whole-source parser produced it. `None` means the
/// streaming parser did, and the caller measures the tree itself.
///
/// # Errors
///
/// Returns a span-aware error when the source is invalid for the selected
/// profile.
pub(crate) fn parse_bytes_with_depth(
    source: &[u8],
    config: ParserConfig,
) -> Result<(Chunk, Option<usize>), SyntaxError> {
    if let Some((chunk, depth)) = fast::parse_complete_with_depth(source, config) {
        return Ok((chunk, Some(depth)));
    }
    parse_bytes_with_config(source, config).map(|chunk| (chunk, None))
}

/// Parses a complete chunk with the whole-source parser only. `None` means that parser
/// gave up: the source is invalid, unfinished, or needs the streaming parser, and the
/// caller decides what to do next. Unlike [`parse_bytes_with_config`] it never starts
/// the streaming parser, so a caller that holds only a prefix of the source can try it cheaply.
pub(crate) fn parse_complete_only(source: &[u8], config: ParserConfig) -> Option<(Chunk, usize)> {
    fast::parse_complete_with_depth(source, config)
}

mod chunk_machine;
mod depth;
mod fast;
pub(crate) use fast::{Tokens, line_starts};
mod frames;
mod lexer;
mod pratt;
mod streaming;

// Four independent dialect feature switches; enum states would obscure them.
#[allow(clippy::struct_excessive_bools)]
struct Lexer<'a> {
    source: &'a [u8],
    position: usize,
    lua52_escapes: bool,
    unicode_escapes: bool,
    integer_literals: bool,
    cint64_literals: bool,
}

#[derive(Clone, Copy)]
enum Boundary {
    NeedInput,
    Complete(usize),
}

enum PendingBoundary {
    Trivia(TriviaBoundary),
    Token(TokenBoundary),
}

struct TriviaBoundary {
    cursor: usize,
    state: TriviaState,
}

enum TriviaState {
    Space,
    LineComment,
    LongOpening { equals: usize },
    LongComment { level: usize, matched: usize },
}

struct TokenBoundary {
    start: usize,
    state: TokenState,
}

enum TokenState {
    Fixed {
        required: usize,
    },
    Identifier {
        cursor: usize,
    },
    Number {
        cursor: usize,
        mode: NumberMode,
        exponent_sign_allowed: bool,
    },
    Dot,
    ShortString {
        cursor: usize,
        quote: u8,
        escaped: bool,
        skip_whitespace: bool,
        escaped_newline: Option<u8>,
    },
    LongOpening {
        cursor: usize,
        equals: usize,
    },
    LongString {
        cursor: usize,
        level: usize,
        matched: usize,
    },
    Complete,
}

#[derive(Clone, Copy)]
enum NumberMode {
    Decimal,
    Hexadecimal,
}

/// Incremental token producer used by the future streaming parser.
///
/// It retains only the unconsumed suffix. A token is emitted only once its
/// final lookahead is available, or after an explicit end marker. In
/// particular, a lone `*` is held until one following byte arrives.
#[allow(clippy::struct_excessive_bools)] // These flags are independent compatibility switches, not lexer states.
struct IncrementalLexer {
    bytes: Vec<u8>,
    /// Start of the unconsumed suffix in `bytes`.
    ///
    /// Advancing this cursor is constant-time. `compact` occasionally moves
    /// the suffix to the front, so a large reader fragment is not shifted once
    /// for every token it contains.
    cursor: usize,
    pending_boundary: Option<PendingBoundary>,
    shebang_scan: usize,
    offset: usize,
    ended: bool,
    emitted_eof: bool,
    lua52_escapes: bool,
    unicode_escapes: bool,
    integer_literals: bool,
    cint64_literals: bool,
}

enum LexPoll {
    NeedInput,
    Token(Token),
    Eof,
}

/// A resumable byte-oriented Lua source parser.
///
/// Bytes are consumed once by the incremental lexer and immediately handed to
/// the statement machine. Call [`StreamingParser::end`] exactly once after the
/// final fragment. Terminal results are stable, so callers can safely inspect
/// them again without requesting more source from a reader.
#[allow(clippy::struct_excessive_bools)] // Profile switches are cached for the per-token checks.
pub struct StreamingParser {
    capabilities: ParserCapabilities,
    luajit: bool,
    lua55: bool,
    /// Whether `goto` and labels are syntax in this profile.
    goto_keyword: bool,
    lexer: IncrementalLexer,
    machine: ChunkMachine,
    previous_token_can_end_statement: bool,
    previous_token_vararg: bool,
    named_vararg_seen: bool,
    previous_token_end: Option<usize>,
    depth: SyntaxDepth,
    source_len: usize,
    line_starts: Vec<usize>,
    trailing_line_ending: Option<u8>,
    terminal: Option<Result<Chunk, SyntaxError>>,
    completion_mode: CompletionMode,
    /// A lexed token to handle before polling the lexer again. `LuaJIT`'s
    /// contextual `goto` looks one token ahead.
    queued: Option<LexPoll>,
}

/// Lua's nesting limits, counted token by token.
#[derive(Default)]
struct SyntaxDepth {
    delimiter_depth: usize,
    block_depth: usize,
    function_depth: usize,
    block_kinds: Vec<bool>,
    first_nested_function_line: Option<usize>,
    right_associative_depth: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CompletionMode {
    RetainForPolling,
    Transfer,
}

/// The outcome of advancing a [`StreamingParser`].
#[derive(Clone, Debug, PartialEq)]
pub enum SourceParsePoll {
    /// More source bytes or an explicit end marker are required.
    NeedInput,
    /// A complete chunk was parsed.
    Parsed(Chunk),
    /// Source is invalid and no more input must be requested.
    Failed(SyntaxError),
}

/// Token-demand expression reducer used by the incremental parser migration.
///
/// It owns scalar primaries, unary and binary operators, parenthesized forms,
/// suffixes, calls, and table constructors. Function bodies remain with the
/// statement parser until their continuation frames are migrated. It never
/// hands its input back to `Parser`.
///
/// Nested expressions (parentheses, brackets, call arguments and table
/// fields) do not get machines of their own. Each opens a [`PrattLevel`] that
/// owns the top of the shared `values` and `operators` stacks, so a token
/// goes straight to the innermost level.
#[allow(dead_code)]
struct PrattMachine {
    source_len: usize,
    values: Vec<Expr>,
    operators: Vec<PrattOperator>,
    /// The outermost expression.
    root: PrattLevel,
    /// Open nested expressions, innermost last. Every level below the
    /// innermost one waits in `PrattLevel::awaiting` for the level above.
    nested: Vec<PrattLevel>,
    terminal: Option<Result<Expr, SyntaxError>>,
    transitions: usize,
    vararg_allowed: bool,
}

/// One expression being reduced: the stack entries at or above its bases
/// belong to it.
struct PrattLevel {
    values_base: usize,
    operators_base: usize,
    expects_operand: bool,
    pending: Option<PrattPending>,
    /// Set while the next level is an open nested expression whose value
    /// this level receives.
    awaiting: Option<PrattAwait>,
}

struct PrattAwait {
    terminator: PrattTerminator,
    target: PrattValueTarget,
}

#[allow(dead_code)]
enum PrattOperator {
    Unary {
        op: UnaryOp,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        precedence: u8,
        span: Span,
    },
}

#[allow(dead_code)]
enum PrattPending {
    Field { base: Box<Expr> },
    Method { receiver: Box<Expr> },
    Call(PrattCall),
    Table(PrattTable),
    Function(Box<FunctionFrame>),
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum PrattTerminator {
    Parenthesis,
    Bracket,
    CommaOrParenthesis,
    CommaOrBrace,
    BracketThenAssign,
}

#[allow(dead_code)]
enum PrattValueTarget {
    Primary,
    Parenthesized(Span),
    Index(Box<Expr>),
    /// An argument of the level's pending call.
    CallArgument,
    TableCall(Box<PrattCall>),
    /// A positional field of the level's pending table.
    TableList,
    /// A bracketed key of the level's pending table.
    TableKey,
    /// The value for the key or name in the pending table's state.
    TableValue,
}

#[allow(dead_code)]
enum PrattCallTarget {
    Plain(Box<Expr>),
    Method { receiver: Box<Expr>, method: Name },
}

#[allow(dead_code)]
struct PrattCall {
    argument_span: Span,
    target: PrattCallTarget,
    arguments: Vec<Expr>,
    parenthesized: bool,
    expect_argument: bool,
    end: Option<Span>,
}

#[allow(dead_code)]
struct PrattTable {
    open: Span,
    fields: Vec<TableField>,
    state: PrattTableState,
    target: Box<PrattValueTarget>,
}

#[allow(dead_code)]
enum PrattTableState {
    Start,
    AfterField,
    NameOrList(Name),
    KeyAwaitAssign(Box<Expr>),
    NameAwaitValue(Name),
    KeyAwaitValue(Box<Expr>),
}

#[allow(dead_code)]
enum PrattInput {
    Token(Token),
    End,
}

#[allow(dead_code)]
enum PrattPoll {
    NeedToken,
    Parsed(Expr),
    Boundary(Expr, Token),
    Failed(SyntaxError),
}

/// Incremental root-statement staging parser.
///
/// This intentionally covers the non-block statement core first. It consumes
/// expressions through `PrattMachine::Boundary`, so it never stores and
/// replays an expression prefix. Block/control-flow frames join this machine
/// in the next migration step.
/// A nested block. The root machine detaches each block into
/// `ChunkMachine::blocks` once it opens, leaving `None` here until the block
/// stops and its result comes back.
type BlockSlot = Option<Box<ChunkMachine>>;

#[allow(dead_code)]
struct ChunkMachine {
    source_len: usize,
    statements: Vec<Statement>,
    /// A completed return, or a Lua 5.1 break, ends this block's grammar.
    /// The optional boolean records whether its optional semicolon was read.
    last_statement: Option<bool>,
    legacy_break_terminates: bool,
    /// The supported `LuaJIT` extension permits empty statements after return.
    trailing_empty_statements: bool,
    state: ChunkState,
    /// The expression that the current state reads, if any. A state holds
    /// at most one, so one machine is reset and reused for every expression
    /// in this block.
    expression: PrattMachine,
    stop: Option<ChunkStop>,
    block_start: Span,
    loop_depth: usize,
    vararg_allowed: bool,
    terminal: Option<Result<Chunk, SyntaxError>>,
    transitions: usize,
    /// Open blocks detached from their holders, innermost last. Only the
    /// root machine uses it: a token goes straight to the innermost block,
    /// so its cost does not grow with nesting depth.
    #[allow(clippy::vec_box)] // Blocks move between slots and this stack by pointer.
    blocks: Vec<Box<ChunkMachine>>,
}

#[allow(dead_code)]
enum ChunkState {
    Start,
    LocalName {
        start: Span,
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        prefix_attribute: LocalAttribute,
    },
    LocalAfterNames {
        start: Span,
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        prefix_attribute: LocalAttribute,
    },
    LocalAttributeName {
        start: Span,
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        prefix_attribute: LocalAttribute,
    },
    LocalAttributeClose {
        start: Span,
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        prefix_attribute: LocalAttribute,
        attribute: LocalAttribute,
    },
    LocalValues {
        start: Span,
        names: Vec<Name>,
        attributes: Vec<LocalAttribute>,
        values: Vec<Expr>,
    },
    GlobalStart {
        start: Span,
        const_prefix: bool,
    },
    GlobalPrefixAttributeName {
        start: Span,
    },
    GlobalPrefixAttributeClose {
        start: Span,
    },
    GlobalName {
        start: Span,
        const_prefix: bool,
        names: Vec<Name>,
        const_names: Vec<bool>,
    },
    GlobalAfterNames {
        start: Span,
        const_prefix: bool,
        names: Vec<Name>,
        const_names: Vec<bool>,
    },
    GlobalPostfixAttributeName {
        start: Span,
        names: Vec<Name>,
        const_names: Vec<bool>,
    },
    GlobalPostfixAttributeClose {
        start: Span,
        names: Vec<Name>,
        const_names: Vec<bool>,
    },
    GlobalValues {
        start: Span,
        names: Vec<Name>,
        const_names: Vec<bool>,
        values: Vec<Expr>,
    },
    GlobalFunctionDeclaration {
        start: Span,
        name: Name,
        frame: Box<FunctionFrame>,
    },
    GotoName {
        start: Span,
    },
    LabelSecondColon {
        start: Span,
    },
    LabelName {
        start: Span,
    },
    LabelEndColon {
        start: Span,
        name: Name,
    },
    LabelFinalColon {
        start: Span,
        name: Name,
    },
    ExpressionStatement,
    AssignmentTargets {
        start: Span,
        targets: Vec<Expr>,
    },
    AssignmentValues {
        start: Span,
        targets: Vec<Expr>,
        values: Vec<Expr>,
    },
    ReturnStart {
        start: Span,
    },
    ReturnValues {
        start: Span,
        values: Vec<Expr>,
    },
    DoBody {
        start: Span,
        body: BlockSlot,
    },
    WhileCondition {
        start: Span,
    },
    WhileBody {
        start: Span,
        condition: Box<Expr>,
        body: BlockSlot,
    },
    RepeatBody {
        start: Span,
        body: BlockSlot,
    },
    RepeatCondition {
        start: Span,
        body: Block,
    },
    IfCondition {
        start: Span,
    },
    IfBody {
        start: Span,
        condition: Box<Expr>,
        body: BlockSlot,
    },
    IfElseIfCondition {
        start: Span,
        condition: Box<Expr>,
        body: Block,
        else_ifs: Vec<ElseIf>,
        arm_start: Span,
    },
    IfElseIfBody {
        start: Span,
        condition: Box<Expr>,
        body: Block,
        else_ifs: Vec<ElseIf>,
        arm_start: Span,
        arm_condition: Box<Expr>,
        arm_body: BlockSlot,
    },
    IfElseBody {
        start: Span,
        condition: Box<Expr>,
        body: Block,
        else_ifs: Vec<ElseIf>,
        else_start: Span,
        else_body: BlockSlot,
    },
    ForName {
        start: Span,
    },
    ForAfterName {
        start: Span,
        name: Name,
    },
    ForGenericName {
        start: Span,
        names: Vec<Name>,
    },
    ForGenericAfterNames {
        start: Span,
        names: Vec<Name>,
    },
    ForGenericExpressions {
        start: Span,
        names: Vec<Name>,
        expressions: Vec<Expr>,
    },
    ForNumericStart {
        start: Span,
        name: Name,
    },
    ForNumericLimit {
        start: Span,
        name: Name,
        initial: Box<Expr>,
    },
    ForNumericStep {
        start: Span,
        name: Name,
        initial: Box<Expr>,
        limit: Box<Expr>,
    },
    ForNumericBody {
        start: Span,
        name: Name,
        initial: Box<Expr>,
        limit: Box<Expr>,
        step: Option<Box<Expr>>,
        body: BlockSlot,
    },
    ForGenericBody {
        start: Span,
        names: Vec<Name>,
        expressions: Vec<Expr>,
        body: BlockSlot,
    },
    FunctionGlobalName {
        start: Span,
    },
    FunctionPath {
        start: Span,
        segments: Vec<Name>,
    },
    FunctionPathSegment {
        start: Span,
        segments: Vec<Name>,
        method: bool,
    },
    FunctionDeclaration {
        start: Span,
        name: FunctionName,
        frame: Box<FunctionFrame>,
    },
    LocalFunctionName {
        start: Span,
    },
    LocalFunctionDeclaration {
        start: Span,
        name: Name,
        frame: Box<FunctionFrame>,
    },
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum ChunkStop {
    End,
    Until,
    IfArm,
}

#[allow(dead_code)]
enum ChunkPoll {
    NeedToken,
    Parsed(Chunk),
    Stopped(ChunkStopped),
    Failed(SyntaxError),
}

/// A child block consumes its own closing token and reports it upward.
///
/// The parent must never feed the terminator into another child. This is what
/// makes an `elseif` arm unambiguous when input arrives one token at a time.
#[allow(dead_code)]
struct ChunkStopped {
    block: Block,
    terminator: ChunkTerminator,
}

#[allow(dead_code)]
enum ChunkTerminator {
    End(Span),
    Until(Span),
    ElseIf(Span),
    Else(Span),
}

struct ParsedFunction {
    parameters: Vec<Name>,
    vararg: bool,
    vararg_name: Option<Name>,
    body: Block,
    end: Span,
}

enum FunctionFramePoll {
    NeedToken,
    Parsed(ParsedFunction),
    Failed(SyntaxError),
}

struct FunctionFrame {
    source_len: usize,
    start: Span,
    state: FunctionFrameState,
}

enum FunctionFrameState {
    ExpectOpen,
    Parameters {
        open: Span,
        parameters: Vec<Name>,
    },
    AfterParameter {
        open: Span,
        parameters: Vec<Name>,
    },
    AfterVararg {
        open: Span,
        parameters: Vec<Name>,
    },
    AfterVarargName {
        open: Span,
        parameters: Vec<Name>,
        vararg_name: Name,
    },
    Body {
        open: Span,
        parameters: Vec<Name>,
        vararg: bool,
        vararg_name: Option<Name>,
        body: BlockSlot,
    },
}

#[cfg(test)]
mod tests;
