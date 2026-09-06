//! Incremental streaming parser that accepts source bytes and yields chunks.

use crate::compat::CompatibilityProfile;
use crate::syntax::{
    Chunk, ChunkMachine, ChunkPoll, CompletionMode, Expr, IncrementalLexer, LexPoll, ParserConfig,
    PrattInput, SourceParsePoll, Span, Statement, StreamingParser, SyntaxDepth, SyntaxError,
    TableField, Token, TokenKind,
};

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingParser {
    /// Creates an empty source parser.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ParserConfig::default())
    }

    /// Creates an empty source parser for one compatibility profile.
    #[must_use]
    pub fn with_profile(compatibility: CompatibilityProfile) -> Self {
        Self::with_config(ParserConfig::new(compatibility))
    }

    /// Creates an empty source parser with explicit settings.
    #[must_use]
    pub fn with_config(config: ParserConfig) -> Self {
        let compatibility = config.compatibility_profile();
        // LuaJIT 2.1 unconditionally accepts the Lua 5.2 `\x` and `\z` and
        // the Lua 5.3 `\u{...}` escapes.
        let luajit = compatibility == CompatibilityProfile::LuaJit21;
        let lua52_escapes = !compatibility.is_lua51_family() || luajit;
        let unicode_escapes = compatibility.is_lua53_or_later() || luajit;
        let integer_literals = compatibility.is_lua53_or_later();
        let mut machine = ChunkMachine::new(0);
        machine.legacy_break_terminates = compatibility.is_lua51_family();
        machine.trailing_empty_statements = luajit;
        Self {
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
            lexer: IncrementalLexer::with_escapes(
                lua52_escapes,
                unicode_escapes,
                integer_literals,
                luajit,
            ),
            machine,
            previous_token_can_end_statement: false,
            previous_token_vararg: false,
            named_vararg_seen: false,
            previous_token_end: None,
            depth: SyntaxDepth::default(),
            queued: None,
            source_len: 0,
            line_starts: vec![0],
            trailing_line_ending: None,
            terminal: None,
            completion_mode: CompletionMode::RetainForPolling,
        }
    }

    /// Supplies one byte fragment.
    #[must_use]
    pub fn push(&mut self, bytes: &[u8]) -> SourceParsePoll {
        if let Some(result) = &self.terminal {
            return Self::poll(result.clone());
        }
        let start = self.source_len;
        self.record_line_starts(start, bytes);
        self.source_len = self.source_len.saturating_add(bytes.len());
        self.machine.set_source_len(self.source_len);
        self.lexer.feed(bytes);
        self.drain(false)
    }

    /// Returns the completed top-level statements in source order.
    ///
    /// An unfinished statement and statements inside an unfinished block are
    /// excluded. The returned prefix grows as input arrives and remains part
    /// of the complete AST returned at EOF. Consumers can track their own
    /// statement count to process each statement once without cloning it.
    /// A later syntax error can still invalidate the source.
    #[must_use]
    pub fn completed_statements(&self) -> &[Statement] {
        match &self.terminal {
            Some(Ok(chunk)) => &chunk.statements,
            _ => &self.machine.statements,
        }
    }

    /// Moves completed top-level statements out for a downstream compiler.
    ///
    /// The final chunk contains only statements not transferred here. Syntax
    /// validation still covers transferred statements, including named varargs.
    /// Callers must retain the transferred statements if they need a full tree.
    ///
    /// # Errors
    /// Returns an unsupported named-vararg declaration before transferring it.
    pub fn take_completed_statements(&mut self) -> Result<Vec<Statement>, SyntaxError> {
        if !self.lua55
            && self.named_vararg_seen
            && let Some(span) = named_vararg_span(self.completed_statements())
        {
            let error = SyntaxError::new(span, "named varargs require Lua 5.5");
            self.fail(error.clone());
            return Err(error);
        }
        Ok(match &mut self.terminal {
            Some(Ok(chunk)) => core::mem::take(&mut chunk.statements),
            _ => core::mem::take(&mut self.machine.statements),
        })
    }

    /// Returns the byte offsets of source line starts seen so far.
    ///
    /// The first entry is zero. A line ending split across fragments can
    /// adjust the final entry on the next push, so consumers should refresh
    /// their line metadata before compiling newly completed statements.
    #[must_use]
    pub fn line_starts(&self) -> &[usize] {
        &self.line_starts
    }

    /// Marks the source stream complete.
    #[must_use]
    pub fn end(&mut self) -> SourceParsePoll {
        if let Some(result) = &self.terminal {
            return Self::poll(result.clone());
        }
        self.machine.set_source_len(self.source_len);
        self.lexer.end();
        self.drain(true)
    }

    /// Ends a parser that will not be polled again and transfers its AST.
    ///
    /// Unlike [`Self::end`], this consumes the parser. That makes repeated
    /// polling impossible, so it does not retain a second owned copy of a
    /// successful chunk merely to satisfy that API contract.
    ///
    /// # Errors
    ///
    /// Returns the terminal syntax error when the completed source is invalid.
    pub fn end_owned(mut self) -> Result<Chunk, SyntaxError> {
        if let Some(result) = self.terminal.take() {
            return result;
        }
        self.completion_mode = CompletionMode::Transfer;
        self.machine.set_source_len(self.source_len);
        self.lexer.end();
        match self.drain(true) {
            SourceParsePoll::Parsed(chunk) => Ok(chunk),
            SourceParsePoll::Failed(error) => Err(error),
            SourceParsePoll::NeedInput => unreachable!("ended parser cannot require more input"),
        }
    }

    #[allow(clippy::too_many_lines)] // This is the single transition loop for the resumable parser.
    pub(crate) fn drain(&mut self, ended: bool) -> SourceParsePoll {
        loop {
            let lexical = match self.queued.take() {
                Some(queued) => queued,
                None => match self.lexer.poll() {
                    Ok(poll) => poll,
                    Err(error) => return self.fail(error),
                },
            };
            match lexical {
                LexPoll::NeedInput => return SourceParsePoll::NeedInput,
                LexPoll::Token(token) => {
                    let token = if self.luajit && matches!(token.kind, TokenKind::Goto) {
                        // LuaJIT reads `goto` as a statement only before a
                        // label name, so `local goto = 1` stays valid.
                        let next = match self.lexer.poll() {
                            Ok(LexPoll::NeedInput) => {
                                self.queued = Some(LexPoll::Token(token));
                                return SourceParsePoll::NeedInput;
                            }
                            Ok(next) => next,
                            Err(error) => return self.fail(error),
                        };
                        let statement = matches!(
                            &next,
                            LexPoll::Token(Token {
                                kind: TokenKind::Identifier(_),
                                ..
                            })
                        );
                        self.queued = Some(next);
                        if statement {
                            token
                        } else {
                            Token {
                                kind: TokenKind::Identifier("goto".to_owned()),
                                span: token.span,
                            }
                        }
                    } else {
                        token
                    };
                    let token = match self.admit(token) {
                        Ok(token) => token,
                        Err(error) => return self.fail(error),
                    };
                    // A named vararg is `...` followed by a name. Only then
                    // can `complete` find one, so most chunks skip that walk.
                    self.named_vararg_seen |= self.previous_token_vararg
                        && matches!(token.kind, TokenKind::Identifier(_));
                    self.previous_token_vararg = matches!(token.kind, TokenKind::Vararg);
                    self.previous_token_can_end_statement =
                        Self::token_can_end_statement(&token.kind);
                    self.previous_token_end = Some(token.span.end);
                    match self.machine.resume(PrattInput::Token(token)) {
                        ChunkPoll::NeedToken => {}
                        ChunkPoll::Parsed(chunk) => return self.complete(chunk),
                        ChunkPoll::Stopped(_) => unreachable!("root parser cannot stop"),
                        ChunkPoll::Failed(error) => return self.fail(error),
                    }
                }
                LexPoll::Eof => {
                    if !ended {
                        return SourceParsePoll::NeedInput;
                    }
                    return match self.machine.resume(PrattInput::End) {
                        ChunkPoll::NeedToken => {
                            unreachable!("ended root parser cannot need a token")
                        }
                        ChunkPoll::Parsed(chunk) => self.complete(chunk),
                        ChunkPoll::Stopped(_) => unreachable!("root parser cannot stop"),
                        ChunkPoll::Failed(error) => self.fail(error),
                    };
                }
            }
        }
    }

    /// Applies the profile's token-level rules before the statement
    /// machine sees a token.
    fn admit(&mut self, token: Token) -> Result<Token, SyntaxError> {
        self.check_syntax_depth(&token)?;
        let capabilities = self.capabilities;
        let token = match token.kind {
            TokenKind::Global if !self.lua55 => Token {
                kind: TokenKind::Identifier("global".to_owned()),
                span: token.span,
            },
            TokenKind::Goto if !self.goto_keyword => Token {
                kind: TokenKind::Identifier("goto".to_owned()),
                span: token.span,
            },
            _ => token,
        };
        let error = match token.kind {
            TokenKind::FloorDivide if !capabilities.integer_division => {
                Some("floor division requires Lua 5.3 or later")
            }
            TokenKind::BitAnd
            | TokenKind::BitOr
            | TokenKind::BitXor
            | TokenKind::ShiftLeft
            | TokenKind::ShiftRight
                if !capabilities.bitwise_operators =>
            {
                Some("bitwise operators require Lua 5.3 or later")
            }
            TokenKind::LessThan if self.machine.accepts_local_prefix_attribute() && !self.lua55 => {
                Some("prefixed local attributes require Lua 5.5")
            }
            TokenKind::LessThan
                if self.machine.accepts_local_attribute() && !capabilities.local_attributes =>
            {
                Some("local attributes require Lua 5.4 or later")
            }
            TokenKind::DoubleColon if self.machine.accepts_label() && !self.goto_keyword => {
                Some("labels require Lua 5.2 or later")
            }
            TokenKind::Semi
                if !capabilities.empty_statements && !self.previous_token_can_end_statement =>
            {
                Some("unexpected ';'")
            }
            TokenKind::LParen
                if capabilities.reject_ambiguous_call_newline
                    && self.line_break_before(token.span.start)
                    && self.machine.accepts_parenthesized_call() =>
            {
                Some("ambiguous syntax (function call x new statement)")
            }
            _ => None,
        };
        match error {
            Some(message) => Err(SyntaxError::new(token.span, message)),
            None => Ok(token),
        }
    }

    pub(crate) fn complete(&mut self, mut chunk: Chunk) -> SourceParsePoll {
        if !self.lua55
            && self.named_vararg_seen
            && let Some(span) = named_vararg_span(&chunk.statements)
        {
            return self.fail(SyntaxError::new(span, "named varargs require Lua 5.5"));
        }
        match self.completion_mode {
            CompletionMode::RetainForPolling => {
                chunk.line_starts.clone_from(&self.line_starts);
                self.terminal = Some(Ok(chunk.clone()));
            }
            CompletionMode::Transfer => {
                chunk.line_starts = core::mem::take(&mut self.line_starts);
            }
        }
        SourceParsePoll::Parsed(chunk)
    }

    pub(crate) const fn token_can_end_statement(kind: &TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Identifier(_)
                | TokenKind::Integer(_)
                | TokenKind::Number(_)
                | TokenKind::Int64(_)
                | TokenKind::UInt64(_)
                | TokenKind::String(_)
                | TokenKind::Nil
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Vararg
                | TokenKind::Break
                | TokenKind::Return
                | TokenKind::End
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::RBrace
        )
    }

    pub(crate) fn line_break_before(&self, token_start: usize) -> bool {
        self.previous_token_end.is_some_and(|previous_end| {
            // Line starts are sorted, so only the first one past the previous
            // token matters.
            let next = self
                .line_starts
                .partition_point(|line_start| *line_start <= previous_end);
            self.line_starts
                .get(next)
                .is_some_and(|line_start| *line_start <= token_start)
        })
    }

    pub(crate) fn check_syntax_depth(&mut self, token: &Token) -> Result<(), SyntaxError> {
        self.depth.check(token, &self.line_starts)
    }

    pub(crate) fn record_line_starts(&mut self, start: usize, bytes: &[u8]) {
        let mut offset = 0;
        if self.trailing_line_ending.is_some_and(|previous| {
            bytes
                .first()
                .is_some_and(|next| *next != previous && matches!(next, b'\r' | b'\n'))
        }) {
            if let Some(line_start) = self.line_starts.last_mut() {
                *line_start = line_start.saturating_add(1);
            }
            offset = 1;
        }
        self.trailing_line_ending = None;
        while let Some(found) = crate::syntax::fast::find_line_break(bytes, offset) {
            offset = found;
            let width = match bytes[offset] {
                b'\n' | b'\r'
                    if bytes.get(offset + 1).is_some_and(|next| {
                        *next != bytes[offset] && matches!(next, b'\r' | b'\n')
                    }) =>
                {
                    2
                }
                _ => 1,
            };
            let line_start = start.saturating_add(offset).saturating_add(width);
            self.line_starts.push(line_start);
            offset += width;
            if width == 1 && offset == bytes.len() {
                self.trailing_line_ending = bytes.last().copied();
            }
        }
    }

    pub(crate) fn fail(&mut self, error: SyntaxError) -> SourceParsePoll {
        self.terminal = Some(Err(error.clone()));
        SourceParsePoll::Failed(error)
    }

    pub(crate) fn poll(result: Result<Chunk, SyntaxError>) -> SourceParsePoll {
        match result {
            Ok(chunk) => SourceParsePoll::Parsed(chunk),
            Err(error) => SourceParsePoll::Failed(error),
        }
    }
}

#[allow(clippy::too_many_lines)] // Iterative AST traversal avoids recursive stack growth on hostile input.
pub(crate) fn named_vararg_span(statements: &[Statement]) -> Option<Span> {
    enum Node<'a> {
        Statements(&'a [Statement]),
        Expressions(&'a [Expr]),
    }

    let mut nodes = vec![Node::Statements(statements)];
    while let Some(node) = nodes.pop() {
        match node {
            Node::Statements(statements) => {
                for statement in statements.iter().rev() {
                    match statement {
                        Statement::Function {
                            vararg_name, body, ..
                        }
                        | Statement::LocalFunction {
                            vararg_name, body, ..
                        } => {
                            if let Some(name) = vararg_name {
                                return Some(name.span);
                            }
                            nodes.push(Node::Statements(&body.statements));
                        }
                        Statement::Local { values, .. }
                        | Statement::Global { values, .. }
                        | Statement::Return { values, .. } => nodes.push(Node::Expressions(values)),
                        Statement::Assign {
                            targets, values, ..
                        } => {
                            nodes.push(Node::Expressions(values));
                            nodes.push(Node::Expressions(targets));
                        }
                        Statement::Call(expression) => {
                            nodes.push(Node::Expressions(core::slice::from_ref(expression)));
                        }
                        Statement::Do(body)
                        | Statement::While { body, .. }
                        | Statement::Repeat { body, .. }
                        | Statement::NumericFor { body, .. }
                        | Statement::GenericFor { body, .. } => {
                            nodes.push(Node::Statements(&body.statements));
                        }
                        Statement::If {
                            body,
                            else_ifs,
                            else_body,
                            ..
                        } => {
                            if let Some(body) = else_body {
                                nodes.push(Node::Statements(&body.statements));
                            }
                            for arm in else_ifs {
                                nodes.push(Node::Statements(&arm.body.statements));
                            }
                            nodes.push(Node::Statements(&body.statements));
                        }
                        Statement::Break { .. }
                        | Statement::Goto { .. }
                        | Statement::Label { .. } => {}
                    }
                }
            }
            Node::Expressions(expressions) => {
                for expression in expressions.iter().rev() {
                    match expression {
                        Expr::Function {
                            vararg_name, body, ..
                        } => {
                            if let Some(name) = vararg_name {
                                return Some(name.span);
                            }
                            nodes.push(Node::Statements(&body.statements));
                        }
                        Expr::Table { fields, .. } => {
                            for field in fields {
                                match field {
                                    TableField::List(value) | TableField::Name { value, .. } => {
                                        nodes.push(Node::Expressions(core::slice::from_ref(value)));
                                    }
                                    TableField::Key { key, value } => {
                                        nodes.push(Node::Expressions(core::slice::from_ref(value)));
                                        nodes.push(Node::Expressions(core::slice::from_ref(key)));
                                    }
                                }
                            }
                        }
                        Expr::Index { table, key, .. }
                        | Expr::Binary {
                            left: table,
                            right: key,
                            ..
                        } => {
                            nodes.push(Node::Expressions(core::slice::from_ref(key)));
                            nodes.push(Node::Expressions(core::slice::from_ref(table)));
                        }
                        Expr::Unary { expression, .. } | Expr::Parenthesized { expression, .. } => {
                            nodes.push(Node::Expressions(core::slice::from_ref(expression)));
                        }
                        Expr::Call {
                            callee, arguments, ..
                        } => {
                            nodes.push(Node::Expressions(arguments));
                            nodes.push(Node::Expressions(core::slice::from_ref(callee)));
                        }
                        Expr::MethodCall {
                            receiver,
                            arguments,
                            ..
                        } => {
                            nodes.push(Node::Expressions(arguments));
                            nodes.push(Node::Expressions(core::slice::from_ref(receiver)));
                        }
                        Expr::Name(_)
                        | Expr::Nil(_)
                        | Expr::Boolean(_, _)
                        | Expr::Integer(_, _)
                        | Expr::IntegerBoundary(_)
                        | Expr::Number(_, _)
                        | Expr::Int64(_, _)
                        | Expr::UInt64(_, _)
                        | Expr::String(_, _)
                        | Expr::Vararg(_) => {}
                    }
                }
            }
        }
    }
    None
}

impl SyntaxDepth {
    /// Counts the token into the nesting levels and rejects it past Lua's
    /// limits. `line_starts` only names the line of a nested function.
    #[inline]
    pub(crate) fn check(
        &mut self,
        token: &Token,
        line_starts: &[usize],
    ) -> Result<(), SyntaxError> {
        // Like Lua's LUAI_MAXCCALLS, delimiters, blocks and functions share
        // one budget of 200 levels; the recursive parser and compiler need
        // well under 1 MiB of stack at that depth. Functions keep a tighter
        // limit of their own because Lua 5.1 diagnoses its 60-upvalue limit
        // inside deeply nested functions.
        const MAX_STRUCTURAL_LEVELS: usize = 200;
        const MAX_FUNCTION_LEVELS: usize = 16;
        const MAX_RIGHT_ASSOCIATIVE_LEVELS: usize = 200;
        match token.kind {
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => {
                self.delimiter_depth += 1;
            }
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => {
                self.delimiter_depth = self.delimiter_depth.saturating_sub(1);
            }
            TokenKind::Do | TokenKind::If | TokenKind::Repeat => {
                self.block_depth += 1;
                self.block_kinds.push(false);
            }
            TokenKind::Function => {
                if self.function_depth == 1 && self.first_nested_function_line.is_none() {
                    self.first_nested_function_line = Some(
                        line_starts.partition_point(|line_start| *line_start <= token.span.start),
                    );
                }
                self.function_depth += 1;
                self.block_kinds.push(true);
            }
            TokenKind::End | TokenKind::Until => {
                if self.block_kinds.pop().unwrap_or(false) {
                    self.function_depth = self.function_depth.saturating_sub(1);
                    if self.function_depth < 2 {
                        self.first_nested_function_line = None;
                    }
                } else {
                    self.block_depth = self.block_depth.saturating_sub(1);
                }
            }
            TokenKind::Caret | TokenKind::Concat => {
                self.right_associative_depth += 1;
            }
            TokenKind::Assign
            | TokenKind::Comma
            | TokenKind::Semi
            | TokenKind::Return
            | TokenKind::Local => {
                self.right_associative_depth = 0;
            }
            _ => {}
        }
        if self.delimiter_depth + self.block_depth + self.function_depth > MAX_STRUCTURAL_LEVELS
            || self.function_depth > MAX_FUNCTION_LEVELS
            || self.right_associative_depth > MAX_RIGHT_ASSOCIATIVE_LEVELS
        {
            let message = self.first_nested_function_line.map_or_else(
                || "chunk has too many syntax levels".to_owned(),
                |line| format!("chunk has too many syntax levels (function at line {line})"),
            );
            Err(SyntaxError::new(token.span, message))
        } else {
            Ok(())
        }
    }
}
