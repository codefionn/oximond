//! Pratt-style expression parsing state machine.

use crate::string::LuaString;
use crate::syntax::{
    BinaryOp, ChunkMachine, ChunkPoll, Expr, FunctionFrame, FunctionFramePoll, Name, PrattAwait,
    PrattCall, PrattCallTarget, PrattInput, PrattLevel, PrattMachine, PrattOperator, PrattPending,
    PrattPoll, PrattTable, PrattTableState, PrattTerminator, PrattValueTarget, Span, SyntaxError,
    TableField, Token, TokenKind, UnaryOp, make_unary_expression,
};

/// Handles one token in the machine's current pending state. Each handler is
/// `#[inline(never)]` so the recursive path only pays for its own frame.
type TokenHandler = fn(&mut PrattMachine, Token) -> PrattPoll;

impl PrattTerminator {
    pub(crate) const fn matches(self, kind: &TokenKind) -> bool {
        match self {
            Self::Parenthesis => matches!(kind, TokenKind::RParen),
            Self::Bracket | Self::BracketThenAssign => matches!(kind, TokenKind::RBracket),
            Self::CommaOrParenthesis => matches!(kind, TokenKind::Comma | TokenKind::RParen),
            Self::CommaOrBrace => {
                matches!(kind, TokenKind::Comma | TokenKind::Semi | TokenKind::RBrace)
            }
        }
    }
}

impl PrattLevel {
    const fn new(values_base: usize, operators_base: usize) -> Self {
        Self {
            values_base,
            operators_base,
            expects_operand: true,
            pending: None,
            awaiting: None,
        }
    }
}

#[allow(dead_code)]
impl PrattMachine {
    pub(crate) const fn new(source_len: usize) -> Self {
        Self::with_vararg(source_len, true)
    }

    pub(crate) const fn with_vararg(source_len: usize, vararg_allowed: bool) -> Self {
        Self {
            source_len,
            values: Vec::new(),
            operators: Vec::new(),
            root: PrattLevel::new(0, 0),
            nested: Vec::new(),
            terminal: None,
            transitions: 0,
            vararg_allowed,
        }
    }

    /// Prepares a finished machine for the next expression, keeping its
    /// stack allocations.
    pub(crate) fn reset(&mut self) {
        self.values.clear();
        self.operators.clear();
        self.root = PrattLevel::new(0, 0);
        self.nested.clear();
        self.terminal = None;
        self.transitions = 0;
    }

    pub(crate) fn resume(&mut self, input: PrattInput) -> PrattPoll {
        if let Some(result) = &self.terminal {
            return Self::poll(result.clone());
        }
        self.transitions += 1;
        match input {
            PrattInput::Token(token) => self.token(token),
            PrattInput::End => self.finish(),
        }
    }

    pub(crate) const fn transition_count(&self) -> usize {
        self.transitions
    }

    pub(crate) fn set_source_len(&mut self, source_len: usize) {
        self.source_len = source_len;
        if let Some(PrattPending::Function(function)) = &mut self.level_mut().pending {
            function.set_source_len(source_len);
        }
    }

    pub(crate) fn accepts_function_parameter_vararg(&self) -> bool {
        match &self.level().pending {
            Some(PrattPending::Function(function)) => function.accepts_vararg_parameter(),
            _ => false,
        }
    }

    pub(crate) fn accepts_parenthesized_call(&self) -> bool {
        let level = self.level();
        match &level.pending {
            Some(PrattPending::Call(call)) => !call.parenthesized,
            Some(PrattPending::Function(function)) => function.accepts_parenthesized_call(),
            None => {
                !level.expects_operand
                    && self.values.len() > level.values_base
                    && self.values.last().is_some_and(Self::is_prefix_expression)
            }
            Some(
                PrattPending::Field { .. } | PrattPending::Method { .. } | PrattPending::Table(_),
            ) => false,
        }
    }

    /// Takes the body block of a function expression once it opens.
    pub(crate) fn take_new_block(&mut self) -> Option<Box<ChunkMachine>> {
        match &mut self.level_mut().pending {
            Some(PrattPending::Function(function)) => function.take_new_block(),
            _ => None,
        }
    }

    /// Gives the pending function expression the result of its body block.
    pub(crate) fn deliver_block(&mut self, poll: ChunkPoll) -> PrattPoll {
        let Some(PrattPending::Function(function)) = &mut self.level_mut().pending else {
            unreachable!("only a function expression holds a block")
        };
        match function.deliver_block(poll) {
            FunctionFramePoll::NeedToken => PrattPoll::NeedToken,
            poll => match self.finish_pending_function(poll) {
                PrattPoll::Failed(error) => self.set_failure(error),
                poll => poll,
            },
        }
    }

    /// The innermost open expression.
    fn level(&self) -> &PrattLevel {
        self.nested.last().unwrap_or(&self.root)
    }

    fn level_mut(&mut self) -> &mut PrattLevel {
        self.nested.last_mut().unwrap_or(&mut self.root)
    }

    /// The level waiting for the innermost nested expression.
    fn parent_level(&self) -> &PrattLevel {
        match self.nested.len() {
            0 => unreachable!("the root expression has no parent"),
            1 => &self.root,
            len => &self.nested[len - 2],
        }
    }

    /// Delivers a token to the innermost expression. A nested expression that
    /// could end here closes first when the token is its terminator; its value
    /// then goes to the enclosing level, which handles the terminator itself.
    pub(crate) fn token(&mut self, token: Token) -> PrattPoll {
        if let Some(inner) = self.nested.last()
            && inner.pending.is_none()
            && !inner.expects_operand
        {
            let Some(awaiting) = &self.parent_level().awaiting else {
                unreachable!("a nested expression has a waiting parent")
            };
            if awaiting.terminator.matches(&token.kind) {
                return self.finish_nested(token);
            }
        }
        let nested = !self.nested.is_empty();
        match self.dispatch(token) {
            // Rebuild the common result rather than moving the whole poll.
            PrattPoll::NeedToken => PrattPoll::NeedToken,
            // A nested expression must end at its own terminator.
            PrattPoll::Boundary(_, boundary) if nested => {
                self.fail(boundary.span, "unexpected token in expression")
            }
            PrattPoll::Failed(error) => self.set_failure(error),
            poll => poll,
        }
    }

    // Handle each pending state in its own `#[inline(never)]` method behind a
    // single call site. Function bodies nest chunk machines, which resume this
    // machine recursively, and debug builds give every binding in a function
    // its own stack slot.
    fn dispatch(&mut self, token: Token) -> PrattPoll {
        let handler = self.token_handler();
        handler(self, token)
    }

    fn token_handler(&self) -> TokenHandler {
        let level = self.level();
        match &level.pending {
            None if level.expects_operand => Self::on_operand,
            None => Self::on_operator,
            Some(PrattPending::Field { .. }) => Self::on_pending_field,
            Some(PrattPending::Method { .. }) => Self::on_pending_method,
            Some(PrattPending::Call(_)) => Self::on_pending_call,
            Some(PrattPending::Table(_)) => Self::on_pending_table,
            Some(PrattPending::Function(_)) => Self::on_pending_function,
        }
    }

    pub(crate) fn start_expression(
        &mut self,
        terminator: PrattTerminator,
        target: PrattValueTarget,
    ) {
        self.level_mut().awaiting = Some(PrattAwait { terminator, target });
        self.nested
            .push(PrattLevel::new(self.values.len(), self.operators.len()));
    }

    pub(crate) fn start_table(&mut self, open: Span, target: PrattValueTarget) {
        self.level_mut().pending = Some(PrattPending::Table(PrattTable {
            open,
            fields: Vec::new(),
            state: PrattTableState::Start,
            target: Box::new(target),
        }));
    }

    fn push_operand(&mut self, value: Expr) {
        self.values.push(value);
        self.level_mut().expects_operand = false;
    }

    fn push_unary(&mut self, op: UnaryOp, span: Span) {
        self.operators.push(PrattOperator::Unary { op, span });
    }

    #[inline(never)]
    fn on_operand(&mut self, token: Token) -> PrattPoll {
        match token.kind {
            TokenKind::Minus => self.push_unary(UnaryOp::Negate, token.span),
            TokenKind::Not => self.push_unary(UnaryOp::Not, token.span),
            TokenKind::Hash => self.push_unary(UnaryOp::Length, token.span),
            TokenKind::BitXor => self.push_unary(UnaryOp::BitNot, token.span),
            TokenKind::Identifier(text) => self.push_operand(Expr::Name(Name {
                text: text.into(),
                span: token.span,
            })),
            TokenKind::Nil => self.push_operand(Expr::Nil(token.span)),
            TokenKind::True => self.push_operand(Expr::Boolean(true, token.span)),
            TokenKind::False => self.push_operand(Expr::Boolean(false, token.span)),
            TokenKind::Integer(value) => self.push_operand(Expr::Integer(value, token.span)),
            TokenKind::IntegerBoundary => self.push_operand(Expr::IntegerBoundary(token.span)),
            TokenKind::Number(value) => self.push_operand(Expr::Number(value, token.span)),
            TokenKind::Int64(value) => self.push_operand(Expr::Int64(value, token.span)),
            TokenKind::UInt64(value) => self.push_operand(Expr::UInt64(value, token.span)),
            TokenKind::String(value) => self.push_operand(Expr::String(value, token.span)),
            // Nested expressions always admit `...`; the statement machine
            // rejects it outside vararg functions.
            TokenKind::Vararg if self.vararg_allowed || !self.nested.is_empty() => {
                self.push_operand(Expr::Vararg(token.span));
            }
            TokenKind::Vararg => {
                return self.fail(token.span, "cannot use '...' outside a vararg function");
            }
            TokenKind::LParen => {
                self.start_expression(
                    PrattTerminator::Parenthesis,
                    PrattValueTarget::Parenthesized(token.span),
                );
            }
            TokenKind::LBrace => self.start_table(token.span, PrattValueTarget::Primary),
            TokenKind::Function => {
                self.level_mut().pending = Some(PrattPending::Function(Box::new(
                    FunctionFrame::new(self.source_len, token.span),
                )));
            }
            _ => return self.fail(token.span, "expected expression"),
        }
        PrattPoll::NeedToken
    }

    #[inline(never)]
    fn on_operator(&mut self, token: Token) -> PrattPoll {
        match token.kind {
            TokenKind::LBracket => {
                let Some(base) = self.take_prefix_expression(token.span) else {
                    return self.fail(token.span, "expected prefix expression");
                };
                self.start_expression(PrattTerminator::Bracket, PrattValueTarget::Index(base));
                return PrattPoll::NeedToken;
            }
            TokenKind::Dot => {
                let Some(base) = self.take_prefix_expression(token.span) else {
                    return self.fail(token.span, "expected prefix expression");
                };
                self.level_mut().pending = Some(PrattPending::Field { base });
                return PrattPoll::NeedToken;
            }
            TokenKind::Colon => {
                let Some(receiver) = self.take_prefix_expression(token.span) else {
                    return self.fail(token.span, "expected prefix expression");
                };
                self.level_mut().pending = Some(PrattPending::Method { receiver });
                return PrattPoll::NeedToken;
            }
            TokenKind::LParen => {
                let Some(callee) = self.take_prefix_expression(token.span) else {
                    // A parenthesized expression may begin the next statement
                    // immediately after a non-prefix RHS such as `{}`.
                    return self.finish_at_boundary(token);
                };
                self.level_mut().pending = Some(PrattPending::Call(PrattCall {
                    argument_span: token.span,
                    target: PrattCallTarget::Plain(callee),
                    arguments: Vec::new(),
                    parenthesized: true,
                    expect_argument: true,
                    end: None,
                }));
                return PrattPoll::NeedToken;
            }
            TokenKind::LBrace => {
                let Some(callee) = self.take_prefix_expression(token.span) else {
                    return self.fail(token.span, "expected prefix expression");
                };
                self.start_table(
                    token.span,
                    PrattValueTarget::TableCall(Box::new(PrattCall {
                        argument_span: token.span,
                        target: PrattCallTarget::Plain(callee),
                        arguments: Vec::new(),
                        parenthesized: false,
                        expect_argument: false,
                        end: None,
                    })),
                );
                return PrattPoll::NeedToken;
            }
            TokenKind::String(value) => {
                let Some(callee) = self.take_prefix_expression(token.span) else {
                    return self.fail(token.span, "expected prefix expression");
                };
                self.push_call(PrattCall {
                    argument_span: token.span,
                    target: PrattCallTarget::Plain(callee),
                    arguments: vec![Expr::String(value, token.span)],
                    parenthesized: false,
                    expect_argument: false,
                    end: Some(token.span),
                });
                return PrattPoll::NeedToken;
            }
            _ => {}
        }

        let Some((op, precedence, right_associative)) = Self::binary_operator(&token.kind) else {
            return self.finish_at_boundary(token);
        };
        while self.should_reduce(precedence, right_associative) {
            if let Err(error) = self.reduce() {
                return self.set_failure(error);
            }
        }
        self.operators.push(PrattOperator::Binary {
            op,
            precedence,
            span: token.span,
        });
        self.level_mut().expects_operand = true;
        PrattPoll::NeedToken
    }

    /// Closes the innermost nested expression at its terminator.
    #[inline(never)]
    fn finish_nested(&mut self, token: Token) -> PrattPoll {
        let value = match self.finish_level() {
            Ok(value) => value,
            Err(error) => return self.set_failure(error),
        };
        self.nested.pop();
        let Some(PrattAwait { target, .. }) = self.level_mut().awaiting.take() else {
            unreachable!("a nested expression has a waiting parent")
        };
        self.accept_value(target, value, Some(token))
    }

    #[inline(never)]
    fn on_pending_field(&mut self, token: Token) -> PrattPoll {
        let Some(PrattPending::Field { base }) = self.level_mut().pending.take() else {
            unreachable!("field handler requires a pending field")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.inject_primary(Expr::Index {
                    span: base.span().join(token.span),
                    table: base,
                    key: Box::new(Expr::String(LuaString::from(text.as_str()), token.span)),
                });
                PrattPoll::NeedToken
            }
            _ => self.fail(token.span, "expected field name after '.'"),
        }
    }

    #[inline(never)]
    fn on_pending_method(&mut self, token: Token) -> PrattPoll {
        let Some(PrattPending::Method { receiver }) = self.level_mut().pending.take() else {
            unreachable!("method handler requires a pending method")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.level_mut().pending = Some(PrattPending::Call(PrattCall {
                    argument_span: token.span,
                    target: PrattCallTarget::Method {
                        receiver,
                        method: Name {
                            text: text.into(),
                            span: token.span,
                        },
                    },
                    arguments: Vec::new(),
                    parenthesized: false,
                    expect_argument: false,
                    end: None,
                }));
                PrattPoll::NeedToken
            }
            _ => self.fail(token.span, "expected method name after ':'"),
        }
    }

    /// The level's pending call. Calls and tables stay in place while their
    /// arguments and fields are parsed.
    fn pending_call(&mut self) -> &mut PrattCall {
        let Some(PrattPending::Call(call)) = &mut self.level_mut().pending else {
            unreachable!("call handler requires a pending call")
        };
        call
    }

    fn pending_table(&mut self) -> &mut PrattTable {
        let Some(PrattPending::Table(table)) = &mut self.level_mut().pending else {
            unreachable!("table handler requires a pending table")
        };
        table
    }

    fn take_pending_call(&mut self) -> PrattCall {
        let Some(PrattPending::Call(call)) = self.level_mut().pending.take() else {
            unreachable!("call handler requires a pending call")
        };
        call
    }

    fn take_pending_table(&mut self) -> PrattTable {
        let Some(PrattPending::Table(table)) = self.level_mut().pending.take() else {
            unreachable!("table handler requires a pending table")
        };
        table
    }

    #[inline(never)]
    fn on_pending_call(&mut self, token: Token) -> PrattPoll {
        let call = self.pending_call();
        if !call.parenthesized {
            call.argument_span = token.span;
            match token.kind {
                TokenKind::LParen => {
                    call.parenthesized = true;
                    call.expect_argument = true;
                }
                TokenKind::String(value) => {
                    call.arguments.push(Expr::String(value, token.span));
                    let call = self.take_pending_call();
                    self.push_call(call);
                }
                TokenKind::LBrace => {
                    let call = self.take_pending_call();
                    self.start_table(token.span, PrattValueTarget::TableCall(Box::new(call)));
                }
                _ => return self.fail(token.span, "expected function arguments"),
            }
            return PrattPoll::NeedToken;
        }
        match token.kind {
            TokenKind::RParen => {
                if call.expect_argument && !call.arguments.is_empty() {
                    return self.fail(token.span, "expected expression after ','");
                }
                call.end = Some(token.span);
                let call = self.take_pending_call();
                self.push_call(call);
                PrattPoll::NeedToken
            }
            TokenKind::Comma => {
                if call.expect_argument {
                    return self.fail(token.span, "expected expression after ','");
                }
                call.expect_argument = true;
                PrattPoll::NeedToken
            }
            _ => {
                call.expect_argument = false;
                self.start_expression(
                    PrattTerminator::CommaOrParenthesis,
                    PrattValueTarget::CallArgument,
                );
                self.token(token)
            }
        }
    }

    #[inline(never)]
    fn on_pending_table(&mut self, token: Token) -> PrattPoll {
        let table = self.pending_table();
        match core::mem::replace(&mut table.state, PrattTableState::Start) {
            PrattTableState::Start => match token.kind {
                TokenKind::RBrace => {
                    let table = self.take_pending_table();
                    self.finish_table(table, token.span)
                }
                TokenKind::LBracket => {
                    self.start_expression(PrattTerminator::Bracket, PrattValueTarget::TableKey);
                    PrattPoll::NeedToken
                }
                TokenKind::Identifier(text) => {
                    table.state = PrattTableState::NameOrList(Name {
                        text: text.into(),
                        span: token.span,
                    });
                    PrattPoll::NeedToken
                }
                kind => {
                    self.start_expression(
                        PrattTerminator::CommaOrBrace,
                        PrattValueTarget::TableList,
                    );
                    self.token(Token {
                        kind,
                        span: token.span,
                    })
                }
            },
            PrattTableState::AfterField => match token.kind {
                TokenKind::Comma | TokenKind::Semi => PrattPoll::NeedToken,
                TokenKind::RBrace => {
                    let table = self.take_pending_table();
                    self.finish_table(table, token.span)
                }
                _ => self.fail(token.span, "expected ',' ';' or '}' in table constructor"),
            },
            PrattTableState::NameOrList(name) => {
                if matches!(token.kind, TokenKind::Assign) {
                    table.state = PrattTableState::NameAwaitValue(name);
                    return PrattPoll::NeedToken;
                }
                let first = Token {
                    kind: TokenKind::Identifier(name.text.to_string()),
                    span: name.span,
                };
                self.start_expression(PrattTerminator::CommaOrBrace, PrattValueTarget::TableList);
                if let PrattPoll::Failed(error) = self.token(first) {
                    return PrattPoll::Failed(error);
                }
                self.token(token)
            }
            PrattTableState::KeyAwaitAssign(key) => {
                if !matches!(token.kind, TokenKind::Assign) {
                    return self.fail(token.span, "expected '=' after table key");
                }
                table.state = PrattTableState::KeyAwaitValue(key);
                PrattPoll::NeedToken
            }
            state @ (PrattTableState::NameAwaitValue(_) | PrattTableState::KeyAwaitValue(_)) => {
                table.state = state;
                self.start_expression(PrattTerminator::CommaOrBrace, PrattValueTarget::TableValue);
                self.token(token)
            }
        }
    }

    // Function bodies nest chunk machines, so this is on a recursive path
    // too; the parsed-function result is handled in its own frame.
    #[inline(never)]
    fn on_pending_function(&mut self, token: Token) -> PrattPoll {
        let Some(PrattPending::Function(function)) = &mut self.level_mut().pending else {
            unreachable!("function handler requires a pending function")
        };
        match function.resume(token) {
            FunctionFramePoll::NeedToken => PrattPoll::NeedToken,
            poll => self.finish_pending_function(poll),
        }
    }

    #[inline(never)]
    fn finish_pending_function(&mut self, poll: FunctionFramePoll) -> PrattPoll {
        let Some(PrattPending::Function(function)) = self.level_mut().pending.take() else {
            unreachable!("function handler requires a pending function")
        };
        match poll {
            FunctionFramePoll::Parsed(parsed) => {
                self.inject_primary(Expr::Function {
                    parameters: parsed.parameters,
                    vararg: parsed.vararg,
                    vararg_name: parsed.vararg_name.map(Box::new),
                    span: function.start.join(parsed.end),
                    body: Box::new(parsed.body),
                });
                PrattPoll::NeedToken
            }
            FunctionFramePoll::Failed(error) => self.set_failure(error),
            FunctionFramePoll::NeedToken => unreachable!("pending function still needs a token"),
        }
    }

    pub(crate) fn finish_table(&mut self, table: PrattTable, close: Span) -> PrattPoll {
        let value = Expr::Table {
            fields: table.fields,
            span: table.open.join(close),
        };
        self.accept_value(*table.target, value, None)
    }

    pub(crate) fn accept_value(
        &mut self,
        target: PrattValueTarget,
        value: Expr,
        delimiter: Option<Token>,
    ) -> PrattPoll {
        match target {
            PrattValueTarget::Primary => {
                self.inject_primary(value);
                PrattPoll::NeedToken
            }
            PrattValueTarget::Parenthesized(open) => {
                let close = delimiter
                    .expect("parenthesized expression has a close delimiter")
                    .span;
                self.inject_primary(Expr::Parenthesized {
                    span: open.join(close),
                    expression: Box::new(value),
                });
                PrattPoll::NeedToken
            }
            PrattValueTarget::Index(base) => {
                let close = delimiter
                    .expect("indexed expression has a close bracket")
                    .span;
                self.inject_primary(Expr::Index {
                    span: base.span().join(close),
                    table: base,
                    key: Box::new(value),
                });
                PrattPoll::NeedToken
            }
            PrattValueTarget::CallArgument => {
                self.pending_call().arguments.push(value);
                self.token(delimiter.expect("call argument has a delimiter"))
            }
            PrattValueTarget::TableCall(mut call) => {
                call.arguments.push(value);
                call.end = call.arguments.last().map(Expr::span);
                self.push_call(*call);
                PrattPoll::NeedToken
            }
            PrattValueTarget::TableList => {
                let table = self.pending_table();
                table.fields.push(TableField::List(value));
                table.state = PrattTableState::AfterField;
                self.token(delimiter.expect("table field has a delimiter"))
            }
            PrattValueTarget::TableKey => {
                self.pending_table().state = PrattTableState::KeyAwaitAssign(Box::new(value));
                PrattPoll::NeedToken
            }
            PrattValueTarget::TableValue => {
                let table = self.pending_table();
                let field = match core::mem::replace(&mut table.state, PrattTableState::AfterField)
                {
                    PrattTableState::KeyAwaitValue(key) => TableField::Key { key: *key, value },
                    PrattTableState::NameAwaitValue(name) => TableField::Name { name, value },
                    _ => unreachable!("a table value follows a key or a name"),
                };
                table.fields.push(field);
                self.token(delimiter.expect("table field has a delimiter"))
            }
        }
    }

    pub(crate) fn inject_primary(&mut self, expression: Expr) {
        self.push_operand(expression);
    }

    pub(crate) fn push_call(&mut self, call: PrattCall) {
        let end = call.end.or_else(|| call.arguments.last().map(Expr::span));
        let expression = match call.target {
            PrattCallTarget::Plain(callee) => Expr::Call {
                span: callee.span().join(end.unwrap_or_else(|| callee.span())),
                callee,
                arguments: call.arguments,
            },
            PrattCallTarget::Method { receiver, method } => Expr::MethodCall {
                argument_span: call.argument_span,
                span: receiver.span().join(end.unwrap_or(method.span)),
                receiver,
                method: Box::new(method),
                arguments: call.arguments,
            },
        };
        self.inject_primary(expression);
    }

    /// Pops the level's last value when it can take a suffix. The value is
    /// boxed because every suffix stores it boxed in the tree.
    pub(crate) fn take_prefix_expression(&mut self, span: Span) -> Option<Box<Expr>> {
        let _ = span;
        if self.values.len() > self.level().values_base
            && self.values.last().is_some_and(Self::is_prefix_expression)
        {
            self.values.pop().map(Box::new)
        } else {
            None
        }
    }

    pub(crate) const fn is_prefix_expression(expression: &Expr) -> bool {
        matches!(
            expression,
            Expr::Name(_)
                | Expr::Index { .. }
                | Expr::Call { .. }
                | Expr::MethodCall { .. }
                | Expr::Parenthesized { .. }
        )
    }

    /// Ends the outermost expression at the end of input. Open nested
    /// expressions are abandoned, as their values never reach it.
    pub(crate) fn finish(&mut self) -> PrattPoll {
        if let Some(first) = self.nested.first() {
            self.values.truncate(first.values_base);
            self.operators.truncate(first.operators_base);
            self.nested.clear();
        }
        match self.finish_level() {
            Ok(expression) => PrattPoll::Parsed(expression),
            Err(error) => self.set_failure(error),
        }
    }

    /// Reduces the innermost level to its single value.
    fn finish_level(&mut self) -> Result<Expr, SyntaxError> {
        let end = Span {
            start: self.source_len,
            end: self.source_len,
        };
        let level = self.level();
        if level.expects_operand {
            return Err(SyntaxError::new(end, "expected expression"));
        }
        let (values_base, operators_base) = (level.values_base, level.operators_base);
        while self.operators.len() > operators_base {
            self.reduce()?;
        }
        if self.values.len() == values_base + 1 {
            Ok(self.values.pop().expect("level holds one value"))
        } else {
            Err(SyntaxError::new(end, "invalid expression state"))
        }
    }

    pub(crate) fn finish_at_boundary(&mut self, token: Token) -> PrattPoll {
        match self.finish_level() {
            Ok(expression) => PrattPoll::Boundary(expression, token),
            Err(error) => self.set_failure(error),
        }
    }

    pub(crate) fn binary_operator(kind: &TokenKind) -> Option<(BinaryOp, u8, bool)> {
        let (op, precedence, right_associative) = match kind {
            TokenKind::Or => (BinaryOp::Or, 1, false),
            TokenKind::And => (BinaryOp::And, 2, false),
            TokenKind::Equal => (BinaryOp::Equal, 3, false),
            TokenKind::NotEqual => (BinaryOp::NotEqual, 3, false),
            TokenKind::LessThan => (BinaryOp::LessThan, 3, false),
            TokenKind::LessEqual => (BinaryOp::LessEqual, 3, false),
            TokenKind::GreaterThan => (BinaryOp::GreaterThan, 3, false),
            TokenKind::GreaterEqual => (BinaryOp::GreaterEqual, 3, false),
            TokenKind::Concat => (BinaryOp::Concat, 4, true),
            TokenKind::BitOr => (BinaryOp::BitOr, 4, false),
            TokenKind::BitXor => (BinaryOp::BitXor, 5, false),
            TokenKind::BitAnd => (BinaryOp::BitAnd, 6, false),
            TokenKind::ShiftLeft => (BinaryOp::ShiftLeft, 7, false),
            TokenKind::ShiftRight => (BinaryOp::ShiftRight, 7, false),
            TokenKind::Plus => (BinaryOp::Add, 5, false),
            TokenKind::Minus => (BinaryOp::Subtract, 5, false),
            TokenKind::Star => (BinaryOp::Multiply, 6, false),
            TokenKind::Slash => (BinaryOp::Divide, 6, false),
            TokenKind::FloorDivide => (BinaryOp::FloorDivide, 6, false),
            TokenKind::Percent => (BinaryOp::Modulo, 6, false),
            TokenKind::Caret => (BinaryOp::Power, 8, true),
            _ => return None,
        };
        Some((op, precedence, right_associative))
    }

    pub(crate) fn should_reduce(
        &self,
        incoming_precedence: u8,
        incoming_right_associative: bool,
    ) -> bool {
        if self.operators.len() <= self.level().operators_base {
            return false;
        }
        match self.operators.last() {
            Some(PrattOperator::Unary { .. }) => 7 >= incoming_precedence,
            Some(PrattOperator::Binary { precedence, .. }) => {
                *precedence > incoming_precedence
                    || (*precedence == incoming_precedence && !incoming_right_associative)
            }
            None => false,
        }
    }

    /// Applies the innermost level's top operator to its operands.
    pub(crate) fn reduce(&mut self) -> Result<(), SyntaxError> {
        let values_base = self.level().values_base;
        let operator = self
            .operators
            .pop()
            .expect("reduction requires an operator");
        let pop_value = |values: &mut Vec<Expr>| {
            if values.len() > values_base {
                values.pop()
            } else {
                None
            }
        };
        match operator {
            PrattOperator::Unary { op, span } => {
                let expression = pop_value(&mut self.values).ok_or_else(|| {
                    SyntaxError::new(span, "expected expression after unary operator")
                })?;
                let span = span.join(expression.span());
                self.values
                    .push(make_unary_expression(op, expression, span));
            }
            PrattOperator::Binary {
                op,
                span: operator_span,
                ..
            } => {
                let right = pop_value(&mut self.values).ok_or_else(|| {
                    SyntaxError::new(
                        Span {
                            start: self.source_len,
                            end: self.source_len,
                        },
                        "expected right operand",
                    )
                })?;
                let left = pop_value(&mut self.values)
                    .ok_or_else(|| SyntaxError::new(right.span(), "expected left operand"))?;
                self.values.push(Expr::Binary {
                    op,
                    operator_span,
                    span: left.span().join(right.span()),
                    left: Box::new(left),
                    right: Box::new(right),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn fail(&mut self, span: Span, message: &str) -> PrattPoll {
        self.set_failure(SyntaxError::new(span, message))
    }

    pub(crate) fn set_failure(&mut self, error: SyntaxError) -> PrattPoll {
        self.terminal = Some(Err(error.clone()));
        PrattPoll::Failed(error)
    }

    pub(crate) fn poll(result: Result<Expr, SyntaxError>) -> PrattPoll {
        match result {
            Ok(expression) => PrattPoll::Parsed(expression),
            Err(error) => PrattPoll::Failed(error),
        }
    }
}
