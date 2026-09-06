// helpers shared by the split test modules
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
pub(crate) struct Parser {
    tokens: Vec<Token>,
    position: usize,
    source_len: usize,
    // Lua chunks accept varargs. Each nested function temporarily replaces
    // this state while its body is parsed.
    vararg_allowed: bool,
    loop_depth: usize,
}

#[cfg(test)]
impl Parser {
    pub(crate) const fn new(tokens: Vec<Token>, source_len: usize) -> Self {
        Self {
            tokens,
            position: 0,
            source_len,
            vararg_allowed: true,
            loop_depth: 0,
        }
    }
    pub(crate) fn chunk(mut self) -> Result<Chunk, SyntaxError> {
        let statements = self.statements(false)?;
        let eof = self.current().span;
        Ok(Chunk {
            statements,
            span: Span {
                start: 0,
                end: eof.start.min(self.source_len),
            },
            line_starts: Vec::new(),
        })
    }
    pub(crate) fn statements(&mut self, until_end: bool) -> Result<Vec<Statement>, SyntaxError> {
        let mut statements = Vec::new();
        loop {
            while self.matches(|kind| matches!(kind, TokenKind::Semi)) {
                self.advance();
            }
            if matches!(self.current().kind, TokenKind::Eof) {
                return if until_end {
                    Err(self.error_here("expected 'end'"))
                } else {
                    Ok(statements)
                };
            }
            if matches!(
                self.current().kind,
                TokenKind::End | TokenKind::Else | TokenKind::ElseIf | TokenKind::Until
            ) {
                return if until_end {
                    Ok(statements)
                } else {
                    Err(self.error_here("unexpected block terminator"))
                };
            }
            statements.push(self.statement()?);
        }
    }
    #[allow(clippy::too_many_lines)]
    pub(crate) fn statement(&mut self) -> Result<Statement, SyntaxError> {
        if self.matches(|kind| matches!(kind, TokenKind::Do)) {
            let start = self.advance().span;
            let statements = self.statements(true)?;
            let end = self
                .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
                .span;
            return Ok(Statement::Do(Block {
                statements,
                span: start.join(end),
            }));
        }
        if self.matches(|kind| matches!(kind, TokenKind::Local)) {
            return self.local();
        }
        if self.matches(|kind| matches!(kind, TokenKind::Function)) {
            return self.function_statement();
        }
        if self.matches(|kind| matches!(kind, TokenKind::Return)) {
            let start = self.advance().span;
            let values = if self.matches(|kind| {
                matches!(
                    kind,
                    TokenKind::End
                        | TokenKind::Else
                        | TokenKind::ElseIf
                        | TokenKind::Until
                        | TokenKind::Semi
                        | TokenKind::Eof
                )
            }) {
                Vec::new()
            } else {
                self.expression_list()?
            };
            let end = values.last().map_or(start, Expr::span);
            return Ok(Statement::Return {
                values,
                span: start.join(end),
            });
        }
        if self.matches(|kind| matches!(kind, TokenKind::If)) {
            let start = self.advance().span;
            let condition = self.expression()?;
            self.expect(|kind| matches!(kind, TokenKind::Then), "expected 'then'")?;
            let statements = self.statements(true)?;
            let mut else_ifs = Vec::new();
            while self.matches(|kind| matches!(kind, TokenKind::ElseIf)) {
                let arm_start = self.advance().span;
                let condition = self.expression()?;
                self.expect(
                    |kind| matches!(kind, TokenKind::Then),
                    "expected 'then' after elseif condition",
                )?;
                let statements = self.statements(true)?;
                let arm_end = condition.span();
                else_ifs.push(ElseIf {
                    condition,
                    body: Block {
                        statements,
                        span: arm_start.join(arm_end),
                    },
                    span: arm_start.join(arm_end),
                });
            }
            let else_body = if self.matches(|kind| matches!(kind, TokenKind::Else)) {
                let else_span = self.advance().span;
                let statements = self.statements(true)?;
                Some(Block {
                    statements,
                    span: else_span,
                })
            } else {
                None
            };
            let end = self
                .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
                .span;
            return Ok(Statement::If {
                condition,
                body: Block {
                    statements,
                    span: start.join(end),
                },
                else_ifs,
                else_body,
                span: start.join(end),
            });
        }
        if self.matches(|kind| matches!(kind, TokenKind::While)) {
            return self.while_statement();
        }
        if self.matches(|kind| matches!(kind, TokenKind::Repeat)) {
            return self.repeat_statement();
        }
        if self.matches(|kind| matches!(kind, TokenKind::Break)) {
            let span = self.advance().span;
            if self.loop_depth == 0 {
                return Err(SyntaxError::new(span, "break outside a loop"));
            }
            return Ok(Statement::Break { span });
        }
        if self.matches(|kind| matches!(kind, TokenKind::For)) {
            return self.numeric_for();
        }
        let first = self.expression()?;
        if self.matches(|kind| matches!(kind, TokenKind::Assign | TokenKind::Comma)) {
            let start = first.span();
            let mut targets = vec![first];
            while self.matches(|kind| matches!(kind, TokenKind::Comma)) {
                self.advance();
                targets.push(self.expression()?);
            }
            if targets.iter().any(|target| !target.is_assignment_target()) {
                let target = targets
                    .iter()
                    .find(|target| !target.is_assignment_target())
                    .expect("known non-target");
                return Err(SyntaxError::new(
                    target.span(),
                    "expected assignment target",
                ));
            }
            self.expect(
                |kind| matches!(kind, TokenKind::Assign),
                "expected '=' in assignment",
            )?;
            let values = self.expression_list()?;
            let end = values.last().map_or(start, Expr::span);
            return Ok(Statement::Assign {
                targets,
                values,
                span: start.join(end),
            });
        }
        if matches!(first, Expr::Call { .. } | Expr::MethodCall { .. }) {
            Ok(Statement::Call(first))
        } else {
            Err(SyntaxError::new(
                first.span(),
                "expected assignment or function call",
            ))
        }
    }
    pub(crate) fn local(&mut self) -> Result<Statement, SyntaxError> {
        let start = self.advance().span;
        if self.matches(|kind| matches!(kind, TokenKind::Function)) {
            self.advance();
            let name = self.name()?;
            let (parameters, vararg, body, end) = self.function_body()?;
            return Ok(Statement::LocalFunction {
                name,
                parameters,
                vararg,
                vararg_name: None,
                body,
                span: start.join(end),
            });
        }
        let mut names = vec![self.name()?];
        let mut attributes = vec![self.local_attribute()?];
        while self.matches(|kind| matches!(kind, TokenKind::Comma)) {
            self.advance();
            names.push(self.name()?);
            attributes.push(self.local_attribute()?);
        }
        let values = if self.matches(|kind| matches!(kind, TokenKind::Assign)) {
            self.advance();
            self.expression_list()?
        } else {
            Vec::new()
        };
        let end = values
            .last()
            .map(Expr::span)
            .or_else(|| names.last().map(|name| name.span))
            .unwrap_or(start);
        Ok(Statement::Local {
            names,
            attributes,
            values,
            span: start.join(end),
        })
    }
    pub(crate) fn local_attribute(&mut self) -> Result<LocalAttribute, SyntaxError> {
        if !self.matches(|kind| matches!(kind, TokenKind::LessThan)) {
            return Ok(LocalAttribute::None);
        }
        self.advance();
        let attribute = match self.advance().kind {
            TokenKind::Identifier(name) if name == "const" => LocalAttribute::Const,
            TokenKind::Identifier(name) if name == "close" => LocalAttribute::Close,
            _ => return Err(self.error_here("expected 'const' or 'close' local attribute")),
        };
        self.expect(
            |kind| matches!(kind, TokenKind::GreaterThan),
            "expected '>' after local attribute",
        )?;
        Ok(attribute)
    }
    pub(crate) fn function_statement(&mut self) -> Result<Statement, SyntaxError> {
        let start = self.advance().span;
        let mut segments = vec![self.name()?];
        while self.matches(|kind| matches!(kind, TokenKind::Dot)) {
            self.advance();
            segments.push(self.name()?);
        }
        let method = if self.matches(|kind| matches!(kind, TokenKind::Colon)) {
            self.advance();
            segments.push(self.name()?);
            true
        } else {
            false
        };
        let (parameters, vararg, body, end) = self.function_body()?;
        Ok(Statement::Function {
            name: FunctionName {
                segments,
                method,
                span: start.join(end),
            },
            parameters,
            vararg,
            vararg_name: None,
            body,
            span: start.join(end),
        })
    }
    pub(crate) fn function_body(&mut self) -> Result<(Vec<Name>, bool, Block, Span), SyntaxError> {
        let start = self
            .expect(
                |kind| matches!(kind, TokenKind::LParen),
                "expected '(' after function name",
            )?
            .span;
        let mut parameters = Vec::new();
        let mut vararg = false;
        if self.matches(|kind| matches!(kind, TokenKind::Vararg)) {
            self.advance();
            vararg = true;
        } else if !self.matches(|kind| matches!(kind, TokenKind::RParen)) {
            parameters.push(self.name()?);
            while self.matches(|kind| matches!(kind, TokenKind::Comma)) {
                self.advance();
                if self.matches(|kind| matches!(kind, TokenKind::Vararg)) {
                    self.advance();
                    vararg = true;
                    break;
                }
                parameters.push(self.name()?);
            }
        }
        self.expect(
            |kind| matches!(kind, TokenKind::RParen),
            "expected ')' after parameters",
        )?;
        let outer_vararg = self.vararg_allowed;
        let outer_loop_depth = self.loop_depth;
        self.vararg_allowed = vararg;
        self.loop_depth = 0;
        let statements = self.statements(true);
        self.vararg_allowed = outer_vararg;
        self.loop_depth = outer_loop_depth;
        let statements = statements?;
        let end = self
            .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
            .span;
        Ok((
            parameters,
            vararg,
            Block {
                statements,
                span: start.join(end),
            },
            end,
        ))
    }
    pub(crate) fn while_statement(&mut self) -> Result<Statement, SyntaxError> {
        let start = self.advance().span;
        let condition = self.expression()?;
        self.expect(
            |kind| matches!(kind, TokenKind::Do),
            "expected 'do' after while condition",
        )?;
        let statements = self.loop_statements()?;
        let end = self
            .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
            .span;
        Ok(Statement::While {
            condition,
            body: Block {
                statements,
                span: start.join(end),
            },
            span: start.join(end),
        })
    }
    pub(crate) fn repeat_statement(&mut self) -> Result<Statement, SyntaxError> {
        let start = self.advance().span;
        let statements = self.loop_statements()?;
        self.expect(
            |kind| matches!(kind, TokenKind::Until),
            "expected 'until' after repeat body",
        )?;
        let condition = self.expression()?;
        let end = condition.span();
        Ok(Statement::Repeat {
            body: Block {
                statements,
                span: start.join(end),
            },
            condition,
            span: start.join(end),
        })
    }
    pub(crate) fn loop_statements(&mut self) -> Result<Vec<Statement>, SyntaxError> {
        self.loop_depth += 1;
        let statements = self.statements(true);
        self.loop_depth -= 1;
        statements
    }
    pub(crate) fn numeric_for(&mut self) -> Result<Statement, SyntaxError> {
        let start_span = self.advance().span;
        let name = self.name()?;
        if !self.matches(|kind| matches!(kind, TokenKind::Assign)) {
            let mut names = vec![name];
            while self.matches(|kind| matches!(kind, TokenKind::Comma)) {
                self.advance();
                names.push(self.name()?);
            }
            self.expect(
                |kind| matches!(kind, TokenKind::In),
                "expected 'in' after loop names",
            )?;
            let expressions = self.expression_list()?;
            self.expect(
                |kind| matches!(kind, TokenKind::Do),
                "expected 'do' after iterator expressions",
            )?;
            let statements = self.loop_statements()?;
            let end = self
                .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
                .span;
            return Ok(Statement::GenericFor {
                names,
                expressions,
                body: Block {
                    statements,
                    span: start_span.join(end),
                },
                span: start_span.join(end),
            });
        }
        self.expect(
            |kind| matches!(kind, TokenKind::Assign),
            "expected '=' after loop name",
        )?;
        let start = self.expression()?;
        self.expect(
            |kind| matches!(kind, TokenKind::Comma),
            "expected ',' after loop start",
        )?;
        let limit = self.expression()?;
        let step = if self.matches(|kind| matches!(kind, TokenKind::Comma)) {
            self.advance();
            Some(self.expression()?)
        } else {
            None
        };
        self.expect(
            |kind| matches!(kind, TokenKind::Do),
            "expected 'do' after numeric for",
        )?;
        let statements = self.loop_statements()?;
        let end = self
            .expect(|kind| matches!(kind, TokenKind::End), "expected 'end'")?
            .span;
        Ok(Statement::NumericFor {
            name,
            start: Box::new(start),
            limit: Box::new(limit),
            step: step.map(Box::new),
            body: Block {
                statements,
                span: start_span.join(end),
            },
            span: start_span.join(end),
        })
    }
    pub(crate) fn expression_list(&mut self) -> Result<Vec<Expr>, SyntaxError> {
        let mut values = vec![self.expression()?];
        while self.matches(|kind| matches!(kind, TokenKind::Comma)) {
            self.advance();
            values.push(self.expression()?);
        }
        Ok(values)
    }
    pub(crate) fn expression(&mut self) -> Result<Expr, SyntaxError> {
        self.binary(0)
    }
    pub(crate) fn binary(&mut self, minimum: u8) -> Result<Expr, SyntaxError> {
        let mut left = self.unary()?;
        loop {
            let (op, precedence) = match self.current().kind {
                // Lua 5.1 order, limited to the operators this parser has.
                TokenKind::Or => (BinaryOp::Or, 1),
                TokenKind::And => (BinaryOp::And, 2),
                TokenKind::Equal => (BinaryOp::Equal, 3),
                TokenKind::NotEqual => (BinaryOp::NotEqual, 3),
                TokenKind::LessThan => (BinaryOp::LessThan, 3),
                TokenKind::LessEqual => (BinaryOp::LessEqual, 3),
                TokenKind::GreaterThan => (BinaryOp::GreaterThan, 3),
                TokenKind::GreaterEqual => (BinaryOp::GreaterEqual, 3),
                TokenKind::Concat => (BinaryOp::Concat, 4),
                TokenKind::BitOr => (BinaryOp::BitOr, 4),
                TokenKind::BitXor => (BinaryOp::BitXor, 5),
                TokenKind::BitAnd => (BinaryOp::BitAnd, 6),
                TokenKind::ShiftLeft => (BinaryOp::ShiftLeft, 7),
                TokenKind::ShiftRight => (BinaryOp::ShiftRight, 7),
                TokenKind::Plus | TokenKind::Minus => (
                    if matches!(self.current().kind, TokenKind::Plus) {
                        BinaryOp::Add
                    } else {
                        BinaryOp::Subtract
                    },
                    5,
                ),
                TokenKind::Star
                | TokenKind::Slash
                | TokenKind::FloorDivide
                | TokenKind::Percent => (
                    match self.current().kind {
                        TokenKind::Star => BinaryOp::Multiply,
                        TokenKind::Slash => BinaryOp::Divide,
                        TokenKind::FloorDivide => BinaryOp::FloorDivide,
                        TokenKind::Percent => BinaryOp::Modulo,
                        _ => unreachable!(),
                    },
                    6,
                ),
                TokenKind::Caret => (BinaryOp::Power, 8),
                _ => break,
            };
            if precedence < minimum {
                break;
            }
            let operator_span = self.current().span;
            self.advance();
            let next_minimum = if matches!(op, BinaryOp::Concat | BinaryOp::Power) {
                precedence
            } else {
                precedence + 1
            };
            let right = self.binary(next_minimum)?;
            let span = left.span().join(right.span());
            left = Expr::Binary {
                op,
                operator_span,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        Ok(left)
    }
    pub(crate) fn unary(&mut self) -> Result<Expr, SyntaxError> {
        if self.matches(|kind| {
            matches!(
                kind,
                TokenKind::Minus | TokenKind::Not | TokenKind::Hash | TokenKind::BitXor
            )
        }) {
            let token = self.advance();
            // Unary operators bind less tightly than exponentiation on their
            // left (`-2^2`), while the recursive prefix parse on the right of
            // `^` still accepts `2^-2`.
            let expression = self.binary(7)?;
            let span = token.span.join(expression.span());
            let op = match token.kind {
                TokenKind::Minus => UnaryOp::Negate,
                TokenKind::Not => UnaryOp::Not,
                TokenKind::Hash => UnaryOp::Length,
                TokenKind::BitXor => UnaryOp::BitNot,
                _ => unreachable!("unary token was matched"),
            };
            Ok(make_unary_expression(op, expression, span))
        } else {
            self.suffixed()
        }
    }
    pub(crate) fn suffixed(&mut self) -> Result<Expr, SyntaxError> {
        let mut expression = self.primary()?;
        while Self::is_prefix_expression(&expression) {
            if self.matches(|kind| matches!(kind, TokenKind::LBracket)) {
                self.advance();
                let key = self.expression()?;
                let close = self
                    .expect(|kind| matches!(kind, TokenKind::RBracket), "expected ']'")?
                    .span;
                let span = expression.span().join(close);
                expression = Expr::Index {
                    table: Box::new(expression),
                    key: Box::new(key),
                    span,
                };
            } else if self.matches(|kind| matches!(kind, TokenKind::Dot)) {
                self.advance();
                let field = self.name()?;
                let span = expression.span().join(field.span);
                expression = Expr::Index {
                    table: Box::new(expression),
                    key: Box::new(Expr::String(LuaString::from(&*field.text), field.span)),
                    span,
                };
            } else if self.matches(|kind| {
                matches!(
                    kind,
                    TokenKind::LParen | TokenKind::LBrace | TokenKind::String(_)
                )
            }) {
                let (arguments, end) = self.call_arguments()?;
                let span = expression.span().join(end);
                expression = Expr::Call {
                    callee: Box::new(expression),
                    arguments,
                    span,
                };
            } else if self.matches(|kind| matches!(kind, TokenKind::Colon)) {
                self.advance();
                let method = self.name()?;
                let argument_span = self.current().span;
                let (arguments, end) = self.call_arguments()?;
                let span = expression.span().join(end);
                expression = Expr::MethodCall {
                    argument_span,
                    receiver: Box::new(expression),
                    method: Box::new(method),
                    arguments,
                    span,
                };
            } else {
                break;
            }
        }
        Ok(expression)
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
    pub(crate) fn call_arguments(&mut self) -> Result<(Vec<Expr>, Span), SyntaxError> {
        if self.matches(|kind| matches!(kind, TokenKind::LParen)) {
            self.advance();
            let arguments = if self.matches(|kind| matches!(kind, TokenKind::RParen)) {
                Vec::new()
            } else {
                self.expression_list()?
            };
            let close = self
                .expect(
                    |kind| matches!(kind, TokenKind::RParen),
                    "expected ')' after arguments",
                )?
                .span;
            Ok((arguments, close))
        } else if self.matches(|kind| matches!(kind, TokenKind::LBrace)) {
            let open = self.advance().span;
            let table = self.table_constructor(open)?;
            let end = table.span();
            Ok((vec![table], end))
        } else {
            let token = self.advance();
            match token.kind {
                TokenKind::String(value) => Ok((vec![Expr::String(value, token.span)], token.span)),
                _ => Err(SyntaxError::new(token.span, "expected function arguments")),
            }
        }
    }
    pub(crate) fn primary(&mut self) -> Result<Expr, SyntaxError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Identifier(text) => Ok(Expr::Name(Name {
                text: text.into(),
                span: token.span,
            })),
            TokenKind::Nil => Ok(Expr::Nil(token.span)),
            TokenKind::True => Ok(Expr::Boolean(true, token.span)),
            TokenKind::False => Ok(Expr::Boolean(false, token.span)),
            TokenKind::Integer(integer) => Ok(Expr::Integer(integer, token.span)),
            TokenKind::IntegerBoundary => Ok(Expr::IntegerBoundary(token.span)),
            TokenKind::Number(number) => Ok(Expr::Number(number, token.span)),
            TokenKind::String(value) => Ok(Expr::String(value, token.span)),
            TokenKind::Function => {
                let (parameters, vararg, body, end) = self.function_body()?;
                Ok(Expr::Function {
                    parameters,
                    vararg,
                    vararg_name: None,
                    body: Box::new(body),
                    span: token.span.join(end),
                })
            }
            TokenKind::LBrace => self.table_constructor(token.span),
            TokenKind::Vararg if self.vararg_allowed => Ok(Expr::Vararg(token.span)),
            TokenKind::Vararg => Err(SyntaxError::new(
                token.span,
                "cannot use '...' outside a vararg function",
            )),
            TokenKind::LParen => {
                let expression = self.expression()?;
                let close = self
                    .expect(|kind| matches!(kind, TokenKind::RParen), "expected ')'")?
                    .span;
                Ok(Expr::Parenthesized {
                    expression: Box::new(expression),
                    span: token.span.join(close),
                })
            }
            _ => Err(SyntaxError::new(token.span, "expected expression")),
        }
    }
    pub(crate) fn table_constructor(&mut self, open: Span) -> Result<Expr, SyntaxError> {
        let mut fields = Vec::new();
        while !self.matches(|kind| matches!(kind, TokenKind::RBrace)) {
            if matches!(self.current().kind, TokenKind::Eof) {
                return Err(self.error_here("expected '}' in table constructor"));
            }
            let field = if self.matches(|kind| matches!(kind, TokenKind::LBracket)) {
                self.advance();
                let key = self.expression()?;
                self.expect(
                    |kind| matches!(kind, TokenKind::RBracket),
                    "expected ']' in table key",
                )?;
                self.expect(
                    |kind| matches!(kind, TokenKind::Assign),
                    "expected '=' after table key",
                )?;
                let value = self.expression()?;
                TableField::Key { key, value }
            } else if let TokenKind::Identifier(text) = &self.current().kind {
                if matches!(
                    self.tokens.get(self.position + 1).map(|token| &token.kind),
                    Some(TokenKind::Assign)
                ) {
                    let name = Name {
                        text: text.clone().into(),
                        span: self.advance().span,
                    };
                    self.advance();
                    let value = self.expression()?;
                    TableField::Name { name, value }
                } else {
                    TableField::List(self.expression()?)
                }
            } else {
                TableField::List(self.expression()?)
            };
            fields.push(field);
            if self.matches(|kind| matches!(kind, TokenKind::Comma | TokenKind::Semi)) {
                self.advance();
            } else if !self.matches(|kind| matches!(kind, TokenKind::RBrace)) {
                return Err(self.error_here("expected ',' ';' or '}' in table constructor"));
            }
        }
        let close = self.advance().span;
        Ok(Expr::Table {
            fields,
            span: open.join(close),
        })
    }
    pub(crate) fn name(&mut self) -> Result<Name, SyntaxError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Identifier(text) => Ok(Name {
                text: text.into(),
                span: token.span,
            }),
            _ => Err(SyntaxError::new(token.span, "expected local name")),
        }
    }
    pub(crate) fn expect(
        &mut self,
        predicate: impl FnOnce(&TokenKind) -> bool,
        message: &str,
    ) -> Result<Token, SyntaxError> {
        if predicate(&self.current().kind) {
            Ok(self.advance())
        } else {
            Err(self.error_here(message))
        }
    }
    pub(crate) fn matches(&self, predicate: impl FnOnce(&TokenKind) -> bool) -> bool {
        predicate(&self.current().kind)
    }
    pub(crate) fn current(&self) -> &Token {
        &self.tokens[self.position]
    }
    pub(crate) fn advance(&mut self) -> Token {
        let token = self.tokens[self.position].clone();
        if !matches!(token.kind, TokenKind::Eof) {
            self.position += 1;
        }
        token
    }
    pub(crate) fn error_here(&self, message: &str) -> SyntaxError {
        SyntaxError::new(self.current().span, message)
    }
}

fn stream_tokens(pieces: &[&[u8]]) -> Result<Vec<Token>, SyntaxError> {
    let mut lexer = IncrementalLexer::new();
    let mut tokens = Vec::new();
    for piece in pieces {
        lexer.feed(piece);
        while let LexPoll::Token(token) = lexer.poll()? {
            tokens.push(token);
        }
    }
    lexer.end();
    loop {
        match lexer.poll()? {
            LexPoll::NeedInput => panic!("ended lexer requested input"),
            LexPoll::Token(token) => tokens.push(token),
            LexPoll::Eof => return Ok(tokens),
        }
    }
}

fn parse_with_chunk_machine(source: &[u8]) -> Result<Chunk, SyntaxError> {
    let tokens = Lexer::new(source).lex()?;
    let mut machine = ChunkMachine::new(source.len());
    for token in tokens {
        match machine.resume(PrattInput::Token(token)) {
            ChunkPoll::NeedToken => {}
            ChunkPoll::Parsed(chunk) => return Ok(chunk),
            ChunkPoll::Stopped(_) => panic!("root parser cannot stop at a block terminator"),
            ChunkPoll::Failed(error) => return Err(error),
        }
    }
    unreachable!("lexer emits an EOF token")
}

fn parse_with_legacy_parser(source: &[u8]) -> Result<Chunk, SyntaxError> {
    Parser::new(Lexer::new(source).lex()?, source.len()).chunk()
}

fn parse_streaming_with_chunk_machine(source: &[u8], split: usize) -> Result<Chunk, SyntaxError> {
    let mut lexer = IncrementalLexer::new();
    let mut machine = ChunkMachine::new(source.len());
    let mut resume = |input| match machine.resume(input) {
        ChunkPoll::NeedToken => Ok(None),
        ChunkPoll::Parsed(chunk) => Ok(Some(chunk)),
        ChunkPoll::Stopped(_) => panic!("root parser cannot stop at a block terminator"),
        ChunkPoll::Failed(error) => Err(error),
    };
    for bytes in source.chunks(split) {
        lexer.feed(bytes);
        loop {
            match lexer.poll()? {
                LexPoll::NeedInput => break,
                LexPoll::Token(token) => {
                    if let Some(chunk) = resume(PrattInput::Token(token))? {
                        return Ok(chunk);
                    }
                }
                LexPoll::Eof => panic!("unterminated lexer reached EOF before end"),
            }
        }
    }
    lexer.end();
    loop {
        match lexer.poll()? {
            LexPoll::NeedInput => panic!("ended lexer requested input"),
            LexPoll::Token(token) => {
                if let Some(chunk) = resume(PrattInput::Token(token))? {
                    return Ok(chunk);
                }
            }
            LexPoll::Eof => match resume(PrattInput::End)? {
                Some(chunk) => return Ok(chunk),
                None => panic!("root parser requested another token after EOF"),
            },
        }
    }
}

mod expr;
mod fast;
mod lexer;
mod misc;
mod statements;
