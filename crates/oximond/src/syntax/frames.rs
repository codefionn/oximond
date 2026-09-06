//! Resumable function-frame state machine for parameters and bodies.

use crate::syntax::{
    Block, ChunkMachine, ChunkPoll, ChunkStopped, ChunkTerminator, FunctionFrame,
    FunctionFramePoll, FunctionFrameState, Name, ParsedFunction, PrattInput, Span, SyntaxError,
    Token, TokenKind,
};

impl FunctionFrame {
    pub(crate) const fn new(source_len: usize, start: Span) -> Self {
        Self {
            source_len,
            start,
            state: FunctionFrameState::ExpectOpen,
        }
    }

    #[allow(clippy::too_many_lines)] // Function headers are parsed by one explicit resumable state machine.
    pub(crate) fn resume(&mut self, token: Token) -> FunctionFramePoll {
        if let FunctionFrameState::Body { body, .. } = &mut self.state {
            let poll = body
                .as_mut()
                .expect("tokens go to the innermost detached block")
                .resume(PrattInput::Token(token));
            return self.deliver_block(poll);
        }
        let state = core::mem::replace(&mut self.state, FunctionFrameState::ExpectOpen);
        match state {
            FunctionFrameState::ExpectOpen => {
                if !matches!(token.kind, TokenKind::LParen) {
                    return Self::fail(token.span, "expected '(' after function name");
                }
                self.state = FunctionFrameState::Parameters {
                    open: token.span,
                    parameters: Vec::new(),
                };
                FunctionFramePoll::NeedToken
            }
            FunctionFrameState::Parameters {
                open,
                mut parameters,
            } => match token.kind {
                TokenKind::RParen if parameters.is_empty() => {
                    self.start_body(open, parameters, false, None)
                }
                TokenKind::Vararg => {
                    self.state = FunctionFrameState::AfterVararg { open, parameters };
                    FunctionFramePoll::NeedToken
                }
                TokenKind::Identifier(text) => {
                    parameters.push(Name {
                        text: text.into(),
                        span: token.span,
                    });
                    self.state = FunctionFrameState::AfterParameter { open, parameters };
                    FunctionFramePoll::NeedToken
                }
                _ => Self::fail(token.span, "expected local name"),
            },
            FunctionFrameState::AfterParameter { open, parameters } => match token.kind {
                TokenKind::Comma => {
                    self.state = FunctionFrameState::Parameters { open, parameters };
                    FunctionFramePoll::NeedToken
                }
                TokenKind::RParen => self.start_body(open, parameters, false, None),
                _ => Self::fail(token.span, "expected ')' after parameters"),
            },
            FunctionFrameState::AfterVararg { open, parameters } => match token.kind {
                TokenKind::RParen => self.start_body(open, parameters, true, None),
                TokenKind::Identifier(text) => {
                    self.state = FunctionFrameState::AfterVarargName {
                        open,
                        parameters,
                        vararg_name: Name {
                            text: text.into(),
                            span: token.span,
                        },
                    };
                    FunctionFramePoll::NeedToken
                }
                _ => Self::fail(token.span, "expected ')' after parameters"),
            },
            FunctionFrameState::AfterVarargName {
                open,
                parameters,
                vararg_name,
            } => {
                if matches!(token.kind, TokenKind::RParen) {
                    self.start_body(open, parameters, true, Some(vararg_name))
                } else {
                    Self::fail(token.span, "expected ')' after vararg name")
                }
            }
            FunctionFrameState::Body { .. } => {
                unreachable!("function bodies are fed in place")
            }
        }
    }

    /// Finishes the function when its body block stops.
    pub(crate) fn deliver_block(&mut self, poll: ChunkPoll) -> FunctionFramePoll {
        if matches!(poll, ChunkPoll::NeedToken) {
            return FunctionFramePoll::NeedToken;
        }
        let FunctionFrameState::Body {
            open,
            parameters,
            vararg,
            vararg_name,
            ..
        } = core::mem::replace(&mut self.state, FunctionFrameState::ExpectOpen)
        else {
            unreachable!("only a function body delivers a block")
        };
        match poll {
            ChunkPoll::Stopped(ChunkStopped {
                block,
                terminator: ChunkTerminator::End(end),
            }) => FunctionFramePoll::Parsed(ParsedFunction {
                parameters,
                vararg,
                vararg_name,
                body: Block {
                    statements: block.statements,
                    span: open.join(end),
                },
                end,
            }),
            ChunkPoll::Stopped(_) => unreachable!("function body only stops at end"),
            ChunkPoll::Parsed(_) => unreachable!("function body cannot reach root EOF"),
            ChunkPoll::Failed(error) => FunctionFramePoll::Failed(error),
            ChunkPoll::NeedToken => unreachable!("handled above"),
        }
    }

    /// Takes the body block once it opens, if it is still attached.
    pub(crate) fn take_new_block(&mut self) -> Option<Box<ChunkMachine>> {
        match &mut self.state {
            FunctionFrameState::Body { body, .. } => body.take(),
            _ => None,
        }
    }

    pub(crate) fn accepts_vararg_parameter(&self) -> bool {
        match &self.state {
            FunctionFrameState::Parameters { .. } => true,
            FunctionFrameState::Body { vararg, body, .. } => {
                *vararg
                    || body
                        .as_ref()
                        .is_some_and(|body| body.accepts_function_parameter_vararg())
            }
            FunctionFrameState::ExpectOpen
            | FunctionFrameState::AfterParameter { .. }
            | FunctionFrameState::AfterVararg { .. }
            | FunctionFrameState::AfterVarargName { .. } => false,
        }
    }

    pub(crate) fn accepts_parenthesized_call(&self) -> bool {
        match &self.state {
            FunctionFrameState::Body { body, .. } => body
                .as_ref()
                .is_some_and(|body| body.accepts_parenthesized_call()),
            FunctionFrameState::ExpectOpen
            | FunctionFrameState::Parameters { .. }
            | FunctionFrameState::AfterParameter { .. }
            | FunctionFrameState::AfterVararg { .. }
            | FunctionFrameState::AfterVarargName { .. } => false,
        }
    }

    pub(crate) fn set_source_len(&mut self, source_len: usize) {
        self.source_len = source_len;
        if let FunctionFrameState::Body {
            body: Some(body), ..
        } = &mut self.state
        {
            body.set_source_len(source_len);
        }
    }

    pub(crate) fn start_body(
        &mut self,
        open: Span,
        parameters: Vec<Name>,
        vararg: bool,
        vararg_name: Option<Name>,
    ) -> FunctionFramePoll {
        self.state = FunctionFrameState::Body {
            open,
            parameters,
            vararg,
            vararg_name,
            body: Some(Box::new(ChunkMachine::block(
                self.source_len,
                open,
                0,
                vararg,
            ))),
        };
        FunctionFramePoll::NeedToken
    }

    pub(crate) fn fail(span: Span, message: &str) -> FunctionFramePoll {
        FunctionFramePoll::Failed(SyntaxError::new(span, message))
    }
}
