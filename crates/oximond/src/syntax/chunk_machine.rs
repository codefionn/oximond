//! Resumable statement-block state machine that assembles chunks.

use core::ops::ControlFlow;

use crate::syntax::{
    Block, BlockSlot, Chunk, ChunkMachine, ChunkPoll, ChunkState, ChunkStop, ChunkStopped,
    ChunkTerminator, ElseIf, Expr, FunctionFrame, FunctionFramePoll, FunctionName,
    GlobalDeclarationMode, LocalAttribute, Name, PrattInput, PrattMachine, PrattPoll, Span,
    Statement, SyntaxError, Token, TokenKind,
};

/// Handles one input in one `ChunkState`. `Continue` carries a token to
/// re-dispatch in the next state; `Break` ends `ChunkMachine::token`.
type StateHandler = fn(&mut ChunkMachine, ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>>;

/// What a state handler consumes: a token, or the result of the child
/// machine its state holds after `ChunkMachine::forward` fed that child.
enum ChunkInput {
    Token(Token),
    Pratt(PrattPoll),
    Chunk(ChunkPoll),
    Frame(FunctionFramePoll),
}

impl ChunkInput {
    fn into_token(self) -> Token {
        match self {
            Self::Token(token) => token,
            Self::Pratt(_) | Self::Chunk(_) | Self::Frame(_) => {
                unreachable!("a state without a child machine received a child result")
            }
        }
    }
}

#[allow(dead_code)]
impl ChunkMachine {
    pub(crate) const fn new(source_len: usize) -> Self {
        Self {
            source_len,
            statements: Vec::new(),
            last_statement: None,
            legacy_break_terminates: true,
            trailing_empty_statements: true,
            state: ChunkState::Start,
            expression: PrattMachine::new(source_len),
            stop: None,
            block_start: Span { start: 0, end: 0 },
            loop_depth: 0,
            vararg_allowed: true,
            terminal: None,
            transitions: 0,
            blocks: Vec::new(),
        }
    }

    pub(crate) const fn block(
        source_len: usize,
        start: Span,
        loop_depth: usize,
        vararg_allowed: bool,
    ) -> Self {
        Self {
            source_len,
            statements: Vec::new(),
            last_statement: None,
            legacy_break_terminates: true,
            trailing_empty_statements: true,
            state: ChunkState::Start,
            expression: PrattMachine::new(source_len),
            stop: Some(ChunkStop::End),
            block_start: start,
            loop_depth,
            vararg_allowed,
            terminal: None,
            transitions: 0,
            blocks: Vec::new(),
        }
    }

    pub(crate) const fn repeat_block(
        source_len: usize,
        start: Span,
        loop_depth: usize,
        vararg_allowed: bool,
    ) -> Self {
        Self {
            source_len,
            statements: Vec::new(),
            last_statement: None,
            legacy_break_terminates: true,
            trailing_empty_statements: true,
            state: ChunkState::Start,
            expression: PrattMachine::new(source_len),
            stop: Some(ChunkStop::Until),
            block_start: start,
            loop_depth,
            vararg_allowed,
            terminal: None,
            transitions: 0,
            blocks: Vec::new(),
        }
    }

    pub(crate) const fn if_block(
        source_len: usize,
        start: Span,
        loop_depth: usize,
        vararg_allowed: bool,
    ) -> Self {
        Self {
            source_len,
            statements: Vec::new(),
            last_statement: None,
            legacy_break_terminates: true,
            trailing_empty_statements: true,
            state: ChunkState::Start,
            expression: PrattMachine::new(source_len),
            stop: Some(ChunkStop::IfArm),
            block_start: start,
            loop_depth,
            vararg_allowed,
            terminal: None,
            transitions: 0,
            blocks: Vec::new(),
        }
    }

    pub(crate) fn resume(&mut self, input: PrattInput) -> ChunkPoll {
        if let Some(result) = &self.terminal {
            return Self::poll(result.clone());
        }
        self.transitions += 1;
        let token = match input {
            PrattInput::Token(token) => token,
            PrattInput::End => Token {
                kind: TokenKind::Eof,
                span: Span {
                    start: self.source_len,
                    end: self.source_len,
                },
            },
        };
        let poll = match self.blocks.last_mut() {
            Some(block) => block.token(token),
            None => self.token(token),
        };
        if matches!(poll, ChunkPoll::NeedToken) {
            self.detach_blocks();
            return ChunkPoll::NeedToken;
        }
        self.settle(poll)
    }

    /// Hands a stopped or failed block's result to whatever holds it, until
    /// some machine needs another token.
    #[inline(never)]
    fn settle(&mut self, mut poll: ChunkPoll) -> ChunkPoll {
        loop {
            if matches!(poll, ChunkPoll::NeedToken) {
                self.detach_blocks();
                return ChunkPoll::NeedToken;
            }
            if self.blocks.pop().is_none() {
                if let ChunkPoll::Failed(error) = &poll {
                    self.terminal = Some(Err(error.clone()));
                }
                return poll;
            }
            poll = match self.blocks.last_mut() {
                Some(holder) => holder.deliver_block(poll),
                None => self.deliver_block(poll),
            };
        }
    }

    /// Moves blocks that the innermost machine just opened onto `blocks`.
    fn detach_blocks(&mut self) {
        loop {
            let opened = match self.blocks.last_mut() {
                Some(block) => block.take_new_block(),
                None => self.take_new_block(),
            };
            match opened {
                Some(mut block) => {
                    block.legacy_break_terminates = self.legacy_break_terminates;
                    block.trailing_empty_statements = self.trailing_empty_statements;
                    self.blocks.push(block);
                }
                None => return,
            }
        }
    }

    /// Takes the block this machine's state opened, if it is still attached.
    fn take_new_block(&mut self) -> Option<Box<Self>> {
        match &mut self.state {
            ChunkState::DoBody { body, .. }
            | ChunkState::WhileBody { body, .. }
            | ChunkState::RepeatBody { body, .. }
            | ChunkState::IfBody { body, .. }
            | ChunkState::IfElseIfBody { arm_body: body, .. }
            | ChunkState::IfElseBody {
                else_body: body, ..
            }
            | ChunkState::ForNumericBody { body, .. }
            | ChunkState::ForGenericBody { body, .. } => body.take(),
            ChunkState::LocalValues { .. }
            | ChunkState::GlobalValues { .. }
            | ChunkState::ExpressionStatement
            | ChunkState::AssignmentTargets { .. }
            | ChunkState::AssignmentValues { .. }
            | ChunkState::ReturnValues { .. }
            | ChunkState::WhileCondition { .. }
            | ChunkState::RepeatCondition { .. }
            | ChunkState::IfCondition { .. }
            | ChunkState::IfElseIfCondition { .. }
            | ChunkState::ForGenericExpressions { .. }
            | ChunkState::ForNumericStart { .. }
            | ChunkState::ForNumericLimit { .. }
            | ChunkState::ForNumericStep { .. } => self.expression.take_new_block(),
            ChunkState::GlobalFunctionDeclaration { frame, .. }
            | ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => frame.take_new_block(),
            _ => None,
        }
    }

    /// Gives this machine the result of the detached block its state holds.
    #[inline(never)]
    fn deliver_block(&mut self, poll: ChunkPoll) -> ChunkPoll {
        let input = match &mut self.state {
            ChunkState::DoBody { .. }
            | ChunkState::WhileBody { .. }
            | ChunkState::RepeatBody { .. }
            | ChunkState::IfBody { .. }
            | ChunkState::IfElseIfBody { .. }
            | ChunkState::IfElseBody { .. }
            | ChunkState::ForNumericBody { .. }
            | ChunkState::ForGenericBody { .. } => ChunkInput::Chunk(poll),
            ChunkState::GlobalFunctionDeclaration { frame, .. }
            | ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => {
                match frame.deliver_block(poll) {
                    FunctionFramePoll::NeedToken => return ChunkPoll::NeedToken,
                    poll => ChunkInput::Frame(poll),
                }
            }
            _ => match self.expression.deliver_block(poll) {
                PrattPoll::NeedToken => return ChunkPoll::NeedToken,
                poll => {
                    self.expression.reset();
                    ChunkInput::Pratt(poll)
                }
            },
        };
        self.run(input)
    }

    pub(crate) const fn transition_count(&self) -> usize {
        self.transitions
    }

    pub(crate) fn accepts_parenthesized_call(&self) -> bool {
        if let Some(block) = self.blocks.last() {
            return block.accepts_parenthesized_call();
        }
        match &self.state {
            ChunkState::DoBody { body, .. }
            | ChunkState::WhileBody { body, .. }
            | ChunkState::RepeatBody { body, .. }
            | ChunkState::IfBody { body, .. }
            | ChunkState::IfElseIfBody { arm_body: body, .. }
            | ChunkState::IfElseBody {
                else_body: body, ..
            }
            | ChunkState::ForNumericBody { body, .. }
            | ChunkState::ForGenericBody { body, .. } => body
                .as_ref()
                .is_some_and(|body| body.accepts_parenthesized_call()),
            ChunkState::LocalValues { .. }
            | ChunkState::GlobalValues { .. }
            | ChunkState::ExpressionStatement
            | ChunkState::AssignmentTargets { .. }
            | ChunkState::AssignmentValues { .. }
            | ChunkState::ReturnValues { .. }
            | ChunkState::WhileCondition { .. }
            | ChunkState::RepeatCondition { .. }
            | ChunkState::IfCondition { .. }
            | ChunkState::IfElseIfCondition { .. }
            | ChunkState::ForGenericExpressions { .. }
            | ChunkState::ForNumericStart { .. }
            | ChunkState::ForNumericLimit { .. }
            | ChunkState::ForNumericStep { .. } => self.expression.accepts_parenthesized_call(),
            ChunkState::GlobalFunctionDeclaration { frame, name, .. } if !name.text.is_empty() => {
                frame.accepts_parenthesized_call()
            }
            ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => {
                frame.accepts_parenthesized_call()
            }
            ChunkState::Start
            | ChunkState::LocalName { .. }
            | ChunkState::LocalAfterNames { .. }
            | ChunkState::LocalAttributeName { .. }
            | ChunkState::LocalAttributeClose { .. }
            | ChunkState::GlobalStart { .. }
            | ChunkState::GlobalPrefixAttributeName { .. }
            | ChunkState::GlobalPrefixAttributeClose { .. }
            | ChunkState::GlobalName { .. }
            | ChunkState::GlobalAfterNames { .. }
            | ChunkState::GlobalPostfixAttributeName { .. }
            | ChunkState::GlobalPostfixAttributeClose { .. }
            | ChunkState::GlobalFunctionDeclaration { .. }
            | ChunkState::GotoName { .. }
            | ChunkState::LabelSecondColon { .. }
            | ChunkState::LabelName { .. }
            | ChunkState::LabelEndColon { .. }
            | ChunkState::LabelFinalColon { .. }
            | ChunkState::ReturnStart { .. }
            | ChunkState::ForName { .. }
            | ChunkState::ForAfterName { .. }
            | ChunkState::ForGenericName { .. }
            | ChunkState::ForGenericAfterNames { .. }
            | ChunkState::FunctionGlobalName { .. }
            | ChunkState::FunctionPath { .. }
            | ChunkState::FunctionPathSegment { .. }
            | ChunkState::LocalFunctionName { .. } => false,
        }
    }

    pub(crate) const fn accepts_local_attribute(&self) -> bool {
        matches!(self.state, ChunkState::LocalAfterNames { .. })
    }

    pub(crate) const fn accepts_local_prefix_attribute(&self) -> bool {
        matches!(
            &self.state,
            ChunkState::LocalName {
                names,
                prefix_attribute: LocalAttribute::None,
                ..
            } if names.is_empty()
        )
    }

    pub(crate) const fn accepts_label(&self) -> bool {
        matches!(self.state, ChunkState::Start)
    }

    pub(crate) fn set_source_len(&mut self, source_len: usize) {
        self.source_len = source_len;
        for block in &mut self.blocks {
            block.set_source_len(source_len);
        }
        match &mut self.state {
            ChunkState::DoBody { body, .. }
            | ChunkState::WhileBody { body, .. }
            | ChunkState::RepeatBody { body, .. }
            | ChunkState::IfBody { body, .. }
            | ChunkState::IfElseIfBody { arm_body: body, .. }
            | ChunkState::IfElseBody {
                else_body: body, ..
            }
            | ChunkState::ForNumericBody { body, .. }
            | ChunkState::ForGenericBody { body, .. } => {
                if let Some(body) = body {
                    body.set_source_len(source_len);
                }
            }
            ChunkState::LocalValues { .. }
            | ChunkState::GlobalValues { .. }
            | ChunkState::ExpressionStatement
            | ChunkState::AssignmentTargets { .. }
            | ChunkState::AssignmentValues { .. }
            | ChunkState::ReturnValues { .. }
            | ChunkState::WhileCondition { .. }
            | ChunkState::RepeatCondition { .. }
            | ChunkState::IfCondition { .. }
            | ChunkState::IfElseIfCondition { .. }
            | ChunkState::ForGenericExpressions { .. }
            | ChunkState::ForNumericStart { .. }
            | ChunkState::ForNumericLimit { .. }
            | ChunkState::ForNumericStep { .. } => {
                self.expression.set_source_len(source_len);
            }
            ChunkState::GlobalFunctionDeclaration { frame, .. }
            | ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => {
                frame.set_source_len(source_len);
            }
            ChunkState::Start
            | ChunkState::LocalName { .. }
            | ChunkState::LocalAfterNames { .. }
            | ChunkState::LocalAttributeName { .. }
            | ChunkState::LocalAttributeClose { .. }
            | ChunkState::GlobalStart { .. }
            | ChunkState::GlobalPrefixAttributeName { .. }
            | ChunkState::GlobalPrefixAttributeClose { .. }
            | ChunkState::GlobalName { .. }
            | ChunkState::GlobalAfterNames { .. }
            | ChunkState::GlobalPostfixAttributeName { .. }
            | ChunkState::GlobalPostfixAttributeClose { .. }
            | ChunkState::GotoName { .. }
            | ChunkState::LabelSecondColon { .. }
            | ChunkState::LabelName { .. }
            | ChunkState::LabelEndColon { .. }
            | ChunkState::LabelFinalColon { .. }
            | ChunkState::ReturnStart { .. }
            | ChunkState::ForName { .. }
            | ChunkState::ForAfterName { .. }
            | ChunkState::ForGenericName { .. }
            | ChunkState::ForGenericAfterNames { .. }
            | ChunkState::FunctionGlobalName { .. }
            | ChunkState::FunctionPath { .. }
            | ChunkState::FunctionPathSegment { .. }
            | ChunkState::LocalFunctionName { .. } => {}
        }
    }

    pub(crate) fn token(&mut self, token: Token) -> ChunkPoll {
        if !self.vararg_allowed
            && matches!(token.kind, TokenKind::Vararg)
            && !self.accepts_function_parameter_vararg()
        {
            return self.fail_message(token.span, "cannot use '...' outside a vararg function");
        }
        match self.forward(token) {
            Some(input) => self.run(input),
            None => ChunkPoll::NeedToken,
        }
    }

    /// Runs state handlers from one input until the machine needs a token.
    fn run(&mut self, mut input: ChunkInput) -> ChunkPoll {
        loop {
            // A machine that holds an attached block resumes it recursively.
            // Handle each state in its own `#[inline(never)]` method behind a
            // single call site: debug builds give every binding, argument and
            // result in a function its own stack slot, so destructuring all
            // states here cost ~145 KB per level and exhausted ordinary thread
            // stacks well before Lua's syntax-depth limit.
            let handler = Self::state_handler(&self.state);
            match handler(self, input) {
                ControlFlow::Continue(Some(token)) => match self.forward(token) {
                    Some(next) => input = next,
                    None => return ChunkPoll::NeedToken,
                },
                ControlFlow::Continue(None) => return ChunkPoll::NeedToken,
                ControlFlow::Break(poll) => return poll,
            }
        }
    }

    pub(crate) fn state_start(&self) -> Span {
        match &self.state {
            ChunkState::AssignmentTargets { start, .. } => *start,
            _ => unreachable!("assignment target state was just installed"),
        }
    }

    pub(crate) fn take_targets(&mut self) -> Vec<Expr> {
        match core::mem::replace(&mut self.state, ChunkState::Start) {
            ChunkState::AssignmentTargets { targets, .. } => targets,
            _ => unreachable!("assignment target state was just installed"),
        }
    }

    pub(crate) fn finish(&mut self) -> ChunkPoll {
        let chunk = Chunk {
            statements: core::mem::take(&mut self.statements),
            span: Span {
                start: 0,
                end: self.source_len,
            },
            line_starts: Vec::new(),
        };
        ChunkPoll::Parsed(chunk)
    }

    pub(crate) fn accepts_function_parameter_vararg(&self) -> bool {
        match &self.state {
            ChunkState::DoBody { body, .. }
            | ChunkState::WhileBody { body, .. }
            | ChunkState::RepeatBody { body, .. }
            | ChunkState::IfBody { body, .. }
            | ChunkState::IfElseIfBody { arm_body: body, .. }
            | ChunkState::IfElseBody {
                else_body: body, ..
            }
            | ChunkState::ForNumericBody { body, .. }
            | ChunkState::ForGenericBody { body, .. } => body
                .as_ref()
                .is_some_and(|body| body.accepts_function_parameter_vararg()),
            ChunkState::GlobalFunctionDeclaration { frame, .. }
            | ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => {
                frame.accepts_vararg_parameter()
            }
            ChunkState::LocalValues { .. }
            | ChunkState::GlobalValues { .. }
            | ChunkState::ExpressionStatement
            | ChunkState::AssignmentTargets { .. }
            | ChunkState::AssignmentValues { .. }
            | ChunkState::ReturnValues { .. }
            | ChunkState::WhileCondition { .. }
            | ChunkState::RepeatCondition { .. }
            | ChunkState::IfCondition { .. }
            | ChunkState::IfElseIfCondition { .. }
            | ChunkState::ForGenericExpressions { .. }
            | ChunkState::ForNumericStart { .. }
            | ChunkState::ForNumericLimit { .. }
            | ChunkState::ForNumericStep { .. } => {
                self.expression.accepts_function_parameter_vararg()
            }
            _ => false,
        }
    }

    pub(crate) fn accepted_terminator(&self, token: &Token) -> Option<ChunkTerminator> {
        match (self.stop, &token.kind) {
            (Some(ChunkStop::End | ChunkStop::IfArm), TokenKind::End) => {
                Some(ChunkTerminator::End(token.span))
            }
            (Some(ChunkStop::Until), TokenKind::Until) => Some(ChunkTerminator::Until(token.span)),
            (Some(ChunkStop::IfArm), TokenKind::ElseIf) => {
                Some(ChunkTerminator::ElseIf(token.span))
            }
            (Some(ChunkStop::IfArm), TokenKind::Else) => Some(ChunkTerminator::Else(token.span)),
            _ => None,
        }
    }

    pub(crate) const fn is_block_terminator(kind: &TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::End | TokenKind::Until | TokenKind::ElseIf | TokenKind::Else
        )
    }

    pub(crate) const fn expected_terminator(&self) -> &'static str {
        match self.stop {
            Some(ChunkStop::End) => "expected 'end'",
            Some(ChunkStop::Until) => "expected 'until'",
            Some(ChunkStop::IfArm) => "expected 'end', 'elseif', or 'else'",
            None => "unexpected end of input",
        }
    }

    pub(crate) fn stop(&mut self, end: Span, terminator: ChunkTerminator) -> ChunkPoll {
        ChunkPoll::Stopped(ChunkStopped {
            block: Block {
                statements: core::mem::take(&mut self.statements),
                span: self.block_start.join(end),
            },
            terminator,
        })
    }

    pub(crate) fn else_if(start: Span, condition: Expr, mut body: Block) -> ElseIf {
        // This preserves the legacy parser's public AST span convention.
        body.span = start.join(condition.span());
        ElseIf {
            span: start.join(condition.span()),
            condition,
            body,
        }
    }

    pub(crate) fn push_if(
        &mut self,
        start: Span,
        condition: Expr,
        mut body: Block,
        else_ifs: Vec<ElseIf>,
        else_body: Option<Block>,
        end: Span,
    ) {
        // The old parser assigns the complete if extent to the primary body.
        // Keep staging differential tests honest until spans are normalized as
        // a separate API change.
        body.span = start.join(end);
        self.statements.push(Statement::If {
            condition,
            body,
            else_ifs,
            else_body,
            span: start.join(end),
        });
    }

    pub(crate) fn fail(&mut self, error: SyntaxError) -> ChunkPoll {
        self.terminal = Some(Err(error.clone()));
        ChunkPoll::Failed(error)
    }

    pub(crate) fn fail_message(&mut self, span: Span, message: &str) -> ChunkPoll {
        self.fail(SyntaxError::new(span, message))
    }

    pub(crate) fn poll(result: Result<Chunk, SyntaxError>) -> ChunkPoll {
        match result {
            Ok(chunk) => ChunkPoll::Parsed(chunk),
            Err(error) => ChunkPoll::Failed(error),
        }
    }
}

// Per-state token handlers (`StateHandler`). Each lives in its own frame so
// the recursion through nested block machines only pays for the handler on
// its path; debug builds otherwise reserve stack for every state at once.
#[allow(clippy::needless_pass_by_value)]
impl ChunkMachine {
    // Each handler moves its state out itself, so the copy lives in the
    // handler's frame and not in `run`.
    fn take_state(&mut self) -> ChunkState {
        core::mem::replace(&mut self.state, ChunkState::Start)
    }

    /// Feeds a token to the child machine that the current state holds, in
    /// place. Most tokens only advance the innermost machine, so a parent
    /// level costs one match and one call. The child's result goes to the
    /// state handler only when the child completes or fails.
    #[inline(never)]
    fn forward(&mut self, token: Token) -> Option<ChunkInput> {
        match &mut self.state {
            ChunkState::DoBody { body, .. }
            | ChunkState::WhileBody { body, .. }
            | ChunkState::RepeatBody { body, .. }
            | ChunkState::IfBody { body, .. }
            | ChunkState::IfElseIfBody { arm_body: body, .. }
            | ChunkState::IfElseBody {
                else_body: body, ..
            }
            | ChunkState::ForNumericBody { body, .. }
            | ChunkState::ForGenericBody { body, .. } => {
                let body = body
                    .as_mut()
                    .expect("tokens go to the innermost detached block");
                match body.resume(PrattInput::Token(token)) {
                    ChunkPoll::NeedToken => None,
                    poll => Some(ChunkInput::Chunk(poll)),
                }
            }
            ChunkState::LocalValues { .. }
            | ChunkState::GlobalValues { .. }
            | ChunkState::ExpressionStatement
            | ChunkState::AssignmentTargets { .. }
            | ChunkState::AssignmentValues { .. }
            | ChunkState::ReturnValues { .. }
            | ChunkState::WhileCondition { .. }
            | ChunkState::RepeatCondition { .. }
            | ChunkState::IfCondition { .. }
            | ChunkState::IfElseIfCondition { .. }
            | ChunkState::ForGenericExpressions { .. }
            | ChunkState::ForNumericStart { .. }
            | ChunkState::ForNumericLimit { .. }
            | ChunkState::ForNumericStep { .. } => {
                match self.expression.resume(PrattInput::Token(token)) {
                    PrattPoll::NeedToken => None,
                    poll => {
                        self.expression.reset();
                        Some(ChunkInput::Pratt(poll))
                    }
                }
            }
            ChunkState::GlobalFunctionDeclaration { frame, name, .. } if !name.text.is_empty() => {
                Self::forward_frame(frame, token)
            }
            ChunkState::FunctionDeclaration { frame, .. }
            | ChunkState::LocalFunctionDeclaration { frame, .. } => {
                Self::forward_frame(frame, token)
            }
            _ => Some(ChunkInput::Token(token)),
        }
    }

    fn forward_frame(frame: &mut FunctionFrame, token: Token) -> Option<ChunkInput> {
        match frame.resume(token) {
            FunctionFramePoll::NeedToken => None,
            poll => Some(ChunkInput::Frame(poll)),
        }
    }

    fn pratt_input(expression: &mut PrattMachine, input: ChunkInput) -> PrattPoll {
        match input {
            ChunkInput::Pratt(poll) => poll,
            ChunkInput::Token(token) => {
                let poll = expression.resume(PrattInput::Token(token));
                if !matches!(poll, PrattPoll::NeedToken) {
                    expression.reset();
                }
                poll
            }
            ChunkInput::Chunk(_) | ChunkInput::Frame(_) => {
                unreachable!("expression state received another machine's result")
            }
        }
    }

    fn chunk_input(body: &mut BlockSlot, input: ChunkInput) -> ChunkPoll {
        match input {
            ChunkInput::Chunk(poll) => poll,
            ChunkInput::Token(token) => body
                .as_mut()
                .expect("tokens go to the innermost detached block")
                .resume(PrattInput::Token(token)),
            ChunkInput::Pratt(_) | ChunkInput::Frame(_) => {
                unreachable!("block state received another machine's result")
            }
        }
    }

    fn frame_input(frame: &mut FunctionFrame, input: ChunkInput) -> FunctionFramePoll {
        match input {
            ChunkInput::Frame(poll) => poll,
            ChunkInput::Token(token) => frame.resume(token),
            ChunkInput::Pratt(_) | ChunkInput::Chunk(_) => {
                unreachable!("function state received another machine's result")
            }
        }
    }

    fn state_handler(state: &ChunkState) -> StateHandler {
        match state {
            ChunkState::Start => Self::on_start,
            ChunkState::LocalName { .. } => Self::on_local_name,
            ChunkState::LocalAfterNames { .. } => Self::on_local_after_names,
            ChunkState::LocalAttributeName { .. } => Self::on_local_attribute_name,
            ChunkState::LocalAttributeClose { .. } => Self::on_local_attribute_close,
            ChunkState::GlobalStart { .. } => Self::on_global_start,
            ChunkState::GlobalPrefixAttributeName { .. } => Self::on_global_prefix_attribute_name,
            ChunkState::GlobalPrefixAttributeClose { .. } => Self::on_global_prefix_attribute_close,
            ChunkState::GlobalAfterNames { .. } => Self::on_global_after_names,
            ChunkState::GlobalPostfixAttributeName { .. } => Self::on_global_postfix_attribute_name,
            ChunkState::GlobalPostfixAttributeClose { .. } => {
                Self::on_global_postfix_attribute_close
            }
            ChunkState::GlobalName { .. } => Self::on_global_name,
            ChunkState::GlobalValues { .. } => Self::on_global_values,
            ChunkState::GlobalFunctionDeclaration { .. } => Self::on_global_function_declaration,
            ChunkState::GotoName { .. } => Self::on_goto_name,
            ChunkState::LabelSecondColon { .. } => Self::on_label_second_colon,
            ChunkState::LabelName { .. } => Self::on_label_name,
            ChunkState::LabelEndColon { .. } => Self::on_label_end_colon,
            ChunkState::LabelFinalColon { .. } => {
                unreachable!("double-colon tokens finish labels in one transition")
            }
            ChunkState::LocalValues { .. } => Self::on_local_values,
            ChunkState::ExpressionStatement => Self::on_expression_statement,
            ChunkState::AssignmentTargets { .. } => Self::on_assignment_targets,
            ChunkState::AssignmentValues { .. } => Self::on_assignment_values,
            ChunkState::ReturnStart { .. } => Self::on_return_start,
            ChunkState::ReturnValues { .. } => Self::on_return_values,
            ChunkState::DoBody { .. } => Self::on_do_body,
            ChunkState::WhileCondition { .. } => Self::on_while_condition,
            ChunkState::WhileBody { .. } => Self::on_while_body,
            ChunkState::RepeatBody { .. } => Self::on_repeat_body,
            ChunkState::RepeatCondition { .. } => Self::on_repeat_condition,
            ChunkState::IfCondition { .. } => Self::on_if_condition,
            ChunkState::IfBody { .. } => Self::on_if_body,
            ChunkState::IfElseIfCondition { .. } => Self::on_if_else_if_condition,
            ChunkState::IfElseIfBody { .. } => Self::on_if_else_if_body,
            ChunkState::IfElseBody { .. } => Self::on_if_else_body,
            ChunkState::ForName { .. } => Self::on_for_name,
            ChunkState::ForAfterName { .. } => Self::on_for_after_name,
            ChunkState::ForGenericName { .. } => Self::on_for_generic_name,
            ChunkState::ForGenericAfterNames { .. } => Self::on_for_generic_after_names,
            ChunkState::ForGenericExpressions { .. } => Self::on_for_generic_expressions,
            ChunkState::ForNumericStart { .. } => Self::on_for_numeric_start,
            ChunkState::ForNumericLimit { .. } => Self::on_for_numeric_limit,
            ChunkState::ForNumericStep { .. } => Self::on_for_numeric_step,
            ChunkState::ForNumericBody { .. } => Self::on_for_numeric_body,
            ChunkState::ForGenericBody { .. } => Self::on_for_generic_body,
            ChunkState::FunctionGlobalName { .. } => Self::on_function_global_name,
            ChunkState::FunctionPath { .. } => Self::on_function_path,
            ChunkState::FunctionPathSegment { .. } => Self::on_function_path_segment,
            ChunkState::FunctionDeclaration { .. } => Self::on_function_declaration,
            ChunkState::LocalFunctionName { .. } => Self::on_local_function_name,
            ChunkState::LocalFunctionDeclaration { .. } => Self::on_local_function_declaration,
        }
    }

    #[inline(never)]
    #[allow(clippy::too_many_lines)]
    fn on_start(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        if let Some(terminator) = self.accepted_terminator(&token) {
            return ControlFlow::Break(self.stop(token.span, terminator));
        }
        if self.stop.is_some() && matches!(token.kind, TokenKind::Eof) {
            return ControlFlow::Break(self.fail_message(token.span, self.expected_terminator()));
        }
        if self.stop.is_some() && Self::is_block_terminator(&token.kind) {
            return ControlFlow::Break(self.fail_message(token.span, self.expected_terminator()));
        }
        if matches!(
            token.kind,
            TokenKind::End | TokenKind::Else | TokenKind::ElseIf | TokenKind::Until
        ) {
            return ControlFlow::Break(
                self.fail_message(token.span, "unexpected block terminator"),
            );
        }
        if let Some(semicolon) = &mut self.last_statement {
            if matches!(token.kind, TokenKind::Semi)
                && (!*semicolon || self.trailing_empty_statements)
            {
                *semicolon = true;
                return ControlFlow::Continue(None);
            }
            if !matches!(token.kind, TokenKind::Eof) {
                let expected = if self.stop.is_some() {
                    self.expected_terminator()
                } else {
                    "expected end of input"
                };
                return ControlFlow::Break(self.fail_message(token.span, expected));
            }
        }
        match token.kind {
            TokenKind::Semi => {}
            TokenKind::Eof => return ControlFlow::Break(self.finish()),
            TokenKind::Do => {
                self.state = ChunkState::DoBody {
                    start: token.span,
                    body: Some(Box::new(Self::block(
                        self.source_len,
                        token.span,
                        self.loop_depth,
                        self.vararg_allowed,
                    ))),
                };
            }
            TokenKind::While => {
                self.state = ChunkState::WhileCondition { start: token.span };
            }
            TokenKind::Repeat => {
                self.state = ChunkState::RepeatBody {
                    start: token.span,
                    body: Some(Box::new(Self::repeat_block(
                        self.source_len,
                        token.span,
                        self.loop_depth + 1,
                        self.vararg_allowed,
                    ))),
                }
            }
            TokenKind::If => {
                self.state = ChunkState::IfCondition { start: token.span };
            }
            TokenKind::For => {
                self.state = ChunkState::ForName { start: token.span };
            }
            TokenKind::Function => {
                self.state = ChunkState::FunctionGlobalName { start: token.span };
            }
            TokenKind::Break => {
                if self.loop_depth == 0 {
                    let near = Span {
                        start: token.span.end,
                        end: self.source_len,
                    };
                    return ControlFlow::Break(self.fail(
                        SyntaxError::new(token.span, "break outside a loop").with_near_span(near),
                    ));
                }
                self.statements.push(Statement::Break { span: token.span });
                if self.legacy_break_terminates {
                    self.last_statement = Some(false);
                }
            }
            TokenKind::Goto => {
                self.state = ChunkState::GotoName { start: token.span };
            }
            TokenKind::DoubleColon => {
                self.state = ChunkState::LabelName { start: token.span };
            }
            TokenKind::Local => {
                self.state = ChunkState::LocalName {
                    start: token.span,
                    names: Vec::new(),
                    attributes: Vec::new(),
                    prefix_attribute: LocalAttribute::None,
                };
            }
            TokenKind::Global => {
                self.state = ChunkState::GlobalStart {
                    start: token.span,
                    const_prefix: false,
                };
            }
            TokenKind::Return => {
                self.state = ChunkState::ReturnStart { start: token.span };
            }
            _ => match self.expression.resume(PrattInput::Token(token)) {
                PrattPoll::NeedToken => self.state = ChunkState::ExpressionStatement,
                PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
                PrattPoll::Parsed(_) | PrattPoll::Boundary(_, _) => {
                    unreachable!("a first expression token cannot complete")
                }
            },
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LocalName {
            start,
            mut names,
            mut attributes,
            prefix_attribute,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Function if names.is_empty() && prefix_attribute == LocalAttribute::None => {
                self.state = ChunkState::LocalFunctionName { start };
            }
            TokenKind::LessThan if names.is_empty() && prefix_attribute == LocalAttribute::None => {
                self.state = ChunkState::LocalAttributeName {
                    start,
                    names,
                    attributes,
                    prefix_attribute,
                };
            }
            TokenKind::Identifier(text) => {
                if names.len() >= 200 {
                    return ControlFlow::Break(
                        self.fail(
                            SyntaxError::new(token.span, "too many local variables")
                                .with_function_span(self.block_start),
                        ),
                    );
                }
                names.push(Name {
                    text: text.into(),
                    span: token.span,
                });
                attributes.push(prefix_attribute);
                self.state = ChunkState::LocalAfterNames {
                    start,
                    names,
                    attributes,
                    prefix_attribute,
                };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected local name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_after_names(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LocalAfterNames {
            start,
            names,
            attributes,
            prefix_attribute,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Comma => {
                self.state = ChunkState::LocalName {
                    start,
                    names,
                    attributes,
                    prefix_attribute,
                };
            }
            TokenKind::LessThan => {
                self.state = ChunkState::LocalAttributeName {
                    start,
                    names,
                    attributes,
                    prefix_attribute,
                };
            }
            TokenKind::Assign => {
                self.state = ChunkState::LocalValues {
                    start,
                    names,
                    attributes,
                    values: Vec::new(),
                };
            }
            _ => {
                self.statements.push(Statement::Local {
                    span: start.join(names.last().expect("local name exists").span),
                    names,
                    attributes,
                    values: Vec::new(),
                });
                return ControlFlow::Continue(Some(token));
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_attribute_name(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LocalAttributeName {
            start,
            names,
            attributes,
            prefix_attribute,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(name) if name == "const" || name == "close" => {
                let attribute = if name == "const" {
                    LocalAttribute::Const
                } else {
                    LocalAttribute::Close
                };
                self.state = ChunkState::LocalAttributeClose {
                    start,
                    names,
                    attributes,
                    prefix_attribute,
                    attribute,
                };
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected 'const' or 'close' local attribute"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_attribute_close(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LocalAttributeClose {
            start,
            names,
            mut attributes,
            mut prefix_attribute,
            attribute,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if !matches!(token.kind, TokenKind::GreaterThan) {
            return ControlFlow::Break(
                self.fail_message(token.span, "expected '>' after local attribute"),
            );
        }
        if names.is_empty() {
            prefix_attribute = attribute;
            self.state = ChunkState::LocalName {
                start,
                names,
                attributes,
                prefix_attribute,
            };
        } else {
            *attributes
                .last_mut()
                .expect("local attribute follows a name") = attribute;
            self.state = ChunkState::LocalAfterNames {
                start,
                names,
                attributes,
                prefix_attribute,
            };
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_start(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalStart {
            start,
            const_prefix,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Function if !const_prefix => {
                self.state = ChunkState::GlobalFunctionDeclaration {
                    start,
                    name: Name {
                        text: "".into(),
                        span: token.span,
                    },
                    frame: Box::new(FunctionFrame::new(self.source_len, start)),
                };
            }
            TokenKind::LessThan if !const_prefix => {
                self.state = ChunkState::GlobalPrefixAttributeName { start };
            }
            TokenKind::Star => self.statements.push(Statement::Global {
                mode: GlobalDeclarationMode::All,
                all_const: const_prefix,
                names: Vec::new(),
                const_names: Vec::new(),
                values: Vec::new(),
                span: start.join(token.span),
            }),
            TokenKind::Identifier(name) if name == "none" => {
                self.statements.push(Statement::Global {
                    mode: GlobalDeclarationMode::None,
                    all_const: false,
                    names: Vec::new(),
                    const_names: Vec::new(),
                    values: Vec::new(),
                    span: start.join(token.span),
                });
            }
            TokenKind::Identifier(text) => {
                self.state = ChunkState::GlobalAfterNames {
                    start,
                    const_prefix,
                    names: vec![Name {
                        text: text.into(),
                        span: token.span,
                    }],
                    const_names: vec![const_prefix],
                }
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected global name, '*' or 'none'"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_prefix_attribute_name(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalPrefixAttributeName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(name) if name == "const" => {
                self.state = ChunkState::GlobalPrefixAttributeClose { start };
            }
            TokenKind::Identifier(name) if name == "close" => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "a global variable cannot be to-be-closed"),
                );
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected 'const' global attribute"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_prefix_attribute_close(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalPrefixAttributeClose { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if !matches!(token.kind, TokenKind::GreaterThan) {
            return ControlFlow::Break(
                self.fail_message(token.span, "expected '>' after global attribute"),
            );
        }
        self.state = ChunkState::GlobalStart {
            start,
            const_prefix: true,
        };

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_after_names(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalAfterNames {
            start,
            const_prefix,
            names,
            const_names,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Comma => {
                self.state = ChunkState::GlobalName {
                    start,
                    const_prefix,
                    names,
                    const_names,
                }
            }
            TokenKind::Assign => {
                self.state = ChunkState::GlobalValues {
                    start,
                    names,
                    const_names,
                    values: Vec::new(),
                }
            }
            TokenKind::LessThan => {
                self.state = ChunkState::GlobalPostfixAttributeName {
                    start,
                    names,
                    const_names,
                };
            }
            _ => {
                self.statements.push(Statement::Global {
                    mode: GlobalDeclarationMode::Names,
                    all_const: false,
                    span: start.join(names.last().expect("global name exists").span),
                    names,
                    const_names,
                    values: Vec::new(),
                });
                return ControlFlow::Continue(Some(token));
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_postfix_attribute_name(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalPostfixAttributeName {
            start,
            names,
            const_names,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(name) if name == "const" => {
                self.state = ChunkState::GlobalPostfixAttributeClose {
                    start,
                    names,
                    const_names,
                };
            }
            TokenKind::Identifier(name) if name == "close" => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "a global variable cannot be to-be-closed"),
                );
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected 'const' global attribute"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_postfix_attribute_close(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalPostfixAttributeClose {
            start,
            names,
            mut const_names,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if !matches!(token.kind, TokenKind::GreaterThan) {
            return ControlFlow::Break(
                self.fail_message(token.span, "expected '>' after global attribute"),
            );
        }
        *const_names
            .last_mut()
            .expect("global attribute follows a name") = true;
        self.state = ChunkState::GlobalAfterNames {
            start,
            const_prefix: false,
            names,
            const_names,
        };

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GlobalName {
            start,
            const_prefix,
            mut names,
            mut const_names,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                names.push(Name {
                    text: text.into(),
                    span: token.span,
                });
                const_names.push(const_prefix);
                self.state = ChunkState::GlobalAfterNames {
                    start,
                    const_prefix,
                    names,
                    const_names,
                };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected global name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_values(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::GlobalValues {
            start,
            names,
            const_names,
            mut values,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::GlobalValues {
                    start,
                    names,
                    const_names,
                    values,
                }
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("global initializer needs a boundary"),
            PrattPoll::Boundary(value, boundary) => {
                values.push(value);
                if matches!(boundary.kind, TokenKind::Comma) {
                    self.state = ChunkState::GlobalValues {
                        start,
                        names,
                        const_names,
                        values,
                    };
                } else {
                    let span = start.join(values.last().expect("global initializer exists").span());
                    self.statements.push(Statement::Global {
                        mode: GlobalDeclarationMode::Names,
                        all_const: false,
                        names,
                        const_names,
                        values,
                        span,
                    });
                    return ControlFlow::Continue(Some(boundary));
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_global_function_declaration(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::GlobalFunctionDeclaration {
            start,
            mut name,
            mut frame,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if name.text.is_empty() {
            let token = input.into_token();
            let TokenKind::Identifier(text) = token.kind else {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected global function name"),
                );
            };
            name = Name {
                text: text.into(),
                span: token.span,
            };
            self.state = ChunkState::GlobalFunctionDeclaration { start, name, frame };
        } else {
            match Self::frame_input(&mut frame, input) {
                FunctionFramePoll::NeedToken => {
                    self.state = ChunkState::GlobalFunctionDeclaration { start, name, frame };
                }
                FunctionFramePoll::Parsed(parsed) => {
                    let span = start.join(parsed.end);
                    self.statements.push(Statement::Global {
                        mode: GlobalDeclarationMode::Names,
                        all_const: false,
                        names: vec![name],
                        const_names: vec![false],
                        values: vec![Expr::Function {
                            parameters: parsed.parameters,
                            vararg: parsed.vararg,
                            vararg_name: parsed.vararg_name.map(Box::new),
                            body: Box::new(parsed.body),
                            span,
                        }],
                        span,
                    });
                }
                FunctionFramePoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            }
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_goto_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::GotoName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => self.statements.push(Statement::Goto {
                name: Name {
                    text: text.into(),
                    span: token.span,
                },
                span: start.join(token.span),
            }),
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected label name after 'goto'"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_label_second_colon(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LabelSecondColon { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if !matches!(token.kind, TokenKind::Colon) {
            return ControlFlow::Break(
                self.fail_message(token.span, "expected ':' to start label"),
            );
        }
        self.state = ChunkState::LabelName { start };

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_label_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LabelName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.state = ChunkState::LabelEndColon {
                    start,
                    name: Name {
                        text: text.into(),
                        span: token.span,
                    },
                }
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected label name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_label_end_colon(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LabelEndColon { start, name } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if !matches!(token.kind, TokenKind::DoubleColon) {
            return ControlFlow::Break(
                self.fail_message(token.span, "expected '::' after label name"),
            );
        }
        self.statements.push(Statement::Label {
            span: start.join(token.span),
            name,
        });

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_values(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::LocalValues {
            start,
            names,
            attributes,
            mut values,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::LocalValues {
                    start,
                    names,
                    attributes,
                    values,
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("value expression needs a boundary"),
            PrattPoll::Boundary(value, boundary) => {
                values.push(value);
                if matches!(boundary.kind, TokenKind::Comma) {
                    self.state = ChunkState::LocalValues {
                        start,
                        names,
                        attributes,
                        values,
                    };
                } else {
                    let span = start.join(values.last().expect("value exists").span());
                    self.statements.push(Statement::Local {
                        names,
                        attributes,
                        values,
                        span,
                    });
                    return ControlFlow::Continue(Some(boundary));
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_expression_statement(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ExpressionStatement = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ExpressionStatement;
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => {
                unreachable!("expression needs a statement boundary")
            }
            PrattPoll::Boundary(first, boundary) => match boundary.kind {
                TokenKind::Assign | TokenKind::Comma => {
                    self.state = ChunkState::AssignmentTargets {
                        start: first.span(),
                        targets: vec![first],
                    };
                    if matches!(boundary.kind, TokenKind::Assign) {
                        self.state = ChunkState::AssignmentValues {
                            start: self.state_start(),
                            targets: self.take_targets(),
                            values: Vec::new(),
                        };
                    }
                }
                _ if matches!(first, Expr::Call { .. } | Expr::MethodCall { .. }) => {
                    self.statements.push(Statement::Call(first));
                    return ControlFlow::Continue(Some(boundary));
                }
                _ => {
                    let near = if matches!(boundary.kind, TokenKind::Eof) {
                        first.span()
                    } else {
                        boundary.span
                    };
                    return ControlFlow::Break(
                        self.fail_message(near, "expected assignment or function call"),
                    );
                }
            },
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_assignment_targets(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::AssignmentTargets { start, mut targets } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::AssignmentTargets { start, targets };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("target expression needs a boundary"),
            PrattPoll::Boundary(target, boundary) => {
                targets.push(target);
                if matches!(boundary.kind, TokenKind::Comma) {
                    self.state = ChunkState::AssignmentTargets { start, targets };
                } else if matches!(boundary.kind, TokenKind::Assign) {
                    self.state = ChunkState::AssignmentValues {
                        start,
                        targets,
                        values: Vec::new(),
                    };
                } else {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected '=' in assignment"),
                    );
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_assignment_values(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::AssignmentValues {
            start,
            targets,
            mut values,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::AssignmentValues {
                    start,
                    targets,
                    values,
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("value expression needs a boundary"),
            PrattPoll::Boundary(value, boundary) => {
                values.push(value);
                if matches!(boundary.kind, TokenKind::Comma) {
                    self.state = ChunkState::AssignmentValues {
                        start,
                        targets,
                        values,
                    };
                } else {
                    if let Some(target) =
                        targets.iter().find(|target| !target.is_assignment_target())
                    {
                        return ControlFlow::Break(
                            self.fail_message(target.span(), "expected assignment target"),
                        );
                    }
                    let span = start.join(values.last().expect("value exists").span());
                    self.statements.push(Statement::Assign {
                        targets,
                        values,
                        span,
                    });
                    return ControlFlow::Continue(Some(boundary));
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_return_start(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::ReturnStart { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        if matches!(
            token.kind,
            TokenKind::Semi
                | TokenKind::Eof
                | TokenKind::End
                | TokenKind::Else
                | TokenKind::ElseIf
                | TokenKind::Until
        ) {
            self.statements.push(Statement::Return {
                values: Vec::new(),
                span: start,
            });
            self.last_statement = Some(false);
            return ControlFlow::Continue(Some(token));
        }
        match self.expression.resume(PrattInput::Token(token)) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ReturnValues {
                    start,
                    values: Vec::new(),
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) | PrattPoll::Boundary(_, _) => {
                unreachable!("a first return expression token cannot complete")
            }
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_return_values(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ReturnValues { start, mut values } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ReturnValues { start, values };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("return expression needs a boundary"),
            PrattPoll::Boundary(value, boundary) => {
                values.push(value);
                if matches!(boundary.kind, TokenKind::Comma) {
                    self.state = ChunkState::ReturnValues { start, values };
                } else {
                    let span = start.join(values.last().expect("value exists").span());
                    self.statements.push(Statement::Return { values, span });
                    self.last_statement = Some(false);
                    return ControlFlow::Continue(Some(boundary));
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_do_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::DoBody { start, mut body } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::DoBody { start, body };
            }
            ChunkPoll::Stopped(ChunkStopped { block, .. }) => {
                self.statements.push(Statement::Do(block));
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_while_condition(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::WhileCondition { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::WhileCondition { start };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("while condition needs 'do'"),
            PrattPoll::Boundary(condition, boundary) => {
                if !matches!(boundary.kind, TokenKind::Do) {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected 'do' after while condition"),
                    );
                }
                self.state = ChunkState::WhileBody {
                    start,
                    condition: Box::new(condition),
                    body: Some(Box::new(Self::block(
                        self.source_len,
                        start,
                        self.loop_depth + 1,
                        self.vararg_allowed,
                    ))),
                };
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_while_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::WhileBody {
            start,
            condition,
            mut body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::WhileBody {
                    start,
                    condition,
                    body,
                };
            }
            ChunkPoll::Stopped(ChunkStopped { block, .. }) => {
                self.statements.push(Statement::While {
                    condition: *condition,
                    span: start.join(block.span),
                    body: block,
                });
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_repeat_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::RepeatBody { start, mut body } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };

        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => self.state = ChunkState::RepeatBody { start, body },
            ChunkPoll::Stopped(ChunkStopped { block: body, .. }) => {
                self.state = ChunkState::RepeatCondition { start, body }
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }

        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_repeat_condition(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::RepeatCondition { start, mut body } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => self.state = ChunkState::RepeatCondition { start, body },
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("repeat condition needs a boundary"),
            PrattPoll::Boundary(condition, boundary) => {
                body.span = start.join(condition.span());
                self.statements.push(Statement::Repeat {
                    span: start.join(condition.span()),
                    body,
                    condition,
                });
                return ControlFlow::Continue(Some(boundary));
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_if_condition(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::IfCondition { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::IfCondition { start };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("if condition needs 'then'"),
            PrattPoll::Boundary(condition, boundary) => {
                if !matches!(boundary.kind, TokenKind::Then) {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected 'then' after if condition"),
                    );
                }
                self.state = ChunkState::IfBody {
                    start,
                    condition: Box::new(condition),
                    body: Some(Box::new(Self::if_block(
                        self.source_len,
                        start,
                        self.loop_depth,
                        self.vararg_allowed,
                    ))),
                };
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_if_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::IfBody {
            start,
            condition,
            mut body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::IfBody {
                    start,
                    condition,
                    body,
                }
            }
            ChunkPoll::Stopped(stopped) => match stopped.terminator {
                ChunkTerminator::End(end) => {
                    self.push_if(start, *condition, stopped.block, Vec::new(), None, end);
                }
                ChunkTerminator::ElseIf(arm_start) => {
                    self.state = ChunkState::IfElseIfCondition {
                        start,
                        condition,
                        body: stopped.block,
                        else_ifs: Vec::new(),
                        arm_start,
                    };
                }
                ChunkTerminator::Else(else_start) => {
                    self.state = ChunkState::IfElseBody {
                        start,
                        condition,
                        body: stopped.block,
                        else_ifs: Vec::new(),
                        else_start,
                        else_body: Some(Box::new(Self::block(
                            self.source_len,
                            else_start,
                            self.loop_depth,
                            self.vararg_allowed,
                        ))),
                    };
                }
                ChunkTerminator::Until(_) => unreachable!("if block cannot stop at until"),
            },
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_if_else_if_condition(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::IfElseIfCondition {
            start,
            condition,
            body,
            else_ifs,
            arm_start,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::IfElseIfCondition {
                    start,
                    condition,
                    body,
                    else_ifs,
                    arm_start,
                }
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("elseif condition needs 'then'"),
            PrattPoll::Boundary(arm_condition, boundary) => {
                if !matches!(boundary.kind, TokenKind::Then) {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected 'then' after elseif condition"),
                    );
                }
                self.state = ChunkState::IfElseIfBody {
                    start,
                    condition,
                    body,
                    else_ifs,
                    arm_start,
                    arm_condition: Box::new(arm_condition),
                    arm_body: Some(Box::new(Self::if_block(
                        self.source_len,
                        arm_start,
                        self.loop_depth,
                        self.vararg_allowed,
                    ))),
                };
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_if_else_if_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::IfElseIfBody {
            start,
            condition,
            body,
            mut else_ifs,
            arm_start,
            arm_condition,
            mut arm_body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut arm_body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::IfElseIfBody {
                    start,
                    condition,
                    body,
                    else_ifs,
                    arm_start,
                    arm_condition,
                    arm_body,
                }
            }
            ChunkPoll::Stopped(stopped) => {
                else_ifs.push(Self::else_if(arm_start, *arm_condition, stopped.block));
                match stopped.terminator {
                    ChunkTerminator::End(end) => {
                        self.push_if(start, *condition, body, else_ifs, None, end);
                    }
                    ChunkTerminator::ElseIf(next_arm_start) => {
                        self.state = ChunkState::IfElseIfCondition {
                            start,
                            condition,
                            body,
                            else_ifs,
                            arm_start: next_arm_start,
                        };
                    }
                    ChunkTerminator::Else(else_start) => {
                        self.state = ChunkState::IfElseBody {
                            start,
                            condition,
                            body,
                            else_ifs,
                            else_start,
                            else_body: Some(Box::new(Self::block(
                                self.source_len,
                                else_start,
                                self.loop_depth,
                                self.vararg_allowed,
                            ))),
                        };
                    }
                    ChunkTerminator::Until(_) => {
                        unreachable!("if block cannot stop at until")
                    }
                }
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_if_else_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::IfElseBody {
            start,
            condition,
            body,
            else_ifs,
            else_start,
            mut else_body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut else_body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::IfElseBody {
                    start,
                    condition,
                    body,
                    else_ifs,
                    else_start,
                    else_body,
                }
            }
            ChunkPoll::Stopped(stopped) => match stopped.terminator {
                ChunkTerminator::End(end) => {
                    let mut else_block = stopped.block;
                    // Match the established parser's source spans. An else block
                    // begins at `else`; its end belongs to the surrounding if.
                    else_block.span = else_start;
                    self.push_if(start, *condition, body, else_ifs, Some(else_block), end);
                }
                ChunkTerminator::Until(_)
                | ChunkTerminator::ElseIf(_)
                | ChunkTerminator::Else(_) => {
                    unreachable!("end-only else block accepted another terminator")
                }
            },
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::ForName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.state = ChunkState::ForAfterName {
                    start,
                    name: Name {
                        text: text.into(),
                        span: token.span,
                    },
                };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_after_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::ForAfterName { start, name } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Assign => {
                self.state = ChunkState::ForNumericStart { start, name };
            }
            TokenKind::Comma => {
                self.state = ChunkState::ForGenericName {
                    start,
                    names: vec![name],
                };
            }
            TokenKind::In => {
                self.state = ChunkState::ForGenericExpressions {
                    start,
                    names: vec![name],
                    expressions: Vec::new(),
                };
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected 'in' after loop names"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_generic_name(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::ForGenericName { start, mut names } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                names.push(Name {
                    text: text.into(),
                    span: token.span,
                });
                self.state = ChunkState::ForGenericAfterNames { start, names };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_generic_after_names(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::ForGenericAfterNames { start, names } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Comma => {
                self.state = ChunkState::ForGenericName { start, names };
            }
            TokenKind::In => {
                self.state = ChunkState::ForGenericExpressions {
                    start,
                    names,
                    expressions: Vec::new(),
                };
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected 'in' after loop names"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_generic_expressions(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForGenericExpressions {
            start,
            names,
            mut expressions,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ForGenericExpressions {
                    start,
                    names,
                    expressions,
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("for iterator expression needs 'do'"),
            PrattPoll::Boundary(value, boundary) => {
                expressions.push(value);
                match boundary.kind {
                    TokenKind::Comma => {
                        self.state = ChunkState::ForGenericExpressions {
                            start,
                            names,
                            expressions,
                        };
                    }
                    TokenKind::Do => {
                        self.state = ChunkState::ForGenericBody {
                            start,
                            names,
                            expressions,
                            body: Some(Box::new(Self::block(
                                self.source_len,
                                start,
                                self.loop_depth + 1,
                                self.vararg_allowed,
                            ))),
                        };
                    }
                    _ => {
                        return ControlFlow::Break(self.fail_message(
                            boundary.span,
                            "expected 'do' after iterator expressions",
                        ));
                    }
                }
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_numeric_start(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForNumericStart { start, name } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ForNumericStart { start, name };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("numeric for start needs ','"),
            PrattPoll::Boundary(initial, boundary) => {
                if !matches!(boundary.kind, TokenKind::Comma) {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected ',' after loop start"),
                    );
                }
                self.state = ChunkState::ForNumericLimit {
                    start,
                    name,
                    initial: Box::new(initial),
                };
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_numeric_limit(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForNumericLimit {
            start,
            name,
            initial,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ForNumericLimit {
                    start,
                    name,
                    initial,
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("numeric for limit needs 'do'"),
            PrattPoll::Boundary(limit, boundary) => match boundary.kind {
                TokenKind::Comma => {
                    self.state = ChunkState::ForNumericStep {
                        start,
                        name,
                        initial,
                        limit: Box::new(limit),
                    };
                }
                TokenKind::Do => {
                    self.state = ChunkState::ForNumericBody {
                        start,
                        name,
                        initial,
                        limit: Box::new(limit),
                        step: None,
                        body: Some(Box::new(Self::block(
                            self.source_len,
                            start,
                            self.loop_depth + 1,
                            self.vararg_allowed,
                        ))),
                    };
                }
                _ => {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected 'do' after numeric for"),
                    );
                }
            },
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_numeric_step(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForNumericStep {
            start,
            name,
            initial,
            limit,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::pratt_input(&mut self.expression, input) {
            PrattPoll::NeedToken => {
                self.state = ChunkState::ForNumericStep {
                    start,
                    name,
                    initial,
                    limit,
                };
            }
            PrattPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
            PrattPoll::Parsed(_) => unreachable!("numeric for step needs 'do'"),
            PrattPoll::Boundary(step, boundary) => {
                if !matches!(boundary.kind, TokenKind::Do) {
                    return ControlFlow::Break(
                        self.fail_message(boundary.span, "expected 'do' after numeric for"),
                    );
                }
                self.state = ChunkState::ForNumericBody {
                    start,
                    name,
                    initial,
                    limit,
                    step: Some(Box::new(step)),
                    body: Some(Box::new(Self::block(
                        self.source_len,
                        start,
                        self.loop_depth + 1,
                        self.vararg_allowed,
                    ))),
                };
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_numeric_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForNumericBody {
            start,
            name,
            initial,
            limit,
            step,
            mut body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::ForNumericBody {
                    start,
                    name,
                    initial,
                    limit,
                    step,
                    body,
                }
            }
            ChunkPoll::Stopped(ChunkStopped { block, .. }) => {
                self.statements.push(Statement::NumericFor {
                    name,
                    start: initial,
                    limit,
                    step,
                    span: start.join(block.span),
                    body: block,
                });
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_for_generic_body(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::ForGenericBody {
            start,
            names,
            expressions,
            mut body,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::chunk_input(&mut body, input) {
            ChunkPoll::NeedToken => {
                self.state = ChunkState::ForGenericBody {
                    start,
                    names,
                    expressions,
                    body,
                }
            }
            ChunkPoll::Stopped(ChunkStopped { block, .. }) => {
                self.statements.push(Statement::GenericFor {
                    names,
                    expressions,
                    span: start.join(block.span),
                    body: block,
                });
            }
            ChunkPoll::Parsed(_) => unreachable!("nested block cannot reach root EOF"),
            ChunkPoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_function_global_name(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::FunctionGlobalName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.state = ChunkState::FunctionPath {
                    start,
                    segments: vec![Name {
                        text: text.into(),
                        span: token.span,
                    }],
                };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected local name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_function_path(&mut self, input: ChunkInput) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::FunctionPath { start, segments } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Dot => {
                self.state = ChunkState::FunctionPathSegment {
                    start,
                    segments,
                    method: false,
                };
            }
            TokenKind::Colon => {
                self.state = ChunkState::FunctionPathSegment {
                    start,
                    segments,
                    method: true,
                };
            }
            TokenKind::LParen => {
                let mut frame = FunctionFrame::new(self.source_len, start);
                match frame.resume(token) {
                    FunctionFramePoll::NeedToken => {
                        self.state = ChunkState::FunctionDeclaration {
                            start,
                            name: FunctionName {
                                segments,
                                method: false,
                                span: start,
                            },
                            frame: Box::new(frame),
                        };
                    }
                    FunctionFramePoll::Parsed(_) => {
                        unreachable!("opening function cannot finish")
                    }
                    FunctionFramePoll::Failed(error) => {
                        return ControlFlow::Break(self.fail(error));
                    }
                }
            }
            _ => {
                return ControlFlow::Break(
                    self.fail_message(token.span, "expected '(' after function name"),
                );
            }
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_function_path_segment(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::FunctionPathSegment {
            start,
            mut segments,
            method,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                segments.push(Name {
                    text: text.into(),
                    span: token.span,
                });
                if method {
                    // A method's path is complete. The next token must be '('.
                    self.state = ChunkState::FunctionDeclaration {
                        start,
                        name: FunctionName {
                            segments,
                            method: true,
                            span: start,
                        },
                        frame: Box::new(FunctionFrame::new(self.source_len, start)),
                    };
                } else {
                    self.state = ChunkState::FunctionPath { start, segments };
                }
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected local name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_function_declaration(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::FunctionDeclaration {
            start,
            mut name,
            mut frame,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::frame_input(&mut frame, input) {
            FunctionFramePoll::NeedToken => {
                self.state = ChunkState::FunctionDeclaration { start, name, frame };
            }
            FunctionFramePoll::Parsed(parsed) => {
                name.span = start.join(parsed.end);
                self.statements.push(Statement::Function {
                    name,
                    parameters: parsed.parameters,
                    vararg: parsed.vararg,
                    vararg_name: parsed.vararg_name,
                    body: parsed.body,
                    span: start.join(parsed.end),
                });
            }
            FunctionFramePoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_function_name(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let token = input.into_token();
        let ChunkState::LocalFunctionName { start } = self.take_state() else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match token.kind {
            TokenKind::Identifier(text) => {
                self.state = ChunkState::LocalFunctionDeclaration {
                    start,
                    name: Name {
                        text: text.into(),
                        span: token.span,
                    },
                    frame: Box::new(FunctionFrame::new(self.source_len, start)),
                };
            }
            _ => return ControlFlow::Break(self.fail_message(token.span, "expected local name")),
        }
        ControlFlow::Continue(None)
    }

    #[inline(never)]
    fn on_local_function_declaration(
        &mut self,
        input: ChunkInput,
    ) -> ControlFlow<ChunkPoll, Option<Token>> {
        let ChunkState::LocalFunctionDeclaration {
            start,
            name,
            mut frame,
        } = self.take_state()
        else {
            unreachable!("token dispatched to the wrong chunk state")
        };
        match Self::frame_input(&mut frame, input) {
            FunctionFramePoll::NeedToken => {
                self.state = ChunkState::LocalFunctionDeclaration { start, name, frame };
            }
            FunctionFramePoll::Parsed(parsed) => {
                self.statements.push(Statement::LocalFunction {
                    name,
                    parameters: parsed.parameters,
                    vararg: parsed.vararg,
                    vararg_name: parsed.vararg_name,
                    body: parsed.body,
                    span: start.join(parsed.end),
                });
            }
            FunctionFramePoll::Failed(error) => return ControlFlow::Break(self.fail(error)),
        }
        ControlFlow::Continue(None)
    }
}
