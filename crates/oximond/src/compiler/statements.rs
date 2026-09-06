//! Statement dispatch, control-flow statements, assignments, local and global declarations, and returns.

use crate::bytecode::{Instruction, OpCode};
use crate::compiler::emit::PendingCondition;
use crate::compiler::{
    CapturedTarget, CompileError, Compiler, GotoInfo, LabelInfo, Register,
    expands_open_call_results,
};
use crate::syntax::{self as syntax, Expr, Name, Span, Statement};

impl Compiler {
    pub(crate) fn statements(&mut self, statements: &[Statement]) -> Result<(), CompileError> {
        self.statements_with_trailing_scope(statements, true)
    }
    pub(crate) fn statements_with_trailing_scope(
        &mut self,
        statements: &[Statement],
        trailing_labels_leave_scope: bool,
    ) -> Result<(), CompileError> {
        for (index, statement) in statements.iter().enumerate() {
            let trailing_label = trailing_labels_leave_scope
                && matches!(statement, Statement::Label { .. })
                && statements[index + 1..]
                    .iter()
                    .all(|statement| matches!(statement, Statement::Label { .. }));
            self.statement(statement, trailing_label)?;
        }
        Ok(())
    }
    pub(crate) fn statement(
        &mut self,
        statement: &Statement,
        trailing_label: bool,
    ) -> Result<(), CompileError> {
        let first_pc = self.code.len();
        let result = self.statement_inner(statement, trailing_label);
        if result.is_ok() {
            let line = self.line_for(statement_span(statement));
            for pc in first_pc..self.code.len() {
                self.instruction_lines.insert_if_absent(pc, line);
            }
            if !matches!(statement, Statement::Return { .. }) {
                self.reclaim_temporaries();
            }
        }
        result
    }
    #[allow(clippy::too_many_lines)] // Statement lowering is one exhaustive AST dispatch.
    pub(crate) fn statement_inner(
        &mut self,
        statement: &Statement,
        trailing_label: bool,
    ) -> Result<(), CompileError> {
        match statement {
            Statement::Do(block) => {
                let local_floor = self.next_register;
                self.push_scope();
                let result = self.statements(&block.statements);
                self.close_current_scope();
                self.pop_scope();
                result?;
                let captured_locals = self.captured_registers_from(local_floor);
                self.close_captured_registers(&captured_locals);
                for register in captured_locals {
                    self.captured_registers.remove(register);
                }
            }
            Statement::Local {
                names,
                attributes,
                values,
                span,
            } => self.local(names, attributes, values, *span)?,
            Statement::Global {
                mode,
                all_const,
                names,
                const_names,
                values,
                span,
            } => {
                self.global(*mode, *all_const, names, const_names, values, *span)?;
            }
            Statement::Assign {
                targets,
                values,
                span,
            } => self.assign(targets, values, *span)?,
            Statement::Function {
                name,
                parameters,
                vararg,
                vararg_name,
                body,
                span,
            } => {
                self.function_statement(
                    name,
                    parameters,
                    *vararg,
                    vararg_name.as_ref(),
                    body,
                    *span,
                )?;
            }
            Statement::LocalFunction {
                name,
                parameters,
                vararg,
                vararg_name,
                body,
                span,
            } => {
                self.local_function(name, parameters, *vararg, vararg_name.as_ref(), body, *span)?;
            }
            Statement::Return { values, span } => self.return_statement(values, *span)?,
            Statement::If {
                condition,
                body,
                else_ifs,
                else_body,
                span,
            } => self.if_statement(condition, body, else_ifs, else_body.as_ref(), *span)?,
            Statement::While {
                condition,
                body,
                span,
            } => self.while_statement(condition, body, *span)?,
            Statement::Repeat {
                body,
                condition,
                span,
            } => self.repeat_statement(body, condition, *span)?,
            Statement::Break { span } => self.break_statement(*span)?,
            Statement::Goto { name, span } => self.goto_statement(name, *span),
            Statement::Label { name, span } => {
                self.label_statement(name, *span, trailing_label)?;
            }
            Statement::NumericFor {
                name,
                start,
                limit,
                step,
                body,
                span,
            } => self.numeric_for(name, start, limit, step.as_deref(), body, *span)?,
            Statement::GenericFor {
                names,
                expressions,
                body,
                span,
            } => self.generic_for(names, expressions, body, *span)?,
            Statement::Call(call) => {
                self.expression(call, 1)?;
            }
        }
        Ok(())
    }
    pub(crate) fn if_statement(
        &mut self,
        condition: &Expr,
        body: &syntax::Block,
        else_ifs: &[syntax::ElseIf],
        else_body: Option<&syntax::Block>,
        span: Span,
    ) -> Result<(), CompileError> {
        self.control_flow_depth += 1;
        let result = self.if_statement_inner(condition, body, else_ifs, else_body, span);
        self.control_flow_depth -= 1;
        result
    }
    pub(crate) fn if_statement_inner(
        &mut self,
        condition: &Expr,
        body: &syntax::Block,
        else_ifs: &[syntax::ElseIf],
        else_body: Option<&syntax::Block>,
        span: Span,
    ) -> Result<(), CompileError> {
        let mut end_jumps = Vec::with_capacity(else_ifs.len() + usize::from(else_body.is_some()));
        for index in 0..=else_ifs.len() {
            let (condition_expression, body) = if index == 0 {
                (condition, body)
            } else {
                let arm = &else_ifs[index - 1];
                (&arm.condition, &arm.body)
            };
            let condition_start = self.code.len();
            let false_jumps = self.condition_false_jumps(condition_expression)?;
            self.record_instruction_line_range(
                condition_start,
                self.code.len(),
                self.line_for(condition_expression.span()),
            );
            let body_start = self.code.len();
            self.push_scope();
            let body_result = self.statements(&body.statements);
            self.close_current_scope();
            self.pop_scope();
            self.record_unmapped_instruction_line_range(
                body_start,
                self.code.len(),
                self.block_exit_line(body),
            );
            body_result?;
            let has_following_arm = index < else_ifs.len();
            if has_following_arm || else_body.is_some() {
                let skip = self.code.len();
                self.code
                    .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
                self.instruction_lines
                    .insert(skip, self.block_exit_line(body));
                end_jumps.push(skip);
            }
            for false_jump in false_jumps {
                self.patch_condition_jump(false_jump, self.code.len(), span)?;
            }
        }
        if let Some(else_body) = else_body {
            let body_start = self.code.len();
            self.push_scope();
            let body_result = self.statements(&else_body.statements);
            self.close_current_scope();
            self.pop_scope();
            self.record_unmapped_instruction_line_range(
                body_start,
                self.code.len(),
                self.block_exit_line(else_body),
            );
            body_result?;
        }
        let end = self.code.len();
        for jump in end_jumps {
            let offset = Self::relative(jump, end, span)?;
            self.code[jump] = Instruction::aj(OpCode::Jump, i32::from(offset))
                .expect("i16 offset fits in a jump");
        }
        Ok(())
    }
    pub(crate) fn while_statement(
        &mut self,
        condition: &Expr,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        self.control_flow_depth += 1;
        let result = self.while_statement_inner(condition, body, span);
        self.control_flow_depth -= 1;
        result
    }
    pub(crate) fn while_statement_inner(
        &mut self,
        condition: &Expr,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        let loop_start = self.code.len();
        let temporary_floor = self.next_register;
        let condition_start = self.code.len();
        let condition_line = self.line_for(condition.span());
        // `while true` tests nothing, as in PUC Lua and LuaJIT: the loop is
        // its body and the jump back.
        let exits = if matches!(Self::strip_parentheses(condition), Expr::Boolean(true, _)) {
            Vec::new()
        } else {
            self.condition_false_jumps(condition)?
        };
        self.record_instruction_line_range(condition_start, self.code.len(), condition_line);
        self.push_scope();
        // The condition's temporaries are dead in the body, and statement
        // boundaries reuse them for body locals. Every register above the
        // enclosing locals may therefore hold a captured body local.
        let loop_local_floor = temporary_floor;
        self.loop_breaks.push(Vec::new());
        self.loop_close_depths.push(self.close_scopes.len() - 1);
        let body_start = self.code.len();
        let body_result = self.statements(&body.statements);
        let breaks = self.loop_breaks.pop().expect("loop break context exists");
        self.loop_close_depths
            .pop()
            .expect("loop close context exists");
        self.close_current_scope();
        self.pop_scope();
        self.record_unmapped_instruction_line_range(
            body_start,
            self.code.len(),
            self.block_exit_line(body),
        );
        body_result?;
        let captured_locals = self.captured_registers_from(loop_local_floor);
        self.close_captured_registers(&captured_locals);
        let back = Self::relative(self.code.len(), loop_start, span)?;
        let back_jump = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, i32::from(back)).expect("i16 offset fits"));
        self.instruction_lines
            .insert(back_jump, self.block_exit_line(body));
        let break_cleanup = self.code.len();
        self.close_captured_registers(&captured_locals);
        let end = self.code.len();
        for exit in exits {
            self.patch_condition_jump(exit, end, span)?;
        }
        self.patch_breaks(breaks, break_cleanup, span)
    }
    pub(crate) fn repeat_statement(
        &mut self,
        body: &syntax::Block,
        condition: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        self.control_flow_depth += 1;
        let result = self.repeat_statement_inner(body, condition, span);
        self.control_flow_depth -= 1;
        result
    }
    pub(crate) fn repeat_statement_inner(
        &mut self,
        body: &syntax::Block,
        condition: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        let loop_start = self.code.len();
        if self.repeat_header_line {
            // LuaJIT's BC_LOOP carries the repeat keyword's line even when
            // the body starts on the following line.
            self.code
                .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
            self.instruction_lines
                .insert(loop_start, self.line_for(span));
        }
        // Repeat's condition runs in the body's lexical scope. This is the
        // small but important difference from `while`.
        self.push_scope();
        let loop_local_floor = self.next_register;
        self.loop_breaks.push(Vec::new());
        self.loop_close_depths.push(self.close_scopes.len() - 1);
        let result = (|| {
            let body_start = self.code.len();
            self.statements_with_trailing_scope(&body.statements, false)?;
            let condition_start = self.code.len();
            // A to-be-closed local runs __close between the condition and
            // the branch, so the test form (which compares at the branch)
            // would reorder the comparison and the close. Compute the
            // boolean first in that case.
            let has_close = self
                .close_scopes
                .last()
                .is_some_and(|scope| !scope.is_empty());
            // Upvalues the body captured are closed between the condition
            // and its branch, which a jump chain has no single place for.
            // A closure in the condition could capture more.
            let chain = !has_close
                && self.captured_registers_from(loop_local_floor).is_empty()
                && !Self::contains_function(condition);
            let pending = if chain {
                self.condition_false_jumps(condition)
                    .map(RepeatCondition::Jumps)
            } else if has_close {
                self.expression(condition, 1)
                    .map(PendingCondition::Value)
                    .map(RepeatCondition::Pending)
            } else {
                self.condition_operands(condition)
                    .map(RepeatCondition::Pending)
            };
            pending.map(|result| {
                (
                    result,
                    body_start,
                    condition_start,
                    self.line_for(condition.span()),
                )
            })
        })();
        let breaks = self.loop_breaks.pop().expect("loop break context exists");
        self.loop_close_depths
            .pop()
            .expect("loop close context exists");
        self.close_current_scope();
        self.pop_scope();
        let (condition, body_start, condition_start, condition_line) = result?;
        self.record_unmapped_instruction_line_range(
            body_start,
            condition_start,
            self.block_exit_line(body),
        );
        let captured_locals = self.captured_registers_from(loop_local_floor);
        self.close_captured_registers(&captured_locals);
        let exits = match condition {
            RepeatCondition::Jumps(jumps) => jumps,
            RepeatCondition::Pending(pending) => vec![self.emit_condition_jump(pending)],
        };
        for exit in exits {
            self.patch_condition_jump(exit, loop_start, span)?;
        }
        self.record_instruction_line_range(condition_start, self.code.len(), condition_line);
        let break_cleanup = self.code.len();
        self.close_captured_registers(&captured_locals);
        self.patch_breaks(breaks, break_cleanup, span)
    }
    pub(crate) fn break_statement(&mut self, span: Span) -> Result<(), CompileError> {
        let Some(&close_depth) = self.loop_close_depths.last() else {
            return Err(CompileError::BreakOutsideLoop { span });
        };
        self.close_scopes_from(close_depth);
        let jump = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
        let Some(breaks) = self.loop_breaks.last_mut() else {
            return Err(CompileError::BreakOutsideLoop { span });
        };
        breaks.push(jump);
        Ok(())
    }
    pub(crate) fn goto_statement(&mut self, name: &Name, span: Span) {
        let instruction = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
        self.gotos.push(GotoInfo {
            name: name.text.to_string(),
            span,
            instruction,
            scope_path: self.scope_path.clone(),
            locals: self.active_local_bindings(),
            captured_locals: self.captured_registers,
            close_locals: self.active_close_registers_in_declaration_order(),
            global_barriers: self.active_global_barriers().keys().copied().collect(),
        });
    }
    pub(crate) fn label_statement(
        &mut self,
        name: &Name,
        span: Span,
        trailing_label: bool,
    ) -> Result<(), CompileError> {
        if self.labels.iter().any(|label| {
            *label.name == *name.text
                && (label.scope_path == self.scope_path
                    || (self.strict_label_shadowing
                        && self.scope_path.starts_with(&label.scope_path)))
        }) {
            return Err(CompileError::DuplicateLabel {
                span,
                name: name.text.to_string(),
            });
        }
        let mut locals = self.active_local_info();
        let mut global_barriers = self.active_global_barriers();
        let mut close_locals = self.active_close_registers();
        if trailing_label {
            for local in self
                .debug_local_scopes
                .last()
                .expect("label has a lexical scope")
            {
                locals.remove(&local.order);
            }
            for (order, _) in self
                .global_barrier_scopes
                .last()
                .expect("label has a lexical global scope")
            {
                global_barriers.remove(order);
            }
            for register in self
                .close_scopes
                .last()
                .expect("label has a lexical close scope")
            {
                close_locals.remove(register);
            }
        }
        self.labels.push(LabelInfo {
            name: name.text.to_string(),
            pc: self.code.len(),
            scope_path: self.scope_path.clone(),
            locals,
            close_locals,
            global_barriers,
        });
        Ok(())
    }
    pub(crate) fn return_statement(
        &mut self,
        expressions: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        if expressions.is_empty() {
            self.close_all_scopes();
            self.code.push(Instruction::halt());
            return Ok(());
        }
        if let [
            Expr::Call {
                callee,
                arguments,
                span: call_span,
            },
        ] = expressions
        {
            return self.tail_call(callee, arguments, *call_span);
        }
        if let [
            Expr::MethodCall {
                argument_span,
                receiver,
                method,
                arguments,
                ..
            },
        ] = expressions
        {
            return self.tail_method_call(receiver, method, arguments, *argument_span);
        }
        if let Some((Expr::Vararg(_), fixed)) = expressions.split_last() {
            self.disable_compatibility_arg();
            let mut values = Vec::with_capacity(fixed.len());
            for expression in fixed {
                values.push(self.expression(expression, 1)?);
            }
            let base = self.pack_or_empty(&values, span)?;
            let count =
                u8::try_from(values.len()).map_err(|_| CompileError::RegisterExhausted { span })?;
            self.close_all_scopes();
            self.code.push(Instruction::return_vararg(base, count));
            return Ok(());
        }
        if let Some((expression, fixed)) = expressions.split_last()
            && expands_open_call_results(expression)
        {
            let mut values = Vec::with_capacity(fixed.len());
            for expression in fixed {
                values.push(self.expression(expression, 1)?);
            }
            let base = self.pack_or_empty(&values, span)?;
            self.open_call_expression(expression)?;
            let count =
                u8::try_from(values.len()).map_err(|_| CompileError::RegisterExhausted { span })?;
            self.close_all_scopes();
            self.code.push(Instruction::return_open(base, count));
            return Ok(());
        }
        if let [expression] = expressions {
            let value = self.expression(expression, 1)?;
            self.close_all_scopes();
            self.code.push(Instruction::return_(value));
            return Ok(());
        }
        let values = self.values(expressions, expressions.len(), span)?;
        // Compact mode already leaves the values in consecutive fresh registers.
        let base = if self.compact_registers {
            values[0]
        } else {
            self.pack(&values, span)?
        };
        let count =
            u8::try_from(values.len()).map_err(|_| CompileError::RegisterExhausted { span })?;
        self.close_all_scopes();
        self.code.push(Instruction::return_values(base, count));
        Ok(())
    }
    pub(crate) fn local(
        &mut self,
        names: &[Name],
        attributes: &[syntax::LocalAttribute],
        values: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        // Initializers run outside the new bindings: `local x = x` reads outer x.
        // A register no instruction has used yet is nil already, and the
        // declaration stays bytecode-free, which keeps `local x` from
        // becoming an executable source line. A register that held a
        // temporary, or a declaration inside a loop or with gotos, which
        // runs again, starts with a LoadNil.
        if values.is_empty() {
            if self.scopes.is_empty() {
                self.push_scope();
            }
            let start_pc = self.code.len();
            let used_registers = self.register_count;
            let reruns = self.goto_labels || !self.loop_breaks.is_empty();
            let mut locals = Vec::with_capacity(names.len());
            for name in names {
                let register = self.fresh(name.span)?;
                locals.push(register);
                self.scopes.insert(name.text.clone(), register);
                self.bind_debug_local(name, register, start_pc);
            }
            self.mark_readonly_locals(names, attributes);
            self.mark_close_locals(&locals, attributes);
            for register in locals {
                if reruns || register < used_registers {
                    self.code.push(Instruction::load_nil(register));
                }
            }
            return Ok(());
        }
        // Lua folds the last variable of a declaration with as many values
        // when it is a `<const>` bound to an integer constant.
        let folded = if names.len() == values.len()
            && matches!(attributes.last(), Some(syntax::LocalAttribute::Const))
        {
            values.last().and_then(|value| self.integer_constant(value))
        } else {
            None
        };
        // Allocate the locals below the initializer temporaries. The names
        // bind only afterwards, so initializers still see outer bindings,
        // and the temporaries above the packed locals clear without moves.
        let mut locals = Vec::with_capacity(names.len());
        for name in names {
            locals.push(self.fresh(name.span)?);
        }
        let values = if let [value @ Expr::Vararg(_)] = values
            && !self.compact_registers
            && locals
                .windows(2)
                .all(|pair| usize::from(pair[1]) == usize::from(pair[0]) + 1)
        {
            // `local a, b, c = ...` reads the varargs into the locals
            // themselves, with no temporaries to move out of.
            let start_pc = self.code.len();
            let count = u8::try_from(locals.len())
                .map_err(|_| CompileError::RegisterExhausted { span: value.span() })?;
            self.disable_compatibility_arg();
            self.code.push(Instruction::vararg(locals[0], count));
            self.record_value_lines(start_pc, value);
            vec![None; locals.len()]
        } else if let ([local], [value]) = (&locals[..], values) {
            let start_pc = self.code.len();
            let register = self.expression(value, 1)?;
            // The retargeted instruction writes the local directly.
            let pending =
                if register == *local || self.retarget_last_scalar_result(register, *local) {
                    Vec::new()
                } else {
                    vec![Some(register)]
                };
            self.record_value_lines(start_pc, value);
            pending
        } else if names.len() == values.len() && !self.compact_registers {
            // One value per local: each initializer lands in its local
            // directly when its last instruction can be retargeted. The
            // locals are not bound yet, so later initializers cannot see them.
            let mut pending = Vec::new();
            for (local, value) in locals.iter().copied().zip(values) {
                let start_pc = self.code.len();
                let register = self.expression(value, 1)?;
                if register == local || self.retarget_last_scalar_result(register, local) {
                    pending.push(None);
                } else if self.register_may_change(register, &[]) {
                    pending.push(Some(self.copy(register, value.span())?));
                } else {
                    pending.push(Some(register));
                }
                self.record_value_lines(start_pc, value);
            }
            pending
        } else {
            self.write_values(values, names.len(), span, &[])?
                .into_iter()
                .map(Some)
                .collect()
        };
        let after_values = self.code.len();
        for (local, value) in locals.iter().copied().zip(values) {
            if let Some(value) = value {
                self.code.push(Instruction::move_(local, value));
            }
        }
        if self.scopes.is_empty() {
            self.push_scope();
        }
        // The locals start after the moves that fill them. Until then their
        // registers hold stale temporaries, which a collection must not root
        // and `debug.getlocal` must not show.
        let start_pc = self.code.len();
        for (name, register) in names.iter().zip(locals.iter().copied()) {
            self.scopes.insert(name.text.clone(), register);
            self.bind_debug_local(name, register, start_pc);
        }
        self.mark_readonly_locals(names, attributes);
        if let Some(last) = names.last() {
            self.mark_constant_local(last, folded);
        }
        // Stock Lua emits what follows the value list (nil fills, moves)
        // after reading the statement's last token.
        let end = self.code.len();
        self.record_unmapped_instruction_line_range(after_values, end, self.end_line(span));
        self.mark_close_locals(&locals, attributes);
        Ok(())
    }
    pub(crate) fn global(
        &mut self,
        mode: syntax::GlobalDeclarationMode,
        all_const: bool,
        names: &[Name],
        const_names: &[bool],
        values: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        if self.scopes.is_empty() {
            self.push_scope();
        }
        // The parser only creates this statement for Lua 5.5 syntax. Until a
        // declaration exists, name resolution can skip the global-scope walk
        // entirely. Carry this fact into nested function compilers.
        self.has_global_declarations = true;
        let global_function = matches!(
            values,
            [Expr::Function { span: function_span, .. }] if names.len() == 1 && *function_span == span
        );
        // As with local declarations, initializer expressions see the bindings
        // that were active before this declaration. The `global function`
        // shorthand is recursive, so its body sees the new global binding.
        let initialized_values = if values.is_empty() || global_function {
            None
        } else {
            Some(self.values(values, names.len(), span)?)
        };
        let barrier_name = match mode {
            syntax::GlobalDeclarationMode::All => "*".to_owned(),
            syntax::GlobalDeclarationMode::None => "none".to_owned(),
            syntax::GlobalDeclarationMode::Names => names
                .first()
                .map_or_else(|| "?".to_owned(), |name| name.text.to_string()),
        };
        let barrier = self.next_global_barrier;
        self.next_global_barrier = self.next_global_barrier.saturating_add(1);
        self.global_barrier_scopes
            .last_mut()
            .expect("global declaration has a barrier scope")
            .push((barrier, barrier_name));
        let globals = self
            .global_scopes
            .last_mut()
            .expect("global declaration has a scope");
        globals.restricted = true;
        if mode == syntax::GlobalDeclarationMode::Names
            && names.iter().any(|name| &*name.text == "_ENV")
        {
            // `_ENV` is the implicit lexical environment binding. A declaration
            // containing it admits none of the listed globals, so later uses
            // are diagnosed under the declaration barrier.
            return Ok(());
        }
        match mode {
            syntax::GlobalDeclarationMode::All => {
                globals.allow_all = true;
                globals.all_readonly = all_const;
                globals.deny_all = false;
            }
            syntax::GlobalDeclarationMode::None => {
                globals.allow_all = false;
                globals.all_readonly = false;
                globals.deny_all = true;
            }
            syntax::GlobalDeclarationMode::Names => {
                for (name, readonly) in names.iter().zip(const_names) {
                    globals.names.insert(name.text.to_string(), *readonly);
                }
            }
        }
        if values.is_empty() {
            return Ok(());
        }
        let values = match initialized_values {
            Some(values) => values,
            None => self.values(values, names.len(), span)?,
        };
        let mut targets = Vec::with_capacity(names.len());
        for name in names {
            targets.push(self.capture_global_name(name)?);
        }
        for (target, value) in targets.into_iter().zip(values) {
            match target {
                CapturedTarget::Global(name) => self.emit_declare_global(value, name),
                CapturedTarget::Index { table, key, .. } => {
                    self.code
                        .push(Instruction::declare_table(table, key, value));
                }
                CapturedTarget::Local(_)
                | CapturedTarget::Upvalue(_)
                | CapturedTarget::Field { .. }
                | CapturedTarget::TabUp { .. } => {
                    unreachable!("a declared global cannot resolve to a local")
                }
            }
        }
        Ok(())
    }
    pub(crate) fn assign(
        &mut self,
        targets: &[Expr],
        expressions: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        for target in targets {
            if let Expr::Name(name) = target
                && self.is_readonly_name(&name.text)
            {
                return Err(CompileError::AssignToReadOnly {
                    span: name.span,
                    name: name.text.to_string(),
                });
            }
            if let Expr::Name(name) = target
                && self.is_readonly_global_target(&name.text)
            {
                return Err(CompileError::AssignToReadOnly {
                    span: name.span,
                    name: name.text.to_string(),
                });
            }
        }
        // Lua evaluates indexed target operands before the RHS. All writes
        // still wait until every target and RHS value has been evaluated.
        let assigned_locals = self.assigned_local_registers(targets);
        let mut captured = Vec::with_capacity(targets.len());
        // Like `check_conflict`: when the statement also assigns the `_ENV`
        // upvalue, global targets read it before the right-hand side.
        let env_by_copy = targets.len() > 1 && self.assigns_environment_upvalue(targets)?;
        for target in targets {
            captured.push(self.capture_target(target, &assigned_locals, env_by_copy)?);
        }
        if captured.len() == 1 && expressions.len() == 1 {
            let target = captured.pop().expect("one assignment target exists");
            let expression = &expressions[0];
            let value = self.expression(expression, 1)?;
            match target {
                CapturedTarget::Local(destination) => {
                    if value != destination && !self.retarget_last_scalar_result(value, destination)
                    {
                        self.code.push(Instruction::move_(destination, value));
                    }
                }
                CapturedTarget::Upvalue(index) => {
                    self.code.push(Instruction::set_upvalue(value, index));
                }
                CapturedTarget::Global(name) => self.emit_set_global(value, name),
                CapturedTarget::Index { table, key, name } => {
                    self.emit_set_table(table, key, value, name);
                }
                CapturedTarget::Field { table, field, name } => {
                    self.emit_set_field(table, field, value, name);
                }
                CapturedTarget::TabUp { upvalue, field } => {
                    self.code
                        .push(Instruction::set_tab_up(upvalue, field, value));
                }
            }
            return Ok(());
        }
        let slots = self.assigned_local_slots(targets);
        let values = self.write_values(expressions, targets.len(), span, &slots)?;
        let after_values = self.code.len();
        for (target, value) in captured.into_iter().zip(values) {
            match target {
                CapturedTarget::Local(destination) => {
                    // A value read in place from its own target needs no copy.
                    if destination != value {
                        self.code.push(Instruction::move_(destination, value));
                    }
                }
                CapturedTarget::Upvalue(index) => {
                    self.code.push(Instruction::set_upvalue(value, index));
                }
                CapturedTarget::Global(name) => {
                    self.emit_set_global(value, name);
                }
                CapturedTarget::Index { table, key, name } => {
                    self.emit_set_table(table, key, value, name);
                }
                CapturedTarget::Field { table, field, name } => {
                    self.emit_set_field(table, field, value, name);
                }
                CapturedTarget::TabUp { upvalue, field } => {
                    self.code
                        .push(Instruction::set_tab_up(upvalue, field, value));
                }
            }
        }
        // As for a declaration, the stores follow the statement's last token.
        let end = self.code.len();
        self.record_unmapped_instruction_line_range(after_values, end, self.end_line(span));
        Ok(())
    }

    pub(crate) fn retarget_last_scalar_result(
        &mut self,
        source: Register,
        destination: Register,
    ) -> bool {
        if source.checked_add(1) != Some(self.next_register) {
            return false;
        }
        let Some(instruction) = self.code.last().copied() else {
            return false;
        };
        let Ok(opcode) = instruction.opcode() else {
            return false;
        };
        // Only a temporary that the last instruction just produced may be
        // renamed. A local initialized in place is also the newest register,
        // but other code still reads it under its own name.
        if instruction.a() != source || self.scopes.registers().any(|register| register == source) {
            return false;
        }
        let replacement = match opcode {
            OpCode::Negate | OpCode::Not | OpCode::Length | OpCode::BitNot => {
                Instruction::unary(opcode, destination, instruction.b())
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::FloorDivide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual => {
                Instruction::binary(opcode, destination, instruction.b(), instruction.c())
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                Instruction::abc(opcode, destination, instruction.b(), instruction.c())
            }
            OpCode::CallOne => Instruction::call_one(destination, instruction.b(), instruction.c()),
            OpCode::GetTable => {
                Instruction::get_table(destination, instruction.b(), instruction.c())
            }
            OpCode::GetField => {
                Instruction::get_field(destination, instruction.b(), instruction.c())
            }
            OpCode::LoadConstant => Instruction::load_constant(destination, instruction.d()),
            OpCode::GetUpvalue => Instruction::get_upvalue(destination, instruction.d()),
            OpCode::GetTabUp => {
                Instruction::get_tab_up(destination, instruction.b(), instruction.c())
            }
            OpCode::GetGlobal => Instruction::get_global(destination, instruction.d()),
            OpCode::NewTable => Instruction::abc(
                OpCode::NewTable,
                destination,
                instruction.b(),
                instruction.c(),
            ),
            _ => return false,
        };
        *self.code.last_mut().expect("the result instruction exists") = replacement;
        self.next_register -= 1;
        true
    }
    pub(crate) fn values(
        &mut self,
        expressions: &[Expr],
        wanted: usize,
        span: Span,
    ) -> Result<Vec<Register>, CompileError> {
        self.values_with(expressions, wanted, span, None)
    }

    /// Evaluates a value list. With `assigned`, a value is copied only when
    /// [`Self::register_may_change`] says so, and otherwise stays in place.
    fn values_with(
        &mut self,
        expressions: &[Expr],
        wanted: usize,
        span: Span,
        assigned: Option<&[Option<Register>]>,
    ) -> Result<Vec<Register>, CompileError> {
        if self.compact_registers {
            return self.compact_values(expressions, wanted, span, assigned.is_some());
        }
        let mut values = Vec::with_capacity(wanted);
        for (index, expression) in expressions.iter().enumerate() {
            let results = if index + 1 == expressions.len()
                && (expands_open_call_results(expression) || matches!(expression, Expr::Vararg(_)))
            {
                wanted.saturating_sub(values.len()).max(1)
            } else {
                1
            };
            let start_pc = self.code.len();
            let value = self.expression(expression, results)?;
            for offset in 0..results {
                if values.len() == wanted {
                    break;
                }
                let source = value
                    .checked_add(u8::try_from(offset).expect("at most 255 registers"))
                    .ok_or(CompileError::RegisterExhausted { span })?;
                if assigned.is_some_and(|assigned| {
                    !self.register_may_change_at(source, values.len(), assigned)
                }) {
                    values.push(source);
                } else {
                    values.push(self.copy(source, span)?);
                }
            }
            if assigned.is_some() {
                self.record_value_lines(start_pc, expression);
            }
        }
        let fill_pc = self.code.len();
        while values.len() < wanted {
            values.push(self.nil(span)?);
        }
        if assigned.is_some() {
            // Stock Lua fills missing values after the list's last token.
            let end = self.code.len();
            self.record_unmapped_instruction_line_range(fill_pc, end, self.end_line(span));
        }
        Ok(values)
    }
    /// [`Self::values`] under compact allocation: the values land in
    /// consecutive fresh registers and scratch registers are released.
    ///
    /// `value_lines` gives each value its own line, for the value list of a
    /// declaration or assignment (see [`Self::record_value_lines`]).
    fn compact_values(
        &mut self,
        expressions: &[Expr],
        wanted: usize,
        span: Span,
        value_lines: bool,
    ) -> Result<Vec<Register>, CompileError> {
        let mut values = Vec::with_capacity(wanted);
        for (index, expression) in expressions.iter().enumerate() {
            let multiple = index + 1 == expressions.len()
                && (expands_open_call_results(expression) || matches!(expression, Expr::Vararg(_)));
            let start_pc = self.code.len();
            if !multiple && values.len() < wanted {
                values.push(self.expression_to_next(expression)?);
                if value_lines {
                    self.record_value_lines(start_pc, expression);
                }
                continue;
            }
            let results = if multiple {
                wanted.saturating_sub(values.len()).max(1)
            } else {
                1
            };
            let mark = self.next_register;
            let value = self.expression(expression, results)?;
            self.release_registers(mark);
            // The copies go to consecutive registers from `mark`, never above
            // their sources, so copying upward in order reads each source
            // before any copy overwrites it.
            for offset in 0..results {
                if values.len() == wanted {
                    break;
                }
                let source = value
                    .checked_add(u8::try_from(offset).expect("at most 255 registers"))
                    .ok_or(CompileError::RegisterExhausted { span })?;
                let slot = self.fresh(span)?;
                if slot != source {
                    self.code.push(Instruction::move_(slot, source));
                }
                values.push(slot);
            }
            if value_lines {
                self.record_value_lines(start_pc, expression);
            }
        }
        let fill_pc = self.code.len();
        while values.len() < wanted {
            values.push(self.nil(span)?);
        }
        if value_lines {
            let end = self.code.len();
            self.record_unmapped_instruction_line_range(fill_pc, end, self.end_line(span));
        }
        Ok(values)
    }
    fn assigns_environment_upvalue(&mut self, targets: &[Expr]) -> Result<bool, CompileError> {
        for target in targets {
            if let Expr::Name(name) = target
                && &*name.text == "_ENV"
                && !self.name_is_forced_global(&name.text)
                && self.local_register(&name.text).is_none()
                && self.resolve_upvalue(name)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Registers of the plain locals that `targets` assign.
    fn assigned_local_registers(&self, targets: &[Expr]) -> Vec<Register> {
        targets
            .iter()
            .filter_map(|target| match target {
                Expr::Name(name) if !self.name_is_forced_global(&name.text) => {
                    self.local_register(&name.text)
                }
                _ => None,
            })
            .collect()
    }

    /// The register each target assigns, `None` for a target that is not a
    /// plain local: aligned with `targets`, for [`Self::register_may_change_at`].
    fn assigned_local_slots(&self, targets: &[Expr]) -> Vec<Option<Register>> {
        targets
            .iter()
            .map(|target| match target {
                Expr::Name(name) if !self.name_is_forced_global(&name.text) => {
                    self.local_register(&name.text)
                }
                _ => None,
            })
            .collect()
    }

    /// [`Self::register_may_change`] for the value at `position` of an
    /// assignment whose targets are written in order after every value is
    /// evaluated: only the writes of the targets before it come ahead of its
    /// read, so `a, b, c = b, c, a` copies `a` alone.
    fn register_may_change_at(
        &self,
        source: Register,
        position: usize,
        slots: &[Option<Register>],
    ) -> bool {
        slots
            .iter()
            .take(position)
            .any(|slot| *slot == Some(source))
            || self
                .closure_assigned_names
                .iter()
                .any(|name| self.local_register(name) == Some(source))
    }

    /// Whether reading `source` later, at write time, could see a different
    /// value than reading it now: an assignment in the same statement or a
    /// nested closure may change it. Fresh temporaries never change.
    fn register_may_change(&self, source: Register, assigned: &[Register]) -> bool {
        assigned.contains(&source)
            || self
                .closure_assigned_names
                .iter()
                .any(|name| self.local_register(name) == Some(source))
    }

    /// Operand of an assignment target: a plain local is used in place unless
    /// the statement or a closure may reassign it before the write. Lua's
    /// `check_conflict` copies the local in the first case.
    fn target_operand(
        &mut self,
        operand: &Expr,
        assigned: &[Register],
    ) -> Result<Register, CompileError> {
        let value = self.expression(operand, 1)?;
        if self.register_may_change(value, assigned) {
            return self.copy(value, operand.span());
        }
        Ok(value)
    }

    /// [`Self::values`] for values that are written to targets afterwards.
    /// Only a value that a write or a closure may change is copied.
    fn write_values(
        &mut self,
        expressions: &[Expr],
        wanted: usize,
        span: Span,
        assigned: &[Option<Register>],
    ) -> Result<Vec<Register>, CompileError> {
        self.values_with(expressions, wanted, span, Some(assigned))
    }

    pub(crate) fn capture_target(
        &mut self,
        target: &Expr,
        assigned: &[Register],
        env_by_copy: bool,
    ) -> Result<CapturedTarget, CompileError> {
        match target {
            Expr::Name(name) => self.capture_name(name, env_by_copy),
            // Compact mode evaluates the operands, before the right-hand
            // side, into registers of their own without extra copies.
            Expr::Index { table, key, .. } if self.compact_registers => {
                let name = self.static_value_name(table);
                let table = self.expression_to_next(table)?;
                if let Some(field) = self.field_key_index(key)? {
                    return Ok(CapturedTarget::Field { table, field, name });
                }
                let key = self.expression_to_next(key)?;
                Ok(CapturedTarget::Index { table, key, name })
            }
            Expr::Index { table, key, .. } => {
                let name = self.static_value_name(table);
                let table = self.target_operand(table, assigned)?;
                if let Some(field) = self.field_key_index(key)? {
                    return Ok(CapturedTarget::Field { table, field, name });
                }
                let key = self.target_operand(key, assigned)?;
                Ok(CapturedTarget::Index { table, key, name })
            }
            _ => unreachable!("syntax validates assignment targets"),
        }
    }
    pub(crate) fn capture_name(
        &mut self,
        name: &Name,
        env_by_copy: bool,
    ) -> Result<CapturedTarget, CompileError> {
        if self.name_is_forced_global(&name.text) {
            self.ensure_global_declared(name)?;
            self.capture_assigned_global_name(name, env_by_copy)
        } else if let Some(register) = self.local_register(&name.text) {
            Ok(CapturedTarget::Local(register))
        } else if let Some(upvalue) = self.resolve_upvalue(name)? {
            Ok(CapturedTarget::Upvalue(upvalue))
        } else {
            self.ensure_global_declared(name)?;
            self.capture_assigned_global_name(name, env_by_copy)
        }
    }
    /// A global written by plain assignment. Through a lexical `_ENV` it is
    /// a field write with a constant key, which has an inline-cache site.
    pub(crate) fn capture_assigned_global_name(
        &mut self,
        name: &Name,
        env_by_copy: bool,
    ) -> Result<CapturedTarget, CompileError> {
        if self.lexical_environment {
            let text = &*name.text;
            if let Some(field) = self.field_name_index_str(text, name.span)? {
                if let Some(environment) = self.local_register("_ENV") {
                    let table = self.copy(environment, name.span)?;
                    return Ok(CapturedTarget::Field {
                        table,
                        field,
                        name: None,
                    });
                }
                let environment = Name {
                    text: "_ENV".into(),
                    span: name.span,
                };
                if !env_by_copy
                    && let Some(upvalue) = self.resolve_upvalue(&environment)?
                    && let Ok(upvalue) = u8::try_from(upvalue)
                {
                    return Ok(CapturedTarget::TabUp { upvalue, field });
                }
            }
        }
        self.capture_global_name(name)
    }
    pub(crate) fn capture_global_name(
        &mut self,
        name: &Name,
    ) -> Result<CapturedTarget, CompileError> {
        if let Some(environment) = self.lexical_environment_value(name.span)? {
            let table = self.copy(environment, name.span)?;
            let key = self.string_constant_register(&name.text, name.span)?;
            return Ok(CapturedTarget::Index {
                table,
                key,
                name: None,
            });
        }
        Ok(CapturedTarget::Global(
            self.string_constant(&name.text, name.span)?,
        ))
    }
}

pub(crate) fn statement_span(statement: &Statement) -> Span {
    match statement {
        Statement::Do(block) => block.span,
        Statement::Local { span, .. }
        | Statement::Global { span, .. }
        | Statement::Assign { span, .. }
        | Statement::Function { span, .. }
        | Statement::LocalFunction { span, .. }
        | Statement::Return { span, .. }
        | Statement::If { span, .. }
        | Statement::While { span, .. }
        | Statement::Repeat { span, .. }
        | Statement::Break { span }
        | Statement::Goto { span, .. }
        | Statement::Label { span, .. }
        | Statement::NumericFor { span, .. }
        | Statement::GenericFor { span, .. } => *span,
        Statement::Call(expression) => expression.span(),
    }
}

/// A `repeat` condition: a finished jump chain, or operands whose branch
/// waits for the closes of captured locals.
enum RepeatCondition {
    Jumps(Vec<usize>),
    Pending(PendingCondition),
}
