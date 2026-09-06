//! Numeric and generic for loops plus goto and break patching.

use crate::bytecode::{CallSite, Constant, Instruction, OpCode};
use crate::compiler::{CompileError, Compiler, Register};
use crate::syntax::{self as syntax, Expr, Name, Span};
use std::collections::BTreeSet;

impl Compiler {
    #[allow(clippy::too_many_lines)] // Resolution validates scope entry and builds cleanup trampolines together.
    pub(crate) fn resolve_gotos(&mut self) -> Result<(), CompileError> {
        let mut resolved = Vec::with_capacity(self.gotos.len());
        for goto in &self.gotos {
            let label = self
                .labels
                .iter()
                .filter(|label| {
                    label.name == goto.name && goto.scope_path.starts_with(&label.scope_path)
                })
                .max_by_key(|label| label.scope_path.len())
                .ok_or_else(|| CompileError::UndefinedLabel {
                    span: goto.span,
                    name: goto.name.clone(),
                })?;
            if goto.instruction < label.pc
                && (!label
                    .locals
                    .keys()
                    .all(|order| goto.locals.contains_key(order))
                    || !label
                        .global_barriers
                        .keys()
                        .all(|order| goto.global_barriers.contains(order)))
            {
                let local = label
                    .locals
                    .iter()
                    .find(|(order, _)| !goto.locals.contains_key(order))
                    .map(|(_, (_, name))| name.clone())
                    .or_else(|| {
                        label
                            .global_barriers
                            .iter()
                            .find(|(order, _)| !goto.global_barriers.contains(order))
                            .map(|(_, name)| name.clone())
                    })
                    .unwrap_or_else(|| "?".to_owned());
                return Err(CompileError::GotoIntoLocalScope {
                    span: goto.span,
                    name: goto.name.clone(),
                    local,
                });
            }
            let close_upvalues = goto
                .locals
                .iter()
                .filter(|(order, _)| !label.locals.contains_key(order))
                .map(|(_, register)| *register)
                .filter(|register| goto.captured_locals.contains(*register))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let close_locals = goto
                .close_locals
                .iter()
                .rev()
                .filter(|register| !label.close_locals.contains(register))
                .copied()
                .collect::<Vec<_>>();
            resolved.push((
                goto.instruction,
                label.pc,
                goto.span,
                close_locals,
                close_upvalues,
            ));
        }

        let needs_cleanup = resolved
            .iter()
            .any(|(_, _, _, close_locals, close_upvalues)| {
                !close_locals.is_empty() || !close_upvalues.is_empty()
            });
        let skip_cleanup = needs_cleanup.then(|| {
            let pc = self.code.len();
            self.code
                .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
            pc
        });
        for (instruction, label, span, close_locals, close_upvalues) in resolved {
            let target = if close_locals.is_empty() && close_upvalues.is_empty() {
                label
            } else {
                let trampoline = self.code.len();
                let line = self.line_for(span);
                for register in close_locals {
                    self.instruction_lines.insert(self.code.len(), line);
                    self.code.push(Instruction::close_local(register));
                }
                for register in close_upvalues {
                    self.instruction_lines.insert(self.code.len(), line);
                    self.code.push(Instruction::close_upvalue(register));
                }
                let jump = Self::relative_jump(self.code.len(), label, span)?;
                self.instruction_lines.insert(self.code.len(), line);
                self.code.push(
                    Instruction::aj(OpCode::Jump, jump)
                        .map_err(|_| CompileError::JumpOutOfRange { span })?,
                );
                trampoline
            };
            let offset = Self::relative_jump(instruction, target, span)?;
            self.code[instruction] = Instruction::aj(OpCode::Jump, offset)
                .map_err(|_| CompileError::JumpOutOfRange { span })?;
        }
        if let Some(skip) = skip_cleanup {
            let span = self
                .gotos
                .first()
                .map_or(Span { start: 0, end: 0 }, |goto| goto.span);
            let offset = Self::relative_jump(skip, self.code.len(), span)?;
            self.code[skip] = Instruction::aj(OpCode::Jump, offset)
                .map_err(|_| CompileError::JumpOutOfRange { span })?;
        }
        Ok(())
    }
    pub(crate) fn patch_breaks(
        &mut self,
        breaks: Vec<usize>,
        target: usize,
        span: Span,
    ) -> Result<(), CompileError> {
        for instruction in breaks {
            let offset = Self::relative(instruction, target, span)?;
            self.code[instruction] = Instruction::aj(OpCode::Jump, i32::from(offset))
                .expect("i16 offset fits in a jump");
        }
        Ok(())
    }
    pub(crate) fn captured_registers_from(&self, first: Register) -> Vec<Register> {
        self.captured_registers.from(first).collect()
    }
    pub(crate) fn close_captured_registers(&mut self, registers: &[Register]) {
        for register in registers {
            self.code.push(Instruction::close_upvalue(*register));
        }
    }
    pub(crate) fn numeric_for(
        &mut self,
        name: &Name,
        start: &Expr,
        limit: &Expr,
        step: Option<&Expr>,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        self.control_flow_depth += 1;
        let result = self.numeric_for_inner(name, start, limit, step, body, span);
        self.control_flow_depth -= 1;
        result
    }
    pub(crate) fn numeric_for_inner(
        &mut self,
        name: &Name,
        start: &Expr,
        limit: &Expr,
        step: Option<&Expr>,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        let start = self.expression(start, 1)?;
        let limit = self.expression(limit, 1)?;
        let step = match step {
            Some(step) => self.expression(step, 1)?,
            None => self.constant(
                if self.tagged_integers {
                    Constant::Integer(1)
                } else {
                    Constant::Number(1.0)
                },
                span,
            )?,
        };
        let base = self.pack(&[start, limit, step], span)?;
        let visible = self.fresh(name.span)?;
        self.code.push(Instruction::load_nil(visible));
        let prep = self.code.len();
        self.code.push(Instruction::numeric_for_prep(base, 0));
        let body_start = self.code.len();
        self.push_scope();
        self.scopes.insert(name.text.clone(), visible);
        if self.readonly_control_variables {
            self.readonly_scopes
                .last_mut()
                .expect("loop read-only scope exists")
                .insert(name.text.clone());
        }
        self.bind_debug_local(name, visible, body_start);
        let reserve = base
            .checked_add(4)
            .ok_or(CompileError::RegisterExhausted { span })?;
        self.reserved_register_floors.push(reserve);
        self.loop_breaks.push(Vec::new());
        self.loop_close_depths.push(self.close_scopes.len() - 1);
        let body_result = self.statements(&body.statements);
        let breaks = self.loop_breaks.pop().expect("loop break context exists");
        self.loop_close_depths
            .pop()
            .expect("loop close context exists");
        self.reserved_register_floors.pop();
        body_result?;
        self.close_current_scope();
        self.pop_scope();
        let captured_locals = self.captured_registers_from(visible);
        self.close_captured_registers(&captured_locals);
        // Body locals above the control window are dead at the latch; the
        // live register counts keep them from being rooted.
        let loop_instruction = self.code.len();
        let back = Self::relative(loop_instruction, body_start, span)?;
        self.code.push(Instruction::numeric_for_loop(base, back));
        let break_cleanup = self.code.len();
        self.close_captured_registers(&captured_locals);
        let forward = Self::relative(prep, loop_instruction, span)?;
        self.code[prep] = Instruction::numeric_for_prep(base, forward);
        self.patch_breaks(breaks, break_cleanup, span)
    }
    pub(crate) fn generic_for(
        &mut self,
        names: &[Name],
        expressions: &[Expr],
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        self.control_flow_depth += 1;
        let result = self.generic_for_inner(names, expressions, body, span);
        self.control_flow_depth -= 1;
        result
    }
    pub(crate) fn generic_for_inner(
        &mut self,
        names: &[Name],
        expressions: &[Expr],
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        // Lua evaluates the iterator expression list once, then repeatedly
        // calls iterator(state, control). The first result becomes control and
        // nil ends the loop.
        //
        // Layout: `base` iterator, `base + 1` state, `base + 2` control, then
        // the loop variables from `base + 3`. `ForInCall` copies the three
        // control registers over the call window at `base + 3` (at least
        // three registers, whatever the variable count) and calls.
        //
        //     setup; Jump -> call; body: <body>; <close captures>;
        //     call: ForInCall; ForInLoop -> body; <break cleanup>
        let setup = self.values(expressions, 3, span)?;
        // Stock Lua (lparser.c forlist) takes the line right after `in`, where
        // the expression list starts, for the iterator call and the loop test.
        let iterator_span = expressions.first().map_or(span, Expr::span);
        let header_line = self.line_for(iterator_span);
        let pack_start = self.code.len();
        let base = self.pack(&setup, span)?;
        self.record_unmapped_instruction_line_range(pack_start, self.code.len(), header_line);
        let variables =
            u8::try_from(names.len()).map_err(|_| CompileError::RegisterExhausted { span })?;
        let mut visible = Vec::with_capacity(names.len());
        for name in names {
            visible.push(self.fresh(name.span)?);
        }
        debug_assert_eq!(
            visible.first().copied(),
            base.checked_add(3),
            "the loop variables follow the control registers"
        );
        // The call window reaches `base + 5` even with fewer variables.
        let window_end = base
            .checked_add(6)
            .ok_or(CompileError::RegisterExhausted { span })?;
        self.register_count = self.register_count.max(window_end);
        let loop_local_floor = *visible
            .first()
            .expect("syntax requires at least one generic-for name");
        let reserve = self.next_register;
        self.reserved_register_floors.push(reserve);
        self.push_scope();
        // The first iteration starts at the call. Keep the jump on the
        // iterator's line so it adds no line event.
        let enter = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
        self.instruction_lines.insert(enter, header_line);
        let body_start = self.code.len();
        for (name, register) in names.iter().zip(visible.iter().copied()) {
            self.scopes.insert(name.text.clone(), register);
            if self.readonly_control_variables {
                self.readonly_scopes
                    .last_mut()
                    .expect("loop read-only scope exists")
                    .insert(name.text.clone());
            }
            self.bind_debug_local(name, register, body_start);
        }
        self.loop_breaks.push(Vec::new());
        self.loop_close_depths.push(self.close_scopes.len() - 1);
        let body_result = self.statements(&body.statements);
        let breaks = self.loop_breaks.pop().expect("loop break context exists");
        self.loop_close_depths
            .pop()
            .expect("loop close context exists");
        self.close_current_scope();
        self.pop_scope();
        self.reserved_register_floors.pop();
        self.record_unmapped_instruction_line_range(
            body_start,
            self.code.len(),
            self.block_exit_line(body),
        );
        body_result?;
        let captured_locals = self.captured_registers_from(loop_local_floor);
        self.close_captured_registers(&captured_locals);
        let call = self.code.len();
        self.record_call_site(None, iterator_span);
        self.instruction_lines.insert(call, header_line);
        // The kind names the call `for iterator`. The profile decides at
        // runtime whether a yield may cross it.
        if let Some(site) = self.call_sites.last_mut() {
            *site = CallSite::for_iterator(site.pc());
        }
        self.code.push(Instruction::for_in_call(base, variables));
        let latch = self.code.len();
        let back = Self::relative(latch, body_start, span)?;
        self.code.push(Instruction::for_in_loop(base, back));
        // The test runs even when the body never does. Keep its line on the
        // loop header so nil does not report a body step.
        self.instruction_lines.insert(latch, header_line);
        if self.generic_for_exit_line {
            // Stock Lua closes its hidden iterator slot after exhaustion.
            let exit = self.code.len();
            self.code
                .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
            self.instruction_lines.insert(
                exit,
                self.line_for(Span {
                    start: span.end.saturating_sub(1),
                    end: span.end,
                }),
            );
        }
        let break_cleanup = self.code.len();
        self.close_captured_registers(&captured_locals);
        let forward = Self::relative(enter, call, span)?;
        self.code[enter] =
            Instruction::aj(OpCode::Jump, i32::from(forward)).expect("i16 offset fits in a jump");
        self.patch_breaks(breaks, break_cleanup, span)
    }
}
