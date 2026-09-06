//! Function definitions, closures, parameters, upvalue resolution, and environment capture.

use crate::bytecode::{Chunk, DebugMetadata, Instruction, UpvalueDescriptor, UpvalueSource};
use crate::compiler::{CapturedTarget, CompileError, Compiler, FreeNameAnalyzer, Ident, Register};
use crate::string::LuaString;
use crate::syntax::{self as syntax, Expr, Name, Span};
use std::collections::BTreeMap;

impl Compiler {
    pub(crate) fn function_statement(
        &mut self,
        name: &syntax::FunctionName,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        let prototype =
            self.function_body(name.method, parameters, vararg, vararg_name, body, span)?;
        let index = u16::try_from(self.prototypes.len())
            .map_err(|_| CompileError::ConstantPoolExhausted { span })?;
        self.prototypes.push(prototype);
        if name.segments.len() == 1 {
            self.store_closure(name.terminal(), index, span)?;
        } else {
            let mut table = self.expression(&Expr::Name(name.segments[0].clone()), 1)?;
            for segment in &name.segments[1..name.segments.len() - 1] {
                let text = &*segment.text;
                if let Some(field) = self.field_name_index_str(text, segment.span)? {
                    let next = self.fresh(segment.span)?;
                    self.code.push(Instruction::get_field(next, table, field));
                    table = next;
                    continue;
                }
                let key = self.string_constant_register(text, segment.span)?;
                let next = self.fresh(segment.span)?;
                self.code.push(Instruction::get_table(next, table, key));
                table = next;
            }
            let terminal = name.terminal();
            let text = &*terminal.text;
            if let Some(field) = self.field_name_index_str(text, terminal.span)? {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, index));
                self.code
                    .push(Instruction::set_field(table, field, closure));
                return Ok(());
            }
            let key = self.string_constant_register(text, terminal.span)?;
            let closure = self.fresh(span)?;
            self.code.push(Instruction::closure(closure, index));
            self.code.push(Instruction::set_table(table, key, closure));
        }
        Ok(())
    }
    pub(crate) fn local_function(
        &mut self,
        name: &Name,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        span: Span,
    ) -> Result<(), CompileError> {
        // Lua binds the name before it creates the closure, which makes the
        // name available to recursive calls in the body.
        if self.scopes.is_empty() {
            self.push_scope();
        }
        let destination = self.fresh(name.span)?;
        self.scopes.insert(name.text.clone(), destination);
        self.bind_debug_local(name, destination, self.code.len());
        let prototype = self.function_body(false, parameters, vararg, vararg_name, body, span)?;
        let index = u16::try_from(self.prototypes.len())
            .map_err(|_| CompileError::ConstantPoolExhausted { span })?;
        self.prototypes.push(prototype);
        self.code.push(Instruction::closure(destination, index));
        Ok(())
    }
    pub(crate) fn bind_parameter(
        &mut self,
        name: &Name,
        count: &mut u8,
    ) -> Result<(), CompileError> {
        let register = self.fresh(name.span)?;
        self.scopes.insert(name.text.clone(), register);
        self.bind_debug_local(name, register, 0);
        *count = count
            .checked_add(1)
            .ok_or(CompileError::RegisterExhausted { span: name.span })?;
        Ok(())
    }
    #[allow(clippy::too_many_lines)] // Child compiler setup mirrors the compatibility configuration explicitly.
    pub(crate) fn function_body(
        &mut self,
        method: bool,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        definition_span: Span,
    ) -> Result<Chunk, CompileError> {
        self.ensure_lexical_environment(definition_span)?;
        let (free_names, assignments) = FreeNameAnalyzer::function(
            method,
            parameters,
            vararg,
            vararg_name,
            body,
            &self.analysis,
        );
        self.forward_child_captures(&free_names, body.span)?;
        let parent_captures = self.child_capture_sources(&free_names);
        let closure_assigned_names = assignments.1.clone();
        let inherited_constants = if self.subtract_immediate {
            self.visible_constants()
        } else {
            BTreeMap::new()
        };
        let child = |compact_registers| Self {
            parent_captures: parent_captures.clone(),
            global_scopes: self.global_scopes.clone(),
            source_name: self.source_name.clone(),
            line_starts: self.line_starts.clone(),
            readonly_control_variables: self.readonly_control_variables,
            goto_labels: self.goto_labels,
            strict_label_shadowing: self.strict_label_shadowing,
            upvalue_limit: self.upvalue_limit,
            legacy_vararg_compat: self.legacy_vararg_compat,
            lexical_environment: self.lexical_environment,
            tagged_integers: self.tagged_integers,
            repeat_header_line: self.repeat_header_line,
            generic_for_exit_line: self.generic_for_exit_line,
            field_key_names: self.field_key_names,
            constant_names: self.constant_names,
            subtract_immediate: self.subtract_immediate,
            inherited_constants: inherited_constants.clone(),
            has_global_declarations: self.has_global_declarations,
            compact_registers,
            closure_assigned_names: closure_assigned_names.clone(),
            string_constants: self.string_constants.clone(),
            analysis: self.analysis.clone(),
            ..Self::default()
        };
        // Normal allocation keeps the bytecode shapes the tracing compiler is
        // tuned for. Only a function that exhausts its registers is compiled
        // again with compact allocation.
        let chunk = match Self::compile_function(
            child(false),
            method,
            parameters,
            vararg,
            vararg_name,
            body,
            definition_span,
        ) {
            Err(CompileError::RegisterExhausted { .. }) => Self::compile_function(
                child(true),
                method,
                parameters,
                vararg,
                vararg_name,
                body,
                definition_span,
            ),
            result => result,
        }?;
        self.pin_direct_captures(&chunk);
        Ok(chunk)
    }

    fn compile_function(
        mut compiler: Self,
        method: bool,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        definition_span: Span,
    ) -> Result<Chunk, CompileError> {
        compiler.push_scope();
        let mut parameter_count = 0_u8;
        if method {
            let self_name = Name {
                text: "self".into(),
                span: body.span,
            };
            compiler.bind_parameter(&self_name, &mut parameter_count)?;
        }
        for parameter in parameters {
            compiler.bind_parameter(parameter, &mut parameter_count)?;
        }
        compiler.parameter_count = parameter_count;
        compiler.is_vararg = vararg;
        if let Some(vararg_name) = vararg_name {
            let table = compiler.fresh(vararg_name.span)?;
            compiler.scopes.insert(vararg_name.text.clone(), table);
            compiler.bind_debug_local(vararg_name, table, 0);
            compiler
                .readonly_scopes
                .last_mut()
                .expect("function scope exists")
                .insert(vararg_name.text.clone());
            compiler.arg_register = Some(table);
        } else if vararg && compiler.legacy_vararg_compat {
            // Lua 5.1 gives every vararg function an implicit `arg` local.
            // It starts as the old compatibility table and becomes nil if the
            // function uses `...`; `vararg` records that later while emitting
            // the body.
            let arg_name = Name {
                text: "arg".into(),
                span: body.span,
            };
            let arg = compiler.fresh(arg_name.span)?;
            compiler.scopes.insert(arg_name.text.clone(), arg);
            compiler.bind_debug_local(&arg_name, arg, 0);
            compiler.arg_register = Some(arg);
        }
        compiler.statements(&body.statements)?;
        compiler.resolve_gotos()?;
        let definition_line = compiler.line_for(definition_span);
        let last_definition_line = compiler.line_for(Span {
            start: body.span.end.saturating_sub(1),
            end: body.span.end,
        });
        // A branch may jump past a final `return`, so every prototype needs a
        // concrete fallthrough target even when its final emitted statement is
        // a return. Lua records this compiler-generated terminal instruction
        // at the closing `end`, which makes that line active for
        // debug.getinfo(function, "L").
        compiler.close_all_scopes();
        let terminal_pc = compiler.code.len();
        compiler.code.push(Instruction::halt());
        compiler
            .instruction_lines
            .insert(terminal_pc, last_definition_line);
        let line_info = compiler.line_info();
        let local_variables = compiler.take_debug_locals();
        let chunk = if compiler.is_vararg {
            Chunk::with_varargs(
                compiler.register_count.max(1),
                compiler.parameter_count,
                compiler.constants,
                compiler.code,
                compiler.prototypes,
                compiler.upvalues,
                compiler.arg_register,
            )
        } else {
            Chunk::with_parts(
                compiler.register_count.max(1),
                compiler.parameter_count,
                compiler.constants,
                compiler.code,
                compiler.prototypes,
                compiler.upvalues,
            )
        }
        .with_debug_metadata(
            DebugMetadata::new(
                compiler
                    .source_name
                    .unwrap_or_else(|| LuaString::from("=?")),
                compiler.call_sites,
            )
            .with_function_lines(definition_line, last_definition_line)
            .with_diagnostic_sites(compiler.diagnostic_sites)
            .with_local_variables(local_variables)
            .with_line_info(line_info),
        );
        // The live register analysis runs on the first call (see
        // `Chunk::entry_register_count`): most loaded functions never run.
        Ok(chunk)
    }
    pub(crate) fn store_closure(
        &mut self,
        name: &Name,
        prototype: u16,
        span: Span,
    ) -> Result<(), CompileError> {
        if self.is_readonly_name(&name.text) || self.is_readonly_global_target(&name.text) {
            return Err(CompileError::AssignToReadOnly {
                span: name.span,
                name: name.text.to_string(),
            });
        }
        match self.capture_name(name, false)? {
            CapturedTarget::Local(destination) => {
                self.code.push(Instruction::closure(destination, prototype));
            }
            CapturedTarget::Upvalue(index) => {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, prototype));
                self.code.push(Instruction::set_upvalue(closure, index));
            }
            CapturedTarget::Global(name) => {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, prototype));
                self.emit_set_global(closure, name);
            }
            CapturedTarget::Index { table, key, name } => {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, prototype));
                self.emit_set_table(table, key, closure, name);
            }
            CapturedTarget::Field { table, field, name } => {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, prototype));
                self.emit_set_field(table, field, closure, name);
            }
            CapturedTarget::TabUp { upvalue, field } => {
                let closure = self.fresh(span)?;
                self.code.push(Instruction::closure(closure, prototype));
                self.code
                    .push(Instruction::set_tab_up(upvalue, field, closure));
            }
        }
        Ok(())
    }
    /// Where a nested function whose free names are `free_names` finds each
    /// of them, and `_ENV`, here: in a local's register, or in an upvalue of
    /// this function. A name it can take from neither is a global. The child
    /// looks up no other name in its parent, since every name it uses or
    /// forwards to its own children is free in it.
    pub(crate) fn child_capture_sources(
        &self,
        free_names: &[Ident],
    ) -> rustc_hash::FxHashMap<Ident, UpvalueSource> {
        let mut sources = rustc_hash::FxHashMap::default();
        let names = free_names
            .iter()
            .map(|name| &**name)
            .chain(core::iter::once("_ENV"));
        for name in names {
            let source = if let Some(register) = self.local_register(name) {
                UpvalueSource::ParentRegister(register)
            } else if let Some(index) = self.upvalue_names.get(name) {
                UpvalueSource::ParentUpvalue(*index)
            } else {
                continue;
            };
            sources.insert(name.into(), source);
        }
        sources
    }
    pub(crate) fn forward_child_captures(
        &mut self,
        free_names: &[Ident],
        span: Span,
    ) -> Result<(), CompileError> {
        for text in free_names {
            if self.local_register(text).is_none() && !self.upvalue_names.contains_key(&**text) {
                self.resolve_upvalue(&Name {
                    text: text.clone(),
                    span,
                })?;
            }
        }
        Ok(())
    }
    pub(crate) fn resolve_upvalue(&mut self, name: &Name) -> Result<Option<u16>, CompileError> {
        if let Some(index) = self.upvalue_names.get(&name.text) {
            return Ok(Some(*index));
        }
        let Some(source) = self.parent_captures.get(&name.text).copied() else {
            return Ok(None);
        };
        if self.upvalues.len() >= self.upvalue_limit {
            return Err(CompileError::UpvalueExhausted { span: name.span });
        }
        let index = u16::try_from(self.upvalues.len())
            .map_err(|_| CompileError::UpvalueExhausted { span: name.span })?;
        self.upvalue_names.insert(name.text.clone(), index);
        self.upvalues.push(UpvalueDescriptor {
            name: self.name_string(&name.text),
            source,
        });
        Ok(Some(index))
    }
    pub(crate) fn bind_implicit_environment(&mut self) -> Result<(), CompileError> {
        if !self.lexical_environment {
            return Ok(());
        }
        if self.upvalues.len() >= self.upvalue_limit {
            return Err(CompileError::UpvalueExhausted {
                span: Span { start: 0, end: 0 },
            });
        }
        let index =
            u16::try_from(self.upvalues.len()).map_err(|_| CompileError::UpvalueExhausted {
                span: Span { start: 0, end: 0 },
            })?;
        self.upvalue_names.insert("_ENV".into(), index);
        self.upvalues.push(UpvalueDescriptor {
            name: LuaString::from("_ENV"),
            source: UpvalueSource::Detached,
        });
        Ok(())
    }
    pub(crate) fn ensure_lexical_environment(&mut self, span: Span) -> Result<(), CompileError> {
        if !self.lexical_environment || self.local_register("_ENV").is_some() {
            return Ok(());
        }
        let environment = Name {
            text: "_ENV".into(),
            span,
        };
        let _ = self.resolve_upvalue(&environment)?;
        Ok(())
    }
    /// Emits a read of global `name` through the lexical `_ENV` as a field
    /// read with a constant key, so the read has an inline-cache site. Returns
    /// false, emitting nothing, when the chunk has no lexical `_ENV` or the
    /// key does not fit a field operand.
    pub(crate) fn emit_environment_get(
        &mut self,
        destination: Register,
        name: &Name,
    ) -> Result<bool, CompileError> {
        if !self.lexical_environment {
            return Ok(false);
        }
        let text = &*name.text;
        let Some(field) = self.field_name_index_str(text, name.span)? else {
            return self.emit_environment_get_by_register(destination, name);
        };
        if let Some(environment) = self.local_register("_ENV") {
            self.code
                .push(Instruction::get_field(destination, environment, field));
            return Ok(true);
        }
        let environment = Name {
            text: "_ENV".into(),
            span: name.span,
        };
        if let Some(upvalue) = self.resolve_upvalue(&environment)? {
            if let Ok(upvalue) = u8::try_from(upvalue) {
                self.code
                    .push(Instruction::get_tab_up(destination, upvalue, field));
            } else {
                self.code
                    .push(Instruction::get_upvalue(destination, upvalue));
                self.code
                    .push(Instruction::get_field(destination, destination, field));
            }
            return Ok(true);
        }
        Ok(false)
    }
    fn emit_environment_get_by_register(
        &mut self,
        destination: Register,
        name: &Name,
    ) -> Result<bool, CompileError> {
        let Some(environment) = self.lexical_environment_value(name.span)? else {
            return Ok(false);
        };
        let key = self.string_constant_register(&name.text, name.span)?;
        self.code
            .push(Instruction::get_table(destination, environment, key));
        Ok(true)
    }
    pub(crate) fn lexical_environment_value(
        &mut self,
        span: Span,
    ) -> Result<Option<Register>, CompileError> {
        if !self.lexical_environment {
            return Ok(None);
        }
        if let Some(environment) = self.local_register("_ENV") {
            return Ok(Some(environment));
        }
        let name = Name {
            text: "_ENV".into(),
            span,
        };
        if let Some(upvalue) = self.resolve_upvalue(&name)? {
            let environment = self.fresh(span)?;
            self.code
                .push(Instruction::get_upvalue(environment, upvalue));
            return Ok(Some(environment));
        }
        Ok(None)
    }
    pub(crate) fn pin_direct_captures(&mut self, child: &Chunk) {
        for upvalue in &child.upvalues {
            if let UpvalueSource::ParentRegister(register) = upvalue.source {
                self.captured_registers.insert(register);
            }
        }
    }
}
