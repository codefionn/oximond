//! Lexical scopes, locals, close handling, debug-local tracking, and scope-binding queries.

use crate::bytecode::{Instruction, LocalVariable};
use crate::compiler::{CompileError, Compiler, GlobalScope, Ident, OpenLocal, Register};
use crate::syntax::{self as syntax, Name};
use std::collections::{BTreeMap, BTreeSet};

impl Compiler {
    pub(crate) fn push_scope(&mut self) {
        self.scopes.push();
        self.readonly_scopes.push(BTreeSet::new());
        // `constant_scopes` grows lazily in `mark_constant_local`, so a
        // function without an integer `<const>` local allocates nothing.
        self.close_scopes.push(Vec::new());
        self.global_scopes.push(GlobalScope::default());
        self.global_barrier_scopes.push(Vec::new());
        self.debug_local_scopes.push(Vec::new());
        let scope = self.next_scope_id;
        self.next_scope_id = self.next_scope_id.saturating_add(1);
        self.scope_path.push(scope);
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scope_path.pop().expect("compiler scope path exists");
        if self.constant_scopes.len() == self.scopes.len() {
            self.constant_scopes.pop();
        }
        self.scopes.pop();
        self.readonly_scopes
            .pop()
            .expect("read-only scope matches compiler scope");
        self.close_scopes
            .pop()
            .expect("to-be-closed scope matches compiler scope");
        self.global_scopes
            .pop()
            .expect("global scope matches compiler scope");
        self.global_barrier_scopes
            .pop()
            .expect("global barrier scope matches compiler scope");
        let locals = self
            .debug_local_scopes
            .pop()
            .expect("debug local scope matches compiler scope");
        self.close_debug_locals(locals, self.code.len());
    }

    /// Emits close operations in reverse declaration order for the current
    /// lexical scope. The VM makes a repeated close harmless, which keeps
    /// jump trampolines and ordinary fallthrough exits composable.
    pub(crate) fn close_current_scope(&mut self) {
        let registers = self
            .close_scopes
            .last()
            .expect("to-be-closed scope exists")
            .clone();
        for register in registers.into_iter().rev() {
            self.code.push(Instruction::close_local(register));
        }
    }

    pub(crate) fn close_scopes_from(&mut self, first_scope: usize) {
        let registers = self.close_scopes[first_scope..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().copied())
            .collect::<Vec<_>>();
        for register in registers {
            self.code.push(Instruction::close_local(register));
        }
    }

    pub(crate) fn close_all_scopes(&mut self) {
        self.close_scopes_from(0);
    }

    pub(crate) fn mark_close_locals(
        &mut self,
        locals: &[Register],
        attributes: &[syntax::LocalAttribute],
    ) {
        let registers = locals
            .iter()
            .copied()
            .zip(attributes)
            .filter_map(|(register, attribute)| {
                matches!(attribute, syntax::LocalAttribute::Close).then_some(register)
            })
            .collect::<Vec<_>>();
        for register in &registers {
            self.code.push(Instruction::mark_close(*register));
        }
        self.close_scopes
            .last_mut()
            .expect("local declaration has a scope")
            .extend(registers);
    }

    pub(crate) fn bind_debug_local(&mut self, name: &Name, register: Register, start_pc: usize) {
        // Every binding of a name goes through here, and a new one hides a
        // constant of the same name declared earlier in the scope.
        if let Some(constants) = self
            .constant_scopes
            .get_mut(self.scopes.len().wrapping_sub(1))
        {
            constants.remove(&name.text);
        }
        let prior = self.debug_local_scopes.last_mut().and_then(|scope| {
            scope
                .iter()
                .rposition(|local| local.name.as_bytes() == name.text.as_bytes())
                .map(|index| scope.remove(index))
        });
        if let Some(prior) = prior {
            self.close_debug_locals(vec![prior], start_pc);
        }
        let order = self.next_debug_local;
        self.next_debug_local += 1;
        let text = self.name_string(&name.text);
        self.debug_local_scopes
            .last_mut()
            .expect("debug local scope exists")
            .push(OpenLocal {
                order,
                name: text,
                register,
                start_pc,
            });
    }

    pub(crate) fn close_debug_locals(&mut self, locals: Vec<OpenLocal>, end_pc: usize) {
        self.debug_locals
            .extend(locals.into_iter().filter_map(|local| {
                (local.start_pc < end_pc).then(|| {
                    (
                        local.order,
                        LocalVariable::new(local.name, local.register, local.start_pc, end_pc),
                    )
                })
            }));
    }

    pub(crate) fn finish_debug_locals(&mut self) {
        let end_pc = self.code.len();
        while !self.scopes.is_empty() {
            self.pop_scope();
        }
        debug_assert!(self.debug_local_scopes.is_empty());
        debug_assert!(self.scopes.is_empty());
        debug_assert!(
            self.debug_locals
                .iter()
                .all(|(_, local)| local.end_pc() <= end_pc)
        );
    }

    pub(crate) fn take_debug_locals(&mut self) -> Vec<LocalVariable> {
        self.finish_debug_locals();
        self.debug_locals.sort_by_key(|(order, _)| *order);
        std::mem::take(&mut self.debug_locals)
            .into_iter()
            .map(|(_, local)| local)
            .collect()
    }
    pub(crate) fn active_local_bindings(&self) -> BTreeMap<usize, Register> {
        self.active_local_info()
            .into_iter()
            .map(|(order, (register, _))| (order, register))
            .collect()
    }
    pub(crate) fn active_global_barriers(&self) -> BTreeMap<usize, String> {
        self.global_barrier_scopes
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
    pub(crate) fn active_close_registers(&self) -> BTreeSet<Register> {
        self.close_scopes.iter().flatten().copied().collect()
    }
    pub(crate) fn active_close_registers_in_declaration_order(&self) -> Vec<Register> {
        self.close_scopes.iter().flatten().copied().collect()
    }
    pub(crate) fn active_local_info(&self) -> BTreeMap<usize, (Register, String)> {
        self.debug_local_scopes
            .iter()
            .flat_map(|scope| scope.iter())
            .map(|local| {
                (
                    local.order,
                    (local.register, local.name.to_lossy().into_owned()),
                )
            })
            .collect()
    }
    pub(crate) fn local_register(&self, name: &str) -> Option<Register> {
        self.scopes.get(name)
    }
    pub(crate) fn name_is_forced_global(&self, name: &str) -> bool {
        if !self.has_global_declarations || name == "_ENV" {
            return false;
        }
        // A Lua 5.5 `global *` declaration affects unresolved names. It must
        // not hide an enclosing local that this nested function receives as
        // an upvalue. A named `global foo`, in contrast, deliberately wins
        // over an enclosing local named `foo`.
        let captured =
            self.upvalue_names.contains_key(name) || self.parent_captures.contains_key(name);
        let inherited = self.global_scopes.len().saturating_sub(self.scopes.len());
        for (index, locals) in self.scopes.iter().enumerate().rev() {
            if locals.iter().any(|(bound, _)| &**bound == name) {
                return false;
            }
            let globals = &self.global_scopes[inherited + index];
            if globals.names.contains_key(name) || (globals.allow_all && !captured) {
                return true;
            }
        }
        for globals in self.global_scopes[..inherited].iter().rev() {
            if globals.names.contains_key(name) || (globals.allow_all && !captured) {
                return true;
            }
        }
        false
    }
    pub(crate) fn is_readonly_name(&self, name: &str) -> bool {
        self.readonly_scopes
            .iter()
            .rev()
            .zip(self.scopes.iter().rev())
            .find_map(|(readonly, scope)| {
                scope
                    .iter()
                    .any(|(bound, _)| &**bound == name)
                    .then(|| readonly.contains(name))
            })
            .unwrap_or(false)
    }
    /// The integer constant that the local `name` stands for, if it is a
    /// `<const>` bound to one in this or an enclosing function.
    pub(crate) fn constant_local(&self, name: &str) -> Option<i64> {
        if !self.subtract_immediate {
            return None;
        }
        for (depth, scope) in self.scopes.iter().enumerate().rev() {
            if scope.iter().any(|(bound, _)| &**bound == name) {
                return self.constant_scopes.get(depth)?.get(name).copied();
            }
        }
        self.inherited_constants.get(name).copied()
    }
    /// Every integer `<const>` local visible here, for a nested function.
    pub(crate) fn visible_constants(&self) -> BTreeMap<Ident, i64> {
        let mut visible = self.inherited_constants.clone();
        for (depth, scope) in self.scopes.iter().enumerate() {
            for (name, _) in scope {
                visible.remove(name);
            }
            for (name, value) in self.constant_scopes.get(depth).into_iter().flatten() {
                visible.insert(name.clone(), *value);
            }
        }
        visible
    }
    /// Records that the last of `names`, just declared, is a `<const>` bound
    /// to `value`. Lua folds only the last variable of a declaration.
    pub(crate) fn mark_constant_local(&mut self, name: &Name, value: Option<i64>) {
        let Some(value) = value else { return };
        if !self.subtract_immediate {
            return;
        }
        // Entries are indexed by scope depth; pad up to the current one.
        while self.constant_scopes.len() < self.scopes.len() {
            self.constant_scopes.push(BTreeMap::new());
        }
        self.constant_scopes
            .last_mut()
            .expect("local declaration has a scope")
            .insert(name.text.clone(), value);
    }
    pub(crate) fn global_readonly_name(&self, name: &str) -> bool {
        for scope in self.global_scopes.iter().rev() {
            if let Some(readonly) = scope.names.get(name) {
                return *readonly;
            }
            if scope.allow_all {
                return scope.all_readonly;
            }
            if scope.deny_all {
                return false;
            }
        }
        false
    }
    pub(crate) fn is_readonly_global_target(&self, name: &str) -> bool {
        let shadowed = !self.name_is_forced_global(name)
            && (self.local_register(name).is_some()
                || self.upvalue_names.contains_key(name)
                || self.parent_captures.contains_key(name));
        !shadowed && self.global_readonly_name(name)
    }
    pub(crate) fn ensure_global_declared(&self, name: &Name) -> Result<(), CompileError> {
        if &*name.text == "_ENV" {
            return Ok(());
        }
        let mut declarations_active = false;
        for scope in self.global_scopes.iter().rev() {
            if scope.names.contains_key(&*name.text) || scope.allow_all {
                return Ok(());
            }
            declarations_active |= scope.restricted;
            if scope.deny_all {
                return Err(CompileError::UndeclaredGlobal {
                    span: name.span,
                    name: name.text.to_string(),
                });
            }
        }
        if declarations_active {
            Err(CompileError::UndeclaredGlobal {
                span: name.span,
                name: name.text.to_string(),
            })
        } else {
            Ok(())
        }
    }
    pub(crate) fn mark_readonly_locals(
        &mut self,
        names: &[Name],
        attributes: &[syntax::LocalAttribute],
    ) {
        let readonly = self
            .readonly_scopes
            .last_mut()
            .expect("local declaration has a scope");
        for (name, attribute) in names.iter().zip(attributes) {
            if matches!(
                attribute,
                syntax::LocalAttribute::Const | syntax::LocalAttribute::Close
            ) {
                readonly.insert(name.text.clone());
            }
        }
    }
    pub(crate) fn disable_compatibility_arg(&mut self) {
        if self.legacy_vararg_compat {
            self.arg_register = None;
        }
    }
}
