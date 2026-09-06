//! Register allocation, constants, jumps, global access, call emission, and call-site and line-info recording.

use crate::bytecode::{
    CallKind, CallSite, Constant, DiagnosticSite, Instruction, LocalVariable, OpCode,
};
use crate::compiler::locals::RegisterBits;
use crate::compiler::statements::statement_span;
use crate::compiler::{CallTail, CompileError, Compiler, ConstantKey, Ident, Register};
use crate::string::LuaString;
use crate::syntax::{self as syntax, Expr, Span};

impl Compiler {
    pub(crate) fn fresh(&mut self, span: Span) -> Result<Register, CompileError> {
        let register = self.next_register;
        self.next_register = self
            .next_register
            .checked_add(1)
            .ok_or(CompileError::RegisterExhausted { span })?;
        self.register_count = self.register_count.max(self.next_register);
        Ok(register)
    }
    /// Frees every register from `mark` up once the values in them have been
    /// consumed, so long expressions reuse scratch registers the way Lua's
    /// own compiler does. `released_high_water` remembers how high the freed
    /// registers reached. Only compact mode releases.
    pub(crate) fn release_registers(&mut self, mark: Register) {
        if mark < self.next_register {
            self.released_high_water = self.released_high_water.max(self.next_register);
            self.next_register = mark;
        }
    }

    /// Compact mode: evaluates `expression` to one value in the next free
    /// register and releases every scratch register the evaluation needed.
    /// Consecutive calls place values in consecutive registers, as call
    /// arguments and concatenation operands require.
    pub(crate) fn expression_to_next(
        &mut self,
        expression: &Expr,
    ) -> Result<Register, CompileError> {
        let slot = self.next_register;
        let value = self.expression(expression, 1)?;
        self.release_registers(slot);
        let slot = self.fresh(expression.span())?;
        if value != slot {
            self.code.push(Instruction::move_(slot, value));
        }
        Ok(slot)
    }

    /// Compact mode: emits a call whose function sits in `base` and whose
    /// `fixed` arguments follow it, then frees the argument registers. The
    /// results occupy `base` onward.
    pub(crate) fn emit_placed_call(
        &mut self,
        base: Register,
        fixed: usize,
        tail: CallTail,
        results: usize,
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
    ) -> Result<Register, CompileError> {
        let count = u8::try_from(fixed).map_err(|_| CompileError::RegisterExhausted { span })?;
        let count_results =
            u8::try_from(results).map_err(|_| CompileError::RegisterExhausted { span })?;
        self.release_registers(base.saturating_add(1));
        // Reserve result registers before validation. The VM writes a nil for absent values.
        for _ in 1..results {
            self.fresh(span)?;
        }
        self.record_call_site(call_name, span);
        self.code.push(match tail {
            CallTail::Fixed => Instruction::call(base, count, count_results),
            CallTail::Vararg => Instruction::call_vararg(base, count, count_results),
            CallTail::Open => Instruction::call_open(base, count, count_results),
        });
        Ok(base)
    }

    /// Compact mode: emits a proper tail call of the function in `base` with
    /// `fixed` arguments after it.
    pub(crate) fn emit_placed_tail_call(
        &mut self,
        base: Register,
        fixed: usize,
        tail: CallTail,
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
    ) -> Result<(), CompileError> {
        let count = u8::try_from(fixed).map_err(|_| CompileError::RegisterExhausted { span })?;
        self.close_all_scopes();
        self.record_call_site(call_name, span);
        self.code.push(match tail {
            CallTail::Fixed => Instruction::tail_call(base, count),
            CallTail::Vararg => Instruction::tail_call_vararg(base, count),
            CallTail::Open => Instruction::tail_call_open(base, count),
        });
        self.code.push(Instruction::return_open(base, 0));
        Ok(())
    }

    /// Ends a statement: frees its temporaries and compacts locals into the
    /// holes they left. Dead temporaries are not cleared. The chunk's live
    /// register counts (`Chunk::live_register_counts`) tell the collector
    /// that registers no later instruction reads are not roots, and every
    /// register is written before it is read, so a stale value is never seen.
    #[allow(
        clippy::too_many_lines,
        reason = "register compaction keeps bytecode, lexical bindings, debug ranges, and close markers synchronized"
    )]
    pub(crate) fn reclaim_temporaries(&mut self) {
        // Registers released inside the statement may hold dead values.
        self.next_register = self.next_register.max(self.released_high_water);
        self.released_high_water = 0;
        self.reclaim_registers();
    }

    /// Frees the registers above the live locals and compacts those locals.
    fn reclaim_registers(&mut self) {
        // The overwhelmingly common case has locals packed from register zero,
        // with no close locals or captured register past the locals. It needs
        // no remap: compaction below would map every local onto itself, as
        // packed locals leave no hole to fill, so every register after them
        // and after the reservations (the tables of the constructors being
        // built, after each of whose fields this runs) is a dead temporary.
        if self.close_scopes.iter().all(Vec::is_empty)
            && let Ok(local_count) = u8::try_from(self.scopes.binding_count())
            && self.scopes.packed()
            && self
                .captured_registers
                .max()
                .is_none_or(|register| register < local_count)
        {
            let reservations = self
                .reserved_register_floors
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            self.next_register = local_count.max(reservations);
            return;
        }
        if self.control_flow_depth != 0 {
            // Branch and loop bodies never compact (see below), so the next
            // register follows the highest local, close marker, reservation
            // or capture, and no register set is needed.
            let close_end = self
                .close_scopes
                .iter()
                .flatten()
                .map(|register| u16::from(*register) + 1)
                .max()
                .unwrap_or(0);
            let locals = u8::try_from(self.scopes.end().max(close_end)).unwrap_or(u8::MAX);
            let reservations = self
                .reserved_register_floors
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            let captures = self
                .captured_registers
                .max()
                .map_or(0, |register| register.saturating_add(1));
            self.next_register = locals.max(reservations).max(captures);
            return;
        }
        // Fixed 256-bit sets and an array remap: this runs after every
        // statement at the top of a function body.
        let mut live_locals = self.scopes.registers().collect::<RegisterBits>();
        // A close marker stores a physical frame register in the VM. Keep
        // that register stable until its matching CloseLocal has run; moving
        // the lexical binding alone would leave the marker pointing at a
        // cleared temporary.
        let close_locals = self
            .close_scopes
            .iter()
            .flatten()
            .copied()
            .collect::<RegisterBits>();
        live_locals = live_locals.union(close_locals);
        let reservations = self
            .reserved_register_floors
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let captures = self
            .captured_registers
            .max()
            .map_or(0, |register| register.saturating_add(1));

        // Register operands are one byte wide, but long source chunks often
        // accumulate locals interleaved with dead expression temporaries.
        // Shadowed bindings add more holes. At straight-line statement
        // boundaries we compact the remaining uncaptured locals into those
        // holes. Captured registers cannot move because existing closures
        // refer to their frame cells by number. Registers reserved by a loop
        // likewise keep their identity until its back edge.
        //
        // Compaction is suppressed while emitting branch and loop bodies.
        // A move emitted on only one control-flow path cannot redefine the
        // register assignment used after paths merge.
        if self.control_flow_depth == 0 {
            let pinned = self.captured_registers.union(close_locals);
            let mut next_destination = reservations;
            let mut remap = [0_u8; 256];
            let mut moved = false;
            let mut destinations = RegisterBits::EMPTY;
            for source in live_locals.iter() {
                let destination = if source < reservations || pinned.contains(source) {
                    source
                } else {
                    while pinned.contains(next_destination) {
                        next_destination = next_destination.saturating_add(1);
                    }
                    let destination = next_destination;
                    next_destination = next_destination.saturating_add(1);
                    destination
                };
                remap[usize::from(source)] = destination;
                destinations.insert(destination);
                moved |= source != destination;
            }
            if moved {
                for source in live_locals.iter() {
                    let destination = remap[usize::from(source)];
                    if source != destination {
                        self.code.push(Instruction::move_(destination, source));
                    }
                }
                self.scopes.remap(&remap);
                let pc = self.code.len();
                let mut previous_locations = Vec::new();
                for scope in &mut self.debug_local_scopes {
                    for local in scope {
                        let destination = remap[usize::from(local.register)];
                        if local.register != destination {
                            if local.start_pc < pc {
                                previous_locations.push((
                                    local.order,
                                    LocalVariable::new(
                                        local.name.clone(),
                                        local.register,
                                        local.start_pc,
                                        pc,
                                    ),
                                ));
                            }
                            local.register = destination;
                            local.start_pc = pc;
                        }
                    }
                }
                self.debug_locals.extend(previous_locations);
            }
            live_locals = destinations;
        }

        let locals = live_locals
            .max()
            .map_or(0, |register| register.saturating_add(1));
        self.next_register = locals.max(reservations).max(captures);
    }
    pub(crate) fn copy(&mut self, source: Register, span: Span) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        self.code.push(Instruction::move_(destination, source));
        Ok(destination)
    }
    pub(crate) fn nil(&mut self, span: Span) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        self.code.push(Instruction::load_nil(destination));
        Ok(destination)
    }
    pub(crate) fn constant(
        &mut self,
        constant: Constant,
        span: Span,
    ) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        let index = self.constant_index(constant, span)?;
        self.emit_load_constant(destination, index);
        Ok(destination)
    }
    pub(crate) fn string_constant(
        &mut self,
        string: &str,
        span: Span,
    ) -> Result<u32, CompileError> {
        self.string_constant_bytes(string.as_bytes(), span)
    }
    /// The index of the string constant `bytes`, adding it on first use. A
    /// hit builds no `LuaString`.
    fn string_constant_bytes(&mut self, bytes: &[u8], span: Span) -> Result<u32, CompileError> {
        let hash = crate::string::cached_hash_bytes(bytes);
        let constants = &self.constants;
        if let Some(index) = self.string_indices.find(hash, |index| {
            matches!(&constants[*index as usize], Constant::String(string) if string.as_bytes() == bytes)
        }) {
            return Ok(*index);
        }
        let string = self.string_constants.intern_bytes(bytes, hash);
        self.add_string_constant(string, hash, span)
    }
    /// [`Self::string_constant_bytes`] for a string that exists already,
    /// which the unit then shares instead of building its own.
    fn string_constant_value(
        &mut self,
        string: &LuaString,
        span: Span,
    ) -> Result<u32, CompileError> {
        let hash = string.cached_hash();
        let constants = &self.constants;
        if let Some(index) = self.string_indices.find(hash, |index| {
            matches!(&constants[*index as usize], Constant::String(other) if other == string)
        }) {
            return Ok(*index);
        }
        let string = self.string_constants.intern(string);
        self.add_string_constant(string, hash, span)
    }
    fn add_string_constant(
        &mut self,
        string: LuaString,
        hash: u64,
        span: Span,
    ) -> Result<u32, CompileError> {
        let index = self.next_constant_index(span)?;
        self.constants.push(Constant::String(string));
        let constants = &self.constants;
        self.string_indices
            .insert_unique(hash, index, |index| match &constants[*index as usize] {
                Constant::String(string) => string.cached_hash(),
                _ => unreachable!("a string index points at a string constant"),
            });
        Ok(index)
    }
    fn next_constant_index(&self, span: Span) -> Result<u32, CompileError> {
        let index = u32::try_from(self.constants.len())
            .map_err(|_| CompileError::ConstantPoolExhausted { span })?;
        if index > Instruction::MAX_WIDE_INDEX {
            return Err(CompileError::ConstantPoolExhausted { span });
        }
        Ok(index)
    }
    pub(crate) fn constant_index(
        &mut self,
        constant: Constant,
        span: Span,
    ) -> Result<u32, CompileError> {
        let Some(key) = ConstantKey::of(&constant) else {
            let Constant::String(string) = constant else {
                unreachable!("only strings have no key");
            };
            return self.string_constant_value(&string, span);
        };
        if let Some(index) = self.constant_indices.get(&key) {
            return Ok(*index);
        }
        let index = self.next_constant_index(span)?;
        self.constants.push(constant);
        self.constant_indices.insert(key, index);
        Ok(index)
    }
    /// The constant index of the string `name` when it fits a field
    /// instruction's byte operand. A larger index leaves the constant
    /// added, so the register-keyed fallback reuses it.
    pub(crate) fn field_name_index(
        &mut self,
        name: &LuaString,
        span: Span,
    ) -> Result<Option<u8>, CompileError> {
        let index = self.string_constant_value(name, span)?;
        Ok(u8::try_from(index).ok())
    }
    /// [`Self::field_name_index`] for a name that is not yet a `LuaString`.
    pub(crate) fn field_name_index_str(
        &mut self,
        name: &str,
        span: Span,
    ) -> Result<Option<u8>, CompileError> {
        let index = self.string_constant(name, span)?;
        Ok(u8::try_from(index).ok())
    }
    /// The unit's string for the name `text`. The unit's pool already holds
    /// most names, so this is usually a reference count, not a new string.
    pub(crate) fn name_string(&self, text: &Ident) -> LuaString {
        self.string_constants.name(text)
    }
    /// Loads the string constant `text` into a fresh register.
    pub(crate) fn string_constant_register(
        &mut self,
        text: &str,
        span: Span,
    ) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        let index = self.string_constant(text, span)?;
        self.emit_load_constant(destination, index);
        Ok(destination)
    }
    /// [`Self::field_name_index`] for an index key that is a string literal.
    pub(crate) fn field_key_index(&mut self, key: &Expr) -> Result<Option<u8>, CompileError> {
        match key {
            Expr::String(name, span) => self.field_name_index(name, *span),
            _ => Ok(None),
        }
    }
    pub(crate) fn emit_load_constant(&mut self, destination: Register, index: u32) {
        if let Ok(index) = u16::try_from(index) {
            self.code
                .push(Instruction::load_constant(destination, index));
        } else {
            self.code.push(Instruction::load_constant_wide(destination));
            self.code.push(
                Instruction::extra_argument(index)
                    .expect("compiler limited the constant pool to wide indices"),
            );
        }
    }

    pub(crate) fn emit_get_global(&mut self, destination: Register, index: u32) {
        if let Ok(index) = u16::try_from(index) {
            self.code.push(Instruction::get_global(destination, index));
        } else {
            self.code.push(Instruction::get_global_wide(destination));
            self.code.push(
                Instruction::extra_argument(index)
                    .expect("compiler limited the constant pool to wide indices"),
            );
        }
    }
    pub(crate) fn emit_set_global(&mut self, source: Register, index: u32) {
        if let Ok(index) = u16::try_from(index) {
            self.code.push(Instruction::set_global(source, index));
        } else {
            self.code.push(Instruction::set_global_wide(source));
            self.code.push(
                Instruction::extra_argument(index)
                    .expect("compiler limited the constant pool to wide indices"),
            );
        }
    }
    pub(crate) fn emit_declare_global(&mut self, source: Register, index: u32) {
        if let Ok(index) = u16::try_from(index) {
            self.code.push(Instruction::declare_global(source, index));
        } else {
            self.code.push(Instruction::declare_global_wide(source));
            self.code.push(
                Instruction::extra_argument(index)
                    .expect("compiler limited the constant pool to wide indices"),
            );
        }
    }
    pub(crate) fn relative(
        instruction: usize,
        target: usize,
        span: Span,
    ) -> Result<i16, CompileError> {
        let next = instruction
            .checked_add(1)
            .ok_or(CompileError::JumpOutOfRange { span })?;
        let target = i64::try_from(target).map_err(|_| CompileError::JumpOutOfRange { span })?;
        let next = i64::try_from(next).map_err(|_| CompileError::JumpOutOfRange { span })?;
        i16::try_from(target - next).map_err(|_| CompileError::JumpOutOfRange { span })
    }
    pub(crate) fn relative_jump(
        instruction: usize,
        target: usize,
        span: Span,
    ) -> Result<i32, CompileError> {
        let next = instruction
            .checked_add(1)
            .ok_or(CompileError::JumpOutOfRange { span })?;
        let target = i64::try_from(target).map_err(|_| CompileError::JumpOutOfRange { span })?;
        let next = i64::try_from(next).map_err(|_| CompileError::JumpOutOfRange { span })?;
        i32::try_from(target - next).map_err(|_| CompileError::JumpOutOfRange { span })
    }
    pub(crate) fn patch_jump_if_false(
        &mut self,
        instruction: usize,
        target: usize,
        span: Span,
    ) -> Result<(), CompileError> {
        let condition = self.code[instruction].a();
        let offset = Self::relative(instruction, target, span)?;
        self.code[instruction] = Instruction::jump_if_false(condition, offset);
        Ok(())
    }
    /// Points the branch [`Self::emit_condition_jump`] emitted at `target`.
    pub(crate) fn patch_condition_jump(
        &mut self,
        instruction: usize,
        target: usize,
        span: Span,
    ) -> Result<(), CompileError> {
        if self.code[instruction].opcode() == Ok(OpCode::Jump) {
            let offset = Self::relative_jump(instruction, target, span)?;
            self.code[instruction] =
                Instruction::aj(OpCode::Jump, offset).expect("a checked offset fits a jump");
            Ok(())
        } else {
            self.patch_jump_if_false(instruction, target, span)
        }
    }
    pub(crate) fn emit_tail_call_open(
        &mut self,
        callee_value: Register,
        fixed_arguments: &[Register],
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<(), CompileError> {
        let base = self.tail_call_base(callee_value, fixed_arguments, span, evaluation)?;
        let count = u8::try_from(fixed_arguments.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        self.close_all_scopes();
        self.record_call_site(call_name, span);
        self.code.push(Instruction::tail_call_open(base, count));
        self.code.push(Instruction::return_open(base, 0));
        Ok(())
    }

    /// Puts a tail call's function and fixed arguments in consecutive
    /// registers, as [`Self::emit_call`] does: an operand that one
    /// instruction of the call's own `evaluation` left in a scratch register
    /// is written to its slot directly, with no copy.
    pub(crate) fn tail_call_base(
        &mut self,
        callee_value: Register,
        arguments: &[Register],
        span: Span,
        evaluation: Option<Evaluation>,
    ) -> Result<Register, CompileError> {
        let base = self.fresh(span)?;
        self.place_call_value(base, callee_value, evaluation);
        for argument in arguments {
            let slot = self.fresh(span)?;
            self.place_call_value(slot, *argument, evaluation);
        }
        Ok(base)
    }
    /// Notes where a call's operand evaluation begins, for
    /// [`Self::place_call_value`].
    pub(crate) fn begin_evaluation(&self) -> Evaluation {
        Evaluation {
            start_pc: self.code.len(),
            mark: self.next_register,
            method: None,
        }
    }

    /// Puts the call operand `value` in its call slot. When a single
    /// instruction of this call's own evaluation produced `value` in a
    /// scratch register that nothing else reads, that instruction writes the
    /// slot directly, as Lua's compiler does, and no copy is needed.
    ///
    /// The slot is above every register the evaluation used, so writing it
    /// early cannot disturb a later operand, and a callee frame starts above
    /// this frame's whole register window.
    fn place_call_value(
        &mut self,
        slot: Register,
        value: Register,
        evaluation: Option<Evaluation>,
    ) {
        if let Some(evaluation) = evaluation
            && slot > value
            && self.retarget_definition(value, slot, evaluation)
        {
            return;
        }
        // The receiver of a method call goes where the method was just
        // placed: the lookup that wrote the method to the slot before this
        // one also copies the receiver here, as Lua's SELF does. The slot
        // lies above every register the evaluation used, so the copy can
        // come before the arguments.
        if let Some(evaluation) = evaluation
            && let Some((pc, receiver)) = evaluation.method
            && value == receiver
            && let Some(lookup) = self.code.get(pc).copied()
            && lookup.opcode() == Ok(OpCode::GetField)
            && lookup.b() == receiver
            && lookup.a().checked_add(1) == Some(slot)
            && slot > receiver
        {
            self.code[pc] = Instruction::self_field(lookup.a(), receiver, lookup.c());
            return;
        }
        self.code.push(Instruction::move_(slot, value));
    }

    /// Stores the right operand of `and` or `or` in the expression's result
    /// register, writing it there directly when [`Self::retarget_definition`]
    /// allows. The result register already holds the left operand, which
    /// only the branch before the right side reads.
    pub(crate) fn place_branch_value(
        &mut self,
        destination: Register,
        value: Register,
        evaluation: Evaluation,
    ) {
        if destination != value && !self.retarget_definition(value, destination, evaluation) {
            self.code.push(Instruction::move_(destination, value));
        }
    }

    /// Rewrites the one instruction that defines scratch register `value`
    /// within `evaluation` to define `slot` instead. That holds when the
    /// instruction is a plain definition, and nothing after it reads or
    /// writes `value` or `slot`; the caller then skips the copy.
    // `pc` is captured for the write-back below, so the loop cannot use an
    // iterator alone.
    #[allow(clippy::needless_range_loop)]
    fn retarget_definition(
        &mut self,
        value: Register,
        slot: Register,
        evaluation: Evaluation,
    ) -> bool {
        if value < evaluation.mark {
            return false;
        }
        let mut definition = None;
        for pc in evaluation.start_pc..self.code.len() {
            let instruction = self.code[pc];
            let Ok(opcode) = instruction.opcode() else {
                return false;
            };
            let mentions = mentions_register(opcode, instruction, value);
            if definition.is_some() {
                // Nothing after the definition may touch either register.
                if mentions
                    || mentions_register(opcode, instruction, slot)
                    || ends_evaluation(opcode)
                {
                    return false;
                }
            } else if mentions {
                // The first mention must be the one definition.
                if instruction.a() != value || !is_retargetable_definition(opcode) {
                    return false;
                }
                definition = Some(pc);
            }
        }
        let Some(pc) = definition else {
            return false;
        };
        let instruction = self.code[pc];
        self.code[pc] =
            Instruction::from_raw((instruction.raw() & !0x0000_ff00) | (u32::from(slot) << 8));
        true
    }

    pub(crate) fn emit_call(
        &mut self,
        callee_value: Register,
        argument_values: &[Register],
        results: usize,
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<Register, CompileError> {
        if argument_values.len() == 1 && results == 1 {
            let destination = self.fresh(span)?;
            self.record_call_site(call_name, span);
            self.code.push(Instruction::call_one(
                destination,
                callee_value,
                argument_values[0],
            ));
            return Ok(destination);
        }
        let base = self.fresh(span)?;
        self.place_call_value(base, callee_value, evaluation);
        for argument in argument_values {
            let slot = self.fresh(span)?;
            self.place_call_value(slot, *argument, evaluation);
        }
        let count = u8::try_from(argument_values.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        let count_results =
            u8::try_from(results).map_err(|_| CompileError::RegisterExhausted { span })?;
        // Reserve result registers before validation. The VM writes a nil for absent values.
        for _ in 1..results {
            self.fresh(span)?;
        }
        self.record_call_site(call_name, span);
        self.code
            .push(Instruction::call(base, count, count_results));
        Ok(base)
    }
    pub(crate) fn emit_call_vararg(
        &mut self,
        callee_value: Register,
        fixed_arguments: &[Register],
        results: usize,
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<Register, CompileError> {
        let base = self.fresh(span)?;
        self.place_call_value(base, callee_value, evaluation);
        for argument in fixed_arguments {
            let slot = self.fresh(span)?;
            self.place_call_value(slot, *argument, evaluation);
        }
        let fixed_count = u8::try_from(fixed_arguments.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        let count_results =
            u8::try_from(results).map_err(|_| CompileError::RegisterExhausted { span })?;
        for _ in 1..results {
            self.fresh(span)?;
        }
        self.record_call_site(call_name, span);
        self.code
            .push(Instruction::call_vararg(base, fixed_count, count_results));
        Ok(base)
    }
    pub(crate) fn emit_call_open(
        &mut self,
        callee_value: Register,
        fixed_arguments: &[Register],
        results: usize,
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<Register, CompileError> {
        let base = self.fresh(span)?;
        self.place_call_value(base, callee_value, evaluation);
        for argument in fixed_arguments {
            let slot = self.fresh(span)?;
            self.place_call_value(slot, *argument, evaluation);
        }
        let fixed_count = u8::try_from(fixed_arguments.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        let count_results =
            u8::try_from(results).map_err(|_| CompileError::RegisterExhausted { span })?;
        for _ in 1..results {
            self.fresh(span)?;
        }
        self.record_call_site(call_name, span);
        self.code
            .push(Instruction::call_open(base, fixed_count, count_results));
        Ok(base)
    }

    pub(crate) fn emit_tail_call(
        &mut self,
        callee_value: Register,
        argument_values: &[Register],
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<(), CompileError> {
        let base = self.tail_call_base(callee_value, argument_values, span, evaluation)?;
        let count = u8::try_from(argument_values.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        self.close_all_scopes();
        self.record_call_site(call_name, span);
        self.code.push(Instruction::tail_call(base, count));
        self.code.push(Instruction::return_open(base, 0));
        Ok(())
    }

    pub(crate) fn emit_tail_call_vararg(
        &mut self,
        callee_value: Register,
        fixed_arguments: &[Register],
        span: Span,
        call_name: Option<(LuaString, CallKind)>,
        evaluation: Option<Evaluation>,
    ) -> Result<(), CompileError> {
        let base = self.tail_call_base(callee_value, fixed_arguments, span, evaluation)?;
        let count = u8::try_from(fixed_arguments.len())
            .map_err(|_| CompileError::RegisterExhausted { span })?;
        self.close_all_scopes();
        self.record_call_site(call_name, span);
        self.code.push(Instruction::tail_call_vararg(base, count));
        self.code.push(Instruction::return_open(base, 0));
        Ok(())
    }
    pub(crate) fn record_call_site(
        &mut self,
        call_name: Option<(LuaString, CallKind)>,
        span: Span,
    ) {
        let pc = self.code.len();
        let site = match call_name {
            Some((name, kind)) => CallSite::named(pc, name, kind),
            None => CallSite::anonymous(pc),
        };
        self.call_sites.push(site);
        self.instruction_lines.insert(pc, self.line_for(span));
    }

    /// Emits a write to the field named by constant `field`, whose table
    /// register `name` describes.
    pub(crate) fn emit_set_field(
        &mut self,
        table: Register,
        field: u8,
        value: Register,
        name: Option<(LuaString, CallKind)>,
    ) {
        let pc = self.code.len();
        self.code.push(Instruction::set_field(table, field, value));
        self.record_diagnostic_site(pc, table, name);
    }

    /// Emits a table write whose table register `name` describes.
    pub(crate) fn emit_set_table(
        &mut self,
        table: Register,
        key: Register,
        value: Register,
        name: Option<(LuaString, CallKind)>,
    ) {
        let pc = self.code.len();
        self.code.push(Instruction::set_table(table, key, value));
        self.record_diagnostic_site(pc, table, name);
    }

    pub(crate) fn record_diagnostic_site(
        &mut self,
        pc: usize,
        register: Register,
        name: Option<(LuaString, CallKind)>,
    ) {
        if let Some((name, kind)) = name {
            let duplicate = self
                .diagnostic_site_keys
                .iter()
                .rev()
                .take_while(|(previous_pc, _)| *previous_pc == pc)
                .any(|(_, previous_register)| *previous_register == register);
            if !duplicate {
                self.diagnostic_site_keys.push((pc, register));
                self.diagnostic_sites
                    .push(DiagnosticSite::new(pc, register, name, kind));
            }
        }
    }

    pub(crate) fn line_for(&self, span: Span) -> u32 {
        let line = self
            .line_starts
            .partition_point(|start| *start <= span.start);
        u32::try_from(line).unwrap_or(0)
    }

    /// The line of the last character of `span`.
    pub(crate) fn end_line(&self, span: Span) -> u32 {
        self.line_for(Span {
            start: span.end.saturating_sub(1).max(span.start),
            end: span.end,
        })
    }

    /// Gives the instructions one value of a declaration's or assignment's
    /// value list emitted from `start_pc` on, and that no nested expression
    /// placed, the value's own line. Stock Lua emits a value's instructions
    /// as it reads the value and the comma after it, so a value list that
    /// spans lines reports each value's line, not the statement's first one.
    /// A value that spans lines itself keeps the statement's line: stock Lua
    /// splits those instructions between its lines, which one line cannot
    /// describe.
    pub(crate) fn record_value_lines(&mut self, start_pc: usize, value: &syntax::Expr) {
        let span = value.span();
        let line = self.line_for(span);
        if line == self.end_line(span) {
            self.record_unmapped_instruction_line_range(start_pc, self.code.len(), line);
        }
    }

    /// Assigns a default source location to a contiguous emitted sequence.
    ///
    /// The parser gives control statements a span beginning at their keyword,
    /// but Lua associates condition evaluation with the condition itself. This
    /// matters when `if`, `while`, or `until` put that expression on a later
    /// physical line.
    pub(crate) fn record_instruction_line_range(&mut self, start: usize, end: usize, line: u32) {
        for pc in start..end {
            self.instruction_lines.insert_if_absent(pc, line);
        }
    }

    /// Assigns locations only to compiler housekeeping that has not already
    /// been tied to a nested statement or expression.
    pub(crate) fn record_unmapped_instruction_line_range(
        &mut self,
        start: usize,
        end: usize,
        line: u32,
    ) {
        for pc in start..end {
            self.instruction_lines.insert_if_absent(pc, line);
        }
    }

    /// Lua assigns a branch's synthetic jump to the last executable line in
    /// that branch. Empty branches have no such line, so retain the enclosing
    /// block's source line instead.
    pub(crate) fn block_exit_line(&self, block: &syntax::Block) -> u32 {
        block.statements.last().map_or_else(
            || self.line_for(block.span),
            |statement| self.line_for(statement_span(statement)),
        )
    }

    pub(crate) fn line_info(&self) -> Vec<u32> {
        if self.line_starts.is_empty() {
            return Vec::new();
        }
        let mut line_info = vec![0; self.code.len()];
        let recorded = self.instruction_lines.as_slice();
        let known = recorded.len().min(line_info.len());
        line_info[..known].copy_from_slice(&recorded[..known]);
        line_info
    }
}

/// Where a call's operands began, so [`Compiler::place_call_value`] can tell
/// the call's own scratch registers from older ones.
#[derive(Clone, Copy)]
pub(crate) struct Evaluation {
    start_pc: usize,
    mark: Register,
    /// For a method call, the `GetField` that reads the method and the
    /// register it reads it from: see [`Compiler::place_call_value`].
    pub(crate) method: Option<(usize, Register)>,
}

/// A branch condition whose operands are evaluated and whose branch is not
/// emitted yet (see [`Compiler::condition_operands`]).
pub(crate) enum PendingCondition {
    /// The condition's value is in the register.
    Value(Register),
    /// A comparison, to run as a compare-and-branch test.
    Test(PendingTest),
}

/// The fields of a compare-and-branch test and what its diagnostics need.
pub(crate) struct PendingTest {
    pub(crate) opcode: OpCode,
    pub(crate) a: Register,
    /// A register, or the constant index of a constant test.
    pub(crate) b: u8,
    pub(crate) expected: bool,
    pub(crate) constant_left: bool,
    pub(crate) line: u32,
    /// The register operands with the names an error message gives them.
    pub(crate) sites: Vec<(Register, Option<(LuaString, CallKind)>)>,
}

/// Instructions that write a fresh value to register `a` and nothing else.
fn is_retargetable_definition(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::LoadConstant
            | OpCode::LoadConstantWide
            | OpCode::GetGlobal
            | OpCode::GetGlobalWide
            | OpCode::GetUpvalue
            | OpCode::GetTabUp
            | OpCode::GetTable
            | OpCode::GetField
            | OpCode::Negate
            | OpCode::Not
            | OpCode::Length
            | OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Power
            | OpCode::FloorDivide
    )
}

/// Instructions that implicitly read or write register ranges, or leave the
/// expression, which the operand fields do not describe.
fn ends_evaluation(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::ClearRange
            | OpCode::Return
            | OpCode::ReturnVararg
            | OpCode::ReturnOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen
            | OpCode::NumericForPrep
            | OpCode::NumericForLoop
            | OpCode::ForInCall
            | OpCode::ForInLoop
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::CloseUpvalue
            | OpCode::MarkClose
            | OpCode::CloseLocal
            | OpCode::Halt
    )
}

/// Whether `instruction` names `register` in one of its operand fields. For
/// opcodes this does not list, all three fields count.
fn mentions_register(opcode: OpCode, instruction: Instruction, register: Register) -> bool {
    match opcode {
        // The rest of the word is a constant or upvalue index.
        OpCode::LoadConstant
        | OpCode::LoadConstantWide
        | OpCode::GetGlobal
        | OpCode::GetGlobalWide
        | OpCode::GetUpvalue
        | OpCode::GetTabUp
        | OpCode::LoadNil => instruction.a() == register,
        // `a` is an upvalue and `b` a constant.
        OpCode::SetTabUp => instruction.c() == register,
        OpCode::ExtraArgument => false,
        OpCode::Move | OpCode::Negate | OpCode::Not | OpCode::Length => {
            instruction.a() == register || instruction.b() == register
        }
        // `c` names the field or arithmetic constant, or holds a test's flags.
        OpCode::TestEqual
        | OpCode::TestLessThan
        | OpCode::TestLessEqual
        | OpCode::GetField
        | OpCode::AddK
        | OpCode::SubtractK
        | OpCode::MultiplyK
        | OpCode::DivideK
        | OpCode::ModuloK => instruction.a() == register || instruction.b() == register,
        // `b` names a constant and `c` holds flags.
        OpCode::TestEqualK | OpCode::TestLessThanK | OpCode::TestLessEqualK => {
            instruction.a() == register
        }
        // `a` and the register after it are written, `b` is read and `c`
        // names the field constant.
        OpCode::SelfField => {
            instruction.a() == register
                || instruction.a().checked_add(1) == Some(register)
                || instruction.b() == register
        }
        // `b` names the field constant.
        OpCode::SetField => instruction.a() == register || instruction.c() == register,
        _ => [instruction.a(), instruction.b(), instruction.c()].contains(&register),
    }
}
