//! Expression compilation: operators, short-circuits, table constructors, and call expressions.

use crate::bytecode::{CallKind, Constant, Instruction, OpCode};
use crate::compiler::emit::{PendingCondition, PendingTest};
use crate::compiler::{
    CallTail, CompileError, Compiler, FieldKeyNames, Register, collect_concat_operands,
    expands_open_call_results,
};
use crate::string::LuaString;
use crate::syntax::{self as syntax, BinaryOp, Expr, Name, Span, TableField, UnaryOp};

impl Compiler {
    pub(crate) fn expression(
        &mut self,
        expression: &Expr,
        results: usize,
    ) -> Result<Register, CompileError> {
        match expression {
            Expr::Name(name) => {
                if self.name_is_forced_global(&name.text) {
                    self.ensure_global_declared(name)?;
                    let destination = self.fresh(name.span)?;
                    if self.emit_environment_get(destination, name)? {
                    } else {
                        let name = self.string_constant(&name.text, name.span)?;
                        self.emit_get_global(destination, name);
                    }
                    Ok(destination)
                } else if let Some(register) = self.local_register(&name.text) {
                    Ok(register)
                } else if let Some(upvalue) = self.resolve_upvalue(name)? {
                    let destination = self.fresh(name.span)?;
                    self.code
                        .push(Instruction::get_upvalue(destination, upvalue));
                    Ok(destination)
                } else {
                    self.ensure_global_declared(name)?;
                    let destination = self.fresh(name.span)?;
                    if self.emit_environment_get(destination, name)? {
                        return Ok(destination);
                    }
                    let name = self.string_constant(&name.text, name.span)?;
                    self.emit_get_global(destination, name);
                    Ok(destination)
                }
            }
            Expr::Nil(span) => self.nil(*span),
            Expr::Boolean(value, span) => self.constant(Constant::Boolean(*value), *span),
            Expr::Integer(value, span) => self.constant(Constant::Integer(*value), *span),
            Expr::IntegerBoundary(span) => {
                self.constant(Constant::Number(9_223_372_036_854_775_808.0), *span)
            }
            Expr::Number(value, span) => self.constant(Constant::Number(*value), *span),
            Expr::Int64(value, span) => self.constant(Constant::Int64(*value), *span),
            Expr::UInt64(value, span) => self.constant(Constant::UInt64(*value), *span),
            Expr::String(value, span) => self.constant(Constant::String(value.clone()), *span),
            Expr::Vararg(span) => self.vararg(*span, results),
            Expr::Table { fields, span } => self.table_constructor(fields, *span),
            Expr::Function {
                parameters,
                vararg,
                vararg_name,
                body,
                span,
            } => self.function_expression(parameters, *vararg, vararg_name.as_deref(), body, *span),
            Expr::Index { table, key, span } => {
                let table_name = self.static_value_name(table);
                let table = self.expression(table, 1)?;
                if let Some(field) = self.field_key_index(key)? {
                    let destination = self.fresh(*span)?;
                    let pc = self.code.len();
                    self.code
                        .push(Instruction::get_field(destination, table, field));
                    self.record_diagnostic_site(pc, table, table_name);
                    return Ok(destination);
                }
                let key = self.expression(key, 1)?;
                let destination = self.fresh(*span)?;
                let pc = self.code.len();
                self.code
                    .push(Instruction::get_table(destination, table, key));
                self.record_diagnostic_site(pc, table, table_name);
                Ok(destination)
            }
            Expr::Binary {
                op,
                operator_span,
                left,
                right,
                span,
            } => self.binary_expression(*op, left, right, *span, *operator_span),
            Expr::Unary {
                op,
                expression,
                span,
            } => self.unary_expression(*op, expression, *span),
            Expr::Call {
                callee,
                arguments,
                span,
            } => self.call(callee, arguments, results, *span),
            Expr::MethodCall {
                argument_span,
                receiver,
                method,
                arguments,
                ..
            } => self.method_call(receiver, method, arguments, results, *argument_span),
            // Parentheses force a call to produce one value, even in final RHS position.
            Expr::Parenthesized { expression, .. } => self.expression(expression, 1),
        }
    }
    pub(crate) fn function_expression(
        &mut self,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        span: Span,
    ) -> Result<Register, CompileError> {
        let prototype = self.function_body(false, parameters, vararg, vararg_name, body, span)?;
        let index = u16::try_from(self.prototypes.len())
            .map_err(|_| CompileError::ConstantPoolExhausted { span })?;
        self.prototypes.push(prototype);
        let destination = self.fresh(span)?;
        self.code.push(Instruction::closure(destination, index));
        Ok(destination)
    }
    pub(crate) fn table_constructor(
        &mut self,
        fields: &[TableField],
        span: Span,
    ) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        let (array_hint, hash_hint) = constructor_size_hints(fields);
        self.code
            .push(Instruction::new_table(destination, array_hint, hash_hint));
        let reservation = destination
            .checked_add(1)
            .ok_or(CompileError::RegisterExhausted { span })?;
        self.reserved_register_floors.push(reservation);
        let result = self.table_fields(destination, fields, span);
        self.reserved_register_floors.pop();
        match result {
            Ok(()) => {
                // The returned table is still an expression value. Keep its
                // register live for the caller so a sibling expression cannot
                // overwrite it before the consuming instruction is emitted.
                self.next_register = self.next_register.max(reservation);
                Ok(destination)
            }
            Err(error) => {
                self.reclaim_temporaries();
                Err(error)
            }
        }
    }
    pub(crate) fn table_fields(
        &mut self,
        destination: Register,
        fields: &[TableField],
        span: Span,
    ) -> Result<(), CompileError> {
        let mut list_index = 1_u32;
        for (field_index, field) in fields.iter().enumerate() {
            if field_index + 1 == fields.len() {
                if matches!(field, TableField::List(Expr::Vararg(_))) {
                    self.disable_compatibility_arg();
                    self.code.push(Instruction::table_vararg(
                        destination,
                        u16::try_from(list_index)
                            .map_err(|_| CompileError::ConstantPoolExhausted { span })?,
                    ));
                    break;
                }
                if let TableField::List(value) = field
                    && expands_open_call_results(value)
                {
                    self.open_call_expression(value)?;
                    self.code.push(Instruction::table_open(
                        destination,
                        u16::try_from(list_index)
                            .map_err(|_| CompileError::ConstantPoolExhausted { span })?,
                    ));
                    break;
                }
            }
            if self.compact_registers {
                // A constant key is loaded after its value, so nested
                // constructors hold one register per level rather than two.
                let (key, value) = match field {
                    TableField::List(value) => {
                        let value = self.expression(value, 1)?;
                        let key = self.constant(Constant::Number(f64::from(list_index)), span)?;
                        list_index = list_index
                            .checked_add(1)
                            .ok_or(CompileError::ConstantPoolExhausted { span })?;
                        (key, value)
                    }
                    TableField::Name { name, value } => {
                        let value = self.expression(value, 1)?;
                        let text = &*name.text;
                        if let Some(field) = self.field_name_index_str(text, name.span)? {
                            self.code
                                .push(Instruction::set_field(destination, field, value));
                            self.reclaim_temporaries();
                            continue;
                        }
                        let key = self.string_constant_register(text, name.span)?;
                        (key, value)
                    }
                    TableField::Key { key, value } => {
                        (self.expression(key, 1)?, self.expression(value, 1)?)
                    }
                };
                self.code
                    .push(Instruction::set_table(destination, key, value));
                self.reclaim_temporaries();
                continue;
            }
            let (key, value) = match field {
                TableField::List(value) => {
                    let key =
                        self.constant(Constant::Number(f64::from(list_index)), value.span())?;
                    list_index = list_index
                        .checked_add(1)
                        .ok_or(CompileError::ConstantPoolExhausted { span })?;
                    (key, self.expression(value, 1)?)
                }
                TableField::Name { name, value } => {
                    let text = &*name.text;
                    if let Some(field) = self.field_name_index_str(text, name.span)? {
                        let value = self.expression(value, 1)?;
                        self.code
                            .push(Instruction::set_field(destination, field, value));
                        self.reclaim_temporaries();
                        continue;
                    }
                    (
                        self.string_constant_register(text, name.span)?,
                        self.expression(value, 1)?,
                    )
                }
                TableField::Key { key, value } => {
                    (self.expression(key, 1)?, self.expression(value, 1)?)
                }
            };
            self.code
                .push(Instruction::set_table(destination, key, value));
            // The table register stays reserved for the whole constructor,
            // while each field's key/value expression is dead after SetTable.
            // This keeps large literal constructors bounded and composes with
            // a nested constructor's own reservation.
            self.reclaim_temporaries();
        }
        Ok(())
    }
    /// Evaluates the operands of a branch condition, without branching yet.
    ///
    /// A comparison (`== ~= < <= > >=`, under any number of `not` and
    /// parentheses) written on one line becomes a compare-and-branch test,
    /// which [`Self::emit_condition_jump`] emits. Anything else is evaluated
    /// as a value, which that method branches on with `JumpIfFalse`.
    pub(crate) fn condition_operands(
        &mut self,
        condition: &Expr,
    ) -> Result<PendingCondition, CompileError> {
        let mut flip = false;
        let mut inner = condition;
        loop {
            match inner {
                Expr::Parenthesized { expression, .. } => inner = expression,
                Expr::Unary {
                    op: UnaryOp::Not,
                    expression,
                    ..
                } => {
                    flip = !flip;
                    inner = expression;
                }
                _ => break,
            }
        }
        // A condition that spans lines keeps the boolean and `JumpIfFalse`:
        // the test's jump word would otherwise sit on another line than the
        // test, and a line hook would see the difference.
        let span = condition.span();
        if let Expr::Binary {
            op:
                op @ (BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::LessThan
                | BinaryOp::LessEqual
                | BinaryOp::GreaterThan
                | BinaryOp::GreaterEqual),
            left,
            right,
            ..
        } = inner
            && self.line_for(span) == self.end_line(span)
        {
            return self.comparison_operands(*op, left, right, flip);
        }
        Ok(PendingCondition::Value(self.expression(condition, 1)?))
    }

    /// Emits a branch condition as a jump chain and returns the pcs of the
    /// jumps taken when the condition is false, for the caller to patch. The
    /// code falls through when the condition is true.
    ///
    /// `and`, `or` and `not` trees of comparisons and other operands compile
    /// to test opcodes and `JumpIfFalse`. A condition that spans lines keeps
    /// the boolean and one `JumpIfFalse`.
    pub(crate) fn condition_false_jumps(
        &mut self,
        condition: &Expr,
    ) -> Result<Vec<usize>, CompileError> {
        let span = condition.span();
        if self.line_for(span) != self.end_line(span) {
            return self.value_false_jump(condition);
        }
        self.jump_when_false(condition)
    }

    /// Whether the expression can jump when it is true, which only
    /// comparisons can. Other operands make their enclosing `or` or negated
    /// `and` fall back to the value form.
    fn can_jump_when_true(expression: &Expr) -> bool {
        match Self::strip_parentheses(expression) {
            Expr::Unary {
                op: UnaryOp::Not, ..
            } => true,
            Expr::Binary {
                op: BinaryOp::And,
                right,
                ..
            } => Self::can_jump_when_true(right),
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
                ..
            } => Self::can_jump_when_true(left) && Self::can_jump_when_true(right),
            Expr::Binary { op, .. } => Self::is_comparison(*op),
            _ => false,
        }
    }

    fn is_comparison(op: BinaryOp) -> bool {
        matches!(
            op,
            BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::LessThan
                | BinaryOp::LessEqual
                | BinaryOp::GreaterThan
                | BinaryOp::GreaterEqual
        )
    }

    pub(super) fn strip_parentheses(mut expression: &Expr) -> &Expr {
        while let Expr::Parenthesized {
            expression: inner, ..
        } = expression
        {
            expression = inner;
        }
        expression
    }

    /// Whether the expression contains a function body, whose captures the
    /// caller would have to close between the condition and its branch.
    pub(crate) fn contains_function(expression: &Expr) -> bool {
        match expression {
            Expr::Function { .. } => true,
            Expr::Table { fields, .. } => fields.iter().any(|field| match field {
                TableField::List(value) | TableField::Name { value, .. } => {
                    Self::contains_function(value)
                }
                TableField::Key { key, value } => {
                    Self::contains_function(key) || Self::contains_function(value)
                }
            }),
            Expr::Index { table, key, .. } => {
                Self::contains_function(table) || Self::contains_function(key)
            }
            Expr::Binary { left, right, .. } => {
                Self::contains_function(left) || Self::contains_function(right)
            }
            Expr::Unary { expression, .. } | Expr::Parenthesized { expression, .. } => {
                Self::contains_function(expression)
            }
            Expr::Call {
                callee, arguments, ..
            } => Self::contains_function(callee) || arguments.iter().any(Self::contains_function),
            Expr::MethodCall {
                receiver,
                arguments,
                ..
            } => Self::contains_function(receiver) || arguments.iter().any(Self::contains_function),
            _ => false,
        }
    }

    fn value_false_jump(&mut self, expression: &Expr) -> Result<Vec<usize>, CompileError> {
        let mark = self.next_register;
        let register = self.expression(expression, 1)?;
        let pc = self.code.len();
        self.code.push(Instruction::jump_if_false(register, 0));
        if self.compact_registers {
            self.release_registers(mark);
        }
        Ok(vec![pc])
    }

    fn comparison_jump(
        &mut self,
        expression: &Expr,
        jump_when_true: bool,
    ) -> Result<usize, CompileError> {
        let Expr::Binary {
            op, left, right, ..
        } = Self::strip_parentheses(expression)
        else {
            unreachable!("the caller checked for a comparison");
        };
        let mark = self.next_register;
        let pending = self.comparison_operands(*op, left, right, jump_when_true)?;
        let pc = self.emit_condition_jump(pending);
        if self.compact_registers {
            self.release_registers(mark);
        }
        Ok(pc)
    }

    fn patch_jumps_here(&mut self, jumps: &[usize], span: Span) -> Result<(), CompileError> {
        let target = self.code.len();
        for jump in jumps {
            self.patch_condition_jump(*jump, target, span)?;
        }
        Ok(())
    }

    /// Emits jumps taken when `expression` is false.
    fn jump_when_false(&mut self, expression: &Expr) -> Result<Vec<usize>, CompileError> {
        match Self::strip_parentheses(expression) {
            Expr::Unary {
                op: UnaryOp::Not,
                expression: operand,
                ..
            } if Self::can_jump_when_true(operand) => self.jump_when_true(operand),
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
                ..
            } => {
                let mut jumps = self.jump_when_false(left)?;
                jumps.extend(self.jump_when_false(right)?);
                Ok(jumps)
            }
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
                ..
            } if Self::can_jump_when_true(left) => {
                let true_jumps = self.jump_when_true(left)?;
                let false_jumps = self.jump_when_false(right)?;
                self.patch_jumps_here(&true_jumps, expression.span())?;
                Ok(false_jumps)
            }
            Expr::Binary { op, .. } if Self::is_comparison(*op) => {
                Ok(vec![self.comparison_jump(expression, false)?])
            }
            _ => self.value_false_jump(expression),
        }
    }

    /// Emits jumps taken when `expression` is true. The caller checked
    /// [`Self::can_jump_when_true`].
    fn jump_when_true(&mut self, expression: &Expr) -> Result<Vec<usize>, CompileError> {
        match Self::strip_parentheses(expression) {
            Expr::Unary {
                op: UnaryOp::Not,
                expression: operand,
                ..
            } => self.jump_when_false(operand),
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
                ..
            } => {
                let false_jumps = self.jump_when_false(left)?;
                let true_jumps = self.jump_when_true(right)?;
                self.patch_jumps_here(&false_jumps, expression.span())?;
                Ok(true_jumps)
            }
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
                ..
            } => {
                let mut jumps = self.jump_when_true(left)?;
                jumps.extend(self.jump_when_true(right)?);
                Ok(jumps)
            }
            Expr::Binary { op, .. } if Self::is_comparison(*op) => {
                Ok(vec![self.comparison_jump(expression, true)?])
            }
            _ => unreachable!("the caller checked that the expression can jump when true"),
        }
    }

    /// The constant a comparison operand stands for, if it is one the test
    /// opcodes take: a number literal (negated or not) for an ordering, and
    /// also `nil`, a boolean or a string for equality.
    fn comparison_constant(equality: bool, expression: &Expr) -> Option<Constant> {
        match expression {
            Expr::Integer(value, _) => Some(Constant::Integer(*value)),
            Expr::Number(value, _) => Some(Constant::Number(*value)),
            Expr::Unary {
                op: UnaryOp::Negate,
                expression,
                ..
            } => match expression.as_ref() {
                Expr::Integer(value, _) => Some(Constant::Integer(value.wrapping_neg())),
                Expr::Number(value, _) => Some(Constant::Number(-value)),
                _ => None,
            },
            Expr::Nil(_) if equality => Some(Constant::Nil),
            Expr::Boolean(value, _) if equality => Some(Constant::Boolean(*value)),
            Expr::String(value, _) if equality => Some(Constant::String(value.clone())),
            _ => None,
        }
    }

    fn comparison_operands(
        &mut self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        flip: bool,
    ) -> Result<PendingCondition, CompileError> {
        // The jump is taken when the condition is false, which is when the
        // comparison's result equals `expected`.
        let expected = flip ^ matches!(op, BinaryOp::NotEqual);
        let equality = matches!(op, BinaryOp::Equal | BinaryOp::NotEqual);
        // `a > b` is `b < a`. Operands are still evaluated left to right.
        let reversed = matches!(op, BinaryOp::GreaterThan | BinaryOp::GreaterEqual);
        let (register_opcode, constant_opcode) = match op {
            BinaryOp::Equal | BinaryOp::NotEqual => (OpCode::TestEqual, OpCode::TestEqualK),
            BinaryOp::LessThan | BinaryOp::GreaterThan => {
                (OpCode::TestLessThan, OpCode::TestLessThanK)
            }
            _ => (OpCode::TestLessEqual, OpCode::TestLessEqualK),
        };
        let line = self.line_for(Span {
            start: right.span().end.saturating_sub(1),
            end: right.span().end,
        });
        let left_name = self.static_value_name(left);
        let right_name = self.static_value_name(right);
        // A constant on the right is preferred; one on the left only when
        // the right is not one.
        let constant_side = if let Some(constant) = Self::comparison_constant(equality, right) {
            Some((constant, false))
        } else {
            Self::comparison_constant(equality, left).map(|constant| (constant, true))
        };
        if let Some((constant, on_left)) = constant_side {
            let (operand, operand_name, constant_span) = if on_left {
                (right, right_name.clone(), left.span())
            } else {
                (left, left_name.clone(), right.span())
            };
            let register = self.expression(operand, 1)?;
            let index = self.constant_index(constant, constant_span)?;
            if let Ok(index) = u8::try_from(index) {
                // In the ordering `R < K`, the constant is on the left of the
                // normalized comparison when it was written on the left of
                // `<` or `<=`, or on the right of `>` or `>=`.
                let constant_left = on_left != reversed;
                return Ok(PendingCondition::Test(PendingTest {
                    opcode: constant_opcode,
                    a: register,
                    b: index,
                    expected,
                    constant_left: !equality && constant_left,
                    line,
                    sites: vec![(register, operand_name)],
                }));
            }
            // The pool index does not fit: load the constant like any
            // operand and use the register form.
            let other = self.expression(if on_left { left } else { right }, 1)?;
            let (first, second) = if on_left {
                ((other, left_name), (register, right_name))
            } else {
                ((register, left_name), (other, right_name))
            };
            return Ok(Self::register_test(
                register_opcode,
                first,
                second,
                reversed,
                expected,
                line,
            ));
        }
        let first = self.expression(left, 1)?;
        let second = self.expression(right, 1)?;
        Ok(Self::register_test(
            register_opcode,
            (first, left_name),
            (second, right_name),
            reversed,
            expected,
            line,
        ))
    }

    fn register_test(
        opcode: OpCode,
        left: (Register, Option<(LuaString, CallKind)>),
        right: (Register, Option<(LuaString, CallKind)>),
        reversed: bool,
        expected: bool,
        line: u32,
    ) -> PendingCondition {
        let (first, second) = if reversed {
            (right, left)
        } else {
            (left, right)
        };
        PendingCondition::Test(PendingTest {
            opcode,
            a: first.0,
            b: second.0,
            expected,
            constant_left: false,
            line,
            sites: vec![(first.0, first.1), (second.0, second.1)],
        })
    }

    /// Emits the branch of a condition whose operands
    /// [`Self::condition_operands`] evaluated. The jump is taken when the
    /// condition is false. Returns the pc of the instruction to patch: a
    /// `JumpIfFalse`, or the `Jump` word after a test.
    pub(crate) fn emit_condition_jump(&mut self, pending: PendingCondition) -> usize {
        match pending {
            PendingCondition::Value(register) => {
                let pc = self.code.len();
                self.code.push(Instruction::jump_if_false(register, 0));
                pc
            }
            PendingCondition::Test(test) => {
                let pc = self.code.len();
                self.instruction_lines.insert(pc, test.line);
                self.code.push(Instruction::test(
                    test.opcode,
                    test.a,
                    test.b,
                    test.expected,
                    test.constant_left,
                ));
                for (register, name) in test.sites {
                    self.record_diagnostic_site(pc, register, name);
                }
                let jump = self.code.len();
                self.code
                    .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
                jump
            }
        }
    }
    pub(crate) fn binary_expression(
        &mut self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        span: Span,
        operator_span: Span,
    ) -> Result<Register, CompileError> {
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            return self.short_circuit(op, left, right, span);
        }
        if matches!(op, BinaryOp::Concat) {
            return self.concat_expression(left, right, span, operator_span);
        }
        let instruction_span = if self.legacy_vararg_compat
            || matches!(
                op,
                BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::LessThan
                    | BinaryOp::LessEqual
                    | BinaryOp::GreaterThan
                    | BinaryOp::GreaterEqual
            ) {
            Span {
                start: right.span().end.saturating_sub(1),
                end: right.span().end,
            }
        } else {
            operator_span
        };
        let subtract_zero = self.subtract_immediate
            && matches!(op, BinaryOp::Subtract)
            && self.integer_constant(right) == Some(0);
        let left_name = self.static_value_name(left);
        let right_name = self.static_value_name(right);
        let constant_form = if subtract_zero {
            None
        } else {
            Self::constant_arithmetic_opcode(op)
        };
        let right_constant = match right {
            Expr::Integer(value, _) => Some((Constant::Integer(*value), right.span())),
            Expr::Number(value, _) => Some((Constant::Number(*value), right.span())),
            _ => None,
        };
        let left = self.expression(left, 1)?;
        if let Some(opcode) = constant_form
            && let Some((constant, constant_span)) = right_constant
        {
            let index = self.constant_index(constant, constant_span)?;
            if let Ok(index) = u8::try_from(index) {
                let destination = self.fresh(span)?;
                let pc = self.code.len();
                self.instruction_lines
                    .insert(pc, self.line_for(instruction_span));
                self.code
                    .push(Instruction::abc(opcode, destination, left, index));
                self.record_diagnostic_site(pc, left, left_name);
                return Ok(destination);
            }
        }
        let right = self.expression(right, 1)?;
        let destination = self.fresh(span)?;
        let opcode = match op {
            BinaryOp::Add => OpCode::Add,
            BinaryOp::Subtract => OpCode::Subtract,
            BinaryOp::Multiply => OpCode::Multiply,
            BinaryOp::Divide => OpCode::Divide,
            BinaryOp::FloorDivide => OpCode::FloorDivide,
            BinaryOp::Modulo => OpCode::Modulo,
            BinaryOp::Power => OpCode::Power,
            BinaryOp::BitAnd => OpCode::BitAnd,
            BinaryOp::BitOr => OpCode::BitOr,
            BinaryOp::BitXor => OpCode::BitXor,
            BinaryOp::ShiftLeft => OpCode::ShiftLeft,
            BinaryOp::ShiftRight => OpCode::ShiftRight,
            BinaryOp::Concat => unreachable!("concatenation has its own range emitter"),
            BinaryOp::Equal => OpCode::Equal,
            BinaryOp::LessThan => OpCode::LessThan,
            BinaryOp::LessEqual => OpCode::LessEqual,
            BinaryOp::NotEqual => {
                let pc = self.code.len();
                self.instruction_lines
                    .insert(pc, self.line_for(instruction_span));
                self.code
                    .push(Instruction::binary(OpCode::Equal, destination, left, right));
                self.record_diagnostic_site(pc, left, left_name);
                self.record_diagnostic_site(pc, right, right_name);
                self.code
                    .push(Instruction::unary(OpCode::Not, destination, destination));
                return Ok(destination);
            }
            // Lua defines `a > b` and `a >= b` in terms of reversed
            // less-than relations. This keeps the VM instruction set compact.
            BinaryOp::GreaterThan => {
                let pc = self.code.len();
                self.instruction_lines
                    .insert(pc, self.line_for(instruction_span));
                self.code.push(Instruction::binary(
                    OpCode::LessThan,
                    destination,
                    right,
                    left,
                ));
                self.record_diagnostic_site(pc, right, right_name);
                self.record_diagnostic_site(pc, left, left_name);
                return Ok(destination);
            }
            BinaryOp::GreaterEqual => {
                let pc = self.code.len();
                self.instruction_lines
                    .insert(pc, self.line_for(instruction_span));
                self.code.push(Instruction::binary(
                    OpCode::LessEqual,
                    destination,
                    right,
                    left,
                ));
                self.record_diagnostic_site(pc, right, right_name);
                self.record_diagnostic_site(pc, left, left_name);
                return Ok(destination);
            }
            BinaryOp::And | BinaryOp::Or => unreachable!("handled above"),
        };
        if subtract_zero {
            return self.subtract_zero(
                destination,
                left,
                right,
                span,
                instruction_span,
                left_name,
                right_name,
            );
        }
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(instruction_span));
        self.code
            .push(Instruction::binary(opcode, destination, left, right));
        self.record_diagnostic_site(pc, left, left_name);
        self.record_diagnostic_site(pc, right, right_name);
        Ok(destination)
    }

    /// The constant-operand opcode for a binary operator, if it has one.
    fn constant_arithmetic_opcode(op: BinaryOp) -> Option<OpCode> {
        match op {
            BinaryOp::Add => Some(OpCode::AddK),
            BinaryOp::Subtract => Some(OpCode::SubtractK),
            BinaryOp::Multiply => Some(OpCode::MultiplyK),
            BinaryOp::Divide => Some(OpCode::DivideK),
            BinaryOp::Modulo => Some(OpCode::ModuloK),
            _ => None,
        }
    }

    /// Lua 5.4 compiles `x - 0` as `x + 0` (`OP_ADDI`), so a float zero of
    /// either sign gives `+0.0`, while a metamethod, a string operand or an
    /// error still sees a subtraction. The register `right` holds the
    /// integer zero. A number equal to zero takes the addition, and
    /// everything else (a table never equals a number) takes the subtraction.
    #[allow(clippy::too_many_arguments)]
    fn subtract_zero(
        &mut self,
        destination: Register,
        left: Register,
        right: Register,
        span: Span,
        instruction_span: Span,
        left_name: Option<(LuaString, CallKind)>,
        right_name: Option<(LuaString, CallKind)>,
    ) -> Result<Register, CompileError> {
        let line = self.line_for(instruction_span);
        let test = self.code.len();
        self.instruction_lines.insert(test, line);
        self.code
            .push(Instruction::binary(OpCode::Equal, destination, left, right));
        let to_subtract = self.code.len();
        self.code.push(Instruction::jump_if_false(destination, 0));
        let add = self.code.len();
        self.instruction_lines.insert(add, line);
        self.code
            .push(Instruction::binary(OpCode::Add, destination, left, right));
        let to_end = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
        let subtract = self.code.len();
        let offset = Self::relative(to_subtract, subtract, span)?;
        self.code[to_subtract] = Instruction::jump_if_false(destination, offset);
        self.instruction_lines.insert(subtract, line);
        self.code.push(Instruction::binary(
            OpCode::Subtract,
            destination,
            left,
            right,
        ));
        self.record_diagnostic_site(subtract, left, left_name);
        self.record_diagnostic_site(subtract, right, right_name);
        // A jump to the next instruction closes the branches. Callers rename
        // the destination of an expression's last instruction, which would
        // leave the other arm writing the old register.
        let end = self.code.len();
        self.code
            .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
        let end_offset = Self::relative(to_end, end, span)?;
        self.code[to_end] = Instruction::aj(OpCode::Jump, i32::from(end_offset))
            .expect("i16 offset fits in a jump");
        Ok(destination)
    }

    pub(crate) fn concat_expression(
        &mut self,
        left: &Expr,
        right: &Expr,
        span: Span,
        operator_span: Span,
    ) -> Result<Register, CompileError> {
        let mut concat_span = operator_span;
        let mut tail = right;
        while let Expr::Binary {
            op: BinaryOp::Concat,
            right,
            operator_span,
            ..
        } = tail
        {
            concat_span = *operator_span;
            tail = right;
        }
        if self.legacy_vararg_compat {
            concat_span = Span {
                start: right.span().end.saturating_sub(1),
                end: right.span().end,
            };
        }
        let mut operands = Vec::new();
        collect_concat_operands(left, &mut operands);
        collect_concat_operands(right, &mut operands);
        if self.compact_registers {
            return self.compact_concat(&operands, span, concat_span);
        }

        let destination = self.fresh(span)?;
        let mut slots = Vec::with_capacity(operands.len());
        for operand in &operands {
            slots.push(self.fresh(operand.span())?);
        }
        let names: Vec<_> = operands
            .iter()
            .map(|operand| self.static_value_name(operand))
            .collect();
        for (operand, slot) in operands.into_iter().zip(slots.iter().copied()) {
            let value = self.expression(operand, 1)?;
            if value != slot {
                self.code.push(Instruction::move_(slot, value));
            }
        }
        let first = *slots.first().expect("a binary concatenation has operands");
        let last = *slots.last().expect("a binary concatenation has operands");
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(concat_span));
        self.code
            .push(Instruction::concat(destination, first, last));
        for (slot, name) in slots.into_iter().zip(names) {
            self.record_diagnostic_site(pc, slot, name);
        }
        Ok(destination)
    }
    pub(crate) fn unary_expression(
        &mut self,
        op: UnaryOp,
        expression: &Expr,
        span: Span,
    ) -> Result<Register, CompileError> {
        let source_name = self.static_value_name(expression);
        let source = self.expression(expression, 1)?;
        let destination = self.fresh(span)?;
        let opcode = match op {
            UnaryOp::Negate => OpCode::Negate,
            UnaryOp::Not => OpCode::Not,
            UnaryOp::Length => OpCode::Length,
            UnaryOp::BitNot => OpCode::BitNot,
        };
        let pc = self.code.len();
        self.code
            .push(Instruction::unary(opcode, destination, source));
        self.record_diagnostic_site(pc, source, source_name);
        Ok(destination)
    }
    pub(crate) fn short_circuit(
        &mut self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        span: Span,
    ) -> Result<Register, CompileError> {
        // `and` and `or` return an operand, rather than a boolean. Hold the
        // left value in the result register before the branch and overwrite
        // it only when evaluation reaches the right side. A scratch register
        // that this expression produced is the result register itself; a
        // local is copied, since the right side must not overwrite it.
        let mark = self.next_register;
        let left = self.expression(left, 1)?;
        let destination = if left >= mark {
            left
        } else {
            self.copy(left, span)?
        };
        match op {
            BinaryOp::And => {
                let skip_right = self.code.len();
                self.code.push(Instruction::jump_if_false(destination, 0));
                let evaluation = self.begin_evaluation();
                let right = self.expression(right, 1)?;
                self.place_branch_value(destination, right, evaluation);
                let offset = Self::relative(skip_right, self.code.len(), span)?;
                self.code[skip_right] = Instruction::jump_if_false(destination, offset);
            }
            BinaryOp::Or => {
                let evaluate_right = self.code.len();
                self.code.push(Instruction::jump_if_false(destination, 0));
                let skip_right = self.code.len();
                self.code
                    .push(Instruction::aj(OpCode::Jump, 0).expect("zero jump fits"));
                let right_start = self.code.len();
                let right_offset = Self::relative(evaluate_right, right_start, span)?;
                self.code[evaluate_right] = Instruction::jump_if_false(destination, right_offset);
                let evaluation = self.begin_evaluation();
                let right = self.expression(right, 1)?;
                self.place_branch_value(destination, right, evaluation);
                let end_offset = Self::relative(skip_right, self.code.len(), span)?;
                self.code[skip_right] = Instruction::aj(OpCode::Jump, i32::from(end_offset))
                    .expect("i16 offset fits in a jump");
            }
            _ => unreachable!("short-circuit operator was checked by caller"),
        }
        Ok(destination)
    }
    pub(crate) fn call(
        &mut self,
        callee: &Expr,
        arguments: &[Expr],
        results: usize,
        span: Span,
    ) -> Result<Register, CompileError> {
        if self.compact_registers {
            return self.compact_call(callee, arguments, results, span);
        }
        let call_name = self.static_call_name(callee);
        let evaluation = self.begin_evaluation();
        let callee_value = self.call_operand(callee, arguments, &[])?;
        if let Some((last, fixed)) = arguments.split_last() {
            if matches!(last, Expr::Vararg(_)) {
                self.disable_compatibility_arg();
                let fixed_values = self.call_operands(fixed, &[])?;
                return self.emit_call_vararg(
                    callee_value,
                    &fixed_values,
                    results,
                    span,
                    call_name,
                    Some(evaluation),
                );
            }
            if expands_open_call_results(last) {
                let fixed_values = self.call_operands(fixed, &arguments[fixed.len()..])?;
                self.open_call_expression(last)?;
                return self.emit_call_open(
                    callee_value,
                    &fixed_values,
                    results,
                    span,
                    call_name,
                    Some(evaluation),
                );
            }
        }
        let argument_values = self.call_operands(arguments, &[])?;
        self.emit_call(
            callee_value,
            &argument_values,
            results,
            span,
            call_name,
            Some(evaluation),
        )
    }
    pub(crate) fn method_call(
        &mut self,
        receiver: &Expr,
        method: &Name,
        arguments: &[Expr],
        results: usize,
        span: Span,
    ) -> Result<Register, CompileError> {
        let call_name = Some((self.name_string(&method.text), CallKind::Method));
        if self.compact_registers {
            let base = self.place_method(receiver, method, span)?;
            let tail = call_tail(arguments);
            let count = self.place_arguments(base, arguments, tail)?;
            return self.emit_placed_call(base, count, tail, results, span, call_name);
        }
        let receiver_name = self.static_value_name(receiver);
        let mut evaluation = Some(self.begin_evaluation());
        let receiver = self.receiver_operand(receiver, span)?;
        let lookup = self.method_lookup(method)?;
        let callee = self.fresh(span)?;
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(method.span));
        self.code.push(lookup.instruction(callee, receiver));
        self.record_diagnostic_site(pc, receiver, receiver_name);
        if let (Some(evaluation), MethodLookup::Field(_)) = (&mut evaluation, lookup) {
            evaluation.method = Some((pc, receiver));
        }
        if let Some((last, fixed)) = arguments.split_last() {
            if matches!(last, Expr::Vararg(_)) {
                self.disable_compatibility_arg();
                let mut fixed_values = Vec::with_capacity(fixed.len() + 1);
                fixed_values.push(receiver);
                fixed_values.extend(self.call_operands(fixed, &[])?);
                return self.emit_call_vararg(
                    callee,
                    &fixed_values,
                    results,
                    span,
                    call_name,
                    evaluation,
                );
            }
            if expands_open_call_results(last) {
                let mut fixed_values = Vec::with_capacity(fixed.len() + 1);
                fixed_values.push(receiver);
                fixed_values.extend(self.call_operands(fixed, &arguments[fixed.len()..])?);
                self.open_call_expression(last)?;
                return self.emit_call_open(
                    callee,
                    &fixed_values,
                    results,
                    span,
                    call_name,
                    evaluation,
                );
            }
        }
        let mut values = Vec::with_capacity(arguments.len() + 1);
        values.push(receiver);
        values.extend(self.call_operands(arguments, &[])?);
        self.emit_call(callee, &values, results, span, call_name, evaluation)
    }

    pub(crate) fn tail_call(
        &mut self,
        callee: &Expr,
        arguments: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        if self.compact_registers {
            let call_name = self.static_call_name(callee);
            let base = self.expression_to_next(callee)?;
            let tail = call_tail(arguments);
            let count = self.place_arguments(base, arguments, tail)?;
            return self.emit_placed_tail_call(base, count, tail, span, call_name);
        }
        let call_name = self.static_call_name(callee);
        let evaluation = Some(self.begin_evaluation());
        let callee_value = self.call_operand(callee, arguments, &[])?;
        if let Some((last, fixed)) = arguments.split_last() {
            if matches!(last, Expr::Vararg(_)) {
                self.disable_compatibility_arg();
                let fixed_values = self.call_operands(fixed, &[])?;
                return self.emit_tail_call_vararg(
                    callee_value,
                    &fixed_values,
                    span,
                    call_name,
                    evaluation,
                );
            }
            if expands_open_call_results(last) {
                let fixed_values = self.call_operands(fixed, &arguments[fixed.len()..])?;
                self.open_call_expression(last)?;
                return self.emit_tail_call_open(
                    callee_value,
                    &fixed_values,
                    span,
                    call_name,
                    evaluation,
                );
            }
        }
        let argument_values = self.call_operands(arguments, &[])?;
        self.emit_tail_call(callee_value, &argument_values, span, call_name, evaluation)
    }

    pub(crate) fn tail_method_call(
        &mut self,
        receiver: &Expr,
        method: &Name,
        arguments: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        let call_name = Some((self.name_string(&method.text), CallKind::Method));
        if self.compact_registers {
            let base = self.place_method(receiver, method, span)?;
            let tail = call_tail(arguments);
            let count = self.place_arguments(base, arguments, tail)?;
            return self.emit_placed_tail_call(base, count, tail, span, call_name);
        }
        let receiver_name = self.static_value_name(receiver);
        let mut evaluation = Some(self.begin_evaluation());
        let receiver = self.receiver_operand(receiver, span)?;
        let lookup = self.method_lookup(method)?;
        let callee = self.fresh(span)?;
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(method.span));
        self.code.push(lookup.instruction(callee, receiver));
        self.record_diagnostic_site(pc, receiver, receiver_name);
        if let (Some(evaluation), MethodLookup::Field(_)) = (&mut evaluation, lookup) {
            evaluation.method = Some((pc, receiver));
        }
        if let Some((last, fixed)) = arguments.split_last() {
            if matches!(last, Expr::Vararg(_)) {
                self.disable_compatibility_arg();
                let mut fixed_values = Vec::with_capacity(fixed.len() + 1);
                fixed_values.push(receiver);
                fixed_values.extend(self.call_operands(fixed, &[])?);
                return self.emit_tail_call_vararg(
                    callee,
                    &fixed_values,
                    span,
                    call_name,
                    evaluation,
                );
            }
            if expands_open_call_results(last) {
                let mut fixed_values = Vec::with_capacity(fixed.len() + 1);
                fixed_values.push(receiver);
                fixed_values.extend(self.call_operands(fixed, &arguments[fixed.len()..])?);
                self.open_call_expression(last)?;
                return self.emit_tail_call_open(
                    callee,
                    &fixed_values,
                    span,
                    call_name,
                    evaluation,
                );
            }
        }
        let mut values = Vec::with_capacity(arguments.len() + 1);
        values.push(receiver);
        values.extend(self.call_operands(arguments, &[])?);
        self.emit_tail_call(callee, &values, span, call_name, evaluation)
    }
    /// Evaluates one call operand, the function or an argument. Lua moves
    /// each operand into its call slot as it goes, but the normal allocator
    /// reads a local's own register only at the call. So a local that a
    /// nested closure assigns is copied at once when a `later` operand could
    /// run that closure.
    fn call_operand(
        &mut self,
        operand: &Expr,
        later: &[Expr],
        tail: &[Expr],
    ) -> Result<Register, CompileError> {
        let value = self.expression(operand, 1)?;
        if self.closure_may_reassign(operand)
            && later
                .iter()
                .chain(tail)
                .any(|later| self.may_run_code(later))
        {
            return self.copy(value, operand.span());
        }
        Ok(value)
    }

    /// Evaluates fixed call arguments, which `tail` (a final open call)
    /// follows.
    fn call_operands(
        &mut self,
        arguments: &[Expr],
        tail: &[Expr],
    ) -> Result<Vec<Register>, CompileError> {
        let mut values = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            values.push(self.call_operand(argument, &arguments[index + 1..], tail)?);
        }
        Ok(values)
    }

    /// Evaluates a method call's receiver. Like Lua's SELF, a local that a
    /// nested closure assigns is copied before the method lookup, whose
    /// __index could run that closure as well as any argument.
    fn receiver_operand(&mut self, receiver: &Expr, span: Span) -> Result<Register, CompileError> {
        let value = self.expression(receiver, 1)?;
        if self.closure_may_reassign(receiver) {
            return self.copy(value, span);
        }
        Ok(value)
    }

    /// Whether `expression` reads a local of this function that one of its
    /// nested closures assigns.
    fn closure_may_reassign(&self, expression: &Expr) -> bool {
        match expression {
            Expr::Name(name) => {
                self.closure_assigned_names.contains(&name.text)
                    && !self.name_is_forced_global(&name.text)
                    && self.local_register(&name.text).is_some()
            }
            Expr::Parenthesized { expression, .. } => self.closure_may_reassign(expression),
            _ => false,
        }
    }

    /// Whether evaluating `expression` could call Lua code: a call, a global
    /// read, or an operation with a metamethod.
    fn may_run_code(&self, expression: &Expr) -> bool {
        match expression {
            Expr::Nil(_)
            | Expr::Boolean(_, _)
            | Expr::Integer(_, _)
            | Expr::IntegerBoundary(_)
            | Expr::Number(_, _)
            | Expr::Int64(_, _)
            | Expr::UInt64(_, _)
            | Expr::String(_, _)
            | Expr::Vararg(_)
            | Expr::Function { .. } => false,
            Expr::Name(name) => {
                self.name_is_forced_global(&name.text)
                    || !matches!(
                        self.static_value_name(expression),
                        Some((_, CallKind::Local | CallKind::Upvalue))
                    )
            }
            Expr::Parenthesized { expression, .. } => self.may_run_code(expression),
            Expr::Table { fields, .. } => fields.iter().any(|field| match field {
                TableField::List(value) | TableField::Name { value, .. } => {
                    self.may_run_code(value)
                }
                TableField::Key { key, value } => {
                    self.may_run_code(key) || self.may_run_code(value)
                }
            }),
            Expr::Index { .. }
            | Expr::Binary { .. }
            | Expr::Unary { .. }
            | Expr::Call { .. }
            | Expr::MethodCall { .. } => true,
        }
    }

    pub(crate) fn static_call_name(&self, expression: &Expr) -> Option<(LuaString, CallKind)> {
        self.static_value_name(expression)
    }

    pub(crate) fn static_value_name(&self, expression: &Expr) -> Option<(LuaString, CallKind)> {
        match expression {
            Expr::Name(name) => {
                let kind = if self.local_register(&name.text).is_some() {
                    CallKind::Local
                } else if self.upvalue_names.contains_key(&name.text)
                    || self.parent_captures.contains_key(&name.text)
                {
                    CallKind::Upvalue
                } else {
                    CallKind::Global
                };
                Some((self.name_string(&name.text), kind))
            }
            Expr::Index { key, .. } => {
                let name = match (key.as_ref(), self.field_key_names) {
                    (Expr::String(name, _), _) => name.clone(),
                    (_, FieldKeyNames::Unnamed) => return None,
                    (Expr::Integer(0..=255, _), FieldKeyNames::IntegerIndex) => {
                        LuaString::from("integer index")
                    }
                    _ => LuaString::from("?"),
                };
                Some((name, CallKind::Field))
            }
            Expr::String(value, _) if self.constant_names => {
                Some((value.clone(), CallKind::Constant))
            }
            // Parentheses force call result adjustment but leave the callee's
            // lexical name intact for diagnostics.
            Expr::Parenthesized { expression, .. } => self.static_call_name(expression),
            _ => None,
        }
    }
    pub(crate) fn open_call_expression(
        &mut self,
        expression: &Expr,
    ) -> Result<Register, CompileError> {
        match expression {
            Expr::Call {
                callee,
                arguments,
                span,
            } => self.call(callee, arguments, 0, *span),
            Expr::MethodCall {
                argument_span,
                receiver,
                method,
                arguments,
                ..
            } => self.method_call(receiver, method, arguments, 0, *argument_span),
            _ => unreachable!("only calls can produce open call results"),
        }
    }
    pub(crate) fn pack(
        &mut self,
        values: &[Register],
        span: Span,
    ) -> Result<Register, CompileError> {
        let base = self.fresh(span)?;
        self.code.push(Instruction::move_(base, values[0]));
        for value in &values[1..] {
            let destination = self.fresh(span)?;
            self.code.push(Instruction::move_(destination, *value));
        }
        Ok(base)
    }
    pub(crate) fn pack_or_empty(
        &mut self,
        values: &[Register],
        span: Span,
    ) -> Result<Register, CompileError> {
        if values.is_empty() {
            self.fresh(span)
        } else {
            self.pack(values, span)
        }
    }
    pub(crate) fn vararg(&mut self, span: Span, results: usize) -> Result<Register, CompileError> {
        debug_assert!(results > 0, "expressions always request at least one value");
        self.disable_compatibility_arg();
        let destination = self.fresh(span)?;
        for _ in 1..results {
            self.fresh(span)?;
        }
        let count = u8::try_from(results).map_err(|_| CompileError::RegisterExhausted { span })?;
        self.code.push(Instruction::vararg(destination, count));
        Ok(destination)
    }
}

/// Compact register allocation, used only for a function that exhausts the
/// 255 registers under the normal allocator. Operands go straight to the
/// next free register and their scratch registers are released at once, as
/// in Lua's own compiler, so long concatenations, deep call nesting and wide
/// assignments fit. Every other function keeps the normal bytecode shapes,
/// which the tracing compiler's call-site recognizers are tuned for.
impl Compiler {
    fn compact_concat(
        &mut self,
        operands: &[&Expr],
        span: Span,
        concat_span: Span,
    ) -> Result<Register, CompileError> {
        let destination = self.fresh(span)?;
        let mut first = None;
        let mut last = destination;
        let mut named_slots = Vec::with_capacity(operands.len());
        for operand in operands {
            let name = self.static_value_name(operand);
            last = self.expression_to_next(operand)?;
            first.get_or_insert(last);
            named_slots.push((last, name));
        }
        let first = first.expect("a binary concatenation has operands");
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(concat_span));
        self.code
            .push(Instruction::concat(destination, first, last));
        for (slot, name) in named_slots {
            self.record_diagnostic_site(pc, slot, name);
        }
        self.release_registers(destination.saturating_add(1));
        Ok(destination)
    }

    fn compact_call(
        &mut self,
        callee: &Expr,
        arguments: &[Expr],
        results: usize,
        span: Span,
    ) -> Result<Register, CompileError> {
        let call_name = self.static_call_name(callee);
        let tail = call_tail(arguments);
        if let ([argument], CallTail::Fixed, 1) = (arguments, tail, results) {
            // CallOne reads its function and argument where they are.
            let callee_value = self.call_operand(callee, arguments, &[])?;
            let argument = self.expression(argument, 1)?;
            let destination = self.fresh(span)?;
            self.record_call_site(call_name, span);
            self.code
                .push(Instruction::call_one(destination, callee_value, argument));
            return Ok(destination);
        }
        let base = self.expression_to_next(callee)?;
        let count = self.place_arguments(base, arguments, tail)?;
        self.emit_placed_call(base, count, tail, results, span, call_name)
    }

    /// Places `arguments` after the registers already filled from `base`
    /// (the function, and for a method call its receiver). Returns how many
    /// fixed arguments, the receiver included, sit in registers.
    fn place_arguments(
        &mut self,
        base: Register,
        arguments: &[Expr],
        tail: CallTail,
    ) -> Result<usize, CompileError> {
        let fixed = match tail {
            CallTail::Fixed => arguments,
            CallTail::Vararg | CallTail::Open => &arguments[..arguments.len() - 1],
        };
        for argument in fixed {
            self.expression_to_next(argument)?;
        }
        match tail {
            CallTail::Fixed => {}
            CallTail::Vararg => self.disable_compatibility_arg(),
            CallTail::Open => {
                // The inner call leaves its results in the frame's open
                // result list, not in registers.
                let mark = self.next_register;
                self.open_call_expression(arguments.last().expect("an open call argument"))?;
                self.release_registers(mark);
            }
        }
        Ok(usize::from(self.next_register - base - 1))
    }

    fn method_lookup(&mut self, method: &Name) -> Result<MethodLookup, CompileError> {
        let text = &*method.text;
        if let Some(field) = self.field_name_index_str(text, method.span)? {
            return Ok(MethodLookup::Field(field));
        }
        Ok(MethodLookup::Key(
            self.string_constant_register(text, method.span)?,
        ))
    }

    /// Like Lua's SELF: looks the method up into a fresh `base` and copies
    /// the receiver, as evaluated now, to the first argument register.
    fn place_method(
        &mut self,
        receiver: &Expr,
        method: &Name,
        span: Span,
    ) -> Result<Register, CompileError> {
        let receiver_name = self.static_value_name(receiver);
        let base = self.fresh(span)?;
        let self_slot = self.fresh(span)?;
        let mark = self.next_register;
        let mut receiver_value = self.expression(receiver, 1)?;
        if self.closure_may_reassign(receiver) {
            // Copy before the lookup, whose __index could reassign it.
            self.code
                .push(Instruction::move_(self_slot, receiver_value));
            receiver_value = self_slot;
        }
        let lookup = self.method_lookup(method)?;
        let pc = self.code.len();
        self.instruction_lines
            .insert(pc, self.line_for(method.span));
        match lookup {
            // One instruction copies the receiver and reads the method.
            MethodLookup::Field(field) if receiver_value != self_slot => {
                self.code
                    .push(Instruction::self_field(base, receiver_value, field));
            }
            _ => {
                self.code.push(lookup.instruction(base, receiver_value));
                if receiver_value != self_slot {
                    self.code
                        .push(Instruction::move_(self_slot, receiver_value));
                }
            }
        }
        self.record_diagnostic_site(pc, receiver_value, receiver_name);
        self.release_registers(mark);
        Ok(base)
    }
}

/// How a method call reads its method: by a field constant, or with the
/// name loaded into a register when the constant index is too large.
#[derive(Clone, Copy)]
enum MethodLookup {
    Field(u8),
    Key(Register),
}

impl MethodLookup {
    const fn instruction(self, destination: Register, receiver: Register) -> Instruction {
        match self {
            Self::Field(field) => Instruction::get_field(destination, receiver, field),
            Self::Key(key) => Instruction::get_table(destination, receiver, key),
        }
    }
}

/// How an argument list ends: `f(a, ...)` forwards varargs and `f(a, g())`
/// forwards every result of `g`.
fn call_tail(arguments: &[Expr]) -> CallTail {
    match arguments.last() {
        Some(Expr::Vararg(_)) => CallTail::Vararg,
        Some(last) if expands_open_call_results(last) => CallTail::Open,
        _ => CallTail::Fixed,
    }
}

/// Lua's `luaV_shiftl`: a negative count shifts right, and a count of 64
/// or more clears the value.
fn shift_left(value: i64, count: i64) -> i64 {
    if count <= -64 || count >= 64 {
        0
    } else if count < 0 {
        ((value as u64) >> count.unsigned_abs()) as i64
    } else {
        ((value as u64) << count) as i64
    }
}

impl Compiler {
    /// The value of an expression Lua 5.4 folds to an integer constant:
    /// integer literals and `<const>` locals bound to such a constant,
    /// combined by the operators `luaO_rawarith` folds on integers. Division
    /// and exponentiation give floats, and `//` and `%` by zero stay
    /// run-time errors, so none of those fold to an integer.
    pub(crate) fn integer_constant(&self, expression: &Expr) -> Option<i64> {
        match expression {
            Expr::Integer(value, _) => Some(*value),
            Expr::Name(name) => self.constant_local(&name.text),
            Expr::Parenthesized { expression, .. } => self.integer_constant(expression),
            Expr::Unary { op, expression, .. } => {
                let value = self.integer_constant(expression)?;
                match op {
                    UnaryOp::Negate => Some(value.wrapping_neg()),
                    UnaryOp::BitNot => Some(!value),
                    UnaryOp::Not | UnaryOp::Length => None,
                }
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let (left, right) = (self.integer_constant(left)?, self.integer_constant(right)?);
                match op {
                    BinaryOp::Add => Some(left.wrapping_add(right)),
                    BinaryOp::Subtract => Some(left.wrapping_sub(right)),
                    BinaryOp::Multiply => Some(left.wrapping_mul(right)),
                    BinaryOp::FloorDivide if right != 0 => Some(floor_div(left, right)),
                    BinaryOp::Modulo if right != 0 => Some(floor_mod(left, right)),
                    BinaryOp::BitAnd => Some(left & right),
                    BinaryOp::BitOr => Some(left | right),
                    BinaryOp::BitXor => Some(left ^ right),
                    BinaryOp::ShiftLeft => Some(shift_left(left, right)),
                    BinaryOp::ShiftRight => Some(shift_left(left, right.wrapping_neg())),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Lua's integer `//`, which rounds toward negative infinity.
fn floor_div(left: i64, right: i64) -> i64 {
    if right == -1 {
        return left.wrapping_neg();
    }
    let quotient = left / right;
    if left % right != 0 && (left ^ right) < 0 {
        quotient - 1
    } else {
        quotient
    }
}

/// Lua's integer `%`, whose result takes the sign of the divisor.
fn floor_mod(left: i64, right: i64) -> i64 {
    if right == -1 {
        return 0;
    }
    let remainder = left % right;
    if remainder != 0 && (remainder ^ right) < 0 {
        remainder + right
    } else {
        remainder
    }
}

/// The sizes `NewTable` presizes from: the positional items a constructor
/// stores one by one, and its keyed items, each capped at what a byte holds.
/// A trailing item that expands to many values is not counted.
fn constructor_size_hints(fields: &[TableField]) -> (u8, u8) {
    let mut array = 0_usize;
    let mut hash = 0_usize;
    for (index, field) in fields.iter().enumerate() {
        match field {
            TableField::List(value) => {
                let open = index + 1 == fields.len()
                    && (matches!(value, Expr::Vararg(_)) || expands_open_call_results(value));
                array += usize::from(!open);
            }
            TableField::Name { .. } | TableField::Key { .. } => hash += 1,
        }
    }
    (
        u8::try_from(array).unwrap_or(u8::MAX),
        u8::try_from(hash).unwrap_or(u8::MAX),
    )
}
