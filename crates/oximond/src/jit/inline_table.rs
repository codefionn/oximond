//! Runtime execution for shared inline table-call plans.

use crate::bytecode::{Constant, OpCode};
use crate::compat::{CompatibilityProfile, FloatArith};
use crate::jit_ir::{InlineTableCallPlan, InlineTableOperation, MAX_INLINE_TABLE_REGISTERS};
use crate::runtime::{Runtime, TableHandle, Value};

#[derive(Clone)]
pub struct PreparedInlineTableCallPlan {
    plan: InlineTableCallPlan,
    constants: Vec<Value>,
    fast_field_update: Option<FastFieldUpdate>,
}

impl PreparedInlineTableCallPlan {
    #[must_use]
    pub fn new(plan: InlineTableCallPlan) -> Self {
        let constants: Vec<Value> = plan
            .operations()
            .iter()
            .map(|operation| match operation {
                InlineTableOperation::Load { value, .. } => {
                    constant_value(value, plan.compatibility())
                }
                _ => Value::Nil,
            })
            .collect();
        let fast_field_update = FastFieldUpdate::build(&plan, &constants);
        Self {
            plan,
            constants,
            fast_field_update,
        }
    }
    #[must_use]
    pub const fn plan(&self) -> &InlineTableCallPlan {
        &self.plan
    }
}

pub fn evaluate_inline_table_call(
    prepared: &PreparedInlineTableCallPlan,
    runtime: &mut Runtime,
    arguments: &[Value],
) -> Option<Value> {
    evaluate(prepared, runtime, arguments, false)
}

pub fn evaluate_inline_table_call_number(
    prepared: &PreparedInlineTableCallPlan,
    runtime: &mut Runtime,
    arguments: &[Value],
) -> Option<f64> {
    match evaluate(prepared, runtime, arguments, true)? {
        Value::Number(value) => Some(value),
        _ => None,
    }
}

fn evaluate(
    prepared: &PreparedInlineTableCallPlan,
    runtime: &mut Runtime,
    arguments: &[Value],
    number_only: bool,
) -> Option<Value> {
    if let Some(descriptor) = &prepared.fast_field_update {
        // A descriptor only accepts operations which are effect-free until its
        // final raw store. A failed runtime guard can therefore go directly to
        // the VM without replaying this plan through the generic evaluator.
        return descriptor.evaluate(prepared, runtime, arguments, number_only);
    }
    evaluate_generic(prepared, runtime, arguments, number_only)
}

fn evaluate_generic(
    prepared: &PreparedInlineTableCallPlan,
    runtime: &mut Runtime,
    arguments: &[Value],
    number_only: bool,
) -> Option<Value> {
    if arguments.len() != usize::from(prepared.plan.parameter_count()) {
        return None;
    }
    let mut registers: [Value; MAX_INLINE_TABLE_REGISTERS] = std::array::from_fn(|_| Value::Nil);
    registers[..arguments.len()].clone_from_slice(arguments);
    let mut staged: Option<(TableHandle, Value, Value)> = None;
    let mut pc = 0;
    while let Some(operation) = prepared.plan.operations().get(pc) {
        let index = pc;
        pc += 1;
        match operation {
            InlineTableOperation::Load { destination, .. } => {
                registers[usize::from(*destination)] = prepared.constants[index].clone();
            }
            InlineTableOperation::Move {
                destination,
                source,
            } => registers[usize::from(*destination)] = registers[usize::from(*source)].clone(),
            InlineTableOperation::ClearRange { start, count } => registers
                [usize::from(*start)..usize::from(*start) + usize::from(*count)]
                .fill(Value::Nil),
            InlineTableOperation::GetTable {
                destination,
                table,
                key,
            } => {
                let Value::Table(table) = registers[usize::from(*table)] else {
                    return None;
                };
                let key = &registers[usize::from(*key)];
                let value = if let Some((staged_table, staged_key, staged_value)) = &staged
                    && table == *staged_table
                    && Runtime::raw_equal(key, staged_key)
                {
                    staged_value.clone()
                } else {
                    runtime.table_get_if_no_index(table, key).ok()??
                };
                registers[usize::from(*destination)] = value;
            }
            InlineTableOperation::SetTable { table, key, value } => {
                let Value::Table(table) = registers[usize::from(*table)] else {
                    return None;
                };
                let key = &registers[usize::from(*key)];
                let value = &registers[usize::from(*value)];
                // Updating __gc can perform fallible finalizer bookkeeping
                // after the raw store. Keep that operation in the VM, where
                // an error cannot be mistaken for an effect-free refusal.
                if invalid_raw_write_key(key) || matches!(value, Value::Nil) {
                    return None;
                }
                if let Some((staged_table, staged_key, staged_value)) = &mut staged {
                    // Every write must replace the same existing slot. Leave
                    // the table untouched if an alias now names another slot.
                    if table != *staged_table || !Runtime::raw_equal(key, staged_key) {
                        return None;
                    }
                    *staged_value = value.clone();
                } else {
                    if matches!(runtime.table_get_if_no_index(table, key).ok()??, Value::Nil) {
                        return None;
                    }
                    staged = Some((table, key.clone(), value.clone()));
                }
            }
            InlineTableOperation::Numeric {
                opcode,
                destination,
                left,
                right,
            } => {
                let mut out = Value::Nil;
                if !numeric_into(
                    *opcode,
                    &registers[usize::from(*left)],
                    &registers[usize::from(*right)],
                    &mut out,
                    prepared.plan.compatibility().float_arith(),
                ) {
                    return None;
                }
                registers[usize::from(*destination)] = out;
            }
            InlineTableOperation::Jump { target } => pc = *target,
            InlineTableOperation::JumpIfFalse { condition, target } => {
                if !registers[usize::from(*condition)].is_truthy() {
                    pc = *target;
                }
            }
            InlineTableOperation::Return { source } => {
                let result = registers[usize::from(*source)].clone();
                if number_only && !matches!(result, Value::Number(_)) {
                    return None;
                }
                let (table, key, value) = staged.take()?;
                if !runtime.table_set_if_no_newindex(table, &key, value).ok()? {
                    return None;
                }
                return Some(result);
            }
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operand {
    Argument(u8),
    Constant(usize),
}

impl Operand {
    fn resolve<'a>(
        self,
        prepared: &'a PreparedInlineTableCallPlan,
        arguments: &'a [Value],
    ) -> Option<&'a Value> {
        match self {
            Self::Argument(register) => arguments.get(usize::from(register)),
            Self::Constant(index) => prepared.constants.get(index),
        }
    }
}

#[derive(Clone, Copy)]
enum Symbol {
    Unknown,
    Operand(Operand),
    PreStoreField,
    Computed,
    PostStoreField,
}

#[derive(Clone, Copy)]
struct FastFieldUpdate {
    table: Operand,
    key: Operand,
    other: Operand,
    field_on_left: bool,
    opcode: OpCode,
}

impl FastFieldUpdate {
    fn build(plan: &InlineTableCallPlan, constants: &[Value]) -> Option<Self> {
        let mut registers = symbolic_registers(plan)?;

        let mut field = None;
        let mut numeric = None;
        let mut stored = false;
        let mut post_store_read = false;
        for (index, operation) in plan.operations().iter().enumerate() {
            match *operation {
                InlineTableOperation::Load { destination, .. } => {
                    registers[usize::from(destination)] = Symbol::Operand(Operand::Constant(index));
                }
                InlineTableOperation::Move {
                    destination,
                    source,
                } => registers[usize::from(destination)] = registers[usize::from(source)],
                InlineTableOperation::ClearRange { start, count } => {
                    registers[usize::from(start)..usize::from(start) + usize::from(count)]
                        .fill(Symbol::Unknown);
                }
                InlineTableOperation::GetTable {
                    destination,
                    table,
                    key,
                } => {
                    let Symbol::Operand(table) = registers[usize::from(table)] else {
                        return None;
                    };
                    let Symbol::Operand(key) = registers[usize::from(key)] else {
                        return None;
                    };
                    if !stored && field.is_none() {
                        field = Some((table, key));
                        registers[usize::from(destination)] = Symbol::PreStoreField;
                    } else if stored && !post_store_read {
                        let (field_table, field_key) = field?;
                        if table != field_table || !same_raw_key(key, field_key, constants) {
                            return None;
                        }
                        post_store_read = true;
                        registers[usize::from(destination)] = Symbol::PostStoreField;
                    } else {
                        return None;
                    }
                }
                InlineTableOperation::Numeric {
                    opcode,
                    destination,
                    left,
                    right,
                } => {
                    if stored || numeric.is_some() {
                        return None;
                    }
                    let (other, field_on_left) =
                        match (registers[usize::from(left)], registers[usize::from(right)]) {
                            (Symbol::PreStoreField, Symbol::Operand(other)) => (other, true),
                            (Symbol::Operand(other), Symbol::PreStoreField) => (other, false),
                            _ => return None,
                        };
                    numeric = Some((opcode, other, field_on_left));
                    registers[usize::from(destination)] = Symbol::Computed;
                }
                InlineTableOperation::SetTable { table, key, value } => {
                    let (field_table, field_key) = field?;
                    if stored
                        || !matches!(registers[usize::from(table)], Symbol::Operand(value) if value == field_table)
                        || !matches!(registers[usize::from(key)], Symbol::Operand(value) if same_raw_key(value, field_key, constants))
                        || !matches!(registers[usize::from(value)], Symbol::Computed)
                        || numeric.is_none()
                    {
                        return None;
                    }
                    stored = true;
                }
                InlineTableOperation::Jump { .. } | InlineTableOperation::JumpIfFalse { .. } => {
                    return None;
                }
                InlineTableOperation::Return { source } => {
                    if index + 1 != plan.operations().len() || !stored {
                        return None;
                    }
                    match registers[usize::from(source)] {
                        Symbol::Computed => {}
                        Symbol::PostStoreField if post_store_read => {}
                        _ => return None,
                    }
                    let (table, key) = field?;
                    let (opcode, other, field_on_left) = numeric?;
                    return Some(Self {
                        table,
                        key,
                        other,
                        field_on_left,
                        opcode,
                    });
                }
            }
        }
        None
    }

    fn evaluate(
        self,
        prepared: &PreparedInlineTableCallPlan,
        runtime: &mut Runtime,
        arguments: &[Value],
        number_only: bool,
    ) -> Option<Value> {
        if arguments.len() != usize::from(prepared.plan.parameter_count()) {
            return None;
        }
        let Value::Table(table) = self.table.resolve(prepared, arguments)? else {
            return None;
        };
        let key = self.key.resolve(prepared, arguments)?;
        if invalid_raw_write_key(key) {
            return None;
        }
        // The raw hit borrows the stored value and returns nil, missing and
        // invalid-key reads as `None`, so no `Result<Option<Value>>` goes
        // through memory. A nil field never satisfies the descriptor.
        let field = runtime.table_get_raw_hit(*table, key)?;
        let other = self.other.resolve(prepared, arguments)?;
        let mut result = Value::Nil;
        let computed = if self.field_on_left {
            numeric_into(
                self.opcode,
                field,
                other,
                &mut result,
                prepared.plan.compatibility().float_arith(),
            )
        } else {
            numeric_into(
                self.opcode,
                other,
                field,
                &mut result,
                prepared.plan.compatibility().float_arith(),
            )
        };
        if !computed {
            return None;
        }
        if number_only && !matches!(result, Value::Number(_)) {
            return None;
        }
        // `field` was non-nil and this is the only effect in the descriptor,
        // so this cannot allocate a new slot or delete one after the guards.
        if !runtime
            .table_set_if_no_newindex(*table, key, result.clone())
            .ok()?
        {
            return None;
        }
        Some(result)
    }
}

fn invalid_raw_write_key(key: &Value) -> bool {
    matches!(key, Value::String(name) if name.as_bytes() == b"__gc")
        || matches!(key, Value::Nil)
        || matches!(key, Value::Number(number) if number.is_nan())
}

fn symbolic_registers(plan: &InlineTableCallPlan) -> Option<[Symbol; MAX_INLINE_TABLE_REGISTERS]> {
    let mut registers = [Symbol::Unknown; MAX_INLINE_TABLE_REGISTERS];
    for (register, slot) in registers
        .iter_mut()
        .take(usize::from(plan.parameter_count()))
        .enumerate()
    {
        *slot = Symbol::Operand(Operand::Argument(u8::try_from(register).ok()?));
    }
    Some(registers)
}

fn same_raw_key(left: Operand, right: Operand, constants: &[Value]) -> bool {
    left == right
        || matches!(
            (left, right),
            (Operand::Constant(left), Operand::Constant(right))
                if Runtime::raw_equal(&constants[left], &constants[right])
        )
}

fn constant_value(value: &Constant, profile: CompatibilityProfile) -> Value {
    match value {
        Constant::Nil => Value::Nil,
        Constant::Boolean(value) => Value::Boolean(*value),
        Constant::Integer(value)
            if matches!(
                profile,
                CompatibilityProfile::Lua53
                    | CompatibilityProfile::Lua54
                    | CompatibilityProfile::Lua55
            ) =>
        {
            Value::Integer(*value)
        }
        // Reached only by profiles without an integer subtype: the arm above
        // keeps `Value::Integer` for lua53, lua54 and lua55, so a plan never
        // turns an integer constant into a float where Lua has integers.
        Constant::Integer(value) => Value::Number(integer_to_float(*value)),
        Constant::Number(value) => Value::Number(*value),
        Constant::String(value) => Value::from(value.clone()),
        Constant::Int64(value) => Value::Int64(*value),
        Constant::UInt64(value) => Value::UInt64(*value),
    }
}

/// Writes the arithmetic result into `out` and reports whether the operands
/// and opcode were supported, so the hot path returns one byte.
#[inline]
fn numeric_into(
    opcode: OpCode,
    left: &Value,
    right: &Value,
    out: &mut Value,
    float_arith: FloatArith,
) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            numeric_float_into(opcode, *left, *right, out, float_arith)
        }
        (Value::Integer(left), Value::Integer(right)) => {
            *out = match opcode {
                OpCode::Add => Value::Integer(left.wrapping_add(*right)),
                OpCode::Subtract => Value::Integer(left.wrapping_sub(*right)),
                OpCode::Multiply => Value::Integer(left.wrapping_mul(*right)),
                OpCode::Divide => Value::Number(integer_to_float(*left) / integer_to_float(*right)),
                OpCode::FloorDivide => match floor_div(*left, *right) {
                    Some(value) => Value::Integer(value),
                    None => return false,
                },
                OpCode::Modulo => match floor_mod(*left, *right) {
                    Some(value) => Value::Integer(value),
                    None => return false,
                },
                OpCode::Power => Value::Number(
                    float_arith.power(integer_to_float(*left), integer_to_float(*right)),
                ),
                _ => return false,
            };
            true
        }
        (Value::Integer(left), Value::Number(right)) => {
            numeric_float_into(opcode, integer_to_float(*left), *right, out, float_arith)
        }
        (Value::Number(left), Value::Integer(right)) => {
            numeric_float_into(opcode, *left, integer_to_float(*right), out, float_arith)
        }
        _ => false,
    }
}
#[inline]
fn numeric_float_into(
    opcode: OpCode,
    left: f64,
    right: f64,
    out: &mut Value,
    float_arith: FloatArith,
) -> bool {
    *out = Value::Number(match opcode {
        OpCode::Add => left + right,
        OpCode::Subtract => left - right,
        OpCode::Multiply => left * right,
        OpCode::Divide => left / right,
        OpCode::FloorDivide => (left / right).floor(),
        OpCode::Modulo => float_arith.modulo(left, right),
        OpCode::Power => float_arith.power(left, right),
        _ => return false,
    });
    true
}
fn floor_div(left: i64, right: i64) -> Option<i64> {
    if right == 0 {
        return None;
    }
    if left == i64::MIN && right == -1 {
        return Some(i64::MIN);
    }
    let quotient = left / right;
    let remainder = left % right;
    Some(if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient - 1
    } else {
        quotient
    })
}
fn floor_mod(left: i64, right: i64) -> Option<i64> {
    if right == 0 {
        return None;
    }
    if left == i64::MIN && right == -1 {
        return Some(0);
    }
    let remainder = left % right;
    Some(if remainder != 0 && (remainder < 0) != (right < 0) {
        remainder + right
    } else {
        remainder
    })
}

// Lua converts integers to floating point for mixed arithmetic and division.
#[allow(clippy::cast_precision_loss)]
fn integer_to_float(value: i64) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{
        PreparedInlineTableCallPlan, evaluate_generic, evaluate_inline_table_call,
        evaluate_inline_table_call_number,
    };
    use crate::bytecode::{Chunk, Constant, Instruction, OpCode};
    use crate::compat::CompatibilityProfile;
    use crate::jit_ir::validate_inline_table_call_plan;
    use crate::runtime::{Runtime, Value};

    fn field_update(opcode: OpCode, aliases: bool, second_numeric: bool) -> Chunk {
        let mut code = vec![
            Instruction::load_constant(2, 0),
            Instruction::get_table(3, 0, 2),
        ];
        if aliases {
            code.push(Instruction::move_(5, 3));
            code.push(Instruction::binary(opcode, 4, 5, 1));
        } else {
            code.push(Instruction::binary(opcode, 4, 3, 1));
        }
        if second_numeric {
            code.push(Instruction::binary(OpCode::Add, 4, 4, 1));
        }
        code.extend([
            Instruction::set_table(0, 2, 4),
            Instruction::get_table(5, 0, 2),
        ]);
        if aliases {
            code.push(Instruction::move_(4, 5));
            code.push(Instruction::return_(4));
        } else {
            code.push(Instruction::return_(5));
        }
        Chunk::with_parts(
            6,
            2,
            vec![Constant::String("value".into())],
            code,
            Vec::new(),
            Vec::new(),
        )
    }

    fn prepared_update(
        opcode: OpCode,
        aliases: bool,
        second_numeric: bool,
    ) -> PreparedInlineTableCallPlan {
        let chunk = field_update(opcode, aliases, second_numeric);
        PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        )
    }

    #[test]
    fn finalizer_field_replacement_refuses_before_the_store() {
        let chunk = Chunk::with_parts(
            3,
            3,
            Vec::new(),
            vec![
                Instruction::abc(OpCode::SetTable, 0, 1, 2),
                Instruction::return_(2),
            ],
            Vec::new(),
            Vec::new(),
        );
        let plan = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("__gc");
        runtime.table_set(table, &key, Value::Number(1.0)).unwrap();
        assert_eq!(
            evaluate_inline_table_call_number(
                &plan,
                &mut runtime,
                &[Value::Table(table), key.clone(), Value::Number(2.0)]
            ),
            None
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Number(1.0));
    }

    #[test]
    fn field_update_descriptor_tracks_aliases_and_post_store_read() {
        let prepared = prepared_update(OpCode::Add, true, false);
        assert!(prepared.fast_field_update.is_some());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let metatable = runtime.create_table().unwrap();
        runtime
            .raw_set_metatable(&Value::Table(table), Some(metatable))
            .unwrap();
        let key = Value::from("value");
        runtime.table_set(table, &key, Value::Integer(4)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(3)],
            ),
            Some(Value::Integer(7))
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(7));
    }

    #[test]
    fn compiler_style_repeated_key_loads_select_the_fast_descriptor() {
        let chunk = Chunk::with_parts(
            8,
            2,
            vec![Constant::String("value".into())],
            vec![
                Instruction::move_(2, 0),
                Instruction::load_constant(3, 0),
                Instruction::move_(4, 3),
                Instruction::load_constant(5, 0),
                Instruction::get_table(6, 0, 5),
                Instruction::binary(OpCode::Add, 7, 6, 1),
                Instruction::set_table(2, 4, 7),
                Instruction::clear_range(2, 6),
                Instruction::load_constant(2, 0),
                Instruction::get_table(3, 0, 2),
                Instruction::return_(3),
                Instruction::halt(),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        assert!(prepared.fast_field_update.is_some());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("value");
        runtime.table_set(table, &key, Value::Integer(1)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(2)],
            ),
            Some(Value::Integer(3))
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(3));
    }

    #[test]
    fn field_update_fast_path_matches_generic_numeric_rules() {
        for (opcode, current, delta) in [
            (OpCode::Add, Value::Integer(i64::MAX), Value::Integer(1)),
            (OpCode::Subtract, Value::Number(-3.5), Value::Number(2.0)),
            (OpCode::FloorDivide, Value::Integer(-7), Value::Integer(3)),
            (OpCode::Modulo, Value::Number(-7.5), Value::Number(3.0)),
        ] {
            let prepared = prepared_update(opcode, false, false);
            assert!(prepared.fast_field_update.is_some());
            let mut fast_runtime = Runtime::new();
            let mut generic_runtime = Runtime::new();
            let fast_table = fast_runtime.create_table().unwrap();
            let generic_table = generic_runtime.create_table().unwrap();
            let key = Value::from("value");
            fast_runtime
                .table_set(fast_table, &key, current.clone())
                .unwrap();
            generic_runtime
                .table_set(generic_table, &key, current)
                .unwrap();
            let fast_arguments = [Value::Table(fast_table), delta.clone()];
            let generic_arguments = [Value::Table(generic_table), delta];
            assert_eq!(
                evaluate_inline_table_call(&prepared, &mut fast_runtime, &fast_arguments),
                evaluate_generic(&prepared, &mut generic_runtime, &generic_arguments, false),
            );
            assert_eq!(
                fast_runtime.table_get(fast_table, &key).unwrap(),
                generic_runtime.table_get(generic_table, &key).unwrap(),
            );
        }
    }

    #[test]
    fn field_update_guards_refuse_before_writing() {
        let prepared = prepared_update(OpCode::Add, false, false);
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("value");
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(1)],
            ),
            None
        );
        runtime
            .table_set(table, &key, Value::from("not a number"))
            .unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(1)],
            ),
            None
        );
        assert_eq!(
            runtime.table_get(table, &key).unwrap(),
            Value::from("not a number")
        );
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Number(1.0), Value::Integer(1)],
            ),
            None
        );
        let floor = prepared_update(OpCode::FloorDivide, false, false);
        runtime.table_set(table, &key, Value::Integer(9)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &floor,
                &mut runtime,
                &[Value::Table(table), Value::Integer(0)],
            ),
            None
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(9));
    }

    #[test]
    fn field_update_refuses_finalizer_key_before_the_store() {
        let mut chunk = field_update(OpCode::Add, false, false);
        chunk.constants[0] = Constant::String("__gc".into());
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        assert!(prepared.fast_field_update.is_some());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("__gc");
        runtime.table_set(table, &key, Value::Integer(1)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(1)],
            ),
            None
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(1));
    }

    #[test]
    fn field_update_refuses_nil_and_nan_argument_keys_before_the_store() {
        let chunk = Chunk::with_parts(
            6,
            3,
            Vec::new(),
            vec![
                Instruction::get_table(3, 0, 2),
                Instruction::binary(OpCode::Add, 4, 3, 1),
                Instruction::set_table(0, 2, 4),
                Instruction::get_table(5, 0, 2),
                Instruction::return_(5),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        assert!(prepared.fast_field_update.is_some());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        for key in [Value::Nil, Value::Number(f64::NAN)] {
            assert_eq!(
                evaluate_inline_table_call(
                    &prepared,
                    &mut runtime,
                    &[Value::Table(table), Value::Integer(1), key],
                ),
                None
            );
        }
    }

    #[test]
    fn two_numeric_operations_stay_on_the_generic_path() {
        let prepared = prepared_update(OpCode::Add, false, true);
        assert!(prepared.fast_field_update.is_none());
    }

    #[test]
    fn repeated_writes_read_the_staged_value_and_commit_the_last_one() {
        let chunk = Chunk::with_parts(
            9,
            2,
            vec![Constant::String("value".into())],
            vec![
                Instruction::load_constant(2, 0),
                Instruction::get_table(3, 0, 2),
                Instruction::binary(OpCode::Add, 4, 3, 1),
                Instruction::set_table(0, 2, 4),
                Instruction::get_table(5, 0, 2),
                Instruction::binary(OpCode::Add, 6, 5, 1),
                Instruction::set_table(0, 2, 6),
                Instruction::get_table(7, 0, 2),
                Instruction::return_(7),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        assert!(prepared.fast_field_update.is_none());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("value");
        runtime.table_set(table, &key, Value::Integer(2)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(3)],
            ),
            Some(Value::Integer(8))
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(8));
    }

    #[test]
    fn repeated_writes_accept_integer_and_number_keys_for_the_same_slot() {
        let chunk = Chunk::with_parts(
            6,
            5,
            Vec::new(),
            vec![
                Instruction::set_table(0, 1, 3),
                Instruction::set_table(0, 2, 4),
                Instruction::get_table(5, 0, 2),
                Instruction::return_(5),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua53).unwrap(),
        );
        assert!(prepared.fast_field_update.is_none());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        runtime
            .table_set(table, &Value::Integer(1), Value::Integer(3))
            .unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[
                    Value::Table(table),
                    Value::Integer(1),
                    Value::Number(1.0),
                    Value::Integer(5),
                    Value::Integer(8),
                ],
            ),
            Some(Value::Integer(8))
        );
        assert_eq!(
            runtime.table_get(table, &Value::Integer(1)).unwrap(),
            Value::Integer(8)
        );
    }

    #[test]
    fn second_write_to_another_key_or_table_refuses_before_mutation() {
        for second_table in [false, true] {
            let chunk = Chunk::with_parts(
                5,
                3,
                vec![Constant::String("value".into())],
                vec![
                    Instruction::load_constant(3, 0),
                    Instruction::set_table(0, 3, 1),
                    Instruction::set_table(
                        if second_table { 2 } else { 0 },
                        if second_table { 3 } else { 2 },
                        1,
                    ),
                    Instruction::return_(1),
                ],
                Vec::new(),
                Vec::new(),
            );
            let prepared = PreparedInlineTableCallPlan::new(
                validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
            );
            let mut runtime = Runtime::new();
            let first = runtime.create_table().unwrap();
            let second = runtime.create_table().unwrap();
            let key = Value::from("value");
            runtime.table_set(first, &key, Value::Integer(1)).unwrap();
            runtime.table_set(second, &key, Value::Integer(9)).unwrap();
            let other = if second_table {
                Value::Table(second)
            } else {
                Value::from("other")
            };
            assert_eq!(
                evaluate_inline_table_call(
                    &prepared,
                    &mut runtime,
                    &[Value::Table(first), Value::Integer(2), other],
                ),
                None
            );
            assert_eq!(runtime.table_get(first, &key).unwrap(), Value::Integer(1));
            assert_eq!(runtime.table_get(second, &key).unwrap(), Value::Integer(9));
        }
    }

    #[test]
    fn nil_second_write_does_not_commit_the_first_write() {
        let chunk = Chunk::with_parts(
            4,
            2,
            vec![Constant::String("value".into())],
            vec![
                Instruction::load_constant(2, 0),
                Instruction::set_table(0, 2, 1),
                Instruction::load_nil(3),
                Instruction::set_table(0, 2, 3),
                Instruction::return_(1),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let key = Value::from("value");
        runtime.table_set(table, &key, Value::Integer(1)).unwrap();
        assert_eq!(
            evaluate_inline_table_call(
                &prepared,
                &mut runtime,
                &[Value::Table(table), Value::Integer(2)],
            ),
            None
        );
        assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(1));
    }

    #[test]
    fn forward_branch_commits_only_the_selected_single_slot() {
        let chunk = Chunk::with_parts(
            5,
            5,
            Vec::new(),
            vec![
                Instruction::set_table(0, 1, 3),
                Instruction::jump_if_false(4, 1),
                Instruction::set_table(0, 2, 3),
                Instruction::return_(3),
            ],
            Vec::new(),
            Vec::new(),
        );
        let prepared = PreparedInlineTableCallPlan::new(
            validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap(),
        );
        assert!(prepared.fast_field_update.is_none());
        let mut runtime = Runtime::new();
        let table = runtime.create_table().unwrap();
        let first = Value::from("first");
        let second = Value::from("second");
        runtime.table_set(table, &first, Value::Integer(1)).unwrap();
        runtime
            .table_set(table, &second, Value::Integer(9))
            .unwrap();
        let arguments = |condition| {
            [
                Value::Table(table),
                first.clone(),
                second.clone(),
                Value::Integer(2),
                condition,
            ]
        };
        assert_eq!(
            evaluate_inline_table_call(&prepared, &mut runtime, &arguments(Value::Boolean(false)),),
            Some(Value::Integer(2))
        );
        assert_eq!(runtime.table_get(table, &first).unwrap(), Value::Integer(2));
        assert_eq!(
            runtime.table_get(table, &second).unwrap(),
            Value::Integer(9)
        );
        assert_eq!(
            evaluate_inline_table_call(&prepared, &mut runtime, &arguments(Value::Boolean(true)),),
            None
        );
        assert_eq!(runtime.table_get(table, &first).unwrap(), Value::Integer(2));
        assert_eq!(
            runtime.table_get(table, &second).unwrap(),
            Value::Integer(9)
        );
    }
}
