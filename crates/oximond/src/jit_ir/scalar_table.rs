//! Bounded scalar replacement for a fresh, nonescaping table.

use crate::bytecode::{Chunk, Constant, OpCode};
use crate::compat::CompatibilityProfile;

const MAX_INSTRUCTIONS: usize = 128;
const MAX_OPERATIONS: usize = 256;

/// One frozen numeric value in a scalar-table plan.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarNumberOperation {
    Input { register: u8 },
    Constant { value: f64 },
    Add { left: usize, right: usize },
    Subtract { left: usize, right: usize },
    Multiply { left: usize, right: usize },
    Negate { source: usize },
}

/// A final register value produced by scalar replacement.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarTableOutput {
    Number { result_index: usize },
    Constant(Constant),
}

/// Immutable straight-line replacement for one fresh table's lifetime.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarTablePlan {
    start_pc: usize,
    end_pc: usize,
    input_registers: Vec<u8>,
    number_operations: Vec<ScalarNumberOperation>,
    outputs: Vec<(u8, ScalarTableOutput)>,
    /// Distinct fields written into the elided table.
    ///
    /// The replacement removes the allocation, but a Lua program can still
    /// observe heap use through `gcinfo`/`collectgarbage("count")`. The VM
    /// charges phantom bytes for this many entries so those observations keep
    /// their pre-elision meaning.
    field_count: usize,
}

impl ScalarTablePlan {
    #[must_use]
    pub const fn start_pc(&self) -> usize {
        self.start_pc
    }
    #[must_use]
    pub const fn end_pc(&self) -> usize {
        self.end_pc
    }
    #[must_use]
    pub fn input_registers(&self) -> &[u8] {
        &self.input_registers
    }
    #[must_use]
    pub fn number_operations(&self) -> &[ScalarNumberOperation] {
        &self.number_operations
    }
    #[must_use]
    pub fn outputs(&self) -> &[(u8, ScalarTableOutput)] {
        &self.outputs
    }
    /// Distinct fields the elided table would have stored.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }
}

#[derive(Clone, Debug)]
enum Value {
    Table,
    Number { index: usize, key: Option<Key> },
    Constant(Constant),
    Unknown(u8),
}

#[derive(Clone, Debug, PartialEq)]
enum Key {
    Number(u64),
    Boolean(bool),
    String(Constant),
}

impl Key {
    // Each arm names a distinct IR constant kind; `Integer` and the final
    // group share the `None` body but stay separate rows to keep the kinds
    // visible.
    #[allow(clippy::match_same_arms)]
    fn from_constant(value: &Constant) -> Option<Self> {
        match value {
            Constant::Integer(_) => None,
            Constant::Number(value) if !value.is_nan() => Some(Self::Number(if *value == 0.0 {
                0.0f64.to_bits()
            } else {
                value.to_bits()
            })),
            Constant::Boolean(value) => Some(Self::Boolean(*value)),
            Constant::String(value) => Some(Self::String(Constant::String(value.clone()))),
            Constant::Nil | Constant::Number(_) | Constant::Int64(_) | Constant::UInt64(_) => None,
        }
    }
}

fn write_value(
    registers: &mut [Value],
    aliases: &mut usize,
    touched: &mut [bool],
    destination: u8,
    value: Value,
) -> Option<()> {
    let slot = registers.get_mut(usize::from(destination))?;
    if matches!(slot, Value::Table) {
        *aliases = aliases.checked_sub(1)?;
    }
    *slot = value;
    *touched.get_mut(usize::from(destination))? = true;
    Some(())
}

/// Validates a fresh-table fragment beginning at `start_pc`.
///
/// `end_limit` is exclusive and must be the next basic-block leader or the
/// end of the chunk. The plan only accepts Lua 5.1 semantics for now.
/// The caller must have validated the enclosing chunk tree. This also accepts
/// nested chunks whose upvalue descriptors refer to that enclosing tree.
#[must_use]
pub fn validate_scalar_table_plan(
    chunk: &Chunk,
    start_pc: usize,
    end_limit: usize,
    compatibility: CompatibilityProfile,
) -> Option<ScalarTablePlan> {
    if compatibility != CompatibilityProfile::Lua51
        || start_pc >= end_limit
        || end_limit > chunk.code.len()
        || chunk.code.get(start_pc)?.opcode().ok()? != OpCode::NewTable
    {
        return None;
    }
    let count = usize::from(chunk.register_count);
    let mut registers = (0..count)
        .map(|register| Value::Unknown(register as u8))
        .collect::<Vec<_>>();
    let mut fields: Vec<(Key, Value)> = Vec::new();
    let mut inputs = Vec::new();
    let mut numbers = Vec::new();
    let mut entry_inputs = vec![None; count];
    let mut touched = vec![false; count];
    let mut pc = start_pc;
    let mut aliases = 0usize;

    let input = |register: u8,
                 entry_inputs: &mut Vec<Option<usize>>,
                 inputs: &mut Vec<u8>,
                 numbers: &mut Vec<ScalarNumberOperation>|
     -> Option<usize> {
        let slot = entry_inputs.get_mut(usize::from(register))?;
        if let Some(index) = *slot {
            return Some(index);
        }
        let index = numbers.len();
        numbers.push(ScalarNumberOperation::Input { register });
        inputs.push(register);
        *slot = Some(index);
        Some(index)
    };
    let bounded_end = end_limit.min(start_pc.saturating_add(MAX_INSTRUCTIONS));
    while pc < bounded_end {
        let instruction = chunk.code[pc];
        let opcode = instruction.opcode().ok()?;
        match opcode {
            OpCode::NewTable if pc == start_pc => {
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    Value::Table,
                )?;
                aliases += 1;
                pc += 1;
            }
            OpCode::LoadNil => {
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    Value::Constant(Constant::Nil),
                )?;
                pc += 1;
            }
            OpCode::LoadConstant | OpCode::LoadConstantWide => {
                let index = if opcode == OpCode::LoadConstant {
                    usize::from(instruction.d())
                } else {
                    usize::try_from(chunk.code.get(pc + 1)?.extra_argument_index()).ok()?
                };
                let constant = chunk.constants.get(index)?.clone();
                let value = match constant {
                    // Keeping integers exact would require an integer lane in
                    // the native plan. Lua 5.1 compiler output uses numbers
                    // here, so refuse direct bytecode with integer constants.
                    // 64-bit integer cdata has no native plan representation.
                    Constant::Integer(_) | Constant::Int64(_) | Constant::UInt64(_) => {
                        return None;
                    }
                    Constant::Number(value) => {
                        let index = numbers.len();
                        numbers.push(ScalarNumberOperation::Constant { value });
                        Value::Number {
                            index,
                            key: Key::from_constant(&Constant::Number(value)),
                        }
                    }
                    value => Value::Constant(value),
                };
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    value,
                )?;
                pc += if opcode == OpCode::LoadConstantWide {
                    2
                } else {
                    1
                };
            }
            OpCode::Move => {
                let value = registers.get(usize::from(instruction.b()))?.clone();
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    value,
                )?;
                if matches!(registers[usize::from(instruction.a())], Value::Table) {
                    aliases += 1;
                }
                pc += 1;
            }
            OpCode::ClearRange => {
                for destination in instruction.a()..instruction.a().checked_add(instruction.b())? {
                    write_value(
                        &mut registers,
                        &mut aliases,
                        &mut touched,
                        destination,
                        Value::Constant(Constant::Nil),
                    )?;
                }
                pc += 1;
            }
            OpCode::SetTable | OpCode::SetField => {
                if !matches!(registers.get(usize::from(instruction.a()))?, Value::Table) {
                    return None;
                }
                let key = if opcode == OpCode::SetField {
                    Key::from_constant(chunk.constants.get(usize::from(instruction.b()))?)?
                } else {
                    match registers.get(usize::from(instruction.b()))? {
                        Value::Constant(value) => Key::from_constant(value)?,
                        Value::Number { key: Some(key), .. } => key.clone(),
                        _ => return None,
                    }
                };
                let value = match registers.get(usize::from(instruction.c()))?.clone() {
                    Value::Unknown(register) => Value::Number {
                        index: input(register, &mut entry_inputs, &mut inputs, &mut numbers)?,
                        key: None,
                    },
                    value => value,
                };
                if matches!(value, Value::Table) {
                    return None;
                }
                if let Some((_, field)) = fields.iter_mut().find(|(old, _)| *old == key) {
                    *field = value;
                } else {
                    fields.push((key, value));
                }
                pc += 1;
            }
            OpCode::GetTable | OpCode::GetField => {
                if !matches!(registers.get(usize::from(instruction.b()))?, Value::Table) {
                    return None;
                }
                let key = if opcode == OpCode::GetField {
                    Key::from_constant(chunk.constants.get(usize::from(instruction.c()))?)?
                } else {
                    match registers.get(usize::from(instruction.c()))? {
                        Value::Constant(value) => Key::from_constant(value)?,
                        Value::Number { key: Some(key), .. } => key.clone(),
                        _ => return None,
                    }
                };
                let value = fields
                    .iter()
                    .find(|(old, _)| *old == key)
                    .map_or(Value::Constant(Constant::Nil), |(_, value)| value.clone());
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    value,
                )?;
                pc += 1;
            }
            OpCode::Negate => {
                let source = match registers.get(usize::from(instruction.b()))?.clone() {
                    Value::Number { index, .. } => index,
                    Value::Unknown(register) => {
                        input(register, &mut entry_inputs, &mut inputs, &mut numbers)?
                    }
                    _ => return None,
                };
                let index = numbers.len();
                numbers.push(ScalarNumberOperation::Negate { source });
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    Value::Number { index, key: None },
                )?;
                pc += 1;
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK => {
                let number =
                    |register: u8,
                     registers: &Vec<Value>,
                     entry_inputs: &mut Vec<Option<usize>>,
                     inputs: &mut Vec<u8>,
                     numbers: &mut Vec<ScalarNumberOperation>| match registers
                        .get(usize::from(register))?
                        .clone()
                    {
                        Value::Number { index, .. } => Some(index),
                        Value::Unknown(source) => input(source, entry_inputs, inputs, numbers),
                        _ => None,
                    };
                let left = number(
                    instruction.b(),
                    &registers,
                    &mut entry_inputs,
                    &mut inputs,
                    &mut numbers,
                )?;
                let right = if opcode.register_form().is_some() {
                    // As `LoadConstant` of a number constant, which does not
                    // keep integer constants exact.
                    let Constant::Number(value) =
                        *chunk.constants.get(usize::from(instruction.c()))?
                    else {
                        return None;
                    };
                    let index = numbers.len();
                    numbers.push(ScalarNumberOperation::Constant { value });
                    index
                } else {
                    number(
                        instruction.c(),
                        &registers,
                        &mut entry_inputs,
                        &mut inputs,
                        &mut numbers,
                    )?
                };
                let operation = match opcode.register_form().unwrap_or(opcode) {
                    OpCode::Add => ScalarNumberOperation::Add { left, right },
                    OpCode::Subtract => ScalarNumberOperation::Subtract { left, right },
                    OpCode::Multiply => ScalarNumberOperation::Multiply { left, right },
                    _ => unreachable!(),
                };
                let index = numbers.len();
                numbers.push(operation);
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    instruction.a(),
                    Value::Number { index, key: None },
                )?;
                pc += 1;
            }
            _ => return None,
        }
        if numbers.len() > MAX_OPERATIONS {
            return None;
        }
        // A register that leaves the live set no longer aliases the table:
        // nothing reads it before writing it again. The compiler does not
        // clear such temporaries, so the plan ends the alias itself and
        // writes nil there, as a clear would have.
        let live = chunk.live_registers(pc);
        for register in 0..registers.len() {
            let Ok(name) = u8::try_from(register) else {
                break;
            };
            if matches!(registers[register], Value::Table) && !live.contains(name) {
                write_value(
                    &mut registers,
                    &mut aliases,
                    &mut touched,
                    name,
                    Value::Constant(Constant::Nil),
                )?;
            }
        }
        if aliases == 0 {
            break;
        }
    }
    if pc == start_pc + 1 || aliases != 0 {
        return None;
    }
    let mut outputs = Vec::new();
    for (register, was_touched) in touched.into_iter().enumerate() {
        if !was_touched {
            continue;
        }
        match registers[register].clone() {
            Value::Number {
                index: result_index,
                ..
            } => outputs.push((register as u8, ScalarTableOutput::Number { result_index })),
            Value::Constant(value) => {
                outputs.push((register as u8, ScalarTableOutput::Constant(value)));
            }
            Value::Table => return None,
            Value::Unknown(source) => outputs.push((
                register as u8,
                ScalarTableOutput::Number {
                    result_index: input(source, &mut entry_inputs, &mut inputs, &mut numbers)?,
                },
            )),
        }
    }
    for (source_pc, instruction) in chunk.code.iter().copied().enumerate() {
        let opcode = instruction.opcode().ok()?;
        let offset = match opcode {
            OpCode::Jump => Some(instruction.jump_offset()),
            OpCode::JumpIfFalse
            | OpCode::NumericForPrep
            | OpCode::NumericForLoop
            | OpCode::ForInLoop => {
                Some(i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())))
            }
            _ => None,
        };
        if let Some(offset) = offset {
            let target = i64::try_from(source_pc)
                .ok()?
                .checked_add(1)?
                .checked_add(i64::from(offset))?;
            let target = usize::try_from(target).ok()?;
            if target > start_pc && target < pc {
                return None;
            }
        }
    }
    Some(ScalarTablePlan {
        start_pc,
        end_pc: pc,
        input_registers: inputs,
        number_operations: numbers,
        outputs,
        field_count: fields.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::{ScalarTableOutput, validate_scalar_table_plan};
    use crate::bytecode::{Chunk, Constant, DebugMetadata, Instruction, LocalVariable, OpCode};
    use crate::compat::CompatibilityProfile;

    /// Plans the fragment with r0 a named local in scope over the whole code,
    /// so only a write ends its alias.
    fn plan(code: Vec<Instruction>, constants: Vec<Constant>) -> Option<super::ScalarTablePlan> {
        let end = code.len();
        let chunk = Chunk::with_parts(3, 0, constants, code, Vec::new(), Vec::new())
            .with_debug_metadata(
                DebugMetadata::new("scalar.lua", Vec::new())
                    .with_local_variables(vec![LocalVariable::new("t", 0, 0, end)]),
            );
        validate_scalar_table_plan(&chunk, 0, end, CompatibilityProfile::Lua51)
    }

    #[test]
    fn a_temporary_table_ends_where_its_register_leaves_the_live_set() {
        // r0 is a temporary: the GetTable is its last read, and no clear
        // follows. The plan ends after the read and writes nil to r0.
        let code = vec![
            Instruction::new_table(0, 0, 0),
            Instruction::ad(OpCode::LoadConstant, 1, 0),
            Instruction::abc(OpCode::SetTable, 0, 1, 1),
            Instruction::abc(OpCode::GetTable, 2, 0, 1),
            Instruction::return_(2),
        ];
        let end = code.len();
        let chunk = Chunk::with_parts(
            3,
            0,
            vec![Constant::Number(1.0)],
            code,
            Vec::new(),
            Vec::new(),
        );
        let plan = validate_scalar_table_plan(&chunk, 0, end, CompatibilityProfile::Lua51)
            .expect("a dead temporary table is scalarizable");
        assert_eq!(plan.end_pc(), 4);
        assert!(
            plan.outputs()
                .contains(&(0, ScalarTableOutput::Constant(Constant::Nil)))
        );
        assert!(matches!(
            plan.outputs().iter().find(|(register, _)| *register == 2),
            Some((_, ScalarTableOutput::Number { .. }))
        ));
    }

    #[test]
    fn moved_entry_values_do_not_follow_later_source_writes() {
        let plan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::move_(1, 2),
                Instruction::load_constant(2, 0),
                Instruction::load_nil(0),
            ],
            vec![Constant::Number(99.0)],
        )
        .unwrap();
        let (_, ScalarTableOutput::Number { result_index }) = plan
            .outputs()
            .iter()
            .find(|(register, _)| *register == 1)
            .unwrap()
        else {
            panic!("the moved input must remain a numeric entry value");
        };
        assert_eq!(
            plan.number_operations()[*result_index],
            super::ScalarNumberOperation::Input { register: 2 }
        );
    }

    #[test]
    fn a_nil_field_write_removes_the_old_value() {
        let plan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::load_constant(1, 0),
                Instruction::set_table(0, 1, 1),
                Instruction::load_nil(2),
                Instruction::set_table(0, 1, 2),
                Instruction::get_table(2, 0, 1),
                Instruction::load_nil(0),
            ],
            vec![Constant::Number(1.0)],
        )
        .unwrap();
        assert_eq!(
            plan.outputs().iter().find(|(register, _)| *register == 2),
            Some(&(2, ScalarTableOutput::Constant(Constant::Nil)))
        );
    }

    #[test]
    fn freezes_an_external_value_before_its_register_is_reused() {
        let plan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::ad(OpCode::LoadConstant, 1, 0),
                Instruction::abc(OpCode::SetTable, 0, 1, 2),
                Instruction::ad(OpCode::LoadConstant, 2, 1),
                Instruction::abc(OpCode::GetTable, 1, 0, 1),
                Instruction::clear_range(0, 1),
                Instruction::abc(OpCode::Halt, 0, 0, 0),
            ],
            vec![Constant::Number(1.0), Constant::Number(99.0)],
        )
        .expect("fresh table fragment is scalarizable");
        assert_eq!(plan.input_registers(), &[2]);
        assert!(matches!(
            plan.outputs().iter().find(|(register, _)| *register == 1),
            Some((_, ScalarTableOutput::Number { .. }))
        ));
    }

    #[test]
    fn rejects_escaping_table_and_nonconstant_key() {
        let escaped = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::abc(OpCode::Move, 1, 0, 0),
                Instruction::abc(OpCode::SetTable, 0, 1, 0),
                Instruction::clear_range(0, 2),
            ],
            Vec::new(),
        );
        assert!(escaped.is_none());
    }

    #[test]
    fn table_aliases_and_field_overwrites_are_tracked_until_every_alias_dies() {
        let plan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::abc(OpCode::Move, 2, 0, 0),
                Instruction::ad(OpCode::LoadConstant, 1, 0),
                Instruction::abc(OpCode::SetTable, 2, 1, 1),
                Instruction::ad(OpCode::LoadConstant, 1, 1),
                Instruction::abc(OpCode::SetTable, 0, 1, 1),
                Instruction::clear_range(0, 1),
                Instruction::clear_range(2, 1),
            ],
            vec![Constant::Number(1.0), Constant::Number(2.0)],
        );
        assert!(plan.is_some());
    }

    #[test]
    fn rejects_calls_branches_bad_keys_integers_and_interior_entries() {
        let call = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::abc(OpCode::Call, 1, 0, 1),
            ],
            Vec::new(),
        );
        assert!(call.is_none());
        let branch = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::jump_if_false(1, 0),
            ],
            Vec::new(),
        );
        assert!(branch.is_none());
        let nil_key = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::abc(OpCode::SetTable, 0, 1, 2),
                Instruction::clear_range(0, 1),
            ],
            Vec::new(),
        );
        assert!(nil_key.is_none());
        let integer = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::ad(OpCode::LoadConstant, 1, 0),
            ],
            vec![Constant::Integer(1)],
        );
        assert!(integer.is_none());
        let interior = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::clear_range(0, 1),
                Instruction::jump_if_false(1, -2),
            ],
            Vec::new(),
        );
        assert!(interior.is_none());
    }

    #[test]
    fn rejects_nan_keys_and_fragments_past_the_instruction_bound() {
        let nan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::ad(OpCode::LoadConstant, 1, 0),
                Instruction::abc(OpCode::SetTable, 0, 1, 2),
                Instruction::clear_range(0, 1),
            ],
            vec![Constant::Number(f64::NAN)],
        );
        assert!(nan.is_none());
        let mut long = vec![Instruction::new_table(0, 0, 0)];
        long.extend((0..128).map(|_| Instruction::abc(OpCode::Move, 1, 1, 0)));
        long.push(Instruction::clear_range(0, 1));
        assert!(plan(long, Vec::new()).is_none());
    }

    #[test]
    fn missing_field_becomes_nil_and_final_clear_ends_the_plan() {
        let plan = plan(
            vec![
                Instruction::new_table(0, 0, 0),
                Instruction::ad(OpCode::LoadConstant, 1, 0),
                Instruction::abc(OpCode::GetTable, 2, 0, 1),
                Instruction::clear_range(0, 1),
                Instruction::abc(OpCode::Halt, 0, 0, 0),
            ],
            vec![Constant::String("missing".into())],
        )
        .expect("fresh table fragment is scalarizable");
        assert_eq!(plan.end_pc(), 4);
        assert!(matches!(
            plan.outputs().iter().find(|(register, _)| *register == 2),
            Some((_, ScalarTableOutput::Constant(Constant::Nil)))
        ));
    }
}
