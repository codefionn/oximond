//! Immutable, bounded descriptions of table-mutating calls.
//! Multiple writes are accepted only when execution can stage them for one
//! existing table slot; the runtime evaluator checks that constraint.

use crate::bytecode::{Chunk, Constant, OpCode};
use crate::compat::CompatibilityProfile;
use crate::jit_ir::digests::ChunkDigest;

pub const MAX_INLINE_TABLE_REGISTERS: usize = 32;
const MAX_INLINE_TABLE_OPERATIONS: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub struct InlineTableCallPlan {
    parameter_count: u8,
    compatibility: CompatibilityProfile,
    callee_digest: ChunkDigest,
    operations: Vec<InlineTableOperation>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InlineTableOperation {
    Load {
        destination: u8,
        value: Constant,
    },
    Move {
        destination: u8,
        source: u8,
    },
    ClearRange {
        start: u8,
        count: u8,
    },
    GetTable {
        destination: u8,
        table: u8,
        key: u8,
    },
    SetTable {
        table: u8,
        key: u8,
        value: u8,
    },
    Numeric {
        opcode: OpCode,
        destination: u8,
        left: u8,
        right: u8,
    },
    Jump {
        target: usize,
    },
    JumpIfFalse {
        condition: u8,
        target: usize,
    },
    Return {
        source: u8,
    },
}

impl InlineTableCallPlan {
    #[must_use]
    pub const fn parameter_count(&self) -> u8 {
        self.parameter_count
    }

    #[must_use]
    pub const fn compatibility(&self) -> CompatibilityProfile {
        self.compatibility
    }

    #[must_use]
    pub const fn callee_digest(&self) -> ChunkDigest {
        self.callee_digest
    }

    #[must_use]
    pub fn operations(&self) -> &[InlineTableOperation] {
        &self.operations
    }
}

#[must_use]
pub fn validate_inline_table_call_plan(
    chunk: &Chunk,
    compatibility: CompatibilityProfile,
) -> Option<InlineTableCallPlan> {
    chunk.validate().ok()?;
    if chunk.is_vararg
        || !chunk.upvalues.is_empty()
        || usize::from(chunk.register_count) > MAX_INLINE_TABLE_REGISTERS
        || chunk.code.len() > MAX_INLINE_TABLE_OPERATIONS
    {
        return None;
    }
    let mut operations = Vec::with_capacity(chunk.code.len());
    let mut writes = 0_u8;
    let mut has_branch = false;
    let operation_len =
        chunk.code.len() - usize::from(chunk.code.last()?.opcode().ok()? == OpCode::Halt);
    // A field instruction becomes a load of its name into a scratch
    // register above the chunk's own, then the register-keyed operation.
    // Branch targets are bytecode pcs until the operations are laid out.
    let scratch = chunk.register_count;
    let mut first_operation = Vec::with_capacity(operation_len + 1);
    for (pc, instruction) in chunk.code.iter().copied().enumerate() {
        let opcode = instruction.opcode().ok()?;
        if opcode == OpCode::Halt && pc + 1 == chunk.code.len() {
            continue;
        }
        first_operation.push(operations.len());
        if matches!(
            opcode,
            OpCode::GetField
                | OpCode::SetField
                | OpCode::AddK
                | OpCode::SubtractK
                | OpCode::MultiplyK
                | OpCode::DivideK
                | OpCode::ModuloK
        ) {
            if usize::from(scratch) >= MAX_INLINE_TABLE_REGISTERS {
                return None;
            }
            let name = if opcode == OpCode::SetField {
                instruction.b()
            } else {
                instruction.c()
            };
            operations.push(InlineTableOperation::Load {
                destination: scratch,
                value: chunk.constants.get(usize::from(name))?.clone(),
            });
        }
        let operation = match opcode {
            OpCode::GetField => InlineTableOperation::GetTable {
                destination: instruction.a(),
                table: instruction.b(),
                key: scratch,
            },
            OpCode::SetField => {
                writes = writes.checked_add(1)?;
                InlineTableOperation::SetTable {
                    table: instruction.a(),
                    key: scratch,
                    value: instruction.c(),
                }
            }
            OpCode::LoadNil => InlineTableOperation::Load {
                destination: instruction.a(),
                value: Constant::Nil,
            },
            OpCode::LoadConstant => InlineTableOperation::Load {
                destination: instruction.a(),
                value: chunk.constants.get(usize::from(instruction.d()))?.clone(),
            },
            OpCode::Move => InlineTableOperation::Move {
                destination: instruction.a(),
                source: instruction.b(),
            },
            OpCode::ClearRange => InlineTableOperation::ClearRange {
                start: instruction.a(),
                count: instruction.b(),
            },
            OpCode::GetTable => InlineTableOperation::GetTable {
                destination: instruction.a(),
                table: instruction.b(),
                key: instruction.c(),
            },
            OpCode::SetTable => {
                writes = writes.checked_add(1)?;
                InlineTableOperation::SetTable {
                    table: instruction.a(),
                    key: instruction.b(),
                    value: instruction.c(),
                }
            }
            OpCode::Jump => {
                has_branch = true;
                InlineTableOperation::Jump {
                    target: forward_target(pc, instruction.jump_offset(), operation_len)?,
                }
            }
            OpCode::JumpIfFalse => {
                has_branch = true;
                InlineTableOperation::JumpIfFalse {
                    condition: instruction.a(),
                    target: forward_target(
                        pc,
                        i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
                        operation_len,
                    )?,
                }
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => InlineTableOperation::Numeric {
                opcode,
                destination: instruction.a(),
                left: instruction.b(),
                right: instruction.c(),
            },
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => InlineTableOperation::Numeric {
                opcode: opcode.register_form()?,
                destination: instruction.a(),
                left: instruction.b(),
                right: scratch,
            },
            OpCode::Return
                if instruction.b() == 1
                    && (pc + 1 == chunk.code.len()
                        || (pc + 2 == chunk.code.len()
                            && chunk.code[pc + 1].opcode().ok() == Some(OpCode::Halt))) =>
            {
                InlineTableOperation::Return {
                    source: instruction.a(),
                }
            }
            _ => return None,
        };
        operations.push(operation);
    }
    first_operation.push(operations.len());
    for operation in &mut operations {
        if let InlineTableOperation::Jump { target }
        | InlineTableOperation::JumpIfFalse { target, .. } = operation
        {
            *target = first_operation[*target];
        }
    }
    // The straight-line liveness pass removes operations and would invalidate
    // branch targets. Branch plans keep every operation.
    if !has_branch {
        remove_dead_assignments(&mut operations);
    }
    (writes > 0 && matches!(operations.last(), Some(InlineTableOperation::Return { .. })))
        .then_some(InlineTableCallPlan {
            parameter_count: chunk.parameter_count,
            compatibility,
            callee_digest: ChunkDigest::of(chunk),
            operations,
        })
}

fn forward_target(pc: usize, offset: i32, operation_len: usize) -> Option<usize> {
    let offset = usize::try_from(offset).ok()?;
    let target = pc.checked_add(1)?.checked_add(offset)?;
    // The optional trailing Halt is omitted from the plan. Every accepted
    // branch must reach an operation, not fall off the evaluator.
    (target < operation_len).then_some(target)
}

fn remove_dead_assignments(operations: &mut Vec<InlineTableOperation>) {
    let mut live = [false; MAX_INLINE_TABLE_REGISTERS];
    let mut keep = vec![true; operations.len()];
    for (index, operation) in operations.iter().enumerate().rev() {
        match *operation {
            InlineTableOperation::ClearRange { start, count } => {
                let slots = &mut live[usize::from(start)..usize::from(start) + usize::from(count)];
                keep[index] = slots.iter().any(|used| *used);
                slots.fill(false);
            }
            InlineTableOperation::Load { destination, .. } => {
                keep[index] = live[usize::from(destination)];
                live[usize::from(destination)] = false;
            }
            InlineTableOperation::Move {
                destination,
                source,
            } => {
                keep[index] = destination != source && live[usize::from(destination)];
                if keep[index] {
                    live[usize::from(destination)] = false;
                    live[usize::from(source)] = true;
                }
            }
            InlineTableOperation::GetTable {
                destination,
                table,
                key,
            } => {
                live[usize::from(destination)] = false;
                live[usize::from(table)] = true;
                live[usize::from(key)] = true;
            }
            InlineTableOperation::SetTable { table, key, value } => {
                live[usize::from(table)] = true;
                live[usize::from(key)] = true;
                live[usize::from(value)] = true;
            }
            InlineTableOperation::Numeric {
                destination,
                left,
                right,
                ..
            } => {
                live[usize::from(destination)] = false;
                live[usize::from(left)] = true;
                live[usize::from(right)] = true;
            }
            InlineTableOperation::Return { source } => live[usize::from(source)] = true,
            InlineTableOperation::Jump { .. } | InlineTableOperation::JumpIfFalse { .. } => {
                unreachable!("dead-assignment removal is limited to straight-line plans")
            }
        }
    }
    let mut index = 0;
    operations.retain(|_| {
        let keep = keep[index];
        index += 1;
        keep
    });
}

#[cfg(test)]
mod validation_tests {
    use super::{InlineTableOperation, validate_inline_table_call_plan};
    use crate::bytecode::{Chunk, Instruction, OpCode};
    use crate::compat::CompatibilityProfile;

    #[test]
    fn malformed_register_ranges_are_rejected_before_plan_analysis() {
        let chunk = Chunk::with_parts(
            2,
            1,
            Vec::new(),
            vec![
                Instruction::clear_range(31, 255),
                Instruction::abc(OpCode::SetTable, 0, 1, 1),
                Instruction::return_(0),
            ],
            Vec::new(),
            Vec::new(),
        );
        assert!(validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).is_none());
    }

    #[test]
    fn straight_line_plan_discards_dead_assignments_and_their_sources() {
        let chunk = Chunk::with_parts(
            5,
            3,
            Vec::new(),
            vec![
                Instruction::clear_range(3, 1),
                Instruction::move_(4, 3),
                Instruction::load_nil(4),
                Instruction::move_(4, 4),
                Instruction::move_(3, 2),
                Instruction::set_table(0, 1, 3),
                Instruction::return_(3),
            ],
            Vec::new(),
            Vec::new(),
        );
        let plan = validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap();
        assert_eq!(
            plan.operations(),
            &[
                InlineTableOperation::Move {
                    destination: 3,
                    source: 2,
                },
                InlineTableOperation::SetTable {
                    table: 0,
                    key: 1,
                    value: 3,
                },
                InlineTableOperation::Return { source: 3 },
            ]
        );
    }

    #[test]
    fn straight_line_plan_keeps_transitive_move_sources() {
        let chunk = Chunk::with_parts(
            5,
            3,
            Vec::new(),
            vec![
                Instruction::move_(3, 2),
                Instruction::move_(4, 3),
                Instruction::set_table(0, 1, 4),
                Instruction::return_(4),
            ],
            Vec::new(),
            Vec::new(),
        );
        let plan = validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap();
        assert_eq!(plan.operations().len(), 4);
    }

    #[test]
    fn forward_targets_keep_bytecode_indices_including_clears() {
        let chunk = Chunk::with_parts(
            4,
            2,
            Vec::new(),
            vec![
                Instruction::clear_range(3, 1),
                Instruction::jump_if_false(1, 1),
                Instruction::set_table(0, 2, 3),
                Instruction::return_(3),
                Instruction::halt(),
            ],
            Vec::new(),
            Vec::new(),
        );
        let plan = validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).unwrap();
        assert_eq!(plan.operations().len(), 4);
        assert!(matches!(
            plan.operations()[1],
            InlineTableOperation::JumpIfFalse { target: 3, .. }
        ));
    }

    #[test]
    fn backward_jumps_and_jumps_to_trailing_halt_are_rejected() {
        for jump in [
            Instruction::aj(OpCode::Jump, -2).unwrap(),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
        ] {
            let chunk = Chunk::with_parts(
                3,
                2,
                Vec::new(),
                vec![
                    Instruction::set_table(0, 1, 2),
                    jump,
                    Instruction::return_(2),
                    Instruction::halt(),
                ],
                Vec::new(),
                Vec::new(),
            );
            assert!(validate_inline_table_call_plan(&chunk, CompatibilityProfile::Lua51).is_none());
        }
    }
}
