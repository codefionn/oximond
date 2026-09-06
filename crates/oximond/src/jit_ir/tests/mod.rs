// helpers shared by the split test modules
#[allow(unused_imports)]
pub(crate) use super::*;
pub(crate) use crate::bytecode::Chunk;
pub(crate) use crate::bytecode::Constant;
pub(crate) use crate::bytecode::OpCode;
pub(crate) use crate::bytecode::UpvalueSource;
pub(crate) use crate::bytecode::{DebugMetadata, Instruction};
pub(crate) use crate::jit_ir::encoder::sha256;
pub(crate) use crate::jit_ir::inline_plans::{
    compose_vararg_forwarding_inline_call_plan_after_validation, validate_inline_call_plan,
    validate_inline_call_plan_after_validation,
};

fn numeric_chunk() -> Chunk {
    Chunk::new(
        4,
        vec![Constant::Number(2.5)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::unary(OpCode::Negate, 3, 2),
            Instruction::halt(),
        ],
    )
}

mod inline;
mod liveness;
