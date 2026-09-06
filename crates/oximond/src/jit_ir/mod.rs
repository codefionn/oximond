#![forbid(unsafe_code)]

//! IR between verified Oximond bytecode and the JIT backends.
//!
//! This crate does not execute code or emit machine instructions. It holds
//! the trace IR the trace engine records and the call and table plans the
//! interpreter evaluates.

pub mod inline_table;
pub mod scalar_table;
pub mod trace;
pub use inline_table::{
    InlineTableCallPlan, InlineTableOperation, MAX_INLINE_TABLE_REGISTERS,
    validate_inline_table_call_plan,
};
pub use scalar_table::{
    ScalarNumberOperation, ScalarTableOutput, ScalarTablePlan, validate_scalar_table_plan,
};

mod digests;
mod encoder;
mod inline_branches;
mod inline_plans;
mod liveness;
mod operation;

pub use digests::ChunkDigest;
pub use inline_plans::{
    ConstantKind, InlineCallPlan, InlineNumericOperation, InlinePlanError, RegisterFile,
    compose_vararg_forwarding_inline_call_plan_after_validation, validate_inline_call_plan,
    validate_inline_call_plan_after_validation,
};
pub use liveness::{RegisterSet, observable_registers, register_set_contains};
pub use operation::Comparison;

#[cfg(test)]
mod tests;
