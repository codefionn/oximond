//! JIT and tiering glue: trace compilation, native entry, and the plan-based
//! interpreter fast paths that share the native backend.

mod completions;
mod entry;
mod inline_calls;
mod inline_plans;
pub(crate) mod inline_table;
pub(crate) mod scalar_table;
pub(crate) mod state;
pub mod trace;
