//! Seeded differential testing of oximond against stock Lua, and of its
//! tracing JIT against its own interpreter ([`jit`]).
//!
//! A seed and a profile determine a valid, bounded Lua program
//! ([`generate`]). The stock interpreter for that profile runs it to give
//! the expected output, and oximond runs it in each execution mode
//! ([`case`]). Campaigns run seed ranges, save every failure with what is
//! needed to replay it, and reduce failing programs ([`reduce`]).

pub mod campaign;
pub mod case;
pub mod compare;
pub mod fixtures;
pub mod generate;
pub mod jit;
pub mod profile;
pub mod program;
pub mod reduce;
pub mod rejections;
pub mod rng;
pub mod run;
pub mod workloads;
