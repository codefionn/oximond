//! Comparison operators shared by the IRs.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Comparison {
    Equal,
    LessThan,
    LessEqual,
}
