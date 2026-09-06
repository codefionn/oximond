//! `LuaJIT`'s optional table allocation and clearing modules.

use crate::runtime::{Runtime, RuntimeError, RuntimeResult, Value};
use crate::stdlib::args::table_argument;

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "LuaJIT truncates allocation hints after checking bounds and clamping negative array hints."
)]
pub(crate) fn new(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let mut sizes = [0; 2];
    for (index, size) in sizes.iter_mut().enumerate() {
        let value = arguments
            .get(index)
            .ok_or_else(|| RuntimeError::bad_argument(index + 1, "number", None))?;
        let number = Runtime::coerce_number(value, "call table.new with")
            .map_err(|_| RuntimeError::bad_argument(index + 1, "number", Some(value)))?;
        if !number.is_finite() || (index == 1 && number < 0.0) || number >= f64::from(u32::MAX) {
            return Err(RuntimeError::message("table overflow"));
        }
        *size = number.max(0.0).trunc() as usize;
    }
    let table = runtime.create_table()?;
    runtime.table_reserve(table, sizes[0], sizes[1])?;
    Ok(vec![Value::Table(table)])
}

pub(crate) fn clear(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let table = table_argument(arguments.first(), 1)?;
    runtime.table_clear(table)?;
    Ok(Vec::new())
}
