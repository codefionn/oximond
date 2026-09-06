// helpers shared by the split test modules
#[allow(unused_imports)]
pub(crate) use super::*;
pub(crate) use crate::bytecode::CallKind;
pub(crate) use crate::bytecode::Constant;
pub(crate) use crate::bytecode::OpCode;
pub(crate) use crate::bytecode::UpvalueSource;
pub(crate) use crate::bytecode::load::LoadInput;
pub(crate) use crate::bytecode::load::LoadPoll;
pub(crate) use crate::compat::CompatibilityProfile;
pub(crate) use crate::runtime::{Runtime, RuntimeResult, Value};
pub(crate) use crate::stdlib::{install_base, install_base_with_profile};
pub(crate) use crate::string::LuaString;
pub(crate) use crate::vm::Vm;
pub(crate) use std::fmt::Write as _;

fn run(source: &str) -> Vm {
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
    vm
}

fn run_with_config(source: &str, config: CompilerConfig) -> Vm {
    let chunk = compile_with_config(source, config).unwrap();
    let mut vm = Vm::with_profile(config.compatibility_profile());
    install_base_with_profile(vm.runtime_mut(), config.compatibility_profile()).unwrap();
    vm.execute(&chunk).unwrap();
    vm
}

#[allow(clippy::unnecessary_wraps)]
fn rhs_changes_i(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    runtime.set_global("i", Value::Number(9.0));
    Ok(vec![Value::Number(42.0)])
}

#[allow(clippy::unnecessary_wraps)]
fn record_rhs_call(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    runtime.set_global("rhs_was_called", Value::Boolean(true));
    Ok(vec![Value::Number(99.0)])
}

#[allow(clippy::unnecessary_wraps)]
fn test_iterator(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Value::Number(control) = arguments[1] else {
        panic!("generic iterator control must be numeric");
    };
    let next = control + 1.0;
    if next > 3.0 {
        Ok(vec![Value::Nil])
    } else {
        Ok(vec![Value::Number(next), Value::Number(next * 10.0)])
    }
}

#[allow(clippy::unnecessary_wraps)]
fn setup_iterator(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let calls = match runtime.get_global("setup_calls") {
        Value::Number(calls) => calls,
        Value::Nil => 0.0,
        value => panic!("unexpected setup call count {value:?}"),
    };
    runtime.set_global("setup_calls", Value::Number(calls + 1.0));
    Ok(vec![
        runtime.get_global("iter"),
        Value::Number(0.0),
        Value::Number(0.0),
    ])
}

mod errors;
mod expressions;
mod misc;
mod statements;
