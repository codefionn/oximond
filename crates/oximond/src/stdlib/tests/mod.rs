// helpers shared by the split test modules
#[allow(unused_imports)]
use super::*;
use crate::compat::CompatibilityProfile;
use crate::runtime::{BuiltinFunction, NativeFunctionHandle, Runtime, RuntimeError, Value};
use crate::stdlib::installers::{install_base, install_base_with_profile};
use std::io::{self, Write};

fn string_functions(runtime: &Runtime) -> (NativeFunctionHandle, NativeFunctionHandle) {
    let Value::Table(string) = runtime.get_global("string") else {
        panic!("string is a table");
    };
    let Value::NativeFunction(sub) = runtime.table_get(string, &Value::from("sub")).unwrap() else {
        panic!("string.sub is native");
    };
    let Value::NativeFunction(rep) = runtime.table_get(string, &Value::from("rep")).unwrap() else {
        panic!("string.rep is native");
    };
    (sub, rep)
}

fn string_function(runtime: &Runtime, name: &str) -> NativeFunctionHandle {
    let Value::Table(string) = runtime.get_global("string") else {
        panic!("string is a table");
    };
    let Value::NativeFunction(function) = runtime.table_get(string, &Value::from(name)).unwrap()
    else {
        panic!("string.{name} is native");
    };
    function
}

fn math_function(runtime: &Runtime, name: &str) -> NativeFunctionHandle {
    let Value::Table(math) = runtime.get_global("math") else {
        panic!("math is a table");
    };
    let Value::NativeFunction(function) = runtime.table_get(math, &Value::from(name)).unwrap()
    else {
        panic!("math.{name} is native");
    };
    function
}

fn bit_function(runtime: &Runtime, name: &str) -> NativeFunctionHandle {
    let Value::Table(bit) = runtime.get_global("bit") else {
        panic!("bit is a table");
    };
    let Value::NativeFunction(function) = runtime.table_get(bit, &Value::from(name)).unwrap()
    else {
        panic!("bit.{name} is native");
    };
    function
}

mod base;
mod core;
mod iolib;
mod math;
mod os;
mod strings;
mod tables;

mod bit32;
