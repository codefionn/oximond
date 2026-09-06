//! The interpreter pushes a plain activation for a Lua call, however long the
//! body, vararg or not: a vararg body keeps its extra arguments on the value
//! stack, so its activation costs no more than a fixed one. A plain call and
//! its return cost less than the evaluation of even a long numeric plan.
//! These programs pin the choice, and that either way the results are the
//! same.

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::runtime::Value;
use oximond::stdlib::install_base_with_profile;
use oximond::vm::Vm;

/// Runs `source` and returns its results and how many calls the interpreter
/// evaluated without a frame.
fn run(source: &str, profile: CompatibilityProfile, mode: ExecutionMode) -> (Vec<Value>, u64) {
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    vm.use_synchronous_jit_compilation();
    install_base_with_profile(vm.runtime_mut(), profile).expect("install standard library");
    let results = vm.execute(&chunk).expect("run");
    (results, vm.jit_statistics().interpreter_inline_calls)
}

fn interpreter(source: &str) -> (Vec<Value>, u64) {
    run(
        source,
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
    )
}

fn float(value: &Value) -> f64 {
    match value {
        Value::Number(number) => *number,
        other => panic!("expected a number, found {other:?}"),
    }
}

#[test]
fn a_short_numeric_callee_gets_an_activation() {
    let (results, inlined) = interpreter(
        r"
        local function step(x) return x * 1.5 + 0.25 end
        local total = 0
        for i = 1, 100 do total = total + step(i) end
        return total
        ",
    );
    assert_eq!(float(&results[0]), 7600.0);
    // The first call, before the closure is classified, may still be a plan.
    assert!(
        inlined <= 1,
        "two operations cost less as bytecode: {inlined}"
    );
}

#[test]
fn a_long_pure_numeric_callee_gets_an_activation_too() {
    let (results, inlined) = interpreter(
        r"
        local function f(a, b)
            local ij = a + b
            return 1 / (ij * (ij + 1) / 2 + a + 1)
        end
        local total = 0
        for i = 1, 100 do total = total + f(i, 2) end
        return total
        ",
    );
    let expected: f64 = (1..=100)
        .map(|i| {
            let ij = f64::from(i) + 2.0;
            1.0 / (ij * (ij + 1.0) / 2.0 + f64::from(i) + 1.0)
        })
        .sum();
    assert!((float(&results[0]) - expected).abs() < 1e-9);
    // The first call, before the closure is classified, may still be a plan.
    assert!(inlined <= 1, "{inlined} calls evaluated as a plan");
}

#[test]
fn a_vararg_body_with_a_plan_gets_an_activation_too() {
    let (results, inlined) = interpreter(
        r"
        local function sum(...)
            local a, b, c = ...
            return a + b + c
        end
        local total = 0
        for i = 1, 100 do total = total + sum(i, 2, 3) end
        return total
        ",
    );
    assert_eq!(float(&results[0]), 5050.0 + 500.0);
    // The first call, before the closure is classified, may still be a plan.
    assert!(inlined <= 1, "{inlined} calls evaluated as a plan");
}

#[test]
fn closures_that_differ_in_how_they_are_called_can_alternate() {
    // A long numeric callee, a plain one and a vararg one share a call site,
    // which remembers only the last callee it left to the general routine.
    let source = r"
        local function long(a, b)
            local ij = a + b
            return 1 / (ij * (ij + 1) / 2 + a + 1)
        end
        local function plain(a, b) return a - b end
        local function va(...) local a, b = ... return a * b end
        local all = {long, plain, va}
        local total = 0
        for i = 1, 300 do
            total = total + all[i % 3 + 1](i, 3)
        end
        return total
    ";
    let (fast, _) = interpreter(source);
    let (traced, _) = run(source, CompatibilityProfile::Lua51, ExecutionMode::Tracing);
    assert!(
        (float(&fast[0]) - float(&traced[0])).abs() < 1e-6,
        "{fast:?} {traced:?}"
    );
}

#[test]
fn an_error_inside_a_plan_callee_reports_the_callers_line() {
    let source = "local function f(a, b)
    local ij = a + b
    return 1 / (ij * (ij + 1) / 2 + a + 1) + nothing
end
local ok, message
for i = 1, 5 do
    ok, message = pcall(f, i, 2)
end
return ok, message
";
    let (results, _) = interpreter(source);
    assert_eq!(results[0], Value::Boolean(false));
    let Value::String(message) = &results[1] else {
        panic!("expected a message, found {:?}", results[1]);
    };
    assert!(
        String::from_utf8_lossy(message.as_bytes()).contains(":3:"),
        "{results:?}"
    );
}
