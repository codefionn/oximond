//! Host-call probe for the trace backends: a traced loop with integers and
//! floats live across a call of a cheap host function, as Mineclonia's
//! mapgen loops call into the engine. The trace calls the function itself
//! (`HelperId::vm_call`), after storing the frame's state for it.
//!
//! ```text
//! cargo run --release -p oximond-cli --example vmcall_probe -- 300000
//! OXIMOND_JIT_BACKEND=cranelift cargo run --release -p oximond-cli --example vmcall_probe -- 300000
//! ```
//!
//! The argument is the number of iterations. Count instructions with
//! `perf stat -e instructions:u` under each backend, at two iteration counts:
//! the difference over the difference of the iterations is the cost of one
//! round. The probe prints the result and the counters that show the calls
//! ran from traces.

use std::env;
use std::error::Error;

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::{Lua, Value, Vm, VmError};

const SOURCE: &str = r"
local function run(n)
local s, acc, count = 0, 0.0, 0
for i = 1, n do
    local x = i % 80
    local y = (i // 80) % 80
    local z = i // 6400
    local fx = x * 0.13 + y * 0.07
    local fy = y * 0.11 - z * 0.05
    local h = cheight(x, z)
    local d = fx * fy + h * 0.5
    if d > 10.0 then count = count + 1 end
    acc = acc + d
    s = s + x + y + z
end
return s, acc, count
end
return run(N)
";

/// `cheight(x, z)`: `(3x + z) % 17`, an integer.
fn cheight(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let int = |value: Option<&Value>| match value {
        Some(Value::Integer(value)) => *value,
        #[allow(clippy::cast_possible_truncation)]
        Some(Value::Number(value)) => *value as i64,
        _ => 0,
    };
    let (x, z) = (int(arguments.first()), int(arguments.get(1)));
    Ok(vec![Value::Integer((3 * x + z).rem_euclid(17))])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let iterations: i64 = args.next().as_deref().unwrap_or("300000").parse()?;
    let mode = match args.next().as_deref() {
        Some("interpreter") => ExecutionMode::Interpreter,
        _ => ExecutionMode::Tracing,
    };
    let profile = CompatibilityProfile::Lua54;
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()?;
    let function = lua
        .vm_mut()
        .register_host_function(cheight, 1, Vec::new())?;
    lua.vm_mut().runtime_mut().set_global("cheight", function);
    lua.vm_mut()
        .runtime_mut()
        .set_global("N", Value::Integer(iterations));
    let results = lua.exec(SOURCE)?;
    let statistics = lua.vm().jit_statistics();
    println!("{results:?}");
    eprintln!(
        "vmcall probe: host_trace_calls={} traces_compiled={} exits={}",
        statistics.host_trace_calls,
        statistics.trace.traces_compiled,
        statistics.trace.exits.iter().sum::<u64>()
    );
    Ok(())
}
