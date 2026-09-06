//! Host-call probe: a traced loop that calls a host function returning a
//! fresh table, as Mineclonia's `locate_biome_in_area` calls
//! `core.get_biome_data` through `index_biomes_cached`.
//!
//! ```text
//! cargo run --release -p oximond-cli --example hostcall_probe -- 2000 700
//! ```
//!
//! The arguments are the number of host entries (calls of `locate`) and the
//! iterations of each. Each iteration writes `v.x`, `v.y` and `v.z` into an
//! upvalue table, calls the host function `get_biome_data(v)` from an inlined
//! method, and reads `data.biome` and `translations[data.biome]`. Count
//! instructions with `perf stat -e instructions:u`; the probe prints the
//! result and the counters that show the calls ran from traces.

use std::env;
use std::error::Error;

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_named};
use oximond::{Lua, Value, Vm, VmError};

const SOURCE: &str = r#"
local entries, iterations = ...
-- The module: its locals are closed upvalues by the time the loop runs.
local function load_module()
    local translations = { "plains", "desert", "forest", "taiga", "ocean" }
    local v = { x = 0, y = 0, z = 0 }
    local get_biome_data = get_biome_data
    local preset = {}
    preset.__index = preset

    function preset:index_biomes_cached(x, y, z)
        v.x = x
        v.y = y
        v.z = z
        local data = get_biome_data(v)
        local biome = data.biome
        return translations[biome] or biome
    end

    local function locate(self, x0, z0, n)
        local found = 0
        for i = 1, n do
            local name = self:index_biomes_cached(x0 + i, 12, z0 + 2 * i)
            if name == "forest" then
                found = found + 1
            end
        end
        return found
    end

    return setmetatable({}, preset), locate
end

local instance, locate = load_module()
local total = 0
for entry = 1, entries do
    total = total + locate(instance, entry * 3, entry * 7, iterations)
end
return total
"#;

/// The engine's `core.get_biome_data(v)`: a fresh table for the position the
/// fields `x`, `y` and `z` of `v` name, with the fields `biome`, `heat` and
/// `humidity`.
fn get_biome_data(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let Some(v) = arguments.first().cloned() else {
        return Ok(vec![Value::Nil]);
    };
    let mut sum = 0.0;
    for field in ["x", "y", "z"] {
        if let Value::Number(n) = vm.get_value(v.clone(), Value::from(field))? {
            sum += n;
        }
    }
    let table = vm.runtime_mut().create_table_value()?;
    let biome = sum.rem_euclid(5.0).floor() + 1.0;
    vm.set_value(table.clone(), Value::from("biome"), Value::Number(biome))?;
    vm.set_value(table.clone(), Value::from("heat"), Value::Number(sum * 0.5))?;
    vm.set_value(
        table.clone(),
        Value::from("humidity"),
        Value::Number(sum * 0.25),
    )?;
    Ok(vec![table])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let entries: f64 = args.next().as_deref().unwrap_or("2000").parse()?;
    let iterations: f64 = args.next().as_deref().unwrap_or("700").parse()?;
    let mode = match args.next().as_deref() {
        Some("interpreter") => ExecutionMode::Interpreter,
        _ => ExecutionMode::Tracing,
    };
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(mode)
        .build()?;
    let function = lua
        .vm_mut()
        .register_host_function(get_biome_data, 0, Vec::new())?;
    lua.vm_mut()
        .runtime_mut()
        .set_global("get_biome_data", function);
    let chunk = compile_named(
        "hostcall_probe",
        SOURCE,
        CompilerConfig::new(CompatibilityProfile::LuaJit21),
    )?;
    let main = lua
        .vm_mut()
        .runtime_mut()
        .create_closure(std::sync::Arc::new(chunk), Vec::new())?;
    let results = lua.vm_mut().call_value(
        Value::Closure(main),
        vec![Value::Number(entries), Value::Number(iterations)],
    )?;
    let statistics = lua.vm().jit_statistics();
    eprintln!("hostcall probe: results={results:?}");
    eprintln!(
        "hostcall probe: host_trace_calls={} traces_compiled={} exits={}",
        statistics.host_trace_calls,
        statistics.trace.traces_compiled,
        statistics.trace.exits.iter().sum::<u64>()
    );
    Ok(())
}
