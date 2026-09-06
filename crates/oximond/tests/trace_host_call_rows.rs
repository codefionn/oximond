//! A trace that calls a host function keeps the rows of its tables across
//! the call when nothing changed them, and checks or fills them again when
//! the call did. These loops change the table the trace reads from inside
//! the host call, in every way a row can go stale, and compare the answer
//! with the interpreter's.

use std::cell::Cell;

use oximond::runtime::{TableHandle, Value};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Vm, VmError};

thread_local! {
    /// The `__index` table `poke` gave `t`.
    static FALLBACK: Cell<Option<TableHandle>> = const { Cell::new(None) };
}

/// `poke(t, i)`: changes `t` at a few chosen steps of the loop, and gives
/// zero. A host function, as a C API callback is, so a trace calls it
/// itself; and one with a result, since a trace makes only those calls.
fn poke(vm: &mut Vm, _token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    let (Some(Value::Table(table)), Some(step)) = (args.first(), args.get(1)) else {
        return Ok(vec![Value::Integer(0)]);
    };
    let runtime = vm.runtime_mut();
    let table = *table;
    let step = match step {
        Value::Integer(step) => *step,
        #[allow(clippy::cast_possible_truncation)]
        Value::Number(step) => *step as i64,
        _ => return Ok(vec![Value::Integer(0)]),
    };
    match step {
        // Overwrites a field with a value of the same type.
        3000 => runtime.table_set(table, &Value::string("x"), Value::Integer(5))?,
        // Adds enough keys to move the table's entries.
        6000 => {
            for key in 0..64 {
                runtime.table_set(
                    table,
                    &Value::string(format!("k{key}")),
                    Value::Integer(key),
                )?;
            }
        }
        // A field the trace saw absent answers through `__index` now.
        9000 => {
            let fallback = runtime.create_table()?;
            runtime.table_set(fallback, &Value::string("w"), Value::Integer(100))?;
            let metatable = runtime.create_table()?;
            runtime.table_set(metatable, &Value::string("__index"), Value::Table(fallback))?;
            runtime.set_table_metatable(table, Some(metatable))?;
            FALLBACK.set(Some(fallback));
        }
        // Changes the table behind the `__index` without touching `t`.
        12000 => {
            if let Some(fallback) = FALLBACK.get() {
                runtime.table_set(fallback, &Value::string("w"), Value::Integer(1000))?;
            }
        }
        // Deletes a field the trace reads: its slot now holds nil.
        15000 => runtime.table_set(table, &Value::string("y"), Value::Nil)?,
        // Brings it back, in the same slot or another.
        17000 => runtime.table_set(table, &Value::string("y"), Value::Integer(7))?,
        // Gives a field a value of another type.
        19000 => runtime.table_set(table, &Value::string("x"), Value::Number(2.5))?,
        _ => {}
    }
    Ok(vec![Value::Integer(0)])
}

fn run(mode: ExecutionMode, source: &str) -> (Vec<Value>, u64) {
    FALLBACK.set(None);
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let poke = lua
        .vm_mut()
        .register_host_function(poke, 0, Vec::new())
        .unwrap();
    lua.vm_mut().runtime_mut().set_global("poke", poke);
    let results = lua.exec(source).unwrap();
    (results, lua.vm().jit_statistics().trace.iterations)
}

/// `swap(env, i)`: changes what the globals `step` and `mode` of the
/// environment table `env` hold at a few chosen steps of the loop, and gives
/// zero. The trace calls `step` and compares `mode`, and after each call
/// here checks that both still hold what it was recorded with, from the
/// slots it found them in.
fn swap(vm: &mut Vm, _token: u64, args: &[Value]) -> Result<Vec<Value>, VmError> {
    let (Some(Value::Table(table)), Some(step)) = (args.first(), args.get(1)) else {
        return Ok(vec![Value::Integer(0)]);
    };
    let runtime = vm.runtime_mut();
    let table = *table;
    // The two functions the table keeps for the swaps.
    let first = runtime.table_get(table, &Value::string("one"))?;
    let second = runtime.table_get(table, &Value::string("two"))?;
    let step = match step {
        Value::Integer(step) => *step,
        #[allow(clippy::cast_possible_truncation)]
        Value::Number(step) => *step as i64,
        _ => return Ok(vec![Value::Integer(0)]),
    };
    let (f, mode) = (Value::string("step"), Value::string("mode"));
    match step {
        // Another function in the same slot: nothing moved.
        3000 => runtime.table_set(table, &f, second)?,
        // Enough keys to move the entries; the fields hold what they held.
        6000 => {
            for key in 0..64 {
                runtime.table_set(
                    table,
                    &Value::string(format!("k{key}")),
                    Value::Integer(key),
                )?;
            }
        }
        // The first function again, where the key is now: after the keys
        // came, and again after they went.
        9000 | 27000 => runtime.table_set(table, &f, first)?,
        // Deleted and brought back with the other function.
        12000 => {
            runtime.table_set(table, &f, Value::Nil)?;
            runtime.table_set(table, &f, second)?;
        }
        // Another string in the same slot.
        15000 => runtime.table_set(table, &mode, Value::string("slow"))?,
        // Deleted: the slot of a hashed table holds nil in place.
        18000 => runtime.table_set(table, &mode, Value::Nil)?,
        21000 => runtime.table_set(table, &mode, Value::string("fast"))?,
        // Keys taken out again, which moves the entries of a small table.
        24000 => {
            for key in 0..64 {
                runtime.table_set(table, &Value::string(format!("k{key}")), Value::Nil)?;
            }
        }
        _ => {}
    }
    Ok(vec![Value::Integer(0)])
}

fn run_swap(mode: ExecutionMode, source: &str) -> (Vec<Value>, u64) {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let swap = lua
        .vm_mut()
        .register_host_function(swap, 0, Vec::new())
        .unwrap();
    lua.vm_mut().runtime_mut().set_global("swap", swap);
    let results = lua.exec(source).unwrap();
    (results, lua.vm().jit_statistics().trace.iterations)
}

/// The loop of the global tests, in the environment `env` names.
fn swap_loop(env: &str) -> String {
    format!(
        "{env}
         function one(i) return i end
         function two(i) return i * 2 end
         step = one
         mode = 'fast'
         local t = {{x = 1, y = 2, z = 3}}
         local s = 0
         for i = 1, 30000 do
           s = s + (t.x or 0) + (t.y or 0) + (t.w or 0) + t.z
           s = s + step(i)
           if mode == 'fast' then s = s + 1 end
           s = s + swap(env, i)
         end
         return s"
    )
}

#[test]
fn globals_found_once_follow_every_change_a_host_call_makes() {
    // The globals table, whose entries are hashed: a key is deleted and
    // revived in place.
    let source = swap_loop("local env = _G");
    let (expected, _) = run_swap(ExecutionMode::Interpreter, &source);
    let (traced, iterations) = run_swap(ExecutionMode::Tracing, &source);
    assert_eq!(traced, expected);
    assert!(iterations > 8_000, "{iterations} native iterations");
}

#[test]
fn globals_of_a_small_environment_follow_every_change_a_host_call_makes() {
    // An environment of a few keys, whose entries are shaped when the
    // loop starts and change representation when keys come and go.
    let source = swap_loop("local env = {swap = swap}; local _ENV = env");
    let (expected, _) = run_swap(ExecutionMode::Interpreter, &source);
    let (traced, iterations) = run_swap(ExecutionMode::Tracing, &source);
    assert_eq!(traced, expected);
    assert!(iterations > 8_000, "{iterations} native iterations");
}

fn same_as_interpreter(source: &str) -> u64 {
    let (expected, _) = run(ExecutionMode::Interpreter, source);
    let (traced, iterations) = run(ExecutionMode::Tracing, source);
    assert_eq!(traced, expected);
    iterations
}

#[test]
fn rows_follow_every_change_a_host_call_makes_to_a_table() {
    let iterations = same_as_interpreter(
        "local t = {x = 1, y = 2, z = 3}
         local s = 0
         for i = 1, 30000 do
           s = s + (t.x or 0) + (t.y or 0) + (t.w or 0) + t.z
           s = s + poke(t, i)
         end
         return s, t.x, t.y, t.w",
    );
    // The trace makes the host calls itself, and carries on through the
    // changes up to the metatable, which sends it out.
    assert!(iterations > 8_000, "{iterations} native iterations");
}

#[test]
fn rows_of_tables_a_host_call_leaves_alone_stand() {
    let iterations = same_as_interpreter(
        "local t = {x = 1, y = 2, z = 3}
         local u = {x = 10, y = 20}
         local s = 0
         for i = 1, 30000 do
           s = s + (t.x or 0) + (t.y or 0) + (t.w or 0) + u.x + u.y
           s = s + poke(t, i)
         end
         return s",
    );
    assert!(iterations > 8_000, "{iterations} native iterations");
}
