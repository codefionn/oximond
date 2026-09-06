//! Host entries: a host calling a Lua function (`Vm::call_value`) again and
//! again, the way Luanti converts every position through `read_vector`.
//! Each entry runs on a stack the VM keeps from the last one (`Vm::run`),
//! and a host callback parks its caller's stack until it returns
//! (`HostSnapshot`). Results must match a VM that interprets, traces must
//! run on the reused stacks, and an entry that fails, a captured variable,
//! a table that changes under a trace and a nested entry must leave the
//! next entry correct.

use super::trace_native::{native_vm, stats_of};
use super::*;

const SOURCE: &str = "
function read_vector(v) return v.x, v.y, v.z end
function sum(t) local s = 0 for i = 1, #t do s = s + t[i] end return s end
function fail(x) if x > 2 then error('too big') end return x end
function fail_captured(x)
  local kept = {x}
  local function get() return kept[1] end
  if x > 2 then error('captured ' .. get()) end
  return get()
end
function counter(n) local c = 0 local function add() c = c + n end add() add() return c end
";

fn loaded(profile: CompatibilityProfile, mode: ExecutionMode) -> Vm {
    let mut vm = match mode {
        ExecutionMode::Tracing => native_vm(profile),
        ExecutionMode::Interpreter => Vm::with_profile_and_execution_mode(profile, mode),
    };
    let chunk = crate::compiler::compile(SOURCE).unwrap();
    vm.execute(&chunk).unwrap();
    vm
}

fn call(vm: &mut Vm, name: &str, arguments: Vec<Value>) -> VmResult<Vec<Value>> {
    let function = vm.runtime().get_global(name);
    vm.call_value(function, arguments)
}

fn vector(vm: &mut Vm, x: f64, y: f64, z: f64) -> Value {
    let table = vm.runtime_mut().create_table().unwrap();
    for (key, value) in [("x", x), ("y", y), ("z", z)] {
        vm.runtime_mut()
            .table_set(table, &Value::from(key), Value::Number(value))
            .unwrap();
    }
    Value::Table(table)
}

/// Everything the host sees over a run of entries: every result, and every
/// error message.
fn session(vm: &mut Vm) -> Vec<String> {
    let mut seen = Vec::new();
    let list = vm.runtime_mut().create_table().unwrap();
    for i in 1..=40 {
        vm.runtime_mut()
            .table_set(
                list,
                &Value::Number(f64::from(i)),
                Value::Number(f64::from(i)),
            )
            .unwrap();
    }
    for round in 0..300 {
        let v = vector(vm, f64::from(round), 2.0, -1.0);
        seen.push(format!("{:?}", call(vm, "read_vector", vec![v]).unwrap()));
        // A field changes type, the table gains a metatable, then the list
        // is swapped for a shorter one: the loop's trace leaves for each.
        if round == 150 {
            vm.runtime_mut()
                .table_set(list, &Value::Number(3.0), Value::Number(0.5))
                .unwrap();
        }
        if round == 200 {
            let meta = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .set_table_metatable(list, Some(meta))
                .unwrap();
        }
        let argument = if round >= 250 {
            let short = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .table_set(short, &Value::Number(1.0), Value::Number(7.0))
                .unwrap();
            Value::Table(short)
        } else {
            Value::Table(list)
        };
        seen.push(format!("{:?}", call(vm, "sum", vec![argument]).unwrap()));
        let failed = call(vm, "fail", vec![Value::Number(f64::from(round % 5))]);
        seen.push(match failed {
            Ok(values) => format!("{values:?}"),
            Err(error) => error.to_string(),
        });
        // An error with a captured variable open on the stack.
        let failed = call(
            vm,
            "fail_captured",
            vec![Value::Number(f64::from(round % 5))],
        );
        seen.push(match failed {
            Ok(values) => format!("{values:?}"),
            Err(error) => error.to_string(),
        });
        seen.push(format!(
            "{:?}",
            call(vm, "counter", vec![Value::Number(f64::from(round))]).unwrap()
        ));
    }
    seen
}

#[test]
fn host_entries_reuse_their_stack_and_match_the_interpreter() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut interpreter = loaded(profile, ExecutionMode::Interpreter);
        let expected = session(&mut interpreter);
        let mut vm = loaded(profile, ExecutionMode::Tracing);
        let seen = session(&mut vm);
        assert_eq!(seen, expected, "{profile:?}");
        // Every entry returned, so one stack serves them all.
        assert_eq!(vm.spare_host_stacks.len(), 1, "{profile:?}");
        let stats = stats_of(&vm);
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

/// A host callback that enters Lua again (`sum`), from inside a Lua loop
/// that a host entry runs: two host entries are live at once, each on a
/// stack of its own, and both go back to the spare list.
fn reenter(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    let sum = vm.runtime().get_global("sum");
    vm.call_value(sum, arguments.to_vec())
}

#[test]
fn nested_host_entries_take_stacks_of_their_own() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = loaded(CompatibilityProfile::Lua54, mode);
        let function = vm.register_host_function(reenter, 0, Vec::new()).unwrap();
        vm.runtime_mut().set_global("reenter", function);
        let chunk = crate::compiler::compile(
            "function outer(n) \
               local t = {1, 2, 3} local s = 0 \
               for i = 1, n do s = s + reenter(t) end \
               return s end",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
        let result = call(&mut vm, "outer", vec![Value::Number(500.0)]).unwrap();
        assert_eq!(result, vec![Value::Number(3000.0)], "{mode:?}");
        assert_eq!(vm.spare_host_stacks.len(), 2, "{mode:?}");
    }
}

/// A callback that runs a full collection and then reads a local of its Lua
/// caller through the debugger, after a nested entry has run on another
/// stack. The caller's stack is parked while the callback runs, and the
/// table the local holds is nowhere else: the collector must root the
/// parked stack, and the debugger's records built late must see the caller
/// as it was.
fn collect_then_read_local(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    let sum = vm.runtime().get_global("sum");
    let list = vm.runtime_mut().create_table()?;
    vm.runtime_mut()
        .table_set(list, &Value::Number(1.0), arguments[0].clone())?;
    vm.call_value(sum, vec![Value::Table(list)])?;
    vm.collect_full_with_values(&[]);
    let local = vm.host_get_local(1, 2)?;
    let frame = vm.host_stack_info(1).expect("the Lua caller is parked");
    let (name, value) = local.expect("the caller has a second local");
    let Value::Table(table) = value else {
        panic!("the second local is the held table, not {value:?}");
    };
    let x = vm.runtime().table_get(table, &Value::from("x"))?;
    Ok(vec![
        Value::String(name),
        x,
        Value::Number(f64::from(frame.current_line)),
    ])
}

#[test]
fn a_parked_stack_stays_rooted_and_inspectable() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = loaded(CompatibilityProfile::Lua54, mode);
        let function = vm
            .register_host_function(collect_then_read_local, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("inspect", function);
        let chunk = crate::compiler::compile(
            "function probe(i)\n\
               local held = {x = i, y = 0, z = 0}\n\
               return inspect(i)\n\
             end",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
        for i in 0..50 {
            let result = call(&mut vm, "probe", vec![Value::Number(f64::from(i))]).unwrap();
            assert_eq!(
                result,
                vec![
                    Value::from("held"),
                    Value::Number(f64::from(i)),
                    Value::Number(3.0)
                ],
                "{mode:?}"
            );
        }
    }
}

/// Returns its argument count and the sum of its numeric arguments, or an
/// argument error for a negative first argument.
fn tally(_: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    if let Some(Value::Number(first)) = arguments.first()
        && *first < 0.0
    {
        return Err(RuntimeError::ArgumentError {
            position: 1,
            message: "negative".into(),
        }
        .into());
    }
    let sum: f64 = arguments
        .iter()
        .map(|argument| match argument {
            Value::Number(x) => *x,
            _ => 0.0,
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    Ok(vec![
        Value::Number(arguments.len() as f64),
        Value::Number(sum),
    ])
}

/// The fast loop runs host callbacks with up to eight fixed arguments
/// itself, on the main thread. Results must be adjusted to each call's
/// count, a statement call must drop them, nine arguments must take the
/// dispatch loop, and an argument error must name the callee as the
/// dispatch loop names it, which runs every host call in a coroutine.
#[test]
fn host_calls_from_the_fast_loop_match_the_dispatch_loop() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = loaded(CompatibilityProfile::Lua51, mode);
        let function = vm.register_host_function(tally, 0, Vec::new()).unwrap();
        vm.runtime_mut().set_global("tally", function);
        vm.runtime_mut().set_global(
            "pcall",
            Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
        );
        vm.runtime_mut().set_global(
            "wrap",
            Value::BuiltinFunction(BuiltinFunction::CoroutineWrap),
        );
        let chunk = crate::compiler::compile(
            "function drive(n)
               local s = 0
               for i = 1, n do
                 local count, sum = tally(i, i)
                 local nine = tally(1, 2, 3, 4, 5, 6, 7, 8, i)
                 local one = tally(i)
                 local a, b, c = tally()
                 tally(i)
                 s = s + count + sum + nine + one + a + b + (c or 100)
               end
               return s
             end
             local function checked(i) local count = tally(i) return count end
             function probe(i) return pcall(checked, i) end
             function in_coroutine(f, i) return wrap(function() return f(i) end)() end",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
        // Each pass adds 2 + 2i + 9 + 1 + 0 + 0 + 100.
        let expected = vec![Value::Number(112.0 * 300.0 + 300.0 * 301.0)];
        let drive = vm.runtime().get_global("drive");
        let result = call(&mut vm, "drive", vec![Value::Number(300.0)]).unwrap();
        assert_eq!(result, expected, "{mode:?}");
        let result = call(&mut vm, "in_coroutine", vec![drive, Value::Number(300.0)]).unwrap();
        assert_eq!(result, expected, "{mode:?} in a coroutine");
        let probe = vm.runtime().get_global("probe");
        let general = call(&mut vm, "in_coroutine", vec![probe, Value::Number(-1.0)]).unwrap();
        assert_eq!(general[0], Value::Boolean(false), "{mode:?}");
        let Value::String(message) = &general[1] else {
            panic!("{mode:?}: {general:?}");
        };
        assert!(
            message
                .as_bytes()
                .ends_with(b":13: bad argument #1 to 'tally' (negative)"),
            "{mode:?}: {general:?}"
        );
        for _ in 0..50 {
            let result = call(&mut vm, "probe", vec![Value::Number(-1.0)]).unwrap();
            assert_eq!(result, general, "{mode:?}");
        }
    }
}

/// `return host(...)` returns the callback's results from the fast loop as
/// the dispatch loop returns them (it runs every host call in a coroutine):
/// all of them to a caller that takes them all, one to a caller that takes
/// one, the first to an `__index` miss, all of them to the host that called
/// the frame, and an argument error that names the callee.
#[test]
fn host_tail_calls_from_the_fast_loop_match_the_dispatch_loop() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = loaded(CompatibilityProfile::Lua51, mode);
        let function = vm.register_host_function(tally, 0, Vec::new()).unwrap();
        vm.runtime_mut().set_global("tally", function);
        for (name, builtin) in [
            ("pcall", BuiltinFunction::ProtectedCall),
            ("wrap", BuiltinFunction::CoroutineWrap),
        ] {
            vm.runtime_mut()
                .set_global(name, Value::BuiltinFunction(builtin));
        }
        let chunk = crate::compiler::compile(
            "local function both(i) return tally(i, i) end
             local function none() return tally() end
             cache, meta = {}, {__index = function(_, k) return tally(k, 1) end}
             function drive(n)
               local s = 0
               for i = 1, n do
                 local count, sum = both(i)
                 local one = both(i)
                 local a, b, c = none()
                 s = s + count + sum + one + a + b + (c or 100) + cache[i]
               end
               return s
             end
             function probe(i) return pcall(both, i) end
             function direct(i) return tally(i, i) end
             function in_coroutine(f, i) return wrap(function() return f(i) end)() end",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
        // A tail call from the frame the host entered returns to the host.
        let result = call(&mut vm, "direct", vec![Value::Number(5.0)]).unwrap();
        assert_eq!(
            result,
            vec![Value::Number(2.0), Value::Number(10.0)],
            "{mode:?}"
        );
        let (Value::Table(cache), Value::Table(meta)) = (
            vm.runtime().get_global("cache"),
            vm.runtime().get_global("meta"),
        ) else {
            panic!("{mode:?}: tables expected");
        };
        vm.runtime_mut()
            .set_table_metatable(cache, Some(meta))
            .unwrap();
        // Each pass adds 2 + 2i + 2 + 0 + 0 + 100 + 2.
        let expected = vec![Value::Number(106.0 * 300.0 + 300.0 * 301.0)];
        let drive = vm.runtime().get_global("drive");
        let result = call(&mut vm, "drive", vec![Value::Number(300.0)]).unwrap();
        assert_eq!(result, expected, "{mode:?}");
        let result = call(&mut vm, "in_coroutine", vec![drive, Value::Number(300.0)]).unwrap();
        assert_eq!(result, expected, "{mode:?} in a coroutine");
        let probe = vm.runtime().get_global("probe");
        let general = call(&mut vm, "in_coroutine", vec![probe, Value::Number(-1.0)]).unwrap();
        assert_eq!(general[0], Value::Boolean(false), "{mode:?}");
        let Value::String(message) = &general[1] else {
            panic!("{mode:?}: {general:?}");
        };
        assert!(
            message
                .as_bytes()
                .ends_with(b"bad argument #1 to 'tally' (negative)"),
            "{mode:?}: {general:?}"
        );
        for _ in 0..50 {
            let result = call(&mut vm, "probe", vec![Value::Number(-1.0)]).unwrap();
            assert_eq!(result, general, "{mode:?}");
        }
    }
}

/// A host entry into a fixed-parameter function returns from the fast loop
/// straight to the host (`Vm::return_to_host`). Every return shape must
/// give the host what the dispatch loop gives it: no values, several, a
/// closure whose captured local closes on the way out, values only the
/// return holds while the collector runs, a vararg body that takes the
/// general entry, and a return hook that must still fire.
#[test]
fn host_entries_return_from_the_fast_loop_like_the_dispatch_loop() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = loaded(CompatibilityProfile::Lua54, mode);
        let chunk = crate::compiler::compile(
            "function none(x) local y = x end
             function three(a, b) local c = a + b return c, a * b, 'k' .. c end
             function make(n) local c = n return function() c = c + 1 return c end end
             function churn(n)
               local t = {}
               for i = 1, n do t[i] = {'' .. i} end
               return t, #t
             end
             function va(...) local a, b = ... return b, a end
             returns = 0
             function hook() returns = returns + 1 end
             function hooked(x) return x + 1 end",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
        for round in 0..200 {
            let n = f64::from(round);
            assert_eq!(
                call(&mut vm, "none", vec![Value::Number(n)]).unwrap(),
                vec![]
            );
            let result = call(&mut vm, "three", vec![Value::Number(n), Value::Number(2.0)]);
            assert_eq!(
                result.unwrap(),
                vec![
                    Value::Number(n + 2.0),
                    Value::Number(n * 2.0),
                    Value::from(format!("k{}", round + 2).as_str()),
                ],
                "{mode:?} round {round}"
            );
        }
        // Each closure counts on in its own closed cell.
        let first = call(&mut vm, "make", vec![Value::Number(10.0)]).unwrap();
        let second = call(&mut vm, "make", vec![Value::Number(20.0)]).unwrap();
        for step in 1..=50 {
            let a = vm.call_value(first[0].clone(), Vec::new()).unwrap();
            let b = vm.call_value(second[0].clone(), Vec::new()).unwrap();
            let step = f64::from(step);
            assert_eq!(a, vec![Value::Number(10.0 + step)], "{mode:?}");
            assert_eq!(b, vec![Value::Number(20.0 + step)], "{mode:?}");
        }
        // The returned table is only in the returned values when the
        // return's collector check runs.
        for _ in 0..20 {
            let kept = call(&mut vm, "churn", vec![Value::Integer(2000)]).unwrap();
            assert_eq!(kept[1], Value::Integer(2000), "{mode:?}");
            vm.collect_full_with_values(&kept);
            let Value::Table(table) = kept[0] else {
                panic!("{mode:?}: {kept:?}");
            };
            let row = vm
                .runtime()
                .table_get(table, &Value::Integer(1999))
                .unwrap();
            let Value::Table(row) = row else {
                panic!("{mode:?}: row 1999 is {row:?}");
            };
            let text = vm.runtime().table_get(row, &Value::Integer(1)).unwrap();
            assert_eq!(text, Value::from("1999"), "{mode:?}");
        }
        let result = call(&mut vm, "va", vec![Value::Number(1.0), Value::Nil]).unwrap();
        assert_eq!(result, vec![Value::Nil, Value::Number(1.0)], "{mode:?}");
        let hook = vm.runtime().get_global("hook");
        vm.debug_hooks.insert(
            None,
            crate::vm::DebugHook {
                function: hook,
                mask: crate::vm::DebugHookMask {
                    return_: true,
                    ..Default::default()
                },
                count: 0,
                remaining: 0,
                running: false,
            },
        );
        vm.refresh_return_watch();
        for _ in 0..10 {
            let result = call(&mut vm, "hooked", vec![Value::Number(1.0)]).unwrap();
            assert_eq!(result, vec![Value::Number(2.0)], "{mode:?}");
        }
        let Value::Number(returns) = vm.runtime().get_global("returns") else {
            panic!("{mode:?}: returns is not a number");
        };
        assert!(returns >= 10.0, "{mode:?}: {returns} return hooks");
        assert!(vm.host_returned.is_empty(), "{mode:?}");
    }
}

/// Reads field `k` of its first argument after a full collection and
/// returns it joined to its second, and whether the call left any suspended
/// roots.
fn collect_then_read(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    vm.collect_full_with_values(&[]);
    let Value::Table(table) = arguments[0] else {
        panic!("the first argument is a table");
    };
    let Value::String(field) = vm.runtime().table_get(table, &Value::from("k"))? else {
        panic!("field k survives the collection");
    };
    let mut joined = field.as_bytes().to_vec();
    joined.extend_from_slice(format!(":{:?}", arguments[1]).as_bytes());
    Ok(vec![
        Value::from(joined),
        Value::Boolean(vm.suspended_roots.is_empty()),
    ])
}

/// A host callback the fast loop calls reads its arguments in the parked
/// stack, and no suspended roots are built for them: the parked registers
/// keep a table that only the call's argument register holds through a
/// collection the callback runs.
#[test]
fn fast_loop_host_call_arguments_survive_a_collection_in_the_callback() {
    let source = "
        local out, lean = {}, 0
        for i = 1, 50 do
          local s, empty = read({k = 'field ' .. i}, i)
          out[i] = s
          if empty then lean = lean + 1 end
        end
        return out[1], out[50], lean
    ";
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_execution_mode(mode);
        let host = vm
            .register_host_function(collect_then_read, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("read", host);
        let chunk = crate::compiler::compile(source).unwrap();
        let results = vm.execute(&chunk).unwrap();
        assert_eq!(results[0], Value::from("field 1:Number(1.0)"), "{mode:?}");
        assert_eq!(results[1], Value::from("field 50:Number(50.0)"), "{mode:?}");
        if mode == ExecutionMode::Interpreter {
            assert_eq!(results[2], Value::Number(50.0), "{mode:?}");
        }
        assert!(vm.host_snapshots.is_empty(), "{mode:?}");
    }
}

/// Calls the global `clear`, which assigns the captured local passed in as
/// the first argument, then returns that argument.
fn clear_then_return(vm: &mut Vm, _: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    let clear = vm.runtime().get_global("clear");
    vm.call_value(clear, Vec::new())?;
    Ok(vec![arguments[0].clone()])
}

/// A captured local passed as the sole argument of a host call (a
/// `CallOne` naming the local's register) keeps the value the call was made
/// with while the callback re-enters Lua and clears the local through its
/// open upvalue cell, which writes the parked stack.
#[test]
fn fast_loop_host_call_argument_survives_its_captured_local_being_cleared() {
    let source = "
        local same = 0
        for i = 1, 50 do
          local x = 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy' .. i
          clear = function() x = nil end
          local seen = probe(x)
          if seen == 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy' .. i and x == nil then same = same + 1 end
        end
        return same
    ";
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_execution_mode(mode);
        let host = vm
            .register_host_function(clear_then_return, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("probe", host);
        let chunk = crate::compiler::compile(source).unwrap();
        let results = vm.execute(&chunk).unwrap();
        assert_eq!(results[0], Value::Number(50.0), "{mode:?}");
        assert!(vm.host_snapshots.is_empty(), "{mode:?}");
    }
}
