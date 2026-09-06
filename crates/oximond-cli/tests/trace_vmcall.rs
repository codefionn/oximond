//! Host C calls made from inside a trace: a hot loop whose body calls a host
//! function (registered through `Vm::register_host_function`, as the C API
//! does) runs compiled, and the trace calls the function itself through the
//! VM's host boundary without leaving it.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results (errors and tracebacks included) must
//! match. Counters show that the calls were made from native code
//! (`host_trace_calls`). A call the trace cannot hold (a function that
//! changes what the trace read, results that are not numbers or booleans, a
//! table set or a written cell beside the call) leaves the trace, or its
//! recording aborts at the call and the loop stays with the interpreter; the
//! results are the same.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value, Vm, VmError};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn has_integers(profile: CompatibilityProfile) -> bool {
    !matches!(
        profile,
        CompatibilityProfile::Lua51 | CompatibilityProfile::Lua52 | CompatibilityProfile::LuaJit21
    )
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(value) => Some(*value as f64),
        Value::Number(value) => Some(*value),
        _ => None,
    }
}

/// A whole number as the profile has it: an integer where there are
/// integers (the host registers with token 1 there), a float elsewhere.
fn whole(token: u64, value: i64) -> Value {
    if token == 1 {
        Value::Integer(value)
    } else {
        #[allow(clippy::cast_precision_loss)]
        Value::Number(value as f64)
    }
}

/// `cadd(a, b)`: the sum, an integer when both are.
fn cadd(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    Ok(vec![match (arguments.first(), arguments.get(1)) {
        (Some(Value::Integer(a)), Some(Value::Integer(b))) => Value::Integer(a.wrapping_add(*b)),
        (a, b) => match (number(a), number(b)) {
            (Some(a), Some(b)) => Value::Number(a + b),
            _ => Value::Nil,
        },
    }])
}

/// `cfail(n)`: raises an error when `n` is 150.
fn cfail(_: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    if number(arguments.first()) == Some(150.0) {
        return Err(VmError::Runtime(oximond::runtime::RuntimeError::message(
            "host function failed",
        )));
    }
    Ok(vec![whole(token, 1)])
}

/// `cbump()`: calls the Lua global `bump`, which changes state the loop
/// reads, and returns what it returns.
fn cbump(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    let bump = vm.runtime().get_global("bump");
    vm.call_value(bump, Vec::new())
}

/// `cmeta(t)`: gives `t` a metatable whose `__index` answers 7, through the
/// Lua global `setmetatable`.
fn cmeta(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let set = vm.runtime().get_global("setmetatable");
    let make = vm.runtime().get_global("make_meta");
    let meta = vm.call_value(make, Vec::new())?;
    let mut call = vec![arguments[0].clone()];
    call.extend(meta);
    vm.call_value(set, call)?;
    Ok(vec![whole(token, 1)])
}

/// `cgc(x)`: collects everything and returns its argument.
fn cgc(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    vm.collect_full_with_values(arguments);
    Ok(vec![arguments.first().cloned().unwrap_or(Value::Nil)])
}

/// `ctrace()`: `debug.traceback` as seen from inside the host function.
fn ctrace(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    let debug = vm.runtime().get_global("debug");
    let traceback = vm.get_value(debug, Value::from("traceback"))?;
    vm.call_value(traceback, vec![Value::from("tb")])
}

/// `cline()`: the current line of the Lua caller, from `debug.getinfo`.
fn cline(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    let debug = vm.runtime().get_global("debug");
    let getinfo = vm.get_value(debug, Value::from("getinfo"))?;
    let info = vm.call_value(getinfo, vec![Value::Integer(2), Value::from("lS")])?;
    let Some(table) = info.first().cloned() else {
        return Ok(vec![Value::Nil]);
    };
    let line = vm.get_value(table.clone(), Value::from("currentline"))?;
    let what = vm.get_value(table, Value::from("short_src"))?;
    Ok(vec![line, what])
}

/// `cyield(x)`: yields `x` to the resumer of the running coroutine.
fn cyield(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    Err(VmError::Yield(arguments.to_vec()))
}

/// `cre(n)`: calls the Lua global `again` with `n`, which runs the same
/// loop the caller is in.
fn cre(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let again = vm.runtime().get_global("again");
    vm.call_value(again, arguments.to_vec())
}

/// `cstr(n)`: a string result, which no word of a trace carries.
fn cstr(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    Ok(vec![Value::from(
        format!("s{}", number(arguments.first()).unwrap_or(0.0)).as_str(),
    )])
}

/// `chook()`: installs a line hook through the Lua global `sethook`.
fn chook(vm: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    let set = vm.runtime().get_global("sethook");
    vm.call_value(set, Vec::new())
}

/// `csetk()`: adds 5 to the local `k` of the calling Lua function, found by
/// name among the first levels.
fn csetk(vm: &mut Vm, token: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    for level in 0..4 {
        for index in 1..20 {
            let Some((name, value)) = vm.host_get_local(level, index)? else {
                break;
            };
            if name.as_bytes() == b"k"
                && let Some(k) = number(Some(&value))
            {
                #[allow(clippy::cast_possible_truncation)]
                vm.host_set_local(level, index, whole(token, k as i64 + 5))?;
                return Ok(vec![whole(token, 1)]);
            }
        }
    }
    Ok(vec![whole(token, 0)])
}

/// `ctwo(a)`: two results, the double of the argument and its quarter as a
/// float.
fn ctwo(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let n = number(arguments.first()).unwrap_or(0.0);
    let double = match arguments.first() {
        Some(Value::Integer(integer)) => Value::Integer(integer.wrapping_mul(2)),
        _ => Value::Number(n * 2.0),
    };
    Ok(vec![double, Value::Number(n / 4.0)])
}

/// `cflip(n)`: a boolean, true for even `n`.
fn cflip(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    Ok(vec![Value::Boolean(
        number(arguments.first()).unwrap_or(0.0) as i64 % 2 == 0,
    )])
}

/// `cmaybe(n)`: a number, but nil when `n` is 777.
#[allow(
    clippy::float_cmp,
    reason = "777.0 is the exact sentinel the calling script sends"
)]
fn cmaybe(_: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let n = number(arguments.first()).unwrap_or(0.0);
    Ok(vec![if n == 777.0 {
        Value::Nil
    } else {
        Value::Number(n + 1.0)
    }])
}

/// Calls `debug.<name>(arguments)` from the host. A profile whose debug
/// library lacks the function (`upvaluejoin` in Lua 5.1) gets `fallback`.
fn debug_call(
    vm: &mut Vm,
    name: &str,
    arguments: Vec<Value>,
    fallback: Option<(&str, Vec<Value>)>,
) -> Result<(), VmError> {
    let debug = vm.runtime().get_global("debug");
    let function = vm.get_value(debug, Value::from(name))?;
    if matches!(function, Value::Nil) {
        if let Some((name, arguments)) = fallback {
            return debug_call(vm, name, arguments, None);
        }
        return Ok(());
    }
    vm.call_value(function, arguments)?;
    Ok(())
}

/// `cjoin(i)`: at set iterations, changes from the host what the running
/// loop reads through the upvalue `a` (upvalue `IA` of `RUNA`, a table) and
/// `k` (upvalue `IK` of `RUNK`, a number), and what the table `HOLD` holds:
///
/// - 1000: joins `a` to the cell of `SRCA`'s first upvalue (another table);
/// - 1600: joins `k` to the cell of `SRCK`'s first upvalue;
/// - 2200: stores a fresh table into the joined cell of `a`;
/// - 2800: rebinds `HOLD.inner` to a fresh table;
/// - 3400: stores a fresh table into `HOLD.inner.t`, which the run reads
///   through two tables.
///
/// Profiles without `debug.upvaluejoin` set the value instead.
fn cjoin(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let i = number(arguments.first()).unwrap_or(0.0) as i64;
    let global = |vm: &mut Vm, name: &str| vm.runtime().get_global(name);
    let fresh = |vm: &mut Vm, base: i64| -> Result<Value, VmError> {
        let table = vm.runtime_mut().create_table_value()?;
        for n in 1..=3 {
            vm.set_value(table.clone(), whole(token, n), whole(token, base * n))?;
        }
        vm.set_value(table.clone(), Value::from("x"), whole(token, base + 1))?;
        Ok(table)
    };
    match i {
        1000 | 1600 => {
            let (run, index, source) = if i == 1000 {
                ("RUNA", "IA", "SRCA")
            } else {
                ("RUNK", "IK", "SRCK")
            };
            let run = global(vm, run);
            let index = global(vm, index);
            let source = global(vm, source);
            let debug = global(vm, "debug");
            let getupvalue = vm.get_value(debug, Value::from("getupvalue"))?;
            let value = vm.call_value(getupvalue, vec![source.clone(), whole(token, 1)])?;
            let value = value.get(1).cloned().unwrap_or(Value::Nil);
            debug_call(
                vm,
                "upvaluejoin",
                vec![run.clone(), index.clone(), source, whole(token, 1)],
                Some(("setupvalue", vec![run, index, value])),
            )?;
        }
        2200 => {
            let table = fresh(vm, 40)?;
            let run = global(vm, "RUNA");
            let index = global(vm, "IA");
            debug_call(vm, "setupvalue", vec![run, index, table], None)?;
        }
        2800 => {
            let table = fresh(vm, 9)?;
            let inner = fresh(vm, 3)?;
            vm.set_value(table.clone(), Value::from("t"), inner)?;
            let hold = global(vm, "HOLD");
            vm.set_value(hold, Value::from("inner"), table)?;
        }
        3400 => {
            let table = fresh(vm, 11)?;
            let hold = global(vm, "HOLD");
            let inner = vm.get_value(hold, Value::from("inner"))?;
            vm.set_value(inner, Value::from("t"), table)?;
        }
        _ => {}
    }
    Ok(vec![whole(token, i % 4)])
}

/// `cnil(n)`: nil.
fn cnil(_: &mut Vm, _: u64, _: &[Value]) -> Result<Vec<Value>, VmError> {
    Ok(vec![Value::Nil])
}

/// A fresh table with the fields `biome`, `n` and `name`, as the engine's
/// `core.get_biome_data` gives.
fn biome_table(vm: &mut Vm, token: u64, n: i64) -> Result<Value, VmError> {
    let table = vm.runtime_mut().create_table_value()?;
    vm.set_value(table.clone(), Value::from("biome"), whole(token, n % 5 + 1))?;
    vm.set_value(table.clone(), Value::from("n"), whole(token, n))?;
    vm.set_value(
        table.clone(),
        Value::from("name"),
        Value::from(format!("b{}", n % 3).as_str()),
    )?;
    Ok(table)
}

/// `cbiome(n)`: a fresh table.
fn cbiome(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let n = number(arguments.first()).unwrap_or(0.0) as i64;
    Ok(vec![biome_table(vm, token, n)?])
}

/// `ckeep(n)`: a fresh table that it hands to the Lua global `keepit`
/// before it returns it, so Lua code holds the table the call gave.
fn ckeep(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let n = number(arguments.first()).unwrap_or(0.0) as i64;
    let table = biome_table(vm, token, n)?;
    let keep = vm.runtime().get_global("keepit");
    vm.call_value(keep, vec![table.clone()])?;
    Ok(vec![table])
}

/// `cdata(v)`: like `core.get_biome_data`, a fresh table for the position
/// the fields `x`, `y` and `z` of the table `v` name.
fn cdata(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let Some(v) = arguments.first().cloned() else {
        return Ok(vec![Value::Nil]);
    };
    let mut sum = 0.0;
    for field in ["x", "y", "z"] {
        sum += number(Some(&vm.get_value(v.clone(), Value::from(field))?)).unwrap_or(0.0);
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(vec![biome_table(vm, token, (sum as i64).rem_euclid(1000))?])
}

/// `cgctab(n)`: a fresh table, made after a full collection, so the table
/// is live nowhere but in the caller's registers and the trace.
fn cgctab(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let n = number(arguments.first()).unwrap_or(0.0) as i64;
    let table = biome_table(vm, token, n)?;
    vm.collect_full_with_values(std::slice::from_ref(&table));
    Ok(vec![table])
}

/// `cvary(n)`: a table below 1500, a string below 2500, nil after.
fn cvary(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let n = number(arguments.first()).unwrap_or(0.0) as i64;
    Ok(vec![if n < 1500 {
        biome_table(vm, token, n)?
    } else if n < 2500 {
        Value::from(format!("v{n}").as_str())
    } else {
        Value::Nil
    }])
}

/// `cname(name)`: an integer for a string, as `core.get_content_id` gives.
fn cname(_: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let length = match arguments.first() {
        Some(Value::String(text)) => text.len() as i64,
        _ => -1,
    };
    Ok(vec![whole(token, length * 3 + 1)])
}

/// `cdyn(n)`: one result, but two for 1777 and none for 2777.
fn cdyn(_: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    #[allow(clippy::cast_possible_truncation)]
    let n = number(arguments.first()).unwrap_or(0.0) as i64;
    Ok(match n {
        1777 => vec![whole(token, 1), whole(token, 2)],
        2777 => Vec::new(),
        _ => vec![whole(token, n % 9)],
    })
}

/// `crawset(t, k, v)`: a raw store, which may resize the array part.
fn crawset(vm: &mut Vm, _: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    let set = vm.runtime().get_global("rawset");
    vm.call_value(set, arguments.to_vec())?;
    Ok(vec![arguments.get(2).cloned().unwrap_or(Value::Nil)])
}

/// `cerr2()`: raises an error at level 2 through the Lua global `error`.
fn cerr2(vm: &mut Vm, token: u64, arguments: &[Value]) -> Result<Vec<Value>, VmError> {
    if number(arguments.first()) != Some(2200.0) {
        return Ok(vec![whole(token, 1)]);
    }
    let error = vm.runtime().get_global("error");
    vm.call_value(error, vec![Value::from("deep"), Value::Integer(2)])
}

struct Report {
    outcome: String,
    stats: TraceStatistics,
    calls: u64,
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Report {
    run_with(profile, mode, source, true)
}

/// [`run`] with the trace's calls of host functions on or off.
fn run_with(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
    vmcall: bool,
) -> Report {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    // The tests run small loops: the bar of work beside a call is out of
    // the way (see `the_default_asks_for_work_in_the_loop`).
    lua.vm_mut().set_vmcall_min_work(0);
    if !vmcall {
        lua.vm_mut().set_trace_vmcall(false);
    }
    let hosts: [(&str, oximond::vm::HostCallback); 27] = [
        ("cjoin", cjoin),
        ("cdyn", cdyn),
        ("cnil", cnil),
        ("cbiome", cbiome),
        ("cgctab", cgctab),
        ("cvary", cvary),
        ("cname", cname),
        ("crawset", crawset),
        ("cerr2", cerr2),
        ("cgetbiome", cbiome),
        ("csetk", csetk),
        ("cadd", cadd),
        ("cfail", cfail),
        ("cbump", cbump),
        ("cmeta", cmeta),
        ("cgc", cgc),
        ("ctrace", ctrace),
        ("cline", cline),
        ("cyield", cyield),
        ("cre", cre),
        ("cstr", cstr),
        ("chook", chook),
        ("ctwo", ctwo),
        ("cflip", cflip),
        ("cmaybe", cmaybe),
        ("ckeep", ckeep),
        ("cdata", cdata),
    ];
    for (name, callback) in hosts {
        let function = lua
            .vm_mut()
            .register_host_function(callback, u64::from(has_integers(profile)), Vec::new())
            .unwrap();
        lua.vm_mut().runtime_mut().set_global(name, function);
    }
    let outcome = match lua.exec(source) {
        Ok(results) => format!("{results:?}"),
        Err(error) => format!("error: {error}"),
    };
    let statistics = lua.vm().jit_statistics();
    Report {
        outcome,
        calls: statistics.host_trace_calls,
        stats: statistics.trace,
    }
}

/// Runs `source` in every profile under both engines, checks the results
/// agree, and returns each profile's tracing report.
fn agree(source: &str) -> Vec<(CompatibilityProfile, Report)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let expected = run(profile, ExecutionMode::Interpreter, source);
            let traced = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(
                traced.outcome, expected.outcome,
                "{profile:?}\n{source}\n{:?}",
                traced.stats
            );
            assert_eq!(expected.calls, 0, "the interpreter makes no trace call");
            (profile, traced)
        })
        .collect()
}

fn assert_called(profile: CompatibilityProfile, report: &Report, least: u64) {
    assert!(
        report.calls >= least,
        "{profile:?}: {} host calls from traces, wanted {least}\n{:?}",
        report.calls,
        report.stats
    );
}

/// A call the trace cannot hold: the first trace that makes it leaves after
/// it, and the next recording of the loop aborts at the call, so traces make
/// few of the loop's calls.
fn assert_declined(profile: CompatibilityProfile, report: &Report) {
    assert!(
        report.calls < 100,
        "{profile:?}: {} host calls from traces\n{:?}",
        report.calls,
        report.stats
    );
}

/// Calls of a host function inside a loop run from the compiled trace, with
/// the Lua work on both sides, for every kind of result the trace takes.
#[test]
fn a_loop_calls_the_host_function_from_its_trace() {
    let source = "local s, t = 0, 1
        for i = 1, 4000 do
            local x = i * 2 + 1
            local y = x * 3 - i + (x % 5)
            s = cadd(s, y)
            local a, b = ctwo(x)
            if cflip(i) then t = t + 1 end
            s = s + a + b - x + (i % 7) * 2
        end
        return s, t";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 3000);
        assert!(report.stats.traces_compiled >= 1, "{profile:?}");
    }
}

/// `while`, `repeat` and generic `for` loops, and a loop that counts down.
#[test]
fn every_loop_kind_calls_from_its_trace() {
    let loops = [
        "local i, s = 0, 0
         while i < 4000 do
            i = i + 1
            local x = i * 5 + 1
            s = cadd(s, x) + i % 3
            s = s - x + 2 * i
         end
         return s",
        "local i, s = 0, 0
         repeat
            i = i + 1
            local x = i * 5 + 1
            s = cadd(s, x) + i % 3
            s = s - x + 2 * i
         until i >= 4000
         return s",
        "local list, s = {}, 0
         for i = 1, 4000 do list[i] = i * 3 end
         for _, v in ipairs(list) do
            s = cadd(s, v * 5 + 1) + v % 3
         end
         return s",
        "local s = 0
         for i = 4000, 1, -1 do
            local x = i * 5 + 1
            s = cadd(s, x) + i % 3
            s = s - x + 2 * i
         end
         return s",
    ];
    for source in loops {
        for (profile, report) in agree(source) {
            // A loop over a table binds it, which the function could change:
            // those stay with the interpreter and give the same results.
            if source.contains("ipairs") {
                continue;
            }
            assert_called(profile, &report, 3000);
        }
    }
}

/// Two host calls in one body.
#[test]
fn two_host_calls_in_one_body() {
    let source = "local s, t = 0, 1
        for i = 1, 4000 do
            local x = i * 2 + 1
            s = cadd(s, x)
            local y = s * 3 - i + x * 5
            t = cadd(t, y % 11)
            s = s - y + t % 7 + i * 3
        end
        return s, t";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 6000);
    }
}

/// A host function that raises an error: the message, the traceback and the
/// values the loop left behind are the interpreter's.
#[test]
fn host_error_matches_the_interpreter() {
    let source = "local ok, message = pcall(function()
            local s, last = 0, 0
            for i = 1, 400 do
                local x = i * 2 + 1
                last = i
                s = s + cfail(i) + x % 3
                s = s - x + i * 4
            end
            return s
        end)
        local ok2, trace = xpcall(function()
            local s = 0
            for i = 1, 400 do
                local x = i * 2 + 1
                s = s + cfail(i) + x % 3
                s = s - x + i * 4
            end
            return s
        end, debug.traceback)
        return ok, message, ok2, trace";
    for (profile, report) in agree(source) {
        assert!(
            report.outcome.contains("host function failed"),
            "{profile:?}"
        );
        assert_called(profile, &report, 100);
    }
    let bare = "local s = 0
        for i = 1, 400 do
            local x = i * 2 + 1
            s = s + cfail(i) + x % 3
            s = s - x + i * 4
        end
        return s";
    for (profile, report) in agree(bare) {
        assert!(
            report.outcome.starts_with("error:"),
            "{profile:?} {}",
            report.outcome
        );
    }
}

/// The callee changes under the trace: a Lua function, another host
/// function, nil.
#[test]
fn a_changed_callee_runs_in_the_interpreter() {
    let source = "local f, s = cadd, 0
        local calls = 0
        local function lua_add(a, b) calls = calls + 1 return a + b + 1 end
        for i = 1, 3000 do
            if i == 1000 then f = lua_add end
            if i == 2000 then f = cfail end
            local x = i * 2 + 1
            s = f(s, x) + i % 5
            s = s - x + i * 2
        end
        return s, calls";
    agree(source);
    let global = "local s = 0
        for i = 1, 3000 do
            if i == 1500 then cadd = function(a, b) return a * 0 + b end end
            local x = i * 2 + 1
            s = cadd(s, x) + i % 5
            s = s - x + i * 2
        end
        return s";
    agree(global);
    let other = "local f, s = cadd, 0
        for i = 1, 3000 do
            if i % 700 == 0 then f = cflip else f = cadd end
            local x = i * 2 + 1
            s = s + (f(s, x) == true and 1 or 0) + i % 5
            s = s - x + i * 2
        end
        return s";
    agree(other);
}

/// The host function itself, through a Lua callback, reassigns the callee the
/// loop calls through: a captured local and a global. The trace makes the
/// call, finds the callee changed and leaves.
#[test]
fn the_host_function_changes_the_callee() {
    let local = "local f, s, n = cadd, 0, 0
        local function other(a, b) return a + b + 1 end
        function bump() n = n + 1 if n == 1000 then f = other end if n == 2000 then f = cfail end return n end
        for i = 1, 3000 do
            local x = i * 2 + 1
            local k = cbump()
            s = f(s, x) + i % 5 + k % 2
            s = s - x + i * 2
        end
        return s, n";
    let global = "local s, n = 0, 0
        function bump() n = n + 1 if n == 1200 then cadd = function(a, b) return a * 0 + b end end return n end
        for i = 1, 3000 do
            local x = i * 2 + 1
            local k = cbump()
            s = cadd(s, x) + i % 5 + k % 2
            s = s - x + i * 2
        end
        return s, n";
    let swapped = "local f, s, n = cadd, 0, 0
        function bump() n = n + 1 if n == 1500 then f = cflip end return n end
        for i = 1, 3000 do
            local x = i * 2 + 1
            local k = cbump()
            s = s + (f(s, x) == true and 1 or 0) + i % 5 + k % 2
            s = s - x + i * 2
        end
        return s, n";
    for source in [local, global, swapped] {
        for (profile, report) in agree(source) {
            assert!(
                report.calls >= 1,
                "{profile:?}\n{source}\n{:?}",
                report.stats
            );
        }
    }
}

/// The host function changes what the trace read. A global, a captured local
/// and an array's length are read again after the call, and the trace makes
/// the call.
#[test]
fn the_call_changes_state_the_trace_read() {
    let upvalue = "local counter = 0
        local n = 0
        function bump() n = n + 1 if n % 500 == 0 then counter = counter + 1 end return counter end
        local s = 0
        for i = 1, 3000 do
            local a = counter * 2 + i
            local k = cbump()
            s = s + k + a
            s = s + counter - a + i % 3
        end
        return s, counter";
    let setupvalue = "local counter, n = 0, 0
        local function bump()
            n = n + 1
            if n % 1000 == 0 then
                local i = 1
                while true do
                    local name = debug.getupvalue(bump, i)
                    if name == nil then break end
                    if name == 'counter' then debug.setupvalue(bump, i, 100 + n) end
                    i = i + 1
                end
            end
            return n % 7 + counter - counter
        end
        _G.bump = bump
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            s = s + k + i % 3
            s = s + counter * 2 - i
        end
        return s, counter, n";
    let length = "local t = {10, 20, 30, 40}
        local n = 0
        function bump()
            n = n + 1
            if n == 1200 then setmetatable(t, {__len = function() return 2 end}) end
            return n
        end
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            s = s + k % 5 + #t + t[i % 4 + 1]
            s = s - i % 7
        end
        return s, n";
    for source in [upvalue, setupvalue, length] {
        for (profile, report) in agree(source) {
            assert!(
                report.calls >= 1,
                "{profile:?}\n{source}\n{:?}",
                report.stats
            );
        }
    }
}

/// The function changes a table field, an array slot and a table's length
/// that the loop reads. The trace holds field reads across no call, so
/// the loop stays with the interpreter.
#[test]
fn the_call_changes_a_field_the_trace_read() {
    let source = "local counter, shared = 0, {n = 1, 10, 20, 30}
        function bump()
            counter = counter + 1
            shared.n = shared.n + 1
            shared[1] = shared[1] + 1
            shared[#shared + 1] = counter
            return counter
        end
        local s = 0
        for i = 1, 3000 do
            local a = counter * 2 + shared.n + shared[1]
            s = s + cbump() + a
            s = s + counter + shared.n + shared[1] + #shared
        end
        return s, counter, shared.n, #shared";
    for (profile, report) in agree(source) {
        assert_declined(profile, &report);
    }
}

/// The loop reads and writes fields of tables on both sides of the call. The
/// trace makes the call and reads the fields again after it.
#[test]
fn fields_are_read_and_written_on_both_sides_of_the_call() {
    let quiet = "local pts, ts = {}, {{n = 0, x = 1.5}}
        for i = 1, 64 do pts[i] = {a = i, b = i * 2} end
        local s = 0
        for i = 1, 4000 do
            local t = ts[1]
            local p = pts[i % 64 + 1]
            local a = p.a
            s = s + cadd(a, i) % 11
            s = s + p.b + t.n + p.a + t.x
        end
        return s, ts[1].n, pts[3].b";
    for (profile, report) in agree(quiet) {
        assert_called(profile, &report, 3000);
    }
}

/// The function behind the call changes what the loop read: a field's value
/// and type, a table's hash part (a rehash), a metatable and a metatable's
/// handler. Each loop gives the interpreter's results.
#[test]
fn the_call_changes_fields_the_trace_reads_again() {
    let retype = "local t = {n = 0, x = 1.5}
        local calls = 0
        function bump()
            calls = calls + 1
            if calls == 1500 then t.n = 0.5 end
            if calls == 2500 then t.n = 7 end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local a = t.n
            local k = cbump()
            s = s + k % 5 + a + t.n + t.x
        end
        return s, t.n, t.x";
    let rehash = "local pts = {}
        for i = 1, 16 do pts[i] = {a = i, b = i * 2} end
        local calls = 0
        function bump()
            calls = calls + 1
            local p = pts[calls % 16 + 1]
            for j = 1, 3 do p['k' .. calls .. '_' .. j] = j end
            if calls % 3 == 0 then p.k = nil end
            p.a = p.a + 1000
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local p = pts[i % 16 + 1]
            local a = p.a
            local k = cbump()
            s = s + k % 5 + a + p.a + p.b
        end
        return s, pts[2].b";
    let meta = "local pts = {}
        for i = 1, 8 do pts[i] = {a = i} end
        local calls = 0
        local handlers = {__index = function(t, k) return 100 end}
        function bump()
            calls = calls + 1
            if calls == 1200 then
                for i = 1, 8 do setmetatable(pts[i], handlers) end
            end
            if calls == 2400 then
                handlers.__index = function(t, k) return 200 end
            end
            if calls == 3000 then
                for i = 1, 8 do pts[i].a = nil end
            end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local p = pts[i % 8 + 1]
            local k = cbump()
            s = s + k % 5 + p.a
        end
        return s";
    let gone = "local pts = {}
        for i = 1, 8 do pts[i] = {a = i, b = 1} end
        local calls = 0
        function bump()
            calls = calls + 1
            if calls == 2000 then pts[3] = {a = 5, b = 6, c = 7} end
            if calls == 2500 then pts[4] = nil end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local p = pts[i % 8 + 1]
            if p then
                local a = p.a
                local k = cbump()
                s = s + k % 5 + a + p.b
            end
        end
        return s";
    // Stores beside the call stay with the interpreter; the results are the same.
    let stores = "local pts, t = {}, {n = 0}
        for i = 1, 16 do pts[i] = {a = i, b = i * 2} end
        function bump()
            t.n = t.n + 1
            pts[t.n % 16 + 1].b = t.n
            return t.n
        end
        local s = 0
        for i = 1, 4000 do
            local p = pts[i % 16 + 1]
            local k = cbump()
            p.b = p.b + 1
            s = s + k % 5 + p.a + p.b
            t.n = t.n + 1
        end
        return s, t.n";
    agree(stores);
    // A field of a table the loop holds in a local is read through the slot
    // the entry bound. A host call could move the slot (a rehash), so the
    // trace refuses such a loop. The results agree.
    for source in [retype, rehash, meta, gone] {
        agree(source);
    }
}

/// The host function writes and grows an array the trace reads by index:
/// the trace reads the array's address and length again after the call.
#[test]
fn the_call_grows_an_array_the_trace_reads() {
    let source = "local t = {}
        for i = 1, 8 do t[i] = i * 10 end
        function bump()
            t[1] = t[1] + 1
            t[#t + 1] = #t
            return t[1]
        end
        local s = 0
        for i = 1, 2000 do
            local a = t[i % 8 + 1]
            local k = cbump()
            local b = t[i % 8 + 1]
            s = s + a + b + k % 3 + #t % 5
        end
        return s, #t, t[1]";
    for (profile, report) in agree(source) {
        assert!(report.stats.traces_compiled >= 1, "{profile:?}");
    }
    let shrink = "local t = {}
        for i = 1, 3000 do t[i] = i end
        function bump()
            t[#t] = nil
            t[#t] = nil
            return #t
        end
        local s = 0
        for i = 1, 1000 do
            local a = t[i % 500 + 1]
            local k = cbump()
            local b = t[i % 500 + 1]
            s = s + a + b + k % 3
        end
        return s, #t";
    agree(shrink);
}

/// A string read out of an array slot, held across the call: the function
/// may move the array, so the recording gives the string a lane of its own
/// before the call, and the call is made from the trace.
#[test]
fn a_string_borrowed_from_an_array_across_the_call() {
    let source = "local keys = {}
        for i = 1, 16 do keys[i] = 'key' .. i end
        local s = 0
        for i = 1, 2000 do
            local k = keys[i % 16 + 1]
            local n = cadd(i, 1)
            s = s + #k + n % 3 + #keys[i % 16 + 1]
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 1500);
    }
}

/// The host function gives a table a metatable the trace did not see: once
/// through a Lua callback (a `__len` the trace reads around the call), once
/// through the host function `cmeta`, whose `__index` reads the loop leaves
/// to the interpreter.
#[test]
fn the_call_changes_a_metatable() {
    let callback = "local t = {1, 2, 3, 4}
        local n = 0
        function bump()
            n = n + 1
            if n == 1200 then setmetatable(t, {__len = function() return 3 end}) end
            return n
        end
        local s = 0
        for i = 1, 3000 do
            local a = #t
            local k = cbump()
            local b = #t
            s = s + a + b + t[i % 4 + 1] + k % 3
            s = s - i + 4
        end
        return s, n";
    for (profile, report) in agree(callback) {
        assert!(report.calls >= 1, "{profile:?}\n{:?}", report.stats);
    }
    let direct = "function make_meta() return {__index = function(t, k) return 7 end} end
        local t = {1, 2, 3, 4}
        local s, n = 0, 0
        for i = 1, 3000 do
            local a = t[i % 6 + 1]
            n = n + 1
            local k = cmaybe(n)
            if i == 2000 then cmeta(t) end
            local b = t[i % 6 + 1]
            s = s + (a or 0) + (b or 0) + i % 3 + (k or 0)
            s = s - i + 4
        end
        return s";
    agree(direct);
}

/// Values live across the call stay rooted while the host function collects.
#[test]
fn values_live_across_the_call_survive_a_collection() {
    let source = "local keep, s = {}, 0
        for i = 1, 3000 do
            local t = {i, i + 1}
            local str = 'v' .. (i % 10)
            local x = i * 2 + 1
            local back = cgc(t)
            if i % 500 == 0 then cgc(false) end
            keep[i % 8 + 1] = t
            s = s + back[1] + back[2] + #str + x % 3
            s = s - x + i * 2
        end
        local n = 0
        for _, v in pairs(keep) do n = n + v[1] end
        return s, n";
    // Fresh tables and a table result are no value of the trace: the loop
    // stays with the interpreter.
    for (profile, report) in agree(source) {
        assert_declined(profile, &report);
    }
    let strings = "local label, s = 'alpha', 0
        local word = string.rep('ab', 3)
        for i = 1, 3000 do
            local x = i * 2 + 1
            local back = cgc(x)
            s = s + #label + #word + back % 3
            s = s - x + i * 2
            if label ~= 'alpha' or word ~= 'ababab' then s = -1 end
        end
        return s, label, word";
    for (profile, report) in agree(strings) {
        assert_called(profile, &report, 1000);
    }
    let scalars = "local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            local y = x * 1.5
            local flag = i % 3 == 0
            cgc(i)
            s = s + x + y + (flag and 1 or 0)
        end
        return s";
    for (profile, report) in agree(scalars) {
        assert_called(profile, &report, 1000);
    }
}

/// `debug.traceback` and `debug.getinfo` through the host frame.
#[test]
fn the_debug_library_sees_the_host_frame() {
    let source = "local out, s = {}, 0
        local function work(i)
            local x = i * 2 + 1
            local tb = ctrace()
            local line, src = cline()
            s = s + x % 3 + line
            s = s - x + i * 2
            return tb, line, src
        end
        for i = 1, 2500 do
            local tb, line, src = work(i)
            if i % 800 == 1 then out[#out + 1] = tb .. '|' .. line .. '|' .. src end
        end
        return table.concat(out, ';'), s";
    for (profile, report) in agree(source) {
        assert!(report.outcome.contains("stack traceback"), "{profile:?}");
    }
    let loop_source = "local out, s = {}, 0
        for i = 1, 2500 do
            local x = i * 2 + 1
            local line = cline()
            s = s + x % 3 + line
            s = s - x + i * 2
            if i % 800 == 1 then out[#out + 1] = ctrace() end
        end
        return table.concat(out, ';'), s";
    for (profile, report) in agree(loop_source) {
        assert!(report.outcome.contains("stack traceback"), "{profile:?}");
        assert_called(profile, &report, 1000);
    }
}

/// Hooks: installed before the loop, and from inside the host function.
#[test]
fn hooks_see_the_same_calls() {
    let before = "local calls, lines, s = 0, 0, 0
        debug.sethook(function(event) if event == 'line' then lines = lines + 1 else calls = calls + 1 end end, 'cl')
        for i = 1, 600 do
            local x = i * 2 + 1
            s = cadd(s, x) + i % 3
            s = s - x + i * 2
        end
        debug.sethook()
        return s, calls, lines";
    agree(before);
    let counting = "local fired, s = 0, 0
        debug.sethook(function() fired = fired + 1 end, '', 100)
        for i = 1, 2000 do
            local x = i * 2 + 1
            s = cadd(s, x) + i % 3
            s = s - x + i * 2
        end
        debug.sethook()
        return s, fired";
    agree(counting);
    // The host function installs a call and return hook in the middle of the
    // loop.
    let inside = "local lines, calls = 0, 0
        function sethook()
            debug.sethook(function(event)
                if event == 'line' then lines = lines + 1 else calls = calls + 1 end
            end, 'cr')
            return 1
        end
        local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + cadd(s, x) % 3
            if i == 2000 then chook() end
            s = s - x + i * 2
        end
        debug.sethook()
        return s, calls";
    agree(inside);
}

/// Runs `source` in every profile with the trace's own calls and without
/// them, and checks the results agree. (A count hook a host function
/// installs in the middle of a loop starts to count at a place that depends
/// on the engine, so its events differ from the interpreter's. The calls
/// from traces must not move it from where tracing without them puts it.)
fn same_without_trace_calls(source: &str) {
    for profile in ALL_PROFILES {
        let called = run(profile, ExecutionMode::Tracing, source);
        let declined = run_with(profile, ExecutionMode::Tracing, source, false);
        assert_eq!(called.outcome, declined.outcome, "{profile:?}\n{source}");
    }
}

/// A host function installs a line or a count hook, or raises an error right
/// after installing one: the hooks see what the interpreter's hooks see.
#[test]
fn hooks_installed_inside_the_host_function() {
    let lines = "local lines, seq = 0, {}
        function sethook()
            debug.sethook(function(e, l)
                lines = lines + 1
                if #seq < 40 then seq[#seq + 1] = l end
            end, 'l')
            return 1
        end
        local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + cadd(s, x) % 3
            if i == 2000 then chook() end
            s = s - x + i * 2
        end
        debug.sethook()
        return s, lines, table.concat(seq, ',')";
    same_without_trace_calls(lines);
    // A line hook fires from the line after the call on, as in stock Lua:
    // the interpreter's fast loop leaves after a host callback that
    // installed a hook, and so does a trace.
    for (profile, report) in agree(lines) {
        assert!(
            report.outcome.contains("7,14,10,11,12,13,14,10"),
            "{profile:?} {}",
            report.outcome
        );
    }
    let count = "local fired = 0
        function sethook()
            debug.sethook(function() fired = fired + 1 end, '', 50)
            return 1
        end
        local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + cadd(s, x) % 3
            if i == 2000 then chook() end
            s = s - x + i * 2
        end
        debug.sethook()
        return s, fired";
    same_without_trace_calls(count);
    let called = "local lines, n = 0, 0
        function bump()
            n = n + 1
            if n == 2000 then
                debug.sethook(function() lines = lines + 1 end, 'l')
                error('boom')
            end
            return n
        end
        local ok, message = pcall(function()
            local s = 0
            for i = 1, 3000 do
                local x = i * 2 + 1
                local k = cbump()
                s = s + k % 3 + x % 5
                s = s - x + i * 2
            end
            return s
        end)
        debug.sethook()
        return ok, message, lines, n";
    agree(called);
    let counted = "local fired, n = 0, 0
        function bump()
            n = n + 1
            if n == 2000 then
                debug.sethook(function() fired = fired + 1 end, '', 10)
                error('boom')
            end
            return n
        end
        local ok, message = pcall(function()
            local s = 0
            for i = 1, 3000 do
                local x = i * 2 + 1
                local k = cbump()
                s = s + k % 3 + x % 5
                s = s - x + i * 2
            end
            return s
        end)
        debug.sethook()
        return ok, message, fired, n";
    agree(counted);
}

/// A host function writes a local of the calling frame (`lua_setlocal`).
#[test]
fn the_call_writes_a_local_of_the_caller() {
    let source = "local s, k = 0, 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            local v = csetk()
            s = s + k + v + x % 3
            s = s - x + i * 2
        end
        return s, k";
    for (profile, report) in agree(source) {
        assert_declined(profile, &report);
    }
}

/// The call that makes the trace stop is one the trace itself made: it
/// installs the hook and the interpreter takes the frame right after it.
#[test]
fn a_hook_installed_by_the_trace_call_itself() {
    let source = "local calls, n = 0, 0
        function sethook()
            n = n + 1
            if n == 2000 then
                debug.sethook(function(event) calls = calls + 1 end, 'cr')
            end
            return n
        end
        local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            local k = chook()
            s = s + k % 3 + x % 5
            s = s - x + i * 2
        end
        debug.sethook()
        return s, calls, n";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 1000);
    }
}

/// Loops inside a coroutine, and a host function that yields.
#[test]
fn coroutines_and_host_yields() {
    let source = "local co = coroutine.wrap(function()
            local s = 0
            for i = 1, 3000 do
                local x = i * 2 + 1
                s = cadd(s, x) + i % 3
                if i % 1000 == 0 then coroutine.yield(s) end
                s = s - x + i * 2
            end
            return s
        end)
        return co(), co(), co(), co()";
    agree(source);
    let hosted = "local co = coroutine.wrap(function()
            local s = 0
            for i = 1, 3000 do
                local x = i * 2 + 1
                s = cadd(s, x) + i % 3
                if i % 1000 == 0 then s = s + cyield(s) end
                s = s - x + i * 2
            end
            return s
        end)
        return co(), co(1), co(2), co(3)";
    agree(hosted);
    let mixed = "local s = 0
        local co = coroutine.create(function()
            for i = 1, 2000 do
                local x = i * 2 + 1
                s = cadd(s, x) + i % 3
                s = s - x + i * 2
                if i % 700 == 0 then coroutine.yield() end
            end
        end)
        while coroutine.resume(co) and coroutine.status(co) ~= 'dead' do
            for j = 1, 100 do
                local y = j * 2 + 1
                s = cadd(s, y) + j % 3
                s = s - y + j * 2
            end
        end
        return s";
    agree(mixed);
}

/// The host function runs the loop's own function again.
#[test]
fn the_call_reenters_the_loop() {
    let source = "local depth = 0
        function again(n)
            if depth >= 2 then return 0 end
            depth = depth + 1
            local s = 0
            for i = 1, n do
                local x = i * 2 + 1
                s = s + cadd(s, x) % 3
                s = s - x + i * 2
                if i == 400 then s = s + cre(math.floor(n / 2)) end
            end
            depth = depth - 1
            return s
        end
        return again(2000), again(1000)";
    agree(source);
}

/// Results the trace takes no word for, a call that changes its answer, and
/// shapes the trace does not call from: all give the interpreter's results.
#[test]
fn other_shapes_give_the_same_results() {
    let shapes = [
        // A string result.
        "local s = 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + #cstr(x) + x % 3
            s = s - x + i * 2
         end
         return s",
        // A result that turns nil in the middle of the run.
        "local s = 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            local r = cmaybe(i)
            s = s + (r or -1) + x % 3
            s = s - x + i * 2
         end
         return s",
        // Varargs.
        "local t, s = {1, 2}, 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + cadd((unpack or table.unpack)(t)) + x % 3
            s = s - x + i * 2
         end
         return s",
        // Open results.
        "local s = 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            s = s + #tostring(cadd(s, x)) + x % 3
            s = s - x + i * 2
         end
         return s",
        // Through a protected call.
        "local s = 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            local ok, v = pcall(cadd, s, x)
            s = s + v + x % 3
            s = s - x + i * 2
         end
         return s",
    ];
    for source in shapes {
        agree(source);
    }
    // The nil in the middle is a mismatch the helper answers by itself.
    let flip = "local s = 0
         for i = 1, 3000 do
            local x = i * 2 + 1
            local r = cmaybe(x)
            s = s + (r or -1) + x % 3
            s = s - x + i * 2
            local q = cmaybe(i)
            s = s + (q or -7)
         end
         return s";
    for (profile, report) in agree(flip) {
        assert_called(profile, &report, 1000);
    }
}

/// The call sits inside a Lua function the trace inlines (an `__index`
/// function too): the trace makes it with the inlined frames rebuilt.
#[test]
fn the_call_sits_in_an_inlined_function() {
    let source = "local obj = setmetatable({}, {__index = function(t, k)
            local a = k * 2 + 1
            return cadd(a, k) % 5
        end})
        local function add_one(v)
            local w = v * 2 + 3
            return cadd(w, v) + 1
        end
        local s = 0
        for i = 1, 3000 do
            local v = obj[i]
            s = s + v + add_one(i) + i % 3
            s = s - i * 2
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 4000);
    }
}

/// The host function looks at the stack from inside inlined frames: the
/// traceback, the caller's line and an error at level 2 are the
/// interpreter's.
#[test]
fn the_host_function_sees_the_inlined_frames() {
    let traceback = "local function inner(v)
            local w = v * 2 + 3
            if v == 2500 then return ctrace() end
            return cadd(w, v) % 7
        end
        local function outer(v)
            local a = inner(v) 
            return a
        end
        local s, tb = 0, nil
        for i = 1, 3000 do
            local r = outer(i)
            if type(r) == 'string' then tb = r else s = s + r + i % 3 end
            s = s - i * 2
        end
        return s, tb";
    for (profile, report) in agree(traceback) {
        assert_called(profile, &report, 2000);
    }
    let line = "local function inner(v)
            local w = v * 2 + 3
            local l = v == 2000 and select(2, cline()) or nil
            return cadd(w, v) % 7, l
        end
        local s, line = 0, nil
        for i = 1, 3000 do
            local r, l = inner(i)
            if l then line = l end
            s = s + r + i % 3
            s = s - i * 2
        end
        return s, line";
    for (profile, report) in agree(line) {
        assert_called(profile, &report, 2000);
    }
    let failing = "local function inner(v)
            local w = v * 2 + 3
            return cfail(v) + cadd(w, v) % 7
        end
        local function outer(v)
            return inner(v) + 1
        end
        local function run()
            local s = 0
            for i = 1, 3000 do
                s = s + outer(i) + i % 3
                s = s - i * 2
            end
            return s
        end
        local ok, err = xpcall(run, debug.traceback)
        return ok, err";
    for (profile, report) in agree(failing) {
        assert_called(profile, &report, 100);
    }
}

/// A call that errors in the middle of an iteration that wrote registers
/// the error handler reads back.
#[test]
fn the_registers_in_front_of_the_call_are_written_back() {
    let source = "local function run()
            local a, b, c = 0, 1.5, true
            for i = 1, 3000 do
                a = a + i
                b = b * 1.001 + a % 3
                c = not c
                local d = a + b
                local k = cfail(i - 1850)
                a = a + k
            end
        end
        local function handler(message)
            local names = {}
            for level = 2, 4 do
                local info = debug.getinfo(level, 'S')
                if info and info.what == 'Lua' then
                    for index = 1, 8 do
                        local name, value = debug.getlocal(level, index)
                        if name then names[#names + 1] = name .. '=' .. tostring(value) end
                    end
                    break
                end
            end
            return message .. '|' .. table.concat(names, ',')
        end
        local ok, err = xpcall(run, handler)
        return ok, err";
    for (profile, report) in agree(source) {
        assert!(report.calls >= 1, "{profile:?} {:?}", report.stats);
    }
    let locals = "local ok, err = pcall(function()
            local a, b = 0, 0.5
            for i = 1, 300 do
                a = a + i
                b = b + a / 7
                if i == 250 then
                    local name, value = debug.getlocal(1, 1)
                    error(name .. '=' .. tostring(value))
                end
                cadd(a, b)
            end
        end)
        return ok, err";
    for (profile, report) in agree(locals) {
        assert_called(profile, &report, 100);
    }
}

/// With the calls from traces turned off (`OXIMOND_JIT_NO_VMCALL`, or
/// `set_trace_vmcall`) the loop runs in the interpreter, and the results are
/// the same.
#[test]
fn the_calls_from_traces_can_be_turned_off() {
    let source = "local s = 0
        for i = 1, 3000 do
            local x = i * 2 + 1
            local y = x * 3 - i + (x % 5) * 2 - (x % 3)
            s = cadd(s, y)
            s = s + x - i + (s % 7) * 2 - (i % 3)
        end
        return s";
    for &profile in &ALL_PROFILES {
        let expected = run(profile, ExecutionMode::Interpreter, source);
        let on = run(profile, ExecutionMode::Tracing, source);
        let off = run_with(profile, ExecutionMode::Tracing, source, false);
        assert_eq!(on.outcome, expected.outcome, "{profile:?}");
        assert_eq!(off.outcome, expected.outcome, "{profile:?}");
        assert!(on.calls >= 2000 && off.calls == 0, "{profile:?}");
    }
}

/// The default asks for some work in the loop besides the call: a call from
/// a trace costs more than the interpreter's, so a body of a few
/// instructions stays with the interpreter.
#[test]
fn the_default_asks_for_work_in_the_loop() {
    let body = |count: usize| {
        let work = "x = x * 3 - i + (x % 5) * 2 - (x % 3) ".repeat(count);
        format!(
            "local s = 0
             for i = 1, 3000 do
                local x = i * 2 + 1
                {work}
                s = cadd(s, x)
             end
             return s"
        )
    };
    for &profile in &ALL_PROFILES {
        for (count, calls) in [(0, false), (6, true)] {
            let source = body(count);
            let expected = run(profile, ExecutionMode::Interpreter, &source);
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(ExecutionMode::Tracing)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            lua.vm_mut().ignore_remaining_work();
            let function = lua
                .vm_mut()
                .register_host_function(cadd, 0, Vec::new())
                .unwrap();
            lua.vm_mut().runtime_mut().set_global("cadd", function);
            let outcome = match lua.exec(&source) {
                Ok(results) => format!("{results:?}"),
                Err(error) => format!("error: {error}"),
            };
            assert_eq!(outcome, expected.outcome, "{profile:?}");
            let made = lua.vm().jit_statistics().host_trace_calls;
            assert_eq!(made >= 1000, calls, "{profile:?} {count} {made}");
        }
    }
}

/// A call used as a statement takes no result and is made from the trace.
#[test]
fn a_call_as_a_statement() {
    let source = "local s = 0
        for i = 1, 4000 do
            cbump()
            s = s + i * 2
        end
        return s";
    for (profile, traced) in agree(&format!(
        "bumps = 0 function bump() bumps = bumps + 1 return 1 end {source}"
    )) {
        assert_called(profile, &traced, 100);
    }
}

/// Results the trace carries by type: nil, a string and a table. The trace
/// reads the fields of the table and the length of the string, and the call
/// is made from the trace.
#[test]
fn nil_string_and_table_results_are_carried() {
    let nil = "local s = 0
        for i = 1, 4000 do
            local x = i * 2 + 1
            local r = cnil(x)
            if r == nil then s = s + x % 7 else s = s - 1 end
            s = s - x + i * 2
        end
        return s";
    let string = "local s, last = 0, nil
        for i = 1, 4000 do
            local x = i * 2 + 1
            local r = cstr(x)
            s = s + #r + x % 7
            s = s - x + i * 2
            last = r
        end
        return s, last";
    let table = "local s, last = 0, nil
        for i = 1, 4000 do
            local x = i * 2 + 1
            local d = cbiome(x)
            s = s + d.biome + d.n % 5 + #d.name
            s = s - x + i * 2
            last = d.n
        end
        return s, last";
    let two = "local s = 0
        for i = 1, 4000 do
            local d, e = cbiome(i), cstr(i)
            s = s + d.biome + #e + i % 3
        end
        return s";
    for source in [nil, string, table, two] {
        for (profile, report) in agree(source) {
            assert_called(profile, &report, 3000);
        }
    }
}

/// A table result that rides to the next iteration in a register has a lane
/// the loop-back would release: the recording is refused, and the loop
/// gives the interpreter's results.
#[test]
fn a_table_result_carried_to_the_next_iteration() {
    let source = "local last, s = nil, 0
        for i = 1, 4000 do
            local x = i * 2 + 1
            local d = cbiome(x)
            s = s + d.biome + (last and last.n or 0) % 5
            s = s - x + i * 2
            last = d
        end
        return s, last.n, last.biome, last.name";
    agree(source);
}

/// The result is of another type than the recording saw: the helper misses
/// after the call, the trace leaves and the interpreter goes on with what
/// the call gave.
#[test]
fn a_result_of_another_type_leaves_the_trace() {
    let source = "local s, nils, kept = 0, 0, {}
        for i = 1, 4000 do
            local r = cvary(i)
            if r == nil then nils = nils + 1 else s = s + i % 7 end
            if r == kept[1] then s = s + 100 end
            if i % 1000 == 0 then kept[1] = r end
        end
        return s, nils, type(kept[1])";
    // The loop stores a table result into a table, so the trace is
    // refused.
    agree(source);
}

/// Tables the call makes are live nowhere but in the trace's registers:
/// the function collects everything before it returns one, the loop keeps
/// some and drops the rest.
#[test]
fn results_live_only_in_the_trace_survive_a_collection() {
    let source = "local keep, s = {}, 0
        for i = 1, 3000 do
            local d = cgctab(i)
            s = s + d.biome + d.n % 7
            if i % 100 == 0 then keep[#keep + 1] = d end
            if i % 500 == 0 then collectgarbage() end
        end
        local n = 0
        for _, d in ipairs(keep) do n = n + d.n + #d.name end
        return s, n, #keep";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// `core.get_content_id` as `builtin/common/item_s.lua` wraps it: an
/// `__index` function that tail calls the host function held in an upvalue,
/// with a string key.
#[test]
fn the_content_id_cache_calls_from_the_trace() {
    let source = "local old_get_content_id = cname
        local name2content = setmetatable({}, {
            __index = function(self, name)
                return old_get_content_id(name)
            end,
        })
        local names = {}
        for i = 1, 24 do names[i] = 'mcl_core:node_' .. i end
        local s = 0
        for i = 1, 4000 do
            local id = name2content[names[i % 24 + 1]]
            s = s + id + i % 5
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 3000);
    }
}

/// The biome search of `mcl_levelgen/biomes.lua`: `index_biomes_cached`
/// writes the position into an upvalue table in three fields and passes that
/// table to the host function `core.get_biome_data`, which reads them and
/// gives a fresh table; a predicate (an inlined function) calls it too. In
/// `LuaJIT` the coordinates are floats.
#[test]
fn the_biome_search_calls_from_the_trace() {
    let source = "local translations = {'Plains', 'Forest', 'Desert', 'Taiga', 'Swamp'}
        local v = {x = 0, y = 0, z = 0}
        local core = {get_biome_data = cdata}
        local function toblock(x) return (x - x % 4) / 4 end
        local mapgen_model = nil
        local preset = {
            index_biomes_cached = function(self, x, y, z)
                v.x = toblock(x)
                v.z = -toblock(z) - 1
                v.y = toblock(y) - 2 + 3
                if mapgen_model then
                    local override = mapgen_model.get_biome_override(v.x, v.z)
                    if override then return override end
                end
                local data = core.get_biome_data(v)
                if data then
                    return translations[data.biome] or 'Plains'
                else
                    return 'Plains'
                end
            end,
        }
        local function predicate(located, arg)
            local data = cbiome(#located + arg)
            return translations[data.biome] == located
        end
        local count, found = 0, nil
        for dz = -30, 30 do
            for dx = -30, 30 do
                local located = preset:index_biomes_cached(dx + 40, 2, dz + 40)
                if predicate(located, dx % 3) then
                    count = count + 1
                    found = located
                end
            end
        end
        return count, found";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// A host function builds a table, hands it to a Lua function that stores it,
/// and returns it. The loop reads the stored table back, so the trace reaches
/// the table a second way, and the lane of the call's result must stay the
/// table's until the run ends.
#[test]
fn a_table_the_call_gave_and_a_field_gives_again_keeps_its_lane() {
    let source = "local holder = {}
        function keepit(d) holder.x = d end
        local last, bad = nil, 0
        for i = 1, 4000 do
            local d = ckeep(i)
            local e = holder.x
            if last and last.n ~= i - 1 then bad = bad + 1 end
            last = e
        end
        return bad, last.n";
    agree(source);
}

/// The same with the call as a statement.
#[test]
fn a_table_the_call_dropped_and_a_field_gives_keeps_its_lane() {
    let source = "local holder = {}
        function keepit(d) holder.x = d end
        local last, bad = nil, 0
        for i = 1, 4000 do
            ckeep(i)
            local e = holder.x
            if last and last.n ~= i - 1 then bad = bad + 1 end
            last = e
        end
        return bad, last.n";
    agree(source);
}

/// The loop writes an upvalue, a global and fields on both sides of the call.
/// The call's restore stores what the trace holds in native values, and the
/// trace makes every call.
#[test]
fn cells_globals_and_fields_are_written_on_both_sides_of_the_call() {
    let source = "local up, ts = 0, {{x = 1, y = 2.5}}
        g = 0
        local s = 0
        for i = 1, 4000 do
            local t = ts[1]
            up = up + 1
            g = g + 2
            t.x = t.x + 1
            local k = cadd(up, g)
            up = up + k % 3
            g = g - 1
            t.y = t.y + 0.5
            s = s + up + g + t.x + t.y + k % 5
        end
        return s, up, g, ts[1].x, ts[1].y";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 3000);
    }
}

/// Globals and fields of a table the loop does not write, read beside the
/// call: the fields are read through the table's lane, which the run makes
/// again after the call.
#[test]
fn globals_and_entry_table_fields_are_read_beside_the_call() {
    let global = "g = 0
        local s = 0
        for i = 1, 4000 do
            g = g + 1
            local k = cadd(i, 1)
            g = g + k % 2
            s = s + g
        end
        return s, g";
    let argument = "g = 3
        local s = 0
        for i = 1, 4000 do
            s = s + cadd(i, g)
        end
        return s";
    let field = "local t = {n = 1, x = 2}
        local s = 0
        for i = 1, 4000 do
            local a = t.n
            local k = cadd(i, 1)
            s = s + a + t.n + t.x + k % 3
        end
        return s";
    for source in [global, argument, field] {
        for (profile, report) in agree(source) {
            assert_called(profile, &report, 3000);
        }
    }
}

/// The function behind the call changes a global and an entry table's field
/// that the loop only reads: the trace leaves at the change.
#[test]
fn the_call_changes_a_global_and_a_field_the_loop_reads() {
    let source = "g = 3
        local t = {n = 1, x = 2}
        local calls = 0
        function bump()
            calls = calls + 1
            if calls == 1500 then g = 10 end
            if calls == 2000 then t.n = 5 end
            if calls == 2500 then t.x = 0.5 end
            if calls == 3000 then g = 'text' end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local a = g
            local k = cbump()
            s = s + k % 3 + t.n + t.x
            s = s + (type(g) == 'number' and g + a or 1)
        end
        return s";
    for (profile, report) in agree(source) {
        // The loop is recorded again after each change.
        assert!(report.calls >= 1000, "{profile:?}: {} calls", report.calls);
    }
}

/// The function behind the call rehashes a table the loop looks up with a
/// variable string key: the lookups after the call find the key where it is.
#[test]
fn the_call_rehashes_a_table_the_loop_looks_up() {
    let source = "local t, keys = {}, {}
        for i = 1, 16 do keys[i] = 'k' .. i; t[keys[i]] = i end
        local n = 0
        function bump()
            n = n + 1
            if n == 1500 then
                for i = 1, 300 do t['k' .. i + 100] = i + 1000 end
            end
            if n == 2500 then
                for i = 1, 300 do t['k' .. i + 100] = nil end
                collectgarbage()
                for i = 1, 300 do t['k' .. i + 100] = i + 2000 end
            end
            return n
        end
        local s = 0
        for i = 1, 4000 do
            local k = keys[i % 16 + 1]
            local a = t[k]
            local c = cbump()
            s = s + a + t[k] + c % 3
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// The function behind the call changes the cell, global, field and array the
/// trace holds in native values or reads: the trace leaves at the change and
/// the interpreter goes on with the function's answer.
#[test]
fn the_call_changes_what_the_trace_wrote() {
    let upvalue = "local up = 0
        local n = 0
        function bump()
            n = n + 1
            if n % 700 == 0 then up = up + 1000 end
            return n
        end
        local s = 0
        for i = 1, 4000 do
            up = up + 1
            local k = cbump()
            up = up + 1
            s = s + up + k % 3
        end
        return s, up, n";
    let global = "g = 0
        local n = 0
        function bump()
            n = n + 1
            if n % 700 == 0 then g = g - 500 end
            return n
        end
        local s = 0
        for i = 1, 4000 do
            g = g + 3
            local k = cbump()
            g = g + 1
            s = s + g + k % 3
        end
        return s, g, n";
    let global_only_written = "x = 0
        local n = 0
        function bump()
            n = n + 1
            if n % 600 == 0 then x = 77 end
            return n
        end
        local s = 0
        for i = 1, 4000 do
            x = i
            local k = cbump()
            s = s + k % 3 + i % 7
        end
        return s, x, n";
    let field = "local t = {x = 1, n = 0}
        local calls = 0
        function bump()
            calls = calls + 1
            if calls % 700 == 0 then t.x = t.x + 1000 end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            t.x = t.x + 1
            local k = cbump()
            t.x = t.x + 1
            s = s + t.x + k % 3
        end
        return s, t.x";
    let rawset = "local t = {1, 2, 3, 4}
        local calls = 0
        function bump()
            calls = calls + 1
            if calls % 500 == 0 then
                for j = #t + 1, #t + 40 do t[j] = j end
            end
            if calls == 3000 then t = {9, 8, 7} end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local a = t[i % 4 + 1]
            local k = crawset(t, i % 4 + 1, i)
            local b = t[i % 4 + 1]
            s = s + a + b + k % 3 + #t
            if i % 2 == 0 then cbump() end
        end
        return s, #t";
    let meta = "local t = {x = 1}
        local calls = 0
        function bump()
            calls = calls + 1
            if calls == 1500 then setmetatable(t, {__index = function(_, k) return 7 end}) end
            if calls == 2500 then t.x = nil end
            return calls
        end
        local s = 0
        for i = 1, 4000 do
            local a = t.x
            local k = cbump()
            s = s + (a or 0) + (t.x or 1) + k % 3
        end
        return s";
    for source in [upvalue, global, global_only_written, field, rawset, meta] {
        agree(source);
    }
    // The ones that change nothing in the middle of the run stay in the trace.
    for source in [upvalue] {
        for (profile, report) in agree(source) {
            assert!(
                report.calls >= 1000,
                "{profile:?}\n{source}\n{:?}",
                report.stats
            );
        }
    }
}

/// An error raised at level 2 by a host function inside an inlined function
/// blames the caller of the inlined function, as in the interpreter.
#[test]
fn an_error_at_level_two_from_an_inlined_call() {
    let source = "local function inner(v)
            local w = v * 2 + 3
            return cerr2(v) + w % 3
        end
        local function outer(v)
            return inner(v) + 1
        end
        local function run()
            local s = 0
            for i = 1, 4000 do
                s = s + outer(i) + i % 3
            end
            return s
        end
        local ok, err = pcall(run)
        return ok, err";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 1000);
    }
}

/// A call hook set before the loop sees the calls of host functions made
/// from an inlined function, so the calls are made by the interpreter; a
/// hook set by the function itself ends the trace.
#[test]
fn hooks_see_host_calls_from_inlined_functions() {
    let before = "local seen = 0
        debug.sethook(function(event) seen = seen + 1 end, 'cr')
        local function inner(v)
            local w = v * 2 + 3
            return cadd(w, v) % 7
        end
        local s = 0
        for i = 1, 2000 do
            s = s + inner(i) + i % 3
        end
        debug.sethook()
        return s, seen";
    agree(before);
    let inside = "local seen = 0
        local function inner(v)
            local w = v * 2 + 3
            if v == 1500 then chook() end
            return cadd(w, v) % 7
        end
        function sethook() debug.sethook(function(event) seen = seen + 1 end, 'cr') end
        local s = 0
        for i = 1, 3000 do
            s = s + inner(i) + i % 3
        end
        debug.sethook()
        return s, seen";
    agree(inside);
}

/// A tail call of a host function gives all its results: a call that gives
/// another number of them than the recording saw leaves the trace after the
/// call, and the interpreter returns what the function gave.
#[test]
fn a_tail_call_gives_all_its_results() {
    let source = "local function get(n) return cdyn(n) end
        local s, extra, none = 0, 0, 0
        for i = 1, 4000 do
            local a, b = get(i)
            if b then extra = extra + 1 end
            if not a then none = none + 1 end
            s = s + (a or 0) + (b or 0) + i % 3
        end
        return s, extra, none";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 3000);
    }
}

/// A call that rehashes a table the loop reads and writes through a field
/// binding: the trace must read the slots again, so it reads through the VM
/// or leaves the loop to the interpreter. Each loop agrees with the interpreter.
#[test]
fn a_rehash_behind_the_call_moves_the_fields_the_trace_reads() {
    let prelude = "local calls = 0
        function bump()
            calls = calls + 1
            if calls == 1500 then
                for j = 1, 300 do T['k' .. j] = j end
            end
            if calls == 2000 then T.n = 1000 end
            return calls
        end";
    let local = format!(
        "{prelude}
        T = {{n = 1, x = 2}}
        local t = T
        local s = 0
        for i = 1, 4000 do
            local a = t.n
            local k = cbump()
            s = s + k % 5 + a + t.n + t.x
        end
        return s, t.n"
    );
    let upvalue = format!(
        "{prelude}
        T = {{n = 1, x = 2}}
        local t = T
        local function run()
            local s = 0
            for i = 1, 4000 do
                local a = t.n
                local k = cbump()
                s = s + k % 5 + a + t.n + t.x
            end
            return s, t.n
        end
        return run()"
    );
    let store = format!(
        "{prelude}
        T = {{n = 1, x = 2}}
        local t = T
        for i = 1, 4000 do
            local k = cbump()
            t.n = t.n + 1
        end
        return t.n, calls"
    );
    for source in [local, upvalue, store] {
        agree(&source);
    }
}

/// A table result stored into an array, a hash key or a global. The
/// recording that does so is refused; the results agree.
#[test]
fn a_table_result_is_stored_into_a_table_or_a_global() {
    let array = "local keep, s = {}, 0
        for i = 1, 4000 do
            local d = cbiome(i)
            s = s + d.biome
            keep[i % 4 + 1] = d
        end
        return s, keep[1].n";
    let hash = "local keep, s = {}, 0
        for i = 1, 4000 do
            local d = cbiome(i)
            s = s + d.biome
            keep.last = d
            keep['k' .. i % 3] = d
        end
        return s, keep.last.n, keep.k1.n";
    let global = "local s = 0
        for i = 1, 4000 do
            local d = cbiome(i)
            s = s + d.biome
            G = d
        end
        return s, G.n";
    for source in [array, hash, global] {
        agree(source);
    }
}

/// After each call the run checks that the upvalues, globals, fields and
/// captured locals it read still hold the objects it entered with, by the
/// lanes the bank gave them. Tables and strings that stay put keep the calls
/// in the trace, while the bank fills with the tables the calls give.
#[test]
fn objects_the_trace_read_hold_across_the_calls() {
    let source = "local pal, tag, mod = {1, 2, 3}, 'abc', {name = 'm'}
        G_TAG = 'gg'
        local s = 0
        for i = 1, 4000 do
            local d = cbiome(i)
            s = s + d.n % 7 + pal[i % 3 + 1] + #tag + #mod.name + #G_TAG
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// The function a call runs swaps an upvalue table, an upvalue string, a
/// field of an upvalue table, a global string and a captured local for other
/// objects. The run that sees one changed leaves; the results agree.
#[test]
fn the_call_swaps_objects_the_trace_read() {
    let swaps = "local pal, other, tag = {1, 2, 3}, {7, 8, 9}, 'a'
        local mod = {name = 'm'}
        G_TAG = 'g'
        local n = 0
        function bump()
            n = n + 1
            if n == 900 then pal = other end
            if n == 1400 then tag = 'bb' end
            if n == 1900 then mod.name = 'mmm' end
            if n == 2400 then G_TAG = 'gggg' end
            return n % 5
        end
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            s = s + k + pal[i % 3 + 1] + #tag + #mod.name + #G_TAG
        end
        return s, n";
    let captured = "local n = 0
        local function run()
            local box, word = {1}, 'x'
            local function swap() box = {5}; word = 'yyy' end
            function bump()
                n = n + 1
                if n == 1500 then swap() end
                return n % 3
            end
            local s = 0
            for i = 1, 3000 do
                local k = cbump()
                s = s + k + box[1] + #word
            end
            return s
        end
        return run(), n";
    // The upvalue `x` is open in a suspended coroutine: the coroutine assigns
    // its register, no cell write, when the call resumes it. The function
    // the call runs writes no cell itself (its counter is a field).
    let open = "local co = coroutine.create(function()
            local x = 1
            get = function() return x end
            coroutine.yield()
            x = 50
            coroutine.yield()
        end)
        coroutine.resume(co)
        local get, st = get, {n = 0}
        function bump()
            st.n = st.n + 1
            if st.n == 1500 then coroutine.resume(co) end
            return st.n % 3
        end
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            s = s + k + get()
        end
        return s, st.n";
    // A call that writes no cell (the counter is a field) keeps the closed
    // upvalues unchecked, and `debug.setupvalue` is a cell write.
    let setupvalue = "local function make()
            local limit = 3
            return function() return limit end
        end
        local read, st = make(), {n = 0}
        function bump()
            st.n = st.n + 1
            if st.n == 1700 then debug.setupvalue(read, 1, 9) end
            return st.n % 4
        end
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            s = s + k + read() * 2 - i % 5
        end
        return s, st.n, read()";
    for source in [swaps, captured, open, setupvalue] {
        for (profile, report) in agree(source) {
            assert!(
                report.calls >= 1,
                "{profile:?}\n{source}\n{:?}",
                report.stats
            );
        }
    }
}

/// Calls that move no table's keys keep the rows of the tables the loop
/// reads (an ordered record and a hashed table) instead of filling them
/// again. Now and then the call changes a value's type, deletes and revives
/// a key of the hashed table in place, adds and removes keys of the record,
/// rehashes the hashed table, turns a shaped record ordered and gives the
/// record a metatable that answers the field it lacks. The results agree.
#[test]
fn rows_stand_across_calls_that_move_no_keys() {
    let source = "local rec = {}
        rec.a = 1; rec.b = 2.5; rec.c = true
        local big = {}
        for i = 1, 40 do big['k' .. i] = i end
        big.x = 3
        local shp = {p = 1, q = 2}
        local st = {n = 0}
        function bump()
            st.n = st.n + 1
            local n = st.n
            if n == 700 then rec.a = 'str' end
            if n == 800 then rec.a = 5 end
            if n == 1100 then big.x = nil end
            if n == 1200 then big.x = 7 end
            if n == 1500 then rec.aa = 1 end
            if n == 1700 then rec.a = nil end
            if n == 1800 then rec.a = 9 end
            if n == 1900 then shp.q = nil; shp.p = 7 end
            if n == 2100 then for j = 1, 200 do big['m' .. j] = j end end
            if n == 2500 then setmetatable(rec, {__index = function() return 4 end}) end
            return n % 3
        end
        local s = 0
        for i = 1, 3000 do
            local k = cbump()
            local a = rec.a
            if a == 'str' or a == nil then a = 1 end
            local b = rec.c and rec.b or 0
            s = s + k + a + b + (big.x or 0) + big.k7 + (rec.q or 2) + shp.p
        end
        return s, st.n";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// Each call makes a table, which moves the count of key layout changes,
/// so the rows of the other tables are checked against their entries: a
/// shaped table that gains a key (and turns ordered), one whose field
/// changes type or goes, and an absent key that comes to exist and goes
/// again must all read as the interpreter reads them after the call.
#[test]
fn rows_kept_across_calls_that_make_tables_follow_their_tables() {
    let source = "local v = {x = 0, y = 0, z = 0}
        local m = {}
        local s = 0
        for i = 1, 4000 do
            v.x = i
            local d = cdata(v)
            s = s + d.biome + v.x + (tonumber(v.z) or 1) + (m.q or 0)
            if i % 701 == 0 then crawset(v, 'k' .. i, i) end
            if i == 1500 then crawset(m, 'q', 7) end
            if i == 2000 then crawset(v, 'z', 0.5) end
            if i == 2600 then crawset(v, 'z', 'str') end
            if i == 3000 then crawset(v, 'z', nil) end
            if i == 3500 then crawset(m, 'q', nil) end
        end
        return s";
    for (profile, report) in agree(source) {
        assert_called(profile, &report, 2000);
    }
}

/// The host function the trace calls changes, through the host API, what
/// the loop reads after the call in the same run: it joins the upvalues of
/// the running function (or of getters the loop calls) to other cells,
/// stores a fresh table into a joined cell, rebinds a field holding a table
/// the run read, and replaces a table two fields deep. Every iteration
/// reads the same cells and tables before and after the call, so a
/// remembered cell read or table read that outlived the call would answer
/// with the old value. The results agree with the interpreter's.
#[test]
fn the_host_rejoins_cells_and_rebinds_tables_the_run_reads() {
    let prelude = "local function upidx(f, name)
            for i = 1, 255 do
                local n = debug.getupvalue(f, i)
                if n == name then return i end
                if n == nil then return nil end
            end
        end
        local a, k = {1, 2, 3, x = 2}, 5
        local srca, srck = {100, 200, 300, x = 8}, 70
        local hold = {inner = {x = 4, t = {x = 6}}}
        local function src_a() return srca end
        local function src_k() return srck end
        SRCA, SRCK, HOLD = src_a, src_k, hold
        ";
    // The loop reads the upvalues of the function it runs in.
    let own = "local function run(n)
            local s = 0
            for i = 1, n do
                s = s + a[i % 3 + 1] + k + hold.inner.x + hold.inner.t.x
                local r = cjoin(i)
                s = s + r + a[i % 3 + 1] * 3 + k * 5 + hold.inner.x * 7
                    + hold.inner.t.x * 11 + a.x
            end
            return s
        end
        RUNA, IA, RUNK, IK = run, upidx(run, 'a'), run, upidx(run, 'k')
        local s = run(4000)
        return s, a[1], k, hold.inner.x, srca[1], srck";
    // The loop reads them through getters it calls, which the trace inlines.
    let getters = "local function geta() return a end
        local function getk() return k end
        RUNA, IA, RUNK, IK = geta, upidx(geta, 'a'), getk, upidx(getk, 'k')
        local s = 0
        for i = 1, 4000 do
            local t = geta()
            s = s + t[i % 3 + 1] + getk() + hold.inner.t.x
            local r = cjoin(i)
            t = geta()
            s = s + r + t[i % 3 + 1] * 3 + getk() * 5 + hold.inner.t.x * 11 + t.x
        end
        return s, a[1], k, hold.inner.x, srca[1], srck";
    // The join gives `c2` the cell of `c1`, which the loop writes: the two
    // names the trace held apart are one variable after the call. At the
    // join both hold 999, so the cell values alone show no change, and the
    // loop writes `c1` between the call and its read of `c2`.
    let apart = "local c1, c2 = 0, 999
        local function src_c() return c1 end
        local function run(n)
            local s = 0
            for i = 1, n do
                local r = cjoin(i)
                c1 = c1 + 1
                s = s + r + c2 * 3 + c1
            end
            return s
        end
        SRCA, SRCK = src_c, src_c
        RUNA, IA, RUNK, IK = run, upidx(run, 'c2'), run, upidx(run, 'c2')
        local s = run(2000)
        return s, c1, c2";
    for body in [own, getters, apart] {
        let source = format!("{prelude}{body}");
        for (profile, report) in agree(&source) {
            assert_called(profile, &report, 1000);
        }
    }
}
