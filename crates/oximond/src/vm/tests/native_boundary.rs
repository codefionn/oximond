//! Native calls from the fast loop: how many results a native returns, what
//! is forwarded as open results, and what an error does. Each case runs on
//! the fast loop, which calls natives in place, and again with profiling on,
//! which keeps the fast loop out. The results must match.

use super::trace_record::interpreter_vm_with;
use super::*;
use crate::runtime::RuntimeResult;
use crate::vm::interp::GENERAL_FALLBACKS;

fn ret0(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(Vec::new())
}

fn ret1(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![Value::from("one")])
}

fn ret3(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![
        Value::from("a"),
        Value::Boolean(true),
        Value::from("c"),
    ])
}

/// Twelve results, more than the pooled result vectors start with room for.
fn ret12(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok((0..12)
        .map(|index| {
            if index % 2 == 0 {
                Value::from(format!("s{index}").as_str())
            } else {
                Value::Number(f64::from(index))
            }
        })
        .collect())
}

fn count(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    Ok(vec![Value::Number(arguments.len() as f64)])
}

fn fail(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Err(RuntimeError::message("boom"))
}

fn scalar(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Value> {
    Ok(Value::Number(7.0))
}

fn scalar_fail(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Value> {
    Err(RuntimeError::message("scalar boom"))
}

/// One value that tells the arguments apart: ten per argument plus the sum
/// of the numbers.
fn scalar_args(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Value> {
    let sum: f64 = arguments
        .iter()
        .map(|value| match value {
            Value::Number(number) => *number,
            #[allow(clippy::cast_precision_loss)]
            Value::Integer(number) => *number as f64,
            _ => 0.0,
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    Ok(Value::Number(arguments.len() as f64 * 10.0 + sum))
}

fn vm_with_natives(profile: CompatibilityProfile, profiling: bool) -> Vm {
    let mut vm = interpreter_vm_with(profile);
    vm.debug_hooks.clear();
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    let runtime = vm.runtime_mut();
    let natives: [(&str, crate::runtime::NativeCallback); 6] = [
        ("ret0", ret0),
        ("ret1", ret1),
        ("ret3", ret3),
        ("ret12", ret12),
        ("count", count),
        ("fail", fail),
    ];
    for (name, callback) in natives {
        runtime.register_global_native(name, callback).unwrap();
    }
    let handle = runtime.register_native_scalar(scalar).unwrap();
    runtime.set_global("scalar", Value::NativeFunction(handle));
    let handle = runtime.register_native_scalar(scalar_fail).unwrap();
    runtime.set_global("scalar_fail", Value::NativeFunction(handle));
    let handle = runtime.register_native_scalar(scalar_args).unwrap();
    runtime.set_global("scalar_args", Value::NativeFunction(handle));
    vm.set_interpreter_profiling(profiling);
    vm
}

const NAMES: [&str; 10] = [
    "ret0",
    "ret1",
    "ret3",
    "ret12",
    "scalar",
    "count",
    "fail",
    "scalar_fail",
    "pcall",
    "scalar_args",
];

fn names() -> Vec<Constant> {
    NAMES
        .iter()
        .map(|name| Constant::String(LuaString::from(*name)))
        .collect()
}

fn global(destination: u8, name: &str) -> Instruction {
    let index = NAMES
        .iter()
        .position(|candidate| *candidate == name)
        .unwrap();
    Instruction::get_global(destination, u16::try_from(index).unwrap())
}

/// Runs `code` on the fast loop and with profiling, which must agree, and
/// returns the shared outcome and the opcodes the fast run left to the
/// dispatch loop.
fn run(
    profile: CompatibilityProfile,
    registers: u8,
    code: Vec<Instruction>,
) -> (Result<Vec<Value>, String>, Vec<OpCode>) {
    let chunk = Chunk::new(registers, names(), code);
    chunk.validate().unwrap();
    GENERAL_FALLBACKS.with(|log| log.borrow_mut().clear());
    let fast = vm_with_natives(profile, false)
        .execute(&chunk)
        .map_err(|error| error.to_string());
    let fallbacks = GENERAL_FALLBACKS.with(|log| log.borrow().clone());
    let general = vm_with_natives(profile, true)
        .execute(&chunk)
        .map_err(|error| error.to_string());
    assert_eq!(fast, general, "{profile:?}");
    (fast, fallbacks)
}

fn text(value: &str) -> Value {
    Value::from(value)
}

#[test]
fn natives_returning_none_one_three_and_many_values_fill_fixed_results() {
    for profile in super::fast_coverage::PROFILES {
        let (outcome, fallbacks) = run(
            profile,
            24,
            vec![
                // No results into two destinations: both become nil.
                global(0, "ret0"),
                Instruction::call(0, 0, 2),
                // One result into two destinations.
                global(2, "ret1"),
                Instruction::call(2, 0, 2),
                // Three results exactly.
                global(4, "ret3"),
                Instruction::call(4, 0, 3),
                // Twelve results into one: the rest are dropped.
                global(7, "ret12"),
                Instruction::call(7, 0, 1),
                // A single-result native into two destinations.
                global(8, "scalar"),
                Instruction::call(8, 0, 2),
                // Twelve results into ten destinations.
                global(10, "ret12"),
                Instruction::call(10, 0, 10),
                Instruction::return_values(0, 20),
            ],
        );
        let mut expected = vec![
            Value::Nil,
            Value::Nil,
            text("one"),
            Value::Nil,
            text("a"),
            Value::Boolean(true),
            text("c"),
            text("s0"),
            Value::Number(7.0),
            Value::Nil,
        ];
        expected.extend((0..10).map(|index| {
            if index % 2 == 0 {
                text(&format!("s{index}"))
            } else {
                Value::Number(f64::from(index))
            }
        }));
        assert_eq!(outcome.unwrap(), expected, "{profile:?}");
        assert!(
            !fallbacks.contains(&OpCode::Call),
            "{profile:?}: {fallbacks:?}"
        );
    }
}

#[test]
fn recycled_result_vectors_carry_nothing_between_calls() {
    for profile in super::fast_coverage::PROFILES {
        let (outcome, _) = run(
            profile,
            12,
            vec![
                global(0, "ret12"),
                Instruction::call(0, 0, 1),
                global(1, "ret1"),
                Instruction::call(1, 0, 3),
                global(4, "ret0"),
                Instruction::call(4, 0, 2),
                global(6, "ret3"),
                Instruction::call(6, 0, 4),
                Instruction::return_values(0, 10),
            ],
        );
        assert_eq!(
            outcome.unwrap(),
            vec![
                text("s0"),
                text("one"),
                Value::Nil,
                Value::Nil,
                Value::Nil,
                Value::Nil,
                text("a"),
                Value::Boolean(true),
                text("c"),
                Value::Nil,
            ],
            "{profile:?}",
        );
    }
}

#[test]
fn native_results_forward_as_open_results() {
    for profile in super::fast_coverage::PROFILES {
        // `count(ret12())` sees all twelve values; `count(ret0())` none.
        let (outcome, _) = run(
            profile,
            6,
            vec![
                global(0, "count"),
                global(1, "ret12"),
                Instruction::call(1, 0, 0),
                Instruction::call_open(0, 0, 1),
                global(1, "count"),
                global(2, "ret0"),
                Instruction::call(2, 0, 0),
                Instruction::call_open(1, 0, 1),
                global(2, "count"),
                global(3, "ret3"),
                Instruction::call(3, 0, 0),
                Instruction::call_open(2, 0, 1),
                Instruction::return_values(0, 3),
            ],
        );
        assert_eq!(
            outcome.unwrap(),
            vec![Value::Number(12.0), Value::Number(0.0), Value::Number(3.0)],
            "{profile:?}",
        );
        // The values themselves come through whole.
        let (outcome, _) = run(
            profile,
            4,
            vec![
                global(0, "ret3"),
                Instruction::call(0, 0, 0),
                Instruction::return_open(0, 0),
            ],
        );
        assert_eq!(
            outcome.unwrap(),
            vec![text("a"), Value::Boolean(true), text("c")],
            "{profile:?}",
        );
    }
}

/// `f (g ())` of two natives that return one value, the shape of
/// `tonumber (band (value, mask))`: the inner call's value is the outer
/// call's open argument, alone or after fixed ones. The fast loop runs both
/// in place (`Vm::open_native_in_window`) as one fused step, so neither goes
/// to the dispatch loop and no open results are stored.
#[test]
fn single_value_natives_pass_open_results_in_the_fast_loop() {
    for profile in super::fast_coverage::PROFILES {
        crate::vm::fast_path::OPEN_NATIVES_IN_PLACE.with(|count| count.set(0));
        crate::vm::fast_path::FUSED_OPEN_NATIVES.with(|count| count.set(0));
        let (outcome, fallbacks) = run(
            profile,
            8,
            vec![
                // r0 = scalar_args (scalar ()): one argument, 7.
                global(0, "scalar_args"),
                global(1, "scalar"),
                Instruction::call(1, 0, 0),
                Instruction::call_open(0, 0, 1),
                // r1 = scalar_args (r0, r0, scalar ()): two fixed, one open.
                global(1, "scalar_args"),
                Instruction::abc(OpCode::Move, 2, 0, 0),
                Instruction::abc(OpCode::Move, 3, 0, 0),
                global(4, "scalar"),
                Instruction::call(4, 0, 0),
                Instruction::call_open(1, 2, 1),
                // r2 = scalar_args (scalar_args ()): the inner one has no
                // arguments.
                global(2, "scalar_args"),
                global(3, "scalar_args"),
                Instruction::call(3, 0, 0),
                Instruction::call_open(2, 0, 1),
                Instruction::return_values(0, 3),
            ],
        );
        assert_eq!(
            outcome.unwrap(),
            vec![
                Value::Number(17.0),
                Value::Number(30.0 + 17.0 + 17.0 + 7.0),
                Value::Number(10.0),
            ],
            "{profile:?}",
        );
        assert!(
            !fallbacks
                .iter()
                .any(|opcode| matches!(opcode, OpCode::Call | OpCode::CallOpen)),
            "{profile:?}: {fallbacks:?}"
        );
        // Each of the three pairs ran in place once, fused, on the fast run;
        // the profiling run keeps out of it.
        let in_place = crate::vm::fast_path::OPEN_NATIVES_IN_PLACE.with(std::cell::Cell::get);
        let fused = crate::vm::fast_path::FUSED_OPEN_NATIVES.with(std::cell::Cell::get);
        assert_eq!((in_place, fused), (3, 3), "{profile:?}");
    }
}

/// The in-place pair keeps the general path's answers where it does not
/// apply or fails: a multi-value native as either call, an open result that
/// stays open, and an error from either call.
#[test]
fn open_results_of_natives_keep_the_general_answers() {
    for profile in super::fast_coverage::PROFILES {
        // `scalar_args (ret3 ())` takes three values; `count (scalar ())`
        // one, through a multi-value native.
        let (outcome, _) = run(
            profile,
            6,
            vec![
                global(0, "scalar_args"),
                global(1, "ret3"),
                Instruction::call(1, 0, 0),
                Instruction::call_open(0, 0, 1),
                global(1, "count"),
                global(2, "scalar"),
                Instruction::call(2, 0, 0),
                Instruction::call_open(1, 0, 1),
                Instruction::return_values(0, 2),
            ],
        );
        assert_eq!(
            outcome.unwrap(),
            vec![Value::Number(30.0), Value::Number(1.0)],
            "{profile:?}",
        );
        // `return scalar_args (scalar ())` leaves the outer value open.
        let (outcome, _) = run(
            profile,
            4,
            vec![
                global(0, "scalar_args"),
                global(1, "scalar"),
                Instruction::call(1, 0, 0),
                Instruction::call_open(0, 0, 0),
                Instruction::return_open(0, 0),
            ],
        );
        assert_eq!(outcome.unwrap(), vec![Value::Number(17.0)], "{profile:?}");
        // An error in the inner and in the outer call.
        for (outer, inner, message) in [
            ("scalar_args", "scalar_fail", "scalar boom"),
            ("scalar_fail", "scalar", "scalar boom"),
        ] {
            let (outcome, _) = run(
                profile,
                4,
                vec![
                    global(0, outer),
                    global(1, inner),
                    Instruction::call(1, 0, 0),
                    Instruction::call_open(0, 0, 1),
                    Instruction::return_(0),
                ],
            );
            let error = outcome.unwrap_err();
            assert!(error.contains(message), "{profile:?}: {error}");
        }
    }
}

#[test]
fn native_errors_inside_pcall_are_caught() {
    for profile in super::fast_coverage::PROFILES {
        for (failing, message) in [("fail", "boom"), ("scalar_fail", "scalar boom")] {
            let (outcome, _) = run(
                profile,
                4,
                vec![
                    global(0, "pcall"),
                    global(1, failing),
                    Instruction::call(0, 1, 3),
                    Instruction::return_values(0, 2),
                ],
            );
            let values = outcome.unwrap();
            assert_eq!(values[0], Value::Boolean(false), "{profile:?}");
            let Value::String(text) = &values[1] else {
                panic!("{profile:?}: {values:?}");
            };
            assert!(text.to_lossy().contains(message), "{profile:?}: {text:?}");
        }
    }
}

#[test]
fn native_errors_outside_pcall_reach_the_caller() {
    for profile in super::fast_coverage::PROFILES {
        for (failing, message) in [("fail", "boom"), ("scalar_fail", "scalar boom")] {
            let (outcome, _) = run(
                profile,
                4,
                vec![
                    global(0, failing),
                    Instruction::call(0, 0, 1),
                    Instruction::return_(0),
                ],
            );
            let error = outcome.unwrap_err();
            assert!(error.contains(message), "{profile:?}: {error}");
        }
    }
}

/// A hook keeps the in-place pair out: with a call hook every call of the
/// inner and the outer native is seen, and with a count hook the counts
/// fire; neither runs `Vm::open_native_in_window`. Without a hook the same
/// loop runs the pair in place.
#[test]
fn hooks_see_every_call_of_open_results_of_natives() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let (prelude, pair) = if profile == CompatibilityProfile::LuaJit21 {
            (
                "local band, tonumber = bit.band, tonumber
                 local x = 0x123456789ull",
                "tonumber(band(x + i, 0xffffffffull))",
            )
        } else {
            (
                "local abs, tonumber = math.abs, tonumber",
                "tonumber(abs(-i))",
            )
        };
        let source = format!(
            "
{prelude}
local inner, outer = {inner}, tonumber
local function loop(n)
  local s = 0
  for i = 1, n do s = s + {pair} end
  return s
end
local calls, counts = 0, 0
local mode = ...
if mode == 'c' then
  debug.sethook(function()
    local f = debug.getinfo(2, 'f').func
    if f == inner or f == outer then calls = calls + 1 end
  end, 'c')
elseif mode == 'count' then
  debug.sethook(function() counts = counts + 1 end, '', 7)
end
local s = loop(1000)
debug.sethook()
return s, calls, counts
",
            inner = if profile == CompatibilityProfile::LuaJit21 {
                "band"
            } else {
                "abs"
            },
        );
        let run_mode = |mode: &str| {
            let mut lua = crate::Lua::builder()
                .profile(profile)
                .execution_mode(ExecutionMode::Interpreter)
                .build()
                .unwrap();
            let chunk = format!("return (function(...) {source} end)('{mode}')");
            crate::vm::fast_path::OPEN_NATIVES_IN_PLACE.with(|count| count.set(0));
            let results = lua
                .exec(&chunk)
                .unwrap_or_else(|error| panic!("{profile:?} {mode}: {error}"));
            let in_place = crate::vm::fast_path::OPEN_NATIVES_IN_PLACE.with(std::cell::Cell::get);
            (results, in_place)
        };
        let (plain, in_place) = run_mode("none");
        assert!(in_place >= 1000, "{profile:?}: {in_place}");
        let (hooked, in_place) = run_mode("c");
        assert_eq!(hooked[0], plain[0], "{profile:?}");
        assert!(
            matches!(hooked[1], Value::Integer(2000)) || hooked[1] == Value::Number(2000.0),
            "{profile:?}: {hooked:?}"
        );
        assert_eq!(in_place, 0, "{profile:?}");
        let (counted, in_place) = run_mode("count");
        assert_eq!(counted[0], plain[0], "{profile:?}");
        assert!(
            matches!(counted[2], Value::Integer(count) if count > 1000)
                || matches!(counted[2], Value::Number(count) if count > 1000.0),
            "{profile:?}: {counted:?}"
        );
        assert_eq!(in_place, 0, "{profile:?}");
    }
}

/// An error in the second call of a fused pair names that call, on its own
/// line, as the unfused pair does. A count hook keeps the pair apart.
#[test]
fn fused_open_natives_report_errors_at_the_second_call() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let source = "
local floor, sub = math.floor, string.sub
local function f(x, fail)
  local y = floor(
    sub(x, fail and 1 or 2))
  return y
end
local mode = ...
if mode == 'count' then debug.sethook(function() end, '', 1000000) end
local good = f('12345', false)
local ok, message = pcall(f, 'abc', true)
debug.sethook()
return good, ok, message
";
        let run_mode = |mode: &str| {
            let mut lua = crate::Lua::builder()
                .profile(profile)
                .execution_mode(ExecutionMode::Interpreter)
                .build()
                .unwrap();
            let chunk = format!("return (function(...) {source} end)('{mode}')");
            crate::vm::fast_path::FUSED_OPEN_NATIVES.with(|count| count.set(0));
            let results = lua
                .exec(&chunk)
                .unwrap_or_else(|error| panic!("{profile:?} {mode}: {error}"));
            let fused = crate::vm::fast_path::FUSED_OPEN_NATIVES.with(std::cell::Cell::get);
            (results, fused)
        };
        let (plain, fused) = run_mode("none");
        assert!(fused >= 1, "{profile:?}: {fused}");
        let (counted, fused) = run_mode("count");
        assert_eq!(fused, 0, "{profile:?}");
        assert_eq!(plain, counted, "{profile:?}");
        assert!(
            matches!(plain[0], Value::Integer(2345)) || plain[0] == Value::Number(2345.0),
            "{profile:?}: {plain:?}"
        );
        assert_eq!(plain[1], Value::Boolean(false), "{profile:?}");
        let Value::String(message) = &plain[2] else {
            panic!("{profile:?}: {plain:?}");
        };
        let message = message.to_lossy();
        assert!(message.contains("floor"), "{profile:?}: {message}");
    }
}
