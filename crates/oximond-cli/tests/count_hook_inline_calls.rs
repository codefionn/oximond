//! A numeric callee evaluated without a frame still costs a count hook the
//! instructions its frame would have run. Tracing execution takes that
//! shortcut under a count hook. The interpreter pushes a lean frame in its
//! counted loop, which costs less than building the shortcut's inputs.
//! These programs check that the hook fires at the same instruction either
//! way, which a count of 1 (never enough budget for a shortcut) pins down.

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

fn number(value: &Value) -> f64 {
    match value {
        Value::Number(number) => *number,
        Value::Integer(integer) => f64::from(i32::try_from(*integer).expect("a small value")),
        other => panic!("expected a number, found {other:?}"),
    }
}

/// Every statement of the workload sits on its own line, so the line a
/// count hook interrupts identifies the instruction.
const POSITIONS: &str = r#"
local captured = 2.5
local function plain(a, b)
    local x = a * b
    return x + 1.5
end
local function with_upvalue(a)
    captured = captured + 0.25
    return a + captured
end
local function branching(a)
    if a > 3.5 then
        return a - 1.5
    end
    return a + 1.5
end
local function work()
    local total = 0.5
    for i = 1.5, 30.5 do
        total = total + plain(i, 0.5)
        total = total + with_upvalue(i)
        total = total + branching(i)
        total = total % 1000.5
    end
    return total
end
local function lines_for(count)
    local lines = {}
    captured = 2.5
    debug.sethook(function()
        lines[#lines + 1] = debug.getinfo(2, "l").currentline
    end, "", count)
    local result = work()
    debug.sethook()
    return lines, result
end
local reference, expected = lines_for(1)
local checked = 0
for _, count in ipairs({2, 3, 4, 5, 7, 11, 13, 50}) do
    local lines, result = lines_for(count)
    if result ~= expected then return -1 end
    if #lines ~= math.floor(#reference / count) then return -2 end
    for j = 1, #lines do
        if lines[j] ~= reference[j * count] then return -3 - j end
        checked = checked + 1
    end
end
return checked
"#;

#[test]
fn a_call_evaluated_without_a_frame_fires_the_hook_at_the_same_instruction() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua54,
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let (results, inlined) = run(POSITIONS, profile, mode);
            let checked = number(&results[0]);
            assert!(checked > 100.0, "{profile:?} {mode:?}: {checked}");
            if mode == ExecutionMode::Tracing {
                assert!(
                    inlined > 100,
                    "{profile:?}: the shortcut never ran: {inlined}"
                );
            } else {
                assert_eq!(inlined, 0, "{profile:?}: the counted loop pushes frames");
            }
        }
    }
}

/// The same count with and without the hook installed gives the same
/// answer, whatever the callee evaluates to.
#[test]
fn the_hook_does_not_change_inlined_results() {
    let source = r#"
        local hooked = ...
        local function f4(a, b, c, d) return a + b - c + d end
        local function f1(a) return a + 1.5 end
        if hooked then debug.sethook(function() end, "", 1000) end
        local total = 0.5
        for i = 1.5, 5000 do
            total = (total + f1(i) + f4(i, total, 2.5, 1.5)) % 1000003.5
        end
        debug.sethook()
        return total
    "#;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let with_hook = |hooked: bool| {
                let source =
                    source.replace("local hooked = ...", &format!("local hooked = {hooked}"));
                run(&source, profile, mode)
            };
            let (plain, hooked) = (with_hook(false), with_hook(true));
            assert_eq!(
                number(&plain.0[0]),
                number(&hooked.0[0]),
                "{profile:?} {mode:?}"
            );
            if mode == ExecutionMode::Tracing {
                assert!(hooked.1 > 5000, "{profile:?}: {}", hooked.1);
            } else {
                assert_eq!(hooked.1, 0, "{profile:?}");
            }
        }
    }
}

/// `opt/call_args_hook` is `opt/call_args` with a count hook installed: both
/// print the same checksum, at a scale small enough for a test build.
#[test]
fn the_call_args_benchmarks_agree() {
    let plain = include_str!("../../../benchmarks/lua/opt/call_args.lua");
    let hooked = include_str!("../../../benchmarks/lua/opt/call_args_hook.lua");
    let scale = "local scale = tonumber(arg and arg[1]) or 1";
    assert!(plain.contains(scale) && hooked.contains(scale));
    let small = |source: &str| {
        source
            .replace(scale, "local scale = 0.02")
            .replace("print(total)", "return total")
    };
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let (plain, _) = run(&small(plain), profile, mode);
            let (hooked, _) = run(&small(hooked), profile, mode);
            assert_eq!(plain, hooked, "{profile:?} {mode:?}");
        }
    }
}
