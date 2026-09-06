//! Count hooks let the fast loop run, counting instructions as it goes.
//! These programs check that it counts exactly what the dispatch loop
//! counts, including across calls, metamethods, and hook changes.

use oximond::compat::{CompatibilityProfile, ExecutionMode};
use oximond::compiler::{CompilerConfig, compile_with_config};
use oximond::runtime::Value;
use oximond::stdlib::install_base_with_profile;
use oximond::vm::Vm;

fn run(source: &str, profile: CompatibilityProfile, mode: ExecutionMode) -> Vec<Value> {
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).expect("compile");
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    vm.use_synchronous_jit_compilation();
    install_base_with_profile(vm.runtime_mut(), profile).expect("install standard library");
    vm.execute(&chunk).expect("run")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Number(number) => *number,
        Value::Integer(integer) => f64::from(i32::try_from(*integer).expect("a small count")),
        other => panic!("expected a number, found {other:?}"),
    }
}

/// A count of 1 fires before every counted instruction, which keeps the
/// fast loop out. Any larger count must then fire `floor(total / count)`
/// times: the hook's own instructions are never counted.
const WORKLOAD: &str = r#"
    local count = ...
    local fires = 0
    local mt = {
        __index = function(t, k) return k * 2 end,
        __add = function(a, b) return 5 end,
    }
    local t = setmetatable({}, mt)
    local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
    local function tail(n, acc) if n == 0 then return acc end return tail(n - 1, acc + n) end
    local s = 0
    debug.sethook(function() fires = fires + 1 end, "", count)
    for i = 1, 200 do
        s = s + fib(i % 8) + t[i] + tail(i % 5, 0) + (t + t)
        s = s + #tostring(i) + math.floor(i / 3)
        local ok = pcall(function() if i % 40 == 0 then error("x") end return i end)
        for k, v in ipairs({1, 2, 3}) do s = s + k * v end
        local word = ("ab"):rep(i % 4) .. i
        s = (s + #word) % 100003
    end
    debug.sethook()
    return fires, s
"#;

#[test]
fn counted_fast_loop_fires_exactly_as_the_dispatch_loop() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let with_count = |count: u32| {
                let source =
                    WORKLOAD.replace("local count = ...", &format!("local count = {count}"));
                let results = run(&source, profile, mode);
                (number(&results[0]), number(&results[1]))
            };
            let (total, checksum) = with_count(1);
            assert!(total > 10_000.0, "{profile:?} {mode:?}: {total}");
            for count in [2, 3, 7, 100, 1000] {
                let (fires, other_checksum) = with_count(count);
                assert_eq!(other_checksum, checksum);
                assert_eq!(
                    fires,
                    (total / f64::from(count)).floor(),
                    "{profile:?} {mode:?} count {count}"
                );
            }
        }
    }
}

#[test]
fn hooks_replaced_from_hooks_and_metamethods_take_effect() {
    let source = r#"
        local events = {count = 0, line = 0, call = 0}
        local fires = 0
        local hook
        hook = function(event)
            events[event] = (events[event] or 0) + 1
            fires = fires + 1
            if fires == 50 then debug.sethook(hook, "l") end
            if fires == 200 then debug.sethook(hook, "", 3) end
        end
        local switched = false
        local t = setmetatable({}, {__index = function(_, k)
            if k == 70 and not switched then
                switched = true
                debug.sethook(hook, "c", 0)
            end
            return k
        end})
        debug.sethook(hook, "", 5)
        local s = 0
        local function f(a) return a + 1 end
        for i = 1, 3000 do
            s = s + f(i)
            if fires > 400 then s = s + t[i % 100] end
            s = s % 9973
        end
        debug.sethook()
        return events.count, events.line, events.call, s
    "#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run(source, CompatibilityProfile::Lua54, mode);
        let (count, line, call) = (
            number(&results[0]),
            number(&results[1]),
            number(&results[2]),
        );
        // 49 count events, then 150 line events, then count events again
        // until the metamethod installs a call hook.
        assert_eq!(line, 150.0, "{mode:?}");
        assert!(count > 250.0, "{mode:?}: {count}");
        assert!(call > 1000.0, "{mode:?}: {call}");
    }
}

/// Call and return hooks let the fast loop run between calls. A line hook
/// keeps it out, and must not change which call and return events fire.
#[test]
fn call_and_return_events_match_the_dispatch_loop() {
    let source = r#"
        local mask, count = ...
        local log = {}
        local function hook(event)
            -- A line event skips its instruction's count, so only call
            -- and return events are comparable.
            if event ~= "line" and event ~= "count" then
                local info = debug.getinfo(2, "nS")
                log[#log + 1] = event .. ":" .. tostring(info.name) .. ":" .. info.what
            end
        end
        local mt = {
            __index = function(t, k) return k * 2 end,
            __add = function(a, b) return 5 end,
            __concat = function(a, b) return "cc" end,
            __lt = function(a, b) return true end,
        }
        local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
        local function tail(n, acc) if n == 0 then return acc end return tail(n - 1, acc + n) end
        local t = setmetatable({}, mt)
        local s = 0
        debug.sethook(hook, mask, count)
        for i = 1, 40 do
            s = s + fib(i % 6) + t[i] + tail(i % 4, 0) + (t + t) + #(t .. "x")
            s = s + (t < t and 1 or 0) + #tostring(i) + math.floor(i / 3)
            pcall(function() if i % 10 == 0 then error("x") end end)
            for _, v in ipairs({1, 2}) do s = s + v end
            local f = function(x) return x + i end
            s = (s + f(1)) % 100003
        end
        debug.sethook()
        return table.concat(log, ","), s
    "#;
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            for (mask, count) in [("c", 0), ("r", 0), ("cr", 0), ("cr", 5)] {
                let with_mask = |mask: &str| {
                    let source = source.replace(
                        "local mask, count = ...",
                        &format!("local mask, count = {mask:?}, {count}"),
                    );
                    run(&source, profile, mode)
                };
                let fast = with_mask(mask);
                let general = with_mask(&format!("{mask}l"));
                assert_eq!(fast, general, "{profile:?} {mode:?} {mask} {count}");
            }
        }
    }
}

/// The interpreter's counted loop pushes and pops Lua frames itself. Every
/// statement sits on its own line, so the line a count hook interrupts
/// identifies the instruction, and a count of 1 (which keeps the fast loop
/// out) gives the reference for each call shape.
const CALL_SHAPES: &str = r##"
local function plain(a, b)
    local x = a + b
    return x
end
local function two(a)
    local x = a + 1
    return x, a
end
local function variadic(...)
    local a, b = ...
    return (a or 0) + (b or 0)
end
local function spread(...)
    return ...
end
local function tail(n, acc)
    if n == 0 then
        return acc
    end
    return tail(n - 1, acc + n)
end
local function fails(i)
    if i % 7 == 0 then
        error("x")
    end
    return i
end
local callable = setmetatable({}, {__call = function(self, a)
    local x = a * 2
    return x
end})
local function work()
    local total = 0
    for i = 1, 40 do
        total = total + plain(i, 1)
        total = total + plain(two(i))
        total = total + variadic(i)
        total = total + variadic(spread(i, i))
        total = total + select("#", spread(two(i)))
        total = total + tail(i % 4, 0)
        total = total + callable(i)
        local ok, value = pcall(fails, i)
        total = total + (ok and value or 1)
        local packed = {two(i)}
        total = total + #packed
    end
    return total
end
local function lines_for(count)
    local lines = {}
    debug.sethook(function()
        lines[#lines + 1] = debug.getinfo(2, "l").currentline
    end, "", count)
    local result = work()
    debug.sethook()
    return lines, result
end
local reference, expected = lines_for(1)
local checked = 0
for _, count in ipairs({2, 3, 4, 5, 7, 11, 13, 50, 1000}) do
    local lines, result = lines_for(count)
    if result ~= expected then return -1 end
    if #lines ~= math.floor(#reference / count) then return -2 end
    for j = 1, #lines do
        if lines[j] ~= reference[j * count] then return -3 - j end
        checked = checked + 1
    end
end
-- A count hook that turns itself into a line hook: the lines reported
-- afterwards depend on the line each caller was left on, which the counted
-- loop restores when a callee returns.
local function lines_after(count, fire)
    local lines, fires = {}, 0
    local function on_line(_, line)
        lines[#lines + 1] = line
    end
    debug.sethook(function()
        fires = fires + 1
        if fires == fire then
            debug.sethook(on_line, "l")
        end
    end, "", count)
    work()
    debug.sethook()
    return lines
end
for _, total in ipairs({210, 630, 1050, 2310}) do
    local wanted = lines_after(1, total)
    if #wanted < 100 then return -1000 end
    for _, count in ipairs({2, 3, 5, 7, 10, 30, 105}) do
        local lines = lines_after(count, total / count)
        if #lines ~= #wanted then return -2000 end
        for j = 1, #lines do
            if lines[j] ~= wanted[j] then return -3000 - j end
            checked = checked + 1
        end
    end
end
return checked
"##;

#[test]
fn counted_calls_fire_the_hook_at_the_same_instruction() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let results = run(CALL_SHAPES, profile, mode);
            let checked = number(&results[0]);
            assert!(checked > 10_000.0, "{profile:?} {mode:?}: {checked}");
        }
    }
}
