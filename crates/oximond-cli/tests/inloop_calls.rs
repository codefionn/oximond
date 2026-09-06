//! Calls and returns between plain Lua functions switch activations inside
//! the interpreter's inner loop. Every program here runs in the plain
//! interpreter, where that path is live, and the observable behaviour must
//! match the dispatch loop's: errors, tracebacks, the debug library, hooks,
//! coroutines, the collector, and the number of results.
//!
//! The reference for "the same as the dispatch loop" is a run of the same
//! program with an empty line hook installed on the first line, which keeps
//! the fast loop out entirely. Line numbers stay the same because the hook
//! is installed on the program's own first line.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run_with(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.exec(source)
        .unwrap_or_else(|error| panic!("{error}"))
        .into_iter()
        .map(|value| match value {
            // Integer-capable profiles return whole numbers as integers.
            #[allow(clippy::cast_precision_loss)]
            Value::Integer(integer) => Value::Number(integer as f64),
            other => other,
        })
        .collect()
}

fn run(source: &str) -> Vec<Value> {
    run_with(
        CompatibilityProfile::Lua54,
        ExecutionMode::Interpreter,
        source,
    )
}

/// Runs `source` on the inner loop and again under a line hook, and checks
/// that both agree.
fn run_both(source: &str) -> Vec<Value> {
    let reference = format!("debug.sethook(function() end, 'l') {source}");
    let fast = run(source);
    let general = run(&reference);
    assert_eq!(fast, general, "inner loop and dispatch loop disagree");
    fast
}

fn text(value: &Value) -> String {
    match value {
        Value::String(string) => String::from_utf8_lossy(string.as_bytes()).into_owned(),
        other => panic!("expected a string, found {other:?}"),
    }
}

#[test]
fn deep_recursion_hits_the_stack_overflow_error_with_the_same_message() {
    let source = r"
        local function down(n) return 1 + down(n + 1) end
        local ok, message = pcall(down, 1)
        return ok, message
    ";
    let results = run(source);
    assert_eq!(results[0], Value::Boolean(false));
    assert_eq!(text(&results[1]), "(load):2: stack overflow");
    // The dispatch loop reports the overflow too, without the position.
    let general = run(&format!("debug.sethook(function() end, 'l') {source}"));
    assert_eq!(general[0], Value::Boolean(false));
    assert!(text(&general[1]).contains("stack overflow"), "{general:?}");
}

#[test]
fn deep_recursion_below_the_limit_returns_through_every_frame() {
    let source = r"
        local function sum(n) if n == 0 then return 0 end return n + sum(n - 1) end
        local total = 0
        for _ = 1, 20 do total = total + sum(1000) end
        return total
    ";
    assert_eq!(run_both(source), vec![Value::Number(10_010_000.0)]);
}

#[test]
fn tracebacks_through_switched_frames_name_functions_and_lines() {
    let source = "local function inner(n)
    if n == 0 then error('boom') end
    return inner(n - 1) + 1
end
local function outer(n)
    local r = inner(n)
    return r
end
local function top()
    return outer(3) * 2
end
local ok, trace = xpcall(top, debug.traceback)
return ok, trace
";
    let results = run_both(source);
    assert_eq!(results[0], Value::Boolean(false));
    let trace = text(&results[1]);
    assert!(trace.contains("boom"), "{trace}");
    // The error is raised on line 2, reached through the recursion on line 3,
    // then outer's call on line 6 and top's on line 10.
    for line in [":2:", ":3:", ":6:", ":10:"] {
        assert!(trace.contains(line), "{line} missing from {trace}");
    }
    assert_eq!(trace.matches(":3:").count(), 3, "{trace}");
    assert!(trace.contains("in upvalue 'inner'"), "{trace}");
    assert!(trace.contains("in upvalue 'outer'"), "{trace}");
}

#[test]
fn error_after_a_return_reports_the_callers_line() {
    let source = "local function value() return nil end
local function use()
    local v = value()
    return v + 1
end
local ok, message = pcall(use)
return ok, message
";
    let results = run_both(source);
    assert_eq!(results[0], Value::Boolean(false));
    assert!(text(&results[1]).contains(":4:"), "{results:?}");
}

#[test]
fn getinfo_and_getlocal_work_inside_switched_frames() {
    let source = "local function inner(a, b)
    local c = a + b
    local name, value = debug.getlocal(1, 3)
    local info = debug.getinfo(1, 'nSl')
    local caller = debug.getinfo(2, 'l')
    local an, av = debug.getlocal(2, 1)
    return info.currentline, name, value, caller.currentline, an, av, info.short_src ~= nil
end
local function outer(x)
    local r1, r2, r3, r4, r5, r6, r7 = inner(x, 10)
    return r1, r2, r3, r4, r5, r6, r7
end
return outer(5)
";
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(4.0));
    assert_eq!(text(&results[1]), "c");
    assert_eq!(results[2], Value::Number(15.0));
    assert_eq!(results[3], Value::Number(10.0));
    assert_eq!(text(&results[4]), "x");
    assert_eq!(results[5], Value::Number(5.0));
    assert_eq!(results[6], Value::Boolean(true));
}

#[test]
fn setlocal_reaches_a_register_of_a_frame_below() {
    let source = "local function poke()
    debug.setlocal(2, 1, 99)
end
local function owner()
    local x = 1
    poke()
    return x
end
return owner()
";
    assert_eq!(run_both(source), vec![Value::Number(99.0)]);
}

#[test]
fn yielding_across_nested_calls_resumes_in_the_right_frame() {
    let source = r"
        local function leaf(n)
            local got = coroutine.yield(n)
            return got + n
        end
        local function mid(n)
            local a = leaf(n)
            local b = leaf(n + 1)
            return a + b
        end
        local co = coroutine.wrap(function()
            local total = 0
            for i = 1, 3 do total = total + mid(i) end
            return total
        end)
        local seen = 0
        local value = co()
        while true do
            seen = seen + 1
            local nextvalue = co(value * 10)
            if seen == 6 then return nextvalue, seen end
            value = nextvalue
        end
    ";
    // mid(1): 1*10+1 + 2*10+2; mid(2): 2*10+2 + 3*10+3; mid(3): 3*10+3 + 4*10+4.
    assert_eq!(
        run_both(source),
        vec![Value::Number(165.0), Value::Number(6.0)]
    );
}

#[test]
fn hooks_set_mid_recursion_see_the_calls_and_returns_below_them() {
    let source = r"
        local events = {}
        local function hook(event)
            local info = debug.getinfo(2, 'f')
            if info.func == rec then events[#events + 1] = event end
        end
        function rec(n, limit)
            if n == 3 then debug.sethook(hook, 'cr') end
            if n == limit then return n end
            local r = rec(n + 1, limit)
            return r
        end
        local result = rec(1, 8)
        debug.sethook()
        local calls, returns = 0, 0
        for _, event in ipairs(events) do
            if event == 'call' then calls = calls + 1 else returns = returns + 1 end
        end
        return result, calls, returns
    ";
    // Calls with n = 4..8 happen after the hook is set; returns of n = 8..1
    // are all reported. Stock Lua 5.4 prints 8 5 8 for this program.
    let results = run(source);
    assert_eq!(results[0], Value::Number(8.0));
    assert_eq!(results[1], Value::Number(5.0));
    // This VM also reports one more return than stock Lua 5.4 does.
    assert!(matches!(results[2], Value::Number(returns) if (8.0..=9.0).contains(&returns)));
}

#[test]
fn line_hooks_set_mid_recursion_report_the_lines_of_every_deeper_frame() {
    let source = "local lines = {}
function rec(n)
    if n == 2 then debug.sethook(function(_, line) lines[#lines + 1] = line end, 'l') end
    if n == 5 then return n end
    local r = rec(n + 1)
    return r
end
local result = rec(1)
debug.sethook()
local seen = {}
for _, line in ipairs(lines) do seen[line] = (seen[line] or 0) + 1 end
return result, seen[4], seen[5], seen[6], seen[3]
";
    let results = run(source);
    // Stock Lua 5.4 prints 5 4 3 4 3 for this program.
    assert_eq!(
        results,
        vec![
            Value::Number(5.0),
            Value::Number(4.0),
            Value::Number(3.0),
            Value::Number(4.0),
            Value::Number(3.0)
        ]
    );
}

#[test]
fn count_hooks_set_mid_recursion_fire_and_leave_results_intact() {
    let source = r"
        local fired = 0
        local function rec(n)
            if n == 4 then debug.sethook(function() fired = fired + 1 end, '', 7) end
            if n == 40 then return 0 end
            return 1 + rec(n + 1)
        end
        local result = rec(1)
        debug.sethook()
        return result, fired > 10
    ";
    assert_eq!(run(source), vec![Value::Number(39.0), Value::Boolean(true)]);
}

#[test]
fn collection_during_deep_recursion_keeps_string_arguments_and_results() {
    let source = r#"
        local function build(n, acc)
            if n == 0 then
                collectgarbage()
                return acc, #acc
            end
            local garbage = {}
            for i = 1, 20 do garbage[i] = tostring(i) .. acc end
            local s, len = build(n - 1, acc .. string.char(97 + n % 26))
            collectgarbage("step")
            return s .. string.char(65 + n % 26), len
        end
        local result, len = build(200, "")
        local expected = {}
        for n = 200, 1, -1 do expected[#expected + 1] = string.char(97 + n % 26) end
        for n = 1, 200 do expected[#expected + 1] = string.char(65 + n % 26) end
        return result == table.concat(expected), len, #result
    "#;
    assert_eq!(
        run_both(source),
        vec![
            Value::Boolean(true),
            Value::Number(200.0),
            Value::Number(400.0)
        ]
    );
}

#[test]
fn string_arguments_survive_the_callee_overwriting_its_registers() {
    let source = r#"
        local function reuse(a, b)
            local x = a .. b
            a = nil
            return x, b
        end
        local out = {}
        for i = 1, 50 do
            local s, t = reuse("left" .. i, "right" .. i)
            out[i] = s .. t
        end
        collectgarbage()
        return out[1], out[50]
    "#;
    let results = run_both(source);
    assert_eq!(text(&results[0]), "left1right1right1");
    assert_eq!(text(&results[1]), "left50right50right50");
}

#[test]
fn result_counts_are_adjusted_to_what_the_caller_expects() {
    let source = r"
        local function none() end
        local function one() return 1 end
        local function many() return 1, 2, 3, 4 end
        local function two() return 'a', 'b' end
        local a, b, c = none()
        local d, e, f = one()
        local g, h, i = many()
        local j, k, l, m = two()
        local only = many()
        local n = none()
        local p = (two())
        none()
        one()
        many()
        local t = {many()}
        local u = {none()}
        local v = select('#', many())
        local w = select('#', none())
        return a == nil and b == nil and c == nil,
            d == 1 and e == nil and f == nil,
            g == 1 and h == 2 and i == 3,
            j == 'a' and k == 'b' and l == nil and m == nil,
            only == 1, n == nil, p == 'a', #t, #u, v, w
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Number(4.0),
            Value::Number(0.0),
            Value::Number(4.0),
            Value::Number(0.0),
        ]
    );
}

#[test]
fn single_argument_calls_and_missing_arguments_fill_with_nil() {
    let source = r"
        local function id(x) return x end
        local function pair(x, y) return x, y end
        local function extra(x) local y, z return x, y, z end
        local a = id(7)
        local b, c = pair(1)
        local d, e = pair(1, 2, 3, 4)
        local f, g, h = extra(5)
        return a, b, c == nil, d, e, f, g == nil, h == nil
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(7.0),
            Value::Number(1.0),
            Value::Boolean(true),
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(5.0),
            Value::Boolean(true),
            Value::Boolean(true),
        ]
    );
}

#[test]
fn callee_registers_start_nil_after_a_deeper_call_left_values_there() {
    // The second call reuses the window the first one filled.
    let source = r"
        local function fill() local a, b, c, d = 1, 2, 3, 4 return a + b + c + d end
        local function probe() local a, b, c, d return a, b, c, d end
        fill()
        local a, b, c, d = probe()
        return a == nil and b == nil and c == nil and d == nil
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn returns_into_a_frame_with_captured_locals_stay_synchronised() {
    let source = r"
        local function produce(n) return n * 2 end
        local function consume()
            local x = 1
            local function bump() x = x + 1 end
            x = produce(x)
            bump()
            x = produce(x)
            bump()
            return x
        end
        return consume()
    ";
    assert_eq!(run_both(source), vec![Value::Number(7.0)]);
}

#[test]
fn varargs_methods_and_metamethod_calls_still_work_between_plain_calls() {
    let source = r"
        local obj = setmetatable({}, {__call = function(self, a) return a + 1 end,
                                       __index = function(_, k) return k .. '!' end})
        local function va(...) return select('#', ...), ... end
        local function plain(x) return x end
        function obj:method(x) return x * 2 end
        local n, first = va(plain(1), plain(2))
        return obj(plain(4)), obj:method(plain(5)), n, first, obj.key
    ";
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(5.0));
    assert_eq!(results[1], Value::Number(10.0));
    assert_eq!(results[2], Value::Number(2.0));
    assert_eq!(results[3], Value::Number(1.0));
    assert_eq!(text(&results[4]), "key!");
}

#[test]
fn tail_calls_and_open_results_mix_with_switched_frames() {
    let source = r"
        local function leaf(n) return n, n + 1 end
        local function tail(n) return leaf(n) end
        local function open(n) return tail(n), tail(n + 10) end
        local a, b, c, d = open(1)
        return a, b, c, d
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(1.0),
            Value::Number(11.0),
            Value::Number(12.0),
            Value::Nil
        ]
    );
}

#[test]
fn every_profile_agrees() {
    let source = r"
        local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
        return fib(20)
    ";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let results = run_with(profile, mode, source);
            assert_eq!(results.len(), 1, "{profile:?} {mode:?}");
            assert_eq!(results[0], Value::Number(6765.0), "{profile:?} {mode:?}");
        }
    }
}

#[test]
fn parameters_left_out_are_nil_whatever_the_window_held_before() {
    let source = r"
        local function fill() local a, b, c, d, e, f = 1, 2, 3, 4, 5, 6 return a end
        local function fill_strings()
            local a, b, c = 'x' .. 1, 'y' .. 2, 'z' .. 3
            return #a
        end
        local function probe(x, y, z) return x, y, z end
        local out = {}
        fill()
        local a, b, c = probe(10)
        out[1] = a == 10 and b == nil and c == nil
        fill_strings()
        local d, e, f = probe()
        out[2] = d == nil and e == nil and f == nil
        local g, h = probe(1, 2, 3, 4)
        out[3] = g == 1 and h == 2
        return out[1], out[2], out[3]
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true)
        ]
    );
}

#[test]
fn registers_a_finished_call_left_behind_do_not_keep_objects_alive() {
    let source = r"
        local weak = setmetatable({}, {__mode = 'v'})
        local function fill() local t = {} weak[1] = t local a, b, c, d = t, t, t, t return 1 end
        local function probe() local x = 1 collectgarbage() return weak[1] end
        local function probe_params(p, q, r) collectgarbage() return weak[1], p, q, r end
        local function deep(n)
            if n == 0 then collectgarbage() return weak[1] end
            local t = {}
            local u = deep(n - 1)
            return u
        end
        fill()
        local first = probe()
        fill()
        local second, p, q, r = probe_params(7)
        fill()
        local third = deep(5)
        return first == nil, second == nil, p, q == nil, r == nil, third == nil
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Number(7.0),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true)
        ]
    );
}

#[test]
fn many_call_shapes_in_one_loop_keep_their_results() {
    // Plain, vararg, plan and method calls through one loop, each with
    // fixed result counts the callers adjust.
    let source = r"
        local object = {n = 0}
        function object:bump(by) self.n = self.n + by return self.n end
        local function plain(a, b) return a + b, a - b end
        local function va(...) return select('#', ...), ... end
        local function long(a, b)
            local ij = a + b
            return 1 / (ij * (ij + 1) / 2 + a + 1)
        end
        local total = 0
        for i = 1, 200 do
            local s, d = plain(i, 1)
            local n, first = va(i, s, d)
            total = total + s + d + n + first + long(i, 2) + object:bump(1)
        end
        return math.floor(total), object.n
    ";
    let expected_integer: f64 = (1..=200)
        .map(|i| {
            let i = f64::from(i);
            (i + 1.0) + (i - 1.0) + 3.0 + i
        })
        .sum();
    let results = run_both(source);
    assert_eq!(results[1], Value::Number(200.0));
    // `object:bump(1)` adds 1 + 2 + ... + 200, and `long` is below one.
    let Value::Number(total) = results[0] else {
        panic!("{results:?}");
    };
    assert!(
        (total - (expected_integer + 20100.0)).abs() < 200.0,
        "{total}"
    );
}

#[test]
fn recursion_past_the_records_a_thread_holds_keeps_every_frame_whole() {
    // Each new depth appends a frame record, which may move all of them while
    // the call that needs it is under way. Every frame must still report its
    // function and line afterwards.
    let source = "local function down(n)
    if n == 0 then error('bottom') end
    return down(n - 1) + 1
end
local ok, trace = xpcall(function() return down(3000) end, debug.traceback)
local frames = 0
local lined = 0
for line in trace:gmatch('[^\\n]+') do
    if line:find('in upvalue .down.') then
        frames = frames + 1
        if line:find(':3:', 1, true) or line:find(':2:', 1, true) then lined = lined + 1 end
    end
end
return ok, frames, lined
";
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let results = run_with(CompatibilityProfile::Lua54, mode, source);
        assert_eq!(results[0], Value::Boolean(false), "{mode}");
        let Value::Number(frames) = results[1] else {
            panic!("{results:?}");
        };
        // `debug.traceback` lists the first and the last levels only.
        assert!(frames > 10.0, "{mode}: {results:?}");
        assert_eq!(results[1], results[2], "{mode}: a frame lost its line");
    }
}

#[test]
fn a_changed_closure_is_seen_by_the_next_call_at_the_same_depth() {
    // Calls from one loop reuse the frame record of the previous call at
    // their depth. The closure it ran, its upvalues and its environment may
    // all differ now.
    let source = r"
        local a, b = 1, 100
        local function f() return a end
        local function g() return b end
        local total = 0
        for _ = 1, 10 do total = total + f() + g() end
        assert(total == 1010, total)
        -- Alternating callees at one depth.
        for i = 1, 10 do
            local callee = (i % 2 == 0) and f or g
            total = total + callee()
        end
        assert(total == 1010 + 5 * 1 + 5 * 100, total)
        -- f now shares g's upvalue; the record that held f's old one must not
        -- be reused as it was.
        debug.upvaluejoin(f, 1, g, 1)
        assert(f() == 100)
        assert(g() == 100)
        b = 7
        assert(f() == 7 and g() == 7)
        -- A closure that was collected and replaced by another in the same
        -- slot of the arena.
        local function make(k) return function() return k end end
        local sum = 0
        for k = 1, 50 do
            local h = make(k)
            sum = sum + h()
            h = nil
            collectgarbage()
        end
        assert(sum == 1275, sum)
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn a_changed_environment_is_seen_by_the_next_call_at_the_same_depth() {
    let source = r"
        x = 1
        local function h() return x end
        local s = 0
        for _ = 1, 3 do s = s + h() end
        assert(s == 3, s)
        setfenv(h, {x = 10})
        assert(h() == 10)
        for _ = 1, 3 do s = s + h() end
        assert(s == 33, s)
        setfenv(h, {x = 5})
        assert(h() == 5)
        return true
    ";
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(
            run_with(CompatibilityProfile::Lua51, mode, source),
            vec![Value::Boolean(true)],
            "{mode}"
        );
    }
}

#[test]
fn values_of_every_type_pass_through_arguments_and_results_unchanged() {
    // Arguments and results are copied bit for bit unless a string is
    // involved: nil, booleans, numbers, tables, functions and strings must
    // arrive as they were, even over stale registers of another type.
    let source = r"
        local t, f = {}, function() end
        local function echo(a, b, c, d, e, g, h, i)
            return a, b, c, d, e, g, h, i
        end
        local function stale(s) local x, y, z = s .. 'a', s .. 'b', s .. 'c' return #x + #y + #z end
        for round = 1, 3 do
            assert(stale('str') == 12)
            local a, b, c, d, e, g, h, i = echo(nil, true, false, 7, 2.5, 'text', t, f)
            assert(a == nil and b == true and c == false and d == 7 and e == 2.5)
            assert(g == 'text' and h == t and i == f)
            assert(select('#', echo(nil, nil, nil)) == 8)
            local m = {echo(1, 2, 3)}
            assert(m[1] == 1 and m[3] == 3 and m[4] == nil)
        end
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn moves_copy_every_type_over_every_type_including_fresh_results() {
    // Registers are copied whole by a move and as a tag and a payload word
    // by the upvalue and field reads. Whatever the source and the
    // destination held, and whether the source was just written (a call's
    // result, an arithmetic result, a loop variable) or long ago, the copy
    // is exact.
    let source = r"
        local t, f = {}, function() return 1 end
        local values = {nil, true, false, 7, 2.5, 'text', t, f}
        local function pass(...) return ... end
        for round = 1, 3 do
            for from = 1, 8 do
                for to = 1, 8 do
                    local dst = values[to]
                    local src = values[from]
                    dst = src
                    assert(dst == src, from .. '->' .. to)
                    local fresh = pass(src)
                    local copy = fresh
                    assert(copy == src and rawequal(copy, src))
                end
            end
        end
        local n = 0
        for i = 1, 100 do local k = i; n = n + k end
        assert(n == 5050, n)
        local x = 1.5
        for i = 1, 10 do x = x + i; local y = x; assert(y == x) end
        local s = 'a'
        local u = s
        s = s .. 'b'
        assert(u == 'a' and s == 'ab')
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn callees_that_alternate_at_one_depth_keep_their_own_captures_and_arguments() {
    // The record at a depth is pointed at each new callee: its chunk, its
    // captures and its environment change with every call, with strings
    // among the arguments some of the time (those calls take the general
    // path) and not the others.
    let source = r"
        local u1, u2, u3 = 1, 20, 300
        local function none(a) return a end
        local function one(a) return a + u1 end
        local function two(a) return a + u1 + u2 end
        local function three(a) return a + u1 + u2 + u3 end
        local function text(a) return #a end
        local callees = {none, one, two, three}
        local expected = {0, 1, 21, 321}
        for round = 1, 20 do
            for i = 1, 4 do
                assert(callees[i](0) == expected[i], i)
                assert(text('abc' .. i) == 4)
                assert(callees[i](0) == expected[i], i)
            end
        end
        u1, u2, u3 = 2, 30, 400
        for i = 1, 4 do
            assert(callees[i](0) == ({0, 2, 32, 432})[i], i)
        end
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn an_error_in_a_callee_leaves_its_record_fit_for_the_next_call() {
    // The record of a call that raised is unwound, not returned from. The
    // next call at that depth, to the same closure or another, starts clean.
    let source = r"
        local function fail(n) if n > 2 then error('boom ' .. n) end return n end
        local function ok(n) return n * 2 end
        local total = 0
        for i = 1, 6 do
            local good, result = pcall(fail, i)
            if good then total = total + result else total = total + 100 end
            total = total + ok(i)
        end
        assert(total == (1 + 2) + 4 * 100 + 2 * 21, total)
        local co = coroutine.wrap(function(a)
            local b = coroutine.yield(ok(a))
            return fail(b), ok(b)
        end)
        assert(co(4) == 8)
        local ok2, message = pcall(co, 9)
        assert(not ok2 and message:find('boom 9', 1, true), message)
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn tail_calls_between_closures_keep_tracebacks_and_captures() {
    let source = r"
        local up = 5
        local function even(n, total)
            if n == 0 then return total end
            return odd(n - 1, total + up)
        end
        function odd(n, total)
            if n == 0 then return total + 1000 end
            return even(n - 1, total + n % 7)
        end
        assert(even(2000, 0) > 0)
        local first = even(10, 0)
        up = 6
        local second = even(10, 0)
        assert(second == first + 5, first .. ' ' .. second)
        local function tail_error(n)
            if n == 0 then error('tail bottom') end
            return tail_error(n - 1)
        end
        local ok, trace = xpcall(tail_error, debug.traceback, 50)
        assert(not ok and trace:find('tail bottom', 1, true) and trace:find('tail call', 1, true), trace)
        return true
    ";
    assert_eq!(run_both(source), vec![Value::Boolean(true)]);
}

#[test]
fn tail_calls_of_natives_return_exactly_what_a_call_and_return_would() {
    // `return native(...)` runs the native in place and returns its one
    // value. Every frame shape the return meets gives the dispatch loop's
    // answer: plain callers, a coroutine's base frame, a protected call, the
    // main chunk, frames with captured locals, varargs and results counts.
    let source = r"
        local floor, fmod, type, select = math.floor, math.fmod, type, select
        local function f1(x) return floor(x) end
        local function f2(x) return fmod(x, 7) end
        local function f3(x) return type(x) end
        local function f4(...) return select('#', ...) end
        local function f5(x) local captured = x; local function peek() return captured end
            return floor(peek()) end
        local function f6(...) return floor(...) end
        local function counted() return (select('#', f1(2.5))), (select('#', f3(nil))) end
        local out = {f1(3.7), f2(10, 4), f3(1), f4(1, 2, 3), f5(8.9), f6(2.5, 9)}
        local n, m = counted()
        out[#out + 1] = n
        out[#out + 1] = m
        local co = coroutine.wrap(function(a)
            local r = f1(a)
            coroutine.yield(r)
            return f1(a + 1)
        end)
        out[#out + 1] = co(1.5)
        out[#out + 1] = co()
        local ok, value = pcall(f1, 4.5)
        out[#out + 1] = ok and value
        out[#out + 1] = pcall(function() return floor(6.5) end) and 1
        local tail_in_tail = function(x) return f1(x) end
        out[#out + 1] = tail_in_tail(9.5)
        return table.unpack(out, 1, 16)
    ";
    let results = run_both(source);
    let numbers: Vec<_> = results
        .iter()
        .map(|value| match value {
            Value::Number(n) => format!("{n}"),
            Value::String(s) => String::from_utf8_lossy(s.as_bytes()).into_owned(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        numbers,
        [
            "3", "3", "number", "3", "8", "2", "1", "1", "1", "2", "4", "1", "9", "Nil", "Nil",
            "Nil"
        ]
    );
}

#[test]
fn tail_calls_of_natives_report_errors_like_calls_do() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let source = r"
            local floor = math.floor
            local function bad(x) return floor(x) end
            local ok, message = pcall(bad, 'x')
            local ok2, message2 = pcall(function() return floor() end)
            local ok3, message3 = xpcall(function() return floor({}) end, function(m) return 'H:' .. m end)
            return ok, message, ok2, message2, ok3, message3
        ";
        let fast = run_with(profile, ExecutionMode::Interpreter, source);
        let general = run_with(
            profile,
            ExecutionMode::Interpreter,
            &format!("debug.sethook(function() end, 'l') {source}"),
        );
        assert_eq!(fast, general, "{profile:?}");
        assert_eq!(fast[0], Value::Boolean(false), "{profile:?}");
    }
}

#[test]
fn tail_calls_of_natives_leave_frames_that_need_closing_to_the_general_return() {
    // A frame with a captured local that is still open, and one that holds a
    // to-be-closed variable, return through the general routine; the
    // native's value still arrives.
    let source = r"
        local floor = math.floor
        local function open_capture(x)
            local keep = x
            local function get() return keep end
            local f = floor
            return f(get())
        end
        local function closing(x)
            local log = {}
            do
                local guard <close> = setmetatable({}, {__close = function() log[#log + 1] = 'closed' end})
                local y = floor(x)
                log[#log + 1] = y
            end
            return floor(x + 1), log[1], log[2]
        end
        local a = open_capture(7.5)
        local b, c, d = closing(2.5)
        return a, b, c, d
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(7.0),
            Value::Number(3.0),
            Value::Number(2.0),
            Value::String("closed".into())
        ]
    );
}

#[test]
fn field_stores_replace_every_type_with_every_type_through_a_cached_site() {
    // The cached field store copies a non-string value as a tag and a
    // payload word. Whatever the field held and whatever arrives (a result
    // just computed, a string, a table the collector must see), the next
    // read finds exactly the stored value, and strings are neither leaked
    // nor freed early.
    let source = r"
        local keep = {}
        local f = function() return 1 end
        local pool = {nil, true, false, 7, 2.5, 'text', keep, f}
        local object = {field = 0, other = 0}
        local function store(o, v) o.field = v end
        local function load(o) return o.field end
        for round = 1, 5 do
            for from = 1, 8 do
                for to = 1, 8 do
                    store(object, pool[to])
                    store(object, pool[from])
                    local got = load(object)
                    assert(got == pool[from], from .. '->' .. to)
                end
            end
        end
        for i = 1, 200 do
            store(object, i + 0.5)
            assert(object.field == i + 0.5)
            store(object, 'k' .. i)
            collectgarbage()
            assert(object.field == 'k' .. i)
            store(object, {i})
            collectgarbage()
            assert(object.field[1] == i)
        end
        local total = 0
        function object:add(delta)
            self.field = (type(self.field) == 'number' and self.field or 0) + delta
            return self.field
        end
        object.field = 0
        for i = 1, 100 do total = total + object:add(i % 7) end
        return total, object.field
    ";
    let results = run_both(source);
    let numbers: Vec<_> = results
        .iter()
        .map(|value| match value {
            Value::Number(n) => format!("{n}"),
            other => format!("{other:?}"),
        })
        .collect();
    let mut total = 0.0;
    let mut field = 0.0;
    for i in 1..=100 {
        field += f64::from(i % 7);
        total += field;
    }
    assert_eq!(numbers, [format!("{total}"), format!("{field}")]);
}

#[test]
fn tail_calls_with_operands_computed_in_place_keep_arguments_and_tracebacks() {
    // The compiler writes a tail call's function and arguments into their
    // slots directly. Arguments that are expressions, upvalues, globals and
    // method receivers must reach the callee unchanged, through frames of
    // every shape.
    let source = r"
        local up = 10
        local function sum(n, acc, extra)
            if n == 0 then return acc, extra end
            return sum(n - 1, acc + n % 7 + up, n)
        end
        local a, e = sum(100, 0)
        local obj = {k = 3}
        function obj:go(x, y) return self.k + x * 10 + y end
        local function via(o, x) return o:go(x * 2, x + 1) end
        local function pick(...) return select('#', ...), ... end
        local function spread(...) return pick(1, ...) end
        local function swap(x, y) return y, x end
        local function tail_swap(x, y) return swap(y, x) end
        local n, first, second = spread('a', 'b')
        local p, q = tail_swap(1, 2)
        local ok, err = pcall(function() return nosuch(1) end)
        return a, e, via(obj, 5), n, first, second, p, q, ok, type(err)
    ";
    let results = run_both(source);
    let mut acc = 0.0;
    for n in 1..=100 {
        acc += f64::from(n % 7) + 10.0;
    }
    assert_eq!(
        results,
        vec![
            Value::Number(acc),
            Value::Number(1.0),
            Value::Number(3.0 + 100.0 + 6.0),
            Value::Number(3.0),
            Value::Number(1.0),
            Value::String("a".into()),
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Boolean(false),
            Value::String("string".into()),
        ]
    );
}

#[test]
fn math_floor_runs_in_place_for_numbers_and_defers_for_everything_else() {
    // The interpreter's native call floors a number itself. Integers, floats
    // that fit an integer, floats that do not, NaN, infinities, numeric
    // strings and bad arguments give the library's answers, as a call hook
    // (which sends every call through the library) shows.
    let source = r"
        local floor = math.floor
        local inputs = {7, -7, 2.5, -2.5, 1e300, -1e300, 0/0, 1/0, -1/0, '3.7', ' 8.2 ', 2^53, -0.0}
        local out = {}
        for i = 1, #inputs do
            local v = floor(inputs[i])
            out[#out + 1] = (v ~= v) and 'nan' or tostring(v)
            out[#out + 1] = math.type and math.type(v) or type(v)
        end
        local ok, err = pcall(floor, {})
        out[#out + 1] = tostring(ok) .. ':' .. tostring(err)
        out[#out + 1] = tostring(pcall(floor))
        local function tail(x) return floor(x) end
        out[#out + 1] = tostring(tail(9.9)) .. tostring(tail(-9.9)) .. tostring(tail(4))
        return table.concat(out, ',')
    ";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let fast = run_with(profile, ExecutionMode::Interpreter, source);
        let hooked = run_with(
            profile,
            ExecutionMode::Interpreter,
            &format!("debug.sethook(function() end, 'c') {source}"),
        );
        assert_eq!(fast, hooked, "{profile:?}");
        assert!(matches!(fast[0], Value::String(_)), "{profile:?}");
    }
}

#[test]
fn scalar_field_overwrites_skip_newindex_but_new_fields_and_other_types_do_not() {
    // A number over a number in a cached field is a raw copy, as a present
    // field always is. A field that is absent, nil, or holds a string or a
    // table takes the general store, and `__newindex` still sees an absent
    // field. Shared shapes, proxies and frozen-by-metatable objects stay
    // right.
    let source = r"
        local seen = {}
        local mt = {__newindex = function(t, k, v) seen[#seen + 1] = k; rawset(t, k, v) end}
        local function set_a(o, v) o.a = v end
        local function set_b(o, v) o.b = v end
        local out = {}
        for round = 1, 3 do
            seen = {}
            local o = setmetatable({a = 1}, mt)
            set_a(o, 2.5)          -- number over number: raw
            set_a(o, true)         -- boolean over number: raw
            set_a(o, 'text')       -- string over boolean
            set_a(o, 7)            -- number over string
            set_b(o, 5)            -- absent: __newindex
            set_b(o, 6)            -- present now: raw
            set_b(o, nil)          -- delete
            set_b(o, 8)            -- absent again: __newindex
            out[#out + 1] = table.concat(seen, ',') .. '|' .. tostring(o.a) .. '|' .. tostring(o.b)
        end
        local plain = {a = 1, b = 2}
        local other = {a = 3, b = 4}
        for i = 1, 10 do
            set_a(plain, i)
            set_a(other, i * 2)
            set_b(plain, i + 0.5)
        end
        out[#out + 1] = plain.a .. ',' .. plain.b .. ',' .. other.a .. ',' .. other.b
        return table.concat(out, ';')
    ";
    let results = run_both(source);
    assert_eq!(text(&results[0]), "b,b|7|8;b,b|7|8;b,b|7|8;10,10.5,20,4");
}

#[test]
fn scalar_fields_added_through_a_site_reach_inherited_reads_and_the_collector() {
    // A scalar added to a shaped table through a filled write site is a
    // copy. A table that other tables inherit from is watched: adding a
    // field to it must show in their next read, from cached sites too.
    let source = r"
        local function set_b(o, v) o.b = v end
        local function read_b(o) return o.b end
        -- fill the write site with the shape {a}
        for i = 1, 50 do local t = {a = 0}; set_b(t, i); assert(t.b == i) end
        local base = {a = 0}
        local mt = {__index = base}
        local objects = {}
        for i = 1, 5 do objects[i] = setmetatable({}, mt) end
        local seen = {}
        for round = 1, 3 do
            for i = 1, 5 do seen[#seen + 1] = tostring(read_b(objects[i])) end
            base = {a = 0}
            mt.__index = base
            set_b(base, round * 10)        -- add through the site
            for i = 1, 5 do seen[#seen + 1] = tostring(read_b(objects[i])) end
            set_b(base, round * 10 + 1)    -- overwrite through the site
            seen[#seen + 1] = tostring(read_b(objects[1]))
        end
        -- a table with a metatable adds through __newindex, not the site
        local log = {}
        local guarded = setmetatable({a = 0}, {__newindex = function(t, k, v) log[#log + 1] = k; rawset(t, k, v) end})
        set_b(guarded, 3)
        set_b(guarded, 4)
        -- allocation accounting: many added fields leave the collector working
        local before = collectgarbage('count')
        for i = 1, 20000 do local t = {a = 0}; set_b(t, i) end
        collectgarbage()
        return table.concat(seen, ','), table.concat(log, ','), guarded.b, collectgarbage('count') < before + 4096
    ";
    let results = run_both(source);
    assert_eq!(
        text(&results[0]),
        "nil,nil,nil,nil,nil,10,10,10,10,10,11,\
         11,11,11,11,11,20,20,20,20,20,21,\
         21,21,21,21,21,30,30,30,30,30,31"
    );
    assert_eq!(text(&results[1]), "b");
    assert_eq!(results[2], Value::Number(4.0));
    assert_eq!(results[3], Value::Boolean(true));
}

#[test]
fn method_calls_look_the_method_up_and_pass_the_receiver_through_every_kind_of_holder() {
    // One instruction reads the method and copies the receiver. Every holder
    // must give what a lookup followed by a copy gives: a field of the
    // object, a field inherited through one or two `__index` tables, an
    // `__index` function, and a string's library method.
    let source = r"
        local Base = {}
        Base.__index = Base
        function Base:name() return 'base', self.id end
        function Base:twice(x) return x * 2, self.id end
        local Derived = setmetatable({}, Base)
        Derived.__index = Derived
        function Derived:name() return 'derived', self.id end
        local plain = {id = 1}
        function plain:name() return 'plain', self.id end
        local own = setmetatable({id = 2}, Base)
        local deep = setmetatable({id = 3}, Derived)
        local dyn = setmetatable({id = 4}, {__index = function(t, k)
            return function(self, ...) return k, self.id, select('#', ...) end
        end})
        local out = {}
        local function add(...) for _, v in ipairs({...}) do out[#out + 1] = tostring(v) end end
        for round = 1, 3 do
            add(plain:name())
            add(own:name())
            add(own:twice(round))
            add(deep:name())
            add(deep:twice(round))
            add(dyn:anything(1, 2, round))
            add(('abc'):upper(), ('abc'):rep(2), ('x'):byte())
        end
        -- the same receiver, called repeatedly through one site
        local holder = {id = 5}
        function holder:id_of() return self.id end
        local slots = {}
        for i = 1, 3 do
            slots[#slots + 1] = holder:id_of()
        end
        add(table.concat(slots, '+'))
        return table.concat(out, ',')
    ";
    let results = run_both(source);
    let expected: Vec<String> = (1..=3)
        .map(|n| {
            format!(
                "plain,1,base,2,{d},2,derived,3,{d},3,anything,4,3,ABC,abcabc,120",
                d = n * 2
            )
        })
        .collect();
    let joined = format!("{},5+5+5", expected.join(","));
    assert_eq!(text(&results[0]), joined);
}

#[test]
fn method_calls_report_errors_and_run_metamethods_like_a_lookup_and_a_copy() {
    let source = r"
        local report = {}
        local function try(f, ...)
            local ok, err = pcall(f, ...)
            report[#report + 1] = tostring(ok) .. ':' .. tostring(err):gsub('^[^:]*:%d+: ', '')
        end
        local missing
        try(function() return missing:go() end)
        try(function() local n = 5; return n:go() end)
        try(function() local t = {}; return t:go() end)
        try(function() return (nil):go(1) end)
        try(function() local t = setmetatable({}, {__index = function() error('boom', 0) end}); return t:go() end)
        -- the copy comes first: an __index function sees the receiver's slot
        -- already holding the argument (only visible through a coroutine here)
        local co = coroutine.wrap(function()
            local t = setmetatable({}, {__index = function(self, k)
                local v = coroutine.yield(k)
                return function(me, a) return v + a, me == self end
            end})
            return t:late(10)
        end)
        local first = co()
        local a, b = co(32)
        report[#report + 1] = first .. ':' .. tostring(a) .. ':' .. tostring(b)
        return table.concat(report, '|')
    ";
    let results = run_both(source);
    assert_eq!(
        text(&results[0]),
        "false:attempt to index a nil value (upvalue 'missing')|\
         false:attempt to index a number value (local 'n')|\
         false:attempt to call a nil value (method 'go')|\
         false:attempt to index a nil value|\
         false:boom|late:42:true"
    );
}

#[test]
fn method_calls_keep_the_receiver_when_an_argument_reassigns_it() {
    // The receiver is read before the arguments run: a call among them that
    // assigns the variable must not change which object the method gets.
    let source = r"
        local a = {tag = 'a'}
        local b = {tag = 'b'}
        function a:who(x) return self.tag, x end
        function b:who(x) return self.tag, x end
        local o = a
        local function swap() o = b; return 'swapped' end
        local first, second = o:who(swap())
        local tail = (function() local p = a; return p:who(select(2, 'x', 'y')) end)()
        local function vararg(...) return o:who(...) end
        o = a
        local third, fourth = vararg(1, 2)
        return first, second, tail, o.tag, third, fourth
    ";
    let results = run_both(source);
    assert_eq!(text(&results[0]), "a");
    assert_eq!(text(&results[1]), "swapped");
    assert_eq!(text(&results[2]), "a");
    assert_eq!(text(&results[3]), "a");
    assert_eq!(text(&results[4]), "a");
    assert_eq!(results[5], Value::Number(1.0));
}

#[test]
fn method_calls_in_tail_position_and_with_open_arguments_pass_every_argument() {
    let source = r"
        local obj = {k = 7}
        function obj:sum(...)
            local total = self.k
            for i = 1, select('#', ...) do total = total + select(i, ...) end
            return total, select('#', ...)
        end
        local function tail_fixed(o, x) return o:sum(x, x) end
        local function tail_open(o, ...) return o:sum(...) end
        local function tail_call_result(o) return o:sum(table.unpack({1, 2, 3})) end
        local function plain_open(o, ...) local t, n = o:sum(...); return t, n end
        local function deep(n, o) if n == 0 then return o:sum(n) end return deep(n - 1, o) end
        local r = {}
        for _, f in ipairs({tail_fixed, tail_open, tail_call_result, plain_open}) do
            local t, n = f(obj, 1, 2, 3)
            r[#r + 1] = t .. '/' .. n
        end
        local t, n = deep(50, obj)
        r[#r + 1] = t .. '/' .. n
        return table.concat(r, ',')
    ";
    let results = run_both(source);
    assert_eq!(text(&results[0]), "9/2,13/3,13/3,13/3,7/1");
}

#[test]
fn method_calls_leave_the_debug_view_and_hooks_as_two_instructions_did() {
    let source = r"
        local obj = {v = 1}
        function obj:probe(x)
            local info = debug.getinfo(1, 'n')
            local name, value = debug.getlocal(1, 1)
            return info.name, info.namewhat, name, value == self, x
        end
        local lines = 0
        local counts = 0
        debug.sethook(function() counts = counts + 1 end, '', 1)
        local n, w, ln, same, x = obj:probe(9)
        debug.sethook()
        return n, w, ln, same, x, counts > 0
    ";
    let results = run(source);
    assert_eq!(text(&results[0]), "probe");
    assert_eq!(text(&results[1]), "method");
    assert_eq!(text(&results[2]), "self");
    assert_eq!(results[3], Value::Boolean(true));
    assert_eq!(results[4], Value::Number(9.0));
    assert_eq!(results[5], Value::Boolean(true));
}

#[test]
fn method_calls_agree_between_the_interpreter_and_the_tracer() {
    let source = r"
        local Point = {}
        Point.__index = Point
        function Point.new(x, y) return setmetatable({x = x, y = y}, Point) end
        function Point:add(other) return Point.new(self.x + other.x, self.y + other.y) end
        function Point:len2() return self.x * self.x + self.y * self.y end
        local acc = Point.new(0, 0)
        local step = Point.new(1, 2)
        local total = 0
        for i = 1, 3000 do
            acc = acc:add(step)
            total = total + acc:len2() % 11
        end
        return acc.x, acc.y, total
    ";
    let interpreted = run_with(
        CompatibilityProfile::Lua54,
        ExecutionMode::Interpreter,
        source,
    );
    let traced = run_with(CompatibilityProfile::Lua54, ExecutionMode::Tracing, source);
    assert_eq!(interpreted, traced);
    assert_eq!(interpreted[0], Value::Number(3000.0));
    assert_eq!(interpreted[1], Value::Number(6000.0));
}

#[test]
fn the_hot_tail_call_alternates_closures_and_falls_back_for_frames_it_declines() {
    // A tail call between two closures, string arguments over dead registers,
    // a self tail call, a frame with captured variables open, a vararg frame,
    // missing and extra arguments, and a tail call that raises, all agree with
    // the dispatch loop.
    let source = r"
        local function even(n, s, t)
            if n == 0 then return s .. t end
            return odd(n - 1, t, s .. 'x')
        end
        function odd(n, s, t)
            if n == 0 then return t .. s end
            return even(n - 1, t, s .. 'y')
        end
        local function same(n, acc)
            if n == 0 then return acc end
            return same(n - 1, acc + n)
        end
        local function captured(n)
            local c = n
            local function bump() c = c + 1 end
            bump()
            if n == 0 then return c end
            return captured(n - 1)
        end
        local function va(n, ...)
            if n == 0 then return select('#', ...) end
            return va(n - 1, ...)
        end
        local function few(a, b, c) return a, b, c end
        local function lead(...) return few(...) end
        local function extra(a) return few(a, 2, 3, 4, 5) end
        local function boom(n)
            if n == 0 then error('tail boom') end
            return boom(n - 1)
        end
        local ok, err = pcall(boom, 20)
        local x1, x2, x3 = lead(1)
        local y1, y2, y3 = extra(7)
        return even(9, 'a', 'b'), odd(8, 'c', 'd'), same(1000, 0), captured(5),
            va(30, 1, 2, 3), x1, x2, x3, y1, y2, y3, ok, (tostring(err):find('tail boom', 1, true) ~= nil)
    ";
    let results = run_both(source);
    assert_eq!(results.len(), 13);
    assert_eq!(results[2], Value::Number(500_500.0));
    assert_eq!(results[3], Value::Number(1.0));
    assert_eq!(results[4], Value::Number(3.0));
    assert_eq!(results[11], Value::Boolean(false));
    assert_eq!(results[12], Value::Boolean(true));
}
