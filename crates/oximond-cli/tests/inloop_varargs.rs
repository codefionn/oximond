//! Vararg functions keep their extra arguments on the value stack, below the
//! window of the activation, and the interpreter's inner loop calls them,
//! forwards them (`f(...)`, `return f(...)`) and returns them without leaving
//! the loop. Tail calls of Lua closures replace the activation in place.
//!
//! The reference for "the same as the dispatch loop" is a run of the same
//! program with an empty line hook installed on the first line, which keeps
//! the fast loop out entirely.

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

fn run_profile(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    run_with(profile, ExecutionMode::Interpreter, source)
}

fn run(source: &str) -> Vec<Value> {
    run_profile(CompatibilityProfile::Lua54, source)
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

fn numbers(values: &[f64]) -> Vec<Value> {
    values.iter().map(|&number| Value::Number(number)).collect()
}

fn text(value: &Value) -> String {
    match value {
        Value::String(string) => String::from_utf8_lossy(string.as_bytes()).into_owned(),
        other => panic!("expected a string, found {other:?}"),
    }
}

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

#[test]
fn select_count_sees_trailing_nils() {
    let source = r"
        local function n(...) return select('#', ...) end
        return n(), n(nil), n(nil, nil), n(1, nil, nil), n(nil, 1, nil)
    ";
    assert_eq!(run_both(source), numbers(&[0.0, 1.0, 2.0, 3.0, 3.0]));
}

#[test]
fn select_with_negative_and_positive_indices() {
    let source = r#"
        local function last(...) return (select(-1, ...)) end
        local function from(i, ...) return select(i, ...) end
        local ok, message = pcall(select, -3, 1, 2)
        return last(1, 2, 3), last("a"), last(1, nil), from(2, "a", "b", "c"),
            from(-2, "a", "b", "c"), ok, message
    "#;
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(3.0));
    assert_eq!(text(&results[1]), "a");
    assert_eq!(results[2], Value::Nil);
    assert_eq!(text(&results[3]), "b");
    assert_eq!(text(&results[4]), "b");
    assert_eq!(results[5], Value::Boolean(false));
    assert!(text(&results[6]).contains("out of range"), "{results:?}");
}

#[test]
fn pack_and_unpack_round_trip_nils_and_long_lists() {
    let source = r"
        local function round(...)
            local t = table.pack(...)
            return table.unpack(t, 1, t.n)
        end
        local big = {}
        for i = 1, 300 do big[i] = i end
        local function count(...) return select('#', ...), (select(300, ...)) end
        local a, b, c, d = round(nil, nil, 3, nil)
        local n, last = count(table.unpack(big))
        local m = select('#', round(table.unpack(big)))
        return select('#', round(nil, nil, 3, nil)), a, b, c, d, n, last, m,
            select('#', round()), (round(1, 2, 3))
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(4.0),
            Value::Nil,
            Value::Nil,
            Value::Number(3.0),
            Value::Nil,
            Value::Number(300.0),
            Value::Number(300.0),
            Value::Number(300.0),
            Value::Number(0.0),
            Value::Number(1.0),
        ]
    );
}

#[test]
fn varargs_in_nested_closures_belong_to_the_closure_that_names_them() {
    let source = r"
        local function outer(...)
            local a, b = ...
            return function(...) return a, b, ... end
        end
        local inner = outer(1, 2, 3)
        local function counter(...)
            local total = 0
            local function add(x) total = total + (x or 0) end
            for i = 1, select('#', ...) do add((select(i, ...))) end
            return total, ...
        end
        return inner(4, 5), counter(10, 20, 30)
    ";
    // `inner(4, 5)` is truncated to its first value by the following call.
    assert_eq!(run_both(source), numbers(&[1.0, 60.0, 10.0, 20.0, 30.0]));
}

#[test]
fn varargs_forward_through_three_levels() {
    let source = r"
        local function f3(...) return select('#', ...), ... end
        local function f2(...) return f3(...) end
        local function f1(...) return f2(...) end
        local function g2(...) local x, y = f2(...) return x, y end
        local function h(...) return f2(1, ...) end
        local function m(...)
            local r = { f1(...) }
            return #r, r[1], r[2], r[5]
        end
        return f1(1, nil, 3, nil), g2('a', 'b'), h(2, 3), m(7, 8, 9)
    ";
    // Each call but the last is cut to one value.
    let results = run_both(source);
    assert_eq!(
        results,
        vec![
            Value::Number(4.0),
            Value::Number(2.0),
            Value::Number(3.0),
            Value::Number(4.0),
            Value::Number(3.0),
            Value::Number(7.0),
            Value::Nil,
        ]
    );
    let source = r"
        local function f3(...) return select('#', ...), ... end
        local function f2(...) return f3(...) end
        local function f1(...) return f2(...) end
        return f1(1, nil, 3, nil)
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(4.0),
            Value::Number(1.0),
            Value::Nil,
            Value::Number(3.0),
            Value::Nil,
        ]
    );
}

#[test]
fn fixed_parameters_and_varargs_mix_in_every_shape() {
    let source = r"
        local function v0(a, b, ...) return a, b, select('#', ...) end
        local function pass(...) return v0(...) end
        local function pass1(...) return v0(1, ...) end
        local function fixed(a, b) return b, a end
        local function toFixed(...) return fixed(...) end
        local function toFixed1(...) return fixed(1, ...) end
        local out = {}
        local function add(...) for i = 1, select('#', ...) do out[#out + 1] = tostring((select(i, ...))) end end
        add(v0()) add(v0(1)) add(v0(1, 2)) add(v0(1, 2, 3, 4))
        add(pass()) add(pass(5)) add(pass(5, 6, 7))
        add(pass1()) add(pass1(8)) add(pass1(8, 9, 10))
        add(toFixed()) add(toFixed(1)) add(toFixed(1, 2, 3))
        add(toFixed1()) add(toFixed1(2, 3))
        return table.concat(out, ',')
    ";
    let results = run_both(source);
    assert_eq!(
        text(&results[0]),
        "nil,nil,0,1,nil,0,1,2,0,1,2,2,\
         nil,nil,0,5,nil,0,5,6,1,\
         1,nil,0,1,8,0,1,8,2,\
         nil,nil,nil,1,2,1,\
         nil,1,2,1"
    );
}

#[test]
fn a_vararg_tail_call_chain_keeps_the_arguments_and_the_stack_bounded() {
    let source = r"
        local function down(n, ...)
            if n == 0 then return select('#', ...), ... end
            return down(n - 1, ...)
        end
        local function grow(n, ...)
            if n == 0 then return select('#', ...) end
            return grow(n - 1, n, ...)
        end
        local function drop(n, a, ...)
            if n == 0 then return a, select('#', ...) end
            return drop(n - 1, ...)
        end
        return down(1000000, 'x', 'y'), grow(150, 0), drop(3, 1, 2, 3, 4, 5)
    ";
    // `down` is cut to one value; `grow` to one; `drop` ends the list.
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(2.0),
            Value::Number(151.0),
            Value::Number(4.0),
            Value::Number(1.0),
        ]
    );
}

#[test]
fn tail_calls_of_a_million_hops_use_no_more_stack_than_one() {
    let source = r"
        local function loop(n, acc)
            if n == 0 then return acc end
            return loop(n - 1, acc + 1)
        end
        local even, odd
        function even(n) if n == 0 then return true end return odd(n - 1) end
        function odd(n) if n == 0 then return false end return even(n - 1) end
        local function open(n, ...)
            if n == 0 then return select('#', ...) end
            return open(n - 1, (function(...) return ... end)(...))
        end
        return loop(1000000, 0), even(1000001), open(100000, 1, 2)
    ";
    assert_eq!(
        run(source),
        vec![
            Value::Number(1_000_000.0),
            Value::Boolean(false),
            Value::Number(2.0)
        ]
    );
}

#[test]
fn tail_called_frames_keep_their_names_and_levels_apart_from_the_caller() {
    let source = r"
        local function g() return (debug.traceback('tb')) end
        local function f(...) return g(...) end
        local function viaOpen(...) return f(g(...)) end
        return f(1, 2), viaOpen()
    ";
    for profile in PROFILES {
        let results = run_profile(profile, source);
        let trace = text(&results[0]);
        match profile {
            CompatibilityProfile::Lua51 => {
                assert!(trace.contains("(tail call): ?"), "{profile:?}: {trace}");
            }
            CompatibilityProfile::LuaJit21 => {
                assert!(
                    !trace.contains("(...tail calls...)"),
                    "{profile:?}: {trace}"
                );
                assert!(!trace.contains("(tail call)"), "{profile:?}: {trace}");
            }
            _ => assert!(trace.contains("(...tail calls...)"), "{profile:?}: {trace}"),
        }
        assert!(
            trace.starts_with("tb\nstack traceback:"),
            "{profile:?}: {trace}"
        );
        // The same program under a hook takes the dispatch loop's tail calls.
        let hooked = run_profile(
            profile,
            &format!("debug.sethook(function() end, 'l') {source}"),
        );
        assert_eq!(text(&hooked[0]), trace, "{profile:?}");
    }
}

#[test]
fn call_and_tail_call_hooks_fire_for_vararg_functions() {
    let source = r"
        local events = {}
        local function hook(event) events[#events + 1] = event end
        local function g(...) return select('#', ...) end
        local function f(...) return g(...) end
        debug.sethook(hook, 'cr')
        local r = f(1, 2, 3)
        debug.sethook()
        return r, table.concat(events, ',')
    ";
    for profile in PROFILES {
        let results = run_profile(profile, source);
        assert_eq!(results[0], Value::Number(3.0), "{profile:?}");
        let events = text(&results[1]);
        let expected = match profile {
            // Lua 5.1 reports a tail call as a call, and its return as one.
            CompatibilityProfile::Lua51 => "return,call,call,call,return,tail return,call",
            CompatibilityProfile::LuaJit21 => "return,call,call,call,return,call",
            _ => "return,call,tail call,call,return,call",
        };
        assert_eq!(events, expected, "{profile:?}");
    }
}

#[test]
fn vararg_functions_run_in_coroutines_and_yield_with_their_extras() {
    let source = r"
        local co = coroutine.wrap(function(...)
            local a = ...
            local b = coroutine.yield(...)
            local function inner(...) return ... end
            return a, b, inner(...)
        end)
        local t1 = { co(1, 2, 3) }
        local t2 = { co('r') }
        local gen = coroutine.wrap(function(...)
            local args = table.pack(...)
            for i = 1, args.n do coroutine.yield(args[i]) end
            return 'done'
        end)
        local seen = {}
        seen[#seen + 1] = tostring(gen('a', nil, 'c'))
        seen[#seen + 1] = tostring(gen())
        seen[#seen + 1] = tostring(gen())
        seen[#seen + 1] = tostring(gen())
        return #t1, t1[3], #t2, t2[1], t2[2], t2[3], table.concat(seen, ',')
    ";
    let results = run_both(source);
    // The first resume yields its three arguments; the second returns
    // `a, b` and then the coroutine function's own varargs again.
    assert_eq!(results[0], Value::Number(3.0));
    assert_eq!(results[1], Value::Number(3.0));
    assert_eq!(results[2], Value::Number(5.0));
    assert_eq!(results[3], Value::Number(1.0));
    assert_eq!(text(&results[4]), "r");
    assert_eq!(results[5], Value::Number(1.0));
    assert_eq!(text(&results[6]), "a,nil,c,done");
}

#[test]
fn vararg_functions_behave_the_same_under_every_kind_of_hook() {
    let body = r"
        local function sum(...)
            local a, b, c = ...
            return (a or 0) + (b or 0) + (c or 0), select('#', ...)
        end
        local function fwd(...) return sum(...) end
        local total = 0
        local counted = 0
        for i = 1, 200 do
            local s, n = fwd(i, 1, 1)
            total = total + s + n
            local s2, n2 = fwd(i)
            total = total + s2 + n2
        end
        return total
    ";
    let plain = run(body);
    for hook in [
        "debug.sethook(function() end, 'l')",
        "debug.sethook(function() end, 'c')",
        "debug.sethook(function() end, 'r')",
        "debug.sethook(function() end, '', 7)",
        "debug.sethook(function() end, 'crl', 3)",
    ] {
        assert_eq!(run(&format!("{hook} {body}")), plain, "{hook}");
    }
    // Per `i`: `fwd(i, 1, 1)` gives `i + 2` and 3, `fwd(i)` gives `i` and 1.
    assert_eq!(plain, vec![Value::Number(41_400.0)]);
}

#[test]
fn the_debug_library_reads_and_writes_varargs_of_inner_loop_frames() {
    let source = r"
        local function loc(...)
            local n1, v1 = debug.getlocal(1, -1)
            local n2, v2 = debug.getlocal(1, -2)
            local n3 = debug.getlocal(1, -3)
            local set = debug.setlocal(1, -1, 'changed')
            return n1, v1, n2, v2, n3, set, (...), select('#', ...)
        end
        local function info(...)
            local i = debug.getinfo(1, 'u')
            return i.isvararg, i.nparams
        end
        return loc('x', 'y'), info(1)
    ";
    // `loc(...)` is cut to its first value by the call that follows.
    let results = run_both(source);
    assert_eq!(text(&results[0]), "(*vararg)");
    assert_eq!(results[1], Value::Boolean(true));
    assert_eq!(results[2], Value::Number(0.0));
}

#[test]
fn errors_unwind_vararg_frames_and_name_the_right_call() {
    let source = r"
        local function boom(...) local x = ... return x.y.z end
        local function callnil(...) return nofn(...) end
        local function callnil2(...) local r = nofn(...) return r end
        local function deep(n, ...) if n == 0 then error('bottom ' .. select('#', ...)) end return 1 + deep(n - 1, n, ...) end
        local ok1, m1 = pcall(boom, {})
        local ok2, m2 = pcall(callnil, 1)
        local ok3, m3 = pcall(callnil2, 1)
        local ok4, m4 = pcall(deep, 20, 'a')
        local ok5, m5 = xpcall(deep, function(m) return 'H:' .. m end, 3)
        return ok1, m1, ok2, m2, ok3, m3, ok4, m4, ok5, m5
    ";
    let results = run_both(source);
    assert_eq!(results[0], Value::Boolean(false));
    assert!(text(&results[1]).contains("attempt to index a nil value (field 'y')"));
    assert!(text(&results[3]).contains("attempt to call a nil value (global 'nofn')"));
    assert!(text(&results[5]).contains("attempt to call a nil value (global 'nofn')"));
    assert!(text(&results[7]).ends_with("bottom 21"), "{results:?}");
    assert!(text(&results[9]).starts_with("H:"), "{results:?}");
    assert!(text(&results[9]).ends_with("bottom 3"), "{results:?}");
}

#[test]
fn unbounded_vararg_recursion_reports_a_stack_overflow() {
    let source = r"
        local function r(n, ...) return 1 + r(n + 1, n, ...) end
        local ok, message = pcall(r, 1)
        return ok, message
    ";
    let results = run(source);
    assert_eq!(results[0], Value::Boolean(false));
    assert!(text(&results[1]).contains("stack overflow"), "{results:?}");
}

#[test]
fn varargs_survive_a_collection_and_keep_their_objects_alive() {
    let source = r"
        local function keep(...)
            local before = { ... }
            collectgarbage()
            collectgarbage()
            local garbage = {}
            for i = 1, 200 do garbage[i] = { i } end
            collectgarbage()
            local ok = true
            for i = 1, select('#', ...) do
                local v = (select(i, ...))
                ok = ok and v == before[i] and v.id == i
            end
            return ok, ...
        end
        local function fwd(...) return keep(...) end
        local a, b, c = { id = 1 }, { id = 2 }, { id = 3 }
        local ok, x, y, z = fwd(a, b, c)
        local s1, s2 = fwd('s' .. 1, 's' .. 2)
        return ok, x == a, y == b, z == c, type(s1), s2
    ";
    let results = run_both(source);
    assert_eq!(
        &results[..4],
        &[
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true),
            Value::Boolean(true)
        ]
    );
}

#[test]
fn string_varargs_pass_through_calls_and_tail_calls_unchanged() {
    let source = r"
        local function id(...) return ... end
        local function twice(...) return id(...) end
        local function mix(a, ...) return a .. select('#', ...), ... end
        local parts = {}
        for i = 1, 300 do
            local s = 'item' .. i
            local x, y, z = twice(s, s .. '!', mix(s, s, s))
            parts[#parts + 1] = x .. y .. tostring(z)
        end
        local last = parts[300]
        return #parts, last, parts[1]
    ";
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(300.0));
    assert_eq!(text(&results[1]), "item300item300!item3002");
    assert_eq!(text(&results[2]), "item1item1!item12");
}

#[test]
fn captured_locals_and_closures_in_vararg_frames_stay_synchronised() {
    let source = r"
        local function make(...)
            local count = select('#', ...)
            local first = ...
            local function bump() count = count + 1 return count, first end
            return bump, ...
        end
        local bump, a, b = make('p', 'q')
        local c1, f1 = bump()
        local c2 = bump()
        return c1, f1, c2, a, b
    ";
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(3.0));
    assert_eq!(text(&results[1]), "p");
    assert_eq!(results[2], Value::Number(4.0));
}

#[test]
fn methods_and_metamethods_take_varargs() {
    let source = r"
        local obj = setmetatable({}, {
            __index = function(t, k) return function(self, ...) return k, select('#', ...), ... end end,
            __call = function(self, ...) return 'called', ... end,
            __add = function(a, b) return select('#', a, b) end,
        })
        local function m(...) return obj:method(...) end
        local function c(...) return obj(...) end
        return m(1, 2, 3), c('x', 'y'), obj + 1
    ";
    let results = run_both(source);
    assert_eq!(text(&results[0]), "method");
    assert_eq!(text(&results[1]), "called");
    assert_eq!(results[2], Value::Number(2.0));
}

#[test]
fn named_varargs_keep_their_table_form_in_lua_55() {
    let source = r"
        local function f(a, ...t) return t.n, t[1], select('#', ...), a end
        local function h(...) local r = { f(...) } return #r, r[1] end
        return f(1, 2, 3), h(4, 5, 6)
    ";
    // `f(1, 2, 3)` is cut to its first value.
    assert_eq!(
        run_profile(CompatibilityProfile::Lua55, source),
        numbers(&[2.0, 4.0, 2.0])
    );
}

#[test]
fn every_profile_agrees_on_varargs_and_tail_calls() {
    let source = r"
        local function f3(...) return select('#', ...), ... end
        local function f2(...) return f3(...) end
        local function f1(a, ...) return f2(a, ...) end
        local function tail(n, ...) if n == 0 then return ... end return tail(n - 1, n, ...) end
        local r = { f1(1, nil, 3) }
        local t = { tail(5) }
        return #r, r[1], #t, t[1], t[5], select('#', f1())
    ";
    let reference = run_profile(CompatibilityProfile::Lua54, source);
    for profile in PROFILES {
        assert_eq!(run_profile(profile, source), reference, "{profile:?}");
    }
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(
            run_with(CompatibilityProfile::Lua54, mode, source),
            reference
        );
    }
}

#[test]
fn tracing_mode_runs_vararg_loops_and_tail_calls_alike() {
    let source = r"
        local function sum(...)
            local a, b, c = ...
            return a + b + c
        end
        local function fwd(...) return sum(...) end
        local function spin(n, acc) if n == 0 then return acc end return spin(n - 1, acc + fwd(n, 1, 1)) end
        local total = 0
        for i = 1, 3000 do total = total + fwd(i, 2, 3) end
        return total, spin(2000, 0)
    ";
    let expected = run(source);
    assert_eq!(
        run_with(CompatibilityProfile::Lua54, ExecutionMode::Tracing, source),
        expected
    );
}

#[test]
fn vararg_functions_called_from_native_code_take_their_arguments() {
    let source = r##"
        local function cmp(...) local a, b = ... return a > b, select('#', ...) end
        local t = { 3, 1, 2, 5, 4 }
        table.sort(t, cmp)
        local replaced = string.gsub('abc', '(%w)', function(...) return select('#', ...) .. (...) end)
        local ok, a, b, c = pcall(function(...) return ... end, 'x', 'y', 'z')
        local xp = { xpcall(function(...) error(select('#', ...)) end, function(...) return 'H' .. select('#', ...) end, 1, 2) }
        local loaded = load('local a, b = ... return b, a, select("#", ...)')
        local l1, l2, l3 = loaded('p', 'q', 'r')
        local mapped = {}
        for k, v in ipairs({ 'u', 'v' }) do mapped[k] = (function(...) return select('#', ...) .. v end)(k, v, nil) end
        return table.concat(t, ','), replaced, ok, a, b, c, xp[2], l1, l2, l3, table.concat(mapped, ',')
    "##;
    let results = run_both(source);
    assert_eq!(text(&results[0]), "5,4,3,2,1");
    assert_eq!(text(&results[1]), "1a1b1c");
    assert_eq!(results[2], Value::Boolean(true));
    assert_eq!(
        [text(&results[3]), text(&results[4]), text(&results[5])],
        ["x", "y", "z"]
    );
    assert_eq!(text(&results[6]), "H1");
    assert_eq!(text(&results[7]), "q");
    assert_eq!(text(&results[8]), "p");
    assert_eq!(results[9], Value::Number(3.0));
    assert_eq!(text(&results[10]), "3u,3v");
}

#[test]
fn a_hook_set_inside_a_vararg_function_sees_the_forwarded_tail_call() {
    let source = r"
        local events = {}
        local function target(...) return select('#', ...) end
        local function install(...)
            debug.sethook(function(event) events[#events + 1] = event end, 'cr')
            return target(...)
        end
        local n = install(1, 2, 3)
        debug.sethook()
        return n, table.concat(events, ',')
    ";
    let results = run(source);
    assert_eq!(results[0], Value::Number(3.0));
    let events = text(&results[1]);
    assert!(events.contains("tail call"), "{events}");
    let hooked = run(&format!("debug.sethook(function() end, 'l') {source}"));
    assert_eq!(text(&hooked[1]), events);
}

#[test]
fn deep_vararg_recursion_and_forwarding_keep_every_argument() {
    let source = r"
        local function build(n, ...)
            if n == 0 then return ... end
            return build(n - 1, n, ...)
        end
        local function sum(...)
            local total = 0
            for i = 1, select('#', ...) do total = total + (select(i, ...)) end
            return total
        end
        local function forward(...) return sum(...) end
        local function viaTable(...) return sum(table.unpack({ ... }, 1, select('#', ...))) end
        local big = { build(1000) }
        return #big, forward(table.unpack(big)), viaTable(table.unpack(big, 1, 250)), big[1], big[1000]
    ";
    assert_eq!(
        run_both(source),
        numbers(&[1000.0, 500_500.0, 31_375.0, 1.0, 1000.0])
    );
}

#[test]
fn a_frame_that_tail_calls_a_fixed_function_drops_its_varargs() {
    // After the tail call the frame runs `inner`, which has no varargs: the
    // debug library sees none, whichever loop made the call.
    let source = r"
        local function inner(a, b)
            local name, value = debug.getlocal(1, -1)
            return name, value, a, b
        end
        local function outer(...) return inner(...) end
        local function outer2(x, ...) return inner(x, ...) end
        local n1, v1, a1, b1 = outer(10, 20, 30)
        local n2, v2, a2, b2 = outer2(40, 50, 60)
        return n1, v1, a1, b1, n2, v2, a2, b2
    ";
    let expected = vec![
        Value::Nil,
        Value::Nil,
        Value::Number(10.0),
        Value::Number(20.0),
        Value::Nil,
        Value::Nil,
        Value::Number(40.0),
        Value::Number(50.0),
    ];
    assert_eq!(run_both(source), expected);
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(
            run_with(CompatibilityProfile::Lua54, mode, source),
            expected,
            "{mode:?}"
        );
    }
}

#[test]
fn fewer_arguments_than_parameters_of_a_vararg_function_are_nil_everywhere() {
    // Two or more arguments to a function with more parameters than that,
    // and varargs behind them, on every loop that makes the call.
    let source = r"
        local function g(a, b, c, d, ...) return a, b, c, d, select('#', ...) end
        local function h(a, b, c, ...) return a, b, c, select('#', ...) end
        local r = table.pack(g(1, 2), g(1, 2, 3), g(1, 2, 3, 4, 5, 6), h(1, 2))
        local out = {}
        for _, f in ipairs({ g, h }) do
            for n = 0, 6 do
                local t = table.pack(f(table.unpack({ 1, 2, 3, 4, 5, 6 }, 1, n)))
                out[#out + 1] = t.n .. ':' .. tostring(t[1]) .. tostring(t[#t])
            end
        end
        return r.n, r[1], table.concat(out, ',')
    ";
    let results = run_both(source);
    assert_eq!(results[0], Value::Number(7.0));
    assert_eq!(results[1], Value::Number(1.0));
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(
            run_with(CompatibilityProfile::Lua54, mode, source),
            results,
            "{mode:?}"
        );
    }
    let source = "
        local function g(a, b, c, ...) return select('#', ...), c end
        local total = 0
        for i = 1, 300 do
            local n, c = g(i, 2)
            total = total + n + (c or 0)
        end
        return total
    ";
    assert_eq!(run_both(source), numbers(&[0.0]));
}
