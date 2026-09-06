//! A `Call` of a vararg function with more arguments than parameters leaves
//! the extra ones where the caller put them: the caller's window is cut short
//! just before them, the callee's window starts right after them, and the
//! caller gets its full window back when the callee is gone, however that
//! happens (return, tail call, error, coroutine switch, unwinding).
//!
//! Every program runs on the interpreter's inner loop and again under a line
//! hook, which keeps the inner loop out; the two must agree. Programs that
//! call a fixed-parameter function after the vararg call are the ones a
//! window left short would corrupt: the new activation would start inside the
//! caller's registers.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run_with(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source)
        .unwrap_or_else(|error| panic!("{error}"))
        .into_iter()
        .map(|value| match value {
            #[allow(clippy::cast_precision_loss)]
            Value::Integer(integer) => Value::Number(integer as f64),
            other => other,
        })
        .collect()
}

/// Runs `source` on the inner loop and again under a line hook.
fn run_both(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let reference = format!("debug.sethook(function() end, 'l') {source}");
    let fast = run_with(profile, source);
    let general = run_with(profile, &reference);
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

/// Results are written over the registers the callee's varargs occupied, so
/// every count of results against every count of arguments is a case: more
/// results than arguments, fewer, none wanted, all of them forwarded.
#[test]
fn results_land_where_the_varargs_were() {
    let source = r"
        local function pack(...) return {n = select('#', ...), ...} end
        local function show(t)
            local s = {}
            for i = 1, t.n do s[#s + 1] = tostring(t[i]) end
            return table.concat(s, ',') .. '#' .. t.n
        end
        local function same(...) return ... end
        local function seven(...) return 1, 2, 3, 4, 5, 6, 7 end
        local function two(a, ...) return a, select('#', ...) end
        local function none(...) end
        local out = {}
        local a, b, c, d, e, f, g = seven(1, 2, 3)
        out[#out + 1] = show(pack(a, b, c, d, e, f, g))
        a, b, c, d, e, f, g = same(1, 2, 3)
        out[#out + 1] = show(pack(a, b, c, d, e, f, g))
        a, b, c, d, e, f, g = same(1, 2, 3, 4, 5, 6, 7, 8, 9)
        out[#out + 1] = show(pack(a, b, c, d, e, f, g))
        a, b, c, d, e, f, g = two(1, 2, 3)
        out[#out + 1] = show(pack(a, b, c, d, e, f, g))
        a, b = none(1, 2, 3)
        out[#out + 1] = show(pack(a, b))
        local h, i = same(nil, nil, nil)
        out[#out + 1] = show(pack(h, i))
        out[#out + 1] = show(pack(same(1, 2, 3), same(4, 5)))
        out[#out + 1] = show(pack((same(1, 2, 3))))
        out[#out + 1] = show(pack(same()))
        out[#out + 1] = show({same(same(1, 2), same(3, 4, 5)), n = 5})
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "1,2,3,4,5,6,7#7 1,2,3,nil,nil,nil,nil#7 1,2,3,4,5,6,7#7 1,2,nil,nil,nil,nil,nil#7 \
             nil,nil#2 nil,nil#2 1,4,5#3 1#1 #0 1,nil,nil,nil,nil#5",
            "{profile:?}"
        );
    }
}

/// Registers the caller holds above the call are dead, and the ones below it
/// are untouched. A new activation after the call starts at the caller's full
/// top; a window left short would put it on the caller's temporaries.
#[test]
fn the_caller_keeps_its_registers_around_a_vararg_call() {
    let source = r"
        local function vf(...) return ... end
        local function g(x, y) return x * 10 + y end
        local function sum(...)
            local s = 0
            for i = 1, select('#', ...) do s = s + (select(i, ...)) end
            return s
        end
        local out = {}
        local keep = 100
        -- a call between operations that hold values in temporaries
        out[#out + 1] = (1 + 2) * sum(1, 2, 3, 4) + g(5, 6) * (7 + vf(8, 9, 10)) + keep
        out[#out + 1] = g(sum(1, 2, 3), g(vf(4, 5), vf(6, 7, 8)))
        out[#out + 1] = (vf(1, 2, 3)) + (vf(4, 5, 6)) + g(vf(7, 8, 9), 1)
        local t = {sum(1, 2), sum(3, 4, 5), g(6, 7), vf(8, 9)}
        out[#out + 1] = #t * 1000 + t[1] * 100 + t[2] * 10 + t[3] + t[4] + t[5]
        for i = 1, 3 do
            local k = i * 2
            local a, b = vf(i, k, i + k)
            local c = g(a, b)
            out[#out + 1] = a * 100 + b * 10 + c + k
        end
        local s = 0
        for k, v in pairs({10, 20, 30}) do
            s = s + vf(k, v, 0, 0) + g(k, 1)
        end
        out[#out + 1] = s
        out[#out + 1] = select('#', vf(1, 2, 3), vf(4, 5, 6))
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        // Worked out by hand:
        //   (1 + 2) * 10 + 56 * (7 + 8) + 100 = 970
        //   g(6, g(4, 6)) = g(6, 46) = 106
        //   1 + 4 + g(7, 1) = 76
        //   t = {3, 12, 67, 8, 9}: 5000 + 300 + 120 + 67 + 8 + 9 = 5504
        //   i = 1..3: a, b = i, 2i, c = g(a, b) = 12 i, k = 2i: 134 i
        //   pairs: a call in the middle of an expression gives one value (the key):
        //   (1 + g(1, 1)) + (2 + g(2, 1)) + (3 + g(3, 1)) = 12 + 23 + 34 = 69
        //   select('#', vf(1, 2, 3), vf(4, 5, 6)) = 1 + 3 = 4
        assert_eq!(
            text(&results[0]),
            "970 106 76 5504 134 268 402 69 4",
            "{profile:?}"
        );
    }
}

/// Strings and tables passed only as extra arguments stay alive through a
/// collection in the callee, at every level of nested in-place calls.
#[test]
fn a_collection_inside_nested_calls_keeps_every_vararg() {
    let source = r"
        local function deepest(...)
            collectgarbage()
            collectgarbage()
            local a, b, c = ...
            return a[1], b, c.n, select('#', ...)
        end
        local function middle(tag, ...)
            local junk = {}
            for i = 1, 50 do junk[i] = {i, tostring(i)} end
            local x, y, z, n = deepest(...)
            collectgarbage()
            return tag, x, y, z, n, #junk, select(2, ...)
        end
        local function outer(...)
            local made = {}
            for i = 1, 20 do made[i] = string.rep('s', i) end
            return middle('t', ...)
        end
        local r = {outer({1}, string.rep('q', 40) .. tostring(7), {n = 3}, {4})}
        collectgarbage()
        return r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8].n, r[9][1]
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "t");
        assert_eq!(results[1], Value::Number(1.0));
        assert_eq!(text(&results[2]), format!("{}7", "q".repeat(40)));
        assert_eq!(results[3], Value::Number(3.0));
        assert_eq!(results[4], Value::Number(4.0));
        assert_eq!(results[5], Value::Number(50.0));
        assert_eq!(text(&results[6]), format!("{}7", "q".repeat(40)));
        assert_eq!(results[7], Value::Number(3.0));
        assert_eq!(results[8], Value::Number(4.0));
    }
}

/// Errors thrown through in-place calls, caught by protected calls at the
/// caller's level or above, leave the catching function with its whole
/// window: it goes on to call fixed-parameter functions with temporaries live.
#[test]
fn errors_through_vararg_calls_leave_the_catcher_a_whole_window() {
    let source = r"
        local function boom(...) error('boom ' .. select('#', ...), 0) end
        local function g(x, y) return x * 10 + y end
        local function catcher(...)
            local ok, message = pcall(boom, ...)
            local a = g(1, 2)
            local ok2, m2 = pcall(function(...) return boom(...) end, ...)
            local b = g(3, 4) + g(5, 6)
            return ok, message, ok2, m2, a, b
        end
        local out = {}
        for i = 1, 5 do
            local ok, message, ok2, m2, a, b = catcher(i, i, i)
            out[#out + 1] = tostring(ok) .. message .. tostring(ok2) .. m2 .. a .. ':' .. b
        end
        local function layered(...)
            local function inner(...) local x = g(7, 8) boom(...) return x end
            local ok, message = pcall(inner, ...)
            return message, g(9, 1)
        end
        local m, r = layered('a', 'b')
        out[#out + 1] = m .. r
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        let one = "falseboom 3falseboom 312:90";
        assert_eq!(
            text(&results[0]),
            format!("{one} {one} {one} {one} {one} boom 291"),
            "{profile:?}"
        );
    }
}

/// A to-be-closed variable of a caller whose vararg callee fails is closed
/// while the callee's frame is being unwound, and the handler runs on the
/// stack that is left.
#[test]
fn closing_a_caller_unwinds_past_a_vararg_callee() {
    let source = r"
        local log = {}
        local function closer(name)
            return setmetatable({}, {__close = function()
                local x = (function(a, b) return a + b end)(1, 2)
                log[#log + 1] = name .. x
            end})
        end
        local function boom(...) error('fail', 0) end
        local function holder(...)
            local c <close> = closer('c')
            local d <close> = closer('d')
            boom(...)
        end
        local ok, message = pcall(holder, 1, 2, 3)
        local function after(...) return select('#', ...) end
        return ok, message, table.concat(log, ','), after(1, 2), (function(a, b) return a * b end)(6, 7)
    ";
    let results = run_both(CompatibilityProfile::Lua54, source);
    assert_eq!(results[0], Value::Boolean(false));
    assert_eq!(text(&results[1]), "fail");
    assert_eq!(text(&results[2]), "d3,c3");
    assert_eq!(results[3], Value::Number(2.0));
    assert_eq!(results[4], Value::Number(42.0));
}

/// A coroutine that yields inside a vararg callee keeps the callee's extra
/// arguments and the cut-short caller until it is resumed, then returns into
/// a caller whose registers are all there.
#[test]
fn coroutines_yield_inside_vararg_calls() {
    let source = r"
        local function g(x, y) return x * 10 + y end
        local function yielder(...)
            local first = coroutine.yield(select('#', ...), ...)
            local second = coroutine.yield(first, (select(2, ...)))
            return first + second, ...
        end
        local co = coroutine.wrap(function(a, b)
            local keep = g(a, b)
            local r1, r2, r3 = yielder(keep, 20, 30)
            local after = g(r1, r2)
            return keep, r1, r2, r3, after
        end)
        local out = {}
        local function add(...) for i = 1, select('#', ...) do out[#out + 1] = tostring((select(i, ...))) end end
        add(co(1, 2))
        add(co(5))
        add(co(7))
        return table.concat(out, ',')
    ";
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let results = run_both(profile, source);
        // yielder(12, 20, 30): yields 3, 12, 20, 30; then 5 -> yields 5, 20;
        // then 7 -> returns 12, 12, 20, 30: r1, r2, r3 = 12, 12, 20 and after = g(12, 12)
        assert_eq!(
            text(&results[0]),
            "3,12,20,30,5,20,12,12,12,20,132",
            "{profile:?}"
        );
    }
}

/// A vararg callee that tail calls leaves the caller's window short until the
/// last function of the chain returns.
#[test]
fn tail_calls_out_of_a_vararg_call_return_to_a_whole_caller() {
    let source = r"
        local function g(x, y) return x * 10 + y end
        local function fixed(a, b, c) return a + b + c end
        local function relay(...) return fixed(...) end
        local function relay2(n, ...) if n == 0 then return fixed(...) end return relay2(n - 1, ...) end
        local function viaflex(...) return relay(...) end
        local function back(...) return g(...) end
        local out = {}
        for i = 1, 4 do
            local a = relay(i, 2, 3, 4, 5)
            local b = g(a, 1)
            local c = relay2(3, i, i, i, i)
            local d = g(c, viaflex(1, 1, 1, 9, 9))
            local e = back(1, 2, 3, 4) + g(2, 2)
            out[#out + 1] = a .. ':' .. b .. ':' .. c .. ':' .. d .. ':' .. e
        end
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "6:61:3:33:34 7:71:6:63:34 8:81:9:93:34 9:91:12:123:34",
            "{profile:?}"
        );
    }
}

/// The debug library sees the same locals in a caller whether its callee is
/// a vararg function or not.
#[test]
fn the_debug_library_sees_the_caller_of_a_vararg_call() {
    let source = r"
        local function locals_of_caller(...)
            local out = {}
            for i = 1, 6 do
                local name, value = debug.getlocal(2, i)
                if not name then break end
                if name:sub(1, 1) ~= '(' then out[#out + 1] = name .. '=' .. tostring(value) end
            end
            return table.concat(out, ';')
        end
        local function plain(a, b) return debug.getinfo(1, 'l').currentline > 0 end
        local function caller(x)
            local y = x * 2
            local seen = locals_of_caller(1, 2, 3)
            local z = x + y
            local again = locals_of_caller()
            return seen, again, z, plain(1, 2)
        end
        return caller(5)
    ";
    let results = run_both(CompatibilityProfile::Lua54, source);
    assert_eq!(text(&results[0]), "x=5;y=10");
    assert_eq!(text(&results[1]), "x=5;y=10;seen=x=5;y=10;z=15");
    assert_eq!(results[2], Value::Number(15.0));
    assert_eq!(results[3], Value::Boolean(true));
}

/// Hooks installed or removed inside a vararg callee do not disturb the
/// window of the caller they return to.
#[test]
fn hooks_set_inside_a_vararg_callee_leave_the_caller_whole() {
    let source = r"
        local count = 0
        local function g(x, y) return x * 10 + y end
        local function inner(...)
            debug.sethook(function() count = count + 1 end, 'cr')
            local a, b = ...
            debug.sethook()
            return a, b, select('#', ...)
        end
        local out = {}
        for i = 1, 3 do
            local x, y, n = inner(i, i + 1, 9, 9)
            out[#out + 1] = g(x, y) + n
        end
        return table.concat(out, ' '), count > 0
    ";
    let results = run_both(CompatibilityProfile::Lua54, source);
    assert_eq!(text(&results[0]), "16 27 38");
    assert_eq!(results[1], Value::Boolean(true));
}

/// A deep chain of non-tail vararg calls, each holding live values in its own
/// registers across the call, returns every one of them intact.
#[test]
fn deep_chains_of_vararg_calls_keep_every_frames_registers() {
    let source = r"
        local function deep(n, ...)
            if n == 0 then return select('#', ...) end
            local a, b, c = n * 2, n * 3, n * 5
            local r = deep(n - 1, a, ...)
            local check = (function(x, y, z) return x + y + z end)(a, b, c)
            return r + check - (a + b + c)
        end
        local function fan(n, ...)
            if n == 0 then return ... end
            local x, y = fan(n - 1, n, ...)
            return x, y
        end
        local t = {fan(30)}
        return deep(150), deep(1), deep(0), #t
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(results, numbers(&[150.0, 1.0, 0.0, 2.0]), "{profile:?}");
    }
}

/// The compiler loads constants and moves values into their final argument
/// registers before it evaluates the arguments around them, so a register
/// above a nested vararg call can hold a value the outer call needs. The
/// callee's window must not cover it.
#[test]
fn values_held_above_a_nested_vararg_call_survive_it() {
    let source = r"
        local function count(...) return select('#', ...), ... end
        local function chk(...) return select('#', ...), ... end
        local function id(...) return ... end
        local function pack(...) return {n = select('#', ...), ...} end
        local function show(t)
            local s = {}
            for i = 1, t.n do s[#s + 1] = tostring(t[i]) end
            return table.concat(s, ',') .. '#' .. t.n
        end
        count()
        local out = {}
        out[#out + 1] = show(pack(chk('#', count('s', 7), 1)))
        out[#out + 1] = show(pack(chk(1, 2, id('a', 'b', 'c'), 3, 4)))
        out[#out + 1] = show(pack(chk('x', 'y', count(1, 2), 'z', id(1, 2, 3, 4))))
        out[#out + 1] = show(pack(chk(count('s', 7, 8), 'k', count('t'), 'm', 5.5)))
        out[#out + 1] = show(pack(id(id('p', 'q'), id('r', 's', 't'), 'u')))
        local name = 'tail'
        out[#out + 1] = show(pack(chk(name, count(name, name), name .. '!', id(name))))
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "3,#,2,1#4 5,1,2,a,3,4#6 8,x,y,2,z,1,2,3,4#9 5,3,k,1,m,5.5#6 p,r,u#3 \
             4,tail,2,tail!,tail#5",
            "{profile:?}"
        );
    }
}
