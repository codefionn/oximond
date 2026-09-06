//! A call of a vararg function whose first statement is `local a, b = ...`
//! runs that `Vararg` itself, from the extra arguments that lie where the
//! caller put them, and starts the callee at the next instruction. The
//! results must be what the instruction alone gives: the first arguments in
//! the locals, nil past the last, strings counted, whatever the registers
//! held before.
//!
//! Every program runs on the interpreter's inner loop and again under a line
//! hook, which keeps the inner loop out; the two must agree.

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

fn run_both(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let reference = format!("debug.sethook(function() end, 'l') {source}");
    let fast = run_with(profile, source);
    let general = run_with(profile, &reference);
    assert_eq!(fast, general, "inner loop and dispatch loop disagree");
    fast
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

/// Fewer, as many and more arguments than locals; trailing nils; strings.
#[test]
fn the_locals_get_the_first_arguments_and_nil_past_the_last() {
    let source = r"
        local function three(...)
            local a, b, c = ...
            return tostring(a) .. ',' .. tostring(b) .. ',' .. tostring(c)
        end
        local function counted(...)
            local a, b = ...
            return tostring(a) .. ',' .. tostring(b) .. '#' .. select('#', ...)
        end
        local out = {}
        out[#out + 1] = three(1)
        out[#out + 1] = three(1, 2)
        out[#out + 1] = three(1, 2, 3)
        out[#out + 1] = three(1, 2, 3, 4, 5, 6)
        out[#out + 1] = three(nil, nil, nil, nil)
        out[#out + 1] = three('x', 'y', 'z')
        out[#out + 1] = counted(1, nil, nil)
        out[#out + 1] = counted(nil, nil, nil, nil, nil)
        out[#out + 1] = counted(false, true)
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "1,nil,nil 1,2,nil 1,2,3 1,2,3 nil,nil,nil x,y,z 1,nil#3 nil,nil#5 false,true#2",
            "{profile:?}"
        );
    }
}

/// The registers the callee's locals take held strings from earlier calls at
/// that depth: they are released, and the argument strings are counted.
#[test]
fn strings_in_the_registers_and_in_the_arguments() {
    let source = r"
        local function swap(...)
            local a, b = ...
            return b, a
        end
        local function keep(...)
            local a, b, c = ...
            local s = a .. b
            return s, c
        end
        local out = {}
        for i = 1, 3 do
            local x, y = swap('first' .. i, 'second' .. i)
            out[#out + 1] = x .. y
            local s, c = keep('ab', 'cd', i, 'tail')
            out[#out + 1] = s .. c
            local n, m = swap(i, i + 1)
            out[#out + 1] = n * 10 + m
            local u, v = swap('only' .. i)
            out[#out + 1] = tostring(u) .. tostring(v)
        end
        collectgarbage()
        collectgarbage()
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "second1first1 abcd1 21 nilonly1 second2first2 abcd2 32 nilonly2 \
             second3first3 abcd3 43 nilonly3",
            "{profile:?}"
        );
    }
}

/// A function with fixed parameters before the locals, and the varargs read
/// again later: the extra arguments stay in place for `select`, `...` and a
/// table constructor.
#[test]
fn fixed_parameters_and_later_uses_of_the_varargs() {
    let source = r"
        local function f(x, ...)
            local a, b = ...
            local t = {...}
            return x, a, b, select('#', ...), (select(-1, ...)), t[1], t[2]
        end
        local function g(...)
            local a = ...
            local t = {...}
            return a, t[1], t[2], select('#', ...), ...
        end
        local out = {}
        local function add(...)
            local parts = {}
            for i = 1, select('#', ...) do parts[i] = tostring((select(i, ...))) end
            out[#out + 1] = table.concat(parts, ',')
        end
        add(f(1, 2, 3, 4, 5))
        add(f(1, 2))
        add(f(1, 2, nil))
        add(g(7, 8, 9))
        add(g(7))
        add(g(nil, 5))
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(
            text(&results[0]),
            "1,2,3,4,5,2,3 1,2,nil,1,2,2,nil 1,2,nil,2,nil,2,nil \
             7,7,8,3,7,8,9 7,7,nil,1,7 nil,nil,5,2,nil,5",
            "{profile:?}"
        );
    }
}

/// An error just after the entry reports the line of the statement that
/// raised it, and a traceback from there names the vararg function.
#[test]
fn an_error_after_the_entry_has_the_right_line() {
    let source = r"
        local function boom(...)
            local a, b = ...
            local c = a + b
            error('late ' .. tostring(c), 0)
        end
        local function typeerr(...)
            local a, b = ...
            return a + b
        end
        local out = {}
        local ok, message = pcall(function() boom(1, 2, 3) end)
        out[#out + 1] = message
        local ok2, message2 = pcall(function() typeerr(1, {}) end)
        out[#out + 1] = (message2:gsub('^[^:]*:', ''))
        local trace
        local function tb(...)
            local a = ...
            trace = debug.traceback('t')
        end
        tb(1, 2)
        out[#out + 1] = (trace:match('tb') and 'named') or 'unnamed'
        return table.concat(out, ' | ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        let rendered = text(&results[0]);
        assert!(rendered.starts_with("late 3 | "), "{profile:?}: {rendered}");
        assert!(rendered.ends_with("| named"), "{profile:?}: {rendered}");
    }
}

/// The callee calls other functions and is called again before the first
/// call returns: each activation has its own locals.
#[test]
fn nested_and_recursive_vararg_calls() {
    let source = r"
        local function depth(n, ...)
            local a, b = ...
            if n == 0 then return a, b end
            local x, y = depth(n - 1, b, a, n)
            return x, y
        end
        local function sum(...)
            local a, b, c = ...
            if c then return a + b + c + sum(a, b) end
            return (a or 0) + (b or 0)
        end
        local out = {}
        for i = 1, 3 do
            local x, y = depth(i, i, i * 2)
            out[#out + 1] = tostring(x) .. ',' .. tostring(y)
            out[#out + 1] = sum(i, i, i)
        end
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "2,1 5 2,4 10 6,3 15", "{profile:?}");
    }
}

/// A coroutine body that is a vararg function, and one that yields from the
/// callee after the entry.
#[test]
fn vararg_functions_in_coroutines() {
    let source = r"
        local function body(...)
            local a, b = ...
            local x = coroutine.yield(a, b)
            return a, b, x
        end
        local out = {}
        local co = coroutine.create(function(...)
            local r = {body(...)}
            return table.concat(r, ',')
        end)
        local ok, p, q = coroutine.resume(co, 'p', 'q', 'r')
        out[#out + 1] = tostring(p) .. tostring(q)
        local ok2, rest = coroutine.resume(co, 'x')
        out[#out + 1] = rest
        local wrapped = coroutine.wrap(function(...)
            local a = ...
            while true do a = a + coroutine.yield(a) end
        end)
        out[#out + 1] = wrapped(1) + wrapped(2) + wrapped(3)
        return table.concat(out, ' ')
    ";
    for profile in PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "pq p,q,x 10", "{profile:?}");
    }
}
