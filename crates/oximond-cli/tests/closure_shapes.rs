//! A frame record that already runs a closure at the shape the closure has
//! now keeps its environment and captures when the closure is called again
//! (`Frame::shape`). Anything that changes either one must make the next
//! call at that depth install them again: the record is reused at the same
//! depth by the very next call, so each program here calls a closure, changes
//! it, and calls it from the same place again.
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

const FENV_PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

const UPVALUE_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

/// `setfenv` of a function whose last call left its record behind: the next
/// call, from the same caller at the same depth, reads the new globals.
#[test]
fn setfenv_reaches_the_next_call_of_the_same_closure() {
    let source = r"
        local function f() return x end
        local function g() local r = f() return r end
        x = 'outer'
        local out = {}
        for round = 1, 4 do
            out[#out + 1] = tostring(g())
            if round == 2 then setfenv(f, {x = 'inner'}) end
            if round == 3 then setfenv(f, {x = round}) end
        end
        out[#out + 1] = tostring(f())
        return table.concat(out, ' ')
    ";
    for profile in FENV_PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "outer outer inner 3 3", "{profile:?}");
    }
}

/// The same, with the closure's own record at the tail of a chain: a tail
/// call installs the callee's shape too.
#[test]
fn setfenv_reaches_a_tail_called_closure() {
    let source = r"
        local function f() return x end
        local function g() return f() end
        x = 1
        local out = {}
        for round = 1, 4 do
            out[#out + 1] = tostring(g())
            if round == 2 then setfenv(f, {x = 20}) end
        end
        return table.concat(out, ' ')
    ";
    for profile in FENV_PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "1 1 20 20", "{profile:?}");
    }
}

/// Another closure of the same prototype keeps its own environment: a record
/// left by one is not taken for the other.
#[test]
fn closures_of_one_prototype_keep_their_own_environment() {
    let source = r"
        local function make() return function() return x end end
        local a, b = make(), make()
        setfenv(a, {x = 'a'})
        setfenv(b, {x = 'b'})
        local function g(f) local r = f() return r end
        local out = {}
        for round = 1, 3 do
            out[#out + 1] = g(a) .. g(b)
        end
        setfenv(a, {x = 'z'})
        out[#out + 1] = g(a) .. g(b)
        return table.concat(out, ' ')
    ";
    for profile in FENV_PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "ab ab ab zb", "{profile:?}");
    }
}

/// `debug.upvaluejoin` replaces which cell an upvalue names: the next call
/// of the closure, which a record at that depth just ran, reads the new one.
#[test]
fn upvaluejoin_reaches_the_next_call_of_the_same_closure() {
    let source = r"
        local one, two = 'one', 'two'
        local function f() return one end
        local function other() return two end
        local function g() local r = f() return r end
        local out = {}
        for round = 1, 4 do
            out[#out + 1] = g()
            if round == 2 then debug.upvaluejoin(f, 1, other, 1) end
        end
        return table.concat(out, ' ')
    ";
    for profile in UPVALUE_PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "one one two two", "{profile:?}");
    }
}

/// With two upvalues the captures are shared storage rather than one inline
/// cell; joining one of them is seen the same way.
#[test]
fn upvaluejoin_of_one_of_several_upvalues() {
    let source = r"
        local a, b, c = 1, 20, 300
        local function f() return a + b end
        local function other() return c end
        local function g() local r = f() return r end
        local out = {}
        for round = 1, 4 do
            out[#out + 1] = g()
            if round == 2 then debug.upvaluejoin(f, 2, other, 1) end
            if round == 3 then c = 4000 end
        end
        return table.concat(out, ' ')
    ";
    for profile in UPVALUE_PROFILES {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "21 21 301 4001", "{profile:?}");
    }
}

/// A closure called through a fixed-parameter call and then a vararg call
/// from the same place keeps the right captures each time.
#[test]
fn a_record_alternates_between_closures_and_keeps_each_ones_captures() {
    let source = r"
        local function counter(start)
            local n = start
            return function(...) n = n + select('#', ...) return n end
        end
        local a, b = counter(0), counter(100)
        local function call(f, ...) local r = f(...) return r end
        local out = {}
        for round = 1, 3 do
            out[#out + 1] = call(a, 1, 2) .. ':' .. call(b, 1)
        end
        return table.concat(out, ' ')
    ";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let results = run_both(profile, source);
        assert_eq!(text(&results[0]), "2:101 4:102 6:103", "{profile:?}");
    }
}

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// A tail call leaves its site in the record; the plain call that reuses the
/// record at that depth must not show it: names, `istailcall` and the
/// traceback of the second call are the plain call's own.
#[test]
fn a_plain_call_after_a_tail_call_does_not_inherit_its_site() {
    let source = r"
        local function inner(tag)
            local trace = debug.traceback(tag, 1)
            local info = debug.getinfo(1, 'nSt')
            return trace .. ' | ' .. tostring(info.name) .. ' ' .. tostring(info.istailcall)
        end
        local function viatail(tag) return inner(tag) end
        local function plain(tag) local r = inner(tag) return r end
        local function run(f, tag) local r = f(tag) return r end
        local out = {}
        for round = 1, 3 do
            out[#out + 1] = run(viatail, 'tail' .. round)
            out[#out + 1] = run(plain, 'plain' .. round)
        end
        return table.concat(out, '\n')
    ";
    for profile in ALL_PROFILES {
        let results = run_both(profile, source);
        let rendered = text(&results[0]);
        // The plain rounds never mention a tail call, whatever the profile
        // prints for the tail ones.
        for block in rendered.split("\nplain").skip(1) {
            let plain_part = block.split("\ntail").next().unwrap();
            assert!(
                !plain_part.contains("tail call"),
                "{profile:?}: a plain call shows a tail site:\n{rendered}"
            );
        }
    }
}
