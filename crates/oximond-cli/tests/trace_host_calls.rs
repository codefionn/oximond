//! Library calls inside traces: a hot loop that calls `tostring`,
//! `string.find`, `table.concat`, `string.format`, `string.char`,
//! `math.max`, `tonumber` or `..` and `#` on the strings they make runs
//! natively, and every call the VM cannot answer inside the trace (a
//! metamethod, a pattern, an error, a result of another type, a hook, a
//! yield, a changed callee) leaves the trace before the call, so the
//! interpreter makes it and everything observable agrees with it.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. Counters show that the loops
//! ran natively and that the exits happened.

use oximond::vm::{TraceExitKind, TraceGuardKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in every profile under both engines, checks the results
/// agree, and returns each profile's result and tracing statistics.
fn agree(source: &str) -> Vec<(CompatibilityProfile, String, TraceStatistics)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
            (profile, expected, stats)
        })
        .collect()
}

fn assert_ran_natively(profile: CompatibilityProfile, stats: &TraceStatistics, least: u64) {
    assert!(
        stats.iterations >= least,
        "{profile:?}: {} native iterations, wanted {least}\n{stats:?}",
        stats.iterations
    );
    assert!(stats.native_entries >= 1, "{profile:?}\n{stats:?}");
    assert!(stats.traces_compiled >= 1, "{profile:?}\n{stats:?}");
}

/// One hot loop per supported call: the loop must run natively in every
/// profile.
#[test]
fn each_supported_call_runs_natively() {
    let loops: &[(&str, &str)] = &[
        (
            "tostring of numbers",
            "local n = 0
             for i = 1, 6000 do local s = tostring(i % 101); n = n + #s end
             return n",
        ),
        (
            "tostring of strings",
            "local n, word = 0, 'hello'
             for i = 1, 6000 do local s = tostring(word); n = n + #s end
             return n",
        ),
        (
            "plain find that finds",
            "local n, hay = 0, string.rep('abcdefghij', 20) .. 'needle'
             for i = 1, 6000 do n = n + (string.find(hay, 'needle', 1, true) or 0) end
             return n",
        ),
        (
            "plain find that does not find",
            "local n, hay = 0, string.rep('abcdefghij', 20)
             for i = 1, 6000 do n = n + (string.find(hay, 'needle', 1, true) or 7) end
             return n",
        ),
        (
            "find of a pattern without special characters",
            "local n, hay = 0, 'a haystack with a needle in it'
             for i = 1, 6000 do n = n + (string.find(hay, 'needle') or 0) end
             return n",
        ),
        (
            "find with an init",
            "local n, hay = 0, 'one two one two one'
             for i = 1, 6000 do n = n + (string.find(hay, 'one', 5, true) or 0) end
             return n",
        ),
        (
            "table.concat",
            "local n, parts = 0, { 'left', 'right', 'x' }
             for i = 1, 6000 do n = n + #table.concat(parts, ':') end
             return n",
        ),
        (
            "table.concat with numbers and a range",
            "local n, parts = 0, { 1, 2, 'three', 4.5, 'five' }
             for i = 1, 6000 do n = n + #table.concat(parts, '-', 2, 4) end
             return n",
        ),
        (
            "string.format",
            "local n = 0
             for i = 1, 6000 do
               local s = string.format('item=%06d value=%.3f name=%s', i, i / 7, 'ab')
               n = n + #s
             end
             return n",
        ),
        (
            "string.char and ..",
            "local n = 0
             for r = 1, 12 do
               local v = ''
               for i = 1, 500 do v = v .. string.char(64 + i % 26) end
               n = n + #v
             end
             return n",
        ),
        (
            ".. of numbers and strings",
            "local n = 0
             for i = 1, 6000 do local s = 'n=' .. i .. ';' .. i / 2; n = n + #s end
             return n",
        ),
        (
            "string.len",
            "local n, s = 0, 'hello world'
             for i = 1, 6000 do n = n + string.len(s) + #s end
             return n",
        ),
        (
            "math.max and math.min",
            "local n = 0
             for i = 1, 6000 do n = n + math.max(i, 100) + math.min(i, 50) end
             return n",
        ),
        (
            "tonumber of numbers",
            "local n = 0
             for i = 1, 6000 do n = n + tonumber(i) end
             return n",
        ),
        (
            "strings stored into a table",
            "local parts = {}
             for i = 1, 4000 do parts[i] = tostring(i % 101) end
             return #table.concat(parts, ',')",
        ),
    ];
    for (name, source) in loops {
        for (profile, result, stats) in agree(source) {
            assert!(!result.contains("Nil"), "{name} {profile:?}: {result}");
            // Lua 5.2 has floats only and reads `math` through `_ENV`: the
            // floats of `math.max` take the numeric natives, which a loop
            // that calls nothing else through the VM leaves to the
            // interpreter.
            if profile == CompatibilityProfile::Lua52 && name.starts_with("math.max") {
                continue;
            }
            assert_ran_natively(profile, &stats, 1_000);
            assert!(
                stats.total_aborts() <= 3,
                "{name} {profile:?}: the loop must not keep aborting\n{stats:?}"
            );
        }
    }
}

#[test]
fn a_run_reuses_the_lanes_of_the_strings_it_released() {
    // 100000 strings, a thousand lanes: the loop drops what the next
    // iteration cannot name and a release puts those lanes back in use, so
    // the run stays in native code. (A run that builds more strings than it
    // has lanes within one iteration leaves instead: see
    // `trace_helpers.rs`.)
    let source = "local n = 0
         for i = 1, 100000 do local s = tostring(i % 997) .. 'x'; n = n + #s end
         return n";
    for (profile, result, stats) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 90_000);
        assert!(
            stats.side_exits_of(TraceGuardKind::Other) < 10,
            "{profile:?}: the lanes must be reused\n{stats:?}"
        );
    }
}

#[test]
fn a_find_that_changes_its_answer_leaves_and_comes_back() {
    // The recording saw a match; one call in five hundred does not find the needle.
    let source = "local hay, n = string.rep('x', 50) .. 'needle', 0
         for i = 1, 5000 do
           local at = string.find(hay, (i % 500 == 0) and 'absent' or 'needle', 1, true)
           if at then n = n + at else n = n - 1 end
         end
         return n";
    for (profile, result, stats) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 1_000);
    }
}

#[test]
fn a_pattern_find_is_the_interpreters() {
    // The recording saw plain text. Later calls pass patterns with special
    // characters and captures; the answer is the interpreter's.
    let source = "local hay, n = 'key=value; other=thing', 0
         local needles = { 'value', 'v.l', 'o%a+er', '(%w+)=', '^key', 'thing$', '%d' }
         for i = 1, 7000 do
           local a, b, c = string.find(hay, needles[i % 7 + 1])
           n = n + (a or 0) * 3 + (b or 1) + (c and #c or 5)
         end
         return n";
    for (profile, result, _) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
    }
    // The same call with the flag set looks for the text, special
    // characters included.
    let source = "local hay, n = 'a.b a.c', 0
         for i = 1, 6000 do
           n = n + (string.find(hay, (i % 2 == 0) and 'a.c' or 'a%.b', 1, i % 2 == 0) or 0)
         end
         return n";
    for (profile, result, _) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
    }
}

#[test]
fn tostring_of_a_table_with_tostring_or_name_is_the_interpreters() {
    // Numbers, then values that carry `__tostring` or `__name`. The type
    // guard of the element leaves the trace for them.
    let source = "local fancy = setmetatable({}, { __tostring = function() return 'fancy!' end })
         local named = setmetatable({}, { __name = 'MyType' })
         local values = {}
         for i = 1, 3000 do values[i] = i end
         values[1500] = fancy
         values[2000] = named
         values[2500] = fancy
         local out, n = {}, 0
         for i = 1, #values do
           local s = tostring(values[i])
           n = n + #s
           if i == 1500 or i == 2500 then out[#out + 1] = s end
           if i == 2000 then out[#out + 1] = s:sub(1, 6) end
         end
         return n > 0, out[1], out[2], out[3]";
    for (profile, result, _) in agree(source) {
        assert!(result.contains("fancy!"), "{profile:?}: {result}");
    }
}

#[test]
fn a_tostring_metamethod_installed_on_strings_mid_loop_is_called() {
    // From iteration 2000 on, `tostring` of a string calls the metamethod of
    // the string metatable. The helper refuses strings once the metatable
    // has one.
    let source = "local word, out, n = 'plain', {}, 0
         local meta = getmetatable('')
         for i = 1, 4000 do
           if i == 2000 then meta.__tostring = function(s) return 'meta:' .. s end end
           local s = tostring(word)
           n = n + #s
           if i == 1999 or i == 2000 then out[#out + 1] = s end
         end
         meta.__tostring = nil
         return n, out[1], out[2]";
    for (profile, result, _) in agree(source) {
        assert!(result.contains("meta:plain"), "{profile:?}: {result}");
    }
}

#[test]
fn errors_raised_by_a_call_are_the_interpreters() {
    // `string.char` of 300 and `table.concat` of a table element raise
    // errors in the middle of a hot loop; the message and the position match.
    let source = "local function char_loop()
           local n = 0
           for i = 1, 5000 do n = n + #string.char(i < 4000 and 65 or 300) end
           return n
         end
         local function concat_loop()
           local parts, n = { 'a', 'b', 'c' }, 0
           for i = 1, 5000 do
             if i == 4500 then parts[2] = {} end
             n = n + #table.concat(parts, ',')
           end
           return n
         end
         local ok1, e1 = pcall(char_loop)
         local ok2, e2 = pcall(concat_loop)
         return ok1, e1, ok2, e2";
    for (profile, result, _) in agree(source) {
        assert!(result.contains("Boolean(false)"), "{profile:?}: {result}");
    }
}

#[test]
fn a_hook_set_mid_loop_sees_the_calls_after_it() {
    // The loop runs natively until iteration 3000, when it installs a call
    // hook. The hook must see every `tostring` call after that, as it does
    // in the interpreter.
    let source = "local calls, n = 0, 0
         for i = 1, 6000 do
           if i == 3000 then
             debug.sethook(function() calls = calls + 1 end, 'c')
           end
           local s = tostring(i % 11)
           n = n + #s
         end
         debug.sethook()
         return calls > 0, n";
    for (profile, result, _) in agree(source) {
        assert!(result.contains("Boolean(true)"), "{profile:?}: {result}");
    }
    // The count itself matches, whatever it is.
    let source = "local calls = 0
         for i = 1, 6000 do
           if i == 3000 then debug.sethook(function() calls = calls + 1 end, 'c') end
           local s = tostring(i % 11) .. string.char(65 + i % 3)
         end
         debug.sethook()
         return calls";
    let _ = agree(source);
}

#[test]
fn a_yield_inside_a_loop_of_calls_resumes_where_it_stopped() {
    let source = "local co = coroutine.wrap(function()
           local n = 0
           for i = 1, 6000 do
             local s = tostring(i % 17) .. ':' .. string.format('%d', i)
             n = n + #s
             if i % 1500 == 0 then coroutine.yield(n) end
           end
           return 'done', n
         end)
         local a, b, c, d = co(), co(), co(), co()
         return a, b, c, d, co()";
    for (profile, result, _) in agree(source) {
        assert!(result.contains("done"), "{profile:?}: {result}");
    }
}

#[test]
fn garbage_made_inside_traces_is_collected() {
    // Strings and a few tables per iteration, a finalizer and a weak table:
    // the collector runs at the interpreter's safepoints between runs, and
    // the live heap stays small.
    let source = "local gone = 0
         setmetatable({}, { __gc = function() gone = gone + 1 end })
         local weak = setmetatable({}, { __mode = 'v' })
         local n = 0
         local pad = string.rep('x', 40)
         for round = 1, 60 do
           for i = 1, 1000 do
             local s = tostring(i) .. pad
             n = n + #s
           end
           weak[#weak + 1] = {}
           collectgarbage('step', 0)
         end
         collectgarbage()
         collectgarbage()
         return n, gone, collectgarbage('count') < 20000";
    for (profile, result, stats) in agree(source) {
        assert!(result.contains("Boolean(true)"), "{profile:?}: {result}");
        assert!(stats.native_entries >= 1, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn a_callee_replaced_mid_loop_leaves_the_trace() {
    let source = "local n = 0
         local real = string.char
         for i = 1, 6000 do
           if i == 3000 then string.char = function(x) return 'ZZ' end end
           n = n + #string.char(65 + i % 26)
         end
         string.char = real
         return n";
    for (profile, result, _) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
    }
    let source = "local n = 0
         local real = tostring
         for i = 1, 6000 do
           if i == 3000 then tostring = function(x) return 'abc' end end
           n = n + #tostring(i % 10)
         end
         tostring = real
         return n";
    for (profile, result, _) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
    }
}

#[test]
fn strings_stored_in_tables_are_kept() {
    let source = "local parts = {}
         for i = 1, 5000 do parts[i] = tostring(i % 101) end
         local again = {}
         for i = 1, 5000 do again[#again + 1] = parts[i] .. '!' end
         return parts[1], parts[1000], parts[5000], again[5000], #parts, #again";
    for (profile, result, stats) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
        assert!(stats.native_entries >= 1, "{profile:?}\n{stats:?}");
    }
}

#[test]
fn integer_and_float_results_keep_their_types() {
    let source = "local out, n = {}, 0
         for i = 1, 6000 do
           local a = math.max(i, 3)
           local b = string.len(tostring(i * 0.5))
           local c = tostring(i % 1 == 0 and i or 0)
           n = n + a + b + #c
           if i == 6000 then out[1] = math.type and math.type(a) or 'n/a' end
         end
         return n, out[1]";
    for (profile, result, _) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
    }
}

#[test]
fn exits_after_a_host_call_restore_the_registers() {
    // A branch leaves after the calls have run; the strings they made are
    // in registers the exit rebuilds.
    let source = "local last, n = nil, 0
         for i = 1, 6000 do
           local s = tostring(i % 13)
           local t = s .. '/' .. string.char(65 + i % 7)
           n = n + #t
           if i == 5000 then last = t; break end
         end
         return last, n";
    for (profile, result, stats) in agree(source) {
        assert!(!result.contains("Nil"), "{profile:?}: {result}");
        assert_ran_natively(profile, &stats, 1_000);
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) >= 1,
            "{profile:?}\n{stats:?}"
        );
    }
}

#[test]
fn rebinding_env_mid_loop_does_not_keep_the_old_library_table() {
    let source = "local e1 = _ENV
         local e2 = setmetatable({math = {max = function() return -1 end}}, {__index = _ENV})
         local envs = {e1, e2}
         local function f()
           local acc = 0
           for i = 1, 3000 do
             acc = acc + math.max(i, 3)
             _ENV = envs[i // 2000 + 1]
           end
           return acc
         end
         return f()";
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (traced, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(traced, expected, "{profile:?}\n{stats:?}");
        assert!(traced.contains("2000003"), "{profile:?}: {traced}");
    }
}
