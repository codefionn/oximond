//! `string.find`, `string.match`, `string.gmatch` and `string.gsub` with
//! patterns in a trace: the loop compiles, the trace calls the helper of the
//! call, and the results (and the errors) are the interpreter's in every
//! profile.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree, that a trace compiled and ran natively and that the
/// trace calls `helper` (when one is named).
fn assert_pattern(source: &str, helper: Option<&str>) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 0, "{profile:?} {source} {stats:?}");
        if let Some(helper) = helper {
            assert!(
                lua.vm().trace_helper_calls().contains(&helper),
                "{profile:?} {source}: wanted {helper}, got {:?}",
                lua.vm().trace_helper_calls()
            );
        }
    }
}

/// [`assert_pattern`] for a loop the recorder may abort part way (a nested
/// loop, a call it cannot hold) and that must still run natively somewhere.
fn assert_agrees(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

/// [`assert_pattern`] for loops whose trace must run most iterations natively
/// (a floor on `iterations`) and call each of `helpers`.
fn assert_running(source: &str, helpers: &[&str]) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations > 10000, "{profile:?} {source} {stats:?}");
        for helper in helpers {
            assert!(
                lua.vm().trace_helper_calls().contains(helper),
                "{profile:?} {source}: wanted {helper}, got {:?}",
                lua.vm().trace_helper_calls()
            );
        }
    }
}

/// A loop of 3000 iterations over `body`, which sums into `n`, around the
/// subject `s`.
fn looped(setup: &str, body: &str) -> String {
    format!("local n = 0 {setup} for i = 1, 20000 do {body} end return n")
}

#[test]
fn find_with_captures_calls_the_helper() {
    let cases = [
        // Two string captures, one, none.
        (
            "local s = 'key=value; other=thing'",
            "local a, b, k, v = string.find(s, '(%w+)=(%w+)') n = n + a + b + #k + #v",
        ),
        (
            "local s = 'key=value; other=thing'",
            "local a, b, k = string.find(s, '(%w+)=') n = n + a + b + #k",
        ),
        (
            "local s = 'key=value; other=thing'",
            "local a, b = string.find(s, '%w+=%w+') n = n + a + b",
        ),
        // Position captures and a mix.
        (
            "local s = 'abc 123 def'",
            "local a, b, p, q = string.find(s, '()%d+()') n = n + a + b + p + q",
        ),
        (
            "local s = 'abc 123 def'",
            "local a, b, w, p = string.find(s, '(%a+) ()') n = n + a + b + #w + p",
        ),
        // One result only: a pattern with specials.
        ("local s = 'abc 123 def'", "n = n + string.find(s, '%d+')"),
    ];
    for (setup, body) in cases {
        assert_pattern(&looped(setup, body), Some("string.find_pattern"));
    }
}

#[test]
fn find_with_an_initial_position_and_the_plain_flag() {
    let cases = [
        // A negative initial position, one past the end, far past it.
        "local s = 'abc 123 def 456' n = n + string.find(s, '%d+', -4)",
        "local s = 'abc 123 def 456' n = n + (string.find(s, '%d+', 16) or 7)",
        "local s = 'abc 123 def 456' n = n + (string.find(s, '%d+', 100) or 7)",
        "local s = 'abc 123 def 456' n = n + (string.find(s, '%d+', -100) or 7)",
        "local s = 'abc 123 def 456' n = n + (string.find(s, '', 17) or 9)",
        // The plain flag, with two results.
        "local s = 'a.b a.b' local a, b = string.find(s, 'a.b', 2, true) n = n + a + b",
        "local s = 'a.b a.b' local a, b = string.find(s, '.', 3, true) n = n + a + b",
        "local s = 'abc' local a, b = string.find(s, '', 2, true) n = n + a + b",
    ];
    for body in cases {
        assert_pattern(&looped("", body), None);
    }
}

#[test]
fn find_with_anchors_balance_frontier_and_classes() {
    let cases = [
        "local s = 'key=value' n = n + string.find(s, '^key')",
        "local s = 'key=value' n = n + string.find(s, 'value$')",
        "local s = 'key=value' n = n + (string.find(s, '^value') or 5)",
        "local s = 'f(a(b)c) g' local a, b = string.find(s, '%b()') n = n + a + b",
        "local s = 'THE (quick) fox' local a, b = string.find(s, '%f[%a]%a+', 5) n = n + a + b",
        "local s = 'one two' local a, b = string.find(s, '[%w_]+', 2) n = n + a + b",
        "local s = 'x  y' local a, b = string.find(s, '%s+') n = n + a + b",
        "local s = 'ab12' local a, b = string.find(s, '%d-$') n = n + a + b",
        "local s = 'aXb' local a, b = string.find(s, '%u') n = n + a + b",
        "local s = 'abc' local a, b = string.find(s, '[^b]+') n = n + a + b",
    ];
    for body in cases {
        assert_pattern(&looped("", body), None);
    }
}

#[test]
fn find_that_fails_calls_the_absent_helper() {
    assert_pattern(
        &looped(
            "local s = 'hello world'",
            "if string.find(s, '%d+') == nil then n = n + i % 7 end",
        ),
        Some("string.find_absent"),
    );
    assert_pattern(
        &looped(
            "local s = 'hello world'",
            "local a, b, c = string.find(s, '(%d)(%d)') n = n + (a or 1) + (b or 2) + (c or 3)",
        ),
        Some("string.find_absent"),
    );
}

#[test]
fn match_with_and_without_captures_calls_the_helper() {
    let cases = [
        // Captures: strings, a position, a mix, none (the whole match).
        "local s = '2024-01-17 INFO' local y, m, d = string.match(s, '(%d+)-(%d+)-(%d+)') n = n + #y + #m + #d",
        "local s = '2024-01-17 INFO' local p = string.match(s, '()INFO') n = n + p",
        "local s = '2024-01-17 INFO' local p, w = string.match(s, '()(%u+)') n = n + p + #w",
        "local s = '2024-01-17 INFO' n = n + #string.match(s, '%d+')",
        "local s = '2024-01-17 INFO' n = n + #string.match(s, '%d+', 6)",
        "local s = '2024-01-17 INFO' n = n + #string.match(s, '%d+', -9) + #(string.match(s, 'zz', 3) or 'q')",
        "local s = '2024-01-17 INFO' n = n + #(string.match(s, '%d+', 100) or 'xy')",
        "local s = '2024-01-17 INFO' n = n + #string.match(s, '^(%d+)')",
        "local s = 'k = v' local k, v = string.match(s, '^(%w+)%s*=%s*(%w+)$') n = n + #k + #v",
    ];
    for body in cases {
        // A start past the end finds nothing.
        let helper = if body.contains("100)") {
            "string.match_absent"
        } else {
            "string.match"
        };
        assert_pattern(&looped("", body), Some(helper));
    }
}

#[test]
fn match_that_fails_calls_the_absent_helper() {
    assert_pattern(
        &looped(
            "local s = 'hello'",
            "if string.match(s, '%d+') == nil then n = n + 1 end",
        ),
        Some("string.match_absent"),
    );
    assert_pattern(
        &looped(
            "local s = 'hello'",
            "local a, b = string.match(s, '(%d)(%d)') n = n + (a and 1 or 2) + (b and 3 or 4)",
        ),
        Some("string.match_absent"),
    );
}

#[test]
fn match_and_find_as_methods_and_with_varying_subjects() {
    let cases = [
        looped(
            "local t = {'a1', 'b22', 'c333', 'd4444'}",
            "local s = t[i % 4 + 1] n = n + #s:match('%d+') + s:find('%d')",
        ),
        looped(
            "local t = {'a1', 'b22', 'c333', 'd4444'}",
            "local s = t[i % 4 + 1] local w, d = s:match('(%a)(%d+)') n = n + #w + #d",
        ),
        // The pattern changes between iterations.
        looped(
            "local t = {'%d+', '%a+', '%w+'}",
            "local p = t[i % 3 + 1] n = n + #string.match('ab12', p)",
        ),
        looped(
            "local t = {'%d+', '%a+', '%w+'}",
            "local p = t[i % 3 + 1] n = n + string.find('ab12', p)",
        ),
    ];
    for source in &cases {
        assert_running(source, &[]);
    }
}

/// A loop over the matches of a long subject, so that the iteration runs
/// long enough to record and repay a trace.
fn over_matches(subject: &str, body: &str) -> String {
    format!(
        "local n = 0 local line = string.rep({subject}, 300) \
         for round = 1, 3 do {body} end return n"
    )
}

#[test]
fn nested_iterators_agree_with_the_interpreter() {
    // The inner loop is long enough to trace on its own.
    assert_running(
        "local n = 0 local line = string.rep('ab cd ', 30) \
         for round = 1, 2000 do for w in string.gmatch(line, '%a+') do n = n + #w end end return n",
        &["gmatch.next"],
    );
    assert_same(&over_matches(
        "'a b, c d, '",
        "for g in string.gmatch(line, '[^,]+') do for w in string.gmatch(g, '%a') do n = n + #w end end",
    ));
}

#[test]
fn gmatch_over_words_calls_the_helpers() {
    let cases = [
        // Words, captures, position captures.
        (
            "'alpha beta gamma delta '",
            "for w in string.gmatch(line, '%a+') do n = n + #w end",
        ),
        (
            "'a=1, b=22, c=333 '",
            "for k, v in string.gmatch(line, '(%w+)=(%w+)') do n = n + #k + #v end",
        ),
        (
            "'a=1, b=22, c=333 '",
            "for p, k in string.gmatch(line, '()(%a)') do n = n + p + #k end",
        ),
        (
            "'a=1, b=22, c=333 '",
            "for p, k, q in string.gmatch(line, '()(%a)()') do n = n + p + #k + q end",
        ),
        // Empty matches, which the profiles rule differently.
        (
            "'abc'",
            "for w in string.gmatch(line, 'x*') do n = n + #w + 1 end",
        ),
        (
            "'aXbXc'",
            "for w in string.gmatch(line, '%u*') do n = n + #w + 1 end",
        ),
        (
            "'abc '",
            "for w in string.gmatch(line, '%a*') do n = n + #w + 1 end",
        ),
        ("'abc'", "for w in string.gmatch(line, '') do n = n + 1 end"),
        // A leading caret, which a profile reads as an anchor, the others as
        // a character.
        (
            "'a^b^c'",
            "for w in string.gmatch(line, '^%a') do n = n + #w + 1 end",
        ),
        (
            "'a^b^c'",
            "for w in string.gmatch(line, '^') do n = n + 1 end",
        ),
        (
            "'one two '",
            "for a, b in string.gmatch(line, '(%a)(%a)') do n = n + #a + #b end",
        ),
        // Broken out of early.
        (
            "'alpha beta gamma delta '",
            "local seen = 0 for w in string.gmatch(line, '%a+') do n = n + #w seen = seen + 1 if seen >= 800 then break end end",
        ),
        // The iterator kept in a variable and called in a loop, and used
        // after the end.
        (
            "'alpha beta gamma '",
            "local it = string.gmatch(line, '%a+') local w = it() while w do n = n + #w w = it() end n = n + (it() and 1 or 0)",
        ),
    ];
    for (subject, body) in cases {
        assert_pattern(&over_matches(subject, body), Some("gmatch.next"));
    }
}

#[test]
fn gmatch_iterators_made_and_run_out_in_the_trace() {
    // The iterator is made in the loop and called until it ends.
    for profile in PROFILES {
        let source = "local n = 0 for i = 1, 20000 do \
               local it = string.gmatch('ab cd', '%a+') \
               local x = it() local y = it() local z = it() \
               n = n + #x + #y + (z and 1 or 2) end return n";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.total_aborts(), 0, "{profile:?} {stats:?}");
        // Every iteration makes an iterator: the lanes are given back, so the
        // loop does not fall back to the interpreter after the first ones.
        assert!(stats.iterations > 19_000, "{profile:?} {stats:?}");
        let calls = lua.vm().trace_helper_calls();
        for helper in ["string.gmatch", "gmatch.next", "gmatch.end"] {
            assert!(
                calls.contains(&helper),
                "{profile:?}: {helper} in {calls:?}"
            );
        }
    }
}

#[test]
fn gmatch_iteration_calls_its_helpers() {
    for profile in PROFILES {
        let source = "local n = 0 local line = string.rep('alpha beta gamma delta ', 400) \
             for w in string.gmatch(line, '%a+') do n = n + #w end return n";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 1000, "{profile:?} {stats:?}");
        assert!(
            lua.vm().trace_helper_calls().contains(&"gmatch.next"),
            "{profile:?}: {:?}",
            lua.vm().trace_helper_calls()
        );
    }
}

#[test]
fn gmatch_with_an_initial_position() {
    for body in [
        "for w in string.gmatch(line, '%a+', 8) do n = n + #w end",
        "for w in string.gmatch(line, '%a+', -#line + 8) do n = n + #w end",
        "for w in string.gmatch(line, '%a*', 0) do n = n + #w + 1 end",
        "for w in string.gmatch(line, '%a*', 2) do n = n + #w + 1 end",
    ] {
        assert_pattern(&over_matches("'alpha beta gamma delta '", body), None);
    }
    for body in [
        "for w in string.gmatch(line, '%a*', 100000) do n = n + #w + 1 end",
        "for w in string.gmatch(line, '%a+', -5) do n = n + #w end",
        "for w in string.gmatch(line, '%a+', 1.5) do n = n + #w end",
    ] {
        assert_same(&over_matches("'alpha beta gamma delta '", body));
    }
}

#[test]
fn gsub_with_a_string_replacement_calls_the_helper() {
    let cases = [
        "local r, c = string.gsub(s, '%d+', '#') n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '<%0>') n = n + #r + c",
        "local r, c = string.gsub(s, '(%a+) (%d+)', '%2 %1') n = n + #r + c",
        "local r, c = string.gsub(s, '(%a+)', '%1%1') n = n + #r + c",
        "local r, c = string.gsub(s, '%a', '%%') n = n + #r + c",
        "local r, c = string.gsub(s, '()%d', '[%1]') n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '#', 1) n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '#', 0) n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '#', 2.0) n = n + #r + c",
        "local r, c = string.gsub(s, '', '-') n = n + #r + c",
        "local r, c = string.gsub(s, '^%a+', 'X') n = n + #r + c",
        "local r, c = string.gsub(s, '%a*', 'X') n = n + #r + c",
        "local r, c = string.gsub(s, 'z', 'y') n = n + #r + c",
        "local r = string.gsub(s, '%d', 'N') n = n + #r",
        "local r, c = s:gsub('%s+', ' ') n = n + #r + c",
    ];
    for body in cases {
        assert_pattern(
            &looped("local s = 'abc 123 def 45 g'", body),
            Some("string.gsub"),
        );
    }
}

#[test]
fn gsub_with_a_table_replacement_calls_the_helper() {
    let cases = [
        // Hits, misses and values that keep the match.
        "local r, c = string.gsub(s, '%a+', t) n = n + #r + c",
        "local r, c = string.gsub(s, '(%a)(%a)', t) n = n + #r + c",
        "local r, c = string.gsub(s, '()%a', t) n = n + #r + c",
        "local r, c = string.gsub(s, '%a+', t, 2) n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', t) n = n + #r + c",
    ];
    for body in cases {
        let setup = "local s = 'one two 3 four' \
            local t = {one = '1', two = false, four = 4.5, [1] = 'x', [3] = 'p', [5] = 7}";
        assert_pattern(&looped(setup, body), Some("string.gsub"));
    }
}

#[test]
fn gsub_inside_loops_with_changing_arguments() {
    let cases = [
        looped(
            "local t = {'a1b2', 'c33', 'd'}",
            "local r, c = string.gsub(t[i % 3 + 1], '%d', '_') n = n + #r + c",
        ),
        looped(
            "local t = {'%d', '%a', '%s'}",
            "local r, c = string.gsub('a1 b2', t[i % 3 + 1], '') n = n + #r + c",
        ),
        looped(
            "local t = {'x', 'yy', 'zzz'}",
            "local r, c = string.gsub('a1 b2', '%d', t[i % 3 + 1]) n = n + #r + c",
        ),
    ];
    for source in &cases {
        assert_running(source, &["string.gsub"]);
    }
}

/// Runs `source` in every profile traced and interpreted, where the loop
/// ends in an error or a case the helper cannot answer, and checks that the
/// outcomes agree and that a trace ran before the exit.
fn assert_exits(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

#[test]
fn errors_leave_to_the_interpreter_that_raises_them() {
    // Each loop records with a good argument `x` and meets a bad one late.
    let cases = [
        // A malformed pattern.
        ("'%d'", "'%'", "string.find(s, x)"),
        ("'%d'", "'(%d'", "string.find(s, x)"),
        ("'%d'", "'[a'", "string.find(s, x)"),
        ("'%d'", "'%b'", "string.match(s, x)"),
        ("'(%d)'", "'%1('", "string.match(s, x)"),
        ("'%d'", "'%f'", "string.match(s, x)"),
        ("'%d'", "'%'", "string.gsub(s, x, 'x')"),
        ("'%d'", "'('", "(string.gmatch(s, x))()"),
        // Too many captures.
        ("'%d'", "string.rep('()', 40)", "string.find(s, x)"),
        // An invalid replacement.
        ("'%0'", "'%'", "string.gsub(s, '%d', x)"),
        ("'%0'", "'%2'", "string.gsub(s, '%d', x)"),
        ("'%0'", "'%z'", "string.gsub(s, '%d', x)"),
        ("'%1'", "'%9'", "string.gsub(s, '(%d)', x)"),
        // A replacement table value that is no string.
        ("{['1'] = 'x'}", "{['1'] = {}}", "string.gsub(s, '%d', x)"),
        // A subject that is a number, a boolean, a table, nil.
        ("s", "12345", "string.find(x, '%d')"),
        ("s", "12345", "string.match(x, '%d')"),
        ("s", "12345", "string.gsub(x, '%d', 'x')"),
        ("s", "{}", "string.find(x, '%d')"),
        ("s", "true", "string.match(x, '%d')"),
        ("s", "nil", "string.gsub(x, '%d', 'x')"),
        ("'%d'", "7", "string.gsub(s, x, 'x')"),
        // An initial position that is no integer.
        ("1", "1.5", "string.find(s, '%d', x)"),
        ("1", "'x'", "string.match(s, '%d', x)"),
        ("1", "{}", "string.gsub(s, '%d', 'x', x)"),
    ];
    for (good, bad, call) in cases {
        let source = format!(
            "local s = 'ab12cd' return pcall(function() local n = 0 local x = {good} \
             for i = 1, 20000 do if i == 19900 then x = {bad} end \
             local r = {call} n = n + (r and 1 or 2) end return n end)"
        );
        assert_exits(&source);
    }
}

#[test]
fn a_table_replacement_with_an_index_metamethod_leaves() {
    for setup in [
        "local t = setmetatable({}, {__index = function(_, k) return k .. '!' end})",
        "local t = setmetatable({}, {__index = {['1'] = 'x'}})",
        "local t = setmetatable({['1'] = 'y'}, {__index = {['2'] = 'x'}})",
    ] {
        let source = format!(
            "local n = 0 {setup} for i = 1, 20000 do \
               local r, c = string.gsub('a1b2', '%d', t) n = n + #r + c end return n"
        );
        assert_same(&source);
    }
    // The metatable arrives late.
    assert_exits(
        "local n = 0 local t = {['1'] = 'q'} for i = 1, 20000 do \
           if i == 19900 then setmetatable(t, {__index = function(_, k) return 'm' end}) end \
           local r, c = string.gsub('a1b2', '%d', t) n = n + #r + c end return n",
    );
}

/// Runs `source` in every profile traced and interpreted, where the loop is
/// one the recorder gives up on, and checks that the outcomes agree.
fn assert_same(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
    }
}

#[test]
fn a_replacement_function_stays_in_the_interpreter() {
    assert_same(
        "local n = 0 local f = function(d) return d .. d end for i = 1, 20000 do \
           local r, c = string.gsub('a1b2', '%d', f) n = n + #r + c end return n",
    );
    assert_same(
        "local n = 0 for i = 1, 20000 do \
           local r, c = string.gsub('a1b2', '%d', string.rep) n = n + #r + c end return n",
    );
    // A loop that meets the function late.
    assert_exits(
        "local n = 0 local r = '#' for i = 1, 20000 do \
           if i == 19900 then r = function(d) return d .. d end end \
           local x, c = string.gsub('a1b2', '%d', r) n = n + #x + c end return n",
    );
}

#[test]
fn a_replaced_string_gsub_stays_the_new_function() {
    // The loop records with the library's `gsub` and runs another later.
    assert_exits(
        "local n = 0 local real = string.gsub for i = 1, 20000 do \
           if i == 19900 then string.gsub = function(s) return 'zz', 7 end end \
           local r, c = string.gsub('a1b2', '%d', '#') n = n + #r + c end return n",
    );
    assert_exits(
        "local n = 0 for i = 1, 20000 do \
           if i == 19900 then string.match = function(s) return 'zz' end end \
           n = n + #string.match('a1b2', '%d') end return n",
    );
}

#[test]
fn a_number_subject_coerces_as_the_interpreter_does() {
    for body in [
        "local a, b = string.find(12345, '3') n = n + a + b",
        "n = n + #string.match(12345, '%d%d')",
        "local r, c = string.gsub(12345, '3', 'x') n = n + #r + c",
        "n = n + #string.match('12345', 3)",
        "n = n + #string.match('12345', '3', '2')",
    ] {
        assert_same(&looped("", body));
    }
}

#[test]
fn a_pattern_that_gives_another_shape_later_leaves() {
    // The captures change from strings to positions, and from two to one.
    assert_exits(&looped(
        "local p = '(%d)(%d)'",
        "if i == 19900 then p = '()()' end local a, b = string.match('12', p) n = n + #tostring(a)",
    ));
    assert_exits(&looped(
        "local p = '(%d)(%d)'",
        "if i == 19900 then p = '(%d)' end local a, b = string.find('12', p) n = n + a + #tostring(b)",
    ));
}

#[test]
fn many_patterns_over_many_subjects_agree_with_the_interpreter() {
    // The subject and the pattern change, so the loop meets every shape of
    // result (strings, positions, none) and the side traces cover them.
    let source = "
        local subjects = {'key=value; other=thing', '  12 ab 345 ', 'f(a(b)c) x', 'THE (quick) fox',
                          'aaa bbb', '', 'x', '2024-01-17 INFO [net] up', 'a,b,,c', 'tab\\tsep'}
        local patterns = {'%a+', '(%d+)', '()%s()', '^(%w+)', '(%w+)$', '%b()', '%f[%w]%w+',
                          '[%a_][%w_]*', '(.)(.)', '(%a+)=(%a+)', 'x*', '%s*', '', '^$',
                          '(%d+)-(%d+)-(%d+)', '[^,]*', '.-', '%u+'}
        local h = 0
        local function fold(v)
          if type(v) == 'number' then h = (h * 31 + v + 7) % 1000003
          else h = (h * 31 + #tostring(v) + 3) % 1000003 end
        end
        for i = 1, 20000 do
          local s = subjects[i % #subjects + 1]
          local p = patterns[math.floor(i / 3) % #patterns + 1]
          local a, b, c, d = string.find(s, p)
          fold(a) fold(b) fold(c) fold(d)
          local x, y = string.match(s, p)
          fold(x) fold(y)
          local r, n = string.gsub(s, p, '<%0>', 3)
          fold(r) fold(n)
          local t, m = string.gsub(s, p, {a = 'A', ab = false, ['12'] = 7})
          fold(t) fold(m)
          local k = 0
          for w in string.gmatch(s, p) do k = k + 1 if k > 20 then break end end
          fold(k)
        end
        return h";
    assert_agrees(source);
}

#[test]
fn gmatch_loops_over_patterns_of_mixed_shapes_do_not_thrash_on_entry() {
    let source = "local pats = {'%a+', '()', '(%a+)', '%d', '[aeiou]()', 'x'}\n\
        local total = 0\n\
        for i = 1, 20000 do\n\
          for _ in string.gmatch('key_1=Value42 next', pats[i % #pats + 1]) do\n\
            total = total + 1\n\
          end\n\
        end\n\
        return total";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let mismatches = stats.exits_of(oximond::vm::TraceExitKind::EntryMismatch);
        assert!(mismatches < 1_000, "{profile:?} {mismatches} {stats:?}");
    }
}

#[test]
fn a_match_that_sometimes_fails_runs_through_a_side_trace() {
    // Match and no match alternate; a side trace covers the other outcome,
    // so the loop does not retire after its first exits.
    let cases = [
        (
            looped(
                "local t = {'k=v', 'nomatch', 'a=b'}",
                "local k, v = string.match(t[i % 3 + 1], '(%w+)=(%w+)') n = n + (k and #k + #v or 7)",
            ),
            ["string.match_absent", "string.match"],
        ),
        (
            looped(
                "local t = {'k=v', 'nomatch', 'a=b'}",
                "local a, b = string.find(t[i % 3 + 1], '(%w+)=(%w+)') n = n + (a or 7)",
            ),
            ["string.find_absent", "string.find_pattern"],
        ),
    ];
    for (source, helpers) in &cases {
        assert_running(source, helpers);
    }
}

#[test]
fn matches_of_alternating_capture_shapes_run_through_side_traces() {
    // The pattern, and with it the shape of the results, changes.
    let cases = [
        looped(
            "local t = {'(%a)(%a)', '(%a)', '()%a'}",
            "local x, y = string.match('ab', t[i % 3 + 1]) n = n + #tostring(x) + #tostring(y)",
        ),
        looped(
            "local t = {'()(%a+)', '%a+', '(%a)()'}",
            "local a, b, x = string.find('ab', t[i % 3 + 1]) n = n + a + b + #tostring(x)",
        ),
    ];
    for source in &cases {
        assert_running(source, &[]);
    }
}

#[test]
fn find_and_match_over_patterns_of_mixed_shapes_run_natively() {
    // The pattern changes every call: string captures, position captures,
    // none, and calls that fail. The results are typed by the shape, and the
    // loop must neither leave at every call nor refuse its entries.
    let source = "local pats = {'%a+', '(%a+)', '()', '(%d)(%d)', '()%a(%a)', 'x', '(k)(e)',\n\
            '%d()', '(%a)(%a)', '^zz'}\n\
        local subject = 'key_1=Value42 next'\n\
        local h = 0\n\
        for i = 1, 20000 do\n\
          local p = pats[i % #pats + 1]\n\
          local a, b, c, d = string.find(subject, p)\n\
          local x, y, z = string.match(subject, p)\n\
          h = (h * 31 + #tostring(a) + #tostring(b) + #tostring(c) + #tostring(d)\n\
               + #tostring(x) + #tostring(y) + #tostring(z)) % 1000003\n\
        end\n\
        return h";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let mismatches = stats.exits_of(oximond::vm::TraceExitKind::EntryMismatch);
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 15_000, "{profile:?} {stats:?}");
        assert!(mismatches < 100, "{profile:?} {mismatches} {stats:?}");
        let calls = lua.vm().trace_helper_calls();
        for helper in ["string.find_pattern", "string.match"] {
            assert!(
                calls.contains(&helper),
                "{profile:?}: {helper} in {calls:?}"
            );
        }
    }
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree, that the loop ran natively for nearly all of `iterations`
/// without entry mismatches and with fewer than `exits` guard exits, and that
/// the trace called each of `helpers`.
fn assert_stable(source: &str, helpers: &[&str], (iterations, exits): (u64, u64)) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        let mismatches = stats.exits_of(oximond::vm::TraceExitKind::EntryMismatch);
        let guards = stats.exits_of(oximond::vm::TraceExitKind::GuardFail);
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > iterations, "{profile:?} {stats:?}");
        assert!(mismatches < 20, "{profile:?} {mismatches} {stats:?}");
        assert!(guards < exits, "{profile:?} {guards} {stats:?}");
        let calls = lua.vm().trace_helper_calls();
        for helper in helpers {
            assert!(calls.contains(helper), "{profile:?}: {helper} in {calls:?}");
        }
    }
}

#[test]
fn find_of_the_first_results_over_patterns_of_any_capture_shape_stays_native() {
    // The call wants the start and the end of the match, so the number and
    // the kinds of the captures past them change nothing: one helper answers
    // every pattern, and what the loop carries keeps its type.
    let source = "local pats = {'%a+', '(%a+)', '()', '(%d)(%d)', '()%a(%a)', 'k', '(k)(e)(y)',\n\
            '%d()', '(%a)()', '(%l+)_(%d)=(%a+)'}\n\
        local subject = 'key_1=Value42 next'\n\
        local h, last = 0, 0\n\
        for i = 1, 20000 do\n\
          local first, stop = string.find(subject, pats[i % #pats + 1])\n\
          last = first or last\n\
          h = (h * 31 + first + stop + last) % 1000003\n\
        end\n\
        return h";
    assert_stable(source, &["string.find_pattern"], (19_000, 50));
}

#[test]
fn match_of_the_first_capture_over_patterns_of_any_capture_count_stays_native() {
    // Every pattern gives a string first, and the later captures differ in
    // number and in kind: the loop is the same trace for all of them.
    let source = "local pats = {'%a+', '(%a+)', '(%a)()', '(%d)(%d)', '(%l)(%a)(%a)', 'k',\n\
            '(k)(e)(y)', '(%d)()', '(%a)(%a)()(%d)', '(%l+)_(%d)=(%a+)'}\n\
        local subject = 'key_1=Value42 next'\n\
        local h, last = 0, ''\n\
        for i = 1, 20000 do\n\
          local first = string.match(subject, pats[i % #pats + 1])\n\
          last = first or last\n\
          h = (h * 31 + #first + #last) % 1000003\n\
        end\n\
        return h";
    assert_stable(source, &["string.match"], (19_000, 50));
}

#[test]
fn gmatch_of_the_first_capture_over_patterns_of_any_capture_count_stays_native() {
    // The loop has one variable, so the captures after the first are not part
    // of the step.
    let source = "local pats = {'%a+', '(%a+)', '(%a)(%a)', '(%l)(%l)(%l)', '(%a)(%a)()(%d)'}\n\
        local subject = string.rep('key_1=Value42 next ', 6)\n\
        local total, last = 0, ''\n\
        for i = 1, 6000 do\n\
          for w in string.gmatch(subject, pats[i % #pats + 1]) do\n\
            last = w\n\
            total = total + #last\n\
          end\n\
        end\n\
        return total";
    assert_stable(source, &["gmatch.next"], (10_000, 7_000));
}

#[test]
fn gmatch_loops_over_patterns_of_mixed_shapes_stay_native_without_entry_mismatches() {
    // The iterator ignores the loop's control value, so the trace does not
    // type it by the shape of the last match (a string, a position, and nil
    // on the loop's first pass): no entry mismatches, and the loop keeps
    // running natively as the shapes change.
    let source = "local pats = {'%a+', '()', '(%a+)', '%d', '[aeiou]()', 'x', '(%d)(%a)'}\n\
        local subject = string.rep('key_1=Value42 next ', 6)\n\
        local total = 0\n\
        for i = 1, 6000 do\n\
          for w, v in string.gmatch(subject, pats[i % #pats + 1]) do\n\
            total = total + #tostring(w) + #tostring(v)\n\
          end\n\
        end\n\
        return total";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let mismatches = stats.exits_of(oximond::vm::TraceExitKind::EntryMismatch);
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 10_000, "{profile:?} {stats:?}");
        assert!(mismatches < 20, "{profile:?} {mismatches} {stats:?}");
    }
}

#[test]
fn leaving_a_gmatch_loop_early_leaves_the_interpreter_state() {
    // A break out of a traced loop resumes the
    // interpreter, which calls the iterator again or drops it.
    let cases = [
        "local n = 0 local s = string.rep('ab ', 50) \
         for i = 1, 3000 do for w in s:gmatch('%a+') do n = n + #w if n % 7 == 0 then break end end end \
         return n",
        "local n = 0 local s = string.rep('ab ', 50) \
         for i = 1, 3000 do local it = s:gmatch('(%a)(%a)') \
           local a, b = it() n = n + #a for x, y in it do n = n + #y if n % 11 == 0 then break end end \
           local c = it() n = n + (c and 1 or 2) end return n",
    ];
    for source in &cases {
        assert_agrees(source);
    }
}

#[test]
fn the_hidden_control_slot_of_a_gmatch_loop_reads_as_in_the_interpreter() {
    // The loop does not carry its control value, so an exit restores the slot
    // from the first loop variable. A function the body calls every few
    // thousand iterations reads the hidden slots of the loop's frame.
    for used in [
        "local u = #w",
        "local u = 0",
        "w = w .. '!' local u = #w",
        "local u = #w w = nil",
    ] {
        let source = format!(
            "local seen = {{}}\n\
            local function peek()\n\
              local parts = {{}}\n\
              for i = 1, 12 do\n\
                local name, value = debug.getlocal(2, i)\n\
                if not name then break end\n\
                if name ~= 'seen' and name ~= 'peek' then\n\
                  parts[#parts + 1] = name .. '=' .. tostring(value)\n\
                end\n\
              end\n\
              seen[#seen + 1] = table.concat(parts, ',')\n\
            end\n\
            local count = 0\n\
            for w in string.rep('ab cd ', 10000):gmatch('%a+') do\n\
              count = count + 1\n\
              {used}\n\
              if count % 4999 == 0 then peek() end\n\
            end\n\
            return table.concat(seen, ';')"
        );
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {used}");
            if profile == CompatibilityProfile::Lua55 && used.contains("w =") {
                // The loop variable is a constant there: both fail to compile.
                assert!(expected.is_err(), "{used}");
                continue;
            }
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.native_entries > 0, "{profile:?} {used} {stats:?}");
            assert!(stats.iterations > 15_000, "{profile:?} {used} {stats:?}");
        }
    }
}

#[test]
fn a_gmatch_trace_entered_with_another_iterator_leaves_its_control_value() {
    // The trace is recorded on a `gmatch` iterator and entered with a Lua
    // function, which the interpreter then calls with the control value it
    // had: the exit must not put the iterator there.
    let source = "local function it(_, c) if #c < 12 then return c .. 'a' end end\n\
        local function run(f, st, init)\n\
          local out = {}\n\
          for w in f, st, init do out[#out + 1] = w .. 'x' end\n\
          return table.concat(out, ' ')\n\
        end\n\
        local s = string.rep('ab cd ', 2000)\n\
        for r = 1, 6 do run(s:gmatch('%a+')) end\n\
        return run(it, nil, '')";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn a_zero_iteration_gmatch_loop_in_an_inlined_callee_agrees() {
    // The callee's loop is not the trace's loop: its control slot is an
    // ordinary register of the callee's frame, which the root's registers
    // must not be confused with.
    let source = "local f = function(s) for w in s:gmatch('=') do end end\n\
        local s, n, acc, other = 'abc', 0, 0, 1000\n\
        for i = 1, 2000 do f(s) if i % 7 == 0 then n = n + 1 end acc = acc + i end\n\
        return acc, n, other";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn a_zero_iteration_gmatch_loop_in_a_loop_body_compiles() {
    let cases = [
        "local s, n, acc = 'abc', 0, 0\n\
         for i = 1, 2000 do\n\
           for w in s:gmatch('=') do end\n\
           if i % 400 == 0 then n = n + 1 end\n\
           acc = acc + i\n\
         end\n\
         return acc, n",
        "local s, n, acc, i = 'abc', 0, 0, 0\n\
         while i < 2000 do\n\
           i = i + 1\n\
           for w in s:gmatch('=') do end\n\
           local p, q, r = i, i + 1, i + 2\n\
           if i % 400 == 0 then n = n + p - i end\n\
           acc = acc + q + r\n\
         end\n\
         return acc, n",
    ];
    for source in cases {
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
            assert!(stats.iterations > 1500, "{profile:?} {source} {stats:?}");
        }
    }
}

#[test]
fn gsub_escapes_run_natively_in_the_lua51_family_and_raise_elsewhere() {
    let cases = [
        "string.gsub(s, '%.', '%.')",
        "string.gsub(s, '%-', '%-')",
        "string.gsub(s, '%a', '%a')",
        "string.gsub(s, '%d+', '<%0%x>')",
        "string.gsub(s, '%d+', '#%')",
        "string.gsub(s, '%d+', '%')",
    ];
    for call in cases {
        let source = format!(
            "local s = 'a.b-c 12 d' return pcall(function() local n = 0 \
             for i = 1, 20000 do local r, c = {call} n = n + #r + c end return n end)"
        );
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            let legacy = matches!(
                profile,
                CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
            );
            if legacy {
                assert!(stats.iterations > 19_000, "{profile:?} {source} {stats:?}");
                assert!(
                    lua.vm().trace_helper_calls().contains(&"string.gsub"),
                    "{profile:?} {source}"
                );
            } else {
                // The interpreter raises on the first call.
                assert_eq!(stats.native_entries, 0, "{profile:?} {source} {stats:?}");
            }
        }
    }
}

#[test]
fn a_short_nested_gmatch_loop_does_not_retire_its_trace() {
    // The inner loop ends after two matches in every outer iteration.
    // The pattern is a literal of the outer loop, or a local, and the subject
    // gives the same words, or one more every so often.
    let sources = [
        "local n = 0 local line = 'ab cd' \
         for round = 1, 20000 do for w in string.gmatch(line, '%a+') do n = n + #w end end \
         return n",
        "local n = 0 local line, pat = 'ab cd', '%a+' \
         for round = 1, 20000 do for w in string.gmatch(line, pat) do n = n + #w end end \
         return n",
        "local n = 0 local lines = {'ab cd', 'ef gh', 'ij kl mn'} \
         for round = 1, 20000 do \
           for w in string.gmatch(lines[round % 3 + 1], '%a+') do n = n + #w end end \
         return n",
    ];
    for source in sources {
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.iterations > 19_000, "{profile:?} {source} {stats:?}");
            assert!(
                stats.native_entries < 200,
                "{profile:?} {source} entered {} times: {stats:?}",
                stats.native_entries
            );
        }
    }
}

#[test]
fn reading_an_absent_library_field_is_a_feature_test() {
    // `math.pow` is absent in Lua 5.4 and 5.5, present in the others.
    let cases = [
        "if math.pow then n = n + x else n = n + x * 2 end",
        "n = n + (math.pow and math.pow(x, 2) or 3)",
        "local f = string.gsubx if f then n = n + 1 else n = n + x end",
        "if not string.nothing and math.nothing == nil then n = n + x end",
    ];
    for body in cases {
        let source = format!(
            "local n = 0 for i = 1, 100000 do \
             local x = i % 13 + i % 7 * 2 + i % 5 * 3 {body} end return n"
        );
        for profile in PROFILES {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, &source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, &source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.iterations > 90_000, "{profile:?} {source} {stats:?}");
            assert_eq!(stats.total_aborts(), 0, "{profile:?} {source} {stats:?}");
        }
    }
}

#[test]
fn an_absent_field_that_appears_late_leaves_the_trace() {
    // The feature test is true from iteration 99000 on (and the loop is
    // still running when it flips).
    let source = "local n = 0 local f = function(a, b) return 1 end \
        for i = 1, 100000 do \
          if i == 99000 then math.pow = f end \
          local x = i % 13 + i % 7 * 2 + i % 5 * 3 \
          n = n + (math.pow and math.pow(x, 2) or x) end \
        return n, math.pow ~= nil";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn plain_closures_carried_beside_strings_do_not_cost_the_release() {
    // Three strings and three plain closures rotate through the loop. The
    // closures come from the entry, so the release keeps only the strings
    // and the loop does not run out of lanes.
    let source = "local function up(s) return s end local function p(s) return s end \
         local function q(s) return s end local a, b, c = 'x', 'yy', 'zzz' \
         local f, g, h = up, p, q local n = 0 \
         for i = 1, 40000 do a, b, c = b, c, a f, g, h = g, h, f \
         n = n + #up(a) + #(a .. b) end return n";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries <= 4, "{profile:?} {stats:?}");
        assert!(stats.iterations > 30000, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_find_or_match_leaves_the_trace_on_the_shape_it_did_not_record() {
    // The call is run on the pattern directly: its answer (a position, a
    // string, a nil, a different number of captures) is the interpreter's
    // for every pattern and every initial position, and the trace leaves
    // through the helper's guard when the shape is not the recorded one.
    let cases = [
        "local a, b = string.find(s, '()%d+', i % 7 - 3) n = n + (a or 0) + (b or 0)",
        "local a, b, c = string.find(s, '(%a+)', i % 9) n = n + (a or 0) + (b or 0) + (c and #c or 0)",
        "local w, d = string.match(s, '(%a+) (%d*)', i % 5 + 1) n = n + (w and #w or 0) + (d and #d or 0)",
        "local m = string.match(s, '%d+', -(i % 11) - 1) n = n + (m and #m or 7)",
        "local m = string.match(s, i % 3 == 0 and '^%a' or '%a$') n = n + (m and 1 or 2)",
        "local a, b = string.find(s, 'b', 1, true) n = n + (a or 0) + (b or 0)",
        "local p, q = string.match(s, '()c()', i % 3) n = n + (p or 0) + (q or 0)",
    ];
    for body in cases {
        assert_agrees(&looped("local s = 'ab cd 12 ef c'", body));
    }
    // A pattern that changes with the iteration, one of them a miss, one
    // malformed (an error the interpreter raises, in the middle of a run).
    for body in [
        "local a = string.find(s, i % 4 == 0 and 'zzz' or 'c') n = n + (a or 100)",
        "local m = string.match(s, i % 5 == 0 and 'c' or '%d+') n = n + (m and #m or 0)",
        "local ok, m = pcall(string.match, s, i % 9000 == 8999 and '(' or 'c') \
         n = n + (ok and 1 or 1000)",
    ] {
        assert_same(&looped("local s = 'ab cd 12 ef c'", body));
    }
}

#[test]
fn a_loop_whose_last_iteration_takes_another_path_runs_inside_its_outer_loop() {
    // The malformed line is always the last one, so the iteration that
    // misses the match is the one that ends the loop. The side trace of the
    // miss runs on through the loop's end and the outer loop, and the
    // outer loop runs in the tree.
    let source = "local lines = { '2024-01-17 INFO a', '2024-01-18 WARN b', \
         '2024-01-19 DEBUG c', 'malformed line' } \
         local n = 0 \
         for round = 1, 4000 do \
           for l = 1, #lines do \
             local y, m = string.match(lines[l], '^(%d+)%-(%d+)') \
             if y then n = n + y + m else n = n - round end \
           end \
         end return n";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.outer_sides > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 10000, "{profile:?} {stats:?}");
        assert!(stats.side_aborts <= 2, "{profile:?} {stats:?}");
    }
}

#[test]
fn gmatch_inside_a_traced_loop_agrees_with_the_interpreter() {
    let setup = "local lines = { 'alpha beta gamma', 'k1=v1, k2=v22, k3=v333', \
         'one two  three   four', '', '12 345 6' }";
    let cases = [
        // Words, with the loop variable read.
        "for w in string.gmatch(lines[i % 5 + 1], '%a+') do n = n + #w end",
        // Captures, and a mix of a capture and a position.
        "for k, v in string.gmatch(lines[i % 5 + 1], '(%w+)=(%w+)') do n = n + #k + #v end",
        "for p, w in string.gmatch(lines[i % 5 + 1], '()(%a+)') do n = n + p + #w end",
        "for p, w, q in string.gmatch(lines[i % 5 + 1], '()(%d+)()') do n = n + p + #w + q end",
        // Empty matches, which the profiles rule differently.
        "for w in string.gmatch(lines[i % 5 + 1], '%a*') do n = n + #w + 1 end",
        "for w in string.gmatch(lines[i % 5 + 1], 'x*') do n = n + 1 end",
        // A leading caret, an anchor in some profiles and a literal in others.
        "for w in string.gmatch(lines[i % 5 + 1], '^%a+') do n = n + #w end",
        // The pattern changes in the middle of the loop.
        "local pattern = i < 10000 and '%a+' or '%d+' \
         for w in string.gmatch(lines[i % 5 + 1], pattern) do n = n + #w end",
        "local pattern = math.floor(i / 1000) % 2 == 0 and '(%a)(%a)' or '(%a+)' \
         for a, b in string.gmatch(lines[i % 5 + 1], pattern) do n = n + #a + (b and #b or 0) end",
        // A side exit in the middle of an iteration of the inner loop, and a
        // break out of it.
        "for w in string.gmatch(lines[i % 5 + 1], '%a+') do \
           if #w > 4 then n = n + 100 else n = n + #w end end",
        "local seen = 0 for w in string.gmatch(lines[i % 5 + 1], '%a+') do \
           seen = seen + 1 if seen > 2 then break end n = n + #w end",
        // An inner loop that runs no trip for some outer iterations.
        "for w in string.gmatch(lines[i % 5 + 1], '%d+') do n = n + #w end",
    ];
    for body in cases {
        assert_agrees(&looped(setup, body));
    }
    // Strings do not compare natively, and two short loops side by side are
    // not carried by the loop around them: the loops stay in the interpreter
    // and the answer is the same.
    for body in [
        "for w in string.gmatch(lines[i % 5 + 1], '%a+') do \
           if w == 'beta' then break end n = n + #w end",
        "for w in string.gmatch(lines[i % 5 + 1], '%a+') do n = n + #w end \
         for d in string.gmatch(lines[i % 5 + 1], '%d+') do n = n + #d * 2 end",
    ] {
        assert_same(&looped(setup, body));
    }
}

#[test]
fn a_gmatch_loop_that_runs_few_trips_is_turned_away_before_it_is_compiled() {
    // Each run of either loop has six trips, and the two loops side by side
    // are more than a recording of the loop around them follows, which cannot
    // carry them either: a trace per loop would enter and leave for six
    // iterations. The cursor of the iterator tells how many trips are left,
    // so the recording is turned away as a short numeric loop is.
    let source = "local s = 'ab cd ef gh ij kl' local n = 0 \
         for i = 1, 20000 do \
           for w in string.gmatch(s, '%a+') do n = n + #w end \
           for w in string.gmatch(s, '%a+') do n = n + #w * 2 end \
         end return n";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        let short = oximond::vm::TraceAbortReason::ShortLoop.index();
        assert!(stats.recording_aborts[short] > 0, "{profile:?} {stats:?}");
        assert_eq!(stats.traces_compiled, 0, "{profile:?} {stats:?}");
    }
    // A loop with trips enough to repay a trace is compiled.
    assert_running(
        "local n = 0 local line = string.rep('ab cd ', 40) \
         for round = 1, 3000 do for w in string.gmatch(line, '%a+') do n = n + #w end \
         for w in string.gmatch(line, '%a+') do n = n + #w end end return n",
        &["gmatch.next"],
    );
}

#[test]
fn a_gmatch_loop_in_a_branch_of_a_loop_nest_agrees_with_the_interpreter() {
    // Every fourth round of the outer loop runs a `gmatch` loop with a few
    // trips (as many as the line has words), the others skip it. The trips of
    // the recorded iteration are followed, and the lines with more or fewer
    // words leave through the guards of the step that ends the loop.
    let source = "local lines = { '2024-01-17 INFO [net] connected to host after 3 tries', \
         '2024-01-17 WARN [disk] usage at 91 percent on sda1', \
         '2024-01-18 ERROR [auth] user alice failed login 5 times', \
         '2024-01-18 INFO [net] sent 4096 bytes in 12 ms', \
         'malformed line without a date' } \
         local n = 0 \
         for round = 1, 3000 do \
           for l = 1, #lines do \
             local line = lines[l] \
             local first, last = string.find(line, '%[(%w+)%]') \
             if first then n = n + first + last end \
             if round % 4 == 0 then \
               for word in string.gmatch(line, '%a+') do n = n + #word end \
             end \
             local _, count = string.gsub(line, '%d+', '#') \
             n = (n + count) % 1000003 \
           end \
         end return n";
    assert_agrees(source);
    // The same with captures.
    let source = source.replace(
        "'%a+') do n = n + #word end",
        "'(%a)(%a*)') do n = n + #word + 1 end",
    );
    assert_agrees(&source);
}

#[test]
fn a_side_trace_follows_the_trips_of_a_gmatch_loop_inside_a_branch() {
    // The arm that runs the inner loop is taken one iteration in four. The
    // side trace of that branch records the trips the line has and the step
    // that ends the loop, instead of giving the arm up at the inner loop's
    // backward branch. A line with more or fewer words leaves through the
    // guard of the step that does not agree.
    let source = "local lines = { 'one two three four five six', 'a b c', \
         'alpha beta gamma delta epsilon zeta eta', 'x' } \
         local n = 0 \
         for i = 1, 40000 do \
           local line = lines[i % 4 + 1] \
           n = n + #line \
           if i % 3 == 0 then \
             for word in string.gmatch(line, '%a+') do n = n + #word end \
           end \
         end return n";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(stats.side_traces_recorded > 0, "{profile:?} {stats:?}");
        let inner = oximond::vm::TraceAbortReason::InnerLoop.index();
        assert_eq!(stats.recording_aborts[inner], 0, "{profile:?} {stats:?}");
        assert_eq!(stats.side_aborts, 0, "{profile:?} {stats:?}");
    }
}
