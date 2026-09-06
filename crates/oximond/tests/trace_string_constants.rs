//! `string.find`, `string.match`, `string.gmatch`, `string.gsub` and
//! `string.format` with a constant pattern or format in a trace compiled at
//! optimization level 1: the trace validates the pattern or format once and
//! calls a specialized helper. Every profile gives the interpreter's results
//! and errors at level 0 and at level 1; only level 1 runs the specialized
//! helper.

use oximond::compat::{TracingOptFlags, TracingOptLevel};
use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

type Outcome = Result<Vec<Value>, String>;

/// Runs `source` and gives its outcome and how many specialized helper calls
/// the run made.
fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    flags: Option<TracingOptFlags>,
    source: &str,
) -> (Outcome, u64, Lua) {
    let mut builder = Lua::builder().profile(profile).execution_mode(mode);
    if let Some(flags) = flags {
        builder = builder.tracing_opt_flags(flags);
    }
    let mut lua = builder.build().unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let before = lua.vm().trace_prepared_helper_calls();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    let calls = lua.vm().trace_prepared_helper_calls() - before;
    (results, calls, lua)
}

/// Checks `source` in every profile: the interpreter, tracing at level 0 and
/// tracing at level 1 agree, and `prepared` says whether level 1 ran the
/// specialized helper (level 0 never does).
fn check(source: &str, prepared: bool) {
    check_with(source, Some(prepared));
}

/// [`check`] where `prepared` of `None` leaves open whether level 1 ran the
/// specialized helper.
fn check_with(source: &str, prepared: Option<bool>) {
    for profile in PROFILES {
        let (expected, _, _) = run(profile, ExecutionMode::Interpreter, None, source);
        let (base, base_calls, lua) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O0.flags()),
            source,
        );
        assert_eq!(base, expected, "{profile:?} O0 {source}");
        assert_eq!(base_calls, 0, "{profile:?} O0 {source}");
        let names = lua.vm().trace_helper_calls();
        let (fast, calls, lua) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O1.flags()),
            source,
        );
        assert_eq!(fast, expected, "{profile:?} O1 {source}");
        // The helper names are the generic ones at both levels.
        assert_eq!(lua.vm().trace_helper_calls(), names, "{profile:?} {source}");
        if prepared == Some(true) {
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
            assert!(
                calls > 0,
                "{profile:?} O1 ran no specialized helper: {source}"
            );
        } else if prepared == Some(false) {
            assert_eq!(calls, 0, "{profile:?} O1 {source}");
        }
    }
}

fn looped(setup: &str, body: &str) -> String {
    format!("local n = 0 {setup} for i = 1, 5000 do {body} end return n")
}

#[test]
fn find_with_zero_one_and_more_captures() {
    let s = "local s = 'key=value; other=thing'";
    for body in [
        "local a, b = string.find(s, '%w+=%w+') n = n + a + b",
        "local a, b, k = string.find(s, '(%w+)=') n = n + a + b + #k",
        "local a, b, k, v = string.find(s, '(%w+)=(%w+)') n = n + a + b + #k + #v",
        "local a, b, p, q = string.find(s, '()%w+()') n = n + a + b + p + q",
        "local a, b, w, p = string.find(s, '(%a+)()') n = n + a + b + #w + p",
        "local a = string.find(s, '(%w+)=(%w+)') n = n + a",
        "local a, b, k = string.find(s, '(%w+)=(%w+)') n = n + a + b + #k",
    ] {
        check(&looped(s, body), true);
    }
}

#[test]
fn find_anchored_unanchored_and_with_an_initial_position() {
    let s = "local s = 'abc 123 def 456'";
    for body in [
        "n = n + string.find(s, '^abc')",
        "n = n + string.find(s, '456$')",
        "n = n + (string.find(s, '^123') or 7)",
        "n = n + string.find(s, '%d+', 6)",
        "n = n + string.find(s, '%d+', -4)",
        "n = n + (string.find(s, '%d+', 17) or 9)",
        "n = n + (string.find(s, '%d+', 100) or 9)",
        "n = n + (string.find(s, '%d+', -100) or 9)",
        "n = n + string.find(s, '%d+', 1.0)",
        "local a, b = string.find(s, '%d+', 2, true) n = n + (a or 3) + (b or 4)",
        "local a, b = string.find(s, 'c 1', 2, true) n = n + a + b",
    ] {
        // A call with the plain flag searches plain text, which the native
        // does at both levels.
        let plain = body.contains("true");
        // A single run is scanned in the trace, or goes to the twin.
        let run = body.contains("'%d+'");
        check_with(&looped(s, body), (!plain && !run).then_some(true));
    }
}

#[test]
fn find_that_finds_nothing() {
    let s = "local s = 'hello world'";
    for body in [
        "if string.find(s, '%d+') == nil then n = n + i % 7 end",
        "if not string.find(s, '^world') then n = n + 1 end",
        "local a, b, c = string.find(s, '(%d)(%d)') n = n + (a or 1) + (b or 2) + (c or 3)",
        "if string.find(s, '%d+', 3) == nil then n = n + 1 end",
    ] {
        // A single run is scanned in the trace.
        check_with(&looped(s, body), (!body.contains("'%d+'")).then_some(true));
    }
}

#[test]
fn match_with_zero_one_and_more_captures() {
    let s = "local s = '2024-01-17 INFO'";
    for body in [
        "n = n + #string.match(s, '%d+')",
        "n = n + #string.match(s, '(%d+)')",
        "local y, m, d = string.match(s, '(%d+)-(%d+)-(%d+)') n = n + #y + #m + #d",
        "local y, m, d = string.match(s, '(%d+)-(%d+)-(%d+)') n = n + tonumber(y) + tonumber(m) + tonumber(d)",
        "n = n + string.match(s, '()INFO')",
        "local p, w = string.match(s, '()(%u+)') n = n + p + #w",
        "local a, b, c, d = string.match(s, '(%d+)-(%d+)-(%d+) (%u+)') n = n + #a + #b + #c + #d",
        "n = n + #string.match(s, '^(%d+)')",
        "n = n + #string.match(s, '%d+', 6)",
        "n = n + #string.match(s, '%d+', -9)",
        "n = n + #(string.match(s, '%d+', 100) or 'xy')",
        "n = n + #(string.match(s, 'zz') or 'q')",
        "n = n + #(string.match(s, '^INFO') or 'q')",
    ] {
        // A single run is scanned in the trace.
        let scanned = ["'%d+'", "'(%d+)'", "'^(%d+)'"]
            .iter()
            .any(|run| body.contains(run));
        check_with(&looped(s, body), (!scanned).then_some(true));
    }
}

#[test]
fn a_bad_constant_pattern_raises_the_same_error() {
    for pattern in ["(", "%", "[a", "%b", "(()", ")", "%1"] {
        let source = format!(
            "local n = 0 local s = 'abc' for i = 1, 3000 do \
             local ok, e = pcall(string.find, s, '{pattern}') n = n + (ok and 1 or 0) \
             if not ok then err = e end end \
             local r = 0 for i = 1, 3000 do local ok, e = pcall(function() return string.match(s, '{pattern}') end) \
             if not ok then err2 = e end end return n, err, err2"
        );
        check(&source, false);
        let source = format!(
            "local s = 'abc' local n = 0 for i = 1, 3000 do n = n + string.find(s, '{pattern}') end return n"
        );
        check(&source, false);
    }
}

#[test]
fn a_constant_pattern_next_to_a_variable_one() {
    let source = "local n = 0 local s = 'key=value; other=thing' \
        local patterns = {'(%w+)=', '=(%w+)', '%a+'} \
        for i = 1, 6000 do \
            local p = patterns[i % 3 + 1] \
            local a, b = string.find(s, '%w+=') \
            local c, d = string.find(s, p) \
            local m = string.match(s, '(%w+)=') \
            local o = string.match(s, p) \
            n = n + a + b + c + d + #m + #o \
        end return n";
    check(source, true);
}

#[test]
fn a_constant_pattern_in_a_nested_function_and_with_two_sites() {
    let source = "local function f(s) return string.match(s, '^(%a+)') end \
        local n = 0 \
        for i = 1, 5000 do \
            n = n + #f('abc1') + #string.match('xy z', '%a+') + #string.match('xy z', '%a+', 2) \
        end return n";
    check(source, true);
}

#[test]
fn the_flag_alone_enables_the_specialized_helper() {
    let source = looped(
        "local s = 'abc 123'",
        "local a, b = string.find(s, '(%d+) ?') n = n + a + b",
    );
    for profile in PROFILES {
        let (expected, _, _) = run(profile, ExecutionMode::Interpreter, None, &source);
        let only = TracingOptFlags::from_bits(0x8);
        let (actual, calls, _) = run(profile, ExecutionMode::Tracing, Some(only), &source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(calls > 0, "{profile:?}");
        let (_, calls, _) = run(
            profile,
            ExecutionMode::Tracing,
            Some(
                TracingOptLevel::O1
                    .flags()
                    .without(TracingOptFlags::STRING_CONSTANTS),
            ),
            &source,
        );
        assert_eq!(calls, 0, "{profile:?}");
    }
}

#[test]
fn locals_that_alias_the_library_and_subjects_from_a_table() {
    let source = "local find, match, gsub = string.find, string.match, string.gsub \
        local lines = {'2024-01-17 INFO [net] connected', '2024-01-18 WARN [disk] usage at 91', 'malformed'} \
        local total = 0 \
        for round = 1, 3000 do for l = 1, #lines do \
            local line = lines[l] \
            local year, month, day, level = match(line, '^(%d+)%-(%d+)%-(%d+) (%u+) ') \
            if year then total = total + year + month * 31 + day + #level end \
            local first, last, tag = find(line, '%[(%w+)%]') \
            if first then total = total + first + last + #tag end \
        end end return total";
    check(source, true);
}

#[test]
fn gmatch_with_and_without_captures() {
    let s = "local s = 'one two  three four'";
    for body in [
        "for w in string.gmatch(s, '%a+') do n = n + #w end",
        "for k, v in string.gmatch('a=1, b=22, c=333', '(%w+)=(%w+)') do n = n + #k + #v end",
        "for p in string.gmatch(s, '()%a') do n = n + p end",
        "for a, p in string.gmatch(s, '(%a)()') do n = n + #a + p end",
        "for w in string.gmatch(s, '%d+') do n = n + #w end n = n + 1",
        "for w in string.gmatch(s, '^%a+') do n = n + #w end n = n + 1",
        "for e in string.gmatch(s, '%a*') do n = n + #e end",
        "for w in string.gmatch(s, '%a+') do n = n + #w if n % 3 == 0 then break end end",
    ] {
        check_with(&looped(s, body), Some(true));
    }
}

#[test]
fn gmatch_over_a_bad_constant_pattern_raises_the_same_error() {
    for pattern in ["'[a'", "'%'", "'(a'", "'%1'"] {
        let source = format!(
            "local n = 0 for i = 1, 3000 do for w in string.gmatch('abc', {pattern}) do n = n + 1 end end return n"
        );
        check_with(&source, None);
    }
}

#[test]
fn gmatch_steps_of_another_pattern_use_the_generic_step() {
    // The iterator comes from a variable pattern, so its steps are not
    // prepared, and the one from the constant is.
    let source = looped(
        "local s = 'ab cd 12 ef' local p = '%d+'",
        "for w in string.gmatch(s, p) do n = n + #w end for w in string.gmatch(s, '%a+') do n = n + #w end",
    );
    check(&source, true);
    let source = looped(
        "local s = 'ab cd 12 ef' local p = '%d+'",
        "for w in string.gmatch(s, p) do n = n + #w end",
    );
    check(&source, false);
    // Two iterators over one subject with different constants at one site.
    let source = looped(
        "local s = 'ab cd 12 ef'",
        "local a = string.gmatch(s, '%a+') local b = string.gmatch(s, '%d+') \
         n = n + #a() + #b() + #a() + (b() or '') :len()",
    );
    check_with(&source, None);
    // The same constant as a gmatch and a find.
    let source = looped(
        "local s = '^ab ^cd'",
        "for w in string.gmatch(s, '^%a+') do n = n + #w end n = n + (string.find(s, '^%a+') or 5)",
    );
    check_with(&source, None);
}

#[test]
fn gsub_with_string_replacements() {
    let s = "local s = 'hello world 123 foo 45'";
    for body in [
        "local r, c = string.gsub(s, '%d+', '#') n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '<%0>') n = n + #r + c",
        "local r, c = string.gsub(s, '(%a+) (%a+)', '%2 %1') n = n + #r + c",
        "local r, c = string.gsub(s, '()%d', '%1,') n = n + #r + c",
        "local r, c = string.gsub(s, '%d+', '100%%') n = n + #r + c",
        "local r, c = string.gsub(s, '%a+', '%1') n = n + #r + c",
        "local r, c = string.gsub(s, 'o', '0', 1) n = n + #r + c",
        "local r, c = string.gsub(s, 'o', '0', 0) n = n + #r + c",
        "local r, c = string.gsub(s, 'o', '0', i % 3) n = n + #r + c",
        "local r, c = string.gsub(s, '^hello', 'bye') n = n + #r + c",
        "local r, c = string.gsub(s, 'zzz', 'bye') n = n + #r + c",
        "local r, c = string.gsub(s, '', '-') n = n + #r + c",
        "local r, c = string.gsub(s, '%w*', '[%0]') n = n + #r + c",
        "local r = string.gsub(s, '%d+', '') n = n + #r",
        "local r, c = s:gsub('%d+', 'n') n = n + #r + c",
    ] {
        check(&looped(s, body), true);
    }
}

#[test]
fn gsub_templates_with_escapes_the_profiles_treat_differently() {
    let s = "local s = 'a1 b22'";
    for template in ["%2", "%9", "%x", "%", "x%", "%%%", "%a%0", "%-"] {
        let body = format!(
            "local ok, r, c = pcall(string.gsub, s, '%d+', '{template}') \
             n = n + (ok and #r + c or 1)"
        );
        check_with(&looped(s, &body), None);
        let direct = format!("local r, c = string.gsub(s, '(%a)', '{template}') n = n + #r + c");
        check_with(&looped(s, &direct), None);
    }
}

#[test]
fn gsub_with_table_and_function_replacements() {
    let s = "local s = 'a b c d' local t = {a = 'x', b = 1, c = 2.5}";
    for body in [
        "local r, c = string.gsub(s, '%a', t) n = n + #r + c",
        "local r, c = string.gsub(s, '(%a)', t) n = n + #r + c",
        "local r, c = string.gsub(s, '%a', function(w) return w .. w end) n = n + #r + c",
        "local r, c = string.gsub(s, '(%a) (%a)', function(x, y) return y end) n = n + #r + c",
        "local r, c = string.gsub(s, '%a', function(w) end) n = n + #r + c",
    ] {
        check_with(&looped(s, body), None);
    }
    check(
        &looped(s, "local r, c = string.gsub(s, '%a', t) n = n + #r + c"),
        true,
    );
    // The table gains an `__index` after the trace compiled.
    check_with(
        "local t = {a = 'x'} local n = 0 for i = 1, 5000 do \
         if i == 4000 then setmetatable(t, {__index = function(_, k) return k:upper() end}) end \
         local r = string.gsub('a b c d', '%a', t) n = n + #r + (r == 'x B C D' and 1 or 0) end \
         return n",
        None,
    );
    // The table gains an `__index` table between two calls of the loop.
    check_with(
        "local v = {a = 'x'} \
         local function run(k) local last for i = 1, k do last = string.gsub('a b c d', '%a', v) end return last end \
         run(3000) setmetatable(v, {__index = {b = 'Q'}}) return run(20000)",
        None,
    );
}

#[test]
fn gsub_with_a_bad_constant_pattern_or_capture_raises_the_same_error() {
    for call in [
        "string.gsub('abc', '[a', 'x')",
        "string.gsub('abc', '(a', 'x')",
        "string.gsub('abc', '%', 'x')",
        "string.gsub('abc', 'a', '%2')",
        "string.gsub('abc', '(a)', '%2')",
        "string.gsub('abc', 'a', '%')",
    ] {
        let source = format!(
            "local n = 0 for i = 1, 3000 do local ok, e = pcall(function() return {call} end) \
             if not ok then n = n + #tostring(e) end end return n"
        );
        check_with(&source, None);
    }
}

#[test]
fn gsub_with_a_constant_pattern_next_to_a_variable_one() {
    let source = looped(
        "local s = 'ab 12 cd 34' local p = '%d+'",
        "local a, x = string.gsub(s, p, '#') local b, y = string.gsub(s, '%a+', '%0%0') \
         n = n + #a + #b + x + y",
    );
    check(&source, true);
    let source = looped(
        "local s = 'ab 12 cd 34' local p = '%d+'",
        "local a, x = string.gsub(s, p, '#') n = n + #a + x",
    );
    check(&source, false);
}

#[test]
fn format_with_a_constant_format_runs_the_prepared_plan() {
    for body in [
        "n = n + #string.format('%d', i)",
        "n = n + #string.format('%5d|%-5d|%05d', i, i, i)",
        "n = n + #string.format('%i %x %X %o', i, i, i, i)",
        "n = n + #string.format('%c%c', 65 + i % 26, 97 + i % 26)",
        "n = n + #string.format('%e %f %g', i / 7, i / 7, i / 7)",
        "n = n + #string.format('%.3f|%10.2f|%-10.1f|', i / 3, i / 3, i / 3)",
        "n = n + #string.format('%s-%s', 'ab', 'cd')",
        "n = n + #string.format('%5s|%-5s|%.1s', 'ab', 'cd', 'ef')",
        "n = n + #string.format('100%% %d%%', i)",
        "n = n + #string.format('no conversions')",
        "n = n + #string.format('')",
        "n = n + #('%d:%s'):format(i, 'x')",
    ] {
        check(&looped("", body), true);
    }
}

#[test]
fn format_numbers_and_subtypes() {
    for body in [
        "n = n + #string.format('%d', 3.0)",
        "n = n + #string.format('%d', i * 1.0)",
        "n = n + #string.format('%.0f %g', i, i)",
        "n = n + #string.format('%x', 255.0)",
        "n = n + #string.format('%f', '1.5')",
        "n = n + #string.format('%d', '12')",
        "n = n + #string.format('%s', 12)",
        "n = n + #string.format('%s %s', i, i / 2)",
        "n = n + #string.format('%s', 1e100)",
    ] {
        check_with(&looped("", body), None);
    }
}

#[test]
fn format_of_plain_numbers_runs_the_prepared_plan_in_every_profile() {
    for body in [
        "n = n + #string.format('%g %e', 1/0, -1/0)",
        "n = n + #string.format('%5.1f', 0/0)",
        "n = n + #string.format('%.0f %g', i, i)",
    ] {
        check(&looped("", body), true);
    }
}

#[test]
fn format_errors_match_the_interpreter() {
    for call in [
        "string.format('%d', 3.5)",
        "string.format('%d', 2^63)",
        "string.format('%d', 'x')",
        "string.format('%d')",
        "string.format('%s %s', 'a')",
        "string.format('%d', {})",
        "string.format('%c', 256)",
        "string.format('%z', 1)",
        "string.format('%', 1)",
        "string.format('%5%')",
        "string.format('%10.123f', 1)",
    ] {
        let source = format!(
            "local n = 0 for i = 1, 3000 do local ok, e = pcall(function() return {call} end) \
             if not ok then n = n + #tostring(e) end end return n"
        );
        check_with(&source, None);
    }
}

#[test]
fn format_leaves_unhandled_arguments_to_the_generic_path() {
    // Each of these gives the interpreter's answer; the prepared plan hands
    // them to the generic path inside the helper.
    let meta = "local t = setmetatable({}, {__tostring = function() return 'T!' end})";
    for body in [
        "n = n + #string.format('%s|%d', t, i)",
        "n = n + #string.format('%q', 'a\\nb\"c')",
        "n = n + #string.format('%q %d', i, i)",
        "n = n + #string.format('%q', i / 4)",
        "n = n + #string.format('%p', t) * 0",
        "n = n + #string.format('%s', tostring(t))",
        "n = n + #string.format('%s', nil)",
        "n = n + #string.format('%s', true)",
        "n = n + #string.format('%s', {})",
        "n = n + #string.format('%a', i)",
    ] {
        check_with(&looped(meta, body), None);
    }
    for body in [
        "n = n + #string.format('%q', 'a\\nb\"c')",
        "n = n + #string.format('%q %d', i, i)",
    ] {
        // The generic path answers inside the twin, so nothing is counted.
        check(&looped("", body), false);
    }
}

#[test]
fn a_find_the_prepared_pattern_leaves_to_the_native_is_not_counted() {
    // A plain search is the native's, in the helper of a constant pattern.
    let s = "local s = 'key=value; other=thing'";
    check(
        &looped(
            s,
            "local a, b = string.find(s, 'ue; ', 1, true) n = n + a + b",
        ),
        false,
    );
    check(
        &looped(s, "local a, b = string.find(s, '%w+=', 1) n = n + a + b"),
        true,
    );
}

#[test]
fn format_with_a_constant_next_to_a_variable_format() {
    let source = looped(
        "local f = '%d,%d'",
        "local a = string.format(f, i, i) local b = string.format('<%s>', a) n = n + #a + #b",
    );
    check(&source, true);
    let source = looped(
        "local f = '%d,%d'",
        "local a = string.format(f, i, i) n = n + #a",
    );
    check(&source, false);
}

#[test]
fn format_keeps_the_interpreters_results_when_the_arguments_change_type() {
    // The loop starts on integers and switches to strings and floats, which
    // sends the trace's guards and the helper's own checks both ways.
    let source = "local n = 0 local vals = {1, 2, 'x', 4.5, '7', 8, 9.25, 'yz'} \
                  for i = 1, 4000 do local v = vals[i % 8 + 1] \
                  local ok, s = pcall(string.format, '[%s]', v) \
                  local ok2, t = pcall(string.format, '%d|%s', v, v) \
                  if ok then n = n + #s end if ok2 then n = n + #t end end return n";
    check_with(source, None);
}

#[test]
fn a_bad_constant_format_raises_the_same_error() {
    for call in [
        "string.format('%y', 1)",
        "string.format('%5', 1)",
        "string.format('%-', 1)",
        "string.format('abc%', 1)",
    ] {
        let source = format!(
            "local n = 0 for i = 1, 3000 do local ok, e = pcall(function() return {call} end) \
             if not ok then n = n + #tostring(e) end end return n"
        );
        check(&source, false);
    }
}

/// The trace statistics of `source` at `flags`.
fn trace_statistics(flags: TracingOptFlags, source: &str) -> TraceStatistics {
    // Not `run`: it switches off the judging of loops that the wait is part of.
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(ExecutionMode::Tracing)
        .tracing_opt_flags(flags)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.exec(source).unwrap();
    lua.vm().jit_statistics().trace
}

/// Whether the one loop of `source` sat out the helper-bound wait before
/// its trace compiled: it recorded, waited, and recorded again. A loop that
/// skips the wait compiles its first recording.
fn waited(flags: TracingOptFlags, source: &str) -> bool {
    let stats = trace_statistics(flags, source);
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert!(stats.iterations > 11_000, "{stats:?}");
    match stats.traces_recorded {
        1 => false,
        2 => true,
        _ => panic!("{stats:?}"),
    }
}

#[test]
fn a_loop_of_constant_patterns_skips_the_helper_bound_wait_at_level_one() {
    let long = |setup: &str, body: &str| {
        format!("local n = 0 {setup} for i = 1, 12000 do {body} end return n")
    };
    let constant = long(
        "local s = 'key=value; other=thing'",
        "local a, b = string.find(s, '%w+=%w+') n = n + a + b",
    );
    // The pattern is built at run time, so the trace cannot prepare it.
    let variable = long(
        "local s = 'key=value; other=thing' local p = string.rep('%w', 1) .. '+=%w+'",
        "local a, b = string.find(s, p) n = n + a + b",
    );
    let o0 = TracingOptLevel::O0.flags();
    let o1 = TracingOptLevel::O1.flags();
    assert!(waited(o0, &constant), "constant at O0 waits");
    assert!(waited(o0, &variable), "variable at O0 waits");
    assert!(!waited(o1, &constant), "constant at O1 does not wait");
    assert!(waited(o1, &variable), "variable at O1 waits");
    // A plain-flag find is the native's, so no prepared pattern runs.
    let plain_flag = long(
        "local s = 'key=value; other=thing'",
        "local a, b = string.find(s, 'ue; ', 1, true) n = n + a + b",
    );
    assert!(waited(o0, &plain_flag), "plain flag at O0 waits");
    assert!(waited(o1, &plain_flag), "plain flag at O1 waits");
    // A one-result find of text with no magic byte is the plain-text helper's
    // and is no pattern call, so it does not wait at either level.
    let no_magic = long(
        "local s = 'key=value; other=thing'",
        "local a = string.find(s, 'ue; ') n = n + a",
    );
    assert!(!waited(o0, &no_magic), "no magic at O0");
    assert!(!waited(o1, &no_magic), "no magic at O1");
    // The iterator is made before the loop from a run-time pattern, so the
    // steps have no twin and the loop waits.
    let stepped = long(
        "local s = string.rep('ab cd ', 4) local p = string.rep('%a', 1) .. '+' \
         local it = string.gmatch(s, p)",
        "local w = it() if w then n = n + #w else it = string.gmatch(s, p) end",
    );
    assert!(waited(o1, &stepped), "variable gmatch at O1 waits");
}

/// `rounds` passes over six lines, with a find and a gsub per line. The
/// find sits in an inner loop of six trips, so each entry of its trace runs
/// six iterations.
fn pattern_nest(rounds: u32) -> String {
    format!(
        "local n = 0 local lines = {{'a=b c', 'dd=ee f', 'g=h', 'ii=jj', 'k=l m', 'none'}} \
         for round = 1, {rounds} do for l = 1, #lines do local line = lines[l] \
           local a, b = string.find(line, '%w+=%w+') if a then n = n + a + b end \
           local _, c = string.gsub(line, '%w+', '#') n = n + c end end return n"
    )
}

#[test]
fn a_prepared_loop_with_a_short_run_per_entry_keeps_the_wait_at_level_one() {
    // 40 rounds are 240 backedges of the inner loop: it is hot after 64 and
    // the wait of 256 outlasts the rest, although its patterns are prepared.
    let stats = trace_statistics(TracingOptLevel::O1.flags(), &pattern_nest(40));
    assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    assert_eq!(stats.traces_compiled, 0, "{stats:?}");
}

#[test]
fn a_helper_bound_loop_inside_a_rejected_loop_ends_its_wait() {
    // The loop around gets hot while the inner one waits, cannot record past
    // it and is rejected, and the skip windows of a rejected loop take the
    // backedges of the loops inside it. A wait longer than the interpreter
    // counts for such a loop never ended, and the nest ran interpreted from
    // start to end (`string/pattern_hot`).
    let source = pattern_nest(3000);
    for level in [TracingOptLevel::O0, TracingOptLevel::O1] {
        let stats = trace_statistics(level.flags(), &source);
        assert_eq!(stats.traces_compiled, 1, "{level:?}: {stats:?}");
        assert!(stats.iterations > 17_000, "{level:?}: {stats:?}");
    }
}

#[test]
fn a_string_metatable_tostring_does_not_change_a_string_argument() {
    // `%s` of a string takes the string itself, whatever `__tostring` the
    // string metatable names.
    let source = looped(
        "getmetatable('').__tostring = function() return 'META' end",
        "n = n + #string.format('%s|%s', 'ab', i)",
    );
    // A number argument aborts the recording in some profiles.
    check_with(&source, None);
    let source = looped(
        "getmetatable('').__tostring = function() return 'META' end local s = 'xyz'",
        "n = n + #string.format('[%s]', s)",
    );
    check_with(&source, None);
    // Where the format traces at all, the twin does the `%s` of a string.
    let mut traced = 0;
    for profile in PROFILES {
        let (expected, _, _) = run(profile, ExecutionMode::Interpreter, None, &source);
        let (fast, calls, lua) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O1.flags()),
            &source,
        );
        assert_eq!(fast, expected, "{profile:?} {source}");
        if lua.vm().jit_statistics().trace.native_entries > 0 {
            traced += 1;
            assert!(calls > 0, "{profile:?} traced without the twin: {source}");
        }
    }
    assert!(traced > 0, "no profile traced: {source}");
}

/// Runs `source` at level 1 in `profile` and gives the side trace counters
/// (recorded, installed, compiled apart) and how many specialized helper calls it made.
fn level_one(profile: CompatibilityProfile, source: &str) -> ([u64; 3], u64) {
    let (_, calls, lua) = run(
        profile,
        ExecutionMode::Tracing,
        Some(TracingOptLevel::O1.flags()),
        source,
    );
    let stats = lua.vm().jit_statistics().trace;
    (
        [
            stats.side_traces_recorded,
            stats.side_installs,
            stats.side_compiles,
        ],
        calls,
    )
}

#[test]
fn a_constant_find_in_a_side_trace_runs_the_prepared_pattern() {
    // The branch is taken one time in five, so its find is recorded as a side
    // trace of the loop. Only the side calls the pattern helpers.
    let sources = [
        looped(
            "local s = 'key=value; other=thing'",
            "if i % 5 == 0 then local a, b = string.find(s, '%w+=%w+') n = n + a + b \
             else n = n + 1 end",
        ),
        looped(
            "local s = 'key=value; other=thing'",
            "if i % 5 == 0 then local k = string.match(s, '(%w+)=') n = n + #k \
             else n = n + 1 end",
        ),
    ];
    for source in &sources {
        check(source, true);
        for profile in PROFILES {
            let (stats, calls) = level_one(profile, source);
            assert!(stats[0] > 0, "{profile:?} {stats:?}");
            assert!(stats[1] > 0, "{profile:?} {stats:?}");
            assert!(calls > 0, "{profile:?} the side ran no twin: {source}");
        }
    }
}

#[test]
fn a_constant_find_in_a_side_compiled_apart_runs_the_prepared_pattern() {
    // The nest is called from a loop and the find sits in a branch of its
    // inner loop, so the side that holds it joins the tree as code of its own.
    let source = "local s = 'key=value; other=thing' \
        local function sweep(n, m) local acc = 0 \
          for i = 1, n do for j = i, m do \
            if j % 5 == 0 then local a, b = string.find(s, '%w+=%w+') acc = acc + a + b \
            else acc = acc + (i * j) % 7 end \
          end end return acc end \
        local t = 0 for k = 1, 12000 do t = (t + sweep(8 + k % 3, 40)) % 1000003 end return t";
    check(source, true);
    let (stats, calls) = level_one(CompatibilityProfile::Lua54, source);
    assert!(stats[2] > 0, "{stats:?}");
    assert!(calls > 0, "{stats:?}");
}

#[test]
fn a_gmatch_step_of_an_iterator_that_entered_the_trace_reads_the_trace_pattern() {
    // `it` is made before the loop and replaced by one with the trace's
    // pattern when it runs dry; the step sees it as an input.
    let source = looped(
        "local s = 'ab cd ef' local it = string.gmatch(s, '%a+')",
        "local w = it() if w then n = n + #w else it = string.gmatch(s, '%a+') end",
    );
    check(&source, true);
    // The root trace holds the step and a side holds the `gmatch`: the steps
    // of the root run the twin too, not only the refills of the side.
    for profile in PROFILES {
        let (_, calls, _) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O1.flags()),
            &source,
        );
        // 3 words in every 4 rounds are 3750 steps.
        assert!(calls >= 3750, "{profile:?} {calls}");
    }
    // The iterator came from another pattern: the step reads its own.
    let source = looped(
        "local s = 'ab cd 12' local it = string.gmatch(s, '%d+')",
        "local w = it() if w then n = n + #w else it = string.gmatch(s, '%a+') end",
    );
    check_with(&source, None);
}

/// [`check`] for a call the level 1 trace scans itself: the results are the
/// interpreter's on every profile, and no specialized helper runs.
fn check_inline(source: &str) {
    for profile in PROFILES {
        let (expected, _, _) = run(profile, ExecutionMode::Interpreter, None, source);
        let (base, _, base_lua) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O0.flags()),
            source,
        );
        assert_eq!(base, expected, "{profile:?} O0 {source}");
        let names = base_lua.vm().trace_helper_calls();
        assert_eq!(
            base_lua.vm().trace_inline_runs(),
            0,
            "{profile:?} O0 {source}"
        );
        let (fast, calls, lua) = run(
            profile,
            ExecutionMode::Tracing,
            Some(TracingOptLevel::O1.flags()),
            source,
        );
        assert_eq!(fast, expected, "{profile:?} O1 {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert_eq!(calls, 0, "{profile:?} O1 called a helper: {source}");
        assert!(
            lua.vm().trace_inline_runs() > 0,
            "{profile:?} O1 scanned nothing: {source}"
        );
        assert_eq!(lua.vm().trace_helper_calls(), names, "{profile:?} {source}");
    }
}

/// A loop that runs `body` over the strings of `subjects` (Lua source of a
/// table's items), one `s` a round.
fn over(subjects: &str, body: &str) -> String {
    format!(
        "local n = 0 local t = {{{subjects}}} \
         for i = 1, 5000 do local s = t[i % #t + 1] {body} end return n"
    )
}

#[test]
fn a_captured_single_run_is_scanned_in_the_trace() {
    let digits =
        "'abc 123 def', 'x9', '007', '9', 'a\\200b12\\0c345', '        42', '12345678901234567890'";
    for body in [
        "local a, b, c = string.find(s, '(%d+)') n = n + a * 7 + b + #c",
        "local a, b, c = string.find(s, '(%d+)', 2) n = n + (a or 1) * 7 + (b or 2) + #(c or '')",
        "local a, b = string.find(s, '(%d+)') n = n + a * 7 + b",
        "n = n + #string.match(s, '(%d+)')",
        "n = n + #(string.match(s, '^([%d ]+)') or '')",
    ] {
        check_inline(&over(digits, body));
    }
}

#[test]
fn a_single_run_find_is_scanned_in_the_trace() {
    let digits =
        "'abc 123 def', 'x9', '007', '9', 'a\\200b12\\0c345', '        42', '12345678901234567890'";
    for body in [
        "local a, b = string.find(s, '%d+') n = n + a * 7 + b",
        "local a = string.find(s, '%d+') n = n + a",
        "local a, b = string.find(s, '[%w_]+') n = n + a * 7 + b",
        "local a, b = string.find(s, '[^ ]+') n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', 1) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', 0) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', nil) n = n + a * 7 + b",
    ] {
        check_inline(&over(digits, body));
    }
}

#[test]
fn a_single_run_find_follows_the_initial_position() {
    let after = "'ab 12', 'xyz 7 9', '1234567', 'a\\0\\200 55'";
    for body in [
        "local a, b = string.find(s, '%d+', 3) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', 4) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', -2) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', -100) n = n + a * 7 + b",
        "local a, b = string.find(s, '%d+', i % 3 + 1) n = n + a * 7 + b",
    ] {
        check_inline(&over(after, body));
    }
    // From the end of the subject on there is nothing to find; past it the
    // profiles differ, and the call goes to the helper.
    let ends = "'abc 12', 'x 99', 'a\\200 07'";
    for body in [
        "if string.find(s, '%d+', #s + 1) == nil then n = n + 1 end",
        "if string.find(s, '%d+', #s) == nil then n = n + 2 end",
    ] {
        check_inline(&over("'abc', 'x', 'a\\200'", body));
        check_with(&over(ends, body), None);
    }
    for body in [
        "if string.find(s, '%d+', #s + 2) == nil then n = n + 1 end",
        "if string.find(s, '%d+', 1000) == nil then n = n + 1 end",
        "if string.find(s, '%d+', -2^31 - 1) == nil then n = n + 1 end",
    ] {
        check_with(&over("'abc', 'x', 'a\\200'", body), None);
    }
}

#[test]
fn an_anchored_single_run_is_scanned_in_the_trace() {
    for body in [
        "local a, b = string.find(s, '^%d+') n = n + a * 7 + b",
        "local m = string.match(s, '^%d+') n = n + #m",
    ] {
        check_inline(&over("'123 a', '9', '007x', '4\\0', '88\\200'", body));
    }
    check_inline(&over(
        "'123 a', '99', '007x', '45\\0', '88\\200'",
        "local a, b = string.find(s, '^%d+', 2) n = n + a * 7 + b",
    ));
    for body in [
        "if string.find(s, '^%d+') == nil then n = n + 1 end",
        "if string.match(s, '^%d+') == nil then n = n + 1 end",
    ] {
        check_inline(&over("' 123', 'a9', '', 'x\\0', '\\200'", body));
    }
    check_inline(&over(
        "' 123', 'a9', 'x\\0', '\\200'",
        "if string.find(s, '^%d+', 2) == nil then n = n + 1 end",
    ));
}

#[test]
fn a_single_run_match_gives_the_text_of_the_run() {
    let words = "'hello world', '_a_', 'ab12\\0cd', 'k=v', '\\200a\\201', 'a 42 b', 'xy'";
    for body in [
        "n = n + #string.match(s, '[%w_]+')",
        "local m = string.match(s, '%w+') n = n + #m + m:byte()",
        "local m = string.match(s, '%a+') n = n + #m + m:byte()",
        "local m = string.match(s, '[^ =]+') n = n + #m + m:byte(-1)",
        "local m = string.match(s, '[%w_]+', 2) n = n + #m",
        "local m = string.match(s, '[%w_]+', -2) n = n + #m",
        "local m = string.match(s, '.+') n = n + #m",
        "local m = string.match(s, '%S+') n = n + #m",
        "local m = string.match(s, '[%w_]+') local u = m .. '!' n = n + #u",
    ] {
        check_inline(&over(words, body));
    }
}

#[test]
fn a_single_run_that_finds_nothing_is_scanned_in_the_trace() {
    let none = "'abc', '', '  x\\200', 'x\\0y', '-', 'hello world'";
    for body in [
        "if string.find(s, '%d+') == nil then n = n + 1 end",
        "if not string.match(s, '%d+') then n = n + 1 end",
        "if string.match(s, '%d+', -1) == nil then n = n + 1 end",
        "local a, b = string.find(s, '%d+') n = n + (a or 3) + (b or 4)",
        "n = n + #(string.match(s, '%d+') or 'q')",
    ] {
        check_inline(&over(none, body));
    }
    check_inline(&over(
        "'abc', '  x\\200', 'x\\0y', '-', 'hello world'",
        "if string.find(s, '%d+', 2) == nil then n = n + 1 end",
    ));
}

#[test]
fn a_single_run_scan_reads_the_strings_the_trace_makes() {
    for body in [
        "local a, b = string.find(tostring(i * 7), '%d+') n = n + a + b",
        "local m = string.match(tostring(i) .. '-x', '%d+') n = n + #m",
        "local m = string.match(string.rep('ab', i % 5 + 1) .. 'z9', '[%a]+') n = n + #m",
        "local s = tostring(i) local a, b = string.find(s, '^%d+', 2) if a then n = n + a + b end",
    ] {
        check_inline(&looped("", body));
    }
}

#[test]
fn a_single_run_in_a_long_subject_goes_to_the_twin() {
    // A string of more than 22 bytes is not stored inline, so the trace does
    // not read it; the helper's twin answers.
    let long = "'abcdefghijklmnopqrstuvwxyz 123', string.rep('x', 40) .. '77', string.rep('0', 30)";
    for body in [
        "local a, b = string.find(s, '%d+') n = n + a * 7 + b",
        "local m = string.match(s, '%d+') n = n + #m",
    ] {
        check(&over(long, body), true);
    }
    // Short and long subjects in one loop.
    let mixed = "'ab 12', string.rep('y', 30) .. '5', '7'";
    check(
        &over(
            mixed,
            "local a, b = string.find(s, '%d+') n = n + a * 7 + b",
        ),
        true,
    );
}

#[test]
fn a_pattern_beyond_a_single_run_keeps_the_twin() {
    for body in [
        "local a, b = string.find(s, '%d+x') n = n + (a or 1) + (b or 2)",
        "local a, b = string.find(s, '(%d+)x') n = n + (a or 1) + (b or 2)",
        "local m = string.match(s, '%d+%a') n = n + #(m or 'q')",
        "local m = string.match(s, '%d*') n = n + #m",
        "local m = string.match(s, '%d-') n = n + #m",
        "local m = string.match(s, '%d?') n = n + #m",
        "local a = string.find(s, '%d+$') n = n + (a or 1)",
    ] {
        check(&over("'12a', '99x', '7', 'a'", body), true);
    }
}
