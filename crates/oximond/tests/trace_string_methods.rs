//! Method calls on strings (`s:upper ()`, `s:sub (i, j)`) in a trace, in every
//! profile.
//!
//! A method call looks the function up through the string metatable, which
//! the trace does not read. A helper checks on every run that the metatable's
//! `__index` still holds the function the recording saw, so each call then
//! records as the `string.f (s, ...)` form does. The loops compile, enter
//! native code, abort nowhere and give the interpreter's results; a string
//! metatable changed after the loop is hot sends the next run back to the
//! interpreter.

use std::fmt::Write as _;

use oximond::runtime::Value;
use oximond::vm::TraceExitKind;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
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
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` interpreted and traced in `profile`, checks that the results
/// agree and that the traced run compiled, entered a trace, aborted nowhere
/// and installed every helper in `helpers`.
fn check_profile(profile: CompatibilityProfile, source: &str, helpers: &[&str]) -> Lua {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected, "{profile:?} {source}");
    assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
    let stats = lua.vm().jit_statistics().trace;
    assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
    assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    assert!(stats.iterations > 10_000, "{profile:?} {source} {stats:?}");
    assert_eq!(
        stats.total_aborts(),
        0,
        "{profile:?} {source} {stats:?} {:?}",
        stats.native_aborts
    );
    let calls = lua.vm().trace_helper_calls();
    for helper in helpers {
        assert!(
            calls.contains(helper),
            "{profile:?} {source}: wanted {helper}, got {calls:?}"
        );
    }
    lua
}

fn check(source: &str, helpers: &[&str]) {
    for profile in PROFILES {
        check_profile(profile, source, helpers);
    }
}

/// Strings with embedded zeros, bytes of the upper half and mixed case, held
/// in a table so a trace reads them as borrowed strings.
const STRINGS: &str =
    r#"local T = {"hello, world", "", "\0b\0c\0", "\200\255x\128", "MiXeD 123 \233 \0Z"}"#;

/// Strings that are at least three bytes long and hold an `o`, so a call
/// that finds a byte or a letter has the same kind of answer for each.
const LONG_STRINGS: &str =
    r#"local T = {"hello, world", "foo\0bar", "xxoxx", "\200\255o\128", "MiXeD o 123"}"#;

/// `body` in a hot loop over the strings of [`STRINGS`] (as `s`), adding
/// `body`'s value `r` to a running hash with the length of the text.
fn over_strings(body: &str) -> String {
    over(STRINGS, body)
}

/// [`over_strings`] over the strings `table` declares as `T`.
fn over(table: &str, body: &str) -> String {
    format!(
        "{table}
local h = 0
for k = 1, 60000 do
  local s = T[k % 5 + 1]
  local r = {body}
  h = (h * 31 + (type(r) == 'string' and #r or (r or -1))) % 1000003
end
return h"
    )
}

#[test]
fn upper_lower_and_reverse_through_self() {
    for (call, helper) in [
        ("s:upper()", "string.upper"),
        ("s:lower()", "string.lower"),
        ("s:reverse()", "string.reverse"),
    ] {
        check(&over_strings(call), &["string.method", helper]);
    }
}

#[test]
fn sub_through_self() {
    for call in [
        "s:sub(2, k % 4 + 1)",
        "s:sub(-3)",
        "s:sub(k % 7 - 3, -2)",
        "s:sub(1, 0)",
    ] {
        check(&over_strings(call), &["string.method", "string.sub"]);
    }
}

#[test]
fn byte_through_self() {
    for call in ["s:byte(k % 3 + 1)", "s:byte(-1)", "s:byte(1, 1)"] {
        check(&over(LONG_STRINGS, call), &["string.method"]);
    }
}

#[test]
fn rep_through_self() {
    for call in ["s:rep(k % 3)", "s:rep(2, '-')"] {
        check(&over_strings(call), &["string.method", "string.rep"]);
    }
}

#[test]
fn find_through_self() {
    for call in ["s:find('o', 1, true)", "s:find('o', k % 3 + 1, true)"] {
        check(&over(LONG_STRINGS, call), &["string.method"]);
    }
}

#[test]
fn len_through_self() {
    check(&over_strings("s:len()"), &["string.method"]);
}

#[test]
fn format_through_self() {
    // The receiver is the format.
    check(
        "local fmt, h = '%d:%s', 0
         for k = 1, 60000 do
           local r = fmt:format(k, 'x')
           h = (h * 31 + #r) % 1000003
         end
         return h",
        &["string.method"],
    );
    check(
        "local fmt, h = '%5.2f|%x', 0
         for k = 1, 60000 do
           local r = fmt:format(k / 7, k)
           h = (h * 31 + #r) % 1000003
         end
         return h",
        &["string.method"],
    );
}

#[test]
fn a_string_constant_receiver_and_chained_methods() {
    check(
        "local h = 0
         for k = 1, 60000 do
           local r = ('abc'):rep(k % 4 + 1):upper()
           h = (h * 31 + #r) % 1000003
         end
         return h",
        &["string.method", "string.rep", "string.upper"],
    );
}

#[test]
fn a_replaced_index_table_entry_leaves_the_trace() {
    // The first run records the loop with the stock `upper`; the second
    // finds another function in the field and must run it.
    let source = "
local s, n = 'hello', 0
local function run(count)
  for k = 1, count do n = n + #s:upper() end
end
run(30000)
string.upper = function(text) return text .. text end
run(30000)
return n";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_replaced_string_metatable_leaves_the_trace() {
    let source = "
local s, n = 'hello', 0
local function run(count)
  for k = 1, count do n = n + #s:upper() + #s:sub(2, 3) end
end
run(30000)
getmetatable('').__index = { upper = function(text) return 'x' end, sub = string.sub }
run(30000)
local meta = getmetatable('')
meta.__index = setmetatable({}, { __index = string })
run(30000)
return n";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn byte_of_a_method_substring_reads_the_source() {
    // `s:sub (i, j)` of a string the loop does not write, read only by
    // `string.byte`: nothing makes the substring (in the profiles without
    // integers), and the method check is the one call left.
    let source = "
local s, n = string.rep('0123456789abcdef', 16), 0
for k = 1, 60000 do
  local i = k % 200 + 1
  local piece = s:sub(i, i + 3)
  local a, b, c, d = string.byte(piece, 1, 4)
  n = n + a + b + c + d
end
return n";
    for profile in PROFILES {
        let lua = check_profile(profile, source, &["string.method", "string.sub"]);
        if !matches!(
            profile,
            CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) {
            assert_eq!(lua.vm().trace_compiled_external_calls(), 1, "{profile:?}");
        }
    }
}

#[test]
fn a_replaced_string_metatable_leaves_the_trace_of_a_method_substring() {
    let source = "
local s, n = string.rep('0123456789abcdef', 16), 0
local function run(count)
  for k = 1, count do
    local i = k % 200 + 1
    local piece = s:sub(i, i + 3)
    local a, b = string.byte(piece, 1, 2)
    n = n + a + b + #piece
  end
end
run(30000)
getmetatable('').__index = { sub = function(text, i, j) return 'ZZZZ' end }
run(30000)
return n";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = run(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(actual, expected, "{profile:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > 0,
            "{profile:?} {stats:?}"
        );
    }
}

#[test]
fn a_method_the_string_library_lacks_runs_in_the_interpreter() {
    // `s:nothing ()` raises in the interpreter, as does a receiver that is
    // not a string once the loop is hot.
    let source = "
local t = {'a', 'b', 3}
local n = 0
local ok, message = pcall(function()
  for k = 1, 60000 do
    local s = t[k < 59000 and 1 or 3]
    n = n + #s:upper()
  end
end)
return n, ok, message";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}

#[test]
fn the_self_instruction_of_a_function_short_of_registers() {
    // A function that nearly runs out of registers is compiled with the
    // register-saving allocator, whose method call is one `SelfField`: the
    // lookup and the copy of the receiver in a single instruction.
    let mut source = String::from("local function run()\n");
    for index in 0..210 {
        let _ = writeln!(source, "local l{index} = {index}");
    }
    source.push_str(
        "local s, h = 'hello, world', 0
         for k = 1, 60000 do h = (h + #s:upper() + #s:sub(2, k % 3 + 2)) % 1000003 end
         return h + l1
         end
         return run()",
    );
    check(&source, &["string.method", "string.upper", "string.sub"]);
}

/// The plain field read `s.upper` records for a string receiver only if the
/// receiver is a string on every entry: a table arriving on a later entry
/// must reach the interpreter, not the string library's function.
#[test]
fn a_field_read_on_a_receiver_that_later_is_a_table() {
    let source = "local function run(s) local n = 0 \
        for i = 1, 30000 do local f = s.upper; n = n + #f('ab') end return n end \
        local a = run('x') \
        local b = run({upper = function() return 'zzz' end}) \
        return a, b";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        let shown = format!("{expected:?}");
        assert!(
            shown.contains("60000") && shown.contains("90000"),
            "{profile:?} {shown}"
        );
    }
}
