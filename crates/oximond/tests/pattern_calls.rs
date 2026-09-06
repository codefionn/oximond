//! The calls into `string.find`, `string.match` and `string.gmatch` that
//! write their results straight into the caller's registers, and string
//! operands of arithmetic, which the fast loop converts in place. The expected
//! text is what stock Lua 5.1 and 5.4 print for the same script, which gives
//! the same lines in both.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const SCRIPT: &str = r##"
local find, match, gmatch, gsub = string.find, string.match, string.gmatch, string.gsub
local function show(...)
  local out = {}
  for i = 1, select("#", ...) do out[i] = tostring((select(i, ...))) end
  return table.concat(out, "|")
end
local log
for round = 1, 40 do
  local line = "2024-01-17 INFO [net] sent 4096 bytes"
  local parts = {}
  local y, m, d, level = match(line, "^(%d+)%-(%d+)%-(%d+) (%u+) ")
  parts[#parts + 1] = show(y, m, d, level)
  parts[#parts + 1] = show(find(line, "%[(%w+)%]"))
  parts[#parts + 1] = show(find(line, "(%d+)%-()(%d+)"))
  parts[#parts + 1] = show(match(line, "%a+"))
  parts[#parts + 1] = show(match(line, "()%d()"))
  parts[#parts + 1] = show(match(line, "nomatch"))
  parts[#parts + 1] = show(find(line, "nomatch"))
  parts[#parts + 1] = show(find(line, "INFO", 12))
  parts[#parts + 1] = show(find(line, "[", 1, true))
  parts[#parts + 1] = show(find(line, "%a+", -5))
  parts[#parts + 1] = show(match(line, "(%d)(%d)(%d)(%d)", 30))
  parts[#parts + 1] = show(match(line, "^(%d+)", 100))
  parts[#parts + 1] = show(find(line, "", 38))
  local a, b, c, d2, e, f, g, h, i2 = match("abcdefghi", "(a)(b)(c)(d)(e)(f)(g)(h)(i)")
  parts[#parts + 1] = show(a, b, c, d2, e, f, g, h, i2)
  local words = {}
  for w in gmatch(line, "%a+") do words[#words + 1] = w end
  parts[#parts + 1] = table.concat(words, ",")
  local pairs_ = {}
  for k, v in gmatch("a=1, b=22, c=333", "(%w+)=(%w+)") do pairs_[#pairs_ + 1] = k .. v end
  parts[#parts + 1] = table.concat(pairs_, ",")
  local positions = {}
  for p in gmatch("abc", "()") do positions[#positions + 1] = p end
  parts[#parts + 1] = table.concat(positions, ",")
  local mixed = {}
  for p, w in gmatch("ab cd", "()(%a+)") do mixed[#mixed + 1] = p .. w end
  parts[#parts + 1] = table.concat(mixed, ",")
  parts[#parts + 1] = show(gsub(line, "%d+", "#"))
  parts[#parts + 1] = show(gsub(line, "(%a+)", "<%1>", 2))
  parts[#parts + 1] = show(gsub(line, "%a+", { INFO = "info", net = false }))
  parts[#parts + 1] = show(gsub(line, "%d+", function(n) return #n end))
  local sum = 0
  for n in gmatch("10 20 30", "%d+") do sum = sum + n * 2 + "1" end
  parts[#parts + 1] = sum
  parts[#parts + 1] = show(pcall(find, line, "[a"))
  parts[#parts + 1] = show(pcall(match, line, "(()"))
  log = table.concat(parts, "\n")
end
return log
"##;

const EXPECTED: &str = r"2024|01|17|INFO
17|21|net
1|7|2024|6|01
INFO
1|2
nil
nil
12|15
17|17
33|37
nil
nil
38|37
a|b|c|d|e|f|g|h|i
INFO,net,sent,bytes
a1,b22,c333
1,2,3,4
1ab,4cd
#-#-# INFO [net] sent # bytes|4
2024-01-17 <INFO> [<net>] sent 4096 bytes|2
2024-01-17 info [net] sent 4096 bytes|4
4-2-2 INFO [net] sent 4 bytes|4
123
false|malformed pattern (missing ']')
false|unfinished capture";

const ARITHMETIC: &str = r#"
local out
for i = 1, 30 do
  local a, b, c, d, e, f, g = "10", " 12 ", "0x10", "1e2", "5", "3", "7"
  local ok, err = pcall(function() return "abc" + i end)
  local ok2, err2 = pcall(function() return i * {} end)
  out = table.concat({
    a + 1, b * 2, c + 0, d - 1, e % f, g / "2", a * i, i - a, "2" ^ 3,
    tostring(ok), tostring(ok2), tostring(err ~= nil), tostring(err2 ~= nil),
  }, "|")
end
return out
"#;

const ARITHMETIC_EXPECTED: &str = "11|24|16|99|2|3.5|300|20|8|false|false|true|true";

fn run(profile: CompatibilityProfile, mode: ExecutionMode) -> String {
    run_script(SCRIPT, profile, mode)
}

fn run_script(script: &str, profile: CompatibilityProfile, mode: ExecutionMode) -> String {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    match lua.exec(script).unwrap().as_slice() {
        [Value::String(text)] => String::from_utf8_lossy(text.as_bytes()).into_owned(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn captures_positions_and_iterators_match_stock_lua() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(run(profile, mode), EXPECTED, "{profile:?} {mode:?}");
        }
    }
}

#[test]
fn numeric_strings_convert_in_arithmetic_before_lua_53() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(
            run_script(ARITHMETIC, CompatibilityProfile::Lua51, mode),
            ARITHMETIC_EXPECTED,
            "{mode:?}"
        );
    }
}

const PAST_THE_END: &str = r##"
local s, out = "abc", nil
for i = 1, 30 do
  local t = {
    tostring(string.match(s, "()", #s + 5)), tostring(string.match(s, "()", #s + 1)),
    tostring(string.find(s, "()", #s + 5)), tostring(string.find(s, "b", #s + 5, true)),
    tostring(string.find(s, "", #s + 5, true)),
    select("#", string.find(s, "()", #s + 5)),
  }
  out = table.concat(t, "|")
end
return out
"##;

/// A start past the end is clamped by Lua 5.1 and `LuaJIT` and fails from 5.2 on
/// (checked against lua5.1, lua5.4 and luajit; 5.2 follows its source), in the native and the in-place
/// path alike.
#[test]
fn a_start_past_the_end_follows_each_profiles_stock_lua() {
    let clamped = "4|4|4|nil|4|3";
    let failed = "nil|4|nil|nil|nil|1";
    for (profile, expected) in [
        (CompatibilityProfile::Lua51, clamped),
        (CompatibilityProfile::LuaJit21, clamped),
        (CompatibilityProfile::Lua52, failed),
        (CompatibilityProfile::Lua53, failed),
        (CompatibilityProfile::Lua54, failed),
        (CompatibilityProfile::Lua55, failed),
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(
                run_script(PAST_THE_END, profile, mode),
                expected,
                "{profile:?} {mode:?}"
            );
        }
    }
}
