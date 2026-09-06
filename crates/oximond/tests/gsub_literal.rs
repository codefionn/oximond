//! `string.gsub` with a replacement that is literal text, which the fast loop
//! runs in place and a trace runs as a helper, against the same call through
//! the general path (a `%` in the replacement, a table, a function). The
//! expected lines are what stock Lua 5.1 and 5.4 print for the same script.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const SCRIPT: &str = r##"
local gsub = string.gsub
local function show(...)
  local out = {}
  for i = 1, select("#", ...) do out[i] = tostring((select(i, ...))) end
  return table.concat(out, "|")
end
local log
for round = 1, 40 do
  local parts = {}
  local line = "2024-01-17 INFO [net] sent 4096 bytes"
  parts[#parts + 1] = show(gsub(line, "%d+", "#"))
  parts[#parts + 1] = show(gsub(line, "%d+", "#", 2))
  parts[#parts + 1] = show(gsub(line, "%d+", "#", 0))
  parts[#parts + 1] = show(gsub(line, "%d+", "#", -3))
  parts[#parts + 1] = show(gsub(line, "%d+", "#", nil))
  parts[#parts + 1] = show(gsub(line, "%d+", ""))
  parts[#parts + 1] = show(gsub(line, "nomatch", "#"))
  parts[#parts + 1] = show(gsub(line, "^%d+", "#"))
  parts[#parts + 1] = show(gsub(line, "^%s*", "#"))
  parts[#parts + 1] = show(gsub(line, "%a+$", "#"))
  parts[#parts + 1] = show(gsub("abc", "x*", "-"))
  parts[#parts + 1] = show(gsub("hello world", "%w*", "-"))
  parts[#parts + 1] = show(gsub("abc", "", "-"))
  parts[#parts + 1] = show(gsub("", "", "-"))
  parts[#parts + 1] = show(gsub("", "x*", "-"))
  parts[#parts + 1] = show(gsub("a b  c", "%s*", "_"))
  parts[#parts + 1] = show(gsub("abc", ".", "xy"))
  parts[#parts + 1] = show(gsub("abc", "(a)(b)", "r"))
  parts[#parts + 1] = show(gsub("abc", "b", "%%"))
  parts[#parts + 1] = show(gsub("abc", "b", "[%0]"))
  parts[#parts + 1] = show(gsub("abc", "(b)", "[%1]"))
  parts[#parts + 1] = show(gsub("abc", "b", 7))
  parts[#parts + 1] = show(gsub("abc", "b", { b = "B" }))
  parts[#parts + 1] = show(gsub("a.b.c", "%.", "/", 1.0))
  local only = gsub(line, "%s+", "_")
  parts[#parts + 1] = only
  local first, second, third = gsub(line, "%s+", "_")
  parts[#parts + 1] = show(first, second, third)
  parts[#parts + 1] = show(select("#", gsub(line, "%s+", "_")))
  parts[#parts + 1] = show((pcall(gsub, line, "[a", "x")))
  parts[#parts + 1] = show((pcall(gsub, line, "%d+", "x", {})))
  parts[#parts + 1] = show((pcall(gsub, line, "%d+", "%")))
  parts[#parts + 1] = show((pcall(gsub, nil, "a", "x")))
  parts[#parts + 1] = show(gsub(12345, "3", "x"))
  parts[#parts + 1] = show(gsub(("ab"):rep(50), "b", "XYZ"))
  log = table.concat(parts, "\n")
end
return log"##;

const LUA51: &str = r"#-#-# INFO [net] sent # bytes|4
#-#-17 INFO [net] sent 4096 bytes|2
2024-01-17 INFO [net] sent 4096 bytes|0
2024-01-17 INFO [net] sent 4096 bytes|0
#-#-# INFO [net] sent # bytes|4
-- INFO [net] sent  bytes|4
2024-01-17 INFO [net] sent 4096 bytes|0
#-01-17 INFO [net] sent 4096 bytes|1
#2024-01-17 INFO [net] sent 4096 bytes|1
2024-01-17 INFO [net] sent 4096 #|1
-a-b-c-|4
-- --|4
-a-b-c-|4
-|1
-|1
_a__b__c_|6
xyxyxy|3
rc|1
a%c|1
a[b]c|1
a[b]c|1
a7c|1
aBc|1
a/b.c|1
2024-01-17_INFO_[net]_sent_4096_bytes
2024-01-17_INFO_[net]_sent_4096_bytes|5|nil
2
false
false
true
false
12x45|1
aXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZ|50";

const LUA54: &str = r"#-#-# INFO [net] sent # bytes|4
#-#-17 INFO [net] sent 4096 bytes|2
2024-01-17 INFO [net] sent 4096 bytes|0
2024-01-17 INFO [net] sent 4096 bytes|0
#-#-# INFO [net] sent # bytes|4
-- INFO [net] sent  bytes|4
2024-01-17 INFO [net] sent 4096 bytes|0
#-01-17 INFO [net] sent 4096 bytes|1
#2024-01-17 INFO [net] sent 4096 bytes|1
2024-01-17 INFO [net] sent 4096 #|1
-a-b-c-|4
- -|2
-a-b-c-|4
-|1
-|1
_a_b_c_|4
xyxyxy|3
rc|1
a%c|1
a[b]c|1
a[b]c|1
a7c|1
aBc|1
a/b.c|1
2024-01-17_INFO_[net]_sent_4096_bytes
2024-01-17_INFO_[net]_sent_4096_bytes|5|nil
2
false
false
false
false
12x45|1
aXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZaXYZ|50";

fn run(profile: CompatibilityProfile, mode: ExecutionMode) -> String {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    match lua.exec(SCRIPT).unwrap().as_slice() {
        [Value::String(text)] => String::from_utf8_lossy(text.as_bytes()).into_owned(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_literal_replacement_gives_what_stock_lua_gives() {
    for (profile, expected) in [
        (CompatibilityProfile::Lua51, LUA51),
        (CompatibilityProfile::Lua54, LUA54),
    ] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            assert_eq!(run(profile, mode), expected, "{profile:?} {mode:?}");
        }
    }
}
