//! `setmetatable` called from Lua code: the fast loop runs the common case
//! inline, and every other case must behave as the native does. The scripts
//! call it in a loop so the interpreter's call path serves them.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(profile: &str, script: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", "interpreter", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn oximond");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write script");
    let output = child.wait_with_output().expect("run oximond");
    assert_eq!(output.status.code(), Some(0), "{profile}: {output:?}");
    String::from_utf8(output.stdout).expect("utf-8 output")
}

fn both(script: &str, expected: &str) {
    for profile in ["lua51", "lua54"] {
        assert_eq!(run(profile, script), expected, "{profile}");
    }
}

#[test]
fn a_rebound_global_is_called() {
    both(
        r"
local real = setmetatable
local calls = 0
setmetatable = function(t, m) calls = calls + 1; return real(t, m) end
local mt = {}
for i = 1, 5 do local t = setmetatable({}, mt) end
print(calls)
",
        "5\n",
    );
}

#[test]
fn another_native_bound_to_the_name_is_called() {
    // `getmetatable` is a single-result native too, so only the handle
    // comparison keeps the fast loop from running it as setmetatable.
    both(
        r"
setmetatable = getmetatable
local mt = {}
local results = {}
for i = 1, 5 do results[i] = setmetatable({}, mt) end
print(results[1], results[5])
",
        "nil\tnil\n",
    );
}

#[test]
fn a_protected_metatable_errors_and_nil_clears() {
    both(
        r#"
local guard = {__metatable = "locked"}
local t = setmetatable({}, guard)
for i = 1, 3 do
  local ok, err = pcall(function() return setmetatable(t, {}) end)
  print(ok, (err:gsub("^.-:%d+: ", "")))
  ok, err = pcall(function() return setmetatable(t, nil) end)
  print(ok, (err:gsub("^.-:%d+: ", "")))
end
print(getmetatable(t))
local u = setmetatable({}, {__index = {x = 1}})
print(u.x)
for i = 1, 3 do setmetatable(u, nil) end
print(u.x, getmetatable(u))
"#,
        &format!(
            "{}locked\n1\nnil\tnil\n",
            "false\tcannot change a protected metatable\n".repeat(6)
        ),
    );
}

#[test]
fn a_bad_argument_errors_as_before() {
    let script = r#"
for _, v in ipairs({1, "s", true}) do
  local ok, err = pcall(function() return setmetatable(v, {}) end)
  print(ok, (err:gsub("^.-:%d+: ", "")))
end
local ok, err = pcall(function() return setmetatable({}, 1) end)
print(ok, (err:gsub("^.-:%d+: ", "")))
"#;
    let first = "false\tbad argument #1 to 'setmetatable' (table expected, got number)\n\
false\tbad argument #1 to 'setmetatable' (table expected, got string)\n\
false\tbad argument #1 to 'setmetatable' (table expected, got boolean)\n\
false\tbad argument #2 to 'setmetatable' (nil or table expected";
    assert_eq!(run("lua51", script), format!("{first})\n"));
    assert_eq!(run("lua54", script), format!("{first}, got number)\n"));
}

#[test]
fn a_call_hook_sees_setmetatable() {
    both(
        r#"
local seen = 0
local mt = {}
local sm = setmetatable
debug.sethook(function()
  local info = debug.getinfo(2, "f")
  if info and info.func == sm then seen = seen + 1 end
end, "c")
for i = 1, 4 do sm({}, mt) end
debug.sethook()
print(seen)
"#,
        "4\n",
    );
}

#[test]
fn extra_results_are_nil() {
    both(
        r"
local mt = {}
for i = 1, 3 do
  local t = {}
  local a, b, c = setmetatable(t, mt)
  print(a == t, b, c)
end
",
        "true\tnil\tnil\n".repeat(3).as_str(),
    );
}

#[test]
fn a_gc_metatable_registers_a_finalizer() {
    let script = r"
local finalized = 0
local mt = {__gc = function() finalized = finalized + 1 end}
for i = 1, 5 do setmetatable({}, mt) end
collectgarbage()
collectgarbage()
print(finalized)
";
    assert_eq!(run("lua54", script), "5\n");
    // Lua 5.1 tables take no finalizer.
    assert_eq!(run("lua51", script), "0\n");
}
