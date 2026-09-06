//! A metatable that gains `__index`, `__mode` or `__len` after a trace
//! warmed up on `setmetatable({..}, mt)`. The absence of those fields is
//! cached per metatable, so each script must behave exactly as the
//! interpreter does once the field appears.

use std::io::Write;
use std::process::{Command, Stdio};

/// Runs the script and returns stdout and stderr. Compilation is
/// synchronous and the trace statistics are printed on stderr, so a test can
/// require that a trace really ran.
fn run(profile: &str, execution: &str, script: &str) -> (String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .env("OXIMOND_JIT_SYNC", "1")
        .env("OXIMOND_JIT_STATS", "1")
        .args(["--compat", profile, "--execution", execution, "-"])
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
    (
        String::from_utf8(output.stdout).expect("utf-8 output"),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The value of `name: N` in the statistics line on `stderr`.
fn counter(stderr: &str, name: &str) -> u64 {
    let key = format!("{name}: ");
    let start = stderr
        .find(&key)
        .unwrap_or_else(|| panic!("no {name} in {stderr}"))
        + key.len();
    stderr[start..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|digits| digits.parse().ok())
        .unwrap_or_else(|| panic!("bad {name} in {stderr}"))
}

/// Runs `script` on the interpreter and in tracing mode and requires the
/// same output, which must be `expected`. The tracing run must have compiled
/// a trace and entered native code, so a regression to interpreter-only
/// coverage fails here.
fn same(script: &str, expected: &str) {
    same_on(&["lua51", "lua54"], script, expected);
}

/// [`same`] for the given profiles only.
fn same_on(profiles: &[&str], script: &str, expected: &str) {
    for profile in profiles.iter().copied() {
        let (reference, _) = run(profile, "interpreter", script);
        assert_eq!(reference, expected, "{profile}: interpreter");
        let (traced, stderr) = run(profile, "tracing", script);
        assert_eq!(traced, expected, "{profile}: tracing");
        assert!(
            counter(&stderr, "traces_compiled") >= 1,
            "{profile}: no trace compiled: {stderr}"
        );
        assert!(
            counter(&stderr, "native_entries") >= 1,
            "{profile}: no native entry: {stderr}"
        );
    }
}

#[test]
fn an_index_added_after_warmup_is_run() {
    same(
        r"
local mt = {}
local hits = 0
local f = function() return 7 end
for i = 1, 300000 do
  if i == 200000 then mt.__index = f end
  local t = setmetatable({a = i}, mt)
  local v = t.missing
  if v then hits = hits + v end
end
print(hits)
",
        "700007\n",
    );
}

#[test]
fn a_len_added_after_warmup_is_run() {
    // Lua 5.1 tables ignore `__len`, so only 5.4 sees the change.
    same_on(
        &["lua54"],
        r"
local mt = {}
local total = 0
local f = function() return 5 end
for i = 1, 300000 do
  if i == 200000 then mt.__len = f end
  local t = setmetatable({i, i}, mt)
  total = total + #t
end
print(total)
",
        "900003\n",
    );
}

#[test]
fn a_mode_added_after_warmup_makes_the_table_weak() {
    // The trace builds tables over `mt` and stores into them. A table made
    // after the change must be weak, which shows once its key is collected.
    same(
        r#"
local mt = {}
local n = 0
for i = 1, 300000 do
  if i == 200000 then mt.__mode = "k" end
  local t = setmetatable({}, mt)
  t[1] = i
  n = n + t[1]
end
local t = setmetatable({}, mt)
t[{}] = 1
collectgarbage()
print(n, next(t) == nil)
"#,
        "45000150000\ttrue\n",
    );
}
