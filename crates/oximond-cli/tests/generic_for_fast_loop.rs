//! The fast loop's `ForInCall` and `ForInLoop` paths, which live outside its
//! dispatch loop and in a helper that skips the bounds check, give the same
//! results in every profile and both execution modes. The scripts pick the
//! cases the loop treats differently: string control values (the copy keeps
//! the reference count), natives run in the window, Lua closures with and
//! without captured locals, and a callable table.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

const PROFILES: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];

fn run(profile: &str, mode: &str, script: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "oximond-generic-for-fast-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    let path = dir.join("case.lua");
    std::fs::write(&path, script).expect("write script");
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg(&path)
        .env("OXIMOND_JIT_SYNC", "1")
        .output()
        .expect("run oximond");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}: {output:?}"
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn check(script: &str, expected: &str) {
    for profile in PROFILES {
        for mode in ["interpreter", "tracing"] {
            assert_eq!(run(profile, mode, script), expected, "{profile}/{mode}");
        }
    }
}

#[test]
fn string_control_values_survive_the_copy() {
    // `pairs` over string keys makes the control value a string, which the
    // loop copies with its reference count. Hot enough to be traced.
    check(
        r#"
local t = {}
for i = 1, 50 do t["key" .. i] = i end
local keys, values = 0, 0
for round = 1, 200 do
  for k, v in pairs(t) do
    keys = keys + #k
    values = values + v
  end
end
print(keys, values)
"#,
        "48200\t255000\n",
    );
}

#[test]
fn native_iterators_run_in_the_window() {
    check(
        r#"
local count, letters = 0, 0
for round = 1, 300 do
  for word in string.gmatch("the quick brown fox", "%a+") do
    count = count + 1
    letters = letters + #word
  end
end
print(count, letters)
"#,
        "1200\t4800\n",
    );
}

#[test]
fn lua_closure_iterators_with_and_without_captured_locals() {
    check(
        r"
local function counter(limit)
  local n = 0
  return function()
    n = n + 1
    if n <= limit then return n end
  end
end
local plain = 0
for round = 1, 300 do
  for i in counter(10) do plain = plain + i end
end
local captured = 0
local function run()
  local extra = 2
  for i in counter(10) do
    captured = captured + i * extra
  end
  return extra
end
for round = 1, 300 do run() end
print(plain, captured)
",
        "16500\t33000\n",
    );
}

#[test]
fn a_callable_table_iterator_still_works() {
    check(
        r"
local iterator = setmetatable({}, {__call = function(self, state, control)
  if control < state then return control + 1 end
end})
local total = 0
for round = 1, 300 do
  for i in iterator, 5, 0 do total = total + i end
end
print(total)
",
        "4500\n",
    );
}
