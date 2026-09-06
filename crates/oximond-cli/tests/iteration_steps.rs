//! The interpreter runs `ipairs` and `next` steps of a generic `for` inside
//! its dispatch loop, and `next` follows a per-table traversal cursor. These
//! scripts pin the behaviour around both: `__index` under `ipairs` per
//! profile, nil termination, `__pairs`, writes during a traversal, the
//! invalid-key error, coroutines and count hooks. Every script is hot enough
//! to build the cursor and to be traced, and runs in both execution modes.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

const PROFILES: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];

fn run(profile: &str, mode: &str, script: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "oximond-iteration-steps-{}-{}",
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

/// Runs `script` in every listed profile and both modes; `expected` names the
/// output for a profile.
fn check_with(profiles: &[&str], script: &str, expected: impl Fn(&str) -> &'static str) {
    for profile in profiles {
        for mode in ["interpreter", "tracing"] {
            assert_eq!(
                run(profile, mode, script),
                expected(profile),
                "{profile}/{mode}"
            );
        }
    }
}

fn check(script: &str, expected: &'static str) {
    check_with(&PROFILES, script, |_| expected);
}

#[test]
fn ipairs_consults_index_only_from_lua_53() {
    // Elements 3 and 4 exist only through `__index`.
    check_with(
        &PROFILES,
        r#"
local t = setmetatable({1, 2}, {__index = function(_, i)
  if i <= 4 then return i * 100 end
end})
local seen
for round = 1, 200 do
  seen = {}
  for i, v in ipairs(t) do seen[#seen + 1] = i .. ":" .. v end
end
print(table.concat(seen, " "))
"#,
        |profile| match profile {
            "lua54" | "lua55" => "1:1 2:2 3:300 4:400\n",
            _ => "1:1 2:2\n",
        },
    );
}

#[test]
fn ipairs_present_elements_ignore_index_in_every_profile() {
    check(
        r"
local t = setmetatable({5, 6, 7}, {__index = function(_, i)
  if i > 3 then return nil end
  return 99
end})
local sum = 0
for round = 1, 200 do
  for _, v in ipairs(t) do sum = sum + v end
end
print(sum)
",
        // Present elements never ask `__index`; the first absent one does.
        "3600\n",
    );
}

#[test]
fn ipairs_stops_at_the_first_nil() {
    check(
        r#"
local t = {1, 2, 3, 4, 5}
t[3] = nil
local count, last = 0, 0
for round = 1, 300 do
  count, last = 0, 0
  for i, v in ipairs(t) do count = count + 1; last = i end
end
print(count, last)
local hashed = {}
hashed[1] = "a"; hashed[2] = "b"; hashed[4] = "d"
local n = 0
for _ in ipairs(hashed) do n = n + 1 end
print(n)
local empty = 0
for _ in ipairs({}) do empty = empty + 1 end
print(empty)
"#,
        "2\t2\n2\n0\n",
    );
}

#[test]
fn ipairs_sees_writes_made_by_the_loop_body() {
    check(
        r#"
local t = {1, 2, 3}
local seen = {}
for i, v in ipairs(t) do
  seen[#seen + 1] = v
  if i < 6 then t[#t + 1] = v * 10 end
end
print(table.concat(seen, ","))
local u = {1, 2, 3, 4}
local count = 0
for i, v in ipairs(u) do
  count = count + 1
  if i == 2 then u[3] = nil end
end
print(count)
"#,
        "1,2,3,10,20,30,100,200\n2\n",
    );
}

#[test]
fn pairs_honours_the_pairs_metamethod() {
    check_with(
        &["lua54", "lua55"],
        r"
local calls = 0
local proxy = setmetatable({}, {__pairs = function(t)
  calls = calls + 1
  local i = 0
  return function()
    i = i + 1
    if i <= 3 then return i, i * i end
  end, t, nil
end})
local sum = 0
for round = 1, 200 do
  for k, v in pairs(proxy) do sum = sum + k + v end
end
print(sum, calls)
",
        |_| "4000\t200\n",
    );
}

#[test]
fn next_sees_values_changed_during_the_traversal() {
    check(
        r#"
local t = {}
for i = 1, 40 do t[i] = i end
for i = 1, 60 do t["k" .. i] = i end
for i = 1, 20 do t[i + 100000] = i end
local visited = 0
for round = 1, 50 do
  for k, v in pairs(t) do
    t[k] = v + 1
    visited = visited + 1
  end
end
local sum = 0
for k, v in pairs(t) do sum = sum + v end
print(visited, sum)
"#,
        // 120 entries, 50 rounds. Each entry gains 50.
        "6000\t8860\n",
    );
}

#[test]
fn next_survives_clearing_fields_during_the_traversal() {
    check(
        r#"
local total = 0
for round = 1, 20 do
  local t = {}
  for i = 1, 30 do t["f" .. i] = i end
  for i = 1, 10 do t[i] = i end
  for k, v in pairs(t) do
    total = total + v
    t[k] = nil
  end
  assert(next(t) == nil)
end
print(total)
"#,
        "10400\n",
    );
}

#[test]
fn next_reports_an_invalid_key() {
    check(
        r#"
local t = {}
for i = 1, 64 do t["key" .. i] = i end
-- A full traversal first, so the table has a cursor.
local n = 0
for round = 1, 20 do
  for _ in pairs(t) do n = n + 1 end
end
print(n)
print(pcall(next, t, "absent"))
print(pcall(next, {}, 7))
print(pcall(next, t, 12.5))
local ok, err = pcall(function()
  for k in next, t, "nope" do end
end)
print(ok, (err:gsub("^.-:%d+: ", "")))
"#,
        // The message is the profile's own, with its position prefix.
        "1280\n\
         false\tinvalid key to 'next'\n\
         false\tinvalid key to 'next'\n\
         false\tinvalid key to 'next'\n\
         false\tinvalid key to 'next'\n",
    );
}

#[test]
fn next_continues_past_a_deleted_key() {
    check(
        r#"
local t = {}
for i = 1, 40 do t["k" .. i] = i end
local count = 0
for round = 1, 10 do
  for k, v in pairs(t) do count = count + 1 end
end
local first = next(t)
local after = next(t, first)
t[first] = nil
-- Lua lets a traversal resume from a key just deleted.
local resumed = next(t, first)
print(count, after == resumed)
"#,
        "400\ttrue\n",
    );
}

#[test]
fn traversals_inside_coroutines_keep_their_cursor() {
    check(
        r#"
local t = {}
for i = 1, 50 do t["k" .. i] = i end
for i = 1, 10 do t[i] = i end
local function walk()
  local sum = 0
  for k, v in pairs(t) do
    sum = sum + v
    coroutine.yield(sum)
  end
  for i, v in ipairs(t) do
    sum = sum + v
    coroutine.yield(sum)
  end
  return sum
end
local a, b = coroutine.create(walk), coroutine.create(walk)
local yields, resultA, resultB = 0, nil, nil
while coroutine.status(a) ~= "dead" do
  local ok1, x = coroutine.resume(a)
  local ok2, y = coroutine.resume(b)
  assert(ok1 and ok2)
  yields = yields + 1
  resultA, resultB = x, y
end
print(yields, resultA, resultB, coroutine.status(b))
"#,
        "71\t1385\t1385\tdead\n",
    );
}

#[test]
fn traversals_with_a_count_hook_see_every_element() {
    check(
        r#"
local t = {}
for i = 1, 100 do t["k" .. i] = i end
for i = 1, 100 do t[i] = i end
local fires = 0
debug.sethook(function() fires = fires + 1 end, "", 7)
local sum, ipairs_sum = 0, 0
for round = 1, 20 do
  for k, v in pairs(t) do sum = sum + v end
  for i, v in ipairs(t) do ipairs_sum = ipairs_sum + v end
end
debug.sethook()
print(sum, ipairs_sum, jit ~= nil or fires > 100)
"#,
        "202000\t101000\ttrue\n",
    );
}
