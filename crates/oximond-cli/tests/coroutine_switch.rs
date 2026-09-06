//! Resume and yield switch stacks without the general call path.
//!
//! `coroutine.resume` and `coroutine.yield` called from the fast loop copy
//! their values between registers through one reused buffer, and a yield
//! parks its stack without building the thread's GC roots: a collection
//! rebuilds them for the stacks that moved since the last one. The counters
//! in the `OXIMOND_COUNTERS` line pin which path ran, and an allocation
//! count (with the `alloc-stats` feature) pins that a round trip allocates
//! nothing once the buffer has grown. The fixtures hold what the stock
//! interpreters print, except where a profile deviates the same way before
//! and after the switch path existed: Lua 5.1 and `LuaJIT` word the resume of
//! a running coroutine differently, `LuaJIT` words a main-thread yield
//! differently, Lua 5.5 prints a nil error as `<no error object>`, and
//! `coroutine.close` of a failing `__close` carries no position.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run_fixture(profile: &str, mode: &str, script: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode, script])
        .current_dir(fixture(""))
        .output()
        .expect("run oximond");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{profile}/{mode}/{script}: {output:?}"
    );
    String::from_utf8(output.stdout).expect("utf-8 output")
}

fn check(script: &str, stem: &str, profiles: &[&str]) {
    for profile in profiles {
        let expected = std::fs::read_to_string(fixture(&format!("{stem}.{profile}.stdout")))
            .expect("read expected output");
        for mode in ["interpreter", "tracing"] {
            assert_eq!(
                run_fixture(profile, mode, script),
                expected,
                "{profile}/{mode}"
            );
        }
    }
}

/// The directories one counted run moves its script through. Dropping it,
/// also while a failed run panics, retires the prefixed directory and
/// removes whichever of them still exists.
struct RunDirectories {
    staging: PathBuf,
    directory: PathBuf,
    retired: PathBuf,
}

impl Drop for RunDirectories {
    fn drop(&mut self) {
        let _ = std::fs::rename(&self.directory, &self.retired);
        for path in [&self.staging, &self.directory, &self.retired] {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

/// Runs `source` with counters on and returns stdout and the counters line.
fn run_counted(profile: &str, mode: &str, source: &str) -> (String, String) {
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // A directory per run, so that removing it cannot race a sibling test.
    // The directory has the `coroutine-switch-` name only while it holds the
    // script: it is built under a staging name and renamed into place, and
    // renamed out again before it is emptied. A sibling that lists the
    // prefix never sees an empty directory.
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let pid = std::process::id();
    let directories = RunDirectories {
        staging: base.join(format!("staging-{pid}-{id}")),
        directory: base.join(format!("coroutine-switch-{pid}-{id}")),
        retired: base.join(format!("retired-{pid}-{id}")),
    };
    let RunDirectories {
        staging, directory, ..
    } = &directories;
    std::fs::create_dir_all(staging).unwrap();
    std::fs::File::create(staging.join(format!("script_{id}.lua")))
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    std::fs::rename(staging, directory).unwrap();
    let script = directory.join(format!("script_{id}.lua"));
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg(&script)
        .env("OXIMOND_COUNTERS", "1")
        .output()
        .unwrap();
    drop(directories);
    assert!(output.status.success(), "{profile}/{mode}: {output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let line = stderr
        .lines()
        .find(|line| line.starts_with("oximond-counters:"))
        .unwrap_or_else(|| panic!("no counters line in {stderr}"))
        .to_string();
    (String::from_utf8(output.stdout).unwrap(), line)
}

fn value(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from {line}"))
        .parse()
        .unwrap()
}

const PROFILES: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];
const MODES: [&str; 2] = ["interpreter", "tracing"];

#[test]
fn resume_yield_errors_values_and_collection_match_stock() {
    check("coroutine_switch.lua", "coroutine_switch", &PROFILES);
}

#[test]
fn hot_loops_of_every_shape_errors_wrap_and_writes_match_stock() {
    // Loops long enough for the tracing mode to resume from native code and
    // for the short run of a thread to take every switch: values of every
    // count both ways, yields from nested calls, errors, dead and running
    // threads, wrap, yield across pcall and threads that write what the
    // driver reads.
    check("coroutine_micro.lua", "coroutine_micro", &PROFILES);
}

#[test]
fn close_with_pending_to_be_closed_variables() {
    check(
        "coroutine_switch_close.lua",
        "coroutine_switch_close",
        &["lua54", "lua55"],
    );
}

const PING_PONG: &str = r"
local rounds = tonumber(arg and arg[1]) or 3000
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local function worker(first)
  local value = first
  while true do value = yield((value * 3 + 7) % 997) end
end
local threads, total = {}, 0
for slot = 1, 8 do
  threads[slot] = create(worker)
  local ok, value = resume(threads[slot], slot)
  total = total + value
end
for i = 1, rounds do
  local ok, value = resume(threads[(i - 1) % 8 + 1], i % 997)
  total = total + value
end
print(total)
";

const PING_PONG_TOTAL: &str = "1489880\n";

#[test]
fn a_ping_pong_loop_takes_the_direct_path_and_builds_no_roots() {
    for profile in PROFILES {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, PING_PONG);
            assert_eq!(stdout, PING_PONG_TOTAL, "{profile}/{mode}");
            // 8 first resumes and 3000 more, each answered by a yield. A
            // recording runs the resume and the yield of the iteration it
            // records through the general path, so the tracing run's loops
            // (the driver's, and the worker's before it is refused) leave up
            // to two of each off the direct path; a trace that resumes
            // counts its resumes apart.
            let recordings = if mode == "tracing" { 2 } else { 0 };
            let resumes =
                value(&line, "coroutine_direct_resumes") + value(&line, "coroutine_trace_resumes");
            let yields = value(&line, "coroutine_direct_yields");
            assert!(
                (3008 - recordings..=3008).contains(&resumes),
                "{profile}/{mode}: {line}"
            );
            assert!(
                (3008 - recordings..=3008).contains(&yields),
                "{profile}/{mode}: {line}"
            );
            assert_eq!(
                value(&line, "coroutine_root_snapshots"),
                0,
                "{profile}/{mode}: {line}"
            );
        }
    }
}

#[test]
fn resetting_a_stale_flag_between_yields_lists_the_thread_once() {
    // debug.setlocal rebuilds a suspended thread's roots and clears its stale
    // flag, so the next yield sets it again. No collection runs in between,
    // so a list that pushed on every flag change would grow by one entry per
    // round. The thread must be listed once, and the collection visits it once.
    let source = r#"
local co = coroutine.create(function() local mine = 0
  while true do coroutine.yield(mine) end end)
for round = 1, 2000 do
  assert(coroutine.resume(co))
  assert(debug.setlocal(co, 1, 1, round))
end
assert(coroutine.resume(co))
collectgarbage()
print("ok")
"#;
    for profile in ["lua51", "lua54"] {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, source);
            assert_eq!(stdout, "ok\n");
            assert_eq!(value(&line, "coroutine_root_visits"), 1, "{line}");
        }
    }
}

#[test]
fn a_collection_rebuilds_the_roots_of_the_stacks_that_moved_since_the_last_one() {
    let source = r#"
local threads = {}
for slot = 1, 8 do
  threads[slot] = coroutine.create(function() local mine = {slot}
    while true do coroutine.yield(mine) end end)
end
for round = 1, 50 do
  for slot = 1, 8 do assert(coroutine.resume(threads[slot])) end
  collectgarbage()
  -- Nothing moved since: the second collection snapshots nothing.
  collectgarbage()
end
print("ok")
"#;
    for profile in ["lua51", "lua54"] {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, source);
            assert_eq!(stdout, "ok\n");
            assert_eq!(value(&line, "coroutine_root_snapshots"), 8 * 50, "{line}");
        }
    }
}

#[test]
fn a_collection_visits_only_the_coroutines_that_yielded_since_the_last_one() {
    // 300 coroutines yield once and a collection refreshes them. Then only
    // three of them move between collections: the refresh looks at those
    // three, not at the 300 that stayed suspended and clean.
    let source = r#"
local threads = {}
for slot = 1, 300 do
  threads[slot] = coroutine.create(function() while true do coroutine.yield(slot) end end)
  assert(coroutine.resume(threads[slot]))
end
collectgarbage()
for round = 1, 20 do
  for slot = 1, 3 do assert(coroutine.resume(threads[slot * 100])) end
  collectgarbage()
  collectgarbage()
end
print("ok")
"#;
    for profile in ["lua51", "lua54"] {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, source);
            assert_eq!(stdout, "ok\n");
            let visits = value(&line, "coroutine_root_visits");
            assert_eq!(visits, 300 + 3 * 20, "{profile}/{mode}: {line}");
            assert_eq!(
                value(&line, "coroutine_root_snapshots"),
                visits,
                "{profile}/{mode}: every visit was a stale stack: {line}"
            );
        }
    }
}

#[test]
fn a_counted_run_leaves_no_directory_behind() {
    run_counted("lua54", "interpreter", "print(1)");
    let prefix = format!("coroutine-switch-{}-", std::process::id());
    let left: Vec<_> = std::fs::read_dir(env!("CARGO_TARGET_TMPDIR"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .filter(|entry| {
            // A directory under the prefix always holds its script while its
            // run is live, so an empty one is a leftover.
            std::fs::read_dir(entry.path()).is_ok_and(|mut files| files.next().is_none())
        })
        .collect();
    assert!(left.is_empty(), "directories left behind: {left:?}");
}

#[test]
fn calls_the_fast_loop_must_not_take_keep_the_general_path() {
    // A hook sees every call and a refused yield raises from the general
    // path. A wrapped coroutine called from the fast loop resumes directly.
    let source = r#"
local fired = 0
local co = coroutine.create(function() while true do coroutine.yield(1) end end)
debug.sethook(function() fired = fired + 1 end, "c")
for i = 1, 20 do coroutine.resume(co) end
debug.sethook()
assert(fired > 0)
local wrapped = coroutine.wrap(function() while true do coroutine.yield(1) end end)
for i = 1, 20 do wrapped() end
print(pcall(coroutine.yield, 1))
local ok, value = coroutine.resume(co, 1, 2, 3)
assert(ok and value == 1)
"#;
    for profile in ["lua51", "lua54"] {
        let (_, line) = run_counted(profile, "interpreter", source);
        // The hooked resumes and their yields and the main-thread yield
        // stayed off the direct path. The unhooked resume, the 20 wrapper
        // calls and the wrapped coroutine's 20 yields took it.
        assert_eq!(value(&line, "coroutine_direct_resumes"), 21, "{line}");
        assert_eq!(value(&line, "coroutine_direct_yields"), 21, "{line}");
    }
}

const WRAPPED_CALLS: &str = r"
local wrap, yield = coroutine.wrap, coroutine.yield
local step = wrap(function(first)
  local value, sum = first, 0
  while true do
    sum = sum + value
    value = yield(sum % 1009)
  end
end)
local total = 0
for i = 1, 3000 do total = total + step(i % 997) end
print(total)
";

#[test]
fn a_wrapper_called_in_a_loop_resumes_without_the_general_call_path() {
    let mut printed = None;
    for profile in PROFILES {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, WRAPPED_CALLS);
            assert_eq!(
                printed.get_or_insert_with(|| stdout.clone()),
                &stdout,
                "{profile}/{mode}"
            );
            // Every call but the ones a recording runs resumes from the
            // fast loop or from a trace, and every yield but those is
            // direct.
            let recordings = if mode == "tracing" { 2 } else { 0 };
            let resumes =
                value(&line, "coroutine_direct_resumes") + value(&line, "coroutine_trace_resumes");
            assert!(
                (3000 - recordings..=3000).contains(&resumes),
                "{profile}/{mode}: {line}"
            );
            assert!(
                (3000 - recordings..=3000).contains(&value(&line, "coroutine_direct_yields")),
                "{profile}/{mode}: {line}"
            );
            // The thread's loop is a sum of two registers, a remainder and
            // the yield: the short run takes it whole.
            if mode == "interpreter" {
                assert!(
                    value(&line, "coroutine_short_runs") >= 2999,
                    "{profile}/{mode}: {line}"
                );
            }
        }
    }
    assert_eq!(printed.as_deref(), Some("1498213\n"));
}

const GENERATORS: &str = r#"
local wrap, yield = coroutine.wrap, coroutine.yield
local function range(n)
  return wrap(function() for i = 1, n do yield(i) end end)
end
local function items(t)
  return wrap(function() for i = 1, #t do yield(t[i], i) end end)
end
local total = 0
for round = 1, 30 do
  for v in range(100) do total = total + v end
end
local list = {}
for i = 1, 100 do list[i] = i * 3 end
for round = 1, 30 do
  for v, i in items(list) do total = total + v * i end
end
print(total)
"#;

#[test]
fn a_generic_for_over_a_generator_resumes_directly_and_steps_in_the_short_run() {
    for profile in PROFILES {
        let (stdout, line) = run_counted(profile, "interpreter", GENERATORS);
        assert_eq!(stdout, "30603000\n", "{profile}");
        // 60 generators of 100 values each. A generator's first call starts
        // its thread, and its last finds the loop over: every call between
        // them resumes into the loop's latch and comes back from the yield
        // without leaving the short run.
        assert_eq!(
            value(&line, "coroutine_direct_resumes"),
            60 * 101,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_direct_yields"),
            60 * 100,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_short_runs"),
            60 * 99,
            "{profile}: {line}"
        );
    }
}

#[test]
fn reads_and_loops_the_short_run_does_not_know_leave_it_with_the_same_values() {
    // A hole answered by `__index`, a string key, a float key that is no
    // array index and a holder that is no table are reads the short run
    // leaves to the fast loop. So are the ends of the loops: a count down, a
    // fractional step and, with integers, the edge of the integer domain.
    let source = r#"
local wrap, yield = coroutine.wrap, coroutine.yield
local function collect(generator)
  local seen = {}
  for value in generator do seen[#seen + 1] = tostring(value) end
  return table.concat(seen, " ")
end
local holes = setmetatable({ 1, 2, nil, 4 }, { __index = function(_, k) return -k end })
print(collect(wrap(function() for i = 1, 5 do yield(holes[i]) end end)))
local keyed = { a = "A", [1.5] = "F", [1] = "one" }
local keys = { "a", 1.5, 1 }
print(collect(wrap(function() for i = 1, #keys do yield(keyed[keys[i]]) end end)))
local text = "xyz"
print(collect(wrap(function() for i = 1, 2 do yield(text[i] == nil) end end)))
print(collect(wrap(function() for i = 5, 1, -2 do yield(i) end end)))
print(collect(wrap(function() for x = 0.5, 2, 0.5 do yield(x) end end)))
local top = math.maxinteger
if top then
  print(collect(wrap(function() for i = top - 2, top do yield(i - top) end end)))
  print(collect(wrap(function() for i = -top, -top - 1, -1 do yield(i + top) end end)))
end
"#;
    let mut printed: Vec<(String, String)> = Vec::new();
    for profile in PROFILES {
        let (interpreted, line) = run_counted(profile, "interpreter", source);
        let (traced, _) = run_counted(profile, "tracing", source);
        assert_eq!(interpreted, traced, "{profile}");
        assert!(
            value(&line, "coroutine_short_runs") > 0,
            "{profile}: {line}"
        );
        printed.push((profile.to_string(), interpreted));
    }
    let lines = |profile: &str| -> Vec<String> {
        printed
            .iter()
            .find(|(name, _)| name == profile)
            .map(|(_, text)| text.lines().map(str::to_string).collect())
            .unwrap()
    };
    let lua51 = lines("lua51");
    assert_eq!(lua51[0], "1 2 -3 4 -5");
    assert_eq!(lua51[1], "A F one");
    assert_eq!(lua51[2], "true true");
    assert_eq!(lua51[3], "5 3 1");
    assert_eq!(lua51[4], "0.5 1 1.5 2");
    let lua54 = lines("lua54");
    assert_eq!(lua54[..4], lua51[..4]);
    assert_eq!(lua54[4], "0.5 1.0 1.5 2.0");
    assert_eq!(lua54[5], "-2 -1 0");
    assert_eq!(lua54[6], "0 -1");
}

#[test]
fn a_wrapper_whose_thread_fails_under_the_direct_resume_raises_as_before() {
    // The thread has run when the fast loop learns it failed, so the general
    // path reports the failure from the answer the fast loop kept: the
    // position of the caller before a string, the value itself for anything
    // else, and a dead thread afterwards. Lua 5.4 closes the thread before
    // it raises.
    let source = r#"
local wrap, yield = coroutine.wrap, coroutine.yield
local function call(f) local value = f() return value end
for _, failure in ipairs({ "text", { 1 }, false }) do
  local step = wrap(function() yield(1); yield(2); error(failure) end)
  local seen = {}
  for i = 1, 5 do
    local ok, value = pcall(call, step)
    seen[#seen + 1] = tostring(ok) .. ":" .. (type(value) == "string" and value:gsub("[^ ]*:%d+: ", "@ ") or type(value))
  end
  print(table.concat(seen, " "))
end
local closed = 0
local guarded = wrap(function()
  for i = 1, 3 do yield(i) end
  error("late")
end)
if _VERSION >= "Lua 5.4" then
  guarded = load([[
    local wrap, yield, count = ...
    return wrap(function()
      local resource <close> = setmetatable({}, { __close = function() count() end })
      for i = 1, 3 do yield(i) end
      error("late")
    end)
  ]])(wrap, yield, function() closed = closed + 1 end)
end
local sum = 0
for i = 1, 6 do
  local ok, value = pcall(call, guarded)
  if ok then sum = sum + value end
end
print(sum, closed)
"#;
    for profile in PROFILES {
        let (interpreted, line) = run_counted(profile, "interpreter", source);
        let (traced, _) = run_counted(profile, "tracing", source);
        assert_eq!(interpreted, traced, "{profile}");
        let lines: Vec<&str> = interpreted.lines().collect();
        assert_eq!(
            lines[0],
            "true:number true:number false:@ @ text false:@ cannot resume dead coroutine \
             false:@ cannot resume dead coroutine",
            "{profile}"
        );
        assert_eq!(
            lines[1],
            "true:number true:number false:table false:@ cannot resume dead coroutine \
             false:@ cannot resume dead coroutine",
            "{profile}"
        );
        assert_eq!(
            lines[2],
            "true:number true:number false:boolean false:@ cannot resume dead coroutine \
             false:@ cannot resume dead coroutine",
            "{profile}"
        );
        let closing = matches!(profile, "lua54" | "lua55");
        assert_eq!(lines[3], if closing { "6\t1" } else { "6\t0" }, "{profile}");
        // The failures came through the direct resume and were replayed.
        assert!(
            value(&line, "coroutine_trace_replays") >= 3,
            "{profile}: {line}"
        );
    }
}

#[test]
fn dropped_suspended_coroutines_and_their_stacks_are_collected() {
    let source = r#"
local weak = setmetatable({}, {__mode = "v"})
collectgarbage(); collectgarbage()
local before = collectgarbage("count")
for i = 1, 100000 do
  local co = coroutine.create(function()
    local held = {i, {i}}
    if i % 1000 == 0 then weak[#weak + 1] = held end
    coroutine.yield(held)
  end)
  coroutine.resume(co)
end
collectgarbage(); collectgarbage()
local grown = collectgarbage("count") - before
local live = 0
for _ in pairs(weak) do live = live + 1 end
print(grown < 2048, live)
"#;
    for (profile, mode) in [
        ("lua51", "interpreter"),
        ("lua54", "interpreter"),
        ("lua54", "tracing"),
    ] {
        let (stdout, _) = run_counted(profile, mode, source);
        assert_eq!(stdout, "true\t0\n", "{profile}/{mode}");
    }
}

#[cfg(feature = "alloc-stats")]
#[test]
fn a_resume_and_yield_pair_allocates_nothing_after_warmup() {
    let allocations = |profile: &str, mode: &str, rounds: u32| {
        let source = PING_PONG.replace("tonumber(arg and arg[1]) or 3000", &rounds.to_string());
        let (_, line) = run_counted(profile, mode, &source);
        value(&line, "alloc_count")
    };
    for profile in PROFILES {
        for mode in MODES {
            // Both runs are long enough for the tracing mode to compile the
            // driver loop, whose compile allocates once and is no part of
            // what a round trip costs.
            let short = allocations(profile, mode, 11_000);
            let long = allocations(profile, mode, 31_000);
            assert!(
                long <= short + 4,
                "{profile}/{mode}: 20000 more round trips allocated {} times",
                long.saturating_sub(short)
            );
        }
    }
}

#[test]
fn threads_that_run_long_between_yields_match_stock_when_resumed_in_place() {
    // Branches, globals, calls, errors, dead threads, hooks, collections
    // and nested resumes in threads the fast loop resumes in place.
    check("coroutine_in_place.lua", "coroutine_in_place", &PROFILES);
}

const LONG_RUNS: &str = r"
local create, resume = coroutine.create, coroutine.resume
local cfg = { bias = 7 }
local function worker(value)
  while true do
    if value > 500 then value = value - cfg.bias else value = value + 1 end
    value = coroutine.yield((value * 3 + cfg.bias) % 997)
  end
end
local threads, total = {}, 0
for slot = 1, 8 do
  threads[slot] = create(worker)
  local ok, value = resume(threads[slot], slot)
  total = total + value
end
for i = 1, 3000 do
  local ok, value = resume(threads[(i - 1) % 8 + 1], i % 997)
  total = total + value
end
print(total)
";

const LONG_RUNS_TOTAL: &str = "1487120\n";

#[test]
fn a_thread_with_a_branch_and_a_field_read_is_resumed_in_place() {
    for profile in PROFILES {
        let (stdout, line) = run_counted(profile, "interpreter", LONG_RUNS);
        assert_eq!(stdout, LONG_RUNS_TOTAL, "{profile}");
        // The worker's loop compares, reads a field and reads a global,
        // none of which the short run knows. Each thread's first resume
        // after its start finds that out. Every one after it runs the
        // thread in the fast loop that resumed it, and every yield goes
        // back to that loop without leaving it.
        let in_place = value(&line, "coroutine_in_place_resumes");
        assert!((3000 - 8..=3000).contains(&in_place), "{profile}: {line}");
        assert_eq!(
            value(&line, "coroutine_in_place_exits"),
            0,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_direct_resumes"),
            3008,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_direct_yields"),
            3008,
            "{profile}: {line}"
        );
    }
}

#[test]
fn no_resume_is_made_in_place_under_a_hook_or_for_a_short_run() {
    // A count hook watches every instruction, so no thread runs in place
    // under it. The loop of `PING_PONG` is a few instructions around its
    // yield, which the short run takes whole.
    let hooked = format!("debug.sethook(function() end, '', 1000000)\n{LONG_RUNS}");
    for profile in ["lua51", "lua54"] {
        let (stdout, line) = run_counted(profile, "interpreter", &hooked);
        assert_eq!(stdout, LONG_RUNS_TOTAL, "{profile}");
        assert_eq!(
            value(&line, "coroutine_in_place_resumes"),
            0,
            "{profile}: {line}"
        );
        let (stdout, line) = run_counted(profile, "interpreter", PING_PONG);
        assert_eq!(stdout, PING_PONG_TOTAL, "{profile}");
        assert_eq!(
            value(&line, "coroutine_in_place_resumes"),
            0,
            "{profile}: {line}"
        );
        assert!(
            value(&line, "coroutine_short_runs") >= 3000,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_thread_that_leaves_the_fast_loop_in_place_goes_on_as_a_nested_resume() {
    // Every second resume the thread installs a call hook and calls a
    // library function, which the fast loop leaves to the dispatch loop.
    // The resume in place turns into a nested one and still hands back
    // what the thread yields. The thread's error and the resume of the
    // dead thread come back the same way.
    let source = r#"
local co = coroutine.create(function(n)
  local sum = 0
  for i = 1, n do
    if i % 2 == 0 then
      debug.sethook(function() end, "c")
      sum = sum + #string.rep("x", i)
      debug.sethook()
    end
    sum = sum + coroutine.yield(sum)
  end
  if sum > 100 then error("too much: " .. sum, 0) end
  return "done", sum
end)
local out = {}
for i = 1, 12 do
  local ok, a, b = coroutine.resume(co, 10, i)
  out[#out + 1] = tostring(ok) .. ":" .. tostring(a) .. ":" .. tostring(b)
end
print(table.concat(out, " "))
"#;
    let expected = "true:0:nil true:12:nil true:22:nil true:36:nil true:46:nil true:62:nil \
                    true:72:nil true:90:nil true:100:nil true:120:nil false:too much: 130:nil \
                    false:cannot resume dead coroutine:nil\n";
    for profile in PROFILES {
        for mode in MODES {
            let (stdout, line) = run_counted(profile, mode, source);
            assert_eq!(stdout, expected, "{profile}/{mode}");
            if mode == "interpreter" {
                assert!(
                    value(&line, "coroutine_in_place_exits") >= 4,
                    "{profile}: {line}"
                );
            }
        }
    }
}

#[test]
fn an_assert_answered_in_the_call_window_matches_stock() {
    // The fast loop answers `assert` of a truthy value in the registers of
    // its call, which is what a resume loop checks every round. A false or
    // missing value, a message of any type, a call hook and a replaced
    // global all leave that path.
    check("assert_in_window.lua", "assert_in_window", &PROFILES);
}
