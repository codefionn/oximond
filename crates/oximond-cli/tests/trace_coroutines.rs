//! Loops that resume coroutines trace instead of aborting.
//!
//! A driver loop that resumes the threads of an array records: the array read
//! gives a thread lane, the call of `coroutine.resume` is a helper that runs
//! the thread through the VM's own switch from inside the native code, and the
//! trace goes on with the values the thread handed back. The counters in the
//! `OXIMOND_COUNTERS` line pin which path ran (a compiled trace, native
//! iterations, the resumes the trace made itself, and the calls the
//! interpreter replayed from what a trace's resume had already given). A
//! resume the trace cannot take (an error, a dead thread, other values than it
//! recorded) leaves before the call and the interpreter makes it without
//! running the thread again, so every script prints what the interpreter
//! prints.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

const PROFILES: [&str; 4] = ["lua51", "lua54", "lua55", "luajit21"];

/// Writes `source` to a script of its own and returns its path.
fn script(source: &str) -> PathBuf {
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("trace-coroutines-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let script = directory.join(format!("script_{id}.lua"));
    std::fs::File::create(&script)
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    script
}

/// Runs the script with counters on and returns stdout and the counters line.
fn run_script(profile: &str, mode: &str, script: &std::path::Path) -> (String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", profile, "--execution", mode])
        .arg(script)
        .env("OXIMOND_COUNTERS", "1")
        .env("OXIMOND_JIT_SYNC", "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "{profile}/{mode}: {output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let line = stderr
        .lines()
        .find(|line| line.starts_with("oximond-counters:"))
        .unwrap_or_else(|| panic!("no counters line in {stderr}"))
        .to_string();
    (String::from_utf8(output.stdout).unwrap(), line)
}

/// Runs `source` with counters on and returns stdout and the counters line.
fn run(profile: &str, mode: &str, source: &str) -> (String, String) {
    let path = script(source);
    let result = run_script(profile, mode, &path);
    let _ = std::fs::remove_file(&path);
    result
}

fn value(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from {line}"))
        .parse()
        .unwrap()
}

/// A counter that is only printed when it is not zero.
fn value_or_zero(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(&format!("{key}=")))
        .map_or(0, |count| count.parse().unwrap())
}

/// Runs `source` in both modes, requires the same output, and returns the
/// counters of the tracing run.
fn agree(profile: &str, source: &str) -> (String, String) {
    let path = script(source);
    let (expected, _) = run_script(profile, "interpreter", &path);
    let (tracing, line) = run_script(profile, "tracing", &path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(tracing, expected, "{profile}\n{source}\n{line}");
    (tracing, line)
}

const PING_PONG: &str = r"
local rounds = 3000
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
  local thread = threads[(i - 1) % 8 + 1]
  local ok, value = resume(thread, i % 997)
  assert(ok)
  total = total + value
end
print(total)
";

const PING_PONG_TOTAL: &str = "1489880\n";

#[test]
fn a_resume_loop_records_and_runs_natively_through_the_helper() {
    for profile in PROFILES {
        let (stdout, line) = run(profile, "tracing", PING_PONG);
        assert_eq!(stdout, PING_PONG_TOTAL, "{profile}");
        // The driver loop records (the array read of a thread and the
        // native-function upvalue-free resume are both typed) and most of its
        // 3000 iterations run in the one native entry.
        assert!(value(&line, "traces_recorded") >= 1, "{profile}: {line}");
        assert!(value(&line, "iterations") >= 2900, "{profile}: {line}");
        assert!(value(&line, "native_entries") <= 4, "{profile}: {line}");
        // Each of those iterations resumed through the helper, not through
        // the fast loop's resume, and nothing needed the interpreter's
        // replay of a call.
        assert!(
            value(&line, "coroutine_trace_resumes") >= 2900,
            "{profile}: {line}"
        );
        // The 8 first resumes and the 3000 of the loop, but the iteration
        // that recorded: a recording runs its resume through the general
        // path, which counts neither.
        assert_eq!(
            value(&line, "coroutine_direct_resumes") + value(&line, "coroutine_trace_resumes"),
            3007,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_trace_replays"),
            0,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_root_snapshots"),
            0,
            "{profile}: {line}"
        );
        // Neither the thread array read nor the resume aborts a recording.
        assert_eq!(
            value_or_zero(&line, "abort.UnsupportedOperand"),
            0,
            "{profile}: {line}"
        );
        assert_eq!(
            value_or_zero(&line, "abort.Divergence"),
            0,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_loop_that_yields_every_iteration_is_rejected_by_its_first_recording() {
    // The worker's loop calls `yield` unconditionally: the thread leaves it
    // in the middle of each iteration, which no trace follows. One recording
    // finds that, and the interpreter runs the loop with no further attempts.
    for profile in PROFILES {
        let (stdout, line) = run(profile, "tracing", PING_PONG);
        assert_eq!(stdout, PING_PONG_TOTAL, "{profile}");
        assert_eq!(value(&line, "abort.YieldingLoop"), 1, "{profile}: {line}");
        assert_eq!(value_or_zero(&line, "abort.Yield"), 0, "{profile}: {line}");
        // The worker's loop and the driver's: two recordings, no retries.
        assert_eq!(value(&line, "recording_starts"), 2, "{profile}: {line}");
    }
}

#[test]
fn a_loop_that_yields_now_and_then_is_not_given_up_on() {
    // Only some iterations reach the yield, so a recording of another one
    // closes and the loop runs natively between the yields.
    let source = r#"
local co = coroutine.wrap(function()
  local sum = 0
  for i = 1, 20000 do
    sum = sum + i % 7
    if i % 5000 == 0 then coroutine.yield(sum) end
  end
  return sum
end)
local parts = {}
for _ = 1, 5 do parts[#parts + 1] = co() end
print(table.concat(parts, " "))
"#;
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_eq!(
            value_or_zero(&line, "abort.YieldingLoop"),
            0,
            "{profile}: {line}"
        );
        assert!(value(&line, "traces_recorded") >= 1, "{profile}: {line}");
        assert!(value(&line, "iterations") > 10_000, "{profile}: {line}");
    }
}

/// A script's driver loop must trace and use the helper: the counters say so.
fn assert_resumed_natively(profile: &str, line: &str, least: u64) {
    assert!(value(line, "traces_recorded") >= 1, "{profile}: {line}");
    assert!(
        value(line, "coroutine_trace_resumes") >= least,
        "{profile}: {line}"
    );
}

#[test]
fn a_coroutine_that_errors_in_the_middle_of_the_loop_answers_false_and_the_message() {
    // Thread 3 raises at its 700th resume. The trace's resume gets `false`
    // and the message, cannot take them, and leaves before the call; the
    // interpreter's replay answers the call without running the thread again
    // (a second run would find it dead and answer another message).
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local threads = {}
for slot = 1, 4 do
  threads[slot] = create(function(first)
    local n = 0
    while true do
      n = n + 1
      if slot == 3 and n == 700 then error("boom " .. n) end
      first = yield(first + n)
    end
  end)
end
local total, fails, msgs = 0, 0, {}
for i = 1, 4000 do
  local ok, value = resume(threads[(i - 1) % 4 + 1], i % 97)
  if ok then total = total + value
  else
    fails = fails + 1
    if fails < 4 then msgs[#msgs + 1] = tostring(value) end
  end
end
print(total, fails, table.concat(msgs, "|"))
"#;
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(
            stdout.contains("boom 700|cannot resume dead coroutine"),
            "{stdout}"
        );
        assert_resumed_natively(profile, &line, 2000);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_table_a_coroutine_raises_survives_collections_around_the_replay() {
    // The error object is a fresh table that nothing but the answer of the
    // trace's resume holds until the interpreter's replay hands it on. A
    // weak table watches the objects, and collections run before and after
    // the failure; the output must be what the interpreter prints. Lua has
    // no safepoint between the trace's exit and the replay, so no collection
    // lands in that window here: the unit test
    // `the_values_a_remembered_resume_holds_survive_a_collection_before_the_replay`
    // pins the rooting there.
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local weak = setmetatable({}, {__mode = "v"})
local threads = {}
for slot = 1, 4 do
  threads[slot] = create(function(first)
    local n = 0
    while true do
      n = n + 1
      if slot == 2 and n == 600 then
        local object = {n = n, name = "fresh" .. n}
        weak[1] = object
        error(object)
      end
      first = yield(first + n)
    end
  end)
end
local total, caught = 0, nil
for i = 1, 4000 do
  local ok, value = resume(threads[(i - 1) % 4 + 1], i % 97)
  if ok then total = total + value
  elseif type(value) == "table" and not caught then
    caught = value
    collectgarbage()
  end
  if i % 500 == 0 then collectgarbage() end
end
collectgarbage()
print(total, caught and caught.n, caught and caught.name, weak[1] == caught)
"#;
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(
            stdout.contains("\t600\tfresh600\ttrue"),
            "{profile}: {stdout}"
        );
        assert_resumed_natively(profile, &line, 1000);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_coroutine_that_finishes_in_the_middle_of_the_loop_is_dead_for_the_rest() {
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local threads = {}
for slot = 1, 4 do
  threads[slot] = create(function(first)
    for n = 1, 300 do first = yield(first + n) end
    return 99
  end)
end
local total, dead, msg, last = 0, 0, "", 0
for i = 1, 4000 do
  local ok, value = resume(threads[(i - 1) % 4 + 1], i % 97)
  if ok then total = total + value; last = value
  else dead = dead + 1; msg = value end
end
print(total, dead, msg, last)
"#;
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(stdout.contains("cannot resume dead coroutine"), "{stdout}");
        assert_resumed_natively(profile, &line, 800);
        // The return of a finished thread (`true, 99`) is a value of the
        // recorded kind, and every resume after it is a replay.
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_coroutine_that_yields_another_number_of_values_is_replayed() {
    // Two values for 600 resumes, then one, then none, then a float: each
    // change leaves the trace before the call, and the replay hands the
    // interpreter what the thread gave.
    let source = r"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local threads = {}
for slot = 1, 2 do
  threads[slot] = create(function(first)
    local n = 0
    while true do
      n = n + 1
      if n < 300 then first = yield(first + n, n)
      elseif n < 450 then first = yield(first + n)
      elseif n < 550 then first = yield()
      else first = yield(first * 0.5, n) end
    end
  end)
end
local ok_count, b, c, nils = 0, 0, 0, 0
for i = 1, 4000 do
  local ok, v, w = resume(threads[(i - 1) % 2 + 1], i % 13)
  ok_count = ok_count + (ok and 1 or 0)
  if v == nil then nils = nils + 1 else b = b + v end
  if w ~= nil then c = c + w end
end
print(ok_count, b, c, nils)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_resumed_natively(profile, &line, 400);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_thread_slot_replaced_by_a_non_thread_raises_the_interpreters_error() {
    // The array read guards the slot's type: a number leaves before the read,
    // and the interpreter resumes it. A thread replaces slot 2 from inside
    // another one, through the array it captured, which the driver (a local
    // the closures capture, so the trace reads it again after each resume)
    // reads on.
    let source = r"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local driver = create(function()
  local threads = {}
  for slot = 1, 4 do
    threads[slot] = create(function(first)
      local n = 0
      while true do
        n = n + 1
        if slot == 1 and n == 500 then threads[2] = 42 end
        first = yield(first + n)
      end
    end)
  end
  local total = 0
  for i = 1, 3000 do
    local ok, value = resume(threads[(i - 1) % 4 + 1], i)
    total = total + value
  end
  return total
end)
print(resume(driver))
";
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(stdout.starts_with("false\t"), "{stdout}");
        assert!(stdout.contains("resume"), "{stdout}");
        assert_resumed_natively(profile, &line, 1000);
    }
}

#[test]
fn a_function_from_coroutine_wrap_records_as_a_call_of_a_helper() {
    // The wrapper is a callable lane of the run's bank, and the helper checks
    // each call that it is one. The 5.4 profiles close a coroutine that fails
    // before the error is raised, which the replay of the failed call does.
    let source = r#"
local function counter(limit)
  return coroutine.wrap(function(first)
    local n = 0
    while true do
      n = n + 1
      if n == limit then error("wrap boom " .. n, 0) end
      first = coroutine.yield(first + n)
    end
  end)
end
local function drive(gen, count)
  local sum = 0
  for i = 1, count do sum = sum + gen(i % 97) end
  return sum
end
print(coroutine.resume(coroutine.create(drive), counter(100000), 3000))
print(coroutine.resume(coroutine.create(drive), counter(2500), 3000))
local again = coroutine.wrap(function() return 1, 2 end)
print(again(), pcall(again))
"#;
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(stdout.contains("wrap boom 2500"), "{stdout}");
        assert!(stdout.contains("cannot resume dead coroutine"), "{stdout}");
        assert_resumed_natively(profile, &line, 2000);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_function_from_coroutine_wrap_called_in_a_loop_answers_and_raises_alike() {
    let source = r#"
local gens = {}
for slot = 1, 4 do
  gens[slot] = coroutine.wrap(function(first)
    local n = 0
    while true do
      n = n + 1
      if slot == 2 and n == 600 then error("wrapped boom " .. n, 0) end
      first = coroutine.yield(first + n)
    end
  end)
end
local total = 0
local function drive(from)
  for i = from, 3000 do total = total + gens[(i - 1) % 4 + 1](i % 97) end
end
local ok, err = pcall(drive, 1)
print(ok, err, total)
local ok2, err2 = pcall(drive, 1)
print(ok2, err2, total)
"#;
    for profile in PROFILES {
        let (stdout, _) = agree(profile, source);
        assert!(stdout.contains("wrapped boom 600"), "{stdout}");
        assert!(stdout.contains("cannot resume dead coroutine"), "{stdout}");
    }
}

#[test]
fn debug_hooks_firing_in_the_resumed_coroutine_see_every_line_the_interpreter_shows() {
    // A line hook on one thread is installed from inside it while the driver
    // is traced. From then on the trace's resume declines (a hook sees every
    // line, which native code does not show) and the interpreter resumes.
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local lines = 0
local threads = {}
for slot = 1, 3 do
  threads[slot] = create(function(first)
    local n = 0
    while true do
      n = n + 1
      if slot == 2 and n == 400 then
        debug.sethook(function() lines = lines + 1 end, "l")
      end
      first = yield(first + n)
    end
  end)
end
local total = 0
for i = 1, 3000 do
  local ok, value = resume(threads[(i - 1) % 3 + 1], i % 11)
  total = total + value
end
print(total, lines > 0, lines)
"#;
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert!(value(&line, "traces_recorded") >= 1, "{profile}: {line}");
        assert!(
            value(&line, "coroutine_trace_resumes") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_count_hook_on_the_driver_runs_the_loop_in_the_interpreter() {
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local threads = {}
for slot = 1, 3 do
  threads[slot] = create(function(first) while true do first = yield(first + 1) end end)
end
local fired, total = 0, 0
for i = 1, 3000 do
  if i == 1500 then debug.sethook(function() fired = fired + 1 end, "", 100) end
  local ok, value = resume(threads[(i - 1) % 3 + 1], i % 11)
  total = total + value
end
debug.sethook()
print(total, fired > 0, fired)
"#;
    for profile in PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_collection_inside_the_resumed_coroutine_keeps_what_the_trace_and_the_stacks_hold() {
    // The coroutines allocate tables and collect in the middle of a resume
    // that a native trace made: the trace's lanes (the thread array and the
    // threads), the driver's stack and the weak table must come out as the
    // interpreter's.
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local weak = setmetatable({}, {__mode = "v"})
local threads = {}
for slot = 1, 4 do
  threads[slot] = create(function(first)
    local mine = {slot}
    local n = 0
    while true do
      n = n + 1
      local scratch = {n, {n}}
      weak[slot * 1000 + n % 50] = scratch
      if n % 40 == 0 then collectgarbage() end
      mine[1] = first
      first = yield(first + n + #scratch)
    end
  end)
end
local keep = {1, 2, 3}
local total = 0
for i = 1, 4000 do
  local ok, value = resume(threads[(i - 1) % 4 + 1], i % 97)
  total = total + value
end
collectgarbage(); collectgarbage()
local live = 0
for _ in pairs(weak) do live = live + 1 end
print(total, #keep, live <= 200, collectgarbage("count") < 20000)
"#;
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert!(stdout.ends_with("true\ttrue\n"), "{stdout}");
        assert_resumed_natively(profile, &line, 2000);
    }
}

#[test]
fn a_global_the_thread_rebinds_is_read_again_after_the_resume() {
    let source = r"
function bump(x) return x + 1 end
local co = coroutine.create(function()
  local n = 0
  while true do
    n = n + 1
    if n == 700 then function bump(x) return x + 1000 end end
    coroutine.yield(n)
  end
end)
local resume = coroutine.resume
local total = 0
for i = 1, 3000 do
  local ok, n = resume(co)
  total = total + bump(n)
end
print(total)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        // The rebinding is seen by the check after the resume, which leaves
        // before the call and has the interpreter replay it.
        assert_resumed_natively(profile, &line, 300);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_captured_local_the_thread_assigns_is_read_again_after_the_resume() {
    // `scale` is a local the thread's closure captured and assigns at its
    // 900th resume. The driver only reads it, so the trace runs, checks the
    // register after each resume and leaves when it changed.
    let source = r"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local scale = 3
local co = create(function()
  local n = 0
  while true do
    n = n + 1
    if n == 900 then scale = 5 end
    yield(n)
  end
end)
local total = 0
for i = 1, 3000 do
  local ok, n = resume(co)
  total = total + n * scale
end
print(total)
";
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert_eq!(stdout, "21698400\n", "{profile}");
        assert_resumed_natively(profile, &line, 1000);
        assert!(
            value(&line, "coroutine_trace_replays") >= 1,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_captured_local_the_loop_writes_keeps_the_loop_in_the_interpreter() {
    // The trace would hold `shared` in a native value that the thread's
    // store never reaches, so no trace calls the resume.
    let source = r"
local shared = 0
local co = coroutine.create(function()
  while true do shared = shared + 100; coroutine.yield() end
end)
local resume = coroutine.resume
for i = 1, 3000 do
  shared = shared + 1
  resume(co)
end
print(shared)
";
    for profile in PROFILES {
        let (stdout, line) = agree(profile, source);
        assert_eq!(stdout, "303000\n", "{profile}");
        assert_eq!(
            value(&line, "coroutine_trace_resumes"),
            0,
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_resume_read_from_the_library_table_or_an_upvalue_records_like_a_local() {
    // `coroutine.resume` found through the library table (a callee the entry
    // pins, which the helper checks again after each resume) and `resume`
    // held in an upvalue of the loop's function (a native-function cell).
    let by_table = r"
local threads = {}
for slot = 1, 4 do
  threads[slot] = coroutine.create(function(first)
    while true do first = coroutine.yield(first + slot) end
  end)
end
local total = 0
for i = 1, 3000 do
  local ok, value = coroutine.resume(threads[(i - 1) % 4 + 1], i % 31)
  total = total + value
end
print(total)
";
    let by_upvalue = r"
local resume = coroutine.resume
local threads = {}
for slot = 1, 4 do
  threads[slot] = coroutine.create(function(first)
    while true do first = coroutine.yield(first + slot) end
  end)
end
local function drive(count)
  local total = 0
  for i = 1, count do
    local ok, value = resume(threads[(i - 1) % 4 + 1], i % 31)
    total = total + value
  end
  return total
end
print(drive(3000))
";
    for source in [by_table, by_upvalue] {
        for profile in PROFILES {
            let (_, line) = agree(profile, source);
            assert_resumed_natively(profile, &line, 2500);
            assert_eq!(
                value_or_zero(&line, "abort.UnsupportedOperand"),
                0,
                "{profile}: {line}"
            );
        }
    }
}

#[test]
fn a_table_the_thread_grows_is_read_through_fresh_bindings_after_the_resume() {
    // The thread appends to the array the driver reads: the array words the
    // trace holds are rewritten from the table after each resume.
    let source = r"
local data = {1, 2, 3, 4}
local co = coroutine.create(function()
  local n = 0
  while true do
    n = n + 1
    data[#data + 1] = n
    coroutine.yield(n)
  end
end)
local resume = coroutine.resume
local total = 0
for i = 1, 3000 do
  local ok, n = resume(co)
  total = total + data[#data % 4 + 1] + n
end
print(total, #data)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_resumed_natively(profile, &line, 2500);
    }
}

#[test]
fn a_thread_that_writes_nothing_is_resumed_without_reading_the_run_again() {
    // The worker's loop is a few instructions around the yield: its run
    // writes no table, cell or global, so the trace has nothing to read
    // again after the resume. Both counters pin the short run, and the
    // total pins that it hands back what the fast loop would.
    for profile in PROFILES {
        let (stdout, line) = run(profile, "tracing", PING_PONG);
        assert_eq!(stdout, PING_PONG_TOTAL, "{profile}");
        assert!(
            value(&line, "coroutine_short_runs") >= 2900,
            "{profile}: {line}"
        );
        assert!(
            value(&line, "coroutine_trace_unchecked") >= 2900,
            "{profile}: {line}"
        );
        assert_eq!(
            value(&line, "coroutine_trace_unchecked"),
            value(&line, "coroutine_trace_resumes"),
            "{profile}: {line}"
        );
    }
}

#[test]
fn a_thread_that_writes_is_read_again_after_every_resume() {
    // Writing a global, a captured local and an array each resume takes the
    // thread out of the short run: the trace reads all three again, and the
    // driver sees every write.
    let source = r"
counter = 0
local shared = 0
local log = {}
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local co = create(function()
  while true do
    counter = counter + 1
    shared = shared + 2
    log[#log + 1] = counter
    yield(shared)
  end
end)
local total = 0
for i = 1, 3000 do
  local ok, n = resume(co)
  total = total + n + shared + #log
end
print(total, counter, shared, #log)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_eq!(
            value(&line, "coroutine_trace_unchecked"),
            0,
            "{profile}: {line}"
        );
        assert_eq!(value(&line, "coroutine_short_runs"), 0, "{profile}: {line}");
    }
}

#[test]
fn a_thread_that_writes_after_many_that_do_not_is_read_again_when_it_does() {
    // The driver resumes a thread that only counts for its first 1000
    // iterations and then one that writes a captured local and a global the
    // driver reads. The first thread's resumes are taken unread, and the
    // driver sees the new values from the second thread's first resume on.
    let source = r"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local scale = 3
function bump(x) return x + 1 end
local threads = {
  create(function() local n = 0; while true do n = n + 1; yield(n) end end),
  create(function()
    local n = 0
    while true do
      n = n + 1
      scale = 5
      function bump(x) return x + 100 end
      yield(n)
    end
  end),
}
local total = 0
for i = 1, 3000 do
  local ok, n = resume(threads[i <= 1000 and 1 or 2])
  total = total + n * scale + bump(n)
end
print(total, scale)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_resumed_natively(profile, &line, 900);
        assert!(
            value(&line, "coroutine_trace_unchecked") >= 900,
            "{profile}: the counting thread's resumes were read again: {line}"
        );
    }
}

#[test]
fn a_thread_with_instructions_outside_the_short_run_is_read_again_and_agrees() {
    // A call, string arithmetic and a metamethod each take the thread out of
    // the short run, to the fast loop, which may call anything. A read of
    // the table's array part does not: the short run makes it, and writes
    // nothing.
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local data = {10, 20, 30}
local function twice(x) return x * 2 end
local meta = setmetatable({}, {__add = function(a, b) data[1] = data[1] + 1; return b end})
local threads = {
  create(function(a) while true do a = yield(data[a % 3 + 1]) end end),
  create(function(a) while true do a = yield(twice(a) % 7) end end),
  create(function(a) while true do a = yield(("5" + a) % 11) end end),
  create(function(a) while true do a = yield(meta + a) end end),
}
local total = 0
for i = 1, 3000 do
  local ok, n = resume(threads[i % 4 + 1], i)
  total = total + n + data[1]
end
print(total)
"#;
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        // Four threads, each resumed 750 times. Only the first, which reads
        // the array, runs as a short run: the driver trace reads its inputs
        // again after every resume of the other three.
        let unchecked = value(&line, "coroutine_trace_unchecked");
        assert!((700..=750).contains(&unchecked), "{profile}: {line}");
    }
}

#[cfg(feature = "alloc-stats")]
#[test]
fn a_resume_from_a_trace_allocates_nothing_after_warmup() {
    let allocations = |profile: &str, rounds: u32| {
        let source = PING_PONG.replace("local rounds = 3000", &format!("local rounds = {rounds}"));
        let (_, line) = run(profile, "tracing", &source);
        assert!(
            value(&line, "coroutine_trace_resumes") >= u64::from(rounds) - 100,
            "{line}"
        );
        value(&line, "alloc_count")
    };
    for profile in PROFILES {
        // Both runs compile the driver loop, which allocates the same.
        let short = allocations(profile, 11_000);
        let long = allocations(profile, 31_000);
        assert!(
            long <= short + 4,
            "{profile}: 20000 more round trips allocated {} times",
            long.saturating_sub(short)
        );
    }
}

#[test]
fn a_thread_that_left_the_short_run_once_takes_it_again() {
    // Each worker makes a call after its first yield, which the short run
    // does not know: the thread counts as one that runs long. A later resume
    // of the trace tries the short run again and finds a thread that yields
    // after a few instructions, so nearly every resume of the loop is a
    // short run.
    let source = r"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local config = { bias = 7 }
local function worker(first)
    local value = yield(first)
    value = value + math.abs(config.bias) - 7
    while true do
        value = yield((value * 3 + 7) % 997)
    end
end
local threads = {}
local total = 0
for slot = 1, 4 do
    threads[slot] = create(worker)
    local ok, value = resume(threads[slot], slot)
    assert(ok)
    total = total + value
end
for i = 1, 6000 do
    local ok, value = resume(threads[(i - 1) % 4 + 1], i % 997)
    assert(ok)
    total = total + value
end
print(total)
";
    for profile in PROFILES {
        let (_, line) = agree(profile, source);
        assert_resumed_natively(profile, &line, 5000);
        assert!(
            value(&line, "coroutine_short_runs") >= 5000,
            "{profile}: {line}"
        );
    }
}

#[test]
fn threads_taken_to_run_long_error_end_and_change_what_they_yield() {
    // Every thread leaves the short run after its first yield and is tried
    // again later. Then one raises, one returns, one is closed by the
    // driver, one yields two values and later a string, and one calls a
    // function every round, which keeps it out of the short run for good.
    let source = r#"
local create, resume, yield = coroutine.create, coroutine.resume, coroutine.yield
local config = { bias = 7 }
local function start(first)
    local value = yield(first)
    return value + math.abs(config.bias) - 7
end
local workers = {
    function(first)
        local value = start(first)
        while true do value = yield((value * 3 + 7) % 997) end
    end,
    function(first)
        local value = start(first)
        for _ = 1, 150 do value = yield((value * 3 + 7) % 997) end
        error("the 150th round", 0)
    end,
    function(first)
        local value = start(first)
        for _ = 1, 220 do value = yield((value * 3 + 7) % 997) end
        return value, "done"
    end,
    function(first)
        local value = start(first)
        while true do value = yield(math.abs(value * 3 - 2000) % 997) end
    end,
    function(first)
        local value = start(first)
        for _ = 1, 260 do value = yield((value * 3 + 7) % 997) end
        value = yield(value, 1)
        for _ = 1, 100 do value = yield((value * 3 + 7) % 997) end
        value = yield("text")
        while true do value = yield((value * 3 + 7) % 997) end
    end,
    function(first)
        local value = start(first)
        while true do value = yield((value * 5 + 1) % 997) end
    end,
}
local threads = {}
for slot = 1, #workers do
    threads[slot] = create(workers[slot])
    assert(resume(threads[slot], slot))
end
local close = coroutine.close
local total, failures, extras, texts = 0, 0, 0, 0
for i = 1, 3600 do
    local slot = (i - 1) % 6 + 1
    if i == 2000 and close then close(threads[6]) end
    local ok, value, extra = resume(threads[slot], i % 997)
    if not ok then
        failures = failures + 1
    elseif type(value) == "number" then
        total = total + value
    else
        texts = texts + 1
    end
    if extra ~= nil then extras = extras + 1 end
end
print(total, failures, extras, texts)
for slot = 1, #workers do print(slot, coroutine.status(threads[slot])) end
"#;
    for profile in PROFILES {
        let (stdout, _) = agree(profile, source);
        assert!(stdout.contains("2\tdead"), "{profile}: {stdout}");
        assert!(stdout.contains("3\tdead"), "{profile}: {stdout}");
        assert!(stdout.contains("1\tsuspended"), "{profile}: {stdout}");
    }
}
