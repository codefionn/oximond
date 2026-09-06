//! Trace recording through the whole stack: compiled Lua source, every
//! profile.
//!
//! Recording must never change what a program computes. Each program runs
//! under the interpreter and under tracing execution, and the results must
//! match exactly. Counters then show that recording really happened where it
//! should: every profile records its numeric loops.

use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::process::Command;

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// A program that runs a hot loop of floats, and whether the recorder can
/// follow it to a closed trace.
struct Program {
    name: &'static str,
    source: &'static str,
    records: bool,
}

const PROGRAMS: &[Program] = &[
    Program {
        name: "float sum",
        source: "local s = 0.5 for i = 1, 300 do s = s + i * 0.5 end return s",
        records: true,
    },
    Program {
        name: "alternating branch",
        source: "local s = 0 for i = 1, 300 do if i % 2 == 0 then s = s + i else s = s - 1 end end return s",
        records: true,
    },
    Program {
        name: "while loop",
        source: "local x, n = 1, 0 while x < 1e12 do x = x * 1.2 + 1 n = n + 1 end return x, n",
        records: true,
    },
    Program {
        name: "nested loops",
        source: "local s = 0 for i = 1, 100 do for j = 1, 5 do s = s + i * j end end return s",
        records: true,
    },
    Program {
        name: "descending by two",
        source: "local s = 0 for i = 300, 1, -2 do s = s + i end return s",
        records: true,
    },
    Program {
        name: "early return",
        source: "local function f(n) local s = 0 for i = 1, n do s = s + i if i > 150 then return s end end return -1 end return f(300), f(100)",
        records: true,
    },
    Program {
        name: "division by zero and nan",
        source: "local s = 0 for i = 1, 300 do s = s + (i % 7) / (i % 3 - 1) end return s ~= s, s",
        records: true,
    },
    Program {
        name: "boolean flag",
        source: "local flag, n = false, 0 for i = 1, 300 do flag = not flag if flag then n = n + 1 end end return n, flag",
        records: true,
    },
    Program {
        name: "nil comparison",
        source: "local x = nil local n = 0 for i = 1, 300 do if x == nil then n = n + 1 end end return n",
        records: true,
    },
    Program {
        name: "break",
        source: "local i = 0 while true do i = i + 1 if i > 200 then break end end return i",
        records: true,
    },
    Program {
        // The fill appends through raw sets, the sum reads the array.
        name: "table reads",
        source: "local t = {} for i = 1, 200 do t[i] = i end local s = 0 for i = 1, 200 do s = s + t[i] end return s",
        records: true,
    },
    Program {
        name: "builtin calls",
        source: "local s = 0 for i = 1, 300 do s = s + math.atan2(i, 2) end return s",
        records: false,
    },
    Program {
        name: "string building",
        source: "local s = '' for i = 1, 200 do s = s .. (os.clock() < 0 and 'y' or 'x') end return #s",
        records: false,
    },
    Program {
        name: "error in loop",
        source: "local ok, e = pcall(function() local s = 0 for i = 1, 300 do s = s + i if i == 200 then error('stop') end end end) return ok, e",
        records: true,
    },
    Program {
        name: "coroutine yield in loop",
        source: "local co = coroutine.wrap(function() local s = 0 for i = 1, 300 do s = s + i coroutine.yield(s) end end) local last for i = 1, 300 do last = co() end return last",
        records: false,
    },
];

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua.exec(source).unwrap();
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

#[test]
fn recording_never_changes_results() {
    for profile in PROFILES {
        for program in PROGRAMS {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, program.source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, program.source);
            assert_eq!(actual, expected, "{} under {profile:?}", program.name);
        }
    }
}

#[test]
fn numeric_loops_record_in_every_profile() {
    let mut failures = Vec::new();
    for profile in PROFILES {
        for program in PROGRAMS.iter().filter(|program| program.records) {
            let (_, stats) = run(profile, ExecutionMode::Tracing, program.source);
            // A branch whose arms leave a temporary register typed in one and
            // cleared in the other is type-unstable for the recording that
            // starts after the "wrong" arm. Which arm that is depends on how
            // the profile counts the first backedge, so a clean
            // `TypeUnstable` refusal is as good as a trace there.
            let unstable = stats.recording_aborts[TraceAbortReason::TypeUnstable.index()];
            let parity = program.name == "alternating branch"
                && stats.traces_recorded == 0
                && unstable == stats.recording_starts;
            // The fast loop ran while recording.
            if !parity && (stats.traces_recorded < 1 || stats.fast_path_records == 0) {
                failures.push(format!("{} under {profile:?}: {stats:?}", program.name));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn unsupported_programs_abort_with_a_counted_reason() {
    for profile in PROFILES {
        for program in PROGRAMS.iter().filter(|program| !program.records) {
            let (_, stats) = run(profile, ExecutionMode::Tracing, program.source);
            // A loop that only reads and writes tables, calls, or builds
            // strings never closes a trace, and says why.
            assert!(
                stats.recording_starts == 0 || stats.total_aborts() > 0,
                "{} under {profile:?}: {stats:?}",
                program.name
            );
            assert_eq!(
                stats.recording_starts,
                stats.traces_recorded + stats.total_aborts(),
                "{} under {profile:?}: every start ends: {stats:?}",
                program.name
            );
        }
    }
}

#[test]
fn interpreter_mode_never_records() {
    for profile in PROFILES {
        let (_, stats) = run(profile, ExecutionMode::Interpreter, PROGRAMS[0].source);
        assert_eq!(stats, TraceStatistics::default(), "{profile:?}");
    }
}

#[test]
fn the_command_line_runs_every_profile_under_tracing_execution() {
    let source = "local s = 0 for i = 1, 500 do if i % 3 == 0 then s = s + i * 0.25 else s = s - 1 end end print(s)";
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        let run = |extra: &[&str]| {
            let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .arg(format!("--compat={profile}"))
                .args(extra)
                .arg("-e")
                .arg(source)
                .output()
                .expect("run oximond");
            assert!(output.status.success(), "{profile}: {output:?}");
            output.stdout
        };
        let expected = run(&["--execution=interpreter"]);
        assert_eq!(run(&["--execution=tracing"]), expected, "{profile}");
    }
}

#[test]
fn the_optimized_ir_dump_and_the_optimizer_counters_are_opt_in() {
    let source = "local s = 0 for i = 1, 3000000 do s = s + i * 0.25 end print(s)";
    let run = |dump: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
        command
            .args(["--execution=tracing", "-e", source])
            .env("OXIMOND_JIT_STATS", "1")
            .env_remove("OXIMOND_JIT_TRACE_IR_OPT");
        if dump {
            command.env("OXIMOND_JIT_TRACE_IR_OPT", "1");
        }
        let output = command.output().expect("run oximond");
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stderr).unwrap()
    };
    let quiet = run(false);
    assert!(!quiet.contains("jit-trace: optimized"), "{quiet}");
    assert!(quiet.contains("loads_forwarded: 0"), "{quiet}");
    assert!(quiet.contains("loads_merged: 0"), "{quiet}");
    assert!(quiet.contains("loads_hoisted: 0"), "{quiet}");
    let dumped = run(true);
    assert!(dumped.contains("jit-trace: optimized member 0"), "{dumped}");
    assert!(dumped.contains("jit-trace: optimizer"), "{dumped}");
}

fn trace_counters(sync: Option<&str>) -> String {
    let source = "local s = 0 for i = 1, 3000000 do s = s + i * 0.25 end print(s)";
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .args(["--execution=tracing", "-e", source])
        .env("OXIMOND_JIT_STATS", "1")
        .env_remove("OXIMOND_JIT_SYNC");
    if let Some(value) = sync {
        command.env("OXIMOND_JIT_SYNC", value);
    }
    let output = command.output().expect("run oximond");
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    let line = stderr
        .lines()
        .find(|line| line.starts_with("trace:"))
        .unwrap_or_else(|| panic!("no trace line: {stderr}"));
    ["native_entries", "iterations", "traces_compiled"]
        .iter()
        .map(|name| {
            let at = line.find(&format!("{name}: ")).expect(name);
            let rest = &line[at + name.len() + 2..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
            format!("{name}={}", &rest[..end])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn synchronous_compilation_gives_identical_trace_counters() {
    let first = trace_counters(Some("1"));
    let second = trace_counters(Some("1"));
    assert_eq!(first, second);
    assert!(first.contains("traces_compiled=1"), "{first}");
    assert!(!first.contains("native_entries=0"), "{first}");
    assert!(!first.contains("iterations=0"), "{first}");
}

#[test]
fn without_the_sync_variable_traces_still_compile() {
    // `0` and unset both leave the worker thread in place; the loop still
    // reaches native code.
    for value in [None, Some("0")] {
        let counters = trace_counters(value);
        assert!(!counters.contains("traces_compiled=0"), "{counters}");
    }
}

/// Runs `source` under the interpreter and under tracing execution with
/// synchronous installs. Returns the program's output and the optimizer
/// counters of the last tree the optimizer compiled.
fn run_with_side_trace(source: &str) -> (String, String) {
    let run = |execution: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args([execution, "-e", source])
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_TRACE_IR_OPT", "1")
            .output()
            .expect("run oximond");
        assert!(output.status.success(), "{output:?}");
        (
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    };
    let (expected, _) = run("--execution=interpreter");
    let (actual, stderr) = run("--execution=tracing");
    assert_eq!(actual, expected, "{stderr}");
    assert!(
        stderr.contains("2 member(s)"),
        "a side trace joined the tree: {stderr}"
    );
    let last = stderr
        .lines()
        .rfind(|line| line.starts_with("jit-trace: optimizer"))
        .expect("optimizer line")
        .to_owned();
    (actual, last)
}

#[test]
fn a_side_trace_that_writes_the_root_s_field_stops_the_root_hoisting_it() {
    let (output, optimizer) = run_with_side_trace(
        "local t = {x = 1.5, y = 0.5} local s = 0 \
         for i = 1, 400000 do s = s + t.x if i % 1000 == 0 then t.x = t.x + 1 end end \
         print(s, t.x, t.y)",
    );
    assert_eq!(output, "80400000\t401.5\t0.5\n");
    assert!(optimizer.contains("loads_hoisted: 0"), "{optimizer}");
}

#[test]
fn a_side_trace_that_writes_another_field_leaves_the_root_read_hoisted() {
    let (output, optimizer) = run_with_side_trace(
        "local t = {x = 1.5, y = 0.5} local s = 0 \
         for i = 1, 400000 do s = s + t.x if i % 1000 == 0 then t.y = t.y + 1 end end \
         print(s, t.x, t.y)",
    );
    assert_eq!(output, "600000\t1.5\t400.5\n");
    assert!(!optimizer.contains("loads_hoisted: 0"), "{optimizer}");
}

#[test]
fn a_side_trace_that_writes_another_array_slot_leaves_the_root_read_hoisted() {
    let (output, optimizer) = run_with_side_trace(
        "local a = {1.5, 0.5} local s = 0 \
         for i = 1, 400000 do s = s + a[1] if i % 1000 == 0 then a[2] = a[2] + 1 end end \
         print(s, a[1], a[2])",
    );
    assert_eq!(output, "600000\t1.5\t400.5\n");
    assert!(!optimizer.contains("loads_hoisted: 0"), "{optimizer}");
}

#[test]
fn a_side_trace_that_writes_the_root_s_array_slot_stops_the_root_hoisting_it() {
    let (output, optimizer) = run_with_side_trace(
        "local a = {1.5, 0.5} local s = 0 \
         for i = 1, 400000 do s = s + a[1] if i % 1000 == 0 then a[1] = a[1] + 1 end end \
         print(s, a[1], a[2])",
    );
    assert_eq!(output, "80400000\t401.5\t0.5\n");
    assert!(optimizer.contains("loads_hoisted: 0"), "{optimizer}");
}

#[test]
fn a_hoisted_array_guard_that_fails_at_entry_matches_the_interpreter() {
    // The slot is a float in some rounds and an integer in others, so a trace
    // compiled for floats fails its hoisted guard when it is entered.
    let (_, optimizer) = run_with_side_trace(
        "local a = {1.5, 0.5} local s = 0 \
         for r = 1, 8 do \
           if r % 2 == 0 then a[1] = 2 else a[1] = 1.5 end \
           for i = 1, 100000 do s = s + a[1] if i % 1000 == 0 then a[2] = a[2] + 1 end end \
         end \
         print(s, a[1], a[2])",
    );
    assert!(!optimizer.contains("loads_hoisted: 0"), "{optimizer}");
}

/// Runs `source` under the interpreter and under tracing execution with
/// synchronous installs and `OXIMOND_JIT_STATS`. Returns the output, the
/// optimizer line of the last tree and the statistics line.
fn run_with_stats(source: &str) -> (String, String, String) {
    let run = |execution: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args([execution, "-e", source])
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_STATS", "1")
            .env("OXIMOND_JIT_TRACE_IR_OPT", "1")
            .output()
            .expect("run oximond");
        assert!(output.status.success(), "{output:?}");
        (
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    };
    let (expected, _) = run("--execution=interpreter");
    let (actual, stderr) = run("--execution=tracing");
    assert_eq!(actual, expected, "{stderr}");
    let optimizer = stderr
        .lines()
        .rfind(|line| line.starts_with("jit-trace: optimizer"))
        .expect("optimizer line")
        .to_owned();
    let stats = stderr
        .lines()
        .rfind(|line| line.starts_with("trace: TraceStatistics"))
        .expect("statistics line")
        .to_owned();
    (actual, optimizer, stats)
}

/// A side trace that rebinds the register a root read hoists from would leave
/// the hoisted value stale, so `plan_tree` refuses it. No side joins the tree
/// today: every swap runs in the interpreter after a guard exit and re-enters
/// through the preheader, which reads the new table again.
#[test]
fn a_side_trace_that_rebinds_the_hoisted_receiver_is_refused_and_matches_the_interpreter() {
    let (output, optimizer, stats) = run_with_stats(
        "local A = {x = 1.5} local B = {x = 100.5} local t = A local s = 0 \
         for i = 1, 200000 do s = s + t.x \
           if i % 1000 == 0 then t = B end \
           if i % 1000 == 500 then t = A end end \
         print(s) \
         local a1, a2 = {1.5, 0.5}, {7.5, 0.5} local a = a1 local u = 0 \
         for i = 1, 200000 do u = u + a[1] \
           if i % 1000 == 0 then a = a2 end \
           if i % 1000 == 500 then a = a1 end end \
         print(u)",
    );
    assert_eq!(output, "10150500\n897000\n");
    assert!(!optimizer.contains("loads_hoisted: 0"), "{optimizer}");
    assert!(stats.contains("side_installs: 0,"), "{stats}");
    assert!(!stats.contains("side_rejections: 0,"), "{stats}");
}

/// A loop trace that paid in one phase and retired in a phase of short
/// runs records again when long runs come back, and the new recording
/// compiles: only a trace that never passed a productivity window is refused
/// as a repeat of itself.
#[test]
fn a_loop_retired_after_paying_compiles_again_when_its_runs_grow() {
    let source = "local function work(n) local s = 0 for i = 1, n do s = s + i end return s end \
                  local t = 0 \
                  for r = 1, 300 do t = t + work(1000) end \
                  for r = 1, 400 do t = t + work(3) end \
                  for r = 1, 200 do t = t + work(5000) end \
                  return t";
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::LuaJit21,
    ] {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        // The first phase runs 300,000 loop-backs natively, the last one
        // 1,000,000 more.
        assert!(
            stats.iterations >= 1_200_000,
            "{profile:?}: the last phase ran in the interpreter\n{stats:?}"
        );
    }
}
