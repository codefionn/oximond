//! Traces run under a count hook and account for it: the hook fires at
//! exactly the instruction the interpreter fires it at, whether the loop
//! ran natively, left a trace by a guard, or never fit a trace at all.
//!
//! Every program installs a count hook that logs the line and the live
//! locals (`i` and `total`) of the interrupted function. A hook with count 1
//! fires before every counted instruction and never lets a trace run, so
//! its log, sampled every `N`th fire, is the reference for a hook with count
//! `N`.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// Defines `makehook(log, stride)`: a hook that appends the position of
/// every `stride`th fire to `log`.
const PRELUDE: &str = r#"
local function makehook(log, stride)
    local n = 0
    return function()
        n = n + 1
        if n % stride ~= 0 then return end
        local line = debug.getinfo(2, "l").currentline
        local parts = {line}
        local k = 1
        while true do
            local name, value = debug.getlocal(2, k)
            if not name then break end
            if name == "i" or name == "total" then
                parts[#parts + 1] = name .. "=" .. tostring(value)
            end
            k = k + 1
        end
        log[#log + 1] = table.concat(parts, " ")
    end
end
"#;

struct Program {
    name: &'static str,
    /// The body after the prelude. `COUNT` and `STRIDE` are numbers the test
    /// substitutes, and the program returns its checksum and the logs.
    body: &'static str,
    /// Whether the program has a hot loop a trace must run natively.
    traced: bool,
}

const PROGRAMS: &[Program] = &[
    Program {
        name: "a plain loop",
        body: r#"
local log = {}
local total = 0
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 6000 do
    total = total + i % 7
    total = total * 3 % 1000003
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#,
        traced: true,
    },
    Program {
        name: "a loop with side exits",
        body: r#"
local log = {}
local total = 0
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 6000 do
    if i % 3 == 0 then
        total = total + 2
    elseif i % 5 == 0 then
        total = total - 1
    else
        total = total + i % 4
    end
    total = total % 1000003
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#,
        traced: true,
    },
    Program {
        name: "inlined Lua calls",
        body: r#"
local log = {}
local total = 0
local function f(a)
    return a + 1
end
local function g(a, b)
    return f(a) * 2 + b
end
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 4000 do
    total = total + g(i % 5, i % 3)
    total = total % 1000003
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#,
        traced: true,
    },
    Program {
        name: "a metamethod called from a traced loop",
        body: r#"
local log = {}
local total = 0
local calls = 0
local t = setmetatable({}, {__index = function(_, k)
    calls = calls + 1
    return k * 2
end})
for j = 1, 100 do rawset(t, j, j) end
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 4000 do
    local v = t[i % 128 + 1]
    total = total + v
    total = total % 1000003
end
debug.sethook()
return total, calls, #log, table.concat(log, ";")
"#,
        traced: true,
    },
    Program {
        name: "a nested loop",
        body: r#"
local log = {}
local total = 0
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 300 do
    for j = 1, 20 do
        total = total + j
        total = total % 1000003
    end
    total = total + i
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#,
        traced: false,
    },
    Program {
        name: "per-thread hooks",
        body: r#"
local main_log, co_log = {}, {}
local total = 0
local co = coroutine.create(function()
    local total = 0
    for i = 1, 3000 do
        total = total + i % 11
        total = total % 10007
        if i % 1000 == 0 then coroutine.yield(total) end
    end
    return total
end)
debug.sethook(co, makehook(co_log, STRIDE), "", COUNT)
debug.sethook(makehook(main_log, STRIDE), "", COUNT)
local co_total = 0
for i = 1, 4000 do
    total = total + i % 5
    total = total % 1000003
    if i % 1300 == 0 then
        local ok, value = coroutine.resume(co)
        co_total = co_total + value
    end
end
debug.sethook()
return total, co_total, #main_log, #co_log, table.concat(main_log, ";"), table.concat(co_log, ";")
"#,
        traced: true,
    },
    Program {
        name: "library calls the trace makes through the VM",
        // `tostring`, `string.find` and `table.concat` in the iteration, and
        // a `tostring` and a `table.concat` of loop-invariant arguments that
        // the optimizer may hoist out of the loop.
        body: r#"
local log = {}
local total = 0
local parts = {"a", "bc", "d"}
local seven = 7
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 4000 do
    local s = tostring(i % 97)
    if string.find(s, "1", 1, true) then total = total + 1 end
    total = total + #table.concat(parts, s)
    total = total + #tostring(seven) + #table.concat(parts, ",")
    total = total % 1000003
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#,
        traced: true,
    },
    Program {
        name: "globals read and written in the loop",
        body: r#"
local log = {}
local total = 0
step, acc = 3, 0
debug.sethook(makehook(log, STRIDE), "", COUNT)
for i = 1, 6000 do
    acc = (acc + i % 7 + step) % 1000003
    total = acc
end
debug.sethook()
return total, acc, #log, table.concat(log, ";")
"#,
        // Lua 5.1's `SetGlobal` keeps the loop out of traces under any hook;
        // `env_globals.rs` checks that the Lua 5.2+ form runs natively.
        traced: false,
    },
];

/// Where two results first differ, with a little context, so that a failure
/// stays readable.
fn first_difference(actual: &str, expected: &str) -> String {
    let at = actual
        .bytes()
        .zip(expected.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    let window = |text: &str| {
        let start = at.saturating_sub(60);
        let end = (at + 60).min(text.len());
        text.get(start..end).unwrap_or("").to_string()
    };
    format!(
        "differs at byte {at}\n    actual   ...{}\n    expected ...{}",
        window(actual),
        window(expected)
    )
}

fn source(program: &Program, count: u32, stride: u32) -> String {
    let body = program
        .body
        .replace("COUNT", &count.to_string())
        .replace("STRIDE", &stride.to_string());
    format!("{PRELUDE}{body}")
}

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
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua.exec(source).unwrap();
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// The hook fires where a count-1 hook, sampled every `count` fires, did.
#[test]
fn counted_traces_fire_where_the_dispatch_loop_does() {
    let mut failures = Vec::new();
    for profile in PROFILES {
        for program in PROGRAMS {
            for count in [997_u32, 100_003, 7, 61] {
                // The reference never lets a trace run.
                let (reference, _) = run(
                    profile,
                    ExecutionMode::Interpreter,
                    &source(program, 1, count),
                );
                let (interpreted, _) = run(
                    profile,
                    ExecutionMode::Interpreter,
                    &source(program, count, 1),
                );
                let (traced, stats) =
                    run(profile, ExecutionMode::Tracing, &source(program, count, 1));
                let label = format!("{} under {profile:?}, count {count}", program.name);
                if interpreted != reference {
                    failures.push(format!(
                        "{label}: interpreter {}",
                        first_difference(&interpreted, &reference)
                    ));
                }
                if traced != reference {
                    failures.push(format!(
                        "{label}: tracing {}",
                        first_difference(&traced, &reference)
                    ));
                }
                // A count that leaves a whole iteration's room runs the loop
                // natively.
                if program.traced && count >= 997 && stats.native_entries < 1 {
                    failures.push(format!("{label}: no native entry: {stats:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The loop of the first program runs as a trace and keeps running as one:
/// a large count fires the hook a handful of times in thousands of
/// iterations.
#[test]
fn a_hot_loop_runs_natively_under_a_count_hook() {
    for profile in PROFILES {
        let program = &PROGRAMS[0];
        let (_, stats) = run(
            profile,
            ExecutionMode::Tracing,
            &source(program, 100_003, 1),
        );
        assert!(stats.traces_recorded >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 3000, "{profile:?}: {stats:?}");
    }
}

/// A hook that changes its own count, or removes itself, takes effect at the
/// same instruction in both modes.
#[test]
fn hooks_that_change_themselves_fire_alike_with_traces() {
    let body = r#"
local log = {}
local total = 0
local fires = 0
local hook
hook = function()
    fires = fires + 1
    log[#log + 1] = debug.getinfo(2, "l").currentline .. ":" .. tostring(total)
    if fires == 3 then debug.sethook(hook, "", 2500) end
    if fires == 6 then debug.sethook(hook, "", 40) end
    if fires == 20 then debug.sethook() end
end
debug.sethook(hook, "", 900)
for i = 1, 8000 do
    total = total + i % 9
    total = total % 1000003
end
return total, fires, table.concat(log, ";")
"#;
    for profile in PROFILES {
        let (interpreted, _) = run(profile, ExecutionMode::Interpreter, body);
        let (traced, stats) = run(profile, ExecutionMode::Tracing, body);
        assert_eq!(traced, interpreted, "{profile:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
    }
}

/// A hook removed from inside itself leaves the loop to run natively
/// afterwards, with the same result.
#[test]
fn a_hook_removed_from_inside_lets_the_loop_run_on() {
    let body = r#"
local total = 0
local fires = 0
debug.sethook(function()
    fires = fires + 1
    if fires == 2 then debug.sethook() end
end, "", 700)
for i = 1, 8000 do
    total = total + i % 9
    total = total % 1000003
end
return total, fires
"#;
    for profile in PROFILES {
        let (interpreted, _) = run(profile, ExecutionMode::Interpreter, body);
        let (traced, stats) = run(profile, ExecutionMode::Tracing, body);
        assert_eq!(traced, interpreted, "{profile:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
    }
}

/// Call, return and line hooks keep traces out, as before.
#[test]
fn call_return_and_line_hooks_still_disable_traces() {
    for mask in ["c", "r", "l", "cr"] {
        let body = format!(
            r#"
local total = 0
local events = 0
debug.sethook(function() events = events + 1 end, "{mask}", 1000)
for i = 1, 3000 do
    total = total + i % 7
    total = total % 1000003
end
debug.sethook()
return total, events > 0
"#
        );
        for profile in PROFILES {
            let (interpreted, _) = run(profile, ExecutionMode::Interpreter, &body);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, &body);
            assert_eq!(traced, interpreted, "{mask} under {profile:?}");
            assert_eq!(
                stats.native_entries, 0,
                "{mask} under {profile:?}: {stats:?}"
            );
        }
    }
}

/// Traces compiled before a hook existed cannot account for it. Installing
/// the hook flushes them, and the loop records again with accounting.
#[test]
fn a_hook_installed_after_a_trace_ran_flushes_it() {
    let body = r#"
local total = 0
local fires = 0
for i = 1, 4000 do
    total = total + i % 7
    total = total % 1000003
end
debug.sethook(function() fires = fires + 1 end, "", 5000)
for i = 1, 4000 do
    total = total + i % 7
    total = total % 1000003
end
debug.sethook()
return total, fires
"#;
    for profile in PROFILES {
        let (interpreted, _) = run(profile, ExecutionMode::Interpreter, body);
        let (traced, stats) = run(profile, ExecutionMode::Tracing, body);
        assert_eq!(traced, interpreted, "{profile:?}");
        // Both loops ran natively: one trace before the hook, one after.
        assert!(stats.traces_recorded >= 2, "{profile:?}: {stats:?}");
    }
}

/// A tree with side traces counts through its links: the sides take over
/// the other arms of the branches, and the hook still fires where the
/// interpreter fires it.
#[test]
fn side_traces_count_through_their_links() {
    let body = r#"
local log = {}
local total = 0
debug.sethook(makehook(log, 1), "", COUNT)
for i = 1, 30000 do
    if i % 3 == 0 then
        total = total + 2
    elseif i % 5 == 0 then
        total = total - 1
    else
        total = total + i % 4
    end
    total = total % 1000003
end
debug.sethook()
return total, #log, table.concat(log, ";")
"#;
    for profile in PROFILES {
        for count in [997_u32, 100_003] {
            let source = body.replace("COUNT", &count.to_string());
            let source = format!("{PRELUDE}{source}");
            let (interpreted, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(traced, interpreted, "{profile:?}, count {count}");
            assert!(
                stats.side_installs >= 1 && stats.side_member_exits >= 1,
                "{profile:?}, count {count}: {stats:?}"
            );
        }
    }
}

/// A count that covers a few iterations at a time cuts every native run
/// short. That says nothing against the trace: it keeps running, and is not
/// retired or rejected for the short runs.
#[test]
fn runs_cut_short_by_the_count_do_not_retire_the_trace() {
    let body = r#"
local total = 0
local fires = 0
debug.sethook(function() fires = fires + 1 end, "", 300)
for i = 1, 20000 do
    total = total + i % 7
    total = total * 3 % 1000003
end
debug.sethook()
return total, fires
"#;
    for profile in PROFILES {
        let (interpreted, _) = run(profile, ExecutionMode::Interpreter, body);
        let (traced, stats) = run(profile, ExecutionMode::Tracing, body);
        assert_eq!(traced, interpreted, "{profile:?}");
        assert!(stats.native_entries > 100, "{profile:?}: {stats:?}");
        assert!(
            stats.iterations > 16000,
            "most iterations run natively, {profile:?}: {stats:?}"
        );
    }
}

/// `debug.sethook` called from the traced loop is a call the trace cannot
/// make: the loop leaves the trace there, and the new hook counts on.
#[test]
fn sethook_from_a_traced_loop_leaves_the_trace() {
    let body = r#"
local total = 0
local fires = 0
local function hook() fires = fires + 1 end
debug.sethook(hook, "", 1500)
for i = 1, 12000 do
    total = total + i % 7
    total = total % 1000003
    if i % 1000 == 0 then
        debug.sethook(hook, "", 1000 + i % 7)
    end
end
debug.sethook()
return total, fires
"#;
    for profile in PROFILES {
        let (interpreted, _) = run(profile, ExecutionMode::Interpreter, body);
        let (traced, _) = run(profile, ExecutionMode::Tracing, body);
        assert_eq!(traced, interpreted, "{profile:?}");
    }
}
