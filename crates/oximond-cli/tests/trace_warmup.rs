//! What a trace costs before it pays: the code the compiler is given for the
//! exits of a loop, the lookup of a table in a run, and the work a table
//! with nothing in its hash part costs a row.
//!
//! Results must match the interpreter bit for bit. The counters pin the
//! shape of the warm-up of a loop nest (one stone, one tree), and a
//! `perf stat` budget pins what that warm-up costs in instructions and page
//! faults.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn canon(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .map(|value| match value {
            Value::Integer(integer) => format!("I{integer}"),
            Value::Number(number) => format!("F{:016x}", number.to_bits()),
            Value::Boolean(flag) => format!("B{flag}"),
            other => format!("{other:?}"),
        })
        .collect()
}

struct Outcome {
    results: Vec<String>,
    trace: TraceStatistics,
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Outcome {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    Outcome {
        results: canon(&results),
        trace: lua.vm().jit_statistics().trace,
    }
}

/// Runs `source` under both modes and returns the tracing outcome, after
/// checking that its result matches the interpreter's.
fn check(source: &str) -> Outcome {
    let profile = CompatibilityProfile::Lua51;
    let expected = run(profile, ExecutionMode::Interpreter, source);
    let tracing = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(tracing.results, expected.results, "tracing\n{source}");
    tracing
}

/// The loops of `benchmarks/lua/algorithm/nbody.lua`, smaller. Its inner
/// loop becomes a stone and then the root of a tree whose root has many
/// exits, all of which save the same values first.
const NBODY: &str = r"
local bodies = {
    {x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0, mass = 39.47841760435743},
    {x = 4.84, y = -1.16, z = -0.10, vx = 0.60, vy = 2.81, vz = -0.025, mass = 0.037},
    {x = 8.34, y = 4.12, z = -0.40, vx = -1.01, vy = 1.82, vz = 0.0084, mass = 0.011},
    {x = 12.89, y = -15.11, z = -0.22, vx = 1.08, vy = 0.86, vz = -0.010, mass = 0.0017},
    {x = 15.37, y = -25.91, z = 0.17, vx = 0.97, vy = 0.59, vz = -0.034, mass = 0.0020},
}
local function advance(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy, dz = first.x - second.x, first.y - second.y, first.z - second.z
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            local first_scale, second_scale = second.mass * magnitude, first.mass * magnitude
            first.vx, first.vy, first.vz = first.vx - dx * first_scale, first.vy - dy * first_scale, first.vz - dz * first_scale
            second.vx, second.vy, second.vz = second.vx + dx * second_scale, second.vy + dy * second_scale, second.vz + dz * second_scale
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x, body.y, body.z = body.x + dt * body.vx, body.y + dt * body.vy, body.z + dt * body.vz
    end
end
for _ = 1, 600 do
    advance(0.01)
end
local total = 0
for i = 1, #bodies do
    local body = bodies[i]
    total = total + body.x + body.y + body.z + body.vx + body.vy + body.vz
end
return total
";

#[test]
fn a_loop_nest_warms_up_with_one_tree() {
    let outcome = check(NBODY);
    let trace = outcome.trace;
    // The inner loop is never compiled alone: the interpreter records the
    // sides of its tree (the outer side, the code beyond the nest and the
    // update loop there as a loop side), and the tree compiles once. The
    // calls link into the loop that makes them after the tree has had
    // `RETURN_THROUGH_ENTRIES` entries. The tree is small, so the sides
    // that do it go into one more compile of the tree, which the MIR
    // backend makes for less than those sides cost compiled beside it
    // (15.0M instructions for this run against 17.8M): two compiles in
    // all, and no side apart.
    assert_eq!(trace.traces_compiled, 1, "{trace:?}");
    assert_eq!(trace.side_compiles, 0, "{trace:?}");
    assert_eq!(trace.compile_submissions, 2, "{trace:?}");
    assert_eq!(trace.link_recompiles, 0, "{trace:?}");
    assert_eq!(trace.baseline_roots, 0, "{trace:?}");
    assert!(trace.side_installs >= 3, "{trace:?}");
    // The update loop beside the nest, and the loop sides of the link.
    assert!(trace.loop_sides >= 1, "{trace:?}");
    // Native code runs the calls: an entry for each before the link, and the
    // iterations of the rest inside the loop around them.
    assert!(trace.native_entries >= 64, "{trace:?}");
    assert!(trace.iterations > 5_000, "{trace:?}");
}

#[test]
fn exits_that_share_their_stores_restore_the_frame() {
    // Each round turns one element into a string at another position, so
    // the guards of the loop fail at different places: every exit of the
    // root runs, with its own values and the ones all exits save.
    let source = r#"
local t = {}
for i = 1, 40 do t[i] = i * 0.5 end
local acc, count, odd = 0.0, 0, 0
for round = 1, 400 do
    local hole = (round % 7) + 1
    local saved = t[hole]
    if round % 5 == 0 then t[hole] = "x" end
    if round % 11 == 0 then t[hole + 1] = false end
    for j = 1, 40 do
        local v = t[j]
        if type(v) == "number" then
            acc = acc + v * round
            count = count + 1
        else
            odd = odd + j
        end
    end
    t[hole] = saved
    t[hole + 1] = (hole + 1) * 0.5
end
return acc, count, odd
"#;
    let trace = check(source).trace;
    assert!(trace.traces_compiled >= 1, "{trace:?}");
}

#[test]
fn a_run_over_more_tables_than_it_scans_finds_each_lane() {
    // One entry admits forty tables, past the bank a run scans for a table,
    // and the next round meets the same tables again.
    let source = r"
local rows = {}
for i = 1, 40 do rows[i] = {a = i, b = i * 2.5, c = i % 3} end
local total = 0.0
for round = 1, 300 do
    for i = 1, #rows do
        local r = rows[i]
        total = total + r.a * 0.5 + r.b - r.c
    end
    rows[(round % 40) + 1].c = round
end
return total
";
    let trace = check(source).trace;
    assert!(trace.native_entries >= 1, "{trace:?}");
}

#[test]
fn tables_of_one_run_keep_their_lanes_when_the_run_makes_more() {
    // The run makes a table per iteration, which takes a lane of the bank
    // and may not be mistaken for one the run was given.
    let source = r"
local rows = {}
for i = 1, 20 do rows[i] = {a = i} end
local total = 0
for round = 1, 200 do
    for i = 1, #rows do
        local r = rows[i]
        local fresh = {a = r.a + round, b = i}
        total = total + fresh.a + fresh.b + r.a
    end
end
return total
";
    check(source);
}

#[test]
fn a_table_the_run_made_and_released_is_still_found_by_its_holder() {
    // `old` holds a table made five iterations ago. The run released its
    // lane and handed it out again, so the lane map must not name it.
    let source = r#"
local total = 0
for round = 1, 3000 do
    local keep = {}
    for i = 1, 60 do
        keep[i] = {a = i * round}
        local old = keep[i > 5 and i - 5 or 1]
        total = total + old.a + #("x" .. i)
    end
end
return total
"#;
    check(source);
}

#[test]
fn a_released_made_table_is_found_when_the_run_was_given_many_tables() {
    // The same, with more entry tables than a run scans for a table.
    let source = r"
local rows = {}
for i = 1, 24 do rows[i] = {a = i} end
local total = 0
for round = 1, 1500 do
    local keep = {}
    for i = 1, 60 do
        keep[i] = {a = i * round}
        local old = keep[i > 5 and i - 5 or 1]
        total = total + old.a + rows[(i % 24) + 1].a
    end
end
return total
";
    check(source);
}

#[cfg(all(target_os = "linux", not(debug_assertions)))]
fn perf_count(report: &str, event: &str) -> Option<u64> {
    report
        .lines()
        .find(|line| line.contains(event))
        .and_then(|line| line.split(',').next())
        .and_then(|field| field.trim().parse().ok())
}

/// What the tree of `algorithm/nbody` costs to warm up, from `perf stat` on
/// the benchmark at scale 3. A run took 96.6M instructions and 638 page
/// faults before the exits shared their stores, floats went to the exit area
/// as they are, a table was found by scanning a small bank, and the code a
/// warm-up runs sat together (`scripts/startup-order.sh`); it takes 88.24M
/// and 490 without huge pages, and 250 to 450 when the kernel backs the
/// 4 MiB heap steps with them. The instruction count moves with the length
/// of the environment and the binary's path, so the run gets a clean
/// environment, and both budgets leave about 5% for a longer path and
/// another kernel's fault-around. Skipped without perf, without the
/// benchmark and in a debug build.
#[cfg(all(target_os = "linux", not(debug_assertions)))]
#[test]
fn nbody_warms_up_within_its_budget() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/lua/algorithm/nbody.lua");
    if !script.exists() {
        return;
    }
    // Found through this process's PATH, but the run gets a fixed one.
    let Some(perf) = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory| directory.join("perf"))
            .find(|candidate| candidate.is_file())
    }) else {
        return;
    };
    let output = match std::process::Command::new(perf)
        .args(["stat", "-x,", "-e", "instructions:u,page-faults", "--"])
        .arg(env!("CARGO_BIN_EXE_oximond"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        // On the thread that runs the program, as with one CPU: a worker
        // thread has an arena of its own, and the run takes 50 faults more.
        .env("OXIMOND_JIT_SYNC", "1")
        .args(["--compat=lua51", "--execution=tracing", "-E"])
        .arg(&script)
        .arg("3")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return,
    };
    let report = String::from_utf8_lossy(&output.stderr);
    if let (Some(instructions), Some(faults)) = (
        perf_count(&report, "instructions"),
        perf_count(&report, "page-faults"),
    ) {
        assert!(instructions <= 92_600_000, "{instructions} instructions");
        assert!(faults <= 515, "{faults} page faults");
    }
}
