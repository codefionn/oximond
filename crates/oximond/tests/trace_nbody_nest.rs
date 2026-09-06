//! The shape of the n-body benchmark: a loop over bodies with a short loop
//! inside it that starts at the next body, work on two shared tables per trip
//! (`bodies[i]`, `bodies[j]`), `math.sqrt`, and a second loop over the bodies
//! after the nest. The inner loop runs four, three, two, one and then no
//! trips per call, so a trace of it leaves the loop at every outer iteration.
//!
//! Every program runs interpreted and traced in each profile and must give
//! the same results. The cases change what the recording saw: the trip
//! counts (none, one, many, different on every call), a `break` out of the
//! inner loop, and the type of a field of a body that the trace has bound.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// The outcome as text, which tells `-0.0` from `0.0` and compares NaN with
/// itself.
fn show(result: &Result<Vec<Value>, String>) -> String {
    format!("{result:?}")
}

/// Runs `source` interpreted and traced in every profile and checks that the
/// outcomes agree and that the traced run entered native code.
fn assert_traced(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?} {source}");
    }
}

/// Checks only that both modes agree, for programs too small to trace.
fn assert_agrees(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
    }
}

/// [`assert_traced`] for a nest whose root exits at every outer iteration, so
/// that Lua 5.1's float loops must also have taken an outer side: the tree
/// runs the outer loop, not just the inner one.
fn assert_traced_through_the_outer_loop(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?} {source}");
        if profile == CompatibilityProfile::Lua51 {
            assert!(stats.outer_sides > 0, "{profile:?} {stats:?} {source}");
        }
    }
}

/// How many times the fast loop left for a lookup at a loop header while
/// `source` ran traced in Lua 5.1.
fn header_lookups(source: &str) -> u64 {
    let (_, lua) = outcome(CompatibilityProfile::Lua51, ExecutionMode::Tracing, source);
    lua.vm().jit_statistics().fast_backedge_lookups
}

/// The benchmark's `advance`, over `count` bodies, run `rounds` times. The
/// bodies are the tables of `n-body`, reduced to the fields it uses.
fn bodies(count: usize) -> String {
    use std::fmt::Write;
    let mut text = String::from("local bodies = {}\n");
    for index in 1..=count {
        let n = f64::from(u32::try_from(index).unwrap());
        writeln!(
            text,
            "bodies[{index}] = {{x = {}, y = {}, z = {}, vx = {}, vy = -0.02, vz = 0.005, mass = {}}}",
            n * 1.5,
            n.powi(2) * 0.25,
            0.125 * n,
            0.01 * n,
            1.0 + n * 0.5,
        )
        .unwrap();
    }
    text
}

const ADVANCE: &str = "
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
";

const TOTAL: &str = "
local total = 0
for i = 1, #bodies do
    local body = bodies[i]
    total = total + body.x + body.y + body.z + body.vx + body.vy + body.vz
end
return string.format('%.9f', total)
";

#[test]
fn the_benchmark_nest_runs_through_its_outer_loop() {
    let source = format!(
        "{}{}for _ = 1, 3000 do advance(0.01) end{}",
        bodies(5),
        ADVANCE,
        TOTAL
    );
    assert_traced_through_the_outer_loop(&source);
}

#[test]
fn the_loops_a_tree_covers_or_the_tracer_rejected_cost_no_lookup_per_call() {
    // Every call of `advance` enters the tree once, at the inner loop, which
    // is one lookup. The outer loop it covers, the second loop over the bodies
    // (too short to trace) and the loop that calls `advance` each have
    // skip windows of their own instead of a lookup on every backedge.
    let rounds = 3000;
    let source = format!(
        "{}{}for _ = 1, {rounds} do advance(0.01) end{}",
        bodies(5),
        ADVANCE,
        TOTAL
    );
    let lookups = header_lookups(&source);
    assert!(
        lookups < rounds * 3 / 2,
        "{lookups} lookups for {rounds} calls"
    );
}

#[test]
fn an_inner_loop_that_runs_no_trips_or_one_trip() {
    // Two bodies: the inner loop runs once for the first and not at all for
    // the second. One body: it never runs. Zero bodies: neither loop does.
    // Such loops are too short to trace, and must give the same answers.
    for count in [0, 1, 2, 3] {
        let source = format!(
            "{}{}for _ = 1, 2500 do advance(0.01) end{}",
            bodies(count),
            ADVANCE,
            TOTAL
        );
        assert_agrees(&source);
    }
    // With bodies to spare the last two outer iterations run one trip and
    // none, and the trace of the loop leaves there on every call.
    let source = format!(
        "{}{}for _ = 1, 2500 do advance(0.01) end{}",
        bodies(7),
        ADVANCE,
        TOTAL
    );
    assert_traced(&source);
}

#[test]
fn an_inner_loop_whose_trip_count_changes_on_every_call() {
    // `n` bodies take part in a call, so the inner loop starts at `i + 1` and
    // stops at `n`: none, one, a few or all the bodies' worth of trips.
    let source = format!(
        "{}local function partial(dt, n)
    for i = 1, n do
        local first = bodies[i]
        for j = i + 1, n do
            local second = bodies[j]
            local dx, dy = first.x - second.x, first.y - second.y
            local distance2 = dx * dx + dy * dy
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            first.vx = first.vx - dx * second.mass * magnitude
            second.vx = second.vx + dx * first.mass * magnitude
        end
    end
    for i = 1, n do
        local body = bodies[i]
        body.x = body.x + dt * body.vx
    end
end
for round = 1, 4000 do partial(0.01, round % 7) end
local sum = 0
for i = 1, #bodies do sum = sum + bodies[i].x + bodies[i].vx end
return string.format('%.9f', sum)",
        bodies(6)
    );
    assert_traced(&source);
}

#[test]
fn an_inner_loop_that_leaves_through_a_break() {
    // The body at index 4 is far away in `x`: a pair that involves it stops
    // the inner loop early, on every call, and on others only after the
    // bodies have drifted.
    let source = format!(
        "{}bodies[4].x = 400
local function advance_until_far(dt)
    local stopped = 0
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy, dz = first.x - second.x, first.y - second.y, first.z - second.z
            if dx * dx > 90000 then stopped = stopped + 1 break end
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            first.vx = first.vx - dx * second.mass * magnitude
            second.vx = second.vx + dx * first.mass * magnitude
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x = body.x + dt * body.vx
    end
    return stopped
end
local stops = 0
for _ = 1, 3000 do stops = stops + advance_until_far(0.01) end
local sum = 0
for i = 1, #bodies do sum = sum + bodies[i].x + bodies[i].vx end
return stops, string.format('%.9f', sum)",
        bodies(6)
    );
    assert_traced(&source);
}

#[test]
fn a_body_field_that_changes_type_in_the_middle_of_the_run() {
    // `mass` of a body becomes a string, a number again, then nil with a
    // default; `vx` of another becomes an integer-valued string. The traced
    // run leaves at the entry or at the load and the interpreter does what
    // the language says (string arithmetic coerces, `nil or 1` defaults).
    let source = format!(
        "{}local function advance_with_defaults(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy, dz = first.x - second.x, first.y - second.y, first.z - second.z
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            local first_scale = (second.mass or 1) * magnitude
            local second_scale = (first.mass or 1) * magnitude
            first.vx = first.vx - dx * first_scale
            second.vx = second.vx + dx * second_scale
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x = body.x + dt * body.vx
    end
end
for round = 1, 6000 do
    if round == 1500 then bodies[3].mass = '2.5' end
    if round == 2500 then bodies[3].mass = 2.5 end
    if round == 3500 then bodies[2].mass = nil end
    if round == 4500 then bodies[2].mass = 0.75 end
    if round == 5000 then bodies[4].vx = '0' end
    advance_with_defaults(0.01)
end
local sum = 0
for i = 1, #bodies do sum = sum + bodies[i].x + bodies[i].vx end
return string.format('%.9f', sum)",
        bodies(5)
    );
    assert_traced(&source);
}

#[test]
fn a_body_that_loses_a_field_or_gains_a_metatable() {
    let source = format!(
        "{}local ghost = setmetatable({{}}, {{__index = function(_, key) return 0.5 end}})
local function advance_all(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy = first.x - second.x, first.y - second.y
            local distance2 = dx * dx + dy * dy
            local magnitude = dt / (distance2 * math.sqrt(distance2))
            first.vx = first.vx - dx * second.mass * magnitude
            second.vx = second.vx + dx * first.mass * magnitude
        end
    end
    for i = 1, #bodies do
        local body = bodies[i]
        body.x = body.x + dt * body.vx
    end
end
for round = 1, 5000 do
    if round == 2000 then
        local replacement = {{x = 9, y = 9, vx = 0.5}}
        setmetatable(replacement, {{__index = {{mass = 3}}}})
        bodies[3] = replacement
    end
    if round == 3500 then bodies[5] = ghost; ghost.x = 7; ghost.y = -7; ghost.vx = 0.25 end
    advance_all(0.01)
end
local sum = 0
for i = 1, #bodies do sum = sum + bodies[i].x + bodies[i].vx end
return string.format('%.9f', sum)",
        bodies(5)
    );
    assert_traced(&source);
}

#[test]
fn math_sqrt_in_a_trace_gives_the_interpreters_answers() {
    // Negative, zero, negative zero, infinite, NaN, subnormal and huge
    // arguments, taken from a table so the trace sees them as numbers of one
    // type. The tostring of each result tells -0 from 0 and shows a NaN.
    let source = "local values = {4, 2, 0, -0, -1, 1/0, -1/0, 0/0, 1e-320, 1e300, 2.25, 1e15 + 0.5}
local out = {}
for round = 1, 500 do
    for i = 1, #values do
        local r = math.sqrt(values[i])
        out[i] = tostring(r)
    end
end
local sum = 0
for round = 1, 3000 do
    for i = 1, 3 do sum = sum + math.sqrt(values[i] + round) end
end
return table.concat(out, ' '), string.format('%.12f', sum)";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn math_sqrt_replaced_in_the_middle_of_the_run() {
    let source = "local real = math.sqrt
local sum = 0
for round = 1, 6000 do
    if round == 3000 then math.sqrt = function(x) return x end end
    for i = 1, 4 do sum = sum + math.sqrt(round + i) end
end
math.sqrt = real
return string.format('%.9f', sum)";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(show(&actual), show(&expected), "{profile:?}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
    }
}
