//! Entering a trace again and again over tables that change between entries.
//!
//! A trace keeps what it learned about the tables of its last entry (their
//! lanes, the offsets of their fields, the rows it built from them) and must
//! not trust it once the tables are not what they were. Every program here
//! calls a function with a hot inner loop many times and changes the tables
//! between calls: replaces one, resizes the array or a hash part, gives one
//! a metatable, drops one for the collector, changes the type of a field.
//! Interpreted and traced runs must give the same results in every profile.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
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

/// Runs `source` interpreted and traced in every profile; the outcomes must
/// agree, and the traced run must have entered native code.
fn assert_traced(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {expected:?}\n{source}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}\n{source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}\n{source}");
    }
}

/// A function that sums fields of every record of `records`, with a loop
/// that is hot from the first call on, then `rounds` calls with `change` run
/// before the call that has `round` as its number.
fn program(setup: &str, change: &str, rounds: u32) -> String {
    format!(
        "
local function record(n)
    return {{x = n * 1.5, y = n * 0.25, z = n * 0.125, mass = 1.0 + n}}
end
{setup}
local function sum(records, scale)
    local total = 0
    for i = 1, #records do
        local r = records[i]
        for k = 1, 3 do
            total = total + (r.x + r.y * scale) * (r.mass or 2) - r.z
            r.z = r.z + 0.0625
        end
    end
    return total
end
local out = {{}}
for round = 1, {rounds} do
    {change}
    out[#out + 1] = string.format('%.6f', sum(bodies, 0.5 + round * 0.01))
end
return table.concat(out, ' ')
"
    )
}

const BODIES: &str = "local bodies = {} for i = 1, 5 do bodies[i] = record(i) end";

#[test]
fn a_table_replaced_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 7 == 0 then bodies[round % 5 + 1] = record(round) end",
        120,
    ));
}

#[test]
fn a_table_replaced_and_collected_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 5 == 0 then bodies[2] = record(round) collectgarbage() end \
         if round % 11 == 0 then local junk = {} for i = 1, 200 do junk[i] = {i} end collectgarbage() end",
        120,
    ));
}

#[test]
fn the_array_grows_and_shrinks_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 9 == 0 then bodies[#bodies + 1] = record(round) end \
         if round % 13 == 0 and #bodies > 3 then table.remove(bodies, 1) end",
        140,
    ));
}

#[test]
fn a_hash_part_gains_a_field_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 8 == 0 then bodies[round % 5 + 1]['extra' .. round] = round end",
        120,
    ));
}

#[test]
fn a_field_changes_its_type_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 6 == 0 then bodies[3].x = round end \
         if round % 6 == 3 then bodies[3].x = round + 0.5 end",
        120,
    ));
}

#[test]
fn a_field_is_removed_and_restored_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round % 10 == 4 then bodies[4].mass = nil end \
         if round % 10 == 6 then bodies[4].mass = 2.5 end",
        120,
    ));
}

#[test]
fn a_table_gets_a_metatable_between_entries() {
    assert_traced(&program(
        BODIES,
        "if round == 30 then setmetatable(bodies[2], {__index = function() return 1 end}) end \
         if round == 70 then setmetatable(bodies[2], nil) end",
        120,
    ));
}

#[test]
fn a_trace_entered_from_two_call_sites_with_different_tables() {
    let source = "
local function record(n)
    return {x = n * 1.5, y = n * 0.25, z = n * 0.125, mass = 1.0 + n}
end
local left, right = {}, {}
for i = 1, 4 do left[i] = record(i) end
for i = 1, 6 do right[i] = record(i + 10) end
local function sum(records)
    local total = 0
    for i = 1, #records do
        local r = records[i]
        for k = 1, 3 do
            total = total + r.x * r.mass + r.y - r.z
            r.y = r.y + 0.03125
        end
    end
    return total
end
local out = {}
for round = 1, 300 do
    if round % 2 == 0 then
        out[#out + 1] = string.format('%.6f', sum(left))
    else
        out[#out + 1] = string.format('%.6f', sum(right))
    end
    if round % 25 == 0 then left, right = right, left end
end
return table.concat(out, ' ')
";
    assert_traced(source);
}

#[test]
fn an_exit_right_after_entry() {
    // The first record's field is a string on some calls: the trace's guard
    // on it fails at the first load of the loop.
    assert_traced(&program(
        BODIES,
        "if round % 4 == 0 then bodies[1].mass = 3 end \
         if round % 4 == 2 then bodies[1].mass = 3.5 end \
         if round % 17 == 0 then bodies[1].z = 0 end",
        120,
    ));
}

#[test]
fn a_nest_over_bodies_with_tables_changing_between_calls() {
    let source = format!(
        "{}
local function advance(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx, dy = first.x - second.x, first.y - second.y
            local d2 = dx * dx + dy * dy + 0.01
            local m = dt / (d2 * math.sqrt(d2))
            first.vx, first.vy = first.vx - dx * second.mass * m, first.vy - dy * second.mass * m
            second.vx, second.vy = second.vx + dx * first.mass * m, second.vy + dy * first.mass * m
        end
    end
    for i = 1, #bodies do
        local b = bodies[i]
        b.x, b.y = b.x + dt * b.vx, b.y + dt * b.vy
    end
end
for round = 1, 200 do
    advance(0.01)
    if round % 37 == 0 then bodies[2] = {{x = round, y = 1.5, vx = 0, vy = 0, mass = 2.0}} end
    if round % 53 == 0 then bodies[#bodies + 1] = {{x = 9, y = round * 0.5, vx = 0, vy = 0, mass = 1.0}} end
    if round % 41 == 0 then bodies[3].mass = 4 end
    if round % 43 == 0 then bodies[3].mass = 4.5 end
end
local total = 0
for i = 1, #bodies do local b = bodies[i] total = total + b.x + b.y + b.vx + b.vy end
return string.format('%.9f', total)
",
        "local bodies = {}
for i = 1, 5 do
    bodies[i] = {x = i * 1.5, y = i * i * 0.25, vx = 0.01 * i, vy = -0.02, mass = 1.0 + i * 0.5}
end"
    );
    assert_traced(&source);
}

/// Closures and tables loaded in one loop: a closure's handle word can equal
/// the identity of an admitted table, and must not get that table's lane.
#[test]
fn closure_array_loads_do_not_hit_table_lanes() {
    assert_traced(
        "local tabs, fns = {}, {} \
         for i = 1, 60 do tabs[i] = {v = i}; fns[i] = function(x) return x + i end end \
         local function run(ts, fs, n) \
           local s, last = 0, nil \
           for i = 1, n do local t = ts[i]; local f = fs[i]; s = s + t.v; last = f end \
           return s, last end \
         local bad = 0 \
         for r = 1, 400 do local n = r % 60 + 1; local _, last = run(tabs, fns, n) \
           if last ~= fns[n] then bad = bad + 1 end end \
         return bad",
    );
}
