//! Exits of a tree that share their stores, and fields a loop reads and then
//! writes: the compiled code computes each table row's address once, reads
//! each field's slot once past its guard, and sends the exits of one tree
//! through shared tail blocks even where the loop's values have changed name
//! (the loop parameters arrive as block parameters after the first guards).
//!
//! The program is the shape of n-body's `advance`: a nest over a table of
//! bodies with seven fields read and written in the inner loop. Midway through
//! the run one field of one body is replaced by a value that makes the guard
//! of that field's access leave the trace: nil, a number written as text, a
//! table, a boolean. Every field gets each replacement, so every guard of the
//! tree is the one that leaves in some program, and the exit has to restore
//! the registers it names whether it stores them itself or through a tail.
//! Interpreted and traced runs in every profile must agree, errors included.

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

const FIELDS: [&str; 7] = ["x", "y", "z", "vx", "vy", "vz", "mass"];

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
/// agree. With `entered`, the traced run must also have entered native code.
fn assert_agree(source: &str, entered: bool) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        if entered {
            let stats = lua.vm().jit_statistics().trace;
            assert!(stats.native_entries > 0, "{profile:?} {stats:?} {source}");
        }
    }
}

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

/// The program: `rounds` calls of `advance`, with `body.field` set to
/// `replacement` before round `at`; the result is the sum of the numeric
/// fields, or the error the run ended with.
fn program(body: usize, field: &str, replacement: &str, at: usize, rounds: usize) -> String {
    format!(
        "{}{}
local ok, err = pcall(function()
    for round = 1, {rounds} do
        if round == {at} then bodies[{body}].{field} = {replacement} end
        advance(0.01)
    end
end)
local total = 0
for i = 1, #bodies do
    for _, key in ipairs({{'x', 'y', 'z', 'vx', 'vy', 'vz', 'mass'}}) do
        local value = bodies[i][key]
        if type(value) == 'number' then total = total + value end
    end
end
return ok, tostring(err), string.format('%.9f', total)",
        bodies(5),
        ADVANCE,
    )
}

#[test]
fn every_field_of_a_body_may_leave_the_trace_through_its_own_guard() {
    for field in FIELDS {
        for replacement in ["nil", "'0.25'", "{}", "true"] {
            // The first body is the outer loop's `first` and a `second` for
            // none; the last is a `second` only. Either way each field is
            // read, and written, by a different guard of the tree.
            for body in [1, 5] {
                let source = program(body, field, replacement, 700, 1200);
                assert_agree(&source, true);
            }
        }
    }
}

#[test]
fn a_field_read_then_written_in_one_iteration_leaves_between_the_two() {
    // `first.vx` is read, then a branch on the data changes what the write
    // after it finds: the table loses the field (so the write is an insert
    // through the VM), or the field turns into a string. The slot read before
    // the branch must not be used after it.
    for change in [
        "first.vx = nil",
        "first.vx = 'text'",
        "first.vx = {}",
        "first.extra = 1",
        "first.vx = nil first.vx = 1.5",
        "rawset(first, 'vx', nil)",
        "setmetatable(first, {__newindex = function(t, k, v) rawset(t, k, v + 1) end})",
    ] {
        let source = format!(
            "{}
local function advance(dt)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            local dx = first.x - second.x
            local before = first.vx
            if round_now == 300 and i == 2 and j == 3 then {change} end
            first.vx = before - dx * dt
            second.vx = second.vx + dx * dt
        end
    end
end
round_now = 0
local ok, err = pcall(function()
    for round = 1, 600 do round_now = round advance(0.01) end
end)
local total = 0
for i = 1, #bodies do
    local vx = bodies[i].vx
    if type(vx) == 'number' then total = total + vx end
end
return ok, tostring(err), string.format('%.9f', total)",
            bodies(5),
        );
        assert_agree(&source, false);
    }
}

#[test]
fn two_bodies_share_no_slot_even_when_they_swap_places() {
    // The row address and the field slot are kept per table. Two tables that
    // trade the roles of `first` and `second` between iterations must not
    // see each other's slots, and a table that is swapped for another of the
    // same shape in the array must be read afresh.
    let source = format!(
        "{}
local function advance(dt, flip)
    for i = 1, #bodies do
        local first = bodies[i]
        for j = i + 1, #bodies do
            local second = bodies[j]
            if flip then first, second = second, first end
            local dx = first.x - second.x
            first.vx = first.vx - dx * dt
            second.vx = second.vx + dx * dt
            local again = first.vx - second.vx
            first.vy = first.vy + again * dt
        end
    end
end
for round = 1, 1500 do
    advance(0.01, round % 3 == 0)
    if round == 800 then bodies[2], bodies[4] = bodies[4], bodies[2] end
end
local total = 0
for i = 1, #bodies do total = total + bodies[i].vx + bodies[i].vy end
return string.format('%.9f', total)",
        bodies(5),
    );
    assert_agree(&source, true);
}

/// Runs `source` traced in every profile and checks that it agrees with the
/// interpreter, that the tree took a side trace and that no side was refused:
/// a lowering that reused a value the side defined in the code after the
/// guard would be refused by Cranelift's verifier (on in debug builds), the
/// tree would not compile, and its side would count as rejected.
fn assert_agree_with_a_side(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?} {source}");
        assert!(stats.side_installs > 0, "{profile:?} {stats:?} {source}");
        assert_eq!(stats.side_rejections, 0, "{profile:?} {stats:?} {source}");
    }
}

#[test]
fn what_a_side_trace_reads_does_not_serve_the_code_after_its_guard() {
    // The branch alternates, so its guard gets a side trace that reads and
    // writes fields the main path does not touch before the branch. The code
    // after the branch reads them again, in the main trace: it must load
    // their slots itself, since the side's loads do not dominate it.
    let source = "
local pool = {
    {a = 1.5, b = 2.5, c = 0.5, d = 4.0},
    {a = 0.5, b = 1.5, c = 2.5, d = 1.0},
    {a = 0.25, b = 3.5, c = 1.5, d = 2.0},
}
for round = 1, 1500 do
    local t = pool[1 + round % 3]
    local u = pool[1 + (round + 1) % 3]
    for i = 1, 60 do
        local x = t.a + u.a
        if i % 3 == 0 then
            t.b = t.b + x * 0.001
            u.c = u.c - t.d * 0.001
            t.d = t.d + u.d * 0.0001
        else
            u.b = u.b + t.c * 0.001
        end
        t.b = t.b * 0.99999
        u.c = u.c * 1.00001
        t.d = t.d + u.a * 0.0001
        t.c = t.c + u.a * 0.001 + x * 0.0001
    end
end
local total = 0
for k = 1, 3 do total = total + pool[k].b + pool[k].c + pool[k].d end
return string.format('%.9f', total)";
    assert_agree_with_a_side(source);
}
