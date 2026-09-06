//! The link of a tree into the loop that calls its function, compiled as a
//! function of its own beside the tree (see "Sides compiled apart" in
//! `oximond::vm::jit::trace`). The loop around the call and the nest it calls
//! run as one native loop, and the results are the interpreter's in every
//! profile whatever the nest or the code around the call does while it runs:
//! raises an error that a `pcall` outside the loop catches, runs under a hook
//! set after the link was made, leaves the tree through a branch of the code
//! after the call, or loses its tree to a flush.

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

const CALLS: u64 = 12_000;

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
    // The subject is the side compiled apart: a tree this small would
    // otherwise be compiled again with the side in it.
    lua.vm_mut().keep_jit_sides_apart();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree, and that the nest's calls ran in the loop that makes them
/// through a side compiled apart: the tree was compiled once, a side joined it
/// without compiling it again, and native code was entered for fewer than one
/// call in `ratio`. The integer profiles record again when a number changes
/// its subtype, so they must only agree.
fn assert_linked_apart(source: &str, ratio: u64) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        let float = matches!(
            profile,
            CompatibilityProfile::Lua51
                | CompatibilityProfile::Lua52
                | CompatibilityProfile::LuaJit21
        );
        if float {
            assert!(
                stats.side_compiles >= 1,
                "{profile:?}: no side was compiled apart: {stats:?}"
            );
            assert!(
                stats.native_entries * ratio < CALLS,
                "{profile:?}: {} entries for {CALLS} calls: {stats:?}",
                stats.native_entries
            );
        }
    }
}

/// The function with the nest, and the loop that calls it.
fn program(body: &str, driver: &str) -> String {
    format!(
        "local function sweep(n, m, probe)
             local acc = 0
             for i = 1, n do
                 for j = i, m do
                     {body}
                 end
             end
             return acc
         end
         {driver}"
    )
}

const PLAIN_BODY: &str = "acc = acc + (i * j) % 7";

#[test]
fn a_nest_called_from_a_loop_runs_through_a_side_compiled_apart() {
    assert_linked_apart(
        &program(
            PLAIN_BODY,
            "local total = 0
             for k = 1, 12000 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total",
        ),
        4,
    );
}

#[test]
fn a_branch_after_the_call_leaves_the_tree_inside_the_callers_frame() {
    // The caller keeps several kinds of state in its registers. A branch it
    // takes now and then is an exit of the side, which rebuilds the caller's
    // frame with the registers the run changed since the loop was entered.
    assert_linked_apart(
        &program(
            PLAIN_BODY,
            "local total, marks, flag, count = 0, 0, true, 0
             for k = 1, 12000 do
                 local value = sweep(8 + k % 3, 40)
                 total = (total + value) % 1000003
                 flag = not flag
                 count = count + 1
                 if k % 1500 == 0 then marks = marks + k end
                 if k % 4001 == 0 then total = total + 0.5 end
             end
             return total, marks, flag, count",
        ),
        4,
    );
}

#[test]
fn an_error_in_the_nest_reaches_a_pcall_outside_the_loop_after_the_link() {
    // The error is raised after thousands of calls ran through the side, so
    // the caller's registers the frame holds are the ones the exit writes.
    assert_linked_apart(
        &program(
            "if probe and i == 5 and j == 17 then error('bad ' .. i .. ' ' .. j) end
                     acc = acc + (i * j) % 7",
            "local total = 0
             local messages = {}
             for round = 1, 3 do
                 local ok, message = pcall(function()
                     local sum, r = 0, round
                     for k = 1, 4000 do
                         local fail = (r == 2 and k == 3999) or (r == 3 and k == 3000)
                         sum = (sum + sweep(8 + k % 3, 40, fail)) % 1000003
                     end
                     total = (total + sum) % 1000003
                 end)
                 messages[#messages + 1] = tostring(ok) .. ':' .. tostring(message)
             end
             return total, table.concat(messages, '|')",
        ),
        4,
    );
}

#[test]
fn an_error_in_the_callers_code_after_the_call_reaches_a_pcall_outside_the_loop() {
    // The nest never fails here. The error comes from the code after the call,
    // which the side that returns into the caller runs: its branch is a guard
    // of that side, so the exit that rebuilds the caller's frame is the side's,
    // and the error the interpreter then raises sees the registers it wrote.
    assert_linked_apart(
        &program(
            PLAIN_BODY,
            "local ok, message = pcall(function()
                 local total, count = 0, 0
                 for k = 1, 12000 do
                     total = (total + sweep(8 + k % 3, 40)) % 1000003
                     count = count + 1
                     if k == 9000 then error('late ' .. total .. ' ' .. count) end
                 end
                 return total
             end)
             return tostring(ok) .. ':' .. tostring(message)",
        ),
        4,
    );
}

#[test]
fn a_hook_set_after_the_link_sees_what_the_interpreter_shows() {
    for kind in ["l", "c", "r"] {
        assert_linked_apart(
            &program(
                PLAIN_BODY,
                &format!(
                    "local total, hits = 0, 0
                     for k = 1, 12000 do
                         if k == 9000 then debug.sethook(function() hits = hits + 1 end, '{kind}') end
                         if k == 9004 then debug.sethook() end
                         total = (total + sweep(8 + k % 3, 40)) % 1000003
                     end
                     return total, hits"
                ),
            ),
            4,
        );
    }
}

#[test]
fn a_count_hook_set_after_the_link_stops_after_its_budget() {
    for count in [1, 7, 100] {
        assert_linked_apart(
            &program(
                PLAIN_BODY,
                &format!(
                    "local total, hits = 0, 0
                     for k = 1, 12000 do
                         if k == 9000 then debug.sethook(function() hits = hits + 1 end, '', {count}) end
                         if k == 9010 then debug.sethook() end
                         total = (total + sweep(8 + k % 3, 40)) % 1000003
                     end
                     return total, hits"
                ),
            ),
            // The hook flushes the trees, which record and link again.
            2,
        );
    }
}

#[test]
fn a_call_that_returns_before_its_loop_runs_ends_the_recording_at_its_second_return() {
    // `n` is zero every seventh call, and the loops of `partial` do not run:
    // the activation returns into the caller again before it reaches the loop
    // the tree was made for. A recording that followed the return into the
    // caller and the call after it would span two activations, each with a
    // caller frame, and the exits of the tree rebuilt the frames wrong (the
    // sum came out 54.238 where the interpreter's is 52.880 with the link
    // from 200 entries on). The recording ends at the second return.
    for profile in PROFILES {
        let source = "local bodies = {}
             for i = 1, 6 do
                 bodies[i] = {x = i * 1.5, y = i * i * 0.25, vx = 0.01 * i, mass = 1.0 + i * 0.5}
             end
             local function partial(dt, n)
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
             return string.format('%.9f', sum)";
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
    }
}
