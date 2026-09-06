//! Hot code that the Luanti emerge ran in the interpreter although a trace
//! owned it. Two mechanisms, each with the loop that showed it:
//!
//! - A side that starts at a call whose callee register is reloaded from the
//!   same local or upvalue on every iteration is guarded on the callee's
//!   identity like the root's call is, not on its prototype. The prototype
//!   guard reads the callee's cells through the live closure, which cannot
//!   hold a function, so every side that inlined a callee reading a function
//!   upvalue aborted with `UnsupportedOperand`.
//! - A tree whose compile of a queued side is held back for a guard about to
//!   ask for its own side counts the entries the run refuses meanwhile as
//!   entries of the wait. They were refused for the queued side, so the hold
//!   never ended and the loop stayed in the interpreter.
//!
//! Every source runs in each profile traced and interpreted and must agree.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const FLOOR_SWITCH: u64 = 1000;
const FLOOR_REPLACED: u64 = 1000;

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

/// Runs `source` traced and interpreted in every profile, checks they agree
/// and returns the traced runs' statistics text and counters
/// `(profile, side_aborts, side_installs, iterations)`.
fn run_all(source: &str) -> Vec<(CompatibilityProfile, u64, u64, u64, String)> {
    run_all_with_exits(source)
        .into_iter()
        .map(|(profile, aborts, installs, iterations, _, text)| {
            (profile, aborts, installs, iterations, text)
        })
        .collect()
}

/// Like [`run_all`], with the native entries and the exits of all kinds.
type Run = (CompatibilityProfile, u64, u64, u64, (u64, u64), String);

fn run_all_with_exits(source: &str) -> Vec<Run> {
    PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            let text = format!("{stats:?}");
            (
                profile,
                stats.side_aborts,
                stats.side_installs,
                stats.iterations,
                (stats.native_entries, stats.exits.iter().sum::<u64>()),
                text,
            )
        })
        .collect()
}

/// A callee moved into the call base from a local, whose own callees are
/// upvalues holding functions, with a branch that goes the other way now and
/// then: the guard has a side that starts at the call.
const RELOADED_CALLEE: &str = "
    local gn = {}
    local ssum
    local function sample(i, s)
        local v = gn[i]
        -- `0.0 % s` (zero) keeps the arm impure: a value select would
        -- leave the branch no guard and so no side.
        ssum = ssum + (v ~= 0 and 1.0 or 0.0 % s) * s
    end
    local function density(i)
        ssum = 0.0
        sample(i, 0.5)
        sample(i + 1, 0.25)
        sample(i + 2, 0.25)
    end
    local function augmented(i)
        density(i)
        return ssum, i
    end
    for i = 1, 4000 do gn[i] = (i % 97 < 60) and 1 or 0 end
    local total = 0
    for rep = 1, 300 do
        for y = 1, 3900 do
            local d, k = augmented(y)
            if d > 0.9 then total = total + 1 end
            gn[y] = gn[y]
        end
    end
    return total";

#[test]
fn a_side_at_a_call_of_a_reloaded_callee_inlines_function_upvalues() {
    for (profile, aborts, installs, iterations, text) in run_all(RELOADED_CALLEE) {
        assert_eq!(aborts, 0, "{profile:?} {text}");
        assert!(installs >= 1, "{profile:?} {text}");
        // 1.17M trips in all; the interpreter runs the few before the tree
        // and its sides are in place.
        assert!(iterations >= 900_000, "{profile:?} {iterations} {text}");
    }
}

/// The same call, but the callee changes between two closures of one
/// prototype halfway through: the identity guard of the call leaves to the
/// interpreter, and the result is the interpreter's.
#[test]
fn a_reloaded_callee_that_changes_still_agrees_with_the_interpreter() {
    let source = "
        local function make(k) return function(i) return i * k end end
        local f = make(2)
        local gn = {}
        for i = 1, 3000 do gn[i] = i % 5 end
        local total = 0
        for rep = 1, 60 do
            if rep == 30 then f = make(3) end
            for y = 1, 2900 do
                local g = f
                local v = g(gn[y])
                if v > 6 then total = total + 1 else total = total - 1 end
            end
        end
        return total";
    for (profile, _, _, iterations, (entries, exits), text) in run_all_with_exits(source) {
        // Natively before the switch, and the call guard leaves at it.
        assert!(entries >= 1, "{profile:?} {text}");
        assert!(
            iterations >= FLOOR_SWITCH,
            "{profile:?} {iterations} {text}"
        );
        assert!(exits >= 1, "{profile:?} {text}");
    }
}

/// A callee that errors after the loop has run natively, and a callee
/// replaced by a table with `__call`: both leave through the guard.
#[test]
fn a_reloaded_callee_replaced_by_a_non_function_leaves_the_trace() {
    let source = "
        local function f(i) return i + 1 end
        local h = f
        local total = 0
        local ok, err = pcall(function()
            for y = 1, 20000 do
                if y == 15000 then h = setmetatable({}, {__call = function(_, i) return i * 2 end}) end
                if y == 19000 then h = nil end
                local g = h
                total = total + g(y)
            end
        end)
        return total, ok, (tostring(err):gsub('^.-:%d+: ', ''))";
    for (profile, _, _, iterations, (entries, exits), text) in run_all_with_exits(source) {
        assert!(entries >= 1, "{profile:?} {text}");
        assert!(
            iterations >= FLOOR_REPLACED,
            "{profile:?} {iterations} {text}"
        );
        // The table with `__call` and then the nil leave through the guard.
        assert!(exits >= 1, "{profile:?} {text}");
    }
}

/// A nest whose inner loop calls a function with a data-dependent branch:
/// the inner root's tree queues sides while its entries are refused for the
/// first of them. The loop must not stay in the interpreter.
const HELD_NEST: &str = "
    local chunk = 24
    local gn, beard = {}, {}
    local function index(x, y, z) return (x * chunk + y) * chunk + z + 1 end
    local ssum, bsum
    local function sample(x, y, z, s)
        local i = index(x, y, z)
        ssum = ssum + (gn[i] ~= 0 and 1.0 or 0.0) * s
        bsum = bsum + beard[i] * s
    end
    local function density(x, y, z)
        ssum, bsum = 0.0, 0.0
        sample(math.max(0, x - 1), y, z, 0.25)
        sample(x, math.max(0, y - 1), z, 0.25)
        sample(x, y, math.max(0, z - 1), 0.25)
        sample(x, y, z, 1.0)
        sample(math.min(chunk - 1, x + 1), y, z, 0.25)
        sample(x, math.min(chunk - 1, y + 1), z, 0.25)
        sample(x, y, math.min(chunk - 1, z + 1), 0.25)
    end
    for x = 0, chunk - 1 do for y = 0, chunk - 1 do for z = 0, chunk - 1 do
        local i = index(x, y, z)
        local h = 10 + (x * 7 + z * 3) % 5
        gn[i] = y < h and 1 or 0
        beard[i] = (y > h - 3 and y < h + 3) and ((x + z) % 3 - 1) * 0.5 or 0.0
    end end end
    local changed = 0
    for rep = 1, 6 do
        for x = 0, chunk - 1 do
            for z = 0, chunk - 1 do
                for y = chunk - 1, 0, -1 do
                    density(x, y, z)
                    if math.abs(bsum) > 0.1 and ssum < 0.8 and gn[index(x, y, z)] ~= 0 then
                        gn[index(x, y, z)] = 0
                        changed = changed + 1
                    end
                end
            end
        end
    end
    return changed";

#[test]
fn a_nest_with_a_queued_side_does_not_wait_on_refused_entries() {
    for (profile, _, _, iterations, text) in run_all(HELD_NEST) {
        // 83k trips in all. With the refused entries counted the loop runs
        // 13k trips or more natively in every profile; with the count
        // removed (checked in a scratch build) the held wait never ends and
        // it runs about 150.
        assert!(iterations >= 10_000, "{profile:?} {iterations} {text}");
    }
}

/// The shape of `form_terrain` in Mineclonia's `mg_ersatz.lua`: a stateful
/// iterator drives an outer loop whose body runs an inner numeric loop that
/// calls functions held in upvalues. A root recording of the outer loop reads
/// those callees through `GetUpvalue` and must pin them as it always did; the
/// identity guard for a reloaded callee is for sides only.
#[test]
fn an_iterator_loop_over_an_inner_loop_calling_upvalue_functions_agrees() {
    let source = "
        local chunk = 16
        local gn = {}
        local function index(x, y, z) return (x * chunk + y) * chunk + z + 1 end
        local function density(x, y, z) return ((x + y * 3 + z * 5) % 7) * 0.2 - 0.5 end
        local function iter(x2, z2)
            local x, z = -1, 0
            return function()
                x = x + 1
                if x > x2 then x = 0; z = z + 1 end
                if z > z2 then return nil end
                return x, 0, z
            end
        end
        for i = 1, chunk * chunk * chunk do gn[i] = 1 end
        local changed = 0
        for rep = 1, 40 do
            for x, _, z in iter(chunk - 1, chunk - 1) do
                for y = chunk - 1, 0, -1 do
                    local i = index(x, y, z)
                    local d = density(x, y, z)
                    if math.abs(d) > 0.4 and gn[i] ~= 0 then
                        gn[i] = 0
                        changed = changed + 1
                    elseif d < -0.3 then
                        gn[i] = 1
                    end
                end
            end
        end
        return changed";
    for (profile, _, _, iterations, (entries, _), text) in run_all_with_exits(source) {
        assert!(entries >= 1, "{profile:?} {text}");
        assert!(iterations >= 1000, "{profile:?} {iterations} {text}");
    }
}
