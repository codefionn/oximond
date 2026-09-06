//! The shortcuts the native backend takes in the bounds-check-free version
//! of a counted loop, against the interpreter: the loop's control value and
//! its visible variable share one register, an integer loop drops the
//! overflow test of its advance, and a float key the loop never reads as a
//! float is computed at the exits only. Each case also has loops that leave
//! that version, or never enter it, with the state those shortcuts skip.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL: [CompatibilityProfile; 6] = [
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

/// Runs `source` in both modes of every profile: the results agree, and
/// the tracing run entered native code.
fn assert_agrees(source: &str) {
    assert_agrees_in(&ALL, source, true);
}

/// [`assert_agrees`] for some profiles. With `native` unset the tracing run
/// need not have entered native code: the recorder may decline the loop.
fn assert_agrees_in(profiles: &[CompatibilityProfile], source: &str, native: bool) {
    for &profile in profiles {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(
            !native || stats.native_entries > 0,
            "{profile:?} {source} {stats:?}"
        );
    }
}

#[test]
fn a_loop_variable_assigned_in_the_body_keeps_its_own_value() {
    // The visible variable differs from the control value inside the
    // iteration and equals it again at the next one. Lua 5.5 makes the
    // variable read-only.
    assert_agrees_in(
        &ALL[..4],
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, last = 0, 0
        for round = 1, 40 do
            for i = 1, n do
                local v = values[i]
                i = i * 2 + round
                total = total + v + i
                last = i
            end
        end
        return total, last
        ",
        true,
    );
    assert_agrees_in(
        &ALL[5..],
        r"
        local values = {}
        for i = 1, 300 do values[i] = i % 97 end
        local total = 0
        for round = 1, 40 do
            for i = 1, 300 do
                local v = values[i]
                i = i * 2 + round
                total = total + v + i
            end
        end
        return total
        ",
        true,
    );
}

#[test]
fn exits_of_the_unchecked_loop_restore_the_control_value_and_the_variable() {
    // An element of another type ends the native loop in the middle of the
    // array; the interpreter goes on from the control value and the variable.
    for odd in ["'7'", "2.5", "false"] {
        let source = r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, seen = 0, 0
        for round = 1, 60 do
            if round == 45 then values[170] = ODD end
            if round == 50 then values[170] = 3 end
            for i = 1, n do
                local v = values[i]
                if v then
                    total = total + v + i
                else
                    seen = seen + i
                end
            end
        end
        return total, seen
        "
        .replace("ODD", odd);
        assert_agrees(&source);
    }
    // The loop ends early through a `break`, with the variable read after.
    assert_agrees(
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, stops = 0, 0
        for round = 1, 60 do
            local stop = 0
            for i = 1, n do
                local v = values[i]
                total = total + v
                if v + i > 250 + round then stop = i break end
            end
            stops = stops + stop
        end
        return total, stops
        ",
    );
}

#[test]
fn integer_loops_at_the_edge_of_the_integer_range_end_as_the_interpreter_ends_them() {
    // The limit is the largest integer, so the advance would overflow: the
    // loop ends there. Reads of keys that far out find nothing. A profile
    // without integers counts to 2^52: at 2^53 adding one changes nothing
    // and the loop never ends, in stock Lua 5.1 as well.
    assert_agrees_in(
        &ALL,
        r"
        local top = math.maxinteger or 2^52
        local values = {}
        for i = 1, 64 do values[i] = i end
        local total, trips = 0, 0
        for round = 1, 300 do
            for i = top - 3, top do
                total = total + (values[i] or 1)
                trips = trips + 1
            end
            for i = 1, 64 do total = total + values[i] end
        end
        return total, trips
        ",
        false,
    );
    // A step so large that one advance leaves the range, over keys in the
    // array, and the same loop with a step that stays.
    assert_agrees_in(
        &ALL,
        r"
        local huge = math.maxinteger and (math.maxinteger - 1) or 2^40
        local values = {}
        for i = 1, 64 do values[i] = i end
        local total, trips = 0, 0
        for round = 1, 400 do
            local step = round % 5 == 0 and huge or 1
            for i = 1, 64, step do
                total = total + values[i]
                trips = trips + 1
            end
        end
        return total, trips
        ",
        false,
    );
    // Downward, to the smallest integer.
    assert_agrees_in(
        &ALL,
        r"
        local bottom = math.mininteger or -2^52
        local values = {}
        for i = 1, 64 do values[i] = i end
        local total, trips = 0, 0
        for round = 1, 300 do
            for i = bottom + 3, bottom, -1 do
                total = total + (values[i] or 1)
                trips = trips + 1
            end
            for i = 64, 1, -1 do total = total + values[i] end
        end
        return total, trips
        ",
        false,
    );
}

#[test]
fn float_keys_read_only_by_exits_are_whole_at_the_exits() {
    // `i - 1.0` keys an array in a float loop. The key is a local that the
    // rare branch and the code after the loop read.
    assert_agrees(
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, marks, last = 0, 0, 0
        for round = 1, 60 do
            for i = 2.0, n do
                local k = i - 1.0
                local v = values[k]
                values[k] = (v + round) % 101
                total = total + v
                if v == 100 then marks = marks + k end
                last = k
            end
        end
        return total, marks, last
        ",
    );
    // A hole opens in the middle: the loop leaves with the key live.
    assert_agrees(
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, holes = 0, 0
        for round = 1, 60 do
            if round == 40 then values[120] = nil end
            if round == 50 then values[120] = 5 end
            for i = 2.0, n do
                local k = i - 1.0
                local v = values[k]
                if v then
                    total = total + v
                else
                    holes = holes + k
                end
            end
        end
        return total, holes
        ",
    );
    // The key is used as a number too, so it is computed in the loop.
    assert_agrees(
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total = 0
        for round = 1, 40 do
            for i = 2.0, n, 1.5 do
                local k = i - 1.0
                local v = values[k]
                total = total + (v or 0.25) + k
            end
        end
        return total
        ",
    );
}
