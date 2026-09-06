//! Sides that start inside inlined calls. A guard that fails in a callee the
//! root inlined, after the call wrote an upvalue cell, leaves through the
//! callee's frames, and the exit pushes those frames for the interpreter. A
//! side continues such a guard: it rebuilds the calls as inlined frames of
//! its own, reads their registers as frame inputs and records on from where
//! the guard failed. The results are the interpreter's in every profile.

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

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree and are not errors, and returns the traced run's statistics
/// by profile.
fn run_all(source: &str) -> Vec<(CompatibilityProfile, oximond::vm::TraceStatistics)> {
    PROFILES
        .into_iter()
        .map(|profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            assert!(expected.is_ok(), "{profile:?} {expected:?}");
            (profile, lua.vm().jit_statistics().trace)
        })
        .collect()
}

/// Checks that the callee's branch took a side that starts inside the call
/// and that hardly any iteration leaves through the call's frames any more.
fn assert_inner_side(profile: CompatibilityProfile, stats: &oximond::vm::TraceStatistics) {
    assert!(stats.inner_sides >= 1, "{profile:?} {stats:?}");
    assert!(stats.iterations >= 100_000, "{profile:?} {stats:?}");
    assert!(
        stats.inner_exits * 20 < stats.iterations,
        "{profile:?}: {} inner exits in {} iterations: {stats:?}",
        stats.inner_exits,
        stats.iterations
    );
}

#[test]
fn a_branch_after_an_upvalue_write_in_a_callee_takes_an_inner_side() {
    let source = "local acc = 0
         local function f(x)
           acc = acc + x
           if x % 3 == 0 then acc = acc * 2 % 1000003 end
           return acc
         end
         local s = 0
         for i = 1, 200000 do s = (s + f(i)) % 1000003 end
         return s, acc";
    for (profile, stats) in run_all(source) {
        assert_inner_side(profile, &stats);
    }
}

#[test]
fn a_branch_in_the_second_of_two_calls_takes_an_inner_side_through_both_frames() {
    // `g` calls `f` twice: the first call writes the cell, and the branch is
    // in the second, two calls deep.
    let source = "local acc = 0
         local function f(x, check)
           acc = acc + x
           if check and x % 3 == 0 then acc = acc * 2 % 1000003 end
           return acc
         end
         local function g(x)
           local a = f(x, false)
           local b = f(x + 1, true)
           return a + b
         end
         local s = 0
         for i = 1, 200000 do s = (s + g(i)) % 1000003 end
         return s, acc";
    for (profile, stats) in run_all(source) {
        assert_inner_side(profile, &stats);
    }
}

#[test]
fn a_side_of_an_inner_side_starts_inside_the_call_too() {
    // The side of `x % 3 == 0` meets `x % 5 == 0`, which fails a third of its
    // runs inside the same call: that guard takes a side of its own.
    let source = "local acc = 0
         local function f(x)
           acc = acc + x
           if x % 3 == 0 then
             acc = acc * 2 % 1000003
             if x % 5 == 0 then acc = acc + 7 end
           end
           return acc
         end
         local s = 0
         for i = 1, 300000 do s = (s + f(i)) % 1000003 end
         return s, acc";
    for (profile, stats) in run_all(source) {
        assert_inner_side(profile, &stats);
        assert!(stats.inner_sides >= 2, "{profile:?} {stats:?}");
        assert!(stats.side_installs >= 2, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_error_after_an_inner_side_exit_has_the_interpreters_message_and_traceback() {
    // The side of `x % 3 == 0` exits at `x == 150000` inside `f`, and the
    // interpreter raises the error in the frames the exit pushed.
    let source = "local acc = 0
         local function f(x)
           acc = acc + x
           if x % 3 == 0 then
             acc = acc * 2 % 1000003
             if x == 150000 then error('stop at ' .. x .. ' with ' .. acc) end
           end
           return acc
         end
         local function run()
           local s = 0
           for i = 1, 200000 do s = (s + f(i)) % 1000003 end
           return s
         end
         local message
         local ok = xpcall(run, function(m)
           message = m
           return debug.traceback(m)
         end)
         local _, trace = xpcall(run, debug.traceback)
         return ok, message, trace, acc";
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        let stats = lua.vm().jit_statistics().trace;
        let expected = expected.unwrap();
        let actual = actual.unwrap();
        assert_eq!(
            expected.first(),
            Some(&Value::Boolean(false)),
            "{profile:?}"
        );
        // The message, then the traceback line by line.
        assert_eq!(actual.get(1), expected.get(1), "{profile:?}");
        let lines = |values: &[Value]| match values.get(2) {
            Some(Value::String(text)) => String::from_utf8_lossy(text.as_bytes())
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            other => panic!("{profile:?}: no traceback: {other:?}"),
        };
        assert_eq!(lines(&actual), lines(&expected), "{profile:?}");
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.inner_sides >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn protected_and_vararg_callees_take_no_inner_side() {
    // A guard inside a `pcall` or inside a vararg callee leaves through the
    // call's frames, but no side starts there.
    let protected = "local acc = 0
         local function f(x)
           acc = acc + x
           if x % 3 == 0 then acc = acc * 2 % 1000003 end
           return acc
         end
         local s = 0
         for i = 1, 200000 do
           local ok, v = pcall(f, i)
           s = (s + v) % 1000003
         end
         return s, acc";
    let vararg = "local acc = 0
         local function f(...)
           local x = ...
           acc = acc + x
           if x % 3 == 0 then acc = acc * 2 % 1000003 end
           return acc
         end
         local s = 0
         for i = 1, 200000 do s = (s + f(i)) % 1000003 end
         return s, acc";
    for source in [protected, vararg] {
        for (profile, stats) in run_all(source) {
            assert_eq!(stats.inner_sides, 0, "{profile:?} {stats:?} {source}");
        }
    }
}
