//! Loops whose conditions compile to compare-and-branch tests record,
//! compile and run natively, and every guard exit resumes the interpreter on
//! the other side of the test.

use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

struct Program {
    name: &'static str,
    source: &'static str,
}

const PROGRAMS: &[Program] = &[
    Program {
        name: "while with a register test",
        source: "local n, limit, s = 0, 3000, 0 while n < limit do n = n + 1 s = s + n end return s",
    },
    Program {
        name: "while with a constant test",
        source: "local n, s = 0, 0 while n < 3000 do n = n + 1 s = s + n % 7 end return s",
    },
    Program {
        name: "repeat until a constant",
        source: "local n, s = 0, 0 repeat n = n + 1 s = s + n % 5 until n >= 3000 return s",
    },
    Program {
        name: "repeat until equality",
        source: "local n, s = 0, 0 repeat n = n + 1 s = s + 1 until n == 3000 return s",
    },
    Program {
        name: "constant on the left",
        source: "local n, s = 0, 0 while 3000 > n do n = n + 1 if 2 < n % 5 then s = s + 1 end end return s",
    },
    Program {
        name: "alternating branch with each operator",
        source: "local lt, le, eq, ne, gt, ge = 0, 0, 0, 0, 0, 0 \
                 for i = 1, 4000 do local m = i % 7 \
                 if m < 3 then lt = lt + 1 end \
                 if m <= 3 then le = le + 1 end \
                 if m == 3 then eq = eq + 1 end \
                 if m ~= 3 then ne = ne + 1 end \
                 if m > 3 then gt = gt + 1 end \
                 if m >= 3 then ge = ge + 1 end end \
                 return lt, le, eq, ne, gt, ge",
    },
    Program {
        name: "negated tests and elseif",
        source: "local a, b, c = 0, 0, 0 \
                 for i = 1, 4000 do local m = i % 10 \
                 if not (m < 4) then a = a + 1 elseif m == 1 then b = b + 1 else c = c + 1 end \
                 if not (m ~= 2) then c = c + 2 end end \
                 return a, b, c",
    },
    Program {
        name: "two registers compared in a nested loop",
        source: "local s = 0 for i = 1, 60 do local j = 0 while j < i do j = j + 1 \
                 if j == i or j * 2 > i then s = s + 1 end end end return s",
    },
    Program {
        name: "float tests with nan",
        source: "local nan, s = 0 / 0, 0 \
                 for i = 1, 3000 do local x = i + 0.5 \
                 if x < nan then s = s + 1000 end \
                 if not (x < nan) then s = s + 1 end \
                 if x ~= x then s = s + 100000 end \
                 if nan == nan then s = s + 1000000 end end return s",
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
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua.exec(source).unwrap();
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

#[test]
fn tests_never_change_what_a_traced_loop_computes() {
    for profile in PROFILES {
        for program in PROGRAMS {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, program.source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, program.source);
            assert_eq!(actual, expected, "{} under {profile:?}", program.name);
        }
    }
}

#[test]
fn loops_with_tests_record_and_run_natively() {
    let mut failures = Vec::new();
    for profile in PROFILES {
        for program in PROGRAMS {
            let (_, stats) = run(profile, ExecutionMode::Tracing, program.source);
            let rejected = stats.recording_aborts[TraceAbortReason::CompileRejected.index()];
            // The alternating branches may refuse one recording as
            // type-unstable; they must still close a trace somewhere.
            if stats.traces_recorded < 1 || stats.native_entries < 1 || rejected > 0 {
                failures.push(format!("{} under {profile:?}: {stats:?}", program.name));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_guard_exit_resumes_on_the_other_side_of_the_test() {
    // The condition flips late, so the trace recorded for one side exits at
    // the test and the interpreter finishes on the other.
    let source = "local s, t = 0, 0 \
                  for i = 1, 6000 do \
                  if i < 5000 then s = s + 1 else t = t + 1 end end \
                  local u = 0 for i = 1, 6000 do if i ~= 5500 then u = u + 1 end end \
                  return s, t, u";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_recorded >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.exits.iter().sum::<u64>() >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}
