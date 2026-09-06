//! Loops that end in `ForInLoop` and call their iterator with `ForInCall`
//! record, compile and run natively like the copies, call and nil test they
//! replace, and guard exits resume the interpreter in the right place.

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

/// Loops over arrays with `ipairs`, which a trace runs as array reads.
const TRACED: &[Program] = &[
    Program {
        name: "ipairs sum",
        source: "local t = {} for i = 1, 200 do t[i] = i end \
                 local s = 0 for rep = 1, 100 do for _, v in ipairs(t) do s = s + v end end return s",
    },
    Program {
        name: "ipairs index and value",
        source: "local t = {} for i = 1, 200 do t[i] = i * 0.5 end \
                 local s = 0 for rep = 1, 100 do for i, v in ipairs(t) do s = s + i * v end end return s",
    },
    Program {
        name: "ipairs with a break",
        source: "local t = {} for i = 1, 300 do t[i] = i end \
                 local s = 0 for rep = 1, 60 do for i, v in ipairs(t) do \
                 if v > 250 then break end s = s + v end end return s",
    },
    Program {
        name: "nested ipairs",
        source: "local g = {} for i = 1, 30 do g[i] = {} for j = 1, 30 do g[i][j] = i + j end end \
                 local s = 0 for rep = 1, 20 do for _, row in ipairs(g) do \
                 for _, v in ipairs(row) do s = s + v end end end return s",
    },
    Program {
        name: "ipairs in a called function",
        source: "local t = {} for i = 1, 100 do t[i] = i end \
                 local function total(x) local s = 0 for _, v in ipairs(x) do s = s + v end return s end \
                 local s = 0 for rep = 1, 300 do s = s + total(t) end return s",
    },
];

/// Loops whose traces the iterator or the data cuts short; they only have to
/// compute the same results.
const MIXED: &[Program] = &[
    Program {
        name: "an element changes type",
        source: "local t = {} for i = 1, 400 do t[i] = i end t[390] = 0.5 \
                 local s = 0 for rep = 1, 30 do for _, v in ipairs(t) do s = s + v end end return s",
    },
    Program {
        name: "a hole ends the loop",
        source: "local t = {} for i = 1, 400 do t[i] = i end t[200] = nil \
                 local s = 0 for rep = 1, 60 do for _, v in ipairs(t) do s = s + v end end return s",
    },
    Program {
        name: "the table grows between runs",
        source: "local t = {1, 2, 3} local s = 0 \
                 for rep = 1, 300 do for _, v in ipairs(t) do s = s + v end t[#t + 1] = rep end return s",
    },
    Program {
        name: "pairs over an array",
        source: "local t = {} for i = 1, 200 do t[i] = i end \
                 local s = 0 for rep = 1, 100 do for k, v in pairs(t) do s = s + k + v end end return s",
    },
    Program {
        name: "pairs over a hash",
        source: "local t = {} for i = 1, 100 do t['k' .. i] = i end \
                 local s = 0 for rep = 1, 100 do for k, v in pairs(t) do s = s + v end end return s",
    },
    Program {
        name: "a closure iterator",
        source: "local function range(n) local i = 0 return function() i = i + 1 if i <= n then return i end end end \
                 local s = 0 for rep = 1, 100 do for i in range(100) do s = s + i end end return s",
    },
    Program {
        name: "a stateless iterator",
        source: "local function step(n, i) if i < n then return i + 1 end end \
                 local s = 0 for rep = 1, 100 do for i in step, 100, 0 do s = s + i end end return s",
    },
    Program {
        name: "captured loop variables",
        source: "local t = {} for i = 1, 200 do t[i] = i end local fs = {} \
                 for rep = 1, 20 do for i, v in ipairs(t) do if i % 100 == 0 then fs[#fs + 1] = function() return i + v end end end end \
                 local s = 0 for _, f in ipairs(fs) do s = s + f() end return s, #fs",
    },
    Program {
        name: "an error in the body",
        source: "local t = {} for i = 1, 200 do t[i] = i end \
                 local ok, e = pcall(function() for rep = 1, 50 do for i, v in ipairs(t) do \
                 if rep == 50 and i == 150 then error('stop') end end end end) return ok, e",
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
fn traced_generic_for_loops_compute_what_the_interpreter_does() {
    for profile in PROFILES {
        for program in TRACED.iter().chain(MIXED) {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, program.source);
            let (actual, _) = run(profile, ExecutionMode::Tracing, program.source);
            assert_eq!(actual, expected, "{} under {profile:?}", program.name);
        }
    }
}

#[test]
fn ipairs_loops_record_and_run_natively() {
    let mut failures = Vec::new();
    for profile in PROFILES {
        for program in TRACED {
            let (_, stats) = run(profile, ExecutionMode::Tracing, program.source);
            let rejected = stats.recording_aborts[TraceAbortReason::CompileRejected.index()];
            if stats.traces_recorded < 1 || stats.native_entries < 1 || rejected > 0 {
                failures.push(format!("{} under {profile:?}: {stats:?}", program.name));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_end_of_an_ipairs_loop_leaves_the_trace_and_the_interpreter_finishes() {
    // The trace records full runs of the array. Each run ends at the first
    // missing element, which is a guard exit before the iterator call.
    let source = "local t = {} for i = 1, 50 do t[i] = i end local s, runs = 0, 0 \
                  for rep = 1, 2000 do for _, v in ipairs(t) do s = s + v end runs = runs + 1 end \
                  return s, runs";
    for profile in PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.exits.iter().sum::<u64>() >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}
