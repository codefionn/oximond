//! Tree nests over table fields. A two-level `while` nest whose inner bound
//! comes from a table field (`local zm = l.nz`) links its outer side into the
//! tree, and an inner loop that hands `l.z` to an inlined function indexing
//! it on a branch (the Luanti `n1` shape) takes its side without a table
//! lookup helper. The results are the interpreter's in every profile.

use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
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
/// outcomes agree and that native code ran, and hands the results and the
/// tracing statistics to `check`.
fn assert_agrees(source: &str, check: impl Fn(CompatibilityProfile, &[Value], &TraceStatistics)) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let results = expected.unwrap_or_else(|error| panic!("{profile:?} {error}"));
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        check(profile, &results, &stats);
    }
}

fn count(value: &Value) -> u64 {
    match value {
        Value::Integer(value) => u64::try_from(*value).unwrap(),
        Value::Number(value) => *value as u64,
        other => panic!("not a count: {other:?}"),
    }
}

const FIELD_BOUND_NEST: &str = "local function run(l)
         local s, trips = 0, 0
         local y = 1
         while y <= 3 do
             local zm = l.nz
             local z = 1
             while z <= zm do
                 s = s + z * y
                 trips = trips + 1
                 z = z + 1
             end
             y = y + 1
         end
         return s, trips
     end
     local l = {nz = 5}
     local total, trips = 0, 0
     for i = 1, 100000 do
         if i == 90000 then l.nz = 4 end
         local s, t = run(l)
         total, trips = total + s, trips + t
     end
     return total, trips";

#[test]
fn a_field_bound_nest_links_its_outer_side() {
    assert_agrees(FIELD_BOUND_NEST, |profile, results, stats| {
        assert!(stats.side_installs >= 2, "{profile:?} {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?} {stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?} {stats:?}");
        let trips = count(&results[1]);
        assert!(
            stats.iterations * 10 > trips * 9,
            "{profile:?} {trips} {stats:?}"
        );
    });
}

#[test]
fn an_inlined_branchy_index_of_a_field_takes_its_side_without_helpers() {
    assert_agrees(
        "local function pick(z, i)
             if i % 3 == 0 then return z[1] end
             return z[2] + z[3]
         end
         local function run(l)
             local s = 0
             for i = 1, 40 do
                 s = s + pick(l.z, i)
             end
             return s
         end
         local l = {z = {1, 2, 3}}
         local total = 0
         for i = 1, 20000 do
             if i == 15000 then l.z = {4, 5, 6} end
             total = total + run(l)
         end
         return total",
        |profile, _, stats| {
            if profile == CompatibilityProfile::Lua51 {
                assert!(stats.side_installs >= 1, "{profile:?} {stats:?}");
                assert_eq!(stats.table_lookup_helper_calls, 0, "{profile:?} {stats:?}");
            }
        },
    );
}

#[test]
fn a_field_bound_that_turns_into_a_float_leaves_for_the_interpreter() {
    assert_agrees(
        "local function run(l)
             local s = 0
             local y = 1
             while y <= 3 do
                 local zm = l.nz
                 local z = 1
                 while z <= zm do s = s + z; z = z + 1 end
                 y = y + 1
             end
             return s
         end
         local l = {nz = 5}
         local total = 0
         for i = 1, 30000 do
             if i == 20000 then l.nz = 2.5 end
             total = total + run(l)
         end
         return total",
        |_, _, _| {},
    );
}

#[test]
fn another_table_in_the_written_register_leaves_at_the_link() {
    assert_agrees(
        "local function run(l)
             local s = 0
             local y = 1
             while y <= 3 do
                 local zm = l.nz
                 local z = 1
                 while z <= zm do s = s + z * y * l.w; z = z + 1 end
                 y = y + 1
             end
             return s
         end
         local a, b = {nz = 5, w = 1}, {nz = 3, w = 2}
         local total = 0
         for i = 1, 60000 do
             total = total + run(i > 40000 and i % 3 == 0 and b or a)
         end
         return total",
        |profile, _, stats| {
            assert!(stats.side_installs >= 1, "{profile:?} {stats:?}");
        },
    );
}
