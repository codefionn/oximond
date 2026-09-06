//! Records read out of an array (tables met by lane) together with callable
//! pinning, `__index` chains, natives, `ipairs` and re-execution regions.
//!
//! Every program runs under the interpreter and under tracing execution.
//! Results must match exactly.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

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
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

fn agree(source: &str) -> TraceStatistics {
    let profile = CompatibilityProfile::Lua54;
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(tracing, expected, "tracing\n{source}\n{stats:?}");
    stats
}

#[test]
fn a_native_in_a_loop_over_records_runs_natively() {
    let stats = agree(
        r"
        local sqrt = math.sqrt
        local bodies = {}
        for i = 1, 8 do bodies[i] = { x = i + 0.5, y = i * 2.0, m = 0.0 } end
        for step = 1, 4000 do
          for i = 1, #bodies do
            local b = bodies[i]
            b.m = b.m + sqrt(b.x * b.x + b.y * b.y) * 0.001
          end
        end
        local s = 0.0
        for i = 1, #bodies do s = s + bodies[i].m end
        return s
        ",
    );
    assert!(stats.iterations > 0, "{stats:?}");
}

#[test]
fn a_field_found_through_index_on_a_loaded_record_stays_correct() {
    let stats = agree(
        r"
        local class = { scale = 3.0 }
        local mt = { __index = class }
        local rows = {}
        for i = 1, 16 do rows[i] = setmetatable({ v = i + 0.0 }, mt) end
        local total = 0.0
        for round = 1, 2000 do
          for i = 1, #rows do
            local r = rows[i]
            total = total + r.v * r.scale
          end
        end
        return total
        ",
    );
    let _ = stats;
}

#[test]
fn a_method_in_a_field_of_a_loaded_record_stays_correct() {
    agree(
        r"
        local function double(x) return x * 2.0 end
        local rows = {}
        for i = 1, 16 do rows[i] = { v = i + 0.0, f = double } end
        local total = 0.0
        for round = 1, 2000 do
          for i = 1, #rows do
            local r = rows[i]
            total = total + r.f(r.v)
          end
        end
        return total
        ",
    );
}

#[test]
fn a_method_through_index_on_a_bound_table_next_to_loaded_records_runs() {
    let stats = agree(
        r"
        local class = {}
        function class.step(x) return x + 1.0 end
        local obj = setmetatable({ n = 0.0 }, { __index = class })
        local rows = {}
        for i = 1, 16 do rows[i] = { v = i + 0.0 } end
        for round = 1, 3000 do
          for i = 1, #rows do
            local r = rows[i]
            r.v = obj.step(r.v)
            obj.n = obj.n + r.v
          end
        end
        return obj.n
        ",
    );
    assert!(stats.iterations > 0, "{stats:?}");
}

#[test]
fn ipairs_over_an_array_of_records_with_stores_stays_correct() {
    agree(
        r"
        local rows = {}
        for i = 1, 16 do rows[i] = { v = i + 0.0, w = 0.0 } end
        for round = 1, 2000 do
          for _, r in ipairs(rows) do
            r.w = r.w + r.v
          end
        end
        local s = 0.0
        for _, r in ipairs(rows) do s = s + r.w end
        return s
        ",
    );
}

#[test]
fn a_vararg_callee_that_stores_into_a_loaded_record_stays_correct() {
    agree(
        r"
        local function bump(r, ...)
          r.v = r.v + select('#', ...)
          if r.v > 1e9 then return 0 end
          return r.v
        end
        local rows = {}
        for i = 1, 16 do rows[i] = { v = i + 0.0 } end
        local total = 0.0
        for round = 1, 2000 do
          for i = 1, #rows do
            total = total + bump(rows[i], 1, 2, 3)
          end
        end
        return total
        ",
    );
}
