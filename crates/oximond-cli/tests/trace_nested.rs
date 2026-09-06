//! Nested loops through the whole stack: Lua programs with loops inside
//! loops, in every profile, under the interpreter and under tracing execution.
//!
//! Results must match the interpreter bit for bit. Counters then prove what
//! equal output cannot: that the inner loop's trace carried on through the
//! loops around it (an outer side trace), that the outer loop therefore ran
//! inside the native tree (`side_entries` counts each pass through its
//! link while `native_entries` stays a handful), and that a covered loop the
//! tree stops carrying is given back to the interpreter.

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

/// A bit-exact rendering of results, so integer against float counts.
fn canon(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .map(|value| match value {
            Value::Integer(integer) => format!("I{integer}"),
            Value::Number(number) => format!("F{:016x}", number.to_bits()),
            Value::Boolean(flag) => format!("B{flag}"),
            other => format!("{other:?}"),
        })
        .collect()
}

struct Outcome {
    results: Vec<String>,
    trace: TraceStatistics,
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Outcome {
    run_with(profile, mode, source, false)
}

/// [`run`], with the sides of small trees compiled apart too when `apart`
/// is set (see `Vm::keep_jit_sides_apart`).
fn run_with(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
    apart: bool,
) -> Outcome {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().count_trace_links();
    if apart {
        lua.vm_mut().keep_jit_sides_apart();
    }
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    Outcome {
        results: canon(&results),
        trace: lua.vm().jit_statistics().trace,
    }
}

/// Runs `source` everywhere and returns the tracing outcome for `profile`,
/// after checking that its result matches the interpreter's.
fn check(profile: CompatibilityProfile, source: &str) -> Outcome {
    let expected = run(profile, ExecutionMode::Interpreter, source);
    let tracing = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing.results, expected.results,
        "{profile:?} tracing\n{source}"
    );
    tracing
}

#[test]
fn a_loop_nest_runs_inside_the_tree_of_its_inner_loop_in_every_profile() {
    let source = "local total = 0 \
                  for i = 1, 600 do \
                      for j = 1, 40 do total = total + (i * j) % 7 end \
                      total = total + i % 3 \
                  end return total";
    for profile in PROFILES {
        let outcome = check(profile, source);
        let stats = outcome.trace;
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 80, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 400, "{profile:?}: {stats:?}");
    }
}

#[test]
fn three_levels_and_a_while_loop_inside_a_for_loop_match() {
    let three = "local a, b, c = 0, 0, 0 \
                 for i = 1, 60 do \
                     for j = 1, 30 do \
                         for k = 1, 6 do a = a + k * j end \
                         b = b + j \
                     end \
                     c = c + i \
                 end return a, b, c";
    let looping = "local acc = 0 \
                   for i = 1, 500 do \
                       local j = 0 \
                       while j < 9 do acc = acc + j * i; j = j + 1 end \
                       acc = acc - i \
                   end return acc";
    for profile in PROFILES {
        for (name, source) in [("three levels", three), ("while in for", looping)] {
            let outcome = check(profile, source);
            let stats = outcome.trace;
            assert!(stats.outer_sides >= 1, "{name} {profile:?}: {stats:?}");
            assert!(stats.native_entries < 150, "{name} {profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_branch_in_the_outer_part_and_a_triangular_inner_loop_match() {
    let branchy = "local acc, evens, odds = 0, 0, 0 \
                   for i = 1, 800 do \
                       for j = 1, 5 do acc = acc + j end \
                       if i % 3 == 0 then evens = evens + i else odds = odds + 1 end \
                   end return acc, evens, odds";
    let triangle = "local acc = 0 \
                    for i = 1, 300 do for j = 1, i do acc = acc + j % 5 end end \
                    return acc";
    for profile in PROFILES {
        for source in [branchy, triangle] {
            let tracing = check(profile, source);
            assert!(tracing.trace.iterations > 1_000, "{profile:?}");
            assert!(
                tracing.trace.outer_sides >= 1,
                "{profile:?}: {:?}",
                tracing.trace
            );
        }
    }
}

#[test]
fn accumulators_in_upvalues_cross_the_outer_links() {
    // `acc` and `rounds` are cells of the running closure: the outer part
    // writes one and the inner loop the other, so the side names cells the
    // root's window has too.
    let source = "local acc, rounds = 0, 0 \
                  local function run(n) \
                      for i = 1, n do \
                          for j = 1, 9 do acc = acc + i + j end \
                          rounds = rounds + 1 \
                      end \
                  end \
                  run(700) \
                  return acc, rounds";
    for profile in PROFILES {
        let outcome = check(profile, source);
        assert!(
            outcome.trace.iterations > 6_000,
            "{profile:?}: {:?}",
            outcome.trace
        );
    }
}

#[test]
fn an_outer_loop_with_a_call_is_never_covered() {
    // The outer part calls a native function, which no side trace holds. No
    // tree runs the outer loop, so it is never covered.
    let source = "local total = 0 \
                  for i = 1, 1500 do \
                      for j = 1, 30 do total = total + j end \
                      total = total + (os.clock() < 0 and 0 or 2) \
                  end return total";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_loops, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_table_read_out_of_a_table_in_the_outer_part_is_covered_through_its_lane() {
    // `local row = m[i]` reads a table out of a table. A side meets it by
    // lane and reads `row[j]` through the table's own array words, so the
    // outer part records and the outer loop is covered.
    let source = "local m = {} \
                  for i = 1, 64 do local r = {} \
                      for j = 1, 16 do r[j] = i * j * 0.5 end m[i] = r end \
                  local s = 0.0 \
                  for rep = 1, 3000 do \
                      local row = m[(rep % 64) + 1] \
                      for j = 1, 16 do s = s + row[j] end \
                  end return s";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_row_copied_to_another_local_before_it_is_indexed_is_covered_too() {
    // As above, but the row reaches the inner loop through a `Move`.
    let source = "local m = {} \
                  for i = 1, 64 do local r = {} \
                      for j = 1, 16 do r[j] = i * j * 0.5 end m[i] = r end \
                  local s = 0.0 \
                  for rep = 1, 3000 do \
                      local row = m[(rep % 64) + 1] \
                      local q = row \
                      for j = 1, 16 do s = s + q[j] end \
                  end return s";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_error_in_the_outer_part_comes_from_the_interpreter() {
    // After 400 outer iterations the outer part indexes a nil: the interpreter
    // raises it, with its message and position, whatever the tree did before.
    let source = "local data = {} \
                  local function go() \
                      local total = 0 \
                      for i = 1, 500 do \
                          for j = 1, 20 do total = total + j end \
                          if i == 400 then total = total + data.missing.field end \
                          total = total + 1 \
                      end \
                      return total \
                  end \
                  local ok, message = pcall(go) \
                  return ok, message";
    for profile in PROFILES {
        let outcome = check(profile, source);
        assert!(
            outcome.trace.iterations > 5_000,
            "{profile:?}: {:?}",
            outcome.trace
        );
    }
}

#[test]
fn a_break_out_of_the_outer_loop_leaves_through_a_guard_of_the_tree() {
    // The outer loop ends early, on an iteration that depends on the data.
    for limit in [97, 98, 99, 100, 101, 250] {
        let source = format!(
            "local total = 0 \
             for i = 1, 1000 do \
                 for j = 1, 11 do total = total + j end \
                 total = total + i \
                 if total > {} then break end \
             end return total",
            limit * 700
        );
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let tracing = check(profile, &source);
            assert!(tracing.trace.outer_sides >= 1, "{profile:?} {limit}");
        }
    }
}

#[test]
fn a_nest_inside_a_coroutine_and_the_main_chunk_both_match() {
    let source = "local function nest(n) \
                      local total = 0 \
                      for i = 1, n do \
                          for j = 1, 10 do total = total + i * j end \
                          total = total % 1000003 \
                      end \
                      return total \
                  end \
                  local co = coroutine.wrap(function() \
                      coroutine.yield(nest(500)) \
                      return nest(700) \
                  end) \
                  local a = co() \
                  local b = co() \
                  return a, b, nest(400)";
    for profile in PROFILES {
        let tracing = check(profile, source);
        assert!(
            tracing.trace.iterations > 10_000,
            "{profile:?}: {:?}",
            tracing.trace
        );
    }
}

#[test]
fn many_nest_shapes_stay_correct() {
    let shapes = [
        "for j = 1, 4 do a = a + j end",
        "for j = 1, i % 7 + 1 do a = a + 1 end",
        "for j = i, i + 5 do a = a + j % 3 end",
        "for j = 10, 1, -3 do a = a - j end",
        "local j = 0 while j < 5 do a = a + 1; j = j + 1 end",
        "for j = 1, 3 do for k = 1, 3 do a = a + j * k end end",
        "for j = 1, 6 do if j % 2 == 0 then a = a + j else b = b + 1 end end",
        "for j = 1, 5 do a = a + 0.5 end",
    ];
    let tails = [
        "",
        "b = b + 1",
        "if i % 4 == 0 then b = b + i end",
        "c = c + a % 10",
    ];
    for (index, shape) in shapes.iter().enumerate() {
        for (tail_index, tail) in tails.iter().enumerate() {
            let source = format!(
                "local a, b, c = 0, 0, 0 \
                 for i = 1, 400 do {shape} {tail} end return a, b, c"
            );
            for profile in [
                CompatibilityProfile::Lua51,
                CompatibilityProfile::Lua53,
                CompatibilityProfile::Lua55,
            ] {
                let tracing = check(profile, &source);
                assert!(
                    tracing.trace.traces_compiled >= 1,
                    "shape {index} tail {tail_index} {profile:?}: {:?}",
                    tracing.trace
                );
            }
        }
    }
}

// ------------------------------------------------------------------ tables

#[test]
fn a_loop_nest_over_an_array_runs_inside_one_tree() {
    // The root of the inner loop binds `values`. In loops without integers
    // its tree takes a side along the outer iteration, so the whole nest runs in native code. An integer loop
    // that runs 2000 iterations per entry spends too little of its time
    // entering and leaving to pay for a second compile of the tree.
    let source = "local values = {} \
                  for i = 1, 2000 do values[i] = i % 101 end \
                  local total = 0 \
                  for round = 1, 80 do \
                      for i = 1, 2000 do total = total + values[i] end \
                  end return total";
    for profile in PROFILES {
        let outcome = check(profile, source);
        let stats = outcome.trace;
        assert!(stats.iterations > 150_000, "{profile:?}: {stats:?}");
        if profile.is_lua53_or_later() {
            assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
        } else {
            assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
            assert!(stats.native_entries < 40, "{profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_matrix_vector_product_stores_the_row_sum_in_the_outer_part() {
    // The outer part stores into `y` and the inner loop reads two arrays, so
    // the side holds a table operation of its own on a table the root never
    // touches.
    let source = "local n = 50 \
                  local a, x, y = {}, {}, {} \
                  for i = 1, n * n do a[i] = (i * 7) % 13 end \
                  for i = 1, n do x[i] = i; y[i] = 0 end \
                  for rep = 1, 30 do \
                      for i = 1, n do \
                          local s = 0 \
                          for j = 1, n do s = s + a[(i - 1) * n + j] * x[j] end \
                          y[i] = s \
                      end \
                  end \
                  return y[1], y[n], y[n // 2]";
    for profile in PROFILES {
        let source = if matches!(
            profile,
            CompatibilityProfile::Lua51
                | CompatibilityProfile::Lua52
                | CompatibilityProfile::LuaJit21
        ) {
            source.replace("n // 2", "n / 2")
        } else {
            source.to_string()
        };
        let outcome = check(profile, &source);
        let stats = outcome.trace;
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 40_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_alternating_branch_over_an_array_gets_sides_that_read_other_tables() {
    // The root binds `t` and `u`. The other arm reads `v`, which only its
    // side binds, so the tree's list of bound tables grows.
    let source = "local t, u, v = {}, {}, {} \
                  for i = 1, 1000 do t[i] = i % 7; u[i] = i % 5; v[i] = i % 3 end \
                  local pos, neg = 0, 0 \
                  for round = 1, 100 do \
                      for i = 1, 1000 do \
                          if t[i] > 3 then pos = pos + u[i] else neg = neg + v[i] end \
                      end \
                  end return pos, neg";
    for profile in PROFILES {
        let outcome = check(profile, source);
        let stats = outcome.trace;
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 90_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_side_cannot_hold_a_table_the_loop_rewrites() {
    // The outer part swaps which table `cur` names, so a binding of `cur`
    // would resolve the wrong table. No tree runs the outer loop, and the
    // answer is the interpreter's.
    let source = "local a, b = {}, {} \
                  for i = 1, 600 do a[i] = i; b[i] = 2 * i end \
                  local cur, total = a, 0 \
                  for round = 1, 200 do \
                      for i = 1, 600 do total = total + cur[i] end \
                      if round % 2 == 0 then cur = a else cur = b end \
                  end return total";
    for profile in PROFILES {
        let outcome = check(profile, source);
        assert_eq!(outcome.trace.outer_sides, 0, "{profile:?}");
    }
}

// ------------------------------------------------------------------- calls

#[test]
fn a_call_in_the_inner_loop_does_not_keep_the_nest_out_of_the_tree() {
    // The root pins `weight` and inlines it. The side along the outer
    // iteration shares the pinned closure and adds none.
    let source = "local function weight(i, j) return (i + j) % 5 + 1 end \
                  local total = 0 \
                  for i = 1, 500 do \
                      for j = 1, 30 do total = total + weight(i, j) * j end \
                      total = total + i % 3 \
                  end return total";
    for profile in PROFILES {
        let outcome = check(profile, source);
        let stats = outcome.trace;
        assert!(stats.inlined_calls >= 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 80, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 14_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_callee_rebound_between_outer_iterations_leaves_the_tree_at_its_guard() {
    // The outer part swaps the function the inner loop calls. The callee guard
    // fails at the first inner iteration after the swap, leaves the tree, and
    // the interpreter finishes with the new callee exactly as without traces.
    let source = "local function one(x) return x + 1 end \
                  local function two(x) return x + 2 end \
                  local f, total = one, 0 \
                  for i = 1, 400 do \
                      for j = 1, 20 do total = total + f(j) end \
                      if i % 50 == 0 then f = (f == one) and two or one end \
                  end return total";
    for profile in PROFILES {
        let tracing = check(profile, source);
        // Each swap sends the trace back to the interpreter, which retires it
        // after a few failed entries and records the new callee. A tree that
        // retires gives the loops it covered back.
        let stats = tracing.trace;
        assert!(stats.iterations > 1_000, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.uncovered_loops <= stats.covered_loops,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_covered_loop_whose_outer_arm_stops_running_is_revoked() {
    // The arm in the outer part runs for the first iterations, so a side
    // lands on the other path and the tree covers the outer loop. Then the
    // arm stops being taken: its guard fails every iteration, its side cannot
    // be recorded (it builds a table, which a side refuses), and every
    // outer iteration enters and leaves the tree with the outer part
    // interpreted. The tree must give the covered header up.
    let source = "local h = {} local total = 0 \
                  for i = 1, 60000 do \
                      for j = 1, 12 do total = total + j end \
                      if i > 3000 then h.x = {total} end \
                  end return total, #h.x";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.uncovered_loops, 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_side_that_names_a_new_root_string_installs_and_grows_the_bank() {
    // The arm loads a string the root does not hold. The side records it,
    // adds the string to the tree's bank, and the tree covers the outer loop.
    let source = "local h = {} local total = 0 \
                  for i = 1, 60000 do \
                      for j = 1, 12 do total = total + j end \
                      if i > 3000 then local key = 'x' h[key] = total end \
                  end return total, h.x";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_float_nest_with_a_long_inner_loop_that_the_tree_stops_carrying_is_revoked() {
    // 70 inner iterations per outer iteration is more than the integer bar
    // (64 per interpreted backedge), but a float tree must carry about 160.
    // After `i > 3000` the arm's guard fails every iteration and its side
    // aborts on the call of a native with no numeric form, so every outer
    // iteration enters and leaves the tree.
    let source = "local name, len = \"abcdef\", os.date local total = 0 \
                  for i = 1, 60000 do \
                      for j = 1, 70 do total = total + j * 0.5 end \
                      if i > 3000 then total = total + #len(name) end \
                  end return total";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_covered_loop_that_its_tree_carries_is_never_revoked() {
    let source = "local total = 0 \
                  for i = 1, 60000 do \
                      for j = 1, 12 do total = total + j end \
                      total = total + i % 3 \
                  end return total";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_string_only_the_outer_part_loads_is_covered() {
    // The outer side loads `count`, which the inner loop does not: the tree's
    // bank of root strings grows to take it, and the tree covers the outer
    // loop.
    let source = "local o = {count = 0} local total = 0 \
                  for i = 1, 20000 do \
                      for j = 1, 12 do total = total + j end \
                      local key = 'count' \
                      o[key] = o[key] + 1 \
                  end return total, o.count";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 0, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 100, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_field_only_the_outer_part_updates_is_covered() {
    // A field instruction names its key by a constant and loads no string,
    // so an outer side records `o.count = o.count + 1` and the tree covers
    // the outer loop.
    let source = "local o = {count = 0} local total = 0 \
                  for i = 1, 20000 do \
                      for j = 1, 12 do total = total + j end \
                      o.count = o.count + 1 \
                  end return total, o.count";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source).trace;
        assert_eq!(stats.covered_loops, 1, "{profile:?}: {stats:?}");
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_aborts, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.covered_revoked, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_float_while_loop_with_two_exits_keeps_its_covered_outer_loop() {
    // An escape-time kernel: about 24 float iterations per pixel and two
    // guards that end the while loop, so the tree needs an outer side per
    // guard. A compile takes many outer iterations, and until the second
    // side lands every pixel that leaves through its guard comes back to the
    // interpreter. That must not read as the tree failing to carry the loop,
    // and the wait for the sides must outlast the compiles. The loop nest
    // stays in the tree.
    let source = "local size, sum = 120, 0 \
                  for rep = 1, 3 do \
                      for y = 0, size - 1 do \
                          local ci = 2 * y / size - 1 \
                          for x = 0, size - 1 do \
                              local cr = 2 * x / size - 1.5 \
                              local zr, zi, n = 0, 0, 0 \
                              while zr * zr + zi * zi <= 4 and n < 50 do \
                                  zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci \
                                  n = n + 1 \
                              end \
                              sum = sum + n \
                          end \
                      end \
                  end return sum";
    let expected = run(
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
        source,
    );
    // Delays are in safepoint polls, which come once per outer iteration
    // here, so each stands for that many pixels of compile time.
    for delay in [0, 100, 400] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        lua.vm_mut().use_delayed_synchronous_jit_compilation(delay);
        lua.vm_mut().count_trace_links();
        let results: Vec<Value> = lua.exec(source).unwrap();
        assert_eq!(canon(&results), expected.results, "delay {delay}");
        let stats = &lua.vm().jit_statistics().trace;
        assert!(stats.covered_loops >= 1, "delay {delay}: {stats:?}");
        assert_eq!(stats.covered_revoked, 0, "delay {delay}: {stats:?}");
    }
}

/// The shape of `algorithm/nbody`: a function whose triangular nest runs four,
/// three, two, one and no inner trips, followed by a five-iteration loop over
/// the same bodies, called many times.
const SHORT_NEST: &str = "local b = {} \
    for i = 1, 5 do b[i] = {x = i * 1.5, v = 0.25 * i, m = 1 + i} end \
    local function advance(dt) \
        for i = 1, #b do \
            local p = b[i] \
            for j = i + 1, #b do \
                local q = b[j] \
                local d = p.x - q.x \
                local f = dt * d / (1 + d * d) \
                p.v = p.v - f * q.m \
                q.v = q.v + f * p.m \
            end \
        end \
        for i = 1, #b do local p = b[i] p.x = p.x + dt * p.v end \
    end \
    for _ = 1, 3000 do advance(0.01) end \
    local t = 0 \
    for i = 1, #b do t = t + b[i].x + b[i].v end \
    return t";

/// The calls of [`SHORT_NEST`].
const SHORT_NEST_CALLS: u64 = 3000;

#[test]
fn short_triangular_nests_run_in_one_tree_and_match_the_interpreter() {
    for profile in PROFILES {
        let stats = check(profile, SHORT_NEST).trace;
        // The inner loop leaves after a trip or two, so only the tree that
        // runs on through the loop around it pays: its outer side carries the
        // loop, and the root that led to it is the only root compiled.
        assert!(stats.outer_sides >= 1, "{profile:?}: {stats:?}");
        assert!(stats.covered_loops >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        // One entry per call until the side that returns into the loop that
        // calls the function joins the tree (after `RETURN_THROUGH_ENTRIES`
        // entries, compiled beside the tree), and the calls after it run in
        // the tree, which the loop around them is: a call's four outer
        // iterations after the first jump through the outer side.
        assert!(
            stats.native_entries <= SHORT_NEST_CALLS / 20,
            "{profile:?}: the calls ran inside the tree of the loop: {stats:?}"
        );
        assert!(
            stats.side_entries >= SHORT_NEST_CALLS * 9 / 10,
            "{profile:?}: the outer iterations ran inside the tree: {stats:?}"
        );
        // Nearly every inner iteration ran native: ten a call.
        assert!(
            stats.iterations >= 7 * SHORT_NEST_CALLS * 9 / 10,
            "{profile:?}: the inner trips ran inside the tree: {stats:?}"
        );
    }
}

#[test]
fn the_root_that_leads_to_a_short_nests_tree_is_not_compiled_alone() {
    // One tree is compiled. The root that gives its exit the outer side is
    // installed as a placeholder while the interpreter records the side,
    // and no baseline code is made for it. The sides that return into the
    // loop that calls the function go into one more compile of the tree,
    // which is small enough for that to cost less than compiling them
    // beside it (14.9M instructions for this run against 19.2M).
    let stats = check(CompatibilityProfile::Lua54, SHORT_NEST).trace;
    assert_eq!(stats.baseline_roots, 0, "{stats:?}");
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert_eq!(stats.side_compiles, 0, "{stats:?}");
    assert_eq!(stats.compile_submissions, 2, "{stats:?}");
    assert!(stats.side_installs >= 1, "{stats:?}");
    // Kept apart, those sides compile beside the tree and the tree's code
    // stays as it is: what a tree too large to compile again does.
    let stats = run_with(
        CompatibilityProfile::Lua54,
        ExecutionMode::Tracing,
        SHORT_NEST,
        true,
    )
    .trace;
    assert_eq!(stats.baseline_roots, 0, "{stats:?}");
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert!(stats.side_compiles >= 1, "{stats:?}");
    assert_eq!(
        stats.compile_submissions,
        1 + stats.side_compiles,
        "{stats:?}"
    );
}

/// Three loops, the innermost short: the nest is deeper than the sides can be
/// recorded ahead for, so its root keeps the stepping stone.
const SHORT_NEST_OF_THREE: &str = "local b = {} \
    for i = 1, 5 do b[i] = {x = i, v = 0.5} end \
    local function step(dt, n) \
        local s = 0 \
        for r = 1, 4 do \
            for i = 1, #b do \
                local p = b[i] \
                for j = i + 1, #b do local q = b[j] s = s + (q.x - p.x) * dt end \
                p.v = p.v + s * dt \
            end \
        end \
        return s \
    end \
    local t = 0 \
    for _ = 1, 3000 do t = t + step(0.001, 1) end \
    return t";

#[test]
fn the_root_that_leads_to_a_deeper_nests_tree_compiles_at_the_baseline_tier() {
    let stats = check(CompatibilityProfile::Lua51, SHORT_NEST_OF_THREE).trace;
    // The stone, and a tree for each side that landed over it.
    assert_eq!(stats.baseline_roots, 1, "{stats:?}");
    assert!(stats.compile_submissions >= 2, "{stats:?}");
    assert!(stats.side_installs >= 1, "{stats:?}");
}

#[test]
fn a_short_loop_in_a_function_is_not_compiled() {
    // Five iterations per call cost less in the interpreter than an entry
    // and exit of a trace, so the loop is turned away before its compile.
    let source = "local b = {} \
        for i = 1, 5 do b[i] = {x = i, v = 0.5} end \
        local function step(dt) \
            for i = 1, #b do local p = b[i] p.x = p.x + dt * p.v end \
        end \
        for _ = 1, 3000 do step(0.01) end \
        local t = 0 \
        for i = 1, #b do t = t + b[i].x end \
        return t";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        assert_eq!(stats.traces_compiled, 0, "{profile:?}: {stats:?}");
    }
}

/// A loop that calls a function whose own loop runs five trips each time.
const CALLED_SHORT_LOOP: &str = "local b = {} \
    for i = 1, 5 do b[i] = {x = i, v = 0.5} end \
    local function step(dt) \
        for i = 1, #b do local p = b[i] p.x = p.x + dt * p.v end \
    end \
    for _ = 1, 60000 do step(0.01) end \
    local t = 0 \
    for i = 1, #b do t = t + b[i].x end \
    return t";

#[test]
fn an_outer_loop_unrolls_the_short_loop_of_the_function_it_calls() {
    for profile in PROFILES {
        let stats = check(profile, CALLED_SHORT_LOOP).trace;
        // The trace of the outer loop holds the call and the five trips of
        // the loop in it: one entry runs every call, where the loop alone
        // never repaid an entry per call.
        assert_eq!(
            stats.aborts(TraceAbortReason::InnerLoop),
            0,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.inlined_calls >= 1, "{profile:?}: {stats:?}");
        assert!(stats.traces_compiled >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries <= 4, "{profile:?}: {stats:?}");
        assert!(stats.iterations >= 40_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_unrolled_loop_that_gets_more_trips_leaves_the_trace_and_matches() {
    // The loop in `inner` runs 3 trips, then 7 (more than the trace holds),
    // then 40 (more than a recording unrolls).
    let source = "local function inner(k) \
            local s = 0 \
            for j = 1, k do s = s + (j * k) % 11 end \
            return s \
        end \
        local total, n = 0, 3 \
        for i = 1, 60000 do \
            if i == 20001 then n = 7 end \
            if i == 40001 then n = 40 end \
            total = total + inner(n) + i % 5 \
        end \
        return total";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations >= 15_000, "{profile:?}: {stats:?}");
        // The guard of the latch fired where the length changed, so the
        // trace was left for the interpreter or a side at least once.
        let left: u64 = stats.exits.iter().sum::<u64>() + stats.side_exits.iter().sum::<u64>();
        assert!(left >= 2, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_long_inner_loop_keeps_its_own_trace_and_is_not_unrolled() {
    let source = "local total = 0 \
        for i = 1, 400 do \
            for j = 1, 100 do total = total + (i * j) % 7 end \
            total = total + i % 3 \
        end \
        return total";
    for profile in PROFILES {
        let stats = check(profile, source).trace;
        // The recording is the inner loop's body, not a hundred copies of it.
        assert!(stats.traces_recorded >= 1, "{profile:?}: {stats:?}");
        assert!(
            stats.recorded_instructions <= 40 * stats.traces_recorded,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.iterations >= 20_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_triangular_nest_inside_a_hot_loop_matches_the_interpreter() {
    // `j` starts at `i + 1`: its trip count differs on every outer iteration
    // of `i`, which an unrolling `i` trace could not repeat.
    let source = "local total = 0 \
        for i = 1, 6 do \
            for j = i + 1, 6 do total = total + i * j end \
        end \
        return total";
    let big = format!(
        "local acc = 0 for r = 1, 4000 do {} acc = acc + total end return acc",
        source.replace("return total", "")
    );
    for profile in PROFILES {
        check(profile, source);
        check(profile, &big);
    }
}

/// The shape of `algorithm/nbody`: a function with a triangular pair loop and
/// a second short loop, called from a hot loop.
fn nbody_shape(calls: u32) -> String {
    NBODY_SHAPE.replace("CALLS", &calls.to_string())
}

const NBODY_SHAPE: &str = "local bodies = {} \
    for i = 1, 5 do bodies[i] = {x = i, y = 2 * i, vx = 0, vy = 0, m = 1 + i / 10} end \
    local function advance(dt) \
        for i = 1, #bodies do \
            local a = bodies[i] \
            for j = i + 1, #bodies do \
                local b = bodies[j] \
                local dx, dy = a.x - b.x, a.y - b.y \
                local d2 = dx * dx + dy * dy \
                local mag = dt / (d2 * math.sqrt(d2)) \
                a.vx, a.vy = a.vx - dx * b.m * mag, a.vy - dy * b.m * mag \
                b.vx, b.vy = b.vx + dx * a.m * mag, b.vy + dy * a.m * mag \
            end \
        end \
        for i = 1, #bodies do \
            local b = bodies[i] \
            b.x, b.y = b.x + dt * b.vx, b.y + dt * b.vy \
        end \
    end \
    for _ = 1, CALLS do advance(0.001) end \
    local t = 0 \
    for i = 1, #bodies do t = t + bodies[i].x + bodies[i].y + bodies[i].vx + bodies[i].vy end \
    return string.format('%.9f', t)";

#[test]
fn a_triangular_pair_loop_in_a_called_function_keeps_the_traces_of_its_loops() {
    // The nest does not fit one trace without a compile that costs more than
    // a run of this length repays, so it is not retried with more room: the
    // loops inside keep their own traces and the result matches. The loop that
    // calls the function joins them through the sides compiled beside the
    // tree, so the calls enter native code only until they have.
    for profile in PROFILES {
        let stats = check(profile, &nbody_shape(40_000)).trace;
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries <= 1_000, "{profile:?}: {stats:?}");
        assert!(stats.iterations >= 25_000, "{profile:?}: {stats:?}");
        assert!(stats.traces_compiled <= 2, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_short_nbody_run_does_not_pay_for_the_big_nest_trace() {
    // A run shorter than the gate of the link (`RETURN_THROUGH_ENTRIES`
    // entries of the tree) is one entry per call, and compiles no side
    // beside the tree: the loops inside keep their own traces and no trace
    // with more room is compiled.
    for profile in PROFILES {
        let stats = check(profile, &nbody_shape(30)).trace;
        assert!(stats.native_entries >= 5, "{profile:?}: {stats:?}");
        assert!(stats.native_entries <= 30, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_compiles, 0, "{profile:?}: {stats:?}");
        assert!(stats.traces_compiled <= 2, "{profile:?}: {stats:?}");
        assert_eq!(stats.negative_evictions, 0, "{profile:?}: {stats:?}");
    }
    // 3000 calls is the length of `algorithm/nbody` at scale 3: the link
    // repays, and the calls run in the loop that makes them (65 entries).
    // The sides of the link go into one more compile of the small tree
    // (18.3M instructions for the run against 22.0M with them apart).
    for profile in PROFILES {
        let stats = check(profile, &nbody_shape(3000)).trace;
        assert!(stats.native_entries >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries <= 100, "{profile:?}: {stats:?}");
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_compiles, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 2, "{profile:?}: {stats:?}");
        assert!(stats.iterations >= 20_000, "{profile:?}: {stats:?}");
        assert!(stats.traces_compiled <= 2, "{profile:?}: {stats:?}");
        assert_eq!(stats.negative_evictions, 0, "{profile:?}: {stats:?}");
    }
}
