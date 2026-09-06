//! Side traces through the whole stack: Lua programs whose loops branch,
//! in every profile, under the interpreter and under tracing execution.
//!
//! Results must match the interpreter bit for bit. Counters then prove what
//! equal output cannot: that the guard which fails often got a side trace, that
//! the tree ran both arms natively (`side_entries` counts the jumps into sides
//! and `iterations` the loop-backs), and that loops a side trace cannot hold
//! stay as they were.

use oximond::vm::TraceStatistics;
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

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Vec<String>, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    lua.vm_mut().count_trace_links();
    // The subject is the sides of branch guards, and most arms here are pure
    // enough for a value select, which leaves no guard.
    lua.vm_mut().keep_trace_branch_guards();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    (canon(&results), lua.vm().jit_statistics().trace)
}

/// Runs `source` everywhere and returns the statistics of the tracing run for
/// `profile`, after checking that its result matches the interpreter's.
fn check(profile: CompatibilityProfile, source: &str) -> TraceStatistics {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(tracing, expected, "{profile:?} tracing\n{source}");
    stats
}

#[test]
fn an_alternating_branch_runs_both_arms_natively_in_every_profile() {
    let source = "local yes, no = 0, 0 \
                  for i = 1, 200000 do \
                      if i % 7 < 3 then yes = yes + i % 101 else no = no + i % 89 end \
                  end return yes, no";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert_eq!(stats.side_installs, 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 195_000, "{profile:?}: {stats:?}");
        // Three iterations in seven take the side.
        assert!(stats.side_entries > 80_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn accumulators_in_upvalues_get_sides_that_name_cells_the_root_does_not() {
    // `yes` and `no` are cells of the running closure. The root records one arm
    // and names one cell. The side names the other: the tree's window grows.
    let source = "local yes, no = 0, 0 \
                  local function run(n) \
                      for i = 1, n do \
                          if i % 7 < 3 then yes = yes + i else no = no + i end \
                      end \
                  end \
                  run(150000) \
                  return yes, no";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 140_000, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 50_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_side_that_reads_a_global_the_root_never_read_still_matches() {
    // Lua 5.1 and LuaJIT read a global with one opcode, so a trace holds it
    // as a window slot read at entry. Later profiles go through `_ENV`
    // tables, which traces do not hold yet: no trace, same answer.
    let source = "gy, gn = 3, 5 \
                  local yes, no = 0, 0 \
                  for i = 1, 150000 do \
                      if i % 7 < 3 then yes = yes + gy else no = no + gn end \
                  end return yes, no";
    for profile in PROFILES {
        let stats = check(profile, source);
        let globals_trace = matches!(
            profile,
            CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
        );
        if globals_trace {
            assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
            assert!(stats.iterations > 140_000, "{profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_three_way_branch_needs_sides_of_sides_or_two_sides() {
    let source = "local a, b, c = 0, 0, 0 \
                  for i = 1, 150000 do \
                      local r = i % 3 \
                      if r < 1 then a = a + i elseif r < 2 then b = b + i else c = c + i end \
                  end return a, b, c";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert_eq!(stats.side_installs, 2, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 145_000, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 90_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn integers_and_floats_keep_their_types_across_a_link() {
    // `n` stays an integer through the arm the side holds and `x` a float
    // through the other, whichever arm runs, so every link hands the loop the
    // types the root's parameters have.
    let source = "local n, x = 0, 0.5 \
                  for i = 1, 120000 do \
                      if i % 5 < 2 then n = n + i // 3 else x = x * 1.0000001 + i % 11 end \
                  end return n, x";
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 110_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_arm_with_an_unsupported_operation_keeps_the_old_behaviour() {
    // The `else` arm calls a native function. No trace holds it, so there is
    // no side trace, and the answer is the interpreter's.
    let source = "local yes, no = 0, 0 \
                  for i = 1, 100000 do \
                      if i % 7 < 3 then yes = yes + i else no = no + (os.clock() < 0 and 0 or 2) end \
                  end return yes, no";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert_eq!(stats.side_installs, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_entries, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_branch_that_leaves_the_loop_is_not_a_side_trace_candidate() {
    // `break` leaves the loop, which the exit already does natively. No loop
    // of the function's chunk encloses this one (the caller's loop is in
    // another chunk), so no side trace could run on from the exit. See
    // `trace_nested` for the loops that do enclose one. The caller's loop
    // runs `walk`'s loop as a loop side in `walk`'s frame (see
    // `trace_callee_loops`): that side and the one that returns out of
    // `walk` from its exits are the only sides.
    let source = "local function walk(n) \
                      for i = 1, 50 do \
                          n = n + i \
                          if n % 1000 == 999 then break end \
                      end \
                      return n \
                  end \
                  local n = 0 \
                  for round = 1, 3000 do n = walk(n) end \
                  return n";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(
            stats.side_installs <= 2 * stats.loop_sides,
            "{profile:?}: {stats:?}"
        );
        assert_eq!(stats.outer_sides, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_loop_that_ends_inside_a_side_returns_the_right_state() {
    // The loop's limit lands on every residue in turn, so its last iteration
    // runs the root's arm for some limits and the side's for others.
    for limit in 20_000..20_014 {
        let source = format!(
            "local yes, no = 0, 0 \
             for i = 1, {limit} do \
                 if i % 7 < 3 then yes = yes + i else no = no + i end \
             end return yes, no"
        );
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let tracing = check(profile, &source);
            assert!(
                tracing.side_installs >= 1,
                "{profile:?} {limit}: {tracing:?}"
            );
        }
    }
}

#[test]
fn a_branch_inside_a_while_loop_and_inside_a_function_gets_a_side() {
    let source = "local function work(n) \
                      local yes, no, k = 0, 0, 0 \
                      while k < n do \
                          k = k + 1 \
                          if k % 4 == 0 then yes = yes + k else no = no + 1 end \
                      end \
                      return yes, no \
                  end \
                  local ya, na = work(100000) \
                  local yb, nb = work(50000) \
                  return ya, na, yb, nb";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 100_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn many_loops_with_branches_stay_correct() {
    // A spread of period and arm shapes, including branches whose other arm
    // is a nested conditional.
    let shapes = [
        "if i % 2 == 0 then a = a + 1 else b = b + 1 end",
        "if i % 3 == 0 then a = a + i elseif i % 3 == 1 then b = b + i else c = c + i end",
        "if i % 5 < 2 then a = a + i else if i % 5 < 4 then b = b + 2 else c = c - 1 end end",
        "if i % 11 == 0 then a = a * 2 % 1000003 else b = (b + i) % 1000003 end",
        "if ((i - i % 100) / 100) % 2 == 0 then a = a + 1 else b = b + 1 end",
        "if i % 13 < 6 then a = a + 0.5 else b = b - 0.25 end",
    ];
    for (index, shape) in shapes.iter().enumerate() {
        let source = format!(
            "local a, b, c = 0, 0, 0 \
             for i = 1, 90000 do {shape} end return a, b, c"
        );
        for profile in PROFILES {
            let tracing = check(profile, &source);
            // The shape that never alternates (period 200) still computes
            // the same, with or without a side.
            assert!(
                tracing.traces_compiled >= 1,
                "shape {index} {profile:?}: {tracing:?}"
            );
        }
    }
}

#[test]
fn a_side_that_changes_a_registers_type_never_joins_the_tree() {
    // `flag` is a boolean in one arm and a number in the other. A side that
    // handed the loop a boolean where the root carries a number cannot link
    // (a loop parameter has one type), so it is rejected, and the loop's work
    // stays bounded: a few records and compiles, not one per iteration.
    let source = "local flag, n = 1, 0 \
                  for i = 1, 100000 do \
                      n = n + 1 \
                      if i % 7 < 3 then flag = true else flag = i end \
                  end return flag, n";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(stats.traces_compiled <= 12, "{profile:?}: {stats:?}");
        assert!(stats.compile_submissions <= 40, "{profile:?}: {stats:?}");
        assert!(stats.side_installs <= 8, "{profile:?}: {stats:?}");
        assert!(stats.recording_starts <= 16, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_loop_whose_arms_write_different_types_to_a_dead_register_still_links() {
    // `t` is a temporary each arm writes with another type and nothing reads
    // across iterations, so it is not a loop parameter and does not stop the
    // side from linking.
    let source = "local n, m = 0, 0 \
                  for i = 1, 120000 do \
                      local t \
                      if i % 7 < 3 then t = i n = n + t else t = true m = m + 1 end \
                  end return n, m";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 110_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_root_that_inlines_a_call_takes_sides_like_any_other() {
    // The root inlines a call, so it pins a closure. A side shares the root's
    // pinned closures and window and adds none, so the alternating branch gets
    // its side, and the loop runs in one tree instead of leaving at every
    // `then` iteration.
    let source = "local function f(x) return x + 1 end \
                  local s, a, b = 0, 0, 0 \
                  for i = 1, 300000 do \
                      s = f(s) \
                      if i % 2 == 0 then a = a + 1 else b = b + 1 end \
                  end return s, a, b";
    for profile in PROFILES {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 200, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 290_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_side_reading_a_global_installs_after_another_side() {
    // The `a` arm records first and the `G` arm second. The first side must
    // not turn the global into a snapshot value, or the tree is refused.
    let source = "G = 5 \
                  local a, b, c = 0, 0, 0 \
                  for i = 1, 300000 do \
                      local m = i % 3 \
                      if m == 0 then c = c + G elseif m == 1 then b = b + 1 else a = a + 1 end \
                  end return a, b, c";
    // Later profiles reach globals through an _ENV upvalue instead.
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let stats = check(profile, source);
        assert!(stats.side_installs >= 2, "{profile:?}: {stats:?}");
        assert_eq!(stats.cache_rejections, 0, "{profile:?}: {stats:?}");
    }
}

/// Runs `source` tracing with compilation delayed by `delay` polls.
fn run_delayed(
    profile: CompatibilityProfile,
    delay: u32,
    source: &str,
) -> (Vec<String>, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    lua.vm_mut().use_delayed_synchronous_jit_compilation(delay);
    lua.vm_mut().count_trace_links();
    lua.vm_mut().keep_trace_branch_guards();
    let results: Vec<Value> = lua.exec(source).unwrap();
    (canon(&results), lua.vm().jit_statistics().trace)
}

#[test]
fn sides_that_each_add_a_window_slot_all_install_whatever_the_compile_delay() {
    // Three arms write three different upvalues. The second side is recorded
    // while the first compiles, so it names a window the tree has outgrown by
    // the time it lands. That must cost it nothing: its guard records again
    // against the grown tree.
    let source = "local up1, up2, up3 = 0, 0, 0 \
                  local function run(n) \
                      local s = 0 \
                      for i = 1, n do \
                          local m = i % 4 \
                          if m == 0 then up1 = up1 + 1 \
                          elseif m == 1 then up2 = up2 + 2 \
                          elseif m == 2 then up3 = up3 + 3 \
                          else s = s + 1 end \
                      end \
                      return s \
                  end \
                  return run(300000), up1, up2, up3";
    let profile = CompatibilityProfile::Lua51;
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    for delay in [0, 1, 2, 3, 5, 8, 13, 21, 34] {
        let (got, stats) = run_delayed(profile, delay, source);
        assert_eq!(got, expected, "delay {delay}");
        assert_eq!(stats.side_installs, 3, "delay {delay}: {stats:?}");
        assert_eq!(stats.side_rejections, 0, "delay {delay}: {stats:?}");
    }
}
