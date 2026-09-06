//! Side traces compiled apart from their tree: a side recorded for a tree that
//! is installed and running becomes a function of its own, which the exit of
//! the guard it continues calls through a cell, so growing the tree costs the
//! compile of the side and not of the tree.
//!
//! The programs are the branchy loops of the side trace tests. The arm the
//! recording starts on is the root's; every other arm takes `1 / modulus` of
//! the iterations, which makes its guard rare enough for its side to be
//! compiled apart (a guard that fails often gets its side in a compile of the
//! tree, see `LINKED_RARITY`). With one accumulator for all arms
//! (`shared_chain_program`) a side writes only a register its root writes too
//! and links back to the loop. With an accumulator of its own per arm it
//! cannot, and leaves at the loop header. Every case compares with an isolated
//! interpreter.

// The subject is the sides of branch guards: value selects stay off.
use super::trace_native::{guarded_vm as native_vm, stats_of};
use super::trace_record::*;
use super::trace_sides::{chain_program_from, flush, run_traced, shared_chain_program};
use super::*;
use crate::vm::jit::trace::TraceExitKind;
use crate::vm::jit::trace::policy::MAX_SIDE_TRACES;

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// A start for the loop variable that puts the recording's iteration on the
/// last arm, which takes all the iterations the others leave.
const START: i64 = 40;

fn has_integers(profile: CompatibilityProfile) -> bool {
    profile.is_lua53_or_later()
}

#[test]
fn a_side_that_links_back_is_compiled_alone_after_its_tree_runs() {
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    for profile in PROFILES {
        let (chunk, header) = shared_chain_program(has_integers(profile), START, 2, 1_024, 400_000);
        let expected = interpreted(profile, &chunk);
        let (vm, results) = run_traced(profile, &chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        // One tree compiled, one side compiled beside it: two compiles, not
        // the tree twice.
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_compiles, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.compile_submissions, 2, "{profile:?}: {stats:?}");
        assert!(
            stats.clif_instructions_side < stats.clif_instructions_tree,
            "{profile:?}: the side is a part of the tree's size: {stats:?}"
        );
        // The side runs natively whenever the arm comes: the loop never
        // leaves for it.
        assert!(stats.side_entries > 200, "{profile:?}: {stats:?}");
        assert!(stats.native_entries < 40, "{profile:?}: {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) < 40,
            "{profile:?}: {stats:?}"
        );
        let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
        assert_eq!(vm.jit.trace.installed_sides(index), Some((1, 1)));
        assert_eq!(
            vm.jit
                .trace
                .installed_compiled(index)
                .unwrap()
                .attached_sides(),
            1
        );
    }
}

#[test]
fn compiling_a_side_costs_less_than_compiling_the_tree_with_it() {
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    let (chunk, _) = shared_chain_program(false, START, 2, 1_024, 400_000);
    let profile = CompatibilityProfile::Lua51;
    let expected = interpreted(profile, &chunk);
    // The same program with every side compiled into its tree.
    let mut together = native_vm(profile);
    together.jit.trace.set_detach_sides(false);
    assert_eq!(together.execute(&chunk).unwrap(), expected);
    let together = stats_of(&together);
    let (apart, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let apart = stats_of(&apart);
    assert_eq!(together.side_compiles, 0, "{together:?}");
    assert_eq!(apart.side_compiles, 1, "{apart:?}");
    // Together the tree is compiled twice, the second time with the side.
    // Apart it is compiled once and the side beside it.
    let together_total = together.clif_instructions_tree + together.clif_instructions_side;
    let apart_total = apart.clif_instructions_tree + apart.clif_instructions_side;
    assert!(
        apart_total < together_total,
        "apart {apart_total}, together {together_total}: {apart:?} {together:?}"
    );
}

#[test]
fn a_side_does_not_recompile_the_tree_it_joins() {
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = shared_chain_program(false, START, 4, 1_024, 800_000);
    let expected = interpreted(profile, &chunk);
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    // Three rare arms: three sides, each compiled apart, and the tree
    // compiled once.
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert_eq!(stats.side_compiles, 3, "{stats:?}");
    assert_eq!(stats.compile_submissions, 4, "{stats:?}");
    assert_eq!(stats.side_installs, 3, "{stats:?}");
    assert!(stats.side_entries > 1_000, "{stats:?}");
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(vm.jit.trace.installed_sides(index), Some((3, 3)));
}

#[test]
fn many_sides_hang_from_one_tree_and_run_natively() {
    for arms in [5_usize, 8, 9] {
        let profile = CompatibilityProfile::Lua51;
        let (chunk, header) = shared_chain_program(false, START, arms, 1_024, 1_600_000);
        let expected = interpreted(profile, &chunk);
        let (vm, results) = run_traced(profile, &chunk);
        assert_eq!(results, expected, "{arms} arms");
        let stats = stats_of(&vm);
        // Up to the tree's room: the rest stay exits.
        let sides = (arms - 1).min(MAX_SIDE_TRACES);
        assert_eq!(stats.side_installs, sides as u64, "{arms} arms: {stats:?}");
        assert_eq!(stats.side_compiles, sides as u64, "{arms} arms: {stats:?}");
        assert_eq!(stats.traces_compiled, 1, "{arms} arms: {stats:?}");
        assert!(stats.side_entries > 1_000, "{arms} arms: {stats:?}");
        let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
        assert_eq!(
            vm.jit.trace.installed_sides(index),
            Some((sides, sides)),
            "{arms} arms"
        );
    }
}

/// The nest of `benchmarks/lua/algorithm/nbody.lua` in small: a function with
/// two pair loops and a loop over the bodies, called from a loop. Its tree is
/// the pair loop; the loops after it are sides that continue each other's
/// terminal exits, and the last returns into the calling loop.
fn advance_program(calls: u64) -> String {
    format!(
        "local bodies = {{}}
         for i = 1, 5 do
             bodies[i] = {{x = i * 1.5, y = i * 0.5, vx = 0.01 * i, vy = 0.02, m = 0.1 + i * 0.01}}
         end
         local function advance(dt)
             for i = 1, #bodies do
                 local a = bodies[i]
                 for j = i + 1, #bodies do
                     local b = bodies[j]
                     local dx, dy = a.x - b.x, a.y - b.y
                     local d2 = dx * dx + dy * dy + 0.5
                     local mag = dt / (d2 * math.sqrt(d2))
                     a.vx, a.vy = a.vx - dx * b.m * mag, a.vy - dy * b.m * mag
                     b.vx, b.vy = b.vx + dx * a.m * mag, b.vy + dy * a.m * mag
                 end
             end
             for i = 1, #bodies do
                 local b = bodies[i]
                 b.x, b.y = b.x + dt * b.vx, b.y + dt * b.vy
             end
         end
         for _ = 1, {calls} do advance(0.01) end
         local total = 0
         for i = 1, #bodies do
             total = total + bodies[i].x + bodies[i].y + bodies[i].vx + bodies[i].vy
         end
         return string.format('%.9f', total)"
    )
}

#[test]
fn a_side_is_compiled_apart_onto_the_exit_of_another_side_compiled_apart() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The side that returns into the calling loop ends in front of a loop of
    // the function, whose side continues its terminal exit, and the last side
    // links back. Each lands in the tree's code in turn, hanging from the exit
    // of the one before it.
    use super::trace_detached_links::{interpreted, traced};
    let profile = CompatibilityProfile::Lua51;
    let source = advance_program(12_000);
    let expected = interpreted(profile, &source);
    let run = traced(profile, &source);
    assert_eq!(run.result, expected.result);
    let stats = &run.stats;
    let vm = run.lua.vm();
    let index = vm.jit.trace.linked_trace_index().expect("a linked tree");
    let apart = vm.jit.trace.installed_apart(index);
    let ir = vm.jit.trace.installed_ir(index).unwrap();
    // A side of the tree compiled apart whose parent is one too.
    let chained = apart
        .iter()
        .filter(|side| {
            ir.sides[**side]
                .parent
                .is_some_and(|parent| apart.contains(&parent))
        })
        .count();
    assert!(chained >= 1, "{apart:?}: {stats:?}");
    assert_eq!(stats.side_compiles as usize, apart.len(), "{stats:?}");
    assert_eq!(stats.link_recompiles, 0, "{stats:?}");
    // The calls ran in the loop around them.
    assert!(stats.native_entries * 4 < 12_000, "{stats:?}");
}

#[test]
fn sides_that_are_not_rare_are_compiled_with_their_tree() {
    // The arms take turns: every guard fails in a third of the iterations,
    // and a call of a side apart would cost the loop what a compile of the
    // tree costs within a short run.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, _) = shared_chain_program(false, 1, 3, 3, 60_000);
    let expected = interpreted(profile, &chunk);
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 0, "{stats:?}");
    assert!(stats.side_installs >= 2, "{stats:?}");
}

#[test]
fn a_side_that_cannot_link_back_leaves_at_the_loop_header() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // Arm 0 runs once in 2048 iterations and adds to an accumulator of its
    // own, which the root does not carry. With the run's help turned off (see
    // `a_side_that_writes_what_the_root_leaves_alone_links_back_through_the_run`)
    // it is compiled apart to resume the interpreter at the loop header,
    // where the tree is entered again.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = chain_program_from(false, START, 2, 20_000, 2_000_000);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    vm.jit.trace.set_detach_link_writes(false);
    let results = vm.execute(&chunk).unwrap();
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert_eq!(stats.link_recompiles, 0, "{stats:?}");
    // The side ran natively, and its exits are polls at the loop header.
    assert!(stats.side_member_exits > 20, "{stats:?}");
    assert!(stats.exits_of(TraceExitKind::Poll) > 20, "{stats:?}");
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(vm.jit.trace.installed_sides(index), Some((1, 1)));
}

#[test]
fn a_side_that_writes_what_the_root_leaves_alone_links_back_through_the_run() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The same program: the accumulator of arm 0 is a register the root never
    // writes, so the root cannot carry it through the loop. The side hands it
    // to the run with the link, which stores it in the entry input the root
    // starts from and writes it into the frame when the run leaves, so that
    // the side is called and links back instead of ending in a poll exit.
    for profile in PROFILES {
        let (chunk, header) =
            chain_program_from(has_integers(profile), START, 2, 20_000, 2_000_000);
        let expected = interpreted(profile, &chunk);
        let (vm, results) = run_traced(profile, &chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.side_compiles, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 20, "{profile:?}: {stats:?}");
        // No poll exit per run of the side: the loop does not leave for it.
        assert!(
            stats.exits_of(TraceExitKind::Poll) < 20,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.native_entries < 40, "{profile:?}: {stats:?}");
        let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
        let compiled = vm.jit.trace.installed_compiled(index).unwrap();
        assert_eq!(compiled.attached_sides(), 1, "{profile:?}");
        assert!(!compiled.writebacks().is_empty(), "{profile:?}");
    }
}

#[test]
fn sides_that_leave_at_the_loop_header_are_compiled_in_when_they_prove_hot() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // Under the baseline policy no tree earns the peak tier, which would put
    // the side inline first.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = chain_program_from(false, START, 2, 20_000, 45_000_000);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    vm.jit.trace.set_detach_link_writes(false);
    vm.jit
        .trace
        .set_compile_tier(crate::jit::OptimizationTier::Baseline);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
    assert_eq!(stats.link_recompiles, 1, "{stats:?}");
    // The tree was compiled again with the side inline.
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(
        vm.jit
            .trace
            .installed_compiled(index)
            .unwrap()
            .attached_sides(),
        0
    );
    assert_eq!(vm.jit.trace.installed_sides(index), Some((1, 1)));
    // After it the arm no longer leaves: the exits stop at the compile.
    assert!(
        stats.exits_of(TraceExitKind::Poll) < 45_000_000 / 20_000,
        "{stats:?}: the exits stop once the side links back"
    );
}

#[test]
fn sides_called_too_often_are_compiled_in() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // Under the baseline policy no tree earns the peak tier, which would put
    // the side inline first.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = shared_chain_program(false, START, 2, 600, 24_000_000);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    vm.jit
        .trace
        .set_compile_tier(crate::jit::OptimizationTier::Baseline);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
    // A call of the side costs the loop a hundred instructions over the side
    // inline. After enough of them the tree is compiled again.
    assert_eq!(stats.link_recompiles, 1, "{stats:?}");
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(
        vm.jit
            .trace
            .installed_compiled(index)
            .unwrap()
            .attached_sides(),
        0
    );
    assert_eq!(vm.jit.trace.installed_sides(index), Some((1, 1)));
}

#[test]
fn a_tree_mir_compiled_stays_off_the_peak_tier_with_a_side_compiled_apart() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The same run under the policy: MIR's loop is the shorter one, with or
    // without a side attached, so the tree keeps its tier.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = shared_chain_program(false, START, 2, 1_024, 2_300_000);
    let expected = interpreted(profile, &chunk);
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(
        vm.jit.trace.installed_tier(index),
        Some(crate::jit::OptimizationTier::Optimized),
        "{stats:?}"
    );
}

#[test]
fn a_tree_at_the_peak_tier_takes_the_sides_compiled_apart_with_it() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The loop runs past the iterations that earn the peak tier with a side
    // attached: the peak compile holds it inline. The tree is MIR's, which
    // the policy keeps off the tier, so the test asks for it.
    crate::jit::debug_env::peak_for_this_test();
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = shared_chain_program(false, START, 2, 1_024, 2_300_000);
    let expected = interpreted(profile, &chunk);
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
    let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
    assert_eq!(
        vm.jit.trace.installed_tier(index),
        Some(crate::jit::OptimizationTier::Peak),
        "{stats:?}"
    );
    let compiled = vm.jit.trace.installed_compiled(index).unwrap();
    assert_eq!(compiled.attached_sides(), 0, "{stats:?}");
    assert_eq!(compiled.side_members(), &[Some(1)]);
}

#[test]
fn a_side_added_while_the_worker_compiles_installs_when_it_is_done() {
    crate::jit::debug_env::sides_apart_for_this_test();
    // The compile result installs only after some number of safepoint polls,
    // as a background worker's would after some time: the loop runs on the
    // tree's code meanwhile, and the guards of the sides keep exiting.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, _) = shared_chain_program(false, START, 3, 1_024, 400_000);
    let chunk = Arc::new(chunk);
    let expected = interpreted(profile, &chunk);
    let mut composed = 0;
    for delay in [0, 1, 2, 4, 9, 30, 120] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.jit.trace.set_value_selects(false);
        vm.use_delayed_synchronous_jit_compilation(delay);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut waited = false;
        let results = loop {
            if vm.jit.trace.side_compile_pending() {
                // A side compile is outstanding while the tree runs.
                waited = true;
            }
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break results;
            }
        };
        assert_eq!(results, expected, "delay {delay}");
        let stats = stats_of(&vm);
        assert!(stats.side_installs >= 2, "delay {delay}: {stats:?}");
        composed += u64::from(waited && stats.side_compiles > 0);
    }
    assert!(composed > 3, "sides were compiled apart in {composed} runs");
}

#[test]
fn a_flush_while_a_side_compiles_drops_it_and_the_program_stays_correct() {
    crate::jit::debug_env::sides_apart_for_this_test();
    let profile = CompatibilityProfile::Lua51;
    let (chunk, header) = shared_chain_program(false, START, 2, 1_024, 600_000);
    let chunk = Arc::new(chunk);
    let expected = interpreted(profile, &chunk);
    for delay in [3_u32, 10, 40] {
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
        vm.jit.trace.set_value_selects(false);
        vm.use_delayed_synchronous_jit_compilation(delay);
        let mut frames = frames_for(&vm, &chunk, None);
        // Run until a side compile is outstanding.
        let mut steps = 0;
        while !vm.jit.trace.side_compile_pending() {
            assert!(vm.step(&mut frames).unwrap().is_none());
            steps += 1;
            assert!(steps < 4_000_000, "delay {delay}: no side compile was made");
        }
        let before = stats_of(&vm);
        flush(&mut vm, &mut frames);
        assert_eq!(vm.jit.trace.installed_count(), 0);
        let results = loop {
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break results;
            }
        };
        assert_eq!(results, expected, "delay {delay}");
        let after = stats_of(&vm);
        // The finished side found no trace to join, and the loop recorded
        // its tree and side again.
        assert!(
            after.stale_completions > before.stale_completions,
            "delay {delay}: {after:?}"
        );
        assert!(after.side_compiles >= 1, "delay {delay}: {after:?}");
        assert!(vm.jit.trace.installed_index_at_pc(header).is_some());
    }
}

#[test]
fn every_state_a_tree_with_sides_apart_publishes_is_on_the_interpreters_path() {
    crate::jit::debug_env::sides_apart_for_this_test();
    for arms in [2_usize, 3, 4] {
        let (chunk, _) = shared_chain_program(false, START, arms, 600, 14_000);
        let chunk = Arc::new(chunk);
        let reference = drive(&chunk, None);
        let live_in = crate::jit_ir::observable_registers(&chunk);
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        let mut frames = frames_for(&vm, &chunk, None);
        let mut seen = 0_usize;
        let results = loop {
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break scalars(&results);
            }
            let state = capture(&frames);
            let found = reference.states[seen..]
                .iter()
                .position(|candidate| candidate.same_as_with(&state, &live_in))
                .unwrap_or_else(|| panic!("{arms} arms: {state:?} is off the path"));
            seen += found;
        };
        assert_eq!(results, reference.results, "{arms} arms");
        assert!(stats_of(&vm).side_compiles >= 1, "{arms} arms");
    }
}

#[test]
fn background_compilation_of_sides_agrees_with_the_interpreter() {
    // The default broker compiles on a worker thread: sides land at whatever
    // safepoint finds them.
    let profile = CompatibilityProfile::Lua51;
    for arms in [2_usize, 3, 5] {
        let (chunk, _) = shared_chain_program(false, START, arms, 1_024, 600_000);
        let expected = interpreted(profile, &chunk);
        for _ in 0..4 {
            let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
            vm.jit.trace.set_value_selects(false);
            let results = vm.execute(&chunk).unwrap();
            assert_eq!(results, expected, "{arms} arms");
        }
    }
}

#[test]
fn turning_detaching_off_compiles_every_side_with_its_tree() {
    let profile = CompatibilityProfile::Lua51;
    let (chunk, _) = shared_chain_program(false, START, 3, 1_024, 120_000);
    let expected = interpreted(profile, &chunk);
    let mut vm = native_vm(profile);
    vm.jit.trace.set_detach_sides(false);
    assert_eq!(vm.execute(&chunk).unwrap(), expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 0, "{stats:?}");
    assert!(stats.side_installs >= 2, "{stats:?}");
}

/// A nest around the calling loop of `advance_program` whose calls end early
/// at a chosen outer step: the exit comes from a side that continues the
/// native link, with the loop variables the interpreter sees after it.
fn early_exit_program(calls: u64) -> String {
    format!(
        "local function sweep(n, m, stop)
             local acc = 0
             for i = 1, n do
                 if i == stop then return acc, i, -1 end
                 for j = i, m do
                     acc = acc + (i * j) % 7
                 end
             end
             return acc, n, 0
         end
         local total, last, hits = 0, 0, 0
         for k = 1, {calls} do
             -- Most calls run to the end; one in 64 stops at an outer step.
             local stop = (k % 64 == 0) and (2 + k % 5) or 100
             local acc, i, early = sweep(8 + k % 3, 40, stop)
             total = (total + acc) % 1000003
             last = i
             hits = hits - early
         end
         return total, last, hits"
    )
}

/// An infinite nest, left by an error a count hook raises. The hook is set
/// after the nest ran hot and linked natively, so a running tree must
/// give way to it.
fn endless_program() -> &'static str {
    "local function sweep(n, m, stop)
         local acc = 0
         for i = 1, n do
             if i == stop then return acc, i, -1 end
             for j = i, m do
                 acc = acc + (i * j) % 7
             end
         end
         return acc, n, 0
     end
     local total, last, hits = 0, 0, 0
     for k = 1, 1000000000 do
         -- The nest runs hot, and its sides link natively, before the hook.
         if k == 8000 then
             debug.sethook(function() error('stopped') end, '', 200000)
         end
         local stop = (k % 64 == 0) and (2 + k % 5) or 100
         local acc, i, early = sweep(8 + k % 3, 40, stop)
         total = (total + acc) % 1000003
         last = i
         hits = hits - early
     end
     return total, last, hits"
}

#[test]
fn native_code_calls_the_side_that_returns_into_the_calling_loop() {
    use super::trace_detached_links::{interpreted, traced};
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    for profile in PROFILES {
        let source = advance_program(12_000);
        let expected = interpreted(profile, &source);
        let run = traced(profile, &source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        let vm = run.lua.vm();
        let index = vm.jit.trace.linked_trace_index().expect("a linked tree");
        let compiled = vm.jit.trace.installed_compiled(index).unwrap();
        assert!(compiled.links_natively(), "{profile:?}: {stats:?}");
        // The calls of the side and the links back to the loop are native
        // code's own: the run is entered for the calls before the side
        // landed and for no others, and every side entry after it is a
        // call native code made.
        assert!(stats.native_side_calls > 10_000, "{profile:?}: {stats:?}");
        assert!(
            stats.side_entries >= stats.native_side_calls,
            "{profile:?}: {stats:?}"
        );
        assert!(stats.native_entries * 4 < 12_000, "{profile:?}: {stats:?}");
        assert_eq!(stats.link_recompiles, 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn the_loop_variables_after_a_run_that_native_links_left_are_the_interpreters() {
    use super::trace_detached_links::{interpreted, traced};
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    for profile in PROFILES {
        let source = early_exit_program(8_000);
        let expected = interpreted(profile, &source);
        let run = traced(profile, &source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        assert!(stats.native_side_calls > 1_000, "{profile:?}: {stats:?}");
        let vm = run.lua.vm();
        let index = vm.jit.trace.linked_trace_index().expect("a linked tree");
        assert!(
            vm.jit
                .trace
                .installed_compiled(index)
                .unwrap()
                .links_natively(),
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_hot_nest_native_code_links_in_still_stops_for_a_count_hook() {
    use super::trace_detached_links::{interpreted, traced};
    // Asserts what Cranelift emits.
    crate::jit::debug_env::cranelift_for_this_test();
    for profile in PROFILES {
        let expected = interpreted(profile, endless_program());
        let run = traced(profile, endless_program());
        assert_eq!(run.result, expected.result, "{profile:?}");
        // The hook's error ends the run, as the interpreter's does. The hook
        // keeps native-link trees from being entered once it is set, so the
        // interpreter raises it; fuel on the native link path is covered by
        // `fuel_is_polled_on_the_native_link_path` in tree_tests.rs.
        let error = run.result.as_ref().expect_err("the hook stops the loop");
        assert!(error.contains("stopped"), "{profile:?}: {error}");
        assert!(
            run.stats.native_side_calls > 0,
            "{profile:?}: {:?}",
            run.stats
        );
    }
}

#[test]
fn a_root_without_loop_parameters_to_pass_on_returns_its_codes_to_the_run() {
    // The accumulators of the arms are the loop's own: nothing is passed
    // through an iteration unchanged, so a link comes back through the run.
    for profile in PROFILES {
        let (chunk, header) = shared_chain_program(has_integers(profile), START, 2, 1_024, 400_000);
        let expected = interpreted(profile, &chunk);
        let (vm, results) = run_traced(profile, &chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        let index = vm.jit.trace.installed_index_at_pc(header).unwrap();
        let compiled = vm.jit.trace.installed_compiled(index).unwrap();
        assert!(!compiled.links_natively(), "{profile:?}: {stats:?}");
        assert_eq!(stats.native_side_calls, 0, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 200, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_nest_shaped_like_mandelbrot_makes_the_same_decisions_with_native_links() {
    // `algorithm/mandelbrot` in small: a loop that runs 25 iterations on
    // average inside two loops. Its outer sides come while the tree is young
    // and stay in its code, so the tree compiles once, no side is compiled
    // apart and nothing is recompiled; that the root can link natively must
    // change none of it, and costs the loop nothing in its own code (the
    // benchmark measured +26% when the link kept the run's buffers in
    // registers across the loop).
    use super::trace_detached_links::{interpreted, traced};
    let source = "local size = 48
         local checksum = 0
         for repeat_index = 1, 2 do
             for y = 0, size - 1 do
                 local ci = 2 * y / size - 1
                 for x = 0, size - 1 do
                     local cr = 2 * x / size - 1.5
                     local zr, zi, iterations = 0, 0, 0
                     while zr * zr + zi * zi <= 4 and iterations < 50 do
                         zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                         iterations = iterations + 1
                     end
                     checksum = checksum + iterations
                 end
             end
         end
         return checksum";
    for profile in PROFILES {
        let expected = interpreted(profile, source);
        let run = traced(profile, source);
        assert_eq!(run.result, expected.result, "{profile:?}");
        let stats = &run.stats;
        // Where the `while` loop's root is given up (Lua 5.4's integer `x`
        // turns its outer side type unstable), the `x` loop's root ends in
        // front of it and the tree compiles once more with its loop side
        // (see "Outer roots" in the module docs of `trace`).
        assert_eq!(
            stats.traces_compiled,
            1 + stats.loop_sides,
            "{profile:?}: {stats:?}"
        );
        assert_eq!(stats.side_compiles, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.link_recompiles, 0, "{profile:?}: {stats:?}");
        assert_eq!(stats.native_side_calls, 0, "{profile:?}: {stats:?}");
    }
}

/// A loop whose rare side (the `x == 0` arm of `nz`) is compiled apart and
/// links back. After a link the root runs from its entry again, and an exit
/// it takes before the iteration writes the loop's variables keeps them as
/// the entry gave them: the frame must get the values of the last link, not
/// those of the run's first entry. With the first entry's values the
/// interpreter ran hundreds of iterations again, which `c[1]` counts.
/// Found by `oximond-difftest run-jit` (lua52 seed 17473173835745953184).
const LINK_THEN_ENTRY_EXIT: &str = "
    local function nz(x) if x == 0 then return 1 end return x end
    local t = {0, 8, 8}
    local k = 239
    local x
    local c = {0}
    local i = 0
    while i < 28931 do
      i = i + 1
      x = i * 0.5
      c[1] = c[1] + 1
      if k > 2741 then k = (k - 2741) % 1000003 else k = (k * 3 + 2741) % 1000003 end
      t[(4294967296 % #t) + 1] = k % nz(t[(k % #t) + 1])
      k = (string.byte(tostring(9), 1) + k) % 1000003
      if i == 18834 then
        t = {8}
      end
    end
    return k, c[1], i";

fn check_exit_after_a_link() {
    use super::trace_detached_links::{interpreted, traced};
    for profile in PROFILES {
        let expected = interpreted(profile, LINK_THEN_ENTRY_EXIT);
        let run = traced(profile, LINK_THEN_ENTRY_EXIT);
        let stats = &run.stats;
        assert_eq!(run.result, expected.result, "{profile:?}: {stats:?}");
        assert!(stats.side_compiles >= 1, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_exit_after_a_link_leaves_the_loop_variables_of_the_link() {
    check_exit_after_a_link();
}

#[test]
fn an_exit_after_a_native_link_leaves_the_loop_variables_of_the_link() {
    // Cranelift's root calls the side and links from its own code.
    crate::jit::debug_env::cranelift_for_this_test();
    check_exit_after_a_link();
}

#[test]
fn a_small_tree_mir_compiled_takes_its_side_in() {
    // The tree is a few operations long, and the MIR backend compiles it
    // again with its one side for less than Cranelift takes for the side
    // alone: no side is compiled apart.
    let profile = CompatibilityProfile::Lua51;
    let (chunk, _) = shared_chain_program(false, START, 2, 1_024, 400_000);
    let expected = interpreted(profile, &chunk);
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 0, "{stats:?}");
    assert_eq!(stats.side_installs, 1, "{stats:?}");
    assert!(stats.side_entries > 200, "{stats:?}");
    // The same side is compiled apart when that rule is off, so it was the
    // rule that took it in.
    crate::jit::debug_env::sides_apart_for_this_test();
    let (vm, results) = run_traced(profile, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_compiles, 1, "{stats:?}");
}

#[test]
fn a_side_that_meets_a_short_inner_loop_is_recorded_again_unrolling_it() {
    use super::trace_detached_links::{interpreted as interpreted_source, traced};
    // The root is the arm without the inner loop. The side of the other arm
    // meets the inner loop, which its first recording does not follow: that
    // abort marks the root's site, so the next recording unrolls the ten
    // trips and the side is compiled, instead of the guard spending its
    // attempts on the same abort.
    let source = "local s = 0\nfor i = 1, 200000 do\n  if i % 4 == 0 then\n    \
                  for j = 1, 10 do s = s + j end\n  else\n    s = s + 1\n  end\nend\nreturn s\n";
    for profile in PROFILES {
        let run = traced(profile, source);
        assert_eq!(
            run.result,
            interpreted_source(profile, source).result,
            "{profile:?}"
        );
        let stats = &run.stats;
        assert!(stats.side_traces_recorded >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.side_aborts <= 1, "{profile:?}: {stats:?}");
    }
}
