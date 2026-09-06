//! Side traces: a guard that fails often continues in a second trace instead
//! of leaving for the interpreter, and the root and its sides run as one
//! native function.
//!
//! The programs are hand-assembled loops with a branch that picks one of
//! several arms. A trace records whichever arm the loop is on when it gets
//! hot, so every other arm is a guard that fails. Every case compares with an
//! isolated interpreter, and counters prove the sides ran natively: the
//! number of exits stays small while the loop does tens of thousands of
//! iterations, and `side_entries` counts the jumps into sides directly.

// The subject is the sides of branch guards: value selects stay off.
use super::trace_native::{guarded_vm, guarded_vm as native_vm, stats_of};
use super::trace_record::*;
use super::*;
use crate::jit_ir::trace::TraceIr;
use crate::vm::jit::trace::policy::{
    MAX_SIDE_DEPTH, MAX_SIDE_TRACES, SIDE_EXIT_THRESHOLD, SiteState,
};
use crate::vm::jit::trace::{TraceExitKind, TraceRun};

/// A number in the profile's number type.
fn number(integers: bool, value: i64) -> Constant {
    if integers {
        Constant::Integer(value)
    } else {
        Constant::Number(value as f64)
    }
}

/// `for i = 1, limit do r = i % modulus; if r < 1 then acc0 = acc0 + i
/// elseif r < 2 then acc1 = acc1 + i ... else acc_{arms-1} = acc_{arms-1} + i
/// end end return acc0, ..., acc_{arms-1}`.
///
/// With `modulus == arms` the arms take turns, one per iteration. With a
/// larger modulus only the first `arms - 1` residues pick an arm of their
/// own, and the last arm takes the rest, so arm 0 runs once in `modulus`
/// iterations. Returns the chunk and its loop header.
pub(super) fn chain_program(
    integers: bool,
    arms: usize,
    modulus: i64,
    limit: i64,
) -> (Chunk, usize) {
    chain_program_from(integers, 1, arms, modulus, limit)
}

/// [`chain_program`] counting from `start`, so a test can choose which arm
/// the iteration that starts a recording is on.
pub(super) fn chain_program_from(
    integers: bool,
    start: i64,
    arms: usize,
    modulus: i64,
    limit: i64,
) -> (Chunk, usize) {
    chain_program_impl(integers, start, arms, modulus, limit, false)
}

/// [`chain_program`] whose arms all add to one accumulator, so that a side
/// writes only a register its root writes too, and links back to the loop
/// without widening the root. Returns the accumulator.
pub(super) fn shared_chain_program(
    integers: bool,
    start: i64,
    arms: usize,
    modulus: i64,
    limit: i64,
) -> (Chunk, usize) {
    chain_program_impl(integers, start, arms, modulus, limit, true)
}

fn chain_program_impl(
    integers: bool,
    start: i64,
    arms: usize,
    modulus: i64,
    limit: i64,
    shared: bool,
) -> (Chunk, usize) {
    let mut constants = vec![
        number(integers, 1),
        number(integers, limit),
        number(integers, 0),
        number(integers, modulus),
    ];
    for threshold in 1..arms {
        constants.push(number(integers, i64::try_from(threshold).unwrap()));
    }
    let accumulator = |arm: usize| u8::try_from(if shared { 4 } else { 4 + arm }).unwrap();
    let selector = u8::try_from(4 + arms).unwrap();
    let remainder = selector + 1;
    let test = selector + 2;
    let threshold = |arm: usize| u8::try_from(7 + arms + arm).unwrap();
    constants.push(number(integers, start));
    let first = u16::try_from(constants.len() - 1).unwrap();
    let mut code = vec![load(0, first), load(1, 1), load(2, 0)];
    for arm in 0..arms {
        code.push(load(accumulator(arm), 2));
    }
    code.push(load(selector, 3));
    for arm in 0..arms - 1 {
        code.push(load(threshold(arm), u16::try_from(4 + arm).unwrap()));
    }
    let header = code.len() + 1;
    let latch = header + 1 + 4 * (arms - 1) + 1;
    code.push(for_prep(0, code.len(), latch));
    code.push(binary(OpCode::Modulo, remainder, 3, selector));
    for arm in 0..arms - 1 {
        let start = code.len();
        code.push(binary(OpCode::LessThan, test, remainder, threshold(arm)));
        code.push(jump_if_false(test, start + 1, start + 4));
        code.push(binary(OpCode::Add, accumulator(arm), accumulator(arm), 3));
        code.push(jump_to(start + 3, latch));
    }
    let last = accumulator(arms - 1);
    code.push(binary(OpCode::Add, last, last, 3));
    assert_eq!(code.len(), latch);
    code.push(for_loop(0, latch, header));
    code.push(Instruction::return_values(
        4,
        if shared {
            1
        } else {
            u8::try_from(arms).unwrap()
        },
    ));
    let registers = u8::try_from(7 + 2 * arms).unwrap();
    (Chunk::new(registers, constants, code), header)
}

/// `jit.flush()` from the program: the request is served at the next
/// consult of a loop header, which the steps below reach.
pub(super) fn flush(vm: &mut Vm, frames: &mut crate::vm::stack::CallStack) {
    vm.runtime_mut().request_jit_flush();
    for _ in 0..10_000 {
        if vm.jit.trace.installed_count() == 0 && !vm.runtime().jit_flush_requested() {
            return;
        }
        assert!(vm.step(frames).unwrap().is_none());
    }
    panic!("the flush was never served");
}

/// Runs `chunk` under the trace engine with synchronous compilation.
/// Branch guards stay guards (see [`guarded_vm`]): these tests are about
/// their sides.
pub(super) fn run_traced(profile: CompatibilityProfile, chunk: &Chunk) -> (Vm, Vec<Value>) {
    let mut vm = guarded_vm(profile);
    let results = vm.execute(chunk).unwrap();
    (vm, results)
}

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn has_integers(profile: CompatibilityProfile) -> bool {
    profile.is_lua53_or_later()
}

#[test]
fn an_alternating_branch_gets_a_side_trace_in_every_profile() {
    for profile in PROFILES {
        let integers = has_integers(profile);
        let (chunk, _) = chain_program(integers, 2, 7, 60_000);
        let expected = interpreted(profile, &chunk);
        let (vm, results) = run_traced(profile, &chunk);
        assert_eq!(results, expected, "{profile:?}");
        let stats = stats_of(&vm);
        assert_eq!(stats.side_installs, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_recordings, 1, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_aborts, 0, "{profile:?}: {stats:?}");
        // The arm the root did not record runs one iteration in seven (the
        // residue 0), and every one of those jumps into the side natively.
        assert!(stats.side_entries > 8_000, "{profile:?}: {stats:?}");
        // Almost every iteration ran natively, behind a handful of exits:
        // the entries before the side landed, the loop's end, and polls.
        assert!(stats.iterations > 59_000, "{profile:?}: {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) < 4_000,
            "{profile:?}: {stats:?}"
        );
        assert_eq!(vm.jit.trace.installed_count(), 1, "{profile:?}");
    }
}

#[test]
fn the_side_trace_is_not_retired_as_unproductive() {
    // Without the side this trace exits at every `then` iteration, so every
    // window of entries would close below the productivity bar. The side is
    // on its way, so the trace is forgiven, and once it lands the trace is
    // productive for good.
    let (chunk, header) = chain_program(false, 2, 7, 90_000);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let (vm, results) = run_traced(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_installs, 1, "{stats:?}");
    assert!(
        matches!(
            vm.jit.trace.site_state_any(header),
            Some(SiteState::Installed { .. })
        ),
        "{:?}",
        vm.jit.trace.site_state_any(header)
    );
    assert_eq!(vm.jit.trace.installed_count(), 1);
}

#[test]
fn arms_that_take_turns_each_get_a_side_trace() {
    for arms in [3_usize, 4, 5] {
        let (chunk, _) = chain_program(false, arms, i64::try_from(arms).unwrap(), 40_000);
        let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
        let (vm, results) = run_traced(CompatibilityProfile::Lua51, &chunk);
        assert_eq!(results, expected, "{arms} arms");
        let stats = stats_of(&vm);
        // The root records one arm, and every other arm needs a side. Some
        // are sides of sides, since a side continues from the arm it starts
        // on.
        assert_eq!(
            stats.side_installs,
            u64::try_from(arms - 1).unwrap(),
            "{arms} arms: {stats:?}"
        );
        assert!(stats.iterations > 39_000, "{arms} arms: {stats:?}");
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) < 4_000,
            "{arms} arms: {stats:?}"
        );
        assert!(
            stats.side_entries > 30_000 * (arms as u64 - 1) / arms as u64,
            "{arms} arms: {stats:?}"
        );
    }
}

#[test]
fn a_side_of_a_side_exists_and_stays_within_the_depth_limit() {
    // Which arm the recording iteration is on depends on where the loop
    // starts. When it is the first arm, the root holds one branch guard, and
    // the arms behind it are reached through sides that hold guards of their
    // own: sides of sides. Try every phase of a five-arm loop.
    let mut deepest = 0;
    let mut most = 0;
    for start in 1..=5 {
        let (chunk, header) = chain_program_from(false, start, 5, 5, 20_000);
        let chunk = Arc::new(chunk);
        let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
        let mut vm = native_vm(CompatibilityProfile::Lua51);
        let mut frames = frames_for(&vm, &chunk, None);
        let results = loop {
            if let Some(index) = vm.jit.trace.installed_index_at_pc(header) {
                let ir = vm.jit.trace.installed_ir(index).unwrap();
                most = most.max(ir.sides.len());
                deepest = deepest.max(depth_of(ir));
            }
            if let Some(results) = vm.step(&mut frames).unwrap() {
                break results;
            }
        };
        assert_eq!(results, expected, "start {start}");
    }
    assert!(
        deepest >= 2,
        "arms behind arms need sides of sides: {deepest}"
    );
    assert!(deepest <= MAX_SIDE_DEPTH as usize, "{deepest}");
    assert!(most <= MAX_SIDE_TRACES, "{most}");
}

/// The longest chain of sides in a tree.
fn depth_of(ir: &TraceIr) -> usize {
    let mut deepest = 0;
    for side in 0..ir.sides.len() {
        let mut depth = 1;
        let mut at = ir.sides[side].parent;
        while let Some(parent) = at {
            depth += 1;
            at = ir.sides[parent].parent;
        }
        deepest = deepest.max(depth);
    }
    deepest
}

#[test]
fn a_wide_branch_stops_at_the_tree_limits_and_stays_correct() {
    // Twelve arms need eleven sides. A tree holds eight. The arms left over
    // keep exiting to the interpreter, and the loop is still right.
    let (chunk, header) = chain_program(false, 12, 12, 30_000);
    let chunk = Arc::new(chunk);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    let mut most = 0;
    let results = loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header) {
            let ir = vm.jit.trace.installed_ir(index).unwrap();
            assert!(depth_of(ir) <= MAX_SIDE_DEPTH as usize);
            most = most.max(ir.sides.len());
        }
        if let Some(results) = vm.step(&mut frames).unwrap() {
            break results;
        }
    };
    assert_eq!(results, expected);
    assert!(most <= MAX_SIDE_TRACES, "{most}");
    let stats = stats_of(&vm);
    // A bounded tree means bounded compiles: a root and at most eight tree
    // rebuilds for each time the loop records a root again, and the loop
    // does not record for ever.
    let roots = stats.recording_starts;
    assert!(roots <= 8, "{stats:?}");
    assert!(
        stats.compile_submissions <= roots * (1 + MAX_SIDE_TRACES as u64 + 1),
        "{stats:?}"
    );
}

#[test]
fn a_rarely_failing_branch_stays_an_exit() {
    // Arm 0 runs once in a hundred iterations: fewer than the threshold's
    // worth of exits in the whole run, so no side trace is recorded for it.
    let limit = 100 * i64::from(SIDE_EXIT_THRESHOLD) - 300;
    let (chunk, _) = chain_program(false, 2, 100, limit);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let (vm, results) = run_traced(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_recordings, 0, "{stats:?}");
    assert_eq!(stats.side_installs, 0, "{stats:?}");
    assert!(stats.iterations > 0, "{stats:?}");
}

#[test]
fn a_branch_that_fails_often_enough_records_exactly_once_per_guard() {
    // Arm 0 runs once in 20 iterations; over 20 000 iterations that is a
    // thousand failures, far past the threshold, yet one side trace covers
    // them all and nothing records again.
    let (chunk, _) = chain_program(false, 2, 20, 20_000);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let (vm, results) = run_traced(CompatibilityProfile::Lua51, &chunk);
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_recordings, 1, "{stats:?}");
    assert_eq!(stats.side_installs, 1, "{stats:?}");
    assert!(stats.side_entries >= 900, "{stats:?}");
}

#[test]
fn a_side_whose_tree_is_refused_for_room_is_recorded_a_bounded_number_of_times() {
    // The cache's metadata cap leaves room for the root alone, so every
    // tree with the side is refused for room. Each refusal spends one of the
    // guard's attempts: the side is recorded at most `MAX_SIDE_ATTEMPTS`
    // times, not again at every threshold's worth of failures (145
    // recordings of one side of Mineclonia's `get_structure_starts`, whose
    // tree outgrew the 16 MB cap, before the bound).
    use crate::vm::jit::trace::cache::CacheLimits;
    use crate::vm::jit::trace::policy::MAX_SIDE_ATTEMPTS;
    let (chunk, header) = chain_program(false, 2, 7, 200_000);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    while vm.jit.trace.installed_index_at_pc(header).is_none() {
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    let meta = vm.jit.trace.cache_mut().meta_bytes();
    vm.jit.trace.cache_mut().set_limits(CacheLimits {
        meta_bytes: meta + 1,
        ..CacheLimits::default()
    });
    let results = loop {
        if let Some(results) = vm.step(&mut frames).unwrap() {
            break results;
        }
    };
    assert_eq!(results, expected);
    let stats = stats_of(&vm);
    assert_eq!(stats.side_installs, 0, "{stats:?}");
    assert!(stats.side_recordings >= 1, "{stats:?}");
    assert!(
        stats.side_recordings <= u64::from(MAX_SIDE_ATTEMPTS),
        "{stats:?}"
    );
}

#[test]
fn a_result_for_an_older_tree_is_never_installed_over_a_newer_one() {
    // Two recompiles of one trace race: the tree changed while the older
    // request compiled. Its result names the version it was asked for, and
    // is dropped.
    let (chunk, header) = chain_program(false, 2, 7, 60_000);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    let index = loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && stats_of(&vm).side_installs == 1
        {
            break index;
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    };
    let before = stats_of(&vm);
    let sides = vm.jit.trace.installed_sides(index);
    assert_eq!(sides, Some((1, 1)));
    // Ask for a compile of the current tree under the version before.
    assert!(vm.jit.trace.queue_stale_retree(index));
    vm.service_trace_compilations();
    let after = stats_of(&vm);
    assert_eq!(after.stale_completions, before.stale_completions + 1);
    assert_eq!(after.side_installs, before.side_installs);
    assert_eq!(vm.jit.trace.installed_sides(index), sides);
    assert!(!vm.jit.trace.side_pending(index));
}

#[test]
fn a_flush_at_any_moment_of_a_side_recording_leaves_a_correct_program() {
    // The compile result installs only after some number of safepoint polls,
    // as a background worker's would after some time. Sweeping that delay and
    // the moment of the flush puts the flush before the recording, during it,
    // while the tree compiles and after it landed. Every run must finish with
    // the interpreter's answer.
    let (chunk, _) = chain_program(false, 3, 3, 12_000);
    let chunk = Arc::new(chunk);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut stale = 0;
    let mut installed_after_flush = 0;
    for delay in [0, 1, 2, 3, 5, 8, 13, 40] {
        for flush_at in [200, 450, 800, 1_500, 3_000, 6_000, 12_000, 30_000] {
            let mut vm = Vm::with_profile_and_execution_mode(
                CompatibilityProfile::Lua51,
                ExecutionMode::Tracing,
            );
            vm.use_delayed_synchronous_jit_compilation(delay);
            let mut frames = frames_for(&vm, &chunk, None);
            let mut steps = 0;
            let mut installs_at_flush = 0;
            let results = loop {
                steps += 1;
                if steps == flush_at {
                    installs_at_flush = stats_of(&vm).side_installs;
                    vm.runtime_mut().request_jit_flush();
                }
                if let Some(results) = vm.step(&mut frames).unwrap() {
                    break results;
                }
            };
            assert_eq!(results, expected, "delay {delay}, flush at {flush_at}");
            let stats = stats_of(&vm);
            stale += stats.stale_completions;
            if stats.side_installs > installs_at_flush && steps > flush_at {
                installed_after_flush += 1;
            }
        }
    }
    assert!(stale > 0, "some flush landed on an outstanding compile");
    assert!(
        installed_after_flush > 0,
        "some tree was rebuilt after a flush"
    );
}

#[test]
fn a_flush_after_the_side_landed_starts_the_tree_over() {
    let (chunk, header) = chain_program(false, 2, 7, 120_000);
    let chunk = Arc::new(chunk);
    let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    for _ in 0..4_000_000 {
        if stats_of(&vm).side_installs == 1 {
            break;
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    }
    assert_eq!(stats_of(&vm).side_installs, 1);
    flush(&mut vm, &mut frames);
    assert_eq!(vm.jit.trace.installed_count(), 0);
    assert_eq!(vm.jit.trace.site_state_any(header), None);
    let results = loop {
        if let Some(results) = vm.step(&mut frames).unwrap() {
            break results;
        }
    };
    assert_eq!(results, expected);
    // The loop recorded again and found its side again.
    let stats = stats_of(&vm);
    assert_eq!(stats.side_installs, 2, "{stats:?}");
}

#[test]
fn a_fuel_poll_inside_a_side_returns_to_the_interpreter_in_step() {
    // Poll every few loop-backs through a tree and check that every state
    // the trace publishes lies on the interpreter's own path, from the
    // iteration it stopped at.
    let (chunk, header) = chain_program(false, 3, 3, 12_000);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    let index = loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && vm
                .jit
                .trace
                .installed_sides(index)
                .is_some_and(|(_, linked)| linked == 2)
        {
            break index;
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    };
    let reference = drive(&chunk, None);
    let visits = header_visits(&reference, header);
    let start = &reference.states[visits[200]];
    let live_in = crate::jit_ir::observable_registers(&chunk);
    let mut frames = frames_for(&vm, &chunk, Some(start));
    let mut polls = 0;
    let mut cursor = visits[200];
    loop {
        let fuel = 5;
        let run = {
            let jit = &mut vm.jit;
            jit.trace.enter(
                &mut jit.statistics.trace,
                index,
                &mut frames,
                0,
                fuel,
                &mut vm.runtime,
            )
        };
        let TraceRun::Ran { poll, .. } = run else {
            panic!("the trace enters at the header");
        };
        let state = capture(&frames);
        let found = reference.states[cursor..]
            .iter()
            .position(|candidate| candidate.same_as_with(&state, &live_in))
            .unwrap_or_else(|| panic!("{state:?} is off the interpreter's path"));
        cursor += found;
        if !poll {
            break;
        }
        polls += 1;
        assert!(polls < 10_000);
        vm.jit.trace.forgive(index);
    }
    assert!(polls > 100, "{polls}");
}

#[test]
fn every_state_a_tree_publishes_is_a_state_the_interpreter_passes_through() {
    for arms in [2_usize, 3, 4] {
        let (chunk, _) = chain_program(false, arms, i64::try_from(arms).unwrap() + 2, 6_000);
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
        assert!(stats_of(&vm).side_installs >= 1, "{arms} arms");
    }
}

#[test]
fn the_restore_plan_of_every_exit_of_a_tree_matches_the_generic_restore() {
    let (chunk, header) = chain_program(false, 3, 3, 12_000);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    let index = loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && vm
                .jit
                .trace
                .installed_sides(index)
                .is_some_and(|(_, linked)| linked == 2)
        {
            break index;
        }
        assert!(vm.step(&mut frames).unwrap().is_none());
    };
    let compiled = vm.jit.trace.installed_compiled(index).unwrap();
    // Exits of the root and of both sides.
    let members: std::collections::BTreeSet<usize> =
        compiled.exits().iter().map(|exit| exit.member).collect();
    assert_eq!(members.len(), 3);
    let reference = drive(&chunk, None);
    let visits = header_visits(&reference, header);
    let base = &reference.states[visits[70]];
    let mut seed = 0x1234_5678_9abc_def1_u64;
    for exit in 0..vm.jit.trace.exit_count(index) {
        for round in 0..4 {
            let area: Vec<i64> = (0..64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if round == 0 { 0 } else { seed.cast_signed() }
                })
                .collect();
            let mut planned = frames_for(&vm, &chunk, Some(base));
            let mut generic = frames_for(&vm, &chunk, Some(base));
            vm.jit
                .trace
                .plan_restore(index, exit, &area, &mut planned.at_mut(0), &mut vm.runtime);
            vm.jit
                .trace
                .generic_restore(index, exit, &area, &mut generic.at_mut(0), &mut vm.runtime)
                .expect("the generic restore succeeds");
            assert_eq!(
                format!("{:?}", capture(&planned)),
                format!("{:?}", capture(&generic)),
                "exit {exit} round {round}"
            );
        }
    }
}

#[test]
fn counters_agree_with_each_other_for_a_tree() {
    let (chunk, _) = chain_program(false, 3, 3, 40_000);
    let (vm, _) = run_traced(CompatibilityProfile::Lua51, &chunk);
    let stats = stats_of(&vm);
    // Every entry that ran code left through exactly one exit.
    let exits: u64 = stats.exits.iter().sum();
    assert_eq!(exits, stats.native_entries, "{stats:?}");
    // Exits taken inside sides are a subset of all exits.
    assert!(stats.side_member_exits <= exits, "{stats:?}");
    // Records, installs and rejections account for every side recording.
    assert_eq!(
        stats.side_recordings,
        stats.side_traces_recorded + stats.side_aborts,
        "{stats:?}"
    );
    assert!(
        stats.side_installs <= stats.side_traces_recorded,
        "{stats:?}"
    );
}

#[test]
fn side_installs_counts_only_sides_the_compiled_tree_holds() {
    let (chunk, header) = chain_program(false, 3, 3, 40_000);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    while !frames.is_empty() {
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
    }
    let index = vm.jit.trace.installed_index_at_pc(header);
    let linked = index
        .and_then(|index| vm.jit.trace.installed_sides(index))
        .map_or(0, |(_, linked)| linked);
    assert!(linked > 0);
    assert_eq!(stats_of(&vm).side_installs, linked as u64);
}

#[test]
fn a_loop_that_ends_inside_a_side_exits_through_the_sides_own_guard() {
    // The loop's last iteration runs an arm that is a side. The side's own
    // latch guard fails and exits with its snapshot: the loop's result must
    // be exactly the interpreter's, for every limit around the end.
    for limit in 3_000..3_007 {
        let (chunk, _) = chain_program(false, 2, 3, limit);
        let expected = interpreted(CompatibilityProfile::Lua51, &chunk);
        let (vm, results) = run_traced(CompatibilityProfile::Lua51, &chunk);
        assert_eq!(results, expected, "limit {limit}");
        assert!(stats_of(&vm).side_installs >= 1);
    }
}

#[test]
fn the_cached_seed_plan_follows_every_change_of_the_side_queue() {
    // A side that went back to the queue after a compile apart the tree could
    // not take changes the tree the next side is seeded from. The cached plan
    // must be that of the tree as it is now, not the one planned before.
    let (chunk, header) = chain_program(false, 4, 4, 40_000);
    let chunk = Arc::new(chunk);
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let mut frames = frames_for(&vm, &chunk, None);
    let mut checked = false;
    loop {
        if let Some(index) = vm.jit.trace.installed_index_at_pc(header)
            && !vm.jit.trace.installed_ir(index).unwrap().sides.is_empty()
            && !vm.jit.trace.has_queued_sides(index)
        {
            let installed_sides = vm.jit.trace.installed_ir(index).unwrap().sides.len();
            let before = vm.jit.trace.seed_plan_of(index).expect("the tree plans");
            assert_eq!(before.0.sides.len(), installed_sides);
            assert_eq!(
                *before.1,
                crate::jit_ir::trace::plan_tree(&vm.jit.trace.pending_tree_of(index).unwrap())
                    .unwrap()
            );
            // The plan is cached: the same call returns the same allocation.
            let again = vm.jit.trace.seed_plan_of(index).unwrap();
            assert!(Arc::ptr_eq(&before.1, &again.1));
            vm.jit.trace.requeue_refused_detach(index, 0);
            let pending = vm.jit.trace.pending_tree_of(index).unwrap();
            assert_eq!(pending.sides.len(), installed_sides + 1);
            let fresh = crate::jit_ir::trace::plan_tree(&pending).ok();
            let cached = vm.jit.trace.seed_plan_of(index);
            // Whatever the planner says of the new tree, the cache says too.
            assert_eq!(
                cached
                    .as_ref()
                    .map(|(tree, members)| (tree.sides.len(), (**members).clone())),
                fresh.map(|members| (pending.sides.len(), members)),
            );
            checked = true;
            break;
        }
        if vm.step(&mut frames).unwrap().is_some() {
            break;
        }
    }
    assert!(
        checked,
        "no tree with a side to send through a refused detach"
    );
}

/// The shape of Mineclonia's `locate_biome_in_area`: a scan over a square
/// whose inner loop calls out when a cell matches. The first matches fall on
/// the last cell of a row, so the side of the match guard records a match
/// that ends the row: it runs through the outer loop and back into the
/// inner one, and holds the inner latch as a guard that the row ended. Every
/// later match in the middle of a row fails that guard, which resumes at the
/// root's own header.
const ROW_END_SCAN: &str = "
    local function predicate(biome, arg) return biome == arg end
    local Rng = {}
    Rng.__index = Rng
    function Rng:next_within(n)
        self.s = (self.s * 1103515245 + 12345) % 2147483648
        return self.s % n
    end
    local function locate(grid, w, r, rng)
        local found, at, matches = nil, 0, 0
        for dz = -r, r do
            for dx = -r, r do
                local cell = grid[(dz + r) * w + (dx + r) + 1]
                if predicate(cell, 'plains') then
                    if not found or rng:next_within(matches + 1) == 0 then
                        found, at = cell, (dz + r) * w + (dx + r)
                    end
                    matches = matches + 1
                end
            end
        end
        return at, matches
    end
    local r = 28
    local w = 2 * r + 1
    local edge, spread = {}, {}
    for i = 1, w * w do
        local row, col = math.floor((i - 1) / w), (i - 1) % w
        edge[i] = col == w - 1 and 'plains' or 'ocean'
        spread[i] = (row * 7 + col * 13) % 11 == 0 and 'plains' or 'ocean'
    end
    local rng = setmetatable({ s = 7 }, Rng)
    local sum = 0
    for call = 1, 12 do
        local at, matches = locate(call <= 3 and edge or spread, w, r, rng)
        sum = sum + at + matches
    end
    return sum";

/// What a tracing run of a program that should end gave, next to the
/// interpreter's results.
struct TracedRun {
    results: Result<Vec<crate::Value>, String>,
    expected: Result<Vec<crate::Value>, String>,
    stats: crate::vm::jit::trace::TraceStatistics,
    /// Installed sides that hold no guard and link back to their root: the
    /// empty sides among them, and short ones that record a few instructions
    /// without a branch.
    empty_links: usize,
}

/// Runs `source` under the interpreter and under tracing with inline
/// compiles, each on a thread with a deadline: an empty side that handed the
/// root the state its guard failed on would never end.
fn traced_run(profile: CompatibilityProfile, source: &'static str) -> TracedRun {
    use crate::jit_ir::trace::TraceTerminator;
    let run = move |mode| {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut lua = crate::Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            let results = lua.exec(source).map_err(|error| format!("{error:?}"));
            let trace = &lua.vm().jit.trace;
            let empty_links = trace
                .installed_irs()
                .flat_map(|ir| ir.sides.iter())
                .filter(|side| {
                    side.ir.guards.is_empty()
                        && matches!(side.ir.terminator, TraceTerminator::Link { .. })
                })
                .count();
            let stats = lua.vm().jit_statistics().trace;
            let _ = sender.send((results, stats, empty_links));
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(120))
            .unwrap_or_else(|_| panic!("{profile:?} {mode:?}: the program never ended"))
    };
    let (expected, _, _) = run(ExecutionMode::Interpreter);
    let (results, stats, empty_links) = run(ExecutionMode::Tracing);
    TracedRun {
        results,
        expected,
        stats,
        empty_links,
    }
}

/// Checks that `source` gives the interpreter's results under every profile
/// in `profiles`, and returns the runs.
fn same_as_interpreter(
    profiles: &[CompatibilityProfile],
    source: &'static str,
) -> Vec<(CompatibilityProfile, TracedRun)> {
    profiles
        .iter()
        .map(|&profile| {
            let run = traced_run(profile, source);
            assert!(run.expected.is_ok(), "{profile:?}: {:?}", run.expected);
            assert_eq!(run.results, run.expected, "{profile:?}: {:?}", run.stats);
            (profile, run)
        })
        .collect()
}

const ALL_PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// The guard that resumes at the root's header takes a side that links back
/// at once: it records no guest instruction and holds no guard. A side that
/// recorded an iteration from the header instead copied the root's, and when
/// that copy aborted (a call the root's own iteration never made, as on the
/// Luanti startup one run in eight) every match in the middle of a row left
/// for the interpreter.
#[test]
fn a_guard_that_resumes_at_the_header_links_back_with_an_empty_side() {
    // The integer profiles: under float numbers the side of the match guard
    // does not fit the tree here (a register it cannot type), so no side
    // reaches the latch.
    let profiles = [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ];
    for (profile, run) in same_as_interpreter(&profiles, ROW_END_SCAN) {
        assert!(run.empty_links >= 1, "{profile:?}: {:?}", run.stats);
        assert!(run.stats.empty_sides >= 1, "{profile:?}: {:?}", run.stats);
    }
}

/// The scan of [`ROW_END_SCAN`] with the grids and calls around it; each
/// program below defines `locate(grid, w, r)`, which gives up to three
/// numbers.
macro_rules! scan_with {
    ($locate:literal) => {
        concat!(
            $locate,
            "
            local r = 28
            local w = 2 * r + 1
            local edge, spread = {}, {}
            for i = 1, w * w do
                local row, col = math.floor((i - 1) / w), (i - 1) % w
                edge[i] = col == w - 1 and 'plains' or 'ocean'
                spread[i] = (row * 7 + col * 13) % 11 == 0 and 'plains' or 'ocean'
            end
            local sum = 0
            for call = 1, 12 do
                local a, b, c = locate(call <= 3 and edge or spread, w, r)
                sum = sum + a + (b or 0) + (c or 0)
            end
            return sum"
        )
    };
}

/// The inner loop as a `repeat`. The guard of `until` sits on the comparison
/// in front of the branch back, not at the latch, so its side records an
/// iteration as before; the run must match either way.
#[test]
fn an_empty_side_of_a_repeat_latch_gives_the_interpreters_results() {
    same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local at, matches = 0, 0
                for dz = -r, r do
                    local dx = -r
                    repeat
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        if cell == 'plains' then
                            at = at + (dz + r) * w + (dx + r)
                            matches = matches + 1
                        end
                        dx = dx + 1
                    until dx > r
                end
                return at, matches
            end"
        ),
    );
}

/// The inner loop counts in floats with a step of one half: the latch's
/// guard compares floats, and the control register the empty side hands
/// back is a float.
#[test]
fn an_empty_side_of_a_float_step_latch_gives_the_interpreters_results() {
    let runs = same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local at, matches = 0, 0
                for dz = -r, r do
                    for dx = -r, r, 0.5 do
                        local cell = grid[(dz + r) * w + math.floor(dx + r) + 1]
                        if cell == 'plains' then
                            at = at + dx * 2
                            matches = matches + 1
                        end
                    end
                end
                return at, matches
            end"
        ),
    );
    for (profile, run) in runs {
        assert!(run.stats.empty_sides >= 1, "{profile:?}: {:?}", run.stats);
    }
}

/// The inner loop is a `while` whose condition calls a host function, and
/// in the second program reads an upvalue through a closure. Its latch is an
/// unconditional jump and the guard that ends the row sits at the header, so
/// no side links back at once; the run must match all the same.
#[test]
fn a_while_latch_with_a_called_condition_gives_the_interpreters_results() {
    same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local at, matches = 0, 0
                for dz = -r, r do
                    local dx = -r
                    while dx <= math.min(r, w) do
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        if cell == 'plains' then
                            at = at + (dz + r) * w + (dx + r)
                            matches = matches + 1
                        end
                        dx = dx + 1
                    end
                end
                return at, matches
            end"
        ),
    );
    same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local at, matches, limit = 0, 0, r
                local function within(dx) return dx <= limit end
                for dz = -r, r do
                    local dx = -r
                    while within(dx) do
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        if cell == 'plains' then
                            at = at + (dz + r) * w + (dx + r)
                            matches = matches + 1
                        end
                        dx = dx + 1
                    end
                    limit = r - (dz + r) % 2
                end
                return at, matches
            end"
        ),
    );
}

/// The root carries accumulators across the inner loop that the empty side
/// does not write: the link hands them back as the root's loop holds them,
/// and every trip of the root adds to them again.
#[test]
fn an_empty_side_keeps_the_accumulators_it_does_not_write() {
    let runs = same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local acc, at, matches = 0, 0, 0
                for dz = -r, r do
                    for dx = -r, r do
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        acc = acc + dx * 3 + dz
                        if cell == 'plains' then
                            at = at + dx * dz
                            matches = matches + 1
                        end
                    end
                end
                return acc, at, matches
            end"
        ),
    );
    for (profile, run) in runs {
        assert!(run.stats.empty_sides >= 1, "{profile:?}: {:?}", run.stats);
    }
}

/// A match bumps a local that a closure captures, through the closure in
/// the first program and in the loop itself in the second, and the root reads
/// it back after the row. The call after the row retires these trees before
/// the latch's guard fails often enough for a side, so this checks results
/// only.
#[test]
fn an_empty_side_keeps_a_captured_local() {
    same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local count, rows = 0, 0
                local function bump(v) count = count + v end
                for dz = -r, r do
                    for dx = -r, r do
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        if cell == 'plains' then bump(dx + r + 1) end
                    end
                    rows = rows + count % 7
                end
                return count, rows
            end"
        ),
    );
    same_as_interpreter(
        &ALL_PROFILES,
        scan_with!(
            "local function locate(grid, w, r)
                local count, rows = 0, 0
                local function seen() return count end
                for dz = -r, r do
                    for dx = -r, r do
                        local cell = grid[(dz + r) * w + (dx + r) + 1]
                        if cell == 'plains' then count = count + dx + r + 1 end
                    end
                    rows = rows + seen() % 7
                end
                return count, rows
            end"
        ),
    );
}

/// Mineclonia's `get_groups` (`mcl_stairs/api.lua:483`): a `pairs` loop whose
/// step guard fails at the top of an iteration. Its exit resumes at the
/// header with the state after the latch, as the guard of a latch does, but
/// the interpreter has to run the step there: a side that linked back at
/// once would hand the root the same state and the same guard forever.
#[test]
fn a_step_guard_at_the_header_does_not_link_back_at_once() {
    const GROUPS: &str = "
        local copied = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k' }
        local function get_groups(base)
            local groups = {}
            for _, group in pairs(copied) do
                groups[group] = base[group]
            end
            return groups
        end
        local n = 0
        for i = 1, 20000 do
            local g = get_groups({ a = 1, c = i, k = 3 })
            if g.c then n = n + g.k end
        end
        return n";
    let run = |profile, mode| {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut lua = crate::Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            let results = lua.exec(GROUPS).map_err(|error| format!("{error:?}"));
            let _ = sender.send(results);
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(120))
            .unwrap_or_else(|_| panic!("{profile:?} {mode:?}: the loop never ended"))
    };
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let expected = run(profile, ExecutionMode::Interpreter);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            run(profile, ExecutionMode::Tracing),
            expected,
            "{profile:?}"
        );
    }
}
