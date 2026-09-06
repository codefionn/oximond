//! Trees that link through the return of their function into the loop that
//! calls it (see "Linking through a return" in `vm::jit::trace`), when the
//! tree is taken away while the calls run.
//!
//! The programs run from Lua source and are stepped by hand, so that a flush
//! or a blacklist lands between two steps, with the link built and its
//! caller frame on the stack. The answer is the interpreter's in every
//! profile, whatever the moment.

use super::trace_native::stats_of;
use super::*;
use crate::jit_ir::trace::TraceType;
use crate::vm::jit::trace::deopt::CellTarget;
use crate::vm::stack::CallStack;

/// A float profile, an integer profile and the one with both kinds of
/// numbers; the other profiles are covered by the integration tests.
const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

const CALLS: u64 = 12_000;

/// A function with a nest, and a loop that calls it.
fn program(driver: &str) -> String {
    format!(
        "local function sweep(n, m)
             local acc = 0
             for i = 1, n do
                 for j = i, m do
                     acc = acc + (i * j) % 7
                 end
             end
             return acc
         end
         {driver}"
    )
}

/// What a stepped run did.
struct Run {
    result: Result<Vec<Value>, String>,
    steps: usize,
    stats: TraceStatistics,
    installed: usize,
}

/// Runs `source` one step at a time, handing the VM to `act` before each step.
fn stepped(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
    mut act: impl FnMut(usize, &mut Vm),
) -> Run {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    let chunk = lua.compile("=(link)", source).unwrap();
    let vm = lua.vm_mut();
    vm.use_synchronous_jit_compilation();
    let mut frames = vm.stack_for(&chunk).unwrap();
    let mut steps = 0;
    let result = loop {
        steps += 1;
        act(steps, vm);
        match vm.step(&mut frames) {
            Ok(Some(results)) => break Ok(results),
            Ok(None) => {}
            Err(error) => break Err(format!("{error:?}")),
        }
    };
    Run {
        result,
        steps,
        stats: stats_of(vm),
        installed: vm.jit.trace.installed_count(),
    }
}

fn interpreted(profile: CompatibilityProfile, source: &str) -> Run {
    stepped(profile, ExecutionMode::Interpreter, source, |_, _| {})
}

/// Takes the tree away at several moments of the run with `take`, and checks
/// that the program is the interpreter's each time. The run without the
/// action must have linked: native code was entered for fewer than one call
/// in four.
fn assert_survives(source: &str, take: impl Fn(&mut Vm) + Copy) {
    for profile in PROFILES {
        let expected = interpreted(profile, source);
        let linked = stepped(profile, ExecutionMode::Tracing, source, |_, _| {});
        assert_eq!(linked.result, expected.result, "{profile:?}");
        assert!(
            linked.stats.native_entries * 4 < CALLS,
            "{profile:?}: {} entries for {CALLS} calls: {:?}",
            linked.stats.native_entries,
            linked.stats
        );
        let mut taken_with_tree = 0;
        for at in [5, 8].map(|tenth| linked.steps * tenth / 10) {
            let mut taken = false;
            let run = stepped(profile, ExecutionMode::Tracing, source, |step, vm| {
                if step == at {
                    taken = vm.jit.trace.installed_count() > 0;
                    take(vm);
                }
            });
            assert_eq!(run.result, expected.result, "{profile:?} taken at {at}");
            taken_with_tree += usize::from(taken);
        }
        assert!(
            taken_with_tree > 0,
            "{profile:?}: no tree was installed when it was taken away"
        );
    }
}

/// The caller of the nest does more than call it: it stores, takes a global
/// and makes a table.
const BUSY_DRIVER: &str = "
    local total, out, keep = 0, {}, {}
    for k = 1, 12000 do
        local row = {k, k + 1}
        keep[k % 50 + 1] = row
        out[k % 10 + 1] = sweep(8 + k % 3, 40)
        total = (total + out[k % 10 + 1] + math.abs(-k)) % 1000003
    end
    return total, #out, #keep";

#[test]
fn a_flush_while_the_calls_run_leaves_the_interpreters_answer() {
    assert_survives(
        &program(
            "local total = 0
             for k = 1, 12000 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total",
        ),
        |vm| vm.runtime_mut().request_jit_flush(),
    );
}

#[test]
fn a_blacklisted_header_while_the_calls_run_leaves_the_interpreters_answer() {
    let blacklist = |vm: &mut Vm| {
        let mut stats = TraceStatistics::default();
        let gone = vm.jit.trace.blacklist_installed(&mut stats);
        assert!(gone > 0 || vm.jit.trace.installed_count() == 0);
        assert_eq!(vm.jit.trace.installed_count(), 0);
    };
    assert_survives(
        &program(
            "local total = 0
             for k = 1, 12000 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total",
        ),
        blacklist,
    );
}

#[test]
fn a_caller_that_stores_and_makes_tables_loses_its_tree_mid_run() {
    assert_survives(&program(BUSY_DRIVER), |vm| {
        vm.runtime_mut().request_jit_flush();
    });
    assert_survives(&program(BUSY_DRIVER), |vm| {
        let mut stats = TraceStatistics::default();
        vm.jit.trace.blacklist_installed(&mut stats);
    });
}

#[test]
fn a_blacklisted_tree_is_not_entered_again() {
    let source = program(
        "local total = 0
         for k = 1, 12000 do
             total = (total + sweep(8 + k % 3, 40)) % 1000003
         end
         return total",
    );
    let profile = CompatibilityProfile::Lua51;
    let whole = stepped(profile, ExecutionMode::Tracing, &source, |_, _| {});
    let at = whole.steps * 6 / 10;
    let mut before = 0;
    let run = stepped(profile, ExecutionMode::Tracing, &source, |step, vm| {
        if step == at {
            before = stats_of(vm).native_entries;
            let mut stats = TraceStatistics::default();
            vm.jit.trace.blacklist_installed(&mut stats);
        }
    });
    assert!(before > 0, "the tree ran before it was taken away");
    assert_eq!(
        run.stats.native_entries, before,
        "native code was entered after the blacklist: {:?}",
        run.stats
    );
    assert_eq!(run.installed, 0);
    assert_eq!(run.result, whole.result);
}

// ------------------------------------------------------------------ exits

/// A VM stepped until the tree that returns into its caller is installed and
/// the root frame is running above the frame that called it (or for `steps`
/// steps). Runs are deterministic, so two worlds stepped alike are the same.
fn world_at_link(
    profile: CompatibilityProfile,
    source: &str,
    steps: Option<usize>,
) -> (crate::Lua, CallStack, u32, usize) {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Tracing)
        .build()
        .unwrap();
    let chunk = lua.compile("=(link)", source).unwrap();
    let vm = lua.vm_mut();
    vm.use_synchronous_jit_compilation();
    let mut frames = vm.stack_for(&chunk).unwrap();
    for step in 1..5_000_000 {
        if let Some(index) = vm.jit.trace.linked_trace_index()
            && frames.len() == 2
            && steps.is_none_or(|steps| steps == step)
        {
            return (lua, frames, index, step);
        }
        assert!(
            vm.step(&mut frames).unwrap().is_none(),
            "{profile:?}: the program ended before its tree linked: {:?}",
            stats_of(vm)
        );
    }
    panic!("{profile:?}: the tree never linked");
}

/// Every frame of the stack, with every register's `Debug`, which keeps
/// the bits of a NaN.
fn shot(frames: &CallStack) -> Vec<String> {
    (0..frames.len())
        .map(|index| {
            let frame = frames.at(index);
            format!(
                "pc {} last {:?} {:?}",
                frame.pc,
                frame.last_interpreted_pc.get(),
                frame.registers
            )
        })
        .collect()
}

#[test]
fn the_restore_plan_of_every_exit_that_pops_the_root_frame_is_the_generic_restore() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let source = program(BUSY_DRIVER);
        let (survey, _frames, index, steps) = world_at_link(profile, &source, None);
        let exits = survey.vm().jit.trace.caller_exits(index);
        assert!(
            !exits.is_empty(),
            "{profile:?}: no exit pops the root frame"
        );
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        // The worlds stay alive to the end of the test: some caches of the
        // runtime are keyed by address, which a freed world's memory may
        // give to the next one.
        let mut keep = Vec::new();
        for &exit in &exits {
            let descriptor = survey
                .vm()
                .jit
                .trace
                .exit_descriptor(index, exit)
                .unwrap()
                .clone();
            // A borrowed string is the address of a string in a table, which
            // arbitrary bits are not.
            if descriptor
                .slots
                .iter()
                .any(|slot| slot.ty == TraceType::StringRef)
            {
                continue;
            }
            for round in 0..3 {
                let mut area: Vec<i64> = (0..96)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        if round == 0 { 0 } else { seed.cast_signed() }
                    })
                    .collect();
                let (mut planned_lua, mut planned, planned_index, _) =
                    world_at_link(profile, &source, Some(steps));
                let (mut generic_lua, mut generic, generic_index, _) =
                    world_at_link(profile, &source, Some(steps));
                assert_eq!((planned_index, generic_index), (index, index));
                let before = shot(&planned);
                assert_eq!(before, shot(&generic));
                // The bank holds the pinned closures, then a table and a
                // string for the lanes of the exit's other slots.
                let pinned = planned_lua.vm().jit.trace.pinned_values(index);
                let (table, string) = (pinned.len(), pinned.len() + 1);
                for slot in &descriptor.slots {
                    area[slot.slot as usize] = match slot.ty {
                        TraceType::Closure => 0,
                        TraceType::Table => table as i64,
                        TraceType::String => string as i64,
                        _ => area[slot.slot as usize],
                    };
                }
                let bank_of = |vm: &mut Vm| {
                    let mut bank = pinned.clone();
                    bank.push(vm.runtime_mut().create_table_value().unwrap());
                    bank.push(Value::from("lane"));
                    bank
                };
                let planned_vm = planned_lua.vm_mut();
                *planned_vm.jit.trace.bank_mut() = bank_of(planned_vm);
                planned_vm.jit.trace.plan_restore_stack(
                    index,
                    exit,
                    &area,
                    &mut planned,
                    &mut planned_vm.runtime,
                );
                let plan_writes = planned_vm.jit.trace.plan_upvalues(index, exit, &area);
                let generic_vm = generic_lua.vm_mut();
                *generic_vm.jit.trace.bank_mut() = bank_of(generic_vm);
                let generic_writes = generic_vm
                    .jit
                    .trace
                    .generic_restore_stack(
                        index,
                        exit,
                        &area,
                        &mut generic,
                        &mut generic_vm.runtime,
                    )
                    .expect("the generic restore succeeds");
                // The plan writes the caller's registers apart from the cells
                // (the stack shows them below).
                let generic_writes: Vec<_> = generic_writes
                    .into_iter()
                    .filter(|(target, _)| !matches!(target, CellTarget::Caller(_)))
                    .collect();
                assert_eq!(
                    format!("{plan_writes:?}"),
                    format!("{generic_writes:?}"),
                    "{profile:?} exit {exit} round {round}: cell writes"
                );
                let after = shot(&planned);
                assert_eq!(
                    after,
                    shot(&generic),
                    "{profile:?} exit {exit} round {round}"
                );
                assert_eq!(
                    planned.len(),
                    1,
                    "{profile:?} exit {exit}: the root frame went"
                );
                assert_ne!(
                    after, before,
                    "{profile:?} exit {exit}: the exit changed nothing"
                );
                planned_vm.jit.trace.bank_mut().clear();
                generic_vm.jit.trace.bank_mut().clear();
                keep.push((planned_lua, planned, generic_lua, generic));
            }
        }
    }
}
