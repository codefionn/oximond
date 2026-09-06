//! Integer traces for Lua 5.3 to 5.5 (and the float-only profiles that share
//! the machinery).
//!
//! The programs are hand-assembled chunks with integer constants. Every case
//! compares native execution with an isolated interpreter of the same
//! profile, as `trace_native` does for floats: a restored state must be a
//! state that interpreter passes through, and running on must finish as it
//! does. Corner values (the integer extremes, zero and minus one divisors,
//! numbers around 2^53 and 2^63) come from the entry-state variants.

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use super::trace_native::*;
use super::trace_record::*;
use super::*;
use crate::jit_ir::trace::reference::{Outcome, evaluate_forced};
use crate::jit_ir::trace::{TraceScalar, TraceType};
use crate::vm::jit::trace::{TraceExitKind, TraceRun};

const INTEGER_PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

fn int(value: i64) -> Constant {
    Constant::Integer(value)
}

/// Assembles `setup`, an integer numeric for loop (control r0, limit r1,
/// step r2, visible index r3) around `body`, and a return of r4. The
/// constants at `start`, `limit` and `step` fill r0 to r2. Returns the chunk
/// and its loop header.
fn for_loop_chunk(
    registers: u8,
    constants: Vec<Constant>,
    [start, limit, step]: [u16; 3],
    setup: Vec<Instruction>,
    body: Vec<Instruction>,
) -> (Chunk, usize) {
    let mut code = vec![load(0, start), load(1, limit), load(2, step)];
    code.extend(setup);
    let prep = code.len();
    let header = prep + 1;
    let latch = header + body.len();
    code.push(for_prep(0, prep, latch));
    code.extend(body);
    code.push(for_loop(0, latch, header));
    code.push(Instruction::return_(4));
    (Chunk::new(registers, constants, code), header)
}

/// Appends `acc = acc + step` guarded by the boolean in `condition`.
fn guarded_add(body: &mut Vec<Instruction>, header: usize, condition: u8) {
    let pc = header + body.len();
    body.push(jump_if_false(condition, pc, pc + 2));
    body.push(binary(OpCode::Add, 4, 4, 2));
}

/// Wrapping arithmetic, floor division and modulo with a negative divisor,
/// bitwise operations, shifts and negation.
fn arithmetic_program(limit: i64) -> (Chunk, usize) {
    for_loop_chunk(
        16,
        vec![int(1), int(limit), int(0), int(7), int(-3), int(3), int(13)],
        [0, 1, 0],
        vec![load(4, 2), load(5, 6), load(6, 3), load(7, 4), load(8, 5)],
        vec![
            binary(OpCode::Multiply, 5, 5, 8),
            binary(OpCode::Add, 5, 5, 3),
            binary(OpCode::FloorDivide, 9, 5, 6),
            binary(OpCode::Modulo, 10, 5, 7),
            binary(OpCode::Add, 4, 4, 9),
            binary(OpCode::BitXor, 4, 4, 10),
            binary(OpCode::ShiftLeft, 11, 3, 8),
            binary(OpCode::ShiftRight, 12, 5, 8),
            binary(OpCode::BitOr, 4, 4, 11),
            binary(OpCode::BitAnd, 13, 12, 5),
            binary(OpCode::Add, 4, 4, 13),
            Instruction::unary(OpCode::BitNot, 14, 4),
            Instruction::unary(OpCode::Negate, 15, 14),
            binary(OpCode::Subtract, 4, 4, 15),
        ],
    )
}

/// Floor division by a divisor `d` that counts down from `divisor`, and
/// modulo by `d + 1`: `1000 // d` and `1000 % (d + 1)`. The first divisor is
/// zero when `d` is, the second when `d` is -1. Body offsets: the division
/// is at the header, the modulo at header + 3.
fn division_program(limit: i64, divisor: i64) -> (Chunk, usize) {
    for_loop_chunk(
        10,
        vec![int(1), int(limit), int(0), int(divisor), int(1000)],
        [0, 1, 0],
        vec![load(4, 2), load(5, 3), load(7, 4)],
        vec![
            binary(OpCode::FloorDivide, 6, 7, 5),
            binary(OpCode::Add, 4, 4, 6),
            binary(OpCode::Add, 9, 5, 2),
            binary(OpCode::Modulo, 8, 7, 9),
            binary(OpCode::Add, 4, 4, 8),
            binary(OpCode::Subtract, 5, 5, 2),
        ],
    )
}

/// Shifts a pattern by a distance that sweeps from -150 up through 64 and
/// beyond.
fn shift_program(limit: i64) -> (Chunk, usize) {
    for_loop_chunk(
        10,
        vec![
            int(1),
            int(limit),
            int(0),
            int(-150),
            int(0x1234_5678_9abc_def1),
        ],
        [0, 1, 0],
        vec![load(4, 2), load(5, 4), load(7, 3)],
        vec![
            binary(OpCode::Add, 6, 3, 7),
            binary(OpCode::ShiftLeft, 8, 5, 6),
            binary(OpCode::ShiftRight, 9, 5, 6),
            binary(OpCode::BitXor, 4, 4, 8),
            binary(OpCode::BitXor, 4, 4, 9),
        ],
    )
}

/// Counts iterations of `for i = start, limit, step`, where the range ends
/// at the edge of the integer domain.
fn edge_program(start: i64, limit: i64, step: i64) -> (Chunk, usize) {
    for_loop_chunk(
        5,
        vec![int(start), int(limit), int(step), int(0), int(1)],
        [0, 1, 2],
        vec![load(4, 3)],
        // The accumulator sums the step, which counts iterations.
        vec![binary(OpCode::Add, 4, 4, 2)],
    )
}

/// Compares floats with integers around 2^53 and 2^63.
fn mixed_compare_program(limit: i64) -> (Chunk, usize) {
    let two_53 = 9_007_199_254_740_992_i64;
    let header = 3 + 5 + 1;
    let mut body = vec![binary(OpCode::Add, 9, 6, 3)];
    let compare = |body: &mut Vec<Instruction>, opcode, register, left, right| {
        body.push(binary(opcode, register, left, right));
        guarded_add(body, header, register);
    };
    // r5 = 2^53 as a float, r6 = 2^53, r7 = 2^63 as a float, r8 = i64::MAX.
    compare(&mut body, OpCode::LessThan, 10, 5, 9);
    compare(&mut body, OpCode::LessEqual, 11, 9, 5);
    compare(&mut body, OpCode::Equal, 12, 5, 9);
    compare(&mut body, OpCode::LessThan, 13, 8, 7);
    compare(&mut body, OpCode::Equal, 14, 8, 7);
    compare(&mut body, OpCode::LessEqual, 15, 7, 8);
    for_loop_chunk(
        16,
        vec![
            int(1),
            int(limit),
            int(0),
            Constant::Number(two_53 as f64),
            int(two_53),
            Constant::Number(9_223_372_036_854_775_808.0),
            int(i64::MAX),
        ],
        [0, 1, 0],
        vec![load(4, 2), load(5, 3), load(6, 4), load(7, 5), load(8, 6)],
        body,
    )
}

/// Integers mixed with floats in arithmetic, true division and power of
/// integers, float floor division and a float modulo.
fn mixed_arithmetic_program(limit: i64) -> (Chunk, usize) {
    for_loop_chunk(
        14,
        vec![
            int(1),
            int(limit),
            Constant::Number(0.0),
            Constant::Number(2.5),
            int(-7),
            int(5),
        ],
        [0, 1, 0],
        vec![load(4, 2), load(5, 3), load(6, 4), load(7, 5)],
        vec![
            binary(OpCode::Add, 8, 3, 5),
            binary(OpCode::Divide, 9, 3, 6),
            binary(OpCode::Power, 10, 3, 7),
            binary(OpCode::FloorDivide, 11, 8, 5),
            binary(OpCode::Modulo, 12, 3, 5),
            binary(OpCode::Add, 4, 4, 8),
            binary(OpCode::Add, 4, 4, 9),
            binary(OpCode::Subtract, 4, 4, 11),
            binary(OpCode::Add, 4, 4, 12),
            binary(OpCode::Subtract, 4, 4, 10),
        ],
    )
}

fn integer_programs() -> Vec<(&'static str, Chunk, usize)> {
    let mut programs = Vec::new();
    for (name, (chunk, header)) in [
        ("arithmetic", arithmetic_program(300)),
        ("division", division_program(300, 400)),
        ("shifts", shift_program(300)),
        ("edge up", edge_program(i64::MAX - 300, i64::MAX, 1)),
        (
            "edge up by seven",
            edge_program(i64::MAX - 704, i64::MAX, 7),
        ),
        ("edge down", edge_program(i64::MIN + 300, i64::MIN, -1)),
        (
            "wide steps",
            edge_program(i64::MIN, i64::MAX, i64::MAX / 150),
        ),
        ("mixed compare", mixed_compare_program(300)),
        ("mixed arithmetic", mixed_arithmetic_program(300)),
    ] {
        programs.push((name, chunk, header));
    }
    programs
}

#[test]
fn integer_loops_record_compile_and_run_natively_in_every_integer_profile() {
    for profile in INTEGER_PROFILES {
        for (name, chunk, header) in integer_programs() {
            let expected = interpreted(profile, &chunk);
            let installed = Installed::with_profile(profile, chunk.clone(), header);
            let ir = installed.ir();
            let has_integers = ir
                .entry
                .inputs
                .iter()
                .any(|input| input.ty == TraceType::Integer);
            assert!(has_integers, "{name} {profile:?}: integer entry guards");
            let mut vm = native_vm(profile);
            assert_eq!(vm.execute(&chunk).unwrap(), expected, "{name} {profile:?}");
            let stats = stats_of(&vm);
            assert!(
                stats.iterations > 20,
                "{name} {profile:?} ran natively: {stats:?}"
            );
        }
    }
}

/// How many loop-header states a differential test enters from. The three
/// profiles run the same integer code, so Lua 5.4 gets the full sweep and the
/// others a smaller one that still reaches every guard.
const fn header_visits_for(profile: CompatibilityProfile) -> usize {
    match profile {
        CompatibilityProfile::Lua54 => 14,
        _ => 5,
    }
}

/// Programs whose guards cannot each fail first from any entry state: the
/// comparisons depend on one another, so the first one to flip hides the rest.
/// The forced-failure test covers their guards one by one instead.
const GUARDS_SHADOW_EACH_OTHER: &[&str] = &["mixed compare"];

#[test]
fn integer_guards_fail_into_states_the_isolated_interpreter_passes_through() {
    for profile in INTEGER_PROFILES {
        for (name, chunk, header) in integer_programs() {
            let mut installed = Installed::with_profile_at(
                profile,
                crate::jit::OptimizationTier::Baseline,
                chunk,
                header,
            );
            let ir = installed.ir();
            let run = drive_with(profile, &installed.chunk, None);
            let visits = header_visits(&run, header);
            assert!(
                visits.len() > 80,
                "{name} {profile:?}: loop runs long enough"
            );
            let mut failed_guards = BTreeSet::new();
            let mut mismatches = 0;
            for &visit in visits
                .iter()
                .skip(66)
                .step_by(9)
                .take(header_visits_for(profile))
            {
                for state in variants(&run.states[visit]) {
                    let (ran, restored) = installed.enter(&state, 600);
                    match evaluate(&ir, &state, 600) {
                        Evaluation::EntryMismatch => {
                            mismatches += 1;
                            assert_eq!(ran, TraceRun::NotEntered, "{name}");
                            assert_eq!(restored, state, "{name}: a mismatch changes nothing");
                        }
                        Evaluation::Guard {
                            guard,
                            iteration,
                            restored: expected,
                        } => {
                            failed_guards.insert(guard);
                            let TraceRun::Ran { poll, .. } = ran else {
                                panic!("{name} {profile:?}: guard {guard} did not run natively");
                            };
                            assert!(!poll, "{name}");
                            assert_eq!(
                                restored, expected,
                                "{name} {profile:?}: guard {guard} in iteration {iteration} \
                                 from {state:?}"
                            );
                            assert!(
                                reaches_with(profile, &installed.chunk, &state, &restored, 20_000),
                                "{name} {profile:?}: guard {guard} restored {restored:?}, which \
                                 the interpreter never reaches from {state:?}"
                            );
                        }
                        Evaluation::Iterations { registers } => {
                            let TraceRun::Ran { poll, .. } = ran else {
                                panic!("{name}: a non-ending loop ran natively");
                            };
                            assert!(poll, "{name}");
                            for (register, scalar) in registers {
                                assert_eq!(
                                    restored.registers[register], scalar,
                                    "{name} {profile:?} r{register}"
                                );
                            }
                        }
                    }
                }
            }
            let guards: BTreeSet<usize> = (0..ir.guards.len()).collect();
            if GUARDS_SHADOW_EACH_OTHER.contains(&name) {
                assert!(!failed_guards.is_empty(), "{name} {profile:?}");
            } else {
                assert_eq!(
                    failed_guards, guards,
                    "{name} {profile:?}: every guard must fail at least once"
                );
            }
            assert!(mismatches > 0, "{name} {profile:?}: entry mismatches");
            let stats = stats_of(&installed.vm);
            assert!(stats.exits_of(TraceExitKind::EntryMismatch) > 0, "{name}");
        }
    }
}

#[test]
fn promoted_integer_traces_exit_like_the_reference_on_the_lowered_ir() {
    let mut promoted_programs = 0;
    for profile in INTEGER_PROFILES {
        for (name, chunk, header) in integer_programs() {
            let mut installed = Installed::with_profile(profile, chunk, header);
            let run = drive_with(profile, &installed.chunk, None);
            let visits = header_visits(&run, header);
            let promoted = visits
                .iter()
                .skip(66)
                .take(12)
                .any(|visit| installed.try_promote(&run.states[*visit], 40));
            if !promoted {
                continue;
            }
            promoted_programs += 1;
            let lowered = installed
                .vm
                .jit
                .trace
                .installed_lowered_ir(installed.index)
                .unwrap()
                .clone();
            lowered.verify().unwrap();
            let mut failed_guards = BTreeSet::new();
            for &visit in visits
                .iter()
                .skip(66)
                .step_by(9)
                .take(header_visits_for(profile))
            {
                for state in variants(&run.states[visit]) {
                    let (ran, restored) = installed.enter(&state, 600);
                    match ran {
                        TraceRun::NotEntered => assert_eq!(restored, state, "{name}"),
                        TraceRun::Failed | TraceRun::Returned { .. } => {
                            unreachable!("these loop traces build no virtual table")
                        }
                        TraceRun::Ran { .. } => {
                            assert!(
                                reaches_with(profile, &installed.chunk, &state, &restored, 20_000),
                                "{name} {profile:?}: {restored:?} is not an interpreter state \
                                 from {state:?}"
                            );
                            if let Some((guard, expected)) = reference_exit(&lowered, &state) {
                                failed_guards.insert(guard);
                                assert_eq!(
                                    restored, expected,
                                    "{name} {profile:?}: lowered guard {guard} from {state:?}"
                                );
                            }
                        }
                    }
                }
            }
            let guards: BTreeSet<usize> = (0..lowered.guards.len()).collect();
            if GUARDS_SHADOW_EACH_OTHER.contains(&name) {
                assert!(!failed_guards.is_empty(), "{name} {profile:?}");
            } else {
                assert_eq!(
                    failed_guards, guards,
                    "{name} {profile:?}: every lowered guard must fail at least once"
                );
            }
            let (ran, restored) = installed.enter(&run.states[visits[70]], 25);
            assert!(matches!(ran, TraceRun::Ran { .. }), "{name}");
            assert_eq!(
                drive_with(profile, &installed.chunk, Some(&restored)).results,
                run.results,
                "{name} {profile:?}"
            );
        }
    }
    assert!(promoted_programs >= 12, "{promoted_programs} promoted");
}

#[test]
fn a_zero_divisor_leaves_before_the_operation_so_the_interpreter_raises_the_error() {
    for profile in INTEGER_PROFILES {
        let (chunk, header) = division_program(300, 400);
        let mut installed = Installed::with_profile(profile, chunk, header);
        let run = drive_with(profile, &installed.chunk, None);
        let visits = header_visits(&run, header);
        let at = &run.states[visits[70]];
        // (the divisor register r5, the register that must hit zero, the
        // pc of the failing operation)
        for (start, zero_register, operation_pc) in [(3, 5, header), (-1, 9, header + 3)] {
            let mut state = at.clone();
            state.registers[5] = TraceScalar::Integer(start);
            let (ran, restored) = installed.enter(&state, 600);
            assert!(matches!(ran, TraceRun::Ran { .. }), "{profile:?}");
            // The restored state is the one before the operation, with its
            // divisor at zero, and the interpreter fails from it.
            assert_eq!(
                restored.registers[zero_register],
                TraceScalar::Integer(0),
                "{profile:?}"
            );
            assert_eq!(
                restored.pc, operation_pc,
                "{profile:?}: before the operation"
            );
            assert!(
                reaches_with(profile, &installed.chunk, &state, &restored, 20_000),
                "{profile:?}: {restored:?}"
            );
            let mut vm = interpreter_vm_with(profile);
            let mut frames = frames_for(&vm, &installed.chunk, Some(&restored));
            let error = loop {
                match vm.step(&mut frames) {
                    Err(error) => break error,
                    Ok(None) => {}
                    Ok(Some(_)) => panic!("{profile:?}: the program finished"),
                }
            };
            assert!(
                error.to_string().contains("divide by zero") || error.to_string().contains("'n%0'"),
                "{profile:?}: {error}"
            );
        }
        assert!(stats_of(&installed.vm).exits_of(TraceExitKind::GuardFail) > 0);
    }
}

#[test]
fn the_most_negative_integer_divided_by_minus_one_wraps_natively() {
    for profile in INTEGER_PROFILES {
        let (chunk, header) = division_program(300, 400);
        let mut installed = Installed::with_profile(profile, chunk, header);
        let run = drive_with(profile, &installed.chunk, None);
        let visits = header_visits(&run, header);
        let mut state = run.states[visits[70]].clone();
        // r7 is the dividend, r5 the divisor.
        state.registers[7] = TraceScalar::Integer(i64::MIN);
        state.registers[5] = TraceScalar::Integer(-1);
        let (ran, restored) = installed.enter(&state, 1);
        assert!(matches!(ran, TraceRun::Ran { .. }), "{profile:?}");
        // One iteration: MIN // -1 is MIN and MIN % -1 is 0.
        let mut interpreter = interpreter_vm_with(profile);
        let mut frames = frames_for(&interpreter, &installed.chunk, Some(&state));
        let mut states = vec![capture(&frames)];
        while !states.last().unwrap().same_as(&restored, &installed.chunk) {
            assert!(interpreter.step(&mut frames).unwrap().is_none());
            states.push(capture(&frames));
            assert!(states.len() < 20, "{profile:?}: {restored:?}");
        }
        assert_eq!(
            restored.registers[4],
            states.last().unwrap().registers[4],
            "{profile:?}"
        );
    }
}

#[test]
fn integer_and_float_entry_guards_are_distinct() {
    for profile in INTEGER_PROFILES {
        let (chunk, header) = arithmetic_program(300);
        let mut installed = Installed::with_profile(profile, chunk, header);
        let run = drive_with(profile, &installed.chunk, None);
        let visits = header_visits(&run, header);
        let at = &run.states[visits[70]];
        let (ran, _) = installed.enter(at, 5);
        assert!(matches!(ran, TraceRun::Ran { .. }), "{profile:?}");
        for (register, scalar) in at.registers.iter().enumerate() {
            let TraceScalar::Integer(value) = scalar else {
                continue;
            };
            let mut state = at.clone();
            state.registers[register] = TraceScalar::Float(*value as f64);
            let (ran, restored) = installed.enter(&state, 5);
            // An integer register the trace reads rejects the float. A
            // register it never reads cannot tell.
            if installed
                .ir()
                .entry
                .inputs
                .iter()
                .any(|input| usize::from(input.register) == register)
            {
                assert_eq!(ran, TraceRun::NotEntered, "{profile:?} r{register}");
                assert_eq!(restored, state);
            }
        }
    }
}

#[test]
fn restored_integers_stay_integers_and_floats_stay_floats() {
    for profile in INTEGER_PROFILES {
        let (chunk, header) = mixed_arithmetic_program(300);
        let mut installed = Installed::with_profile(profile, chunk, header);
        let run = drive_with(profile, &installed.chunk, None);
        let visits = header_visits(&run, header);
        let at = &run.states[visits[70]];
        let (ran, restored) = installed.enter(at, 5);
        assert!(matches!(ran, TraceRun::Ran { .. }), "{profile:?}");
        // The loop control and the visible index are integers, the sums
        // floats, exactly as the interpreter holds them.
        assert!(matches!(restored.registers[0], TraceScalar::Integer(_)));
        assert!(matches!(restored.registers[3], TraceScalar::Integer(_)));
        assert!(matches!(restored.registers[4], TraceScalar::Float(_)));
        // r8 is a float temporary. Where it is dead the exit leaves a nil
        // placeholder, which the interpreter overwrites before reading.
        let live = crate::jit_ir::observable_registers(&installed.chunk);
        if crate::jit_ir::register_set_contains(&live[restored.pc], 8) {
            assert!(matches!(restored.registers[8], TraceScalar::Float(_)));
        } else {
            assert!(matches!(
                restored.registers[8],
                TraceScalar::Float(_) | TraceScalar::Nil
            ));
        }
    }
}

#[test]
fn string_and_float_operands_of_integer_operations_are_never_specialised() {
    // `3.0 << 1` converts an integral float. The recorder leaves it to the
    // interpreter. (String operands need the string library, so the
    // source-level tests in the CLI cover them.)
    let cases: Vec<(&str, Chunk)> = vec![
        (
            "float shift",
            for_loop_chunk(
                7,
                vec![int(1), int(300), int(0), Constant::Number(3.0)],
                [0, 1, 0],
                vec![load(4, 2), load(5, 3)],
                vec![
                    binary(OpCode::ShiftLeft, 6, 5, 0),
                    binary(OpCode::Add, 4, 4, 6),
                ],
            )
            .0,
        ),
        (
            "float bitwise and",
            for_loop_chunk(
                7,
                vec![int(1), int(300), int(0), Constant::Number(255.0)],
                [0, 1, 0],
                vec![load(4, 2), load(5, 3)],
                vec![
                    binary(OpCode::BitAnd, 6, 5, 3),
                    binary(OpCode::Add, 4, 4, 6),
                ],
            )
            .0,
        ),
    ];
    for profile in INTEGER_PROFILES {
        for (name, chunk) in &cases {
            let expected = interpreted(profile, chunk);
            let mut vm = native_vm(profile);
            assert_eq!(vm.execute(chunk).unwrap(), expected, "{name} {profile:?}");
            let stats = stats_of(&vm);
            assert_eq!(stats.traces_compiled, 0, "{name} {profile:?}: {stats:?}");
            assert_eq!(stats.iterations, 0, "{name} {profile:?}: {stats:?}");
            assert!(stats.recording_starts > 0, "{name} {profile:?}: {stats:?}");
            assert!(stats.total_aborts() > 0, "{name} {profile:?}: {stats:?}");
        }
    }
}

#[test]
fn profiles_without_integers_read_integer_constants_as_floats() {
    let (chunk, header) = for_loop_chunk(
        7,
        vec![int(1), int(300), int(0), int(3)],
        [0, 1, 0],
        vec![load(4, 2), load(5, 3)],
        vec![
            binary(OpCode::Multiply, 6, 5, 3),
            binary(OpCode::Add, 4, 4, 6),
        ],
    );
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::LuaJit21,
    ] {
        let installed = Installed::with_profile(profile, chunk.clone(), header);
        let ir = installed.ir();
        assert!(
            ir.entry
                .inputs
                .iter()
                .all(|input| input.ty == TraceType::Float),
            "{profile:?}: {:?}",
            ir.entry
        );
        let mut vm = native_vm(profile);
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            interpreted(profile, &chunk),
            "{profile:?}"
        );
        assert!(stats_of(&vm).iterations > 100, "{profile:?}");
    }
}

#[test]
fn forced_failures_of_before_guards_restore_a_state_the_interpreter_passes_through() {
    // A `Before` guard (a zero divisor, the sign of a for step, the overflow
    // headroom) resumes by re-executing the instruction from the state
    // before it. That state is the interpreter's whatever the guard's
    // condition says, so forcing the guard to fail, even where the data
    // would pass it, must still restore a reachable state. This covers
    // guards no entry variant can trip first.
    let mut forced_guards = 0;
    for profile in INTEGER_PROFILES {
        for (name, chunk, header) in integer_programs() {
            let installed = Installed::with_profile(profile, chunk, header);
            let ir = installed.ir();
            let run = drive_with(profile, &installed.chunk, None);
            let visits = header_visits(&run, header);
            let state = &run.states[visits[70]];
            for (guard_index, guard) in ir.guards.iter().enumerate() {
                if !matches!(
                    ir.snapshots[guard.snapshot.index()].phase,
                    crate::jit_ir::trace::SnapshotPhase::Before { .. }
                ) {
                    continue;
                }
                let forced = crate::jit_ir::trace::reference::ForcedExit {
                    frame: guard.site.frame,
                    pc: guard.site.pc,
                    guard: Some(crate::jit_ir::trace::GuardId(
                        u32::try_from(guard_index).unwrap(),
                    )),
                    iteration: 2,
                };
                let evaluation =
                    evaluate_forced(&ir, &state.registers, &[], 600, Some(forced), |_, _| None);
                let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
                    panic!("{name} {profile:?}: the forced guard did not fail");
                };
                forced_guards += 1;
                let restored = State {
                    pc: restored.pc,
                    last: restored.last_interpreted_pc,
                    registers: restored.registers,
                };
                assert!(
                    reaches_with(profile, &installed.chunk, state, &restored, 20_000),
                    "{name} {profile:?}: forced guard at pc {} restored {restored:?}",
                    guard.site.pc
                );
            }
        }
    }
    assert!(forced_guards > 30, "{forced_guards} guards forced");
}

#[test]
fn forced_failures_of_before_guards_take_the_native_exit_stubs() {
    // The same forcing as above, but the compiled code is rebuilt so the
    // guard's branch always exits. Each exit stub then writes its snapshot
    // natively, and the restored state must equal the reference's.
    let mut forced_guards = 0;
    for profile in INTEGER_PROFILES {
        for (name, chunk, header) in integer_programs() {
            let mut installed = Installed::with_profile(profile, chunk, header);
            let run = drive_with(profile, &installed.chunk, None);
            let visits = header_visits(&run, header);
            let lowered = installed
                .vm
                .jit
                .trace
                .installed_lowered_ir(installed.index)
                .unwrap()
                .clone();
            for (guard_index, guard) in lowered.guards.iter().enumerate() {
                if !matches!(
                    lowered.snapshots[guard.snapshot.index()].phase,
                    crate::jit_ir::trace::SnapshotPhase::Before { .. }
                ) {
                    continue;
                }
                if !installed
                    .vm
                    .jit
                    .trace
                    .force_guard(installed.index, guard_index)
                {
                    continue;
                }
                for &visit in &[visits[70], visits[77]] {
                    let state = &run.states[visit];
                    let forced = crate::jit_ir::trace::reference::ForcedExit {
                        frame: guard.site.frame,
                        pc: guard.site.pc,
                        guard: Some(crate::jit_ir::trace::GuardId(
                            u32::try_from(guard_index).unwrap(),
                        )),
                        iteration: 0,
                    };
                    let evaluation = evaluate_forced(
                        &lowered,
                        &state.registers,
                        &[],
                        600,
                        Some(forced),
                        |_, _| None,
                    );
                    let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
                        panic!("{name} {profile:?}: the forced guard did not fail");
                    };
                    let expected = State {
                        pc: restored.pc,
                        last: restored.last_interpreted_pc,
                        registers: restored.registers,
                    };
                    let (ran, actual) = installed.enter(state, 600);
                    assert!(
                        matches!(ran, TraceRun::Ran { .. }),
                        "{name} {profile:?}: guard {guard_index} did not run"
                    );
                    assert_eq!(
                        actual, expected,
                        "{name} {profile:?}: native exit of guard {guard_index} at pc {}",
                        guard.site.pc
                    );
                    assert!(
                        reaches_with(profile, &installed.chunk, state, &actual, 20_000),
                        "{name} {profile:?}: guard {guard_index} restored {actual:?}"
                    );
                    forced_guards += 1;
                }
            }
        }
    }
    assert!(forced_guards > 60, "{forced_guards} native forced exits");
}

#[test]
fn the_shared_trace_semantics_equal_the_interpreters_helpers() {
    use crate::jit_ir::Comparison;
    use crate::jit_ir::trace::{ArithmeticOp, arithmetic, compare};
    use crate::vm::arith::{bitwise_result, integer_floor_div, integer_floor_mod, numeric_order};

    let integers = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        7,
        -7,
        63,
        64,
        65,
        -63,
        -64,
        -65,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        i64::MIN + 1,
        9_007_199_254_740_992,
        9_007_199_254_740_993,
        -9_007_199_254_740_993,
        4_611_686_018_427_387_904,
    ];
    let floats = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.5,
        9_007_199_254_740_992.0,
        9_007_199_254_740_994.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        4_611_686_018_427_387_904.0,
        1e19,
        -1e19,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let integer = TraceScalar::Integer;
    let float = TraceScalar::Float;
    for &a in &integers {
        for &b in &integers {
            let apply = |op| arithmetic(op, integer(a), integer(b));
            if b != 0 {
                assert_eq!(
                    apply(ArithmeticOp::FloorDivide),
                    Some(integer(integer_floor_div(a, b).unwrap())),
                    "{a} // {b}"
                );
                assert_eq!(
                    apply(ArithmeticOp::Modulo),
                    Some(integer(integer_floor_mod(a, b).unwrap())),
                    "{a} % {b}"
                );
            }
            for (op, opcode) in [
                (ArithmeticOp::BitAnd, OpCode::BitAnd),
                (ArithmeticOp::BitOr, OpCode::BitOr),
                (ArithmeticOp::BitXor, OpCode::BitXor),
                (ArithmeticOp::ShiftLeft, OpCode::ShiftLeft),
                (ArithmeticOp::ShiftRight, OpCode::ShiftRight),
            ] {
                assert_eq!(
                    apply(op),
                    Some(integer(bitwise_result(opcode, a, b))),
                    "{op:?}({a}, {b})"
                );
            }
        }
    }
    // Mixed comparisons: order from the interpreter's ordering, equality from
    // raw equality.
    let mixed = |integer_value: i64, float_value: f64| {
        (Value::Integer(integer_value), Value::Number(float_value))
    };
    for &a in &integers {
        for &b in &floats {
            let (left, right) = mixed(a, b);
            for (comparison, expected) in [
                (
                    Comparison::LessThan,
                    numeric_order(&left, &right).is_some_and(std::cmp::Ordering::is_lt),
                ),
                (
                    Comparison::LessEqual,
                    numeric_order(&left, &right).is_some_and(std::cmp::Ordering::is_le),
                ),
                (Comparison::Equal, Runtime::raw_equal(&left, &right)),
            ] {
                assert_eq!(
                    compare(comparison, integer(a), float(b)),
                    Some(expected),
                    "{a} {comparison:?} {b:?}"
                );
            }
            for (comparison, expected) in [
                (
                    Comparison::LessThan,
                    numeric_order(&right, &left).is_some_and(std::cmp::Ordering::is_lt),
                ),
                (
                    Comparison::LessEqual,
                    numeric_order(&right, &left).is_some_and(std::cmp::Ordering::is_le),
                ),
                (Comparison::Equal, Runtime::raw_equal(&right, &left)),
            ] {
                assert_eq!(
                    compare(comparison, float(b), integer(a)),
                    Some(expected),
                    "{b:?} {comparison:?} {a}"
                );
            }
        }
    }
}
