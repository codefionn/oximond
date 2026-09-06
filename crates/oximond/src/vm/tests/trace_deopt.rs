//! Snapshot restoration onto real frames.
//!
//! The harness steps a program under the interpreter and records the state at
//! every instruction boundary. For each boundary it builds a snapshot from the
//! recorded registers, restores it onto a fresh frame full of garbage, and
//! checks two things. The restored frame equals the recorded state, and
//! running on from it finishes with the same results in exactly the number of
//! steps that remained. A replayed or skipped instruction changes one of them.

use super::*;
use crate::jit_ir::trace::{
    FrameId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TraceScalar, ValueId,
};
use crate::vm::jit::trace::deopt::{
    DeoptError, ExitValues, restore_root_snapshot, scalar_to_value, value_to_scalar,
};
use crate::vm::stack::CallStack;

#[derive(Clone, Debug)]
struct State {
    pc: usize,
    last: Option<usize>,
    registers: Vec<TraceScalar>,
}

struct Run {
    /// `states[k]` is the state before step `k`.
    states: Vec<State>,
    results: Vec<TraceScalar>,
}

impl Run {
    fn steps(&self) -> usize {
        self.states.len()
    }
}

fn interpreter_vm() -> Vm {
    Vm::with_profile_and_execution_mode(CompatibilityProfile::Lua54, ExecutionMode::Interpreter)
}

fn new_frames(vm: &Vm, chunk: &Arc<Chunk>) -> CallStack {
    CallStack::from(vec![Frame::new(
        Arc::clone(chunk),
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        None,
    )])
}

fn capture(frames: &CallStack) -> State {
    let frame = frames.last().expect("a live frame");
    State {
        pc: frame.pc,
        last: frame.last_interpreted_pc.get(),
        registers: frame
            .registers
            .iter()
            .map(|value| value_to_scalar(value).expect("test programs use scalars"))
            .collect(),
    }
}

fn scalars(values: &[Value]) -> Vec<TraceScalar> {
    values
        .iter()
        .map(|value| value_to_scalar(value).expect("scalar result"))
        .collect()
}

/// Steps `frames` to completion. `flip` negates the truthiness of a register
/// once the given number of steps have run.
fn drive(vm: &mut Vm, frames: &mut CallStack, already: usize, flip: Option<(usize, usize)>) -> Run {
    let mut states = Vec::new();
    loop {
        let step = already + states.len();
        if let Some((at, register)) = flip
            && at == step
        {
            let frame = frames.last_mut().unwrap();
            let flipped = if frame.registers[register].is_truthy() {
                Value::Nil
            } else {
                Value::Boolean(true)
            };
            frame.registers[register] = flipped;
        }
        states.push(capture(frames));
        assert!(states.len() < 10_000, "program does not terminate");
        if let Some(results) = vm.step(frames).unwrap() {
            return Run {
                states,
                results: scalars(&results),
            };
        }
    }
}

fn reference(chunk: &Arc<Chunk>, flip: Option<(usize, usize)>) -> Run {
    let mut vm = interpreter_vm();
    let mut frames = new_frames(&vm, chunk);
    drive(&mut vm, &mut frames, 0, flip)
}

/// Value ids are spread out so a restore cannot pass by reading index `i`.
fn value_id(register: usize) -> ValueId {
    ValueId(u32::try_from(register * 2 + 1).unwrap())
}

/// A register map that mixes every entry kind, plus the exit values behind it.
fn mixed_entries(registers: &[TraceScalar]) -> (Vec<RegisterEntry>, Vec<Option<TraceScalar>>) {
    let mut values = vec![None; registers.len() * 2 + 1];
    let entries = registers
        .iter()
        .enumerate()
        .map(|(index, scalar)| match index % 3 {
            0 => {
                values[value_id(index).index()] = Some(*scalar);
                RegisterEntry::Value(value_id(index))
            }
            1 => RegisterEntry::Const(*scalar),
            _ => RegisterEntry::Unchanged,
        })
        .collect();
    (entries, values)
}

fn snapshot(phase: SnapshotPhase, last: Option<usize>, entries: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: last,
            registers: entries.into(),
        }],
    )
}

/// Restores `snapshot` onto a garbage-filled fresh frame. `Unchanged` registers
/// keep the true value, as they do in the interpreter during a native run.
fn restore_fresh(
    chunk: &Arc<Chunk>,
    snapshot: &Snapshot,
    values: &[Option<TraceScalar>],
    truth: &[TraceScalar],
) -> (Vm, CallStack) {
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, chunk);
    {
        let mut frame = frames.last_mut().unwrap();
        frame.pc = 0;
        frame.last_interpreted_pc = None.into();
        for (index, register) in frame.registers.iter_mut().enumerate() {
            *register = match snapshot.frames[0].registers[index] {
                RegisterEntry::Unchanged => scalar_to_value(truth[index]),
                _ => Value::from("garbage"),
            };
        }
        restore_root_snapshot(&mut frame, snapshot, values, &[]).unwrap();
    }
    (vm, frames)
}

fn assert_state(actual: &State, expected: &State, context: &str) {
    assert_eq!(actual.pc, expected.pc, "pc: {context}");
    assert_eq!(actual.last, expected.last, "last_interpreted_pc: {context}");
    assert_eq!(actual.registers, expected.registers, "registers: {context}");
}

/// Restores before and after every dynamic instruction of `chunk`.
fn check_every_boundary(chunk: &Chunk) {
    let chunk = Arc::new(chunk.clone());
    let run = reference(&chunk, None);
    assert!(run.steps() > 3, "program is too short to mean anything");
    for k in 0..run.steps() {
        let expected = &run.states[k];
        let (entries, values) = mixed_entries(&expected.registers);

        // Before step k: the interpreter executes the instruction itself.
        let before = snapshot(
            SnapshotPhase::Before { pc: expected.pc },
            expected.last,
            entries.clone(),
        );
        let (mut vm, mut frames) = restore_fresh(&chunk, &before, &values, &expected.registers);
        let context = format!("before step {k} at pc {}", expected.pc);
        assert_state(&capture(&frames), expected, &context);
        let tail = drive(&mut vm, &mut frames, 0, None);
        assert_eq!(tail.results, run.results, "results: {context}");
        assert_eq!(tail.steps(), run.steps() - k, "steps: {context}");

        // After step k - 1: it committed once and must not run again.
        if k > 0 {
            let executed = expected.last.expect("a step ran before");
            let after = snapshot(
                SnapshotPhase::After {
                    pc: executed,
                    successor: expected.pc,
                },
                None,
                entries,
            );
            let (mut vm, mut frames) = restore_fresh(&chunk, &after, &values, &expected.registers);
            let context = format!("after pc {executed}, resume at {}", expected.pc);
            assert_state(&capture(&frames), expected, &context);
            let tail = drive(&mut vm, &mut frames, 0, None);
            assert_eq!(tail.results, run.results, "results: {context}");
            assert_eq!(tail.steps(), run.steps() - k, "steps: {context}");
        }
    }
}

/// For every conditional branch: restore its post-state with the alternative
/// successor and compare against an interpreter run where the branch
/// condition had the other truthiness.
fn check_branch_alternatives(chunk: &Chunk) -> usize {
    let chunk = Arc::new(chunk.clone());
    let run = reference(&chunk, None);
    let mut checked = 0;
    for (j, state) in run.states.iter().enumerate() {
        let instruction = chunk.code[state.pc];
        if instruction.opcode() != Ok(OpCode::JumpIfFalse) {
            continue;
        }
        let condition = usize::from(instruction.a());
        let flipped = reference(&chunk, Some((j, condition)));
        // The flipped run starts the branch with the other condition.
        let at_branch = &flipped.states[j];
        let alternative = flipped.states[j + 1].pc;
        assert_ne!(alternative, run.states[j + 1].pc, "the successor differs");
        // A branch writes nothing, so its post-state is the pre-state.
        let (entries, values) = mixed_entries(&at_branch.registers);
        let after = snapshot(
            SnapshotPhase::After {
                pc: state.pc,
                successor: alternative,
            },
            None,
            entries,
        );
        let (mut vm, mut frames) = restore_fresh(&chunk, &after, &values, &at_branch.registers);
        let tail = drive(&mut vm, &mut frames, 0, None);
        let context = format!("alternative of the branch at pc {}", state.pc);
        assert_eq!(tail.results, flipped.results, "{context}");
        assert_eq!(tail.steps(), flipped.steps() - (j + 1), "{context}");
        checked += 1;
    }
    checked
}

const REGISTERS: u8 = 10;

fn constants() -> Vec<Constant> {
    vec![
        Constant::Number(7.5),           // 0
        Constant::Number(2.0),           // 1
        Constant::Number(-3.0),          // 2
        Constant::Boolean(true),         // 3
        Constant::Integer(5),            // 4
        Constant::Integer(3),            // 5
        Constant::Boolean(false),        // 6
        Constant::Integer(1),            // 7
        Constant::Integer(0),            // 8
        Constant::Number(1.0),           // 9
        Constant::Number(2.5),           // 10
        Constant::Number(0.5),           // 11
        Constant::Integer(i64::MAX - 1), // 12
        Constant::Integer(i64::MAX),     // 13
        Constant::Integer(-1),           // 14
    ]
}

fn program(code: Vec<Instruction>) -> Chunk {
    let mut code = code;
    code.push(Instruction::return_values(0, REGISTERS));
    Chunk::new(REGISTERS, constants(), code)
}

fn load(destination: u8, constant: u16) -> Instruction {
    Instruction::load_constant(destination, constant)
}

fn binary(opcode: OpCode, destination: u8, left: u8, right: u8) -> Instruction {
    Instruction::binary(opcode, destination, left, right)
}

fn jump(offset: i32) -> Instruction {
    Instruction::aj(OpCode::Jump, offset).unwrap()
}

/// A numeric for with a body of `body.len()` instructions. Registers 0..4 are
/// the loop control, limit, step and visible index.
fn for_loop(start: u16, limit: u16, step: u16, body: &[Instruction]) -> Chunk {
    let length = i16::try_from(body.len()).unwrap();
    let mut code = vec![
        load(0, start),
        load(1, limit),
        load(2, step),
        load(5, 8),
        load(6, 9),
        load(7, 0),
        Instruction::numeric_for_prep(0, length),
    ];
    code.extend_from_slice(body);
    code.push(Instruction::numeric_for_loop(0, -(length + 1)));
    program(code)
}

fn arithmetic_program() -> Chunk {
    program(vec![
        load(0, 0),
        load(1, 1),
        load(2, 2),
        load(3, 3),
        load(4, 4),
        load(5, 5),
        binary(OpCode::Add, 6, 0, 1),
        binary(OpCode::Add, 0, 0, 1),
        binary(OpCode::Subtract, 7, 0, 2),
        binary(OpCode::Multiply, 8, 1, 2),
        binary(OpCode::Divide, 9, 0, 1),
        binary(OpCode::Modulo, 6, 0, 2),
        Instruction::power(7, 1, 1),
        binary(OpCode::FloorDivide, 8, 0, 1),
        Instruction::unary(OpCode::Negate, 9, 2),
        Instruction::unary(OpCode::Not, 6, 3),
        binary(OpCode::Equal, 7, 0, 1),
        binary(OpCode::LessThan, 8, 2, 1),
        binary(OpCode::LessEqual, 9, 1, 0),
        Instruction::move_(6, 4),
        binary(OpCode::Add, 4, 4, 5),
        binary(OpCode::Multiply, 5, 4, 5),
        binary(OpCode::Subtract, 4, 4, 5),
        Instruction::load_nil(8),
        Instruction::clear_range(6, 2),
        load(9, 0),
    ])
}

fn branch_program() -> Chunk {
    program(vec![
        load(0, 3),
        load(1, 6),
        load(2, 0),
        Instruction::jump_if_false(0, 1), // not taken
        binary(OpCode::Add, 2, 2, 2),
        Instruction::jump_if_false(1, 1), // taken
        binary(OpCode::Add, 2, 2, 2),     // skipped
        jump(1),
        binary(OpCode::Add, 2, 2, 2), // skipped
        Instruction::unary(OpCode::Not, 3, 1),
        binary(OpCode::Equal, 4, 2, 2),
        Instruction::jump_if_false(4, 1), // not taken
        binary(OpCode::Add, 2, 2, 2),
    ])
}

#[test]
fn restores_every_boundary_of_straight_line_arithmetic() {
    check_every_boundary(&arithmetic_program());
}

#[test]
fn restores_every_boundary_around_branches() {
    let chunk = branch_program();
    check_every_boundary(&chunk);
    assert_eq!(check_branch_alternatives(&chunk), 3);
}

#[test]
fn restores_numeric_for_control_and_visible_index_writes() {
    let body = [binary(OpCode::Add, 5, 5, 3), Instruction::move_(8, 3)];
    for (name, chunk) in [
        // integer loop, ascending
        ("integer", for_loop(7, 5, 7, &body)),
        // float loop
        ("float", for_loop(9, 10, 11, &body)),
        // descending integer loop: 5, 4, 3 with step -1
        ("descending", for_loop(4, 5, 14, &body)),
        // empty range: prep skips the latch
        ("empty", for_loop(5, 7, 7, &body)),
        // control reaches i64::MAX, the latch's checked add stops the loop
        ("overflow", for_loop(12, 13, 7, &body)),
    ] {
        eprintln!("numeric for: {name}");
        check_every_boundary(&chunk);
    }
}

#[test]
fn numeric_for_restores_both_write_registers_after_the_latch() {
    // `After` the latch that continues: the control register and the visible
    // index both hold the new value, and the interpreter resumes at the body.
    let chunk = Arc::new(for_loop(7, 5, 7, &[binary(OpCode::Add, 5, 5, 3)]));
    let run = reference(&chunk, None);
    let latch = chunk
        .code
        .iter()
        .position(|i| i.opcode() == Ok(OpCode::NumericForLoop))
        .unwrap();
    let first = run
        .states
        .iter()
        .position(|state| state.last == Some(latch))
        .unwrap();
    let state = &run.states[first];
    assert_eq!(state.registers[0], TraceScalar::Integer(2));
    assert_eq!(state.registers[3], TraceScalar::Integer(2));
    assert_eq!(state.pc, latch - 1, "back at the body");
    check_every_boundary(&chunk);
}

#[test]
fn restores_loop_carried_swaps() {
    // r6 and r7 swap through r8 every iteration.
    let body = [
        Instruction::move_(8, 6),
        Instruction::move_(6, 7),
        Instruction::move_(7, 8),
    ];
    let chunk = for_loop(7, 5, 7, &body);
    check_every_boundary(&chunk);
}

#[test]
fn restores_a_register_that_is_live_only_on_the_cold_branch() {
    // r6 is the cold-only product. The hot path never reads it. When the
    // loop test fails, the cold successor does.
    let body = [
        binary(OpCode::Add, 6, 3, 3),
        binary(OpCode::LessThan, 8, 3, 4),
        Instruction::jump_if_false(8, 1),
        binary(OpCode::Add, 5, 5, 3),
        Instruction::move_(9, 6),
    ];
    let mut chunk = for_loop(7, 5, 7, &body);
    // r4 = 3 so the comparison flips on the last iteration.
    chunk.code[5] = load(4, 5);
    check_every_boundary(&chunk);
    assert!(check_branch_alternatives(&chunk) >= 3);
}

#[test]
fn integer_and_float_bits_keep_their_types() {
    let chunk = Arc::new(program(vec![load(0, 0)]));
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let nan = f64::from_bits(0x7ff8_0000_dead_beef);
    let scalars = [
        TraceScalar::Integer(i64::MAX),
        TraceScalar::Integer(i64::MIN),
        TraceScalar::Integer(5),
        TraceScalar::Float(5.0),
        TraceScalar::Float(-0.0),
        TraceScalar::Float(nan),
        TraceScalar::Float(f64::INFINITY),
        TraceScalar::Boolean(false),
        TraceScalar::Nil,
        TraceScalar::Integer(1 << 60),
    ];
    let values: Vec<Option<TraceScalar>> = scalars.iter().copied().map(Some).collect();
    let entries = (0..scalars.len())
        .map(|i| RegisterEntry::Value(ValueId(u32::try_from(i).unwrap())))
        .collect();
    let snap = snapshot(SnapshotPhase::Before { pc: 0 }, None, entries);
    let mut frame = frames.last_mut().unwrap();
    restore_root_snapshot(&mut frame, &snap, values.as_slice(), &[]).unwrap();
    assert_eq!(frame.registers[0], Value::Integer(i64::MAX));
    assert_eq!(frame.registers[1], Value::Integer(i64::MIN));
    assert_eq!(frame.registers[2], Value::Integer(5));
    assert_eq!(frame.registers[3], Value::Number(5.0));
    assert!(matches!(frame.registers[4], Value::Number(v) if v.to_bits() == (-0.0f64).to_bits()));
    assert!(matches!(frame.registers[5], Value::Number(v) if v.to_bits() == nan.to_bits()));
    assert_eq!(frame.registers[7], Value::Boolean(false));
    assert_eq!(frame.registers[8], Value::Nil);
    assert_eq!(frame.registers[9], Value::Integer(1 << 60));
}

#[test]
fn a_borrowed_string_restores_as_a_clone_of_its_table_slot() {
    // The array part of a table the exit's owner holds: slot 1 is the
    // borrowed string, slot 2 another one. Two registers borrow the same
    // slot and one the other. Each restores as the very string, sharing its
    // allocation, and the slots keep theirs.
    let slots = [Value::from("alpha"), Value::from("beta"), Value::Integer(7)];
    let address = |slot: &Value| std::ptr::from_ref(slot) as u64;
    let chunk = Arc::new(program(vec![load(0, 0)]));
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let mut frame = frames.last_mut().unwrap();
    let mut entries = vec![RegisterEntry::Unchanged; usize::from(REGISTERS)];
    let mut values = vec![None; 3];
    values[0] = Some(TraceScalar::StringRef(address(&slots[0])));
    values[1] = Some(TraceScalar::StringRef(address(&slots[1])));
    entries[0] = RegisterEntry::Value(ValueId(0));
    entries[1] = RegisterEntry::Value(ValueId(0));
    entries[2] = RegisterEntry::Value(ValueId(1));
    let snap = snapshot(SnapshotPhase::Before { pc: 0 }, None, entries);
    restore_root_snapshot(&mut frame, &snap, values.as_slice(), &[]).unwrap();
    let same_string = |restored: &Value, slot: &Value| match (restored, slot) {
        (Value::String(a), Value::String(b)) => a.ptr_eq(b),
        _ => false,
    };
    assert!(same_string(&frame.registers[0], &slots[0]));
    assert!(same_string(&frame.registers[1], &slots[0]));
    assert!(same_string(&frame.registers[2], &slots[1]));
    assert_eq!(frame.registers[0], Value::from("alpha"));
    assert_eq!(slots[2], Value::Integer(7));
}

#[test]
fn root_entries_restore_opaque_values() {
    let chunk = Arc::new(program(vec![load(0, 0)]));
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let mut frame = frames.last_mut().unwrap();
    let mut entries = vec![RegisterEntry::Unchanged; usize::from(REGISTERS)];
    entries[2] = RegisterEntry::Root(1);
    let snap = snapshot(SnapshotPhase::Before { pc: 0 }, None, entries);
    let roots = [Value::from("zero"), Value::from("one")];
    let none: &[Option<TraceScalar>] = &[];
    restore_root_snapshot(&mut frame, &snap, none, &roots).unwrap();
    assert_eq!(frame.registers[2], Value::from("one"));
}

#[test]
fn restore_reports_resume_point_and_sets_frame_fields() {
    let chunk = Arc::new(program(vec![load(0, 0), load(1, 1)]));
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let mut frame = frames.last_mut().unwrap();
    let none: &[Option<TraceScalar>] = &[];
    let entries = vec![RegisterEntry::Unchanged; usize::from(REGISTERS)];
    let before = snapshot(SnapshotPhase::Before { pc: 2 }, Some(1), entries.clone());
    let resume = restore_root_snapshot(&mut frame, &before, none, &[]).unwrap();
    assert_eq!((resume.pc, resume.last_interpreted_pc), (2, Some(1)));
    assert_eq!((frame.pc, frame.last_interpreted_pc.get()), (2, Some(1)));
    let after = snapshot(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        Some(0),
        entries,
    );
    restore_root_snapshot(&mut frame, &after, none, &[]).unwrap();
    assert_eq!((frame.pc, frame.last_interpreted_pc.get()), (2, Some(1)));
}

#[test]
fn failed_restores_leave_the_frame_untouched() {
    let chunk = Arc::new(program(vec![load(0, 0), load(1, 1)]));
    let vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let mut frame = frames.last_mut().unwrap();
    frame.pc = 1;
    frame.last_interpreted_pc = Some(0).into();
    frame.registers[0] = Value::Integer(11);
    let before_registers = frame.registers.to_vec();
    let none: &[Option<TraceScalar>] = &[];
    let good =
        |entries: Vec<RegisterEntry>| snapshot(SnapshotPhase::Before { pc: 2 }, None, entries);
    let full = || vec![RegisterEntry::Const(TraceScalar::Nil); usize::from(REGISTERS)];

    // A missing value after earlier entries that would have written.
    let mut missing = full();
    missing[9] = RegisterEntry::Value(ValueId(40));
    assert_eq!(
        restore_root_snapshot(&mut frame, &good(missing), none, &[]),
        Err(DeoptError::MissingValue(ValueId(40)))
    );
    let mut bad_root = full();
    bad_root[9] = RegisterEntry::Root(3);
    assert_eq!(
        restore_root_snapshot(&mut frame, &good(bad_root), none, &[]),
        Err(DeoptError::MissingRoot(3))
    );
    assert_eq!(
        restore_root_snapshot(&mut frame, &good(full()[1..].to_vec()), none, &[]),
        Err(DeoptError::RegisterCount {
            snapshot: usize::from(REGISTERS) - 1,
            frame: usize::from(REGISTERS),
        })
    );
    let far = snapshot(SnapshotPhase::Before { pc: 999 }, None, full());
    assert_eq!(
        restore_root_snapshot(&mut frame, &far, none, &[]),
        Err(DeoptError::PcOutOfBounds(999))
    );
    let mut two = good(full());
    two.frames.push(two.frames[0].clone());
    assert_eq!(
        restore_root_snapshot(&mut frame, &two, none, &[]),
        Err(DeoptError::NotRootOnly { frames: 2 })
    );
    assert_eq!(frame.pc, 1);
    assert_eq!(frame.last_interpreted_pc.get(), Some(0));
    assert_eq!(frame.registers.to_vec(), before_registers);
}

#[test]
fn a_restore_writes_registers_and_an_open_cell_reads_them() {
    let chunk = Arc::new(program(vec![load(0, 0), load(1, 1)]));
    let mut vm = interpreter_vm();
    let mut frames = new_frames(&vm, &chunk);
    let cell = vm.capture_slot(&mut frames, 0, 0).unwrap();
    let mut frame = frames.last_mut().unwrap();
    frame.registers[0] = Value::Integer(7);
    let none: &[Option<TraceScalar>] = &[];
    let mut entries = vec![RegisterEntry::Const(TraceScalar::Nil); usize::from(REGISTERS)];
    entries[0] = RegisterEntry::Const(TraceScalar::Integer(9));
    let snapshot = snapshot(SnapshotPhase::Before { pc: 1 }, None, entries);
    restore_root_snapshot(&mut frame, &snapshot, none, &[]).unwrap();
    assert_eq!(frame.pc, 1);
    assert_eq!(frame.registers[0], Value::Integer(9));
    // The cell stands for the register: it reads the restored value.
    assert_eq!(vm.runtime().cell_get(cell).unwrap(), Value::Integer(9));
    vm.close_upvalues(&mut frames, 0);
}

#[test]
fn exit_values_slice_reads_by_value_id() {
    let values = [None, Some(TraceScalar::Integer(4))];
    assert_eq!(values[..].scalar(ValueId(1)), Some(TraceScalar::Integer(4)));
    assert_eq!(values[..].scalar(ValueId(0)), None);
    assert_eq!(values[..].scalar(ValueId(9)), None);
}
