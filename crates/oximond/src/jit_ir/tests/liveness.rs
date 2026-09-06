use super::*;
use crate::bytecode::LocalVariable;
use crate::jit_ir::{observable_registers, register_set_contains};

fn observable(set: &crate::jit_ir::RegisterSet) -> Vec<u8> {
    (0..=255)
        .filter(|register| register_set_contains(set, *register))
        .collect()
}

#[test]
fn straight_line_code_observes_only_later_reads() {
    // r2 = r0 + r1; r3 = -r2; return r3
    let chunk = Chunk::new(
        4,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::unary(OpCode::Negate, 3, 2),
            Instruction::return_(3),
        ],
    );
    let live = observable_registers(&chunk);
    assert_eq!(observable(&live[0]), vec![0, 1]);
    assert_eq!(observable(&live[1]), vec![2]);
    assert_eq!(observable(&live[2]), vec![3]);
}

#[test]
fn halt_and_overwrites_end_liveness() {
    let chunk = Chunk::new(
        3,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::load_constant(2, 0),
            Instruction::halt(),
        ],
    );
    let live = observable_registers(&chunk);
    // The sum is overwritten before anything reads it.
    assert_eq!(observable(&live[1]), Vec::<u8>::new());
    assert_eq!(observable(&live[2]), Vec::<u8>::new());
}

#[test]
fn open_return_keeps_registers_observable_and_closes_read_only_their_register() {
    let chunk = Chunk::new(
        4,
        Vec::new(),
        vec![Instruction::load_nil(0), Instruction::return_values(1, 0)],
    );
    let live = observable_registers(&chunk);
    assert!(register_set_contains(&live[1], 3));
    assert!(!register_set_contains(&live[1], 0));

    let chunk = Chunk::new(
        4,
        Vec::new(),
        vec![
            Instruction::load_nil(0),
            Instruction::close_upvalue(0),
            Instruction::halt(),
        ],
    );
    let live = observable_registers(&chunk);
    assert!(register_set_contains(&live[1], 0));
    assert!(!register_set_contains(&live[1], 3));
}

#[test]
fn a_dead_temporary_is_not_observable_without_a_clear() {
    // r1 = r0 + r0 is read once; the compiler leaves no clear behind it, and
    // the closure and concatenation after it do not make it observable.
    let mut chunk = Chunk::new(
        4,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::Add, 1, 0, 0),
            Instruction::move_(2, 1),
            Instruction::concat(3, 2, 2),
            Instruction::ad(OpCode::Closure, 2, 0),
            Instruction::return_(2),
        ],
    );
    chunk.prototypes.push(std::sync::Arc::new(Chunk::new(
        1,
        Vec::new(),
        vec![Instruction::halt()],
    )));
    let live = observable_registers(&chunk);
    assert_eq!(observable(&live[1]), vec![1]);
    assert_eq!(observable(&live[2]), vec![2]);
    assert_eq!(observable(&live[3]), Vec::<u8>::new());
}

#[test]
fn loops_named_locals_and_captures_stay_observable() {
    // for r0..r3 do r4 = r3 * r3 end; halt
    let chunk = Chunk::new(
        6,
        Vec::new(),
        vec![
            Instruction::numeric_for_prep(0, 1),
            Instruction::binary(OpCode::Multiply, 4, 3, 3),
            Instruction::numeric_for_loop(0, -2),
            Instruction::load_nil(5),
            Instruction::halt(),
        ],
    )
    .with_debug_metadata(
        DebugMetadata::new("liveness.lua", Vec::new())
            .with_local_variables(vec![LocalVariable::new("kept", 5, 0, 5)]),
    );
    let live = observable_registers(&chunk);
    // The loop control is read by the latch; r4 is a dead temporary; r5 is
    // a named local in scope.
    assert_eq!(observable(&live[1]), vec![0, 1, 2, 3, 5]);
    assert_eq!(observable(&live[2]), vec![0, 1, 2, 3, 5]);

    let mut parent = Chunk::new(
        2,
        Vec::new(),
        vec![Instruction::load_nil(1), Instruction::halt()],
    );
    let mut child = Chunk::new(1, Vec::new(), vec![Instruction::halt()]);
    child.upvalues.push(crate::bytecode::UpvalueDescriptor {
        name: "captured".into(),
        source: UpvalueSource::ParentRegister(1),
    });
    parent.prototypes.push(std::sync::Arc::new(child));
    let live = observable_registers(&parent);
    assert!(register_set_contains(&live[1], 1));
}
