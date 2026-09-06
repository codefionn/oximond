//! `Add` and `AddK` followed by `NumericForLoop` run as one step on the
//! dispatch loop, with the result `Add` or `AddK` gives alone.
//!
//! The dispatch loop runs here because profiling keeps the fast loop out. A
//! fused step skips `scalar_instruction`, so its counters count only the
//! additions that were not fused.

use super::*;

const ITERATIONS: i64 = 5;

fn whole(profile: CompatibilityProfile, value: i64) -> Constant {
    if profile.is_lua53_or_later() {
        Constant::Integer(value)
    } else {
        #[allow(clippy::cast_precision_loss)]
        Constant::Number(value as f64)
    }
}

/// `sum = <initial>; for i = 1, 5 do sum = sum + <addend> end return sum`,
/// with `opcode` reading the addend as a constant (`AddK`) or from a
/// register (`Add`).
fn sum_loop(
    profile: CompatibilityProfile,
    initial: Constant,
    addend: Constant,
    opcode: OpCode,
) -> Chunk {
    // Constants: 0 the start and step, 1 the limit, 2 the initial sum, 3 the
    // addend. Registers: 0..=2 the loop control, 3 the loop variable, 4 the
    // sum, 5 the addend for `Add`.
    let add = match opcode {
        OpCode::AddK => Instruction::abc(OpCode::AddK, 4, 4, 3),
        _ => Instruction::binary(OpCode::Add, 4, 4, 5),
    };
    let code = vec![
        Instruction::load_constant(0, 0),
        Instruction::load_constant(1, 1),
        Instruction::load_constant(2, 0),
        Instruction::load_constant(4, 2),
        Instruction::load_constant(5, 3),
        Instruction::numeric_for_prep(0, 1),
        add,
        Instruction::numeric_for_loop(0, -2),
        Instruction::return_(4),
    ];
    Chunk::with_parts(
        6,
        0,
        vec![
            whole(profile, 1),
            whole(profile, ITERATIONS),
            initial,
            addend,
        ],
        code,
        Vec::new(),
        Vec::new(),
    )
}

/// The results, the additions that ran as a plain scalar step, and those that
/// took the slow path.
fn run(profile: CompatibilityProfile, chunk: &Chunk) -> (Vec<Value>, u64, u64) {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    vm.debug_hooks.clear();
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm.set_interpreter_profiling(true);
    chunk.validate().unwrap();
    let results = vm.execute_with_arguments(chunk, Vec::new()).unwrap();
    let counters = vm.interpreter_counters();
    (
        results,
        counters.scalar_fast.get(),
        counters.scalar_slow.get(),
    )
}

#[test]
fn a_constant_addition_before_the_latch_is_fused() {
    for opcode in [OpCode::AddK, OpCode::Add] {
        // Lua 5.4 keeps integers integers, and a float addend makes floats.
        let profile = CompatibilityProfile::Lua54;
        let cases = [
            (
                Constant::Integer(0),
                Constant::Integer(3),
                Value::Integer(15),
            ),
            (
                Constant::Integer(0),
                Constant::Number(0.5),
                Value::Number(2.5),
            ),
            (
                Constant::Number(1.5),
                Constant::Integer(2),
                Value::Number(11.5),
            ),
            (
                Constant::Number(1.0),
                Constant::Integer(1),
                Value::Number(6.0),
            ),
            (
                Constant::Integer(i64::MAX - 2),
                Constant::Integer(1),
                Value::Integer(i64::MIN + 2),
            ),
        ];
        for (initial, addend, expected) in cases {
            let chunk = sum_loop(profile, initial.clone(), addend.clone(), opcode);
            let (results, fast, slow) = run(profile, &chunk);
            assert_eq!(
                results,
                vec![expected],
                "{opcode:?} {initial:?} + {addend:?}"
            );
            assert_eq!((fast, slow), (0, 0), "{opcode:?}: not fused");
        }
        // Lua 5.1 has floats only.
        let profile = CompatibilityProfile::Lua51;
        let chunk = sum_loop(
            profile,
            Constant::Number(0.0),
            Constant::Number(3.0),
            opcode,
        );
        let (results, fast, slow) = run(profile, &chunk);
        assert_eq!(results, vec![Value::Number(15.0)], "{opcode:?}");
        assert_eq!((fast, slow), (0, 0), "{opcode:?}: not fused");
    }
}

#[test]
fn a_string_sum_takes_the_slow_path_once_and_then_fuses() {
    // `"10" + 1` coerces through the slow path to the integer 11 (a float in
    // Lua 5.1), and the next additions are fused.
    for (profile, expected) in [
        (CompatibilityProfile::Lua54, Value::Integer(15)),
        (CompatibilityProfile::Lua51, Value::Number(15.0)),
    ] {
        for opcode in [OpCode::AddK, OpCode::Add] {
            let chunk = sum_loop(
                profile,
                Constant::String(LuaString::from("10")),
                whole(profile, 1),
                opcode,
            );
            let (results, fast, slow) = run(profile, &chunk);
            assert_eq!(results, vec![expected.clone()], "{profile:?} {opcode:?}");
            assert_eq!((fast, slow), (0, 1), "{profile:?} {opcode:?}");
        }
    }
}

#[test]
fn a_constant_addition_that_is_not_before_a_latch_is_not_fused() {
    // The counter would stay at zero for every addition if it did not count
    // the unfused ones.
    let profile = CompatibilityProfile::Lua54;
    let chunk = Chunk::with_parts(
        5,
        0,
        vec![Constant::Integer(40), Constant::Integer(2)],
        vec![
            Instruction::load_constant(4, 0),
            Instruction::abc(OpCode::AddK, 4, 4, 1),
            Instruction::return_(4),
        ],
        Vec::new(),
        Vec::new(),
    );
    let (results, fast, slow) = run(profile, &chunk);
    assert_eq!(results, vec![Value::Integer(42)]);
    assert_eq!((fast, slow), (1, 0));
}
