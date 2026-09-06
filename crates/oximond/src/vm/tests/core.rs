use super::*;

#[test]
fn execution_policy_is_independent_from_compatibility() {
    let vm = Vm::with_profile_and_execution_mode(
        crate::compat::CompatibilityProfile::Lua54,
        ExecutionMode::Tracing,
    );
    assert_eq!(
        vm.compatibility_profile(),
        crate::compat::CompatibilityProfile::Lua54
    );
    assert_eq!(vm.execution_mode(), ExecutionMode::Tracing);
}

#[test]
fn halt_has_no_results() {
    let chunk = Chunk::new(0, vec![], vec![Instruction::halt()]);
    assert_eq!(Vm::new().execute(&chunk).unwrap(), Vec::<Value>::new());
}

#[test]
fn interpreter_backedges_advance_an_active_cycle_without_new_allocations() {
    let loop_chunk = Chunk::new(
        5,
        vec![
            Constant::Number(1.0),
            Constant::Number(5_000.0),
            Constant::Number(1.0),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(4, 3),
            Instruction::numeric_for_prep(0, 2),
            Instruction::binary(OpCode::Add, 4, 4, 3),
            Instruction::aj(OpCode::Jump, 0).unwrap(),
            Instruction::numeric_for_loop(0, -3),
            Instruction::return_(4),
        ],
    );
    let mut vm = Vm::with_execution_mode(ExecutionMode::Interpreter);
    vm.runtime_mut().stop_gc();
    make_gc_due(vm.runtime_mut());
    vm.runtime_mut().restart_gc();
    assert!(vm.runtime().gc_needs_collection());

    assert_eq!(
        vm.execute(&loop_chunk).unwrap(),
        vec![Value::Number(12_502_500.0)]
    );
    assert!(
        !vm.runtime().gc_needs_collection(),
        "allocation-free interpreter backedges must finish active GC work"
    );
}

#[test]
fn interpreter_backedges_poll_again_after_the_countdown_rearms() {
    let loop_chunk = Chunk::new(
        5,
        vec![
            Constant::Number(1.0),
            Constant::Number(5_000.0),
            Constant::Number(1.0),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(4, 3),
            Instruction::numeric_for_prep(0, 2),
            Instruction::binary(OpCode::Add, 4, 4, 3),
            Instruction::aj(OpCode::Jump, 0).unwrap(),
            Instruction::numeric_for_loop(0, -3),
            Instruction::return_(4),
        ],
    );
    let mut vm = Vm::with_execution_mode(ExecutionMode::Interpreter);
    // Start with a full countdown so the first poll comes only after a rearm
    // and the initial value of 1 cannot make this test pass.
    vm.backedge_gc_countdown = crate::vm::BACKEDGE_GC_POLL_INTERVAL;
    vm.runtime_mut().stop_gc();
    make_gc_due(vm.runtime_mut());
    vm.runtime_mut().restart_gc();
    assert!(vm.runtime().gc_needs_collection());

    assert_eq!(
        vm.execute(&loop_chunk).unwrap(),
        vec![Value::Number(12_502_500.0)]
    );
    assert!(
        !vm.runtime().gc_needs_collection(),
        "a rearmed countdown must still reach a collector poll"
    );
}

#[test]
fn integer_constants_follow_lua_number_rounding() {
    let chunk = Chunk::new(
        1,
        vec![Constant::Integer(i64::MAX)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    assert_eq!(
        Vm::new().execute(&chunk).unwrap(),
        vec![Value::Number(9_223_372_036_854_775_808.0)]
    );
}

#[test]
fn modulo_uses_floor_and_keeps_zero_division_ieee() {
    let modulo = Chunk::new(
        3,
        vec![Constant::Number(-3.0), Constant::Number(2.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&modulo).unwrap(),
        vec![Value::Number(1.0)]
    );

    let zero_divisor = Chunk::new(
        3,
        vec![Constant::Number(1.0), Constant::Number(0.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    let result = Vm::new().execute(&zero_divisor).unwrap();
    assert!(matches!(result.as_slice(), [Value::Number(value)] if value.is_nan()));
}

#[test]
fn numeric_for_handles_fractional_and_negative_steps() {
    let positive = numeric_sum_chunk(1.0, 3.0, 1.0);
    assert_eq!(
        Vm::new().execute(&positive).unwrap(),
        vec![Value::Number(6.0)]
    );

    let negative = numeric_sum_chunk(1.5, 0.5, -0.5);
    assert_eq!(
        Vm::new().execute(&negative).unwrap(),
        vec![Value::Number(3.0)]
    );
}

#[test]
fn integer_numeric_for_rounds_fractional_limits_toward_loop_direction() {
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        for (start, limit, step, expected) in [
            (1, Constant::Number(3.5), 1, 6),
            (3, Constant::Number(0.5), -1, 6),
            (1, Constant::String(LuaString::from("3.5")), 1, 6),
            (-3, Constant::Number(-0.5), 1, -6),
            (-1, Constant::Number(-3.5), -1, -6),
        ] {
            let chunk = Chunk::new(
                5,
                vec![
                    Constant::Integer(start),
                    limit,
                    Constant::Integer(step),
                    Constant::Integer(0),
                ],
                vec![
                    Instruction::load_constant(0, 0),
                    Instruction::load_constant(1, 1),
                    Instruction::load_constant(2, 2),
                    Instruction::load_constant(4, 3),
                    Instruction::numeric_for_prep(0, 1),
                    Instruction::binary(OpCode::Add, 4, 4, 3),
                    Instruction::numeric_for_loop(0, -2),
                    Instruction::return_(4),
                ],
            );
            let (values, _) = super::fast_coverage::run_both(profile, &chunk, vec![]);
            assert!(
                matches!(values.as_slice(), [Value::Integer(n)] if *n == expected),
                "{profile:?}: {values:?}"
            );
        }
    }
}

#[test]
fn integer_numeric_for_clamps_out_of_range_limits_and_skips_empty_ranges() {
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        for (control, limit, step, expected_limit, enters) in [
            (i64::MAX, f64::INFINITY, 1, i64::MAX, true),
            (i64::MIN, f64::NEG_INFINITY, -1, i64::MIN, true),
            (i64::MIN, f64::INFINITY, -1, i64::MAX, false),
            (i64::MAX, f64::NEG_INFINITY, 1, i64::MIN, false),
            (1, f64::NAN, 1, i64::MIN, false),
        ] {
            let mut registers = vec![
                Value::Integer(control),
                Value::Number(limit),
                Value::Integer(step),
                Value::Nil,
            ];
            let mut pc = 10;
            numeric_for_instruction(
                &mut registers,
                &mut pc,
                Instruction::numeric_for_prep(0, 2),
                OpCode::NumericForPrep,
                profile,
            )
            .unwrap();
            if enters {
                assert_eq!(registers[1], Value::Integer(expected_limit));
                assert_eq!(registers[3], Value::Integer(control));
                assert_eq!(pc, 10);
            } else {
                assert_eq!(pc, 13);
            }
        }
    }
}

#[test]
fn integer_numeric_for_stops_at_integer_bounds() {
    let instruction = Instruction::numeric_for_loop(0, 0);
    for (control, limit, step) in [(i64::MAX - 1, i64::MAX, 1), (i64::MIN + 1, i64::MIN, -1)] {
        let mut registers = vec![
            Value::Integer(control),
            Value::Integer(limit),
            Value::Integer(step),
            Value::Nil,
        ];
        let mut pc = 17;
        numeric_for_instruction(
            &mut registers,
            &mut pc,
            instruction,
            OpCode::NumericForLoop,
            CompatibilityProfile::Lua54,
        )
        .unwrap();
        assert_eq!(registers[0], Value::Integer(limit));
        pc = 17;
        numeric_for_instruction(
            &mut registers,
            &mut pc,
            instruction,
            OpCode::NumericForLoop,
            CompatibilityProfile::Lua54,
        )
        .unwrap();
        assert_eq!(registers[0], Value::Integer(limit));
        assert_eq!(pc, 17);
    }
}

#[test]
fn integer_numeric_for_enters_and_exits_at_integer_bounds() {
    for (start, limit, step) in [(i64::MAX, i64::MAX, 1), (i64::MIN, i64::MIN, -1)] {
        let chunk = Chunk::with_parts(
            5,
            0,
            vec![
                Constant::Integer(start),
                Constant::Integer(limit),
                Constant::Integer(step),
            ],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::load_constant(2, 2),
                Instruction::numeric_for_prep(0, 2),
                Instruction::move_(4, 3),
                Instruction::move_(4, 3),
                Instruction::numeric_for_loop(0, -3),
                Instruction::return_(4),
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            Vm::with_profile(CompatibilityProfile::Lua54)
                .execute(&chunk)
                .unwrap(),
            vec![Value::Integer(start)]
        );
    }

    let empty = Chunk::with_parts(
        5,
        0,
        vec![
            Constant::Integer(2),
            Constant::Integer(1),
            Constant::Integer(1),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::numeric_for_prep(0, 2),
            Instruction::move_(4, 3),
            Instruction::move_(4, 3),
            Instruction::numeric_for_loop(0, -3),
            Instruction::return_(4),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(
        Vm::with_profile(CompatibilityProfile::Lua54)
            .execute(&empty)
            .unwrap(),
        vec![Value::Nil]
    );
}
