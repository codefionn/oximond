use super::*;

#[test]
fn semantic_digest_uses_standard_sha256() {
    assert_eq!(
        sha256(b"abc"),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );
}

#[test]
fn semantic_digest_ignores_source_metadata_but_changes_with_code() {
    let first = numeric_chunk().with_debug_metadata(DebugMetadata::new("first.lua", Vec::new()));
    let renamed = numeric_chunk()
        .with_debug_metadata(DebugMetadata::new("elsewhere/renamed.lua", Vec::new()));
    assert_eq!(ChunkDigest::of(&first), ChunkDigest::of(&renamed));
    let changed = Chunk::new(4, vec![Constant::Number(3.5)], first.code.clone());
    assert_ne!(ChunkDigest::of(&first), ChunkDigest::of(&changed));
}

#[test]
fn validates_and_evaluates_a_capture_free_numeric_inline_plan() {
    let callee = Chunk::with_parts(
        3,
        1,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let plan = validate_inline_call_plan(&callee).expect("numeric callee is inlineable");
    // A count hook sees the load, the add and the return; the `Halt` after
    // the return never runs.
    assert_eq!(plan.counted_instructions(), Some(3));

    assert_eq!(plan.register_count(), 3);
    assert_eq!(plan.result_register(), 2);
    assert_eq!(plan.evaluate(41.0).to_bits(), 42.0_f64.to_bits());
    assert_eq!(
        plan.operations(),
        &[
            InlineNumericOperation::LoadNumber {
                destination: 1,
                value: 1.0,
            },
            InlineNumericOperation::Add {
                destination: 2,
                left: 0,
                right: 1,
            },
        ]
    );
}

#[test]
fn inline_plan_accepts_fixed_arity_and_fixed_vararg_loads() {
    let callee = Chunk::with_varargs(
        4,
        2,
        Vec::new(),
        vec![
            Instruction::vararg(3, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::binary(OpCode::Add, 2, 2, 3),
            Instruction::return_(2),
        ],
        Vec::new(),
        Vec::new(),
        None,
    );
    let plan = validate_inline_call_plan(&callee).unwrap();
    assert_eq!(plan.parameter_count(), 2);
    assert_eq!(plan.minimum_argument_count(), 3);
    assert_eq!(
        plan.evaluate_arguments_with_upvalues(&[1.0, 2.0, 4.0], &mut [])
            .to_bits(),
        7.0_f64.to_bits()
    );
}

#[test]
fn validates_and_evaluates_a_single_captured_numeric_cell() {
    let callee = Chunk::with_parts(
        3,
        1,
        Vec::new(),
        vec![
            Instruction::get_upvalue(1, 0),
            Instruction::binary(OpCode::Add, 2, 1, 0),
            Instruction::set_upvalue(2, 0),
            Instruction::clear_range(1, 2),
            Instruction::get_upvalue(1, 0),
            Instruction::return_(1),
        ],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "value".into(),
            source: UpvalueSource::Detached,
        }],
    );
    let plan = validate_inline_call_plan(&callee).expect("numeric counter is inlineable");
    let mut upvalue = [40.0];

    assert_eq!(plan.upvalue_count(), 1);
    assert!(plan.writes_upvalue());
    assert_eq!(
        plan.evaluate_with_upvalues(2.0, &mut upvalue).to_bits(),
        42.0_f64.to_bits()
    );
    assert_eq!(upvalue[0].to_bits(), 42.0_f64.to_bits());
}

#[test]
fn nested_captured_plan_uses_the_after_validation_entry_point() {
    let nested = Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::get_upvalue(1, 0),
            Instruction::binary(OpCode::Add, 1, 1, 0),
            Instruction::set_upvalue(1, 0),
            Instruction::return_(1),
        ],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "value".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );

    assert_eq!(
        validate_inline_call_plan(&nested),
        Err(InlinePlanError::InvalidChunk)
    );
    assert!(validate_inline_call_plan_after_validation(&nested).is_ok());
}

#[test]
fn composes_an_exact_vararg_forwarding_wrapper_with_a_capture_free_target() {
    let target = Chunk::with_parts(
        3,
        1,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let target_plan = validate_inline_call_plan(&target).unwrap();
    let wrapper = Chunk::with_varargs(
        2,
        0,
        Vec::new(),
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::move_(1, 0),
            Instruction::tail_call_vararg(1, 0),
            Instruction::return_open(1, 0),
            Instruction::halt(),
        ],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "target".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
        None,
    );
    let root = Chunk::with_parts(
        1,
        0,
        Vec::new(),
        vec![Instruction::halt()],
        vec![wrapper.clone()],
        Vec::new(),
    );
    root.validate().expect("the enclosing chunk tree is valid");
    assert!(validate_inline_call_plan(&wrapper).is_err());

    let plan = compose_vararg_forwarding_inline_call_plan_after_validation(&wrapper, &target_plan)
        .expect("the wrapper only forwards its varargs to the captured target");

    // The wrapper's own instructions run too, so the composed plan does not
    // know how many a count hook sees; everything else is the target's.
    assert_eq!(plan.counted_instructions(), None);
    assert_eq!(plan.operations(), target_plan.operations());
    assert_eq!(plan.result_register(), target_plan.result_register());
    assert_eq!(plan.callee_digest(), target_plan.callee_digest());
    assert_eq!(plan.parameter_count(), 1);
    assert_eq!(plan.evaluate(41.0).to_bits(), 42.0_f64.to_bits());
}

#[test]
fn vararg_forwarding_composer_rejects_nontransparent_moves_and_wrong_registers() {
    let target = validate_inline_call_plan(&Chunk::with_parts(
        1,
        1,
        Vec::new(),
        vec![Instruction::return_(0), Instruction::halt()],
        Vec::new(),
        Vec::new(),
    ))
    .unwrap();
    let wrapper = Chunk::with_varargs(
        3,
        0,
        Vec::new(),
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::move_(1, 0),
            Instruction::tail_call_vararg(1, 0),
            Instruction::return_open(1, 0),
            Instruction::halt(),
        ],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "target".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
        None,
    );

    let mut nontransparent = wrapper.clone();
    nontransparent.code[1] = Instruction::move_(1, 2);
    assert!(
        compose_vararg_forwarding_inline_call_plan_after_validation(&nontransparent, &target,)
            .is_none()
    );

    let mut wrong_tail_base = wrapper.clone();
    wrong_tail_base.code[2] = Instruction::tail_call_vararg(0, 0);
    assert!(
        compose_vararg_forwarding_inline_call_plan_after_validation(&wrong_tail_base, &target,)
            .is_none()
    );

    let mut wrong_return_base = wrapper;
    wrong_return_base.code[3] = Instruction::return_open(0, 0);
    assert!(
        compose_vararg_forwarding_inline_call_plan_after_validation(&wrong_return_base, &target,)
            .is_none()
    );
}

#[test]
fn vararg_forwarding_composer_rejects_fixed_arguments() {
    let target = validate_inline_call_plan(&Chunk::with_parts(
        1,
        1,
        Vec::new(),
        vec![Instruction::return_(0), Instruction::halt()],
        Vec::new(),
        Vec::new(),
    ))
    .unwrap();
    let wrapper = Chunk::with_varargs(
        2,
        0,
        Vec::new(),
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::tail_call_vararg(0, 1),
            Instruction::return_open(0, 0),
            Instruction::halt(),
        ],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "target".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
        None,
    );

    assert!(
        compose_vararg_forwarding_inline_call_plan_after_validation(&wrapper, &target).is_none()
    );
}

#[test]
fn inline_plan_rejects_lua_observable_or_uninitialized_bodies() {
    let captured = Chunk::with_parts(
        1,
        1,
        Vec::new(),
        vec![Instruction::return_(0)],
        Vec::new(),
        vec![crate::bytecode::UpvalueDescriptor {
            name: "state".into(),
            source: UpvalueSource::Detached,
        }],
    );
    assert_eq!(
        validate_inline_call_plan(&captured),
        Err(InlinePlanError::Captures)
    );

    let vararg = Chunk::with_varargs(
        1,
        1,
        Vec::new(),
        vec![Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
        None,
    );
    assert!(validate_inline_call_plan(&vararg).is_ok());

    let global = Chunk::with_parts(
        2,
        1,
        vec![Constant::String("outside".into())],
        vec![Instruction::get_global(1, 0), Instruction::return_(1)],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(
        validate_inline_call_plan(&global),
        Err(InlinePlanError::UnsupportedOpcode {
            pc: 0,
            opcode: OpCode::GetGlobal,
        })
    );

    let uninitialized = Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::return_(0),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(
        validate_inline_call_plan(&uninitialized),
        Err(InlinePlanError::UninitializedRegister { pc: 0, register: 1 })
    );
}

/// `function(x) if x % 2 == 0 then return x + 1 else return x + 2 end end`
fn parity_bump() -> Chunk {
    Chunk::with_parts(
        9,
        1,
        vec![
            Constant::Number(2.0),
            Constant::Number(0.0),
            Constant::Number(1.0),
        ],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::load_constant(3, 1),
            Instruction::binary(OpCode::Equal, 4, 2, 3),
            Instruction::jump_if_false(4, 4),
            Instruction::load_constant(5, 2),
            Instruction::binary(OpCode::Add, 6, 0, 5),
            Instruction::return_(6),
            Instruction::aj(OpCode::Jump, 3).unwrap(),
            Instruction::load_constant(7, 0),
            Instruction::binary(OpCode::Add, 8, 0, 7),
            Instruction::return_(8),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    )
}

#[test]
fn if_converts_a_branching_numeric_callee() {
    let plan = validate_inline_call_plan(&parity_bump()).expect("branching callee is inlineable");

    assert_eq!(plan.evaluate(4.0).to_bits(), 5.0_f64.to_bits());
    assert_eq!(plan.evaluate(7.0).to_bits(), 9.0_f64.to_bits());
    assert_eq!(plan.evaluate(-3.0).to_bits(), (-1.0_f64).to_bits());
    assert!(plan.evaluate(f64::NAN).is_nan());
    assert!(
        plan.operations()
            .iter()
            .any(|operation| matches!(operation, InlineNumericOperation::Select { .. }))
    );
}

#[test]
fn if_conversion_merges_registers_assigned_on_both_arms() {
    // local y; if x < 0 then y = -x else y = x end; if not (y <= 10) then y = 10 end; return y
    let callee = Chunk::with_parts(
        4,
        1,
        vec![Constant::Number(0.0), Constant::Number(10.0)],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::binary(OpCode::LessThan, 3, 0, 2),
            Instruction::jump_if_false(3, 2),
            Instruction::unary(OpCode::Negate, 1, 0),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::move_(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::binary(OpCode::LessEqual, 3, 1, 2),
            Instruction::unary(OpCode::Not, 3, 3),
            Instruction::jump_if_false(3, 1),
            Instruction::move_(1, 2),
            Instruction::return_(1),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let plan = validate_inline_call_plan(&callee).expect("merging callee is inlineable");

    for (argument, expected) in [(-3.0, 3.0), (4.0, 4.0), (-20.0, 10.0), (12.0, 10.0)] {
        assert_eq!(plan.evaluate(argument).to_bits(), f64::to_bits(expected));
    }
}

#[test]
fn if_conversion_refuses_loops_missing_returns_and_condition_results() {
    let looping = Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::Add, 1, 0, 0),
            Instruction::binary(OpCode::LessThan, 1, 0, 1),
            Instruction::jump_if_false(1, -3),
            Instruction::return_(0),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert!(validate_inline_call_plan(&looping).is_err());

    let falls_off = Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::LessThan, 1, 0, 0),
            Instruction::jump_if_false(1, 1),
            Instruction::return_(0),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert!(validate_inline_call_plan(&falls_off).is_err());

    let returns_condition = Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::LessThan, 1, 0, 0),
            Instruction::jump_if_false(1, 1),
            Instruction::return_(0),
            Instruction::return_(1),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert!(validate_inline_call_plan(&returns_condition).is_err());
}

#[test]
fn plans_record_whether_the_callee_loads_an_integer_constant() {
    let callee = |constant| {
        Chunk::with_parts(
            2,
            1,
            vec![constant],
            vec![
                Instruction::load_constant(1, 0),
                Instruction::return_(1),
                Instruction::halt(),
            ],
            Vec::new(),
            Vec::new(),
        )
    };
    let integer = validate_inline_call_plan(&callee(Constant::Integer(9))).unwrap();
    assert!(integer.loads_integer_constant());
    let float = validate_inline_call_plan(&callee(Constant::Number(9.0))).unwrap();
    assert!(!float.loads_integer_constant());
}

/// `function(x) return x % 3.001, x ^ 2` style callee, one result per plan:
/// the plan evaluates `%` and `^` as the profile it was given does.
#[test]
fn plans_compute_float_modulo_and_power_as_the_profile_does() {
    use crate::compat::CompatibilityProfile;
    let callee = |opcode: OpCode, constant: f64| {
        Chunk::with_parts(
            3,
            1,
            vec![Constant::Number(constant)],
            vec![
                Instruction::load_constant(1, 0),
                Instruction::binary(opcode, 2, 0, 1),
                Instruction::return_(2),
                Instruction::halt(),
            ],
            Vec::new(),
            Vec::new(),
        )
    };
    let modulo = validate_inline_call_plan(&callee(OpCode::Modulo, 3.001)).unwrap();
    let power = validate_inline_call_plan(&callee(OpCode::Power, 2.0)).unwrap();
    let cube = 483.0_f64.powf(3.0);
    // The C library's pow, as stock Lua calls it: glibc rounds this tie of
    // 12_696_463_968_316_569 up to ..570, the Windows C runtime to ..568.
    let pow_squared = cube.powf(2.0);
    for (profile, floor_or_fmod, squared) in [
        // The default form is LuaJIT's.
        (
            CompatibilityProfile::LuaJit21,
            2.194_999_999_999_936_3,
            pow_squared,
        ),
        (
            CompatibilityProfile::Lua51,
            2.194_999_999_999_936_3,
            pow_squared,
        ),
        (
            CompatibilityProfile::Lua54,
            2.194_999_999_999_978_5,
            12_696_463_968_316_568.0,
        ),
        (
            CompatibilityProfile::Lua55,
            2.194_999_999_999_978_5,
            12_696_463_968_316_568.0,
        ),
    ] {
        let arith = profile.float_arith();
        let modulo = modulo.clone().with_float_arith(arith);
        let power = power.clone().with_float_arith(arith);
        assert_eq!(
            modulo.evaluate(-583.0).to_bits(),
            f64::to_bits(floor_or_fmod),
            "{profile:?}"
        );
        assert_eq!(
            power.evaluate(cube).to_bits(),
            f64::to_bits(squared),
            "{profile:?}"
        );
    }
}

#[test]
fn constant_arithmetic_inlines_as_a_scratch_load_and_the_register_form() {
    let callee = Chunk::with_parts(
        2,
        1,
        vec![Constant::Number(1.5), Constant::Integer(4)],
        vec![
            Instruction::abc(OpCode::AddK, 1, 0, 0),
            Instruction::abc(OpCode::MultiplyK, 1, 1, 1),
            Instruction::return_(1),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let plan = validate_inline_call_plan(&callee).expect("constant arithmetic is inlineable");
    assert_eq!(plan.register_count(), 3);
    assert!(plan.loads_integer_constant());
    assert_eq!(plan.evaluate(1.5).to_bits(), 12.0_f64.to_bits());
    assert_eq!(
        plan.operations()[..2],
        [
            InlineNumericOperation::LoadNumber {
                destination: 2,
                value: 1.5,
            },
            InlineNumericOperation::Add {
                destination: 1,
                left: 0,
                right: 2,
            },
        ]
    );
}

#[test]
fn constant_arithmetic_inlines_in_branching_callees() {
    let callee = Chunk::with_parts(
        3,
        1,
        vec![Constant::Number(3.0), Constant::Number(2.0)],
        vec![
            Instruction::abc(OpCode::ModuloK, 1, 0, 0),
            Instruction::load_constant(2, 1),
            Instruction::binary(OpCode::LessThan, 2, 1, 2),
            Instruction::jump_if_false(2, 1),
            Instruction::abc(OpCode::SubtractK, 1, 0, 0),
            Instruction::return_(1),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let plan = validate_inline_call_plan(&callee).expect("branching callee is inlineable");
    assert_eq!(plan.evaluate(5.0).to_bits(), 2.0_f64.to_bits());
    assert_eq!(plan.evaluate(4.0).to_bits(), 1.0_f64.to_bits());
}

/// `function(x, y) if <comparison> then return x else return y end end`,
/// written with a test instruction, and with the comparison opcode and a
/// `JumpIfFalse` (and a `Not` when the test jumps on a true result).
fn comparison_callees(
    comparison: OpCode,
    constant: Option<(f64, bool)>,
    expected: bool,
) -> (Chunk, Chunk) {
    let constants = vec![Constant::Number(constant.map_or(0.0, |(value, _)| value))];
    let register_form = |first: Instruction, extra: &[Instruction]| {
        let mut code = vec![first];
        code.extend_from_slice(extra);
        code
    };
    // Registers: x = 0, y = 1, the constant = 2, the comparison result = 3.
    let (test, reference_prefix) = match constant {
        None => (
            Instruction::test(
                match comparison {
                    OpCode::Equal => OpCode::TestEqual,
                    OpCode::LessThan => OpCode::TestLessThan,
                    _ => OpCode::TestLessEqual,
                },
                0,
                1,
                expected,
                false,
            ),
            vec![Instruction::binary(comparison, 3, 0, 1)],
        ),
        Some((_, constant_left)) => (
            Instruction::test(
                match comparison {
                    OpCode::Equal => OpCode::TestEqualK,
                    OpCode::LessThan => OpCode::TestLessThanK,
                    _ => OpCode::TestLessEqualK,
                },
                0,
                0,
                expected,
                constant_left,
            ),
            vec![
                Instruction::load_constant(2, 0),
                if constant_left {
                    Instruction::binary(comparison, 3, 2, 0)
                } else {
                    Instruction::binary(comparison, 3, 0, 2)
                },
            ],
        ),
    };
    let tail = [
        Instruction::return_(0),
        Instruction::return_(1),
        Instruction::halt(),
    ];
    let tested = Chunk::with_parts(
        4,
        2,
        constants.clone(),
        register_form(
            test,
            &[
                Instruction::aj(OpCode::Jump, 1).unwrap(),
                tail[0],
                tail[1],
                tail[2],
            ],
        ),
        Vec::new(),
        Vec::new(),
    );
    // The jump is taken when the result equals `expected`: `JumpIfFalse`
    // jumps on false, so a true `expected` first negates the result.
    let mut reference = reference_prefix;
    if expected {
        reference.push(Instruction::unary(OpCode::Not, 3, 3));
    }
    reference.push(Instruction::jump_if_false(3, 1));
    reference.extend_from_slice(&tail);
    let reference = Chunk::with_parts(4, 2, constants, reference, Vec::new(), Vec::new());
    (tested, reference)
}

#[test]
fn comparison_tests_if_convert_like_the_comparison_and_jump() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut checked = 0;
    for comparison in [OpCode::Equal, OpCode::LessThan, OpCode::LessEqual] {
        for expected in [false, true] {
            for constant in [
                None,
                Some((1.0, false)),
                Some((f64::NAN, false)),
                Some((-1.0, true)),
                Some((2.5, true)),
            ] {
                // Only the ordered constant tests take `constant_left`.
                if comparison == OpCode::Equal && matches!(constant, Some((_, true))) {
                    continue;
                }
                let (tested, reference) = comparison_callees(comparison, constant, expected);
                assert_eq!(tested.validate(), Ok(()), "{comparison:?} {constant:?}");
                let tested_plan = validate_inline_call_plan(&tested)
                    .unwrap_or_else(|error| panic!("{comparison:?} {constant:?}: {error:?}"));
                let reference_plan = validate_inline_call_plan(&reference).unwrap();
                for x in values {
                    for y in values {
                        let arguments = [x, y];
                        let got = tested_plan.evaluate_arguments_with_upvalues(&arguments, &mut []);
                        let want =
                            reference_plan.evaluate_arguments_with_upvalues(&arguments, &mut []);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{comparison:?} {constant:?} expected={expected} x={x} y={y}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
}

#[test]
fn comparison_tests_with_a_non_numeric_constant_are_refused() {
    let callee = Chunk::with_parts(
        2,
        1,
        vec![Constant::Nil],
        vec![
            Instruction::test(OpCode::TestEqualK, 0, 0, false, false),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::return_(0),
            Instruction::return_(0),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(callee.validate(), Ok(()));
    assert!(validate_inline_call_plan(&callee).is_err());
}

#[test]
fn generic_for_loops_are_refused_by_every_inline_plan() {
    // A function that iterates with `ForInCall`/`ForInLoop` calls its iterator
    // and loops back, which none of the straight-line or if-converted plans
    // model. The backward `ForInLoop` and the call are refused, not misread.
    let iterating = Chunk::with_parts(
        8,
        1,
        Vec::new(),
        vec![
            Instruction::move_(1, 0),
            Instruction::load_nil(2),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::binary(OpCode::Add, 7, 7, 4),
            Instruction::for_in_call(0, 1),
            Instruction::for_in_loop(0, -3),
            Instruction::return_(7),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    iterating.validate().unwrap();
    assert!(validate_inline_call_plan(&iterating).is_err());
    for compatibility in [
        crate::compat::CompatibilityProfile::Lua51,
        crate::compat::CompatibilityProfile::Lua54,
    ] {
        assert!(validate_inline_table_call_plan(&iterating, compatibility).is_none());
    }
}
