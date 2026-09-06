//! Arithmetic, comparison and MOVE on the fast loop. Every case runs on the
//! fast loop and on the general loop (see `fast_coverage::run_both`), which
//! must agree; the expected values pin the result itself.

use super::fast_coverage::{PROFILES, run_both};
use super::trace_record::interpreter_vm_with;
use super::*;

fn integer(value: i64) -> Constant {
    Constant::Integer(value)
}

fn float(value: f64) -> Constant {
    Constant::Number(value)
}

fn text(value: &str) -> Constant {
    Constant::String(LuaString::from(value))
}

/// Loads `constants` into registers `0..n`, applies `opcode` to every
/// ordered pair `(i, j)` and returns all results.
fn pairwise(opcode: OpCode, constants: Vec<Constant>) -> Chunk {
    let count = constants.len() as u8;
    let mut code = Vec::new();
    for index in 0..count {
        code.push(Instruction::load_constant(index, u16::from(index)));
    }
    let mut destination = count;
    for left in 0..count {
        for right in 0..count {
            code.push(Instruction::binary(opcode, destination, left, right));
            destination += 1;
        }
    }
    code.push(Instruction::return_values(count, destination - count));
    let chunk = Chunk::with_parts(destination, 0, constants, code, vec![], vec![]);
    chunk.validate().unwrap();
    chunk
}

fn run_result(
    profile: CompatibilityProfile,
    chunk: &Chunk,
    general: bool,
) -> Result<Vec<Value>, String> {
    let mut vm = interpreter_vm_with(profile);
    vm.debug_hooks.clear();
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm.set_interpreter_profiling(general);
    vm.execute_with_arguments(chunk, Vec::new())
        .map_err(|error| error.to_string())
}

#[test]
fn mixed_integer_and_float_arithmetic_matches_the_general_loop() {
    let constants = vec![integer(7), float(2.5), integer(3), float(0.5)];
    for opcode in [
        OpCode::Add,
        OpCode::Subtract,
        OpCode::Multiply,
        OpCode::Divide,
        OpCode::Modulo,
    ] {
        let chunk = pairwise(opcode, constants.clone());
        for profile in PROFILES {
            let (_, fallbacks) = run_both(profile, &chunk, Vec::new());
            assert!(!fallbacks.contains(&opcode), "{opcode:?} {profile:?}");
        }
    }
}

#[test]
fn arithmetic_results_keep_their_numeric_subtype() {
    // 7 + 3, 7 + 2.5, 7 / 3, 7 % 3 and 7.5 % 2
    let chunk = Chunk::with_parts(
        8,
        0,
        vec![integer(7), integer(3), float(2.5), float(7.5), integer(2)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::load_constant(4, 4),
            Instruction::binary(OpCode::Add, 5, 0, 1),
            Instruction::binary(OpCode::Add, 6, 0, 2),
            Instruction::binary(OpCode::Divide, 7, 0, 1),
            Instruction::binary(OpCode::Modulo, 0, 0, 1),
            Instruction::binary(OpCode::Modulo, 1, 3, 4),
            Instruction::return_values(5, 3),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    let (lua54, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
    assert_eq!(
        lua54,
        vec![
            Value::Integer(10),
            Value::Number(9.5),
            Value::Number(7.0 / 3.0)
        ]
    );
    let (lua51, _) = run_both(CompatibilityProfile::Lua51, &chunk, Vec::new());
    assert_eq!(
        lua51,
        vec![
            Value::Number(10.0),
            Value::Number(9.5),
            Value::Number(7.0 / 3.0)
        ]
    );
}

#[test]
fn integer_overflow_wraps_in_lua53_and_later() {
    let chunk = Chunk::with_parts(
        6,
        0,
        vec![integer(i64::MAX), integer(i64::MIN), integer(1), integer(2)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::binary(OpCode::Add, 4, 0, 2),
            Instruction::binary(OpCode::Subtract, 5, 1, 2),
            Instruction::binary(OpCode::Multiply, 0, 0, 3),
            Instruction::return_values(4, 2),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in [CompatibilityProfile::Lua53, CompatibilityProfile::Lua54] {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![Value::Integer(i64::MIN), Value::Integer(i64::MAX)],
            "{profile:?}"
        );
        assert!(!fallbacks.contains(&OpCode::Add), "{profile:?}");
    }
    // Multiplication wraps too.
    let product = Chunk::with_parts(
        3,
        0,
        vec![integer(i64::MAX), integer(2)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Multiply, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let (results, _) = run_both(CompatibilityProfile::Lua54, &product, Vec::new());
    assert_eq!(results, vec![Value::Integer(-2)]);
}

#[test]
fn division_and_modulo_by_zero() {
    // Floats: x / 0 is infinite, x % 0 is NaN in every profile.
    let chunk = Chunk::with_parts(
        5,
        0,
        vec![float(1.0), float(0.0), float(-1.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::binary(OpCode::Divide, 3, 0, 1),
            Instruction::binary(OpCode::Divide, 4, 2, 1),
            Instruction::binary(OpCode::Modulo, 0, 0, 1),
            Instruction::return_values(3, 2),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, _) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![
                Value::Number(f64::INFINITY),
                Value::Number(f64::NEG_INFINITY)
            ],
            "{profile:?}"
        );
    }
    let nan = Chunk::with_parts(
        3,
        0,
        vec![float(5.0), float(0.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    for profile in PROFILES {
        // NaN is unequal to itself, so the loops are compared by their debug output.
        let fast = run_result(profile, &nan, false).unwrap();
        let general = run_result(profile, &nan, true).unwrap();
        assert!(
            matches!(fast[..], [Value::Number(value)] if value.is_nan()),
            "{profile:?}"
        );
        assert_eq!(format!("{fast:?}"), format!("{general:?}"), "{profile:?}");
    }

    // Integers (Lua 5.3+): `/` is a float division, `%` by zero is an error.
    let ints = |opcode| {
        Chunk::with_parts(
            3,
            0,
            vec![integer(5), integer(0)],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::binary(opcode, 2, 0, 1),
                Instruction::return_(2),
            ],
            vec![],
            vec![],
        )
    };
    let (results, _) = run_both(
        CompatibilityProfile::Lua54,
        &ints(OpCode::Divide),
        Vec::new(),
    );
    assert_eq!(results, vec![Value::Number(f64::INFINITY)]);
    let chunk = ints(OpCode::Modulo);
    let fast = run_result(CompatibilityProfile::Lua54, &chunk, false);
    let general = run_result(CompatibilityProfile::Lua54, &chunk, true);
    assert!(fast.is_err(), "{fast:?}");
    assert_eq!(fast, general);
    assert!(fast.unwrap_err().contains("'n%0'"));
}

#[test]
// exact comparison is required here (NaN semantics)
#[allow(clippy::float_cmp)]
fn lua53_numeric_string_arithmetic_produces_floats() {
    for (opcode, expected) in [
        (OpCode::Add, 11.0),
        (OpCode::Subtract, 9.0),
        (OpCode::Multiply, 10.0),
        (OpCode::FloorDivide, 10.0),
        (OpCode::Modulo, 0.0),
        (OpCode::Negate, -10.0),
    ] {
        let chunk = Chunk::with_parts(
            3,
            0,
            vec![text("10"), integer(1)],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::binary(opcode, 2, 0, 1),
                Instruction::return_(2),
            ],
            vec![],
            vec![],
        );
        let (results, _) = run_both(CompatibilityProfile::Lua53, &chunk, Vec::new());
        assert!(
            matches!(results.as_slice(), [Value::Number(n)] if *n == expected),
            "{opcode:?}: {results:?}"
        );
        let mut vm = Vm::with_profile_and_execution_mode(
            CompatibilityProfile::Lua53,
            ExecutionMode::Tracing,
        );
        for _ in 0..100 {
            let results = vm.execute(&chunk).unwrap();
            assert!(
                matches!(results.as_slice(), [Value::Number(n)] if *n == expected),
                "tracing {opcode:?}: {results:?}"
            );
        }
    }
}

#[test]
// exact comparison is required here (NaN semantics)
#[allow(clippy::float_cmp)]
fn lua53_string_arithmetic_rounds_integers_and_wraps_hex_before_float_conversion() {
    let chunk = pairwise(
        OpCode::Add,
        vec![
            integer(1),
            text("9007199254740993"),
            text("0xffffffffffffffff"),
        ],
    );
    let (results, _) = run_both(CompatibilityProfile::Lua53, &chunk, Vec::new());
    assert!(matches!(results[0], Value::Integer(2)));
    assert!(matches!(results[1], Value::Number(n) if n == 9_007_199_254_740_992.0));
    assert!(matches!(results[2], Value::Number(n) if n == 0.0));
    assert!(matches!(results[3], Value::Number(n) if n == 9_007_199_254_740_992.0));
    assert!(matches!(results[6], Value::Number(n) if n == 0.0));
    let chunk = pairwise(OpCode::BitAnd, vec![text("10"), integer(3)]);
    let (results, _) = run_both(CompatibilityProfile::Lua53, &chunk, Vec::new());
    assert!(matches!(results[1], Value::Integer(2)));
}

#[test]
fn strings_coerce_to_numbers_in_arithmetic() {
    for (opcode, expected54, expected51) in [
        (OpCode::Add, Value::Integer(15), Value::Number(15.0)),
        (OpCode::Multiply, Value::Integer(50), Value::Number(50.0)),
        (OpCode::Subtract, Value::Integer(5), Value::Number(5.0)),
    ] {
        let chunk = Chunk::with_parts(
            3,
            0,
            vec![text("10"), integer(5)],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::binary(opcode, 2, 0, 1),
                Instruction::return_(2),
            ],
            vec![],
            vec![],
        );
        let (results, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
        assert_eq!(results, vec![expected54], "{opcode:?}");
        let (results, _) = run_both(CompatibilityProfile::Lua51, &chunk, Vec::new());
        assert_eq!(results, vec![expected51], "{opcode:?}");
    }
    // A string on the right, a hexadecimal string, and a float string.
    let chunk = Chunk::with_parts(
        5,
        0,
        vec![float(1.5), text("0x10"), text("2.5")],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::binary(OpCode::Add, 3, 0, 1),
            Instruction::binary(OpCode::Multiply, 4, 2, 0),
            Instruction::return_values(3, 2),
        ],
        vec![],
        vec![],
    );
    for profile in PROFILES {
        let (results, _) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![Value::Number(17.5), Value::Number(3.75)],
            "{profile:?}"
        );
    }
    // A string that is not a number is an error, on both loops.
    let bad = Chunk::with_parts(
        3,
        0,
        vec![text("abc"), integer(1)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let fast = run_result(CompatibilityProfile::Lua54, &bad, false);
    assert!(fast.is_err());
    assert_eq!(fast, run_result(CompatibilityProfile::Lua54, &bad, true));
}

#[test]
fn comparisons_of_mixed_numbers_and_strings() {
    for opcode in [OpCode::Equal, OpCode::LessThan, OpCode::LessEqual] {
        let chunk = pairwise(
            opcode,
            vec![integer(3), float(3.0), float(2.5), integer(-4)],
        );
        for profile in PROFILES {
            run_both(profile, &chunk, Vec::new());
        }
    }
    let strings = pairwise(OpCode::LessThan, vec![text("a"), text("b"), text("a")]);
    for profile in PROFILES {
        let (results, _) = run_both(profile, &strings, Vec::new());
        assert_eq!(results[1], Value::Boolean(true));
        assert_eq!(results[3], Value::Boolean(false));
    }
}

#[test]
fn add_falls_back_to_the_metamethod() {
    // setmetatable({}, {__add = function() return 42 end}) + 1, and 1 + t.
    let handler = Chunk::with_parts(
        3,
        2,
        vec![float(42.0)],
        vec![Instruction::load_constant(2, 0), Instruction::return_(2)],
        vec![],
        vec![],
    );
    let chunk = Chunk::with_parts(
        12,
        0,
        vec![text("setmetatable"), text("__add"), float(1.0)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::new_table(1, 0, 0),
            Instruction::new_table(2, 0, 0),
            Instruction::load_constant(3, 1),
            Instruction::closure(4, 0),
            Instruction::set_table(2, 3, 4),
            Instruction::move_(5, 0),
            Instruction::move_(6, 1),
            Instruction::move_(7, 2),
            Instruction::call(5, 2, 1),
            Instruction::load_constant(8, 2),
            Instruction::binary(OpCode::Add, 9, 1, 8),
            Instruction::binary(OpCode::Add, 10, 8, 1),
            Instruction::return_values(9, 2),
        ],
        vec![handler],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, _) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![Value::Number(42.0), Value::Number(42.0)],
            "{profile:?}"
        );
    }
}

#[test]
fn move_of_strings_over_strings_stays_on_the_fast_loop() {
    let chunk = Chunk::with_parts(
        4,
        0,
        vec![text("first"), text("second"), float(1.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(3, 2),
            Instruction::move_(1, 0), // string over string
            Instruction::move_(2, 1), // string over nil
            Instruction::move_(2, 2), // string over itself
            Instruction::move_(0, 3), // number over string
            Instruction::move_(3, 3), // number over itself
            Instruction::return_values(0, 4),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![
                Value::Number(1.0),
                Value::from("first"),
                Value::from("first"),
                Value::Number(1.0)
            ],
            "{profile:?}"
        );
        assert!(!fallbacks.contains(&OpCode::Move), "{profile:?}");
    }
}

#[test]
fn every_opcode_is_in_the_dispatch_list() {
    // The fast loop's dispatch has no wildcard fallback: an opcode missing
    // from its list would be undefined behavior.
    let mut listed = crate::vm::fast_path::dispatched_opcodes().to_vec();
    listed.sort_unstable();
    let all: Vec<u8> = (0..=u8::MAX)
        .filter(|byte| OpCode::try_from(*byte).is_ok())
        .collect();
    assert_eq!(listed, all);
}

/// Floor modulo computed in i128, independent of the implementation.
fn reference_floor_mod(left: i64, right: i64) -> i64 {
    let (left, right) = (i128::from(left), i128::from(right));
    (((left % right) + right) % right) as i64
}

#[test]
fn integer_modulo_agrees_on_both_loops_in_lua54() {
    // Both signs, +-1, MIN % -1 and the extremes; no zero divisor. `fallbacks`
    // only sees opcodes handed back to dispatch, so it cannot tell the fast arm
    // from `arithmetic_slow`; the op-stats CLI test
    // `integer_modulo_stays_on_the_fast_arm` checks that in its own process.
    let values = [
        0,
        1,
        -1,
        2,
        -2,
        5,
        -5,
        7,
        -7,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        i64::MIN + 1,
    ];
    let constants: Vec<Constant> = values.iter().map(|value| integer(*value)).collect();
    let divisors: Vec<usize> = (0..values.len())
        .filter(|index| values[*index] != 0)
        .collect();
    // Built by hand because `pairwise` would divide by zero.
    let count = values.len() as u8;
    let mut code = Vec::new();
    for index in 0..count {
        code.push(Instruction::load_constant(index, u16::from(index)));
    }
    let mut destination = count;
    let mut expected = Vec::new();
    for left in 0..values.len() {
        for right in &divisors {
            code.push(Instruction::binary(
                OpCode::Modulo,
                destination,
                left as u8,
                *right as u8,
            ));
            destination += 1;
            expected.push(Value::Integer(reference_floor_mod(
                values[left],
                values[*right],
            )));
        }
    }
    code.push(Instruction::return_values(count, destination - count));
    let chunk = Chunk::with_parts(destination, 0, constants, code, vec![], vec![]);
    chunk.validate().unwrap();
    let (results, fallbacks) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
    assert_eq!(results, expected);
    assert!(!fallbacks.contains(&OpCode::Modulo), "{fallbacks:?}");
    assert!(results.contains(&Value::Integer(0)));
}

#[test]
fn integer_modulo_known_values() {
    for (left, right, expected) in [
        (7, 3, 1),
        (-7, 3, 2),
        (7, -3, -2),
        (-7, -3, -1),
        (6, 3, 0),
        (-6, 3, 0),
        (5, 1, 0),
        (5, -1, 0),
        (i64::MIN, -1, 0),
        (i64::MIN, 1, 0),
        (i64::MAX, -1, 0),
        (i64::MIN, i64::MAX, i64::MAX - 1),
        (i64::MAX, i64::MIN, -1),
        (i64::MIN, i64::MIN, 0),
    ] {
        assert_eq!(
            reference_floor_mod(left, right),
            expected,
            "{left} % {right}"
        );
        let chunk = Chunk::with_parts(
            3,
            0,
            vec![integer(left), integer(right)],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::binary(OpCode::Modulo, 2, 0, 1),
                Instruction::return_(2),
            ],
            vec![],
            vec![],
        );
        let (results, fallbacks) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
        assert_eq!(results, vec![Value::Integer(expected)], "{left} % {right}");
        assert!(!fallbacks.contains(&OpCode::Modulo), "{left} % {right}");
    }
}

#[test]
fn integer_modulo_by_zero_keeps_its_error() {
    let chunk = Chunk::with_parts(
        3,
        0,
        vec![integer(i64::MIN), integer(0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let fast = run_result(CompatibilityProfile::Lua54, &chunk, false).unwrap_err();
    let general = run_result(CompatibilityProfile::Lua54, &chunk, true).unwrap_err();
    assert_eq!(fast, general);
    assert!(fast.contains("attempt to perform 'n%0'"), "{fast}");
}

#[test]
fn modulo_with_floats_and_mixed_operands() {
    // 7.5 % 2 (float, integer), 7 % 2.5 (integer, float), -7 % 2.5, 7 % -2.5.
    let chunk = Chunk::with_parts(
        10,
        0,
        vec![
            float(7.5),
            integer(2),
            integer(7),
            float(2.5),
            integer(-7),
            float(-2.5),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::load_constant(4, 4),
            Instruction::load_constant(5, 5),
            Instruction::binary(OpCode::Modulo, 6, 0, 1),
            Instruction::binary(OpCode::Modulo, 7, 2, 3),
            Instruction::binary(OpCode::Modulo, 8, 4, 3),
            Instruction::binary(OpCode::Modulo, 9, 2, 5),
            Instruction::return_values(6, 4),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, _) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![
                Value::Number(1.5),
                Value::Number(2.0),
                Value::Number(0.5),
                Value::Number(-0.5)
            ],
            "{profile:?}"
        );
    }
}

#[test]
fn integer_looking_modulo_is_float_in_lua51() {
    let chunk = Chunk::with_parts(
        3,
        0,
        vec![integer(-7), integer(3)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let (lua51, _) = run_both(CompatibilityProfile::Lua51, &chunk, Vec::new());
    assert_eq!(lua51, vec![Value::Number(2.0)]);
    let (lua54, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
    assert_eq!(lua54, vec![Value::Integer(2)]);
}

/// Float bits of every result, on the fast loop and on the general loop,
/// which must agree bit for bit (`run_both` cannot compare NaN).
fn bits_on_both_loops(profile: CompatibilityProfile, chunk: &Chunk) -> Vec<u64> {
    let bits = |general: bool| -> Vec<u64> {
        run_result(profile, chunk, general)
            .unwrap()
            .iter()
            .map(|value| match value {
                Value::Number(number) => number.to_bits(),
                other => panic!("expected a float, got {other:?}"),
            })
            .collect()
    };
    let fast = bits(false);
    assert_eq!(fast, bits(true), "{profile:?}: fast and general loops");
    fast
}

/// Float `%` and `^` on the fast loop and the general loop give what each
/// stock interpreter prints (values from lua5.1, lua5.4, lua5.5 and luajit),
/// including infinities, NaN and signed zero.
#[test]
fn float_modulo_and_power_follow_each_profile() {
    let cube = 483.0_f64.powf(3.0);
    let quotient = 4.0 / -2.25;
    let inputs = [
        -583.0,
        3.001,
        1e10,
        quotient,
        f64::INFINITY,
        f64::NEG_INFINITY,
        5.0,
        -0.0,
        cube,
        2.0,
        f64::NAN,
    ];
    let constants: Vec<Constant> = inputs.iter().map(|value| float(*value)).collect();
    let index = |value: f64| {
        inputs
            .iter()
            .position(|input| input.to_bits() == value.to_bits())
            .unwrap()
    };
    let cell = |left: f64, right: f64| index(left) * inputs.len() + index(right);
    for profile in CompatibilityProfile::ALL {
        let modulo = bits_on_both_loops(profile, &pairwise(OpCode::Modulo, constants.clone()));
        let power = bits_on_both_loops(profile, &pairwise(OpCode::Power, constants.clone()));
        let fmod = profile.is_lua53_or_later();
        let expect = |left: f64, right: f64, unfused_value: f64, fmod_value: f64| {
            let want = if fmod { fmod_value } else { unfused_value };
            let got = f64::from_bits(modulo[cell(left, right)]);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{profile:?}: {left:?} % {right:?} = {got:?}, stock {want:?}"
            );
        };
        let nan = f64::NAN;
        expect(
            -583.0,
            3.001,
            2.194_999_999_999_936_3,
            2.194_999_999_999_978_5,
        );
        expect(1e10, quotient, 0.0, -1.777_777_222_666_265_4);
        expect(5.0, f64::INFINITY, nan, 5.0);
        expect(5.0, f64::NEG_INFINITY, nan, f64::NEG_INFINITY);
        expect(f64::INFINITY, 5.0, nan, nan);
        expect(-0.0, 5.0, 0.0, -0.0);
        expect(5.0, 5.0, 0.0, 0.0);
        expect(5.0, nan, nan, nan);
        // `x ^ 2`: Lua 5.4 and 5.5 multiply, the others call pow.
        let squared = f64::from_bits(power[cell(cube, 2.0)]);
        // Without the shortcut, the C library's pow: glibc rounds this tie of
        // 12_696_463_968_316_569 up to ..570, the Windows C runtime to ..568.
        let want: f64 = if profile.float_arith().square_shortcut {
            12_696_463_968_316_568.0
        } else {
            cube.powf(2.0)
        };
        assert_eq!(squared.to_bits(), want.to_bits(), "{profile:?}: x ^ 2");
        assert_eq!(
            f64::from_bits(power[cell(2.0, 2.0)]),
            4.0,
            "{profile:?}: 2 ^ 2"
        );
    }
}

/// An integer operand converts to float before the float form runs.
#[test]
fn mixed_integer_and_float_modulo_uses_the_float_form() {
    let chunk = Chunk::with_parts(
        4,
        0,
        vec![integer(-583), float(3.001)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Modulo, 2, 0, 1),
            Instruction::return_values(2, 1),
        ],
        vec![],
        vec![],
    );
    chunk.validate().unwrap();
    for (profile, want) in [
        (CompatibilityProfile::Lua51, 2.194_999_999_999_936_3),
        (CompatibilityProfile::LuaJit21, 2.194_999_999_999_936_3),
        (CompatibilityProfile::Lua54, 2.194_999_999_999_978_5),
        (CompatibilityProfile::Lua55, 2.194_999_999_999_978_5),
    ] {
        let (result, _) = run_both(profile, &chunk, Vec::new());
        assert_eq!(result, vec![Value::Number(want)], "{profile:?}");
    }
}

/// Lua 5.3 and later take an exact integer path when both operands are
/// integral and below 2^53. It must match `fmod` and its sign fix, including
/// the sign of a zero result, on the fast and the general loop.
#[test]
fn integral_float_modulo_matches_fmod_on_both_loops() {
    let inputs = [
        0.0,
        -0.0,
        3.0,
        -3.0,
        -6.0,
        7.0,
        -7.0,
        9_007_199_254_740_991.0,
        -9_007_199_254_740_992.0,
        1e300,
        5.5,
    ];
    let constants: Vec<Constant> = inputs.iter().map(|value| float(*value)).collect();
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let got = bits_on_both_loops(profile, &pairwise(OpCode::Modulo, constants.clone()));
        for (i, &left) in inputs.iter().enumerate() {
            for (j, &right) in inputs.iter().enumerate() {
                let m = left % right;
                let want = if profile == CompatibilityProfile::Lua53 {
                    if m * right < 0.0 { m + right } else { m }
                } else if (m > 0.0 && right < 0.0) || (m < 0.0 && right > 0.0) {
                    m + right
                } else {
                    m
                };
                let have = f64::from_bits(got[i * inputs.len() + j]);
                assert!(
                    have.to_bits() == want.to_bits() || (have.is_nan() && want.is_nan()),
                    "{profile:?}: {left:?} % {right:?} = {have:?}, fmod form {want:?}"
                );
            }
        }
    }
}

/// Integers beyond 2^53, the ends of the integer range, signed zeros,
/// infinities and NaN, with fractions on both sides of the integers.
fn mixed_edge_constants() -> Vec<Constant> {
    vec![
        integer(0),
        integer(-7),
        integer(9_007_199_254_740_993),
        integer(i64::MAX),
        integer(i64::MIN),
        float(0.0),
        float(-0.0),
        float(2.5),
        float(-2.5),
        float(9_007_199_254_740_992.0),
        float(9_223_372_036_854_775_808.0),
        float(-9_223_372_036_854_775_808.0),
        float(f64::INFINITY),
        float(f64::NEG_INFINITY),
        float(f64::NAN),
    ]
}

/// A debug rendering of a run, so a NaN result equals itself and an error
/// compares by message.
fn rendered(profile: CompatibilityProfile, chunk: &Chunk, general: bool) -> String {
    format!("{:?}", run_result(profile, chunk, general))
}

#[test]
fn mixed_pairs_match_the_general_loop_for_every_arithmetic_operator() {
    let constants = mixed_edge_constants();
    for opcode in [
        OpCode::Add,
        OpCode::Subtract,
        OpCode::Multiply,
        OpCode::Divide,
        OpCode::FloorDivide,
        OpCode::Modulo,
        OpCode::Power,
    ] {
        let chunk = pairwise(opcode, constants.clone());
        for profile in PROFILES {
            assert_eq!(
                rendered(profile, &chunk, false),
                rendered(profile, &chunk, true),
                "{opcode:?} {profile:?}"
            );
        }
    }
}

#[test]
fn mixed_pairs_compare_exactly_and_stay_in_the_fast_arm() {
    let constants = mixed_edge_constants();
    for opcode in [OpCode::Equal, OpCode::LessThan, OpCode::LessEqual] {
        let chunk = pairwise(opcode, constants.clone());
        for profile in PROFILES {
            let (_, fallbacks) = run_both(profile, &chunk, Vec::new());
            assert!(!fallbacks.contains(&opcode), "{opcode:?} {profile:?}");
        }
    }
    // 2^53 + 1 is not the float 2^53, and is greater than it.
    let chunk = pairwise(
        OpCode::LessThan,
        vec![
            integer(9_007_199_254_740_993),
            float(9_007_199_254_740_992.0),
        ],
    );
    let (values, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
    assert_eq!(
        values,
        vec![
            Value::Boolean(false),
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false)
        ]
    );
    // i64::MAX is below the float 2^63, and the two are not equal.
    let chunk = pairwise(
        OpCode::Equal,
        vec![integer(i64::MAX), float(9_223_372_036_854_775_808.0)],
    );
    let (values, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
    assert_eq!(values[1], Value::Boolean(false));
    assert_eq!(values[2], Value::Boolean(false));
}

#[test]
fn mixed_modulo_and_division_by_zero_keep_their_results() {
    // 7 % 0.0 is NaN in 5.4, 7 // 0.0 and 7 / 0 are infinite, and the integer
    // pair 7 % 0 is an error, none of them through a fast-arm shortcut.
    let chunk = pairwise(OpCode::Modulo, vec![integer(7), integer(0), float(0.0)]);
    for profile in PROFILES {
        assert_eq!(
            rendered(profile, &chunk, false),
            rendered(profile, &chunk, true),
            "{profile:?}"
        );
    }
}

/// Every value on both sides of the 32-bit line, where the integer `%`
/// switches between its unsigned 32-bit division and the signed 64-bit one.
#[test]
fn integer_modulo_agrees_across_the_32_bit_line() {
    let values = [
        0,
        1,
        7,
        97,
        (1 << 31) - 1,
        1 << 31,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        -1,
        -7,
        -(1 << 31),
        -(1 << 32),
        i64::MAX,
        i64::MIN,
    ];
    for left in values {
        for right in values {
            let register_form = Chunk::with_parts(
                3,
                0,
                vec![integer(left), integer(right)],
                vec![
                    Instruction::load_constant(0, 0),
                    Instruction::load_constant(1, 1),
                    Instruction::binary(OpCode::Modulo, 2, 0, 1),
                    Instruction::return_(2),
                ],
                vec![],
                vec![],
            );
            let constant_form = Chunk::with_parts(
                2,
                0,
                vec![integer(left), integer(right)],
                vec![
                    Instruction::load_constant(0, 0),
                    Instruction::binary(OpCode::ModuloK, 1, 0, 1),
                    Instruction::return_(1),
                ],
                vec![],
                vec![],
            );
            for chunk in [register_form, constant_form] {
                chunk.validate().unwrap();
                if right == 0 {
                    // The fast arm leaves a zero divisor to the general
                    // path, which raises the error.
                    for general in [false, true] {
                        let error =
                            run_result(CompatibilityProfile::Lua54, &chunk, general).unwrap_err();
                        assert!(error.contains("n%0"), "{left} % 0: {error}");
                    }
                    continue;
                }
                let (results, _) = run_both(CompatibilityProfile::Lua54, &chunk, Vec::new());
                assert_eq!(
                    results,
                    vec![Value::Integer(reference_floor_mod(left, right))],
                    "{left} % {right}"
                );
            }
        }
    }
}

/// An integer that reaches a register of a profile without integers (a
/// host can pass one) is not the fast arm's integer: the arm's one compare
/// of the tag must send it where the general loop sends it.
#[test]
fn an_integer_register_of_a_float_profile_leaves_the_integer_arm() {
    for (opcode, constant_form) in [
        (OpCode::Add, OpCode::AddK),
        (OpCode::Subtract, OpCode::SubtractK),
        (OpCode::Multiply, OpCode::MultiplyK),
        (OpCode::Modulo, OpCode::ModuloK),
    ] {
        let chunk = Chunk::with_parts(
            4,
            2,
            vec![integer(3)],
            vec![
                Instruction::binary(opcode, 2, 0, 1),
                Instruction::binary(constant_form, 3, 0, 0),
                Instruction::return_values(2, 2),
            ],
            vec![],
            vec![],
        );
        chunk.validate().unwrap();
        let arguments = || vec![Value::Integer(-7), Value::Integer(3)];
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ] {
            // `run_both` asserts that the fast and the general loop agree.
            let (results, _) = run_both(profile, &chunk, arguments());
            assert_eq!(results.len(), 2, "{profile:?} {opcode:?}");
        }
        let (results, _) = run_both(CompatibilityProfile::Lua54, &chunk, arguments());
        assert!(
            results
                .iter()
                .all(|value| matches!(value, Value::Integer(_))),
            "{opcode:?}: {results:?}"
        );
    }
}
