//! Operator metamethods whose handler is a plain Lua closure, entered from
//! the fast loop (see `Vm::operator_miss_hot`): no operator in a hot loop
//! leaves the loop for the scalar path, and the results are the dispatch
//! loop's.

use super::trace_record::{for_loop, for_prep, load};
use super::*;

fn number(value: f64) -> Constant {
    Constant::Number(value)
}

fn text(value: &str) -> Constant {
    Constant::String(LuaString::from(value))
}

/// A two-parameter handler that returns constant `result`, or nothing.
fn handler(result: Option<Constant>) -> Chunk {
    match result {
        Some(result) => Chunk::with_parts(
            3,
            2,
            vec![result],
            vec![load(2, 0), Instruction::return_(2)],
            vec![],
            vec![],
        ),
        None => Chunk::with_parts(3, 2, vec![], vec![Instruction::halt()], vec![], vec![]),
    }
}

/// Sets the handlers of `meta` (r6) and runs `limit` iterations of every
/// operator over the tables `object` (r4) and `other` (r5):
/// r10 `object + other`, r11 `object + 1`, r12 `1 + object`, r13 `-object`,
/// r14 `#object`, r17 `object .. other`, r18 `<`, r19 `<=`, r20 `==`, and r21
/// counts the iterations a `TestLessThan` branch sees `object < other`.
/// Returns r10..r21.
fn operator_loop(limit: f64) -> Chunk {
    let metamethods = [
        ("__add", Some(number(10.0))),
        ("__unm", Some(number(11.0))),
        ("__len", Some(number(12.0))),
        ("__concat", Some(text("cc"))),
        ("__lt", Some(number(1.0))),
        ("__le", None),
        ("__eq", Some(number(1.0))),
    ];
    let mut constants = vec![
        number(1.0),
        number(limit),
        text("object"),
        text("other"),
        text("meta"),
        number(0.0),
    ];
    let mut code = vec![
        Instruction::get_global(4, 2),
        Instruction::get_global(5, 3),
        Instruction::get_global(6, 4),
    ];
    let mut children = Vec::new();
    for (index, (name, result)) in metamethods.into_iter().enumerate() {
        constants.push(text(name));
        children.push(handler(result));
        code.push(Instruction::closure(7, index as u16));
        code.push(Instruction::set_field(6, 6 + index as u8, 7));
    }
    code.extend([load(0, 0), load(1, 1), load(2, 0), load(8, 0), load(21, 5)]);
    let prep = code.len();
    code.push(Instruction::halt());
    let header = code.len();
    code.extend([
        Instruction::binary(OpCode::Add, 10, 4, 5),
        Instruction::binary(OpCode::Add, 11, 4, 8),
        Instruction::binary(OpCode::Add, 12, 8, 4),
        Instruction::unary(OpCode::Negate, 13, 4),
        Instruction::length(14, 4),
        Instruction::move_(15, 4),
        Instruction::move_(16, 5),
        Instruction::concat(17, 15, 16),
        Instruction::binary(OpCode::LessThan, 18, 4, 5),
        Instruction::binary(OpCode::LessEqual, 19, 4, 5),
        Instruction::binary(OpCode::Equal, 20, 4, 5),
        Instruction::test(OpCode::TestLessThan, 4, 5, false, false),
        Instruction::aj(OpCode::Jump, 1).unwrap(),
        Instruction::binary(OpCode::Add, 21, 21, 8),
    ]);
    let latch = code.len();
    code.push(for_loop(0, latch, header));
    code.push(Instruction::return_values(10, 12));
    code[prep] = for_prep(0, prep, latch);
    Chunk::with_parts(22, 0, constants, code, children, vec![])
}

fn operator_vm(profile: CompatibilityProfile, profiling: bool) -> (Vm, Value, Value) {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm.set_interpreter_profiling(profiling);
    let object = vm.runtime_mut().create_table().unwrap();
    let other = vm.runtime_mut().create_table().unwrap();
    let meta = vm.runtime_mut().create_table().unwrap();
    for table in [object, other] {
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(table), Some(meta))
            .unwrap();
    }
    vm.runtime_mut().set_global("object", Value::Table(object));
    vm.runtime_mut().set_global("other", Value::Table(other));
    vm.runtime_mut().set_global("meta", Value::Table(meta));
    (vm, Value::Table(object), Value::Table(other))
}

/// Every operator of the loop enters its handler from the fast loop, and the
/// results are those of the dispatch loop.
#[test]
fn plain_operator_handlers_are_entered_without_the_scalar_path() {
    const LIMIT: f64 = 300.0;
    let chunk = operator_loop(LIMIT);
    chunk.validate().unwrap();
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (mut general, ..) = operator_vm(profile, true);
        let expected = general.execute(&chunk).unwrap();
        assert_eq!(general.operator_handler_entries, 0, "{profile:?}");
        let (mut fast, object, other) = operator_vm(profile, false);
        let results = fast.execute(&chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        assert_eq!(results[0], Value::Number(10.0), "{profile:?}");
        assert_eq!(results[1], Value::Number(10.0), "{profile:?}");
        assert_eq!(results[2], Value::Number(10.0), "{profile:?}");
        assert_eq!(results[3], Value::Number(11.0), "{profile:?}");
        assert_eq!(results[5], object, "{profile:?}");
        assert_eq!(results[6], other, "{profile:?}");
        assert_eq!(results[7], Value::from("cc"), "{profile:?}");
        assert_eq!(results[8], Value::Boolean(true), "{profile:?}");
        assert_eq!(results[9], Value::Boolean(false), "{profile:?}");
        assert_eq!(results[10], Value::Boolean(true), "{profile:?}");
        assert_eq!(results[11], Value::Number(LIMIT), "{profile:?}");
        // Lua 5.1 takes no `__len` of a table.
        let length = if profile == CompatibilityProfile::Lua51 {
            0
        } else {
            assert_eq!(results[4], Value::Number(12.0), "{profile:?}");
            1
        };
        // Nine or ten handler calls an iteration. An entry appends the
        // activation record when no earlier call left one at its depth, so
        // at most the first entry of the run declines.
        let per_iteration = 9 + length;
        assert!(
            fast.operator_handler_entries >= per_iteration * (LIMIT as usize) - 1,
            "{profile:?}: {} entries",
            fast.operator_handler_entries
        );
    }
}

/// Host calls run between the operators, so a handler may find no retired
/// activation record at its depth. The entry appends one and stays on the
/// fast loop.
#[test]
fn handlers_are_entered_after_a_host_call_dropped_their_record() {
    const LIMIT: f64 = 200.0;
    let constants = vec![
        number(1.0),
        number(LIMIT),
        text("object"),
        text("other"),
        text("meta"),
        text("type"),
        text("__add"),
        text("__lt"),
    ];
    let mut code = vec![
        Instruction::get_global(4, 2),
        Instruction::get_global(5, 3),
        Instruction::get_global(6, 4),
        Instruction::closure(7, 0),
        Instruction::set_field(6, 6, 7),
        Instruction::closure(7, 1),
        Instruction::set_field(6, 7, 7),
        load(0, 0),
        load(1, 1),
        load(2, 0),
    ];
    let prep = code.len();
    code.push(Instruction::halt());
    let header = code.len();
    code.extend([
        Instruction::get_global(15, 5),
        Instruction::move_(16, 4),
        Instruction::call(15, 1, 2),
        Instruction::binary(OpCode::Add, 10, 4, 5),
        Instruction::get_global(15, 5),
        Instruction::move_(16, 4),
        Instruction::call(15, 1, 2),
        Instruction::binary(OpCode::LessThan, 11, 4, 5),
        Instruction::get_global(15, 5),
        Instruction::move_(16, 4),
        Instruction::call(15, 1, 2),
        Instruction::binary(OpCode::Add, 12, 4, 0),
    ]);
    let latch = code.len();
    code.push(for_loop(0, latch, header));
    code.push(Instruction::return_values(10, 3));
    code[prep] = for_prep(0, prep, latch);
    let children = vec![handler(Some(number(10.0))), handler(Some(number(1.0)))];
    let chunk = Chunk::with_parts(18, 0, constants, code, children, vec![]);
    chunk.validate().unwrap();
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (mut general, ..) = operator_vm(profile, true);
        let expected = general.execute(&chunk).unwrap();
        let (mut fast, ..) = operator_vm(profile, false);
        let results = fast.execute(&chunk).unwrap();
        assert_eq!(results, expected, "{profile:?}");
        assert_eq!(
            results,
            vec![
                Value::Number(10.0),
                Value::Boolean(true),
                Value::Number(10.0)
            ],
            "{profile:?}"
        );
        assert!(
            fast.operator_handler_entries >= 3 * (LIMIT as usize) - 1,
            "{profile:?}: {} entries",
            fast.operator_handler_entries
        );
    }
}
