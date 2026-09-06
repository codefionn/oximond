//! Instructions the fast loop runs itself, instead of handing them to the
//! dispatch loop, and the fetch that no longer checks its bounds.
//!
//! Each case runs a hand-assembled chunk on the fast loop and again with
//! profiling on, which keeps the fast loop out. The results must match, and
//! the fast run must not have handed the covered opcodes to the dispatch loop.

use super::trace_record::interpreter_vm_with;
use super::*;
use crate::vm::interp::GENERAL_FALLBACKS;

pub(super) const PROFILES: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54];

fn number(value: f64) -> Constant {
    Constant::Number(value)
}

fn text(value: &str) -> Constant {
    Constant::String(LuaString::from(value))
}

/// Runs `chunk` on the fast loop and on the general loop. Returns the shared
/// result and the opcodes the fast run handed to the dispatch loop.
pub(super) fn run_both(
    profile: CompatibilityProfile,
    chunk: &Chunk,
    arguments: Vec<Value>,
) -> (Vec<Value>, Vec<OpCode>) {
    let build = |profiling: bool| {
        let mut vm = interpreter_vm_with(profile);
        vm.debug_hooks.clear();
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        vm.set_interpreter_profiling(profiling);
        vm
    };
    GENERAL_FALLBACKS.with(|log| log.borrow_mut().clear());
    let fast = build(false)
        .execute_with_arguments(chunk, arguments.clone())
        .unwrap();
    let fallbacks = GENERAL_FALLBACKS.with(|log| log.borrow().clone());
    let general = build(true)
        .execute_with_arguments(chunk, arguments)
        .unwrap();
    assert_eq!(fast, general, "{profile:?}");
    (fast, fallbacks)
}

fn counter_child() -> Chunk {
    // upvalue 0 += 1
    Chunk::with_parts(
        2,
        0,
        vec![number(1.0)],
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::set_upvalue(0, 0),
            Instruction::return_values(0, 0),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: LuaString::from("count"),
            source: UpvalueSource::ParentRegister(0),
        }],
    )
}

#[test]
fn upvalue_stores_and_closures_stay_on_the_fast_loop() {
    let chunk = Chunk::with_parts(
        3,
        0,
        vec![number(0.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::move_(2, 1),
            Instruction::call(2, 0, 1),
            Instruction::move_(2, 1),
            Instruction::call(2, 0, 1),
            Instruction::move_(2, 1),
            Instruction::call(2, 0, 1),
            Instruction::return_(0),
        ],
        vec![counter_child()],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        assert_eq!(results, vec![Value::Number(3.0)], "{profile:?}");
        assert!(
            !fallbacks.contains(&OpCode::SetUpvalue) && !fallbacks.contains(&OpCode::Closure),
            "{profile:?}: {fallbacks:?}"
        );
    }
}

#[test]
fn vararg_expansion_stays_on_the_fast_loop() {
    let chunk = Chunk::with_varargs(
        4,
        0,
        vec![],
        vec![
            Instruction::load_nil(3),
            Instruction::vararg(0, 2),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
        None,
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(
            profile,
            &chunk,
            vec![Value::Number(4.0), Value::Number(5.0)],
        );
        assert_eq!(results, vec![Value::Number(9.0)], "{profile:?}");
        assert!(
            !fallbacks.contains(&OpCode::Vararg),
            "{profile:?}: {fallbacks:?}"
        );
    }
}

#[test]
fn open_result_calls_of_natives_stay_on_the_fast_loop() {
    // return math.max(1, 5, 3): the call leaves its results open.
    let chunk = Chunk::new(
        5,
        vec![
            text("math"),
            text("max"),
            number(1.0),
            number(5.0),
            number(3.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_table(0, 0, 1),
            Instruction::load_constant(1, 2),
            Instruction::load_constant(2, 3),
            Instruction::load_constant(3, 4),
            Instruction::call(0, 3, 0),
            Instruction::return_open(0, 0),
        ],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        assert_eq!(results, vec![Value::Number(5.0)], "{profile:?}");
        assert!(
            !fallbacks.contains(&OpCode::Call),
            "{profile:?}: {fallbacks:?}"
        );
    }
}

#[test]
fn open_result_calls_of_lua_functions_stay_on_the_fast_loop() {
    // local function pair() return 7, 8 end; pair(); return pair()
    // The first call classifies the closure on the dispatch loop.
    let pair = Chunk::new(
        2,
        vec![number(7.0), number(8.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::return_values(0, 2),
        ],
    );
    let chunk = Chunk::with_parts(
        2,
        0,
        vec![],
        vec![
            Instruction::load_nil(1),
            Instruction::closure(0, 0),
            Instruction::move_(1, 0),
            Instruction::call(1, 0, 0),
            Instruction::call(0, 0, 0),
            Instruction::return_open(0, 0),
        ],
        vec![pair],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        assert_eq!(
            results,
            vec![Value::Number(7.0), Value::Number(8.0)],
            "{profile:?}"
        );
        let calls = fallbacks.iter().filter(|&&op| op == OpCode::Call).count();
        assert!(
            calls <= 1 && !fallbacks.contains(&OpCode::Return),
            "{profile:?}: {fallbacks:?}"
        );
    }
}

#[test]
fn empty_integer_loop_ending_the_chunk_does_not_read_past_the_code() {
    // A validated chunk may point a loop's prep at its last instruction. The
    // fast loop skips the latch of an empty range, so it must check that the
    // instruction after the target exists before fetching it unchecked.
    let chunk = Chunk::new(
        4,
        vec![
            Constant::Integer(5),
            Constant::Integer(1),
            Constant::Integer(1),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::numeric_for_prep(0, 0),
            Instruction::halt(),
        ],
    );
    chunk.validate().unwrap();
    let outcome = |profiling: bool| {
        let mut vm = interpreter_vm_with(CompatibilityProfile::Lua54);
        vm.debug_hooks.clear();
        vm.set_interpreter_profiling(profiling);
        format!("{:?}", vm.execute(&chunk))
    };
    assert_eq!(outcome(false), outcome(true));
}

/// Chunks that never went through `Chunk::validate` could reach the fast loop
/// through `create_closure`, and its unchecked fetch would read past the
/// code. The runtime validates them, so they are refused instead.
#[test]
fn unvalidated_chunks_cannot_become_closures() {
    let wild_jump = Instruction::aj(OpCode::Jump, 100).unwrap();
    let invalid = [
        vec![Instruction::load_nil(0), Instruction::move_(1, 0)],
        vec![wild_jump, Instruction::return_(0)],
    ];
    for code in invalid {
        let chunk = Chunk::new(2, vec![], code);
        assert!(chunk.validate().is_err());
        let mut vm = interpreter_vm_with(CompatibilityProfile::Lua54);
        let created = vm.runtime_mut().create_closure(Arc::new(chunk), vec![]);
        assert!(created.is_err());
    }
}

/// A chunk that builds `{ 10, 20, 30 }`, runs the iterator function named
/// `constructor` ("ipairs" or "pairs") on it into r10..r12, and loops with
/// `ForInCall`/`ForInLoop` over `variables` loop variables (from r13). The
/// body is `body`, which may use r1 as an accumulator starting at 0 and
/// returns it.
fn for_in_chunk(constructor: &str, variables: u8, body: Vec<Instruction>) -> Chunk {
    let mut code = vec![
        Instruction::get_global(10, 0),
        Instruction::new_table(11, 3, 0),
    ];
    for index in 0..3_u16 {
        code.push(Instruction::load_constant(2, 1 + index));
        code.push(Instruction::load_constant(3, 4 + index));
        code.push(Instruction::set_table(11, 2, 3));
    }
    code.extend([
        Instruction::call(10, 1, 3),
        Instruction::load_constant(1, 7),
    ]);
    // The entry jump skips the body to the call.
    let length = i16::try_from(body.len()).unwrap();
    code.push(Instruction::aj(OpCode::Jump, i32::from(length)).unwrap());
    code.extend(body);
    code.push(Instruction::for_in_call(10, variables));
    let back = -(length + 2);
    code.push(Instruction::for_in_loop(10, back));
    code.push(Instruction::return_(1));
    Chunk::new(
        20,
        vec![
            text(constructor),
            Constant::Integer(1),
            Constant::Integer(2),
            Constant::Integer(3),
            Constant::Integer(10),
            Constant::Integer(20),
            Constant::Integer(30),
            Constant::Integer(0),
        ],
        code,
    )
}

#[test]
fn generic_for_over_ipairs_and_pairs_stays_on_the_fast_loop() {
    let sum_values = vec![Instruction::binary(OpCode::Add, 1, 1, 14)];
    let sum_keys = vec![Instruction::binary(OpCode::Add, 1, 1, 13)];
    let cases = [
        ("ipairs", 2, sum_values.clone(), 60),
        ("ipairs", 1, sum_keys.clone(), 6),
        ("pairs", 2, sum_values, 60),
        ("pairs", 1, sum_keys, 6),
    ];
    for (constructor, variables, body, expected) in cases {
        let chunk = for_in_chunk(constructor, variables, body);
        chunk.validate().unwrap();
        for profile in PROFILES {
            let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
            let [total] = results.as_slice() else {
                panic!("{constructor} {variables}: {results:?}");
            };
            let total = match total {
                Value::Integer(value) => *value as f64,
                Value::Number(value) => *value,
                other => panic!("{other:?}"),
            };
            assert_eq!(total, f64::from(expected), "{constructor} {profile:?}");
            assert!(
                !fallbacks.contains(&OpCode::ForInCall) && !fallbacks.contains(&OpCode::ForInLoop),
                "{constructor} {variables} {profile:?}: {fallbacks:?}"
            );
        }
    }
}

#[test]
fn generic_for_with_extra_variables_pads_with_nil_on_the_fast_loop() {
    // Three variables over pairs: the third is always nil. The body adds the
    // loop variable r15 to the accumulator only when it is not nil, so a
    // stale register would change the sum.
    let body = vec![
        Instruction::binary(OpCode::Add, 1, 1, 14),
        Instruction::load_nil(2),
        Instruction::test(OpCode::TestEqual, 15, 2, true, false),
        Instruction::aj(OpCode::Jump, 1).unwrap(),
        Instruction::load_constant(1, 7),
    ];
    let chunk = for_in_chunk("pairs", 3, body);
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        let total = match results.as_slice() {
            [Value::Integer(value)] => *value as f64,
            [Value::Number(value)] => *value,
            other => panic!("{other:?}"),
        };
        assert_eq!(total, 60.0, "{profile:?}");
        assert!(!fallbacks.contains(&OpCode::ForInCall), "{fallbacks:?}");
    }
}

#[test]
fn generic_for_over_a_lua_closure_pushes_activations_on_the_fast_loop() {
    // function(state, control) control = control + 1; if control <= 3 then
    // return control end end
    let iterator = Chunk::with_parts(
        4,
        2,
        vec![Constant::Integer(1), Constant::Integer(3)],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::binary(OpCode::Add, 1, 1, 2),
            Instruction::test(OpCode::TestLessEqualK, 1, 1, true, false),
            Instruction::aj(OpCode::Jump, 2).unwrap(),
            Instruction::load_nil(1),
            Instruction::return_values(1, 1),
            Instruction::return_values(1, 1),
        ],
        vec![],
        vec![],
    );
    let chunk = Chunk::with_parts(
        20,
        0,
        vec![Constant::Integer(0)],
        vec![
            Instruction::closure(10, 0),
            Instruction::load_nil(11),
            Instruction::load_constant(12, 0),
            Instruction::load_constant(1, 0),
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::binary(OpCode::Add, 1, 1, 13),
            Instruction::for_in_call(10, 1),
            Instruction::for_in_loop(10, -3),
            Instruction::return_(1),
        ],
        vec![iterator],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let (results, fallbacks) = run_both(profile, &chunk, Vec::new());
        let total = match results.as_slice() {
            [Value::Integer(value)] => *value as f64,
            [Value::Number(value)] => *value,
            other => panic!("{other:?}"),
        };
        assert_eq!(total, 6.0, "{profile:?}");
        // The first call classifies the closure on the dispatch loop, as a
        // `Call` does. The three calls after it push activations.
        let calls = fallbacks
            .iter()
            .filter(|opcode| **opcode == OpCode::ForInCall)
            .count();
        assert!(calls <= 1, "{profile:?}: {fallbacks:?}");
        assert!(!fallbacks.contains(&OpCode::ForInLoop), "{fallbacks:?}");
    }
}

#[test]
fn a_test_without_its_jump_word_reports_the_opcode_it_found() {
    // Validation rejects a test that no `Jump` follows, so build the frame
    // by hand. The dispatch error names the word that stood there.
    let word_after = |word: Instruction| {
        let chunk = Arc::new(Chunk::new(
            2,
            Vec::new(),
            vec![
                Instruction::test(OpCode::TestEqual, 0, 1, true, false),
                word,
                Instruction::halt(),
            ],
        ));
        let vm = Vm::new();
        let environment = vm.runtime().global_environment();
        let mut frame = Frame::new(chunk, Vec::new(), Vec::new(), None, None, environment, None);
        // The test has run, so `pc` is at its jump word.
        frame.pc = 1;
        let outcome = crate::vm::instructions::branch_after_test(&mut frame, true, true);
        drop(vm);
        outcome
    };
    assert!(matches!(
        word_after(Instruction::move_(0, 1)),
        Err(VmError::InvalidDispatch(OpCode::Move))
    ));
    // A word that does not decode falls back to `Jump`.
    assert!(matches!(
        word_after(Instruction::from_raw(0xff)),
        Err(VmError::InvalidDispatch(OpCode::Jump))
    ));
    assert!(matches!(
        word_after(Instruction::aj(OpCode::Jump, 0).unwrap()),
        Ok(())
    ));
}

/// `__index` handler `function(t, k) return 7 end` and `__newindex` handler
/// `function(t, k, v) total = total + v end`, with `total` in register 8 of
/// the chunk below, run `limit` times from a loop that reads `object.foo` and
/// writes `object.bar`, then `object[key]` and `object[key] = v` with a
/// register key (a string or a number). Returns the sums of both.
fn index_handler_loop(limit: f64, key: Constant) -> Chunk {
    use super::trace_record::{for_loop, for_prep, load};
    let index_handler = Chunk::with_parts(
        3,
        2,
        vec![number(7.0)],
        vec![load(2, 0), Instruction::return_(2)],
        vec![],
        vec![],
    );
    let newindex_handler = Chunk::with_parts(
        4,
        3,
        vec![],
        vec![
            Instruction::get_upvalue(3, 0),
            Instruction::binary(OpCode::Add, 3, 3, 2),
            Instruction::set_upvalue(3, 0),
            Instruction::halt(),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: LuaString::from("total"),
            source: UpvalueSource::ParentRegister(8),
        }],
    );
    // r0..r3 the loop, r4 the object, r5 its metatable, r6 a closure,
    // r7 the sum of reads, r8 the sum of writes, r9 a read, r10 a key.
    Chunk::with_parts(
        11,
        0,
        vec![
            number(0.0),
            number(1.0),
            number(limit),
            text("object"),
            text("meta"),
            text("__index"),
            text("__newindex"),
            text("foo"),
            text("bar"),
            key,
        ],
        vec![
            load(7, 0),
            load(8, 0),
            Instruction::get_global(4, 3),
            Instruction::get_global(5, 4),
            Instruction::closure(6, 0),
            Instruction::set_field(5, 5, 6),
            Instruction::closure(6, 1),
            Instruction::set_field(5, 6, 6),
            load(0, 1),
            load(1, 2),
            load(2, 1),
            for_prep(0, 11, 19),
            Instruction::get_field(9, 4, 7),
            Instruction::binary(OpCode::Add, 7, 7, 9),
            Instruction::set_field(4, 8, 9),
            load(10, 9),
            Instruction::get_table(9, 4, 10),
            Instruction::binary(OpCode::Add, 7, 7, 9),
            Instruction::set_table(4, 10, 9),
            for_loop(0, 19, 12),
            Instruction::return_values(7, 2),
        ],
        vec![index_handler, newindex_handler],
        vec![],
    )
}

/// The fast loop enters a plain Lua `__index` or `__newindex` handler itself:
/// no iteration of a hot loop leaves the loop for its miss or resolves it out
/// of line, and the results are those of the dispatch loop alone.
#[test]
fn plain_function_handlers_are_entered_without_leaving_the_fast_loop() {
    const LIMIT: f64 = 400.0;
    for key in [text("key"), number(5.0)] {
        let chunk = index_handler_loop(LIMIT, key);
        chunk.validate().unwrap();
        check_index_handler_loop(&chunk, LIMIT);
    }
}

fn check_index_handler_loop(chunk: &Chunk, limit: f64) {
    for profile in PROFILES {
        let build = |profiling: bool| {
            let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
            install_base_with_profile(vm.runtime_mut(), profile).unwrap();
            vm.set_interpreter_profiling(profiling);
            let object = vm.runtime_mut().create_table().unwrap();
            let meta = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .raw_set_metatable(&Value::Table(object), Some(meta))
                .unwrap();
            vm.runtime_mut().set_global("object", Value::Table(object));
            vm.runtime_mut().set_global("meta", Value::Table(meta));
            vm
        };
        let mut fast = build(false);
        let results = fast.execute(chunk).unwrap();
        // Reads are 7 per `foo` and per `key`; the writes store 7 to `bar`
        // and the read value to `key`, which is 7 as well.
        let expected = vec![Value::Number(limit * 14.0), Value::Number(limit * 14.0)];
        assert_eq!(results, expected, "{profile:?}");
        assert_eq!(build(true).execute(chunk).unwrap(), expected, "{profile:?}");
        // Four handler calls an iteration. Only the entries that find no
        // record at their depth yet (the first of each handler) may resolve
        // out of line.
        assert!(
            fast.index_handler_entries >= 4 * (limit as usize) - 4,
            "{profile:?}: {} entries",
            fast.index_handler_entries
        );
        assert_eq!(
            fast.index_miss_exits, 0,
            "{profile:?}: a handler loop left the fast loop"
        );
        assert!(
            fast.index_miss_resolves <= 4,
            "{profile:?}: {} out-of-line resolutions",
            fast.index_miss_resolves
        );
    }
}

/// Handlers with an empty body (which is also what a bare `return` compiles
/// to) start on a `Halt`. The loop reads `object.foo` and writes `object.bar`
/// through them: the read gives nil, and no iteration leaves the fast loop or
/// resolves its miss out of line after the first of each handler.
#[test]
fn empty_handlers_are_entered_without_leaving_the_fast_loop() {
    use super::trace_record::{for_loop, for_prep, load};
    const LIMIT: f64 = 400.0;
    let handler = |parameters| {
        Chunk::with_parts(
            parameters,
            parameters,
            vec![],
            vec![Instruction::halt()],
            vec![],
            vec![],
        )
    };
    // r0..r3 the loop, r4 the object, r5 its metatable, r6 a closure, r9 a read.
    let chunk = Chunk::with_parts(
        10,
        0,
        vec![
            number(1.0),
            number(LIMIT),
            text("object"),
            text("meta"),
            text("__index"),
            text("__newindex"),
            text("foo"),
            text("bar"),
        ],
        vec![
            Instruction::get_global(4, 2),
            Instruction::get_global(5, 3),
            Instruction::closure(6, 0),
            Instruction::set_field(5, 4, 6),
            Instruction::closure(6, 1),
            Instruction::set_field(5, 5, 6),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 9, 12),
            Instruction::get_field(9, 4, 6),
            Instruction::set_field(4, 7, 9),
            for_loop(0, 12, 10),
            Instruction::return_values(9, 1),
        ],
        vec![handler(2), handler(3)],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let object = vm.runtime_mut().create_table().unwrap();
        let meta = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(object), Some(meta))
            .unwrap();
        vm.runtime_mut().set_global("object", Value::Table(object));
        vm.runtime_mut().set_global("meta", Value::Table(meta));
        assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Nil], "{profile:?}");
        assert!(
            vm.index_handler_entries >= 2 * (LIMIT as usize) - 2,
            "{profile:?}: {} entries",
            vm.index_handler_entries
        );
        assert_eq!(vm.index_miss_exits, 0, "{profile:?}");
        assert!(
            vm.index_miss_resolves <= 2,
            "{profile:?}: {} out-of-line resolutions",
            vm.index_miss_resolves
        );
    }
}

/// A VM whose globals `object` and `meta` are a table and its metatable.
fn object_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    let object = vm.runtime_mut().create_table().unwrap();
    let meta = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(object), Some(meta))
        .unwrap();
    vm.runtime_mut().set_global("object", Value::Table(object));
    vm.runtime_mut().set_global("meta", Value::Table(meta));
    vm
}

/// The method `function(self, x) return x end`, and an `__index` handler
/// `function(t, k) return method end` that makes a closure of it each call.
fn method_and_handler() -> Chunk {
    let method = Chunk::with_parts(2, 2, vec![], vec![Instruction::return_(1)], vec![], vec![]);
    Chunk::with_parts(
        3,
        2,
        vec![],
        vec![Instruction::closure(2, 0), Instruction::return_(2)],
        vec![method],
        vec![],
    )
}

fn check_handler_pins(vm: &Vm, profile: CompatibilityProfile, limit: f64) {
    assert!(
        vm.index_handler_entries >= limit as usize - 1,
        "{profile:?}: {} entries",
        vm.index_handler_entries
    );
    assert_eq!(
        vm.index_miss_exits, 0,
        "{profile:?}: a handler loop left the fast loop"
    );
    assert!(
        vm.index_miss_resolves <= 2,
        "{profile:?}: {} out-of-line resolutions",
        vm.index_miss_resolves
    );
}

/// `object:m(1)` where `m` comes from a plain `__index` closure is a
/// `SelfField` miss that the fast loop enters the handler for.
#[test]
fn method_calls_through_a_plain_index_handler_stay_in_the_fast_loop() {
    use super::trace_record::{for_loop, for_prep, load};
    const LIMIT: f64 = 400.0;
    // r0..r3 the loop, r4 the object, r5 its metatable, r6 a closure, r7 the
    // sum, r9 the method and r10 its receiver, r11 the argument.
    let chunk = Chunk::with_parts(
        12,
        0,
        vec![
            number(0.0),
            number(1.0),
            number(LIMIT),
            text("object"),
            text("meta"),
            text("__index"),
            text("m"),
        ],
        vec![
            load(7, 0),
            Instruction::get_global(4, 3),
            Instruction::get_global(5, 4),
            Instruction::closure(6, 0),
            Instruction::set_field(5, 5, 6),
            load(0, 1),
            load(1, 2),
            load(2, 1),
            for_prep(0, 8, 13),
            Instruction::self_field(9, 4, 6),
            load(11, 1),
            Instruction::call(9, 2, 2),
            Instruction::binary(OpCode::Add, 7, 7, 9),
            for_loop(0, 13, 9),
            Instruction::return_values(7, 1),
        ],
        vec![method_and_handler()],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let mut vm = object_vm(profile);
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT)],
            "{profile:?}"
        );
        check_handler_pins(&vm, profile, LIMIT);
        let mut general = object_vm(profile);
        general.set_interpreter_profiling(true);
        assert_eq!(
            general.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT)],
            "{profile:?}"
        );
    }
}

/// A read of `object.m` through the table in an upvalue (`GetTabUp`) enters a
/// plain `__index` closure from the fast loop.
#[test]
fn upvalue_table_reads_through_a_plain_index_handler_stay_in_the_fast_loop() {
    use super::trace_record::{for_loop, for_prep, load};
    const LIMIT: f64 = 400.0;
    // The reader `function() local s = 0; for ... do s = s + up.m(0, 1) end end`:
    // r0..r3 the loop, r7 the sum, r9 the function, r10 its argument.
    let reader = Chunk::with_parts(
        12,
        0,
        vec![number(0.0), number(1.0), number(LIMIT), text("m")],
        vec![
            load(7, 0),
            load(0, 1),
            load(1, 2),
            load(2, 1),
            for_prep(0, 4, 10),
            Instruction::get_tab_up(9, 0, 3),
            load(10, 0),
            load(11, 1),
            Instruction::call(9, 2, 2),
            Instruction::binary(OpCode::Add, 7, 7, 9),
            for_loop(0, 10, 5),
            Instruction::return_values(7, 1),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: LuaString::from("up"),
            source: UpvalueSource::ParentRegister(4),
        }],
    );
    // r4 the object, r5 its metatable, r6 a closure.
    let chunk = Chunk::with_parts(
        12,
        0,
        vec![text("object"), text("meta"), text("__index")],
        vec![
            Instruction::get_global(4, 0),
            Instruction::get_global(5, 1),
            Instruction::closure(6, 0),
            Instruction::set_field(5, 2, 6),
            Instruction::closure(7, 1),
            Instruction::call(7, 0, 2),
            Instruction::return_values(7, 1),
        ],
        vec![method_and_handler(), reader],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let mut vm = object_vm(profile);
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT)],
            "{profile:?}"
        );
        check_handler_pins(&vm, profile, LIMIT);
    }
}

/// A VM in the plain interpreter with the base library.
fn plain_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm
}

/// `pcall(f, i)` of a plain closure pushes its activation in the fast loop
/// and the closure's return writes `true` and the value into the caller's
/// window, with no dispatch-loop fallback in between.
#[test]
fn pcall_of_a_plain_closure_is_pushed_and_returned_by_the_fast_loop() {
    use super::trace_record::{for_loop, for_prep, load};
    const LIMIT: f64 = 400.0;
    // function(x) return x end
    let child = Chunk::with_parts(1, 1, vec![], vec![Instruction::return_(0)], vec![], vec![]);
    // r0..r3 the loop, r5 the closure, r7 the sum, r8 `pcall`, r9 the status
    // and r10 the value.
    let chunk = Chunk::with_parts(
        11,
        0,
        vec![number(0.0), number(1.0), number(LIMIT), text("pcall")],
        vec![
            load(7, 0),
            Instruction::closure(5, 0),
            load(0, 1),
            load(1, 2),
            load(2, 1),
            for_prep(0, 5, 11),
            Instruction::get_global(8, 3),
            Instruction::move_(9, 5),
            Instruction::move_(10, 3),
            Instruction::call(8, 2, 2),
            Instruction::binary(OpCode::Add, 7, 7, 10),
            for_loop(0, 11, 6),
            Instruction::return_values(7, 1),
        ],
        vec![child],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let mut vm = plain_vm(profile);
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT * (LIMIT + 1.0) / 2.0)],
            "{profile:?}"
        );
        let calls = LIMIT as usize;
        assert!(
            vm.protected_fast_pushes >= calls,
            "{profile:?}: {} pushes",
            vm.protected_fast_pushes
        );
        assert!(
            vm.protected_fast_returns >= calls,
            "{profile:?}: {} returns",
            vm.protected_fast_returns
        );
        assert!(
            vm.fast_fallback_exits < 8,
            "{profile:?}: {} fallbacks",
            vm.fast_fallback_exits
        );
        // The general loop agrees.
        let mut general = plain_vm(profile);
        general.set_interpreter_profiling(true);
        assert_eq!(
            general.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT * (LIMIT + 1.0) / 2.0)],
            "{profile:?}"
        );
        assert_eq!(general.protected_fast_pushes, 0, "{profile:?}");
    }
}

/// A call of a table with a plain `__call` closure, as `Call` and as
/// `CallOne`, enters the handler from the fast loop with the table as its
/// first argument.
#[test]
fn call_metamethod_closures_are_entered_by_the_fast_loop() {
    use super::trace_record::{for_loop, for_prep, load};
    const LIMIT: f64 = 400.0;
    // function(self, x) return x end
    let handler = Chunk::with_parts(2, 2, vec![], vec![Instruction::return_(1)], vec![], vec![]);
    // r0..r3 the loop, r4 the object, r5 its metatable, r6 the handler,
    // r7 the sum, r8 and r9 a call, r10 a one-argument call.
    let chunk = Chunk::with_parts(
        11,
        0,
        vec![
            number(0.0),
            number(1.0),
            number(LIMIT),
            text("object"),
            text("meta"),
            text("__call"),
        ],
        vec![
            load(7, 0),
            Instruction::get_global(4, 3),
            Instruction::get_global(5, 4),
            Instruction::closure(6, 0),
            Instruction::set_field(5, 5, 6),
            load(0, 1),
            load(1, 2),
            load(2, 1),
            for_prep(0, 8, 15),
            Instruction::move_(8, 4),
            Instruction::move_(9, 3),
            Instruction::call(8, 1, 1),
            Instruction::binary(OpCode::Add, 7, 7, 8),
            Instruction::call_one(10, 4, 3),
            Instruction::binary(OpCode::Add, 7, 7, 10),
            for_loop(0, 15, 9),
            Instruction::return_values(7, 1),
        ],
        vec![handler],
        vec![],
    );
    chunk.validate().unwrap();
    for profile in PROFILES {
        let mut vm = object_vm(profile);
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT * (LIMIT + 1.0))],
            "{profile:?}"
        );
        assert!(
            vm.call_handler_pushes >= 2 * LIMIT as usize,
            "{profile:?}: {} pushes",
            vm.call_handler_pushes
        );
        assert!(
            vm.fast_fallback_exits < 8,
            "{profile:?}: {} fallbacks",
            vm.fast_fallback_exits
        );
        let mut general = object_vm(profile);
        general.set_interpreter_profiling(true);
        assert_eq!(
            general.execute(&chunk).unwrap(),
            vec![Value::Number(LIMIT * (LIMIT + 1.0))],
            "{profile:?}"
        );
        assert_eq!(general.call_handler_pushes, 0, "{profile:?}");
    }
}
