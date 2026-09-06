use super::*;
use crate::vm::stack::CallStack;

#[test]
fn gmatch_iterator_yields_captures_positions_and_empty_matches() {
    let mut vm = Vm::new();
    let iterator = vm
        .runtime_mut()
        .create_gmatch_iterator("ab".into(), "()".into())
        .unwrap();

    assert_eq!(
        vm.call_gmatch_iterator(iterator).unwrap(),
        vec![Value::Number(1.0)]
    );
    assert_eq!(
        vm.call_gmatch_iterator(iterator).unwrap(),
        vec![Value::Number(2.0)]
    );
    assert_eq!(
        vm.call_gmatch_iterator(iterator).unwrap(),
        vec![Value::Number(3.0)]
    );
    assert_eq!(vm.call_gmatch_iterator(iterator).unwrap(), vec![Value::Nil]);

    let words = vm
        .runtime_mut()
        .create_gmatch_iterator("one two".into(), "(%a+)".into())
        .unwrap();
    assert_eq!(
        vm.call_gmatch_iterator(words).unwrap(),
        vec![Value::from("one")]
    );
    assert_eq!(
        vm.call_gmatch_iterator(words).unwrap(),
        vec![Value::from("two")]
    );
    assert_eq!(vm.call_gmatch_iterator(words).unwrap(), vec![Value::Nil]);
}

#[test]
fn thread_environment_applies_to_newly_loaded_closures_and_is_rooted() {
    #[allow(clippy::unnecessary_wraps)]
    fn loader(_: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        Ok(Chunk::new(
            1,
            vec![Constant::String("answer".into())],
            vec![Instruction::get_global(0, 0), Instruction::return_(0)],
        ))
    }

    let mut vm = Vm::new();
    let environment = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(environment, &Value::from("answer"), Value::Number(41.0))
        .unwrap();
    assert_eq!(
        vm.set_fenv(
            &mut CallStack::new(),
            &[Value::Number(0.0), Value::Table(environment)],
        )
        .unwrap(),
        vec![Value::Number(0.0)]
    );
    assert_eq!(
        vm.get_fenv(&CallStack::new(), &[Value::Number(0.0)])
            .unwrap(),
        vec![Value::Table(environment)]
    );
    set_function_loader(&mut vm, loader);
    let Value::Closure(closure) = vm
        .load_string(
            &LuaString::from("return answer"),
            &LuaString::from("=(test)"),
        )
        .remove(0)
    else {
        panic!("load did not return a closure");
    };
    assert_eq!(
        vm.runtime().closure_environment(closure).unwrap(),
        environment
    );
    vm.runtime_mut()
        .table_set(
            environment,
            &Value::from("test_closure"),
            Value::Closure(closure),
        )
        .unwrap();
    assert_eq!(
        call_closure(&mut vm, Value::Closure(closure)),
        vec![Value::Number(41.0)]
    );
    let roots = Vm::gc_roots(vm.thread_environment, &mut CallStack::new(), &[]);
    assert_eq!(vm.runtime_mut().collect_full(roots).reclaimed(), 0);
}

/// A table with one field, which only the caller's handle names.
fn probe_table(vm: &mut Vm) -> TableHandle {
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("k"), Value::Integer(7))
        .unwrap();
    table
}

fn alive(vm: &Vm, table: TableHandle) -> bool {
    vm.runtime().table_get(table, &Value::from("k")).ok() == Some(Value::Integer(7))
}

#[test]
fn the_values_a_remembered_resume_holds_survive_a_collection_before_the_replay() {
    // A native trace's resume that the trace cannot take waits in the VM for
    // the interpreter's replay. Whatever it holds is reachable from nothing
    // else in that window: a success's values, the object a raised error
    // carries, the values of a host yield, and an error a hook raised.
    type Answer = fn(Value) -> crate::vm::VmResult<Vec<Value>>;
    let answers: [(&str, Answer); 4] = [
        ("success", |table| Ok(vec![Value::Boolean(true), table])),
        ("raised", |table| {
            Err(crate::vm::errors::raised_value(table))
        }),
        ("host yield", |table| {
            Err(crate::vm::VmError::Yield(vec![table]))
        }),
        ("hook error", |table| {
            Err(crate::vm::VmError::HookError(Box::new(
                crate::vm::errors::raised_value(table),
            )))
        }),
    ];
    for (name, answer) in answers {
        let mut vm = Vm::new();
        let thread = vm.create_host_coroutine().unwrap();
        let table = probe_table(&mut vm);
        vm.remember_resume(thread, answer(Value::Table(table)));
        vm.collect_full_with_values(&[]);
        assert!(alive(&vm, table), "{name}: the memo roots its values");

        vm.forget_resume();
        vm.collect_full_with_values(&[]);
        assert!(!alive(&vm, table), "{name}: a forgotten memo roots nothing");
    }
}

/// The loop of a thread that yields `y(v * 3 + 7)` and takes the answer of
/// the resume as its next `v`: eight instructions around the yield, the shape
/// the short run (`Vm::run_micro`) is made for. `y` is the thread's first
/// argument, so the chunk needs no upvalue.
fn yielding_chunk(multiplier: Constant, addend: Constant) -> Chunk {
    Chunk::with_parts(
        3,
        2,
        vec![multiplier, addend],
        vec![
            Instruction::abc(OpCode::MultiplyK, 2, 1, 0),
            Instruction::abc(OpCode::AddK, 2, 2, 1),
            Instruction::call_one(1, 0, 2),
            Instruction::aj(OpCode::Jump, -4).unwrap(),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    )
}

fn yield_function() -> Value {
    Value::BuiltinFunction(BuiltinFunction::CoroutineYield)
}

#[test]
fn a_thread_that_loops_around_its_yield_runs_the_short_run_on_every_resume() {
    for (profile, multiplier, addend, expected) in [
        (
            CompatibilityProfile::Lua51,
            Constant::Number(3.0),
            Constant::Number(7.0),
            Value::Number(22.0),
        ),
        (
            CompatibilityProfile::Lua54,
            Constant::Integer(3),
            Constant::Integer(7),
            Value::Integer(22),
        ),
    ] {
        let number = |value: i64| match profile {
            CompatibilityProfile::Lua51 => Value::Number(value as f64),
            _ => Value::Integer(value),
        };
        let mut vm = Vm::with_profile(profile);
        let entry = vm.load_chunk(&yielding_chunk(multiplier, addend)).unwrap();
        let thread = vm.create_host_coroutine().unwrap();
        // The run that starts the thread is a short run too.
        let started = vm
            .start_host_coroutine(thread, entry, vec![yield_function(), number(1)])
            .unwrap();
        assert_eq!(started, vec![Value::Boolean(true), number(10)]);
        assert_eq!(vm.jit_statistics().coroutine_short_runs, 1);
        // Every resume after it is the short run: the values are those of
        // the fast loop, and a yield hands back exactly the one it parked.
        for round in 1..=50 {
            let given = round % 9;
            let answer = vm
                .resume_host_coroutine(thread, vec![number(given)])
                .unwrap();
            assert_eq!(
                answer,
                vec![Value::Boolean(true), number(given * 3 + 7)],
                "{profile:?} round {round}"
            );
        }
        assert_eq!(vm.jit_statistics().coroutine_short_runs, 51, "{profile:?}");
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![number(5)]).unwrap()[1],
            expected,
            "{profile:?}"
        );
        assert!(!vm.quick.pending, "{profile:?}: nothing waits to be taken");
        assert!(vm.quick.values().is_empty(), "{profile:?}");
    }
}

#[test]
fn an_operand_the_short_run_does_not_know_goes_to_the_fast_loop_unchanged() {
    // A numeric string, nil and a table with a metamethod are the operands
    // of the multiplication that the short run leaves to the fast loop,
    // which coerces, raises or calls the metamethod. The thread then goes on
    // in the short run.
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua51);
    let entry = vm
        .load_chunk(&yielding_chunk(
            Constant::Number(3.0),
            Constant::Number(7.0),
        ))
        .unwrap();
    let thread = vm.create_host_coroutine().unwrap();
    vm.start_host_coroutine(thread, entry, vec![yield_function(), Value::Number(1.0)])
        .unwrap();
    let short = |vm: &Vm| vm.jit_statistics().coroutine_short_runs;
    assert_eq!(short(&vm), 1, "the start");
    assert_eq!(
        vm.resume_host_coroutine(thread, vec![Value::Number(2.0)])
            .unwrap(),
        vec![Value::Boolean(true), Value::Number(13.0)]
    );
    assert_eq!(short(&vm), 2);
    // "5" * 3 + 7 in Lua 5.1 is 22.
    assert_eq!(
        vm.resume_host_coroutine(thread, vec![Value::from("5")])
            .unwrap(),
        vec![Value::Boolean(true), Value::Number(22.0)]
    );
    assert_eq!(short(&vm), 2, "the string went through the fast loop");
    assert_eq!(
        vm.resume_host_coroutine(thread, vec![Value::Number(1.0)])
            .unwrap(),
        vec![Value::Boolean(true), Value::Number(10.0)]
    );
    assert_eq!(short(&vm), 3, "the next resume is a short run again");
    // nil raises, with the thread dead after it and a message.
    let failed = vm.resume_host_coroutine(thread, vec![Value::Nil]).unwrap();
    assert_eq!(failed[0], Value::Boolean(false));
    let Value::String(message) = &failed[1] else {
        panic!("the error is a string: {failed:?}");
    };
    assert!(
        message
            .to_string()
            .contains("attempt to perform arithmetic on"),
        "{message}"
    );
    assert_eq!(short(&vm), 3);
    assert_eq!(
        vm.host_coroutine_status(thread).unwrap(),
        HostCoroutineStatus::Dead
    );
}

#[test]
fn the_short_run_leaves_a_hooked_thread_to_the_dispatch_loop() {
    // A hook sees every instruction of the thread, so none of them is run
    // by the short run, though the hook here never fires.
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua51);
    vm.debug_hooks.insert(
        None,
        crate::vm::DebugHook {
            function: Value::Nil,
            mask: crate::vm::DebugHookMask::default(),
            count: 0,
            remaining: 0,
            running: false,
        },
    );
    vm.refresh_return_watch();
    let entry = vm
        .load_chunk(&yielding_chunk(
            Constant::Number(3.0),
            Constant::Number(7.0),
        ))
        .unwrap();
    let thread = vm.create_host_coroutine().unwrap();
    vm.start_host_coroutine(thread, entry, vec![yield_function(), Value::Number(1.0)])
        .unwrap();
    for given in [2.0, 3.0, 4.0] {
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(given)])
                .unwrap(),
            vec![Value::Boolean(true), Value::Number(given * 3.0 + 7.0)]
        );
    }
    assert_eq!(vm.jit_statistics().coroutine_short_runs, 0);
}

#[test]
fn a_yield_with_more_values_than_the_quick_buffer_holds_goes_through_a_vector() {
    // `y(v, v, v, v, v)`: five values, one more than the quick buffer.
    let chunk = Chunk::with_parts(
        8,
        2,
        Vec::new(),
        vec![
            Instruction::move_(2, 0),
            Instruction::move_(3, 1),
            Instruction::move_(4, 1),
            Instruction::move_(5, 1),
            Instruction::move_(6, 1),
            Instruction::move_(7, 1),
            Instruction::call(2, 5, 1),
            Instruction::move_(1, 2),
            Instruction::aj(OpCode::Jump, -9).unwrap(),
            Instruction::halt(),
        ],
        Vec::new(),
        Vec::new(),
    );
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua51);
    let entry = vm.load_chunk(&chunk).unwrap();
    let thread = vm.create_host_coroutine().unwrap();
    let five = |value: f64| {
        let mut values = vec![Value::Boolean(true)];
        values.extend(std::iter::repeat_n(Value::Number(value), 5));
        values
    };
    assert_eq!(
        vm.start_host_coroutine(thread, entry, vec![yield_function(), Value::Number(4.0)])
            .unwrap(),
        five(4.0)
    );
    for round in 0..20 {
        assert_eq!(
            vm.resume_host_coroutine(thread, vec![Value::Number(f64::from(round))])
                .unwrap(),
            five(f64::from(round)),
            "round {round}"
        );
    }
}

// An internal error ends a coroutine's run before the assignments of the
// normal ends; the coroutine must not stay running.
#[test]
fn a_run_cut_short_by_an_internal_error_leaves_the_coroutine_dead() {
    let mut vm = Vm::new();
    let thread = vm.create_host_coroutine().unwrap();
    vm.coroutines.get_mut(&thread).unwrap().state = CoroutineState::Running;
    vm.kill_if_running(thread);
    assert!(matches!(vm.coroutines[&thread].state, CoroutineState::Dead));
}

// The short run stops before an instruction it does not know, right after a
// backward jump. The frame must keep that jump as its last instruction, so
// the interpreter counts the landing as a backedge (`pc <= last`).
#[test]
fn the_short_run_stopping_after_a_backward_jump_keeps_the_jump_as_last() {
    let mut vm = Vm::new();
    let chunk = Arc::new(Chunk::new(
        1,
        Vec::new(),
        vec![
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::halt(),
            Instruction::aj(OpCode::Jump, -2).unwrap(),
        ],
    ));
    let mut frames = CallStack::from(vec![Frame::new(
        chunk,
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        None,
    )]);
    assert!(matches!(
        vm.run_micro(&mut frames),
        crate::vm::fast_path::Micro::Left(None)
    ));
    assert_eq!(frames[0].pc, 1);
    assert_eq!(frames[0].last_interpreted_pc.get(), Some(2));
}
