use super::*;

fn reenter_captured_callback(vm: &mut Vm, token: u64, arguments: &[Value]) -> VmResult<Vec<Value>> {
    let values = vm.call_value(arguments[0].clone(), Vec::new())?;
    if token == 1 {
        return Err(RuntimeError::message("callback failed after mutation").into());
    }
    if token == 2 {
        return Ok(vec![Value::Number(100.0)]);
    }
    if token == 3 {
        return Err(VmError::Yield(values));
    }
    Ok(values)
}

fn reentrant_capture_chunk(call: Instruction) -> Chunk {
    let increment = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::set_upvalue(0, 0),
            Instruction::return_(0),
        ],
        Vec::new(),
        vec![UpvalueDescriptor {
            name: "counter".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    Chunk::with_parts(
        4,
        0,
        vec![
            Constant::Number(0.0),
            Constant::Number(7.0),
            Constant::String("reenter".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(3, 0),
            Instruction::load_constant(0, 1),
            Instruction::get_global(1, 2),
            Instruction::move_(2, 3),
            call,
            Instruction::return_values(0, 4),
        ],
        vec![increment],
        Vec::new(),
    )
}

#[test]
fn reentrant_host_callback_preserves_captured_scalar_and_result_assignment() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        for call in [
            Instruction::call(1, 1, 2),
            Instruction::call_one(1, 1, 2),
            Instruction::call_one(0, 1, 2),
        ] {
            let mut vm = Vm::with_execution_mode(mode);
            let assigns_capture = call == Instruction::call_one(0, 1, 2);
            let host = vm
                .register_host_function(
                    reenter_captured_callback,
                    if assigns_capture { 2 } else { 0 },
                    Vec::new(),
                )
                .unwrap();
            vm.runtime_mut().set_global("reenter", host);
            let results = vm.execute(&reentrant_capture_chunk(call)).unwrap();
            let expected = if assigns_capture { 100.0 } else { 8.0 };
            assert_eq!(results[0], Value::Number(expected), "{mode:?} {call:?}");
            assert_eq!(
                vm.call_value(results[3].clone(), Vec::new()).unwrap(),
                vec![Value::Number(expected + 1.0)]
            );
        }
    }
}

#[test]
fn reentrant_host_callback_preserves_captured_scalar_on_protected_error() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_execution_mode(mode);
        let host = vm
            .register_host_function(reenter_captured_callback, 1, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("reenter", host);
        vm.runtime_mut().set_global(
            "pcall",
            Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
        );
        let mut chunk = reentrant_capture_chunk(Instruction::call(1, 2, 2));
        chunk.register_count = 5;
        chunk.constants.push(Constant::String("pcall".into()));
        chunk.code = vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(4, 0),
            Instruction::load_constant(0, 1),
            Instruction::get_global(1, 3),
            Instruction::get_global(2, 2),
            Instruction::move_(3, 4),
            Instruction::call(1, 2, 2),
            Instruction::return_values(0, 5),
        ];
        let results = vm.execute(&chunk).unwrap();
        assert_eq!(results[0], Value::Number(8.0));
        assert_eq!(results[1], Value::Boolean(false));
        assert!(
            matches!(&results[2], Value::String(message) if message.as_bytes().ends_with(b"callback failed after mutation"))
        );
        assert_eq!(
            vm.call_value(results[4].clone(), Vec::new()).unwrap(),
            vec![Value::Number(9.0)]
        );
    }
}

#[test]
fn reentrant_host_boundary_publishes_dirty_owner_before_callback() {
    let mut vm = Vm::new();
    let prototype = reentrant_capture_chunk(Instruction::call(1, 1, 1))
        .prototypes
        .remove(0);
    let frame = Frame::new(
        Arc::new(Chunk::new(1, Vec::new(), vec![Instruction::halt()])),
        vec![Value::Number(0.0)],
        Vec::new(),
        None,
        None,
        vm.runtime.global_environment(),
        None,
    );
    let mut frames = CallStack::from(vec![frame]);
    let cell = vm.capture_slot(&mut frames, 0, 0).unwrap();
    let callback = vm
        .runtime_mut()
        .create_closure(prototype, vec![cell])
        .unwrap();
    frames.at_mut(0).registers[0] = Value::Number(7.0);
    let Value::NativeFunction(host) = vm
        .register_host_function(reenter_captured_callback, 0, Vec::new())
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        vm.call_native(&mut frames, host, &[Value::Closure(callback)])
            .unwrap(),
        vec![Value::Number(8.0)]
    );
    assert_eq!(frames.at(0).registers[0], Value::Number(8.0));
    vm.close_upvalues(&mut frames, 0);
}

#[test]
fn reentrant_host_callback_preserves_captured_scalar_on_tail_call() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = Vm::with_execution_mode(mode);
        let host = vm
            .register_host_function(reenter_captured_callback, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("reenter", host);
        let mut chunk = reentrant_capture_chunk(Instruction::tail_call(1, 1));
        chunk
            .constants
            .push(Constant::String("saved_counter".into()));
        chunk.code.insert(5, Instruction::set_global(3, 3));
        *chunk.code.last_mut().unwrap() = Instruction::return_open(1, 0);
        assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(8.0)]);
        let saved = vm.runtime().get_global("saved_counter");
        assert_eq!(
            vm.call_value(saved, Vec::new()).unwrap(),
            vec![Value::Number(9.0)]
        );
    }
}

#[test]
fn reentrant_host_callback_preserves_captured_scalar_through_yield() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua51);
    let host = vm
        .register_host_function(reenter_captured_callback, 3, Vec::new())
        .unwrap();
    vm.runtime_mut().set_global("reenter", host);
    let entry = vm
        .load_chunk(&reentrant_capture_chunk(Instruction::call(1, 1, 1)))
        .unwrap();
    let thread = vm.create_host_coroutine().unwrap();
    assert_eq!(
        vm.start_host_coroutine(thread, entry, Vec::new()).unwrap(),
        vec![Value::Boolean(true), Value::Number(8.0)]
    );
    let results = vm
        .resume_host_coroutine(thread, vec![Value::Number(100.0)])
        .unwrap();
    assert_eq!(results[0], Value::Boolean(true));
    assert_eq!(results[1], Value::Number(8.0));
    assert_eq!(results[2], Value::Number(100.0));
    assert_eq!(
        vm.call_value(results[4].clone(), Vec::new()).unwrap(),
        vec![Value::Number(9.0)]
    );
}

#[test]
fn reentrant_host_callback_preserves_captured_scalar_in_hot_loop() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = if mode == ExecutionMode::Tracing {
            synchronous_tracing_vm()
        } else {
            Vm::new()
        };
        let host = vm
            .register_host_function(reenter_captured_callback, 0, Vec::new())
            .unwrap();
        vm.runtime_mut().set_global("reenter", host);
        let mut chunk = reentrant_capture_chunk(Instruction::call_one(1, 1, 2));
        chunk.register_count = 9;
        chunk
            .constants
            .extend([Constant::Number(1.0), Constant::Number(512.0)]);
        // Make the reentrant Lua callback itself execute a compiled numeric
        // loop too.
        let callback = Arc::make_mut(&mut chunk.prototypes[0]);
        callback.register_count = 7;
        callback
            .constants
            .extend([Constant::Number(32.0), Constant::Number(0.0)]);
        callback.code.splice(
            0..0,
            [
                Instruction::load_constant(2, 0),
                Instruction::load_constant(3, 1),
                Instruction::load_constant(4, 0),
                Instruction::load_constant(6, 2),
                Instruction::numeric_for_prep(2, 1),
                Instruction::binary(OpCode::Add, 6, 6, 5),
                Instruction::numeric_for_loop(2, -2),
            ],
        );
        chunk.code = vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(3, 0),
            Instruction::load_constant(0, 1),
            Instruction::load_constant(4, 3),
            Instruction::load_constant(5, 4),
            Instruction::load_constant(6, 3),
            Instruction::load_constant(8, 0),
            Instruction::numeric_for_prep(4, 4),
            Instruction::get_global(1, 2),
            Instruction::move_(2, 3),
            Instruction::call_one(1, 1, 2),
            Instruction::binary(OpCode::Add, 8, 8, 7),
            Instruction::numeric_for_loop(4, -5),
            Instruction::return_values(0, 4),
        ];
        let results = vm.execute(&chunk).unwrap();
        assert_eq!(results[0], Value::Number(519.0));
        assert_eq!(
            vm.call_value(results[3].clone(), Vec::new()).unwrap(),
            vec![Value::Number(520.0)]
        );
        if mode == ExecutionMode::Tracing {
            assert!(
                vm.jit_statistics().trace.native_entries > 0,
                "{:?}",
                vm.jit_statistics()
            );
        }
    }
}

#[test]
fn close_local_runs_lua_closure_handler_before_continuing() {
    let mut vm = Vm::new();
    let handler = Arc::new(Chunk::new(
        3,
        vec![Constant::Boolean(true), Constant::String("closed".into())],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::set_global(2, 1),
            Instruction::return_(0),
        ],
    ));
    let closure = vm
        .runtime_mut()
        .create_closure(handler, Vec::new())
        .unwrap();
    let resource = vm.runtime_mut().create_table().unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(metatable, &Value::from("__close"), Value::Closure(closure))
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(resource), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("resource", Value::Table(resource));

    let caller = Chunk::new(
        2,
        vec![
            Constant::String("resource".into()),
            Constant::String("closed".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::mark_close(0),
            Instruction::close_local(0),
            Instruction::get_global(1, 1),
            Instruction::return_(1),
        ],
    );

    assert_eq!(vm.execute(&caller).unwrap(), vec![Value::Boolean(true)]);
}

#[test]
fn close_locals_run_lua_handlers_in_reverse_declaration_order() {
    fn record_close(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let [Value::Number(value)] = arguments else {
            return Err(RuntimeError::message(
                "close handler received wrong arguments",
            ));
        };
        CLOSE_HANDLER_ORDER.lock().unwrap().push(*value);
        Ok(Vec::new())
    }

    CLOSE_HANDLER_ORDER.lock().unwrap().clear();
    let mut vm = Vm::new();
    let record = vm.runtime_mut().register_native(record_close).unwrap();
    vm.runtime_mut()
        .set_global("record_close", Value::NativeFunction(record));
    let make_handler = |order| {
        Arc::new(Chunk::new(
            4,
            vec![
                Constant::String("record_close".into()),
                Constant::Number(order),
            ],
            vec![
                Instruction::get_global(2, 0),
                Instruction::load_constant(3, 1),
                Instruction::call(2, 1, 0),
                Instruction::return_(0),
            ],
        ))
    };
    let first_handler = vm
        .runtime_mut()
        .create_closure(make_handler(1.0), Vec::new())
        .unwrap();
    let second_handler = vm
        .runtime_mut()
        .create_closure(make_handler(2.0), Vec::new())
        .unwrap();
    let mut make_resource = |handler| {
        let resource = vm.runtime_mut().create_table().unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(metatable, &Value::from("__close"), Value::Closure(handler))
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(resource), Some(metatable))
            .unwrap();
        resource
    };
    let first = make_resource(first_handler);
    let second = make_resource(second_handler);
    vm.runtime_mut().set_global("first", Value::Table(first));
    vm.runtime_mut().set_global("second", Value::Table(second));

    let caller = Chunk::new(
        2,
        vec![
            Constant::String("first".into()),
            Constant::String("second".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::mark_close(0),
            Instruction::get_global(1, 1),
            Instruction::mark_close(1),
            Instruction::close_local(1),
            Instruction::close_local(0),
            Instruction::return_(0),
        ],
    );

    vm.execute(&caller).unwrap();
    assert_eq!(*CLOSE_HANDLER_ORDER.lock().unwrap(), vec![2.0, 1.0]);
}

#[test]
fn call_one_clones_aliased_inputs_and_uses_call_metamethods() {
    fn call_metamethod(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let [Value::Table(_), Value::Number(value)] = arguments else {
            return Err(RuntimeError::message("__call received the wrong values"));
        };
        Ok(vec![Value::Number(value + 1.0)])
    }

    let echo = Chunk::with_parts(1, 1, vec![], vec![Instruction::return_(0)], vec![], vec![]);
    let aliasing = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(41.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::call_one(0, 0, 1),
            Instruction::return_(0),
        ],
        vec![echo],
        vec![],
    );
    let mut vm = Vm::new();
    assert_eq!(vm.execute(&aliasing).unwrap(), vec![Value::Number(41.0)]);

    let callable = vm.runtime_mut().create_table().unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let handler = vm.runtime_mut().register_native(call_metamethod).unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__call"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(callable), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("call_one_callable", Value::Table(callable));
    let metamethod = Chunk::new(
        3,
        vec![
            Constant::String("call_one_callable".into()),
            Constant::Number(9.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call_one(2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(vm.execute(&metamethod).unwrap(), vec![Value::Number(10.0)]);
}

#[test]
fn retryable_forwarding_refusal_is_reconsidered_after_capture_rebinding() {
    let mut vm = Vm::new();
    let unsupported_target = vm
        .runtime_mut()
        .create_closure(
            Arc::new(Chunk::with_parts(
                1,
                1,
                vec![Constant::String("not numeric".into())],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
                Vec::new(),
                Vec::new(),
            )),
            Vec::new(),
        )
        .unwrap();
    let target_cell = vm
        .runtime_mut()
        .create_cell(Value::Closure(unsupported_target))
        .unwrap();
    let wrapper = vm
        .runtime_mut()
        .create_closure(
            Arc::new(Chunk::with_varargs(
                2,
                0,
                Vec::new(),
                vec![
                    Instruction::get_upvalue(0, 0),
                    Instruction::tail_call_vararg(0, 0),
                    Instruction::return_open(0, 0),
                    Instruction::halt(),
                ],
                Vec::new(),
                vec![crate::bytecode::UpvalueDescriptor {
                    name: "target".into(),
                    source: crate::bytecode::UpvalueSource::Detached,
                }],
                None,
            )),
            vec![target_cell],
        )
        .unwrap();

    assert!(vm.inline_call_plan(wrapper).is_none());
    assert!(vm.jit.inline_call_retryable_refusals.contains(&wrapper));
    assert_ne!(
        vm.jit
            .inline_call_hot_cache
            .as_ref()
            .map(|(cached, _)| *cached),
        Some(wrapper)
    );

    let supported_target = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    vm.runtime_mut()
        .cell_set(target_cell, Value::Closure(supported_target))
        .unwrap();

    let plan = vm
        .inline_call_plan(wrapper)
        .expect("the forwarding wrapper should be reconsidered after rebinding");
    assert!(plan.forwarded.is_some());
    assert_eq!(plan.plan.evaluate(41.0), 42.0);
}

#[test]
fn permanent_refusal_hot_cache_preserves_stale_closure_fallback_error() {
    let mut vm = Vm::new();
    let stale = vm
        .runtime_mut()
        .create_closure(
            Arc::new(Chunk::with_parts(
                1,
                1,
                vec![Constant::String("not numeric".into())],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
                Vec::new(),
                Vec::new(),
            )),
            Vec::new(),
        )
        .unwrap();
    assert!(vm.inline_call_plan(stale).is_none());
    assert!(matches!(
        vm.jit.inline_call_hot_cache,
        Some((cached, None)) if cached == stale
    ));

    vm.runtime_mut().collect_full(GcRootBuilder::new().finish());
    let error = vm
        .execute_with_arguments(
            &one_argument_call_one_parent(1),
            vec![Value::Closure(stale), Value::Number(1.0)],
        )
        .unwrap_err();
    assert!(matches!(error, VmError::MissingClosureBody(handle) if handle == stale));
}

#[test]
fn call_one_uses_the_capture_free_numeric_interpreter_plan_and_preserves_aliases() {
    let parent = one_argument_call_one_parent(1);
    let mut vm = Vm::new();
    let increment = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    assert_eq!(
        vm.execute_with_arguments(
            &parent,
            vec![Value::Closure(increment), Value::Number(41.0)],
        )
        .unwrap(),
        vec![Value::Number(42.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 1);

    // The result may overwrite the callee as well. This call is still
    // exact because the fast evaluator snapshots function and argument
    // before writing the caller destination.
    let callee_alias_parent = one_argument_call_one_parent(0);
    let increment = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    assert_eq!(
        vm.execute_with_arguments(
            &callee_alias_parent,
            vec![Value::Closure(increment), Value::Number(9.0)],
        )
        .unwrap(),
        vec![Value::Number(10.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 2);
}

#[test]
fn open_upvalues_are_the_registers_they_stand_for_until_closed() {
    let mut vm = Vm::new();
    let frame = Frame::new(
        Arc::new(Chunk::new(2, Vec::new(), vec![Instruction::halt()])),
        vec![Value::Number(1.0), Value::Number(2.0)],
        Vec::new(),
        None,
        None,
        vm.runtime.global_environment(),
        None,
    );
    let mut frames = CallStack::from(vec![frame]);
    let first = vm.capture_slot(&mut frames, 0, 0).unwrap();
    let second = vm.capture_slot(&mut frames, 0, 1).unwrap();
    // The same register names the same cell.
    assert_eq!(vm.capture_slot(&mut frames, 0, 0).unwrap(), first);
    assert_eq!(frames.open_upvalues().len(), 2);

    // A register write is visible through the cell, and the other way.
    frames.at_mut(0).registers[0] = Value::Number(3.0);
    assert_eq!(vm.runtime.cell_get(first).unwrap(), Value::Number(3.0));
    vm.runtime.cell_set(second, Value::Number(9.0)).unwrap();
    assert_eq!(
        frames.at(0).registers,
        vec![Value::Number(3.0), Value::Number(9.0)]
    );

    // Closing keeps the value in the cell and cuts it off from the stack.
    vm.close_upvalues(&mut frames, 0);
    assert!(frames.open_upvalues().is_empty());
    assert_eq!(frames.open_top(), 0);
    frames.at_mut(0).registers[0] = Value::Number(100.0);
    assert_eq!(vm.runtime.cell_get(first).unwrap(), Value::Number(3.0));
    assert_eq!(vm.runtime.cell_get(second).unwrap(), Value::Number(9.0));
}

#[test]
fn closing_from_a_level_leaves_the_lower_upvalues_open() {
    let mut vm = Vm::new();
    let frame = Frame::new(
        Arc::new(Chunk::new(3, Vec::new(), vec![Instruction::halt()])),
        vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)],
        Vec::new(),
        None,
        None,
        vm.runtime.global_environment(),
        None,
    );
    let mut frames = CallStack::from(vec![frame]);
    let cells: Vec<_> = (0..3)
        .map(|register| vm.capture_slot(&mut frames, 0, register).unwrap())
        .collect();
    vm.close_upvalues(&mut frames, 1);
    assert_eq!(frames.open_upvalues().len(), 1);
    assert!(vm.runtime.cell_open_slot(cells[0]).is_some());
    assert!(vm.runtime.cell_open_slot(cells[1]).is_none());
    assert!(vm.runtime.cell_open_slot(cells[2]).is_none());
    vm.close_upvalues(&mut frames, 0);
}

#[test]
fn fixed_one_argument_closure_call_returns_directly_to_its_caller() {
    let increment = Chunk::with_parts(
        2,
        1,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(41.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
        vec![increment],
        vec![],
    );

    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(42.0)]
    );
}

#[test]
fn fixed_zero_argument_closure_call_returns_directly_to_its_caller() {
    let answer = Chunk::with_parts(
        1,
        0,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        vec![],
        vec![],
    );
    let program = Chunk::with_parts(
        1,
        0,
        Vec::new(),
        vec![
            Instruction::closure(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
        vec![answer],
        vec![],
    );

    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(42.0)]
    );
}

#[test]
fn fixed_one_argument_frame_has_no_vararg_tail() {
    let runtime = Runtime::default();
    let frame = Frame::new_fixed_one_argument(
        Arc::new(Chunk::with_parts(
            1,
            1,
            Vec::new(),
            vec![Instruction::return_(0)],
            vec![],
            vec![],
        )),
        Value::Number(9.0),
        Vec::new(),
        None,
        None,
        runtime.global_environment(),
        None,
    );

    assert_eq!(frame.registers, vec![Value::Number(9.0)]);
    assert_eq!(frame.varargs, [] as [Value; 0]);
}

#[test]
fn frame_assigns_fixed_arguments_and_preserves_the_vararg_tail() {
    let runtime = Runtime::default();
    let frame = Frame::new(
        Arc::new(Chunk::with_varargs(
            4,
            2,
            Vec::new(),
            vec![Instruction::return_(0)],
            vec![],
            vec![],
            None,
        )),
        vec![
            Value::Number(1.0),
            Value::Nil,
            Value::from("third"),
            Value::Boolean(false),
        ],
        Vec::new(),
        None,
        None,
        runtime.global_environment(),
        None,
    );

    assert_eq!(frame.registers[0], Value::Number(1.0));
    assert_eq!(frame.registers[1], Value::Nil);
    assert_eq!(
        frame.varargs,
        vec![Value::from("third"), Value::Boolean(false)]
    );

    let missing_argument_frame = Frame::new(
        Arc::new(Chunk::with_varargs(
            2,
            2,
            Vec::new(),
            vec![Instruction::return_(0)],
            vec![],
            vec![],
            None,
        )),
        vec![Value::Number(1.0)],
        Vec::new(),
        None,
        None,
        runtime.global_environment(),
        None,
    );
    assert_eq!(missing_argument_frame.registers[0], Value::Number(1.0));
    assert_eq!(missing_argument_frame.registers[1], Value::Nil);
    assert_eq!(missing_argument_frame.varargs, [] as [Value; 0]);
}

#[test]
fn call_errors_use_static_callee_names_but_not_pcall_names() {
    let named = Chunk::new(
        1,
        vec![Constant::Number(2.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::halt(),
        ],
    )
    .with_debug_metadata(DebugMetadata::new(
        "named.lua",
        vec![CallSite::named(1, "bbbb", CallKind::Global)],
    ));
    let Err(VmError::Runtime(error)) = Vm::new().execute(&named) else {
        panic!("named non-function call fails at runtime");
    };
    assert_eq!(
        error,
        RuntimeError::message("attempt to call global 'bbbb' (a number value)")
    );

    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    let protected = Chunk::new(
        2,
        vec![Constant::String("pcall".into()), Constant::Number(2.0)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    )
    .with_debug_metadata(DebugMetadata::new(
        "protected.lua",
        vec![CallSite::named(2, "pcall", CallKind::Global)],
    ));
    assert_eq!(
        vm.execute(&protected).unwrap(),
        vec![
            Value::Boolean(false),
            Value::from("attempt to call a number value")
        ]
    );

    let named_index =
        Chunk::new(
            3,
            vec![Constant::String("field".into())],
            vec![
                Instruction::load_nil(0),
                Instruction::load_constant(1, 0),
                Instruction::get_table(2, 0, 1),
                Instruction::halt(),
            ],
        )
        .with_debug_metadata(
            DebugMetadata::new("index.lua", vec![])
                .with_diagnostic_sites(vec![DiagnosticSite::new(2, 0, "source", CallKind::Global)]),
        );
    let Err(VmError::Runtime(error)) = Vm::new().execute(&named_index) else {
        panic!("named nil index fails at runtime");
    };
    assert_eq!(
        error.to_string(),
        "attempt to index global 'source' (a nil value)"
    );

    let named_arithmetic = Chunk::new(
        3,
        vec![Constant::String("x".into()), Constant::Number(1.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::halt(),
        ],
    )
    .with_debug_metadata(
        DebugMetadata::new("arithmetic.lua", vec![])
            .with_diagnostic_sites(vec![DiagnosticSite::new(2, 0, "value", CallKind::Local)]),
    );
    let Err(VmError::Runtime(error)) = Vm::new().execute(&named_arithmetic) else {
        panic!("named string arithmetic fails at runtime");
    };
    assert_eq!(
        error.to_string(),
        "attempt to perform arithmetic on local 'value' (a string value)"
    );
}

#[test]
fn executes_arithmetic_and_globals() {
    let chunk = Chunk::new(
        3,
        vec![
            Constant::Integer(2),
            Constant::Integer(40),
            Constant::String("answer".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::set_global(2, 2),
            Instruction::return_(2),
        ],
    );
    let mut vm = Vm::new();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(42.0)]);
    assert_eq!(vm.runtime().get_global("answer"), Value::Number(42.0));
}

#[test]
fn protected_file_method_errors_resume_the_caller() {
    let chunk = Chunk::new(
        4,
        vec![
            Constant::String("pcall".into()),
            Constant::String("file".into()),
            Constant::String("read".into()),
            Constant::String("*a".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(2, 1),
            Instruction::load_constant(3, 2),
            Instruction::get_table(1, 2, 3),
            Instruction::load_constant(3, 3),
            Instruction::call(0, 3, 2),
            Instruction::return_values(0, 2),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let file = vm.runtime_mut().create_temp_file().unwrap();
    vm.runtime_mut().file_close(file).unwrap();
    vm.runtime_mut().set_global("file", Value::File(file));
    let values = vm.execute(&chunk).unwrap();
    assert!(
        matches!(values.as_slice(), [Value::Boolean(false), Value::String(message)] if message.as_bytes().windows(b"closed file".len()).any(|window| window == b"closed file"))
    );
}

#[test]
fn executes_unary_arithmetic_comparisons_and_jumps() {
    let chunk = Chunk::new(
        3,
        vec![Constant::Integer(8), Constant::Integer(2)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::unary(OpCode::Negate, 2, 1),
            Instruction::binary(OpCode::Equal, 2, 0, 0),
            Instruction::binary(OpCode::LessEqual, 2, 1, 0),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::binary(OpCode::Subtract, 2, 2, 1),
            Instruction::binary(OpCode::Multiply, 2, 2, 1),
            Instruction::binary(OpCode::Divide, 2, 2, 1),
            Instruction::binary(OpCode::LessThan, 2, 1, 0),
            Instruction::unary(OpCode::Not, 2, 2),
            Instruction::jump_if_false(2, 1),
            Instruction::load_nil(0),
            Instruction::aj(OpCode::Jump, 0).unwrap(),
            Instruction::return_(0),
        ],
    );
    assert_eq!(Vm::new().execute(&chunk).unwrap(), vec![Value::Number(8.0)]);
}

#[test]
fn zero_argument_calls_do_not_form_an_invalid_register_slice() {
    fn zero_argument_native(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        runtime.create_table()?;
        Ok(vec![Value::Number(99.0)])
    }

    let chunk = Chunk::new(
        1,
        vec![Constant::String("zero".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("zero", zero_argument_native)
        .unwrap();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(99.0)]);
}

#[test]
fn closures_pass_explicit_self_and_parameters_through_nested_calls() {
    let getter = Chunk::with_parts(
        3,
        2,
        vec![],
        vec![Instruction::get_table(2, 0, 1), Instruction::return_(2)],
        vec![],
        vec![],
    );
    let method = Chunk::with_parts(
        5,
        2,
        vec![],
        vec![
            Instruction::closure(2, 0),
            Instruction::move_(3, 0),
            Instruction::move_(4, 1),
            Instruction::call(2, 2, 1),
            Instruction::return_(2),
        ],
        vec![getter],
        vec![],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::String("value".into()),
            Constant::String("hello".into()),
            Constant::String("greet".into()),
        ],
        vec![
            Instruction::new_table(0, 0, 2),
            Instruction::load_constant(2, 0),
            Instruction::load_constant(3, 1),
            Instruction::set_table(0, 2, 3),
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 2),
            Instruction::set_table(0, 2, 1),
            Instruction::get_table(1, 0, 2),
            Instruction::move_(2, 0),
            Instruction::load_constant(3, 0),
            Instruction::call(1, 2, 1),
            Instruction::return_(1),
        ],
        vec![method],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::from("hello")]
    );
}

#[test]
fn closures_read_open_and_closed_parent_slots() {
    let captured = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(1.0), Constant::Number(2.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::load_constant(0, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_(1),
        ],
        vec![captured],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(2.0)]
    );
}

#[test]
fn captured_mutation_survives_caller_return_and_repeated_calls() {
    let increment = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::set_upvalue(0, 0),
            Instruction::return_(0),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        3,
        0,
        vec![Constant::Number(7.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::move_(2, 1),
            Instruction::call(1, 0, 1),
            Instruction::call(2, 0, 1),
            Instruction::return_(0),
        ],
        vec![increment],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(9.0)]
    );
}

/// Caller registers 1..=3 are captured by a reader closure. A call based at
/// register 1 puts two results directly into registers 1 and 2, and the
/// callee also writes the captured register 3 through its own upvalue. After
/// the call the caller reads all three directly and through the closure.
fn two_captured_results_program(callee: Chunk) -> Chunk {
    let capture = |name: &str, register: u8| UpvalueDescriptor {
        name: name.into(),
        source: UpvalueSource::ParentRegister(register),
    };
    let reader = Chunk::with_parts(
        3,
        0,
        vec![],
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::get_upvalue(1, 1),
            Instruction::get_upvalue(2, 2),
            Instruction::return_values(0, 3),
        ],
        vec![],
        vec![capture("a", 1), capture("b", 2), capture("c", 3)],
    );
    // The callee captures register 3 only; its proto is index 0, the
    // reader's is index 1.
    Chunk::with_parts(
        7,
        0,
        vec![Constant::Number(2.0), Constant::Number(3.0)],
        vec![
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 0),
            Instruction::load_constant(3, 1),
            Instruction::closure(0, 1),
            // Results land in registers 1 and 2.
            Instruction::call(1, 0, 2),
            Instruction::move_(4, 0),
            Instruction::call(4, 0, 3),
            Instruction::return_values(1, 6),
        ],
        vec![callee, reader],
        vec![],
    )
}

fn expected_two_captured_results() -> Vec<Value> {
    [10.0, 20.0, 100.0, 10.0, 20.0, 100.0]
        .into_iter()
        .map(Value::Number)
        .collect()
}

#[test]
fn fast_return_flushes_two_captured_result_registers() {
    let callee = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::Number(100.0),
            Constant::Number(10.0),
            Constant::Number(20.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::set_upvalue(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::return_values(1, 2),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "c".into(),
            source: UpvalueSource::ParentRegister(3),
        }],
    );
    assert_eq!(
        Vm::new()
            .execute(&two_captured_results_program(callee))
            .unwrap(),
        expected_two_captured_results()
    );
}

#[test]
fn fast_return_open_flushes_two_captured_result_registers() {
    // The callee returns one fixed value followed by the open results of an
    // inner call, so it leaves through ReturnOpen with fixed results.
    let inner = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(20.0), Constant::Number(21.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::return_values(0, 2),
        ],
        vec![],
        vec![],
    );
    let callee = Chunk::with_parts(
        3,
        0,
        vec![Constant::Number(100.0), Constant::Number(10.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::set_upvalue(0, 0),
            Instruction::load_constant(0, 1),
            Instruction::closure(1, 0),
            Instruction::call(1, 0, 0),
            Instruction::return_open(0, 1),
        ],
        vec![inner],
        vec![UpvalueDescriptor {
            name: "c".into(),
            source: UpvalueSource::ParentRegister(3),
        }],
    );
    assert_eq!(
        Vm::new()
            .execute(&two_captured_results_program(callee))
            .unwrap(),
        expected_two_captured_results()
    );
}

#[test]
fn returned_closure_keeps_a_closed_parent_cell_alive() {
    let child = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let maker = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(12.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::return_(1),
        ],
        vec![child],
        vec![],
    );
    let mut vm = Vm::new();
    let closure = vm.execute(&maker).unwrap().pop().unwrap();
    vm.runtime_mut().set_global("saved", closure);
    let call = Chunk::new(
        1,
        vec![Constant::String("saved".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&call).unwrap(), vec![Value::Number(12.0)]);
}

#[test]
fn tail_call_closes_a_local_captured_by_its_returned_result() {
    let captured = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "captured".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let return_argument =
        Chunk::with_parts(1, 1, vec![], vec![Instruction::return_(0)], vec![], vec![]);
    let tailer = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(42.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(3, 0),
            Instruction::closure(2, 1),
            Instruction::tail_call(2, 1),
            Instruction::return_open(2, 0),
        ],
        vec![captured, return_argument],
        vec![],
    );
    let program = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![
            Instruction::closure(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
        vec![tailer],
        vec![],
    );
    let mut vm = Vm::new();
    let closure = vm.execute(&program).unwrap().pop().unwrap();
    vm.runtime_mut().set_global("captured_tail_result", closure);
    let call = Chunk::new(
        1,
        vec![Constant::String("captured_tail_result".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&call).unwrap(), vec![Value::Number(42.0)]);
}

#[test]
fn tail_call_close_error_preserves_the_detached_tail_target_behavior() {
    fn failing_close(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Err(RuntimeError::message("tail close failed"))
    }

    let mut vm = Vm::new();
    let close = vm.runtime_mut().register_native(failing_close).unwrap();
    let resource = vm.runtime_mut().create_table().unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__close"),
            Value::NativeFunction(close),
        )
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(resource), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("tail_resource", Value::Table(resource));
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );

    let target = Chunk::new(
        1,
        vec![Constant::Number(7.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let tailer = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("tail_resource".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::mark_close(0),
            Instruction::closure(1, 0),
            Instruction::tail_call(1, 0),
            Instruction::return_open(1, 0),
        ],
        vec![target],
        vec![],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("pcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
        vec![tailer],
        vec![],
    );
    let error = vm.execute(&program).unwrap_err();
    assert!(error.to_string().contains("tail close failed"));
}

#[test]
fn tail_calls_resize_reused_registers_for_smaller_and_larger_callees() {
    let small = Chunk::new(
        1,
        vec![Constant::Number(3.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let large = Chunk::new(
        4,
        vec![Constant::Number(9.0)],
        vec![Instruction::load_constant(3, 0), Instruction::return_(3)],
    );
    let larger_caller = Chunk::with_parts(
        4,
        0,
        vec![],
        vec![
            Instruction::closure(0, 0),
            Instruction::tail_call(0, 0),
            Instruction::return_open(0, 0),
        ],
        vec![small],
        vec![],
    );
    let smaller_caller = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![
            Instruction::closure(0, 0),
            Instruction::tail_call(0, 0),
            Instruction::return_open(0, 0),
        ],
        vec![large],
        vec![],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![],
        vec![
            Instruction::closure(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::closure(1, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_values(0, 2),
        ],
        vec![larger_caller, smaller_caller],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(3.0), Value::Number(9.0)]
    );
}

#[test]
fn fixed_tail_call_moves_a_nonzero_window_and_fills_missing_parameters() {
    let target = Chunk::with_parts(
        3,
        3,
        Vec::new(),
        vec![Instruction::return_values(0, 3)],
        Vec::new(),
        Vec::new(),
    );
    let tailer = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(11.0), Constant::Number(22.0)],
        vec![
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 0),
            Instruction::load_constant(3, 1),
            Instruction::tail_call(1, 2),
            Instruction::return_open(1, 0),
        ],
        vec![target],
        Vec::new(),
    );
    let program = Chunk::with_parts(
        3,
        0,
        Vec::new(),
        vec![
            Instruction::closure(0, 0),
            Instruction::call(0, 0, 3),
            Instruction::return_values(0, 3),
        ],
        vec![tailer],
        Vec::new(),
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(11.0), Value::Number(22.0), Value::Nil]
    );
}

#[test]
fn protected_calls_return_status_results_and_resume_the_caller() {
    let echo = Chunk::with_parts(
        2,
        2,
        vec![],
        vec![Instruction::return_values(0, 2)],
        vec![],
        vec![],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::Number(4.0),
            Constant::Number(5.0),
            Constant::String("pcall".into()),
        ],
        vec![
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 0),
            Instruction::load_constant(3, 1),
            Instruction::get_global(0, 2),
            Instruction::call(0, 3, 3),
            Instruction::binary(OpCode::Add, 3, 1, 2),
            Instruction::return_(3),
        ],
        vec![echo],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(9.0)]);
}

#[test]
fn protected_calls_preserve_nil_error_objects() {
    let raise_nil = Chunk::new(
        2,
        vec![Constant::String("error".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_nil(1),
            Instruction::call(0, 1, 0),
            Instruction::return_values(0, 0),
        ],
    );
    let nested = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("pcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
        vec![raise_nil],
        vec![],
    );
    let direct = Chunk::new(
        2,
        vec![
            Constant::String("pcall".into()),
            Constant::String("error".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(
        vm.execute(&direct).unwrap(),
        vec![Value::Boolean(false), Value::Nil]
    );
    assert_eq!(
        vm.execute(&nested).unwrap(),
        vec![Value::Boolean(false), Value::Nil]
    );
}

#[test]
fn extended_protected_calls_use_lua_handlers_and_contain_handler_failures() {
    let raise = Chunk::new(
        3,
        vec![
            Constant::String("error".into()),
            Constant::String("original".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::return_values(0, 0),
        ],
    );
    let identity = Chunk::with_parts(1, 1, vec![], vec![Instruction::return_(0)], vec![], vec![]);
    let handler_failure = Chunk::new(
        3,
        vec![
            Constant::String("error".into()),
            Constant::String("handler failure".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::return_values(0, 0),
        ],
    );
    let program = Chunk::with_parts(
        3,
        0,
        vec![Constant::String("xpcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::closure(2, 1),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
        vec![raise.clone(), identity],
        vec![],
    );
    let handler_error_program = Chunk::with_parts(
        3,
        0,
        vec![Constant::String("xpcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::closure(2, 1),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
        vec![raise, handler_failure],
        vec![],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Boolean(false), Value::from("original")]
    );
    assert_eq!(
        vm.execute(&handler_error_program).unwrap(),
        vec![
            Value::Boolean(false),
            Value::from("error in error handling")
        ]
    );
}

#[test]
fn protected_calls_also_catch_vm_builtin_errors() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("present"), Value::Number(1.0))
        .unwrap();
    vm.runtime_mut().set_global("table", Value::Table(table));
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    vm.runtime_mut()
        .set_global("next", Value::BuiltinFunction(BuiltinFunction::Next));
    let chunk = Chunk::new(
        4,
        vec![
            Constant::String("pcall".into()),
            Constant::String("next".into()),
            Constant::String("table".into()),
            Constant::String("missing".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::call(0, 3, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&chunk).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));
}

#[test]
fn vararg_copy_pads_and_return_vararg_does_not_leak_registers() {
    let padded = Chunk::with_varargs(
        4,
        1,
        vec![],
        vec![Instruction::vararg(1, 3), Instruction::return_values(1, 3)],
        vec![],
        vec![],
        None,
    );
    let program = Chunk::with_parts(
        5,
        0,
        vec![
            Constant::Number(7.0),
            Constant::Number(8.0),
            Constant::Number(9.0),
        ],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::load_nil(3),
            Instruction::load_constant(4, 2),
            Instruction::call(0, 4, 3),
            Instruction::return_values(0, 3),
        ],
        vec![padded],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(8.0), Value::Nil, Value::Number(9.0)]
    );

    let returns_only_extras = Chunk::with_varargs(
        3,
        0,
        vec![
            Constant::Number(40.0),
            Constant::Number(41.0),
            Constant::Number(99.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::return_vararg(2, 0),
        ],
        vec![],
        vec![],
        None,
    );
    let program = Chunk::with_parts(
        3,
        0,
        vec![Constant::Number(4.0), Constant::Number(5.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
        vec![returns_only_extras],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(4.0), Value::Number(5.0)]
    );

    let returns_prefix_then_extras = Chunk::with_varargs(
        1,
        0,
        vec![Constant::Number(99.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::return_vararg(0, 1),
        ],
        vec![],
        vec![],
        None,
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(4.0), Constant::Number(5.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::call(0, 2, 3),
            Instruction::return_values(0, 3),
        ],
        vec![returns_prefix_then_extras],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(99.0), Value::Number(4.0), Value::Number(5.0)]
    );
}

#[test]
fn call_vararg_forwards_to_closures_and_protected_calls() {
    let echo = Chunk::with_varargs(
        1,
        0,
        vec![],
        vec![Instruction::return_vararg(0, 0)],
        vec![],
        vec![],
        None,
    );
    let forward = Chunk::with_varargs(
        3,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::String("echo".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call_vararg(0, 1, 3),
            Instruction::return_values(0, 3),
        ],
        vec![],
        vec![],
        None,
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(4.0), Constant::Number(5.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::call(0, 2, 3),
            Instruction::return_values(0, 3),
        ],
        vec![forward],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    let echo_handle = vm
        .install_closure(echo, Vec::new(), vm.runtime.global_environment())
        .unwrap();
    vm.runtime_mut()
        .set_global("echo", Value::Closure(echo_handle));
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Boolean(true), Value::Number(4.0), Value::Number(5.0)]
    );
}

#[test]
fn frames_retain_more_than_255_varargs() {
    let chunk = Chunk::with_varargs(
        2,
        1,
        vec![],
        vec![Instruction::halt()],
        vec![],
        vec![],
        Some(1),
    );
    let mut arguments = vec![Value::Number(0.0)];
    arguments.extend((1..=300).map(|value| Value::Number(f64::from(value))));
    let mut vm = Vm::new();
    let frame = vm
        .make_frame(
            chunk,
            arguments,
            Vec::new(),
            None,
            None,
            vm.runtime().global_environment(),
            None,
        )
        .unwrap();
    assert_eq!(frame.varargs.len(), 300);
    let Value::Table(table) = &frame.registers[1] else {
        panic!("missing compatibility arg table");
    };
    assert_eq!(
        vm.runtime().table_get(*table, &Value::from("n")).unwrap(),
        Value::Number(300.0)
    );
    assert_eq!(
        vm.runtime()
            .table_get(*table, &Value::Number(300.0))
            .unwrap(),
        Value::Number(300.0)
    );
}

#[test]
fn open_results_preserve_closure_and_protected_call_results() {
    let echo = Chunk::with_parts(
        2,
        2,
        vec![],
        vec![Instruction::return_values(0, 2)],
        vec![],
        vec![],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::Number(4.0),
            Constant::Number(5.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::load_constant(3, 2),
            Instruction::call(0, 3, 0),
            Instruction::return_open(0, 0),
        ],
        vec![echo],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Boolean(true), Value::Number(4.0), Value::Number(5.0)]
    );
}

#[test]
fn inline_call_with_fewer_arguments_than_the_plan_minimum_falls_back() {
    // Two parameters plus a fixed vararg read: the plan needs three values.
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
    let parent = |arguments: u8| {
        Chunk::with_parts(
            5,
            4,
            Vec::new(),
            vec![Instruction::call(0, arguments, 1), Instruction::return_(0)],
            Vec::new(),
            Vec::new(),
        )
    };
    let mut vm = Vm::new();
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(callee), Vec::new())
        .unwrap();
    let plan = vm.inline_call_plan(closure).expect("the callee has a plan");
    assert_eq!(plan.plan.minimum_argument_count(), 3);

    // Two arguments: the plan must decline and the ordinary call path runs,
    // where the missing vararg is nil.
    let error = vm
        .execute_with_arguments(
            &parent(2),
            vec![
                Value::Closure(closure),
                Value::Number(1.0),
                Value::Number(2.0),
            ],
        )
        .unwrap_err();
    assert!(error.to_string().contains("arithmetic"), "{error}");
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 0);

    // Three arguments meet the minimum. The fast loop calls a vararg body with
    // a plain activation (its varargs stay on the value stack), which beats
    // evaluating the plan, so the plan is left for the general routine.
    assert_eq!(
        vm.execute_with_arguments(
            &parent(3),
            vec![
                Value::Closure(closure),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(4.0),
            ],
        )
        .unwrap(),
        vec![Value::Number(7.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 0);
}

#[test]
fn inline_table_call_that_raises_reports_the_ordinary_index_error() {
    let method = Chunk::with_parts(
        5,
        1,
        vec![
            Constant::String("value".into()),
            Constant::Number(2.0),
            Constant::Number(3.0),
        ],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::set_table(0, 1, 2),
            Instruction::get_table(3, 0, 1),
            Instruction::load_constant(4, 2),
            Instruction::binary(OpCode::Add, 3, 3, 4),
            Instruction::set_table(0, 1, 3),
            Instruction::return_(3),
        ],
        Vec::new(),
        Vec::new(),
    );
    let call = Chunk::with_parts(
        2,
        2,
        Vec::new(),
        vec![Instruction::call(0, 1, 1), Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
    );
    let mut vm = Vm::new();
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(method), Vec::new())
        .unwrap();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("value"), Value::Number(5.0))
        .unwrap();
    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(closure), Value::Table(table)])
            .unwrap(),
        vec![Value::Number(5.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_table_calls, 1);

    // A non-table argument makes the plan decline; the ordinary path then
    // raises the usual indexing error rather than a plan-specific one.
    let error = vm
        .execute_with_arguments(&call, vec![Value::Closure(closure), Value::Number(3.0)])
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("attempt to index"), "{message}");
    assert!(message.contains("number"), "{message}");
    assert_eq!(vm.jit_statistics().interpreter_inline_table_calls, 1);
}

/// `local a, b = 5, 1; local function f() a = 4; return b + 3 [, g()] end;
/// b = f(nil); return a, b`. The callee writes `a`'s cell, so the caller's
/// cells are newer than its registers when the result lands in `b`.
fn captured_destination_chunk(open_return: bool) -> Chunk {
    let upvalues = vec![
        UpvalueDescriptor {
            name: "a".into(),
            source: UpvalueSource::ParentRegister(0),
        },
        UpvalueDescriptor {
            name: "b".into(),
            source: UpvalueSource::ParentRegister(1),
        },
    ];
    let mut code = vec![
        Instruction::load_constant(0, 0),
        Instruction::set_upvalue(0, 0),
        Instruction::get_upvalue(1, 1),
        Instruction::load_constant(2, 1),
        Instruction::binary(OpCode::Add, 0, 1, 2),
    ];
    let mut prototypes = Vec::new();
    if open_return {
        // `return b + 3, g()`
        prototypes.push(Chunk::with_parts(
            2,
            0,
            vec![Constant::Number(100.0), Constant::Number(200.0)],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::return_values(0, 2),
            ],
            Vec::new(),
            Vec::new(),
        ));
        code.push(Instruction::closure(1, 0));
        code.push(Instruction::call(1, 0, 0));
        code.push(Instruction::return_open(0, 1));
    } else {
        code.push(Instruction::return_(0));
    }
    let callee = Chunk::with_parts(
        3,
        0,
        vec![Constant::Number(4.0), Constant::Number(3.0)],
        code,
        prototypes,
        upvalues,
    );
    Chunk::with_parts(
        4,
        0,
        vec![
            Constant::Number(5.0),
            Constant::Number(1.0),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::closure(2, 0),
            Instruction::load_constant(3, 2),
            Instruction::call_one(1, 2, 3),
            Instruction::return_values(0, 2),
        ],
        vec![callee],
        Vec::new(),
    )
}

#[test]
fn a_return_into_a_captured_destination_keeps_the_result_after_the_callee_wrote_a_cell() {
    for open_return in [false, true] {
        let chunk = captured_destination_chunk(open_return);
        // Three VMs in one stack array is deliberate; a heap allocation
        // would change nothing but the loop's shape.
        #[allow(clippy::large_stack_arrays)]
        for mut vm in [
            Vm::with_execution_mode(ExecutionMode::Interpreter),
            Vm::with_execution_mode(ExecutionMode::Tracing),
            synchronous_tracing_vm(),
        ] {
            let results = vm.execute(&chunk).unwrap();
            assert_eq!(
                results,
                vec![Value::Number(4.0), Value::Number(4.0)],
                "open_return={open_return}"
            );
        }
    }
}

#[test]
fn an_owned_chunk_loads_after_the_same_validation() {
    let invalid = Chunk::new(
        1,
        vec![],
        vec![Instruction::call(0, 0, 2), Instruction::halt()],
    );
    assert!(matches!(
        Vm::new().load_owned_chunk(invalid),
        Err(VmError::InvalidChunk(
            ValidationError::RegisterOutOfBounds { .. }
        ))
    ));

    let mut vm = Vm::new();
    let chunk = crate::compiler::compile("return 40 + 2").unwrap();
    let function = vm.load_owned_chunk(chunk).unwrap();
    assert_eq!(
        vm.call_value(function, Vec::new()).unwrap(),
        vec![Value::Number(42.0)]
    );
}
