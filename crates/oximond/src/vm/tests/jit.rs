use super::*;
use crate::vm::stack::CallStack;

#[test]
fn direct_inline_plan_hot_cache_bypasses_the_backing_map() {
    let mut vm = Vm::new();
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();

    let first = vm.inline_call_plan(closure).unwrap();
    assert_eq!(
        vm.jit
            .inline_call_hot_cache
            .as_ref()
            .map(|(cached, _)| *cached),
        Some(closure)
    );
    vm.jit.inline_call_plans.clear();

    let second = vm.inline_call_plan(closure).unwrap();
    assert_eq!(second.callee_digest, first.callee_digest);
    assert!(vm.jit.inline_call_plans.is_empty());
}

#[test]
fn inline_plans_of_alternating_closures_hit_without_the_backing_map() {
    let mut vm = Vm::new();
    // The fast loop pushes a real activation for a plain call. The dispatch
    // loop, which profiling selects, still evaluates inline plans.
    vm.set_interpreter_profiling(true);
    let increment = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    let double = vm
        .runtime_mut()
        .create_closure(
            Arc::new(Chunk::with_parts(
                2,
                1,
                Vec::new(),
                vec![
                    Instruction::binary(OpCode::Add, 1, 0, 0),
                    Instruction::return_(1),
                ],
                Vec::new(),
                Vec::new(),
            )),
            Vec::new(),
        )
        .unwrap();
    let call = one_argument_call_one_parent(1);
    let run = |vm: &mut Vm, closure: ClosureHandle, argument: f64| {
        vm.execute_with_arguments(
            &call,
            vec![Value::Closure(closure), Value::Number(argument)],
        )
        .unwrap()
    };
    assert_eq!(run(&mut vm, increment, 1.0), vec![Value::Number(2.0)]);
    assert_eq!(run(&mut vm, double, 4.0), vec![Value::Number(8.0)]);
    let plans = [increment, double].map(|closure| vm.inline_call_plan(closure).unwrap().plan);
    vm.jit.inline_call_plans.clear();
    vm.jit.inline_call_hot_cache = None;
    let probes = plans.map(|plan| {
        let count = Arc::strong_count(&plan);
        (plan, count)
    });
    let calls = vm.jit_statistics().interpreter_inline_calls;

    for argument in 0..16 {
        let argument = f64::from(argument);
        assert_eq!(
            run(&mut vm, increment, argument),
            vec![Value::Number(argument + 1.0)]
        );
        assert_eq!(
            run(&mut vm, double, argument),
            vec![Value::Number(argument * 2.0)]
        );
    }

    assert_eq!(vm.jit_statistics().interpreter_inline_calls, calls + 32);
    assert!(vm.jit.inline_call_plans.is_empty());
    for (plan, count) in &probes {
        assert_eq!(Arc::strong_count(plan), *count);
    }
}

#[test]
fn permanent_inline_refusal_hot_cache_bypasses_the_backing_map_during_calls() {
    let mut vm = Vm::new();
    let unsupported = vm
        .runtime_mut()
        .create_closure(
            Arc::new(Chunk::with_parts(
                1,
                1,
                vec![Constant::String("ordinary fallback".into())],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
                Vec::new(),
                Vec::new(),
            )),
            Vec::new(),
        )
        .unwrap();
    let call = one_argument_call_one_parent(1);

    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(unsupported), Value::Number(1.0)],)
            .unwrap(),
        vec![Value::from("ordinary fallback")]
    );
    assert!(matches!(
        vm.jit.inline_call_plans.get(&unsupported),
        Some(None)
    ));
    assert!(matches!(
        vm.jit.inline_call_hot_cache,
        Some((cached, None)) if cached == unsupported
    ));

    let supported = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    assert!(vm.inline_call_plan(supported).is_some());
    assert_eq!(
        vm.jit
            .inline_call_hot_cache
            .as_ref()
            .map(|(cached, _)| *cached),
        Some(supported)
    );

    // The first call recorded the permanent refusal on the closure itself,
    // so later calls consult neither the one-entry cache nor the map.
    assert_eq!(
        vm.runtime()
            .closure_call_parts(unsupported)
            .map(|parts| parts.call_hint),
        Some(crate::vm::CALL_HINT_PLAIN)
    );
    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(unsupported), Value::Number(2.0)],)
            .unwrap(),
        vec![Value::from("ordinary fallback")]
    );
    assert!(matches!(
        vm.jit.inline_call_hot_cache,
        Some((cached, _)) if cached == supported
    ));
    vm.jit.inline_call_plans.clear();
    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(unsupported), Value::Number(3.0)],)
            .unwrap(),
        vec![Value::from("ordinary fallback")]
    );
    assert!(vm.jit.inline_call_plans.is_empty());
}

#[test]
fn direct_inline_plan_hot_cache_rejects_stale_generational_handles() {
    let mut vm = Vm::new();
    let stale = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    assert!(vm.inline_call_plan(stale).is_some());

    vm.runtime_mut().collect_full(GcRootBuilder::new().finish());
    let replacement = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();

    assert_ne!(replacement, stale);
    assert!(vm.inline_call_plan(stale).is_none());
    assert!(vm.inline_call_plan(replacement).is_some());
    assert_eq!(
        vm.jit
            .inline_call_hot_cache
            .as_ref()
            .map(|(cached, _)| *cached),
        Some(replacement)
    );
}

#[test]
fn inline_table_replacements_do_not_allocate_and_normalize_numeric_keys() {
    let mut runtime = Runtime::default();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Integer(1), Value::Number(4.0))
        .unwrap();
    let plan = InlineTablePlan::build(
        &Chunk::with_parts(
            5,
            2,
            vec![Constant::Integer(1)],
            vec![
                Instruction::load_constant(2, 0),
                Instruction::get_table(3, 0, 1),
                Instruction::binary(OpCode::Add, 3, 3, 2),
                Instruction::set_table(0, 1, 3),
                Instruction::get_table(4, 0, 2),
                Instruction::return_(4),
            ],
            Vec::new(),
            Vec::new(),
        ),
        CompatibilityProfile::Lua55,
    )
    .expect("existing numeric-key table update is inlineable");
    let serial = runtime.allocation_serial();
    assert_eq!(
        plan.evaluate(&mut runtime, &[Value::Table(table), Value::Number(1.0)],),
        Some(Value::Number(5.0))
    );
    assert_eq!(runtime.allocation_serial(), serial);
    assert_eq!(
        runtime.table_get(table, &Value::Integer(1)).unwrap(),
        Value::Number(5.0)
    );
}

#[test]
fn inline_table_plan_bails_before_insertions_and_nil_deletions() {
    let mut runtime = Runtime::default();
    let table = runtime.create_table().unwrap();
    let replace = InlineTablePlan::build(
        &Chunk::with_parts(
            4,
            1,
            vec![Constant::String("value".into()), Constant::Number(2.0)],
            vec![
                Instruction::load_constant(1, 0),
                Instruction::load_constant(2, 1),
                Instruction::set_table(0, 1, 2),
                Instruction::get_table(3, 0, 1),
                Instruction::return_(3),
            ],
            Vec::new(),
            Vec::new(),
        ),
        CompatibilityProfile::Lua55,
    )
    .expect("straight-line table write is a candidate");
    let serial = runtime.allocation_serial();
    assert_eq!(replace.evaluate(&mut runtime, &[Value::Table(table)]), None);
    assert_eq!(runtime.allocation_serial(), serial);

    runtime
        .table_set(table, &Value::from("value"), Value::Number(1.0))
        .unwrap();
    let delete = InlineTablePlan::build(
        &Chunk::with_parts(
            3,
            1,
            vec![Constant::String("value".into())],
            vec![
                Instruction::load_constant(1, 0),
                Instruction::load_nil(2),
                Instruction::set_table(0, 1, 2),
                Instruction::return_(2),
            ],
            Vec::new(),
            Vec::new(),
        ),
        CompatibilityProfile::Lua55,
    )
    .expect("nil write remains a candidate until its runtime guard");
    let serial = runtime.allocation_serial();
    assert_eq!(delete.evaluate(&mut runtime, &[Value::Table(table)]), None);
    assert_eq!(runtime.allocation_serial(), serial);
    assert_eq!(
        runtime.table_get(table, &Value::from("value")).unwrap(),
        Value::Number(1.0)
    );
}

#[test]
fn repeated_existing_field_writes_commit_once_and_replay_other_keys() {
    let same_key = Chunk::with_parts(
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
    let other_key = Chunk::with_parts(
        5,
        1,
        vec![
            Constant::String("value".into()),
            Constant::String("other".into()),
            Constant::Number(2.0),
            Constant::Number(3.0),
        ],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 2),
            Instruction::set_table(0, 1, 2),
            Instruction::load_constant(3, 1),
            Instruction::load_constant(4, 3),
            Instruction::set_table(0, 3, 4),
            Instruction::return_(4),
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

    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = match mode {
            ExecutionMode::Interpreter => Vm::new(),
            ExecutionMode::Tracing => synchronous_tracing_vm(),
        };
        let table = vm.runtime_mut().create_table().unwrap();
        let key = Value::from("value");
        vm.runtime_mut()
            .table_set(table, &key, Value::Number(1.0))
            .unwrap();
        let plan = InlineTablePlan::build(&same_key, CompatibilityProfile::Lua55).unwrap();
        assert_eq!(
            plan.evaluate(vm.runtime_mut(), &[Value::Table(table)]),
            Some(Value::Number(5.0))
        );
        assert_eq!(
            vm.runtime().table_get(table, &key).unwrap(),
            Value::Number(5.0)
        );

        let fallback = InlineTablePlan::build(&other_key, CompatibilityProfile::Lua55).unwrap();
        assert_eq!(
            fallback.evaluate(vm.runtime_mut(), &[Value::Table(table)]),
            None
        );
        assert_eq!(
            vm.runtime().table_get(table, &key).unwrap(),
            Value::Number(5.0)
        );
        assert_eq!(
            vm.runtime()
                .table_get(table, &Value::from("other"))
                .unwrap(),
            Value::Nil
        );

        let same = vm
            .runtime_mut()
            .create_closure(Arc::new(same_key.clone()), Vec::new())
            .unwrap();
        let other = vm
            .runtime_mut()
            .create_closure(Arc::new(other_key.clone()), Vec::new())
            .unwrap();
        assert_eq!(
            vm.execute_with_arguments(&call, vec![Value::Closure(same), Value::Table(table)])
                .unwrap(),
            vec![Value::Number(5.0)]
        );
        assert_eq!(vm.jit_statistics().interpreter_inline_table_calls, 1);
        assert_eq!(
            vm.execute_with_arguments(&call, vec![Value::Closure(other), Value::Table(table)])
                .unwrap(),
            vec![Value::Number(3.0)]
        );
        assert_eq!(
            vm.runtime().table_get(table, &key).unwrap(),
            Value::Number(2.0)
        );
        assert_eq!(
            vm.runtime()
                .table_get(table, &Value::from("other"))
                .unwrap(),
            Value::Number(3.0)
        );
    }
}

#[test]
fn branched_table_call_commits_one_field_or_replays_both() {
    let callee = Chunk::with_parts(
        4,
        2,
        vec![
            Constant::String("value".into()),
            Constant::String("other".into()),
            Constant::Number(7.0),
        ],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::load_constant(3, 2),
            Instruction::set_table(0, 2, 3),
            Instruction::jump_if_false(1, 2),
            Instruction::load_constant(2, 1),
            Instruction::set_table(0, 2, 3),
            Instruction::return_(3),
        ],
        Vec::new(),
        Vec::new(),
    );
    let call = Chunk::with_parts(
        3,
        3,
        Vec::new(),
        vec![Instruction::call(0, 2, 1), Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
    );

    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut vm = match mode {
            ExecutionMode::Interpreter => Vm::new(),
            ExecutionMode::Tracing => synchronous_tracing_vm(),
        };
        let table = vm.runtime_mut().create_table().unwrap();
        let key = Value::from("value");
        let other = Value::from("other");
        vm.runtime_mut()
            .table_set(table, &key, Value::Number(1.0))
            .unwrap();
        let plan = InlineTablePlan::build(&callee, CompatibilityProfile::Lua55).unwrap();
        assert_eq!(
            plan.evaluate(
                vm.runtime_mut(),
                &[Value::Table(table), Value::Boolean(false)]
            ),
            Some(Value::Number(7.0))
        );
        assert_eq!(
            vm.runtime().table_get(table, &key).unwrap(),
            Value::Number(7.0)
        );
        assert_eq!(vm.runtime().table_get(table, &other).unwrap(), Value::Nil);
        assert_eq!(
            plan.evaluate(
                vm.runtime_mut(),
                &[Value::Table(table), Value::Boolean(true)]
            ),
            None
        );
        assert_eq!(
            vm.runtime().table_get(table, &key).unwrap(),
            Value::Number(7.0)
        );
        assert_eq!(vm.runtime().table_get(table, &other).unwrap(), Value::Nil);

        let closure = vm
            .runtime_mut()
            .create_closure(Arc::new(callee.clone()), Vec::new())
            .unwrap();
        assert_eq!(
            vm.execute_with_arguments(
                &call,
                vec![
                    Value::Closure(closure),
                    Value::Table(table),
                    Value::Boolean(false)
                ],
            )
            .unwrap(),
            vec![Value::Number(7.0)]
        );
        assert_eq!(vm.jit_statistics().interpreter_inline_table_calls, 1);
        assert_eq!(
            vm.execute_with_arguments(
                &call,
                vec![
                    Value::Closure(closure),
                    Value::Table(table),
                    Value::Boolean(true)
                ],
            )
            .unwrap(),
            vec![Value::Number(7.0)]
        );
        assert_eq!(
            vm.runtime().table_get(table, &other).unwrap(),
            Value::Number(7.0)
        );
    }
}

#[test]
fn vm_call_bridge_reads_a_replaced_closure_at_the_original_call() {
    let _test_lock = VM_CALL_BRIDGE_TEST_LOCK.lock().unwrap();
    let limit = 8_u32;
    let caller = vm_call_bridge_caller(f64::from(limit));
    let mut vm = synchronous_tracing_vm();
    let recorder = vm
        .runtime_mut()
        .register_native(record_vm_call_bridge_effect)
        .unwrap();
    vm.runtime_mut()
        .set_global("bridge_record", Value::NativeFunction(recorder));
    let first = vm
        .runtime_mut()
        .create_closure(Arc::new(vm_call_bridge_callee(1.0)), Vec::new())
        .unwrap();
    let replacement = vm
        .runtime_mut()
        .create_closure(Arc::new(vm_call_bridge_callee(2.0)), Vec::new())
        .unwrap();

    for _ in 0..3 {
        assert_eq!(
            vm.execute_with_arguments(&caller, vec![Value::Closure(first)])
                .unwrap(),
            vec![Value::Number(vm_call_bridge_expected(limit, 1))]
        );
    }
    VM_CALL_BRIDGE_EFFECTS.store(0, Ordering::SeqCst);
    assert_eq!(
        vm.execute_with_arguments(&caller, vec![Value::Closure(replacement)])
            .unwrap(),
        vec![Value::Number(vm_call_bridge_expected(limit, 2))]
    );
    assert_eq!(
        VM_CALL_BRIDGE_EFFECTS.load(Ordering::SeqCst),
        usize::try_from(limit).unwrap()
    );
}

#[test]
fn tracing_vm_call_bridge_reports_callee_error_after_one_effect() {
    let _test_lock = VM_CALL_BRIDGE_TEST_LOCK.lock().unwrap();
    let caller = vm_call_bridge_caller(8.0);
    let failing = Chunk::with_parts(
        3,
        1,
        vec![
            Constant::String("bridge_record".into()),
            Constant::Number(1.0),
        ],
        vec![
            Instruction::get_global(1, 0),
            Instruction::move_(2, 0),
            Instruction::call(1, 1, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(1, 0, 0),
            Instruction::return_(0),
        ],
        Vec::new(),
        Vec::new(),
    );
    let mut vm = synchronous_tracing_vm();
    let recorder = vm
        .runtime_mut()
        .register_native(record_vm_call_bridge_effect)
        .unwrap();
    vm.runtime_mut()
        .set_global("bridge_record", Value::NativeFunction(recorder));
    let successful = vm
        .runtime_mut()
        .create_closure(Arc::new(vm_call_bridge_callee(1.0)), Vec::new())
        .unwrap();
    let failed = vm
        .runtime_mut()
        .create_closure(Arc::new(failing), Vec::new())
        .unwrap();
    for _ in 0..4 {
        vm.execute_with_arguments(&caller, vec![Value::Closure(successful)])
            .unwrap();
    }

    VM_CALL_BRIDGE_EFFECTS.store(0, Ordering::SeqCst);
    let error = vm
        .execute_with_arguments(&caller, vec![Value::Closure(failed)])
        .unwrap_err();
    assert!(
        error.to_string().contains("attempt to call a number value"),
        "unexpected error: {error}"
    );
    assert_eq!(VM_CALL_BRIDGE_EFFECTS.load(Ordering::SeqCst), 1);
}

#[test]
// `caller` and `callee` are the domain's paired terms.
#[allow(clippy::similar_names)]
fn tracing_vm_call_bridge_warmed_site_observes_new_call_hooks() {
    static HOOK_EVENTS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn record_hook(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        if matches!(arguments, [Value::String(event), Value::Nil] if event.as_bytes() == b"call") {
            HOOK_EVENTS.fetch_add(1, Ordering::SeqCst);
        }
        Ok(Vec::new())
    }

    let _test_lock = VM_CALL_BRIDGE_TEST_LOCK.lock().unwrap();
    let caller = vm_call_bridge_caller(8.0);
    let mut vm = synchronous_tracing_vm();
    let recorder = vm
        .runtime_mut()
        .register_native(record_vm_call_bridge_effect)
        .unwrap();
    vm.runtime_mut()
        .set_global("bridge_record", Value::NativeFunction(recorder));
    let callee = vm
        .runtime_mut()
        .create_closure(Arc::new(vm_call_bridge_callee(1.0)), Vec::new())
        .unwrap();
    for _ in 0..4 {
        vm.execute_with_arguments(&caller, vec![Value::Closure(callee)])
            .unwrap();
    }

    let hook = Value::NativeFunction(vm.runtime_mut().register_native(record_hook).unwrap());
    vm.debug_sethook(&[hook, Value::from("c")]).unwrap();
    let native_entries = vm.jit_statistics().trace.native_entries;
    VM_CALL_BRIDGE_EFFECTS.store(0, Ordering::SeqCst);
    HOOK_EVENTS.store(0, Ordering::SeqCst);
    assert_eq!(
        vm.execute_with_arguments(&caller, vec![Value::Closure(callee)])
            .unwrap(),
        vec![Value::Number(vm_call_bridge_expected(8, 1))]
    );
    assert_eq!(VM_CALL_BRIDGE_EFFECTS.load(Ordering::SeqCst), 8);
    assert!(HOOK_EVENTS.load(Ordering::SeqCst) >= 8);
    assert_eq!(vm.jit_statistics().trace.native_entries, native_entries);
}

#[test]
fn ordinary_calls_inline_fixed_arity_and_vararg_numeric_closures() {
    let fixed = Chunk::with_parts(
        3,
        2,
        Vec::new(),
        vec![
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        Vec::new(),
        Vec::new(),
    );
    let vararg = Chunk::with_varargs(
        3,
        1,
        Vec::new(),
        vec![
            Instruction::vararg(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
        Vec::new(),
        Vec::new(),
        None,
    );
    let parent = Chunk::with_parts(
        3,
        3,
        Vec::new(),
        vec![Instruction::call(0, 2, 1), Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
    );
    let mut vm = Vm::new();
    let fixed = vm
        .runtime_mut()
        .create_closure(Arc::new(fixed), Vec::new())
        .unwrap();
    let vararg = vm
        .runtime_mut()
        .create_closure(Arc::new(vararg), Vec::new())
        .unwrap();
    assert_eq!(
        vm.execute_with_arguments(
            &parent,
            vec![
                Value::Closure(fixed),
                Value::Number(20.0),
                Value::Number(22.0)
            ],
        )
        .unwrap(),
        vec![Value::Number(42.0)]
    );
    assert_eq!(
        vm.execute_with_arguments(
            &parent,
            vec![
                Value::Closure(vararg),
                Value::Number(20.0),
                Value::Number(22.0)
            ],
        )
        .unwrap(),
        vec![Value::Number(42.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 2);
}

#[test]
fn call_one_inline_plan_uses_a_single_numeric_capture_but_respects_hooks() {
    static HOOK_EVENTS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn record_hook(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        HOOK_EVENTS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    let parent = one_argument_call_one_parent(2);
    let mut vm = Vm::new();
    let captured = Chunk::with_parts(
        1,
        1,
        Vec::new(),
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        Vec::new(),
        vec![UpvalueDescriptor {
            name: "saved".into(),
            source: UpvalueSource::Detached,
        }],
    );
    let cell = vm.runtime_mut().create_cell(Value::Number(17.0)).unwrap();
    let captured = vm
        .runtime_mut()
        .create_closure(Arc::new(captured), vec![cell])
        .unwrap();
    assert_eq!(
        vm.execute_with_arguments(&parent, vec![Value::Closure(captured), Value::Number(2.0)],)
            .unwrap(),
        vec![Value::Number(17.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 1);

    let increment = vm
        .runtime_mut()
        .create_closure(Arc::new(inline_increment_callee()), Vec::new())
        .unwrap();
    let callback = Value::NativeFunction(vm.runtime_mut().register_native(record_hook).unwrap());
    HOOK_EVENTS.store(0, Ordering::SeqCst);
    vm.debug_sethook(&[callback, Value::from("cr")]).unwrap();
    assert_eq!(
        vm.execute_with_arguments(&parent, vec![Value::Closure(increment), Value::Number(4.0)],)
            .unwrap(),
        vec![Value::Number(5.0)]
    );
    assert_eq!(vm.jit_statistics().interpreter_inline_calls, 1);
    assert!(HOOK_EVENTS.load(Ordering::SeqCst) >= 2);
}

#[test]
fn captured_inline_plan_tracks_debug_upvaluejoin_replacements() {
    let mut vm = Vm::new();
    let original_cell = vm.runtime_mut().create_cell(Value::Number(1.0)).unwrap();
    let source_cell = vm.runtime_mut().create_cell(Value::Number(100.0)).unwrap();
    let body = Arc::new(captured_counter_call_one_callee());
    let counter = vm
        .runtime_mut()
        .create_closure(Arc::clone(&body), vec![original_cell])
        .unwrap();
    let source = vm
        .runtime_mut()
        .create_closure(body, vec![source_cell])
        .unwrap();
    let call = one_argument_call_one_parent(1);

    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(counter), Value::Number(1.0)])
            .unwrap(),
        vec![Value::Number(2.0)]
    );
    vm.debug_upvaluejoin(
        &mut CallStack::new(),
        &[
            Value::Closure(counter),
            Value::Number(1.0),
            Value::Closure(source),
            Value::Number(1.0),
        ],
    )
    .unwrap();
    assert_eq!(
        vm.execute_with_arguments(&call, vec![Value::Closure(counter), Value::Number(1.0)])
            .unwrap(),
        vec![Value::Number(101.0)]
    );
    assert_eq!(
        vm.runtime().cell_get(original_cell).unwrap(),
        Value::Number(2.0)
    );
    assert_eq!(
        vm.runtime().cell_get(source_cell).unwrap(),
        Value::Number(101.0)
    );
}

#[test]
fn tracing_non_hot_steps_leave_installed_hooks_for_step() {
    static HOOK_EVENTS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn record_count_hook(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert!(
            matches!(arguments, [Value::String(event), Value::Nil] if event.as_bytes() == b"count")
        );
        HOOK_EVENTS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    HOOK_EVENTS.store(0, Ordering::SeqCst);
    let mut vm = Vm::with_execution_mode(ExecutionMode::Tracing);
    let environment = vm.runtime().global_environment();
    let mut frame = Frame::new(
        Arc::new(numeric_return_chunk()),
        Vec::new(),
        Vec::new(),
        None,
        None,
        environment,
        None,
    );
    // The second instruction is a forward interpreter step, not a tracing
    // entry or backedge.
    frame.pc = 1;
    frame.last_interpreted_pc = Some(0).into();
    let callback =
        Value::NativeFunction(vm.runtime_mut().register_native(record_count_hook).unwrap());
    vm.debug_sethook(&[callback, Value::from(""), Value::Number(1.0)])
        .unwrap();

    let mut frames = CallStack::from(vec![frame]);
    // A real tracing entry still checks the hook and cannot start native
    // execution while instruction events are observable.
    frames[0].pc = 0;
    frames[0].last_interpreted_pc = None.into();
    assert!(!vm.maybe_run_native(&mut frames).unwrap());
    assert_eq!(vm.jit_statistics().trace.recording_starts, 0);

    frames[0].pc = 1;
    frames[0].last_interpreted_pc = Some(0).into();
    assert!(
        !vm.maybe_run_native(&mut frames).unwrap(),
        "a non-hot tracing step stays in the interpreter even with a live hook"
    );
    assert_eq!(frames[0].pc, 1);

    assert_eq!(vm.step(&mut frames).unwrap(), None);
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 1);
    assert_eq!(frames[0].pc, 1, "the hook runs before the instruction");
}

#[test]
fn interpreter_mode_never_compiles_numeric_bytecode() {
    let mut vm = Vm::new();
    assert!(vm.jit.broker.is_none());
    assert_eq!(
        vm.execute(&numeric_return_chunk()).unwrap(),
        vec![Value::Number(5.0)]
    );
    assert_eq!(vm.jit_statistics(), crate::vm::JitStatistics::default());
}

#[test]
fn tracing_mode_defers_the_compile_broker_until_a_compile() {
    assert!(
        Vm::with_execution_mode(ExecutionMode::Tracing)
            .jit
            .broker
            .is_none()
    );
}

#[test]
fn tracing_mode_executes_a_validated_nested_captured_closure() {
    let child = Chunk::with_parts(
        1,
        0,
        Vec::new(),
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        Vec::new(),
        vec![UpvalueDescriptor {
            name: "captured".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(7.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(1, 0, 1),
            Instruction::return_(1),
        ],
        vec![child],
        Vec::new(),
    );

    let mut vm = synchronous_tracing_vm();
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(7.0)]);
}

#[test]
fn executes_tables_and_native_calls() {
    let chunk = Chunk::new(
        5,
        vec![
            Constant::String("assert".into()),
            Constant::Integer(1),
            Constant::String("ok".into()),
        ],
        vec![
            Instruction::new_table(0, 0, 1),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::set_table(0, 1, 2),
            Instruction::get_table(3, 0, 1),
            Instruction::get_global(0, 0),
            Instruction::move_(1, 3),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::from("ok")]);
}

#[test]
fn dofile_keeps_lua51s_native_yield_boundary() {
    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);
    #[allow(clippy::unnecessary_wraps)]
    fn loader(_: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        Ok(Chunk::new(
            2,
            vec![
                Constant::String("coroutine".into()),
                Constant::String("yield".into()),
                Constant::Number(1.0),
            ],
            vec![
                Instruction::get_global(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::get_table(0, 0, 1),
                Instruction::load_constant(1, 2),
                Instruction::call(0, 1, 0),
                Instruction::return_(0),
            ],
        ))
    }

    let path = std::env::temp_dir().join(format!(
        "oximond-dofile-yield-{}.lua",
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, b"dofile source").unwrap();
    let entry = Chunk::new(
        2,
        vec![
            Constant::String("dofile".into()),
            Constant::String(LuaString::from(path.to_string_lossy().into_owned())),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
    );
    let program = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::String("coroutine".into()),
            Constant::String("create".into()),
            Constant::String("resume".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_table(0, 0, 1),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 1),
            Instruction::get_global(1, 0),
            Instruction::load_constant(2, 2),
            Instruction::get_table(1, 1, 2),
            Instruction::move_(2, 0),
            Instruction::call(1, 1, 2),
            Instruction::return_values(1, 2),
        ],
        vec![entry],
        vec![],
    );
    let mut vm = Vm::with_profile(crate::compat::CompatibilityProfile::Lua51);
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    assert!(matches!(
        vm.execute(&program).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(message)]
            if message.as_bytes() == b"attempt to yield across metamethod/C-call boundary"
    ));
    fs::remove_file(path).unwrap();
}

#[test]
fn collectgarbage_runs_native_userdata_finalizers() {
    fn native_finalizer(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        if let Some(value) = arguments.first() {
            runtime.raw_metatable(value)?;
        }
        runtime.set_global(
            "native_finalized",
            arguments.first().cloned().unwrap_or(Value::Nil),
        );
        Ok(Vec::new())
    }

    let collect = Chunk::new(
        2,
        vec![
            Constant::String("doomed".into()),
            Constant::String("collectgarbage".into()),
            Constant::String("collect".into()),
            Constant::String("native_finalized".into()),
        ],
        vec![
            Instruction::load_nil(0),
            Instruction::set_global(0, 0),
            Instruction::get_global(0, 1),
            Instruction::load_constant(1, 2),
            Instruction::call(0, 1, 0),
            Instruction::get_global(0, 3),
            Instruction::return_(0),
        ],
    );

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.runtime_mut().stop_gc();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let finalizer = vm.runtime_mut().register_native(native_finalizer).unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(finalizer),
        )
        .unwrap();
    let userdata = vm.runtime_mut().create_userdata().unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("doomed", Value::Userdata(userdata));

    assert_eq!(
        vm.execute(&collect).unwrap(),
        vec![Value::Userdata(userdata)]
    );
}

#[test]
fn uncaught_native_error_preserves_a_deferred_gc_event_for_the_next_run() {
    fn allocate_then_fail(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        runtime.create_table()?;
        Err(RuntimeError::message("expected failure"))
    }

    let failing = Chunk::new(
        1,
        vec![Constant::String("allocate_then_fail".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::halt(),
        ],
    );
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
    for _ in 0..128 {
        vm.runtime_mut().create_table().unwrap();
    }
    let native = vm
        .runtime_mut()
        .register_native(allocate_then_fail)
        .unwrap();
    vm.runtime_mut()
        .set_global("allocate_then_fail", Value::NativeFunction(native));
    vm.runtime_mut().restart_gc();

    assert!(vm.execute(&failing).is_err());
    assert_ne!(
        vm.gc_observed_allocation_serial,
        vm.runtime().allocation_serial(),
        "the failing call must leave its allocation for the next run"
    );
    assert_eq!(
        vm.execute(&loop_chunk).unwrap(),
        vec![Value::Number(12_502_500.0)]
    );
    assert!(!vm.runtime().gc_needs_collection());
}

#[test]
fn load_reader_collects_native_byte_chunks_without_recursion() {
    let load_and_call = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, reader_loader);
    install_sequence_reader(
        &mut vm,
        &[Value::from("an"), Value::from("swer"), Value::Nil],
    );
    assert_eq!(
        vm.execute(&load_and_call).unwrap(),
        vec![Value::Number(42.0)]
    );

    install_sequence_reader(
        &mut vm,
        &[Value::from(LuaString::from(vec![0xff])), Value::Nil],
    );
    assert_eq!(
        vm.execute(&load_and_call).unwrap(),
        vec![Value::Number(9.0)]
    );

    install_sequence_reader(
        &mut vm,
        &[Value::from("answer"), Value::from(""), Value::from("")],
    );
    assert_eq!(
        vm.execute(&load_and_call).unwrap(),
        vec![Value::Number(42.0)]
    );
    assert_eq!(vm.runtime().get_global("reader_index"), Value::Number(3.0));
}

#[test]
fn invalid_call_result_range_and_jump_are_rejected_before_execution() {
    let invalid_results = Chunk::new(
        1,
        vec![],
        vec![Instruction::call(0, 0, 2), Instruction::halt()],
    );
    assert!(matches!(
        Vm::new().execute(&invalid_results),
        Err(VmError::InvalidChunk(
            ValidationError::RegisterOutOfBounds { .. }
        ))
    ));

    let invalid_jump = Chunk::new(
        1,
        vec![],
        vec![
            Instruction::aj(OpCode::Jump, 1).unwrap(),
            Instruction::halt(),
        ],
    );
    assert!(matches!(
        Vm::new().execute(&invalid_jump),
        Err(VmError::InvalidChunk(
            ValidationError::JumpOutOfBounds { .. }
        ))
    ));
}

#[test]
fn call_one_inline_capture_updates_the_callers_open_register() {
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
        4,
        0,
        vec![Constant::Number(7.0), Constant::Number(0.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::move_(2, 1),
            Instruction::load_constant(3, 1),
            Instruction::call_one(1, 1, 3),
            Instruction::call_one(2, 2, 3),
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

#[test]
fn numeric_conversion_rejects_invalid_strings_and_comparisons_keep_types_separate() {
    let invalid_arithmetic = Chunk::new(
        2,
        vec![Constant::String("nope".into()), Constant::Number(1.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(
        Vm::new()
            .execute(&invalid_arithmetic)
            .unwrap_err()
            .to_string(),
        "attempt to perform arithmetic on a string value"
    );

    let mixed_comparison = Chunk::new(
        3,
        vec![Constant::String("2".into()), Constant::Number(3.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::LessThan, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new()
            .execute(&mixed_comparison)
            .unwrap_err()
            .to_string(),
        "attempt to compare string with number"
    );

    let table_comparison = Chunk::new(
        3,
        vec![],
        vec![
            Instruction::new_table(0, 0, 0),
            Instruction::new_table(1, 0, 0),
            Instruction::binary(OpCode::LessThan, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new()
            .execute(&table_comparison)
            .unwrap_err()
            .to_string(),
        "attempt to compare two table values"
    );

    let string_comparison = Chunk::new(
        3,
        vec![
            Constant::String(LuaString::from(vec![0x7f])),
            Constant::String(LuaString::from(vec![0x80])),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::LessThan, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&string_comparison).unwrap(),
        vec![Value::Boolean(true)]
    );
}

#[test]
fn length_uses_table_borders_string_bytes_and_rejects_scalars() {
    let table_length = Chunk::new(
        4,
        vec![
            Constant::Number(1.0),
            Constant::Number(2.0),
            Constant::Number(4.0),
        ],
        vec![
            Instruction::new_table(0, 0, 3),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 0),
            Instruction::set_table(0, 1, 2),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 1),
            Instruction::set_table(0, 1, 2),
            Instruction::load_constant(1, 2),
            Instruction::load_constant(2, 2),
            Instruction::set_table(0, 1, 2),
            Instruction::length(3, 0),
            Instruction::return_(3),
        ],
    );
    assert_eq!(
        Vm::new().execute(&table_length).unwrap(),
        vec![Value::Number(2.0)]
    );

    let string_length = Chunk::new(
        2,
        vec![Constant::String("h\u{e9}".into())],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::length(0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(
        Vm::new().execute(&string_length).unwrap(),
        vec![Value::Number(3.0)]
    );

    let scalar_length = Chunk::new(
        2,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::length(0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(
        Vm::new().execute(&scalar_length).unwrap_err().to_string(),
        "attempt to get length of a number value"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn table_sort_runs_default_native_and_lua_closure_comparators() {
    fn descending(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let (Some(Value::Number(left)), Some(Value::Number(right))) =
            (arguments.first(), arguments.get(1))
        else {
            return Err(RuntimeError::message("numbers required"));
        };
        Ok(vec![Value::Boolean(left > right)])
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let Value::Table(table_library) = vm.runtime().get_global("table") else {
        panic!("table is a table");
    };
    let sort = vm
        .runtime()
        .table_get(table_library, &Value::from("sort"))
        .unwrap();
    let items = vm.runtime_mut().create_table().unwrap();
    for (index, value) in [(1, 3), (2, 1), (3, 2)] {
        vm.runtime_mut()
            .table_set(
                items,
                &Value::Number(f64::from(index)),
                Value::Number(f64::from(value)),
            )
            .unwrap();
    }
    vm.runtime_mut().set_global("sort", sort);
    vm.runtime_mut().set_global("items", Value::Table(items));
    vm.runtime_mut()
        .register_global_native("descending", descending)
        .unwrap();

    let default = Chunk::new(
        4,
        vec![
            Constant::String("sort".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_nil(2),
            Instruction::load_nil(3),
            Instruction::call(0, 3, 0),
            Instruction::return_(1),
        ],
    );
    vm.execute(&default).unwrap();
    assert_eq!(
        vm.runtime().table_get(items, &Value::Number(1.0)).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(vm.table_sort_fast_path_hits, 1);

    let sparse = vm.runtime_mut().create_table().unwrap();
    for (index, value) in [(1, 3.0), (2, 2.0), (3, 1.0)] {
        vm.runtime_mut()
            .table_set(
                sparse,
                &Value::Number(f64::from(index)),
                Value::Number(value),
            )
            .unwrap();
    }
    vm.runtime_mut()
        .table_set(sparse, &Value::Number(2.0), Value::Nil)
        .unwrap();
    vm.runtime_mut().set_global("items", Value::Table(sparse));
    vm.execute(&default).unwrap();
    assert_eq!(vm.table_sort_fast_path_hits, 1);

    let non_number = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(non_number, &Value::Number(1.0), Value::Number(3.0))
        .unwrap();
    vm.runtime_mut()
        .table_set(non_number, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    vm.runtime_mut()
        .set_global("items", Value::Table(non_number));
    assert!(vm.execute(&default).is_err());
    assert_eq!(vm.table_sort_fast_path_hits, 1);

    let nan = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(nan, &Value::Number(1.0), Value::Number(f64::NAN))
        .unwrap();
    vm.runtime_mut()
        .table_set(nan, &Value::Number(2.0), Value::Number(1.0))
        .unwrap();
    vm.runtime_mut().set_global("items", Value::Table(nan));
    vm.execute(&default).unwrap();
    assert_eq!(vm.table_sort_fast_path_hits, 1);
    vm.runtime_mut().set_global("items", Value::Table(items));

    let native = Chunk::new(
        4,
        vec![
            Constant::String("sort".into()),
            Constant::String("items".into()),
            Constant::String("descending".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::return_(1),
        ],
    );
    vm.execute(&native).unwrap();
    assert_eq!(
        vm.runtime().table_get(items, &Value::Number(1.0)).unwrap(),
        Value::Number(3.0)
    );
    assert_eq!(vm.table_sort_fast_path_hits, 1);

    let ascending = Chunk::with_parts(
        3,
        2,
        vec![],
        vec![
            Instruction::binary(OpCode::LessThan, 2, 0, 1),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let closure = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::String("sort".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::closure(2, 0),
            Instruction::call(0, 2, 0),
            Instruction::return_(1),
        ],
        vec![ascending],
        vec![],
    );
    vm.execute(&closure).unwrap();
    assert_eq!(
        vm.runtime().table_get(items, &Value::Number(1.0)).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(vm.table_sort_fast_path_hits, 1);
}

#[test]
fn jit_switch_controls_native_entry_without_changing_execution_mode() {
    let chunk = numeric_sum_chunk(1.0, 10_000.0, 1.0);
    let mut vm = synchronous_tracing_vm();
    assert!(vm.jit_enabled());
    vm.runtime_mut().set_jit_enabled(false);
    assert!(!vm.jit_enabled());
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Number(50_005_000.0)]
    );
    assert_eq!(vm.jit_statistics().trace.native_entries, 0);

    vm.set_jit_enabled(true);
    assert!(vm.runtime().jit_enabled());
    for _ in 0..4 {
        assert_eq!(
            vm.execute(&chunk).unwrap(),
            vec![Value::Number(50_005_000.0)]
        );
    }
    assert!(vm.jit_statistics().trace.native_entries > 0);
    vm.set_jit_enabled(false);
    let runs = vm.jit_statistics().trace.native_entries;
    assert!(vm.jit.trace.cache().live() > 0);
    vm.runtime_mut().request_jit_flush();
    assert!(vm.runtime().jit_flush_requested());
    vm.execute(&chunk).unwrap();
    assert_eq!(vm.jit_statistics().trace.native_entries, runs);
    assert!(!vm.runtime().jit_flush_requested());
    assert_eq!(vm.jit.trace.cache().live(), 0);

    let mut interpreter = Vm::with_execution_mode(ExecutionMode::Interpreter);
    interpreter.set_jit_enabled(true);
    assert!(!interpreter.jit_enabled());
    interpreter.runtime_mut().set_jit_enabled(true);
    assert!(!interpreter.runtime().jit_enabled());
}

#[test]
fn jit_flush_keeps_tracing_enabled() {
    let mut vm = Vm::with_profile_and_execution_mode(
        CompatibilityProfile::default(),
        ExecutionMode::Tracing,
    );
    vm.runtime_mut().request_jit_flush();
    vm.service_jit_flush();
    assert!(vm.jit.trace.enabled());
}

#[test]
fn trace_policy_is_consulted_at_loop_headers_only_when_enabled() {
    for (mode, expect_consult) in [
        (ExecutionMode::Tracing, true),
        (ExecutionMode::Interpreter, false),
    ] {
        let mut vm = Vm::with_profile_and_execution_mode(CompatibilityProfile::default(), mode);
        vm.execute(&numeric_sum_chunk(1.0, 5000.0, 1.0)).unwrap();
        assert_eq!(vm.jit.trace.consults() > 0, expect_consult, "{mode:?}");
    }
}

/// `function(x) return <constant> end`: one parameter, one constant load.
fn constant_result_callee(constant: Constant) -> Chunk {
    Chunk::with_parts(
        2,
        1,
        vec![constant],
        vec![Instruction::load_constant(1, 0), Instruction::return_(1)],
        Vec::new(),
        Vec::new(),
    )
}

fn call_constant_callee(profile: CompatibilityProfile, constant: Constant) -> (Value, u64) {
    let mut vm = Vm::with_profile(profile);
    // The dispatch loop, which profiling selects, evaluates inline plans; the
    // fast loop pushes a real activation for a plain call.
    vm.set_interpreter_profiling(true);
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(constant_result_callee(constant)), Vec::new())
        .unwrap();
    // Run twice so a plan cached by the first call is also exercised.
    let mut result = Vec::new();
    for _ in 0..2 {
        result = vm
            .execute_with_arguments(
                &one_argument_call_one_parent(1),
                vec![Value::Closure(closure), Value::Number(1.5)],
            )
            .unwrap();
    }
    (
        result[0].clone(),
        vm.jit_statistics().interpreter_inline_calls,
    )
}

#[test]
fn integer_constant_callee_is_not_inlined_under_integer_profiles() {
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (value, inlined) = call_constant_callee(profile, Constant::Integer(9));
        assert_eq!(value, Value::Integer(9), "{profile:?}");
        assert_eq!(inlined, 0, "{profile:?} must not inline an integer result");
    }
}

#[test]
fn float_constant_callee_is_still_inlined_under_integer_profiles() {
    let (value, inlined) = call_constant_callee(CompatibilityProfile::Lua54, Constant::Number(9.5));
    assert_eq!(value, Value::Number(9.5));
    assert_eq!(inlined, 2);
}

#[test]
fn integer_constant_callee_is_still_inlined_without_an_integer_subtype() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let (value, inlined) = call_constant_callee(profile, Constant::Integer(9));
        assert_eq!(value, Value::Number(9.0), "{profile:?}");
        assert_eq!(inlined, 2, "{profile:?} keeps its inline plan");
    }
}

#[test]
fn tail_cycles_stay_detected_after_the_tail_call_count_saturates() {
    use crate::vm::jit::state::TailCycles;
    let (even, odd) = (0x1000, 0x2000);
    let mut cycles = TailCycles::default();
    // A frame at depth 1 goes even -> odd -> even just below saturation.
    assert!(!cycles.landed(1, even, u32::MAX - 2));
    assert!(!cycles.landed(1, odd, u32::MAX - 1));
    assert!(cycles.landed(1, even, u32::MAX));
    // From here every landing carries the saturated count.
    for _ in 0..16 {
        assert!(cycles.landed(1, odd, u32::MAX));
        assert!(cycles.landed(1, even, u32::MAX));
    }
    // A new frame at the same depth starts counting from one again: the
    // saturated landings of the old frame do not make it a cycle.
    assert!(!cycles.landed(1, even, 1));
    // Nor does a frame at another depth.
    assert!(!cycles.landed(2, odd, u32::MAX));
}

#[test]
fn tail_cycles_tell_a_cycle_from_a_chain_that_passes_through() {
    use crate::vm::jit::state::TailCycles;
    let mut cycles = TailCycles::default();
    // A chain of distinct functions never lands on one twice.
    for (count, chunk) in (1..=8).zip((1..=8).map(|index| index * 0x100)) {
        assert!(!cycles.landed(3, chunk, count));
    }
    // A function that comes back after more than eight tail calls is too far
    // apart to count.
    let mut cycles = TailCycles::default();
    assert!(!cycles.landed(3, 0x100, 1));
    assert!(!cycles.landed(3, 0x100, 10));
    assert!(cycles.landed(3, 0x100, 11));
}
