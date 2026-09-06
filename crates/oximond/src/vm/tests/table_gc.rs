use super::*;

#[test]
fn table_length_metamethod_follows_each_compatibility_profile() {
    fn length(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert_eq!(arguments.len(), 2);
        assert!(Runtime::raw_equal(&arguments[0], &arguments[1]));
        Ok(vec![Value::Number(31.0)])
    }
    for profile in CompatibilityProfile::ALL {
        let mut vm = Vm::with_profile(profile);
        let table = vm.runtime_mut().create_table().unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        let function = vm.runtime_mut().register_native(length).unwrap();
        vm.runtime_mut()
            .table_set(
                metatable,
                &Value::from("__len"),
                Value::NativeFunction(function),
            )
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(table), Some(metatable))
            .unwrap();
        vm.runtime_mut()
            .set_global("length_operand", Value::Table(table));
        let chunk = Chunk::new(
            2,
            vec![Constant::String("length_operand".into())],
            vec![
                Instruction::get_global(0, 0),
                Instruction::length(1, 0),
                Instruction::return_(1),
            ],
        );
        let result = vm.execute(&chunk).unwrap();
        if profile.is_lua51_family() {
            assert_eq!(result, vec![Value::Number(0.0)], "{profile}");
        } else {
            assert_eq!(result, vec![Value::Number(31.0)], "{profile}");
        }
    }
}

#[test]
fn lua55_gc_parameters_change_collector_policy_and_use_compact_encoding() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua55);
    for (index, (name, default)) in [
        ("minormul", 20),
        ("majorminor", 50),
        ("minormajor", 68),
        ("pause", 250),
        ("stepmul", 200),
        ("stepsize", 9_600),
    ]
    .into_iter()
    .enumerate()
    {
        let action = vm
            .collect_garbage(
                &mut CallStack::new(),
                &[Value::from("param"), Value::from(name), Value::Integer(30)],
            )
            .unwrap();
        let CollectGarbageAction::Immediate(values) = action else {
            panic!("parameter performs no collection");
        };
        assert_eq!(values, vec![Value::Integer(default)]);
        let expected = if index == 5 { 29.0 } else { 29.6875 };
        assert_eq!(vm.runtime.gc_parameter(index), Some(expected));
        let CollectGarbageAction::Immediate(values) = vm
            .collect_garbage(
                &mut CallStack::new(),
                &[Value::from("param"), Value::from(name), Value::Integer(-1)],
            )
            .unwrap()
        else {
            panic!("parameter performs no collection");
        };
        assert_eq!(values, vec![Value::Integer(29)]);
    }
    assert_eq!(quantize_lua55_gc_parameter(396_800), 396_800.0);
    assert_eq!(quantize_lua55_gc_parameter(u32::MAX), 396_800.0);
}
use crate::vm::stack::CallStack;

#[test]
fn borrowed_table_reads_allow_receiver_and_key_destination_aliases() {
    for destination in [0, 1] {
        let mut vm = Vm::new();
        let table = vm.runtime.create_table().unwrap();
        vm.runtime
            .table_set(table, &Value::from("key"), Value::from("result"))
            .unwrap();
        let chunk = Chunk::with_parts(
            2,
            2,
            Vec::new(),
            vec![
                Instruction::abc(OpCode::GetTable, destination, 0, 1),
                Instruction::return_(destination),
            ],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            vm.execute_with_arguments(&chunk, vec![Value::Table(table), Value::from("key")])
                .unwrap(),
            vec![Value::from("result")]
        );
    }
}

#[test]
fn zero_argument_vararg_closure_builds_its_empty_arg_table() {
    let inspect_arg_table = Chunk::with_varargs(
        2,
        0,
        vec![Constant::String("n".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::get_table(0, 1, 0),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
        Some(1),
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
        vec![inspect_arg_table],
        vec![],
    );

    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(0.0)]
    );

    let mut tracing = synchronous_tracing_vm();
    assert_eq!(tracing.execute(&program).unwrap(), vec![Value::Number(0.0)]);
}

#[test]
fn lua55_named_vararg_table_controls_later_expansion() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua55);
    let chunk = Arc::new(Chunk::with_varargs(
        1,
        0,
        Vec::new(),
        vec![Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
        Some(0),
    ));
    let environment = vm.thread_environment;
    let frame = vm
        .make_frame(
            chunk,
            vec![Value::Number(1.0), Value::Number(2.0)],
            Vec::new(),
            None,
            None,
            environment,
            None,
        )
        .unwrap();
    let Value::Table(varargs) = frame.registers[0] else {
        panic!("named varargs must bind a table")
    };
    vm.runtime
        .table_set(varargs, &Value::Number(1.0), Value::Number(11.0))
        .unwrap();
    vm.runtime
        .table_set(varargs, &Value::Number(3.0), Value::Number(33.0))
        .unwrap();
    vm.runtime
        .table_set(varargs, &Value::from("n"), Value::Number(3.0))
        .unwrap();

    let stack = crate::vm::stack::CallStack::with_frame(frame);
    assert_eq!(
        vm.frame_varargs(&stack, 0).unwrap(),
        vec![Value::Number(11.0), Value::Number(2.0), Value::Number(33.0)]
    );
}

#[test]
fn raw_table_paths_skip_continuations_but_keep_index_metamethod_fallbacks() {
    static INDEX_CALLS: AtomicUsize = AtomicUsize::new(0);
    static NEWINDEX_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn index(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        INDEX_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(vec![Value::Number(42.0)])
    }
    #[allow(clippy::unnecessary_wraps)]
    fn newindex(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        NEWINDEX_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(Vec::new())
    }

    INDEX_CALLS.store(0, Ordering::Relaxed);
    NEWINDEX_CALLS.store(0, Ordering::Relaxed);
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let index = vm.runtime_mut().register_native(index).unwrap();
    let newindex = vm.runtime_mut().register_native(newindex).unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("present"), Value::Number(1.0))
        .unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__index"),
            Value::NativeFunction(index),
        )
        .unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__newindex"),
            Value::NativeFunction(newindex),
        )
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    vm.runtime_mut().set_global("target", Value::Table(table));

    let program = Chunk::new(
        5,
        vec![
            Constant::String("target".into()),
            Constant::String("present".into()),
            Constant::String("missing".into()),
            Constant::Number(9.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 3),
            Instruction::set_table(0, 1, 2),
            Instruction::get_table(3, 0, 1),
            Instruction::load_constant(1, 2),
            Instruction::get_table(4, 0, 1),
            Instruction::set_table(0, 1, 2),
            Instruction::return_values(3, 2),
        ],
    );

    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Number(9.0), Value::Number(42.0)]
    );
    assert_eq!(INDEX_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(NEWINDEX_CALLS.load(Ordering::Relaxed), 1);
}

#[test]
fn temporary_file_methods_dispatch_as_opaque_userdata_methods() {
    let chunk = Chunk::new(
        6,
        vec![
            Constant::String("io".into()),
            Constant::String("tmpfile".into()),
            Constant::String("write".into()),
            Constant::String("seek".into()),
            Constant::String("read".into()),
            Constant::String("set".into()),
            Constant::String(LuaString::from(vec![0xff, b'x'])),
            Constant::String("*a".into()),
            Constant::Number(2.5),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(2, 1),
            Instruction::get_table(1, 0, 2),
            Instruction::call(1, 0, 1),
            Instruction::load_constant(2, 2),
            Instruction::get_table(2, 1, 2),
            Instruction::move_(3, 1),
            Instruction::load_constant(4, 6),
            Instruction::load_constant(5, 8),
            Instruction::call(2, 3, 1),
            Instruction::load_constant(2, 3),
            Instruction::get_table(2, 1, 2),
            Instruction::move_(3, 1),
            Instruction::load_constant(4, 5),
            Instruction::load_constant(5, 9),
            Instruction::call(2, 3, 1),
            Instruction::load_constant(2, 4),
            Instruction::get_table(2, 1, 2),
            Instruction::move_(3, 1),
            Instruction::load_constant(4, 7),
            Instruction::call(2, 2, 1),
            Instruction::return_(2),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::bytes(vec![0xff, b'x', b'2', b'.', b'5'])]
    );
}

#[test]
fn pairs_and_ipairs_return_vm_callable_iteration_triples() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    vm.runtime_mut().set_global("table", Value::Table(table));
    vm.runtime_mut()
        .set_global("pairs", Value::BuiltinFunction(BuiltinFunction::Pairs));
    vm.runtime_mut()
        .set_global("ipairs", Value::BuiltinFunction(BuiltinFunction::IPairs));

    let pairs = Chunk::new(
        6,
        vec![
            Constant::String("pairs".into()),
            Constant::String("table".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 3),
            Instruction::move_(3, 0),
            Instruction::move_(4, 1),
            Instruction::move_(5, 2),
            Instruction::call(3, 2, 2),
            Instruction::return_values(3, 2),
        ],
    );
    assert_eq!(
        vm.execute(&pairs).unwrap(),
        vec![Value::Number(1.0), Value::from("one")]
    );

    let ipairs = Chunk::new(
        6,
        vec![
            Constant::String("ipairs".into()),
            Constant::String("table".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 3),
            Instruction::move_(3, 0),
            Instruction::move_(4, 1),
            Instruction::move_(5, 2),
            Instruction::call(3, 2, 2),
            Instruction::return_values(3, 2),
        ],
    );
    assert_eq!(
        vm.execute(&ipairs).unwrap(),
        vec![Value::Number(1.0), Value::from("one")]
    );
}

#[test]
fn ipairs_stops_at_the_first_nil_value() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(3.0), Value::from("three"))
        .unwrap();
    vm.runtime_mut().set_global("table", Value::Table(table));
    vm.runtime_mut().set_global(
        "iterator",
        Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
    );
    let chunk = Chunk::new(
        4,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::Number(1.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Nil, Value::Nil]);
}

#[test]
fn fixed_ipairs_iterator_uses_register_results_and_preserves_false_values() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(1.0), Value::Boolean(false))
        .unwrap();
    vm.runtime_mut().set_global(
        "iterator",
        Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
    );
    vm.runtime_mut().set_global("table", Value::Table(table));
    let chunk = Chunk::new(
        3,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            // Results overlap the iterator and state registers.
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );

    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Number(1.0), Value::Boolean(false)]
    );
    assert_eq!(vm.jit_statistics().interpreter_ipairs_steps, 1);
}

#[test]
fn fixed_ipairs_iterator_writes_two_nils_at_termination() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut().set_global(
        "iterator",
        Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
    );
    vm.runtime_mut().set_global("table", Value::Table(table));
    let chunk = Chunk::new(
        3,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );

    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Nil, Value::Nil]);
    assert_eq!(vm.jit_statistics().interpreter_ipairs_steps, 1);
}

#[test]
fn fixed_ipairs_iterator_falls_back_for_coercion_and_metatable_misses() {
    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    vm.runtime_mut().set_global(
        "iterator",
        Value::BuiltinFunction(BuiltinFunction::IPairsIterator),
    );
    vm.runtime_mut().set_global("table", Value::Table(table));
    let chunk = Chunk::new(
        3,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::String("0".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );

    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Number(1.0), Value::from("one")]
    );
    assert_eq!(vm.jit_statistics().interpreter_ipairs_steps, 0);

    let metatable = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    let metatabled_chunk = Chunk::new(
        3,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    // A present value is the result under every profile, so a metatable
    // does not stop the fixed step from answering it.
    assert_eq!(
        vm.execute(&metatabled_chunk).unwrap(),
        vec![Value::Number(1.0), Value::from("one")]
    );
    assert_eq!(vm.jit_statistics().interpreter_ipairs_steps, 1);

    // A miss on a table with a metatable may consult `__index`; that keeps
    // the ordinary builtin call.
    let missing_chunk = Chunk::new(
        3,
        vec![
            Constant::String("iterator".into()),
            Constant::String("table".into()),
            Constant::Number(1.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert_eq!(
        vm.execute(&missing_chunk).unwrap(),
        vec![Value::Nil, Value::Nil]
    );
    assert_eq!(vm.jit_statistics().interpreter_ipairs_steps, 1);
}

#[test]
fn collectgarbage_keeps_tables_and_captured_cells_live_in_the_current_frame() {
    let capture = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "value".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::Number(17.0),
            Constant::String("collectgarbage".into()),
            Constant::String("collect".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::get_global(2, 1),
            Instruction::load_constant(3, 2),
            Instruction::call(2, 1, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_(1),
        ],
        vec![capture],
        vec![],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(17.0)]);
}

#[test]
fn collectgarbage_reports_accounted_bytes_and_retains_control_settings() {
    let pause = Chunk::new(
        3,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("setpause".into()),
            Constant::Number(150.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    let count = Chunk::new(
        2,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("count".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    );
    let step_multiplier = Chunk::new(
        3,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("setstepmul".into()),
            Constant::Number(125.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(vm.execute(&pause).unwrap(), vec![Value::Number(200.0)]);
    assert_eq!(vm.runtime().gc_pause().to_bits(), 150.0_f64.to_bits());
    assert!(
        matches!(vm.execute(&count).unwrap().as_slice(), [Value::Number(value)] if *value > 0.0)
    );
    assert_eq!(
        vm.execute(&step_multiplier).unwrap(),
        vec![Value::Number(200.0)]
    );
    assert_eq!(
        vm.runtime().gc_step_multiplier().to_bits(),
        125.0_f64.to_bits()
    );
}

#[test]
fn modern_full_collection_preserves_an_explicit_gc_stop() {
    fn collect_chunk() -> Chunk {
        Chunk::new(
            2,
            vec![
                Constant::String("collectgarbage".into()),
                Constant::String("collect".into()),
            ],
            vec![
                Instruction::get_global(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::call(0, 1, 0),
                Instruction::return_(0),
            ],
        )
    }

    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua53).unwrap();
    runtime.stop_gc();
    let mut modern = Vm::with_runtime_and_profile(runtime, CompatibilityProfile::Lua53);
    modern.execute(&collect_chunk()).unwrap();
    assert!(!modern.runtime().gc_is_running());

    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    runtime.stop_gc();
    let mut lua51 = Vm::with_runtime_and_profile(runtime, CompatibilityProfile::Lua51);
    lua51.execute(&collect_chunk()).unwrap();
    assert!(lua51.runtime().gc_is_running());
}

#[test]
fn collectgarbage_runs_lua_userdata_finalizers_before_resuming_the_caller() {
    let finalizer = Chunk::with_parts(
        1,
        1,
        vec![Constant::Number(1.0), Constant::String("finalized".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::set_global(0, 1),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
    );
    let collect = Chunk::new(
        2,
        vec![
            Constant::String("doomed".into()),
            Constant::String("collectgarbage".into()),
            Constant::String("collect".into()),
            Constant::String("finalized".into()),
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
    let collect_again = Chunk::new(
        2,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("collect".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
    );

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.runtime_mut().stop_gc();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(finalizer), Vec::new())
        .unwrap();
    vm.runtime_mut()
        .table_set(metatable, &Value::from("__gc"), Value::Closure(closure))
        .unwrap();
    let userdata = vm.runtime_mut().create_userdata().unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("doomed", Value::Userdata(userdata));

    assert_eq!(vm.execute(&collect).unwrap(), vec![Value::Number(1.0)]);
    assert!(
        vm.runtime()
            .raw_metatable(&Value::Userdata(userdata))
            .is_ok()
    );
    vm.execute(&collect_again).unwrap();
    assert!(matches!(
        vm.runtime().raw_metatable(&Value::Userdata(userdata)),
        Err(RuntimeError::InvalidUserdata(handle)) if handle == userdata
    ));
}

#[test]
fn pcall_catches_lua_userdata_finalizer_errors() {
    let finalizer = Chunk::with_parts(
        2,
        1,
        vec![
            Constant::String("error".into()),
            Constant::String("finalizer failed".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
    );
    let program = Chunk::new(
        2,
        vec![
            Constant::String("doomed".into()),
            Constant::String("pcall".into()),
            Constant::String("collectgarbage".into()),
        ],
        vec![
            Instruction::load_nil(0),
            Instruction::set_global(0, 0),
            Instruction::get_global(0, 1),
            Instruction::get_global(1, 2),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.runtime_mut().stop_gc();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let closure = vm
        .runtime_mut()
        .create_closure(Arc::new(finalizer), Vec::new())
        .unwrap();
    vm.runtime_mut()
        .table_set(metatable, &Value::from("__gc"), Value::Closure(closure))
        .unwrap();
    let userdata = vm.runtime_mut().create_userdata().unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("doomed", Value::Userdata(userdata));

    let results = vm.execute(&program).unwrap();
    assert!(
        matches!(
        results.as_slice(),
        [Value::Boolean(false), Value::String(message)]
            if message.as_bytes() == b"finalizer failed"
        ),
        "unexpected protected finalizer results: {results:?}"
    );
}

#[test]
fn automatic_gc_runs_at_instruction_safe_points_and_controls_manual_steps() {
    let mut code = Vec::new();
    // Exercise several collection cycles, not just startup debt: about
    // three nurseries of tables.
    for _ in 0..32_768 {
        code.push(Instruction::new_table(0, 0, 0));
        code.push(Instruction::load_nil(0));
    }
    code.push(Instruction::halt());
    let allocation_loop = Chunk::new(1, vec![], code);
    let stop = Chunk::new(
        2,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("stop".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
    );
    let step_chunk = Chunk::new(
        3,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("step".into()),
            Constant::Number(10_000.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    let restart = Chunk::new(
        2,
        vec![
            Constant::String("collectgarbage".into()),
            Constant::String("restart".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
    );

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let baseline = vm.runtime().gc_count_bytes();
    vm.execute(&allocation_loop).unwrap();
    let after = vm.runtime().gc_count_bytes();
    // The incremental collector conservatively counts allocations made
    // during a cycle as survivors. Its pause threshold is therefore not
    // a fixed multiple of the initial heap. Compare against the same
    // workload with collection disabled, independent of Value layout.
    let mut stopped = Vm::new();
    install_base(stopped.runtime_mut()).unwrap();
    let stopped_baseline = stopped.runtime().gc_count_bytes();
    stopped.runtime_mut().stop_gc();
    stopped.execute(&allocation_loop).unwrap();
    let uncollected = stopped.runtime().gc_count_bytes() - stopped_baseline;
    assert!(
        after.saturating_sub(baseline) < uncollected / 2,
        "automatic collection retained {} of {uncollected} otherwise uncollected bytes",
        after.saturating_sub(baseline)
    );

    vm.execute(&stop).unwrap();
    assert!(!vm.runtime().gc_is_running());
    assert_eq!(vm.execute(&step_chunk).unwrap(), vec![Value::Boolean(true)]);

    vm.execute(&restart).unwrap();
    assert!(vm.runtime().gc_is_running());
}

#[test]
fn top_level_return_services_a_deferred_gc_slice_with_result_roots() {
    let chunk = Chunk::new(
        1,
        Vec::new(),
        vec![Instruction::new_table(0, 0, 0), Instruction::return_(0)],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().stop_gc();
    make_gc_due(vm.runtime_mut());
    vm.runtime_mut().restart_gc();

    let values = vm.execute(&chunk).unwrap();
    let Value::Table(result) = values[0] else {
        panic!("chunk must return its freshly allocated table");
    };
    assert!(vm.runtime().raw_metatable(&Value::Table(result)).is_ok());
    assert_eq!(
        vm.gc_observed_allocation_serial,
        vm.runtime().allocation_serial(),
        "top-level return must not lose a deferred allocation event"
    );
}

#[test]
fn executing_closure_identity_is_a_gc_root() {
    let closure = Chunk::new(
        1,
        vec![Constant::Number(29.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let mut vm = Vm::new();
    let handle = vm
        .install_closure(closure, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    let frame = Frame::new(
        Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        Some(handle),
    );
    let roots = Vm::gc_roots(
        vm.thread_environment,
        &mut CallStack::from(vec![frame]),
        &[],
    );
    assert_eq!(vm.runtime_mut().collect_full(roots).reclaimed(), 0);
    assert_eq!(
        call_closure(&mut vm, Value::Closure(handle)),
        vec![Value::Number(29.0)]
    );
}

#[test]
fn live_roots_clear_dead_registers_but_keep_captured_and_close_slots() {
    // At pc 0 only register 4 is read. Registers 0 to 3 are dead.
    let chunk = Arc::new(Chunk::new(5, Vec::new(), vec![Instruction::return_(4)]));
    let mut vm = Vm::new();
    let mut table = || Value::Table(vm.runtime_mut().create_table().unwrap());
    let registers = vec![table(), table(), table(), Value::Integer(7), table()];
    let mut frame = Frame::new_with_registers(
        Arc::clone(&chunk),
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        None,
        registers,
    );
    frame.extras().close_slots.push(2);
    let mut frames = CallStack::from(vec![frame]);
    vm.capture_slot(&mut frames, 0, 1).unwrap();
    let roots = Vm::gc_roots(vm.thread_environment, &mut frames, &[]);
    let registers = frames.last_registers().unwrap();
    assert!(matches!(registers[0], Value::Nil), "dead table register");
    assert!(matches!(registers[3], Value::Nil), "dead scalar register");
    assert!(matches!(registers[1], Value::Table(_)), "captured register");
    assert!(matches!(registers[2], Value::Table(_)), "close register");
    assert!(matches!(registers[4], Value::Table(_)), "live register");
    // Everything still held was rooted: nothing but the cleared table is
    // reclaimed, and the kept ones stay valid.
    vm.runtime_mut().collect_full(roots);
    let registers = frames.last_registers().unwrap().to_vec();
    for register in [1, 2, 4] {
        assert!(vm.runtime().raw_metatable(&registers[register]).is_ok());
    }
    vm.close_upvalues(&mut frames, 0);
}

#[test]
fn scalar_metamethods_use_dynamic_call_and_unary_length_arguments() {
    fn callable_add(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        if arguments.len() != 3 {
            return Err(RuntimeError::message(
                "__call did not receive itself and both operands",
            ));
        }
        Ok(vec![Value::Number(91.0)])
    }

    fn unary_length(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        if arguments.len() != 1 {
            return Err(RuntimeError::message(
                "__len received an unexpected second operand",
            ));
        }
        Ok(vec![Value::Number(17.0)])
    }

    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("callable_add", callable_add)
        .unwrap();
    vm.runtime_mut()
        .register_global_native("unary_length", unary_length)
        .unwrap();
    let callable_add = vm.runtime().get_global("callable_add");
    let unary_length = vm.runtime().get_global("unary_length");

    let operand = vm.runtime_mut().create_table().unwrap();
    let callable = vm.runtime_mut().create_table().unwrap();
    let callable_meta = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(callable_meta, &Value::from("__call"), callable_add)
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(callable), Some(callable_meta))
        .unwrap();
    let operand_meta = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(operand_meta, &Value::from("__add"), Value::Table(callable))
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Table(operand), Some(operand_meta))
        .unwrap();
    vm.runtime_mut()
        .set_global("metamethod_operand", Value::Table(operand));

    let add = Chunk::new(
        3,
        vec![Constant::String("metamethod_operand".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::move_(1, 0),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(vm.execute(&add).unwrap(), vec![Value::Number(91.0)]);

    let userdata = vm.runtime_mut().create_userdata().unwrap();
    let userdata_meta = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(userdata_meta, &Value::from("__len"), unary_length)
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::Userdata(userdata), Some(userdata_meta))
        .unwrap();
    vm.runtime_mut()
        .set_global("metamethod_userdata", Value::Userdata(userdata));
    let length = Chunk::new(
        2,
        vec![Constant::String("metamethod_userdata".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::length(1, 0),
            Instruction::return_(1),
        ],
    );
    assert_eq!(vm.execute(&length).unwrap(), vec![Value::Number(17.0)]);
}

#[test]
fn concat_preserves_operand_order_and_reports_type_errors() {
    let values = Chunk::new(
        4,
        vec![
            Constant::String("x".into()),
            Constant::Number(2.5),
            Constant::String("!".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::concat(3, 0, 2),
            Instruction::return_(3),
        ],
    );
    assert_eq!(
        Vm::new().execute(&values).unwrap(),
        vec![Value::from("x2.5!")]
    );

    let bad_operand = Chunk::new(
        3,
        vec![],
        vec![
            Instruction::new_table(0, 0, 0),
            Instruction::load_nil(1),
            Instruction::concat(2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&bad_operand).unwrap_err().to_string(),
        "attempt to concatenate a table value"
    );
}

#[test]
fn concat_ranges_reduce_metamethod_operands_from_the_right() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CONCAT_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn concat_handler(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert_eq!(arguments.len(), 2);
        CONCAT_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(vec![Value::from("joined")])
    }

    CONCAT_CALLS.store(0, Ordering::Relaxed);
    let mut vm = Vm::new();
    let handler = vm.runtime_mut().register_native(concat_handler).unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__concat"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    for name in ["left", "middle", "right"] {
        let table = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(table), Some(metatable))
            .unwrap();
        vm.runtime_mut().set_global(name, Value::Table(table));
    }
    let chunk = Chunk::new(
        4,
        vec![
            Constant::String("left".into()),
            Constant::String("middle".into()),
            Constant::String("right".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::concat(3, 0, 2),
            Instruction::return_(3),
        ],
    );
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::from("joined")]);
    assert_eq!(CONCAT_CALLS.load(Ordering::Relaxed), 2);
}

#[test]
fn vararg_compatibility_table_and_table_expansion_preserve_nil_positions() {
    let compatibility = Chunk::with_varargs(
        2,
        0,
        vec![],
        vec![Instruction::return_(1)],
        vec![],
        vec![],
        Some(1),
    );
    let expanded = Chunk::with_varargs(
        1,
        0,
        vec![],
        vec![
            Instruction::new_table(0, 0, 0),
            Instruction::table_vararg(0, 1),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
        None,
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(1.0), Constant::Number(3.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_nil(2),
            Instruction::load_constant(3, 1),
            Instruction::call(0, 3, 1),
            Instruction::return_(0),
        ],
        vec![compatibility],
        vec![],
    );
    let mut vm = Vm::new();
    let result = vm.execute(&program).unwrap();
    let Value::Table(table) = result[0] else {
        panic!("vararg compatibility table was not returned");
    };
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(1.0)).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(2.0)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(3.0)).unwrap(),
        Value::Number(3.0)
    );
    assert_eq!(
        vm.runtime().table_get(table, &Value::from("n")).unwrap(),
        Value::Number(3.0)
    );

    let program = Chunk::with_parts(
        4,
        0,
        vec![Constant::Number(1.0), Constant::Number(3.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::load_nil(2),
            Instruction::load_constant(3, 1),
            Instruction::call(0, 3, 1),
            Instruction::return_(0),
        ],
        vec![expanded],
        vec![],
    );
    let result = vm.execute(&program).unwrap();
    let Value::Table(table) = result[0] else {
        panic!("expanded vararg table was not returned");
    };
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(1.0)).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(2.0)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        vm.runtime().table_get(table, &Value::Number(3.0)).unwrap(),
        Value::Number(3.0)
    );
}

#[test]
fn open_call_results_are_unbounded_and_feed_calls_returns_and_tables() {
    #[allow(clippy::unnecessary_wraps)]
    fn many(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok((1..=300)
            .map(|value| Value::Number(f64::from(value)))
            .collect())
    }
    #[allow(clippy::unnecessary_wraps)]
    fn select_last(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![arguments.last().cloned().unwrap_or(Value::Nil)])
    }

    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("many", many)
        .unwrap();
    vm.runtime_mut()
        .register_global_native("select_last", select_last)
        .unwrap();

    let returned = Chunk::new(
        1,
        vec![Constant::String("many".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::return_open(0, 0),
        ],
    );
    let values = vm.execute(&returned).unwrap();
    assert_eq!(values.len(), 300);
    assert_eq!(values.first(), Some(&Value::Number(1.0)));
    assert_eq!(values.last(), Some(&Value::Number(300.0)));

    let forwarded = Chunk::new(
        3,
        vec![
            Constant::String("many".into()),
            Constant::String("select_last".into()),
            Constant::Number(-1.0),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call_open(1, 1, 1),
            Instruction::return_(1),
        ],
    );
    assert_eq!(vm.execute(&forwarded).unwrap(), vec![Value::Number(300.0)]);

    let tabled = Chunk::new(
        2,
        vec![Constant::String("many".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::new_table(1, 0, 0),
            Instruction::table_open(1, 1),
            Instruction::return_(1),
        ],
    );
    let result = vm.execute(&tabled).unwrap();
    let Value::Table(table) = result[0] else {
        panic!("table constructor did not return a table");
    };
    assert_eq!(
        vm.runtime()
            .table_get(table, &Value::Number(300.0))
            .unwrap(),
        Value::Number(300.0)
    );
}

#[test]
fn table_sort_uses_logarithmic_comparisons_for_large_reverse_input() {
    static COMPARISONS: AtomicUsize = AtomicUsize::new(0);

    fn counted_less(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        COMPARISONS.fetch_add(1, Ordering::Relaxed);
        let (Some(Value::Number(left)), Some(Value::Number(right))) =
            (arguments.first(), arguments.get(1))
        else {
            return Err(RuntimeError::message("numbers required"));
        };
        Ok(vec![Value::Boolean(left < right)])
    }

    const LENGTH: u16 = 4_096;
    COMPARISONS.store(0, Ordering::Relaxed);
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
    for index in 1..=LENGTH {
        vm.runtime_mut()
            .table_set(
                items,
                &Value::Number(f64::from(index)),
                Value::Number(f64::from(LENGTH - index)),
            )
            .unwrap();
    }
    vm.runtime_mut().set_global("sort", sort);
    vm.runtime_mut().set_global("items", Value::Table(items));
    vm.runtime_mut()
        .register_global_native("counted_less", counted_less)
        .unwrap();
    let program = Chunk::new(
        3,
        vec![
            Constant::String("sort".into()),
            Constant::String("items".into()),
            Constant::String("counted_less".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::return_(0),
        ],
    );

    vm.execute(&program).unwrap();
    for index in 1..=LENGTH {
        assert_eq!(
            vm.runtime()
                .table_get(items, &Value::Number(f64::from(index)))
                .unwrap(),
            Value::Number(f64::from(index - 1)),
        );
    }
    assert!(COMPARISONS.load(Ordering::Relaxed) < usize::from(LENGTH) * 32);
}

#[test]
fn table_sort_default_comparator_uses_shared_less_than_metamethod() {
    fn less_by_value(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let (Some(Value::Table(left)), Some(Value::Table(right))) =
            (arguments.first(), arguments.get(1))
        else {
            return Err(RuntimeError::message("tables required"));
        };
        let left = runtime.table_get(*left, &Value::from("value"))?;
        let right = runtime.table_get(*right, &Value::from("value"))?;
        let (Value::Number(left), Value::Number(right)) = (left, right) else {
            return Err(RuntimeError::message("numeric values required"));
        };
        Ok(vec![Value::Boolean(left < right)])
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
    let metatable = vm.runtime_mut().create_table().unwrap();
    let less = vm.runtime_mut().register_native(less_by_value).unwrap();
    vm.runtime_mut()
        .table_set(metatable, &Value::from("__lt"), Value::NativeFunction(less))
        .unwrap();
    let items = vm.runtime_mut().create_table().unwrap();
    for (index, value) in [(1, 3), (2, 1), (3, 2)] {
        let item = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(item, &Value::from("value"), Value::Number(f64::from(value)))
            .unwrap();
        vm.runtime_mut()
            .raw_set_metatable(&Value::Table(item), Some(metatable))
            .unwrap();
        vm.runtime_mut()
            .table_set(items, &Value::Number(f64::from(index)), Value::Table(item))
            .unwrap();
    }
    vm.runtime_mut().set_global("sort", sort);
    vm.runtime_mut().set_global("items", Value::Table(items));
    let program = Chunk::new(
        2,
        vec![
            Constant::String("sort".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_(0),
        ],
    );

    vm.execute(&program).unwrap();
    for (index, expected) in [(1, 1), (2, 2), (3, 3)] {
        let Value::Table(item) = vm
            .runtime()
            .table_get(items, &Value::Number(f64::from(index)))
            .unwrap()
        else {
            panic!("sorted item is a table");
        };
        assert_eq!(
            vm.runtime().table_get(item, &Value::from("value")).unwrap(),
            Value::Number(f64::from(expected)),
        );
    }
}

#[test]
fn table_sort_comparator_errors_are_caught_by_pcall_without_recursion() {
    let failing_comparator = Chunk::with_parts(
        3,
        2,
        vec![Constant::String("bad comparator".into())],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::unary(OpCode::Negate, 2, 2),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::String("sort".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::closure(3, 0),
            Instruction::call(0, 3, 2),
            Instruction::return_values(0, 2),
        ],
        vec![failing_comparator],
        vec![],
    );
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
    vm.runtime_mut()
        .table_set(items, &Value::Number(1.0), Value::Number(2.0))
        .unwrap();
    vm.runtime_mut()
        .table_set(items, &Value::Number(2.0), Value::Number(1.0))
        .unwrap();
    vm.runtime_mut().set_global("sort", sort);
    vm.runtime_mut().set_global("items", Value::Table(items));
    assert!(matches!(
        vm.execute(&program).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));
    assert_eq!(
        vm.runtime().table_get(items, &Value::Number(1.0)).unwrap(),
        Value::Number(2.0)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn table_foreach_and_foreachi_resume_callbacks_and_preserve_call_rules() {
    #[allow(clippy::unnecessary_wraps)]
    fn stop_at_second(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let Some(Value::Number(index)) = arguments.first() else {
            return Err(RuntimeError::message("missing table key"));
        };
        Ok(if index.to_bits() == 2.0_f64.to_bits() {
            vec![Value::Boolean(false)]
        } else {
            vec![Value::Nil]
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn callable_stop(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        let (Some(Value::Table(_)), Some(Value::Number(index))) =
            (arguments.first(), arguments.get(1))
        else {
            return Err(RuntimeError::message("callable arguments are wrong"));
        };
        Ok(if index.to_bits() == 2.0_f64.to_bits() {
            vec![Value::Boolean(false)]
        } else {
            vec![Value::Nil]
        })
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let Value::Table(library) = vm.runtime().get_global("table") else {
        panic!("table is a table");
    };
    let foreach = vm
        .runtime()
        .table_get(library, &Value::from("foreach"))
        .unwrap();
    let foreachi = vm
        .runtime()
        .table_get(library, &Value::from("foreachi"))
        .unwrap();
    let items = vm.runtime_mut().create_table().unwrap();
    for (index, value) in [(1, 10), (2, 20), (3, 30)] {
        vm.runtime_mut()
            .table_set(
                items,
                &Value::Number(f64::from(index)),
                Value::Number(f64::from(value)),
            )
            .unwrap();
    }
    vm.runtime_mut().set_global("foreach", foreach);
    vm.runtime_mut().set_global("foreachi", foreachi);
    vm.runtime_mut().set_global("items", Value::Table(items));
    vm.runtime_mut()
        .register_global_native("stop_at_second", stop_at_second)
        .unwrap();

    let native = Chunk::new(
        3,
        vec![
            Constant::String("foreachi".into()),
            Constant::String("items".into()),
            Constant::String("stop_at_second".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&native).unwrap(), vec![Value::Boolean(false)]);

    let return_value =
        Chunk::with_parts(2, 2, vec![], vec![Instruction::return_(1)], vec![], vec![]);
    let closure = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::String("foreachi".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::closure(2, 0),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
        vec![return_value],
        vec![],
    );
    assert_eq!(vm.execute(&closure).unwrap(), vec![Value::Number(10.0)]);

    let callback = vm.runtime_mut().create_table().unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let native_callback = vm.runtime_mut().register_native(callable_stop).unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__call"),
            Value::NativeFunction(native_callback),
        )
        .unwrap();
    vm.runtime_mut()
        .set_table_metatable(callback, Some(metatable))
        .unwrap();
    vm.runtime_mut()
        .set_global("callback", Value::Table(callback));
    let callable = Chunk::new(
        3,
        vec![
            Constant::String("foreachi".into()),
            Constant::String("items".into()),
            Constant::String("callback".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&callable).unwrap(), vec![Value::Boolean(false)]);

    let inner = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(inner, &Value::Number(1.0), Value::Number(1.0))
        .unwrap();
    let outer = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(outer, &Value::Table(inner), Value::Boolean(true))
        .unwrap();
    vm.runtime_mut().set_global("outer", Value::Table(outer));
    let builtin = Chunk::new(
        3,
        vec![
            Constant::String("foreach".into()),
            Constant::String("outer".into()),
            Constant::String("ipairs".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(
        vm.execute(&builtin).unwrap(),
        vec![Value::BuiltinFunction(BuiltinFunction::IPairsIterator)]
    );

    let failing_callback = Chunk::with_parts(
        3,
        2,
        vec![Constant::String("callback failure".into())],
        vec![
            Instruction::load_constant(2, 0),
            Instruction::unary(OpCode::Negate, 2, 2),
            Instruction::return_(2),
        ],
        vec![],
        vec![],
    );
    let protected = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::String("foreachi".into()),
            Constant::String("items".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::closure(3, 0),
            Instruction::call(0, 3, 2),
            Instruction::return_values(0, 2),
        ],
        vec![failing_callback],
        vec![],
    );
    assert!(matches!(
        vm.execute(&protected).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));
}
