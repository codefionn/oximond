use super::*;
use crate::vm::stack::CallStack;

#[test]
fn debug_uservalues_follow_profile_rules() {
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut vm = Vm::with_profile(profile);
        let userdata = vm.runtime_mut().create_userdata().unwrap();
        let subject = Value::Userdata(userdata);
        let indexed = profile.capabilities().debug.multiple_uservalues;
        assert_eq!(
            vm.debug_getuservalue(std::slice::from_ref(&subject))
                .unwrap(),
            if indexed {
                vec![Value::Nil, Value::Boolean(true)]
            } else {
                vec![Value::Nil]
            }
        );
        let value = if profile == CompatibilityProfile::Lua52 {
            Value::Table(vm.runtime_mut().create_table().unwrap())
        } else {
            Value::from("stored")
        };
        assert_eq!(
            vm.debug_setuservalue(&[subject.clone(), value.clone()])
                .unwrap(),
            vec![subject.clone()]
        );
        assert_eq!(
            vm.debug_getuservalue(std::slice::from_ref(&subject))
                .unwrap(),
            if indexed {
                vec![value.clone(), Value::Boolean(true)]
            } else {
                vec![value.clone()]
            }
        );
        if indexed {
            assert_eq!(
                vm.debug_getuservalue(&[subject.clone(), Value::Integer(2)])
                    .unwrap(),
                vec![Value::Nil]
            );
            vm.runtime_mut()
                .set_userdata_uservalue_count(userdata, 2)
                .unwrap();
            assert_eq!(
                vm.debug_setuservalue(&[subject.clone(), Value::from("second"), Value::Integer(2)])
                    .unwrap(),
                vec![subject.clone()]
            );
            assert_eq!(
                vm.debug_getuservalue(&[subject.clone(), Value::Integer(2)])
                    .unwrap(),
                vec![Value::from("second"), Value::Boolean(true)]
            );
            assert_eq!(
                vm.debug_setuservalue(&[subject.clone(), Value::Nil, Value::Integer(3)])
                    .unwrap(),
                vec![Value::Nil]
            );
        }
    }
}

#[test]
fn debug_file_uservalues_follow_profile_rules() {
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut vm = Vm::with_profile(profile);
        let file = Value::File(vm.runtime_mut().create_temp_file().unwrap());
        let value = Value::Table(vm.runtime_mut().create_table().unwrap());
        let indexed = profile.capabilities().debug.multiple_uservalues;
        assert_eq!(
            vm.debug_getuservalue(std::slice::from_ref(&file)).unwrap(),
            vec![Value::Nil]
        );
        assert_eq!(
            vm.debug_setuservalue(&[file.clone(), value.clone()])
                .unwrap(),
            if indexed {
                vec![Value::Nil]
            } else {
                vec![file.clone()]
            }
        );
        assert_eq!(
            vm.debug_getuservalue(&[file]).unwrap(),
            if indexed {
                vec![Value::Nil]
            } else {
                vec![value]
            }
        );
    }
}

#[test]
fn no_hook_steps_leave_hook_bookkeeping_untouched() {
    // The fused dispatch runs instructions until an observable pause. End
    // the program in a nil index error so the frame survives for the
    // hook-state inspection below.
    let chunk = Arc::new(Chunk::new(
        1,
        Vec::new(),
        vec![Instruction::load_nil(0), Instruction::get_table(0, 0, 0)],
    ));
    let mut vm = Vm::new();
    let environment = vm.runtime().global_environment();
    let mut frame = Frame::new(chunk, Vec::new(), Vec::new(), None, None, environment, None);
    frame.extras().last_hook_line = Some(99);
    frame.extras().last_hook_pc = Some(42);
    frame.extras().skip_hook_pc_once = Some(0);

    let mut frames = CallStack::from(vec![frame]);
    let outcome = vm.step(&mut frames);
    assert!(matches!(outcome, Ok(None) | Err(_)) || outcome.is_err());
    let hook_state = frames[0].extras_ref().expect("hook state exists");
    assert_eq!(hook_state.last_hook_line, Some(99));
    assert_eq!(hook_state.last_hook_pc, Some(42));
    assert_eq!(hook_state.skip_hook_pc_once, Some(0));
}

#[test]
fn removing_and_reinstalling_a_hook_keeps_events_observable() {
    static HOOK_EVENTS: AtomicUsize = AtomicUsize::new(0);

    // Native callbacks use the runtime's fallible callback ABI.
    #[allow(clippy::unnecessary_wraps)]
    fn record_hook(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert!(matches!(arguments, [Value::String(_), Value::Nil]));
        HOOK_EVENTS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    HOOK_EVENTS.store(0, Ordering::SeqCst);
    let mut vm = Vm::new();
    let callback = Value::NativeFunction(vm.runtime_mut().register_native(record_hook).unwrap());
    let environment = vm.runtime().global_environment();
    let mut frames = CallStack::from(vec![Frame::new(
        Arc::new(Chunk::new(1, Vec::new(), vec![Instruction::return_(0)])),
        Vec::new(),
        Vec::new(),
        None,
        None,
        environment,
        None,
    )]);

    vm.debug_sethook(&[callback.clone(), Value::from("c")])
        .unwrap();
    assert!(
        vm.fire_debug_hook(&mut frames, DebugHookEvent::Call)
            .unwrap()
    );
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 1);

    vm.debug_sethook(&[Value::Nil]).unwrap();
    assert!(
        !vm.fire_debug_hook(&mut frames, DebugHookEvent::Call)
            .unwrap()
    );
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 1);

    vm.debug_sethook(&[callback, Value::from("r")]).unwrap();
    assert!(
        vm.fire_debug_hook(&mut frames, DebugHookEvent::Return)
            .unwrap()
    );
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 2);
}

#[test]
fn native_call_subject_only_allocates_for_an_active_call_hook() {
    #[allow(clippy::unnecessary_wraps)]
    fn record_hook(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert!(
            matches!(arguments, [Value::String(event), Value::Nil] if event.as_bytes() == b"call")
        );
        Ok(Vec::new())
    }

    let mut vm = Vm::new();
    let callback = Value::NativeFunction(vm.runtime_mut().register_native(record_hook).unwrap());
    let environment = vm.runtime().global_environment();
    let mut frames = CallStack::from(vec![Frame::new(
        Arc::new(Chunk::new(1, Vec::new(), vec![Instruction::return_(0)])),
        Vec::new(),
        Vec::new(),
        None,
        None,
        environment,
        None,
    )]);

    assert!(
        !vm.observe_native_call(&mut frames, callback.clone(), 0)
            .unwrap()
    );
    assert!(frames[0].extras_ref().is_none());

    vm.debug_sethook(&[callback.clone(), Value::from("r")])
        .unwrap();
    assert!(
        !vm.observe_native_call(&mut frames, callback.clone(), 0)
            .unwrap()
    );
    assert!(frames[0].extras_ref().is_none());

    vm.debug_sethook(&[callback.clone(), Value::from("c")])
        .unwrap();
    assert!(vm.observe_native_call(&mut frames, callback, 0).unwrap());
    assert!(
        frames[0]
            .extras_ref()
            .is_some_and(|extras| extras.debug_hook_subject.is_none())
    );
}

#[test]
fn clear_range_clears_each_register_and_does_not_advance_count_hooks() {
    static HOOK_EVENTS: AtomicUsize = AtomicUsize::new(0);

    #[allow(clippy::unnecessary_wraps)]
    fn record_count_hook(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert!(
            matches!(arguments, [Value::String(event), Value::Nil] if event.as_bytes() == b"count")
        );
        HOOK_EVENTS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    let chunk = Arc::new(Chunk::new(
        3,
        vec![
            Constant::Number(1.0),
            Constant::Number(2.0),
            Constant::Number(3.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::clear_range(0, 3),
            Instruction::return_values(0, 3),
        ],
    ));
    let mut vm = Vm::new();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Nil, Value::Nil, Value::Nil]
    );

    HOOK_EVENTS.store(0, Ordering::SeqCst);
    let callback =
        Value::NativeFunction(vm.runtime_mut().register_native(record_count_hook).unwrap());
    vm.debug_sethook(&[callback, Value::from(""), Value::Number(1.0)])
        .unwrap();
    let environment = vm.runtime().global_environment();
    let mut frames = CallStack::from(vec![Frame::new(
        Arc::new(Chunk::new(
            3,
            Vec::new(),
            vec![Instruction::clear_range(0, 3), Instruction::halt()],
        )),
        Vec::new(),
        Vec::new(),
        None,
        None,
        environment,
        None,
    )]);
    assert!(!vm.fire_count_hook(&mut frames).unwrap());
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 0);

    frames[0].pc = 1;
    assert!(vm.fire_count_hook(&mut frames).unwrap());
    assert_eq!(HOOK_EVENTS.load(Ordering::SeqCst), 1);
}

#[test]
fn setfenv_changes_the_active_frame_global_environment() {
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let environment = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .set_global("private", Value::Table(environment));
    vm.runtime_mut()
        .table_set(
            environment,
            &Value::from("collectgarbage"),
            Value::BuiltinFunction(BuiltinFunction::CollectGarbage),
        )
        .unwrap();
    let chunk = Chunk::new(
        3,
        vec![
            Constant::String(LuaString::from("setfenv")),
            Constant::String(LuaString::from("private")),
            Constant::String(LuaString::from("x")),
            Constant::Number(20.0),
            Constant::Number(1.0),
            Constant::String(LuaString::from("collectgarbage")),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 4),
            Instruction::get_global(2, 1),
            Instruction::call(0, 2, 0),
            Instruction::get_global(0, 5),
            Instruction::call(0, 0, 0),
            Instruction::load_constant(0, 3),
            Instruction::set_global(0, 2),
            Instruction::get_global(1, 2),
            Instruction::return_(1),
        ],
    );

    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(20.0)]);
    assert_eq!(vm.runtime().get_global("x"), Value::Nil);
    assert_eq!(
        vm.runtime()
            .table_get(environment, &Value::from("x"))
            .unwrap(),
        Value::Number(20.0)
    );
}

#[test]
fn level_setfenv_persists_the_target_closure_environment() {
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let environment = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .set_global("changed_environment", Value::Table(environment));

    let inner = Chunk::new(
        3,
        vec![
            Constant::String("setfenv".into()),
            Constant::Number(2.0),
            Constant::String("changed_environment".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::halt(),
        ],
    );
    let inner = vm
        .install_closure(inner, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    vm.runtime_mut().set_global("inner", Value::Closure(inner));
    vm.runtime_mut()
        .table_set(environment, &Value::from("inner"), Value::Closure(inner))
        .unwrap();
    vm.runtime_mut()
        .table_set(environment, &Value::from("answer"), Value::Number(10.0))
        .unwrap();

    let outer = Chunk::new(
        2,
        vec![
            Constant::String("inner".into()),
            Constant::String("answer".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 0),
            Instruction::get_global(1, 1),
            Instruction::return_(1),
        ],
    );
    let outer = vm
        .install_closure(outer, Vec::new(), vm.runtime().global_environment())
        .unwrap();

    assert_eq!(
        call_closure(&mut vm, Value::Closure(outer)),
        vec![Value::Number(10.0)]
    );
    assert_eq!(
        vm.runtime().closure_environment(outer).unwrap(),
        environment
    );
    assert_eq!(
        call_closure(&mut vm, Value::Closure(outer)),
        vec![Value::Number(10.0)]
    );
}

#[test]
fn closure_setfenv_updates_matching_active_frames() {
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let environment = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .set_global("private", Value::Table(environment));
    vm.runtime_mut()
        .table_set(environment, &Value::from("answer"), Value::Number(42.0))
        .unwrap();

    let target = Chunk::new(
        3,
        vec![
            Constant::String("setfenv".into()),
            Constant::String("target".into()),
            Constant::String("private".into()),
            Constant::String("answer".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 0),
            Instruction::get_global(0, 3),
            Instruction::return_(0),
        ],
    );
    let target = vm
        .install_closure(target, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    vm.runtime_mut()
        .set_global("target", Value::Closure(target));
    vm.runtime_mut()
        .table_set(environment, &Value::from("target"), Value::Closure(target))
        .unwrap();

    assert_eq!(
        call_closure(&mut vm, Value::Closure(target)),
        vec![Value::Number(42.0)]
    );
    assert_eq!(
        vm.runtime().closure_environment(target).unwrap(),
        environment
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn debug_getinfo_reads_a_closure_source_and_named_call_site() {
    let source_closure = Chunk::new(
        1,
        vec![Constant::Number(1.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    )
    .with_debug_metadata(DebugMetadata::new("@function.lua", vec![]).with_function_lines(17, 23));
    let function_form = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("getinfo".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
        vec![source_closure],
        vec![],
    );
    let mut vm = Vm::new();
    install_getinfo(&mut vm);
    let Value::Table(info) = vm.execute(&function_form).unwrap().remove(0) else {
        panic!("getinfo returns a table");
    };
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("source"))
            .unwrap(),
        Value::from("@function.lua")
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("short_src"))
            .unwrap(),
        Value::from("function.lua")
    );
    assert_eq!(
        vm.runtime().table_get(info, &Value::from("what")).unwrap(),
        Value::from("Lua")
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("namewhat"))
            .unwrap(),
        Value::from("")
    );
    assert_eq!(
        vm.runtime().table_get(info, &Value::from("name")).unwrap(),
        Value::Nil
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("linedefined"))
            .unwrap(),
        Value::Number(17.0)
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("lastlinedefined"))
            .unwrap(),
        Value::Number(23.0)
    );

    let inspector = Chunk::new(
        3,
        vec![
            Constant::String("getinfo".into()),
            Constant::Number(1.0),
            Constant::String("n".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    let numeric_form = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("F".into())],
        vec![
            Instruction::closure(0, 0),
            Instruction::set_global(0, 0),
            Instruction::get_global(0, 0),
            Instruction::load_nil(1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
        vec![inspector],
        vec![],
    )
    .with_debug_metadata(DebugMetadata::new(
        "@calls.lua",
        vec![CallSite::named(4, "F", CallKind::Global)],
    ));
    let Value::Table(info) = vm.execute(&numeric_form).unwrap().remove(0) else {
        panic!("getinfo returns a table");
    };
    assert_eq!(
        vm.runtime().table_get(info, &Value::from("name")).unwrap(),
        Value::from("F")
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("namewhat"))
            .unwrap(),
        Value::from("global")
    );
}

#[test]
fn named_call_sites_survive_fixed_argument_calls() {
    for arguments in [1, 2] {
        let mut vm = Vm::new();
        install_getinfo(&mut vm);
        let inspector = Chunk::with_parts(
            3,
            arguments,
            vec![
                Constant::String("getinfo".into()),
                Constant::Number(1.0),
                Constant::String("n".into()),
            ],
            vec![
                Instruction::get_global(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::load_constant(2, 2),
                Instruction::call(0, 2, 1),
                Instruction::return_(0),
            ],
            vec![],
            vec![],
        );
        let mut code = vec![Instruction::closure(0, 0)];
        for register in 1..=arguments {
            code.push(Instruction::load_nil(register));
        }
        let call_pc = code.len();
        code.push(Instruction::call(0, arguments, 1));
        code.push(Instruction::return_(0));
        let parent = Chunk::with_parts(3, 0, vec![], code, vec![inspector], vec![])
            .with_debug_metadata(DebugMetadata::new(
                "@named.lua",
                vec![CallSite::named(call_pc, "inspector", CallKind::Global)],
            ));
        let Value::Table(info) = vm.execute(&parent).unwrap().remove(0) else {
            panic!("getinfo returns a table");
        };
        assert_eq!(
            vm.runtime().table_get(info, &Value::from("name")).unwrap(),
            Value::from("inspector")
        );
    }
}

#[test]
fn debug_short_source_matches_lua51_chunk_id_rules() {
    assert_eq!(
        debug_short_source(&LuaString::from("function f () end")),
        LuaString::from("[string \"function f () end\"]")
    );
    assert_eq!(
        debug_short_source(&LuaString::from("\nfunction f () end")),
        LuaString::from("[string \"...\"]")
    );
    assert_eq!(
        debug_short_source(&LuaString::from("@xuxu")),
        LuaString::from("xuxu")
    );
    assert_eq!(
        debug_short_source(&LuaString::from(format!("@{}t", "p".repeat(1_000)))),
        LuaString::from(format!("...{}t", "p".repeat(55)))
    );
    assert_eq!(
        debug_short_source(&LuaString::from("=named chunk")),
        LuaString::from("named chunk")
    );
}

#[test]
fn traceback_keeps_lua_frames_and_non_string_error_objects() {
    let trace = Traceback {
        debug_frames: Vec::new(),
        frames: vec![
            TraceFrame {
                source: LuaString::from("@trace.lua"),
                line: Some(42),
                call_site: Some(ActiveCallSite {
                    name: LuaString::from("recurse"),
                    kind: CallKind::Global,
                }),
                is_main: false,
                extra: TraceExtra::default(),
            },
            TraceFrame {
                source: LuaString::from("@trace.lua"),
                line: Some(11),
                call_site: None,
                is_main: false,
                extra: TraceExtra::default(),
            },
        ],
    };
    assert_eq!(
        format_traceback(Some("trace.lua:42: boom"), &trace, 0),
        "trace.lua:42: boom\nstack traceback:\n\ttrace.lua:42: in function 'recurse'\n\ttrace.lua:11: in function <trace.lua:11>"
    );

    let mut vm = Vm::new();
    let table = vm.runtime_mut().create_table().unwrap();
    assert_eq!(
        vm.traceback_handler_result(&[Value::Table(table)], Some(&trace)),
        Value::Table(table)
    );
}

#[test]
fn traceback_resolves_nested_call_names_and_leaves_tail_called_frames_unnamed() {
    use crate::vm::debug_info::capture_traceback;

    let vm = Vm::new();
    let environment = vm.runtime().global_environment();
    let parent_chunk = Chunk::new(1, vec![], vec![Instruction::return_(0)]).with_debug_metadata(
        DebugMetadata::new(
            "@parent.lua",
            vec![CallSite::named(0, "ordinary", CallKind::Global)],
        ),
    );
    let mut parent = Frame::new(
        Arc::new(parent_chunk),
        vec![],
        vec![],
        None,
        None,
        environment,
        None,
    );
    parent.pc = 1;
    let child_chunk = Arc::new(
        Chunk::new(1, vec![], vec![Instruction::return_(0)]).with_debug_metadata(
            DebugMetadata::new("@child.lua", vec![]).with_function_lines(1, 1),
        ),
    );
    let child = Frame::new(
        child_chunk,
        vec![],
        vec![],
        Some(ReturnTarget::Direct(super::CallReturn {
            base: 0,
            results: 1,
        })),
        None,
        environment,
        None,
    );
    let ordinary = capture_traceback(&CallStack::from(vec![parent, child]));
    assert!(format_traceback(None, &ordinary, 0).contains("in function 'ordinary'"));

    let mut tail = Frame::new(
        Arc::new(
            Chunk::new(1, vec![], vec![Instruction::return_(0)]).with_debug_metadata(
                DebugMetadata::new("@tail.lua", vec![]).with_function_lines(1, 1),
            ),
        ),
        vec![],
        vec![],
        None,
        Some(ActiveCallSite {
            name: LuaString::from("tail_target"),
            kind: CallKind::Global,
        }),
        environment,
        None,
    );
    tail.tail_calls = 1;
    let tail_trace = capture_traceback(&CallStack::from(vec![tail]));
    let text = format_traceback(None, &tail_trace, 0);
    // Stock Lua names no function that tail calls replaced the caller of:
    // `lua5.1` prints `in function <tail.lua:1>` before `(tail call): ?`.
    assert!(text.contains("in function <tail.lua:1>"));
    assert!(!text.contains("tail_target"));
    assert!(text.contains("(tail call): ?"));
}

#[test]
fn debug_getinfo_returns_nil_for_an_out_of_range_level() {
    let chunk = Chunk::new(
        2,
        vec![Constant::String("getinfo".into()), Constant::Number(99.0)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    install_getinfo(&mut vm);
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Nil]);
}

#[test]
fn debug_getinfo_reports_the_active_instruction_line_or_minus_one() {
    let chunk = Chunk::new(
        2,
        vec![Constant::String("getinfo".into()), Constant::Number(1.0)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    )
    .with_debug_metadata(
        DebugMetadata::new(
            "lines.lua",
            vec![CallSite::named(2, "getinfo", CallKind::Global)],
        )
        .with_line_info(vec![9, 9, 42, 42]),
    );
    let mut vm = Vm::new();
    install_getinfo(&mut vm);
    let Value::Table(info) = vm.execute(&chunk).unwrap().remove(0) else {
        panic!("getinfo returns a table");
    };
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("currentline"))
            .unwrap(),
        Value::Number(42.0)
    );

    let Value::Table(info) = vm
        .debug_get_info(
            &CallStack::new(),
            &[Value::BuiltinFunction(BuiltinFunction::DebugGetInfo)],
        )
        .unwrap()
        .remove(0)
    else {
        panic!("getinfo returns a table");
    };
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("currentline"))
            .unwrap(),
        Value::Number(-1.0)
    );
}

#[test]
fn debug_getinfo_exposes_distinct_executable_lines_for_lua_closures() {
    let closure = Chunk::new(
        1,
        vec![],
        vec![Instruction::load_nil(0), Instruction::halt()],
    )
    .with_debug_metadata(DebugMetadata::new("@lines.lua", vec![]).with_line_info(vec![8, 11]));
    let entry = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::String("getinfo".into()),
            Constant::String("L".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
        vec![closure],
        vec![],
    );
    let mut vm = Vm::new();
    install_getinfo(&mut vm);
    let Value::Table(info) = vm.execute(&entry).unwrap().remove(0) else {
        panic!("getinfo returns a table");
    };
    let Value::Table(active_lines) = vm
        .runtime()
        .table_get(info, &Value::from("activelines"))
        .unwrap()
    else {
        panic!("L returns an active-line table for a Lua closure");
    };
    assert_eq!(
        vm.runtime()
            .table_get(active_lines, &Value::Number(8.0))
            .unwrap(),
        Value::Boolean(true)
    );
    assert_eq!(
        vm.runtime()
            .table_get(active_lines, &Value::Number(11.0))
            .unwrap(),
        Value::Boolean(true)
    );
    assert_eq!(
        vm.runtime()
            .table_get(active_lines, &Value::Number(9.0))
            .unwrap(),
        Value::Nil
    );
}

#[test]
fn debug_upvalue_identity_and_join_use_shared_cells() {
    let body = Arc::new(Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "captured".into(),
            source: UpvalueSource::Detached,
        }],
    ));
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua52);
    let first_cell = vm.runtime_mut().create_cell(Value::Number(11.0)).unwrap();
    let second_cell = vm.runtime_mut().create_cell(Value::Number(22.0)).unwrap();
    let first = vm
        .runtime_mut()
        .create_closure(Arc::clone(&body), vec![first_cell])
        .unwrap();
    let second = vm
        .runtime_mut()
        .create_closure(body, vec![second_cell])
        .unwrap();

    let first_id = vm
        .debug_upvalueid(&[Value::Closure(first), Value::Number(1.0)])
        .unwrap()
        .remove(0);
    assert_eq!(
        vm.debug_upvalueid(&[Value::Closure(first), Value::Number(1.0)])
            .unwrap(),
        vec![first_id.clone()]
    );
    assert_eq!(
        vm.debug_upvalueid(&[Value::Closure(first), Value::Integer(1)])
            .unwrap(),
        vec![first_id.clone()]
    );
    let second_id = vm
        .debug_upvalueid(&[Value::Closure(second), Value::Number(1.0)])
        .unwrap()
        .remove(0);
    assert!(matches!(first_id, Value::Userdata(_)));
    assert_ne!(first_id, second_id);

    assert_eq!(
        vm.debug_upvaluejoin(
            &mut CallStack::new(),
            &[
                Value::Closure(first),
                Value::Number(1.0),
                Value::Closure(second),
                Value::Number(1.0),
            ],
        )
        .unwrap(),
        [] as [Value; 0]
    );
    assert_eq!(
        vm.debug_upvalueid(&[Value::Closure(first), Value::Number(1.0)])
            .unwrap(),
        vec![second_id]
    );
    vm.runtime_mut()
        .cell_set(second_cell, Value::Number(37.0))
        .unwrap();
    assert_eq!(
        call_closure(&mut vm, Value::Closure(first)),
        vec![Value::Number(37.0)]
    );
}

#[test]
fn debug_getinfo_omits_names_for_an_indirect_protected_call() {
    let inspector = Chunk::new(
        3,
        vec![
            Constant::String("getinfo".into()),
            Constant::Number(1.0),
            Constant::String("n".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("pcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 2),
            Instruction::return_(1),
        ],
        vec![inspector],
        vec![],
    );
    let mut vm = Vm::new();
    install_getinfo(&mut vm);
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    let Value::Table(info) = vm.execute(&program).unwrap().remove(0) else {
        panic!("getinfo returns a table");
    };
    assert_eq!(
        vm.runtime().table_get(info, &Value::from("name")).unwrap(),
        Value::Nil
    );
    assert_eq!(
        vm.runtime()
            .table_get(info, &Value::from("namewhat"))
            .unwrap(),
        Value::from("")
    );
}

#[test]
fn lua52_loaded_chunk_environment_is_an_upvalue_and_accepts_any_value() {
    let chunk = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "_ENV".into(),
            source: UpvalueSource::Detached,
        }],
    );
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua52);
    let closure = vm.install_loaded_closure(chunk).unwrap();
    assert_eq!(
        vm.debug_getupvalue(&[Value::Closure(closure), Value::Number(1.0)])
            .unwrap(),
        vec![
            Value::from("_ENV"),
            Value::Table(vm.runtime().global_environment())
        ]
    );

    let loaded =
        vm.apply_load_environment(vec![Value::Closure(closure)], Some(&Value::Number(123.0)));
    assert_eq!(loaded, vec![Value::Closure(closure)]);
    assert_eq!(
        call_closure(&mut vm, Value::Closure(closure)),
        vec![Value::Number(123.0)]
    );
    assert_eq!(
        vm.debug_getupvalue(&[Value::Closure(closure), Value::Number(1.0)])
            .unwrap(),
        vec![Value::from("_ENV"), Value::Number(123.0)]
    );
    assert_eq!(
        vm.debug_setupvalue(
            &mut CallStack::new(),
            &[
                Value::Closure(closure),
                Value::Number(1.0),
                Value::from("changed"),
            ],
        )
        .unwrap(),
        vec![Value::from("_ENV")]
    );
    assert_eq!(
        call_closure(&mut vm, Value::Closure(closure)),
        vec![Value::from("changed")]
    );
}

#[test]
fn binary_load_detaches_captures_and_setupvalue_uses_descriptor_names() {
    let chunk = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: LuaString::from(vec![0xff, b'x']),
            source: UpvalueSource::Detached,
        }],
    );
    let mut vm = Vm::new();
    let live_cell = vm.create_detached_cell().unwrap();
    vm.runtime_mut()
        .cell_set(live_cell, Value::Number(7.0))
        .unwrap();
    let original = vm
        .install_closure(chunk, vec![live_cell], vm.runtime.global_environment())
        .unwrap();
    let Value::String(bytes) = vm
        .string_dump(&[Value::Closure(original)])
        .unwrap()
        .remove(0)
    else {
        panic!("string.dump returns bytes");
    };
    let Value::Closure(first) = vm
        .load_string(&bytes, &LuaString::from("ignored"))
        .remove(0)
    else {
        panic!("binary load returns a closure");
    };
    let Value::Closure(second) = vm
        .load_string(&bytes, &LuaString::from("ignored"))
        .remove(0)
    else {
        panic!("binary load returns a closure");
    };
    assert_eq!(
        call_closure(&mut vm, Value::Closure(first)),
        vec![Value::Nil]
    );
    assert_eq!(
        call_closure(&mut vm, Value::Closure(second)),
        vec![Value::Nil]
    );
    assert_eq!(
        vm.debug_setupvalue(
            &mut CallStack::new(),
            &[
                Value::Closure(first),
                Value::Number(1.0),
                Value::Number(11.0),
            ]
        )
        .unwrap(),
        vec![Value::bytes(vec![0xff, b'x'])]
    );
    assert_eq!(
        call_closure(&mut vm, Value::Closure(first)),
        vec![Value::Number(11.0)]
    );
    assert_eq!(
        call_closure(&mut vm, Value::Closure(second)),
        vec![Value::Nil]
    );
    assert_eq!(
        vm.debug_setupvalue(
            &mut CallStack::new(),
            &[Value::Closure(first), Value::Number(3.0), Value::Nil],
        )
        .unwrap(),
        [] as [Value; 0]
    );
    assert_eq!(
        vm.debug_setupvalue(
            &mut CallStack::new(),
            &[
                Value::BuiltinFunction(BuiltinFunction::Next),
                Value::Number(1.0),
                Value::Nil,
            ]
        )
        .unwrap(),
        [] as [Value; 0]
    );
}

#[test]
fn close_upvalue_detaches_a_closure_from_later_register_writes() {
    let captured = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "iteration".into(),
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
            Instruction::close_upvalue(0),
            Instruction::load_constant(0, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_(1),
        ],
        vec![captured],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(1.0)]
    );
}

#[test]
fn siblings_share_cells_and_parent_upvalues_chain() {
    let setter = Chunk::with_parts(
        1,
        0,
        vec![Constant::Number(7.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::set_upvalue(0, 0),
            Instruction::return_(0),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let getter = Chunk::with_parts(
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
        3,
        0,
        vec![Constant::Number(0.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::closure(2, 1),
            Instruction::call(1, 0, 1),
            Instruction::call(2, 0, 1),
            Instruction::return_(2),
        ],
        vec![setter, getter],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(7.0)]
    );
}

#[test]
fn parent_upvalue_reuses_the_same_cell_across_levels() {
    let leaf = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentUpvalue(0),
        }],
    );
    let middle = Chunk::with_parts(
        1,
        0,
        vec![],
        vec![Instruction::closure(0, 0), Instruction::return_(0)],
        vec![leaf],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(9.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(1, 0, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_(1),
        ],
        vec![middle],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(9.0)]
    );
}

#[test]
fn proper_tail_calls_replace_lua_frames_and_keep_upvalues() {
    let recursive = Chunk::with_parts(
        3,
        1,
        vec![Constant::Number(0.0), Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::LessEqual, 2, 0, 1),
            Instruction::jump_if_false(2, 1),
            Instruction::return_(0),
            Instruction::get_upvalue(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::binary(OpCode::Subtract, 2, 0, 2),
            Instruction::tail_call(1, 1),
            Instruction::return_open(1, 0),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(30_000.0)],
        vec![
            Instruction::closure(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
        vec![recursive],
        vec![],
    );
    assert_eq!(
        Vm::new().execute(&program).unwrap(),
        vec![Value::Number(0.0)]
    );
}

#[test]
fn protected_failures_resume_nearest_handler_and_preserve_closed_upvalues() {
    let fail = Chunk::with_parts(
        1,
        0,
        vec![Constant::String("bad".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::unary(OpCode::Negate, 0, 0),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
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
        vec![fail.clone()],
        vec![],
    );
    let saved = Chunk::with_parts(
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
    let captures_then_fails = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::Number(13.0),
            Constant::String("saved".into()),
            Constant::String("bad".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::closure(1, 0),
            Instruction::set_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::unary(OpCode::Negate, 2, 2),
            Instruction::return_(0),
        ],
        vec![saved],
        vec![],
    );
    let program = Chunk::with_parts(
        4,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::String("saved".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 3),
            Instruction::get_global(2, 1),
            Instruction::call(2, 0, 1),
            Instruction::return_(2),
        ],
        vec![captures_then_fails],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(13.0)]);

    let nested_program = Chunk::with_parts(
        3,
        0,
        vec![Constant::String("pcall".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 3),
            Instruction::return_values(0, 3),
        ],
        vec![nested],
        vec![],
    );
    assert!(matches!(
        vm.execute(&nested_program).unwrap().as_slice(),
        [
            Value::Boolean(true),
            Value::Boolean(false),
            Value::String(_)
        ]
    ));
}

#[test]
fn hook_transfer_metadata_tracks_live_results_and_clears_after_callback() {
    fn inspect(vm: &mut Vm, expected: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        let info = vm.host_stack_info(1).expect("interrupted Lua frame");
        assert_eq!(
            (info.first_transfer, info.transfer_count),
            ((expected >> 32) as usize, (expected & 0xffff_ffff) as usize)
        );
        let hook = vm.host_stack_info(0).expect("native hook");
        assert_eq!((hook.first_transfer, hook.transfer_count), (0, 0));
        Ok(Vec::new())
    }
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut vm = Vm::with_profile(profile);
        let environment = vm.runtime().global_environment();
        let mut chunk = Chunk::new(8, Vec::new(), vec![Instruction::return_values(3, 2)]);
        chunk.parameter_count = 2;
        let mut frames = CallStack::from(vec![Frame::new(
            Arc::new(chunk),
            Vec::new(),
            Vec::new(),
            None,
            None,
            environment,
            None,
        )]);
        let modern = profile != CompatibilityProfile::Lua53;
        for (event, expected) in [
            (DebugHookEvent::Call, if modern { (1, 2) } else { (0, 0) }),
            (DebugHookEvent::Return, if modern { (4, 2) } else { (0, 0) }),
            (DebugHookEvent::Line(1), (0, 0)),
        ] {
            let callback = vm
                .register_host_function(
                    inspect,
                    ((expected.0 as u64) << 32) | expected.1 as u64,
                    Vec::new(),
                )
                .unwrap();
            vm.debug_sethook(&[callback, Value::from("crl")]).unwrap();
            frames[0].pc = usize::from(matches!(event, DebugHookEvent::Return));
            assert!(vm.fire_debug_hook(&mut frames, event).unwrap());
            assert_eq!(frames[0].extras_ref().unwrap().debug_hook_transfer, (0, 0));
        }
    }
}

#[test]
fn return_hooks_count_open_and_vararg_results_without_consuming_them() {
    fn inspect(vm: &mut Vm, expected: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        let info = vm.host_stack_info(1).unwrap();
        assert_eq!(
            (info.first_transfer, info.transfer_count),
            (2, expected as usize)
        );
        Ok(Vec::new())
    }
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        for opcode in [OpCode::ReturnOpen, OpCode::ReturnVararg] {
            let mut vm = Vm::with_profile(profile);
            let callback = vm.register_host_function(inspect, 3, Vec::new()).unwrap();
            vm.debug_sethook(&[callback, Value::from("r")]).unwrap();
            let environment = vm.runtime().global_environment();
            let mut chunk = Chunk::new(5, Vec::new(), vec![Instruction::abc(opcode, 1, 1, 0)]);
            chunk.is_vararg = true;
            let mut frame = Frame::new(
                Arc::new(chunk),
                vec![Value::from("a"), Value::from("b")],
                Vec::new(),
                None,
                None,
                environment,
                None,
            );
            frame.set_open_results(vec![Value::from("x"), Value::from("y")]);
            frame.pc = 1;
            let mut frames = CallStack::from(vec![frame]);
            assert!(
                vm.fire_debug_hook(&mut frames, DebugHookEvent::Return)
                    .unwrap()
            );
            assert_eq!(frames[0].open_values().unwrap().len(), 2);
            assert_eq!(frames.varargs(0).len(), 2);
        }
    }
}

#[test]
fn failed_lua_hook_unwinds_hook_state_before_the_next_callback() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    let environment = vm.runtime().global_environment();
    let hook = vm
        .install_closure(
            Chunk::new(
                2,
                Vec::new(),
                vec![Instruction::load_nil(0), Instruction::get_table(1, 0, 0)],
            ),
            Vec::new(),
            environment,
        )
        .unwrap();
    vm.debug_sethook(&[Value::Closure(hook), Value::from("c")])
        .unwrap();
    let mut chunk = Chunk::new(2, Vec::new(), vec![Instruction::return_values(0, 2)]);
    chunk.parameter_count = 2;
    let mut frames = CallStack::from(vec![Frame::new(
        Arc::new(chunk),
        Vec::new(),
        Vec::new(),
        None,
        None,
        environment,
        None,
    )]);
    assert!(
        vm.fire_debug_hook(&mut frames, DebugHookEvent::Call)
            .unwrap()
    );
    assert_eq!(frames[0].extras_ref().unwrap().debug_hook_transfer, (1, 2));
    let error = vm.interpret(&mut frames, false).unwrap_err();
    assert!(matches!(
        vm.begin_error_unwind(&mut frames, error).unwrap(),
        UnwindAction::Return(_)
    ));
    assert_eq!(vm.debug_hook_depth, 0);
    assert!(!vm.debug_hooks.get(&None).unwrap().running);
    assert!(frames.is_empty());
}

#[test]
fn host_hook_replacement_does_not_execute_the_previous_callback() {
    fn unexpected(_: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        panic!("setting a host hook must not run Lua or the previous hook")
    }
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    let old = vm
        .register_host_function(unexpected, 0, Vec::new())
        .unwrap();
    let new = vm
        .register_host_function(unexpected, 1, Vec::new())
        .unwrap();
    vm.set_host_debug_hook(None, Some(old.clone()), 8, 1)
        .unwrap();
    let thread = vm.create_host_coroutine().unwrap();
    vm.current_coroutine = Some(thread);
    vm.set_host_debug_hook(None, Some(new.clone()), 3, 0)
        .unwrap();
    assert!(vm.unregister_host_function(old));
    assert_eq!(vm.debug_hooks[&None].function, new);
    assert!(vm.debug_hooks[&None].mask.call);
    assert!(vm.debug_hooks[&None].mask.return_);
    assert_eq!(vm.debug_hooks[&None].count, 0);
    assert!(!vm.debug_hooks.contains_key(&Some(thread)));
    vm.set_host_debug_hook(Some(thread), Some(new), 12, 2)
        .unwrap();
    assert!(vm.debug_hooks[&Some(thread)].mask.line);
    assert_eq!(vm.debug_hooks[&Some(thread)].count, 2);
    vm.set_host_debug_hook(None, None, 0, 0).unwrap();
    assert!(!vm.debug_hooks.contains_key(&None));
    assert!(vm.debug_hooks.contains_key(&Some(thread)));
}

#[test]
fn host_error_handler_does_not_reenter_a_hook_that_replaced_itself_before_error() {
    fn kill(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        let calls = match vm.runtime().get_global("kill_count") {
            Value::Integer(calls) => calls,
            _ => 0,
        };
        vm.runtime_mut()
            .set_global("kill_count", Value::Integer(calls + 1));
        let hook = vm.runtime().get_global("kill_hook");
        vm.set_host_debug_hook(None, Some(hook), 4, 0)?;
        Err(RuntimeError::Raised(Value::from("killed by SCRIPT KILL")).into())
    }
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut vm = Vm::with_profile(profile);
        let hook = vm.register_host_function(kill, 0, Vec::new()).unwrap();
        vm.runtime_mut().set_global("kill_hook", hook.clone());
        vm.set_host_debug_hook(None, Some(hook), 8, 1).unwrap();
        let target = Chunk::new(
            1,
            vec![Constant::Number(1.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        )
        .with_debug_metadata(
            DebugMetadata::new("@script.lua", Vec::new()).with_line_info(vec![1, 1]),
        );
        let error = vm.execute(&target).unwrap_err();
        assert_eq!(error.to_string(), "killed by SCRIPT KILL");
        assert!(error.is_hook_error());
        let mut handler = Chunk::new(
            2,
            vec![Constant::String("handled kill".into())],
            vec![Instruction::load_constant(1, 0), Instruction::return_(1)],
        )
        .with_debug_metadata(
            DebugMetadata::new("@handler.lua", Vec::new()).with_line_info(vec![1, 2]),
        );
        handler.parameter_count = 1;
        let handler = vm.load_chunk(&handler).unwrap();
        assert_eq!(
            vm.call_value_without_hooks(
                handler.clone(),
                vec![Value::from("killed by SCRIPT KILL")]
            )
            .unwrap(),
            vec![Value::from("handled kill")]
        );
        assert_eq!(vm.runtime().get_global("kill_count"), Value::Integer(1));
        assert_eq!(vm.debug_hook_depth, 0);
        assert!(vm.execute(&target).is_err());
        assert_eq!(vm.runtime().get_global("kill_count"), Value::Integer(2));
    }
}

#[test]
fn lua_message_handlers_suppress_only_errors_that_escape_a_hook() {
    fn kill(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        let count = match vm.runtime().get_global("hook_count") {
            Value::Integer(count) => count,
            _ => 0,
        };
        vm.runtime_mut()
            .set_global("hook_count", Value::Integer(count + 1));
        let hook = vm.runtime().get_global("kill_hook");
        vm.set_host_debug_hook(None, Some(hook), 4, 0)?;
        Err(RuntimeError::Raised(Value::from("kill")).into())
    }
    fn record(vm: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        let count = match vm.runtime().get_global("hook_count") {
            Value::Integer(count) => count,
            _ => 0,
        };
        vm.runtime_mut()
            .set_global("hook_count", Value::Integer(count + 1));
        Ok(Vec::new())
    }
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut vm = Vm::with_profile(profile);
        let hook = vm.register_host_function(kill, 0, Vec::new()).unwrap();
        vm.runtime_mut().set_global("kill_hook", hook.clone());
        let mut handler = Chunk::new(
            2,
            vec![Constant::String("handled kill".into())],
            vec![Instruction::load_constant(1, 0), Instruction::return_(1)],
        )
        .with_debug_metadata(
            DebugMetadata::new("@handler.lua", Vec::new()).with_line_info(vec![1, 2]),
        );
        handler.parameter_count = 1;
        let handler = vm.load_chunk(&handler).unwrap();
        vm.set_host_debug_hook(None, Some(hook), 8, 1).unwrap();
        let target = Chunk::new(
            1,
            vec![Constant::Number(1.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        )
        .with_debug_metadata(
            DebugMetadata::new("@script.lua", Vec::new()).with_line_info(vec![1, 1]),
        );
        assert_eq!(
            vm.execute_with_message_handler(&target, Vec::new(), handler.clone())
                .unwrap_err()
                .to_string(),
            "handled kill"
        );
        assert_eq!(vm.runtime().get_global("hook_count"), Value::Integer(1));
        assert_eq!(vm.debug_hook_depth, 0);
        let hook = vm.register_host_function(record, 0, Vec::new()).unwrap();
        vm.set_host_debug_hook(None, Some(hook), 4, 0).unwrap();
        vm.runtime_mut().set_global("hook_count", Value::Integer(0));
        let plain_error = Chunk::new(
            1,
            Vec::new(),
            vec![
                Instruction::load_nil(0),
                Instruction::get_table(0, 0, 0),
                Instruction::halt(),
            ],
        );
        assert_eq!(
            vm.execute_with_message_handler(&plain_error, Vec::new(), handler)
                .unwrap_err()
                .to_string(),
            "handled kill"
        );
        assert!(
            matches!(vm.runtime().get_global("hook_count"), Value::Integer(count) if count >= 2)
        );
    }
}

#[test]
fn hook_error_tags_preserve_objects_memory_status_and_suppression_cleanup() {
    fn memory(_: &mut Vm, _: u64, _: &[Value]) -> VmResult<Vec<Value>> {
        Err(RuntimeError::memory_error().into())
    }
    let mut vm = Vm::new();
    let table = Value::Table(vm.runtime_mut().create_table().unwrap());
    let error = VmError::HookError(Box::new(RuntimeError::Raised(table.clone()).into()));
    assert!(error.is_hook_error());
    assert_eq!(crate::vm::errors::vm_error_object(&error).unwrap(), table);
    let error = VmError::HookError(Box::new(RuntimeError::memory_error().into()));
    assert!(error.is_memory_error());
    assert_eq!(error.to_string(), "not enough memory");
    let native = vm.register_host_function(memory, 0, Vec::new()).unwrap();
    assert!(
        vm.call_value_without_hooks(native, Vec::new())
            .unwrap_err()
            .is_memory_error()
    );
    assert_eq!(vm.debug_hook_depth, 0);
}
