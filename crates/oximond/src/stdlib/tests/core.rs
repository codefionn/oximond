use super::*;

#[test]
fn numeric_index_conversions_match_the_decimal_implementations() {
    let signed = [
        isize::MIN,
        isize::MIN + 1,
        -1,
        0,
        1,
        isize::MAX - 1,
        isize::MAX,
    ];
    for value in signed {
        let old = value.to_string().parse::<f64>().unwrap();
        assert_eq!(
            crate::stdlib::args::isize_to_number(value)
                .unwrap()
                .to_bits(),
            old.to_bits()
        );
    }
    for value in [-9_007_199_254_740_992_i64, 9_007_199_254_740_992_i64]
        .into_iter()
        .filter_map(|value| isize::try_from(value).ok())
    {
        let old = value.to_string().parse::<f64>().unwrap();
        assert_eq!(
            crate::stdlib::args::isize_to_number(value)
                .unwrap()
                .to_bits(),
            old.to_bits()
        );
    }
    let unsigned = [0_usize, 1, 17, usize::MAX - 1, usize::MAX];
    for value in unsigned {
        let old = value.to_string().parse::<f64>().unwrap();
        assert_eq!(
            crate::stdlib::args::usize_to_number(value, "test")
                .unwrap()
                .to_bits(),
            old.to_bits()
        );
    }
    if let Ok(value) = usize::try_from(9_007_199_254_740_992_u64) {
        let old = value.to_string().parse::<f64>().unwrap();
        assert_eq!(
            crate::stdlib::args::usize_to_number(value, "test")
                .unwrap()
                .to_bits(),
            old.to_bits()
        );
    }

    let exact = 9_007_199_254_740_992.0_f64;
    let values = [
        -0.0,
        -1.0,
        0.0,
        1.0,
        123_456.0,
        -123_456.0,
        -exact,
        -f64::from_bits(exact.to_bits() + 1),
        exact,
        f64::from_bits(exact.to_bits() + 1),
        isize::MIN as f64,
        isize::MAX as f64,
        1.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for value in values {
        let old = if value.is_finite() && value.fract() == 0.0 {
            value.to_string().parse::<isize>().ok()
        } else {
            None
        };
        assert_eq!(
            crate::stdlib::args::number_to_isize(value, "test").ok(),
            old
        );
    }
}

#[test]
fn host_output_failures_remain_lua_io_results() {
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("broken pipe"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("broken pipe"))
        }
    }

    let mut runtime = Runtime::new();
    runtime.set_output_writer(FailingWriter);
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let Value::NativeFunction(write) = runtime.table_get(io, &Value::from("write")).unwrap() else {
        panic!("io.write is native");
    };
    let Value::NativeFunction(flush) = runtime.table_get(io, &Value::from("flush")).unwrap() else {
        panic!("io.flush is native");
    };
    let Value::File(stdout) = runtime.table_get(io, &Value::from("stdout")).unwrap() else {
        panic!("io.stdout is a file");
    };
    let Some(file_metatable) = runtime.raw_metatable(&Value::File(stdout)).unwrap() else {
        panic!("io.stdout has a metatable");
    };
    let Value::Table(file_methods) = runtime
        .table_get(file_metatable, &Value::from("__index"))
        .unwrap()
    else {
        panic!("file methods are a table");
    };
    let Value::NativeFunction(file_write) = runtime
        .table_get(file_methods, &Value::from("write"))
        .unwrap()
    else {
        panic!("file:write is native");
    };
    let Value::NativeFunction(file_flush) = runtime
        .table_get(file_methods, &Value::from("flush"))
        .unwrap()
    else {
        panic!("file:flush is native");
    };

    for (function, arguments) in [
        (write, vec![Value::from("output")]),
        (flush, Vec::new()),
        (file_write, vec![Value::File(stdout), Value::from("output")]),
        (file_flush, vec![Value::File(stdout)]),
    ] {
        let result = runtime.call_native(function, &arguments).unwrap();
        assert!(matches!(
            result.as_slice(),
            [Value::Nil, Value::String(_), Value::Number(_)]
        ));
    }
}

#[test]
fn collectgarbage_is_installed_as_a_vm_builtin() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    assert_eq!(
        runtime.get_global("collectgarbage"),
        Value::BuiltinFunction(BuiltinFunction::CollectGarbage)
    );
}

#[test]
fn base_installation_exposes_protected_calls_as_functions() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let protected_call = runtime.get_global("pcall");
    assert_eq!(
        protected_call,
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall)
    );
    assert_eq!(protected_call.type_name(), "function");
    assert!(Runtime::raw_equal(
        &protected_call,
        &Value::BuiltinFunction(BuiltinFunction::ProtectedCall)
    ));
    assert_eq!(
        runtime.get_global("xpcall"),
        Value::BuiltinFunction(BuiltinFunction::ExtendedProtectedCall)
    );
}

#[test]
fn base_installation_exposes_vm_debug_builtins() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(debug) = runtime.get_global("debug") else {
        panic!("debug is a table");
    };
    assert_eq!(
        runtime.table_get(debug, &Value::from("getinfo")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::DebugGetInfo)
    );
    assert_eq!(
        runtime.table_get(debug, &Value::from("traceback")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::DebugTraceback)
    );
    assert_eq!(
        runtime
            .table_get(debug, &Value::from("setupvalue"))
            .unwrap(),
        Value::BuiltinFunction(BuiltinFunction::DebugSetupValue)
    );
    for (name, builtin) in [
        ("gethook", BuiltinFunction::DebugGetHook),
        ("sethook", BuiltinFunction::DebugSetHook),
        ("getlocal", BuiltinFunction::DebugGetLocal),
        ("setlocal", BuiltinFunction::DebugSetLocal),
        ("getupvalue", BuiltinFunction::DebugGetUpvalue),
        ("getregistry", BuiltinFunction::DebugGetRegistry),
    ] {
        assert_eq!(
            runtime.table_get(debug, &Value::from(name)).unwrap(),
            Value::BuiltinFunction(builtin)
        );
    }
    let Value::Table(string) = runtime.get_global("string") else {
        panic!("string is a table");
    };
    assert_eq!(
        runtime.table_get(string, &Value::from("dump")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::StringDump)
    );
}

#[test]
fn upvalue_identity_debug_functions_are_generation_gated() {
    let mut legacy = Runtime::new();
    install_base_with_profile(&mut legacy, CompatibilityProfile::Lua51).unwrap();
    let Value::Table(legacy_debug) = legacy.get_global("debug") else {
        panic!("debug is a table");
    };
    assert_eq!(
        legacy
            .table_get(legacy_debug, &Value::from("upvalueid"))
            .unwrap(),
        Value::Nil
    );
    assert_eq!(
        legacy
            .table_get(legacy_debug, &Value::from("upvaluejoin"))
            .unwrap(),
        Value::Nil
    );

    let mut modern = Runtime::new();
    install_base_with_profile(&mut modern, CompatibilityProfile::Lua52).unwrap();
    let Value::Table(modern_debug) = modern.get_global("debug") else {
        panic!("debug is a table");
    };
    assert_eq!(
        modern
            .table_get(modern_debug, &Value::from("upvalueid"))
            .unwrap(),
        Value::BuiltinFunction(BuiltinFunction::DebugUpvalueId)
    );
    assert_eq!(
        modern
            .table_get(modern_debug, &Value::from("upvaluejoin"))
            .unwrap(),
        Value::BuiltinFunction(BuiltinFunction::DebugUpvalueJoin)
    );
}
