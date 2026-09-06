use super::*;

#[test]
fn assert_returns_all_arguments_on_success() {
    let mut runtime = Runtime::new();
    let function = install_base(&mut runtime).unwrap();
    assert_eq!(
        runtime
            .call_native(function, &[Value::from(true), Value::from("detail")])
            .unwrap(),
        vec![Value::from(true), Value::from("detail")]
    );
}

#[test]
fn assert_rejects_false_and_nil() {
    let mut runtime = Runtime::new();
    let function = install_base(&mut runtime).unwrap();
    assert_eq!(
        runtime.call_native(function, &[Value::from(false), Value::from("bad")]),
        Err(RuntimeError::message("bad"))
    );
    assert_eq!(
        runtime.call_native(function, &[Value::Nil]),
        Err(RuntimeError::message("assertion failed!"))
    );
}

#[test]
fn base_installation_exposes_lua_type_names() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(type_of) = runtime.get_global("type") else {
        panic!("type is native");
    };
    let table = Value::Table(runtime.create_table().unwrap());
    let cases = [
        (Value::Nil, "nil"),
        (Value::from(true), "boolean"),
        (Value::from(1.0), "number"),
        (Value::from("oximond"), "string"),
        (table, "table"),
        (Value::NativeFunction(type_of), "function"),
    ];
    for (value, expected) in cases {
        assert_eq!(
            runtime.call_native(type_of, &[value]).unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert_eq!(
        runtime.call_native(type_of, &[]).unwrap_err(),
        RuntimeError::argument_error(1, "value expected")
    );
    assert_eq!(
        runtime
            .call_native(type_of, &[Value::from("first"), Value::from(2.0)])
            .unwrap(),
        vec![Value::from("string")]
    );
    // `type` is a single-value native whose names are shared: two calls
    // return one string, not two allocations.
    assert!(matches!(
        runtime.native_plan(type_of),
        crate::runtime::NativePlan::Scalar(_)
    ));
    let names: Vec<_> = (0..2)
        .map(|_| {
            match runtime
                .call_native(type_of, &[Value::Nil])
                .unwrap()
                .as_slice()
            {
                [Value::String(name)] => name.clone(),
                other => panic!("{other:?}"),
            }
        })
        .collect();
    assert!(names[0].ptr_eq(&names[1]));
    for (value, expected) in [
        (Value::Integer(3), "number"),
        (Value::Int64(-1), "cdata"),
        (Value::UInt64(1), "cdata"),
    ] {
        assert_eq!(
            runtime.call_native(type_of, &[value]).unwrap(),
            vec![Value::from(expected)]
        );
    }
}

#[test]
fn type_hands_out_one_shared_string_per_name() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(type_of) = runtime.get_global("type") else {
        panic!("type is native");
    };
    let name = |runtime: &mut Runtime, value: Value| match runtime
        .call_native(type_of, &[value])
        .unwrap()
        .as_slice()
    {
        [Value::String(name)] => name.clone(),
        _ => panic!("type returns one string"),
    };
    let first = name(&mut runtime, Value::from(1.0));
    let second = name(&mut runtime, Value::from(2.0));
    assert_eq!(first.as_bytes(), b"number");
    assert_eq!(first.identity(), second.identity());
    let cdata = name(&mut runtime, Value::Int64(3));
    assert_eq!(cdata.as_bytes(), b"cdata");
}

#[test]
fn base_installation_exposes_iteration_and_loading_builtins() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    for (name, builtin) in [
        ("next", BuiltinFunction::Next),
        ("pairs", BuiltinFunction::Pairs),
        ("ipairs", BuiltinFunction::IPairs),
        ("load", BuiltinFunction::Load),
        ("loadstring", BuiltinFunction::Load),
        ("dofile", BuiltinFunction::DoFile),
    ] {
        let value = runtime.get_global(name);
        assert_eq!(value, Value::BuiltinFunction(builtin));
        assert_eq!(value.type_name(), "function");
        assert!(Runtime::raw_equal(&value, &Value::BuiltinFunction(builtin)));
    }
}

#[test]
fn pure_installation_keeps_in_memory_libraries_without_host_access() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let mut runtime = Runtime::new();
        crate::stdlib::install_pure_with_profile(&mut runtime, profile).unwrap();
        for name in [
            "assert",
            "type",
            "load",
            "math",
            "string",
            "table",
            "coroutine",
        ] {
            assert_ne!(
                runtime.get_global(name),
                Value::Nil,
                "{name} in {profile:?}"
            );
        }
        for name in [
            "io", "os", "package", "debug", "print", "warn", "loadfile", "dofile", "require",
            "module",
        ] {
            assert_eq!(
                runtime.get_global(name),
                Value::Nil,
                "{name} in {profile:?}"
            );
        }
        assert_eq!(
            runtime.get_global("load"),
            Value::BuiltinFunction(BuiltinFunction::Load)
        );
    }
}

#[test]
fn full_installation_can_keep_custom_io() {
    let mut runtime = Runtime::new();
    let io = runtime.create_table().unwrap();
    runtime.set_global("io", Value::Table(io));
    crate::stdlib::install_base_with_profile_without_io(&mut runtime, CompatibilityProfile::Lua51)
        .unwrap();
    assert_eq!(runtime.get_global("io"), Value::Table(io));
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap() else {
        panic!("package.loaded is a table");
    };
    assert_eq!(
        runtime.table_get(loaded, &Value::from("io")).unwrap(),
        Value::Table(io)
    );
    assert_ne!(runtime.get_global("os"), Value::Nil);
    assert_ne!(runtime.get_global("require"), Value::Nil);
}

#[test]
fn full_installation_can_keep_custom_io_and_os_independently() {
    for (default_io, default_os) in [(false, false), (false, true), (true, false)] {
        let mut runtime = Runtime::new();
        let custom_io = runtime.create_table().unwrap();
        let custom_os = runtime.create_table().unwrap();
        runtime.set_global("io", Value::Table(custom_io));
        runtime.set_global("os", Value::Table(custom_os));
        crate::stdlib::install_full_with_profile_and_host_libraries(
            &mut runtime,
            CompatibilityProfile::Lua51,
            default_io,
            default_os,
        )
        .unwrap();

        let io = runtime.get_global("io");
        let os = runtime.get_global("os");
        assert_eq!(io == Value::Table(custom_io), !default_io);
        assert_eq!(os == Value::Table(custom_os), !default_os);
        let Value::Table(package) = runtime.get_global("package") else {
            panic!("package is a table");
        };
        let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap()
        else {
            panic!("package.loaded is a table");
        };
        assert_eq!(runtime.table_get(loaded, &Value::from("io")).unwrap(), io);
        assert_eq!(runtime.table_get(loaded, &Value::from("os")).unwrap(), os);
        assert_ne!(runtime.get_global("print"), Value::Nil);
    }
}

#[test]
fn pure_loading_installs_package_without_host_libraries() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let mut runtime = Runtime::new();
        crate::stdlib::install_pure_with_profile_and_loading(&mut runtime, profile).unwrap();
        for name in ["loadfile", "dofile", "require", "package"] {
            assert_ne!(
                runtime.get_global(name),
                Value::Nil,
                "{name} in {profile:?}"
            );
        }
        for name in ["io", "os", "print", "debug", "warn"] {
            assert_eq!(
                runtime.get_global(name),
                Value::Nil,
                "{name} in {profile:?}"
            );
        }
        let Value::Table(package) = runtime.get_global("package") else {
            panic!("package is a table");
        };
        let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap()
        else {
            panic!("package.loaded is a table");
        };
        assert_eq!(
            runtime.table_get(loaded, &Value::from("io")).unwrap(),
            Value::Nil
        );
        assert_eq!(
            runtime.table_get(loaded, &Value::from("os")).unwrap(),
            Value::Nil
        );
    }
}

#[test]
fn tonumber_parses_trimmed_decimal_hexadecimal_and_bases() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(tonumber) = runtime.get_global("tonumber") else {
        panic!("tonumber is native");
    };
    assert_eq!(
        runtime.call_native(tonumber, &[]).unwrap_err(),
        RuntimeError::argument_error(1, "value expected")
    );
    assert_eq!(
        runtime.call_native(tonumber, &[Value::Nil]).unwrap(),
        vec![Value::Nil]
    );
    for (source, expected) in [("  -12.5 ", -12.5), ("0x10", 16.0), ("-0x1.8p1", -3.0)] {
        assert_eq!(
            runtime
                .call_native(tonumber, &[Value::from(source)])
                .unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert_eq!(
        runtime
            .call_native(tonumber, &[Value::from("-ff"), Value::from(16.0)])
            .unwrap(),
        vec![Value::from(-255.0)]
    );
    assert_eq!(
        runtime
            .call_native(tonumber, &[Value::from("+101"), Value::from(2.0)])
            .unwrap(),
        vec![Value::from(5.0)]
    );
    assert_eq!(
        runtime
            .call_native(tonumber, &[Value::from("bad")])
            .unwrap(),
        vec![Value::Nil]
    );
    assert_eq!(
        runtime.call_native(tonumber, &[Value::from(true)]).unwrap(),
        vec![Value::Nil]
    );
    assert_eq!(
        runtime.call_native(tonumber, &[Value::from(12.0)]).unwrap(),
        vec![Value::from(12.0)]
    );
    assert!(
        runtime
            .call_native(tonumber, &[Value::from("10"), Value::from(1.0)])
            .is_err()
    );
    assert_eq!(
        runtime
            .call_native(tonumber, &[Value::from(12.0), Value::from(10.0)])
            .unwrap(),
        vec![Value::from(12.0)]
    );
}

#[test]
fn tonumber_accepts_numbers_with_a_base_only_in_lua51_profiles() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::NativeFunction(tonumber) = runtime.get_global("tonumber") else {
            panic!("tonumber is native");
        };
        let mut call = |value: Value, base: f64| {
            runtime
                .call_native(tonumber, &[value, Value::from(base)])
                .unwrap()
        };
        // Base 10 converts as without a base, so any value is accepted.
        assert_eq!(call(Value::from(5.0), 10.0), vec![Value::from(5.0)]);
        assert_eq!(call(Value::from(5.5), 10.0), vec![Value::from(5.5)]);
        assert_eq!(call(Value::from("0x10"), 10.0), vec![Value::from(16.0)]);
        assert_eq!(call(Value::from("1e2"), 10.0), vec![Value::from(100.0)]);
        assert_eq!(call(Value::from(true), 10.0), vec![Value::Nil]);
        assert_eq!(call(Value::Nil, 10.0), vec![Value::Nil]);
        // Other bases read a number through its string spelling.
        assert_eq!(call(Value::from(255.0), 16.0), vec![Value::from(597.0)]);
        assert_eq!(call(Value::from(5.5), 16.0), vec![Value::Nil]);
        assert!(
            runtime
                .call_native(tonumber, &[Value::from(true), Value::from(16.0)])
                .is_err()
        );
    }
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::NativeFunction(tonumber) = runtime.get_global("tonumber") else {
            panic!("tonumber is native");
        };
        for base in [10.0, 16.0] {
            assert_eq!(
                runtime
                    .call_native(tonumber, &[Value::from(5.0), Value::from(base)])
                    .unwrap_err(),
                RuntimeError::BadArgument {
                    position: 1,
                    expected: "string",
                    found: "number",
                }
            );
        }
        assert_eq!(
            runtime
                .call_native(tonumber, &[Value::from("0x10"), Value::from(10.0)])
                .unwrap(),
            vec![Value::Nil]
        );
    }
}

#[test]
fn tonumber_keeps_lua52_hexadecimal_numerals_as_doubles() {
    let source = format!("0x{}", "f".repeat(150));

    let mut lua52 = Runtime::new();
    install_base_with_profile(&mut lua52, CompatibilityProfile::Lua52).unwrap();
    let Value::NativeFunction(tonumber) = lua52.get_global("tonumber") else {
        panic!("tonumber is native");
    };
    assert_eq!(
        lua52
            .call_native(tonumber, &[Value::from(source.as_str())])
            .unwrap(),
        vec![Value::Number(2_f64.powi(600) - 1.0)]
    );

    let mut lua53 = Runtime::new();
    install_base_with_profile(&mut lua53, CompatibilityProfile::Lua53).unwrap();
    let Value::NativeFunction(tonumber) = lua53.get_global("tonumber") else {
        panic!("tonumber is native");
    };
    assert_eq!(
        lua53
            .call_native(tonumber, &[Value::from("0xffffffffffffffff")])
            .unwrap(),
        vec![Value::Integer(-1)]
    );
    let scaled = format!("0xe03{}p-4000", "0".repeat(1_000));
    assert_eq!(
        lua53
            .call_native(tonumber, &[Value::from(scaled.as_str())])
            .unwrap(),
        vec![Value::Number(3587.0)]
    );
}

#[test]
fn numeric_error_objects_follow_the_profile_and_error_level() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::NativeFunction(function) = runtime.get_global("error") else {
            panic!("error native")
        };
        for level in [0, 1] {
            let converted = level > 0
                && matches!(
                    profile,
                    CompatibilityProfile::Lua51
                        | CompatibilityProfile::Lua52
                        | CompatibilityProfile::LuaJit21
                );
            let value = if converted {
                Value::from("42")
            } else {
                Value::Number(42.0)
            };
            assert_eq!(
                runtime.call_native(
                    function,
                    &[Value::Number(42.0), Value::Number(level as f64)]
                ),
                Err(RuntimeError::RaisedAtLevel { value, level }),
                "{profile:?} level {level}",
            );
        }
    }
}

#[test]
fn lua52_assert_without_an_argument_fails_with_the_default_message() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua52).unwrap();
    let Value::NativeFunction(function) = runtime.get_global("assert") else {
        panic!("assert is native");
    };
    assert_eq!(
        runtime.call_native(function, &[]),
        Err(RuntimeError::message("assertion failed!"))
    );
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::NativeFunction(function) = runtime.get_global("assert") else {
            panic!("assert is native");
        };
        assert_eq!(
            runtime.call_native(function, &[]),
            Err(RuntimeError::argument_error(1, "value expected")),
            "{profile:?}"
        );
    }
}

/// The integrality test the table and string natives use agrees with
/// `fract` on every finite value, around the 2^52 and 2^63 edges too.
#[test]
fn finite_is_integral_matches_fract() {
    let two_52 = 4_503_599_627_370_496.0_f64;
    for value in [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.5,
        1e-300,
        two_52 - 0.5,
        -(two_52 - 0.5),
        two_52,
        two_52 + 1.0,
        9.3e18,
        -9.3e18,
        1e300,
        f64::MAX,
        f64::MIN,
    ] {
        assert_eq!(
            crate::stdlib::args::finite_is_integral(value),
            value.fract() == 0.0,
            "{value}"
        );
    }
}
