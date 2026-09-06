use super::*;

#[test]
fn lua_warnings_reach_host_callback_as_validated_fragments() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut runtime = Runtime::new();
    let recorded = Rc::new(RefCell::new(Vec::new()));
    let capture = Rc::clone(&recorded);
    runtime.set_warning_handler(Some(Box::new(move |bytes, continued| {
        capture.borrow_mut().push((bytes.to_vec(), continued));
        Ok(())
    })));
    crate::stdlib::base::warn(&mut runtime, &[Value::from("first"), Value::from("second")])
        .unwrap();
    crate::stdlib::base::warn(&mut runtime, &[Value::from("@off")]).unwrap();
    assert_eq!(
        &*recorded.borrow(),
        &[
            (b"first".to_vec(), true),
            (b"second".to_vec(), false),
            (b"@off".to_vec(), false),
        ]
    );
    assert!(
        crate::stdlib::base::warn(
            &mut runtime,
            &[Value::from("partial"), Value::Boolean(true)]
        )
        .is_err()
    );
    assert_eq!(
        recorded.borrow().len(),
        3,
        "invalid later arguments produce no partial warning"
    );
}

#[test]
fn bit_can_be_installed_after_any_lua_profile() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        assert_eq!(runtime.get_global("bit"), Value::Nil);

        crate::stdlib::install_bit(&mut runtime).unwrap();
        let bit = runtime.get_global("bit");
        let band = bit_function(&runtime, "band");
        assert_eq!(
            runtime
                .call_native(band, &[Value::from(12.0), Value::from(10.0)])
                .unwrap(),
            vec![Value::from(8.0)]
        );

        let Value::Table(package) = runtime.get_global("package") else {
            panic!("package is a table");
        };
        let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap()
        else {
            panic!("package.loaded is a table");
        };
        assert_eq!(runtime.table_get(loaded, &Value::from("bit")).unwrap(), bit);
    }
}

#[test]
fn bit_can_be_installed_in_pure_mode_before_package() {
    let mut runtime = Runtime::new();
    crate::stdlib::install_pure_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    crate::stdlib::install_bit(&mut runtime).unwrap();
    assert!(matches!(runtime.get_global("bit"), Value::Table(_)));
    assert_eq!(runtime.get_global("package"), Value::Nil);

    crate::stdlib::install_package(&mut runtime).unwrap();
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap() else {
        panic!("package.loaded is a table");
    };
    assert_eq!(
        runtime.table_get(loaded, &Value::from("bit")).unwrap(),
        runtime.get_global("bit")
    );
}

#[test]
fn bit_accepts_modern_lua_integer_values() {
    let mut runtime = Runtime::new();
    crate::stdlib::install_pure_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    crate::stdlib::install_bit(&mut runtime).unwrap();

    let bxor = bit_function(&runtime, "bxor");
    assert_eq!(
        runtime
            .call_native(bxor, &[Value::Integer(6), Value::Integer(3)])
            .unwrap(),
        vec![Value::from(5.0)]
    );

    let tobit = bit_function(&runtime, "tobit");
    for (integer, expected) in [
        (-1, -1.0),
        (4_294_967_297, 1.0),
        (i64::MAX, -1.0),
        (i64::MIN, 0.0),
    ] {
        assert_eq!(
            runtime
                .call_native(tobit, &[Value::Integer(integer)])
                .unwrap(),
            vec![Value::from(expected)]
        );
    }
}

#[test]
fn io_stderr_is_a_distinct_opaque_error_stream() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let stderr = runtime.table_get(io, &Value::from("stderr")).unwrap();
    let Value::File(stderr) = stderr else {
        panic!("io.stderr is a file");
    };
    runtime
        .file_write(stderr, &[Value::bytes(b"warn\xff")])
        .unwrap();
    assert_eq!(runtime.take_output(), [] as [u8; 0]);
    assert_eq!(runtime.take_error_output(), b"warn\xff");
}

#[test]
fn math_random_is_runtime_local_and_validates_lua_intervals() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua51).unwrap();
    let random = math_function(&runtime, "random");
    let randomseed = math_function(&runtime, "randomseed");

    runtime
        .call_native(randomseed, &[Value::Number(1234.0)])
        .unwrap();
    let first = runtime.call_native(random, &[]).unwrap();
    let bounded = runtime.call_native(random, &[Value::Number(5.0)]).unwrap();
    let ranged = runtime
        .call_native(random, &[Value::Number(-3.0), Value::Number(4.0)])
        .unwrap();
    runtime
        .call_native(randomseed, &[Value::Number(1234.0)])
        .unwrap();
    assert_eq!(runtime.call_native(random, &[]).unwrap(), first);
    assert!(matches!(first.as_slice(), [Value::Number(value)] if *value >= 0.0 && *value < 1.0));
    assert!(matches!(bounded.as_slice(), [Value::Number(value)] if *value >= 1.0 && *value <= 5.0));
    assert!(matches!(ranged.as_slice(), [Value::Number(value)] if *value >= -3.0 && *value <= 4.0));
    assert!(runtime.call_native(random, &[Value::Number(0.0)]).is_err());
    assert!(
        runtime
            .call_native(random, &[Value::Number(4.0), Value::Number(3.0)])
            .is_err()
    );
    assert!(runtime.call_native(random, &[Value::Number(1.5)]).is_err());
}

#[test]
fn lua54_random_matches_seeded_reference_sequence() {
    let mut runtime = Runtime::new();
    runtime.set_random_modern(true);
    let seeds = crate::stdlib::math::math_randomseed_modern(
        &mut runtime,
        &[Value::Integer(1234), Value::Integer(5678)],
    )
    .unwrap();
    assert_eq!(seeds, vec![Value::Integer(1234), Value::Integer(5678)]);
    assert_eq!(
        crate::stdlib::math::math_random_lua54(&mut runtime, &[Value::Integer(0)]).unwrap(),
        vec![Value::Integer(5_173_903_662_814_325_993)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_lua54(&mut runtime, &[]).unwrap(),
        vec![Value::Number(0.991_611_477_994_134_5)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_lua54(&mut runtime, &[Value::Integer(10)]).unwrap(),
        vec![Value::Integer(1)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_lua54(
            &mut runtime,
            &[Value::Integer(-10), Value::Integer(10)]
        )
        .unwrap(),
        vec![Value::Integer(2)]
    );

    assert_eq!(
        crate::stdlib::math::math_randomseed_modern(&mut runtime, &[Value::Integer(-1)]).unwrap(),
        vec![Value::Integer(-1), Value::Integer(0)]
    );
    assert_eq!(
        crate::stdlib::math::math_randomseed_modern(&mut runtime, &[])
            .unwrap()
            .len(),
        2
    );
    assert!(crate::stdlib::math::math_randomseed_modern(&mut runtime, &[Value::Nil]).is_err());
}

#[test]
fn modern_atan_accepts_an_optional_second_coordinate() {
    let mut runtime = Runtime::new();
    assert_eq!(
        crate::stdlib::math::math_atan_modern(
            &mut runtime,
            &[Value::Integer(1), Value::Integer(0)]
        )
        .unwrap(),
        Value::Number(std::f64::consts::FRAC_PI_2)
    );
    assert_eq!(
        crate::stdlib::math::math_atan_modern(&mut runtime, &[Value::Integer(1), Value::Nil])
            .unwrap(),
        Value::Number(std::f64::consts::FRAC_PI_4)
    );
    assert!(
        crate::stdlib::math::math_atan_modern(
            &mut runtime,
            &[Value::Integer(1), Value::from("bad")]
        )
        .is_err()
    );
}

#[test]
fn modern_math_registration_uses_profile_specific_atan_and_randomseed() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let atan = math_function(&runtime, "atan");
        let randomseed = math_function(&runtime, "randomseed");
        let random = math_function(&runtime, "random");
        assert_eq!(
            runtime
                .call_native(atan, &[Value::Integer(1), Value::Integer(0)])
                .unwrap(),
            vec![Value::Number(std::f64::consts::FRAC_PI_2)]
        );
        assert_eq!(
            runtime
                .call_native(randomseed, &[Value::Integer(1234), Value::Integer(5678)])
                .unwrap(),
            vec![Value::Integer(1234), Value::Integer(5678)]
        );
        assert_eq!(
            runtime.call_native(random, &[Value::Integer(0)]).unwrap(),
            vec![Value::Integer(5_173_903_662_814_325_993)]
        );
    }
}

#[test]
fn luajit_random_matches_seeded_reference_sequence_and_bounds() {
    let mut runtime = Runtime::new();
    runtime.set_random_luajit(true);
    crate::stdlib::math::math_randomseed_luajit(&mut runtime, &[Value::Number(1.0)]).unwrap();
    assert_eq!(
        crate::stdlib::math::math_random_luajit(&mut runtime, &[]).unwrap(),
        vec![Value::Number(0.323_810_562_378_636_7)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_luajit(&mut runtime, &[Value::Number(10.0)]).unwrap(),
        vec![Value::Number(1.0)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_luajit(&mut runtime, &[Value::Number(0.0)]).unwrap(),
        vec![Value::Number(1.0)]
    );
    assert_eq!(
        crate::stdlib::math::math_random_luajit(
            &mut runtime,
            &[Value::Number(-3.0), Value::Number(4.0)]
        )
        .unwrap(),
        vec![Value::Number(-1.0)]
    );
    assert!(
        crate::stdlib::math::math_random_luajit(&mut runtime, &[Value::from("invalid")]).is_err()
    );
}

#[test]
fn base_installation_exposes_math_floor() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(math) = runtime.get_global("math") else {
        panic!("math global is a table");
    };
    let Value::NativeFunction(floor) = runtime.table_get(math, &Value::from("floor")).unwrap()
    else {
        panic!("math.floor is native");
    };
    assert_eq!(
        runtime.call_native(floor, &[Value::Number(-1.2)]).unwrap(),
        vec![Value::Number(-2.0)]
    );
}

#[test]
fn math_library_registers_the_stateless_lua_51_surface() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(math) = runtime.get_global("math") else {
        panic!("math is a table");
    };

    for name in [
        "abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "deg", "exp", "floor",
        "fmod", "frexp", "ldexp", "log", "log10", "max", "min", "mod", "modf", "pow", "rad", "sin",
        "sinh", "sqrt", "tan", "tanh",
    ] {
        assert!(matches!(
            runtime.table_get(math, &Value::from(name)).unwrap(),
            Value::NativeFunction(_)
        ));
    }
    assert_eq!(
        runtime.table_get(math, &Value::from("pi")).unwrap(),
        Value::Number(std::f64::consts::PI)
    );
    let Value::Number(huge) = runtime.table_get(math, &Value::from("huge")).unwrap() else {
        panic!("math.huge is a number");
    };
    assert!(huge.is_infinite() && huge.is_sign_positive());
}

#[test]
fn math_unary_numeric_natives_publish_fast_paths_without_changing_lua_dispatch() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();

    for name in [
        "acos", "asin", "cos", "cosh", "deg", "exp", "log", "log10", "rad", "sin", "sinh", "sqrt",
        "tan", "tanh",
    ] {
        assert!(
            runtime
                .native_numeric_unary(math_function(&runtime, name))
                .is_some()
        );
    }
    // `abs`, `ceil` and `floor` are recorded as operations of their own.
    for name in ["abs", "atan", "atan2", "ceil", "floor", "pow"] {
        assert!(
            runtime
                .native_numeric_unary(math_function(&runtime, name))
                .is_none()
        );
    }
    assert!(
        runtime
            .native_numeric_binary(math_function(&runtime, "fmod"))
            .is_some()
    );
    assert!(
        runtime
            .native_numeric_binary(math_function(&runtime, "pow"))
            .is_some()
    );
    // A profile with integers floors a float to an integer, which is no
    // float function: no profile has one for `floor` or `abs`.
    let mut modern = Runtime::new();
    install_base_with_profile(&mut modern, CompatibilityProfile::Lua54).unwrap();
    for name in ["floor", "abs"] {
        assert!(
            modern
                .native_numeric_unary(math_function(&modern, name))
                .is_none()
        );
    }

    // `log` with a base is still `log` when the profile takes one.
    assert!(
        modern
            .native_numeric_unary(math_function(&modern, "log"))
            .is_some()
    );

    let sin = math_function(&runtime, "sin");
    let numeric_sin = runtime.native_numeric_unary(sin).unwrap();
    assert!((numeric_sin(std::f64::consts::PI / 2.0) - 1.0).abs() < f64::EPSILON);
    assert_eq!(
        runtime
            .call_native(sin, &[Value::from("0"), Value::from("ignored")])
            .unwrap(),
        vec![Value::Number(0.0)]
    );
    assert!(
        runtime
            .call_native(sin, &[Value::from("not a number")])
            .is_err()
    );
}

#[test]
fn modern_math_log_accepts_a_base_without_changing_lua51() {
    let mut modern = Runtime::new();
    install_base_with_profile(&mut modern, CompatibilityProfile::Lua55).unwrap();
    let log = math_function(&modern, "log");
    assert_eq!(
        modern
            .call_native(log, &[Value::from(9.0), Value::from(3.0)])
            .unwrap(),
        vec![Value::from(2.0)]
    );

    let mut legacy = Runtime::new();
    install_base_with_profile(&mut legacy, CompatibilityProfile::Lua51).unwrap();
    let log = math_function(&legacy, "log");
    assert_eq!(
        legacy
            .call_native(log, &[Value::from(9.0), Value::from(3.0)])
            .unwrap(),
        vec![Value::from(9.0_f64.ln())]
    );
}

#[test]
fn math_remainders_decomposition_and_scaling_handle_edges() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let fmod = math_function(&runtime, "fmod");
    let modulo = math_function(&runtime, "mod");
    let frexp = math_function(&runtime, "frexp");
    let ldexp = math_function(&runtime, "ldexp");
    let modf = math_function(&runtime, "modf");

    assert_eq!(
        runtime
            .call_native(fmod, &[Value::from(-5.0), Value::from(3.0)])
            .unwrap(),
        vec![Value::from(-2.0)]
    );
    assert_eq!(
        runtime
            .call_native(modulo, &[Value::from(-5.0), Value::from(3.0)])
            .unwrap(),
        vec![Value::from(1.0)]
    );
    assert_eq!(
        runtime.call_native(modf, &[Value::from(-3.5)]).unwrap(),
        vec![Value::from(-3.0), Value::from(-0.5)]
    );
    assert_eq!(
        runtime
            .call_native(modf, &[Value::Number(f64::INFINITY)])
            .unwrap(),
        vec![Value::Number(f64::INFINITY), Value::from(0.0)]
    );
    assert_eq!(
        runtime
            .call_native(frexp, &[Value::from(std::f64::consts::PI)])
            .unwrap(),
        vec![Value::from(std::f64::consts::PI / 4.0), Value::from(2.0)]
    );
    assert_eq!(
        runtime
            .call_native(frexp, &[Value::from(f64::from_bits(1))])
            .unwrap(),
        vec![Value::from(0.5), Value::from(-1073.0)]
    );
    assert_eq!(
        runtime
            .call_native(ldexp, &[Value::from(0.5), Value::from(2.0)])
            .unwrap(),
        vec![Value::from(2.0)]
    );
    assert_eq!(
        runtime
            .call_native(ldexp, &[Value::from(1.0), Value::from(-1074.0)])
            .unwrap(),
        vec![Value::from(f64::from_bits(1))]
    );
    assert_eq!(
        runtime
            .call_native(
                ldexp,
                &[Value::from(f64::from_bits(1)), Value::from(1074.0)]
            )
            .unwrap(),
        vec![Value::from(1.0)]
    );
    assert_eq!(
        runtime
            .call_native(ldexp, &[Value::Number(f64::INFINITY), Value::from(-2.0)])
            .unwrap(),
        vec![Value::Number(f64::INFINITY)]
    );
    assert!(
        runtime
            .call_native(ldexp, &[Value::from(1.0), Value::from(1.5)])
            .is_err()
    );
}

#[test]
fn math_extrema_and_binary_functions_follow_lua_number_rules() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let atan2 = math_function(&runtime, "atan2");
    let max = math_function(&runtime, "max");
    let min = math_function(&runtime, "min");
    let pow = math_function(&runtime, "pow");

    assert_eq!(
        runtime
            .call_native(max, &[Value::from("2"), Value::from(5.0), Value::from("3")],)
            .unwrap(),
        vec![Value::from(5.0)]
    );
    assert_eq!(
        runtime
            .call_native(min, &[Value::from(-2.0), Value::from("-5")])
            .unwrap(),
        vec![Value::from(-5.0)]
    );
    // The default profile is LuaJIT, whose math.max replaces the result
    // unless the new value compares less: a leading NaN is replaced and a
    // trailing NaN wins (stock luajit: max(nan, 1) == 1, max(1, nan) is nan).
    assert_eq!(
        runtime
            .call_native(max, &[Value::Number(f64::NAN), Value::from(1.0)])
            .unwrap(),
        vec![Value::from(1.0)]
    );
    let max_nan = runtime
        .call_native(max, &[Value::from(1.0), Value::Number(f64::NAN)])
        .unwrap();
    let [Value::Number(max_nan)] = max_nan.as_slice() else {
        panic!("math.max returns a number");
    };
    assert!(max_nan.is_nan());
    assert_eq!(
        runtime
            .call_native(atan2, &[Value::from(1.0), Value::from(0.0)])
            .unwrap(),
        vec![Value::from(std::f64::consts::FRAC_PI_2)]
    );
    assert_eq!(
        runtime
            .call_native(pow, &[Value::from("2"), Value::from(3.0)])
            .unwrap(),
        vec![Value::from(8.0)]
    );
    assert!(runtime.call_native(min, &[]).is_err());
}

#[test]
#[allow(clippy::too_many_lines)]
fn bit_library_matches_luajit_number_mode() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let tobit = bit_function(&runtime, "tobit");
    for (input, expected) in [
        (0.0, 0.0),
        (2_147_483_647.0, 2_147_483_647.0),
        (2_147_483_648.0, -2_147_483_648.0),
        (4_294_967_295.0, -1.0),
        (4_294_967_296.0, 0.0),
        (-1.0, -1.0),
        (-4_294_967_298.0, -2.0),
        (3.9, 4.0),
        (-3.9, -4.0),
        (2.5, 2.0),
        (1.5, 2.0),
    ] {
        assert_eq!(
            runtime.call_native(tobit, &[Value::from(input)]).unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert_eq!(
        runtime
            .call_native(tobit, &[Value::from("0x80000000")])
            .unwrap(),
        vec![Value::from(-2_147_483_648.0)]
    );
    assert_eq!(
        runtime
            .call_native(tobit, &[Value::Number(f64::NAN)])
            .unwrap(),
        vec![Value::from(0.0)]
    );

    let bnot = bit_function(&runtime, "bnot");
    let bswap = bit_function(&runtime, "bswap");
    assert_eq!(
        runtime.call_native(bnot, &[Value::from(0.0)]).unwrap(),
        vec![Value::from(-1.0)]
    );
    assert_eq!(
        runtime
            .call_native(bswap, &[Value::from(305_419_896.0)])
            .unwrap(),
        vec![Value::from(2_018_915_346.0)]
    );

    for (name, expected) in [("band", 0.0), ("bor", 15.0), ("bxor", 15.0)] {
        let function = bit_function(&runtime, name);
        assert_eq!(
            runtime
                .call_native(
                    function,
                    &[
                        Value::from(1.0),
                        Value::from(2.0),
                        Value::from(4.0),
                        Value::from(8.0)
                    ],
                )
                .unwrap(),
            vec![Value::from(expected)]
        );
        assert_eq!(
            runtime.call_native(function, &[Value::from(1.0)]).unwrap(),
            vec![Value::from(1.0)]
        );
        assert!(runtime.call_native(function, &[]).is_err());
        assert!(runtime.call_native(function, &[Value::from(true)]).is_err());
    }

    for (name, expected) in [
        ("lshift", 591_751_040.0),
        ("rshift", 19_088_743.0),
        ("arshift", 19_088_743.0),
        ("rol", 591_751_041.0),
        ("ror", -2_128_394_905.0),
    ] {
        let function = bit_function(&runtime, name);
        assert_eq!(
            runtime
                .call_native(function, &[Value::from(305_419_896.0), Value::from(4.0)])
                .unwrap(),
            vec![Value::from(expected)]
        );
        assert!(runtime.call_native(function, &[Value::from(1.0)]).is_err());
    }

    let tohex = bit_function(&runtime, "tohex");
    for (width, expected) in [
        (None, "1234abcd"),
        (Some(4.0), "abcd"),
        (Some(-4.0), "ABCD"),
        (Some(12.0), "00001234abcd"),
        (Some(0.0), ""),
    ] {
        let mut arguments = vec![Value::from(305_441_741.0)];
        if let Some(width) = width {
            arguments.push(Value::from(width));
        }
        assert_eq!(
            runtime.call_native(tohex, &arguments).unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert!(runtime.call_native(tohex, &[]).is_err());
    assert!(
        runtime
            .call_native(tohex, &[Value::from(1.0), Value::Nil])
            .is_err()
    );
    assert!(runtime.call_native(tobit, &[Value::from(true)]).is_err());
}

#[test]
fn math_sin_coerces_its_first_argument_and_ignores_extras() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(math) = runtime.get_global("math") else {
        panic!("math is a table");
    };
    let Value::NativeFunction(sin) = runtime.table_get(math, &Value::from("sin")).unwrap() else {
        panic!("math.sin is native");
    };

    let result = runtime
        .call_native(
            sin,
            &[Value::from(" 1.5707963267948966 "), Value::from("ignored")],
        )
        .unwrap();
    let [Value::Number(result)] = result.as_slice() else {
        panic!("math.sin returns one number");
    };
    assert!((*result - 1.0).abs() < f64::EPSILON);
    assert!(
        runtime
            .call_native(sin, &[Value::from("not a number")])
            .is_err()
    );
    assert!(runtime.call_native(sin, &[]).is_err());
}

#[test]
fn math_mod_uses_floor_based_remainder_and_numeric_coercion() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(math) = runtime.get_global("math") else {
        panic!("math is a table");
    };
    let Value::NativeFunction(math_mod) = runtime.table_get(math, &Value::from("mod")).unwrap()
    else {
        panic!("math.mod is native");
    };

    for (arguments, expected) in [
        (vec![Value::from(5.0), Value::from(2.0)], 1.0),
        (vec![Value::from(-3.0), Value::from(2.0)], 1.0),
        (vec![Value::from(3.0), Value::from(-2.0)], -1.0),
        (vec![Value::from(-3.0), Value::from(-2.0)], -1.0),
        (vec![Value::from(" 5.5 "), Value::from("2")], 1.5),
    ] {
        assert_eq!(
            runtime.call_native(math_mod, &arguments).unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert_eq!(
        runtime
            .call_native(
                math_mod,
                &[Value::from(5.0), Value::from(2.0), Value::from("ignored")],
            )
            .unwrap(),
        vec![Value::from(1.0)]
    );
    assert!(
        runtime
            .call_native(math_mod, &[Value::from("not a number"), Value::from(2.0)])
            .is_err()
    );
    assert!(runtime.call_native(math_mod, &[Value::from(2.0)]).is_err());
}

#[test]
fn base_installation_exposes_error_and_select() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(error) = runtime.get_global("error") else {
        panic!("error is native");
    };
    let Value::NativeFunction(select) = runtime.get_global("select") else {
        panic!("select is native");
    };

    assert_eq!(
        runtime.call_native(error, &[Value::from(&b"bad\xff"[..])]),
        Err(RuntimeError::RaisedAtLevel {
            value: Value::from(&b"bad\xff"[..]),
            level: 1,
        })
    );
    assert_eq!(
        runtime.call_native(error, &[Value::from(false), Value::from(2.0)]),
        Err(RuntimeError::RaisedAtLevel {
            value: Value::from(false),
            level: 2,
        })
    );
    assert_eq!(
        runtime.call_native(error, &[]),
        Err(RuntimeError::RaisedNil)
    );
    assert_eq!(
        runtime.call_native(error, &[Value::Nil]),
        Err(RuntimeError::RaisedNil)
    );
    let table = Value::Table(runtime.create_table().unwrap());
    assert_eq!(
        runtime.call_native(error, std::slice::from_ref(&table)),
        Err(RuntimeError::RaisedAtLevel {
            value: table,
            level: 1,
        })
    );

    assert_eq!(
        runtime
            .call_native(
                select,
                &[Value::from("#"), Value::Nil, Value::from(7.0), Value::Nil],
            )
            .unwrap(),
        vec![Value::from(3.0)]
    );
    assert_eq!(
        runtime
            .call_native(
                select,
                &[
                    Value::from(1.0),
                    Value::Nil,
                    Value::from("middle"),
                    Value::Nil
                ],
            )
            .unwrap(),
        vec![Value::Nil, Value::from("middle"), Value::Nil]
    );
    assert_eq!(
        runtime
            .call_native(
                select,
                &[
                    Value::from(-2.0),
                    Value::Nil,
                    Value::from("middle"),
                    Value::Nil
                ],
            )
            .unwrap(),
        vec![Value::from("middle"), Value::Nil]
    );
    assert_eq!(
        runtime
            .call_native(select, &[Value::from(2.0), Value::from("only")])
            .unwrap(),
        Vec::<Value>::new()
    );
    assert_eq!(
        runtime.call_native(select, &[Value::from(1.0)]).unwrap(),
        Vec::<Value>::new()
    );
}

#[test]
fn table_remove_defaults_an_explicit_nil_position_to_the_last_element() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::from("a"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(2.0), Value::from("b"))
        .unwrap();
    assert_eq!(
        crate::stdlib::base::table_remove(&mut runtime, &[Value::Table(table), Value::Nil])
            .unwrap(),
        vec![Value::from("b")]
    );
    assert_eq!(runtime.table_length(table).unwrap(), 1.0);
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)).unwrap(),
        Value::from("a")
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(2.0)).unwrap(),
        Value::Nil
    );
}

#[test]
fn table_insert_and_remove_shift_raw_list_slots_and_observe_holes() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(library) = runtime.get_global("table") else {
        panic!("table is a table");
    };
    let native = |name| {
        let Value::NativeFunction(function) =
            runtime.table_get(library, &Value::from(name)).unwrap()
        else {
            panic!("table.{name} is native");
        };
        function
    };
    let insert = native("insert");
    let remove = native("remove");
    let table = runtime.create_table().unwrap();
    for (index, value) in [(1.0, "a"), (2.0, "c"), (3.0, "d")] {
        runtime
            .table_set(table, &Value::Number(index), Value::from(value))
            .unwrap();
    }

    assert_eq!(
        runtime
            .call_native(
                insert,
                &[Value::Table(table), Value::Number(2.0), Value::from("b")]
            )
            .unwrap(),
        Vec::<Value>::new()
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)).unwrap(),
        Value::from("a")
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(2.0)).unwrap(),
        Value::from("b")
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(4.0)).unwrap(),
        Value::from("d")
    );
    assert_eq!(
        runtime
            .call_native(remove, &[Value::Table(table), Value::Number(2.0)])
            .unwrap(),
        vec![Value::from("b")]
    );
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        3.0_f64.to_bits()
    );
    assert_eq!(
        runtime
            .call_native(remove, &[Value::Table(table), Value::Number(9.0)])
            .unwrap(),
        vec![Value::Nil]
    );

    runtime
        .table_set(table, &Value::Number(2.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        runtime
            .call_native(insert, &[Value::Table(table), Value::from("tail")])
            .unwrap(),
        Vec::<Value>::new()
    );
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        3.0_f64.to_bits()
    );
    assert!(
        runtime
            .call_native(
                insert,
                &[Value::Table(table), Value::Number(5.0), Value::Nil]
            )
            .is_err()
    );
}

#[test]
fn bit_numeric_fast_paths_match_their_callbacks() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let inputs = [
        0.0,
        1.0,
        -1.0,
        2.5,
        -3.9,
        31.0,
        33.0,
        2_147_483_648.0,
        4_294_967_295.0,
        -4_294_967_298.0,
        1e300,
        f64::NAN,
        f64::INFINITY,
    ];
    for name in ["tobit", "bnot", "bswap"] {
        let function = bit_function(&runtime, name);
        let unary = runtime.native_numeric_unary(function).unwrap();
        for input in inputs {
            assert_eq!(
                runtime
                    .call_native(function, &[Value::from(input)])
                    .unwrap(),
                vec![Value::from(unary(input))],
                "bit.{name}({input})"
            );
        }
    }
    for name in [
        "band", "bor", "bxor", "lshift", "rshift", "arshift", "rol", "ror",
    ] {
        let function = bit_function(&runtime, name);
        let binary = runtime.native_numeric_binary(function).unwrap();
        for left in inputs {
            for right in inputs {
                assert_eq!(
                    runtime
                        .call_native(function, &[Value::from(left), Value::from(right)])
                        .unwrap(),
                    vec![Value::from(binary(left, right))],
                    "bit.{name}({left}, {right})"
                );
            }
        }
    }
}

#[test]
fn error_gives_number_messages_a_position_only_in_lua51_family() {
    for (profile, positioned) in [
        (CompatibilityProfile::Lua51, true),
        (CompatibilityProfile::LuaJit21, true),
        (CompatibilityProfile::Lua54, false),
        (CompatibilityProfile::Lua55, false),
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::NativeFunction(error) = runtime.get_global("error") else {
            panic!("error is native");
        };
        for (message, text) in [(Value::Integer(42), "42"), (Value::Number(1.5), "1.5")] {
            // Level 1 and up: the VM prefixes the position to a string, so
            // the number must already be one. Level 0 never adds a position.
            let expected = if positioned && profile.is_lua51_family() {
                Value::from(text)
            } else {
                message.clone()
            };
            let level_one = runtime.call_native(error, std::slice::from_ref(&message));
            let level_two = runtime.call_native(error, &[message.clone(), Value::Integer(2)]);
            for result in [level_one, level_two] {
                let Err(RuntimeError::RaisedAtLevel { value, .. }) = result else {
                    panic!("error always raises");
                };
                assert_eq!(value, expected, "{profile:?}");
            }
            let level_zero = runtime.call_native(error, &[message.clone(), Value::Integer(0)]);
            assert_eq!(
                level_zero,
                Err(RuntimeError::RaisedAtLevel {
                    value: message,
                    level: 0
                }),
                "{profile:?}"
            );
        }
    }
}

/// `%.17g` of `math.log(x, b)` over a sweep of arguments, as the stock
/// interpreter prints it; `None` when the binary is not installed.
fn stock_log_sweep(binary: &str) -> Option<Vec<(f64, f64, f64)>> {
    let script = "local xs = {1000, 8, 100, 2, 10, math.exp(1), 3, 0.5, 1, 0, -1, math.huge, 0/0, 7, 1e300, 0.1, 27, 1024, 5e-324}\n\
        local bs = {2, 10, math.exp(1), 3, 0.5, 1, 0, -1, math.huge, 0/0, -math.huge, 100, 8}\n\
        for _, x in ipairs(xs) do for _, b in ipairs(bs) do\n\
        io.write(string.format('%.17g %.17g %.17g\\n', x, b, math.log(x, b))) end end";
    let path = std::env::temp_dir().join(format!(
        "oximond-log-sweep-{}-{binary}.lua",
        std::process::id()
    ));
    std::fs::write(&path, script).unwrap();
    let output = std::process::Command::new(format!("/usr/bin/{binary}"))
        .arg(&path)
        .output();
    std::fs::remove_file(&path).ok();
    let output = output.ok()?;
    let parse = |text: &str| -> f64 {
        match text.trim_start_matches('-') {
            "nan" => f64::NAN,
            "inf" => text.parse().unwrap_or(f64::INFINITY),
            _ => text.parse().unwrap(),
        }
    };
    Some(
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                let mut parts = line.split(' ').map(parse);
                (
                    parts.next().unwrap(),
                    parts.next().unwrap(),
                    parts.next().unwrap(),
                )
            })
            .collect(),
    )
}

fn assert_log_matches_stock(profile: CompatibilityProfile, binary: &str) {
    let Some(sweep) = stock_log_sweep(binary) else {
        eprintln!("skipping {profile:?}: no {binary} here");
        return;
    };
    assert!(sweep.len() > 200);
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, profile).unwrap();
    let log = math_function(&runtime, "log");
    for (value, base, expected) in sweep {
        let actual = runtime
            .call_native(log, &[Value::from(value), Value::from(base)])
            .unwrap();
        let [Value::Number(actual)] = actual[..] else {
            panic!("math.log gave {actual:?}");
        };
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{profile:?} log({value}, {base}): {actual:?} against {binary}'s {expected:?}"
        );
    }
}

#[test]
fn math_log_with_a_base_matches_the_stock_interpreters() {
    assert_log_matches_stock(CompatibilityProfile::Lua54, "lua5.4");
    assert_log_matches_stock(CompatibilityProfile::Lua55, "lua5.5");
    assert_log_matches_stock(CompatibilityProfile::LuaJit21, "luajit");
}

#[test]
#[allow(clippy::float_cmp, reason = "the bases are exactly 2 and 10")]
fn math_log_uses_exact_logarithms_for_bases_two_and_ten() {
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let log = math_function(&runtime, "log");
        for (value, base) in [
            (1000.0, 10.0),
            (8.0, 2.0),
            (1e15, 10.0),
            (2f64.powi(50), 2.0),
        ] {
            let expected = if base == 2.0 {
                value.log2()
            } else {
                value.log10()
            };
            assert_eq!(
                runtime
                    .call_native(log, &[Value::from(value), Value::from(base)])
                    .unwrap(),
                vec![Value::Number(expected)],
                "{profile:?} log({value}, {base})"
            );
        }
        // A nil base is no base, as `lua_isnoneornil` has it.
        assert_eq!(
            runtime
                .call_native(log, &[Value::from(8.0), Value::Nil])
                .unwrap(),
            vec![Value::Number(8.0_f64.ln())]
        );
    }
}

#[test]
fn math_pow_is_installed_everywhere_but_lua55() {
    for (profile, installed) in [
        (CompatibilityProfile::Lua51, true),
        (CompatibilityProfile::Lua52, true),
        (CompatibilityProfile::Lua53, true),
        (CompatibilityProfile::Lua54, true),
        (CompatibilityProfile::Lua55, false),
        (CompatibilityProfile::LuaJit21, true),
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::Table(math) = runtime.get_global("math") else {
            panic!("math is a table");
        };
        let pow = runtime.table_get(math, &Value::from("pow")).unwrap();
        assert_eq!(
            matches!(pow, Value::NativeFunction(_)),
            installed,
            "{profile:?}"
        );
    }
}

/// The fraction `math.modf` gives, as the stock interpreter prints its bits
/// with `string.format ('%a')`: the sign of a zero fraction is the argument's
/// before Lua 5.3.
#[test]
fn math_modf_fraction_keeps_the_sign_of_the_argument_like_the_stock_c_library() {
    let values = [
        "-3",
        "3",
        "-2.5",
        "2.5",
        "math.huge",
        "-math.huge",
        "-2147483648",
        "-0.5",
    ];
    for (profile, binary, signed) in [
        (CompatibilityProfile::Lua51, "lua5.1", true),
        (CompatibilityProfile::LuaJit21, "luajit", true),
        (CompatibilityProfile::Lua54, "lua5.4", false),
    ] {
        let script = format!(
            "for _, v in ipairs({{{}}}) do io.write(1 / select(2, math.modf(v)), '\\n') end",
            values.join(", ")
        );
        let path =
            std::env::temp_dir().join(format!("oximond-modf-{}-{binary}.lua", std::process::id()));
        std::fs::write(&path, &script).unwrap();
        let output = std::process::Command::new(format!("/usr/bin/{binary}"))
            .arg(&path)
            .output();
        std::fs::remove_file(&path).ok();
        let Ok(output) = output else {
            eprintln!("skipping {profile:?}: no {binary} here");
            continue;
        };
        let expected: Vec<f64> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| line.parse().unwrap_or(f64::NAN))
            .collect();
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let modf = math_function(&runtime, "modf");
        let arguments = [
            -3.0,
            3.0,
            -2.5,
            2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -2_147_483_648.0,
            -0.5,
        ];
        for (argument, expected) in arguments.iter().zip(&expected) {
            let results = runtime
                .call_native(modf, &[Value::from(*argument)])
                .unwrap();
            let Value::Number(fraction) = results[1] else {
                panic!("a float fraction: {results:?}");
            };
            assert_eq!(
                (1.0 / fraction).to_bits(),
                expected.to_bits(),
                "{profile:?} modf({argument}) fraction {fraction:?} (signed {signed})"
            );
        }
    }
}

#[test]
fn table_insert_and_remove_keep_their_errors_and_hash_keys_on_the_shift_path() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(library) = runtime.get_global("table") else {
        panic!("table is a table");
    };
    let native = |name| {
        let Value::NativeFunction(function) =
            runtime.table_get(library, &Value::from(name)).unwrap()
        else {
            panic!("table.{name} is native");
        };
        function
    };
    let insert = native("insert");
    let remove = native("remove");
    let table = runtime.create_table().unwrap();
    for index in 1..=3 {
        runtime
            .table_set(
                table,
                &Value::Number(f64::from(index)),
                Value::Number(f64::from(index)),
            )
            .unwrap();
    }
    let error = runtime
        .call_native(
            insert,
            &[Value::Table(table), Value::Number(5.0), Value::Nil],
        )
        .unwrap_err();
    assert!(error.to_string().contains("position out of bounds"));
    assert_eq!(
        runtime
            .call_native(remove, &[Value::Table(table), Value::Number(9.0)])
            .unwrap(),
        vec![Value::Nil]
    );
    // A hash-part key after the end is absorbed by the insert.
    runtime
        .table_set(table, &Value::Number(5.0), Value::Number(5.0))
        .unwrap();
    runtime
        .call_native(
            insert,
            &[Value::Table(table), Value::Number(1.0), Value::Number(0.0)],
        )
        .unwrap();
    let got: Vec<_> = (1..=5)
        .map(|index| {
            runtime
                .table_get(table, &Value::Number(f64::from(index)))
                .unwrap()
        })
        .collect();
    assert_eq!(got, [0.0, 1.0, 2.0, 3.0, 5.0].map(Value::Number).to_vec());
    // Readonly tables still raise.
    runtime.table_set_readonly(table, true).unwrap();
    assert!(
        runtime
            .call_native(remove, &[Value::Table(table), Value::Number(1.0)])
            .is_err()
    );
    // An empty table gives nil.
    let empty = runtime.create_table().unwrap();
    assert_eq!(
        runtime.call_native(remove, &[Value::Table(empty)]).unwrap(),
        vec![Value::Nil]
    );
}
