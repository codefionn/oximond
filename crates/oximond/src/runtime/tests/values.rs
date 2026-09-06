use super::*;

#[test]
fn value_fits_two_eight_byte_words() {
    assert!(std::mem::size_of::<crate::runtime::Value>() <= 16);
}

#[test]
fn scalar_native_dispatch_preserves_nil_and_does_not_call_general_callbacks() {
    let mut runtime = Runtime::new();
    let general = runtime
        .register_native(|runtime, _| {
            runtime.set_global("general_called", Value::Boolean(true));
            Ok(vec![Value::Number(1.0), Value::Nil])
        })
        .unwrap();
    assert_eq!(runtime.call_native_scalar(general, &[]).unwrap(), None);
    assert_eq!(runtime.get_global("general_called"), Value::Nil);
    assert_eq!(
        runtime.call_native(general, &[]).unwrap(),
        vec![Value::Number(1.0), Value::Nil]
    );
    let scalar = runtime
        .register_native_scalar(|_, arguments| Ok(arguments.first().cloned().unwrap_or(Value::Nil)))
        .unwrap();
    assert_eq!(
        runtime.call_native_scalar(scalar, &[]).unwrap(),
        Some(Value::Nil)
    );
    assert_eq!(runtime.call_native(scalar, &[]).unwrap(), vec![Value::Nil]);
    assert_eq!(
        runtime
            .call_native_scalar(scalar, &[Value::Number(3.0), Value::Nil])
            .unwrap(),
        Some(Value::Number(3.0))
    );
    let failure = runtime
        .register_native_scalar(|_, _| Err(RuntimeError::message("scalar failed")))
        .unwrap();
    assert!(runtime.call_native_scalar(failure, &[]).is_err());
    assert!(runtime.call_native(failure, &[]).is_err());
}

#[test]
fn native_closure_captures_state_and_can_mutate_runtime() {
    let mut runtime = Runtime::new();
    let offset = 7.0;
    let function = runtime
        .register_native_closure(move |runtime, arguments| {
            let Value::Number(value) = arguments[0] else {
                return Err(RuntimeError::message("expected a number"));
            };
            let result = Value::Number(value + offset);
            runtime.set_global("last_result", result.clone());
            Ok(vec![result])
        })
        .unwrap();
    assert_eq!(runtime.call_native_scalar(function, &[]).unwrap(), None);
    assert_eq!(
        runtime
            .call_native(function, &[Value::Number(3.0)])
            .unwrap(),
        vec![Value::Number(10.0)]
    );
    assert_eq!(runtime.get_global("last_result"), Value::Number(10.0));
}

#[test]
fn raw_index_fast_path_distinguishes_nil_from_metamethod_fallback() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let key = Value::from("missing");
    assert_eq!(
        runtime.table_get_if_no_index(table, &key).unwrap(),
        Some(Value::Nil)
    );
    let metatable = runtime.create_table().unwrap();
    runtime.set_table_metatable(table, Some(metatable)).unwrap();
    assert_eq!(runtime.table_get_if_no_index(table, &key).unwrap(), None);
    runtime
        .table_set(table, &key, Value::Boolean(false))
        .unwrap();
    assert_eq!(
        runtime.table_get_if_no_index(table, &key).unwrap(),
        Some(Value::Boolean(false))
    );
    assert_eq!(
        runtime
            .table_get_if_no_index(table, &Value::Number(f64::NAN))
            .unwrap(),
        None
    );
}

#[test]
fn format_value_spells_integer_boundaries_like_lua51() {
    // Stock Lua prints floats with "%.14g", so integral floats beyond
    // fourteen digits switch to exponent notation (checked against lua5.1).
    let runtime = Runtime::new();
    let threshold = 9_223_372_036_854_775_808.0_f64;
    let cases = [
        (-0.0, "-0"),
        (-threshold, "-9.2233720368548e+18"),
        (-4_611_686_018_427_387_904.0, "-4.6116860184274e+18"),
        (-1.0, "-1"),
        (0.0, "0"),
        (1.0, "1"),
        (42.0, "42"),
        (99_999_999_999_999.0, "99999999999999"),
        (100_000_000_000_000.0, "1e+14"),
        (4_611_686_018_427_387_904.0, "4.6116860184274e+18"),
        (threshold, "9.2233720368548e+18"),
        (
            f64::from_bits(threshold.to_bits() + 1),
            "9.2233720368548e+18",
        ),
        (
            -f64::from_bits(threshold.to_bits() + 1),
            "-9.2233720368548e+18",
        ),
        (1.5, "1.5"),
        (-1.5, "-1.5"),
        (f64::NAN, "nan"),
        (-f64::NAN, "-nan"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
    ];
    for (value, expected) in cases {
        assert_eq!(
            runtime
                .format_value(&Value::Number(value))
                .unwrap()
                .as_bytes(),
            expected.as_bytes(),
            "value bits: {:x}",
            value.to_bits(),
        );
    }
}

#[test]
fn globals_and_natives_use_handles() {
    fn answer(runtime: &mut Runtime, _: &[Value]) -> crate::runtime::RuntimeResult<Vec<Value>> {
        runtime.create_table()?;
        Ok(vec![Value::from(42.0)])
    }
    let mut runtime = Runtime::new();
    let function = runtime.register_global_native("answer", answer).unwrap();
    assert_eq!(
        runtime.get_global("answer"),
        Value::NativeFunction(function)
    );
    assert_eq!(
        runtime.call_native(function, &[]).unwrap(),
        vec![Value::from(42.0)]
    );
}

#[test]
fn raw_numeric_equality_normalizes_zero_and_excludes_nan() {
    assert!(Runtime::raw_equal(&Value::from(-0.0), &Value::from(0.0)));
    assert!(!Runtime::raw_equal(
        &Value::from(f64::NAN),
        &Value::from(f64::NAN)
    ));
}

#[test]
fn numeric_strings_convert_without_affecting_value_identity() {
    assert_eq!(
        Runtime::coerce_number(&Value::from("  -1.25e2  "), "perform arithmetic"),
        Ok(-125.0)
    );
    assert_eq!(
        Runtime::coerce_number(&Value::from(" -0xa "), "perform arithmetic"),
        Ok(-10.0)
    );
    assert_eq!(
        Runtime::coerce_number(&Value::from("0x1.8p1"), "perform arithmetic"),
        Ok(3.0)
    );
    assert_eq!(
        Runtime::coerce_number(&Value::from("nope"), "perform arithmetic"),
        Err(RuntimeError::TypeError {
            operation: "perform arithmetic",
            expected: "number",
            found: "string",
        })
    );
    assert!(!Runtime::raw_equal(&Value::from("2"), &Value::Number(2.0)));
}

#[test]
fn nonfinite_numerals_follow_strtod_or_the_luajit_scanner() {
    use crate::runtime::NonFiniteNumerals::{LuaJit, Rejected, Strtod};
    for style in [Strtod, LuaJit] {
        for text in ["inf", "+INF", " Infinity\t", "iNfInItY"] {
            assert_eq!(style.parse(text.as_bytes()), Some(f64::INFINITY), "{text}");
        }
        assert_eq!(style.parse(b"-inf"), Some(f64::NEG_INFINITY));
        assert!(style.parse(b" NaN ").is_some_and(f64::is_nan));
        for text in [
            "", "in", "inff", "infinit", "nanx", "nan(", "0xinf", "--inf",
        ] {
            assert_eq!(style.parse(text.as_bytes()), None, "{text}");
        }
    }
    assert!(Strtod.parse(b"nan(12_ab)").is_some_and(f64::is_nan));
    assert!(Strtod.parse(b"nan()").is_some_and(f64::is_nan));
    assert!(Strtod.parse(b"-nan").is_some_and(f64::is_sign_negative));
    assert!(Strtod.parse(b"nan").is_some_and(f64::is_sign_positive));
    assert_eq!(LuaJit.parse(b"nan(1)"), None);
    assert_eq!(
        LuaJit.parse(b"nan").map(f64::to_bits),
        Some(0xfff8_0000_0000_0000)
    );
    assert_eq!(Rejected.parse(b"inf"), None);
    assert_eq!(
        Runtime::coerce_number_in(&Value::from(" -inf "), "perform arithmetic", Strtod),
        Ok(f64::NEG_INFINITY)
    );
    assert!(Runtime::coerce_number(&Value::from("inf"), "perform arithmetic").is_err());
}

#[test]
fn raised_tagged_integers_display_as_numbers() {
    assert_eq!(RuntimeError::Raised(Value::Integer(42)).to_string(), "42");
    assert_eq!(
        RuntimeError::RaisedAtLevel {
            value: Value::Integer(-7),
            level: 1,
        }
        .to_string(),
        "-7"
    );
}

#[test]
fn cint64_conversions_follow_luajit_on_x86() {
    use crate::runtime::cint64::{number_to_i64, number_to_u64};
    assert_eq!(number_to_i64(-2.5), -2);
    assert_eq!(number_to_i64(9.2e18), 9_200_000_000_000_000_000);
    for out_of_range in [9_223_372_036_854_775_808.0, 1e300, f64::INFINITY, f64::NAN] {
        assert_eq!(number_to_i64(out_of_range), i64::MIN);
    }
    assert_eq!(number_to_u64(-1.0), u64::MAX);
    assert_eq!(number_to_u64(1.9e19), 553_255_926_290_448_384);
    assert_eq!(number_to_u64(18_446_744_073_709_551_616.0), 0);
    assert_eq!(number_to_u64(-1e19), 1 << 63);
}

#[test]
fn cint64_arithmetic_uses_the_common_c_type() {
    use crate::runtime::cint64::{CIntArith, arith, equal, less, negate};
    assert_eq!(
        arith(CIntArith::Add, &Value::Int64(1), &Value::Number(2.7)),
        Some(Value::Int64(3))
    );
    assert_eq!(
        arith(CIntArith::Subtract, &Value::UInt64(0), &Value::Number(1.0)),
        Some(Value::UInt64(u64::MAX))
    );
    assert_eq!(
        arith(CIntArith::Multiply, &Value::Int64(-2), &Value::UInt64(3)),
        Some(Value::UInt64(u64::MAX - 5))
    );
    assert_eq!(
        arith(CIntArith::Divide, &Value::Int64(7), &Value::Number(0.0)),
        Some(Value::Int64(i64::MIN))
    );
    assert_eq!(
        arith(CIntArith::Modulo, &Value::Int64(-7), &Value::Number(3.0)),
        Some(Value::Int64(-1))
    );
    assert_eq!(
        arith(CIntArith::Power, &Value::Int64(0), &Value::Number(-1.0)),
        Some(Value::Int64(i64::MAX))
    );
    assert_eq!(
        arith(CIntArith::Add, &Value::Int64(1), &Value::from("2")),
        None
    );
    assert_eq!(negate(&Value::UInt64(5)), Some(Value::UInt64(u64::MAX - 4)));
    assert_eq!(equal(&Value::Int64(1), &Value::Number(1.5)), Some(true));
    assert_eq!(equal(&Value::Int64(1), &Value::from("1")), Some(false));
    assert_eq!(
        less(&Value::Int64(-1), &Value::UInt64(1), false),
        Some(false)
    );
    assert_eq!(
        less(&Value::Int64(-1), &Value::Number(1.0), false),
        Some(true)
    );
}

#[test]
fn small_integers_share_one_string_and_format_like_any_other() {
    use crate::runtime::{NumberFormat, small_integer_string};
    for value in [0_u64, 7, 99, 255] {
        let shared = small_integer_string(value).expect("below the sharing limit");
        assert_eq!(shared.as_bytes(), value.to_string().as_bytes());
        assert_eq!(
            shared.identity(),
            small_integer_string(value).unwrap().identity()
        );
    }
    assert!(small_integer_string(256).is_none());
    // Floats that print as plain digits take the shared string; a `.0`
    // suffix, a sign, or a large value never does.
    let plain = NumberFormat::Lua51.format_string(42.0);
    assert_eq!(plain.as_bytes(), b"42");
    assert_eq!(
        plain.identity(),
        NumberFormat::Lua51.format_string(42.0).identity()
    );
    assert_eq!(NumberFormat::Lua53.format_string(42.0).as_bytes(), b"42.0");
    assert_eq!(NumberFormat::Lua51.format_string(-0.0).as_bytes(), b"-0");
    assert_eq!(NumberFormat::Lua51.format_string(-5.0).as_bytes(), b"-5");
    assert_eq!(NumberFormat::Lua51.format_string(256.0).as_bytes(), b"256");
    assert_eq!(
        Runtime::new().format_value(&Value::Integer(255)).unwrap(),
        small_integer_string(255).unwrap()
    );
    assert_eq!(
        Runtime::new().format_value(&Value::Integer(-1)).unwrap(),
        crate::string::LuaString::from("-1")
    );
}

#[test]
fn shared_metatables_are_kept_per_primitive_kind() {
    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    assert_eq!(runtime.raw_metatable(&Value::Number(1.0)).unwrap(), None);
    runtime
        .raw_set_metatable(&Value::Number(1.0), Some(metatable))
        .unwrap();
    // Integers and floats share the number metatable; nothing else does.
    assert_eq!(
        runtime.raw_metatable(&Value::Integer(3)).unwrap(),
        Some(metatable)
    );
    assert_eq!(
        runtime.raw_metatable(&Value::Number(2.5)).unwrap(),
        Some(metatable)
    );
    assert_eq!(runtime.raw_metatable(&Value::Boolean(true)).unwrap(), None);
    assert_eq!(runtime.raw_metatable(&Value::from("text")).unwrap(), None);
    runtime
        .raw_set_metatable(&Value::Number(1.0), None)
        .unwrap();
    assert_eq!(runtime.raw_metatable(&Value::Integer(3)).unwrap(), None);
}

#[test]
fn native_plan_tells_each_registration_form_apart() {
    extern "C" fn twice(value: f64) -> f64 {
        value * 2.0
    }
    extern "C" fn plus(left: f64, right: f64) -> f64 {
        left + right
    }

    let mut runtime = Runtime::new();
    let scalar = runtime
        .register_native_scalar(|_, _| Ok(Value::Nil))
        .unwrap();
    let unary = runtime
        .register_native_numeric_unary(|_, _| Ok(Value::Nil), twice)
        .unwrap();
    let binary = runtime
        .register_native_numeric_binary(|_, _| Ok(Value::Nil), plus)
        .unwrap();
    let multiple = runtime.register_native(|_, _| Ok(Vec::new())).unwrap();
    let closure = runtime
        .register_native_closure(|_, _| Ok(Vec::new()))
        .unwrap();
    let dispatched = runtime
        .register_native_dispatched(|_, _| Ok(Value::Nil))
        .unwrap();

    assert!(matches!(runtime.native_plan(scalar), NativePlan::Scalar(_)));
    match runtime.native_plan(unary) {
        NativePlan::Unary(_, function) => assert_eq!(function(1.5), 3.0),
        _ => panic!("a numeric unary registration plans as unary"),
    }
    match runtime.native_plan(binary) {
        NativePlan::Binary(_, function) => assert_eq!(function(1.5, 2.0), 3.5),
        _ => panic!("a numeric binary registration plans as binary"),
    }
    assert!(matches!(
        runtime.native_plan(multiple),
        NativePlan::Multiple(_)
    ));
    let paired = runtime
        .register_native_paired(
            |_, _| Ok(vec![Value::Nil, Value::Nil]),
            |_, _| Ok(Some(NativePair::two(Value::Nil, Value::Nil))),
        )
        .unwrap();
    assert!(matches!(
        runtime.native_plan(paired),
        NativePlan::Paired(_, _)
    ));
    // Every caller that is not the window path runs the general callback.
    assert_eq!(runtime.call_native(paired, &[]).unwrap().len(), 2);
    assert!(matches!(
        runtime.native_call_kind(paired),
        NativeCallKind::Multiple(_)
    ));
    assert!(matches!(runtime.native_plan(closure), NativePlan::Other));
    assert!(matches!(runtime.native_plan(dispatched), NativePlan::Other));
    assert!(matches!(
        runtime.native_plan(NativeFunctionHandle(u32::MAX)),
        NativePlan::Other
    ));
}

#[test]
fn number_metatable_follows_the_shared_number_metatable() {
    let mut runtime = Runtime::new();
    assert_eq!(runtime.number_metatable(), None);
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Number(1.0), Some(metatable))
        .unwrap();
    assert_eq!(runtime.number_metatable(), Some(metatable));
    // Integers and floats share it.
    runtime.raw_set_metatable(&Value::Integer(1), None).unwrap();
    assert_eq!(runtime.number_metatable(), None);
}

#[test]
fn table_handle_is_the_payload_word() {
    let mut runtime = Runtime::new();
    let mut handles = Vec::new();
    // Slots and generations that differ in both halves of the word.
    for _ in 0..70 {
        handles.push(runtime.create_table().unwrap());
    }
    handles.push(crate::runtime::TableHandle::from_native_identity(
        0x1234_5678_9abc_def0,
    ));
    for handle in handles {
        let value = Value::Table(handle);
        assert_eq!(value.table_handle_word(), Some(handle));
        assert_eq!(
            value
                .table_handle_word()
                .map(crate::runtime::TableHandle::native_identity),
            Some(handle.native_identity())
        );
    }
    for other in [
        Value::Nil,
        Value::Boolean(true),
        Value::Integer(1),
        Value::Number(2.5),
        Value::string("text"),
    ] {
        assert_eq!(other.table_handle_word(), None);
    }
}
