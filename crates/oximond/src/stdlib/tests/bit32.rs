// The tables hold raw bit32 expectations; digit separators would obscure them.
#![allow(clippy::unreadable_literal)]

use super::*;

#[test]
fn bit32_unsigned_words_and_shift_boundaries() {
    for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::Table(table) = runtime.get_global("bit32") else {
            panic!("bit32 table")
        };
        let cases: &[(&str, &[f64], f64)] = &[
            ("bnot", &[297.0], 4294966998.0),
            // bit32 keeps mathematical low words for wide numbers rather
            // than using LuaJIT BitOp's biased IEEE-754 conversion.
            ("band", &[3002399751580330.0], 2863311530.0),
            ("band", &[-3002399751580330.0], 1431655766.0),
            ("band", &[], 4294967295.0),
            ("bor", &[], 0.0),
            ("bxor", &[], 0.0),
            ("band", &[-1.0, 4294967295.0], 4294967295.0),
            ("bor", &[2147483648.0, 1.0], 2147483649.0),
            ("bxor", &[0.0, -1.0], 4294967295.0),
            ("lshift", &[16.0, -4.0], 1.0),
            ("rshift", &[1.0, -4.0], 16.0),
            ("lshift", &[1.0, 32.0], 0.0),
            ("rshift", &[-1.0, 32.0], 0.0),
            ("arshift", &[-1.0, 32.0], 4294967295.0),
            ("arshift", &[-1.0, -1.0], 4294967294.0),
            ("lrotate", &[1.0, -1.0], 2147483648.0),
            ("rrotate", &[1.0, 33.0], 2147483648.0),
            ("extract", &[4294967295.0, 0.0, 32.0], 4294967295.0),
            ("replace", &[0.0, 15.0, 28.0, 4.0], 4026531840.0),
        ];
        for &(name, args, expected) in cases {
            let Value::NativeFunction(function) =
                runtime.table_get(table, &Value::from(name)).unwrap()
            else {
                panic!("{name} native")
            };
            let args: Vec<_> = args.iter().copied().map(Value::Number).collect();
            let result = runtime.call_native(function, &args).unwrap();
            assert_eq!(
                result,
                vec![if profile == CompatibilityProfile::Lua53 {
                    Value::Integer(expected as i64)
                } else {
                    Value::Number(expected)
                }],
                "{profile:?} {name} {args:?}"
            );
        }
    }
}

#[test]
fn bit32_profiles_validate_inputs_and_luajit_keeps_signed_results() {
    for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua53] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::Table(table) = runtime.get_global("bit32") else {
            panic!("bit32 table")
        };
        let Value::NativeFunction(bnot) = runtime.table_get(table, &Value::from("bnot")).unwrap()
        else {
            panic!("bnot native")
        };
        let result = runtime.call_native(bnot, &[Value::Number(1.5)]);
        if profile == CompatibilityProfile::Lua52 {
            assert_eq!(result.unwrap(), vec![Value::Number(4294967293.0)]);
        } else {
            assert!(result.is_err());
        }
        for (name, args) in [
            ("extract", vec![0.0, -1.0]),
            ("extract", vec![0.0, 31.0, 2.0]),
            ("extract", vec![0.0, 0.0, 0.0]),
            ("replace", vec![0.0, 1.0, 32.0]),
        ] {
            let Value::NativeFunction(function) =
                runtime.table_get(table, &Value::from(name)).unwrap()
            else {
                panic!("{name} native")
            };
            let args: Vec<_> = args.into_iter().map(Value::Number).collect();
            assert!(runtime.call_native(function, &args).is_err());
        }
        let Value::NativeFunction(btest) = runtime.table_get(table, &Value::from("btest")).unwrap()
        else {
            panic!("btest native")
        };
        assert_eq!(
            runtime.call_native(btest, &[]).unwrap(),
            vec![Value::Boolean(true)]
        );
    }
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::LuaJit21).unwrap();
    let bnot = bit_function(&runtime, "bnot");
    assert_eq!(
        runtime.call_native(bnot, &[Value::Number(297.0)]).unwrap(),
        vec![Value::Number(-298.0)]
    );
    let lshift = bit_function(&runtime, "lshift");
    assert_eq!(
        runtime
            .call_native(lshift, &[Value::Number(1.0), Value::Number(32.0)])
            .unwrap(),
        vec![Value::Number(1.0)]
    );
}

#[test]
fn the_trace_recognizes_the_table_and_bit_functions_by_kind() {
    use crate::runtime::DirectNativeKind as Kind;
    let named = |runtime: &Runtime, library: Option<&str>, name: &str| {
        let table = match library {
            None => runtime.global_environment(),
            Some(library) => match runtime.get_global(library) {
                Value::Table(table) => table,
                other => panic!("{library}: {other:?}"),
            },
        };
        match runtime.table_get(table, &Value::from(name)).unwrap() {
            Value::NativeFunction(function) => function,
            other => panic!("{name}: {other:?}"),
        }
    };
    for profile in CompatibilityProfile::ALL {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        for (name, kind) in [
            ("rawget", Kind::RawGet),
            ("rawset", Kind::RawSet),
            ("getmetatable", Kind::GetMetatable),
            ("setmetatable", Kind::SetMetatable),
        ] {
            let function = named(&runtime, None, name);
            assert_eq!(
                runtime.direct_native_kind(function),
                Some(kind),
                "{profile:?} {name}"
            );
        }
        if profile.is_lua51_family() {
            let function = named(&runtime, None, "unpack");
            assert_eq!(runtime.direct_native_kind(function), Some(Kind::Unpack));
        } else {
            let function = named(&runtime, None, "rawlen");
            assert_eq!(runtime.direct_native_kind(function), Some(Kind::RawLen));
            let function = named(&runtime, Some("table"), "unpack");
            let kind = if profile == CompatibilityProfile::Lua52 {
                Kind::Unpack
            } else {
                Kind::TableUnpack
            };
            assert_eq!(
                runtime.direct_native_kind(function),
                Some(kind),
                "{profile:?}"
            );
        }
        if matches!(
            profile,
            CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53
        ) {
            for (name, kind) in [
                ("band", Kind::Bit32Band),
                ("bor", Kind::Bit32Bor),
                ("bxor", Kind::Bit32Bxor),
                ("bnot", Kind::Bit32Bnot),
                ("lshift", Kind::Bit32Lshift),
                ("rshift", Kind::Bit32Rshift),
                ("arshift", Kind::Bit32Arshift),
                ("lrotate", Kind::Bit32Lrotate),
                ("rrotate", Kind::Bit32Rrotate),
                ("btest", Kind::Bit32Btest),
                ("extract", Kind::Bit32Extract),
                ("replace", Kind::Bit32Replace),
            ] {
                let function = named(&runtime, Some("bit32"), name);
                assert_eq!(
                    runtime.direct_native_kind(function),
                    Some(kind),
                    "{profile:?} {name}"
                );
            }
        }
    }
}
