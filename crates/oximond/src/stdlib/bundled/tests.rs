use super::*;
use crate::compat::CompatibilityProfile;

/// Installs the loaders and runs each as `require` does, storing its module
/// in the returned `package.loaded` stand-in.
fn load_all(runtime: &mut Runtime) -> TableHandle {
    let loaded = runtime.create_table().unwrap();
    let preload = runtime.create_table().unwrap();
    install(runtime, preload).unwrap();
    for name in ["cjson", "cjson.safe", "struct", "cmsgpack", "cmsgpack_safe"] {
        let Value::NativeFunction(factory) =
            runtime.table_get(preload, &Value::from(name)).unwrap()
        else {
            panic!("{name} must have a loader");
        };
        let module = runtime.call_native(factory, &[Value::from(name)]).unwrap();
        runtime
            .table_set(loaded, &Value::from(name), module[0].clone())
            .unwrap();
    }
    loaded
}

fn setup() -> (Runtime, TableHandle) {
    let mut runtime = Runtime::new();
    let loaded = load_all(&mut runtime);
    (runtime, loaded)
}

fn module(runtime: &Runtime, loaded: TableHandle, name: &str) -> TableHandle {
    let Value::Table(table) = runtime.table_get(loaded, &Value::from(name)).unwrap() else {
        panic!("{name} must be a table");
    };
    table
}

fn call(
    runtime: &mut Runtime,
    table: TableHandle,
    name: &str,
    args: &[Value],
) -> RuntimeResult<Vec<Value>> {
    let Value::NativeFunction(function) = runtime.table_get(table, &Value::from(name))? else {
        panic!("{name} must be native");
    };
    runtime.call_native(function, args)
}

#[test]
fn messagepack_scalar_wire_widths_and_binary_strings() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cmsgpack");
    for (number, bytes) in [
        (127.0, vec![0x7f]),
        (128.0, vec![0xcc, 0x80]),
        (256.0, vec![0xcd, 1, 0]),
        (65536.0, vec![0xce, 0, 1, 0, 0]),
        (4_294_967_296.0, vec![0xcf, 0, 0, 0, 1, 0, 0, 0, 0]),
        (-32.0, vec![0xe0]),
        (-33.0, vec![0xd0, 0xdf]),
        (-129.0, vec![0xd1, 0xff, 0x7f]),
        (-32769.0, vec![0xd2, 0xff, 0xff, 0x7f, 0xff]),
    ] {
        assert_eq!(
            call(&mut runtime, table, "pack", &[Value::Number(number)]).unwrap(),
            vec![Value::bytes(bytes.clone())]
        );
        assert_eq!(
            call(&mut runtime, table, "unpack", &[Value::bytes(bytes)]).unwrap(),
            vec![runtime.integer_value(number as i64)]
        );
    }
    assert_eq!(
        call(
            &mut runtime,
            table,
            "pack",
            &[
                Value::Nil,
                Value::Boolean(false),
                Value::Boolean(true),
                Value::bytes(vec![0, 255])
            ]
        )
        .unwrap(),
        vec![Value::bytes(vec![0xc0, 0xc2, 0xc3, 0xa2, 0, 255])]
    );
    assert_eq!(
        call(&mut runtime, table, "pack", &[Value::Number(1.5)]).unwrap(),
        vec![Value::bytes(vec![0xca, 0x3f, 0xc0, 0, 0])]
    );
    let mut double = vec![0xcb];
    double.extend_from_slice(&0.1f64.to_be_bytes());
    assert_eq!(
        call(&mut runtime, table, "pack", &[Value::Number(0.1)]).unwrap(),
        vec![Value::bytes(double)]
    );
}

#[test]
fn messagepack_offsets_nil_results_errors_and_depth_limit() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cmsgpack");
    let bytes = Value::bytes(vec![1, 0xc0, 2]);
    assert_eq!(
        call(&mut runtime, table, "unpack", std::slice::from_ref(&bytes)).unwrap(),
        vec![
            runtime.integer_value(1),
            Value::Nil,
            runtime.integer_value(2)
        ]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "unpack_one",
            &[bytes.clone(), Value::Number(1.0)]
        )
        .unwrap(),
        vec![runtime.integer_value(2), Value::Nil]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "unpack_limit",
            &[bytes, Value::Number(2.0), Value::Number(1.0)]
        )
        .unwrap(),
        vec![
            runtime.integer_value(-1),
            Value::Nil,
            runtime.integer_value(2)
        ]
    );
    for bytes in [vec![0xc1], vec![0xcd, 1], vec![0x92, 1]] {
        assert!(
            call(
                &mut runtime,
                table,
                "unpack",
                &[Value::bytes(bytes.clone())]
            )
            .is_err()
        );
        let safe = module(&runtime, loaded, "cmsgpack_safe");
        let result = call(&mut runtime, safe, "unpack", &[Value::bytes(bytes)]).unwrap();
        assert!(matches!(result.as_slice(), [Value::Nil, Value::String(_)]));
    }
    let nested = runtime.create_table().unwrap();
    runtime
        .table_set(nested, &Value::Number(1.0), Value::Table(nested))
        .unwrap();
    let mut expected = vec![0x91; 16];
    expected.push(0xc0);
    assert_eq!(
        call(&mut runtime, table, "pack", &[Value::Table(nested)]).unwrap(),
        vec![Value::bytes(expected)]
    );
}

#[test]
fn struct_endian_alignment_signed_wrap_and_variable_strings() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "struct");
    assert_eq!(
        call(
            &mut runtime,
            table,
            "pack",
            &[
                Value::from("<!4BI2"),
                Value::Number(1.0),
                Value::Number(f64::from(0x1234))
            ]
        )
        .unwrap(),
        vec![Value::bytes(vec![1, 0, 0x34, 0x12])]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "pack",
            &[
                Value::from(">I2b"),
                Value::Number(f64::from(0x1234)),
                Value::Number(255.0)
            ]
        )
        .unwrap(),
        vec![Value::bytes(vec![0x12, 0x34, 0xff])]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "unpack",
            &[Value::from(">I2b"), Value::bytes(vec![0x12, 0x34, 0xff])]
        )
        .unwrap(),
        vec![
            Value::Number(4660.0),
            Value::Number(-1.0),
            runtime.integer_value(4)
        ]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "unpack",
            &[Value::from("Bc0s"), Value::bytes(vec![2, 0, 255, b'a', 0])]
        )
        .unwrap(),
        vec![
            Value::bytes(vec![0, 255]),
            Value::from("a"),
            runtime.integer_value(6)
        ]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "pack",
            &[Value::from("s"), Value::bytes(vec![b'a', 0, b'b'])]
        )
        .unwrap(),
        vec![Value::bytes(vec![b'a', 0, b'b', 0])]
    );
    assert_eq!(
        call(&mut runtime, table, "size", &[Value::from("!4BI4")]).unwrap(),
        vec![runtime.integer_value(8)]
    );
    assert_eq!(
        call(&mut runtime, table, "size", &[Value::from("z")]).unwrap(),
        vec![runtime.integer_value(0)]
    );
    assert!(call(&mut runtime, table, "pack", &[Value::from("z")]).is_err());
    for format in ["s", "c0", "!3I4", "I33"] {
        assert!(call(&mut runtime, table, "size", &[Value::from(format)]).is_err());
    }
    assert!(
        call(
            &mut runtime,
            table,
            "unpack",
            &[Value::from("s"), Value::from("unfinished")]
        )
        .is_err()
    );
    assert!(
        call(
            &mut runtime,
            table,
            "unpack",
            &[Value::from("I4"), Value::bytes(vec![1])]
        )
        .is_err()
    );
}

#[test]
fn cjson_byte_escapes_unicode_null_and_decode_failures() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    assert_eq!(
        call(
            &mut runtime,
            table,
            "encode",
            &[Value::bytes(vec![0, b'\n', b'/', b'"', 255])]
        )
        .unwrap(),
        vec![Value::bytes(b"\"\\u0000\\n\\/\\\"\xff\"".to_vec())]
    );
    assert_eq!(
        call(
            &mut runtime,
            table,
            "decode",
            &[Value::from("\"\\u20ac\\ud83d\\ude00\"")]
        )
        .unwrap(),
        vec![Value::from("€😀")]
    );
    let null = runtime.table_get(table, &Value::from("null")).unwrap();
    assert_eq!(
        call(&mut runtime, table, "decode", &[Value::from("null")]).unwrap(),
        vec![null.clone()]
    );
    assert_eq!(
        call(&mut runtime, table, "encode", &[null]).unwrap(),
        vec![Value::from("null")]
    );
    for source in [
        "",
        "[1,]",
        "{\"x\":}",
        "true false",
        "\"\\ud800\"",
        "\"\\uZZZZ\"",
    ] {
        assert!(
            call(&mut runtime, table, "decode", &[Value::from(source)]).is_err(),
            "{source}"
        );
        let safe = module(&runtime, loaded, "cjson.safe");
        assert!(matches!(
            call(&mut runtime, safe, "decode", &[Value::from(source)])
                .unwrap()
                .as_slice(),
            [Value::Nil, Value::String(_)]
        ));
    }
}

#[test]
fn cjson_configuration_sparse_arrays_depth_and_new_are_independent() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    call(
        &mut runtime,
        table,
        "encode_number_precision",
        &[Value::Number(3.0)],
    )
    .unwrap();
    assert_eq!(
        call(&mut runtime, table, "encode", &[Value::Number(1.234_567)]).unwrap(),
        vec![Value::from("1.23")]
    );
    let Value::Table(new) = call(&mut runtime, table, "new", &[]).unwrap()[0] else {
        panic!("new table")
    };
    assert_eq!(
        call(&mut runtime, new, "encode_number_precision", &[]).unwrap(),
        vec![runtime.integer_value(14)]
    );
    let safe = module(&runtime, loaded, "cjson.safe");
    assert_eq!(
        call(&mut runtime, safe, "encode_number_precision", &[]).unwrap(),
        vec![runtime.integer_value(14)]
    );
    assert!(
        call(
            &mut runtime,
            table,
            "encode",
            &[Value::Number(f64::INFINITY)]
        )
        .is_err()
    );
    call(
        &mut runtime,
        table,
        "encode_invalid_numbers",
        &[Value::from("null")],
    )
    .unwrap();
    assert_eq!(
        call(&mut runtime, table, "encode", &[Value::Number(f64::NAN)]).unwrap(),
        vec![Value::from("null")]
    );
    call(
        &mut runtime,
        table,
        "decode_invalid_numbers",
        &[Value::Boolean(false)],
    )
    .unwrap();
    assert!(call(&mut runtime, table, "decode", &[Value::from("NaN")]).is_err());
    let sparse = runtime.create_table().unwrap();
    runtime
        .table_set(sparse, &Value::Number(100.0), Value::from("x"))
        .unwrap();
    assert!(call(&mut runtime, table, "encode", &[Value::Table(sparse)]).is_err());
    call(
        &mut runtime,
        table,
        "encode_sparse_array",
        &[Value::Boolean(true)],
    )
    .unwrap();
    assert_eq!(
        call(&mut runtime, table, "encode", &[Value::Table(sparse)]).unwrap(),
        vec![Value::from("{\"100\":\"x\"}")]
    );
    call(
        &mut runtime,
        table,
        "decode_max_depth",
        &[Value::Number(1.0)],
    )
    .unwrap();
    assert!(call(&mut runtime, table, "decode", &[Value::from("[[1]]")]).is_err());
    let cycle = runtime.create_table().unwrap();
    runtime
        .table_set(cycle, &Value::Number(1.0), Value::Table(cycle))
        .unwrap();
    call(
        &mut runtime,
        table,
        "encode_max_depth",
        &[Value::Number(2.0)],
    )
    .unwrap();
    assert!(call(&mut runtime, table, "encode", &[Value::Table(cycle)]).is_err());
}

#[test]
fn bundled_numeric_results_follow_each_profile() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
        CompatibilityProfile::LuaJit21,
    ] {
        let mut runtime = Runtime::new();
        crate::stdlib::install_pure_with_profile(&mut runtime, profile).unwrap();
        let loaded = load_all(&mut runtime);
        let table = module(&runtime, loaded, "cmsgpack");
        assert_eq!(
            call(&mut runtime, table, "unpack", &[Value::bytes(vec![42])]).unwrap(),
            vec![runtime.integer_value(42)],
            "{profile:?}"
        );
    }
}

#[test]
fn cjson_null_survives_collection_with_only_native_closures_retained() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    let Value::NativeFunction(decode) = runtime.table_get(table, &Value::from("decode")).unwrap()
    else {
        panic!("decoder")
    };
    let Value::NativeFunction(new) = runtime.table_get(table, &Value::from("new")).unwrap() else {
        panic!("constructor")
    };
    runtime
        .table_set(table, &Value::from("null"), Value::Nil)
        .unwrap();
    runtime
        .table_set(loaded, &Value::from("cjson"), Value::Nil)
        .unwrap();
    runtime
        .table_set(loaded, &Value::from("cjson.safe"), Value::Nil)
        .unwrap();
    runtime.collect_full(crate::runtime::GcRootBuilder::new().finish());
    let value = runtime
        .call_native(decode, &[Value::from("null")])
        .unwrap()
        .remove(0);
    assert!(matches!(value, Value::Userdata(_)));
    runtime.raw_metatable(&value).unwrap();
    let Value::Table(fresh) = runtime.call_native(new, &[]).unwrap()[0] else {
        panic!("fresh module")
    };
    assert_eq!(
        runtime.table_get(fresh, &Value::from("null")).unwrap(),
        value
    );
}

#[test]
fn messagepack_decodes_deep_arrays_without_host_recursion() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cmsgpack");
    let mut bytes = vec![0x91; 1000];
    bytes.push(0xc0);
    let mut value = call(&mut runtime, table, "unpack", &[Value::bytes(bytes)])
        .unwrap()
        .remove(0);
    for _ in 0..1000 {
        let Value::Table(array) = value else {
            panic!("array at each nesting level")
        };
        value = runtime.table_get(array, &runtime.integer_value(1)).unwrap();
    }
    assert_eq!(value, Value::Nil);
    assert!(
        call(
            &mut runtime,
            table,
            "unpack",
            &[Value::bytes(vec![0x91; 1000])]
        )
        .is_err()
    );
}

#[test]
fn cjson_precision_matches_significant_digits_and_exponent_spelling() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    for (number, expected) in [
        (0.00001, "1e-05"),
        (0.0001, "0.0001"),
        (99_999_999_999_999.0, "99999999999999"),
        (99_999_999_999_999.5, "1e+14"),
        (100_000_000_000_000.0, "1e+14"),
    ] {
        assert_eq!(
            call(&mut runtime, table, "encode", &[Value::Number(number)]).unwrap(),
            vec![Value::from(expected)],
            "{number}"
        );
    }
    call(
        &mut runtime,
        table,
        "encode_number_precision",
        &[Value::Number(3.0)],
    )
    .unwrap();
    assert_eq!(
        call(&mut runtime, table, "encode", &[Value::Number(999.5)]).unwrap(),
        vec![Value::from("1e+03")]
    );
}

#[test]
fn cjson_default_depth_limit_allows_1000_levels_and_rejects_1001() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    let source = format!("{}0{}", "[".repeat(1000), "]".repeat(1000));
    let root = call(
        &mut runtime,
        table,
        "decode",
        &[Value::from(source.as_str())],
    )
    .unwrap()
    .remove(0);
    let mut value = root.clone();
    for _ in 0..1000 {
        let Value::Table(array) = value else {
            panic!("array at each nesting level")
        };
        value = runtime.table_get(array, &runtime.integer_value(1)).unwrap();
    }
    assert_eq!(value, Value::Number(0.0));
    assert_eq!(
        call(&mut runtime, table, "encode", std::slice::from_ref(&root)).unwrap(),
        vec![Value::from(source)]
    );
    let too_deep = format!("{}0{}", "[".repeat(1001), "]".repeat(1001));
    assert!(call(&mut runtime, table, "decode", &[Value::from(too_deep)]).is_err());
    let outer = runtime.create_table().unwrap();
    runtime
        .table_set(outer, &runtime.integer_value(1), root)
        .unwrap();
    assert!(call(&mut runtime, table, "encode", &[Value::Table(outer)]).is_err());
}

#[test]
fn cjson_accepts_nan_payload_and_rejects_boolean_object_keys() {
    let (mut runtime, loaded) = setup();
    let table = module(&runtime, loaded, "cjson");
    let safe = module(&runtime, loaded, "cjson.safe");
    assert!(call(&mut runtime, safe, "encode", &[]).is_err());
    assert!(call(&mut runtime, safe, "encode", &[Value::Nil, Value::Nil]).is_err());
    let decoded = call(
        &mut runtime,
        table,
        "decode",
        &[Value::from("nan(payload)")],
    )
    .unwrap();
    assert!(matches!(decoded.as_slice(), [Value::Number(number)] if number.is_nan()));
    assert_eq!(
        call(&mut runtime, table, "decode", &[Value::from("0x")])
            .unwrap_err()
            .to_string(),
        "Expected the end but found invalid token at character 2"
    );
    let object = runtime.create_table().unwrap();
    runtime
        .table_set(object, &Value::Boolean(true), Value::Number(1.0))
        .unwrap();
    let error = call(&mut runtime, table, "encode", &[Value::Table(object)]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Cannot serialise boolean: table key must be a number or string"
    );
}

#[test]
fn struct_integer_fields_convert_through_lua_number() {
    let (mut runtime, loaded) = setup();
    runtime.set_tagged_integers(true);
    let table = module(&runtime, loaded, "struct");
    assert_eq!(
        call(
            &mut runtime,
            table,
            "pack",
            &[Value::from(">I8"), Value::Integer(9_007_199_254_740_993)]
        )
        .unwrap(),
        vec![Value::bytes(vec![0, 0x20, 0, 0, 0, 0, 0, 0])]
    );
}

#[test]
fn bundled_preload_loaders_recreate_modules_after_cache_clear() {
    let mut runtime = Runtime::new();
    crate::stdlib::install_base(&mut runtime).unwrap();
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package table");
    };
    let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap() else {
        panic!("loaded table");
    };
    let Value::Table(preload) = runtime.table_get(package, &Value::from("preload")).unwrap() else {
        panic!("preload table");
    };
    // Nothing is built until a script requires it.
    for name in ["cjson", "cjson.safe", "struct", "cmsgpack", "cmsgpack_safe"] {
        assert_eq!(
            runtime.table_get(loaded, &Value::from(name)).unwrap(),
            Value::Nil
        );
    }
    let Value::NativeFunction(first) = runtime.table_get(preload, &Value::from("cjson")).unwrap()
    else {
        panic!("preload loader for cjson");
    };
    let first = runtime.call_native(first, &[Value::from("cjson")]).unwrap();
    let [Value::Table(cjson)] = first.as_slice() else {
        panic!("module table");
    };
    let cjson = *cjson;
    runtime
        .table_set(loaded, &Value::from("cjson"), Value::Table(cjson))
        .unwrap();
    let null = runtime.table_get(cjson, &Value::from("null")).unwrap();
    call(
        &mut runtime,
        cjson,
        "encode_number_precision",
        &[Value::Number(5.0)],
    )
    .unwrap();
    for name in ["cjson", "cjson.safe", "struct", "cmsgpack", "cmsgpack_safe"] {
        assert_eq!(runtime.get_global(name), Value::Nil);
        runtime
            .table_set(loaded, &Value::from(name), Value::Nil)
            .unwrap();
        let Value::NativeFunction(factory) =
            runtime.table_get(preload, &Value::from(name)).unwrap()
        else {
            panic!("preload loader for {name}");
        };
        let reloaded = runtime.call_native(factory, &[Value::from(name)]).unwrap();
        let [Value::Table(table)] = reloaded.as_slice() else {
            panic!("module table");
        };
        if name == "cjson" {
            assert_ne!(*table, cjson);
            assert_eq!(
                runtime.table_get(*table, &Value::from("null")).unwrap(),
                null
            );
            assert_eq!(
                call(&mut runtime, *table, "encode_number_precision", &[]).unwrap(),
                vec![runtime.integer_value(14)]
            );
        }
    }
}
