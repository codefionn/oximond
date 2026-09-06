use super::*;

#[test]
fn error_nil_retains_its_value_until_the_vm_handles_it() {
    let mut legacy = Runtime::new();
    install_base_with_profile(&mut legacy, CompatibilityProfile::Lua54).unwrap();
    let Value::NativeFunction(error_function) = legacy.get_global("error") else {
        panic!("error is native")
    };
    assert_eq!(
        legacy.call_native(error_function, &[Value::Nil]),
        Err(RuntimeError::RaisedNil)
    );
    let mut modern = Runtime::new();
    install_base_with_profile(&mut modern, CompatibilityProfile::Lua55).unwrap();
    let Value::NativeFunction(error_function) = modern.get_global("error") else {
        panic!("error is native")
    };
    assert_eq!(
        modern.call_native(error_function, &[Value::Nil]),
        Err(RuntimeError::RaisedNil)
    );
}

#[test]
fn lua55_utf8_helpers_preserve_byte_positions() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua55).unwrap();
    let Value::Table(utf8) = runtime.get_global("utf8") else {
        panic!("utf8 is installed");
    };
    let (char, codepoint, len, offset) = {
        let native = |name| match runtime.table_get(utf8, &Value::from(name)).unwrap() {
            Value::NativeFunction(function) => function,
            _ => panic!("utf8.{name} is native"),
        };
        (
            native("char"),
            native("codepoint"),
            native("len"),
            native("offset"),
        )
    };
    assert_eq!(
        runtime
            .call_native(char, &[Value::Number(f64::from(0x20ac))])
            .unwrap(),
        vec![Value::from("€")]
    );
    assert_eq!(
        runtime.call_native(codepoint, &[Value::from("€")]).unwrap(),
        vec![Value::Integer(0x20ac)]
    );
    assert_eq!(
        runtime.call_native(len, &[Value::from("a€")]).unwrap(),
        vec![Value::Integer(2)]
    );
    assert_eq!(
        runtime
            .call_native(
                offset,
                &[Value::from("a€"), Value::Number(1.0), Value::Number(2.0)]
            )
            .unwrap(),
        // Lua 5.5 also returns the character's final byte.
        vec![Value::Integer(2), Value::Integer(4)]
    );
}

#[test]
fn lua55_string_pack_round_trips_portable_numeric_and_string_forms() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua55).unwrap();
    let Value::Table(string) = runtime.get_global("string") else {
        panic!("string is installed");
    };
    let (pack, unpack, packsize) = {
        let native = |name| match runtime.table_get(string, &Value::from(name)).unwrap() {
            Value::NativeFunction(function) => function,
            _ => panic!("string.{name} is native"),
        };
        (native("pack"), native("unpack"), native("packsize"))
    };
    let bytes = runtime
        .call_native(
            pack,
            &[
                Value::from(">!4 b I2 f c3 z s2"),
                Value::Number(-2.0),
                Value::Number(513.0),
                Value::Number(1.5),
                Value::from("xy"),
                Value::from("end"),
                Value::from("hello"),
            ],
        )
        .unwrap()
        .remove(0);
    let values = runtime
        .call_native(unpack, &[Value::from(">!4 b I2 f c3 z s2"), bytes.clone()])
        .unwrap();
    assert_eq!(
        values[..6],
        [
            Value::Integer(-2),
            Value::Integer(513),
            Value::Number(1.5),
            Value::from(vec![b'x', b'y', 0]),
            Value::from("end"),
            Value::from("hello")
        ]
    );
    assert_eq!(
        runtime
            .call_native(packsize, &[Value::from("!4 b I2 f c3")])
            .unwrap(),
        vec![Value::Integer(11)]
    );
    assert!(runtime.call_native(packsize, &[Value::from("z")]).is_err());
}

#[test]
fn math_functions_coerce_byte_strings_and_ignore_unary_extras() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let abs = math_function(&runtime, "abs");
    let ceil = math_function(&runtime, "ceil");
    let cos = math_function(&runtime, "cos");
    let deg = math_function(&runtime, "deg");
    let rad = math_function(&runtime, "rad");
    let sqrt = math_function(&runtime, "sqrt");

    assert_eq!(
        runtime
            .call_native(abs, &[Value::from(" -2.5 "), Value::from("ignored")])
            .unwrap(),
        vec![Value::from(2.5)]
    );
    assert_eq!(
        runtime.call_native(ceil, &[Value::from("-1.2")]).unwrap(),
        vec![Value::from(-1.0)]
    );
    let cosine = runtime
        .call_native(cos, &[Value::from(std::f64::consts::PI)])
        .unwrap();
    let [Value::Number(cosine)] = cosine.as_slice() else {
        panic!("math.cos returns one number");
    };
    assert!((*cosine + 1.0).abs() < f64::EPSILON);
    assert_eq!(
        runtime
            .call_native(deg, &[Value::from(std::f64::consts::PI)])
            .unwrap(),
        vec![Value::from(180.0)]
    );
    assert_eq!(
        runtime.call_native(rad, &[Value::from(180.0)]).unwrap(),
        vec![Value::from(std::f64::consts::PI)]
    );
    assert_eq!(
        runtime.call_native(sqrt, &[Value::from("81")]).unwrap(),
        vec![Value::from(9.0)]
    );
    assert!(runtime.call_native(abs, &[Value::from("no")]).is_err());
    assert!(runtime.call_native(abs, &[]).is_err());
}

#[test]
fn base_installation_exposes_byte_indexed_string_sub() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let (sub, _) = string_functions(&runtime);
    assert_eq!(
        runtime
            .call_native_scalar(
                sub,
                &[Value::from("abcdef"), Value::from(2.0), Value::from(4.0)],
            )
            .unwrap(),
        Some(Value::from("bcd"))
    );
    let cases = [
        (
            vec![Value::from("abcdef"), Value::from(2.0), Value::from(4.0)],
            "bcd",
        ),
        (
            vec![Value::from("abcdef"), Value::from(-3.0), Value::from(-1.0)],
            "def",
        ),
        (
            vec![Value::from("abcdef"), Value::from(-99.0), Value::from(99.0)],
            "abcdef",
        ),
        (vec![Value::from("abcdef"), Value::from(4.0)], "def"),
        (
            vec![Value::from("abcdef"), Value::from(5.0), Value::from(2.0)],
            "",
        ),
    ];
    for (arguments, expected) in cases {
        assert_eq!(
            runtime.call_native(sub, &arguments).unwrap(),
            vec![Value::from(expected)]
        );
    }
    assert_eq!(
        runtime
            .call_native(
                sub,
                &[
                    Value::from(&b"a\xffb"[..]),
                    Value::from(2.0),
                    Value::from(2.0),
                ],
            )
            .unwrap(),
        vec![Value::from(&b"\xff"[..])]
    );
    assert_eq!(
        runtime
            .call_native(
                sub,
                &[
                    Value::from("abcdef"),
                    Value::from(-9_223_372_036_854_775_808.0),
                    Value::from(9_223_372_036_854_775_808.0),
                ],
            )
            .unwrap(),
        // An end index outside the integer range becomes the minimum
        // integer on LuaJIT, as in stock `string.sub("abcdef", -2^63, 2^63)`.
        vec![Value::from("")]
    );
}

#[test]
fn string_rep_validates_count_and_checked_size() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let (_, rep) = string_functions(&runtime);
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from("ab"), Value::from(3.0)])
            .unwrap(),
        vec![Value::from("ababab")]
    );
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from("ab"), Value::from(0.0)])
            .unwrap(),
        vec![Value::from("")]
    );
    // Stock Lua returns the empty string for a negative count.
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from("ab"), Value::from(-1.0)])
            .unwrap(),
        vec![Value::from("")]
    );
    // LuaJIT truncates a fractional count, and a count outside the integer
    // range becomes the minimum integer, which repeats nothing.
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from("ab"), Value::from(1.5)])
            .unwrap(),
        vec![Value::from("ab")]
    );
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from("ab"), Value::Number(f64::MAX)])
            .unwrap(),
        vec![Value::from("")]
    );
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from(&b"\xff"[..]), Value::from(2.0)])
            .unwrap(),
        vec![Value::from(&b"\xff\xff"[..])]
    );
}

#[test]
fn modern_string_rep_inserts_the_optional_separator() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua52).unwrap();
    let rep = string_function(&runtime, "rep");
    assert_eq!(
        runtime
            .call_native(
                rep,
                &[Value::from("ab"), Value::from(3.0), Value::from(".")],
            )
            .unwrap(),
        vec![Value::from("ab.ab.ab")]
    );
    assert_eq!(
        runtime
            .call_native(rep, &[Value::from(""), Value::from(4.0), Value::from(".")],)
            .unwrap(),
        vec![Value::from("...")]
    );
}

#[test]
fn package_loaded_contains_the_installed_standard_libraries() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap() else {
        panic!("package.loaded is a table");
    };
    for name in ["string", "table", "math", "io", "os", "debug", "bit"] {
        assert_eq!(
            runtime.table_get(loaded, &Value::from(name)).unwrap(),
            runtime.get_global(name)
        );
    }
}

#[test]
fn package_exposes_lua51_paths_and_rust_only_loadlib_failure() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    assert!(matches!(
        runtime.table_get(package, &Value::from("cpath")).unwrap(),
        Value::String(_)
    ));
    assert!(matches!(
        runtime.table_get(package, &Value::from("loaders")).unwrap(),
        Value::Table(_)
    ));
    let Value::NativeFunction(loadlib) =
        runtime.table_get(package, &Value::from("loadlib")).unwrap()
    else {
        panic!("package.loadlib is a function");
    };
    let results = runtime
        .call_native(
            loadlib,
            &[Value::from("missing.so"), Value::from("luaopen_x")],
        )
        .unwrap();
    assert!(matches!(
        results.as_slice(),
        [Value::Nil, Value::String(message), Value::String(when)]
            if message.as_bytes().windows(b"Rust-only".len()).any(|part| part == b"Rust-only")
                && when.as_bytes() == b"absent"
    ));
}

#[test]
fn base_installation_exposes_unpack_and_table_getn() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(unpack) = runtime.get_global("unpack") else {
        panic!("unpack is native");
    };
    let Value::Table(table_library) = runtime.get_global("table") else {
        panic!("table is a table");
    };
    let Value::NativeFunction(getn) = runtime
        .table_get(table_library, &Value::from("getn"))
        .unwrap()
    else {
        panic!("table.getn is native");
    };
    let Value::NativeFunction(maxn) = runtime
        .table_get(table_library, &Value::from("maxn"))
        .unwrap()
    else {
        panic!("table.maxn is native");
    };
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from(1.0), Value::from("first"))
        .unwrap();
    runtime
        .table_set(table, &Value::from(3.0), Value::from("third"))
        .unwrap();

    assert_eq!(
        runtime
            .call_native(
                unpack,
                &[Value::Table(table), Value::from(1.0), Value::from(3.0)],
            )
            .unwrap(),
        vec![Value::from("first"), Value::Nil, Value::from("third")]
    );
    assert_eq!(
        runtime.call_native(unpack, &[Value::Table(table)]).unwrap(),
        vec![Value::from("first")]
    );
    assert_eq!(
        runtime.call_native(getn, &[Value::Table(table)]).unwrap(),
        vec![Value::from(1.0)]
    );
    runtime
        .table_set(table, &Value::Number(24.5), Value::Boolean(true))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(-90.0), Value::Boolean(true))
        .unwrap();
    assert_eq!(
        runtime.call_native(maxn, &[Value::Table(table)]).unwrap(),
        vec![Value::Number(24.5)]
    );
    assert!(
        runtime
            .call_native(
                unpack,
                &[
                    Value::Table(table),
                    Value::from(1.0),
                    Value::Number(f64::MAX)
                ],
            )
            .is_err()
    );
}

#[test]
fn string_length_byte_and_char_are_byte_oriented() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let byte = string_function(&runtime, "byte");
    let char = string_function(&runtime, "char");
    let len = string_function(&runtime, "len");

    assert_eq!(
        runtime.call_native(len, &[Value::from(123.0)]).unwrap(),
        vec![Value::from(3.0)]
    );
    assert_eq!(
        runtime
            .call_native_scalar(len, &[Value::from(123.0)])
            .unwrap(),
        Some(Value::from(3.0))
    );
    assert_eq!(
        runtime
            .call_native(byte, &[Value::from(&b"\0\0alo\0x"[..]), Value::from(-1.0)],)
            .unwrap(),
        vec![Value::from(f64::from(b'x'))]
    );
    assert_eq!(
        runtime
            .call_native(
                byte,
                &[Value::from("\n\n"), Value::from(2.0), Value::from(-1.0)],
            )
            .unwrap(),
        vec![Value::from(10.0)]
    );
    assert_eq!(
        runtime
            .call_native(byte, &[Value::from("hi"), Value::from(-3.0)])
            .unwrap(),
        [] as [Value; 0]
    );
    assert_eq!(
        runtime
            .call_native(
                char,
                &[Value::from(0.0), Value::from(255.0), Value::from(0.0)]
            )
            .unwrap(),
        vec![Value::from(&b"\0\xff\0"[..])]
    );
    assert_eq!(
        runtime.call_native(char, &[Value::from(65.9)]).unwrap(),
        vec![Value::from("A")]
    );
    assert_eq!(
        runtime
            .call_native(byte, &[Value::from("abc"), Value::from(1.9)])
            .unwrap(),
        vec![Value::from(f64::from(b'a'))]
    );
    assert!(runtime.call_native(char, &[Value::from(256.0)]).is_err());
}

#[test]
fn string_find_pair_matches_string_find_and_declines_the_rest() {
    use crate::stdlib::strings::{string_find, string_find_pair};
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let plain = Value::Boolean(true);
    let cases: Vec<Vec<Value>> = vec![
        vec![
            Value::from("haystack"),
            Value::from("stack"),
            Value::from(1.0),
            plain.clone(),
        ],
        vec![
            Value::from("haystack"),
            Value::from("none"),
            Value::from(1.0),
            plain.clone(),
        ],
        vec![
            Value::from("haystack"),
            Value::from("hay"),
            Value::from(-3.0),
            plain.clone(),
        ],
        vec![
            Value::from("haystack"),
            Value::from(""),
            Value::from(20.0),
            plain.clone(),
        ],
        vec![
            Value::from("haystack"),
            Value::from(""),
            Value::Nil,
            plain.clone(),
        ],
    ];
    for arguments in &cases {
        let general = string_find(&mut runtime, arguments).unwrap();
        let pair = string_find_pair(&mut runtime, arguments)
            .unwrap()
            .expect("a plain find takes the fixed buffer");
        assert_eq!(pair.as_slice(), general.as_slice(), "{arguments:?}");
    }
    // Patterns, non-strings, and a non-plain find take the general callback.
    for arguments in [
        vec![Value::from("abc"), Value::from("b+")],
        vec![
            Value::from("abc"),
            Value::from("b"),
            Value::from(1.0),
            Value::Nil,
        ],
        vec![Value::from(12.0), Value::from("2"), Value::from(1.0), plain],
        vec![],
    ] {
        assert!(
            string_find_pair(&mut runtime, &arguments)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn string_find_and_ascii_transformations_preserve_bytes() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let find = string_function(&runtime, "find");
    let lower = string_function(&runtime, "lower");
    let reverse = string_function(&runtime, "reverse");
    let upper = string_function(&runtime, "upper");

    assert_eq!(
        runtime
            .call_native(find, &[Value::from("123456789"), Value::from("345")])
            .unwrap(),
        vec![Value::from(3.0), Value::from(5.0)]
    );
    assert_eq!(
        runtime
            .call_native(
                find,
                &[
                    Value::from("1234567890123456789"),
                    Value::from(".45"),
                    Value::from(-9.0),
                ],
            )
            .unwrap(),
        vec![Value::from(13.0), Value::from(15.0)]
    );
    assert_eq!(
        runtime
            .call_native(
                find,
                &[
                    Value::from("alo(.)alo"),
                    Value::from("(.)"),
                    Value::from(1.0),
                    Value::from(true),
                ],
            )
            .unwrap(),
        vec![Value::from(4.0), Value::from(6.0)]
    );
    assert_eq!(
        runtime
            .call_native(find, &[Value::from(""), Value::from("")])
            .unwrap(),
        vec![Value::from(1.0), Value::from(0.0)]
    );
    assert_eq!(
        runtime
            .call_native(find, &[Value::from(""), Value::from(""), Value::from(2.0)],)
            .unwrap(),
        // The default profile is LuaJIT, which starts a search past the end
        // at the end, as `luajit` prints `1 0` here.
        vec![Value::from(1.0), Value::from(0.0)]
    );
    assert_eq!(
        runtime
            .call_native(find, &[Value::from("abc"), Value::from("missing")])
            .unwrap(),
        vec![Value::Nil]
    );
    assert_eq!(
        runtime
            .call_native(find, &[Value::from("abc"), Value::from("a+")])
            .unwrap(),
        vec![Value::from(1.0), Value::from(1.0)]
    );

    assert_eq!(
        runtime
            .call_native(reverse, &[Value::from(&b"\0\x01\x02\x03"[..])])
            .unwrap(),
        vec![Value::from(&b"\x03\x02\x01\0"[..])]
    );
    assert_eq!(
        runtime
            .call_native(upper, &[Value::from(&b"ab\0c\xff"[..])])
            .unwrap(),
        vec![Value::from(&b"AB\0C\xff"[..])]
    );
    assert_eq!(
        runtime
            .call_native(lower, &[Value::from("\0ABCc%$")])
            .unwrap(),
        vec![Value::from("\0abcc%$")]
    );
}

#[test]
fn string_find_plain_search_handles_binary_needles_and_offsets() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let find = string_function(&runtime, "find");
    let source = Value::from(&b"\xff\0needle\0needle\x80"[..]);
    let pattern = Value::from(&b"\0needle"[..]);

    assert_eq!(
        runtime
            .call_native(
                find,
                &[
                    source.clone(),
                    pattern.clone(),
                    Value::from(1.0),
                    Value::from(true),
                ],
            )
            .unwrap(),
        vec![Value::from(2.0), Value::from(8.0)]
    );
    assert_eq!(
        runtime
            .call_native(
                find,
                &[
                    source.clone(),
                    pattern.clone(),
                    Value::from(3.0),
                    Value::from(true),
                ],
            )
            .unwrap(),
        vec![Value::from(9.0), Value::from(15.0)]
    );
    assert_eq!(
        runtime
            .call_native(
                find,
                &[
                    source.clone(),
                    pattern,
                    Value::from(10.0),
                    Value::from(true),
                ],
            )
            .unwrap(),
        vec![Value::Nil]
    );
    assert_eq!(
        runtime
            .call_native(
                find,
                &[source, Value::from(""), Value::from(4.0), Value::from(true)],
            )
            .unwrap(),
        vec![Value::from(4.0), Value::from(3.0)]
    );
}

#[test]
fn gmatch_and_gfind_are_the_same_native_function_and_allocate_state() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(string) = runtime.get_global("string") else {
        panic!("string is a table");
    };
    let gmatch = runtime.table_get(string, &Value::from("gmatch")).unwrap();
    let gfind = runtime.table_get(string, &Value::from("gfind")).unwrap();
    assert_eq!(gmatch, gfind);
    let Value::NativeFunction(gmatch) = gmatch else {
        panic!("string.gmatch is native");
    };

    let values = runtime
        .call_native(gmatch, &[Value::from("one two"), Value::from("%a+")])
        .unwrap();
    let [Value::GmatchIterator(iterator)] = values.as_slice() else {
        panic!("string.gmatch returns its callable iterator");
    };
    let state = runtime.gmatch_iterator_snapshot(*iterator).unwrap();
    assert_eq!(state.source().as_bytes(), b"one two");
    assert_eq!(state.pattern().as_bytes(), b"%a+");
    assert_eq!(state.search(), 0);
}

#[test]
fn string_find_and_match_return_pattern_captures() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let find = string_function(&runtime, "find");
    let Value::NativeFunction(r#match) = runtime
        .table_get(
            match runtime.get_global("string") {
                Value::Table(table) => table,
                _ => unreachable!(),
            },
            &Value::from("match"),
        )
        .unwrap()
    else {
        panic!("string.match is native");
    };
    assert_eq!(
        runtime
            .call_native(
                find,
                &[Value::from("abc=xyz"), Value::from("(%w*)(%p)(%w+)")]
            )
            .unwrap(),
        vec![
            Value::from(1.0),
            Value::from(7.0),
            Value::from("abc"),
            Value::from("="),
            Value::from("xyz")
        ]
    );
    assert_eq!(
        runtime
            .call_native(
                r#match,
                &[Value::from("abc=xyz"), Value::from("(%w*)(%p)(%w+)")]
            )
            .unwrap(),
        vec![Value::from("abc"), Value::from("="), Value::from("xyz")]
    );
}

#[test]
fn install_base_registers_the_vm_tostring_builtin() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    assert_eq!(
        runtime.get_global("tostring"),
        Value::BuiltinFunction(BuiltinFunction::ToString)
    );
}

#[test]
fn os_date_uses_lua_calendar_fields_and_fixed_c_formatting() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(date) = runtime.table_get(os, &Value::from("date")).unwrap() else {
        panic!("os.date is native");
    };
    let timestamp = Value::Number(946_684_800.0); // 2000-01-01 00:00:00 UTC.
    assert_eq!(
        runtime
            .call_native(
                date,
                &[
                    Value::from("!%Y-%m-%d %H:%M:%S %w %u %j %z %Z"),
                    timestamp.clone()
                ]
            )
            .unwrap(),
        vec![Value::from("2000-01-01 00:00:00 6 6 001 +0000 UTC")]
    );
    let Value::Table(fields) = runtime
        .call_native(date, &[Value::from("!*t"), timestamp])
        .unwrap()
        .remove(0)
    else {
        panic!("os.date !*t returns a table");
    };
    assert_eq!(
        runtime.table_get(fields, &Value::from("year")).unwrap(),
        Value::Number(2000.0)
    );
    assert_eq!(
        runtime.table_get(fields, &Value::from("wday")).unwrap(),
        Value::Number(7.0)
    );
    assert_eq!(
        runtime.table_get(fields, &Value::from("yday")).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(
        runtime.table_get(fields, &Value::from("isdst")).unwrap(),
        Value::Boolean(false)
    );
}

#[test]
fn string_format_preserves_bytes_and_supports_early_lua_forms() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let format = string_function(&runtime, "format");
    let value = Value::from(&b"\"\xfflo\"\n\\"[..]);
    let mut expected = b"\"\\\"\xfflo\\\"\\\n\\\\\"".to_vec();
    expected.extend_from_slice(&b"\"\xfflo\"\n\\"[..]);
    assert_eq!(
        runtime
            .call_native(format, &[Value::from("%q%s"), value.clone(), value])
            .unwrap(),
        vec![Value::from(expected)]
    );
    assert_eq!(
        runtime
            .call_native(format, &[Value::from("%q"), Value::from(&b"\0"[..])])
            .unwrap(),
        // The default profile is LuaJIT, which writes `\0` before a non-digit.
        vec![Value::from(&b"\"\\0\""[..])]
    );
    assert_eq!(
        runtime
            .call_native(
                format,
                &[
                    Value::from(&b"\0%c\0%c%x\0"[..]),
                    Value::from(255.0),
                    Value::from(f64::from(b'b')),
                    Value::from(140.0),
                ],
            )
            .unwrap(),
        vec![Value::from(&b"\0\xff\0b8c\0"[..])]
    );
    assert_eq!(
        runtime
            .call_native(
                format,
                &[
                    Value::from("%%%d %010d"),
                    Value::from(10.0),
                    Value::from(23.0)
                ],
            )
            .unwrap(),
        vec![Value::from("%10 0000000023")]
    );
    assert_eq!(
        runtime
            .call_native(
                format,
                &[
                    Value::from("%+d|% i|%u|%#o|%#x|%#X|%.2e|%.2E|%.4g|%.4G"),
                    Value::from(12.0),
                    Value::from(12.0),
                    Value::from(12.0),
                    Value::from(8.0),
                    Value::from(140.0),
                    Value::from(140.0),
                    Value::from(12.3),
                    Value::from(12.3),
                    Value::from(12.3),
                    Value::from(12.3),
                ],
            )
            .unwrap(),
        vec![Value::from(
            "+12| 12|12|010|0x8c|0X8C|1.23e+01|1.23E+01|12.3|12.3"
        )]
    );
    let padded = runtime
        .call_native(format, &[Value::from("\"%-50s\""), Value::from("a")])
        .unwrap();
    let [Value::String(padded)] = padded.as_slice() else {
        panic!("string.format returns one string");
    };
    assert_eq!(padded.as_bytes().len(), 52);
    assert_eq!(&padded.as_bytes()[..4], b"\"a  ");
    assert!(
        runtime
            .call_native(format, &[Value::from("%p"), Value::from(1.0)])
            .is_err()
    );
}

#[test]
fn modern_string_format_quotes_literals_and_formats_pointers() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua55).unwrap();
    assert_eq!(
        crate::stdlib::strings::string_format_with_pointer(
            &mut runtime,
            &[Value::from("%q"), Value::from(&b"\0"[..])],
        )
        .unwrap(),
        vec![Value::from(&b"\"\\0\""[..])]
    );
    for (value, expected) in [
        (Value::Nil, "nil"),
        (Value::Boolean(true), "true"),
        (Value::Boolean(false), "false"),
    ] {
        assert_eq!(
            crate::stdlib::strings::string_format_with_pointer(
                &mut runtime,
                &[Value::from("%q"), value]
            )
            .unwrap(),
            vec![Value::from(expected)]
        );
    }
    let table = Value::Table(runtime.create_table().unwrap());
    let first = crate::stdlib::strings::string_format_with_pointer(
        &mut runtime,
        &[Value::from("%p"), table.clone()],
    )
    .unwrap();
    let second = crate::stdlib::strings::string_format_with_pointer(
        &mut runtime,
        &[Value::from("%p"), table],
    )
    .unwrap();
    assert_eq!(first, second);
    assert_ne!(first, vec![Value::from("(null)")]);
}

fn format_in(
    profile: CompatibilityProfile,
    arguments: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, profile).unwrap();
    match profile {
        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21 => {
            crate::stdlib::strings::string_format(&mut runtime, arguments)
        }
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53 => {
            crate::stdlib::strings::string_format_modern(&mut runtime, arguments)
        }
        _ => crate::stdlib::strings::string_format_with_pointer(&mut runtime, arguments),
    }
}

#[test]
fn format_q_escapes_follow_the_stock_profile() {
    let input = Value::from(&b"a\tb\x01c\x012\r\n\0\x7f\xc8\x019"[..]);
    let legacy = b"\"a\tb\x01c\x012\\r\\\n\\000\x7f\xc8\x019\"".to_vec();
    let modern = b"\"a\\9b\\1c\\0012\\13\\\n\\0\\127\xc8\\0019\"".to_vec();
    for (profile, expected) in [
        (CompatibilityProfile::Lua51, &legacy),
        (CompatibilityProfile::Lua52, &modern),
        (CompatibilityProfile::Lua53, &modern),
        (CompatibilityProfile::Lua54, &modern),
        (CompatibilityProfile::Lua55, &modern),
        (CompatibilityProfile::LuaJit21, &modern),
    ] {
        assert_eq!(
            format_in(profile, &[Value::from("%q"), input.clone()]).unwrap(),
            vec![Value::from(expected.clone())],
            "{profile}"
        );
    }
}

#[test]
fn format_d_needs_an_integer_value_from_lua53() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        let error = format_in(profile, &[Value::from("%d"), Value::Number(1.5)]).unwrap_err();
        assert_eq!(
            error.argument_detail(),
            Some((2, "number has no integer representation".to_string()))
        );
        assert!(format_in(profile, &[Value::from("%d"), Value::from("1.5")]).is_err());
        assert!(
            format_in(
                profile,
                &[Value::from("%x"), Value::Number(2.0_f64.powi(63))]
            )
            .is_err()
        );
        assert_eq!(
            format_in(profile, &[Value::from("%d"), Value::Number(2.0)]).unwrap(),
            vec![Value::from("2")]
        );
    }
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        assert_eq!(
            format_in(profile, &[Value::from("%d"), Value::Number(1.5)]).unwrap(),
            vec![Value::from("1")]
        );
    }
}

#[test]
fn format_c_of_zero_is_dropped_only_by_lua51() {
    let arguments = [Value::from("a%cb"), Value::Number(0.0)];
    assert_eq!(
        format_in(CompatibilityProfile::Lua51, &arguments).unwrap(),
        vec![Value::from("ab")]
    );
    for profile in [
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        assert_eq!(
            format_in(profile, &arguments).unwrap(),
            vec![Value::from(&b"a\0b"[..])],
            "{profile}"
        );
    }
}

#[test]
fn rep_separator_is_ignored_by_stock_lua51_only() {
    for (profile, expected) in [
        (CompatibilityProfile::Lua51, "ababab"),
        (CompatibilityProfile::LuaJit21, "ab,ab,ab"),
        (CompatibilityProfile::Lua54, "ab,ab,ab"),
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let rep = string_function(&runtime, "rep");
        assert_eq!(
            runtime
                .call_native(
                    rep,
                    &[Value::from("ab"), Value::Number(3.0), Value::from(",")]
                )
                .unwrap(),
            vec![Value::from(expected)],
            "{profile}"
        );
    }
}

#[test]
fn byte_index_conversion_follows_the_profile() {
    for (profile, huge_float_fails) in [
        (CompatibilityProfile::Lua51, false),
        (CompatibilityProfile::LuaJit21, false),
        (CompatibilityProfile::Lua54, true),
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let byte = string_function(&runtime, "byte");
        for index in [i64::MAX, i64::MIN] {
            let arguments = [Value::from("abc"), runtime.integer_value(index)];
            assert_eq!(runtime.call_native(byte, &arguments).unwrap(), vec![]);
        }
        let huge = [Value::from("abc"), Value::Number(2.0_f64.powi(63))];
        assert_eq!(
            runtime.call_native(byte, &huge).is_err(),
            huge_float_fails,
            "{profile}"
        );
    }
}

#[test]
fn char_range_message_follows_the_profile() {
    for (profile, message) in [
        (CompatibilityProfile::Lua51, "invalid value"),
        (CompatibilityProfile::LuaJit21, "invalid value"),
        (CompatibilityProfile::Lua54, "value out of range"),
        (CompatibilityProfile::Lua55, "value out of range"),
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let char = string_function(&runtime, "char");
        for byte in [256.0, -1.0] {
            let error = runtime
                .call_native(char, &[Value::Number(65.0), Value::Number(byte)])
                .unwrap_err();
            let (position, detail) = error.argument_detail().expect("an argument error");
            assert_eq!((position, detail.as_str()), (2, message), "{profile}");
        }
    }
}

#[test]
fn luajit_string_indices_wrap_to_int_min_outside_32_bits() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::LuaJit21).unwrap();
    let (sub, _) = string_functions(&runtime);
    for (start, end, expected) in [
        (2.0_f64.powi(53), 229.0, "ab"),
        (2.0_f64.powi(31), 5.0, "ab"),
        (2.0_f64.powi(31) - 1.0, 5.0, ""),
        (-(2.0_f64.powi(53)), 2.0, "ab"),
        (1.0, 2.0_f64.powi(32), ""),
        (1.0, f64::NAN, ""),
    ] {
        let arguments = [Value::from("ab"), Value::Number(start), Value::Number(end)];
        assert_eq!(
            runtime.call_native_scalar(sub, &arguments).unwrap(),
            Some(Value::from(expected)),
            "{start} {end}"
        );
    }
}

#[test]
fn utf8_profile_rules_ranges_and_stateless_iterators() {
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        let Value::Table(table) = runtime.get_global("utf8") else {
            panic!("UTF-8 table");
        };
        let lookup = |name| match runtime.table_get(table, &Value::from(name)).unwrap() {
            Value::NativeFunction(handle) => handle,
            _ => panic!("UTF-8 function"),
        };
        let char = lookup("char");
        let codepoint = lookup("codepoint");
        let len = lookup("len");
        let codes = lookup("codes");
        let offset = lookup("offset");
        let surrogate = Value::bytes(vec![0xed, 0xa0, 0x80]);
        assert_eq!(
            runtime
                .call_native(char, &[Value::Integer(0xd800)])
                .unwrap(),
            vec![surrogate.clone()]
        );
        assert_eq!(
            runtime
                .call_native(len, std::slice::from_ref(&surrogate))
                .unwrap(),
            if profile == CompatibilityProfile::Lua53 {
                vec![Value::Integer(1)]
            } else {
                vec![Value::Nil, Value::Integer(1)]
            },
        );
        assert!(
            runtime
                .call_native(
                    codepoint,
                    &[
                        surrogate,
                        Value::Integer(1),
                        Value::Integer(1),
                        Value::Boolean(true)
                    ]
                )
                .is_ok(),
        );
        let extended = runtime.call_native(char, &[Value::Integer(0x7fff_ffff)]);
        if profile == CompatibilityProfile::Lua53 {
            assert!(extended.is_err());
        } else {
            let extended = extended.unwrap().remove(0);
            assert_eq!(
                extended,
                Value::bytes(vec![0xfd, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf])
            );
            assert_eq!(
                runtime
                    .call_native(
                        codepoint,
                        &[
                            extended.clone(),
                            Value::Integer(1),
                            Value::Integer(1),
                            Value::Boolean(true)
                        ]
                    )
                    .unwrap(),
                vec![Value::Integer(0x7fff_ffff)]
            );
            assert_eq!(
                runtime
                    .call_native(len, std::slice::from_ref(&extended))
                    .unwrap(),
                vec![Value::Nil, Value::Integer(1)]
            );
            assert_eq!(
                runtime
                    .call_native(
                        len,
                        &[
                            extended,
                            Value::Integer(1),
                            Value::Integer(1),
                            Value::Boolean(true)
                        ]
                    )
                    .unwrap(),
                vec![Value::Integer(1)]
            );
        }
        // The final index limits character starts, not all bytes of each character.
        assert_eq!(
            runtime
                .call_native(
                    len,
                    &[Value::from("€"), Value::Integer(1), Value::Integer(1)]
                )
                .unwrap(),
            vec![Value::Integer(1)]
        );
        assert_eq!(
            runtime
                .call_native(
                    codepoint,
                    &[Value::bytes(vec![b'a', 0xff]), Value::Integer(1)]
                )
                .unwrap(),
            vec![Value::Integer(97)]
        );
        assert!(
            runtime
                .call_native(len, &[Value::from("abc"), Value::Integer(0)])
                .is_err()
        );
        assert!(
            runtime
                .call_native(
                    codepoint,
                    &[Value::from("abc"), Value::Integer(1), Value::Integer(4)]
                )
                .is_err()
        );
        assert!(
            runtime
                .call_native(
                    offset,
                    &[Value::from("abc"), Value::Integer(1), Value::Integer(0)]
                )
                .is_err()
        );
        let bad = Value::bytes(vec![b'a', 0xff]);
        assert_eq!(
            runtime
                .call_native(offset, &[bad.clone(), Value::Integer(2)])
                .unwrap()[0],
            Value::Integer(2)
        );
        let iterator = runtime
            .call_native(codes, std::slice::from_ref(&bad))
            .unwrap();
        assert_eq!(iterator.len(), 3);
        assert_eq!(iterator[1], bad);
        assert_eq!(iterator[2], Value::Integer(0));
        assert_eq!(
            runtime.call_native(codes, &[Value::from("other")]).unwrap()[0],
            iterator[0]
        );
        let Value::NativeFunction(next) = iterator[0] else {
            panic!("native iterator");
        };
        assert_eq!(
            runtime
                .call_native(next, &[iterator[1].clone(), Value::Integer(0)])
                .unwrap(),
            vec![Value::Integer(1), Value::Integer(97)]
        );
        assert!(
            runtime
                .call_native(next, &[iterator[1].clone(), Value::Integer(1)])
                .is_err()
        );
        // State and control arguments determine each call, including repeated calls.
        assert_eq!(
            runtime
                .call_native(next, &[Value::from("a€"), Value::Integer(1)])
                .unwrap(),
            vec![Value::Integer(2), Value::Integer(0x20ac)]
        );
        assert_eq!(
            runtime
                .call_native(next, &[Value::from("a€"), Value::Integer(1)])
                .unwrap(),
            vec![Value::Integer(2), Value::Integer(0x20ac)]
        );
        assert_eq!(
            runtime
                .call_native(next, &[Value::from("a€"), Value::Integer(4)])
                .unwrap(),
            [] as [Value; 0]
        );
    }
}

/// The `utf8` natives of `profile`, by name.
fn utf8_natives(profile: CompatibilityProfile) -> (Runtime, [NativeFunctionHandle; 5]) {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, profile).unwrap();
    let Value::Table(utf8) = runtime.get_global("utf8") else {
        panic!("utf8 is installed");
    };
    let natives = ["char", "len", "codepoint", "offset", "codes"].map(|name| {
        match runtime.table_get(utf8, &Value::from(name)).unwrap() {
            Value::NativeFunction(function) => function,
            _ => panic!("utf8.{name} is native"),
        }
    });
    (runtime, natives)
}

/// Strings of every shape the decoder tells apart.
const UTF8_SUBJECTS: [&[u8]; 16] = [
    b"",
    b"abc",
    b"h\xc3\xa9llo",
    b"\xe2\x82\xac\xf0\x9f\x98\x80z",
    b"a\xffb",
    b"\xe2\x82",
    b"\x80",
    b"\xed\xa0\x80",
    b"\xf4\x90\x80\x80",
    b"\xc0\x80",
    b"\xfd\xbf\xbf\xbf\xbf\xbf",
    b"\xf8\x88\x80\x80\x80",
    b"ab\xc3",
    b"\xc3\xa9\xc3\xa9\xc3\xa9",
    b"a\x80\x80b",
    b"\xf0\x9f\x98\x80",
];

#[test]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn the_utf8_cores_agree_with_the_natives() {
    use crate::string::utf8::{Len, Offset, Utf8Mode, codepoints, len, offset};
    let positions: Vec<Option<i64>> = std::iter::once(None)
        .chain((-14..=14).map(Some))
        .chain([Some(i64::MAX), Some(i64::MIN)])
        .collect();
    let argument = |position: Option<i64>| position.map_or(Value::Nil, Value::Integer);
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (mut runtime, [_, length, codepoint, offsets, _]) = utf8_natives(profile);
        let lua53 = profile == CompatibilityProfile::Lua53;
        for subject in UTF8_SUBJECTS {
            let text = Value::bytes(subject.to_vec());
            for &first in &positions {
                for &last in &positions {
                    for lax in [false, true] {
                        if lax && lua53 {
                            continue;
                        }
                        let mode = if lua53 {
                            Utf8Mode::Lua53
                        } else if lax {
                            Utf8Mode::Lax
                        } else {
                            Utf8Mode::Strict
                        };
                        let mut arguments = vec![text.clone(), argument(first), argument(last)];
                        if lax {
                            arguments.push(Value::Boolean(true));
                        }
                        let case = format!("{profile:?} {subject:?} {first:?} {last:?} {lax}");
                        // utf8.len
                        let native = runtime.call_native(length, &arguments).ok();
                        match (len(subject, first, last, mode), native.as_deref()) {
                            (Some(Len::Count(count)), Some([Value::Integer(native)])) => {
                                assert_eq!(count, *native, "len {case}");
                            }
                            (
                                Some(Len::Invalid(at)),
                                Some([Value::Nil, Value::Integer(native)]),
                            ) => {
                                assert_eq!(at as i64, *native, "len {case}");
                            }
                            (None, None) => {}
                            (core, native) => panic!("len {case}: {core:?} against {native:?}"),
                        }
                        // utf8.codepoint
                        let native = runtime.call_native(codepoint, &arguments).ok();
                        let mut codes = [0_u32; 64];
                        match (codepoints(subject, first, last, mode, &mut codes), native) {
                            (Some(count), Some(native)) => {
                                let wanted: Vec<_> = codes[..count]
                                    .iter()
                                    .map(|code| Value::Integer(i64::from(*code)))
                                    .collect();
                                assert_eq!(wanted, native, "codepoint {case}");
                            }
                            (None, None) => {}
                            // More code points than the buffer holds.
                            (None, Some(native)) => {
                                assert!(native.len() > 64, "codepoint {case}: {native:?}");
                            }
                            (Some(_), None) => panic!("codepoint {case}: core answers an error"),
                        }
                    }
                }
            }
            // utf8.offset
            for &shift in &positions[1..] {
                for &position in &positions {
                    let arguments = [text.clone(), argument(shift), argument(position)];
                    let case = format!("{profile:?} {subject:?} {shift:?} {position:?}");
                    let native = runtime.call_native(offsets, &arguments).ok();
                    let core = offset(subject, shift.unwrap(), position);
                    match (core, native.as_deref()) {
                        (Some(Offset::Absent), Some([Value::Nil])) => {}
                        (
                            Some(Offset::Found { start, end }),
                            Some([Value::Integer(native_start), rest @ ..]),
                        ) => {
                            assert_eq!(start as i64, *native_start, "offset {case}");
                            if profile == CompatibilityProfile::Lua55 {
                                assert_eq!(rest, [Value::Integer(end as i64)], "offset {case}");
                            }
                        }
                        // The end of Lua 5.5 raises where the others do not.
                        (None, _) => {
                            assert!(
                                native.is_none() || profile != CompatibilityProfile::Lua55,
                                "offset {case}: {native:?}"
                            );
                        }
                        (core, native) => panic!("offset {case}: {core:?} against {native:?}"),
                    }
                }
            }
        }
    }
}

#[test]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn the_utf8_encoder_and_the_codes_step_agree_with_the_natives() {
    use crate::string::utf8::{CodesStep, Utf8Mode, codes_step, encode};
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let (mut runtime, [char, _, _, _, codes]) = utf8_natives(profile);
        let lua53 = profile == CompatibilityProfile::Lua53;
        let maximum = if lua53 { 0x10_ffff } else { 0x7fff_ffff };
        for code in (-3..0x900)
            .chain([
                0xd7ff, 0xd800, 0xdfff, 0xe000, 0xffff, 0x10000, 0x10_ffff, 0x11_0000,
            ])
            .chain([
                0x1f_ffff,
                0x20_0000,
                0x3ff_ffff,
                0x400_0000,
                0x7fff_ffff,
                0x8000_0000,
            ])
        {
            let native = runtime.call_native(char, &[Value::Integer(code)]).ok();
            let mut bytes = [0_u8; 8];
            let core = encode(code, maximum, &mut bytes).map(|count| bytes[..count].to_vec());
            match (core, native.as_deref()) {
                (Some(core), Some([Value::String(native)])) => {
                    assert_eq!(core, native.as_bytes(), "{profile:?} {code:#x}");
                }
                (None, None) => {}
                (core, native) => panic!("{profile:?} {code:#x}: {core:?} against {native:?}"),
            }
        }
        // The iterator the call gives.
        let Value::Table(utf8) = runtime.get_global("utf8") else {
            panic!("utf8 is installed");
        };
        let _ = (utf8, codes);
        for lax in [false, true] {
            if lax && lua53 {
                continue;
            }
            let step = {
                let results = runtime
                    .call_native(codes, &[Value::from("a"), Value::Boolean(lax)])
                    .unwrap();
                match &results[0] {
                    Value::NativeFunction(function) => *function,
                    other => panic!("utf8.codes gives {other:?}"),
                }
            };
            let mode = if lua53 {
                Utf8Mode::Lua53
            } else if lax {
                Utf8Mode::Lax
            } else {
                Utf8Mode::Strict
            };
            for subject in UTF8_SUBJECTS {
                for index in -3..=14 {
                    let arguments = [Value::bytes(subject.to_vec()), Value::Integer(index)];
                    let native = runtime.call_native(step, &arguments).ok();
                    let case = format!("{profile:?} {subject:?} {index} {lax}");
                    match (codes_step(subject, index as isize, mode), native.as_deref()) {
                        (CodesStep::End, Some([])) | (CodesStep::Invalid, None) => {}
                        (
                            CodesStep::Character { position, code },
                            Some([Value::Integer(at), Value::Integer(native)]),
                        ) => {
                            assert_eq!(
                                (position as i64, i64::from(code)),
                                (*at, *native),
                                "{case}"
                            );
                        }
                        (_, native) => panic!("codes {case}: against {native:?}"),
                    }
                }
            }
        }
    }
}

/// The native `library.name` of a runtime with the profile installed.
fn library_native(runtime: &Runtime, library: &str, name: &str) -> NativeFunctionHandle {
    let Value::Table(table) = runtime.get_global(library) else {
        panic!("{library} is installed");
    };
    match runtime.table_get(table, &Value::from(name)).unwrap() {
        Value::NativeFunction(function) => function,
        _ => panic!("{library}.{name} is native"),
    }
}

#[test]
fn fixed_results_probes_agree_with_the_results_probes_and_the_natives() {
    use crate::runtime::NativeResults;
    let text = Value::from("a\u{e9}\u{20ac}x\u{1f600}z");
    let long = Value::from("abcdefghij");
    let record = Value::bytes(b"\x07\x00\x00\x00\x02\x00\x01\x02ab\x03cde".to_vec());
    let calls: Vec<(&str, &str, Vec<Value>)> = vec![
        (
            "string",
            "pack",
            vec!["<i4 s1".into(), Value::Integer(7), "ab".into()],
        ),
        ("string", "pack", vec!["<i2".into(), Value::Integer(70000)]),
        (
            "string",
            "pack",
            vec!["<i2".into(), Value::Integer(1), Value::Integer(2)],
        ),
        ("string", "pack", vec!["<c100".into(), "x".into()]),
        ("string", "pack", vec!["<s1".into(), "x".repeat(300).into()]),
        (
            "string",
            "pack",
            vec![
                "<!8 i1 d i1".into(),
                Value::Integer(1),
                2.5.into(),
                Value::Integer(3),
            ],
        ),
        ("string", "unpack", vec!["<i4 i2".into(), record.clone()]),
        (
            "string",
            "unpack",
            vec!["<i4 i2 i1 i1".into(), record.clone()],
        ),
        (
            "string",
            "unpack",
            vec!["<i4 i2 i1 i1 s1 s1".into(), record.clone()],
        ),
        (
            "string",
            "unpack",
            vec!["<i4 i2 i1 i1 s1 s1 i1".into(), record.clone()],
        ),
        (
            "string",
            "unpack",
            vec!["<i4".into(), record.clone(), Value::Integer(99)],
        ),
        (
            "string",
            "unpack",
            vec!["<i4".into(), record.clone(), Value::Integer(-4)],
        ),
        (
            "string",
            "unpack",
            vec!["<i8".into(), record.clone(), Value::Integer(10)],
        ),
        ("utf8", "char", vec![]),
        (
            "utf8",
            "char",
            vec![
                Value::Integer(72),
                Value::Integer(0x20ac),
                Value::Integer(0x1f600),
            ],
        ),
        ("utf8", "char", vec![Value::Integer(-1)]),
        ("utf8", "codepoint", vec![text.clone()]),
        (
            "utf8",
            "codepoint",
            vec![text.clone(), Value::Integer(1), Value::Integer(4)],
        ),
        (
            "utf8",
            "codepoint",
            vec![text.clone(), Value::Integer(3), Value::Integer(2)],
        ),
        (
            "utf8",
            "codepoint",
            vec![text.clone(), Value::Integer(1), Value::Integer(99)],
        ),
        (
            "utf8",
            "codepoint",
            vec![long.clone(), Value::Integer(1), Value::Integer(6)],
        ),
        (
            "utf8",
            "codepoint",
            vec![long.clone(), Value::Integer(1), Value::Integer(7)],
        ),
        ("utf8", "len", vec![text.clone()]),
        ("utf8", "len", vec!["a\u{e9}".into(), Value::Integer(3)]),
        ("utf8", "len", vec![Value::bytes(b"a\xffb".to_vec())]),
        ("utf8", "offset", vec![text.clone(), Value::Integer(-1)]),
        ("utf8", "offset", vec![text.clone(), Value::Integer(9)]),
        (
            "utf8",
            "offset",
            vec![text.clone(), Value::Integer(2), Value::Integer(3)],
        ),
    ];
    for profile in [
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let mut runtime = Runtime::new();
        install_base_with_profile(&mut runtime, profile).unwrap();
        for (library, name, arguments) in &calls {
            let function = library_native(&runtime, library, name);
            let expected = runtime.probe_native_results(function, arguments);
            let mut fixed = NativeResults::new();
            let found = runtime.probe_native_results_fixed(function, arguments, &mut fixed);
            let label = format!("{profile:?} {library}.{name} {arguments:?}");
            match expected {
                // An error, or more results than a list holds: no answer.
                None => assert!(!found, "{label}"),
                Some(values) if values.len() > crate::runtime::NATIVE_RESULTS_CAPACITY => {
                    assert!(!found, "{label}");
                }
                Some(values) => {
                    assert!(found, "{label}");
                    assert_eq!(fixed.as_slice(), values.as_slice(), "{label}");
                }
            }
        }
    }
}

#[test]
fn string_pack_builds_records_on_both_sides_of_its_inline_buffer() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    let pack = library_native(&runtime, "string", "pack");
    let unpack = library_native(&runtime, "string", "unpack");
    for size in (0..=60).chain([100, 255]) {
        let payload = "x".repeat(size);
        let format = Value::from(format!("<i2 c{size} i2"));
        let arguments = [
            format.clone(),
            Value::Integer(5),
            payload.clone().into(),
            Value::Integer(9),
        ];
        let packed = runtime.probe_native_results(pack, &arguments).unwrap();
        let [Value::String(record)] = packed.as_slice() else {
            panic!("one string for {size}");
        };
        let mut expected = vec![5, 0];
        expected.extend(payload.bytes());
        expected.extend([9, 0]);
        assert_eq!(record.as_bytes(), expected.as_slice(), "{size}");
        let again = runtime
            .probe_native_results(unpack, &[format, Value::String(record.clone())])
            .unwrap();
        assert_eq!(
            again,
            vec![
                Value::Integer(5),
                Value::from(payload),
                Value::Integer(9),
                Value::Integer(i64::try_from(expected.len()).unwrap() + 1)
            ]
        );
    }
}

#[test]
fn a_prepared_format_matches_the_generic_formatter() {
    let runtime = Runtime::new();
    let numbers = runtime.number_format();
    let numerals = runtime.nonfinite_numerals();
    let cases: [(&[u8], Vec<Value>); 6] = [
        (
            b"%d|%5s|%-6.2f|%x",
            vec![
                Value::Integer(7),
                "ab".into(),
                1.5.into(),
                Value::Integer(255),
            ],
        ),
        (b"100%% %s", vec!["x".into()]),
        (b"plain", vec![]),
        (b"%c%c", vec![Value::Integer(72), Value::Integer(105)]),
        (b"%5.1f", vec!["2.25".into()]),
        (b"%d %d", vec![Value::Integer(1), Value::Integer(2)]),
    ];
    for (format, rest) in cases {
        let mut arguments = vec![Value::from(format)];
        arguments.extend(rest);
        let plan = crate::stdlib::PreparedFormat::new(format).unwrap();
        let expected =
            crate::stdlib::strings::string_format_inner(&arguments, false, true, numbers, numerals)
                .unwrap();
        assert_eq!(
            plan.format(&arguments, true, numbers, numerals),
            Some(expected[0].clone())
        );
    }
    // A call that would raise is declined, and a bad format does not prepare.
    let plan = crate::stdlib::PreparedFormat::new(b"%d").unwrap();
    assert_eq!(
        plan.format(
            &[Value::from("%d"), Value::from("x")],
            true,
            numbers,
            numerals
        ),
        None
    );
    assert!(crate::stdlib::PreparedFormat::new(b"%y").is_none());
    assert!(crate::stdlib::PreparedFormat::new(b"%5%").is_none());
    assert!(
        crate::stdlib::PreparedFormat::new(b"%q")
            .unwrap()
            .generic_only()
    );
}

const DIRECT_FORMAT: &[u8] = b"item=%06d value=%.3f hex=%x %-5d|%8.2f|%X";

/// Arguments for `DIRECT_FORMAT`: numbers the direct writers take, then
/// fractions, strings and huge values that leave them.
fn direct_conversion_rows() -> [[Value; 6]; 4] {
    [
        [
            Value::Integer(1),
            Value::Number(1.0 / 7.0),
            Value::Number(65535.0),
            Value::Integer(-12),
            Value::Number(-0.004),
            Value::Integer(-1),
        ],
        [
            Value::Number(240_000.0),
            Value::Integer(5),
            Value::Integer(0),
            Value::Number(0.0),
            Value::Number(1e9),
            Value::Number(255.0),
        ],
        [
            Value::Integer(i64::MIN),
            Value::Number(0.0005),
            Value::Integer(i64::MAX),
            Value::Integer(100_000),
            Value::Number(-0.0),
            Value::Integer(48879),
        ],
        // Fractions, strings and huge values leave the direct writers.
        [
            Value::Number(2.5),
            Value::from("1.25"),
            Value::Number(3.0),
            Value::from("7"),
            Value::Number(1e300),
            Value::Number(1e19),
        ],
    ]
}

#[test]
fn a_prepared_format_with_direct_conversions_matches_the_generic_formatter() {
    use crate::runtime::NumberFormat;
    let runtime = Runtime::new();
    let numerals = runtime.nonfinite_numerals();
    let format = DIRECT_FORMAT;
    let plan = crate::stdlib::PreparedFormat::new(format).unwrap();
    let rows = direct_conversion_rows();
    for numbers in [
        NumberFormat::Lua51,
        NumberFormat::LuaJit,
        NumberFormat::Lua53,
        NumberFormat::Lua55,
    ] {
        for row in &rows {
            let mut arguments = vec![Value::from(format)];
            arguments.extend(row.iter().cloned());
            let expected = crate::stdlib::strings::string_format_inner(
                &arguments, false, true, numbers, numerals,
            );
            let actual = plan.format(&arguments, true, numbers, numerals);
            match expected {
                Ok(expected) => {
                    assert_eq!(actual, Some(expected[0].clone()), "{numbers:?} {row:?}");
                }
                Err(_) => assert_eq!(actual, None, "{numbers:?} {row:?}"),
            }
        }
    }
}

#[test]
fn a_repeated_format_takes_the_direct_writers_and_matches_the_generic_formatter() {
    use crate::runtime::NumberFormat;
    let runtime = Runtime::new();
    let numerals = runtime.nonfinite_numerals();
    let format = Value::from(DIRECT_FORMAT);
    let call = |format: Value, row: &[Value], numbers| {
        let mut arguments = vec![format];
        arguments.extend(row.iter().cloned());
        crate::stdlib::strings::string_format_inner(&arguments, false, true, numbers, numerals)
    };
    for numbers in [
        NumberFormat::Lua51,
        NumberFormat::LuaJit,
        NumberFormat::Lua53,
        NumberFormat::Lua55,
    ] {
        for row in &direct_conversion_rows() {
            // A fresh format string misses the one-entry plan and formats
            // generically. The same string after a successful call formats
            // through its plan.
            let expected = call(Value::from(DIRECT_FORMAT), row, numbers);
            call(format.clone(), &[const { Value::Integer(0) }; 6], numbers).unwrap();
            let actual = call(format.clone(), row, numbers);
            assert_eq!(actual.ok(), expected.ok(), "{numbers:?} {row:?}");
        }
    }
}

/// What the float parser makes of `text` once it is trimmed, where the
/// number parsers hand it a string at all: one that starts, after a sign,
/// with a digit or a point.
fn decimal_reference(text: &str) -> Option<f64> {
    let trimmed = text.trim_matches(|byte: char| byte.is_ascii_whitespace());
    let body = trimmed
        .strip_prefix('+')
        .or_else(|| trimmed.strip_prefix('-'))
        .unwrap_or(trimmed);
    if !body.starts_with(|first: char| first.is_ascii_digit() || first == '.') {
        return None;
    }
    trimmed.parse().ok()
}

#[test]
fn the_plain_numeral_scan_agrees_with_the_float_parser() {
    use crate::runtime::{PlainNumeral, scan_plain_numeral};
    let check = |text: &str| {
        let reference = decimal_reference(text);
        match scan_plain_numeral(text.as_bytes()) {
            PlainNumeral::Whole {
                negative,
                magnitude,
            } => {
                #[allow(clippy::cast_precision_loss)]
                let value = magnitude as f64;
                let value = if negative { -value } else { value };
                assert_eq!(
                    reference.map(f64::to_bits),
                    Some(value.to_bits()),
                    "{text:?}"
                );
            }
            PlainNumeral::Fraction(value) => {
                assert_eq!(
                    reference.map(f64::to_bits),
                    Some(value.to_bits()),
                    "{text:?}"
                );
            }
            PlainNumeral::Invalid => {
                assert_eq!(reference, None, "{text:?}");
                // No hexadecimal numeral, infinity or NaN either.
                assert_eq!(
                    crate::stdlib::strings::parse_hex_number(text.as_bytes()),
                    None,
                    "{text:?}"
                );
                for numerals in [
                    crate::runtime::NonFiniteNumerals::Strtod,
                    crate::runtime::NonFiniteNumerals::LuaJit,
                ] {
                    assert_eq!(numerals.parse(text.as_bytes()), None, "{text:?}");
                }
            }
            PlainNumeral::Other => {}
        }
    };
    // What each kind of string is taken for.
    let kind = |text: &str| match scan_plain_numeral(text.as_bytes()) {
        PlainNumeral::Whole { .. } => 'w',
        PlainNumeral::Fraction(_) => 'f',
        PlainNumeral::Invalid => 'i',
        PlainNumeral::Other => 'o',
    };
    for (text, expected) in [
        ("0", 'w'),
        ("17", 'w'),
        ("-29", 'w'),
        ("+7", 'w'),
        (" 304\t", 'w'),
        ("-0", 'w'),
        ("999999999999999", 'w'),
        ("3.25", 'f'),
        ("-0.0", 'f'),
        ("5.", 'f'),
        (".5", 'f'),
        ("0.00000000000001", 'f'),
        ("12345678.9012345", 'f'),
        ("", 'i'),
        (" ", 'i'),
        ("-", 'i'),
        ("bad!", 'i'),
        ("1x", 'i'),
        ("--2", 'i'),
        ("+-2", 'i'),
        ("3.4.5", 'i'),
        ("1 2", 'i'),
        ("1\0", 'i'),
        ("12abc", 'i'),
        ("\u{b}1", 'i'),
        // These leave the scan: the full parsers know them.
        (".", 'o'),
        ("1e5", 'o'),
        ("1E-3", 'o'),
        ("2.5e3", 'o'),
        ("0x10", 'o'),
        ("0X1p4", 'o'),
        ("-0x.8", 'o'),
        ("inf", 'o'),
        ("-Infinity", 'o'),
        ("nan", 'o'),
        ("NaN(12)", 'o'),
        ("1234567890123456", 'o'),
        ("0.1234567890123456", 'o'),
        ("00000000000000000001", 'o'),
    ] {
        assert_eq!(kind(text), expected, "{text:?}");
        check(text);
    }
    // Every placement of a point in runs of digits, with signs and
    // whitespace, against the float parser bit for bit.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..40_000 {
        let digits = 1 + usize::try_from(next() % 17).unwrap();
        let mut text: String = (0..digits)
            .map(|_| char::from(b'0' + u8::try_from(next() % 10).unwrap()))
            .collect();
        let places = 2 * (u64::try_from(digits).unwrap() + 1);
        let point = usize::try_from(next() % places).unwrap();
        if point <= digits {
            text.insert(point, '.');
        }
        match next() % 8 {
            0 => text.insert(0, '-'),
            1 => text.insert(0, '+'),
            2 => text = format!(" {text}\n"),
            3 => text.push('x'),
            _ => {}
        }
        check(&text);
    }
}

#[test]
fn tonumber_of_strings_follows_each_profile_on_and_off_the_scan() {
    let float = Value::from;
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::LuaJit21,
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
        let tagged = runtime.tagged_integers();
        let lua51 = matches!(
            profile,
            CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
        );
        let whole = |value: i64| {
            if tagged {
                Value::Integer(value)
            } else {
                #[allow(clippy::cast_precision_loss)]
                float(value as f64)
            }
        };
        let cases: Vec<(&str, Option<i64>, Value)> = vec![
            ("17", None, whole(17)),
            ("-29", None, whole(-29)),
            ("+7", None, whole(7)),
            (" 304 ", None, whole(304)),
            ("999999999999999", None, whole(999_999_999_999_999)),
            // Sixteen digits leave the scan and convert the same way.
            ("1234567890123456", None, whole(1_234_567_890_123_456)),
            ("3.25", None, float(3.25)),
            ("5.", None, float(5.0)),
            (".5", None, float(0.5)),
            ("1e2", None, float(100.0)),
            ("0x10", None, whole(16)),
            ("", None, Value::Nil),
            (".", None, Value::Nil),
            ("bad!", None, Value::Nil),
            ("1x", None, Value::Nil),
            ("--2", None, Value::Nil),
            ("3.4.5", None, Value::Nil),
            (
                "inf",
                None,
                if lua51 {
                    float(f64::INFINITY)
                } else {
                    Value::Nil
                },
            ),
            // A base.
            ("ff", Some(16), whole(255)),
            ("7F", Some(16), whole(127)),
            ("zz", Some(36), whole(1295)),
            ("101", Some(2), whole(5)),
            ("8", Some(8), Value::Nil),
            ("", Some(16), Value::Nil),
            ("1e", Some(16), whole(30)),
            (" ff ", Some(16), whole(255)),
            ("-ff", Some(16), {
                if profile == CompatibilityProfile::Lua51 {
                    // `strtoul` negates in 64 bits.
                    #[allow(clippy::cast_precision_loss)]
                    float(255_u64.wrapping_neg() as f64)
                } else {
                    whole(-255)
                }
            }),
            // Thirteen digits leave the short digit loop.
            ("1000000000000", Some(16), whole(1 << 48)),
            ("0x1F", Some(16), if lua51 { whole(31) } else { Value::Nil }),
            ("17", Some(10), whole(17)),
            (
                "1e2",
                Some(10),
                if lua51 { float(100.0) } else { Value::Nil },
            ),
        ];
        for (text, base, expected) in cases {
            let mut arguments = vec![Value::from(text)];
            if let Some(base) = base {
                arguments.push(whole(base));
            }
            let answer = runtime.call_native(tonumber, &arguments).unwrap();
            assert_eq!(
                answer,
                vec![expected.clone()],
                "{profile} {text:?} {base:?}"
            );
            // What a compiled trace asks for is the same answer.
            assert_eq!(
                crate::stdlib::tonumber_of_text(&runtime, text.as_bytes(), base),
                Some(expected),
                "{profile} {text:?} {base:?}"
            );
        }
        // Negative zero keeps its sign where numbers are floats.
        let zero = runtime.call_native(tonumber, &[Value::from("-0")]).unwrap();
        match zero.as_slice() {
            [Value::Integer(0)] => assert!(tagged, "{profile}"),
            [Value::Number(zero)] => {
                assert!(
                    !tagged && zero.to_bits() == (-0.0_f64).to_bits(),
                    "{profile}"
                );
            }
            other => panic!("{profile}: {other:?}"),
        }
        // A base out of range is the native's error, not an answer.
        assert_eq!(
            crate::stdlib::tonumber_of_text(&runtime, b"1", Some(37)),
            None
        );
        assert!(
            runtime
                .call_native(tonumber, &[Value::from("1"), whole(37)])
                .is_err()
        );
    }
}

/// The bytes of `value` in two's complement, `size` of them.
fn reference_integer_bytes(value: i128, size: usize, little: bool) -> Vec<u8> {
    let mut bytes = value.to_le_bytes()[..size].to_vec();
    if !little {
        bytes.reverse();
    }
    bytes
}

#[test]
fn string_pack_integers_and_floats_match_their_byte_forms_at_every_width() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    let pack = library_native(&runtime, "string", "pack");
    let unpack = library_native(&runtime, "string", "unpack");
    for size in 1..=16_usize {
        for (signed, letter) in [(true, 'i'), (false, 'I')] {
            for (little, mark) in [(true, '<'), (false, '>')] {
                let format = Value::from(format!("{mark}{letter}{size}"));
                let bits = u32::try_from(8 * size.min(8)).unwrap();
                // The ends of the range, their neighbours and a few inside.
                let (low, high): (i128, i128) = if signed {
                    (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
                } else {
                    (0, (1_i128 << bits) - 1)
                };
                let limit = 1_i128 << 53;
                for value in [
                    low,
                    low + 1,
                    -1,
                    0,
                    1,
                    0x5a,
                    high - 1,
                    high,
                    low - 1,
                    high + 1,
                ] {
                    // Past 2^53 an argument converts through a double.
                    if value.abs() > limit {
                        continue;
                    }
                    let whole = i64::try_from(value).unwrap();
                    let fits = if signed {
                        size >= 8 || (low..=high).contains(&value)
                    } else {
                        value >= 0 && (size >= 8 || value <= high)
                    };
                    // An integer and the float that equals it pack alike.
                    #[allow(clippy::cast_precision_loss)]
                    let arguments = [Value::Integer(whole), Value::Number(whole as f64)];
                    for argument in arguments {
                        let packed = runtime.call_native(pack, &[format.clone(), argument.clone()]);
                        if !fits {
                            assert!(packed.is_err(), "{format:?} {argument:?}");
                            continue;
                        }
                        let packed = packed.unwrap();
                        let [Value::String(record)] = packed.as_slice() else {
                            panic!("{format:?} {argument:?}: {packed:?}");
                        };
                        assert_eq!(
                            record.as_bytes(),
                            reference_integer_bytes(value, size, little),
                            "{format:?} {argument:?}"
                        );
                        // The bytes read back as the integer, at the start of
                        // a string that ends with them and inside a longer
                        // one, where a whole word is loaded.
                        for padding in [0, 9] {
                            let mut source = record.as_bytes().to_vec();
                            source.extend(std::iter::repeat_n(0xa5, padding));
                            let again = runtime
                                .call_native(unpack, &[format.clone(), Value::from(source)])
                                .unwrap();
                            assert_eq!(
                                again,
                                vec![
                                    Value::Integer(whole),
                                    Value::Integer(i64::try_from(size).unwrap() + 1)
                                ],
                                "{format:?} {argument:?} {padding}"
                            );
                        }
                    }
                }
            }
        }
    }
    // The smallest integer is past what a double holds exactly and takes the
    // general path.
    assert_eq!(
        runtime
            .call_native(pack, &[Value::from(">i8"), Value::Integer(i64::MIN)])
            .unwrap(),
        vec![Value::from(&i64::MIN.to_be_bytes()[..])]
    );
    // Not integers: a fraction, a string that is no number, nothing at all.
    for argument in [Value::Number(1.5), Value::from("x"), Value::Nil] {
        assert!(
            runtime
                .call_native(pack, &[Value::from("<i4"), argument])
                .is_err()
        );
    }
    // A numeric string converts, as before.
    assert_eq!(
        runtime
            .call_native(pack, &[Value::from("<i2"), Value::from("258")])
            .unwrap(),
        vec![Value::from(&[2_u8, 1][..])]
    );
    // Floats in both orders and widths, and after an odd offset.
    for value in [0.0, -0.0, 1.5, -2.75e300, f64::INFINITY, 1.0e-310] {
        #[allow(clippy::cast_possible_truncation)]
        let narrow = value as f32;
        for (format, bytes) in [
            ("<d", value.to_le_bytes().to_vec()),
            (">d", value.to_be_bytes().to_vec()),
            ("<f", narrow.to_le_bytes().to_vec()),
            (">f", narrow.to_be_bytes().to_vec()),
        ] {
            let format = Value::from(format!("b{format}"));
            let packed = runtime
                .call_native(
                    pack,
                    &[format.clone(), Value::Integer(7), Value::Number(value)],
                )
                .unwrap();
            let mut expected = vec![7_u8];
            expected.extend(&bytes);
            assert_eq!(packed, vec![Value::from(expected.as_slice())], "{format:?}");
            let again = runtime
                .call_native(unpack, &[format.clone(), packed[0].clone()])
                .unwrap();
            let read = if bytes.len() == 4 {
                f64::from(narrow)
            } else {
                value
            };
            let [Value::Integer(7), Value::Number(number), Value::Integer(_)] = again.as_slice()
            else {
                panic!("{format:?}: {again:?}");
            };
            assert_eq!(number.to_bits(), read.to_bits(), "{format:?}");
        }
    }
    // A record longer than the inline buffer: the word stores stop where
    // the buffer has no room for a whole word.
    let format = Value::from("<i3".repeat(20));
    let mut arguments = vec![format.clone()];
    arguments.extend((0..20).map(|index| Value::Integer(index * 70_000 - 600_000)));
    let packed = runtime.call_native(pack, &arguments).unwrap();
    let [Value::String(record)] = packed.as_slice() else {
        panic!("one record");
    };
    let expected: Vec<u8> = (0..20)
        .flat_map(|index| reference_integer_bytes(index * 70_000 - 600_000, 3, true))
        .collect();
    assert_eq!(record.as_bytes(), expected);
    let again = runtime
        .call_native(unpack, &[format, packed[0].clone()])
        .unwrap();
    assert_eq!(
        again[..20],
        arguments[1..],
        "every integer of a long record"
    );
    // Data that ends inside an item is an error at either end of the word
    // load.
    for (format, source) in [
        ("<i4", &b"\x01\x02\x03"[..]),
        ("<i4i4", &b"\x01\x02\x03\x04\x05\x06\x07"[..]),
        ("<d", &b"\x01\x02\x03\x04\x05\x06\x07"[..]),
        ("<s2", &b"\x05\x00abcd"[..]),
        ("<s8", &b"\xff\xff\xff\xff\xff\xff\xff\xffab"[..]),
    ] {
        assert!(
            runtime
                .call_native(unpack, &[Value::from(format), Value::from(source)])
                .is_err(),
            "{format}"
        );
    }
    // A string with its length before it, in both orders.
    for (format, source) in [
        ("<s2", &b"\x03\x00abc"[..]),
        (">s2", &b"\x00\x03abc"[..]),
        ("<s1", &b"\x03abc"[..]),
        (">s8", &b"\x00\x00\x00\x00\x00\x00\x00\x03abc"[..]),
    ] {
        let again = runtime
            .call_native(unpack, &[Value::from(format), Value::from(source)])
            .unwrap();
        assert_eq!(
            again,
            vec![
                Value::from("abc"),
                Value::Integer(i64::try_from(source.len()).unwrap() + 1)
            ],
            "{format}"
        );
        assert_eq!(
            runtime
                .call_native(pack, &[Value::from(format), Value::from("abc")])
                .unwrap(),
            vec![Value::from(source)],
            "{format}"
        );
    }
}

#[test]
fn more_pack_formats_than_the_cache_holds_keep_their_own_items() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    let pack = library_native(&runtime, "string", "pack");
    // Twelve formats in turn, three times over: each call must find or parse
    // its own format whichever entries the others pushed out.
    for round in 0..3 {
        for size in 1..=12_usize {
            let packed = runtime
                .call_native(
                    pack,
                    &[
                        Value::from(format!("<I{size}")),
                        Value::Integer(0x41 + round),
                    ],
                )
                .unwrap();
            let mut expected = vec![0_u8; size];
            expected[0] = u8::try_from(0x41 + round).unwrap();
            assert_eq!(packed, vec![Value::from(expected.as_slice())], "{size}");
        }
        // A format that does not parse raises every time.
        assert!(
            runtime
                .call_native(pack, &[Value::from("<I17"), Value::Integer(1)])
                .is_err()
        );
    }
}

#[test]
fn table_concat_of_short_strings_copies_every_length_and_separator() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    let Value::Table(library) = runtime.get_global("table") else {
        panic!("table is installed");
    };
    let Value::NativeFunction(concat) = runtime.table_get(library, &Value::from("concat")).unwrap()
    else {
        panic!("table.concat is native");
    };
    // Elements of every inline length, a few on the heap among them, and
    // separators on both sides of the one-word copy.
    let lists: Vec<Vec<String>> = vec![
        (0..200).map(|index| (index % 101).to_string()).collect(),
        (0..=22).map(|length| "x".repeat(length)).collect(),
        (0..60)
            .map(|index| {
                if index % 20 == 7 {
                    "long".repeat(15)
                } else {
                    format!("w{index}")
                }
            })
            .collect(),
        vec![String::new(); 70],
        vec!["y".repeat(22); 5],
        // Mostly long: the sized copy of whole strings.
        vec!["z".repeat(40); 5],
    ];
    for list in &lists {
        for separator in ["", ",", ", ", "12345678", "123456789", &"-".repeat(30)] {
            let table = runtime.create_table().unwrap();
            for (index, element) in list.iter().enumerate() {
                runtime
                    .table_set(
                        table,
                        &Value::Integer(i64::try_from(index).unwrap() + 1),
                        Value::from(element.as_str()),
                    )
                    .unwrap();
            }
            let joined = runtime
                .call_native(concat, &[Value::Table(table), Value::from(separator)])
                .unwrap();
            assert_eq!(
                joined,
                vec![Value::from(list.join(separator))],
                "{} elements, separator {separator:?}",
                list.len()
            );
        }
    }
}
