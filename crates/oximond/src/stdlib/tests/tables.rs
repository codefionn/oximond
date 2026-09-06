use super::*;

#[test]
fn base_installation_sets_the_selected_stock_version() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    assert_eq!(runtime.get_global("_VERSION"), Value::from("Lua 5.4"));

    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    assert_eq!(runtime.get_global("_VERSION"), Value::from("Lua 5.1"));
}

#[test]
fn lua55_installs_modern_tables_and_hides_51_only_globals() {
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua55).unwrap();

    for name in [
        "gcinfo",
        "newproxy",
        "unpack",
        "loadstring",
        "getfenv",
        "setfenv",
        "module",
    ] {
        assert_eq!(
            runtime.get_global(name),
            Value::Nil,
            "{name} is not Lua 5.5 API"
        );
    }
    let Value::Table(table) = runtime.get_global("table") else {
        panic!("table is installed");
    };
    let (pack, move_values) = {
        let native = |name| match runtime.table_get(table, &Value::from(name)).unwrap() {
            Value::NativeFunction(function) => function,
            _ => panic!("table.{name} is native"),
        };
        (native("pack"), native("move"))
    };
    let packed = runtime
        .call_native(
            pack,
            &[Value::from("first"), Value::Nil, Value::Number(3.0)],
        )
        .unwrap();
    let Value::Table(packed) = packed[0] else {
        panic!("table.pack returns a table");
    };
    assert_eq!(
        runtime.table_get(packed, &Value::from("n")).unwrap(),
        Value::Integer(3)
    );
    assert_eq!(
        runtime.table_get(packed, &Value::Number(2.0)).unwrap(),
        Value::Nil
    );

    runtime
        .call_native(
            move_values,
            &[
                Value::Table(packed),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(2.0),
            ],
        )
        .unwrap();
    assert_eq!(
        runtime.table_get(packed, &Value::Number(2.0)).unwrap(),
        Value::from("first")
    );
    assert_eq!(
        runtime.table_get(packed, &Value::Number(3.0)).unwrap(),
        Value::Nil
    );

    let Value::NativeFunction(rawlen) = runtime.get_global("rawlen") else {
        panic!("rawlen is native");
    };
    assert_eq!(
        runtime.call_native(rawlen, &[Value::from("\0a")]).unwrap(),
        vec![Value::Integer(2)]
    );

    let Value::NativeFunction(warn) = runtime.get_global("warn") else {
        panic!("warn is native");
    };
    runtime
        .call_native(warn, &[Value::from("heads up")])
        .unwrap();
    assert_eq!(runtime.take_error_output(), b"Lua warning: heads up\n");
    runtime.call_native(warn, &[Value::from("@off")]).unwrap();
    runtime.call_native(warn, &[Value::from("quiet")]).unwrap();
    assert_eq!(runtime.take_error_output(), [] as [u8; 0]);
}

#[test]
fn base_installation_exposes_raw_table_access_and_raw_equality() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(rawget) = runtime.get_global("rawget") else {
        panic!("rawget is native");
    };
    let Value::NativeFunction(rawset) = runtime.get_global("rawset") else {
        panic!("rawset is native");
    };
    let Value::NativeFunction(rawequal) = runtime.get_global("rawequal") else {
        panic!("rawequal is native");
    };
    let table = runtime.create_table().unwrap();

    assert_eq!(
        runtime
            .call_native(
                rawset,
                &[
                    Value::Table(table),
                    Value::from("key"),
                    Value::from("value"),
                    Value::from("ignored"),
                ],
            )
            .unwrap(),
        vec![Value::Table(table)]
    );
    assert_eq!(
        runtime
            .call_native(
                rawget,
                &[
                    Value::Table(table),
                    Value::from("key"),
                    Value::from("ignored")
                ],
            )
            .unwrap(),
        vec![Value::from("value")]
    );
    runtime
        .call_native(
            rawset,
            &[Value::Table(table), Value::from("key"), Value::Nil],
        )
        .unwrap();
    assert_eq!(
        runtime
            .call_native(rawget, &[Value::Table(table), Value::from("key")])
            .unwrap(),
        vec![Value::Nil]
    );
    assert_eq!(
        runtime
            .call_native(rawequal, &[Value::from(-0.0), Value::from(0.0)])
            .unwrap(),
        vec![Value::from(true)]
    );
    assert_eq!(
        runtime
            .call_native(
                rawequal,
                &[Value::Number(f64::NAN), Value::Number(f64::NAN)]
            )
            .unwrap(),
        vec![Value::from(false)]
    );
}

#[test]
fn base_metatable_apis_honor_protection_and_debug_bypasses_it() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(getmetatable) = runtime.get_global("getmetatable") else {
        panic!("getmetatable is native");
    };
    let Value::NativeFunction(setmetatable) = runtime.get_global("setmetatable") else {
        panic!("setmetatable is native");
    };
    let Value::Table(debug) = runtime.get_global("debug") else {
        panic!("debug is a table");
    };
    let Value::NativeFunction(inspector) = runtime
        .table_get(debug, &Value::from("getmetatable"))
        .unwrap()
    else {
        panic!("debug.getmetatable is native");
    };
    let Value::NativeFunction(mutator) = runtime
        .table_get(debug, &Value::from("setmetatable"))
        .unwrap()
    else {
        panic!("debug.setmetatable is native");
    };

    let table = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__metatable"),
            Value::from("locked"),
        )
        .unwrap();
    assert_eq!(
        runtime
            .call_native(
                setmetatable,
                &[Value::Table(table), Value::Table(metatable)],
            )
            .unwrap(),
        vec![Value::Table(table)]
    );
    assert_eq!(
        runtime
            .call_native(getmetatable, &[Value::Table(table)])
            .unwrap(),
        vec![Value::from("locked")]
    );
    assert_eq!(
        runtime
            .call_native(inspector, &[Value::Table(table)])
            .unwrap(),
        vec![Value::Table(metatable)]
    );
    assert!(
        runtime
            .call_native(setmetatable, &[Value::Table(table), Value::Nil])
            .is_err()
    );
    assert_eq!(
        runtime
            .call_native(mutator, &[Value::Table(table), Value::Nil])
            .unwrap(),
        vec![Value::Table(table)]
    );
    assert_eq!(
        runtime
            .call_native(getmetatable, &[Value::Table(table)])
            .unwrap(),
        vec![Value::Nil]
    );

    assert_eq!(
        runtime
            .call_native(mutator, &[Value::Number(1.0), Value::Table(metatable)],)
            .unwrap(),
        vec![Value::Number(1.0)]
    );
    assert_eq!(
        runtime
            .call_native(getmetatable, &[Value::Number(2.0)])
            .unwrap(),
        vec![Value::from("locked")]
    );
}

#[test]
fn newproxy_creates_distinct_userdata_and_shares_requested_metatables() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(newproxy) = runtime.get_global("newproxy") else {
        panic!("newproxy is native");
    };

    let bare = runtime.call_native(newproxy, &[]).unwrap().remove(0);
    let false_proxy = runtime
        .call_native(newproxy, &[Value::Boolean(false)])
        .unwrap()
        .remove(0);
    assert!(matches!(bare, Value::Userdata(_)));
    assert!(matches!(false_proxy, Value::Userdata(_)));
    assert!(!Runtime::raw_equal(&bare, &false_proxy));
    assert_eq!(runtime.raw_metatable(&bare).unwrap(), None);
    assert_eq!(runtime.raw_metatable(&false_proxy).unwrap(), None);

    let source = runtime
        .call_native(newproxy, &[Value::Boolean(true)])
        .unwrap()
        .remove(0);
    let duplicate = runtime
        .call_native(newproxy, std::slice::from_ref(&source))
        .unwrap()
        .remove(0);
    assert!(matches!(source, Value::Userdata(_)));
    assert!(matches!(duplicate, Value::Userdata(_)));
    assert!(!Runtime::raw_equal(&source, &duplicate));
    let metatable = runtime.raw_metatable(&source).unwrap();
    assert!(metatable.is_some());
    assert_eq!(runtime.raw_metatable(&duplicate).unwrap(), metatable);
    let table = runtime.create_table().unwrap();
    assert!(
        runtime
            .call_native(newproxy, &[Value::Table(table)])
            .is_err()
    );
    let file = runtime.create_temp_file().unwrap();
    assert!(runtime.call_native(newproxy, &[Value::File(file)]).is_err());
}

#[test]
fn select_rejects_zero_and_out_of_range_indices() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(select) = runtime.get_global("select") else {
        panic!("select is native");
    };
    for index in [Value::from(0.0), Value::from(-2.0)] {
        assert!(
            runtime
                .call_native(select, &[index, Value::from("only")])
                .is_err()
        );
    }
    // An index past the last argument gives no values, as in stock Lua.
    for index in [Value::from(2.0), Value::from(3.0), Value::from(1000.0)] {
        assert_eq!(
            runtime
                .call_native(select, &[index, Value::from("only")])
                .unwrap(),
            Vec::new()
        );
    }
    assert!(runtime.call_native(select, &[Value::from(1.5)]).is_err());
    assert!(
        runtime
            .call_native(select, &[Value::from("not-a-count")])
            .is_err()
    );
}

#[test]
fn table_installation_exposes_vm_table_builtins() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(table) = runtime.get_global("table") else {
        panic!("table is a table");
    };
    assert_eq!(
        runtime.table_get(table, &Value::from("sort")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::TableSort)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("foreach")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::TableForeach)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("foreachi")).unwrap(),
        Value::BuiltinFunction(BuiltinFunction::TableForeachi)
    );
}

#[test]
fn table_concat_joins_bytes_with_inclusive_ranges() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(table_library) = runtime.get_global("table") else {
        panic!("table is a table");
    };
    let Value::NativeFunction(concat) = runtime
        .table_get(table_library, &Value::from("concat"))
        .unwrap()
    else {
        panic!("table.concat is native");
    };
    let table = runtime.create_table().unwrap();
    for (index, value) in [
        (1.0, Value::from(&b"\0"[..])),
        (2.0, Value::from(&b"\0\x01"[..])),
        (3.0, Value::from(&b"\0\x01\x02"[..])),
    ] {
        runtime
            .table_set(table, &Value::from(index), value)
            .unwrap();
    }
    assert_eq!(
        runtime
            .call_native(concat, &[Value::Table(table), Value::from(&b".\0."[..])])
            .unwrap(),
        vec![Value::from(&b"\0.\0.\0\x01.\0.\0\x01\x02"[..])]
    );
    assert_eq!(
        runtime
            .call_native(
                concat,
                &[
                    Value::Table(table),
                    Value::from(""),
                    Value::from(2.0),
                    Value::from(3.0),
                ],
            )
            .unwrap(),
        vec![Value::from(&b"\0\x01\0\x01\x02"[..])]
    );
    assert_eq!(
        runtime
            .call_native(
                concat,
                &[
                    Value::Table(table),
                    Value::from(","),
                    Value::from(3.0),
                    Value::from(2.0)
                ],
            )
            .unwrap(),
        vec![Value::from("")]
    );
    runtime
        .table_set(table, &Value::from(2.0), Value::from(42.0))
        .unwrap();
    assert_eq!(
        runtime
            .call_native(concat, &[Value::Table(table), Value::from(",")])
            .unwrap(),
        vec![Value::from(&b"\0,42,\0\x01\x02"[..])]
    );
    runtime
        .table_set(table, &Value::from(2.0), Value::Nil)
        .unwrap();
    assert!(
        runtime
            .call_native(
                concat,
                &[
                    Value::Table(table),
                    Value::from(","),
                    Value::from(1.0),
                    Value::from(3.0),
                ],
            )
            .is_err()
    );
    assert!(
        runtime
            .call_native(
                concat,
                &[
                    Value::Table(table),
                    Value::from(","),
                    Value::from(1.0),
                    Value::from(4.0),
                ],
            )
            .is_err()
    );
}

#[test]
fn table_functions_decline_only_tables_with_element_metamethods() {
    // From Lua 5.3 on the natives hand a table whose metatable has `__index`,
    // `__newindex` or `__len` to the VM; everything else keeps the raw path.
    let mut runtime = Runtime::new();
    install_base_with_profile(&mut runtime, CompatibilityProfile::Lua54).unwrap();
    let Value::Table(library) = runtime.get_global("table") else {
        panic!("table is installed");
    };
    let native = |runtime: &Runtime, name: &str| match runtime
        .table_get(library, &Value::from(name))
        .unwrap()
    {
        Value::NativeFunction(function) => function,
        _ => panic!("table.{name} is native"),
    };
    let plain = runtime.create_table().unwrap();
    let with_metatable = runtime.create_table().unwrap();
    let ignored = runtime.create_table().unwrap();
    runtime
        .table_set(ignored, &Value::from("__tostring"), Value::Boolean(true))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(with_metatable), Some(ignored))
        .unwrap();
    let with_len = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__len"), Value::Boolean(true))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(with_len), Some(metatable))
        .unwrap();
    for name in ["insert", "remove", "unpack", "concat", "move"] {
        let function = native(&runtime, name);
        let arguments = |table| match name {
            "insert" => vec![Value::Table(table), Value::Number(1.0)],
            "move" => vec![
                Value::Table(table),
                Value::Number(1.0),
                Value::Number(1.0),
                Value::Number(2.0),
            ],
            _ => vec![Value::Table(table)],
        };
        for table in [plain, with_metatable] {
            assert!(
                runtime.call_native(function, &arguments(table)).is_ok(),
                "table.{name} runs a table without element metamethods"
            );
        }
        assert_eq!(
            runtime.call_native(function, &arguments(with_len)),
            Err(RuntimeError::Redispatch),
            "table.{name} declines a table with __len"
        );
    }
    // Lua 5.1 and LuaJIT never decline.
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(library) = runtime.get_global("table") else {
        panic!("table is installed");
    };
    let Value::NativeFunction(insert) = runtime.table_get(library, &Value::from("insert")).unwrap()
    else {
        panic!("table.insert is native");
    };
    let table = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__len"), Value::Boolean(true))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    assert!(
        runtime
            .call_native(insert, &[Value::Table(table), Value::Number(1.0)])
            .is_ok()
    );
}

/// A table's observable state: every raw element around the border, the
/// order of its keys, and what `next` answers for each key around the array
/// part (a key deleted from the array still resumes a traversal).
fn table_state(runtime: &Runtime, table: crate::runtime::TableHandle) -> String {
    use std::fmt::Write as _;
    let mut text = format!("len={:?}", runtime.table_length(table).unwrap());
    for index in 0..40 {
        let key = Value::Number(f64::from(index));
        let _ = write!(
            text,
            " [{index}]={:?} next={:?}",
            runtime.table_get(table, &key).unwrap(),
            runtime.table_next(table, &key).ok()
        );
    }
    let mut key = Value::Nil;
    while let Ok((next, value)) = runtime.table_next(table, &key) {
        if matches!(next, Value::Nil) {
            break;
        }
        let _ = write!(text, " {next:?}:{value:?}");
        key = next;
    }
    text
}

/// The fast paths of `table.insert` and `table.remove` a trace calls
/// (`Runtime::table_trace_insert`, `table_trace_remove`) leave a table in the
/// state the library functions leave it, for every sequence of calls that
/// they accept; the ones they decline are left to the library.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a long randomized comparison over small, non-negative indices"
)]
fn the_trace_forms_of_insert_and_remove_agree_with_the_library_functions() {
    use crate::runtime::Runtime as Rt;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = move |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let (mut slow, mut fast) = (Rt::new(), Rt::new());
    let (mut reference, mut subject) = (slow.create_table().unwrap(), fast.create_table().unwrap());
    let mut taken = 0_u32;
    let mut declined = 0_u32;
    for step in 0..40_000_u32 {
        let value = match random(6) {
            0 => Value::Integer(i64::from(step)),
            1 => Value::Number(f64::from(step) + 0.5),
            2 => Value::Boolean(step % 2 == 0),
            3 => Value::from(format!("s{step}").as_str()),
            4 => Value::Integer(-1),
            _ => Value::Number(0.0),
        };
        // A damaged table stays damaged, so the pair starts over now and then.
        if step % 300 == 0 {
            (reference, subject) = (slow.create_table().unwrap(), fast.create_table().unwrap());
        }
        let size = u64::try_from(slow.table_length(reference).unwrap() as i64).unwrap();
        let place = random(size + 4) as f64 - 1.0;
        // The tables mostly stay lists; now and then one is damaged so the
        // paths that decline it are met.
        if random(500) == 0 {
            let key = Value::Number(random(size + 6) as f64 + 1.0);
            let stored = if random(3) == 0 {
                Value::Nil
            } else {
                value.clone()
            };
            slow.table_set(reference, &key, stored.clone()).unwrap();
            fast.table_set(subject, &key, stored).unwrap();
        }
        if random(900) == 0 {
            let field = Value::from("field");
            slow.table_set(reference, &field, value.clone()).unwrap();
            fast.table_set(subject, &field, value.clone()).unwrap();
        }
        let (kind, with_place) = (random(4), random(2) == 0);
        let (expected, got): (Result<Vec<Value>, RuntimeError>, Option<Vec<Value>>);
        match kind {
            0 | 1 => {
                let mut arguments = vec![Value::Table(reference)];
                if with_place {
                    arguments.push(Value::Number(place));
                }
                arguments.push(value.clone());
                expected = crate::stdlib::base::table_insert_loop(&mut slow, &arguments);
                let position = with_place.then_some(place);
                let accepted = position.is_none_or(|place| place >= 1.0)
                    && fast
                        .table_trace_insert(
                            subject,
                            position.map(|place| place as usize),
                            &value,
                            false,
                        )
                        .is_some();
                got = accepted.then(Vec::new);
                if accepted {
                    taken += 1;
                } else {
                    let mut arguments = vec![Value::Table(subject)];
                    if with_place {
                        arguments.push(Value::Number(place));
                    }
                    arguments.push(value.clone());
                    let again = crate::stdlib::base::table_insert(&mut fast, &arguments);
                    assert_eq!(again.is_ok(), expected.is_ok(), "step {step}");
                    declined += 1;
                }
            }
            _ => {
                let mut arguments = vec![Value::Table(reference)];
                if with_place {
                    arguments.push(Value::Number(place));
                }
                expected = crate::stdlib::base::table_remove_loop(&mut slow, &arguments);
                let position = with_place.then_some(place);
                let nothing = fast.table_trace_remove_nothing(
                    subject,
                    position.map(|place| place as i64),
                    false,
                );
                let removed = if nothing {
                    Some(Value::Nil)
                } else {
                    fast.table_trace_remove(
                        subject,
                        position.and_then(|place| (place >= 1.0).then_some(place as usize)),
                        false,
                        |_| true,
                    )
                    .map(|(value, _)| value)
                };
                got = removed.clone().map(|value| vec![value]);
                if removed.is_none() {
                    let mut arguments = vec![Value::Table(subject)];
                    if with_place {
                        arguments.push(Value::Number(place));
                    }
                    let again = crate::stdlib::base::table_remove(&mut fast, &arguments);
                    assert_eq!(again.is_ok(), expected.is_ok(), "step {step}");
                    declined += 1;
                } else {
                    taken += 1;
                }
            }
        }
        if let Some(got) = got {
            assert_eq!(Ok(got), expected, "step {step}");
        }
        assert_eq!(
            table_state(&slow, reference),
            table_state(&fast, subject),
            "step {step}, kind {kind}, place {place}, {value:?}"
        );
    }
    assert!(taken > 10_000 && declined > 100, "{taken} {declined}");
}

/// The library functions, which shift a dense list in one move, leave a table
/// in the state their element loops leave it, damaged tables included.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a randomized comparison over small, non-negative indices"
)]
fn the_library_shift_agrees_with_the_element_loops() {
    use crate::runtime::Runtime as Rt;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut random = move |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let (mut slow, mut fast) = (Rt::new(), Rt::new());
    let (mut reference, mut subject) = (slow.create_table().unwrap(), fast.create_table().unwrap());
    for step in 0..40_000_u32 {
        let value = match random(4) {
            0 => Value::Integer(i64::from(step)),
            1 => Value::from(format!("s{step}").as_str()),
            2 => Value::Boolean(true),
            _ => Value::Number(f64::from(step) + 0.5),
        };
        if step % 300 == 0 {
            (reference, subject) = (slow.create_table().unwrap(), fast.create_table().unwrap());
        }
        let size = u64::try_from(slow.table_length(reference).unwrap() as i64).unwrap();
        if random(60) == 0 {
            let key = Value::Number(random(size + 6) as f64 + 1.0);
            let stored = if random(2) == 0 {
                Value::Nil
            } else {
                value.clone()
            };
            slow.table_set(reference, &key, stored.clone()).unwrap();
            fast.table_set(subject, &key, stored).unwrap();
        }
        let place = Value::Number(random(size + 4) as f64 - 1.0);
        let mut arguments = vec![Value::Table(reference)];
        if random(2) == 0 {
            arguments.push(place);
        }
        let mut other = arguments.clone();
        other[0] = Value::Table(subject);
        let (expected, got) = if random(2) == 0 {
            arguments.push(value);
            other.push(arguments[arguments.len() - 1].clone());
            (
                crate::stdlib::base::table_insert_loop(&mut slow, &arguments),
                crate::stdlib::base::table_insert(&mut fast, &other),
            )
        } else {
            (
                crate::stdlib::base::table_remove_loop(&mut slow, &arguments),
                crate::stdlib::base::table_remove(&mut fast, &other),
            )
        };
        assert_eq!(got.is_ok(), expected.is_ok(), "step {step}");
        if let (Ok(got), Ok(expected)) = (got, expected) {
            assert_eq!(got, expected, "step {step}");
        }
        assert_eq!(
            table_state(&slow, reference),
            table_state(&fast, subject),
            "step {step}"
        );
    }
}
