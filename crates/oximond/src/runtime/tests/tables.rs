use super::*;
use crate::string::LuaString;

#[test]
fn table_reserve_keeps_empty_capacity_through_clear() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let initial_hash_bytes = runtime.table(table).unwrap().entries.allocated_bytes();
    runtime.table_reserve(table, 48, 48).unwrap();
    let array_capacity = runtime.table(table).unwrap().array.capacity();
    let hash_bytes = runtime.table(table).unwrap().entries.allocated_bytes();
    assert!(array_capacity >= 48);
    assert!(hash_bytes > initial_hash_bytes);
    assert_eq!(
        runtime.table_next(table, &Value::Nil),
        Ok((Value::Nil, Value::Nil))
    );

    runtime.table_clear(table).unwrap();
    assert_eq!(
        runtime.table(table).unwrap().array.capacity(),
        array_capacity
    );
    assert_eq!(
        runtime.table(table).unwrap().entries.allocated_bytes(),
        hash_bytes
    );
}

#[test]
fn table_clear_keeps_identity_metatable_capacity_and_invalidates_cached_lookups() {
    use crate::runtime::table_storage::entries::StringReadCache;
    use crate::string::LuaString;

    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime.table_reserve(table, 64, 64).unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    for index in 1..=32 {
        runtime
            .table_set(table, &Value::Integer(index), Value::Integer(index * 2))
            .unwrap();
        runtime
            .table_set(
                table,
                &Value::from(format!("field_{index}")),
                Value::Integer(index),
            )
            .unwrap();
    }
    runtime
        .table_set(table, &Value::Boolean(true), Value::from("bool"))
        .unwrap();
    let key = LuaString::from("field_20");
    let mut cache = StringReadCache::default();
    assert_eq!(
        runtime.table_get_string_cached(table, &key, &mut cache),
        Some(&Value::Integer(20))
    );
    assert_eq!(
        runtime.table_get_string_cached(table, &key, &mut cache),
        Some(&Value::Integer(20))
    );
    assert_eq!(
        runtime.table_get_string_cached(table, &key, &mut cache),
        Some(&Value::Integer(20))
    );
    assert!(cache.hits() > 0);
    assert!(runtime.table(table).unwrap().entries.is_indexed());
    let array_capacity = runtime.table(table).unwrap().array.capacity();
    let hash_capacity = match &runtime.table(table).unwrap().entries {
        TableEntries::Other(other) => match other.view() {
            OtherRef::Indexed(entries) => entries.values.capacity(),
            OtherRef::Shaped(_) => unreachable!(),
        },
        TableEntries::Ordered(_) => unreachable!(),
    };

    runtime.table_clear(table).unwrap();

    assert_eq!(
        runtime.raw_metatable(&Value::Table(table)).unwrap(),
        Some(metatable)
    );
    assert_eq!(
        runtime.table(table).unwrap().array.capacity(),
        array_capacity
    );
    let after_hash_capacity = match &runtime.table(table).unwrap().entries {
        TableEntries::Other(other) => match other.view() {
            OtherRef::Indexed(entries) => entries.values.capacity(),
            OtherRef::Shaped(_) => unreachable!(),
        },
        TableEntries::Ordered(_) => unreachable!(),
    };
    assert_eq!(after_hash_capacity, hash_capacity);
    assert_eq!(runtime.table_length(table), Ok(0.0));
    assert_eq!(
        runtime.table_next(table, &Value::Nil),
        Ok((Value::Nil, Value::Nil))
    );
    for key in [
        Value::Integer(1),
        Value::Integer(32),
        Value::from("field_1"),
        Value::from("field_20"),
        Value::Boolean(true),
    ] {
        assert_eq!(runtime.table_get(table, &key), Ok(Value::Nil));
    }
    assert_eq!(
        runtime.table_get_string_cached(table, &key, &mut cache),
        None
    );
    runtime
        .table_set(table, &Value::from("field_20"), Value::Integer(99))
        .unwrap();
    assert_eq!(
        runtime.table_get_string_cached(table, &key, &mut cache),
        Some(&Value::Integer(99))
    );
}

#[test]
fn table_clear_preserves_shaped_storage_and_readonly_tables_reject_both_operations() {
    use crate::runtime::TableShape;
    use crate::string::LuaString;

    let shape = TableShape::new(["first", "second"].map(LuaString::from)).unwrap();
    let mut runtime = Runtime::new();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    runtime
        .table_set(table, &Value::from("first"), Value::Integer(7))
        .unwrap();
    runtime
        .table_set(table, &Value::from("second"), Value::Integer(8))
        .unwrap();
    runtime.table_clear(table).unwrap();
    assert!(runtime.table(table).unwrap().entries.is_shaped());
    assert_eq!(
        runtime.table_get(table, &Value::from("first")),
        Ok(Value::Nil)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("second")),
        Ok(Value::Nil)
    );
    runtime
        .table_set(table, &Value::from("first"), Value::Integer(9))
        .unwrap();

    runtime.table_set_readonly(table, true).unwrap();
    let error = RuntimeError::Message("Attempt to modify a readonly table".into());
    assert_eq!(runtime.table_clear(table), Err(error.clone()));
    assert_eq!(runtime.table_reserve(table, 4, 4), Err(error));
    assert_eq!(
        runtime.table_get(table, &Value::from("first")),
        Ok(Value::Integer(9))
    );
    assert!(runtime.table(table).unwrap().entries.is_shaped());
}

#[test]
fn readonly_tables_reject_raw_and_optimized_writes_and_mutable_native_views() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::Number(3.0))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(2.0), Value::Number(1.0))
        .unwrap();
    runtime
        .table_set(table, &Value::from("field"), Value::Number(7.0))
        .unwrap();
    runtime.table_set_readonly(table, true).unwrap();
    assert_eq!(runtime.table_is_readonly(table), Ok(true));
    let error = RuntimeError::Message("Attempt to modify a readonly table".into());
    assert_eq!(
        runtime.table_set(table, &Value::Number(1.0), Value::Nil),
        Err(error.clone())
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(2.0)),
        Err(error.clone())
    );
    assert_eq!(
        runtime.sort_dense_finite_number_prefix(table, 2),
        Err(error.clone())
    );
    assert_eq!(
        runtime.raw_set_metatable(&Value::Table(table), None),
        Err(error)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)),
        Ok(Value::Number(3.0))
    );
    runtime.table_set_readonly(table, false).unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::Number(2.0))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)),
        Ok(Value::Number(2.0))
    );
}

#[test]
fn table_handles_and_numeric_keys_have_lua_identity_rules() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for _ in 0..32 {
        runtime.create_table().unwrap();
    }
    runtime
        .table_set(table, &Value::from(-0.0), Value::from("zero"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &Value::from(0.0)).unwrap(),
        Value::from("zero")
    );
    assert_eq!(
        runtime.table_set(table, &Value::from(f64::NAN), Value::Nil),
        Err(RuntimeError::InvalidTableKey("number"))
    );
    assert_eq!(
        runtime.table_get(table, &Value::from(f64::NAN)),
        Ok(Value::Nil)
    );
}

#[test]
fn large_integral_doubles_remain_lossless_hash_keys() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let huge = 10e30_f64;
    let adjacent = f64::from_bits(huge.to_bits() + 1);

    runtime
        .table_set(table, &Value::Number(huge), Value::from("huge"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(adjacent), Value::from("adjacent"))
        .unwrap();

    assert_eq!(
        runtime.table_get(table, &Value::Number(huge)).unwrap(),
        Value::from("huge")
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(adjacent)).unwrap(),
        Value::from("adjacent")
    );

    let (first_key, first_value) = runtime.table_next(table, &Value::Nil).unwrap();
    let (second_key, second_value) = runtime.table_next(table, &first_key).unwrap();
    let entries = [(first_key, first_value), (second_key, second_value)];
    assert!(entries.iter().any(|(key, value)| {
        Runtime::raw_equal(key, &Value::Number(huge)) && *value == Value::from("huge")
    }));
    assert!(entries.iter().any(|(key, value)| {
        Runtime::raw_equal(key, &Value::Number(adjacent)) && *value == Value::from("adjacent")
    }));
}

#[test]
fn globals_alias_a_self_referential_environment_table() {
    let mut runtime = Runtime::new();
    let environment = runtime.global_environment();
    assert_eq!(
        runtime.table_get(environment, &Value::from("_G")).unwrap(),
        Value::Table(environment)
    );
    assert_eq!(runtime.get_global("_G"), Value::Table(environment));

    runtime.set_global("from_global", Value::Number(1.0));
    assert_eq!(
        runtime
            .table_get(environment, &Value::from("from_global"))
            .unwrap(),
        Value::Number(1.0)
    );
    runtime
        .table_set(environment, &Value::from("from_table"), Value::Number(2.0))
        .unwrap();
    assert_eq!(runtime.get_global("from_table"), Value::Number(2.0));

    runtime.set_global("from_global", Value::Nil);
    assert_eq!(
        runtime
            .table_get(environment, &Value::from("from_global"))
            .unwrap(),
        Value::Nil
    );
    runtime
        .table_set(environment, &Value::from("from_table"), Value::Nil)
        .unwrap();
    assert_eq!(runtime.get_global("from_table"), Value::Nil);

    let binary_name = b"\xffglobal";
    runtime.set_global(&binary_name[..], Value::from("binary"));
    assert_eq!(
        runtime
            .table_get(environment, &Value::from(&binary_name[..]))
            .unwrap(),
        Value::from("binary")
    );
    assert_eq!(runtime.get_global(&binary_name[..]), Value::from("binary"));
}

#[test]
fn table_values_alias_the_same_runtime_owned_table() {
    let mut runtime = Runtime::new();
    let table = Value::Table(runtime.create_table().unwrap());
    let alias = table.clone();
    let Value::Table(table_handle) = table else {
        unreachable!("table value was constructed above");
    };
    let Value::Table(alias_handle) = alias else {
        unreachable!("table value was cloned above");
    };
    runtime
        .table_set(table_handle, &Value::from("key"), Value::from("value"))
        .unwrap();
    assert_eq!(
        runtime
            .table_get(alias_handle, &Value::from("key"))
            .unwrap(),
        Value::from("value")
    );
}

#[test]
fn direct_table_set_skips_only_a_missing_key_with_a_metatable() {
    let mut runtime = Runtime::default();
    let table = runtime.create_table().unwrap();
    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(10.0))
            .unwrap()
    );

    let metatable = runtime.create_table().unwrap();
    let handler = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Table(handler))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(11.0))
            .unwrap()
    );
    assert!(
        !runtime
            .table_set_if_no_newindex(table, &Value::Number(2.0), Value::Number(20.0))
            .unwrap()
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)).unwrap(),
        Value::Number(11.0)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(2.0)).unwrap(),
        Value::Nil
    );
}

#[test]
fn dense_finite_number_prefix_sort_fast_path_and_fallbacks() {
    let mut runtime = Runtime::new();
    let sorted = runtime.create_table().unwrap();
    for (index, value) in [(1, 3.0), (2, -1.0), (3, 2.0)] {
        runtime
            .table_set(
                sorted,
                &Value::Number(f64::from(index)),
                Value::Number(value),
            )
            .unwrap();
    }
    assert!(runtime.sort_dense_finite_number_prefix(sorted, 3).unwrap());
    for (index, expected) in [(1, -1.0), (2, 2.0), (3, 3.0)] {
        assert_eq!(
            runtime
                .table_get(sorted, &Value::Number(f64::from(index)))
                .unwrap(),
            Value::Number(expected)
        );
    }

    let hole = runtime.create_table().unwrap();
    runtime
        .table_set(hole, &Value::Number(1.0), Value::Number(3.0))
        .unwrap();
    runtime
        .table_set(hole, &Value::Number(3.0), Value::Number(1.0))
        .unwrap();
    assert!(!runtime.sort_dense_finite_number_prefix(hole, 3).unwrap());

    let retained_hole = runtime.create_table().unwrap();
    for (index, value) in [(1, 3.0), (2, 2.0), (3, 1.0)] {
        runtime
            .table_set(
                retained_hole,
                &Value::Number(f64::from(index)),
                Value::Number(value),
            )
            .unwrap();
    }
    runtime
        .table_set(retained_hole, &Value::Number(2.0), Value::Nil)
        .unwrap();
    assert!(
        !runtime
            .sort_dense_finite_number_prefix(retained_hole, 1)
            .unwrap()
    );

    let non_number = runtime.create_table().unwrap();
    runtime
        .table_set(non_number, &Value::Number(1.0), Value::Number(3.0))
        .unwrap();
    runtime
        .table_set(non_number, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    assert!(
        !runtime
            .sort_dense_finite_number_prefix(non_number, 2)
            .unwrap()
    );

    let nan = runtime.create_table().unwrap();
    runtime
        .table_set(nan, &Value::Number(1.0), Value::Number(3.0))
        .unwrap();
    runtime
        .table_set(nan, &Value::Number(2.0), Value::Number(f64::NAN))
        .unwrap();
    assert!(!runtime.sort_dense_finite_number_prefix(nan, 2).unwrap());
}

#[test]
fn table_length_stops_at_the_first_missing_positive_integer_key() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(4.0), Value::from("four"))
        .unwrap();
    runtime
        .table_set(table, &Value::from("name"), Value::from("value"))
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        2.0_f64.to_bits()
    );
    runtime
        .table_set(table, &Value::Number(2.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn array_border_tracks_holes_sparse_keys_refills_and_tail_deletes() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for (key, value) in [(1.0, "one"), (3.0, "three"), (90.0, "ninety")] {
        runtime
            .table_set(table, &Value::Number(key), Value::from(value))
            .unwrap();
    }
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(90.0)).unwrap(),
        Value::from("ninety")
    );

    runtime
        .table_set(table, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        3.0_f64.to_bits()
    );

    runtime
        .table_set(table, &Value::Number(2.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    runtime
        .table_set(table, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        3.0_f64.to_bits()
    );

    runtime
        .table_set(table, &Value::Number(3.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        2.0_f64.to_bits()
    );
    runtime
        .table_set(table, &Value::Number(2.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    runtime
        .table_set(table, &Value::Number(1.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_length(table).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn table_next_crosses_array_holes_and_sparse_entries() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for (key, value) in [(1.0, "one"), (2.0, "two"), (3.0, "three"), (90.0, "ninety")] {
        runtime
            .table_set(table, &Value::Number(key), Value::from(value))
            .unwrap();
    }
    runtime
        .table_set(table, &Value::Number(2.0), Value::Nil)
        .unwrap();

    let first = runtime.table_next(table, &Value::Nil).unwrap();
    assert_eq!(first, (Value::Number(1.0), Value::from("one")));
    let after_hole = runtime.table_next(table, &first.0).unwrap();
    assert_eq!(after_hole, (Value::Number(3.0), Value::from("three")));
    let sparse = runtime.table_next(table, &after_hole.0).unwrap();
    assert_eq!(sparse, (Value::Number(90.0), Value::from("ninety")));
    assert_eq!(
        runtime.table_next(table, &sparse.0).unwrap(),
        (Value::Nil, Value::Nil)
    );
}

#[test]
fn table_next_resumes_after_the_current_array_or_map_key_is_deleted() {
    let mut runtime = Runtime::new();
    let array = runtime.create_table().unwrap();
    runtime
        .table_set(array, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    runtime
        .table_set(array, &Value::Number(2.0), Value::from("two"))
        .unwrap();
    let (first, _) = runtime.table_next(array, &Value::Nil).unwrap();
    assert_eq!(first, Value::Number(1.0));
    runtime.table_set(array, &first, Value::Nil).unwrap();
    assert_eq!(
        runtime.table_next(array, &first).unwrap(),
        (Value::Number(2.0), Value::from("two"))
    );

    let map = runtime.create_table().unwrap();
    runtime
        .table_set(map, &Value::from("a"), Value::Number(1.0))
        .unwrap();
    runtime
        .table_set(map, &Value::from("b"), Value::Number(2.0))
        .unwrap();
    let (first, _) = runtime.table_next(map, &Value::Nil).unwrap();
    assert_eq!(first, Value::from("a"));
    runtime.table_set(map, &first, Value::Nil).unwrap();
    assert_eq!(
        runtime.table_next(map, &first).unwrap(),
        (Value::from("b"), Value::Number(2.0))
    );
}

#[test]
fn indexed_numeric_maps_preserve_key_identity_and_absorb_dense_prefixes() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in 1000..1200 {
        runtime
            .table_set(table, &Value::Integer(index), Value::Integer(index))
            .unwrap();
    }
    assert!(runtime.table(table).unwrap().entries.is_indexed());
    for (key, alias) in [
        (Value::Integer(0), Value::Number(-0.0)),
        (Value::Integer(-7), Value::Number(-7.0)),
        (Value::Integer(1050), Value::Number(1050.0)),
    ] {
        runtime
            .table_set(table, &key, Value::Boolean(true))
            .unwrap();
        assert_eq!(
            runtime.table_get(table, &alias).unwrap(),
            Value::Boolean(true)
        );
        runtime
            .table_set(table, &alias, Value::Boolean(false))
            .unwrap();
        assert_eq!(
            runtime.table_get(table, &key).unwrap(),
            Value::Boolean(false)
        );
    }
    let adjacent = f64::from_bits(1.5_f64.to_bits() + 1);
    for (key, value) in [
        (1.5, 1.0),
        (adjacent, 2.0),
        (1e31, 3.0),
        (f64::INFINITY, 4.0),
    ] {
        runtime
            .table_set(table, &Value::Number(key), Value::Number(value))
            .unwrap();
    }
    for (key, value) in [
        (1.5, 1.0),
        (adjacent, 2.0),
        (1e31, 3.0),
        (f64::INFINITY, 4.0),
    ] {
        assert_eq!(
            runtime.table_get(table, &Value::Number(key)).unwrap(),
            Value::Number(value)
        );
    }
    assert!(
        runtime
            .table_set(table, &Value::Number(f64::NAN), Value::Boolean(true))
            .is_err()
    );
    runtime
        .table_set(table, &Value::Number(1100.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(table, &Value::Number(1100.0)).unwrap().0,
        Value::Number(1101.0)
    );
    runtime
        .table_set(table, &Value::Integer(1100), Value::Boolean(true))
        .unwrap();
    for index in 1..1000 {
        runtime
            .table_set(table, &Value::Integer(index), Value::Boolean(true))
            .unwrap();
    }
    assert_eq!(runtime.table(table).unwrap().array.len(), 1199);
    assert_eq!(runtime.table_length(table).unwrap(), 1199.0);
    assert_eq!(
        runtime.table_get(table, &Value::Number(1050.0)).unwrap(),
        Value::Boolean(false)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Integer(1100)).unwrap(),
        Value::Boolean(true)
    );
    let TableEntries::Other(other) = &runtime.table(table).unwrap().entries else {
        panic!("keep indexed non-array entries")
    };
    let OtherRef::Indexed(entries) = other.view() else {
        panic!("keep indexed non-array entries")
    };
    assert_eq!(entries.values.len(), 6);
    assert_eq!(entries.order().len(), entries.values.len());
}

#[test]
fn sparse_array_entries_are_absorbed_when_the_dense_prefix_reaches_them() {
    let mut table = crate::runtime::table_storage::Table::default();
    table.set_array(3, Value::from("three"));
    table.set_array(1, Value::from("one"));
    table.set_array(2, Value::from("two"));

    assert_eq!(
        table.array,
        vec![Value::from("one"), Value::from("two"), Value::from("three")]
    );
    assert!(table.entries.is_empty());
    assert_eq!(table.border, 3);
}

#[test]
fn dense_array_updates_stay_out_of_sparse_storage() {
    let mut table = crate::runtime::table_storage::Table::default();
    table.set_array(1, Value::from("first"));
    table.set_array(1, Value::from("replacement"));

    assert_eq!(table.array, vec![Value::from("replacement")]);
    assert!(table.entries.is_empty());
    assert!(table.deleted_array_keys.is_empty(table.array.len()));
    assert!(table.deleted_entries.is_empty());
}

#[test]
fn array_border_recovers_after_hole_is_refilled() {
    let mut table = crate::runtime::table_storage::Table::default();
    table.set_array(2, Value::from("two"));
    table.set_array(1, Value::from("one"));
    assert_eq!(table.border, 2);

    table.set_array(1, Value::Nil);
    assert_eq!(table.border, 0);
    table.set_array(1, Value::from("one"));
    assert_eq!(table.border, 2);
}

#[test]
fn byte_strings_remain_valid_values_keys_globals_and_concat_operands() {
    let mut runtime = Runtime::new();
    let bytes = Value::bytes(vec![0xff, b'a']);
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &bytes, Value::from("stored"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &bytes).unwrap(),
        Value::from("stored")
    );

    runtime.set_global(&b"\xffname"[..], bytes.clone());
    assert!(Runtime::raw_equal(
        &runtime.get_global(&b"\xffname"[..]),
        &bytes
    ));

    let Value::String(joined) = runtime
        .concatenate(&bytes, &Value::from("!"))
        .expect("byte-string concatenation is valid")
    else {
        panic!("concatenation returns a string");
    };
    assert_eq!(joined.as_bytes(), b"\xffa!");
}

#[test]
fn userdata_has_distinct_identity_and_is_a_valid_table_key() {
    let mut runtime = Runtime::new();
    let first = Value::Userdata(runtime.create_userdata().unwrap());
    let second = Value::Userdata(runtime.create_userdata().unwrap());
    let table = runtime.create_table().unwrap();

    assert!(Runtime::raw_equal(&first, &first));
    assert!(!Runtime::raw_equal(&first, &second));
    runtime
        .table_set(table, &first, Value::from("first"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &first).unwrap(),
        Value::from("first")
    );
    assert_eq!(runtime.table_get(table, &second).unwrap(), Value::Nil);
}

#[test]
fn metatables_are_protected_or_raw_and_shared_by_scalar_type() {
    let mut runtime = Runtime::new();
    let protected = runtime.create_table().unwrap();
    runtime
        .table_set(
            protected,
            &Value::from("__metatable"),
            Value::from("locked"),
        )
        .unwrap();
    let table = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(protected))
        .unwrap();
    assert_eq!(
        runtime.protected_metatable(&Value::Table(table)).unwrap(),
        Value::from("locked")
    );
    assert!(runtime.set_table_metatable(table, None).is_err());
    runtime
        .raw_set_metatable(&Value::Table(table), None)
        .unwrap();

    let number_metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Number(1.0), Some(number_metatable))
        .unwrap();
    assert_eq!(
        runtime.raw_metatable(&Value::Number(2.0)).unwrap(),
        Some(number_metatable)
    );
    runtime
        .raw_set_metatable(&Value::Number(3.0), None)
        .unwrap();
}

#[test]
fn userdata_metatable_is_traced_and_stale_userdata_is_rejected() {
    let mut runtime = Runtime::new();
    let userdata = runtime.create_userdata().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Userdata(userdata));
    assert_eq!(runtime.collect_full(roots.finish()).reclaimed(), 0);
    assert_eq!(
        runtime.raw_metatable(&Value::Userdata(userdata)).unwrap(),
        Some(metatable)
    );

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.userdatas, 1);
    assert!(matches!(
        runtime.raw_metatable(&Value::Userdata(userdata)),
        Err(RuntimeError::InvalidUserdata(handle)) if handle == userdata
    ));
}

#[test]
fn raw_length_ignores_metatables_and_warnings_have_runtime_state() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::from("item"))
        .unwrap();
    assert!((runtime.raw_length(&Value::Table(table)).unwrap() - 1.0).abs() < f64::EPSILON);
    assert!((runtime.raw_length(&Value::bytes(b"a\0")).unwrap() - 2.0).abs() < f64::EPSILON);
    assert!(runtime.warnings_enabled());
    runtime.set_warnings_enabled(false);
    assert!(!runtime.warnings_enabled());
}

#[test]
fn incremental_dead_table_teardown_is_spread_over_steps() {
    let mut runtime = Runtime::new();
    let dead = runtime.create_table().unwrap();
    for index in 1..=40_000 {
        runtime
            .table_set(dead, &Value::Number(f64::from(index)), Value::Boolean(true))
            .unwrap();
    }
    advance_to_incremental_sweep(&mut runtime, &Value::Nil);
    let mut largest_drain = 0;
    let mut saw_partial_teardown = false;
    for _ in 0..100_000 {
        let before = runtime.gc_detached.front().map(|table| table.array.len());
        let result = runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 0);
        let after = runtime.gc_detached.front().map(|table| table.array.len());
        if let (Some(before), Some(after)) = (before, after) {
            largest_drain = largest_drain.max(before.saturating_sub(after));
            saw_partial_teardown |= after > 0 && after < before;
        }
        if result == GcStepResult::Complete {
            assert!(saw_partial_teardown);
            assert!(largest_drain <= 64 * 64);
            assert!(runtime.gc_detached.is_empty());
            assert!(runtime.tables.get(dead.0).is_none());
            return;
        }
    }
    panic!("collector did not finish dead table teardown");
}

#[test]
fn pair_concatenation_matches_many_parts_for_strings_and_numbers() {
    let mut runtime = Runtime::new();
    let pairs = [
        (Value::from(&b"a\0"[..]), Value::from(&b"\xff"[..])),
        (Value::Integer(-12), Value::from("x")),
        (Value::Number(f64::INFINITY), Value::Number(f64::NAN)),
    ];
    for (left, right) in pairs {
        assert_eq!(
            runtime.concatenate(&left, &right).unwrap(),
            runtime.concatenate_many(&[left, right]).unwrap()
        );
    }
}

#[test]
fn raw_array_range_borrows_dense_slots_and_rejects_out_of_bounds() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Integer(1), Value::from("a"))
        .unwrap();
    runtime
        .table_set(table, &Value::Integer(2), Value::from("b"))
        .unwrap();
    assert_eq!(
        runtime.table_raw_array_range(table, 1, 2).unwrap(),
        Some(&[Value::from("a"), Value::from("b")][..])
    );
    assert_eq!(runtime.table_raw_array_range(table, 0, 1).unwrap(), None);
    assert_eq!(runtime.table_raw_array_range(table, 1, 3).unwrap(), None);
}

#[test]
fn trace_views_bind_array_slots_and_typed_fields_with_or_without_a_metatable() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in 1_i32..=4 {
        runtime
            .table_set(
                table,
                &Value::Integer(i64::from(index)),
                Value::Number(f64::from(index)),
            )
            .unwrap();
    }
    runtime
        .table_set(table, &Value::from("x"), Value::Number(2.5))
        .unwrap();
    runtime
        .table_set(table, &Value::from("on"), Value::Boolean(true))
        .unwrap();
    runtime
        .table_set(table, &Value::from("n"), Value::Integer(7))
        .unwrap();
    runtime
        .table_set(table, &Value::from("s"), Value::from("text"))
        .unwrap();
    assert_eq!(runtime.table_array_len(table), Ok(4));
    let view = runtime.table_trace_array(table).unwrap();
    assert_eq!(view.len(), 4);
    assert!(!view.as_ptr().is_null());

    // A slot binds only for the type the caller asks for, and only when it
    // holds a value.
    let key = |name: &str| Value::from(name);
    for (name, kind, expected) in [
        ("x", NativeSlotKind::Number, true),
        ("x", NativeSlotKind::Integer, false),
        ("x", NativeSlotKind::Boolean, false),
        ("on", NativeSlotKind::Boolean, true),
        ("on", NativeSlotKind::Number, false),
        ("n", NativeSlotKind::Integer, true),
        ("n", NativeSlotKind::Number, false),
        ("s", NativeSlotKind::Number, false),
        ("missing", NativeSlotKind::Number, false),
    ] {
        let slot = runtime
            .table_native_value_slot(table, &key(name), kind)
            .unwrap();
        assert_eq!(slot.is_some(), expected, "{name} as {kind:?}");
    }
    // An array element binds through the same call.
    assert!(
        runtime
            .table_native_value_slot(table, &Value::Integer(2), NativeSlotKind::Number)
            .unwrap()
            .is_some()
    );
    // A metatable changes neither the array view nor a present slot: Lua
    // consults `__index` and `__newindex` only for an absent key.
    let meta = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(meta))
        .unwrap();
    assert_eq!(runtime.table_trace_array(table).unwrap().len(), 4);
    assert!(
        runtime
            .table_native_value_slot(table, &key("x"), NativeSlotKind::Number)
            .unwrap()
            .is_some()
    );
}

#[test]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn native_value_slots_of_one_small_table_address_their_own_entries() {
    let mut runtime = Runtime::default();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("x"), Value::Number(1.0))
        .unwrap();
    runtime
        .table_set(table, &Value::from("y"), Value::Number(2.0))
        .unwrap();
    let x = runtime
        .table_native_value_slot(table, &Value::from("x"), NativeSlotKind::Number)
        .unwrap()
        .unwrap();
    let y = runtime
        .table_native_value_slot(table, &Value::from("y"), NativeSlotKind::Number)
        .unwrap()
        .unwrap();
    assert_ne!(x.as_ptr(), y.as_ptr());
    // The first address still names x after the second lookup.
    assert!(matches!(
        runtime.table_get(table, &Value::from("x")),
        Ok(Value::Number(value)) if value == 1.0
    ));
    let again = runtime
        .table_native_value_slot(table, &Value::from("x"), NativeSlotKind::Number)
        .unwrap()
        .unwrap();
    assert_eq!(x.as_ptr(), again.as_ptr());
}

#[test]
fn string_key_write_matches_the_general_write() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let key = crate::string::LuaString::from("field");
    // Without a metatable the write stores, replaces, and deletes.
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Number(1.0)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Number(2.0)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field")),
        Ok(Value::Number(2.0))
    );
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Nil),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field")),
        Ok(Value::Nil)
    );
    // A metatable with `__newindex` takes an absent key but not a present one.
    let metatable = runtime.create_table().unwrap();
    let handler = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Table(handler))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Number(3.0)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field")),
        Ok(Value::Nil)
    );
    runtime
        .table_set(table, &Value::from("field"), Value::Number(4.0))
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Number(5.0)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field")),
        Ok(Value::Number(5.0))
    );
    // A stored nil counts as absent.
    let nil_key = crate::string::LuaString::from("nil_field");
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &nil_key, Value::Number(1.0)),
        Ok(false)
    );
    runtime.table_set_readonly(table, true).unwrap();
    assert!(
        runtime
            .table_set_string_if_no_newindex(table, &key, Value::Number(6.0))
            .is_err()
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field")),
        Ok(Value::Number(5.0))
    );
}

#[test]
fn string_key_write_updates_large_tables_and_ignores_newindex_for_present_keys() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in 0..40 {
        runtime
            .table_set(
                table,
                &Value::from(format!("key{index}").as_str()),
                Value::Number(0.0),
            )
            .unwrap();
    }
    assert!(runtime.table(table).unwrap().entries.is_indexed());
    let key = crate::string::LuaString::from("key17");
    let inner = runtime.create_table().unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Table(inner)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("key17")),
        Ok(Value::Table(inner))
    );
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Boolean(true))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &key, Value::Number(9.0)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("key17")),
        Ok(Value::Number(9.0))
    );
    let absent = crate::string::LuaString::from("absent");
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &absent, Value::Number(1.0)),
        Ok(false)
    );
}

#[test]
fn string_holder_follows_table_valued_index_chains_to_the_first_value() {
    let mut runtime = Runtime::new();
    let key = crate::string::LuaString::from("x");
    let index = Value::from("__index");
    let mut chain = |parent: Option<TableHandle>| {
        let table = runtime.create_table().unwrap();
        if let Some(parent) = parent {
            let meta = runtime.create_table().unwrap();
            runtime
                .table_set(meta, &index, Value::Table(parent))
                .unwrap();
            runtime.set_table_metatable(table, Some(meta)).unwrap();
        }
        table
    };
    let base = chain(None);
    let middle = chain(Some(base));
    let object = chain(Some(middle));
    drop(chain);
    runtime
        .table_set(base, &Value::from("x"), Value::Number(1.0))
        .unwrap();
    // No value anywhere before `base`: three tables, two hops.
    assert_eq!(
        runtime.table_string_holder(object, &key, 4),
        Some((base, 2))
    );
    assert_eq!(
        runtime.table_string_holder(object, &key, 2),
        Some((base, 2))
    );
    // A hop budget that ends before the holder gives up.
    assert_eq!(runtime.table_string_holder(object, &key, 1), None);
    assert_eq!(runtime.table_string_holder(base, &key, 0), Some((base, 0)));
    // A value nearer the receiver shadows the one behind it, a nil does not.
    runtime
        .table_set(middle, &Value::from("x"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_string_holder(object, &key, 4),
        Some((base, 2))
    );
    runtime
        .table_set(middle, &Value::from("x"), Value::Boolean(false))
        .unwrap();
    assert_eq!(
        runtime.table_string_holder(object, &key, 4),
        Some((middle, 1))
    );
    // A key nobody has, a missing `__index` and a function handler end it.
    let other = crate::string::LuaString::from("y");
    assert_eq!(runtime.table_string_holder(object, &other, 4), None);
    let handler = runtime
        .register_native(|_, _| Ok(vec![Value::Nil]))
        .unwrap();
    let meta = runtime.create_table().unwrap();
    runtime
        .table_set(meta, &index, Value::NativeFunction(handler))
        .unwrap();
    let guarded = runtime.create_table().unwrap();
    runtime.set_table_metatable(guarded, Some(meta)).unwrap();
    assert_eq!(runtime.table_string_holder(guarded, &key, 4), None);
}

#[test]
fn many_native_value_slots_agree_with_one_lookup_each() {
    let mut runtime = Runtime::default();
    // Once as a small ordered table and once past the hashed threshold.
    for extra in [0_i64, 40] {
        let table = runtime.create_table().unwrap();
        for (name, value) in [
            ("x", Value::Number(1.0)),
            ("n", Value::Integer(3)),
            ("on", Value::Boolean(false)),
            ("s", Value::from("text")),
            ("t", Value::Table(table)),
        ] {
            runtime.table_set(table, &Value::from(name), value).unwrap();
        }
        for index in 0..extra {
            runtime
                .table_set(
                    table,
                    &Value::from(format!("filler{index}").as_str()),
                    Value::Number(0.0),
                )
                .unwrap();
        }
        // A key allocated apart from the table's own still matches by content.
        let wanted: Vec<(Value, NativeSlotKind)> = [
            ("x", NativeSlotKind::Number),
            ("x", NativeSlotKind::Integer),
            ("n", NativeSlotKind::Integer),
            ("on", NativeSlotKind::Boolean),
            ("s", NativeSlotKind::Number),
            ("t", NativeSlotKind::Table),
            ("missing", NativeSlotKind::Number),
        ]
        .into_iter()
        .map(|(name, kind)| (Value::from(String::from(name).as_str()), kind))
        .collect();
        let mut found = vec![0_usize; wanted.len()];
        // Hints that are wrong, stale or out of range change nothing.
        for hints in [
            vec![0; 7],
            vec![3; 7],
            vec![99; 7],
            vec![0, 1, 2, 3, 4, 5, 6],
        ] {
            let mut hints = hints;
            let mut again = vec![0_usize; wanted.len()];
            runtime
                .table_native_value_slots(table, &wanted, &mut hints, &mut again)
                .unwrap();
            if found.iter().all(|address| *address == 0) {
                found = again;
            } else {
                assert_eq!(found, again, "hints {hints:?}");
            }
        }
        for ((key, kind), address) in wanted.iter().zip(&found) {
            let single = runtime
                .table_native_value_slot(table, key, *kind)
                .unwrap()
                .map_or(0, |slot| slot.as_ptr() as usize);
            assert_eq!(*address, single, "{key:?} as {kind:?} with {extra} extra");
        }
        assert_ne!(found[0], 0);
        assert_eq!(found[1], 0);
        assert_eq!(found[4], 0);
        assert_eq!(found[6], 0);
        assert_eq!(
            found.iter().filter(|address| **address != 0).count(),
            4,
            "x, n, on and t are present with their kind"
        );
        // Distinct slots have distinct addresses.
        let mut present: Vec<usize> = found.iter().copied().filter(|a| *a != 0).collect();
        present.sort_unstable();
        present.dedup();
        assert_eq!(present.len(), 4);
    }
}

#[test]
fn a_read_only_table_gets_no_native_value_slots() {
    let mut runtime = Runtime::default();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("x"), Value::Number(1.0))
        .unwrap();
    runtime.table_set_readonly(table, true).unwrap();
    let wanted = [(Value::from("x"), NativeSlotKind::Number)];
    let mut found = [7_usize];
    runtime
        .table_native_value_slots(table, &wanted, &mut [0], &mut found)
        .unwrap();
    assert_eq!(found, [0]);
}

/// Applies `writes` to two fresh tables, one through the general raw write
/// and one through the array store of the fast loop, and checks that
/// contents, length, `next` order, and GC accounting agree.
fn assert_fast_store_matches_raw_write(writes: &[(Value, Value)]) {
    let mut raw = Runtime::new();
    let mut fast = Runtime::new();
    let raw_table = raw.create_table().unwrap();
    let fast_table = fast.create_table().unwrap();
    for (key, value) in writes {
        raw.table_set(raw_table, key, value.clone()).unwrap();
        // The fast loop tries the borrowing store first.
        if !fast.table_set_dense_scalar(fast_table, key, value) {
            assert_eq!(
                fast.table_set_if_no_newindex(fast_table, key, value.clone()),
                Ok(true)
            );
        }
    }
    assert_eq!(raw.gc_count_bytes(), fast.gc_count_bytes());
    assert_eq!(raw.allocation_serial(), fast.allocation_serial());
    assert_eq!(raw.table_length(raw_table), fast.table_length(fast_table));
    assert_eq!(
        raw.table_next(raw_table, &Value::Nil),
        fast.table_next(fast_table, &Value::Nil)
    );
    for index in 0..40 {
        let key = Value::Number(f64::from(index));
        assert_eq!(
            raw.table_get(raw_table, &key),
            fast.table_get(fast_table, &key),
            "index {index}"
        );
    }
}

#[test]
fn fast_array_store_appends_overwrites_and_keeps_the_border() {
    let number = Value::Number;
    let mut writes = Vec::new();
    for index in 1..=20 {
        writes.push((number(f64::from(index)), number(f64::from(index) * 2.0)));
    }
    // Overwrites, a boolean, an integer key, and stores past the border.
    writes.push((number(3.0), Value::Boolean(true)));
    writes.push((Value::Integer(4), number(0.5)));
    writes.push((number(25.0), number(1.0)));
    writes.push((number(21.0), number(1.0)));
    writes.push((number(2.0), Value::Nil));
    writes.push((number(2.0), number(9.0)));
    writes.push((number(1.5), number(1.0)));
    writes.push((number(0.0), number(1.0)));
    writes.push((number(-1.0), number(1.0)));
    assert_fast_store_matches_raw_write(&writes);
}

#[test]
fn fast_array_store_replaces_owned_slots_and_leaves_other_tables_alone() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::from("text"))
        .unwrap();
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(4.0)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(1.0)),
        Ok(Value::Number(4.0))
    );
    // A hash entry sends the store down the general path, which still lands.
    runtime
        .table_set(table, &Value::from("field"), Value::Number(1.0))
        .unwrap();
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Number(2.0), Value::Boolean(true)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(2.0)),
        Ok(Value::Boolean(true))
    );
    // The borrowing store declines a string slot, a hash entry, a metatable,
    // and a readonly table without changing anything.
    let owned = runtime.create_table().unwrap();
    runtime
        .table_set(owned, &Value::Number(1.0), Value::from("text"))
        .unwrap();
    assert!(!runtime.table_set_dense_scalar(owned, &Value::Number(1.0), &Value::Number(1.0)));
    assert_eq!(
        runtime.table_get(owned, &Value::Number(1.0)),
        Ok(Value::from("text"))
    );
    assert!(!runtime.table_set_dense_scalar(table, &Value::Number(3.0), &Value::Number(1.0)));
    let guarded = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(guarded), Some(metatable))
        .unwrap();
    assert!(!runtime.table_set_dense_scalar(guarded, &Value::Number(1.0), &Value::Number(1.0)));
    assert_eq!(
        runtime.table_get(guarded, &Value::Number(1.0)),
        Ok(Value::Nil)
    );
    // A readonly table refuses with its usual error.
    let other = runtime.create_table().unwrap();
    runtime.table_set_readonly(other, true).unwrap();
    assert!(!runtime.table_set_dense_scalar(other, &Value::Number(1.0), &Value::Number(1.0)));
    assert!(
        runtime
            .table_set_if_no_newindex(other, &Value::Number(1.0), Value::Number(1.0))
            .is_err()
    );
}

#[test]
fn dense_reference_stores_write_array_slots_and_decline_other_stores() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let other = runtime.create_table().unwrap();
    let text = Value::from("x");
    // Appends at the border, then an overwrite that releases the old string.
    for index in 1..=3 {
        assert!(runtime.table_set_dense_reference(table, &Value::Number(f64::from(index)), &text));
    }
    assert!(runtime.table_set_dense_reference(table, &Value::Integer(2), &Value::Table(other)));
    assert_eq!(
        runtime.table_get(table, &Value::Number(2.0)),
        Ok(Value::Table(other))
    );
    assert_eq!(
        runtime.table_get(table, &Value::Number(3.0)),
        Ok(text.clone())
    );
    // A gap, a string key, and a scalar value take the general path.
    assert!(!runtime.table_set_dense_reference(table, &Value::Number(9.0), &text));
    assert!(!runtime.table_set_dense_reference(table, &Value::from("k"), &text));
    assert!(!runtime.table_set_dense_reference(table, &Value::Number(4.0), &Value::Number(1.0)));
    assert_eq!(
        runtime.table_get(table, &Value::Number(4.0)),
        Ok(Value::Nil)
    );
    // A metatable and a readonly table decline, changing nothing.
    runtime.table_set_readonly(table, true).unwrap();
    assert!(!runtime.table_set_dense_reference(table, &Value::Number(1.0), &text));
    runtime.table_set_readonly(table, false).unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(other))
        .unwrap();
    assert!(!runtime.table_set_dense_reference(table, &Value::Number(4.0), &text));
    assert_eq!(
        runtime.table_get(table, &Value::Number(4.0)),
        Ok(Value::Nil)
    );
}

#[test]
fn index_walk_after_a_raw_miss_follows_the_same_chain_as_the_full_walk() {
    let mut runtime = Runtime::new();
    let base = runtime.create_table().unwrap();
    runtime
        .table_set(base, &Value::from("inherited"), Value::Number(7.0))
        .unwrap();
    let meta = runtime.create_table().unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Table(base))
        .unwrap();
    let object = runtime.create_table().unwrap();
    runtime
        .table_set(object, &Value::from("own"), Value::Number(1.0))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(object), Some(meta))
        .unwrap();
    for name in ["inherited", "missing"] {
        let key = Value::from(name);
        let full = runtime
            .table_index_walk(object, &key)
            .map(std::result::Result::ok);
        let after_miss = runtime
            .table_index_walk_after_miss(object, &key)
            .map(std::result::Result::ok);
        assert_eq!(full, after_miss, "{name}");
    }
    let inherited = Value::from("inherited");
    assert_eq!(
        runtime
            .table_index_walk_after_miss(object, &inherited)
            .and_then(Result::ok),
        Some(&Value::Number(7.0))
    );
    // Without a metatable the walk ends in nil whatever the table holds.
    assert_eq!(
        runtime
            .table_index_walk_after_miss(base, &Value::from("missing"))
            .and_then(Result::ok),
        Some(&Value::Nil)
    );
}

/// An object whose class chain is `depth` tables deep, the value at the end.
fn inherited_chain(runtime: &mut Runtime, depth: usize) -> (TableHandle, Vec<TableHandle>) {
    let end = runtime.create_table().unwrap();
    runtime
        .table_set(end, &Value::from("field"), Value::Number(5.0))
        .unwrap();
    let mut holders = vec![end];
    for _ in 1..depth {
        let meta = runtime.create_table().unwrap();
        runtime
            .table_set(meta, &Value::from("__index"), Value::Table(holders[0]))
            .unwrap();
        let mid = runtime.create_table().unwrap();
        runtime
            .raw_set_metatable(&Value::Table(mid), Some(meta))
            .unwrap();
        holders.insert(0, mid);
    }
    let meta = runtime.create_table().unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Table(holders[0]))
        .unwrap();
    let object = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(object), Some(meta))
        .unwrap();
    (object, holders)
}

#[test]
fn inherited_reads_hit_after_one_resolve_at_every_depth() {
    for depth in 1..=3 {
        let mut runtime = Runtime::new();
        let (object, _) = inherited_chain(&mut runtime, depth);
        let key = crate::string::LuaString::from("field");
        assert_eq!(runtime.table_inherited_get(object, &key), None);
        assert_eq!(
            runtime.table_inherited_resolve(object, &key),
            Some(&Value::Number(5.0)),
            "depth {depth}"
        );
        assert_eq!(
            runtime.table_inherited_get(object, &key),
            Some(&Value::Number(5.0)),
            "depth {depth}"
        );
        // A key found nowhere caches as nil.
        let absent = crate::string::LuaString::from("absent");
        assert_eq!(
            runtime.table_inherited_resolve(object, &absent),
            Some(&Value::Nil)
        );
        assert_eq!(
            runtime.table_inherited_get(object, &absent),
            Some(&Value::Nil)
        );
    }
}

#[test]
fn inherited_reads_follow_every_change_to_the_chain() {
    let key = crate::string::LuaString::from("field");
    for depth in 1..=3 {
        // Overwriting the value in the table that holds it shows at once.
        let mut runtime = Runtime::new();
        let (object, holders) = inherited_chain(&mut runtime, depth);
        runtime.table_inherited_resolve(object, &key);
        let end = *holders.last().unwrap();
        runtime
            .table_set(end, &Value::from("field"), Value::Number(6.0))
            .unwrap();
        assert_eq!(
            runtime.table_inherited_get(object, &key),
            Some(&Value::Number(6.0))
        );
        // Removing it leaves no stale answer, and the full walk takes over.
        runtime
            .table_set(end, &Value::from("field"), Value::Nil)
            .unwrap();
        assert_eq!(runtime.table_inherited_get(object, &key), None);

        // A table in front of the holder shadows it.
        runtime
            .table_set(end, &Value::from("field"), Value::Number(5.0))
            .unwrap();
        runtime.table_inherited_resolve(object, &key);
        assert!(runtime.table_inherited_get(object, &key).is_some());
        runtime
            .table_set(holders[0], &Value::from("field"), Value::Number(9.0))
            .unwrap();
        if depth > 1 {
            assert_eq!(runtime.table_inherited_get(object, &key), None, "{depth}");
            assert_eq!(
                runtime.table_inherited_resolve(object, &key),
                Some(&Value::Number(9.0))
            );
        }

        // A new metatable, or none, changes the answer.
        let other = runtime.create_table().unwrap();
        runtime
            .raw_set_metatable(&Value::Table(object), Some(other))
            .unwrap();
        assert_eq!(runtime.table_inherited_get(object, &key), None);
        runtime
            .raw_set_metatable(&Value::Table(object), None)
            .unwrap();
        assert_eq!(runtime.table_inherited_get(object, &key), None);
    }
}

#[test]
fn inherited_reads_stop_at_a_replaced_or_removed_index() {
    let mut runtime = Runtime::new();
    let (object, _) = inherited_chain(&mut runtime, 1);
    let key = crate::string::LuaString::from("field");
    runtime.table_inherited_resolve(object, &key);
    let meta = runtime
        .raw_metatable(&Value::Table(object))
        .unwrap()
        .unwrap();
    let replacement = runtime.create_table().unwrap();
    runtime
        .table_set(replacement, &Value::from("field"), Value::Number(8.0))
        .unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Table(replacement))
        .unwrap();
    assert_eq!(runtime.table_inherited_get(object, &key), None);
    assert_eq!(
        runtime.table_inherited_resolve(object, &key),
        Some(&Value::Number(8.0))
    );
    runtime
        .table_set(meta, &Value::from("__index"), Value::Nil)
        .unwrap();
    assert_eq!(runtime.table_inherited_get(object, &key), None);
    assert_eq!(
        runtime.table_inherited_resolve(object, &key),
        Some(&Value::Nil)
    );
}

#[test]
fn inherited_reads_never_cache_a_handler_function() {
    let mut runtime = Runtime::new();
    let (object, _) = inherited_chain(&mut runtime, 1);
    let meta = runtime
        .raw_metatable(&Value::Table(object))
        .unwrap()
        .unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Boolean(true))
        .unwrap();
    let key = crate::string::LuaString::from("field");
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
    assert_eq!(runtime.table_inherited_get(object, &key), None);
    // A loop is not cached either.
    runtime
        .table_set(meta, &Value::from("__index"), Value::Table(object))
        .unwrap();
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
}

#[test]
fn inherited_reads_know_a_closure_handler_on_the_receivers_own_metatable() {
    let mut runtime = Runtime::new();
    let chunk = Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]));
    let handler = runtime
        .create_closure(Arc::clone(&chunk), Vec::new())
        .unwrap();
    let meta = runtime.create_table().unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Closure(handler))
        .unwrap();
    let object = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(object), Some(meta))
        .unwrap();
    let key = LuaString::from("field");
    assert_eq!(runtime.table_inherited_handler(object, &key), None);
    // The handler is no value: reads still go the full way.
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
    assert_eq!(runtime.table_inherited_get(object, &key), None);
    assert_eq!(runtime.table_inherited_handler(object, &key), Some(handler));

    // Replacing the handler, or the metatable, drops the answer.
    let other = runtime.create_closure(chunk, Vec::new()).unwrap();
    runtime
        .table_set(meta, &Value::from("__index"), Value::Closure(other))
        .unwrap();
    assert_eq!(runtime.table_inherited_handler(object, &key), None);
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
    assert_eq!(runtime.table_inherited_handler(object, &key), Some(other));
    let fresh = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(object), Some(fresh))
        .unwrap();
    assert_eq!(runtime.table_inherited_handler(object, &key), None);

    // A handler behind a table hop is called with the inner table, so it is
    // not the receiver's own and is not kept.
    let inner_meta = runtime.create_table().unwrap();
    runtime
        .table_set(inner_meta, &Value::from("__index"), Value::Closure(other))
        .unwrap();
    let inner = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(inner), Some(inner_meta))
        .unwrap();
    runtime
        .table_set(fresh, &Value::from("__index"), Value::Table(inner))
        .unwrap();
    assert_eq!(runtime.table_inherited_resolve(object, &key), None);
    assert_eq!(runtime.table_inherited_handler(object, &key), None);
}

/// Walks `table` with the cursor step (and the general step when the cursor
/// declines), checking each pair against the general step alone. `mutate`
/// runs after every pair with the key just returned.
fn walk_against_general_step(
    runtime: &mut Runtime,
    table: TableHandle,
    mut mutate: impl FnMut(&mut Runtime, &Value),
) -> Vec<Value> {
    let mut keys = Vec::new();
    let mut previous = Value::Nil;
    loop {
        let expected = runtime.table_next(table, &previous).unwrap();
        let fast = runtime.table_next_fast(table, &previous);
        if let Some(fast) = fast {
            assert_eq!(fast, expected, "after {previous:?}");
        }
        if expected.0 == Value::Nil {
            return keys;
        }
        mutate(runtime, &expected.0);
        keys.push(expected.0.clone());
        previous = expected.0;
    }
}

fn mixed_table(runtime: &mut Runtime) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for index in 1..=10 {
        runtime
            .table_set(table, &Value::Number(f64::from(index)), Value::Number(1.0))
            .unwrap();
    }
    for index in 0..100 {
        runtime
            .table_set(
                table,
                &Value::from(format!("k{index}").as_str()),
                Value::Number(f64::from(index)),
            )
            .unwrap();
    }
    for index in 0..20 {
        runtime
            .table_set(
                table,
                &Value::Number(f64::from(1000 + 7 * index)),
                Value::Boolean(true),
            )
            .unwrap();
    }
    runtime
        .table_set(table, &Value::Number(0.5), Value::Boolean(false))
        .unwrap();
    table
}

#[test]
fn next_cursor_agrees_with_the_general_step() {
    let mut runtime = Runtime::new();
    let table = mixed_table(&mut runtime);
    let first = walk_against_general_step(&mut runtime, table, |_, _| {});
    assert_eq!(first.len(), 131);
    // A second walk runs on the cursor the first one built.
    let second = walk_against_general_step(&mut runtime, table, |_, _| {});
    assert_eq!(first, second);
}

#[test]
fn next_cursor_follows_overwrites_and_gives_way_to_deletions() {
    let mut runtime = Runtime::new();
    let table = mixed_table(&mut runtime);
    // Warm the cursor, then overwrite every value mid-walk: same layout.
    walk_against_general_step(&mut runtime, table, |_, _| {});
    walk_against_general_step(&mut runtime, table, |runtime, key| {
        runtime.table_set(table, key, Value::Number(-1.0)).unwrap();
    });
    // Deleting the key just returned changes the layout under the cursor.
    let mut walked = 0;
    walk_against_general_step(&mut runtime, table, |runtime, key| {
        walked += 1;
        runtime.table_set(table, key, Value::Nil).unwrap();
    });
    assert_eq!(walked, 131);
    assert_eq!(
        runtime.table_next(table, &Value::Nil),
        Ok((Value::Nil, Value::Nil))
    );
}

fn dense_table(runtime: &mut Runtime, values: &[Value]) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for (index, value) in values.iter().enumerate() {
        runtime
            .table_set(table, &Value::Integer(index as i64 + 1), value.clone())
            .unwrap();
    }
    table
}

#[test]
fn scalar_overwrite_ignores_the_metatable() {
    let mut runtime = Runtime::default();
    let table = dense_table(
        &mut runtime,
        &[Value::Number(1.0), Value::Integer(2), Value::Boolean(true)],
    );
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Boolean(true))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    runtime
        .table_set(table, &Value::from("name"), Value::Number(5.0))
        .unwrap();
    runtime
        .table_set(table, &Value::from("name"), Value::Nil)
        .unwrap();
    assert!(runtime.table_set_dense_scalar(table, &Value::Number(1.0), &Value::Integer(9)));
    assert!(runtime.table_set_dense_scalar(table, &Value::Integer(2), &Value::Boolean(false)));
    assert!(runtime.table_set_dense_scalar(table, &Value::Integer(3), &Value::Number(0.5)));
    assert_eq!(
        runtime.table_get(table, &Value::Integer(1)).unwrap(),
        Value::Integer(9)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Integer(2)).unwrap(),
        Value::Boolean(false)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Integer(3)).unwrap(),
        Value::Number(0.5)
    );
    // An absent key still takes the general path past the metatable.
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(4), &Value::Number(1.0)));
}

#[test]
fn scalar_overwrite_matches_table_set_with_a_live_hash_part_and_deleted_array_keys() {
    let build = |runtime: &mut Runtime| {
        let values: Vec<Value> = (1..=5).map(Value::Integer).collect();
        let table = dense_table(runtime, &values);
        runtime
            .table_set(table, &Value::from("keep"), Value::Integer(1))
            .unwrap();
        runtime
            .table_set(table, &Value::Integer(4), Value::Nil)
            .unwrap();
        table
    };
    let mut fast = Runtime::default();
    let fast_table = build(&mut fast);
    let mut general = Runtime::default();
    let general_table = build(&mut general);

    assert!(fast.table_set_dense_scalar(fast_table, &Value::Integer(1), &Value::Number(7.5)));
    assert!(fast.table_set_dense_scalar(fast_table, &Value::Number(2.0), &Value::Boolean(true)));
    general
        .table_set(general_table, &Value::Integer(1), Value::Number(7.5))
        .unwrap();
    general
        .table_set(general_table, &Value::Integer(2), Value::Boolean(true))
        .unwrap();

    for key in [
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
        Value::Integer(5),
        Value::from("keep"),
    ] {
        assert_eq!(
            fast.table_get(fast_table, &key).unwrap(),
            general.table_get(general_table, &key).unwrap(),
            "key {key:?}"
        );
    }
    assert_eq!(
        fast.table_get(fast_table, &Value::Integer(1)).unwrap(),
        Value::Number(7.5)
    );
    assert_eq!(
        fast.table_length(fast_table).unwrap(),
        general.table_length(general_table).unwrap()
    );
}

#[test]
fn scalar_overwrite_declines_a_hole_in_the_array() {
    let mut runtime = Runtime::default();
    let table = dense_table(
        &mut runtime,
        &[
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
            Value::Number(4.0),
        ],
    );
    runtime
        .table_set(table, &Value::Integer(2), Value::Nil)
        .unwrap();
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(2), &Value::Number(1.0)));
    assert_eq!(
        runtime.table_get(table, &Value::Integer(2)).unwrap(),
        Value::Nil
    );
}

#[test]
fn scalar_overwrite_leaves_string_slots_and_replaces_table_slots() {
    let mut runtime = Runtime::default();
    let inner = runtime.create_table().unwrap();
    let table = dense_table(
        &mut runtime,
        &[Value::Number(1.0), Value::from("text"), Value::Table(inner)],
    );
    // A string slot owns a reference, so the general path keeps it.
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(2), &Value::Number(1.0)));
    assert_eq!(
        runtime.table_get(table, &Value::Integer(2)).unwrap(),
        Value::from("text")
    );
    // A table slot is replaced by the dense path, as before the overwrite path.
    assert!(runtime.table_set_dense_scalar(table, &Value::Integer(3), &Value::Number(2.0)));
    assert_eq!(
        runtime.table_get(table, &Value::Integer(3)).unwrap(),
        Value::Number(2.0)
    );
}

#[test]
fn scalar_overwrite_declines_non_keys_and_readonly_tables() {
    let mut runtime = Runtime::default();
    let table = dense_table(&mut runtime, &[Value::Number(1.0), Value::Number(2.0)]);
    assert!(!runtime.table_set_dense_scalar(table, &Value::Number(0.5), &Value::Number(1.0)));
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(0), &Value::Number(1.0)));
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(i64::MIN), &Value::Number(1.0)));

    runtime.table_set_readonly(table, true).unwrap();
    assert!(!runtime.table_set_dense_scalar(table, &Value::Integer(1), &Value::Number(7.0)));
    assert_eq!(
        runtime.table_get(table, &Value::Integer(1)).unwrap(),
        Value::Number(1.0)
    );
}

#[test]
fn next_cursor_declines_keys_it_cannot_place() {
    let mut runtime = Runtime::new();
    let table = mixed_table(&mut runtime);
    walk_against_general_step(&mut runtime, table, |_, _| {});
    // Warm: the cursor must not invent an answer for an absent key.
    for previous in [
        Value::from("absent"),
        Value::Number(99.0),
        Value::Boolean(true),
        Value::Table(table),
    ] {
        assert!(runtime.table_next_fast(table, &previous).is_none());
        assert!(runtime.table_next(table, &previous).is_err());
    }
}

#[test]
fn scalar_append_still_extends_a_plain_array() {
    let mut runtime = Runtime::default();
    let table = dense_table(&mut runtime, &[Value::Number(1.0)]);
    assert!(runtime.table_set_dense_scalar(table, &Value::Integer(2), &Value::Number(2.0)));
    assert!(runtime.table_set_dense_scalar(table, &Value::Number(3.0), &Value::Boolean(true)));
    assert_eq!(runtime.table_length(table).unwrap(), 3.0);
    assert_eq!(
        runtime.table_get(table, &Value::Integer(3)).unwrap(),
        Value::Boolean(true)
    );
}

#[test]
fn an_empty_site_never_hits_a_table_of_any_layout() {
    use crate::bytecode::FieldSite;
    use crate::string::LuaString;
    let mut runtime = Runtime::new();
    let site = FieldSite::default();
    // Ordered, then enough keys to leave the ordered layout.
    let table = runtime.create_table().unwrap();
    for round in 0..64 {
        let key = LuaString::from(format!("k{round}").as_str());
        runtime
            .table_set_string_if_no_newindex(table, &key, Value::Integer(round))
            .unwrap();
        assert!(runtime.table_site_get(table, &site).is_none());
    }
}

#[test]
fn a_probe_of_an_empty_site_hands_back_the_table_and_finds_no_other_holder() {
    use crate::bytecode::FieldSite;
    use crate::runtime::{SiteProbe, StringReadCache};
    use crate::string::LuaString;
    let mut runtime = Runtime::new();
    let site = FieldSite::default();
    let table = runtime.create_table().unwrap();
    let key = LuaString::from("x");
    runtime
        .table_set_string_if_no_newindex(table, &key, Value::Integer(7))
        .unwrap();
    let mut cache = StringReadCache::default();
    let SiteProbe::Miss(found) = runtime.table_site_probe(&site, &Value::Table(table)) else {
        panic!("an empty site misses a live table");
    };
    assert_eq!(
        runtime.found_table_get_string_cached_hit(found, &key, &mut cache),
        Some(&Value::Integer(7))
    );
    assert!(matches!(
        runtime.table_site_probe(&site, &Value::Integer(1)),
        SiteProbe::Absent
    ));
}

/// A metatable that stores can cache the absence of `__newindex` for, and a
/// table that uses it.
fn newindex_pair(runtime: &mut Runtime) -> (TableHandle, TableHandle) {
    let table = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(metatable))
        .unwrap();
    (table, metatable)
}

fn newindex_handler(runtime: &mut Runtime) -> Value {
    fn handler(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(Vec::new())
    }
    Value::NativeFunction(runtime.register_native(handler).unwrap())
}

#[test]
fn metatable_without_newindex_takes_absent_keys_raw() {
    let mut runtime = Runtime::new();
    let (table, _) = newindex_pair(&mut runtime);
    let name = LuaString::from("field");
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &name, Value::Integer(1)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::from("other"), Value::Integer(2)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Integer(5), Value::Integer(3)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::Integer(5)),
        Ok(Value::Integer(3))
    );
    assert_eq!(
        runtime.table_get_string(table, &name),
        Ok(Value::Integer(1))
    );
}

#[test]
fn newindex_added_by_any_path_stops_raw_stores_at_once() {
    type Add = fn(&mut Runtime, TableHandle, Value);
    let adds: [(&str, Add); 3] = [
        ("table_set", |runtime, metatable, handler| {
            runtime
                .table_set(metatable, &Value::from("__newindex"), handler)
                .unwrap();
        }),
        ("string store", |runtime, metatable, handler| {
            let key = LuaString::from("__newindex");
            assert_eq!(
                runtime.table_set_string_if_no_newindex(metatable, &key, handler),
                Ok(true)
            );
        }),
        ("generic store", |runtime, metatable, handler| {
            assert_eq!(
                runtime.table_set_if_no_newindex(metatable, &Value::from("__newindex"), handler),
                Ok(true)
            );
        }),
    ];
    for (label, add) in adds {
        let mut runtime = Runtime::new();
        let (table, metatable) = newindex_pair(&mut runtime);
        // Fill the cache through both entry points.
        assert_eq!(
            runtime.table_set_string_if_no_newindex(table, &LuaString::from("a"), Value::Nil),
            Ok(true),
            "{label}"
        );
        assert_eq!(
            runtime.table_set_if_no_newindex(table, &Value::Integer(1), Value::Integer(1)),
            Ok(true),
            "{label}"
        );
        let handler = newindex_handler(&mut runtime);
        add(&mut runtime, metatable, handler);
        assert_eq!(
            runtime.table_set_string_if_no_newindex(
                table,
                &LuaString::from("b"),
                Value::Integer(1)
            ),
            Ok(false),
            "{label}"
        );
        assert_eq!(
            runtime.table_set_if_no_newindex(table, &Value::from("c"), Value::Integer(1)),
            Ok(false),
            "{label}"
        );
        assert_eq!(
            runtime.table_set_if_no_newindex(table, &Value::Integer(9), Value::Integer(1)),
            Ok(false),
            "{label}"
        );
    }
}

#[test]
fn newindex_set_to_nil_counts_as_absent() {
    let mut runtime = Runtime::new();
    let (table, metatable) = newindex_pair(&mut runtime);
    let handler = newindex_handler(&mut runtime);
    runtime
        .table_set(metatable, &Value::from("__newindex"), handler)
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("a"), Value::Integer(1)),
        Ok(false)
    );
    runtime
        .table_set(metatable, &Value::from("__newindex"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("a"), Value::Integer(1)),
        Ok(true)
    );
}

#[test]
fn present_key_stores_raw_whatever_newindex_says() {
    let mut runtime = Runtime::new();
    let (table, metatable) = newindex_pair(&mut runtime);
    let name = LuaString::from("a");
    runtime
        .table_set(table, &Value::from("a"), Value::Integer(1))
        .unwrap();
    let handler = newindex_handler(&mut runtime);
    runtime
        .table_set(metatable, &Value::from("__newindex"), handler)
        .unwrap();
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &name, Value::Integer(2)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get_string(table, &name),
        Ok(Value::Integer(2))
    );
}

#[test]
fn table_that_is_its_own_metatable_sees_a_raw_newindex_store() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(table), Some(table))
        .unwrap();
    // Caches the absence of __newindex in the table itself, then adds the
    // very key it cached.
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("a"), Value::Integer(1)),
        Ok(true)
    );
    let handler = newindex_handler(&mut runtime);
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("__newindex"), handler),
        Ok(true)
    );
    assert_eq!(
        runtime.table_set_string_if_no_newindex(table, &LuaString::from("b"), Value::Integer(1)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_set_if_no_newindex(table, &Value::Integer(4), Value::Integer(1)),
        Ok(false)
    );
}

#[test]
fn metatable_field_added_after_the_cache_protects_the_metatable() {
    let mut runtime = Runtime::new();
    let (table, metatable) = newindex_pair(&mut runtime);
    // Caches the absence of __metatable.
    assert_eq!(
        runtime.protected_metatable(&Value::Table(table)),
        Ok(Value::Table(metatable))
    );
    runtime.set_table_metatable(table, Some(metatable)).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__metatable"),
            Value::from("locked"),
        )
        .unwrap();
    assert_eq!(
        runtime.protected_metatable(&Value::Table(table)),
        Ok(Value::from("locked"))
    );
    let error = runtime.set_table_metatable(table, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot change a protected metatable"),
        "{error}"
    );
}

#[test]
fn gc_field_added_after_the_cache_is_seen_by_the_finalizer_check() {
    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    assert_eq!(runtime.metatable_has_finalizer(metatable), Ok(false));
    let handler = newindex_handler(&mut runtime);
    runtime
        .table_set(metatable, &Value::from("__gc"), handler)
        .unwrap();
    assert_eq!(runtime.metatable_has_finalizer(metatable), Ok(true));
    runtime
        .table_set(metatable, &Value::from("__gc"), Value::Nil)
        .unwrap();
    assert_eq!(runtime.metatable_has_finalizer(metatable), Ok(false));
}

/// A hashed table of `count` string keys `k1`..`kN`, each holding its index.
fn hashed_string_table(runtime: &mut Runtime, count: i64) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for index in 1..=count {
        runtime
            .table_set(
                table,
                &Value::from(format!("k{index}").as_str()),
                Value::Integer(index),
            )
            .unwrap();
    }
    assert!(runtime.table(table).unwrap().entries.is_indexed());
    table
}

/// The keys and tombstones a hashed table holds, and its live count.
fn indexed_sizes(runtime: &Runtime, table: TableHandle) -> (usize, usize, usize) {
    let stored = &runtime.table(table).unwrap().entries;
    let TableEntries::Other(other) = stored else {
        panic!("a hashed table")
    };
    let OtherRef::Indexed(entries) = other.view() else {
        panic!("a hashed table")
    };
    (entries.values.len(), entries.order().len(), stored.len())
}

/// Every key a full `next` traversal visits, through either entry point.
fn traversal(runtime: &mut Runtime, table: TableHandle, fast: bool) -> Vec<Value> {
    let mut keys = Vec::new();
    let mut key = Value::Nil;
    loop {
        let step = if fast {
            runtime.table_next_fast(table, &key)
        } else {
            None
        };
        let (next, value) = match step {
            Some(step) => step,
            None => runtime.table_next(table, &key).unwrap(),
        };
        if matches!(next, Value::Nil) {
            return keys;
        }
        assert!(!matches!(value, Value::Nil), "a tombstone was visited");
        keys.push(next.clone());
        key = next;
    }
}

#[test]
fn a_deleted_string_key_of_a_hashed_table_stays_as_a_tombstone_until_reinserted() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = Value::from("k5");
    runtime.table_set(table, &key, Value::Nil).unwrap();

    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 199));
    assert!(runtime.table(table).unwrap().deleted_entries.is_empty());
    assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Nil);
    for fast in [false, true] {
        let keys = traversal(&mut runtime, table, fast);
        assert_eq!(keys.len(), 199);
        assert!(!keys.contains(&key));
    }

    // Deleting it again changes nothing, and the reinsert reuses the slot.
    runtime.table_set(table, &key, Value::Nil).unwrap();
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 199));
    runtime.table_set(table, &key, Value::Integer(55)).unwrap();
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 200));
    assert_eq!(runtime.table_get(table, &key).unwrap(), Value::Integer(55));
    assert_eq!(traversal(&mut runtime, table, true).len(), 200);
}

#[test]
fn next_resumes_from_a_tombstoned_key_and_rejects_a_key_never_stored() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 100);
    // Clear every key as the traversal reaches it, as `pairs` loops may.
    let mut visited = 0;
    let mut key = Value::Nil;
    loop {
        let (next, _) = runtime.table_next(table, &key).unwrap();
        if matches!(next, Value::Nil) {
            break;
        }
        runtime.table_set(table, &next, Value::Nil).unwrap();
        visited += 1;
        key = next;
    }
    assert_eq!(visited, 100);
    assert_eq!(indexed_sizes(&runtime, table).2, 0);
    assert_eq!(
        runtime.table_next(table, &Value::Nil).unwrap(),
        (Value::Nil, Value::Nil)
    );
    assert_eq!(
        runtime.table_next(table, &Value::from("k1")).unwrap(),
        (Value::Nil, Value::Nil)
    );
    assert!(runtime.table_next(table, &Value::from("absent")).is_err());
}

#[test]
fn a_new_key_compacts_tombstones_that_outnumber_the_live_entries() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    for index in 1..=150 {
        runtime
            .table_set(
                table,
                &Value::from(format!("k{index}").as_str()),
                Value::Nil,
            )
            .unwrap();
    }
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 50));
    // Reviving a tombstone is no new key, so nothing is compacted yet.
    runtime
        .table_set(table, &Value::from("k1"), Value::Integer(1))
        .unwrap();
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 51));
    runtime
        .table_set(table, &Value::from("fresh"), Value::Integer(0))
        .unwrap();
    assert_eq!(indexed_sizes(&runtime, table), (52, 52, 52));
    let mut keys = traversal(&mut runtime, table, true);
    keys.sort_by_key(|key| format!("{key:?}"));
    assert_eq!(keys.len(), 52);
    assert_eq!(
        runtime.table_get(table, &Value::from("k2")).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("k151")).unwrap(),
        Value::Integer(151)
    );
}

/// The entry `table_next_ref` gives as `table_next` gives it, or `None` where
/// the cursor declined.
fn next_by_reference(
    runtime: &mut Runtime,
    table: TableHandle,
    previous: &Value,
) -> Option<(Value, Value)> {
    use crate::runtime::{NextKey, NextRef};
    Some(match runtime.table_next_ref(table, previous)? {
        NextRef::End => (Value::Nil, Value::Nil),
        NextRef::Array(index, value) => (
            // The key's representation is the profile's.
            Value::Integer(i64::try_from(index).unwrap()),
            value.clone(),
        ),
        NextRef::Hash(NextKey::String(text), value) => (Value::String(text.clone()), value.clone()),
        NextRef::Hash(NextKey::Plain(key), value) => (key, value.clone()),
    })
}

#[test]
fn table_next_ref_agrees_with_table_next_through_holes_and_tombstones() {
    for tagged in [false, true] {
        let mut runtime = Runtime::new();
        runtime.set_tagged_integers(tagged);
        let table = runtime.create_table().unwrap();
        // An array part with holes, integer keys past it, floats, booleans
        // and many string keys, some deleted during the traversal.
        for index in 1..=20 {
            runtime
                .table_set(table, &Value::Integer(index), Value::Integer(index * 2))
                .unwrap();
        }
        for index in [3, 4, 11] {
            runtime
                .table_set(table, &Value::Integer(index), Value::Nil)
                .unwrap();
        }
        for index in 0..60 {
            runtime
                .table_set(
                    table,
                    &Value::from(format!("key{index}").as_str()),
                    Value::Integer(index),
                )
                .unwrap();
        }
        runtime
            .table_set(table, &Value::Integer(5000), Value::Integer(1))
            .unwrap();
        runtime
            .table_set(table, &Value::Number(2.5), Value::Integer(2))
            .unwrap();
        runtime
            .table_set(table, &Value::Boolean(true), Value::Integer(3))
            .unwrap();
        for round in 0..4 {
            let mut key = Value::Nil;
            let mut seen = 0;
            loop {
                let expected = runtime.table_next(table, &key).unwrap();
                // The cursor may decline (a young table, a changed layout):
                // then the general step stands in, as the helpers do.
                let actual =
                    next_by_reference(&mut runtime, table, &key).unwrap_or(expected.clone());
                // An integer key of the array part is a number on the
                // profiles without integers.
                let same_key = match (&actual.0, &expected.0) {
                    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
                    (Value::Integer(a), Value::Number(b)) if !tagged => *a as f64 == *b,
                    (a, b) => a == b,
                };
                assert!(
                    same_key,
                    "tagged {tagged} round {round}: {actual:?} {expected:?}"
                );
                assert_eq!(actual.1, expected.1, "tagged {tagged} round {round}");
                if matches!(expected.0, Value::Nil) {
                    break;
                }
                key = expected.0;
                seen += 1;
                // Delete a string key now and then, as a clearing loop does:
                // the tombstone stays and the traversal skips it.
                if round % 2 == 1
                    && let Value::String(text) = &key
                    && seen % 3 == 0
                {
                    runtime
                        .table_set(table, &Value::String(text.clone()), Value::Nil)
                        .unwrap();
                }
            }
        }
    }
}

/// Checks that the position of `view` names the entry (`key`, `value`) a
/// step gave, and that the key, the integer and the value read from the slot
/// agree with it. Returns the position.
#[allow(unsafe_code)]
fn check_native_cursor(
    runtime: &Runtime,
    view: &crate::runtime::NativeCursor,
    (key, value): (&Value, &Value),
) -> usize {
    use crate::runtime::{
        NATIVE_CURSOR_SLOT_KEY_BITS, NATIVE_CURSOR_SLOT_KEY_TAG, NATIVE_CURSOR_SLOT_SIZE,
        NATIVE_CURSOR_SLOT_VALUE, NATIVE_VALUE_TAG_INTEGER, NATIVE_VALUE_TAG_STRING,
    };
    let position = usize::try_from(unsafe { *(view.position as *const u64) }).unwrap();
    assert!(position < usize::try_from(view.len).unwrap());
    let slot = view.slots + position * NATIVE_CURSOR_SLOT_SIZE;
    let key_tag = unsafe { *((slot + NATIVE_CURSOR_SLOT_KEY_TAG) as *const u64) };
    let key_bits = unsafe { *((slot + NATIVE_CURSOR_SLOT_KEY_BITS) as *const i64) };
    let value_at = unsafe { *((slot + NATIVE_CURSOR_SLOT_VALUE) as *const *const Value) };
    assert_eq!(unsafe { &*value_at }, value);
    let committed = unsafe { runtime.native_cursor_key(slot) };
    assert_eq!(&committed, key);
    if matches!(key, Value::String(_)) {
        assert_eq!(key_tag, u64::from(NATIVE_VALUE_TAG_STRING));
    } else {
        assert_eq!(key_tag, u64::from(NATIVE_VALUE_TAG_INTEGER));
        assert!((1000..1040).contains(&key_bits));
    }
    position
}

#[test]
fn the_native_cursor_describes_the_snapshot_a_traversal_built() {
    for tagged in [false, true] {
        let mut runtime = Runtime::new();
        runtime.set_tagged_integers(tagged);
        let mut scratch = crate::runtime::NativeCursorScratch::default();
        let table = runtime.create_table().unwrap();
        for index in 0..40 {
            runtime
                .table_set(
                    table,
                    &Value::from(format!("key{index}").as_str()),
                    Value::Integer(index),
                )
                .unwrap();
            runtime
                .table_set(table, &Value::Integer(1000 + index), Value::Integer(-index))
                .unwrap();
        }
        // No snapshot before a traversal asked for one.
        assert!(
            runtime
                .table_native_cursor(table, &Value::Nil, &mut scratch)
                .is_none()
        );
        let mut key = Value::Nil;
        let mut steps = 0;
        let mut seen = Vec::new();
        loop {
            let (next, value) = match next_by_reference(&mut runtime, table, &key) {
                Some(entry) => entry,
                None => runtime.table_next(table, &key).unwrap(),
            };
            if matches!(next, Value::Nil) {
                break;
            }
            steps += 1;
            if let Some(view) = runtime.table_native_cursor(table, &next, &mut scratch) {
                seen.push(check_native_cursor(&runtime, &view, (&next, &value)));
            }
            key = next;
        }
        assert_eq!(steps, 80);
        // Once built, the snapshot stays for the rest of the traversal, in
        // the order the positions count.
        assert!(seen.len() > 60, "{} steps had a snapshot", seen.len());
        assert!(seen.windows(2).all(|pair| pair[1] == pair[0] + 1));
        // A key added to the table changes the layout: no snapshot.
        runtime
            .table_set(table, &Value::from("new"), Value::Integer(1))
            .unwrap();
        assert!(
            runtime
                .table_native_cursor(table, &key, &mut scratch)
                .is_none()
        );
    }
}

#[test]
fn a_small_table_gets_its_snapshot_built_for_the_loop() {
    for tagged in [false, true] {
        let mut runtime = Runtime::new();
        runtime.set_tagged_integers(tagged);
        let mut scratch = crate::runtime::NativeCursorScratch::default();
        let table = runtime.create_table().unwrap();
        for index in 0..3 {
            runtime
                .table_set(
                    table,
                    &Value::from(format!("key{index}").as_str()),
                    Value::Integer(index),
                )
                .unwrap();
            runtime
                .table_set(table, &Value::Integer(1000 + index), Value::Integer(-index))
                .unwrap();
        }
        let mut key = Value::Nil;
        let mut positions = Vec::new();
        loop {
            let (next, value) = runtime.table_next(table, &key).unwrap();
            if matches!(next, Value::Nil) {
                break;
            }
            // The snapshot has the table's entries in traversal order, and
            // the position of the key the step gave.
            let view = runtime
                .table_native_cursor(table, &next, &mut scratch)
                .unwrap_or_else(|| panic!("a small table builds one on request: {next:?}"));
            assert_eq!(view.len, 6);
            positions.push(check_native_cursor(&runtime, &view, (&next, &value)));
            key = next;
        }
        assert_eq!(positions, (0..6).collect::<Vec<_>>());
        // A key that is not in the table names no position.
        assert!(
            runtime
                .table_native_cursor(table, &Value::from("absent"), &mut scratch)
                .is_none()
        );
    }
}

#[test]
fn a_scalar_overwrite_in_place_keeps_the_layout_and_declines_other_stores() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = LuaString::from("k5");
    let before = indexed_sizes(&runtime, table);
    // The value is an integer: an integer replaces it, a float or a boolean
    // does not.
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &key, &Value::Number(2.5)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &key, &Value::Boolean(true)),
        Ok(false)
    );
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &key, &Value::from("text")),
        Ok(false)
    );
    assert!(runtime.table_can_overwrite_scalar_string(table, &key, &Value::Integer(0)));
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &key, &Value::Integer(77)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("k5")).unwrap(),
        Value::Integer(77)
    );
    // An absent key and a read-only table decline.
    let absent = LuaString::from("absent");
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &absent, &Value::Integer(1)),
        Ok(false)
    );
    runtime.table_set_readonly(table, true).unwrap();
    assert_eq!(
        runtime.table_overwrite_scalar_string(table, &key, &Value::Integer(1)),
        Ok(false)
    );
    runtime.table_set_readonly(table, false).unwrap();
    assert_eq!(indexed_sizes(&runtime, table), before);
}

#[test]
fn a_delete_in_place_leaves_a_tombstone_the_traversal_skips() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = LuaString::from("k5");
    assert!(runtime.table_can_delete_string_in_place(table, &key));
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 199));
    for fast in [false, true] {
        let keys = traversal(&mut runtime, table, fast);
        assert_eq!(keys.len(), 199);
        assert!(!keys.contains(&Value::from("k5")));
    }
    // The key is absent now and stays so, with no metatable to take the store.
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 199));
    // A metatable with __newindex takes the store of an absent key.
    let handler = runtime.create_table().unwrap();
    runtime
        .table_set(handler, &Value::from("__newindex"), Value::Boolean(true))
        .unwrap();
    runtime.set_table_metatable(table, Some(handler)).unwrap();
    assert!(!runtime.table_can_delete_string_in_place(table, &key));
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(false));
    // A key it holds is deleted whatever the metatable says.
    let held = LuaString::from("k6");
    assert_eq!(runtime.table_delete_string_in_place(table, &held), Ok(true));
    assert_eq!(indexed_sizes(&runtime, table), (200, 200, 198));
    // The key __gc is left to the general store, and so is a small table's
    // key: it shifts its entries.
    let gc = LuaString::from("__gc");
    runtime.set_table_metatable(table, None).unwrap();
    runtime
        .table_set(table, &Value::from("__gc"), Value::Integer(1))
        .unwrap();
    assert_eq!(runtime.table_delete_string_in_place(table, &gc), Ok(false));
    let small = runtime.create_table().unwrap();
    runtime
        .table_set(small, &Value::from("a"), Value::Integer(1))
        .unwrap();
    let a = LuaString::from("a");
    assert!(!runtime.table_can_delete_string_in_place(small, &a));
    assert_eq!(runtime.table_delete_string_in_place(small, &a), Ok(false));
    assert_eq!(
        runtime.table_get(small, &Value::from("a")).unwrap(),
        Value::Integer(1)
    );
}

fn list_of(runtime: &mut Runtime, count: u32) -> TableHandle {
    let table = runtime.create_table().unwrap();
    for index in 1..=count {
        runtime
            .table_set(
                table,
                &Value::Number(f64::from(index)),
                Value::Number(f64::from(index)),
            )
            .unwrap();
    }
    table
}

fn list_values(runtime: &Runtime, table: TableHandle) -> Vec<f64> {
    let mut values = Vec::new();
    let mut index = 1.0;
    while let Ok(Value::Number(number)) = runtime.table_get(table, &Value::Number(index)) {
        values.push(number);
        index += 1.0;
    }
    values
}

#[test]
fn array_shift_moves_a_dense_list_in_place() {
    let mut runtime = Runtime::new();
    let table = list_of(&mut runtime, 4);
    assert!(runtime.table_array_shift_insert(table, Some(2), &Value::Number(9.0)));
    assert_eq!(list_values(&runtime, table), [1.0, 9.0, 2.0, 3.0, 4.0]);
    // At length + 1 and at the end.
    assert!(runtime.table_array_shift_insert(table, Some(6), &Value::Number(7.0)));
    assert!(runtime.table_array_shift_insert(table, None, &Value::Number(8.0)));
    assert_eq!(
        list_values(&runtime, table),
        [1.0, 9.0, 2.0, 3.0, 4.0, 7.0, 8.0]
    );
    assert_eq!(
        runtime.table_array_shift_remove(table, Some(2)),
        Some(Value::Number(9.0))
    );
    assert_eq!(
        runtime.table_array_shift_remove(table, None),
        Some(Value::Number(8.0))
    );
    assert_eq!(list_values(&runtime, table), [1.0, 2.0, 3.0, 4.0, 7.0]);
}

#[test]
fn array_shift_declines_what_the_element_loop_must_handle() {
    let mut runtime = Runtime::new();
    let table = list_of(&mut runtime, 3);
    // Out of range, nil value.
    assert!(!runtime.table_array_shift_insert(table, Some(5), &Value::Number(1.0)));
    assert!(!runtime.table_array_shift_insert(table, Some(0), &Value::Number(1.0)));
    assert!(!runtime.table_array_shift_insert(table, Some(1), &Value::Nil));
    assert_eq!(runtime.table_array_shift_remove(table, Some(4)), None);
    assert_eq!(runtime.table_array_shift_remove(table, Some(0)), None);
    assert_eq!(list_values(&runtime, table), [1.0, 2.0, 3.0]);
    // A hash-part key right after the end.
    runtime
        .table_set(table, &Value::Number(5.0), Value::Number(5.0))
        .unwrap();
    assert!(!runtime.table_array_shift_insert(table, None, &Value::Number(1.0)));
    // A readonly table.
    let frozen = list_of(&mut runtime, 3);
    runtime.table_set_readonly(frozen, true).unwrap();
    assert!(!runtime.table_array_shift_insert(frozen, Some(1), &Value::Number(1.0)));
    assert_eq!(runtime.table_array_shift_remove(frozen, Some(1)), None);
    assert_eq!(list_values(&runtime, frozen), [1.0, 2.0, 3.0]);
    // An empty table has nothing to remove.
    let empty = runtime.create_table().unwrap();
    assert_eq!(runtime.table_array_shift_remove(empty, None), None);
    assert!(runtime.table_array_shift_insert(empty, None, &Value::Number(1.0)));
}

#[test]
fn operator_handlers_are_cached_per_metatable_and_dropped_when_it_changes() {
    use crate::runtime::OperatorHandler;
    let mut runtime = Runtime::new();
    let chunk = Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]));
    let first = runtime
        .create_closure(Arc::clone(&chunk), Vec::new())
        .unwrap();
    let second = runtime.create_closure(chunk, Vec::new()).unwrap();
    let meta = runtime.create_table().unwrap();
    let key = LuaString::from("__add");
    const ADD: u8 = 0;
    // An absent handler is known too, until the key is added.
    assert_eq!(runtime.operator_handler_cached(meta, ADD), None);
    assert_eq!(
        runtime.operator_handler(meta, ADD, &key),
        OperatorHandler::Absent
    );
    assert_eq!(
        runtime.operator_handler_cached(meta, ADD),
        Some(OperatorHandler::Absent)
    );
    runtime
        .table_set(meta, &Value::from("__add"), Value::Closure(first))
        .unwrap();
    assert_eq!(runtime.operator_handler_cached(meta, ADD), None);
    assert_eq!(
        runtime.operator_handler(meta, ADD, &key),
        OperatorHandler::Closure(first)
    );
    // Another event of the same metatable has its own entry.
    assert_eq!(runtime.operator_handler_cached(meta, 1), None);
    // Replacing the closure, by a closure or another value, or removing it,
    // drops the answer.
    runtime
        .table_set(meta, &Value::from("__add"), Value::Closure(second))
        .unwrap();
    assert_eq!(runtime.operator_handler_cached(meta, ADD), None);
    assert_eq!(
        runtime.operator_handler(meta, ADD, &key),
        OperatorHandler::Closure(second)
    );
    runtime
        .table_set(meta, &Value::from("__add"), Value::Integer(1))
        .unwrap();
    assert_eq!(runtime.operator_handler_cached(meta, ADD), None);
    // A value that is not a closure is not kept: every read looks again.
    assert_eq!(
        runtime.operator_handler(meta, ADD, &key),
        OperatorHandler::Other
    );
    assert_eq!(runtime.operator_handler_cached(meta, ADD), None);
    runtime
        .table_set(meta, &Value::from("__add"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.operator_handler(meta, ADD, &key),
        OperatorHandler::Absent
    );
    // Another metatable never reads this one's entry.
    let other = runtime.create_table().unwrap();
    assert_eq!(runtime.operator_handler_cached(other, ADD), None);
}

/// How many hashed string probes `work` makes.
fn probes_of(work: impl FnOnce()) -> u64 {
    let before = crate::runtime::table_storage::entries::string_probes();
    work();
    crate::runtime::table_storage::entries::string_probes() - before
}

#[test]
fn a_delete_or_a_store_into_a_present_key_of_a_hashed_table_makes_one_probe() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = LuaString::from("k5");
    let _ = key.cached_hash();
    // Overwriting a value with a scalar, from the interpreter's raw store or
    // from the helper that keeps the layout, finds the key once.
    for value in [Value::Integer(8), Value::Number(2.5), Value::Boolean(true)] {
        let probes = probes_of(|| {
            assert_eq!(runtime.table_set_string_from(table, &key, &value), Ok(true));
        });
        assert_eq!(probes, 1, "{value:?}");
    }
    runtime
        .table_set_string_from(table, &key, &Value::Integer(8))
        .unwrap();
    let probes = probes_of(|| {
        assert_eq!(
            runtime.table_overwrite_scalar_string(table, &key, &Value::Integer(9)),
            Ok(true)
        );
    });
    assert_eq!(probes, 1);
    // Deleting and reinserting an existing key is one probe each, whether
    // the reinsertion is the raw store or the revive helper.
    for revive_by_helper in [false, true] {
        let probes = probes_of(|| {
            assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
        });
        assert_eq!(probes, 1, "delete");
        assert_eq!(indexed_sizes(&runtime, table), (200, 200, 199));
        let probes = probes_of(|| {
            if revive_by_helper {
                assert_eq!(
                    runtime.table_revive_string(table, &key, &Value::Integer(5)),
                    Ok(true)
                );
            } else {
                assert_eq!(
                    runtime.table_set_string_from(table, &key, &Value::Integer(5)),
                    Ok(true)
                );
            }
        });
        assert_eq!(probes, 1, "reinsert, helper: {revive_by_helper}");
        assert_eq!(indexed_sizes(&runtime, table), (200, 200, 200));
    }
    // A key the table lacks is found absent by the same one probe.
    let absent = LuaString::from("absent");
    let _ = absent.cached_hash();
    let probes = probes_of(|| {
        assert_eq!(
            runtime.table_delete_string_in_place(table, &absent),
            Ok(true)
        );
    });
    assert_eq!(probes, 1);
}

#[test]
fn a_deleted_key_revives_where_it_stood_and_other_stores_decline() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = LuaString::from("k5");
    let order = traversal(&mut runtime, table, false);
    // A live key, an absent key and a key that was never there are no revive.
    assert!(!runtime.table_can_revive_string(table, &key));
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Integer(1)),
        Ok(false)
    );
    let absent = LuaString::from("absent");
    assert!(!runtime.table_can_revive_string(table, &absent));
    assert_eq!(
        runtime.table_revive_string(table, &absent, &Value::Integer(1)),
        Ok(false)
    );
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
    assert!(runtime.table_can_revive_string(table, &key));
    // Only a number or a boolean: a string or a table needs the barrier and
    // the charge of the general store.
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::from("text")),
        Ok(false)
    );
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Nil),
        Ok(false)
    );
    // A read-only table and a table with a metatable decline.
    runtime.table_set_readonly(table, true).unwrap();
    assert!(!runtime.table_can_revive_string(table, &key));
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Integer(1)),
        Ok(false)
    );
    runtime.table_set_readonly(table, false).unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime.set_table_metatable(table, Some(metatable)).unwrap();
    assert!(!runtime.table_can_revive_string(table, &key));
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Integer(1)),
        Ok(false)
    );
    runtime.set_table_metatable(table, None).unwrap();
    // A key that names `__gc` changes the finalizer: the general store's.
    let finalizer = LuaString::from("__gc");
    runtime
        .table_set(table, &Value::String(finalizer.clone()), Value::Integer(1))
        .unwrap();
    runtime
        .table_set(table, &Value::String(finalizer.clone()), Value::Nil)
        .unwrap();
    assert!(!runtime.table_can_revive_string(table, &finalizer));
    // Nothing changed until now.
    assert_eq!(
        runtime.table_get(table, &Value::from("k5")).unwrap(),
        Value::Nil
    );
    // The revive stores in place: the traversal order of the keys is the one
    // the table had, and the entry counts live again.
    let before = indexed_sizes(&runtime, table);
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Number(2.5)),
        Ok(true)
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("k5")).unwrap(),
        Value::Number(2.5)
    );
    let after = indexed_sizes(&runtime, table);
    assert_eq!(
        (after.0, after.1, after.2),
        (before.0, before.1, before.2 + 1)
    );
    let mut again = traversal(&mut runtime, table, false);
    let mut expected = order.clone();
    again.retain(|key| !matches!(key, Value::String(text) if text.as_bytes() == b"__gc"));
    expected.retain(|key| !matches!(key, Value::String(text) if text.as_bytes() == b"__gc"));
    assert_eq!(again, expected);
}

#[test]
fn a_store_notifies_what_the_tables_around_it_cache() {
    let mut runtime = Runtime::new();
    let table = hashed_string_table(&mut runtime, 200);
    let key = LuaString::from("k5");
    // A table that cached inherited reads depend on (`watched`) advances the
    // epoch for a delete and a revive, and a table used as a metatable forgets
    // the metamethods it lacked.
    runtime.table_mut(table).unwrap().watched = true;
    let epoch = runtime.inherit_epoch;
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
    assert!(runtime.inherit_epoch > epoch);
    let epoch = runtime.inherit_epoch;
    assert_eq!(
        runtime.table_revive_string(table, &key, &Value::Integer(1)),
        Ok(true)
    );
    assert!(runtime.inherit_epoch > epoch);
    runtime.table_mut(table).unwrap().watched = false;
    let known = runtime
        .table(table)
        .unwrap()
        .absent_metamethods
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(known, 0);
    runtime
        .table_mut(table)
        .unwrap()
        .absent_metamethods
        .store(0b101, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(runtime.table_delete_string_in_place(table, &key), Ok(true));
    assert_eq!(
        runtime
            .table(table)
            .unwrap()
            .absent_metamethods
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn a_header_is_solid_while_the_hash_part_is_empty_and_appendable_at_the_border() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in [1, 2, 3] {
        runtime
            .table_set(table, &Value::Integer(index), Value::Boolean(true))
            .unwrap();
    }
    let header = runtime.table_native_header(table).unwrap();
    assert!(header.appendable && header.solid);
    assert_eq!(header.word(), NativeTableHeader::DENSE);

    // A boolean two past the end grows the array part with a hole: the border
    // stays below the end, and the table is solid but not appendable.
    runtime
        .table_set(table, &Value::Integer(5), Value::Boolean(true))
        .unwrap();
    let header = runtime.table_native_header(table).unwrap();
    assert!(!header.appendable && header.solid);
    assert_eq!(header.word(), NativeTableHeader::HOLEY);

    // A nil stored in the middle records a deleted key: not solid.
    runtime
        .table_set(table, &Value::Integer(2), Value::Nil)
        .unwrap();
    assert!(!runtime.table_native_header(table).unwrap().solid);

    // A key in the hash part (a string) makes nothing past the array's end nil.
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Integer(1), Value::Boolean(true))
        .unwrap();
    runtime
        .table_set(
            table,
            &Value::String(LuaString::from("name")),
            Value::Integer(1),
        )
        .unwrap();
    let header = runtime.table_native_header(table).unwrap();
    assert!(!header.appendable && !header.solid);
    assert_eq!(header.word(), 0);
}

/// The kinds a trace reads of a record with `value`, `left` and `right`: a
/// number, two tables, and `left` as an absent key.
fn record_kinds() -> Vec<(Value, NativeSlotKind)> {
    [
        ("value", NativeSlotKind::Number),
        ("left", NativeSlotKind::Table),
        ("right", NativeSlotKind::Table),
        ("left", NativeSlotKind::Absent),
    ]
    .into_iter()
    .map(|(name, kind)| (Value::from(name), kind))
    .collect()
}

fn record_shape() -> std::sync::Arc<crate::runtime::TableShape> {
    crate::runtime::TableShape::from_bytes([&b"value"[..], b"left", b"right"]).unwrap()
}

fn absent_marker() -> usize {
    std::ptr::addr_of!(crate::runtime::NATIVE_ABSENT_FIELD) as usize
}

#[test]
fn a_shape_names_the_slot_of_each_key() {
    let shape = record_shape();
    // Keys sort, so the slots do not follow the order they were given in.
    let mut slots: Vec<_> = [&b"value"[..], b"left", b"right"]
        .into_iter()
        .map(|key| shape.slot_of(key).unwrap())
        .collect();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1, 2]);
    assert_eq!(shape.slot_of(b"left"), Some(0));
    assert_eq!(shape.slot_of(b"missing"), None);
    assert!(crate::runtime::TableShape::from_bytes([]).is_none());
}

#[test]
#[allow(unsafe_code)]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn rows_of_shaped_tables_follow_the_values_and_the_absent_keys() {
    let mut runtime = Runtime::new();
    let shape = record_shape();
    let inner = runtime.create_table_with_shape(&shape).unwrap();
    let leaf_shape = crate::runtime::TableShape::from_bytes([&b"value"[..]]).unwrap();
    let leaf = runtime.create_table_with_shape(&leaf_shape).unwrap();
    for (table, value) in [(inner, 1.0), (leaf, 2.0)] {
        runtime
            .table_set(table, &Value::from("value"), Value::Number(value))
            .unwrap();
    }
    for key in ["left", "right"] {
        runtime
            .table_set(inner, &Value::from(key), Value::Table(leaf))
            .unwrap();
    }
    let wanted = record_kinds();
    let mut hints = vec![0; crate::runtime::shape_cache_len(wanted.len())];
    let mut row = vec![0_usize; wanted.len()];
    // The tables alternate, as the nodes of a tree do, and the shapes stay on
    // the lines of the hints: the row is the same every time.
    let mut rows = Vec::new();
    for table in [inner, leaf, inner, leaf, inner] {
        runtime
            .table_native_value_slots(table, &wanted, &mut hints, &mut row)
            .unwrap();
        rows.push((table, row.clone()));
    }
    for (table, row) in &rows {
        let reference = if *table == inner {
            &rows[0].1
        } else {
            &rows[1].1
        };
        assert_eq!(row, reference);
    }
    let [value, left, right, no_left] = rows[0].1[..] else {
        panic!("four fields");
    };
    // The record has all three, and `left` is no absent key of it.
    assert!(value != 0 && left != 0 && right != 0);
    assert_eq!(no_left, 0);
    // SAFETY: the addresses are slots of tables the test keeps, and nothing
    // writes them meanwhile.
    unsafe {
        assert!(matches!(*(value as *const Value), Value::Number(n) if n == 1.0));
        assert!(matches!(*(left as *const Value), Value::Table(held) if held == leaf));
        assert!(matches!(*(right as *const Value), Value::Table(held) if held == leaf));
    }
    // The leaf has a value only, and `left` is absent from it: the marker.
    let [value, left, right, no_left] = rows[1].1[..] else {
        panic!("four fields");
    };
    assert!(value != 0);
    assert_eq!((left, right), (0, 0));
    assert_eq!(no_left, absent_marker());
}

#[test]
fn a_table_with_no_hash_entries_answers_a_row_with_the_marker_only() {
    let mut runtime = Runtime::new();
    let wanted = record_kinds();
    let mut hints = vec![0; crate::runtime::shape_cache_len(wanted.len())];
    let mut row = vec![7_usize; wanted.len()];
    // An array has no string key: only the field wanted absent has an
    // answer, and the marker is it.
    let array = runtime.create_table().unwrap();
    for index in 1..=3_i64 {
        runtime
            .table_set(array, &Value::Integer(index), Value::Integer(index))
            .unwrap();
    }
    runtime
        .table_native_value_slots(array, &wanted, &mut hints, &mut row)
        .unwrap();
    assert_eq!(row, [0, 0, 0, absent_marker()]);
    // The same table with a metatable might answer through `__index`.
    let metatable = runtime.create_table().unwrap();
    runtime.set_table_metatable(array, Some(metatable)).unwrap();
    let mut row = vec![7_usize; wanted.len()];
    runtime
        .table_native_value_slots(array, &wanted, &mut hints, &mut row)
        .unwrap();
    assert_eq!(row, [0, 0, 0, 0]);
    // A fresh empty table answers alike.
    let empty = runtime.create_table().unwrap();
    let mut row = vec![7_usize; wanted.len()];
    runtime
        .table_native_value_slots(empty, &wanted, &mut hints, &mut row)
        .unwrap();
    assert_eq!(row, [0, 0, 0, absent_marker()]);
}

#[test]
fn a_row_reads_nil_only_from_a_table_without_a_metatable() {
    let mut runtime = Runtime::new();
    let shape = crate::runtime::TableShape::from_bytes([&b"value"[..]]).unwrap();
    let wanted = record_kinds();
    let mut hints = vec![0; crate::runtime::shape_cache_len(wanted.len())];
    let mut found = |runtime: &mut Runtime, table| {
        let mut row = vec![0_usize; wanted.len()];
        runtime
            .table_native_value_slots(table, &wanted, &mut hints, &mut row)
            .unwrap();
        row[3]
    };
    let plain = runtime.create_table_with_shape(&shape).unwrap();
    assert_eq!(found(&mut runtime, plain), absent_marker());
    // `__index` may supply the key, so the table answers for it itself only
    // when no metatable stands behind it.
    let behind = runtime.create_table_with_shape(&shape).unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(behind), Some(metatable))
        .unwrap();
    assert_eq!(found(&mut runtime, behind), 0);
    // A key that was there and was deleted is nil as well; one that holds
    // false is not.
    let record = runtime.create_table_with_shape(&record_shape()).unwrap();
    runtime
        .table_set(record, &Value::from("left"), Value::Boolean(false))
        .unwrap();
    assert_eq!(found(&mut runtime, record), 0);
    runtime
        .table_set(record, &Value::from("left"), Value::Nil)
        .unwrap();
    assert_eq!(found(&mut runtime, record), absent_marker());
    // An ordered table (not shaped) and a hashed one answer the same.
    let ordered = runtime.create_table().unwrap();
    runtime
        .table_set(ordered, &Value::from("value"), Value::Number(1.0))
        .unwrap();
    assert_eq!(found(&mut runtime, ordered), absent_marker());
    runtime
        .table_set(ordered, &Value::from("left"), Value::Number(1.0))
        .unwrap();
    assert_eq!(found(&mut runtime, ordered), 0);
}

#[test]
fn more_shapes_than_the_hints_remember_still_give_the_right_rows() {
    let mut runtime = Runtime::new();
    let wanted = [
        (Value::from("a"), NativeSlotKind::Number),
        (Value::from("z"), NativeSlotKind::Absent),
    ];
    let mut hints = vec![0; crate::runtime::shape_cache_len(wanted.len())];
    // Six shapes with `a` in a different place each, visited in turn twice.
    let keys = ["a", "b", "c", "d", "e", "f"];
    let mut tables = Vec::new();
    for count in 0..keys.len() {
        let names: Vec<&[u8]> = keys[count..].iter().map(|key| key.as_bytes()).collect();
        let shape = crate::runtime::TableShape::from_bytes(names).unwrap();
        let table = runtime.create_table_with_shape(&shape).unwrap();
        runtime
            .table_set(table, &Value::from("f"), Value::Number(count as f64))
            .unwrap();
        tables.push(table);
    }
    for _ in 0..2 {
        for (count, table) in tables.iter().enumerate() {
            let mut row = [7_usize; 2];
            runtime
                .table_native_value_slots(*table, &wanted, &mut hints, &mut row)
                .unwrap();
            // Only the first table has an `a`, and it holds no number.
            assert_eq!(row[0], 0, "table {count}");
            assert_eq!(row[1], absent_marker(), "table {count}");
        }
    }
    // Hints too short for the lines of shapes work as well, one search each.
    let mut short = [0_usize; 2];
    let mut row = [0_usize; 2];
    runtime
        .table_native_value_slots(tables[5], &wanted, &mut short, &mut row)
        .unwrap();
    assert_eq!(row, [0, absent_marker()]);
}

#[test]
fn a_constructor_field_is_stored_in_a_slot_the_shape_names() {
    let mut runtime = Runtime::new();
    let shape = record_shape();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    let before = runtime.gc_debt;
    let name = Value::from("value");
    let Value::String(name) = &name else {
        unreachable!("a string")
    };
    assert!(runtime.table_init_shaped(table, name, &Value::Number(4.0)));
    assert_eq!(
        runtime.table_get(table, &Value::from("value")).unwrap(),
        Value::Number(4.0)
    );
    // The store is charged as the interpreter's first store of a key is.
    assert_eq!(
        runtime.gc_debt - before,
        std::mem::size_of::<crate::runtime::table_storage::key::TableKey>()
            + std::mem::size_of::<Value>()
    );
    // A slot that holds a value is no empty slot, and a key the shape lacks
    // names none.
    assert!(!runtime.table_init_shaped(table, name, &Value::Number(5.0)));
    let other = Value::from("other");
    let Value::String(other) = &other else {
        unreachable!("a string")
    };
    assert!(!runtime.table_init_shaped(table, other, &Value::Number(5.0)));
    // Nor does a table that is not shaped, has a metatable or is read-only.
    let plain = runtime.create_table().unwrap();
    assert!(!runtime.table_init_shaped(plain, name, &Value::Number(1.0)));
    let behind = runtime.create_table_with_shape(&shape).unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(behind), Some(metatable))
        .unwrap();
    assert!(!runtime.table_init_shaped(behind, name, &Value::Number(1.0)));
    let locked = runtime.create_table_with_shape(&shape).unwrap();
    runtime.table_set_readonly(locked, true).unwrap();
    assert!(!runtime.table_init_shaped(locked, name, &Value::Number(1.0)));
}

#[test]
#[allow(unsafe_code)]
fn native_code_that_fills_the_slots_of_a_shaped_table_leaves_it_whole() {
    let mut runtime = Runtime::new();
    let shape = record_shape();
    let table = runtime.create_table_with_shape(&shape).unwrap();
    let slots = runtime.table_shaped_slots(table).unwrap() as *mut Value;
    // The slot of `value` takes a number the way native code writes it.
    let slot = shape.slot_of(b"value").unwrap();
    // SAFETY: the slots belong to the table, which nothing else touches.
    unsafe { *slots.add(slot) = Value::Number(9.5) };
    runtime.charge_native_inits(1);
    assert_eq!(
        runtime.table_get(table, &Value::from("value")).unwrap(),
        Value::Number(9.5)
    );
    // Only a shaped table has slots to give.
    let plain = runtime.create_table().unwrap();
    assert!(runtime.table_shaped_slots(plain).is_none());
    // A major cycle in progress asks for the write barrier.
    assert!(runtime.gc_barrier_idle());
}

#[test]
fn new_string_keys_fill_an_empty_table_and_keep_key_order() {
    let mut runtime = Runtime::new();
    for count in [1_usize, 2, 4, 5, 8, 9, 16, 17, 40] {
        let table = runtime.create_table().unwrap();
        // Keys arrive out of order; iteration still walks them sorted.
        let names: Vec<String> = (0..count).map(|index| format!("key{index:03}")).collect();
        for (index, name) in names.iter().enumerate().rev() {
            let key = crate::string::LuaString::from(name.as_str());
            let number = i64::try_from(index).unwrap();
            assert_eq!(
                runtime.table_set_string_if_no_newindex(table, &key, Value::Integer(number)),
                Ok(true)
            );
        }
        let mut walked = Vec::new();
        let mut previous = Value::Nil;
        loop {
            let (key, value) = runtime.table_next(table, &previous).unwrap();
            if key == Value::Nil {
                break;
            }
            walked.push((key.clone(), value));
            previous = key;
        }
        assert_eq!(walked.len(), count);
        for (index, (key, value)) in walked.iter().enumerate() {
            assert_eq!(*key, Value::from(names[index].as_str()));
            assert_eq!(*value, Value::Integer(i64::try_from(index).unwrap()));
        }
        assert_eq!(runtime.table_length(table), Ok(0.0));
    }
}

#[test]
fn a_new_string_key_charges_the_collector_once_per_key() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let key = crate::string::LuaString::from("only");
    let before = runtime.allocation_serial();
    runtime
        .table_set_string_if_no_newindex(table, &key, Value::Integer(1))
        .unwrap();
    let after_first = runtime.allocation_serial();
    assert!(after_first > before);
    runtime
        .table_set_string_if_no_newindex(table, &key, Value::Integer(2))
        .unwrap();
    assert_eq!(runtime.allocation_serial(), after_first);
}

#[test]
fn integer_key_get_and_set_match_the_generic_table_calls() {
    let keys: [i64; 10] = [-3, 0, 1, 2, 3, 5, 9, 100, 1 << 53, i64::MAX];
    for tagged in [true, false] {
        let mut fast = Runtime::new();
        let mut slow = Runtime::new();
        fast.set_tagged_integers(tagged);
        slow.set_tagged_integers(tagged);
        let a = fast.create_table().unwrap();
        let b = slow.create_table().unwrap();
        let inner = fast.create_table().unwrap();
        let inner_slow = slow.create_table().unwrap();
        // Appends, overwrites, holes, gaps, far keys, and a collectable value.
        for (step, key) in [1, 2, 3, 2, 5, 4, 1 << 53, 100, 0, -3, i64::MAX, 3, 9]
            .into_iter()
            .enumerate()
        {
            let nil = step % 5 == 4;
            let (x, y) = if nil {
                (Value::Nil, Value::Nil)
            } else if step == 7 {
                (Value::Table(inner), Value::Table(inner_slow))
            } else {
                (Value::Number(step as f64), Value::Number(step as f64))
            };
            match x {
                Value::Number(number) if step % 2 == 0 => {
                    fast.table_set_integer_number(a, key, number).unwrap();
                }
                x => fast.table_set_integer(a, key, x).unwrap(),
            }
            slow.table_set(b, &slow.integer_value(key), y).unwrap();
            for probe in keys.iter().copied().chain(1..=6) {
                let got = fast.table_get_integer(a, probe).unwrap();
                let want = slow.table_get(b, &slow.integer_value(probe)).unwrap();
                match (&got, &want) {
                    (Value::Table(_), Value::Table(_)) => {}
                    _ => assert_eq!(got, want, "key {probe} after storing key {key}"),
                }
            }
            assert_eq!(fast.table_length(a), slow.table_length(b));
        }
    }
}

#[test]
fn integer_key_set_respects_readonly_tables() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set_integer(table, 1, Value::Number(1.0))
        .unwrap();
    runtime.table_set_readonly(table, true).unwrap();
    assert!(
        runtime
            .table_set_integer(table, 1, Value::Number(2.0))
            .is_err()
    );
    assert!(
        runtime
            .table_set_integer(table, 2, Value::Number(2.0))
            .is_err()
    );
    assert!(runtime.table_set_integer_number(table, 1, 2.0).is_err());
    assert!(runtime.table_set_integer_number(table, 2, 2.0).is_err());
    assert_eq!(
        runtime.table_get_integer(table, 1).unwrap(),
        Value::Number(1.0)
    );
}

#[test]
fn integer_number_store_falls_back_when_the_table_has_hash_entries_or_holes() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for n in 1..=4 {
        runtime
            .table_set_integer_number(table, n, n as f64)
            .unwrap();
    }
    // A hash entry for the next key must be absorbed, not duplicated.
    runtime
        .table_set(table, &Value::Number(6.0), Value::from("six"))
        .unwrap();
    runtime
        .table_set(table, &Value::from("name"), Value::from("x"))
        .unwrap();
    runtime.table_set_integer_number(table, 5, 5.0).unwrap();
    assert_eq!(runtime.table_length(table), Ok(6.0));
    assert_eq!(
        runtime.table_get_integer(table, 6).unwrap(),
        Value::from("six")
    );
    // A hole, then an overwrite above the border.
    runtime.table_set_integer(table, 2, Value::Nil).unwrap();
    runtime.table_set_integer_number(table, 4, 40.0).unwrap();
    runtime.table_set_integer_number(table, 2, 20.0).unwrap();
    for (n, want) in [(1, 1.0), (2, 20.0), (3, 3.0), (4, 40.0), (5, 5.0)] {
        assert_eq!(
            runtime.table_get_integer(table, n).unwrap(),
            Value::Number(want)
        );
    }
    assert_eq!(runtime.table_length(table), Ok(6.0));
}

/// The live variants the C API uses for tables on its stacks skip the
/// arena's generation and occupancy checks. A rooted table keeps its slot
/// through arena growth and collections that free and reuse the slots around
/// it, so the live calls agree with the checked ones, and they decline the
/// same stores: any into a read-only table, and appends to the table a major
/// cycle traces in chunks.
#[test]
#[allow(unsafe_code)]
fn live_array_stores_and_reads_match_the_checked_calls_across_growth_and_collection() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let root = Value::Table(table);
    let mut length = 0;
    for round in 0..4_i32 {
        // Thousands of unrooted tables grow the arena, which reallocates it,
        // and the collection frees them for the next round to reuse.
        for _ in 0..5000 {
            runtime.create_table().unwrap();
        }
        length += 100;
        for n in 1..=length {
            let value = Value::Number(f64::from(round) * 1000.0 + n as f64);
            // SAFETY: `root` roots the table at every collection below.
            if !unsafe { runtime.table_raw_store_scalar_live(table, n, &value) } {
                runtime.table_set_integer(table, n, value).unwrap();
            }
        }
        runtime.collect_full(incremental_roots(&root));
        let fresh = runtime.create_table().unwrap();
        // SAFETY: nothing collects between the creation and the read.
        assert_eq!(unsafe { runtime.table_array_slot_live(fresh, 1) }, None);
        for n in 1..=length {
            let expected = Value::Number(f64::from(round) * 1000.0 + n as f64);
            // SAFETY: as above.
            assert_eq!(
                unsafe { runtime.table_array_slot_live(table, n) },
                Some(&expected)
            );
            assert_eq!(runtime.table_array_slot(table, n), Some(&expected));
        }
    }
    // SAFETY: the table is still rooted, and nothing collects from here on.
    unsafe {
        assert_eq!(runtime.table_array_slot_live(table, 0), None);
        assert_eq!(runtime.table_array_slot_live(table, length + 1), None);
        let next = length + 1;
        runtime
            .table_reserve(table, usize::try_from(next).unwrap() + 16, 0)
            .unwrap();
        // The table a major cycle traces in chunks takes overwrites, not
        // appends, which the general path makes so it can mark the table.
        runtime.gc_partial_table = Some(table);
        assert!(!runtime.table_raw_store_scalar_live(table, next, &Value::Number(1.0)));
        assert!(runtime.table_raw_store_scalar_live(table, 1, &Value::Number(-1.0)));
        runtime.gc_partial_table = None;
        assert_eq!(runtime.table_array_slot(table, next), None);
        assert!(runtime.table_raw_store_scalar_live(table, next, &Value::Number(1.0)));
        assert_eq!(
            runtime.table_array_slot(table, next),
            Some(&Value::Number(1.0))
        );
        // A read-only table takes no store at all.
        runtime.table_set_readonly(table, true).unwrap();
        assert!(!runtime.table_raw_store_scalar_live(table, 1, &Value::Number(2.0)));
        assert!(!runtime.table_raw_store_scalar_live(table, next + 1, &Value::Number(2.0)));
        assert_eq!(
            runtime.table_array_slot(table, 1),
            Some(&Value::Number(-1.0))
        );
        assert_eq!(runtime.table_array_slot(table, next + 1), None);
    }
}

/// The live store appends a scalar one past a full array part with room in
/// its buffer, charging the collector one value per append as the general
/// path does, and declines every append that needs bookkeeping: past the
/// buffer, past the next key, of a string, into a table with a hash entry,
/// a deleted key or a hole, and into the table a major cycle traces in
/// chunks, but not into another table that the cycle is not tracing.
#[test]
#[allow(unsafe_code)]
fn live_appends_fill_presized_tables_and_leave_bookkeeping_to_the_general_path() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table_sized(4).unwrap();
    let debt = runtime.gc_debt;
    let serial = runtime.allocation_serial();
    // SAFETY: nothing collects in this test.
    unsafe {
        for n in 1..=4 {
            assert!(runtime.table_raw_store_scalar_live(table, n, &Value::Integer(n * 10)));
        }
        assert_eq!(runtime.gc_debt - debt, 4 * std::mem::size_of::<Value>());
        assert_eq!(runtime.allocation_serial() - serial, 4);
        assert_eq!(runtime.table_length(table), Ok(4.0));
        // The buffer is full; the general path grows it.
        assert!(!runtime.table_raw_store_scalar_live(table, 5, &Value::Boolean(true)));
        runtime
            .table_set_integer(table, 5, Value::Boolean(true))
            .unwrap();
        assert!(runtime.table_raw_store_scalar_live(table, 6, &Value::Number(0.5)));
        // Only the next key appends, and only a scalar.
        assert!(!runtime.table_raw_store_scalar_live(table, 8, &Value::Number(1.0)));
        assert!(!runtime.table_raw_store_scalar_live(table, 7, &Value::from("seven")));
        assert_eq!(runtime.table_length(table), Ok(6.0));
        assert_eq!(
            runtime.table_array_slot(table, 5),
            Some(&Value::Boolean(true))
        );
        assert_eq!(
            runtime.table_array_slot(table, 6),
            Some(&Value::Number(0.5))
        );
    }

    let declines = |setup: &dyn Fn(&mut Runtime, TableHandle)| {
        let mut runtime = Runtime::new();
        let table = runtime.create_table_sized(8).unwrap();
        for n in 1..=3 {
            runtime
                .table_set_integer(table, n, Value::Integer(n))
                .unwrap();
        }
        setup(&mut runtime, table);
        let length = runtime.table_length(table).unwrap();
        // SAFETY: nothing collects here.
        assert!(!unsafe { runtime.table_raw_store_scalar_live(table, 4, &Value::Integer(4)) });
        assert_eq!(runtime.table_length(table), Ok(length));
        runtime.gc_partial_table = None;
        // The general path takes the store.
        runtime
            .table_set_integer(table, 4, Value::Integer(4))
            .unwrap();
        assert_eq!(runtime.table_get_integer(table, 4), Ok(Value::Integer(4)));
    };
    declines(&|runtime, table| {
        runtime
            .table_set(table, &Value::from("name"), Value::Boolean(true))
            .unwrap();
    });
    declines(&|runtime, table| {
        runtime
            .table_set(table, &Value::from("gone"), Value::Integer(1))
            .unwrap();
        runtime
            .table_set(table, &Value::from("gone"), Value::Nil)
            .unwrap();
    });
    declines(&|runtime, table| {
        runtime.table_set_integer(table, 2, Value::Nil).unwrap();
    });
    declines(&|runtime, table| {
        runtime.table_set_integer(table, 3, Value::Nil).unwrap();
        runtime
            .table_set_integer(table, 3, Value::Integer(3))
            .unwrap();
        runtime.table_set_integer(table, 3, Value::Nil).unwrap();
    });
    declines(&|runtime, table| {
        runtime.gc_partial_table = Some(table);
    });

    let mut runtime = Runtime::new();
    let other = runtime.create_table().unwrap();
    let table = runtime.create_table_sized(2).unwrap();
    assert_ne!(other.index(), table.index());
    runtime.gc_partial_table = Some(other);
    // SAFETY: nothing collects here.
    assert!(unsafe { runtime.table_raw_store_scalar_live(table, 1, &Value::Integer(1)) });
}

#[test]
fn a_set_of_absent_metamethods_is_one_answer_until_the_metatable_changes() {
    use std::sync::atomic::Ordering;
    let mut runtime = Runtime::new();
    let meta = runtime.create_table().unwrap();
    let plain = Metamethod::Gc.bit() | Metamethod::Mode.bit() | Metamethod::Index.bit();
    let every = Metamethod::ALL
        .into_iter()
        .fold(0, |set, event| set | event.bit());
    // Nothing is cached yet: the slow path answers and caches what it asked.
    assert_eq!(
        runtime
            .table(meta)
            .unwrap()
            .absent_metamethods
            .load(Ordering::Relaxed),
        0
    );
    assert!(runtime.metatable_lacks_all(meta, plain));
    assert_eq!(
        runtime
            .table(meta)
            .unwrap()
            .absent_metamethods
            .load(Ordering::Relaxed),
        plain
    );
    // The cached answer and a wider set that still has events to ask about.
    assert!(runtime.metatable_lacks_all(meta, plain));
    assert!(runtime.metatable_lacks_all(meta, every));
    assert!(runtime.metatable_lacks_all(meta, 0));

    // A store drops the cache, and each event is refused by itself and in
    // any set that holds it, whichever other events of the set are absent.
    for event in Metamethod::ALL {
        let key = Value::String(runtime.metamethod_key(event).clone());
        runtime.table_set(meta, &key, Value::Integer(1)).unwrap();
        assert!(!runtime.metatable_lacks_all(meta, event.bit()));
        assert!(!runtime.metatable_lacks_all(meta, every));
        assert!(runtime.metatable_lacks_all(meta, every & !event.bit()));
        // A second ask, with the other absences cached, says the same.
        assert!(!runtime.metatable_lacks_all(meta, every));
        assert_eq!(
            runtime.metatable_lacks_all(meta, plain),
            plain & event.bit() == 0
        );
        // A stored nil is absent again.
        runtime.table_set(meta, &key, Value::Nil).unwrap();
        assert!(runtime.metatable_lacks_all(meta, every));
    }

    // An unrelated key leaves every answer as it was.
    runtime
        .table_set(meta, &Value::from("name"), Value::Integer(1))
        .unwrap();
    assert!(runtime.metatable_lacks_all(meta, every));

    // A stale handle lacks nothing.
    assert!(runtime.tables.remove(meta.0).is_some());
    assert!(!runtime.metatable_lacks_all(meta, plain));
    assert!(!runtime.metatable_lacks_all(meta, 0));
}
