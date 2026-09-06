use super::*;

#[test]
fn borrowed_string_lookup_matches_owned_hash_and_distinct_allocations() {
    use std::hash::BuildHasher;
    let state = std::collections::hash_map::RandomState::new();
    for bytes in [b"".as_slice(), b"key", b"key\0\xff"] {
        let stored = crate::string::LuaString::new(bytes);
        let queried = crate::string::LuaString::new(bytes);
        assert_eq!(
            state.hash_one(crate::runtime::table_storage::key::TableKey::String(stored)),
            state.hash_one(crate::runtime::table_storage::key::StringTableKey(&queried)),
        );
    }
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in 0..200 {
        runtime
            .table_set(
                table,
                &Value::from(format!("key_{index}")),
                Value::Number(f64::from(index)),
            )
            .unwrap();
    }
    for index in 0..200 {
        assert_eq!(
            runtime
                .table_get(table, &Value::from(format!("key_{index}")))
                .unwrap(),
            Value::Number(f64::from(index)),
        );
    }
    assert_eq!(
        runtime.table_get(table, &Value::from("absent")).unwrap(),
        Value::Nil
    );
}

#[test]
fn numeric_unary_registration_keeps_scalar_dispatch_and_excludes_other_natives() {
    extern "C" fn double(value: f64) -> f64 {
        value * 2.0
    }

    let mut runtime = Runtime::new();
    let ordinary = runtime
        .register_native_scalar(|_, _| Ok(Value::Number(1.0)))
        .unwrap();
    let unary = runtime
        .register_native_numeric_unary(
            |_, arguments| {
                Ok(match arguments.first() {
                    Some(Value::Number(number)) => Value::Number(number * 2.0),
                    value => value.cloned().unwrap_or(Value::Nil),
                })
            },
            double,
        )
        .unwrap();

    assert!(runtime.native_numeric_unary(ordinary).is_none());
    assert!(
        runtime
            .native_numeric_unary(NativeFunctionHandle(u32::MAX))
            .is_none()
    );
    let fast_path = runtime.native_numeric_unary(unary).unwrap();
    assert_eq!(fast_path(1.5), 3.0);
    assert_eq!(
        runtime
            .call_native(unary, &[Value::from("2.5"), Value::from("ignored")])
            .unwrap(),
        vec![Value::from("2.5")]
    );
}

#[test]
fn nil_and_nan_table_reads_miss_but_writes_remain_invalid() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();

    assert_eq!(runtime.table_get(table, &Value::Nil).unwrap(), Value::Nil);
    assert_eq!(
        runtime.table_get(table, &Value::Number(f64::NAN)).unwrap(),
        Value::Nil
    );
    assert!(
        runtime
            .table_set(table, &Value::Nil, Value::from(1.0))
            .is_err()
    );
    assert!(
        runtime
            .table_set(table, &Value::Number(f64::NAN), Value::from(1.0))
            .is_err()
    );
}

#[test]
fn format_value_uses_lua_primitive_tostring_representations() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let userdata = runtime.create_userdata().unwrap();

    assert_eq!(
        runtime.format_value(&Value::Nil).unwrap().as_bytes(),
        b"nil"
    );
    assert_eq!(
        runtime
            .format_value(&Value::Boolean(true))
            .unwrap()
            .as_bytes(),
        b"true"
    );
    assert_eq!(
        runtime
            .format_value(&Value::Number(f64::INFINITY))
            .unwrap()
            .as_bytes(),
        b"inf"
    );
    assert_eq!(
        runtime
            .format_value(&Value::Number(4_611_686_018_427_387_904.0))
            .unwrap()
            .as_bytes(),
        b"4.6116860184274e+18"
    );
    assert_eq!(
        runtime
            .format_value(&Value::from(&b"\0"[..]))
            .unwrap()
            .as_bytes(),
        b"\0"
    );
    assert_eq!(
        runtime
            .format_value(&Value::Table(table))
            .unwrap()
            .as_bytes(),
        format!("table: {}", table.index()).as_bytes()
    );
    assert_eq!(
        runtime
            .format_value(&Value::Userdata(userdata))
            .unwrap()
            .as_bytes(),
        format!("userdata: {}", userdata.index()).as_bytes()
    );
}

#[test]
fn output_buffer_preserves_bytes_and_drains() {
    let mut runtime = Runtime::new();
    runtime.write_output(b"first\xff").unwrap();
    runtime.write_output(b" second").unwrap();
    assert_eq!(runtime.take_output(), b"first\xff second");
    assert_eq!(runtime.take_output(), [] as [u8; 0]);
}

#[test]
fn host_output_writer_receives_writes_and_flushes() {
    #[derive(Clone)]
    struct SharedWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
        flushed: Arc<Mutex<bool>>,
    }

    impl Write for SharedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            *self.flushed.lock().unwrap() = true;
            Ok(())
        }
    }

    let bytes = Arc::new(Mutex::new(Vec::new()));
    let flushed = Arc::new(Mutex::new(false));
    let writer = SharedWriter {
        bytes: Arc::clone(&bytes),
        flushed: Arc::clone(&flushed),
    };
    let mut runtime = Runtime::new();
    let stdout = runtime.create_stdout_file().unwrap();
    runtime.set_output_writer(writer);

    runtime
        .file_write(stdout, &[Value::bytes(b"ready")])
        .unwrap();
    assert_eq!(&*bytes.lock().unwrap(), b"ready");
    assert_eq!(runtime.take_output(), [] as [u8; 0]);
    assert!(!*flushed.lock().unwrap());

    runtime
        .file_setvbuf(stdout, &Value::from("no"), None)
        .unwrap();
    runtime.file_write(stdout, &[Value::bytes(b"!")]).unwrap();
    assert!(*flushed.lock().unwrap());
    runtime.file_flush(stdout).unwrap();
    assert!(*flushed.lock().unwrap());
}

#[test]
fn raw_value_table_read_borrows_stored_values_without_indexing() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::Table(child))
        .unwrap();
    runtime
        .table_set(table, &Value::from("compact"), Value::Number(3.0))
        .unwrap();

    assert!(matches!(
        runtime
            .table_get_raw_value(table, &Value::Number(1.0))
            .unwrap(),
        Some(Value::Table(handle)) if *handle == child
    ));
    assert!(matches!(
        runtime
            .table_get_raw_value(table, &Value::from("compact"))
            .unwrap(),
        Some(Value::Number(value)) if value.to_bits() == 3.0_f64.to_bits()
    ));
    assert_eq!(
        runtime
            .table_get_raw_value(table, &Value::Number(2.0))
            .unwrap(),
        None
    );
    assert_eq!(
        runtime
            .table_get_raw_value(table, &Value::Number(f64::NAN))
            .unwrap(),
        None
    );

    let metatable = runtime.create_table().unwrap();
    runtime.set_table_metatable(table, Some(metatable)).unwrap();
    assert!(matches!(
        runtime
            .table_get_raw_value(table, &Value::Number(1.0))
            .unwrap(),
        Some(Value::Table(handle)) if *handle == child
    ));

    let promoted = runtime.create_table().unwrap();
    for index in 0..HASHED_TABLE_ENTRY_THRESHOLD {
        runtime
            .table_set(
                promoted,
                &Value::from(format!("value-{index}")),
                Value::Number(f64::from(u32::try_from(index).unwrap())),
            )
            .unwrap();
    }
    assert!(matches!(
        runtime
            .table_get_raw_value(promoted, &Value::from(format!("value-{}", HASHED_TABLE_ENTRY_THRESHOLD - 1)))
            .unwrap(),
        Some(Value::Number(value)) if value.to_bits() == f64::from(u32::try_from(HASHED_TABLE_ENTRY_THRESHOLD - 1).unwrap()).to_bits()
    ));
}

#[test]
fn raw_value_table_read_rejects_stale_handles() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    assert!(runtime.tables.remove(table.0).is_some());
    assert_eq!(
        runtime.table_get_raw_value(table, &Value::from("key")),
        Err(RuntimeError::InvalidTable(table))
    );
}

#[test]
fn numeric_array_index_conversion_keeps_invalid_and_huge_keys_out_of_arrays() {
    for value in [
        -f64::INFINITY,
        -1.0,
        -0.0,
        0.0,
        0.5,
        f64::NAN,
        f64::INFINITY,
        I64_EXCLUSIVE_UPPER_AS_F64,
        f64::MAX,
    ] {
        assert_eq!(array_index_number(value), None, "{value:?}");
    }
    assert_eq!(array_index_number(1.0), Some(1));
    assert_eq!(
        array_index_number(4_294_967_295.0),
        usize::try_from(4_294_967_295_i64).ok()
    );

    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let huge = I64_EXCLUSIVE_UPPER_AS_F64;
    runtime
        .table_set(table, &Value::Number(huge), Value::from("huge"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &Value::Number(huge)).unwrap(),
        Value::from("huge")
    );
}

#[test]
fn closures_are_stable_function_values_and_table_entries() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    let closure = Value::Closure(
        runtime
            .create_closure(
                Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
                Vec::new(),
            )
            .unwrap(),
    );
    runtime
        .table_set(table, &Value::from("method"), closure.clone())
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &Value::from("method")).unwrap(),
        closure
    );
    assert_eq!(closure.type_name(), "function");
}

#[test]
fn concatenation_formats_numbers_and_rejects_other_values() {
    let mut runtime = Runtime::new();
    assert_eq!(
        runtime
            .concatenate(&Value::from("left"), &Value::Number(2.0))
            .unwrap(),
        Value::from("left2")
    );
    assert_eq!(
        runtime
            .concatenate(&Value::Number(-1.25), &Value::from("x"))
            .unwrap(),
        Value::from("-1.25x")
    );
    assert_eq!(
        runtime
            .concatenate(&Value::Number(f64::NAN), &Value::Number(f64::INFINITY))
            .unwrap(),
        Value::from("naninf")
    );
    assert_eq!(
        runtime
            .concatenate(&Value::Number(f64::NEG_INFINITY), &Value::Number(f64::NAN),)
            .unwrap(),
        Value::from("-infnan")
    );

    let table = Value::Table(runtime.create_table().unwrap());
    assert!(matches!(
        runtime.concatenate(&table, &Value::from("x")),
        Err(RuntimeError::TypeError {
            operation: "concatenate",
            expected: "string or number",
            found: "table",
        })
    ));
}

#[test]
fn vm_builtins_have_function_identity_and_are_valid_table_keys() {
    let mut runtime = Runtime::new();
    let protected = Value::BuiltinFunction(BuiltinFunction::ProtectedCall);
    assert_eq!(protected.type_name(), "function");
    assert!(Runtime::raw_equal(&protected, &protected));
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &protected, Value::from("works"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &protected).unwrap(),
        Value::from("works")
    );
}

#[test]
fn table_next_keeps_storage_private_and_rejects_invalid_continuations() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("first"), Value::Number(1.0))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(2.0), Value::from("second"))
        .unwrap();

    let (first_key, first_value) = runtime.table_next(table, &Value::Nil).unwrap();
    assert_ne!(first_key, Value::Nil);
    assert_ne!(first_value, Value::Nil);
    let (second_key, second_value) = runtime.table_next(table, &first_key).unwrap();
    assert_ne!(second_key, Value::Nil);
    assert_ne!(second_value, Value::Nil);
    assert_eq!(
        runtime.table_next(table, &second_key).unwrap(),
        (Value::Nil, Value::Nil)
    );
    assert_eq!(
        runtime.table_next(table, &Value::from("missing")),
        Err(RuntimeError::InvalidTableContinuationKey)
    );
}

#[test]
fn large_string_maps_keep_ordered_next_after_hash_index_promotion() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for index in 0..=HASHED_TABLE_ENTRY_THRESHOLD {
        runtime
            .table_set(
                table,
                &Value::from(format!("field-{index:02}")),
                Value::Number(f64::from(u32::try_from(index).unwrap())),
            )
            .unwrap();
    }
    assert!(runtime.table(table).unwrap().entries.is_indexed());

    let first = Value::from("field-00");
    runtime.table_set(table, &first, Value::Nil).unwrap();
    assert_eq!(
        runtime.table_next(table, &first).unwrap(),
        (Value::from("field-01"), Value::Number(1.0))
    );
    assert_eq!(
        runtime.table_get(table, &Value::from("field-15")).unwrap(),
        Value::Number(15.0)
    );
}

#[test]
fn deleted_sparse_array_keys_remain_valid_next_continuations() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(3.0), Value::from("three"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(4.0), Value::from("four"))
        .unwrap();

    runtime
        .table_set(table, &Value::Number(3.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(table, &Value::Number(3.0)).unwrap(),
        (Value::Number(4.0), Value::from("four"))
    );
}

#[test]
fn nil_writes_remove_sparse_array_entries_without_creating_new_ones() {
    let mut table = crate::runtime::table_storage::Table::default();
    table.set_array(1, Value::from("one"));
    table.set_array(3, Value::from("three"));
    table.set_array(3, Value::Nil);
    table.set_array(4, Value::Nil);

    assert!(table.entries.is_empty());
    assert!(
        table
            .deleted_entries
            .contains(&crate::runtime::table_storage::array_table_key(3).unwrap())
    );
    assert!(
        !table
            .deleted_entries
            .contains(&crate::runtime::table_storage::array_table_key(4).unwrap())
    );
}

#[test]
fn many_part_concatenation_checks_lua51s_32_bit_length_limit_before_copying() {
    let mut runtime = Runtime::new();
    let part = Value::bytes(vec![0; 1 << 25]);
    let values = vec![part; 129];
    let error = runtime.concatenate_many(&values).unwrap_err();
    assert!(error.to_string().contains("overflow"));
}

#[test]
fn table_storage_growth_charges_debt_but_overwrites_do_not() {
    let mut runtime = Runtime::new();
    runtime.set_gc_mode(GcMode::Incremental);
    let table = runtime.create_table().unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(table));
    runtime.collect_full(roots.finish());
    assert_eq!(runtime.gc_debt, 0);

    runtime
        .table_set(table, &Value::from("map"), Value::Number(1.0))
        .unwrap();
    let map_growth = runtime.gc_debt;
    assert!(map_growth > 0);
    runtime
        .table_set(table, &Value::from("map"), Value::Number(2.0))
        .unwrap();
    assert_eq!(runtime.gc_debt, map_growth);

    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::from("raw"), Value::Number(3.0))
            .unwrap()
    );
    let raw_growth = runtime.gc_debt;
    assert!(raw_growth > map_growth);
    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::from("raw"), Value::Number(4.0))
            .unwrap()
    );
    assert_eq!(runtime.gc_debt, raw_growth);

    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(1.0))
            .unwrap()
    );
    let dense_growth = runtime.gc_debt;
    assert!(dense_growth > raw_growth);
    assert!(
        runtime
            .table_set_if_no_newindex(table, &Value::Number(1.0), Value::Number(2.0))
            .unwrap()
    );
    assert_eq!(runtime.gc_debt, dense_growth);

    for index in 2..=512 {
        assert!(
            runtime
                .table_set_if_no_newindex(
                    table,
                    &Value::Number(f64::from(index)),
                    Value::Number(f64::from(index))
                )
                .unwrap()
        );
    }
    assert!(runtime.gc_needs_collection());
}

#[test]
fn arena_retires_a_slot_when_its_generation_would_wrap() {
    let mut arena = Arena::default();
    let first = arena.insert("first").unwrap();
    arena.slots[first.slot as usize].generation = u32::MAX;
    let exhausted = ArenaKey {
        slot: first.slot,
        generation: u32::MAX,
    };
    assert_eq!(arena.remove(exhausted), Some("first"));
    assert!(arena.slots[exhausted.slot as usize].retired);
    let second = arena.insert("second").unwrap();
    assert_ne!(second.slot, exhausted.slot);
    assert_eq!(arena.get(exhausted), None);
}

#[test]
fn capture_cells_inline_small_captures_and_copy_on_write_large_ones() {
    let mut runtime = Runtime::new();
    let first = runtime.create_cell(Value::Number(1.0)).unwrap();
    let second = runtime.create_cell(Value::Number(2.0)).unwrap();
    let replacement = runtime.create_cell(Value::Number(3.0)).unwrap();

    let empty_shared = ClosureCells::from_slice(&[]);
    let mut empty = CaptureCells::from_shared(&empty_shared);
    let empty_clone = empty.clone();
    assert_eq!(&*empty_clone, &[] as &[CellHandle]);
    assert!(!empty.replace(0, replacement));

    let one_shared = ClosureCells::from_slice(&[first]);
    assert!(matches!(one_shared, ClosureCells::One(_)));
    let mut one = CaptureCells::from_shared(&one_shared);
    let one_clone = one.clone();
    assert_eq!(&*one_clone, &[first]);
    assert!(!one.replace(1, replacement));
    assert!(one.replace(0, replacement));
    assert_eq!(&*one, &[replacement]);

    let shared_cells = ClosureCells::from_slice(&[first, second]);
    let ClosureCells::Many(shared) = &shared_cells else {
        panic!("two cells are shared");
    };
    let mut many = CaptureCells::from_shared(&shared_cells);
    assert_eq!(Arc::strong_count(shared), 2);
    let prior_snapshot = many.clone();
    assert_eq!(Arc::strong_count(shared), 3);
    assert!(!many.replace(2, replacement));
    assert_eq!(Arc::strong_count(shared), 3);
    assert!(many.replace(1, replacement));
    assert_eq!(&*many, &[first, replacement]);
    assert_eq!(&*prior_snapshot, &[first, second]);
    assert_eq!(Arc::strong_count(shared), 2);

    assert!(std::mem::size_of::<CaptureCells>() <= 24);
}

#[test]
#[allow(unsafe_code)]
fn capture_cell_lookup_agrees_with_the_slice_in_every_storage_form() {
    let mut runtime = Runtime::new();
    let first = runtime.create_cell(Value::Number(1.0)).unwrap();
    let second = runtime.create_cell(Value::Number(2.0)).unwrap();
    let third = runtime.create_cell(Value::Number(3.0)).unwrap();

    let check = |cells: &CaptureCells| {
        for index in 0..5 {
            assert_eq!(
                cells.cell(index),
                cells.get(index).copied(),
                "index {index}"
            );
        }
    };
    let empty_shared = ClosureCells::from_slice(&[]);
    let one_shared = ClosureCells::from_slice(&[first]);
    let many_shared = ClosureCells::from_slice(&[first, second, third]);
    check(&CaptureCells::default());
    check(&CaptureCells::from_shared(&empty_shared));
    check(&CaptureCells::from_shared(&one_shared));
    // Shared storage, and the same cells borrowed from a closure's slice.
    check(&CaptureCells::from_shared(&many_shared));
    // SAFETY: `many_shared` outlives `borrowed`.
    let borrowed = unsafe { CaptureCells::borrow_shared(&many_shared) };
    check(&borrowed);
    assert_eq!(borrowed.cell(2), Some(third));
    assert_eq!(borrowed.cell(3), None);
    // A single cell answers index 0 only.
    let one = CaptureCells::from_shared(&one_shared);
    assert_eq!(one.cell(0), Some(first));
    assert_eq!(one.cell(1), None);
}

#[test]
#[allow(unsafe_code)]
fn borrowed_capture_cells_skip_counting_and_own_their_copies() {
    let mut runtime = Runtime::new();
    let first = runtime.create_cell(Value::Number(1.0)).unwrap();
    let second = runtime.create_cell(Value::Number(2.0)).unwrap();
    let replacement = runtime.create_cell(Value::Number(3.0)).unwrap();

    let shared_cells = ClosureCells::from_slice(&[first, second]);
    let ClosureCells::Many(shared) = &shared_cells else {
        panic!("two cells are shared");
    };
    // SAFETY: `shared_cells` outlives `borrowed` and its moves.
    let mut borrowed = unsafe { CaptureCells::borrow_shared(&shared_cells) };
    assert_eq!(Arc::strong_count(shared), 1);
    assert_eq!(&*borrowed, &[first, second]);

    let copy = borrowed.clone();
    assert_eq!(Arc::strong_count(shared), 1);
    assert!(borrowed.replace(0, replacement));
    assert_eq!(&*borrowed, &[replacement, second]);
    assert_eq!(&*copy, &[first, second]);
    assert_eq!(&**shared, &[first, second]);
    assert!(!borrowed.replace(2, replacement));
}

#[test]
#[allow(unsafe_code)]
fn capture_cells_know_which_closure_cells_they_borrow() {
    let mut runtime = Runtime::new();
    let first = runtime.create_cell(Value::Number(1.0)).unwrap();
    let second = runtime.create_cell(Value::Number(2.0)).unwrap();
    let none = ClosureCells::from_slice(&[]);
    let one = ClosureCells::from_slice(&[first]);
    let other_one = ClosureCells::from_slice(&[second]);
    let two = ClosureCells::from_slice(&[first, second]);
    let same_two = ClosureCells::from_slice(&[first, second]);

    // SAFETY: each source outlives the captures built from it.
    let (empty_cells, one_cell, many) = unsafe {
        (
            CaptureCells::borrow_shared(&none),
            CaptureCells::borrow_shared(&one),
            CaptureCells::borrow_shared(&two),
        )
    };
    assert!(empty_cells.is_borrow_of(&none));
    assert!(!empty_cells.is_borrow_of(&one));
    // A single cell is copied inline: it matches by handle.
    assert!(one_cell.is_borrow_of(&one));
    assert!(!one_cell.is_borrow_of(&other_one));
    assert!(!one_cell.is_borrow_of(&none));
    // A larger borrow matches the allocation, not equal contents.
    assert!(many.is_borrow_of(&two));
    assert!(!many.is_borrow_of(&same_two));
    // An owning clone no longer borrows anything.
    assert!(!many.clone().is_borrow_of(&two));
    assert!(!CaptureCells::from_shared(&two).is_borrow_of(&two));
}

#[test]
fn numeric_cell_accessors_preserve_bits_for_identical_writes() {
    let mut runtime = Runtime::new();
    let first_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let second_nan = f64::from_bits(0x7ff8_0000_0000_0002);
    let cell = runtime.create_cell(Value::Number(-0.0)).unwrap();

    assert_eq!(
        runtime.cell_get_number(cell).unwrap().unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(runtime.cell_set_number(cell, first_nan).unwrap());
    assert_eq!(
        runtime.cell_get_number(cell).unwrap().unwrap().to_bits(),
        first_nan.to_bits()
    );

    assert!(runtime.cell_set_number(cell, first_nan).unwrap());
    assert!(runtime.cell_set_number(cell, second_nan).unwrap());
    assert_eq!(
        runtime.cell_get_number(cell).unwrap().unwrap().to_bits(),
        second_nan.to_bits()
    );
}

#[test]
fn numeric_cell_set_refuses_other_value_kinds_without_mutation() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::from("unchanged")).unwrap();

    assert_eq!(runtime.cell_get_number(cell).unwrap(), None);
    assert!(!runtime.cell_set_number(cell, 42.0).unwrap());
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::from("unchanged"));
}

#[test]
fn temporary_files_are_opaque_byte_streams_with_closed_lifecycle() {
    let mut runtime = Runtime::new();
    let first = runtime.create_temp_file().unwrap();
    let second = runtime.create_temp_file().unwrap();
    assert!(Runtime::raw_equal(&Value::File(first), &Value::File(first)));
    assert!(!Runtime::raw_equal(
        &Value::File(first),
        &Value::File(second)
    ));

    runtime
        .file_write(first, &[Value::bytes(vec![0xff, b'x']), Value::Number(2.5)])
        .unwrap();
    assert_eq!(
        runtime
            .file_seek(first, &Value::from("set"), &Value::Number(0.0))
            .unwrap()
            .to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        runtime.file_read_all(first).unwrap().as_bytes(),
        &[0xff, b'x', b'2', b'.', b'5']
    );
    runtime.file_close(first).unwrap();
    assert!(matches!(
        runtime.file_read_all(first),
        Err(RuntimeError::Message(message)) if message == "attempt to use a closed file"
    ));
    assert!(matches!(
        runtime.file_close(first),
        Err(RuntimeError::Message(message)) if message == "attempt to use a closed file"
    ));
}

#[test]
fn named_files_support_lua_modes_byte_reads_and_default_handles() {
    let path = named_file_path();
    let _ = std::fs::remove_file(&path);

    let mut runtime = Runtime::new();
    let writer = runtime.open_file(&path, FileOpenMode::Write).unwrap();
    runtime.set_default_output(Some(writer)).unwrap();
    assert_eq!(runtime.default_output(), Some(writer));
    runtime
        .file_write(writer, &[Value::bytes(b"first\r\nsecond")])
        .unwrap();
    runtime.file_close(writer).unwrap();
    assert_eq!(runtime.default_output(), Some(writer));

    let reader = runtime.open_file(&path, FileOpenMode::Read).unwrap();
    runtime.set_default_input(Some(reader)).unwrap();
    assert_eq!(
        runtime.file_read_line(reader).unwrap().unwrap().as_bytes(),
        b"first"
    );
    assert_eq!(
        runtime
            .file_read_some(reader, 3)
            .unwrap()
            .unwrap()
            .as_bytes(),
        b"sec"
    );
    assert_eq!(runtime.file_read_all(reader).unwrap().as_bytes(), b"ond");
    runtime.file_close(reader).unwrap();
    assert_eq!(runtime.default_input(), Some(reader));

    std::fs::remove_file(path).unwrap();
}

#[test]
fn explicit_steps_make_proportional_progress_while_stopped() {
    let mut runtime = Runtime::new();
    runtime.collect_full(GcRootBuilder::new().finish());
    runtime.stop_gc();
    // A zero-sized step sweeps a few hundred slots.
    for _ in 0..4096 {
        runtime.create_table().unwrap();
    }

    let mut steps = 0;
    while runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 0) == GcStepResult::InProgress
    {
        steps += 1;
        assert!(steps < 100, "zero-sized GC steps must make progress");
    }
    assert!(steps > 10);
    assert_eq!(
        runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 10_000),
        GcStepResult::Complete
    );
}

#[test]
fn named_file_mode_parser_accepts_binary_suffixes() {
    assert_eq!(FileOpenMode::parse(b"rb"), Ok(FileOpenMode::Read));
    assert_eq!(FileOpenMode::parse(b"a+b"), Ok(FileOpenMode::AppendUpdate));
    assert!(FileOpenMode::parse(b"x").is_err());
}

#[test]
fn temporary_file_handles_are_kept_by_values_and_reclaimed_when_unrooted() {
    let mut runtime = Runtime::new();
    let file = runtime.create_temp_file().unwrap();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("file"), Value::File(file))
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(table));
    assert_eq!(runtime.collect_full(roots.finish()).files, 0);
    assert!(runtime.file_read_all(file).is_ok());

    assert_eq!(runtime.collect_full(GcRootBuilder::new().finish()).files, 1);
    assert!(matches!(
        runtime.file_read_all(file),
        Err(RuntimeError::InvalidFile(handle)) if handle == file
    ));
}

#[test]
fn standard_error_file_has_a_separate_byte_sink() {
    let mut runtime = Runtime::new();
    let stderr = runtime.create_stderr_file().unwrap();
    runtime
        .file_write(stderr, &[Value::bytes(b"diagnostic\xff")])
        .unwrap();
    assert_eq!(runtime.take_output(), [] as [u8; 0]);
    assert_eq!(runtime.take_error_output(), b"diagnostic\xff");
}

#[test]
fn repeated_table_writes_enqueue_one_unmarked_reference() {
    let mut runtime = Runtime::new();
    let owner = runtime.create_table().unwrap();
    let root = Value::Table(owner);
    advance_to_incremental_sweep(&mut runtime, &root);

    let child = runtime.create_table().unwrap();
    for index in 0..32 {
        runtime
            .table_set(
                child,
                &Value::Number(f64::from(index)),
                Value::Boolean(true),
            )
            .unwrap();
    }
    for _ in 0..1_000 {
        runtime
            .table_set(owner, &Value::from("child"), Value::Table(child))
            .unwrap();
    }
    let cycle = runtime.incremental_gc.as_ref().unwrap();
    assert!(
        cycle
            .marker
            .gray
            .iter()
            .filter(|reference| matches!(reference, crate::runtime::gc::GcReference::Table(handle) if *handle == child))
            .count()
            <= 1,
        "repeated writes must not queue duplicate child marks"
    );

    for steps in 1..=256 {
        if runtime.gc_step_with_roots(incremental_roots(&root), 0) == GcStepResult::Complete {
            assert!(steps < 256, "collection work grew with duplicate writes");
            return;
        }
    }
    panic!("repeated writes starved incremental collection");
}

#[test]
fn io_error_text_drops_the_os_error_suffix() {
    let missing = std::fs::read("/nonexistent/oximond-test-file").unwrap_err();
    assert_eq!(
        crate::runtime::io_error_text(&missing),
        "No such file or directory"
    );
    assert_eq!(crate::runtime::io_error_code(&missing), Some(2));
    let custom = std::io::Error::other("custom message");
    assert_eq!(crate::runtime::io_error_text(&custom), "custom message");
}
