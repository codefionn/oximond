use super::*;

#[test]
fn native_userdata_payload_is_counted_until_collection() {
    let mut runtime = Runtime::new();
    let baseline = runtime.gc_count_bytes();
    let userdata = runtime.create_userdata().unwrap();
    let header_bytes = runtime.gc_count_bytes();
    runtime.set_userdata_external_size(userdata, 4096).unwrap();
    assert_eq!(runtime.gc_count_bytes(), header_bytes + 4096);
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Userdata(userdata));
    runtime.collect_full(roots.finish());
    assert_eq!(runtime.gc_count_bytes(), header_bytes + 4096);
    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(runtime.gc_count_bytes(), baseline);
    assert!(runtime.set_userdata_external_size(userdata, 1).is_err());
}

#[test]
fn userdata_environment_retains_its_graph_only_while_the_userdata_is_reachable() {
    let mut runtime = Runtime::new();
    let userdata = runtime.create_userdata().unwrap();
    let environment = runtime.create_table().unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .table_set(environment, &Value::from("child"), Value::Table(child))
        .unwrap();
    runtime
        .set_userdata_environment(userdata, environment)
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Userdata(userdata));
    runtime.collect_full(roots.finish());
    assert_eq!(
        runtime.userdata_environment(userdata).unwrap(),
        Some(environment)
    );
    assert_eq!(
        runtime
            .table_get(environment, &Value::from("child"))
            .unwrap(),
        Value::Table(child)
    );
    assert!(runtime.table_length(child).is_ok());
    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(runtime.userdata_environment(userdata).is_err());
    assert!(runtime.table_length(environment).is_err());
    assert!(runtime.table_length(child).is_err());
}

#[test]
fn userdata_uservalues_retain_their_referents() {
    let mut runtime = Runtime::new();
    let userdata = runtime.create_userdata().unwrap();
    runtime.set_userdata_uservalue_count(userdata, 3).unwrap();
    let child = runtime.create_table().unwrap();
    assert!(
        runtime
            .set_userdata_uservalue(userdata, 3, Value::Table(child))
            .unwrap()
    );
    assert!(
        !runtime
            .set_userdata_uservalue(userdata, 4, Value::Table(child))
            .unwrap()
    );
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Userdata(userdata));
    runtime.collect_full(roots.finish());
    assert!(runtime.table_length(child).is_ok());
    assert_eq!(
        runtime.userdata_uservalue(userdata, 3).unwrap(),
        Some(Value::Table(child))
    );
    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(runtime.table_length(child).is_err());
}

#[test]
fn file_uservalue_retains_its_referent() {
    let mut runtime = Runtime::new();
    let file = runtime.create_temp_file().unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .set_file_uservalue(file, Value::Table(child))
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::File(file));
    runtime.collect_full(roots.finish());
    assert_eq!(runtime.file_uservalue(file).unwrap(), Value::Table(child));
    assert!(runtime.table_length(child).is_ok());
    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(runtime.table_length(child).is_err());
}

#[test]
fn mixed_tombstones_resume_next_and_survive_collection_cleanup() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::Number(1.0), Value::from("one"))
        .unwrap();
    runtime
        .table_set(table, &Value::Number(3.0), Value::from("three"))
        .unwrap();
    runtime
        .table_set(table, &Value::from("name"), Value::from("value"))
        .unwrap();

    runtime
        .table_set(table, &Value::Number(1.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(table, &Value::Number(1.0)).unwrap(),
        (Value::Number(3.0), Value::from("three"))
    );
    runtime
        .table_set(table, &Value::Number(3.0), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(table, &Value::Number(3.0)).unwrap(),
        (Value::from("name"), Value::from("value"))
    );
    runtime
        .table_set(table, &Value::from("name"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(table, &Value::from("name")).unwrap(),
        (Value::Nil, Value::Nil)
    );

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(table));
    assert_eq!(runtime.collect_full(roots.finish()).reclaimed(), 0);
}

#[test]
fn arena_sweep_and_accounting_visit_only_live_slots() {
    let mut arena = Arena::default();
    let handles = (0..128)
        .map(|value| arena.insert(value).unwrap())
        .collect::<Vec<_>>();
    for handle in handles.iter().copied() {
        assert!(arena.remove(handle).is_some());
    }
    assert_eq!(arena.slots.len(), 128);
    assert!(arena.live().next().is_none());

    let accounted = std::cell::Cell::new(0);
    assert_eq!(
        arena.accounted_bytes(|_| {
            accounted.set(accounted.get() + 1);
            std::mem::size_of::<usize>()
        }),
        0
    );
    assert_eq!(accounted.get(), 0);
    assert_eq!(arena.sweep(1), 0);

    let fresh = arena.insert(99).unwrap();
    assert_eq!(arena.get(fresh), Some(&99));
    assert_eq!(arena.get(handles[0]), None);
}

/// An arena of `count` objects in which every third one is marked with
/// `epoch`, as a mark phase would leave it, and the keys of the marked ones.
fn marked_every_third(count: u32, epoch: u64) -> (Arena<u32>, Vec<ArenaKey>) {
    let mut arena = Arena::default();
    let mut kept = Vec::new();
    for value in 0..count {
        let key = arena.insert(value).unwrap();
        if value % 3 == 0 {
            assert!(arena.mark(key, epoch, false));
            kept.push(key);
        }
    }
    (arena, kept)
}

#[test]
fn a_batched_major_sweep_takes_the_steps_of_a_slot_by_slot_one() {
    use crate::runtime::arena::{MajorSweepCursor, SweepStep};
    let (mut by_slot, kept) = marked_every_third(50, 7);
    let mut cursor = MajorSweepCursor::default();
    let (mut freed_by_slot, mut steps) = (0, 0);
    while let Some(step) =
        by_slot.sweep_major_one(7, &mut cursor, &mut |_: &u32| {}, &mut |_: &mut u32| {})
    {
        freed_by_slot += usize::from(matches!(step, SweepStep::Freed));
        steps += 1;
    }

    let (mut batched, _) = marked_every_third(50, 7);
    let mut cursor = MajorSweepCursor::default();
    let (mut kept_seen, mut freed_seen) = (0, 0);
    let mut calls = 0;
    let mut slots_used = 0;
    loop {
        let mut slots = 8;
        let swept = batched.sweep_major_run(
            7,
            &mut cursor,
            &mut slots,
            |_| kept_seen += 1,
            |_| freed_seen += 1,
        );
        slots_used += 8 - slots;
        calls += 1;
        if swept {
            break;
        }
        assert_eq!(slots, 0, "a batch that is not done uses all its slots");
    }
    // One slot per step, and the step that finds the arena swept costs one.
    assert_eq!(slots_used, steps + 1);
    assert_eq!(calls, (steps + 1).div_ceil(8));
    assert_eq!(freed_seen, freed_by_slot);
    assert_eq!(kept_seen, kept.len());
    assert_eq!(freed_seen + kept_seen, 50);
    for key in &kept {
        assert!(batched.get(*key).is_some(), "a marked object survives");
    }
    assert_eq!(batched.live().count(), kept.len());
    assert_eq!(by_slot.live().count(), kept.len());
}

#[test]
fn a_major_sweep_with_no_slots_left_changes_nothing() {
    use crate::runtime::arena::MajorSweepCursor;
    let (mut arena, _) = marked_every_third(10, 3);
    let mut cursor = MajorSweepCursor::default();
    let mut slots = 0;
    let touched = std::cell::Cell::new(false);
    let swept = arena.sweep_major_run(
        3,
        &mut cursor,
        &mut slots,
        |_| touched.set(true),
        |_| touched.set(true),
    );
    assert!(!swept);
    assert!(!touched.get());
    assert_eq!(arena.live().count(), 10);
}

#[test]
fn cell_writes_keep_number_bits() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::Number(0.0)).unwrap();

    runtime.cell_set(cell, Value::Number(-0.0)).unwrap();
    let Value::Number(value) = runtime.cell_get(cell).unwrap() else {
        panic!("cell write must retain a number");
    };
    assert_eq!(value.to_bits(), (-0.0_f64).to_bits());

    assert!(runtime.cells.remove(cell.0).is_some());
    assert!(matches!(
        runtime.cell_set(cell, Value::Number(4.0)),
        Err(RuntimeError::InvalidCell(handle)) if handle == cell
    ));
}

#[test]
fn collection_traces_closure_environments() {
    let mut runtime = Runtime::new();
    let environment = runtime.create_table().unwrap();
    let closure = runtime
        .create_closure_in_environment(
            Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
            Vec::new(),
            environment,
        )
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.closure(closure);
    assert_eq!(runtime.collect_full(roots.finish()).reclaimed(), 0);
    runtime
        .table_set(environment, &Value::from("alive"), Value::Boolean(true))
        .unwrap();

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.closures, 1);
    assert_eq!(stats.tables, 1);
    assert!(matches!(
        runtime.closure_environment(closure),
        Err(RuntimeError::InvalidClosure(handle)) if handle == closure
    ));
    assert!(matches!(
        runtime.table_get(environment, &Value::from("alive")),
        Err(RuntimeError::InvalidTable(handle)) if handle == environment
    ));
}

#[test]
fn collection_reclaims_unrooted_table_cycles_and_leaves_stale_handles_invalid() {
    let mut runtime = Runtime::new();
    let first = runtime.create_table().unwrap();
    let second = runtime.create_table().unwrap();
    runtime
        .table_set(first, &Value::from("next"), Value::Table(second))
        .unwrap();
    runtime
        .table_set(second, &Value::from("next"), Value::Table(first))
        .unwrap();

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.tables, 2);
    assert!(matches!(
        runtime.table_get(first, &Value::from("next")),
        Err(RuntimeError::InvalidTable(handle)) if handle == first
    ));
    assert!(matches!(
        runtime.table_get(second, &Value::from("next")),
        Err(RuntimeError::InvalidTable(handle)) if handle == second
    ));
}

#[test]
fn collection_keeps_live_cycles_and_closure_cells_from_explicit_roots() {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    runtime
        .table_set(table, &Value::from("self"), Value::Table(table))
        .unwrap();
    let cell = runtime.create_cell(Value::Table(table)).unwrap();
    let closure = runtime
        .create_closure(
            Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
            vec![cell],
        )
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.closure(closure);
    assert_eq!(runtime.collect_full(roots.finish()).reclaimed(), 0);
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Table(table));
    assert_eq!(
        runtime.table_get(table, &Value::from("self")).unwrap(),
        Value::Table(table)
    );

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.tables, 1);
    assert_eq!(stats.closures, 1);
    assert_eq!(stats.cells, 1);
    assert!(matches!(
        runtime.closure_snapshot(closure),
        Err(RuntimeError::InvalidClosure(handle)) if handle == closure
    ));
    assert!(matches!(
        runtime.cell_get(cell),
        Err(RuntimeError::InvalidCell(handle)) if handle == cell
    ));
}

#[test]
fn weak_keys_drop_unreachable_objects_but_keep_value_cycles_and_strings() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("k"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();

    let dead = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Table(dead), Value::Number(1.0))
        .unwrap();
    let retained_by_value = runtime.create_table().unwrap();
    runtime
        .table_set(
            weak,
            &Value::Table(retained_by_value),
            Value::Table(retained_by_value),
        )
        .unwrap();
    runtime
        .table_set(
            weak,
            &Value::from("string key"),
            Value::from("string value"),
        )
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    let stats = runtime.collect_full(roots.finish());
    assert_eq!(stats.tables, 1);
    assert_eq!(
        runtime.table_get(weak, &Value::Table(dead)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime
            .table_get(weak, &Value::Table(retained_by_value))
            .unwrap(),
        Value::Table(retained_by_value)
    );
    assert_eq!(
        runtime.table_get(weak, &Value::from("string key")).unwrap(),
        Value::from("string value")
    );
}

fn weak_key_table(runtime: &mut Runtime) -> TableHandle {
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("k"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();
    weak
}

/// Stores `weak[key] = {key}`: a value that refers back to its own key.
fn store_key_cycle(runtime: &mut Runtime, weak: TableHandle, key: TableHandle) -> TableHandle {
    let value = runtime.create_table().unwrap();
    runtime
        .table_set(value, &Value::Number(1.0), Value::Table(key))
        .unwrap();
    runtime
        .table_set(weak, &Value::Table(key), Value::Table(value))
        .unwrap();
    value
}

fn is_freed(runtime: &Runtime, table: TableHandle) -> bool {
    runtime.table_get(table, &Value::Nil).is_err()
}

#[test]
fn ephemerons_collect_a_key_reachable_only_through_its_value() {
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(true);
    let weak = weak_key_table(&mut runtime);
    let dead_key = runtime.create_table().unwrap();
    let dead_value = store_key_cycle(&mut runtime, weak, dead_key);
    let live_key = runtime.create_table().unwrap();
    let live_value = store_key_cycle(&mut runtime, weak, live_key);

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    roots.value(&Value::Table(live_key));
    let stats = runtime.collect_full(roots.finish());

    assert_eq!(stats.tables, 2);
    assert!(is_freed(&runtime, dead_key));
    assert!(is_freed(&runtime, dead_value));
    assert_eq!(
        runtime.table_get(weak, &Value::Table(live_key)).unwrap(),
        Value::Table(live_value)
    );
    assert_eq!(
        runtime.table_next(weak, &Value::Table(live_key)).unwrap().0,
        Value::Nil
    );
}

#[test]
fn ephemerons_mark_values_whose_keys_are_reached_after_the_table_is_traced() {
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(true);
    let weak = weak_key_table(&mut runtime);
    // `weak[k1] = {k2}` and `weak[k2] = {k3}`, with only `k1` rooted, and
    // `k1` reached only after `weak`: each value is marked in a later round.
    let keys = [(); 3].map(|()| runtime.create_table().unwrap());
    let mut values = Vec::new();
    for pair in keys.windows(2) {
        let value = runtime.create_table().unwrap();
        runtime
            .table_set(value, &Value::Number(1.0), Value::Table(pair[1]))
            .unwrap();
        runtime
            .table_set(weak, &Value::Table(pair[0]), Value::Table(value))
            .unwrap();
        values.push(value);
    }
    let holder = runtime.create_table().unwrap();
    runtime
        .table_set(holder, &Value::Number(1.0), Value::Table(keys[0]))
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    roots.value(&Value::Table(holder));
    assert_eq!(runtime.collect_full(roots.finish()).tables, 0);
    for (key, value) in keys.iter().zip(&values) {
        assert_eq!(
            runtime.table_get(weak, &Value::Table(*key)).unwrap(),
            Value::Table(*value)
        );
    }
}

#[test]
fn ephemeron_chains_across_two_tables_are_collected_together() {
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(true);
    let first = weak_key_table(&mut runtime);
    let second = weak_key_table(&mut runtime);
    // `first[a] = {b}` and `second[b] = {a}`: each key is reachable only
    // through the other table's value.
    let a = runtime.create_table().unwrap();
    let b = runtime.create_table().unwrap();
    let a_value = runtime.create_table().unwrap();
    runtime
        .table_set(a_value, &Value::Number(1.0), Value::Table(b))
        .unwrap();
    runtime
        .table_set(first, &Value::Table(a), Value::Table(a_value))
        .unwrap();
    let b_value = runtime.create_table().unwrap();
    runtime
        .table_set(b_value, &Value::Number(1.0), Value::Table(a))
        .unwrap();
    runtime
        .table_set(second, &Value::Table(b), Value::Table(b_value))
        .unwrap();

    let roots = |a: Option<TableHandle>| {
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(first));
        roots.value(&Value::Table(second));
        if let Some(a) = a {
            roots.value(&Value::Table(a));
        }
        roots.finish()
    };
    // Rooting `a` keeps the whole chain through both tables.
    assert_eq!(runtime.collect_full(roots(Some(a))).tables, 0);
    assert_eq!(
        runtime.table_get(second, &Value::Table(b)).unwrap(),
        Value::Table(b_value)
    );
    // Without it, both entries go at once.
    assert_eq!(runtime.collect_full(roots(None)).tables, 4);
    for table in [a, b, a_value, b_value] {
        assert!(is_freed(&runtime, table));
    }
    for weak in [first, second] {
        assert_eq!(runtime.table_next(weak, &Value::Nil).unwrap().0, Value::Nil);
    }
}

#[test]
fn incremental_and_minor_collections_treat_weak_keys_as_ephemerons() {
    for minor in [false, true] {
        let mut runtime = Runtime::new();
        runtime.set_ephemerons_enabled(true);
        let weak = weak_key_table(&mut runtime);
        let live_key = runtime.create_table().unwrap();
        let root = runtime.create_table().unwrap();
        runtime
            .table_set(root, &Value::from("weak"), Value::Table(weak))
            .unwrap();
        runtime
            .table_set(root, &Value::from("key"), Value::Table(live_key))
            .unwrap();
        if minor {
            // Old table, young entries: a minor collection must clear them.
            runtime.collect_full(incremental_roots(&Value::Table(root)));
        }
        let live_value = store_key_cycle(&mut runtime, weak, live_key);
        let dead_key = runtime.create_table().unwrap();
        let dead_value = store_key_cycle(&mut runtime, weak, dead_key);
        if minor {
            assert_eq!(
                runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(root))), 1),
                GcStepResult::Complete
            );
        } else {
            advance_to_incremental_sweep(&mut runtime, &Value::Table(root));
            finish_incremental_cycle(&mut runtime, &Value::Table(root));
        }
        assert!(is_freed(&runtime, dead_key), "minor: {minor}");
        assert!(is_freed(&runtime, dead_value), "minor: {minor}");
        assert_eq!(
            runtime.table_get(weak, &Value::Table(live_key)).unwrap(),
            Value::Table(live_value)
        );
    }
}

#[test]
fn lua51_weak_keys_keep_a_key_reachable_through_its_value() {
    // Lua 5.1 and LuaJIT mark weak-key values strongly, so a value that
    // refers back to its key keeps the entry forever.
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(false);
    let weak = weak_key_table(&mut runtime);
    let key = runtime.create_table().unwrap();
    let value = store_key_cycle(&mut runtime, weak, key);
    for _ in 0..3 {
        assert_eq!(
            runtime
                .collect_full(incremental_roots(&Value::Table(weak)))
                .tables,
            0
        );
    }
    assert_eq!(
        runtime.table_get(weak, &Value::Table(key)).unwrap(),
        Value::Table(value)
    );
}

#[test]
fn weak_values_clear_array_and_map_objects_but_keep_strings_and_strong_keys() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("v"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();

    let dead_array_value = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Number(1.0), Value::Table(dead_array_value))
        .unwrap();
    let dead_map_value = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::from("dead"), Value::Table(dead_map_value))
        .unwrap();
    let retained_by_key = runtime.create_table().unwrap();
    runtime
        .table_set(
            weak,
            &Value::Table(retained_by_key),
            Value::Table(retained_by_key),
        )
        .unwrap();
    runtime
        .table_set(weak, &Value::from("text"), Value::from("not weak"))
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    let stats = runtime.collect_full(roots.finish());
    assert_eq!(stats.tables, 2);
    assert_eq!(
        runtime.table_length(weak).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Number(1.0)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime.table_get(weak, &Value::from("dead")).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime
            .table_get(weak, &Value::Table(retained_by_key))
            .unwrap(),
        Value::Table(retained_by_key)
    );
    assert_eq!(
        runtime.table_get(weak, &Value::from("text")).unwrap(),
        Value::from("not weak")
    );
}

#[test]
fn incremental_weak_values_clear_promoted_maps_and_keep_ordered_next() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("v"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();

    for index in 0..=HASHED_TABLE_ENTRY_THRESHOLD {
        let key = Value::from(format!("field-{index:02}"));
        let value = if index % 2 == 0 {
            Value::Table(runtime.create_table().unwrap())
        } else {
            Value::from("retained string")
        };
        runtime.table_set(weak, &key, value).unwrap();
    }

    advance_to_incremental_sweep(&mut runtime, &Value::Table(weak));
    finish_incremental_cycle(&mut runtime, &Value::Table(weak));

    assert_eq!(
        runtime.table_get(weak, &Value::from("field-00")).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime.table_get(weak, &Value::from("field-01")).unwrap(),
        Value::from("retained string")
    );
    runtime
        .table_set(weak, &Value::from("field-01"), Value::Nil)
        .unwrap();
    assert_eq!(
        runtime.table_next(weak, &Value::from("field-01")).unwrap(),
        (Value::from("field-03"), Value::from("retained string"))
    );
}

#[test]
fn incremental_weak_keys_clear_promoted_maps_but_keep_rooted_keys() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("k"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();

    let mut rooted_key = None;
    let mut dead_key = None;
    for index in 0..=HASHED_TABLE_ENTRY_THRESHOLD {
        let key = runtime.create_table().unwrap();
        if index == 0 {
            dead_key = Some(key);
        }
        if index == HASHED_TABLE_ENTRY_THRESHOLD {
            rooted_key = Some(key);
        }
        runtime
            .table_set(weak, &Value::Table(key), Value::from("value"))
            .unwrap();
    }
    let rooted_key = rooted_key.expect("last table key was retained");
    let dead_key = dead_key.expect("first table key was retained for inspection");
    let root = runtime.create_table().unwrap();
    runtime
        .table_set(root, &Value::from("weak"), Value::Table(weak))
        .unwrap();
    runtime
        .table_set(root, &Value::from("key"), Value::Table(rooted_key))
        .unwrap();
    advance_to_incremental_sweep(&mut runtime, &Value::Table(root));
    finish_incremental_cycle(&mut runtime, &Value::Table(root));

    assert_eq!(
        runtime.table_get(weak, &Value::Table(rooted_key)).unwrap(),
        Value::from("value")
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Table(dead_key)).unwrap(),
        Value::Nil
    );
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(root));
    assert_eq!(runtime.collect_full(roots.finish()).tables, 0);
}

#[test]
fn fully_weak_tables_keep_only_externally_reachable_objects_and_strings() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("vk"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();

    let dead_key = runtime.create_table().unwrap();
    let dead_value = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Table(dead_key), Value::Table(dead_value))
        .unwrap();
    let live_key = runtime.create_table().unwrap();
    let live_value = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Table(live_key), Value::Table(live_value))
        .unwrap();
    runtime
        .table_set(weak, &Value::from("$"), Value::from("$"))
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.values([
        &Value::Table(weak),
        &Value::Table(live_key),
        &Value::Table(live_value),
    ]);
    let stats = runtime.collect_full(roots.finish());
    assert_eq!(stats.tables, 2);
    assert_eq!(
        runtime.table_get(weak, &Value::Table(dead_key)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Table(live_key)).unwrap(),
        Value::Table(live_value)
    );
    assert_eq!(
        runtime.table_get(weak, &Value::from("$")).unwrap(),
        Value::from("$")
    );
}

#[test]
fn default_file_handles_are_gc_roots_until_replaced() {
    let mut runtime = Runtime::new();
    let file = runtime.create_temp_file().unwrap();
    runtime.set_default_input(Some(file)).unwrap();
    assert_eq!(runtime.collect_full(GcRootBuilder::new().finish()).files, 0);

    runtime.set_default_input(None).unwrap();
    assert_eq!(runtime.collect_full(GcRootBuilder::new().finish()).files, 1);
}

#[test]
fn allocation_debt_waits_for_restart_before_requesting_a_collection() {
    let mut runtime = Runtime::new();
    runtime.set_gc_mode(GcMode::Incremental);
    runtime.collect_full(GcRootBuilder::new().finish());
    for _ in 0..128 {
        runtime.create_table().unwrap();
    }
    assert!(runtime.gc_needs_collection());

    runtime.stop_gc();
    assert!(!runtime.gc_is_running());
    assert!(!runtime.gc_needs_collection());

    runtime.restart_gc();
    assert!(runtime.gc_is_running());
    assert!(runtime.gc_needs_collection());
}

#[test]
fn userdata_finalizers_are_queued_in_reverse_registration_order() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let first = runtime.create_userdata().unwrap();
    let second = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(first), Some(metatable))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(second), Some(metatable))
        .unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.userdatas, 0);
    let queued = runtime.take_pending_finalizer().unwrap().unwrap();
    assert_eq!(userdata_object(&queued.object), second);
    assert_eq!(queued.handler, Value::NativeFunction(handler));
    let queued = runtime.take_pending_finalizer().unwrap().unwrap();
    assert_eq!(userdata_object(&queued.object), first);
    assert!(runtime.take_pending_finalizer().unwrap().is_none());

    let stats = runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(stats.userdatas, 2);
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
}

#[test]
fn shared_finalizer_graph_keeps_all_unreachable_candidates_queued() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let members = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    runtime
        .table_set(metatable, &Value::from("members"), Value::Table(members))
        .unwrap();

    let mut objects = Vec::new();
    for index in 1..=11 {
        let userdata = runtime.create_userdata().unwrap();
        runtime
            .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
            .unwrap();
        runtime
            .table_set(
                members,
                &Value::Number(f64::from(index)),
                Value::Userdata(userdata),
            )
            .unwrap();
        objects.push(userdata);
    }

    assert_eq!(
        runtime
            .collect_full(GcRootBuilder::new().finish())
            .userdatas,
        0
    );
    objects.reverse();
    for expected in objects {
        assert_eq!(finalized_userdata(&mut runtime), expected);
    }
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
}

#[test]
fn finalized_userdata_can_be_resurrected_until_a_later_collection() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();

    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(finalized_userdata(&mut runtime), userdata);
    let mut resurrected = GcRootBuilder::new();
    resurrected.value(&Value::Userdata(userdata));
    assert_eq!(runtime.collect_full(resurrected.finish()).userdatas, 0);
    assert!(runtime.raw_metatable(&Value::Userdata(userdata)).is_ok());
    assert_eq!(
        runtime
            .collect_full(GcRootBuilder::new().finish())
            .userdatas,
        1
    );
    assert!(matches!(
        runtime.raw_metatable(&Value::Userdata(userdata)),
        Err(RuntimeError::InvalidUserdata(handle)) if handle == userdata
    ));
}

#[test]
fn pending_userdata_keeps_weak_keys_but_not_weak_values_until_finalized() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let userdata_metatable = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            userdata_metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(userdata_metatable))
        .unwrap();

    let weak_metatable = runtime.create_table().unwrap();
    runtime
        .table_set(weak_metatable, &Value::from("__mode"), Value::from("kv"))
        .unwrap();
    let weak = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(weak_metatable))
        .unwrap();
    runtime
        .table_set(weak, &Value::Userdata(userdata), Value::Number(11.0))
        .unwrap();
    runtime
        .table_set(weak, &Value::Number(11.0), Value::Userdata(userdata))
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    runtime.collect_full(roots.finish());
    assert_eq!(
        runtime.table_get(weak, &Value::Userdata(userdata)).unwrap(),
        Value::Number(11.0)
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Number(11.0)).unwrap(),
        Value::Nil
    );
    assert_eq!(finalized_userdata(&mut runtime), userdata);

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    assert_eq!(runtime.collect_full(roots.finish()).userdatas, 1);
    assert_eq!(
        runtime.table_get(weak, &Value::Userdata(userdata)).unwrap(),
        Value::Nil
    );
}

#[test]
fn collector_mode_defaults_to_incremental_and_can_switch() {
    let mut runtime = Runtime::new();
    assert_eq!(runtime.gc_mode(), GcMode::Incremental);
    assert_eq!(
        runtime.set_gc_mode(GcMode::Generational),
        GcMode::Incremental
    );
    assert_eq!(runtime.gc_mode(), GcMode::Generational);
}

/// Adds `count` tables to `holder` and runs one automatic step, a minor
/// collection, in which every one of them survives.
fn grow_and_collect_minor(runtime: &mut Runtime, holder: TableHandle, count: usize) {
    for _ in 0..count {
        let table = runtime.create_table().unwrap();
        let key = Value::Table(table);
        runtime
            .table_set(holder, &key, Value::Boolean(true))
            .unwrap();
    }
    assert_eq!(
        runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1),
        GcStepResult::Complete
    );
}

#[test]
fn default_policy_pauses_minor_collections_in_which_everything_survives() {
    let mut runtime = Runtime::new();
    let holder = runtime.create_table().unwrap();
    grow_and_collect_minor(&mut runtime, holder, 1000);
    assert!(runtime.gc_minors_active(), "one bad minor is not a verdict");
    grow_and_collect_minor(&mut runtime, holder, 1000);
    assert!(!runtime.gc_minors_active());
    // Major cycles run until the pause is over; then minors resume.
    let mut majors = 0;
    while !runtime.gc_minors_active() {
        // The major cycle is due, not a trial.
        runtime.gc_debt = runtime.gc_pause_major_debt;
        while runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1)
            != GcStepResult::Complete
        {}
        majors += 1;
        assert!(majors < 10, "the pause never ended");
    }
    assert_eq!(majors, 2);
}

/// Pauses minor collections with two minors in which everything survives.
fn pause_minor_collections(runtime: &mut Runtime, holder: TableHandle) {
    grow_and_collect_minor(runtime, holder, 1000);
    grow_and_collect_minor(runtime, holder, 1000);
    assert!(!runtime.gc_minors_active());
}

#[test]
fn a_trial_minor_ends_the_pause_once_the_young_objects_die() {
    let mut runtime = Runtime::new();
    let holder = runtime.create_table().unwrap();
    pause_minor_collections(&mut runtime, holder);
    // The program stopped building: what it allocates is garbage. The first
    // automatic step after a quarter nursery must not wait for a major cycle
    // that marks the whole old heap.
    for _ in 0..200_000 {
        runtime.create_table().unwrap();
    }
    runtime.gc_debt = runtime.gc_threshold;
    assert!(runtime.gc_trial_minor_due());
    // The trial passes and starts the major cycle that clears the pause's
    // floating garbage. That cycle is incremental: the step that ends the
    // pause does not collect the whole heap at once.
    let tables = runtime.tables.live_slots().count();
    assert_eq!(
        runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1),
        GcStepResult::InProgress
    );
    assert!(runtime.incremental_gc.is_some());
    assert!(!runtime.gc_minors_active());
    assert_eq!(
        runtime.tables.live_slots().count(),
        tables,
        "the first step swept nothing"
    );
    let mut steps = 1;
    while runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1)
        != GcStepResult::Complete
    {
        steps += 1;
        assert!(steps < 10_000, "the cycle never finished");
    }
    assert!(steps > 2, "the cycle ran in {steps} steps");
    // Its end lifts the pause, and its garbage is gone.
    assert!(runtime.gc_minors_active());
    assert!(runtime.incremental_gc.is_none());
    assert!(runtime.tables.live_slots().count() < tables / 10);
}

#[test]
fn a_trial_minor_keeps_the_pause_while_the_young_objects_survive() {
    let mut runtime = Runtime::new();
    let holder = runtime.create_table().unwrap();
    pause_minor_collections(&mut runtime, holder);
    for _ in 0..4000 {
        let table = runtime.create_table().unwrap();
        runtime
            .table_set(holder, &Value::Table(table), Value::Boolean(true))
            .unwrap();
    }
    runtime.gc_debt = runtime.gc_threshold;
    let young = runtime.tables.young.len();
    assert_eq!(
        runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1),
        GcStepResult::Complete
    );
    assert!(!runtime.gc_minors_active());
    // The trial neither collected nor promoted anything.
    assert_eq!(runtime.tables.young.len(), young);
    assert!(runtime.gc_threshold > runtime.gc_debt);
}

#[test]
fn minor_collections_that_free_their_garbage_stay_active() {
    let mut runtime = Runtime::new();
    let holder = runtime.create_table().unwrap();
    for _ in 0..4 {
        for _ in 0..1000 {
            runtime.create_table().unwrap();
        }
        assert_eq!(
            runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1),
            GcStepResult::Complete
        );
        assert!(runtime.gc_minors_active());
    }
}

#[test]
fn selected_generational_mode_never_pauses_minor_collections() {
    let mut runtime = Runtime::new();
    runtime.set_gc_mode(GcMode::Generational);
    let holder = runtime.create_table().unwrap();
    for _ in 0..4 {
        grow_and_collect_minor(&mut runtime, holder, 1000);
        assert!(runtime.gc_minors_active());
    }
}

#[test]
fn minor_gc_parameter_changes_when_allocation_requests_collection() {
    let mut runtime = Runtime::new();
    runtime.set_gc_mode(GcMode::Generational);
    let retained = runtime.create_table().unwrap();
    for index in 1..=50_000 {
        runtime
            .table_set(
                retained,
                &Value::Number(f64::from(index)),
                Value::Number(1.0),
            )
            .unwrap();
    }
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(retained));
    runtime.collect_full(roots.finish());
    runtime.set_gc_parameter(0, 1000.0).unwrap();
    // More than the smallest nursery, less than ten times the retained heap.
    for _ in 0..32_000 {
        runtime.create_table().unwrap();
    }
    assert!(
        !runtime.gc_needs_collection(),
        "larger nursery defers collection"
    );
    runtime.set_gc_parameter(0, 1.0).unwrap();
    assert!(
        runtime.gc_needs_collection(),
        "smaller nursery requests collection with the same allocation debt"
    );
}

#[test]
fn gmatch_iterators_have_function_identity_and_are_collected() {
    let mut runtime = Runtime::new();
    let iterator = runtime
        .create_gmatch_iterator("one two".into(), "%a+".into())
        .unwrap();
    let value = Value::GmatchIterator(iterator);
    let table = runtime.create_table().unwrap();

    assert_eq!(value.type_name(), "function");
    assert!(Runtime::raw_equal(&value, &value));
    runtime
        .table_set(table, &value, Value::from("stored"))
        .unwrap();
    assert_eq!(
        runtime.table_get(table, &value).unwrap(),
        Value::from("stored")
    );

    let mut roots = GcRootBuilder::new();
    roots.value(&value);
    assert_eq!(runtime.collect_full(roots.finish()).gmatch_iterators, 0);
    assert_eq!(
        runtime.gmatch_iterator_snapshot(iterator).unwrap().search(),
        0
    );

    assert_eq!(
        runtime
            .collect_full(GcRootBuilder::new().finish())
            .gmatch_iterators,
        1
    );
    assert!(matches!(
        runtime.gmatch_iterator_snapshot(iterator),
        Err(RuntimeError::InvalidGmatchIterator(handle)) if handle == iterator
    ));
}

#[test]
fn collection_threshold_excludes_garbage_allocated_before_the_cycle() {
    for incremental in [false, true] {
        let mut runtime = Runtime::new();
        runtime.set_gc_mode(GcMode::Incremental);
        for _ in 0..256 {
            runtime.create_table().unwrap();
        }
        if incremental {
            finish_incremental_cycle(&mut runtime, &Value::Nil);
        } else {
            runtime.collect_full(incremental_roots(&Value::Nil));
        }
        let expected =
            crate::runtime::gc::heap::scale_gc_work(runtime.gc_count_bytes(), runtime.gc_pause)
                .max(crate::runtime::gc::GC_MIN_THRESHOLD_BYTES);
        assert_eq!(runtime.gc_threshold, expected);
        assert!(!runtime.gc_needs_collection());
    }
}

#[test]
fn incremental_finalizer_candidate_becoming_reachable_is_not_queued() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }
    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    runtime.set_gc_step_multiplier(0.01);
    // Marking starts while the userdata is unreachable. It becomes a root
    // before the atomic step, which selects finalizers from fresh roots.
    assert_eq!(
        runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 0),
        GcStepResult::InProgress
    );
    assert!(
        runtime
            .incremental_gc
            .as_ref()
            .is_some_and(|cycle| cycle.phase == crate::runtime::gc::IncrementalGcPhase::Mark)
    );
    finish_incremental_cycle(&mut runtime, &Value::Userdata(userdata));
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
    runtime.collect_full(incremental_roots(&Value::Nil));
    assert_eq!(finalized_userdata(&mut runtime), userdata);
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
}

/// Builds a parent -> child graph whose child stores `answer`.
fn parent_child_graph(runtime: &mut Runtime, answer: f64) -> (TableHandle, TableHandle) {
    let parent = runtime.create_table().unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .table_set(parent, &Value::from("child"), Value::Table(child))
        .unwrap();
    runtime
        .table_set(child, &Value::from("answer"), Value::Number(answer))
        .unwrap();
    (parent, child)
}

/// Starts a major cycle whose first step traces exactly the last root.
/// The other roots stay gray, so values they hold are still white.
fn trace_last_root(runtime: &mut Runtime, roots: &[Value]) {
    runtime.set_gc_step_multiplier(0.01);
    let mut builder = GcRootBuilder::new();
    builder.values(roots);
    assert_eq!(
        runtime.gc_step_with_roots(builder.finish(), 0),
        GcStepResult::InProgress
    );
    assert!(
        runtime
            .incremental_gc
            .as_ref()
            .is_some_and(|cycle| cycle.phase == crate::runtime::gc::IncrementalGcPhase::Mark)
    );
}

fn assert_graph_alive(runtime: &Runtime, parent: TableHandle, child: TableHandle, answer: f64) {
    assert_eq!(
        runtime.table_get(parent, &Value::from("child")).unwrap(),
        Value::Table(child)
    );
    assert_eq!(
        runtime.table_get(child, &Value::from("answer")).unwrap(),
        Value::Number(answer)
    );
}

/// Moves `parent` out of an untraced holder, the way Lua code would after
/// loading it into a register. Deleting a field has no barrier.
fn take_from_holder(runtime: &mut Runtime, holder: TableHandle) {
    runtime
        .table_set(holder, &Value::from("parent"), Value::Nil)
        .unwrap();
}

fn holder_of(runtime: &mut Runtime, value: Value) -> TableHandle {
    let holder = runtime.create_table().unwrap();
    runtime
        .table_set(holder, &Value::from("parent"), value)
        .unwrap();
    holder
}

#[test]
fn atomic_step_rescans_roots_that_appeared_during_marking() {
    let mut runtime = Runtime::new();
    let (parent, child) = parent_child_graph(&mut runtime, 42.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    let other = runtime.create_table().unwrap();
    trace_last_root(&mut runtime, &[Value::Table(holder), Value::Table(other)]);
    // The graph now lives only in a VM register, which has no barrier. The
    // atomic step must find it in the fresh root snapshot.
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Table(parent));
    assert_graph_alive(&runtime, parent, child, 42.0);

    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(runtime.table_get(parent, &Value::from("child")).is_err());
    assert!(runtime.table_get(child, &Value::from("answer")).is_err());
}

#[test]
fn insertion_barrier_keeps_a_graph_stored_into_a_traced_table() {
    let mut runtime = Runtime::new();
    let owner = runtime.create_table().unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 17.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(&mut runtime, &[Value::Table(holder), Value::Table(owner)]);
    // The owner was already traced. Only the insertion barrier can
    // discover this new edge.
    runtime
        .table_set(owner, &Value::from("new"), Value::Table(parent))
        .unwrap();
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Table(owner));
    assert_eq!(
        runtime.table_get(owner, &Value::from("new")).unwrap(),
        Value::Table(parent)
    );
    assert_graph_alive(&runtime, parent, child, 17.0);
}

#[test]
fn insertion_barrier_keeps_a_graph_added_through_thread_roots() {
    let mut runtime = Runtime::new();
    let thread = runtime.create_thread().unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 41.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(&mut runtime, &[Value::Table(holder), Value::Thread(thread)]);
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(parent));
    runtime.set_thread_roots(thread, roots.finish()).unwrap();
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Thread(thread));
    assert_graph_alive(&runtime, parent, child, 41.0);
}

#[test]
fn insertion_barrier_keeps_a_graph_added_through_closure_environment() {
    let mut runtime = Runtime::new();
    let closure = runtime
        .create_closure(
            Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()])),
            Vec::new(),
        )
        .unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 42.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(
        &mut runtime,
        &[Value::Table(holder), Value::Closure(closure)],
    );
    runtime.set_closure_environment(closure, parent).unwrap();
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Closure(closure));
    assert_eq!(runtime.closure_environment(closure).unwrap(), parent);
    assert_graph_alive(&runtime, parent, child, 42.0);
}

#[test]
fn insertion_barrier_keeps_a_graph_added_through_closure_cell() {
    let mut runtime = Runtime::new();
    let old_cell = runtime.create_cell(Value::Nil).unwrap();
    let chunk = Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()]));
    let closure = runtime
        .create_closure(Arc::clone(&chunk), vec![old_cell])
        .unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 43.0);
    let replacement = runtime.create_cell(Value::Table(parent)).unwrap();
    // The replacement cell is reachable only through an untraced closure.
    let other = runtime.create_closure(chunk, vec![replacement]).unwrap();
    let holder = holder_of(&mut runtime, Value::Closure(other));
    trace_last_root(
        &mut runtime,
        &[Value::Table(holder), Value::Closure(closure)],
    );
    assert!(runtime.set_closure_cell(closure, 0, replacement).unwrap());
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Closure(closure));
    assert_eq!(runtime.cell_get(replacement).unwrap(), Value::Table(parent));
    assert_graph_alive(&runtime, parent, child, 43.0);
}

#[test]
fn insertion_barrier_keeps_a_graph_added_through_cell_set() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::Nil).unwrap();
    let closure = runtime
        .create_closure(
            Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()])),
            vec![cell],
        )
        .unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 44.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(
        &mut runtime,
        &[Value::Table(holder), Value::Closure(closure)],
    );
    runtime.cell_set(cell, Value::Table(parent)).unwrap();
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Closure(closure));
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Table(parent));
    assert_graph_alive(&runtime, parent, child, 44.0);
}

#[test]
fn objects_allocated_during_a_cycle_survive_its_sweep() {
    let mut runtime = Runtime::new();
    let owner = runtime.create_table().unwrap();
    let root = Value::Table(owner);
    advance_to_incremental_sweep(&mut runtime, &root);
    let (parent, child) = parent_child_graph(&mut runtime, 45.0);
    runtime
        .table_set(owner, &Value::from("late"), Value::Table(parent))
        .unwrap();
    finish_incremental_cycle(&mut runtime, &root);
    assert_graph_alive(&runtime, parent, child, 45.0);
}

#[test]
fn incremental_cycle_clears_weak_values_at_the_atomic_step() {
    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("v"))
        .unwrap();
    let weak = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();
    let child = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Number(1.0), Value::Table(child))
        .unwrap();

    let root = Value::Table(weak);
    advance_to_incremental_sweep(&mut runtime, &root);
    // Clearing happened atomically, before the mutator could load the dead
    // value from the weak table.
    assert_eq!(
        runtime.table_get(weak, &Value::Number(1.0)).unwrap(),
        Value::Nil
    );
    finish_incremental_cycle(&mut runtime, &root);
    assert!(runtime.table_get(child, &Value::from("anything")).is_err());
}

#[test]
fn full_collection_preserves_staged_finalizers_and_their_shared_graph() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }
    for publication_started in [false, true] {
        let mut runtime = Runtime::new();
        let metatable = runtime.create_table().unwrap();
        let handler = runtime.register_native(finalizer).unwrap();
        runtime
            .table_set(
                metatable,
                &Value::from("__gc"),
                Value::NativeFunction(handler),
            )
            .unwrap();
        let mut objects = Vec::new();
        for index in 1..=11 {
            let userdata = runtime.create_userdata().unwrap();
            runtime
                .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
                .unwrap();
            runtime
                .table_set(
                    metatable,
                    &Value::Number(f64::from(index)),
                    Value::Userdata(userdata),
                )
                .unwrap();
            objects.push(userdata);
        }
        runtime.set_gc_step_multiplier(0.01);
        let steps = if publication_started { 40 } else { 2 };
        for _ in 0..steps {
            if runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 0)
                == GcStepResult::Complete
            {
                break;
            }
        }
        // Whether or not the interrupted cycle already queued finalizers,
        // the full collection must queue each userdata exactly once.
        assert_eq!(
            runtime
                .collect_full(incremental_roots(&Value::Nil))
                .userdatas,
            0
        );
        for expected in objects.into_iter().rev() {
            assert_eq!(finalized_userdata(&mut runtime), expected);
        }
        assert!(runtime.take_pending_finalizer().unwrap().is_none());
        assert_eq!(
            runtime
                .collect_full(incremental_roots(&Value::Nil))
                .userdatas,
            11
        );
    }
}

#[test]
fn full_collection_interrupting_an_incremental_cycle_does_not_duplicate_finalizers() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let first = runtime.create_userdata().unwrap();
    let second = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(first), Some(metatable))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(second), Some(metatable))
        .unwrap();

    runtime.set_gc_step_multiplier(0.01);
    assert_eq!(
        runtime.gc_step_with_roots(incremental_roots(&Value::Nil), 0),
        GcStepResult::InProgress
    );
    runtime.collect_full(GcRootBuilder::new().finish());

    let mut queued = vec![
        finalized_userdata(&mut runtime),
        finalized_userdata(&mut runtime),
    ];
    queued.sort_unstable();
    assert_eq!(queued, vec![first, second]);
    assert!(runtime.take_pending_finalizer().unwrap().is_none());

    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
}

#[test]
fn one_unit_steps_are_not_starved_by_scalar_writes_or_refreshed_marked_roots() {
    let mut runtime = Runtime::new();
    let owner = runtime.create_table().unwrap();
    let root = Value::Table(owner);
    advance_to_incremental_sweep(&mut runtime, &root);

    for index in 0..10_000 {
        // Both key and value are scalars. Their write barrier must not
        // restart a finished mark phase, and repeatedly supplied roots
        // already marked in this epoch must stay out of the work queue.
        runtime
            .table_set(
                owner,
                &Value::from("counter"),
                Value::Number(f64::from(index)),
            )
            .unwrap();
        if runtime.gc_step_with_roots(incremental_roots(&root), 0) == GcStepResult::Complete {
            return;
        }
    }
    panic!("scalar writes starved one-unit incremental collection");
}

#[test]
fn finalizer_registered_after_the_incremental_scan_survives_until_next_cycle() {
    fn finalizer(runtime: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        runtime.write_output(&[])?;
        Ok(Vec::new())
    }

    let mut runtime = Runtime::new();
    let metatable = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            metatable,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();

    advance_to_incremental_sweep(&mut runtime, &Value::Table(metatable));
    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(metatable))
        .unwrap();
    finish_incremental_cycle(&mut runtime, &Value::Table(metatable));

    // The registration happened after this epoch's finalizer cursor. It
    // must remain registered rather than being reclaimed unqueued.
    assert_eq!(
        runtime.raw_metatable(&Value::Userdata(userdata)).unwrap(),
        Some(metatable)
    );
    assert!(runtime.take_pending_finalizer().unwrap().is_none());

    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(finalized_userdata(&mut runtime), userdata);
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
}

/// One object of the reference model: a table, a weak-key table, or a
/// closure whose single captured cell holds its only field. A weak-key
/// table's integer fields are strong; its entries keyed by other nodes are
/// ephemerons when the runtime enables them.
#[derive(Clone, Copy)]
enum ModelNode {
    Table(TableHandle),
    WeakKeys(TableHandle),
    Closure(ClosureHandle, CellHandle),
}

impl ModelNode {
    fn value(self) -> Value {
        match self {
            Self::Table(handle) | Self::WeakKeys(handle) => Value::Table(handle),
            Self::Closure(handle, _) => Value::Closure(handle),
        }
    }

    const fn fields(self) -> usize {
        match self {
            Self::Table(_) | Self::WeakKeys(_) => 4,
            Self::Closure(..) => 1,
        }
    }
}

/// A reference model of an object graph. Tables have fields `1..=4`,
/// closures one field in their captured cell, and `roots` plays the VM
/// registers. Several arenas are involved, because a major sweep visits
/// arenas one after another.
struct GraphModel {
    runtime: Runtime,
    chunk: Arc<Chunk>,
    nodes: Vec<Option<ModelNode>>,
    edges: Vec<[Option<usize>; 4]>,
    /// Entries of weak-key tables keyed by other nodes: key to value.
    weak_entries: Vec<BTreeMap<usize, usize>>,
    roots: Vec<usize>,
    state: u64,
    /// Whether weak-key entries are ephemerons (Lua 5.2+) or hold their
    /// values strongly (Lua 5.1).
    ephemerons: bool,
}

impl GraphModel {
    fn new(seed: u64, ephemerons: bool) -> Self {
        let mut runtime = Runtime::new();
        runtime.set_ephemerons_enabled(ephemerons);
        Self {
            runtime,
            chunk: Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()])),
            nodes: Vec::new(),
            edges: Vec::new(),
            weak_entries: Vec::new(),
            roots: Vec::new(),
            state: seed | 1,
            ephemerons,
        }
    }

    fn next(&mut self, bound: usize) -> usize {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state % bound as u64) as usize
    }

    fn live_nodes(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|index| self.nodes[*index].is_some())
            .collect()
    }

    fn node(&self, index: usize) -> ModelNode {
        self.nodes[index].expect("model node is reachable")
    }

    fn gc_roots(&self) -> GcRoots {
        let mut builder = GcRootBuilder::new();
        for index in &self.roots {
            builder.value(&self.node(*index).value());
        }
        builder.finish()
    }

    /// Forgets nodes the model cannot reach; the collector may free them.
    /// Weak-key entries whose key is unreachable are forgotten too.
    fn prune_unreachable(&mut self) {
        let mut reached = vec![false; self.nodes.len()];
        let mut stack = self.roots.clone();
        loop {
            while let Some(index) = stack.pop() {
                if std::mem::replace(&mut reached[index], true) {
                    continue;
                }
                stack.extend(self.edges[index].iter().flatten());
                if !self.ephemerons {
                    stack.extend(self.weak_entries[index].values());
                }
            }
            if self.ephemerons {
                for (table, entries) in self.weak_entries.iter().enumerate() {
                    if reached[table] {
                        stack.extend(
                            entries
                                .iter()
                                .filter(|(key, value)| reached[**key] && !reached[**value])
                                .map(|(_, value)| *value),
                        );
                    }
                }
            }
            if stack.is_empty() {
                break;
            }
        }
        for (index, reached) in reached.iter().enumerate() {
            if !reached {
                self.nodes[index] = None;
                self.weak_entries[index].clear();
            }
        }
        for entries in &mut self.weak_entries {
            entries.retain(|key, _| reached[*key]);
        }
    }

    fn is_alive(&self, node: ModelNode) -> bool {
        match node {
            ModelNode::Table(handle) | ModelNode::WeakKeys(handle) => {
                self.runtime.table_get(handle, &Value::Nil).is_ok()
            }
            ModelNode::Closure(handle, _) => self.runtime.closure_environment(handle).is_ok(),
        }
    }

    fn live_weak_tables(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|index| matches!(self.nodes[*index], Some(ModelNode::WeakKeys(_))))
            .collect()
    }

    fn weak_table(&self, index: usize) -> TableHandle {
        match self.node(index) {
            ModelNode::WeakKeys(handle) => handle,
            _ => panic!("model node {index} is not a weak-key table"),
        }
    }

    fn store_entry(&mut self, table: usize, key: usize, value: Option<usize>) {
        let handle = self.weak_table(table);
        let stored = value.map_or(Value::Nil, |value| self.node(value).value());
        self.runtime
            .table_set(handle, &self.node(key).value(), stored)
            .unwrap();
        match value {
            Some(value) => self.weak_entries[table].insert(key, value),
            None => self.weak_entries[table].remove(&key),
        };
    }

    fn stored(&self, node: ModelNode, field: usize) -> Value {
        match node {
            ModelNode::Table(handle) | ModelNode::WeakKeys(handle) => self
                .runtime
                .table_get(handle, &Value::Number((field + 1) as f64))
                .expect("a reachable table survives"),
            ModelNode::Closure(handle, cell) => {
                assert!(
                    self.runtime.closure_environment(handle).is_ok(),
                    "a reachable closure survives"
                );
                self.runtime
                    .cell_get(cell)
                    .expect("a reachable cell survives")
            }
        }
    }

    fn store(&mut self, from: usize, field: usize, to: Option<usize>) {
        let value = to.map_or(Value::Nil, |to| self.node(to).value());
        match self.node(from) {
            ModelNode::Table(handle) | ModelNode::WeakKeys(handle) => self
                .runtime
                .table_set(handle, &Value::Number((field + 1) as f64), value)
                .unwrap(),
            ModelNode::Closure(_, cell) => self.runtime.cell_set(cell, value).unwrap(),
        }
        self.edges[from][field] = to;
    }

    fn check(&self) {
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(node) = *node else { continue };
            for field in 0..node.fields() {
                let expected =
                    self.edges[index][field].map_or(Value::Nil, |target| self.node(target).value());
                assert_eq!(self.stored(node, field), expected);
            }
        }
        for table in self.live_weak_tables() {
            let handle = self.weak_table(table);
            for (key, value) in &self.weak_entries[table] {
                assert_eq!(
                    self.runtime
                        .table_get(handle, &self.node(*key).value())
                        .expect("a reachable weak-key table survives"),
                    self.node(*value).value(),
                    "weak-key entry {key} -> {value} of node {table}"
                );
            }
        }
        if let Some((parent, child)) = self.runtime.gc_generational_violation() {
            panic!("untraced old {parent:?} points at young {child:?}");
        }
    }

    fn step(&mut self) {
        let live = self.live_nodes();
        match self.next(25) {
            0..=3 => {
                let handle = self.runtime.create_table().unwrap();
                let node = if self.next(4) == 0 {
                    let metatable = self.runtime.create_table().unwrap();
                    self.runtime
                        .table_set(metatable, &Value::from("__mode"), Value::from("k"))
                        .unwrap();
                    self.runtime
                        .raw_set_metatable(&Value::Table(handle), Some(metatable))
                        .unwrap();
                    ModelNode::WeakKeys(handle)
                } else {
                    ModelNode::Table(handle)
                };
                self.nodes.push(Some(node));
                self.edges.push([None; 4]);
                self.weak_entries.push(BTreeMap::new());
                // New objects start in a register.
                self.roots.push(self.nodes.len() - 1);
            }
            4..=5 => {
                // A closure whose cell is born holding an existing object:
                // a reference no barrier sees.
                let target =
                    (!live.is_empty() && self.next(3) != 0).then(|| live[self.next(live.len())]);
                let initial = target.map_or(Value::Nil, |target| self.node(target).value());
                let cell = self.runtime.create_cell(initial).unwrap();
                let closure = self
                    .runtime
                    .create_closure(Arc::clone(&self.chunk), vec![cell])
                    .unwrap();
                self.nodes.push(Some(ModelNode::Closure(closure, cell)));
                self.edges.push([target, None, None, None]);
                self.weak_entries.push(BTreeMap::new());
                self.roots.push(self.nodes.len() - 1);
            }
            6..=10 if !live.is_empty() => {
                let from = live[self.next(live.len())];
                let to = live[self.next(live.len())];
                let field = self.next(self.node(from).fields());
                self.store(from, field, Some(to));
            }
            11..=12 if !live.is_empty() => {
                let from = live[self.next(live.len())];
                let field = self.next(self.node(from).fields());
                self.store(from, field, None);
            }
            13..=14 if !self.roots.is_empty() => {
                let slot = self.next(self.roots.len());
                self.roots.swap_remove(slot);
            }
            15 if !live.is_empty() => {
                // A register loads a value from the heap.
                let index = live[self.next(live.len())];
                self.roots.push(index);
            }
            16..=18 => {
                self.runtime.gc_automatic_step(Some(self.gc_roots()), 1);
                self.prune_unreachable();
            }
            19..=20 => {
                let _ = self.runtime.gc_step_with_roots(self.gc_roots(), 0);
                self.prune_unreachable();
            }
            21 if self.next(8) == 0 => {
                self.runtime.collect_full(self.gc_roots());
                let before = self.nodes.clone();
                self.prune_unreachable();
                // A full collection traces exactly what the roots reach, so
                // everything the model forgot is freed, including keys and
                // values held only through ephemerons. Lua 5.1 keeps the
                // value of an entry with a dead key alive until the
                // collection that clears the entry, which the model does
                // not track.
                for (index, node) in before.iter().enumerate() {
                    if let (true, Some(node), None) = (self.ephemerons, node, self.nodes[index]) {
                        assert!(!self.is_alive(*node), "unreachable node {index} survived");
                    }
                }
            }
            22..=24 if !live.is_empty() => {
                // A weak-key entry keyed by an object, as in `t[k] = v`.
                let tables = self.live_weak_tables();
                if !tables.is_empty() {
                    let table = tables[self.next(tables.len())];
                    let key = live[self.next(live.len())];
                    let value = (self.next(4) != 0).then(|| live[self.next(live.len())]);
                    self.store_entry(table, key, value);
                }
            }
            _ => {}
        }
        if self.roots.len() > 24 {
            self.roots.drain(..8);
        }
        self.check();
    }
}

#[test]
fn generational_and_incremental_collections_preserve_a_random_graph() {
    let seeds = std::env::var("OXIMOND_GC_GRAPH_SEEDS")
        .ok()
        .and_then(|seeds| seeds.parse().ok())
        .unwrap_or(60_u64);
    for seed in 1..=seeds {
        // One seed in three keeps Lua 5.1's strong weak-key values.
        let mut model = GraphModel::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15), seed % 3 != 0);
        if seed % 2 == 0 {
            // Tiny major steps interleave mutations with marking and sweep.
            model.runtime.set_gc_step_multiplier(1.0);
        }
        for _ in 0..3_000 {
            model.step();
        }
    }
}

#[test]
fn minor_collection_frees_young_garbage_and_keeps_old_to_young_edges() {
    let mut runtime = Runtime::new();
    let old = runtime.create_table().unwrap();
    runtime.collect_full(incremental_roots(&Value::Table(old)));
    assert!(!runtime.gc_is_young(crate::runtime::gc::GcReference::Table(old)));

    let young = runtime.create_table().unwrap();
    let garbage = runtime.create_table().unwrap();
    runtime
        .table_set(old, &Value::from("young"), Value::Table(young))
        .unwrap();
    for _ in 0..3 {
        assert_eq!(
            runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(old))), 1),
            GcStepResult::Complete
        );
        assert_eq!(
            runtime.table_get(old, &Value::from("young")).unwrap(),
            Value::Table(young)
        );
    }
    assert!(runtime.table_get(garbage, &Value::Nil).is_err());
    // Two minor collections promote the young table.
    assert!(!runtime.gc_is_young(crate::runtime::gc::GcReference::Table(young)));
}

#[test]
fn minor_collection_clears_weak_entries_and_queues_young_finalizers() {
    fn finalizer(_: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        Ok(Vec::new())
    }
    let mut runtime = Runtime::new();
    // Only a program that selected generational mode finalizes in minor
    // collections.
    runtime.set_gc_mode(GcMode::Generational);
    let weak_meta = runtime.create_table().unwrap();
    runtime
        .table_set(weak_meta, &Value::from("__mode"), Value::from("k"))
        .unwrap();
    let weak = runtime.create_table().unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(weak_meta))
        .unwrap();
    let gc_meta = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            gc_meta,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let holder = runtime.create_table().unwrap();
    runtime
        .table_set(holder, &Value::from("weak"), Value::Table(weak))
        .unwrap();
    runtime
        .table_set(holder, &Value::from("gc"), Value::Table(gc_meta))
        .unwrap();
    runtime.collect_full(incremental_roots(&Value::Table(holder)));

    let key = runtime.create_table().unwrap();
    let kept = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Table(key), Value::Boolean(true))
        .unwrap();
    runtime
        .table_set(weak, &Value::Table(kept), Value::Boolean(true))
        .unwrap();
    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(gc_meta))
        .unwrap();

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(holder));
    roots.value(&Value::Table(kept));
    assert_eq!(
        runtime.gc_automatic_step(Some(roots.finish()), 1),
        GcStepResult::Complete
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Table(kept)).unwrap(),
        Value::Boolean(true)
    );
    assert!(runtime.table_next(weak, &Value::Nil).is_ok());
    assert_eq!(runtime.table_length(weak).unwrap(), 0.0);
    assert!(runtime.table_get(key, &Value::Nil).is_err());
    let mut entries = 0;
    let mut cursor = Value::Nil;
    loop {
        let (next, _) = runtime.table_next(weak, &cursor).unwrap();
        if next == Value::Nil {
            break;
        }
        entries += 1;
        cursor = next;
    }
    assert_eq!(entries, 1);
    assert_eq!(finalized_userdata(&mut runtime), userdata);
}

#[test]
fn incremental_mode_defers_finalizers_from_minor_collections_to_a_major_cycle() {
    #[allow(clippy::unnecessary_wraps)] // The native-function signature.
    fn finalizer(_: &mut Runtime, _: &[Value]) -> Result<Vec<Value>, RuntimeError> {
        Ok(Vec::new())
    }
    // Lua's `gc.lua` waits for a finalizer, then expects older garbage to
    // be gone: here an old ephemeron entry whose key only its value holds.
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(true);
    assert_eq!(runtime.gc_mode(), GcMode::Incremental);
    let weak = weak_key_table(&mut runtime);
    let gc_meta = runtime.create_table().unwrap();
    let handler = runtime.register_native(finalizer).unwrap();
    runtime
        .table_set(
            gc_meta,
            &Value::from("__gc"),
            Value::NativeFunction(handler),
        )
        .unwrap();
    let holder = runtime.create_table().unwrap();
    runtime
        .table_set(holder, &Value::from("weak"), Value::Table(weak))
        .unwrap();
    runtime
        .table_set(holder, &Value::from("gc"), Value::Table(gc_meta))
        .unwrap();
    let old_key = runtime.create_table().unwrap();
    store_key_cycle(&mut runtime, weak, old_key);
    runtime
        .table_set(holder, &Value::from("key"), Value::Table(old_key))
        .unwrap();
    runtime.collect_full(incremental_roots(&Value::Table(holder)));
    runtime
        .table_set(holder, &Value::from("key"), Value::Nil)
        .unwrap();

    let userdata = runtime.create_userdata().unwrap();
    runtime
        .raw_set_metatable(&Value::Userdata(userdata), Some(gc_meta))
        .unwrap();
    // The minor collection keeps the dead userdata for the major cycle.
    assert_eq!(
        runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1),
        GcStepResult::Complete
    );
    assert!(runtime.take_pending_finalizer().unwrap().is_none());
    assert!(!is_freed(&runtime, old_key));
    let mut steps = 0;
    while runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(holder))), 1)
        != GcStepResult::Complete
    {
        steps += 1;
        assert!(steps < 10_000, "the requested major cycle never finished");
    }
    assert_eq!(finalized_userdata(&mut runtime), userdata);
    assert!(is_freed(&runtime, old_key));
    assert_eq!(runtime.table_next(weak, &Value::Nil).unwrap().0, Value::Nil);
}

fn userdata_object(object: &Value) -> UserdataHandle {
    match object {
        Value::Userdata(handle) => *handle,
        object => panic!("expected a userdata finalizer, found {object:?}"),
    }
}

fn finalized_userdata(runtime: &mut Runtime) -> UserdataHandle {
    userdata_object(&runtime.take_pending_finalizer().unwrap().unwrap().object)
}

#[test]
fn objects_born_during_marking_keep_what_they_hold_from_birth() {
    let mut runtime = Runtime::new();
    runtime.collect_full(GcRootBuilder::new().finish());
    runtime.stop_gc();
    let held = runtime.create_table().unwrap();
    let captured = runtime.create_table().unwrap();
    let captured_cell = runtime.create_cell(Value::Table(captured)).unwrap();
    runtime.start_gc_cycle(Some(&GcRootBuilder::new().finish()));

    // Born marked during the cycle, so never traced: only a birth barrier
    // marks what they reference.
    let cell = runtime.create_cell(Value::Table(held)).unwrap();
    let chunk = Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()]));
    let closure = runtime.create_closure(chunk, vec![captured_cell]).unwrap();
    let mut roots = GcRootBuilder::new();
    roots.cell(cell);
    roots.closure(closure);
    let roots = roots.finish();
    while runtime.gc_step_with_roots(roots.clone(), 0) == GcStepResult::InProgress {}

    assert!(runtime.table_get(held, &Value::Nil).is_ok());
    assert!(runtime.table_get(captured, &Value::Nil).is_ok());
    assert_eq!(
        runtime.cell_get(captured_cell).unwrap(),
        Value::Table(captured)
    );
}

#[test]
fn host_thread_roots_are_independent_and_die_with_the_thread() {
    let mut runtime = Runtime::new();
    let thread = runtime.create_thread().unwrap();
    let vm_table = runtime.create_table().unwrap();
    let host_table = runtime.create_table().unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(vm_table));
    runtime.set_thread_roots(thread, roots.finish()).unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(host_table));
    runtime
        .set_thread_host_roots(thread, roots.finish())
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Thread(thread));
    runtime.collect_full(roots.finish());
    assert!(runtime.table_length(vm_table).is_ok());
    assert!(runtime.table_length(host_table).is_ok());
    runtime.collect_full(GcRootBuilder::new().finish());
    assert!(!runtime.thread_exists(thread));
    assert!(runtime.table_length(vm_table).is_err());
    assert!(runtime.table_length(host_table).is_err());
}

#[test]
fn insertion_barrier_keeps_a_graph_added_through_host_thread_roots() {
    let mut runtime = Runtime::new();
    let thread = runtime.create_thread().unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 43.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(&mut runtime, &[Value::Table(holder), Value::Thread(thread)]);
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(parent));
    runtime
        .set_thread_host_roots(thread, roots.finish())
        .unwrap();
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Thread(thread));
    assert_graph_alive(&runtime, parent, child, 43.0);
}

#[test]
fn table_and_elision_totals_survive_a_collection() {
    let mut runtime = Runtime::new();
    assert_eq!(runtime.tables_created(), 0);
    runtime.create_table().unwrap();
    runtime.create_table().unwrap();
    runtime.charge_elided_table(3);
    runtime.charge_elided_table(0);
    let (tables, bytes) = runtime.elided_tables_total();
    assert_eq!(tables, 2);
    assert_eq!(bytes as usize, runtime.elided_table_bytes());
    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(runtime.elided_table_bytes(), 0);
    assert_eq!(runtime.elided_tables_total(), (tables, bytes));
    assert_eq!(runtime.tables_created(), 2);
}

#[test]
fn refunding_an_elided_table_undoes_its_charge_once() {
    let mut runtime = Runtime::new();
    runtime.charge_elided_table(5);
    runtime.charge_elided_table(2);
    let both = runtime.elided_table_bytes();
    assert_eq!(
        both,
        Runtime::elided_footprint(5) + Runtime::elided_footprint(2)
    );
    runtime.refund_elided_table(5);
    assert_eq!(runtime.elided_table_bytes(), Runtime::elided_footprint(2));
    assert_eq!(runtime.elided_tables_total().0, 1);
    runtime.refund_elided_table(2);
    assert_eq!(runtime.elided_table_bytes(), 0);
    assert_eq!(runtime.elided_tables_total(), (0, 0));
    // A charge a finished cycle already released leaves the pending bytes
    // alone and shrinks only the total.
    runtime.charge_elided_table(3);
    runtime.collect_full(GcRootBuilder::new().finish());
    assert_eq!(runtime.elided_table_bytes(), 0);
    runtime.refund_elided_table(3);
    assert_eq!(runtime.elided_tables_total(), (0, 0));
    assert_eq!(runtime.elided_table_bytes(), 0);
}

/// The random graph model again, with every stop-the-world trace on four
/// threads and arrays split into ranges of two.
#[test]
fn parallel_marking_preserves_a_random_graph() {
    for seed in 1..=24_u64 {
        let mut model = GraphModel::new(seed.wrapping_mul(0x2545_f491_4f6c_dd1d), seed % 3 != 0);
        model.runtime.force_parallel_marking(4);
        if seed % 2 == 0 {
            model.runtime.set_gc_step_multiplier(1.0);
        }
        for _ in 0..3_000 {
            model.step();
        }
    }
}

/// Builds a heap past the default parallel thresholds: a long array of
/// small trees, half of them dropped, a ring of closures, and an ephemeron
/// table whose values are reachable only through live keys.
fn large_graph(threads: usize) -> (Runtime, Value) {
    let mut runtime = Runtime::new();
    runtime.set_ephemerons_enabled(true);
    runtime.set_gc_mark_threads(threads);
    let root = runtime.create_table().unwrap();
    let chunk = Arc::new(Chunk::new(0, Vec::new(), vec![Instruction::halt()]));
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("k"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();
    runtime
        .table_set(root, &Value::from("weak"), Value::Table(weak))
        .unwrap();
    let mut previous = Value::Nil;
    for index in 1..=40_000_i64 {
        let node = runtime.create_table().unwrap();
        let leaf = runtime.create_table().unwrap();
        runtime
            .table_set(node, &Value::from("leaf"), Value::Table(leaf))
            .unwrap();
        let cell = runtime.create_cell(previous.clone()).unwrap();
        let closure = runtime
            .create_closure(Arc::clone(&chunk), vec![cell])
            .unwrap();
        runtime
            .table_set(node, &Value::from("f"), Value::Closure(closure))
            .unwrap();
        previous = Value::Closure(closure);
        let only_by_key = runtime.create_table().unwrap();
        runtime
            .table_set(weak, &Value::Table(node), Value::Table(only_by_key))
            .unwrap();
        if index % 2 == 0 {
            runtime
                .table_set(root, &Value::Integer(index / 2), Value::Table(node))
                .unwrap();
        }
    }
    (runtime, Value::Table(root))
}

/// Tracing on several threads frees exactly what tracing on one does, on
/// a heap large enough for the default thresholds.
#[test]
fn parallel_marking_matches_serial_marking_on_a_large_heap() {
    let (mut serial, root) = large_graph(1);
    let serial_stats = serial.collect_full(incremental_roots(&root));
    let (mut parallel, root) = large_graph(4);
    let parallel_stats = parallel.collect_full(incremental_roots(&root));
    assert_eq!(serial_stats, parallel_stats);
    assert!(serial_stats.tables > 0);
    assert_eq!(serial.gc_count_bytes(), parallel.gc_count_bytes());
    // A minor collection over a young heap of the same shape.
    let (mut serial, root) = large_graph(1);
    let serial_stats = serial.gc_minor_collection(&incremental_roots(&root));
    let (mut parallel, root) = large_graph(4);
    let parallel_stats = parallel.gc_minor_collection(&incremental_roots(&root));
    assert_eq!(serial_stats, parallel_stats);
    assert_eq!(serial.gc_count_bytes(), parallel.gc_count_bytes());
}

#[test]
fn array_shift_insert_keeps_a_graph_stored_into_a_traced_table() {
    let mut runtime = Runtime::new();
    let owner = runtime.create_table().unwrap();
    runtime
        .table_set(owner, &Value::Number(1.0), Value::Number(1.0))
        .unwrap();
    let (parent, child) = parent_child_graph(&mut runtime, 23.0);
    let holder = holder_of(&mut runtime, Value::Table(parent));
    trace_last_root(&mut runtime, &[Value::Table(holder), Value::Table(owner)]);
    // The owner was already traced; only the shift's barrier finds the edge.
    assert!(runtime.table_array_shift_insert(owner, Some(1), &Value::Table(parent)));
    take_from_holder(&mut runtime, holder);
    finish_incremental_cycle(&mut runtime, &Value::Table(owner));
    assert_eq!(
        runtime.table_get(owner, &Value::Number(1.0)).unwrap(),
        Value::Table(parent)
    );
    assert_graph_alive(&runtime, parent, child, 23.0);
}

/// An old table that only receives appends keeps every young child through
/// repeated minor collections, though each minor traces only the slots
/// stored since the previous one.
#[test]
fn minor_collections_trace_appended_slots_of_an_old_table() {
    let mut runtime = Runtime::new();
    let kept = runtime.create_table().unwrap();
    runtime.collect_full(incremental_roots(&Value::Table(kept)));
    let mut children = Vec::new();
    for round in 1..=40_i64 {
        let child = runtime.create_table().unwrap();
        children.push(child);
        assert!(runtime.table_set_dense_reference(
            kept,
            &Value::Integer(round),
            &Value::Table(child)
        ));
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        for (index, child) in children.iter().enumerate() {
            assert_eq!(
                runtime
                    .table_get(kept, &Value::Integer(index as i64 + 1))
                    .unwrap(),
                Value::Table(*child)
            );
            assert!(runtime.table_length(*child).is_ok());
        }
    }
    for _ in 0..3 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
    }
    for child in &children {
        assert!(runtime.table_length(*child).is_ok());
    }
}

/// A young store into an old table's array followed by a shifting insert
/// moves the stored child to a slot the barrier did not index. The shift must
/// widen the traced range to the whole array, or two minors free the child.
#[test]
fn insert_that_shifts_a_remembered_slot_keeps_the_shifted_child() {
    let mut runtime = Runtime::new();
    let kept = runtime.create_table().unwrap();
    let old_child = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(1), &Value::Table(old_child)));
    runtime.collect_full(incremental_roots(&Value::Table(kept)));
    let young = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(2), &Value::Table(young)));
    assert!(runtime.table_array_shift_insert(kept, Some(1), &Value::Integer(7)));
    for _ in 0..2 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        assert_eq!(
            runtime.table_get(kept, &Value::Integer(3)).unwrap(),
            Value::Table(young)
        );
        assert!(runtime.table_length(young).is_ok());
    }
}

/// The removing counterpart: a young child moves down one slot, which a
/// ranged remembered table must not miss.
#[test]
fn remove_that_shifts_a_remembered_slot_keeps_the_shifted_child() {
    let mut runtime = Runtime::new();
    let kept = runtime.create_table().unwrap();
    let old_child = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(1), &Value::Table(old_child)));
    runtime.collect_full(incremental_roots(&Value::Table(kept)));
    let young = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(2), &Value::Table(young)));
    assert!(runtime.table_array_shift_remove(kept, Some(1)).is_some());
    for _ in 0..2 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        assert_eq!(
            runtime.table_get(kept, &Value::Integer(1)).unwrap(),
            Value::Table(young)
        );
        assert!(runtime.table_length(young).is_ok());
    }
}

/// The automatic collector, which mixes minor collections with incremental
/// major cycles, keeps every child a program stores into an old table.
#[test]
fn automatic_steps_keep_children_stored_into_an_old_table() {
    let mut runtime = Runtime::new();
    let kept = runtime.create_table().unwrap();
    let mut children = Vec::new();
    for round in 0..20_000_i64 {
        let child = runtime.create_table().unwrap();
        if round % 2 == 0 {
            children.push(child);
            let index = children.len() as i64;
            // A major cycle that traces `kept` in chunks sends the store to the
            // general path, which is what the interpreter does too.
            if !runtime.table_set_dense_reference(
                kept,
                &Value::Integer(index),
                &Value::Table(child),
            ) {
                runtime
                    .table_set(kept, &Value::Integer(index), Value::Table(child))
                    .unwrap();
            }
        }
        runtime.gc_automatic_step(Some(incremental_roots(&Value::Table(kept))), 1);
        if let Some((slot, index)) = runtime.gc_unranged_young_child() {
            panic!("old table {slot} slot {index} young and untraced at round {round}");
        }
        if round % 97 == 0 {
            for (index, child) in children.iter().enumerate() {
                assert!(
                    runtime.table_length(*child).is_ok(),
                    "child {index} freed at round {round}"
                );
            }
        }
    }
    for child in &children {
        assert!(runtime.table_length(*child).is_ok());
    }
}

/// An old list `kept` holding `count` old tables in slots `1..=count`.
fn old_list(runtime: &mut Runtime, count: i64) -> TableHandle {
    let kept = runtime.create_table().unwrap();
    for index in 1..=count {
        let child = runtime.create_table().unwrap();
        assert!(runtime.table_set_dense_reference(
            kept,
            &Value::Integer(index),
            &Value::Table(child)
        ));
    }
    runtime.collect_full(incremental_roots(&Value::Table(kept)));
    kept
}

/// A major collection leaves an old list with no pending range. Two indexed
/// stores after it (an overwrite and an append, with an old slot between)
/// widen one range, and the minor collections that follow trace both.
#[test]
fn stores_after_a_major_collection_union_their_slots() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 5);
    let low = runtime.create_table().unwrap();
    let high = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(2), &Value::Table(low)));
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(6), &Value::Table(high)));
    for _ in 0..4 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        assert_eq!(
            runtime.table_get(kept, &Value::Integer(2)).unwrap(),
            Value::Table(low)
        );
        assert_eq!(
            runtime.table_get(kept, &Value::Integer(6)).unwrap(),
            Value::Table(high)
        );
        assert!(runtime.table_length(low).is_ok());
        assert!(runtime.table_length(high).is_ok());
    }
}

/// A store that lands while a carried table still holds the previous
/// minor's survivors adds to the range, it does not replace it.
#[test]
fn stores_between_minors_keep_the_survivors_of_the_last_one() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 3);
    let first = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(1), &Value::Table(first)));
    runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
    let second = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(3), &Value::Table(second)));
    for _ in 0..4 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        assert!(runtime.table_length(first).is_ok());
        assert!(runtime.table_length(second).is_ok());
    }
}

/// A weak-valued old table is traced in full, not by range: its young entry
/// that nothing else holds is cleared by the next minor collection, and one
/// that a root holds stays. A weak-keyed one keeps its array values.
#[test]
fn weak_old_tables_fall_back_to_a_full_traversal() {
    let mut runtime = Runtime::new();
    let weak = runtime.create_table().unwrap();
    let metatable = runtime.create_table().unwrap();
    runtime
        .table_set(metatable, &Value::from("__mode"), Value::from("v"))
        .unwrap();
    runtime
        .raw_set_metatable(&Value::Table(weak), Some(metatable))
        .unwrap();
    let old = runtime.create_table().unwrap();
    runtime
        .table_set(weak, &Value::Integer(1), Value::Table(old))
        .unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    roots.value(&Value::Table(old));
    runtime.collect_full(roots.finish());
    let dead = runtime.create_table().unwrap();
    let held = runtime.create_table().unwrap();
    // Native code stores into the array of a table with a metatable and
    // reports each slot, as `run_array_store` does.
    for (index, child) in [(1, dead), (2, held)] {
        let table = runtime.tables.get_mut(weak.0).unwrap();
        table.array.push(Value::Table(child));
        table.border = table.array.len();
        runtime.table_native_stored_at(weak, Some(index), &Value::Table(child));
    }
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(weak));
    roots.value(&Value::Table(old));
    roots.value(&Value::Table(held));
    runtime.gc_minor_collection(&roots.finish());
    assert_eq!(
        runtime.table_get(weak, &Value::Integer(2)).unwrap(),
        Value::Nil
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Integer(3)).unwrap(),
        Value::Table(held)
    );
    assert_eq!(
        runtime.table_get(weak, &Value::Integer(1)).unwrap(),
        Value::Table(old)
    );
}

/// The debug check that every minor collection runs finds a young child
/// outside an old table's traced range, and only then.
#[cfg(debug_assertions)]
#[test]
fn the_range_check_flags_a_young_child_outside_the_traced_slots() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 4);
    let young = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(kept, &Value::Integer(2), &Value::Table(young)));
    assert_eq!(runtime.gc_unranged_young_child(), None);
    // Narrow the range to a slot that does not hold the child.
    runtime.tables.get_mut(kept.0).unwrap().gc_dirty =
        crate::runtime::table_storage::GcSlotRange::one(0);
    assert_eq!(
        runtime.gc_unranged_young_child(),
        Some((kept.0.slot, 1)),
        "the child at slot 1 is outside [0, 1)"
    );
}

/// Writes `child` into array slot `index` (zero-based) of `table` with no
/// barrier, as native code does, and reports the slot by its address.
fn native_table_store(runtime: &mut Runtime, table: TableHandle, index: usize, child: TableHandle) {
    let entry = runtime.tables.get_mut(table.0).unwrap();
    entry.array[index] = Value::Table(child);
    let slot = std::ptr::from_ref(&entry.array[index]);
    runtime.table_native_stored_table_slot(table, slot, child);
}

/// The slots the next minor collection traces for `table`.
fn pending_slots(runtime: &Runtime, table: TableHandle) -> (u32, u32) {
    let range = runtime.tables.get(table.0).unwrap().gc_dirty;
    (range.lo, range.hi)
}

/// A table that native code stores into an old list is remembered with its
/// slot. A second store into a slot already pending adds nothing, a store
/// outside widens the range, and the minor collections that follow keep
/// every stored table.
#[test]
fn a_native_table_store_into_an_old_list_is_remembered_by_its_slot() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 8);
    assert_eq!(runtime.gc_remembered.len(), 0);
    let first = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 2, first);
    assert_eq!(
        runtime.gc_remembered,
        vec![crate::runtime::gc::GcReference::Table(kept)]
    );
    assert_eq!(pending_slots(&runtime, kept), (2, 3));
    // The same slot again: the list is remembered with it.
    let second = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 2, second);
    assert_eq!(runtime.gc_remembered.len(), 1);
    assert_eq!(pending_slots(&runtime, kept), (2, 3));
    // A slot above and a slot below the range widen it.
    let above = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 6, above);
    assert_eq!(pending_slots(&runtime, kept), (2, 7));
    let below = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 0, below);
    assert_eq!(pending_slots(&runtime, kept), (0, 7));
    // A slot between them is pending already.
    let between = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 4, between);
    assert_eq!(pending_slots(&runtime, kept), (0, 7));
    assert_eq!(runtime.gc_remembered.len(), 1);
    #[cfg(debug_assertions)]
    assert_eq!(runtime.gc_unranged_young_child(), None);
    for _ in 0..4 {
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        #[cfg(debug_assertions)]
        assert_eq!(runtime.gc_unranged_young_child(), None);
    }
    for child in [second, above, below, between] {
        assert!(runtime.table_length(child).is_ok());
    }
    assert!(runtime.table_length(first).is_err(), "nothing holds it");
}

/// After a minor collection has consumed the range, a store into a slot the
/// range held before is remembered again.
#[test]
fn a_native_table_store_after_a_minor_collection_is_remembered_again() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 4);
    for round in 0..12 {
        let child = runtime.create_table().unwrap();
        native_table_store(&mut runtime, kept, 1, child);
        #[cfg(debug_assertions)]
        assert_eq!(runtime.gc_unranged_young_child(), None, "round {round}");
        runtime.gc_minor_collection(&incremental_roots(&Value::Table(kept)));
        assert!(runtime.table_length(child).is_ok(), "round {round}");
        assert_eq!(
            runtime.table_get(kept, &Value::Integer(2)).unwrap(),
            Value::Table(child)
        );
    }
}

/// The stores the short barrier leaves alone: an old table stored into an
/// old list, any table stored into a young list, and a list whose handle is
/// stale.
#[test]
fn a_native_table_store_that_needs_no_barrier_remembers_nothing() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 4);
    let Value::Table(old_child) = runtime.table_get(kept, &Value::Integer(1)).unwrap() else {
        panic!("slot 1 holds a table");
    };
    native_table_store(&mut runtime, kept, 3, old_child);
    assert_eq!(runtime.gc_remembered.len(), 0);
    assert_eq!(pending_slots(&runtime, kept), (0, 0));

    let young = runtime.create_table().unwrap();
    let child = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(young, &Value::Integer(1), &Value::Table(child)));
    let other = runtime.create_table().unwrap();
    native_table_store(&mut runtime, young, 0, other);
    assert_eq!(runtime.gc_remembered.len(), 0);

    // A handle whose table is gone: the report changes nothing.
    let gone = runtime.create_table().unwrap();
    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(kept));
    roots.value(&Value::Table(young));
    runtime.collect_full(roots.finish());
    assert!(runtime.table_length(gone).is_err());
    let fresh = runtime.create_table().unwrap();
    runtime.table_native_stored_table_slot(gone, std::ptr::null(), fresh);
    assert_eq!(runtime.gc_remembered.len(), 0);
}

/// An address that is no slot of the list's array part makes the next minor
/// collection trace the whole list, as the general barrier does, even when
/// a range is pending.
#[test]
fn a_native_table_store_at_an_address_that_names_no_slot_marks_the_whole_list() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 8);
    let first = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 3, first);
    assert_eq!(pending_slots(&runtime, kept), (3, 4));
    let child = runtime.create_table().unwrap();
    let base = runtime.tables.get(kept.0).unwrap().array.as_ptr();
    // Inside slot 3, but not at its start.
    let inside = base
        .wrapping_add(3)
        .cast::<u8>()
        .wrapping_add(1)
        .cast::<Value>();
    runtime.table_native_stored_table_slot(kept, inside, child);
    assert!(runtime.tables.get(kept.0).unwrap().gc_dirty.is_whole());

    for before in [false, true] {
        let mut runtime = Runtime::new();
        let kept = old_list(&mut runtime, 8);
        let child = runtime.create_table().unwrap();
        let capacity = runtime.tables.get(kept.0).unwrap().array.capacity();
        let base = runtime.tables.get(kept.0).unwrap().array.as_ptr();
        let outside = if before {
            base.wrapping_sub(1)
        } else {
            base.wrapping_add(capacity)
        };
        runtime.table_native_stored_table_slot(kept, outside, child);
        assert!(runtime.tables.get(kept.0).unwrap().gc_dirty.is_whole());
        assert_eq!(runtime.gc_remembered.len(), 1);
    }
}

/// The table barrier answers a store in line when no major cycle runs and
/// the list is young, or is old and already traces the slot. While a major
/// cycle runs it takes the full check for every store, whatever the list's
/// age and range: a table born and stored then, into a slot the list
/// already traces and into one it does not, survives the cycle and the
/// minor collections after it.
#[test]
fn a_native_table_store_during_a_major_cycle_keeps_the_stored_table() {
    let mut runtime = Runtime::new();
    let kept = old_list(&mut runtime, 8);
    let young = runtime.create_table().unwrap();
    let first = runtime.create_table().unwrap();
    assert!(runtime.table_set_dense_reference(young, &Value::Integer(1), &Value::Table(first)));
    // Slot 2 of the old list is pending before the cycle starts.
    let before = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 2, before);
    assert_eq!(pending_slots(&runtime, kept), (2, 3));
    let roots = || {
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(kept));
        roots.value(&Value::Table(young));
        roots.finish()
    };
    let step_roots = roots();
    assert_eq!(
        runtime.gc_step_with_roots(step_roots, 0),
        GcStepResult::InProgress
    );
    assert!(runtime.incremental_gc.is_some());
    // Three stores during the cycle: over the pending slot, into a slot
    // outside the range, and into the young list.
    let pending = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 2, pending);
    let outside = runtime.create_table().unwrap();
    native_table_store(&mut runtime, kept, 5, outside);
    let into_young = runtime.create_table().unwrap();
    native_table_store(&mut runtime, young, 0, into_young);
    let mut steps = 0;
    loop {
        let step_roots = roots();
        if runtime.gc_step_with_roots(step_roots, 0) == GcStepResult::Complete {
            break;
        }
        steps += 1;
        assert!(steps < 100_000, "the cycle never ended");
    }
    assert!(runtime.incremental_gc.is_none());
    for _ in 0..3 {
        let minor_roots = roots();
        runtime.gc_minor_collection(&minor_roots);
        #[cfg(debug_assertions)]
        assert_eq!(runtime.gc_unranged_young_child(), None);
    }
    for stored in [pending, outside, into_young] {
        assert!(runtime.table_length(stored).is_ok());
    }
    assert_eq!(
        runtime.table_get(kept, &Value::Integer(3)).unwrap(),
        Value::Table(pending)
    );
    assert_eq!(
        runtime.table_get(kept, &Value::Integer(6)).unwrap(),
        Value::Table(outside)
    );
    assert_eq!(
        runtime.table_get(young, &Value::Integer(1)).unwrap(),
        Value::Table(into_young)
    );
}

/// A table born with its array part is the table that `create_table_sized`
/// and `table_prefill_array` make: the same length and slots, and the same
/// bytes on the collector's count, which a collection gives back.
#[test]
fn a_table_born_prefilled_matches_one_prefilled_after_its_creation() {
    for count in [1_usize, 2, 3, 8, 16] {
        let length = i64::try_from(count).unwrap();
        let mut runtime = Runtime::new();
        let baseline = runtime.gc_count_bytes();
        let (born, slots) = runtime.create_table_prefilled(count).unwrap();
        let born_bytes = runtime.gc_count_bytes() - baseline;
        let sized = runtime.create_table_sized(count).unwrap();
        let sized_bytes = runtime.gc_count_bytes() - baseline - born_bytes;
        let address = runtime.table_prefill_array(sized, count).unwrap();
        assert_eq!(born_bytes, sized_bytes, "count {count}");
        for handle in [born, sized] {
            let table = runtime.tables.get(handle.0).unwrap();
            assert_eq!(table.array.len(), count);
            assert_eq!(table.border, count);
            assert!(table.array.iter().all(|value| *value == Value::Nil));
        }
        assert_eq!(
            slots,
            runtime.tables.get(born.0).unwrap().array.as_ptr() as usize
        );
        assert_eq!(
            address,
            runtime.tables.get(sized.0).unwrap().array.as_ptr() as usize
        );
        // Both take values as a list does.
        for handle in [born, sized] {
            for index in 1..=length {
                runtime
                    .table_set(handle, &Value::Integer(index), Value::Integer(7))
                    .unwrap();
            }
            assert_eq!(runtime.table_length(handle).unwrap(), count as f64);
            runtime
                .table_set(handle, &Value::Integer(length + 1), Value::Integer(8))
                .unwrap();
            assert_eq!(runtime.table_length(handle).unwrap(), (count + 1) as f64);
        }
        runtime.collect_full(GcRootBuilder::new().finish());
        assert_eq!(runtime.gc_count_bytes(), baseline, "count {count}");
    }
}
