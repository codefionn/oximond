#![allow(unsafe_code)]
use super::*;

#[test]
fn closure_snapshots_share_chunk_identity_and_cells_stay_opaque() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::Number(4.0)).unwrap();
    let chunk = Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]));
    let closure = runtime
        .create_closure(Arc::clone(&chunk), vec![cell])
        .unwrap();
    let snapshot = runtime.closure_snapshot(closure).unwrap();
    assert!(Arc::ptr_eq(&chunk, &snapshot.chunk()));
    assert_eq!(snapshot.cells(), &[cell]);
    let (owned_chunk, owned_cells, environment) = snapshot.into_parts();
    assert!(Arc::ptr_eq(&chunk, &owned_chunk));
    assert_eq!(&*owned_cells, &[cell]);
    assert_eq!(environment, runtime.global_environment());
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Number(4.0));
    runtime.cell_set(cell, Value::Number(9.0)).unwrap();
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Number(9.0));
}

#[test]
fn closure_snapshots_keep_old_cells_after_a_replacement() {
    let mut runtime = Runtime::new();
    let old_cell = runtime.create_cell(Value::Number(4.0)).unwrap();
    let replacement = runtime.create_cell(Value::Number(9.0)).unwrap();
    let closure = runtime
        .create_closure(
            Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
            vec![old_cell],
        )
        .unwrap();

    let old_snapshot = runtime.closure_snapshot(closure).unwrap();
    assert!(runtime.set_closure_cell(closure, 0, replacement).unwrap());
    let new_snapshot = runtime.closure_snapshot(closure).unwrap();

    assert_eq!(old_snapshot.cells(), &[old_cell]);
    assert_eq!(new_snapshot.cells(), &[replacement]);
}

#[test]
fn numeric_cell_accessors_reject_stale_handles() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::Number(1.0)).unwrap();
    assert!(runtime.cells.remove(cell.0).is_some());

    assert!(matches!(
        runtime.cell_get_number(cell),
        Err(RuntimeError::InvalidCell(handle)) if handle == cell
    ));
    assert!(matches!(
        runtime.cell_set_number(cell, 2.0),
        Err(RuntimeError::InvalidCell(handle)) if handle == cell
    ));
}

#[test]
fn cell_set_keeps_moved_string_storage() {
    let mut runtime = Runtime::new();
    let cell = runtime.create_cell(Value::Nil).unwrap();
    let string = crate::string::LuaString::from("cell-owned");
    let identity = string.identity();

    runtime.cell_set(cell, Value::String(string)).unwrap();

    let Value::String(stored) = runtime.cell_get(cell).unwrap() else {
        panic!("cell write must retain a string");
    };
    assert_eq!(stored.identity(), identity);
}

#[test]
fn closures_keep_explicit_environments_and_children_can_inherit_them() {
    let mut runtime = Runtime::new();
    let first_environment = runtime.create_table().unwrap();
    let second_environment = runtime.create_table().unwrap();
    let chunk = Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]));

    let parent = runtime
        .create_closure_in_environment(Arc::clone(&chunk), Vec::new(), first_environment)
        .unwrap();
    let child = runtime
        .create_closure_in_environment(
            Arc::clone(&chunk),
            Vec::new(),
            runtime.closure_environment(parent).unwrap(),
        )
        .unwrap();
    let default_closure = runtime.create_closure(chunk, Vec::new()).unwrap();

    assert_eq!(
        runtime.closure_environment(parent).unwrap(),
        first_environment
    );
    assert_eq!(
        runtime.closure_environment(child).unwrap(),
        first_environment
    );
    assert_eq!(
        runtime.closure_environment(default_closure).unwrap(),
        runtime.global_environment()
    );
    assert_eq!(
        runtime.closure_snapshot(parent).unwrap().environment(),
        first_environment
    );

    runtime
        .set_closure_environment(child, second_environment)
        .unwrap();
    assert_eq!(
        runtime.closure_environment(child).unwrap(),
        second_environment
    );
}

#[test]
fn a_closure_shape_changes_with_its_environment_and_captures() {
    let mut runtime = Runtime::new();
    let environment = runtime.create_table().unwrap();
    let first = runtime.create_cell(Value::Nil).unwrap();
    let second = runtime.create_cell(Value::Nil).unwrap();
    let chunk = Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]));
    let closure = runtime
        .create_closure(Arc::clone(&chunk), vec![first, first])
        .unwrap();
    let other = runtime.create_closure(chunk, vec![first]).unwrap();
    let shape = |runtime: &Runtime, closure| runtime.closure_call_parts(closure).unwrap().shape;

    let created = shape(&runtime, closure);
    assert_ne!(created, crate::runtime::NO_SHAPE);
    // Reading changes nothing, and neither does another closure's change.
    assert_eq!(shape(&runtime, closure), created);
    runtime.set_closure_environment(other, environment).unwrap();
    assert_eq!(shape(&runtime, closure), created);

    runtime
        .set_closure_environment(closure, environment)
        .unwrap();
    let after_environment = shape(&runtime, closure);
    assert_ne!(after_environment, created);

    assert!(runtime.set_closure_cell(closure, 1, second).unwrap());
    let after_cell = shape(&runtime, closure);
    assert_ne!(after_cell, after_environment);
    assert_ne!(after_cell, crate::runtime::NO_SHAPE);

    // An index outside the captures changes nothing.
    assert!(!runtime.set_closure_cell(closure, 2, second).unwrap());
    assert_eq!(shape(&runtime, closure), after_cell);
}

#[test]
fn utf8_code_iterators_retain_source_and_cursor() {
    let mut runtime = Runtime::new();
    let iterator = runtime.create_utf8_codes_iterator("a€".into()).unwrap();
    let snapshot = runtime.gmatch_iterator_snapshot(iterator).unwrap();
    assert!(snapshot.is_utf8_codes());
    assert_eq!(snapshot.source().as_bytes(), "a€".as_bytes());
    runtime.set_gmatch_iterator_search(iterator, 1).unwrap();
    assert_eq!(
        runtime.gmatch_iterator_snapshot(iterator).unwrap().search(),
        1
    );
}

/// A value stack with an anchor, as a thread owns one.
struct TestStack {
    values: Vec<Value>,
    anchor: Box<crate::runtime::StackAnchor>,
}

impl TestStack {
    fn new(size: usize) -> Self {
        let mut stack = Self {
            values: vec![Value::Nil; size],
            anchor: Box::new(crate::runtime::StackAnchor::new()),
        };
        stack.anchor.set_base(stack.values.as_mut_ptr());
        stack
    }

    fn open(&mut self, runtime: &mut Runtime, slot: usize) -> CellHandle {
        let anchor = std::ptr::NonNull::from(&*self.anchor);
        // SAFETY: the slot is inside the buffer and the anchor names it.
        unsafe { runtime.create_open_cell(anchor, slot as u32) }.unwrap()
    }

    fn grow(&mut self, size: usize) {
        self.values.resize(size, Value::Nil);
        self.anchor.set_base(self.values.as_mut_ptr());
    }
}

#[test]
fn an_open_cell_reads_and_writes_its_stack_slot() {
    let mut runtime = Runtime::new();
    let mut stack = TestStack::new(4);
    let cell = stack.open(&mut runtime, 2);
    stack.values[2] = Value::Number(5.0);

    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Number(5.0));
    assert_eq!(runtime.cell_get_number(cell).unwrap(), Some(5.0));
    runtime.cell_set(cell, Value::Number(6.0)).unwrap();
    assert_eq!(stack.values[2], Value::Number(6.0));
    assert!(runtime.cell_set_number(cell, 7.5).unwrap());
    assert_eq!(stack.values[2], Value::Number(7.5));
    stack.values[2] = Value::from("text");
    assert!(!runtime.cell_set_number(cell, 1.0).unwrap());
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::from("text"));
    assert!(runtime.cell_open_slot(cell).is_some());
}

#[test]
fn an_open_cell_follows_a_stack_that_moves() {
    let mut runtime = Runtime::new();
    let mut stack = TestStack::new(2);
    let cell = stack.open(&mut runtime, 1);
    stack.values[1] = Value::Number(1.0);
    // A buffer this much larger cannot stay where it was.
    stack.grow(1 << 16);
    stack.values[1] = Value::Number(2.0);
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Number(2.0));
    runtime.cell_set(cell, Value::Number(3.0)).unwrap();
    assert_eq!(stack.values[1], Value::Number(3.0));
}

#[test]
fn closing_an_open_cell_keeps_its_value_and_frees_it_from_the_stack() {
    let mut runtime = Runtime::new();
    let mut stack = TestStack::new(3);
    let cell = stack.open(&mut runtime, 0);
    stack.values[0] = Value::from("kept");

    // SAFETY: the slot still holds the variable.
    assert!(unsafe { runtime.close_open_cell(cell) });
    assert!(runtime.cell_open_slot(cell).is_none());
    stack.values[0] = Value::Number(1.0);
    assert_eq!(runtime.cell_get(cell).unwrap(), Value::from("kept"));
    runtime.cell_set(cell, Value::Number(8.0)).unwrap();
    assert_eq!(stack.values[0], Value::Number(1.0));
    // SAFETY: a closed cell has nothing to read.
    assert!(!unsafe { runtime.close_open_cell(cell) });
}

#[test]
fn collection_keeps_what_an_open_cell_holds_in_its_stack_slot() {
    let mut runtime = Runtime::new();
    let mut stack = TestStack::new(2);
    let cell = stack.open(&mut runtime, 1);
    let table = runtime.create_table().unwrap();
    stack.values[1] = Value::Table(table);
    let closure = runtime
        .create_closure(
            Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
            vec![cell],
        )
        .unwrap();

    let mut roots = crate::runtime::GcRootBuilder::new();
    roots.value(&Value::Closure(closure));
    runtime.collect_full(roots.finish());

    assert_eq!(runtime.cell_get(cell).unwrap(), Value::Table(table));
    assert!(runtime.table_get(table, &Value::Nil).is_ok());
}

fn halt_chunk() -> Arc<Chunk> {
    Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()]))
}

#[test]
fn closures_keep_none_and_one_cell_inline_and_share_a_slice_for_more() {
    let mut runtime = Runtime::new();
    let cells: Vec<CellHandle> = (0..9)
        .map(|index| {
            runtime
                .create_cell(Value::Number(f64::from(index)))
                .unwrap()
        })
        .collect();
    for count in 0..=cells.len() {
        let closure = runtime
            .create_closure(halt_chunk(), cells[..count].to_vec())
            .unwrap();
        let parts = runtime.closure_call_parts(closure).unwrap();
        assert_eq!(&**parts.cells, &cells[..count]);
        assert_eq!(
            matches!(parts.cells, ClosureCells::Many(_)),
            count >= 2,
            "{count} cells"
        );
        assert_eq!(
            runtime.closure_snapshot(closure).unwrap().cells(),
            &cells[..count]
        );
        assert_eq!(
            runtime.closure_cell(closure, 0).unwrap(),
            cells.first().filter(|_| count > 0).copied()
        );
    }
}

#[test]
fn replacing_a_cell_changes_the_shape_in_every_storage_form() {
    let mut runtime = Runtime::new();
    let cells: Vec<CellHandle> = (0..4)
        .map(|index| {
            runtime
                .create_cell(Value::Number(f64::from(index)))
                .unwrap()
        })
        .collect();
    let replacement = runtime.create_cell(Value::Number(9.0)).unwrap();
    for count in 1..=cells.len() {
        let closure = runtime
            .create_closure(halt_chunk(), cells[..count].to_vec())
            .unwrap();
        let before = runtime.closure_call_parts(closure).unwrap().shape;
        // A frame that borrowed the old slice keeps reading it.
        let borrowed = unsafe {
            CaptureCells::borrow_shared(runtime.closure_call_parts(closure).unwrap().cells)
        };
        assert!(
            !runtime
                .set_closure_cell(closure, count, replacement)
                .unwrap()
        );
        assert_eq!(runtime.closure_call_parts(closure).unwrap().shape, before);
        assert!(
            runtime
                .set_closure_cell(closure, count - 1, replacement)
                .unwrap()
        );
        let parts = runtime.closure_call_parts(closure).unwrap();
        assert_ne!(parts.shape, before);
        let mut expected = cells[..count].to_vec();
        expected[count - 1] = replacement;
        assert_eq!(&**parts.cells, &expected[..]);
        if count == 1 {
            // A single cell was copied into the frame's captures.
            assert_eq!(&*borrowed, &cells[..count]);
        } else {
            // A shared slice nobody else holds is edited in place.
            assert_eq!(&*borrowed, &expected[..], "{count} cells");
        }
    }
    let empty = runtime.create_closure(halt_chunk(), vec![]).unwrap();
    assert!(!runtime.set_closure_cell(empty, 0, replacement).unwrap());
}

#[test]
fn closures_made_from_identities_match_those_made_from_handles() {
    let mut runtime = Runtime::new();
    let cells: Vec<CellHandle> = (0..5)
        .map(|index| {
            runtime
                .create_cell(Value::Number(f64::from(index)))
                .unwrap()
        })
        .collect();
    let environment = runtime.global_environment();
    for count in 0..=cells.len() {
        let words: Vec<u64> = cells[..count]
            .iter()
            .map(|cell| cell.native_identity())
            .collect();
        let identity =
            unsafe { runtime.create_closure_from_identities(halt_chunk(), &words, environment) };
        assert_ne!(identity, crate::runtime::NO_TABLE_IDENTITY);
        let closure = ClosureHandle::from_native_identity(identity);
        assert_eq!(
            runtime.closure_snapshot(closure).unwrap().cells(),
            &cells[..count]
        );
    }
    // A stale environment makes no closure and says why.
    let stale = runtime.create_table().unwrap();
    assert!(runtime.tables.remove(stale.0).is_some());
    let identity = unsafe { runtime.create_closure_from_identities(halt_chunk(), &[], stale) };
    assert_eq!(identity, crate::runtime::NO_TABLE_IDENTITY);
    assert!(matches!(
        runtime.closure_creation_error(stale),
        RuntimeError::InvalidTable(handle) if handle == stale
    ));
}

#[test]
fn collection_keeps_the_cells_of_closures_in_every_storage_form() {
    let mut runtime = Runtime::new();
    let mut closures = Vec::new();
    for count in 0..5 {
        let cells: Vec<CellHandle> = (0..count)
            .map(|index| {
                runtime
                    .create_cell(Value::Number(f64::from(index)))
                    .unwrap()
            })
            .collect();
        closures.push((
            runtime.create_closure(halt_chunk(), cells.clone()).unwrap(),
            cells,
        ));
    }
    let mut roots = GcRootBuilder::new();
    for (closure, _) in &closures {
        roots.closure(*closure);
    }
    let roots = roots.finish();
    runtime.collect_full(roots.clone());
    runtime.collect_full(roots);
    for (closure, cells) in &closures {
        assert_eq!(
            runtime.closure_snapshot(*closure).unwrap().cells(),
            &cells[..]
        );
        for cell in cells {
            runtime.cell_get(*cell).unwrap();
        }
    }
}
