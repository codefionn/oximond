//! Native traces over upvalue cells, captured locals, globals, opaque lanes
//! and string roots, driven through hand-assembled chunks.
//!
//! Frames here carry real cells, so every case can compare what native code
//! leaves behind (registers and cells together) with an isolated interpreter
//! that starts from the same state. Counters prove the native paths ran.

use super::trace_native::{Installed, native_vm, stats_of};
use super::trace_record::*;
use super::*;
use crate::jit_ir::trace::reference::{ForcedExit, Outcome, evaluate_forced};
use crate::jit_ir::trace::{GuardId, TraceScalar};
use crate::runtime::{CellHandle, TableHandle};
use crate::vm::jit::trace::TraceExitKind;
use crate::vm::stack::CallStack;

/// What a frame holds, registers and upvalue cells both.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    pc: usize,
    last: Option<usize>,
    registers: Vec<Value>,
    cells: Vec<Value>,
}

/// A VM, a chunk, and the cells its frames use as upvalues.
struct World {
    vm: Vm,
    chunk: Arc<Chunk>,
    cells: Vec<CellHandle>,
}

impl World {
    fn with_vm(mut vm: Vm, chunk: &Arc<Chunk>, values: &[Value]) -> Self {
        let cells = values
            .iter()
            .map(|value| vm.runtime_mut().create_cell(value.clone()).unwrap())
            .collect();
        Self {
            vm,
            chunk: Arc::clone(chunk),
            cells,
        }
    }

    fn tracing(profile: CompatibilityProfile, chunk: &Arc<Chunk>, values: &[Value]) -> Self {
        Self::with_vm(native_vm(profile), chunk, values)
    }

    fn isolated(profile: CompatibilityProfile, chunk: &Arc<Chunk>, values: &[Value]) -> Self {
        Self::with_vm(interpreter_vm_with(profile), chunk, values)
    }

    /// New cells, for another closure over the same chunk.
    fn with_cells(&mut self, values: &[Value]) {
        self.cells = values
            .iter()
            .map(|value| self.vm.runtime_mut().create_cell(value.clone()).unwrap())
            .collect();
    }

    fn frames(&mut self, at: Option<&Entry>) -> CallStack {
        let mut frames = CallStack::from(vec![Frame::new(
            Arc::clone(&self.chunk),
            Vec::new(),
            self.cells.clone(),
            None,
            None,
            self.vm.runtime().global_environment(),
            None,
        )]);
        if let Some(entry) = at {
            let mut frame = frames.last_mut().unwrap();
            frame.pc = entry.pc;
            frame.last_interpreted_pc = entry.last.into();
            for (register, value) in frame.registers.iter_mut().zip(&entry.registers) {
                *register = value.clone();
            }
            for (cell, value) in self.cells.iter().zip(&entry.cells) {
                self.vm
                    .runtime_mut()
                    .cell_set(*cell, value.clone())
                    .unwrap();
            }
        }
        frames
    }

    fn cell_values(&self) -> Vec<Value> {
        self.cells
            .iter()
            .map(|cell| self.vm.runtime().cell_get(*cell).unwrap())
            .collect()
    }

    fn capture(&self, frames: &CallStack) -> Entry {
        let frame = frames.last().unwrap();
        Entry {
            pc: frame.pc,
            last: frame.last_interpreted_pc.get(),
            registers: frame.registers.to_vec(),
            cells: self.cell_values(),
        }
    }

    /// Steps the frames to completion and returns the results.
    fn finish(&mut self, frames: &mut CallStack) -> Vec<Value> {
        loop {
            if let Some(results) = self.vm.step(frames).unwrap() {
                return results;
            }
        }
    }

    /// Runs the chunk from its start until a trace is installed at `header`.
    /// Returns the trace index and the finished run's results.
    fn install(&mut self, header: usize) -> (u32, Vec<Value>) {
        let mut frames = self.frames(None);
        let results = self.finish(&mut frames);
        let index = self
            .vm
            .jit
            .trace
            .installed_index_at_pc(header)
            .unwrap_or_else(|| panic!("no trace at {header}: {:?}", stats_of(&self.vm)));
        (index, results)
    }
}

fn float(value: f64) -> Value {
    Value::Number(value)
}

// ------------------------------------------------------------- programs

/// `acc = acc + step` through two upvalue cells, `limit` times. Upvalue 0 is
/// the accumulator and upvalue 1 the increment. Header pc 4, latch pc 8.
fn upvalue_sum_program(limit: f64) -> Chunk {
    Chunk::new(
        6,
        vec![Constant::Number(1.0), Constant::Number(limit)],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 3, 8),
            Instruction::get_upvalue(4, 0),
            Instruction::get_upvalue(5, 1),
            binary(OpCode::Add, 4, 4, 5),
            Instruction::set_upvalue(4, 0),
            for_loop(0, 8, 4),
            Instruction::get_upvalue(4, 0),
            Instruction::return_(4),
        ],
    )
}

/// Swaps two upvalue cells `limit` times. Header pc 4, latch pc 8.
fn upvalue_swap_program(limit: f64) -> Chunk {
    Chunk::new(
        6,
        vec![Constant::Number(1.0), Constant::Number(limit)],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 3, 8),
            Instruction::get_upvalue(4, 0),
            Instruction::get_upvalue(5, 1),
            Instruction::set_upvalue(5, 0),
            Instruction::set_upvalue(4, 1),
            for_loop(0, 8, 4),
            Instruction::get_upvalue(4, 0),
            Instruction::return_(4),
        ],
    )
}

/// The registers of a for loop at its header: control, limit, step, the
/// visible variable and two temporaries.
fn for_registers(control: f64, limit: f64) -> Vec<Value> {
    vec![
        float(control),
        float(limit),
        float(1.0),
        float(control),
        Value::Nil,
        Value::Nil,
    ]
}

fn header_entry(registers: Vec<Value>, cells: Vec<Value>, header: usize, latch: usize) -> Entry {
    Entry {
        pc: header,
        last: Some(latch),
        registers,
        cells,
    }
}

fn float_scalars(values: &[Value]) -> Vec<TraceScalar> {
    values
        .iter()
        .map(|value| match value {
            Value::Number(number) => TraceScalar::Float(*number),
            Value::Nil => TraceScalar::Nil,
            other => panic!("not a float program value: {other:?}"),
        })
        .collect()
}

// ---------------------------------------------------------------- tests

#[test]
fn upvalue_cell_traces_match_the_interpreter_end_to_end() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let chunk = Arc::new(upvalue_sum_program(2_000.0));
        let cells = [float(0.0), float(2.0)];
        let mut traced = World::tracing(profile, &chunk, &cells);
        let (_, results) = traced.install(4);
        let mut isolated = World::isolated(profile, &chunk, &cells);
        let mut frames = isolated.frames(None);
        let expected = isolated.finish(&mut frames);
        assert_eq!(results, expected, "{profile:?}");
        assert_eq!(results, vec![float(4_000.0)], "{profile:?}");
        assert_eq!(traced.cell_values(), isolated.cell_values(), "{profile:?}");
        let stats = stats_of(&traced.vm);
        assert!(stats.iterations > 1_800, "{profile:?}: {stats:?}");
        assert_eq!(
            stats.exits_of(TraceExitKind::LoopComplete),
            1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn one_chunk_with_different_cells_shares_one_trace_and_never_a_cell() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = Arc::new(upvalue_sum_program(2_000.0));
    let mut world = World::tracing(profile, &chunk, &[float(0.0), float(2.0)]);
    let (index, _) = world.install(4);
    assert_eq!(stats_of(&world.vm).traces_compiled, 1);
    let entries = stats_of(&world.vm).native_entries;
    let first_cells = world.cell_values();
    assert_eq!(first_cells, vec![float(4_000.0), float(2.0)]);

    // The same chunk, other cells: the trace reads them at entry.
    let first = world.cells.clone();
    world.with_cells(&[float(1_000.0), float(0.5)]);
    let entry = header_entry(for_registers(1.0, 2_000.0), world.cell_values(), 4, 8);
    let mut frames = world.frames(Some(&entry));
    let ran = world.vm.enter_native_trace(&mut frames, 0, index).unwrap();
    assert!(ran.is_some(), "the trace entered with other cells");
    let results = world.finish(&mut frames);
    assert_eq!(results, vec![float(2_000.0)]);
    assert_eq!(world.cell_values(), vec![float(2_000.0), float(0.5)]);
    // The first closure's cells were not touched.
    let untouched: Vec<Value> = first
        .iter()
        .map(|cell| world.vm.runtime().cell_get(*cell).unwrap())
        .collect();
    assert_eq!(untouched, first_cells);
    let stats = stats_of(&world.vm);
    assert_eq!(stats.traces_compiled, 1, "{stats:?}");
    assert_eq!(stats.native_entries, entries + 1, "{stats:?}");
}

#[test]
fn every_guard_of_a_cell_trace_exits_into_a_state_the_interpreter_reaches() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = Arc::new(upvalue_sum_program(2_000.0));
    let mut world = World::tracing(profile, &chunk, &[float(0.0), float(2.0)]);
    let (index, _) = world.install(4);
    let lowered = world
        .vm
        .jit
        .trace
        .installed_lowered_ir(index)
        .unwrap()
        .clone();
    assert_eq!(lowered.external.len(), 2, "both upvalues are window slots");
    let live_in = crate::jit_ir::observable_registers(&chunk);
    let mut forced_exits = 0;
    let mut interpreter_checked = 0;
    for (guard_index, guard) in lowered.guards.iter().enumerate() {
        let guard_snapshot = guard.snapshot;
        assert!(world.vm.jit.trace.force_guard(index, guard_index));
        for control in [700.0, 1_999.0] {
            let entry = header_entry(
                for_registers(control, 2_000.0),
                vec![float(control * 2.0), float(2.0)],
                4,
                8,
            );
            let mut window = float_scalars(&entry.registers);
            window.extend(float_scalars(&entry.cells));
            let forced = ForcedExit {
                frame: guard.site.frame,
                pc: guard.site.pc,
                guard: Some(GuardId(u32::try_from(guard_index).unwrap())),
                iteration: 0,
            };
            let evaluation =
                evaluate_forced(&lowered, &window, &[], 600, Some(forced), |_, _| None);
            let Outcome::GuardFailed { restored, .. } = evaluation.outcome else {
                panic!("guard {guard_index} did not fail when forced");
            };
            let real = entry.registers.len();
            let to_values = |scalars: &[TraceScalar]| -> Vec<Value> {
                scalars
                    .iter()
                    .map(|scalar| match scalar {
                        TraceScalar::Float(value) => float(*value),
                        TraceScalar::Nil => Value::Nil,
                        other => panic!("{other:?}"),
                    })
                    .collect()
            };
            let expected = Entry {
                pc: restored.pc,
                last: restored.last_interpreted_pc,
                registers: to_values(&restored.registers[..real]),
                cells: to_values(&restored.registers[real..]),
            };
            // Native code, on a frame whose cells hold the entry values.
            world.with_cells(&entry.cells);
            let mut frames = world.frames(Some(&entry));
            let ran = world
                .vm
                .enter_native_trace_with_fuel(&mut frames, 0, index, 600)
                .unwrap();
            assert!(ran.is_some(), "guard {guard_index} did not run");
            let actual = world.capture(&frames);
            assert_eq!(actual.pc, expected.pc, "guard {guard_index}");
            assert_eq!(actual.last, expected.last, "guard {guard_index}");
            assert_eq!(actual.cells, expected.cells, "guard {guard_index} cells");
            assert_same_registers(&actual, &expected, &live_in, guard_index);
            // A guard that fails before its instruction restores a state the
            // interpreter passes through. A guard after one (the loop test,
            // forced here to fail early) restores the state of a different
            // run, which only the reference evaluator describes.
            if !matches!(
                lowered.snapshots[guard_snapshot.index()].phase,
                crate::jit_ir::trace::SnapshotPhase::Before { .. }
            ) {
                forced_exits += 1;
                continue;
            }
            let mut isolated = World::isolated(profile, &chunk, &entry.cells);
            let mut iso_frames = isolated.frames(Some(&entry));
            let mut reached = false;
            for _ in 0..20_000 {
                let state = isolated.capture(&iso_frames);
                if state.pc == actual.pc
                    && state.last == actual.last
                    && state.cells == actual.cells
                    && registers_agree(&state, &actual, &live_in)
                {
                    reached = true;
                    break;
                }
                if isolated.vm.step(&mut iso_frames).unwrap().is_some() {
                    break;
                }
            }
            assert!(reached, "guard {guard_index} restored {actual:?}");
            forced_exits += 1;
            interpreter_checked += 1;
        }
    }
    assert!(forced_exits >= 4, "{forced_exits} forced exits");
    assert!(
        interpreter_checked >= 2,
        "{interpreter_checked} exits checked against the interpreter"
    );
}

fn registers_agree(left: &Entry, right: &Entry, live_in: &[crate::jit_ir::RegisterSet]) -> bool {
    let live = live_in.get(left.pc);
    left.registers
        .iter()
        .zip(&right.registers)
        .enumerate()
        .all(|(register, (a, b))| {
            a == b
                || live.is_some_and(|set| {
                    u8::try_from(register)
                        .is_ok_and(|register| !crate::jit_ir::register_set_contains(set, register))
                })
        })
}

fn assert_same_registers(
    actual: &Entry,
    expected: &Entry,
    live_in: &[crate::jit_ir::RegisterSet],
    guard: usize,
) {
    assert!(
        registers_agree(actual, expected, live_in),
        "guard {guard}: registers {:?} against {:?}",
        actual.registers,
        expected.registers
    );
}

#[test]
fn a_poll_publishes_swapped_cells_and_the_collector_keeps_both_tables() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = Arc::new(upvalue_swap_program(300.0));
    let mut vm = native_vm(profile);
    let a = vm.runtime_mut().create_table().unwrap();
    let b = vm.runtime_mut().create_table().unwrap();
    for (table, name) in [(a, "a"), (b, "b")] {
        vm.runtime_mut()
            .table_set(table, &Value::from("name"), Value::from(name))
            .unwrap();
    }
    let mut world = World::with_vm(vm, &chunk, &[Value::Table(a), Value::Table(b)]);
    let (index, results) = world.install(4);
    // 300 swaps leave the cells as they began.
    assert_eq!(results, vec![Value::Table(a)]);
    assert_eq!(world.cell_values(), vec![Value::Table(a), Value::Table(b)]);
    let entry_stats = stats_of(&world.vm);
    assert!(entry_stats.iterations > 200, "{entry_stats:?}");

    // Three loop-backs, then a poll. The restored cells are swapped an odd
    // number of times, and only the poll's exit put them there.
    let entry = header_entry(
        vec![
            float(1.0),
            float(300.0),
            float(1.0),
            float(1.0),
            Value::Nil,
            Value::Nil,
        ],
        vec![Value::Table(a), Value::Table(b)],
        4,
        8,
    );
    let mut frames = world.frames(Some(&entry));
    let polls = stats_of(&world.vm).exits_of(TraceExitKind::Poll);
    let ran = world
        .vm
        .enter_native_trace_with_fuel(&mut frames, 0, index, 3)
        .unwrap();
    assert_eq!(ran, Some(true), "a poll asks for the safepoint");
    assert_eq!(stats_of(&world.vm).exits_of(TraceExitKind::Poll), polls + 1);
    assert_eq!(world.cell_values(), vec![Value::Table(b), Value::Table(a)]);
    assert!(
        world.vm.jit.trace.root_values().is_empty(),
        "the bank holds nothing once the exit restored the frame"
    );

    // A collection at the poll's safepoint keeps both tables: they live in
    // the cells the exit wrote.
    let roots = world.vm.gc_roots_with_coroutines(&mut frames, &[]);
    world.vm.runtime_mut().collect_full(roots);
    for (table, name) in [(a, "a"), (b, "b")] {
        assert_eq!(
            world
                .vm
                .runtime()
                .table_get(table, &Value::from("name"))
                .unwrap(),
            Value::from(name)
        );
    }
}

#[test]
fn the_root_bank_alone_keeps_an_opaque_value_alive_and_empties_between_runs() {
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(table, &Value::from("k"), Value::Integer(7))
        .unwrap();
    // Nothing but the bank reaches the table.
    vm.jit.trace.bank_mut().push(Value::Table(table));
    vm.collect_full_with_values(&[]);
    assert_eq!(
        vm.runtime().table_get(table, &Value::from("k")).unwrap(),
        Value::Integer(7),
        "a lane's table survives a collection while its run is in flight"
    );
    vm.jit.trace.bank_mut().clear();
    vm.collect_full_with_values(&[]);
    assert!(
        vm.runtime().table_get(table, &Value::from("k")).is_err(),
        "an empty bank pins nothing"
    );
}

#[test]
fn a_collection_while_a_run_is_in_flight_roots_whole_windows_and_clears_nothing() {
    // At pc 0 only register 2 is read: registers 0 and 1 are dead there. A
    // native run may have written them, though, and its frame's pc does not
    // say so until the exit restores it.
    let chunk = Arc::new(Chunk::new(3, Vec::new(), vec![Instruction::return_(2)]));
    let mut vm = native_vm(CompatibilityProfile::Lua51);
    let frames_with_tables = |vm: &mut Vm| {
        let mut table = |tag: i64| {
            let table = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .table_set(table, &Value::from("tag"), Value::Integer(tag))
                .unwrap();
            Value::Table(table)
        };
        let registers = vec![table(0), table(1), table(2)];
        let frame = Frame::new_with_registers(
            Arc::clone(&chunk),
            Vec::new(),
            Vec::new(),
            None,
            None,
            vm.runtime().global_environment(),
            None,
            registers,
        );
        CallStack::from(vec![frame])
    };
    let tag = |vm: &Vm, value: &Value| {
        let Value::Table(table) = value else {
            panic!("expected a table, got {value:?}");
        };
        vm.runtime().table_get(*table, &Value::from("tag")).unwrap()
    };

    // In flight, with an empty root bank (a run whose lanes name no table):
    // the dead registers keep their tables and the collection keeps them.
    let mut frames = frames_with_tables(&mut vm);
    assert_eq!(vm.jit.trace.root_values(), []);
    vm.jit.trace.set_run_in_flight(true);
    let roots = vm.gc_roots_with_coroutines(&mut frames, &[]);
    vm.runtime_mut().collect_full(roots);
    vm.jit.trace.set_run_in_flight(false);
    let registers = frames.last_registers().unwrap().to_vec();
    for (register, value) in registers.iter().enumerate() {
        assert_eq!(
            tag(&vm, value),
            Value::Integer(i64::try_from(register).unwrap()),
            "register {register} kept through a collection in flight"
        );
    }

    // Between runs the same frame loses its dead registers, and the live
    // one stays intact.
    let mut frames = frames_with_tables(&mut vm);
    let roots = vm.gc_roots_with_coroutines(&mut frames, &[]);
    vm.runtime_mut().collect_full(roots);
    let registers = frames.last_registers().unwrap().to_vec();
    assert!(matches!(registers[0], Value::Nil));
    assert!(matches!(registers[1], Value::Nil));
    assert_eq!(tag(&vm, &registers[2]), Value::Integer(2));
}

#[test]
fn a_cell_aliased_with_another_upvalue_or_a_captured_local_is_refused() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = Arc::new(upvalue_sum_program(2_000.0));
    let mut world = World::tracing(profile, &chunk, &[float(0.0), float(2.0)]);
    let (index, _) = world.install(4);
    let entries = stats_of(&world.vm).native_entries;
    let iterations = stats_of(&world.vm).iterations;

    // Both upvalues are one cell: two window slots for one variable.
    let shared = world.cells[0];
    world.cells = vec![shared, shared];
    world.vm.runtime_mut().cell_set(shared, float(5.0)).unwrap();
    let entry = header_entry(for_registers(1.0, 2_000.0), vec![float(5.0); 2], 4, 8);
    let mut frames = world.frames(Some(&entry));
    let before = world.capture(&frames);
    let ran = world.vm.enter_native_trace(&mut frames, 0, index).unwrap();
    assert_eq!(ran, None);
    assert_eq!(world.capture(&frames), before);
    let mismatches = stats_of(&world.vm).exits_of(TraceExitKind::EntryMismatch);

    // An upvalue that is also a captured local of the same frame.
    world.with_cells(&[float(1.0), float(2.0)]);
    let entry = header_entry(for_registers(1.0, 2_000.0), world.cell_values(), 4, 8);
    let mut frames = world.frames(Some(&entry));
    let cell = world.cells[0];
    // The cell now stands for register 5, which holds what it held.
    frames.at_mut(0).registers[5] = float(1.0);
    open_cell_at_register(&mut world.vm, &mut frames, 0, 5, cell);
    let ran = world.vm.enter_native_trace(&mut frames, 0, index).unwrap();
    assert_eq!(ran, None);
    assert_eq!(world.cell_values(), vec![float(1.0), float(2.0)]);
    // Both refusals count as entries that matched no version, and no
    // iteration ran.
    let stats = stats_of(&world.vm);
    assert_eq!(
        stats.exits_of(TraceExitKind::EntryMismatch),
        mismatches + 1,
        "{stats:?}"
    );
    assert_eq!(stats.native_entries, entries + 2, "{stats:?}");
    assert_eq!(stats.iterations, iterations, "{stats:?}");
    // The interpreter still computes the right sum from there.
    let mut isolated = World::isolated(profile, &chunk, &[float(1.0), float(2.0)]);
    let entry = header_entry(for_registers(1.0, 2_000.0), isolated.cell_values(), 4, 8);
    let mut iso_frames = isolated.frames(Some(&entry));
    let expected = isolated.finish(&mut iso_frames);
    assert_eq!(expected, vec![float(4_001.0)]);
}

#[test]
fn a_captured_register_reconciles_with_its_cell_at_every_exit() {
    // The running sum (register 4) is a captured local: its open cell is the
    // register. Every way out of the trace must leave the register at the
    // value the interpreter holds there: the end of the loop, a poll, a
    // failed guard. The cell then reads the same.
    let mut installed =
        Installed::with_profile(CompatibilityProfile::Lua51, sum_program(5_000.0), 5);
    let run = drive(&installed.chunk, None);
    let visits = header_visits(&run, 5);
    let state = run.states[visits[80]].clone();
    let lowered = installed
        .vm
        .jit
        .trace
        .installed_lowered_ir(installed.index)
        .unwrap()
        .clone();
    let mut cases: Vec<(String, Option<usize>, i64)> = vec![
        ("loop complete".to_string(), None, 1_000_000),
        ("poll".to_string(), None, 7),
    ];
    for guard in 0..lowered.guards.len() {
        cases.push((format!("guard {guard}"), Some(guard), 600));
    }
    for (name, guard, fuel) in cases {
        if let Some(guard) = guard
            && !installed.vm.jit.trace.force_guard(installed.index, guard)
        {
            continue;
        }
        let mut frames = frames_for(&installed.vm, &installed.chunk, Some(&state));
        let cell = installed.vm.capture_slot(&mut frames, 0, 4).unwrap();
        // The cell is the register: it reads what the frame holds.
        assert_eq!(
            installed.vm.runtime().cell_get(cell).unwrap(),
            frames.last().unwrap().registers[4]
        );
        let ran = installed
            .vm
            .enter_native_trace_with_fuel(&mut frames, 0, installed.index, fuel)
            .unwrap();
        assert!(ran.is_some(), "{name}");
        let register = frames.last().unwrap().registers[4].clone();
        assert_eq!(
            installed.vm.runtime().cell_get(cell).unwrap(),
            register,
            "{name}: the cell holds what the register does"
        );
        installed.vm.close_upvalues(&mut frames, 0);
        // A consistent but wrong value would pass the checks above, so also
        // find the exit's state on the interpreter's own path. A guard that
        // fails after its instruction describes a different run, so only the
        // other exits are compared.
        let before = guard.is_none_or(|guard| {
            matches!(
                lowered.snapshots[lowered.guards[guard].snapshot.index()].phase,
                crate::jit_ir::trace::SnapshotPhase::Before { .. }
            )
        });
        if before {
            let pc = frames.last().unwrap().pc;
            let interpreter = interpreter_vm_with(CompatibilityProfile::Lua51);
            let mut iso = frames_for(&interpreter, &installed.chunk, Some(&state));
            let mut interpreter = interpreter;
            let mut reached = false;
            for _ in 0..20_000 {
                let frame = iso.last().unwrap();
                if frame.pc == pc && frame.registers[4] == register {
                    reached = true;
                    break;
                }
                if interpreter.step(&mut iso).unwrap().is_some() {
                    break;
                }
            }
            assert!(
                reached,
                "{name}: the interpreter never holds {register:?} at pc {pc}"
            );
        }
    }
}

#[test]
fn a_trace_reads_a_global_function_at_entry_and_a_rebound_one_is_not_stale() {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let chunk = Arc::new(global_compare_program(300.0, "gf"));
        let mut vm = native_vm(profile);
        let environment = vm.runtime().global_environment();
        let body = Arc::new(Chunk::new(1, Vec::new(), vec![Instruction::return_(0)]));
        let f1 = vm
            .install_closure(Arc::clone(&body), Vec::new(), environment)
            .unwrap();
        let f2 = vm
            .install_closure(Arc::clone(&body), Vec::new(), environment)
            .unwrap();
        vm.runtime_mut().set_global("gf", Value::Closure(f1));
        let mut world = World::with_vm(vm, &chunk, &[]);
        // Registers 8 holds the function the loop compares the global with.
        let run = |world: &mut World, held: crate::runtime::ClosureHandle| {
            let mut frames = world.frames(None);
            frames.at_mut(0).registers[8] = Value::Closure(held);
            world.finish(&mut frames)
        };
        let first = run(&mut world, f1);
        assert_eq!(first, vec![float(300.0), Value::from("tag")], "{profile:?}");
        let stats = stats_of(&world.vm);
        assert!(stats.iterations > 200, "{profile:?}: {stats:?}");
        // Rebind the global. The compiled trace names no constant, so the
        // new value decides every comparison.
        world.vm.runtime_mut().set_global("gf", Value::Closure(f2));
        let second = run(&mut world, f1);
        assert_eq!(second, vec![float(0.0), Value::from("tag")], "{profile:?}");
        let third = run(&mut world, f2);
        assert_eq!(third, vec![float(300.0), Value::from("tag")], "{profile:?}");
        let mut isolated_vm = interpreter_vm_with(profile);
        let env = isolated_vm.runtime().global_environment();
        let g1 = isolated_vm
            .install_closure(Arc::clone(&body), Vec::new(), env)
            .unwrap();
        isolated_vm
            .runtime_mut()
            .set_global("gf", Value::Closure(g1));
        let mut isolated = World::with_vm(isolated_vm, &chunk, &[]);
        let mut frames = isolated.frames(None);
        frames.at_mut(0).registers[8] = Value::Closure(g1);
        assert_eq!(isolated.finish(&mut frames), first, "{profile:?}");
    }
}

#[test]
fn tables_with_eq_are_never_compared_natively() {
    // Once at the baseline tier, where the comparison runs inside the loop,
    // and once at the optimized tier, where the optimizer hoists it into the
    // preheader. Either way a pair that is only equal through `__eq` must
    // leave the trace before the comparison.
    for (limit, promoted) in [(100.0, false), (700.0, true)] {
        let profile = CompatibilityProfile::Lua51;
        let chunk = Arc::new(global_compare_program(limit, "gt"));
        let mut vm = native_vm(profile);
        if !promoted {
            vm.jit
                .trace
                .set_compile_tier(crate::jit::OptimizationTier::Baseline);
        }
        let environment = vm.runtime().global_environment();
        let eq_body = Arc::new(Chunk::new(
            1,
            vec![Constant::Boolean(true)],
            vec![load(0, 0), Instruction::return_(0)],
        ));
        let always = vm
            .install_closure(eq_body, Vec::new(), environment)
            .unwrap();
        let metatable = vm.runtime_mut().create_table().unwrap();
        vm.runtime_mut()
            .table_set(metatable, &Value::from("__eq"), Value::Closure(always))
            .unwrap();
        let a = vm.runtime_mut().create_table().unwrap();
        let b = vm.runtime_mut().create_table().unwrap();
        for table in [a, b] {
            vm.runtime_mut()
                .raw_set_metatable(&Value::Table(table), Some(metatable))
                .unwrap();
        }
        vm.runtime_mut().set_global("gt", Value::Table(a));
        let mut world = World::with_vm(vm, &chunk, &[]);
        let run = |world: &mut World, held: TableHandle| {
            let mut frames = world.frames(None);
            frames.at_mut(0).registers[8] = Value::Table(held);
            world.finish(&mut frames)
        };
        let expected = |count: f64| vec![float(count), Value::from("tag")];
        // The same table: identity, natively.
        assert_eq!(run(&mut world, a), expected(limit));
        if promoted {
            // Optimized code arrives at a safepoint after the run that asked
            // for it.
            assert_eq!(run(&mut world, a), expected(limit));
        }
        let index = world.vm.jit.trace.installed_index_at_pc(6).unwrap();
        let tier = world.vm.jit.trace.installed_tier(index).unwrap();
        assert_eq!(
            tier == crate::jit::OptimizationTier::Optimized,
            promoted,
            "the run was meant to end at the {} tier",
            if promoted { "optimized" } else { "baseline" }
        );
        assert!(stats_of(&world.vm).iterations > 20);
        // A different table with the same `__eq`: only the interpreter can
        // say it is equal, so the trace must leave before comparing.
        let guard_fails = stats_of(&world.vm).exits_of(TraceExitKind::GuardFail);
        assert_eq!(run(&mut world, b), expected(limit));
        let stats = stats_of(&world.vm);
        assert!(
            stats.exits_of(TraceExitKind::GuardFail) > guard_fails,
            "the native compare saw two tables and left: {stats:?}"
        );
        // Without the metamethod they differ.
        world
            .vm
            .runtime_mut()
            .raw_set_metatable(&Value::Table(b), None)
            .unwrap();
        assert_eq!(run(&mut world, b), expected(0.0));
    }
}

/// Counts the iterations where global `name` equals register 8, then leaves
/// the string "tag" in register 7. Header pc 6, latch pc 11. Register 9 holds
/// the increment and register 6 the count.
fn global_compare_program(limit: f64, name: &str) -> Chunk {
    Chunk::new(
        10,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
            Constant::String(name.into()),
            Constant::String("tag".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(6, 2),
            load(9, 0),
            for_prep(0, 5, 11),
            Instruction::get_global(4, 3),
            binary(OpCode::Equal, 5, 4, 8),
            jump_if_false(5, 8, 10),
            binary(OpCode::Add, 6, 6, 9),
            load(7, 4),
            for_loop(0, 11, 6),
            Instruction::return_values(6, 2),
        ],
    )
}

// ------------------------------------------------------------- recording

/// A VM that records but leaves closed traces in the queue.
fn recording_vm(profile: CompatibilityProfile) -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    vm.jit.trace.set_native_compilation(false);
    vm
}

fn queued_ir(world: &World) -> crate::jit_ir::trace::TraceIr {
    let queued = world.vm.jit.trace.queued();
    assert_eq!(queued.len(), 1, "{:?}", stats_of(&world.vm));
    queued[0].ir.clone()
}

#[test]
fn the_recorder_names_upvalues_as_window_slots() {
    use crate::jit_ir::trace::{ExternalSlot, TraceType};
    let chunk = Arc::new(upvalue_sum_program(300.0));
    let mut world = World::with_vm(
        recording_vm(CompatibilityProfile::Lua51),
        &chunk,
        &[float(0.0), float(2.0)],
    );
    let mut frames = world.frames(None);
    world.finish(&mut frames);
    let ir = queued_ir(&world);
    ir.verify().unwrap();
    assert_eq!(
        ir.external,
        vec![ExternalSlot::Upvalue(0), ExternalSlot::Upvalue(1)]
    );
    // Six guest registers, then the two cells.
    assert_eq!(ir.frames[0].register_count, 8);
    let input_type = |register: u16| {
        ir.entry
            .inputs
            .iter()
            .find(|input| input.register == register)
            .map(|input| input.ty)
    };
    assert_eq!(input_type(6), Some(TraceType::Float), "the accumulator");
    assert_eq!(input_type(7), Some(TraceType::Float), "the increment");
    // Only the accumulator is written.
    assert!(
        ir.written_registers
            .contains(&(crate::jit_ir::trace::FrameId(0), 6))
    );
    assert!(
        !ir.written_registers
            .contains(&(crate::jit_ir::trace::FrameId(0), 7))
    );
    // The accumulator is loop-carried, the increment is not.
    assert!(ir.loop_params.iter().any(
        |param| matches!(param.init, crate::jit_ir::trace::LoopParamInit::Input(value)
                if ir.entry.inputs.iter().any(|input| input.value == value && input.register == 6))
    ));
}

#[test]
fn two_upvalue_indices_for_one_cell_share_one_window_slot() {
    use crate::jit_ir::trace::{CellAlias, CellSource, ExternalSlot};
    let chunk = Arc::new(upvalue_sum_program(300.0));
    let mut world = World::with_vm(
        recording_vm(CompatibilityProfile::Lua51),
        &chunk,
        &[float(0.0), float(2.0)],
    );
    let shared = world.cells[0];
    world.cells = vec![shared, shared];
    let mut frames = world.frames(None);
    world.finish(&mut frames);
    let stats = stats_of(&world.vm);
    assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    // One variable, one window slot. The second name is an alias the VM
    // checks at entry.
    let ir = world
        .vm
        .jit
        .trace
        .queued()
        .front()
        .expect("recorded")
        .ir
        .clone();
    assert_eq!(ir.external, vec![ExternalSlot::Upvalue(0)]);
    assert_eq!(
        ir.cell_aliases,
        vec![CellAlias {
            cell: CellSource::Upvalue(1),
            is: CellSource::Upvalue(0),
        }]
    );
}

#[test]
fn a_shared_cell_runs_natively_and_a_frame_that_breaks_the_promise_is_refused() {
    let profile = CompatibilityProfile::Lua51;
    let chunk = Arc::new(upvalue_sum_program(2_000.0));
    let mut traced = World::tracing(profile, &chunk, &[float(1.0), float(1.0)]);
    let shared = traced.cells[0];
    traced.cells = vec![shared, shared];
    let mut frames = traced.frames(None);
    let results = traced.finish(&mut frames);
    // The interpreter alone, with the same shared cell.
    let mut isolated = World::isolated(profile, &chunk, &[float(1.0), float(1.0)]);
    let shared = isolated.cells[0];
    isolated.cells = vec![shared, shared];
    let mut frames = isolated.frames(None);
    let expected = isolated.finish(&mut frames);
    assert_eq!(results, expected);
    assert_eq!(traced.cell_values(), isolated.cell_values());
    let stats = stats_of(&traced.vm);
    assert!(stats.iterations > 1_000, "{stats:?}");

    // A frame whose two upvalues are two cells is not the one the trace was
    // recorded for. It does not enter.
    let index = traced
        .vm
        .jit
        .trace
        .installed_index_at_pc(4)
        .expect("installed");
    traced.with_cells(&[float(1.0), float(5.0)]);
    let entry = header_entry(for_registers(1.0, 2_000.0), traced.cell_values(), 4, 8);
    let mut frames = traced.frames(Some(&entry));
    let ran = traced.vm.enter_native_trace(&mut frames, 0, index).unwrap();
    assert_eq!(ran, None);
}

#[test]
fn globals_the_recorder_cannot_repeat_abort_it() {
    use crate::vm::jit::trace::TraceAbortReason::UnsupportedOperand;
    // A missing global and `_ENV` by name.
    for (name, setup, profile) in [
        ("missing", None, CompatibilityProfile::Lua51),
        ("_ENV", None, CompatibilityProfile::Lua52),
    ] {
        let global = if name == "_ENV" { "_ENV" } else { "gx" };
        let chunk = Arc::new(Chunk::new(
            6,
            vec![
                Constant::Number(1.0),
                Constant::Number(300.0),
                Constant::String(global.into()),
            ],
            vec![
                load(0, 0),
                load(1, 1),
                load(2, 0),
                for_prep(0, 3, 6),
                Instruction::get_global(4, 2),
                Instruction::move_(5, 4),
                for_loop(0, 6, 4),
                Instruction::return_(5),
            ],
        ));
        let mut vm = recording_vm(profile);
        if let Some(value) = setup {
            vm.runtime_mut().set_global("gx", value);
        }
        let mut world = World::with_vm(vm, &chunk, &[]);
        let mut frames = world.frames(None);
        world.finish(&mut frames);
        let stats = stats_of(&world.vm);
        assert_eq!(stats.traces_recorded, 0, "{name}: {stats:?}");
        assert!(stats.aborts(UnsupportedOperand) > 0, "{name}: {stats:?}");
    }
}

#[test]
fn a_global_function_or_table_records_as_a_read_only_window_slot() {
    use crate::jit_ir::trace::{ExternalSlot, RegisterEntry, TraceOp, TraceType};
    let chunk = Arc::new(global_compare_program(300.0, "gt"));
    let mut vm = recording_vm(CompatibilityProfile::Lua51);
    let environment = vm.runtime().global_environment();
    let table = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut().set_global("gt", Value::Table(table));
    let mut world = World::with_vm(vm, &chunk, &[]);
    let mut frames = world.frames(None);
    frames.at_mut(0).registers[8] = Value::Table(table);
    world.finish(&mut frames);
    let _ = environment;
    let ir = queued_ir(&world);
    ir.verify().unwrap();
    assert_eq!(ir.external, vec![ExternalSlot::Global(3)]);
    // The global is a Table input in the window slot after the registers.
    let slot = ir.frames[0].register_count - 1;
    assert!(
        ir.entry
            .inputs
            .iter()
            .any(|input| input.register == slot && input.ty == TraceType::Table)
    );
    // It is never written, so no snapshot maps it to anything but Unchanged.
    assert!(
        !ir.written_registers
            .iter()
            .any(|(_, register)| *register == slot)
    );
    for snapshot in &ir.snapshots {
        assert_eq!(
            snapshot.frames[0].registers[usize::from(slot)],
            RegisterEntry::Unchanged
        );
    }
    // Two tables compare behind a guard that leaves before the comparison.
    let compare = ir
        .body
        .iter()
        .position(|instruction| matches!(instruction.op, TraceOp::Compare { .. }))
        .expect("the comparison");
    let guard = &ir.body[compare + 1];
    let TraceOp::GuardBranch {
        expected, guard, ..
    } = guard.op
    else {
        panic!("a branch guard follows the table comparison");
    };
    assert!(expected);
    let snapshot = &ir.snapshots[ir.guards[guard.index()].snapshot.index()];
    assert!(matches!(
        snapshot.phase,
        crate::jit_ir::trace::SnapshotPhase::Before { pc: 7 }
    ));
    // The string the loop stores rides in a root slot.
    assert_eq!(
        ir.root_constants,
        vec![std::sync::Arc::<[u8]>::from(&b"tag"[..])]
    );
    assert!(ir.snapshots.iter().any(|snapshot| {
        snapshot.frames[0]
            .registers
            .iter()
            .any(|entry| *entry == RegisterEntry::Root(0))
    }));
}

#[test]
fn comparing_two_different_tables_without_a_handler_records_false() {
    // Two tables that are not the same are unequal when no metatable has
    // `__eq`: the trace checks they differ and that no handler answers.
    let chunk = Arc::new(global_compare_program(300.0, "gt"));
    let mut vm = recording_vm(CompatibilityProfile::Lua51);
    let a = vm.runtime_mut().create_table().unwrap();
    let b = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut().set_global("gt", Value::Table(a));
    let mut world = World::with_vm(vm, &chunk, &[]);
    let mut frames = world.frames(None);
    frames.at_mut(0).registers[8] = Value::Table(b);
    world.finish(&mut frames);
    let stats = stats_of(&world.vm);
    assert_eq!(
        stats.aborts(crate::vm::jit::trace::TraceAbortReason::UnsupportedOperand),
        0,
        "{stats:?}"
    );
    queued_ir(&world).verify().unwrap();
}
