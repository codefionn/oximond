//! Native table access in the VM, driven through hand-assembled chunks over
//! real tables.
//!
//! The plan's worked recovery case runs here for real:
//!
//! ```lua
//! for i = 1, n do
//!   local x = u[i]          -- the load; u and t may be one table
//!   t[i] = x + 1            -- the store, committed once
//!   if x > 0 then sum = sum + x else sum = sum - x end
//! end
//! ```
//!
//! A trace of the "then" arm loads, stores, and then fails its branch guard in
//! an iteration whose `x` is not positive: the store has committed, so the
//! restored state must resume after it. Every exit a case takes is compared
//! with an isolated interpreter that starts from a copy of the same tables.
//! The exit state must be a state that interpreter passes through, with the
//! same registers (dead ones aside) and the same table contents, which means
//! no store was replayed or lost. Counters prove the native paths ran.

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_borrow,
    clippy::single_match_else
)]

use super::trace_native::{Installed, guarded_vm, stats_of};
use super::trace_record::*;
use super::*;
use crate::runtime::TableHandle;
use crate::vm::jit::trace::{TraceExitKind, TraceGuardKind, TraceRun};

// --------------------------------------------------------------- programs

/// A numeric constant of the profile.
fn constant(integers: bool, value: i64) -> Constant {
    if integers {
        Constant::Integer(value)
    } else {
        Constant::Number(value as f64)
    }
}

/// Index of the loop's first instruction, the trace header.
const HEADER: usize = 7;
/// Index of the numeric-for latch.
const LATCH: usize = 15;

/// The worked case. `r5` is the table `t` the loop stores into and `r11` the
/// table `u` it loads from.
///
/// ```text
///  0..5  constants into r0 r1 r2 r4 r9 r10, 6 for_prep
///  7  r6 = u[i]            header, pc 7
///  8  r7 = r6 + 1
///  9  t[i] = r7
/// 10  r8 = 0 < r6
/// 11  jump if false to 14
/// 12  sum = sum + r6
/// 13  jump to 15
/// 14  sum = sum - r6
/// 15  for loop to 7        latch
/// 16  return sum
/// ```
fn worked_program(integers: bool, limit: i64) -> Chunk {
    Chunk::new(
        12,
        vec![
            constant(integers, 1),
            constant(integers, limit),
            constant(integers, 0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(9, 0),
            load(10, 2),
            for_prep(0, 6, LATCH),
            Instruction::get_table(6, 11, 3),
            binary(OpCode::Add, 7, 6, 9),
            Instruction::set_table(5, 3, 7),
            binary(OpCode::LessThan, 8, 10, 6),
            jump_if_false(8, 11, 14),
            binary(OpCode::Add, 4, 4, 6),
            jump_to(13, LATCH),
            binary(OpCode::Subtract, 4, 4, 6),
            for_loop(0, LATCH, HEADER),
            Instruction::return_(4),
        ],
    )
}

/// `t.x = t.x + 1` and `sum = sum + t.x` on the field "x" of `r5`, `limit`
/// times. Header pc 7, latch pc 12.
fn field_program(integers: bool, limit: i64) -> Chunk {
    Chunk::new(
        13,
        vec![
            constant(integers, 1),
            constant(integers, limit),
            constant(integers, 0),
            Constant::String("x".into()),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(9, 0),
            load(10, 2),
            for_prep(0, 6, 12),
            load(12, 3),
            Instruction::get_table(6, 5, 12),
            binary(OpCode::Add, 7, 6, 9),
            Instruction::set_table(5, 12, 7),
            binary(OpCode::Add, 4, 4, 6),
            for_loop(0, 12, 7),
            Instruction::return_(4),
        ],
    )
}

/// `n = #t` summed into `r4` each iteration. Header pc 7, latch pc 9.
fn length_program(integers: bool, limit: i64) -> Chunk {
    Chunk::new(
        13,
        vec![
            constant(integers, 1),
            constant(integers, limit),
            constant(integers, 0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 2),
            load(9, 0),
            load(10, 2),
            for_prep(0, 6, 9),
            Instruction::binary(OpCode::Length, 6, 5, 0),
            binary(OpCode::Add, 4, 4, 6),
            for_loop(0, 9, 7),
            Instruction::return_(4),
        ],
    )
}

// ------------------------------------------------------------------ world

fn number(profile: CompatibilityProfile, value: i64) -> Value {
    if profile.is_lua53_or_later() {
        Value::Integer(value)
    } else {
        Value::Number(value as f64)
    }
}

fn key(profile: CompatibilityProfile, index: usize) -> Value {
    number(profile, i64::try_from(index).unwrap())
}

/// A table holding `values` at 1, 2, ...
fn array(vm: &mut Vm, profile: CompatibilityProfile, values: &[Value]) -> TableHandle {
    let table = vm.runtime_mut().create_table().unwrap();
    for (index, value) in values.iter().enumerate() {
        if *value != Value::Nil {
            vm.runtime_mut()
                .table_set(table, &key(profile, index + 1), value.clone())
                .unwrap();
        }
    }
    table
}

/// What the interpreter can see of a state: pc, `last_interpreted_pc`, every
/// live register (a table by the order it first appears in), and the contents
/// of those tables at the probe keys.
#[derive(Debug, PartialEq)]
struct Observed {
    pc: usize,
    last: Option<usize>,
    registers: Vec<String>,
    tables: Vec<Vec<String>>,
}

fn show(value: &Value) -> String {
    match value {
        Value::Number(number) => format!("n{:016x}", number.to_bits()),
        Value::Integer(integer) => format!("i{integer}"),
        Value::Boolean(flag) => format!("b{flag}"),
        Value::Nil => "nil".to_owned(),
        Value::String(text) => format!("s{}", String::from_utf8_lossy(text.as_bytes())),
        other => format!("other:{}", other.type_name()),
    }
}

fn observe(
    runtime: &Runtime,
    chunk: &Chunk,
    pc: usize,
    last: Option<usize>,
    registers: &[Value],
    probes: &[Value],
) -> Observed {
    let live_in = crate::jit_ir::observable_registers(chunk);
    let live = live_in.get(pc);
    let mut seen: Vec<TableHandle> = Vec::new();
    let mut shown = Vec::new();
    for (register, value) in registers.iter().enumerate() {
        let is_live = live.is_none_or(|set| {
            u8::try_from(register)
                .is_ok_and(|register| crate::jit_ir::register_set_contains(set, register))
        });
        shown.push(match value {
            _ if !is_live => "dead".to_owned(),
            Value::Table(handle) => {
                let position = seen.iter().position(|known| known == handle);
                let position = position.unwrap_or_else(|| {
                    seen.push(*handle);
                    seen.len() - 1
                });
                format!("table{position}")
            }
            other => show(other),
        });
    }
    let tables = seen
        .iter()
        .map(|handle| {
            probes
                .iter()
                .map(|probe| show(&runtime.table_get(*handle, probe).unwrap()))
                .collect()
        })
        .collect();
    Observed {
        pc,
        last,
        registers: shown,
        tables,
    }
}

/// A VM with a trace installed for a worked-case program.
struct World {
    profile: CompatibilityProfile,
    vm: Vm,
    chunk: Arc<Chunk>,
    index: u32,
    probes: Vec<Value>,
}

impl World {
    /// Runs `chunk` with `prepare` filling the registers a program does not
    /// load itself, until a trace is installed at `header`. The loop runs on
    /// positive data, so the trace is of the "then" arm.
    fn install(
        profile: CompatibilityProfile,
        chunk: Chunk,
        header: usize,
        prepare: impl FnOnce(&mut Vm, &mut [Value]),
        probes: Vec<Value>,
    ) -> Self {
        let chunk = Arc::new(chunk);
        // The exits under test include the guards of two-arm branches.
        let mut vm = guarded_vm(profile);
        let mut frames = frames_for(&vm, &chunk, None);
        {
            let mut frame = frames.last_mut().unwrap();
            prepare(&mut vm, &mut frame.registers);
        }
        for _ in 0..1_000_000 {
            if let Some(index) = vm.jit.trace.installed_index_at_pc(header) {
                return Self {
                    profile,
                    vm,
                    chunk,
                    index,
                    probes,
                };
            }
            if vm.step(&mut frames).unwrap().is_some() {
                break;
            }
        }
        panic!(
            "no trace was installed at pc {header} under {profile:?}: {:?}",
            stats_of(&vm)
        );
    }

    /// Enters the trace on a fresh frame with `registers` at the header, and
    /// returns what the run did and what it left, tables included.
    fn enter(&mut self, registers: &[Value], fuel: i64) -> (TraceRun, Observed) {
        let mut frames = frames_for(&self.vm, &self.chunk, None);
        {
            let mut frame = frames.last_mut().unwrap();
            frame.pc = HEADER;
            frame.last_interpreted_pc = Some(self.chunk.code.len() - 2).into();
            for (slot, value) in frame.registers.iter_mut().zip(registers) {
                slot.clone_from(value);
            }
        }
        let run = {
            let jit = &mut self.vm.jit;
            jit.trace.enter(
                &mut jit.statistics.trace,
                self.index,
                &mut frames,
                0,
                fuel,
                &mut self.vm.runtime,
            )
        };
        self.vm.jit.trace.forgive(self.index);
        let frame = frames.last().unwrap();
        let observed = observe(
            self.vm.runtime(),
            &self.chunk,
            frame.pc,
            frame.last_interpreted_pc.get(),
            &frame.registers,
            &self.probes,
        );
        (run, observed)
    }

    /// Whether an isolated interpreter, started from `registers` over copies
    /// of the same tables (read before the native run changed them), passes
    /// through `target`. `before` is what the tables held at entry.
    fn interpreter_reaches(
        &self,
        before: &[(TableHandle, Vec<(Value, Value)>)],
        registers: &[Value],
        target: &Observed,
    ) -> bool {
        let mut vm = interpreter_vm_with(self.profile);
        // One copy of each table, however many registers name it.
        let mut copies: Vec<(TableHandle, TableHandle)> = Vec::new();
        let mut mapped = registers.to_vec();
        for value in &mut mapped {
            let Value::Table(handle) = value else {
                continue;
            };
            let copy = match copies.iter().find(|(from, _)| from == handle) {
                Some((_, copy)) => *copy,
                None => {
                    let copy = vm.runtime_mut().create_table().unwrap();
                    let contents = before
                        .iter()
                        .find(|(table, _)| table == handle)
                        .map(|(_, contents)| contents.clone())
                        .expect("every table was recorded before the run");
                    for (key, stored) in contents {
                        vm.runtime_mut().table_set(copy, &key, stored).unwrap();
                    }
                    copies.push((*handle, copy));
                    copy
                }
            };
            *value = Value::Table(copy);
        }
        let mut frames = frames_for(&vm, &self.chunk, None);
        {
            let mut frame = frames.last_mut().unwrap();
            frame.pc = HEADER;
            frame.last_interpreted_pc = Some(self.chunk.code.len() - 2).into();
            for (slot, value) in frame.registers.iter_mut().zip(&mapped) {
                slot.clone_from(value);
            }
        }
        for _ in 0..20_000 {
            let frame = frames.last().unwrap();
            let seen = observe(
                vm.runtime(),
                &self.chunk,
                frame.pc,
                frame.last_interpreted_pc.get(),
                &frame.registers,
                &self.probes,
            );
            if &seen == target {
                return true;
            }
            match vm.step(&mut frames) {
                Ok(None) => {}
                Ok(Some(_)) | Err(_) => return false,
            }
        }
        false
    }

    /// The contents of `table` at every probe key, as stored.
    fn contents(&self, table: TableHandle) -> Vec<(Value, Value)> {
        self.probes
            .iter()
            .map(|probe| {
                (
                    probe.clone(),
                    self.vm.runtime().table_get(table, probe).unwrap(),
                )
            })
            .filter(|(_, value)| *value != Value::Nil)
            .collect()
    }
}

/// The registers of the worked case at its header, about to run iteration
/// `index` of `limit`, with the tables `t` and `u`.
fn worked_registers(
    profile: CompatibilityProfile,
    index: i64,
    limit: i64,
    t: TableHandle,
    u: TableHandle,
) -> Vec<Value> {
    let mut registers = vec![Value::Nil; 12];
    registers[0] = number(profile, index);
    registers[1] = number(profile, limit);
    registers[2] = number(profile, 1);
    registers[3] = number(profile, index);
    registers[4] = number(profile, 0);
    registers[5] = Value::Table(t);
    registers[9] = number(profile, 1);
    registers[10] = number(profile, 0);
    registers[11] = Value::Table(u);
    registers
}

fn positives(profile: CompatibilityProfile, count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| number(profile, i64::try_from(index % 7).unwrap() + 1))
        .collect()
}

fn worked_world(profile: CompatibilityProfile, aliased: bool) -> World {
    let probes: Vec<Value> = (1..=40).map(|index| key(profile, index)).collect();
    World::install(
        profile,
        worked_program(profile.is_lua53_or_later(), 600),
        HEADER,
        |vm, registers| {
            let t = array(vm, profile, &positives(profile, 700));
            let u = if aliased {
                t
            } else {
                array(vm, profile, &positives(profile, 700))
            };
            registers[5] = Value::Table(t);
            registers[11] = Value::Table(u);
        },
        probes,
    )
}

const PROFILES: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::LuaJit21,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
];

/// Runs one exit scenario: table contents, an entry index and a fuel. Enters
/// the trace, then requires the interpreter to reach the same state over a
/// copy of the same tables.
fn check_exit(
    world: &mut World,
    t_values: &[Value],
    u_values: Option<&[Value]>,
    index: i64,
    fuel: i64,
    what: &str,
) -> (TraceRun, Observed) {
    let profile = world.profile;
    let t = array(&mut world.vm, profile, t_values);
    let u = match u_values {
        Some(values) => array(&mut world.vm, profile, values),
        None => t,
    };
    let registers = worked_registers(profile, index, 50, t, u);
    let before: Vec<(TableHandle, Vec<(Value, Value)>)> = [t, u]
        .into_iter()
        .map(|table| (table, world.contents(table)))
        .collect();
    let (run, observed) = world.enter(&registers, fuel);
    assert!(
        world.interpreter_reaches(&before, &registers, &observed),
        "{profile:?} {what}: {run:?} left {observed:?}, a state the interpreter never reaches"
    );
    (run, observed)
}

#[test]
fn a_table_loop_records_one_array_binding_per_table_and_compiles() {
    for profile in PROFILES {
        for aliased in [false, true] {
            let world = worked_world(profile, aliased);
            let stats = stats_of(&world.vm);
            assert_eq!(stats.traces_compiled, 1, "{profile:?}: {stats:?}");
            let ir = world.vm.jit.trace.installed_ir(world.index).unwrap();
            // The load reads `u` (r11) and the store writes `t` (r5): two
            // registers, so two bindings, even when they hold one table.
            let registers: Vec<u16> = ir.bindings.iter().map(|b| b.register).collect();
            assert_eq!(registers.len(), 2, "{profile:?}: {registers:?}");
            assert!(registers.contains(&5) && registers.contains(&11));
            assert_eq!(ir.sides.len(), 0);
            assert_eq!(ir.effect_count(), 1, "{profile:?}");
        }
    }
}

#[test]
fn a_branch_guard_after_a_store_resumes_after_the_store() {
    for profile in PROFILES {
        for aliased in [false, true] {
            let mut world = worked_world(profile, aliased);
            let mut committed = 0;
            for negative_at in 1..=6_usize {
                let mut values = positives(profile, 12);
                values[negative_at - 1] = number(profile, -3);
                let separate = (!aliased).then_some(&values[..]);
                let (run, observed) = check_exit(
                    &mut world,
                    &values,
                    separate,
                    1,
                    10_000,
                    &format!("negative at {negative_at}, aliased {aliased}"),
                );
                let TraceRun::Ran {
                    poll, resume_pc, ..
                } = run
                else {
                    panic!("{profile:?}: the trace did not run");
                };
                assert!(!poll);
                // The branch guard fails after the store, at the else arm.
                assert_eq!(resume_pc, 14, "{profile:?}");
                committed += observed
                    .tables
                    .iter()
                    .flatten()
                    .filter(|slot| slot.starts_with('i') || slot.starts_with('n'))
                    .count();
            }
            assert!(committed > 0);
            let stats = stats_of(&world.vm);
            assert!(
                stats.exits_of(TraceExitKind::CommittedEffect) >= 6,
                "{stats:?}"
            );
        }
    }
}

#[test]
fn an_exit_after_the_store_leaves_the_store_in_the_table_and_runs_it_once() {
    for profile in PROFILES {
        let mut world = worked_world(profile, false);
        let mut values = positives(profile, 12);
        values[3] = number(profile, -5);
        let t = array(&mut world.vm, profile, &values);
        let registers = worked_registers(profile, 1, 50, t, t);
        let (run, _) = world.enter(&registers, 10_000);
        assert!(
            matches!(run, TraceRun::Ran { resume_pc: 14, .. }),
            "{run:?}"
        );
        // Slots 1..=4 hold their old value plus one: slot 4 (the negative
        // one) was stored before the guard failed, and nothing after it ran.
        let after: Vec<Value> = (1..=6)
            .map(|index| {
                world
                    .vm
                    .runtime()
                    .table_get(t, &key(profile, index))
                    .unwrap()
            })
            .collect();
        let expected: Vec<Value> = values[..6]
            .iter()
            .enumerate()
            .map(|(slot, value)| match (slot, value) {
                (0..=3, Value::Number(x)) => Value::Number(x + 1.0),
                (0..=3, Value::Integer(x)) => Value::Integer(x + 1),
                _ => value.clone(),
            })
            .collect();
        assert_eq!(after, expected, "{profile:?}");
    }
}

#[test]
fn a_budget_exit_after_a_store_is_a_state_the_interpreter_passes_through() {
    for profile in PROFILES {
        for aliased in [false, true] {
            let mut world = worked_world(profile, aliased);
            for fuel in [1_i64, 2, 3, 7] {
                let values = positives(profile, 40);
                let (run, observed) = check_exit(
                    &mut world,
                    &values,
                    None,
                    1,
                    fuel,
                    &format!("fuel {fuel}, aliased {aliased}"),
                );
                let TraceRun::Ran {
                    poll, resume_pc, ..
                } = run
                else {
                    panic!("{profile:?}: the trace did not run");
                };
                assert!(poll, "{profile:?}: fuel {fuel} ends in a poll");
                assert_eq!(resume_pc, HEADER);
                // `fuel` iterations stored `x + 1` into the slots they read.
                let stored = observed
                    .tables
                    .iter()
                    .flatten()
                    .take(usize::try_from(fuel).unwrap())
                    .count();
                assert_eq!(stored, usize::try_from(fuel).unwrap());
            }
        }
    }
}

#[test]
fn receivers_that_alias_see_each_others_stores() {
    // `u[i]` reads what the store through `t` wrote in an earlier iteration
    // when the two are one table: a trace that cached the read, or ordered
    // the load before the store, would miss it.
    for profile in PROFILES {
        let mut world = worked_world(profile, true);
        // The key 1 is stored (+1) and then read again at key 1: one table,
        // so slot 1 after the run is the only slot that changed.
        let values = positives(profile, 6);
        let (run, _) = check_exit(&mut world, &values, None, 1, 3, "aliased budget exit");
        assert!(matches!(run, TraceRun::Ran { poll: true, .. }));
        // The same loop over two different tables gives the same table
        // states for `t` because `u` is only read.
        let (run, _) = check_exit(
            &mut world,
            &values,
            Some(&positives(profile, 6)),
            1,
            3,
            "separate tables",
        );
        assert!(matches!(run, TraceRun::Ran { poll: true, .. }));
    }
}

#[test]
fn every_way_an_array_read_can_fail_leaves_before_the_load() {
    for profile in PROFILES {
        let mut world = worked_world(profile, false);
        let cases: Vec<(&str, Vec<Value>, i64)> = vec![
            ("past the end", positives(profile, 4), 4),
            (
                "a nil slot",
                {
                    let mut values = positives(profile, 6);
                    values[2] = Value::Nil;
                    values
                },
                1,
            ),
            (
                "a string slot",
                {
                    let mut values = positives(profile, 6);
                    values[2] = Value::from("7");
                    values
                },
                1,
            ),
            (
                "a boolean slot",
                {
                    let mut values = positives(profile, 6);
                    values[2] = Value::Boolean(true);
                    values
                },
                1,
            ),
            (
                "a slot of the other number type",
                {
                    let mut values = positives(profile, 6);
                    values[2] = if profile.is_lua53_or_later() {
                        Value::Number(2.5)
                    } else {
                        Value::from("2")
                    };
                    values
                },
                1,
            ),
            ("key zero", positives(profile, 6), 0),
        ];
        for (what, values, start) in cases {
            let (run, observed) = check_exit(&mut world, &values, None, start, 10_000, what);
            let TraceRun::Ran {
                poll, resume_pc, ..
            } = run
            else {
                panic!("{profile:?} {what}: the trace did not run");
            };
            assert!(!poll, "{what}");
            // The load guard leaves before the instruction, so the
            // interpreter performs the load itself.
            assert_eq!(resume_pc, HEADER, "{profile:?} {what}: {observed:?}");
        }
        let stats = stats_of(&world.vm);
        assert!(
            stats.side_exits_of(TraceGuardKind::ArrayBounds) >= 6,
            "{stats:?}"
        );
    }
}

#[test]
fn fractional_and_non_numeric_array_keys_leave_before_the_load() {
    // A float key that is not an integer, in a profile where the loop
    // variable is a float: start the loop at 1.5.
    let profile = CompatibilityProfile::Lua51;
    let mut world = worked_world(profile, false);
    let values = positives(profile, 6);
    let t = array(&mut world.vm, profile, &values);
    let mut registers = worked_registers(profile, 1, 50, t, t);
    registers[0] = Value::Number(1.5);
    registers[3] = Value::Number(1.5);
    let before = vec![(t, world.contents(t))];
    let (run, observed) = world.enter(&registers, 10_000);
    let TraceRun::Ran { resume_pc, .. } = run else {
        panic!("the trace did not run")
    };
    assert_eq!(resume_pc, HEADER);
    assert!(world.interpreter_reaches(&before, &registers, &observed));
}

#[test]
fn a_metatable_on_the_table_changes_nothing_for_present_slots_and_misses_run_index() {
    for profile in PROFILES {
        let mut world = worked_world(profile, false);
        // `__index` answers for a missing key. The trace never sees it: the
        // load guard leaves and the interpreter dispatches the metamethod.
        let mut values = positives(profile, 6);
        values[4] = Value::Nil;
        let t = array(&mut world.vm, profile, &values);
        let meta = world.vm.runtime_mut().create_table().unwrap();
        // The metatable's `__index` is a table holding a value for slot 5.
        let fallback = array(&mut world.vm, profile, &[Value::Nil; 0]);
        world
            .vm
            .runtime_mut()
            .table_set(fallback, &key(profile, 5), number(profile, 9))
            .unwrap();
        world
            .vm
            .runtime_mut()
            .table_set(meta, &Value::from("__index"), Value::Table(fallback))
            .unwrap();
        world
            .vm
            .runtime_mut()
            .raw_set_metatable(&Value::Table(t), Some(meta))
            .unwrap();
        let registers = worked_registers(profile, 1, 50, t, t);
        let (run, observed) = world.enter(&registers, 10_000);
        let TraceRun::Ran { resume_pc, .. } = run else {
            panic!("{profile:?}: the trace did not run")
        };
        // Slots 1..=4 are present: the trace ran them natively. Slot 5 is
        // missing: it left before the load for the interpreter to index.
        assert_eq!(resume_pc, HEADER, "{profile:?}");
        assert_eq!(observed.registers[3], show(&number(profile, 5)));
    }
}

#[test]
fn field_reads_and_writes_run_natively_and_leave_in_interpreter_states() {
    for profile in PROFILES {
        let probes = vec![Value::from("x"), Value::from("y")];
        let world = World::install(
            profile,
            field_program(profile.is_lua53_or_later(), 600),
            7,
            |vm, registers| {
                let table = vm.runtime_mut().create_table().unwrap();
                vm.runtime_mut()
                    .table_set(table, &Value::from("x"), number(profile, 0))
                    .unwrap();
                registers[5] = Value::Table(table);
            },
            probes,
        );
        let ir = world
            .vm
            .jit
            .trace
            .installed_ir(world.index)
            .unwrap()
            .clone();
        assert!(ir.bindings.iter().any(|binding| matches!(
            &binding.kind,
            crate::jit_ir::trace::BindingKind::Field { key, .. } if &key[..] == b"x"
        )));
    }
}

#[test]
fn a_field_that_changed_type_or_vanished_is_an_entry_mismatch_and_changes_nothing() {
    let profile = CompatibilityProfile::Lua51;
    let probes = vec![Value::from("x"), Value::from("y")];
    let mut world = World::install(
        profile,
        field_program(profile.is_lua53_or_later(), 600),
        7,
        |vm, registers| {
            let table = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .table_set(table, &Value::from("x"), number(profile, 0))
                .unwrap();
            registers[5] = Value::Table(table);
        },
        probes,
    );
    let mut registers = vec![Value::Nil; 13];
    registers[0] = number(profile, 5);
    registers[1] = number(profile, 600);
    registers[2] = number(profile, 1);
    registers[3] = number(profile, 5);
    registers[4] = number(profile, 0);
    registers[9] = number(profile, 1);
    registers[10] = number(profile, 0);
    for (what, replacement) in [
        ("a string", Some(Value::from("7"))),
        ("a boolean", Some(Value::Boolean(false))),
        ("nil", None),
        (
            "an integer where the trace holds a float",
            Some(Value::Integer(3)),
        ),
    ] {
        let table = world.vm.runtime_mut().create_table().unwrap();
        if let Some(value) = replacement
            && (profile.is_lua53_or_later() || !matches!(value, Value::Integer(_)))
        {
            world
                .vm
                .runtime_mut()
                .table_set(table, &Value::from("x"), value)
                .unwrap();
        }
        registers[5] = Value::Table(table);
        let before = observe(
            world.vm.runtime(),
            &world.chunk,
            7,
            Some(world.chunk.code.len() - 2),
            &registers,
            &world.probes,
        );
        let entries = stats_of(&world.vm).exits_of(TraceExitKind::EntryMismatch);
        let (run, after) = world.enter(&registers, 10_000);
        assert_eq!(run, TraceRun::NotEntered, "{what}");
        assert_eq!(after, before, "{what}: a mismatch changes nothing");
        assert!(
            stats_of(&world.vm).exits_of(TraceExitKind::EntryMismatch) >= entries,
            "{what}"
        );
    }
}

#[test]
fn a_length_binding_is_read_once_per_run_and_refuses_a_len_metamethod() {
    let profile = CompatibilityProfile::Lua52;
    let probes = vec![number(profile, 1)];
    let mut world = World::install(
        profile,
        length_program(profile.is_lua53_or_later(), 600),
        7,
        |vm, registers| {
            let table = array(vm, profile, &positives(profile, 5));
            registers[5] = Value::Table(table);
        },
        probes,
    );
    let ir = world
        .vm
        .jit
        .trace
        .installed_ir(world.index)
        .unwrap()
        .clone();
    assert!(ir.bindings.iter().any(|binding| matches!(
        binding.kind,
        crate::jit_ir::trace::BindingKind::Length { .. }
    )));
    let table = array(&mut world.vm, profile, &positives(profile, 9));
    let mut registers = vec![Value::Nil; 13];
    registers[0] = number(profile, 1);
    registers[1] = number(profile, 4);
    registers[2] = number(profile, 1);
    registers[3] = number(profile, 1);
    registers[4] = number(profile, 0);
    registers[5] = Value::Table(table);
    let (run, observed) = world.enter(&registers, 10_000);
    assert!(matches!(run, TraceRun::Ran { .. }), "{run:?}");
    // Four iterations each add the length 9.
    assert_eq!(observed.registers[4], show(&number(profile, 36)));
    // A `__len` metamethod makes entry refuse: the interpreter runs it.
    let meta = world.vm.runtime_mut().create_table().unwrap();
    world
        .vm
        .runtime_mut()
        .table_set(meta, &Value::from("__len"), Value::Boolean(true))
        .unwrap();
    world
        .vm
        .runtime_mut()
        .raw_set_metatable(&Value::Table(table), Some(meta))
        .unwrap();
    let (run, _) = world.enter(&registers, 10_000);
    assert_eq!(run, TraceRun::NotEntered);
}

#[test]
fn an_array_whose_first_slot_has_another_type_is_an_entry_mismatch() {
    // The trace was recorded over arrays of one element type. Over an array
    // of the other every access would leave through its guard, which costs
    // a whole exit each, so entry refuses it and another version (or the
    // interpreter) takes the loop.
    for profile in [CompatibilityProfile::Lua53, CompatibilityProfile::Lua54] {
        let mut world = worked_world(profile, false);
        let ir = world
            .vm
            .jit
            .trace
            .installed_ir(world.index)
            .unwrap()
            .clone();
        assert!(ir.bindings.iter().all(|binding| matches!(
            binding.kind,
            crate::jit_ir::trace::BindingKind::Array {
                first: Some(crate::jit_ir::trace::TraceType::Integer)
            }
        )));
        // Floats where the trace saw integers.
        let floats: Vec<Value> = (0..12)
            .map(|index| Value::Number(f64::from(index) + 1.5))
            .collect();
        let t = array(&mut world.vm, profile, &floats);
        let registers = worked_registers(profile, 1, 50, t, t);
        let before = observe(
            world.vm.runtime(),
            &world.chunk,
            HEADER,
            Some(world.chunk.code.len() - 2),
            &registers,
            &world.probes,
        );
        let (run, after) = world.enter(&registers, 10_000);
        assert_eq!(run, TraceRun::NotEntered, "{profile:?}");
        assert_eq!(after, before, "{profile:?}: a mismatch changes nothing");
        // An empty array and one with a nil first slot are refused too.
        for values in [vec![], vec![Value::Nil, number(profile, 1)]] {
            let t = array(&mut world.vm, profile, &values);
            let registers = worked_registers(profile, 1, 50, t, t);
            let (run, _) = world.enter(&registers, 10_000);
            assert_eq!(run, TraceRun::NotEntered, "{profile:?} {values:?}");
        }
        // The recorded type runs.
        let ints = positives(profile, 12);
        let t = array(&mut world.vm, profile, &ints);
        let registers = worked_registers(profile, 1, 50, t, t);
        let (run, _) = world.enter(&registers, 3);
        assert!(matches!(run, TraceRun::Ran { .. }), "{profile:?}: {run:?}");
    }
}

#[test]
fn a_table_miss_is_retried_after_an_odd_wait() {
    use crate::vm::jit::trace::TraceAbortReason;
    use crate::vm::jit::trace::policy::{AbortClass, attempt_limit, classify_abort};
    // The same even wait would start the next recording on the same
    // iteration of a loop whose length divides it. The waits keep growing over
    // the longer budget of a table miss.
    let mut last = 0;
    for attempts in 0..attempt_limit(TraceAbortReason::TableMiss) {
        let AbortClass::Retry { cooldown } = classify_abort(TraceAbortReason::TableMiss, attempts)
        else {
            panic!("a table miss is retried");
        };
        assert_eq!(cooldown % 2, 1, "attempt {attempts}");
        assert!(cooldown > last || attempts > 5, "attempt {attempts}");
        last = cooldown;
    }
}

#[test]
fn a_filling_table_gets_a_longer_retry_budget() {
    use crate::vm::jit::trace::TraceAbortReason;
    use crate::vm::jit::trace::policy::{
        MAX_RECORDING_ATTEMPTS, MAX_TABLE_MISS_ATTEMPTS, attempt_limit,
    };
    // A loop that fills an array misses on every iteration of the fill, so a
    // table miss waits longer than other aborts.
    assert_eq!(
        attempt_limit(TraceAbortReason::TableMiss),
        MAX_TABLE_MISS_ATTEMPTS
    );
    assert_eq!(
        attempt_limit(TraceAbortReason::UnsupportedOperand),
        MAX_RECORDING_ATTEMPTS
    );
}

// ------------------------------------------------- tables read out of tables

/// `r6 = u[i]; r7 = r6.x; r6.x = r7 + 1; sum = sum + r7` over an array `u`
/// (r11) of records. Header pc 7, latch pc 13.
fn records_program(integers: bool, limit: i64) -> Chunk {
    Chunk::new(
        13,
        vec![
            constant(integers, 1),
            constant(integers, limit),
            constant(integers, 0),
            Constant::String("x".into()),
            Constant::Number(0.0),
        ],
        vec![
            load(0, 0),
            load(1, 1),
            load(2, 0),
            load(4, 4),
            load(9, 0),
            load(10, 2),
            for_prep(0, 6, 13),
            Instruction::get_table(6, 11, 3),
            load(12, 3),
            Instruction::get_table(7, 6, 12),
            binary(OpCode::Add, 8, 7, 9),
            Instruction::set_table(6, 12, 8),
            binary(OpCode::Add, 4, 4, 7),
            for_loop(0, 13, 7),
            Instruction::return_(4),
        ],
    )
}

/// How the records of a scenario differ from `{x = k * 0.5}`.
#[derive(Clone, Copy, Debug)]
enum Corruption {
    None,
    /// Record 10 has no `x`.
    MissingField,
    /// Record 10 holds a boolean in `x`.
    WrongType,
    /// Slot 10 of the array holds a string.
    NotATable,
    /// Slot 5 names the table of slot 3.
    Aliased,
}

/// An array of `count` records and the records, with `corruption` applied.
fn record_array(
    vm: &mut Vm,
    profile: CompatibilityProfile,
    count: usize,
    corruption: Corruption,
) -> (TableHandle, Vec<TableHandle>) {
    let records: Vec<TableHandle> = (0..count)
        .map(|index| {
            let record = vm.runtime_mut().create_table().unwrap();
            vm.runtime_mut()
                .table_set(record, &Value::from("x"), Value::Number(index as f64 * 0.5))
                .unwrap();
            record
        })
        .collect();
    match corruption {
        Corruption::MissingField => vm
            .runtime_mut()
            .table_set(records[9], &Value::from("x"), Value::Nil)
            .unwrap(),
        Corruption::WrongType => vm
            .runtime_mut()
            .table_set(records[9], &Value::from("x"), Value::Boolean(true))
            .unwrap(),
        _ => {}
    }
    let values: Vec<Value> = records
        .iter()
        .enumerate()
        .map(|(index, record)| match corruption {
            Corruption::NotATable if index == 9 => Value::from("text"),
            Corruption::Aliased if index == 4 => Value::Table(records[2]),
            _ => Value::Table(*record),
        })
        .collect();
    let table = array(vm, profile, &values);
    // The records outlive the frame that runs over them: a collection after
    // the program returns must not free what the test reads back.
    let environment = vm.runtime().global_environment();
    vm.runtime_mut()
        .table_set(environment, &Value::from("records"), Value::Table(table))
        .unwrap();
    (table, records)
}

/// Runs the program from its header to the end, starting from `registers`.
/// `native` enters the trace of `world` first, with `fuel`. Returns the
/// result and what the records hold afterwards.
fn finish_records(
    vm: &mut Vm,
    chunk: &Arc<Chunk>,
    trace: Option<(u32, i64)>,
    registers: &[Value],
    records: &[TableHandle],
) -> (String, Vec<String>) {
    let mut frames = frames_for(vm, chunk, None);
    {
        let mut frame = frames.last_mut().unwrap();
        frame.pc = HEADER;
        frame.last_interpreted_pc = Some(chunk.code.len() - 2).into();
        for (slot, value) in frame.registers.iter_mut().zip(registers) {
            slot.clone_from(value);
        }
    }
    if let Some((index, fuel)) = trace {
        let run = {
            let jit = &mut vm.jit;
            jit.trace.enter(
                &mut jit.statistics.trace,
                index,
                &mut frames,
                0,
                fuel,
                &mut vm.runtime,
            )
        };
        assert!(matches!(run, TraceRun::Ran { .. }), "{run:?}");
    }
    let result = loop {
        match vm.step(&mut frames) {
            Ok(None) => {}
            Ok(Some(values)) => break format!("{values:?}"),
            Err(error) => break format!("error: {error}"),
        }
    };
    let fields = records
        .iter()
        .map(|record| show(&vm.runtime().table_get(*record, &Value::from("x")).unwrap()))
        .collect();
    (result, fields)
}

#[test]
fn a_loop_over_records_leaves_in_states_the_interpreter_agrees_with() {
    for profile in PROFILES {
        let integers = profile.is_lua53_or_later();
        let mut world = World::install(
            profile,
            records_program(integers, 600),
            HEADER,
            |vm, registers| {
                let (u, _) = record_array(vm, profile, 700, Corruption::None);
                registers[11] = Value::Table(u);
            },
            Vec::new(),
        );
        let ir = world
            .vm
            .jit
            .trace
            .installed_ir(world.index)
            .unwrap()
            .clone();
        // The array read hands out tables, and the fields are read by lane.
        assert_eq!(ir.fields.len(), 1, "{profile:?}");
        assert_eq!(&ir.fields[0].key[..], b"x");
        assert!(ir.body.iter().any(|instruction| matches!(
            instruction.op,
            crate::jit_ir::trace::TraceOp::ArrayLoad {
                expected: crate::jit_ir::trace::TraceType::Table,
                ..
            }
        )));
        assert!(ir.body.iter().any(|instruction| matches!(
            instruction.op,
            crate::jit_ir::trace::TraceOp::TableFieldStore { .. }
        )));
        for (corruption, count, limit, fuel) in [
            (Corruption::None, 700, 50, 5),
            (Corruption::None, 700, 50, 10_000),
            (Corruption::None, 700, 690, 10_000),
            (Corruption::MissingField, 700, 50, 10_000),
            (Corruption::WrongType, 700, 50, 10_000),
            (Corruption::NotATable, 700, 50, 10_000),
            (Corruption::Aliased, 700, 50, 10_000),
            (Corruption::None, 30, 40, 10_000),
        ] {
            let build = |vm: &mut Vm| {
                let (u, records) = record_array(vm, profile, count, corruption);
                let mut registers = vec![Value::Nil; 13];
                registers[0] = number(profile, 1);
                registers[1] = number(profile, limit);
                registers[2] = number(profile, 1);
                registers[3] = number(profile, 1);
                registers[4] = Value::Number(0.0);
                registers[9] = number(profile, 1);
                registers[10] = number(profile, 0);
                registers[11] = Value::Table(u);
                (registers, records)
            };
            let (registers, records) = build(&mut world.vm);
            let native = finish_records(
                &mut world.vm,
                &world.chunk,
                Some((world.index, fuel)),
                &registers,
                &records,
            );
            let mut reference = interpreter_vm_with(profile);
            let (registers, records) = build(&mut reference);
            let expected = finish_records(&mut reference, &world.chunk, None, &registers, &records);
            assert_eq!(
                native, expected,
                "{profile:?} {corruption:?} {count} records, limit {limit}, fuel {fuel}"
            );
        }
    }
}

#[test]
fn a_body_that_reads_and_writes_many_fields_needs_fewer_iterations_per_entry() {
    use crate::vm::jit::trace::native::{TRACE_MIN_ITERATIONS_PER_ENTRY, min_iterations_of};
    // A loop over records does enough work per iteration that a few of them
    // pay for an entry and exit.
    let profile = CompatibilityProfile::Lua51;
    let world = World::install(
        profile,
        records_program(false, 600),
        HEADER,
        |vm, registers| {
            let (u, _) = record_array(vm, profile, 700, Corruption::None);
            registers[11] = Value::Table(u);
        },
        Vec::new(),
    );
    let ir = world.vm.jit.trace.installed_ir(world.index).unwrap();
    let few = min_iterations_of(ir);
    assert!(few <= TRACE_MIN_ITERATIONS_PER_ENTRY, "{few}");
    // The same loop with ten more field accesses.
    let mut heavy = ir.clone();
    let loads: Vec<_> = heavy
        .body
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.op,
                crate::jit_ir::trace::TraceOp::TableFieldStore { .. }
            )
        })
        .cloned()
        .collect();
    for _ in 0..12 {
        heavy.body.extend(loads.iter().cloned());
    }
    assert!(min_iterations_of(&heavy) < TRACE_MIN_ITERATIONS_PER_ENTRY);
    assert!(min_iterations_of(&heavy) >= 1);
    // A body of plain arithmetic keeps the bar it always had.
    {
        let (name, chunk, header) = ("sum", sum_program(300.0), 5);
        let installed = Installed::with_profile(profile, chunk, header);
        let ir = installed
            .vm
            .jit
            .trace
            .installed_ir(installed.index)
            .unwrap();
        assert_eq!(
            min_iterations_of(ir),
            TRACE_MIN_ITERATIONS_PER_ENTRY,
            "{name}"
        );
    }
}
