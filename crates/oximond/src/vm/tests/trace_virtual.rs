//! Virtual tables at trace exits, from hand-built IR.
//!
//! No recorder makes a virtual table yet, so these cases build the IR by hand:
//! a trace that leaves through an exit whose snapshot has `Virtual` entries
//! and recipes. Each case runs the compiled trace for real, then restores the
//! exit three ways and compares them: the symbolic reference evaluator, the
//! generic restore of `deopt` and the plan the native policy applies
//! (`ExitRestore`). The tables they build must hold the same contents, and two
//! registers that name one id must hold one table.

#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]

use super::trace_record::{for_loop, for_prep, load, only_trace, record};
use super::*;
use crate::jit::{TraceControl, TraceExitClass, TraceOutcome, compile_trace};
use crate::jit_ir::trace::reference::{Outcome, RestoredTable, evaluate};
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, HelperId, LoopParamInit, MAX_VIRTUAL_SHAPE_KEYS, RegisterEntry,
    Snapshot, SnapshotFrame, SnapshotPhase, TraceBuilder, TraceFrameInfo, TraceId, TraceIr,
    TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType, VirtualAlloc, VirtualId,
    VirtualKey, VirtualTable,
};
use crate::runtime::{MAX_SHAPE_KEYS, TableHandle};
use crate::vm::jit::trace::deopt::{materialize_virtuals, restore_window, scalar_to_value};
use crate::vm::jit::trace::native::{ExitRestore, ExitView, input_slot, restores_of, side_keys_of};
use crate::vm::stack::CallStack;

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn vm_for(profile: CompatibilityProfile) -> Vm {
    Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter)
}

/// A number of the profile, as the recorder types one.
fn number(profile: CompatibilityProfile, value: i64) -> TraceScalar {
    if profile.is_lua53_or_later() {
        TraceScalar::Integer(value)
    } else {
        TraceScalar::Float(value as f64)
    }
}

fn number_type(profile: CompatibilityProfile) -> TraceType {
    number(profile, 0).ty()
}

fn name(text: &str) -> std::sync::Arc<[u8]> {
    std::sync::Arc::from(text.as_bytes())
}

fn nil() -> RegisterEntry {
    RegisterEntry::Const(TraceScalar::Nil)
}

const REGISTERS: u16 = 8;

/// The trace of one case and the entry values it runs on.
struct Case {
    ir: TraceIr,
    /// Registers at entry, as the VM holds them.
    entry: Vec<Value>,
    table: TableHandle,
}

/// A trace over `r0 = a`, `r1 = b`, `r2` = a real table, `r3` = a boolean and,
/// under `LuaJIT`, `r4`/`r5` = an `int64_t` and a `uint64_t`. It computes
/// `sum = a + b` and leaves through an exit that restores eight registers:
///
/// ```text
/// r0 a   r1 V0   r2 V0   r3 V1   r4 V2   r5 t   r6 nil   r7 sum
/// V0 plain:  [a, sum, hole, 40, t]  100 -> flag, 0.5 -> b, "name" -> root 0
/// V1 shaped: x -> a, y -> b, z -> nil
/// V2 plain:  [i64, u64] (LuaJIT)  -3 -> sum
/// V3 plain:  [a], named by no register
/// ```
fn case(vm: &mut Vm, profile: CompatibilityProfile, phantom: bool) -> Case {
    use RegisterEntry::{Const, Root, Value as Val, Virtual};
    let wide = profile == CompatibilityProfile::LuaJit21;
    let ty = number_type(profile);
    let mut b = TraceBuilder::new(
        TraceId(5),
        TraceFrameInfo {
            register_count: REGISTERS,
            code_len: 4,
            inlined: None,
            activation: None,
        },
    );
    for register in 1..REGISTERS {
        b.mark_written(FrameId(0), register);
    }
    let a = b.input(0, ty);
    let bb = b.input(1, ty);
    let t = b.input(2, TraceType::Table);
    let flag = b.input(3, TraceType::Boolean);
    let wide_values = wide.then(|| (b.input(4, TraceType::Int64), b.input(5, TraceType::UInt64)));
    let sum = b
        .emit(
            site(1),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: a,
                right: bb,
            },
            Some(ty),
        )
        .unwrap();
    let forty = number(profile, 40);
    let v0 = VirtualTable {
        alloc: VirtualAlloc::Plain,
        array: vec![Val(a), Val(sum), nil(), Const(forty), Val(t)],
        fields: vec![
            (VirtualKey::Integer(100), Val(flag)),
            (VirtualKey::Float(0.5), Val(bb)),
            (VirtualKey::String(name("name")), Root(0)),
        ],
        phantom_entries: phantom.then_some(8),
        metatable: None,
    };
    let v1 = VirtualTable {
        alloc: VirtualAlloc::Shaped {
            keys: vec![name("x"), name("y"), name("z")],
        },
        array: Vec::new(),
        fields: vec![
            (VirtualKey::String(name("x")), Val(a)),
            (VirtualKey::String(name("y")), Val(bb)),
            (VirtualKey::String(name("z")), nil()),
        ],
        phantom_entries: phantom.then_some(3),
        metatable: None,
    };
    let v2 = VirtualTable {
        alloc: VirtualAlloc::Plain,
        array: wide_values.map_or_else(Vec::new, |(i, u)| vec![Val(i), Val(u)]),
        fields: vec![(VirtualKey::Integer(-3), Val(sum))],
        phantom_entries: None,
        metatable: None,
    };
    let v3 = VirtualTable {
        alloc: VirtualAlloc::Plain,
        array: vec![Val(a)],
        fields: Vec::new(),
        phantom_entries: phantom.then_some(1),
        // The exit sets the table the trace was entered with.
        metatable: Some(Val(t)),
    };
    let exit = b.snapshot(
        Snapshot::new(
            SnapshotPhase::After {
                pc: 1,
                successor: 2,
            },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(1),
                registers: vec![
                    Val(a),
                    Virtual(VirtualId(0)),
                    Virtual(VirtualId(0)),
                    Virtual(VirtualId(1)),
                    Virtual(VirtualId(2)),
                    Val(t),
                    nil(),
                    Val(sum),
                ]
                .into(),
            }],
        )
        .with_virtuals(vec![v0, v1, v2, v3]),
    );
    let mut ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.root_slot_count = 1;
    ir.verify().unwrap();
    let table = vm.runtime_mut().create_table().unwrap();
    let value = |v: i64| match number(profile, v) {
        TraceScalar::Integer(i) => Value::Integer(i),
        TraceScalar::Float(f) => Value::Number(f),
        _ => unreachable!(),
    };
    let mut entry = vec![
        value(5),
        value(9),
        Value::Table(table),
        Value::Boolean(true),
        Value::Nil,
        Value::Nil,
        Value::Nil,
        Value::Nil,
    ];
    if wide {
        entry[4] = Value::Int64(-77);
        entry[5] = Value::UInt64(u64::MAX - 2);
    }
    Case { ir, entry, table }
}

fn chunk() -> Arc<Chunk> {
    Arc::new(Chunk::new(
        u8::try_from(REGISTERS).unwrap(),
        Vec::new(),
        vec![Instruction::return_(0); 4],
    ))
}

fn frames_of(vm: &Vm, chunk: &Arc<Chunk>) -> CallStack {
    CallStack::from(vec![Frame::new(
        Arc::clone(chunk),
        Vec::new(),
        Vec::new(),
        None,
        None,
        vm.runtime().global_environment(),
        None,
    )])
}

/// The scalar a value had in the reference evaluator.
fn scalar_of(value: &Value, bank: &[Value]) -> TraceScalar {
    match value {
        Value::Nil => TraceScalar::Nil,
        Value::Boolean(flag) => TraceScalar::Boolean(*flag),
        Value::Integer(integer) => TraceScalar::Integer(*integer),
        Value::Number(float) => TraceScalar::Float(*float),
        Value::Int64(wide) => TraceScalar::Int64(*wide),
        Value::UInt64(wide) => TraceScalar::UInt64(*wide),
        Value::Table(_) => TraceScalar::Table(
            bank.iter()
                .position(|held| held == value)
                .expect("the table is in the bank") as u64,
        ),
        other => panic!("no scalar for {other:?}"),
    }
}

/// The VM value a reference scalar stands for.
fn value_of(scalar: TraceScalar, bank: &[Value], roots: &[Value]) -> Value {
    match scalar {
        TraceScalar::Table(lane) => bank[lane as usize].clone(),
        TraceScalar::String(slot) => roots[slot as usize].clone(),
        scalar => scalar_to_value(scalar),
    }
}

fn key_value(profile: CompatibilityProfile, key: &VirtualKey) -> Value {
    match key {
        VirtualKey::Integer(index) if profile.is_lua53_or_later() => Value::Integer(*index),
        VirtualKey::Integer(index) => Value::Number(*index as f64),
        VirtualKey::Float(number) => Value::Number(*number),
        VirtualKey::String(bytes) => Value::from(String::from_utf8(bytes.to_vec()).unwrap()),
    }
}

/// Checks that `handle` holds exactly what `expected` says, no more.
fn check_table(
    vm: &Vm,
    profile: CompatibilityProfile,
    handle: TableHandle,
    expected: &RestoredTable,
    bank: &[Value],
    roots: &[Value],
    context: &str,
) {
    let runtime = vm.runtime();
    let wanted = expected
        .metatable
        .map(|scalar| match value_of(scalar, bank, roots) {
            Value::Table(metatable) => metatable,
            other => panic!("{context}: a metatable is a table, not {other:?}"),
        });
    assert_eq!(
        runtime.raw_metatable(&Value::Table(handle)).unwrap(),
        wanted,
        "{context}: metatable"
    );
    let mut stored = 0;
    for (position, scalar) in expected.array.iter().enumerate() {
        let key = key_value(
            profile,
            &VirtualKey::Integer(i64::try_from(position + 1).unwrap()),
        );
        let want = value_of(*scalar, bank, roots);
        assert_eq!(
            runtime.table_get(handle, &key).unwrap(),
            want,
            "{context}: array position {}",
            position + 1
        );
        stored += usize::from(want != Value::Nil);
    }
    for (key, scalar) in &expected.fields {
        let want = value_of(*scalar, bank, roots);
        assert_eq!(
            runtime.table_get(handle, &key_value(profile, key)).unwrap(),
            want,
            "{context}: field {key:?}"
        );
        stored += usize::from(want != Value::Nil);
    }
    // Nothing else is in the table: a nil stored nothing, and a hole is a hole.
    let mut seen = 0;
    let mut previous = Value::Nil;
    loop {
        let (key, _) = runtime.table_next(handle, &previous).unwrap();
        if key == Value::Nil {
            break;
        }
        seen += 1;
        previous = key;
    }
    assert_eq!(seen, stored, "{context}: entries stored");
}

/// One run of a case at `tier`: the exit area, the bank and the descriptor.
struct Exited {
    compiled: crate::jit::CompiledTrace,
    area: Vec<i64>,
    bank: Vec<Value>,
    exit: usize,
}

fn run(case: &Case, tier: OptimizationTier) -> Exited {
    let compiled = compile_trace(&case.ir, tier).unwrap();
    let mut bank = Vec::new();
    let inputs: Vec<_> = compiled
        .entry_contract()
        .iter()
        .map(|(register, _)| input_slot(&case.entry[usize::from(*register)], &mut bank))
        .collect();
    let mut area = vec![0_i64; compiled.exit_area_len()];
    let mut control = TraceControl::new(100);
    let TraceOutcome::Exit { index } = compiled.run(&inputs, &mut control, &mut area) else {
        panic!("the trace exits")
    };
    assert_eq!(compiled.exits()[index].class, TraceExitClass::LoopComplete);
    Exited {
        compiled,
        area,
        bank,
        exit: index,
    }
}

fn reference(
    case: &Case,
    bank: &[Value],
    roots: &[TraceScalar],
) -> crate::jit_ir::trace::reference::Restored {
    let registers: Vec<TraceScalar> = case
        .entry
        .iter()
        .map(|value| scalar_of(value, bank))
        .collect();
    match evaluate(&case.ir, &registers, roots, 4, |_, _| None).outcome {
        Outcome::Exited(restored) => restored,
        other => panic!("the reference exits: {other:?}"),
    }
}

#[test]
fn the_shape_limit_is_the_runtimes() {
    assert_eq!(MAX_VIRTUAL_SHAPE_KEYS, MAX_SHAPE_KEYS);
}

#[test]
fn an_exit_materializes_the_same_tables_three_ways() {
    for profile in PROFILES {
        for tier in TIERS {
            let context = format!("{profile:?} {tier:?}");
            let mut vm = vm_for(profile);
            let case = case(&mut vm, profile, false);
            let exited = run(&case, tier);
            let roots = [Value::from("hello")];
            let root_scalars = [TraceScalar::String(0)];
            let expected = reference(&case, &exited.bank, &root_scalars);
            // Only the tables a register names exist.
            assert_eq!(
                expected
                    .tables
                    .iter()
                    .map(|(id, _)| id.0)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2]
            );
            let chunk = chunk();

            // The native plan.
            let restores = restores_of(&exited.compiled, usize::from(REGISTERS), &[]).unwrap();
            let restore: &ExitRestore = &restores[exited.exit];
            assert_eq!(restore.virtuals.len(), 3);
            let shaped: Vec<bool> = restore
                .virtuals
                .iter()
                .map(|plan| plan.shape.is_some())
                .collect();
            assert_eq!(shaped, vec![false, true, false], "{context}");
            let before = vm.runtime().tables_created();
            let mut planned = frames_of(&vm, &chunk);
            let tables = restore
                .materialize(&exited.area, &exited.bank, &roots, vm.runtime_mut())
                .unwrap();
            assert_eq!(
                vm.runtime().tables_created() - before,
                3,
                "{context}: one allocation per named table"
            );
            restore.apply(
                &mut planned.at_mut(0),
                &exited.area,
                &exited.bank,
                &roots,
                &tables,
                vm.runtime_mut(),
            );

            // The generic restore.
            let lowered = exited.compiled.lowered_member(0);
            let descriptor = &exited.compiled.exits()[exited.exit];
            let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
            let view = ExitView {
                descriptor,
                area: &exited.area,
            };
            let virtuals =
                materialize_virtuals(vm.runtime_mut(), snapshot, &view, &exited.bank, &roots)
                    .unwrap();
            let mut generic = frames_of(&vm, &chunk);
            restore_window(
                &mut generic.at_mut(0),
                snapshot,
                &view,
                &exited.bank,
                &roots,
                &[],
                &virtuals,
            )
            .unwrap();

            for (label, frames) in [("native", &planned), ("generic", &generic)] {
                let registers = &frames.at(0).registers;
                let context = format!("{context} {label}");
                // Plain registers are what the reference says.
                for register in [0, 5, 6, 7] {
                    assert_eq!(
                        scalar_of(&registers[register], &exited.bank),
                        expected.registers[register],
                        "{context}: r{register}"
                    );
                }
                // Two registers, one table: rawequal. Others differ.
                let handle = |register: usize| match registers[register] {
                    Value::Table(handle) => handle,
                    ref other => panic!("{context}: r{register} is {other:?}"),
                };
                assert_eq!(registers[1], registers[2], "{context}: aliases");
                assert_ne!(registers[1], registers[3], "{context}");
                assert_ne!(registers[1], registers[4], "{context}");
                assert_ne!(registers[3], registers[4], "{context}");
                assert_ne!(registers[1], registers[5], "the input table stays itself");
                assert_eq!(registers[5], Value::Table(case.table));
                for (id, table) in &expected.tables {
                    let register = match id.0 {
                        0 => 1,
                        1 => 3,
                        _ => 4,
                    };
                    check_table(
                        &vm,
                        profile,
                        handle(register),
                        table,
                        &exited.bank,
                        &roots,
                        &format!("{context}: V{}", id.0),
                    );
                }
                // Sanity on the first table, spelled out.
                let v0 = handle(1);
                let get = |key: Value| vm.runtime().table_get(v0, &key).unwrap();
                assert_eq!(get(key_value(profile, &VirtualKey::Integer(3))), Value::Nil);
                assert_eq!(get(Value::from("name")), Value::from("hello"));
                assert_eq!(
                    get(Value::Number(0.5)),
                    scalar_to_value(case_number(profile, 9))
                );
                assert_eq!(
                    get(key_value(profile, &VirtualKey::Integer(5))),
                    Value::Table(case.table)
                );
                if profile == CompatibilityProfile::LuaJit21 {
                    let v2 = handle(4);
                    let wide = |index: i64| {
                        vm.runtime()
                            .table_get(v2, &key_value(profile, &VirtualKey::Integer(index)))
                            .unwrap()
                    };
                    assert_eq!(wide(1), Value::Int64(-77));
                    assert_eq!(wide(2), Value::UInt64(u64::MAX - 2));
                }
            }
        }
    }
}

/// A trace that leaves through an exit whose register `r1` names V0, a table
/// that holds V1 twice (at array position 2 and under the key `inner`), and
/// `r2` names V1 itself:
///
/// ```text
/// r0 a   r1 V0   r2 V1
/// V0 plain:  [a, V1]  inner -> V1
/// V1 plain:  [sum]
/// ```
fn nested_case(vm: &mut Vm, profile: CompatibilityProfile, phantom: bool) -> Case {
    use RegisterEntry::{Value as Val, Virtual};
    let ty = number_type(profile);
    let mut b = TraceBuilder::new(
        TraceId(6),
        TraceFrameInfo {
            register_count: REGISTERS,
            code_len: 4,
            inlined: None,
            activation: None,
        },
    );
    for register in 1..REGISTERS {
        b.mark_written(FrameId(0), register);
    }
    let a = b.input(0, ty);
    let bb = b.input(1, ty);
    let t = b.input(2, TraceType::Table);
    let sum = b
        .emit(
            site(1),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: a,
                right: bb,
            },
            Some(ty),
        )
        .unwrap();
    let outer = VirtualTable {
        alloc: VirtualAlloc::Plain,
        array: vec![Val(a), Virtual(VirtualId(1))],
        fields: vec![(VirtualKey::String(name("inner")), Virtual(VirtualId(1)))],
        phantom_entries: phantom.then_some(3),
        metatable: None,
    };
    let inner = VirtualTable {
        alloc: VirtualAlloc::Plain,
        array: vec![Val(sum)],
        fields: Vec::new(),
        phantom_entries: phantom.then_some(1),
        metatable: None,
    };
    let exit = b.snapshot(
        Snapshot::new(
            SnapshotPhase::After {
                pc: 1,
                successor: 2,
            },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(1),
                registers: vec![
                    Val(a),
                    Virtual(VirtualId(0)),
                    Virtual(VirtualId(1)),
                    nil(),
                    nil(),
                    Val(t),
                    nil(),
                    nil(),
                ]
                .into(),
            }],
        )
        .with_virtuals(vec![outer, inner]),
    );
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    ir.verify().unwrap();
    let table = vm.runtime_mut().create_table().unwrap();
    let value = |v: i64| match number(profile, v) {
        TraceScalar::Integer(i) => Value::Integer(i),
        TraceScalar::Float(f) => Value::Number(f),
        _ => unreachable!(),
    };
    let mut entry = vec![value(5), value(9), Value::Table(table)];
    entry.resize(usize::from(REGISTERS), Value::Nil);
    Case { ir, entry, table }
}

#[test]
fn a_table_inside_a_table_is_built_before_either_is_filled_three_ways() {
    for profile in PROFILES {
        for tier in TIERS {
            let context = format!("{profile:?} {tier:?}");
            let mut vm = vm_for(profile);
            let case = nested_case(&mut vm, profile, false);
            let exited = run(&case, tier);
            let expected = reference(&case, &exited.bank, &[]);
            // The inner table is built because the outer one holds it, and
            // the register that names it holds the same one.
            assert_eq!(
                expected
                    .tables
                    .iter()
                    .map(|(id, _)| id.0)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            let chunk = chunk();
            let restores = restores_of(&exited.compiled, usize::from(REGISTERS), &[]).unwrap();
            let restore: &ExitRestore = &restores[exited.exit];
            assert_eq!(restore.virtuals.len(), 2, "{context}");

            let before = vm.runtime().tables_created();
            let mut planned = frames_of(&vm, &chunk);
            let tables = restore
                .materialize(&exited.area, &exited.bank, &[], vm.runtime_mut())
                .unwrap();
            assert_eq!(vm.runtime().tables_created() - before, 2, "{context}");
            restore.apply(
                &mut planned.at_mut(0),
                &exited.area,
                &exited.bank,
                &[],
                &tables,
                vm.runtime_mut(),
            );

            let lowered = exited.compiled.lowered_member(0);
            let descriptor = &exited.compiled.exits()[exited.exit];
            let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
            let view = ExitView {
                descriptor,
                area: &exited.area,
            };
            let virtuals =
                materialize_virtuals(vm.runtime_mut(), snapshot, &view, &exited.bank, &[]).unwrap();
            let mut generic = frames_of(&vm, &chunk);
            restore_window(
                &mut generic.at_mut(0),
                snapshot,
                &view,
                &exited.bank,
                &[],
                &[],
                &virtuals,
            )
            .unwrap();

            for (label, frames) in [("native", &planned), ("generic", &generic)] {
                let context = format!("{context} {label}");
                let registers = &frames.at(0).registers;
                let handle = |register: usize| match registers[register] {
                    Value::Table(handle) => handle,
                    ref other => panic!("{context}: r{register} is {other:?}"),
                };
                let (outer, inner) = (handle(1), handle(2));
                assert_ne!(outer, inner, "{context}");
                let get = |table: TableHandle, key: Value| vm.runtime().table_get(table, &key);
                // Both slots of the outer table hold the one inner table.
                assert_eq!(
                    get(outer, key_value(profile, &VirtualKey::Integer(2))).unwrap(),
                    Value::Table(inner),
                    "{context}"
                );
                assert_eq!(
                    get(outer, Value::from("inner")).unwrap(),
                    Value::Table(inner),
                    "{context}"
                );
                assert_eq!(
                    get(inner, key_value(profile, &VirtualKey::Integer(1))).unwrap(),
                    scalar_to_value(case_number(profile, 14)),
                    "{context}"
                );
            }
        }
    }
}

fn case_number(profile: CompatibilityProfile, value: i64) -> TraceScalar {
    number(profile, value)
}

#[test]
fn the_length_of_a_materialized_table_is_the_border_of_its_array() {
    for profile in PROFILES {
        let mut vm = vm_for(profile);
        let case = case(&mut vm, profile, false);
        let exited = run(&case, OptimizationTier::Optimized);
        let roots = [Value::from("hello")];
        let restores = restores_of(&exited.compiled, usize::from(REGISTERS), &[]).unwrap();
        let restore = &restores[exited.exit];
        let tables = restore
            .materialize(&exited.area, &exited.bank, &roots, vm.runtime_mut())
            .unwrap();
        // V0 is [a, sum, hole, 40, t]: a table built by hand with the same
        // stores has the same length, whichever border that is.
        let twin = vm.runtime_mut().create_table().unwrap();
        for (position, value) in [Some(0), Some(1), None, Some(3), Some(4)]
            .into_iter()
            .enumerate()
        {
            if value.is_some() {
                let key = key_value(
                    profile,
                    &VirtualKey::Integer(i64::try_from(position + 1).unwrap()),
                );
                vm.runtime_mut()
                    .table_set(twin, &key, Value::Boolean(true))
                    .unwrap();
            }
        }
        let length = |table| vm.runtime().table_length(table).unwrap();
        assert_eq!(length(tables[0]), length(twin), "{profile:?}");
        // The second array (V2) has two slots under LuaJIT and none elsewhere.
        let wide = if profile == CompatibilityProfile::LuaJit21 {
            2.0
        } else {
            0.0
        };
        assert_eq!(length(tables[2]), wide, "{profile:?}");
        // V1 has no array part.
        assert_eq!(length(tables[1]), 0.0);
    }
}

#[test]
fn materializing_a_table_the_trace_charged_as_phantom_debt_counts_it_once() {
    for phantom in [false, true] {
        let mut vm = vm_for(CompatibilityProfile::Lua54);
        let case = case(&mut vm, CompatibilityProfile::Lua54, phantom);
        let exited = run(&case, OptimizationTier::Optimized);
        let roots = [Value::from("hello")];
        let restores = restores_of(&exited.compiled, usize::from(REGISTERS), &[]).unwrap();
        let restore = &restores[exited.exit];
        // The trace charged the three tables that carry the flag, at their
        // recipe footprints (V0: 5 + 3, V1: 3, V3 is never named).
        if phantom {
            vm.runtime_mut().charge_elided_table(8);
            vm.runtime_mut().charge_elided_table(3);
        }
        let charged = vm.runtime().elided_tables_total();
        assert_eq!(charged.0, if phantom { 2 } else { 0 });
        let pending = vm.runtime().elided_table_bytes();
        restore
            .materialize(&exited.area, &exited.bank, &roots, vm.runtime_mut())
            .unwrap();
        // V0 and V1 gave their phantom charge back, V2 never had one, and
        // V3 (named by no register) is not built, so its charge stays: a
        // trace charges only constructors it ran, and it never charged V3.
        assert_eq!(vm.runtime().elided_tables_total().0, 0, "phantom {phantom}");
        assert_eq!(vm.runtime().elided_table_bytes(), 0, "phantom {phantom}");
        assert_eq!(pending > 0, phantom);
    }
}

#[test]
fn a_failed_materialization_leaves_the_frame_alone_and_reports() {
    let profile = CompatibilityProfile::Lua54;
    let mut vm = vm_for(profile);
    let case = case(&mut vm, profile, false);
    let exited = run(&case, OptimizationTier::Optimized);
    let restores = restores_of(&exited.compiled, usize::from(REGISTERS), &[]).unwrap();

    // A snapshot a verifier would refuse: a NaN key. The plan builds, and
    // the store fails after the allocations.
    let mut broken = case.ir.clone();
    broken.snapshots[0].virtuals[0].fields[1].0 = VirtualKey::Float(f64::NAN);
    let descriptor = &exited.compiled.exits()[exited.exit];
    let plan = ExitRestore::new(
        &broken,
        &broken.snapshots[0],
        descriptor,
        usize::from(REGISTERS),
        &[],
    )
    .expect("the plan does not look at keys");
    let roots = [Value::from("hello")];
    let chunk = chunk();
    let mut frames = frames_of(&vm, &chunk);
    for (index, register) in frames.at_mut(0).registers.iter_mut().enumerate() {
        *register = Value::Integer(index as i64 + 1000);
    }
    let before = frames.at(0).registers.to_vec();
    let error = plan
        .materialize(&exited.area, &exited.bank, &roots, vm.runtime_mut())
        .expect_err("a NaN key cannot be stored");
    assert!(error.to_string().contains("NaN"), "{error}");
    // `apply` never ran: the registers are as they were.
    assert_eq!(frames.at(0).registers.to_vec(), before);
    // The generic path says the same thing as a DeoptError, writing nothing.
    let lowered = exited.compiled.lowered_member(0);
    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
    let mut snapshot = snapshot.clone();
    snapshot.virtuals[0].fields[1].0 = VirtualKey::Float(f64::NAN);
    let view = ExitView {
        descriptor,
        area: &exited.area,
    };
    let result = materialize_virtuals(vm.runtime_mut(), &snapshot, &view, &exited.bank, &roots);
    assert!(result.is_err());
    // The exit that would have worked is untouched by all of it.
    let tables = restores[exited.exit]
        .materialize(&exited.area, &exited.bank, &roots, vm.runtime_mut())
        .unwrap();
    assert_eq!(tables.len(), 3);

    // A value the exit cannot name fails before any allocation.
    let created = vm.runtime().tables_created();
    let missing = materialize_virtuals(vm.runtime_mut(), &snapshot, &view, &exited.bank, &[]);
    assert!(missing.is_err());
    assert_eq!(
        vm.runtime().tables_created(),
        created,
        "nothing was allocated"
    );
}

#[test]
fn a_trace_without_virtual_entries_plans_no_tables() {
    use RegisterEntry::Value as Val;
    let mut b = TraceBuilder::new(
        TraceId(6),
        TraceFrameInfo {
            register_count: 2,
            code_len: 4,
            inlined: None,
            activation: None,
        },
    );
    b.mark_written(FrameId(0), 1);
    let x = b.input(0, TraceType::Float);
    let exit = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 1,
            successor: 2,
        },
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(1),
            registers: vec![Val(x), Val(x)].into(),
        }],
    ));
    let ir = b.finish(TraceTerminator::Exit { snapshot: exit });
    let compiled = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
    let restores = restores_of(&compiled, 2, &[]).unwrap();
    assert!(restores.iter().all(|restore| restore.virtuals.is_empty()));
}

/// A loop `no += i` whose first guard (`i % 7 < 3` expected false) restores
/// r3: as the comparison when `tables` is zero, and otherwise as a virtual
/// table at the head of a chain of `tables`, each holding the next and the
/// last holding `no`.
// The IR-builder test uses the builder's conventional one-letter names.
#[allow(clippy::many_single_char_names)]
fn loop_with_guard(tables: usize) -> TraceIr {
    use RegisterEntry::{Unchanged, Value as Val, Virtual};
    let f = TraceScalar::Float;
    let snap = |phase, last, registers| {
        Snapshot::new(
            phase,
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(last),
                registers,
            }],
        )
    };
    let mut b = TraceBuilder::new(
        TraceId(7),
        TraceFrameInfo {
            register_count: 4,
            code_len: 16,
            inlined: None,
            activation: None,
        },
    );
    for register in [1, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let no_in = b.input(1, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let no = b.loop_param(TraceType::Float, LoopParamInit::Input(no_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let seven = b
        .emit(site(0), TraceOp::Constant(f(7.0)), Some(TraceType::Float))
        .unwrap();
    let rem = b
        .emit(
            site(0),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Modulo,
                left: i,
                right: seven,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let three = b
        .emit(site(1), TraceOp::Constant(f(3.0)), Some(TraceType::Float))
        .unwrap();
    let c = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: crate::jit_ir::Comparison::LessThan,
                left: rem,
                right: three,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let held = if tables > 0 {
        Virtual(VirtualId(0))
    } else {
        Val(c)
    };
    let mut first = snap(
        SnapshotPhase::After {
            pc: 2,
            successor: 3,
        },
        2,
        vec![Unchanged, Val(no), Val(i), held].into(),
    );
    if tables > 0 {
        let recipes = (0..tables)
            .map(|index| VirtualTable {
                alloc: VirtualAlloc::Plain,
                array: vec![if index + 1 < tables {
                    Virtual(VirtualId(u32::try_from(index + 1).unwrap()))
                } else {
                    Val(no)
                }],
                fields: Vec::new(),
                phantom_entries: None,
                metatable: None,
            })
            .collect();
        first = first.with_virtuals(recipes);
    }
    let first = b.snapshot(first);
    let g = b.guard(site(2), first);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: c,
            expected: false,
            guard: g,
        },
        None,
    );
    let no2 = b
        .emit(
            site(5),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: no,
                right: i,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let one = b
        .emit(site(6), TraceOp::Constant(f(1.0)), Some(TraceType::Float))
        .unwrap();
    let i2 = b
        .emit(
            site(6),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let top = b
        .emit(site(6), TraceOp::Constant(f(100.0)), Some(TraceType::Float))
        .unwrap();
    let more = b
        .emit(
            site(6),
            TraceOp::Compare {
                comparison: crate::jit_ir::Comparison::LessEqual,
                left: i2,
                right: top,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let after = vec![Unchanged, Val(no2), Val(i2), Val(c)];
    let s = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 7,
        },
        6,
        after.clone().into(),
    ));
    let g = b.guard(site(6), s);
    b.emit(
        site(6),
        TraceOp::GuardBranch {
            condition: more,
            expected: true,
            guard: g,
        },
        None,
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 6,
            successor: 0,
        },
        6,
        after.into(),
    ));
    b.set_poll_snapshot(poll);
    let mut ir = b.finish(TraceTerminator::LoopBack {
        args: vec![no2, i2],
    });
    ir.header_live = vec![0, 1, 2];
    ir
}

/// A side can start from an exit that builds virtual tables when the run can
/// build them on the way in, which it does for as many as a helper's result
/// words name lanes (`ExitRestore::bridges_to_side`). An exit that builds
/// more gets no side.
#[test]
fn a_guard_that_builds_more_tables_than_a_side_takes_gets_no_side_trace() {
    let most = crate::jit_ir::trace::MAX_HELPER_RESULTS;
    for tables in [0, 1, most, most + 1] {
        let ir = loop_with_guard(tables);
        ir.verify().unwrap();
        let compiled = compile_trace(&ir, OptimizationTier::Optimized).unwrap();
        let restores = restores_of(&compiled, 4, &[]).unwrap();
        let keys = side_keys_of(&compiled, &restores, &ir, false, None);
        // The first branch guard (origin 0) is the one with the table.
        let first_guard_exit = compiled
            .exits()
            .iter()
            .position(|exit| exit.class == TraceExitClass::GuardFail && exit.origin == Some(0))
            .expect("the first guard has an exit");
        assert_eq!(
            keys[first_guard_exit].is_some(),
            tables <= most,
            "{tables} tables"
        );
        // Only its exit is refused: the loop's own back-branch guard is not
        // a virtual's concern.
        assert_eq!(restores[first_guard_exit].virtuals.len(), tables);
    }
}

#[test]
fn a_restore_without_materialized_tables_refuses_a_virtual_entry() {
    use crate::vm::jit::trace::deopt::{DeoptError, restore_root_snapshot};
    let profile = CompatibilityProfile::Lua54;
    let mut vm = vm_for(profile);
    let case = case(&mut vm, profile, false);
    let exited = run(&case, OptimizationTier::Optimized);
    let lowered = exited.compiled.lowered_member(0);
    let descriptor = &exited.compiled.exits()[exited.exit];
    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
    let view = ExitView {
        descriptor,
        area: &exited.area,
    };
    let chunk = chunk();
    let mut frames = frames_of(&vm, &chunk);
    for register in frames.at_mut(0).registers.iter_mut() {
        *register = Value::from("garbage");
    }
    let result = restore_root_snapshot(&mut frames.at_mut(0), snapshot, &view, &[Value::Nil]);
    assert!(
        matches!(result, Err(DeoptError::MissingVirtual(_))),
        "{result:?}"
    );
    // Nothing was written.
    assert!(
        frames
            .at(0)
            .registers
            .iter()
            .all(|register| *register == Value::from("garbage"))
    );
}

#[test]
fn a_virtual_exit_that_reaches_the_side_threshold_is_refused_and_counted() {
    use crate::vm::jit::trace::policy::SIDE_EXIT_THRESHOLD;
    let profile = CompatibilityProfile::Lua54;
    let mut vm = Vm::with_profile_and_execution_mode(profile, ExecutionMode::Tracing);
    let chunk = Arc::new(Chunk::new(4, Vec::new(), vec![Instruction::return_(0); 16]));
    // One table more than a side can start with.
    let ir = loop_with_guard(crate::jit_ir::trace::MAX_HELPER_RESULTS + 1);
    let index = {
        let jit = &mut vm.jit;
        jit.trace
            .install_hand_built(&mut jit.statistics.trace, &chunk, 0, ir)
            .expect("the hand-built trace installs")
    };
    for run in 1..=2 * SIDE_EXIT_THRESHOLD {
        let mut frames = frames_of(&vm, &chunk);
        {
            let frame = frames.at_mut(0);
            frame.registers[1] = Value::Number(0.0);
            frame.registers[2] = Value::Number(0.0);
        }
        let jit = &mut vm.jit;
        let outcome = jit.trace.enter(
            &mut jit.statistics.trace,
            index,
            &mut frames,
            0,
            100,
            &mut vm.runtime,
        );
        assert!(
            matches!(
                outcome,
                crate::vm::jit::trace::TraceRun::Ran { poll: false, .. }
            ),
            "run {run}: {outcome:?}"
        );
        vm.jit.trace.forgive(index);
        // The refusal is counted once, when the guard reaches the threshold.
        let expected = u64::from(run >= SIDE_EXIT_THRESHOLD);
        let stats = vm.jit_statistics().trace;
        assert_eq!(stats.side_rejections, expected, "run {run}");
        assert_eq!(stats.side_recordings, 0, "run {run}");
        assert_eq!(stats.side_traces_recorded, 0, "run {run}");
    }
    assert_eq!(vm.jit_statistics().trace.side_installs, 0);
}

// ------------------------------------------------------ recorded constructors

/// A number constant of the profile.
fn constant(profile: CompatibilityProfile, value: i64) -> Constant {
    if profile.is_lua53_or_later() {
        Constant::Integer(value)
    } else {
        Constant::Number(value as f64)
    }
}

/// `local ring = {}; for i = 1, limit do local t = {}; t[1] = i; ring[i] = t end`.
/// Header pc 5, latch pc 9. The table `t` is stored in another table, so it
/// escapes at the store at pc 8.
fn ring_program(profile: CompatibilityProfile, limit: i64) -> Chunk {
    Chunk::with_parts(
        7,
        0,
        vec![constant(profile, 1), constant(profile, limit)],
        vec![
            Instruction::new_table(4, 0, 0),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 4, 9),
            Instruction::new_table(5, 0, 0),
            load(6, 0),
            Instruction::set_table(5, 6, 3),
            Instruction::set_table(4, 3, 5),
            for_loop(0, 9, 5),
            Instruction::return_(4),
        ],
        vec![],
        vec![],
    )
}

/// `local acc = 0; for i = 1, limit do local t, u = {}, {}; u[1] = i; t[1] = u;
/// acc = acc + t[1][1] end return acc`. The table `u` is stored in `t` and
/// read back, and neither ever leaves the iteration. Header pc 5, latch pc 13.
fn nested_program(profile: CompatibilityProfile, limit: i64) -> Chunk {
    Chunk::with_parts(
        11,
        0,
        vec![
            constant(profile, 1),
            constant(profile, limit),
            constant(profile, 0),
        ],
        vec![
            load(10, 2),
            load(0, 0),
            load(1, 1),
            load(2, 0),
            for_prep(0, 4, 13),
            Instruction::new_table(5, 0, 0),
            Instruction::new_table(7, 0, 0),
            load(6, 0),
            Instruction::set_table(7, 6, 3),
            Instruction::set_table(5, 6, 7),
            Instruction::get_table(8, 5, 6),
            Instruction::get_table(9, 8, 6),
            binary_add(10, 10, 9),
            for_loop(0, 13, 5),
            Instruction::return_(10),
        ],
        vec![],
        vec![],
    )
}

fn binary_add(destination: u8, left: u8, right: u8) -> Instruction {
    super::trace_record::binary(OpCode::Add, destination, left, right)
}

fn helper_calls(ir: &TraceIr, wanted: HelperId) -> usize {
    ir.body
        .iter()
        .filter(|i| matches!(&i.op, TraceOp::CallHelper { helper, .. } if *helper == wanted))
        .count()
}

fn phantom_tables(ir: &TraceIr) -> usize {
    ir.body
        .iter()
        .filter(|i| matches!(i.op, TraceOp::PhantomTable { .. }))
        .count()
}

#[test]
fn a_table_stored_in_another_is_made_by_a_call_and_filled_by_another() {
    for profile in PROFILES {
        let (vm, _) = record(profile, &ring_program(profile, 400));
        let ir = only_trace(&vm);
        ir.verify().unwrap();
        // The escape made the table and stored its one entry: no phantom
        // charge stands for it, and the trace is one that makes tables.
        assert_eq!(helper_calls(ir, HelperId::TABLE_NEW), 1, "{profile:?}");
        assert_eq!(helper_calls(ir, HelperId::TABLE_INIT), 1, "{profile:?}");
        assert_eq!(phantom_tables(ir), 0, "{profile:?}");
        assert!(ir.makes_tables(), "{profile:?}");
        // The calls come before the store of the table, in order, and the
        // release at the end of the body drops the lane.
        let at = |wanted: HelperId| {
            ir.body
                .iter()
                .position(
                    |i| matches!(&i.op, TraceOp::CallHelper { helper, .. } if *helper == wanted),
                )
                .unwrap()
        };
        assert!(
            at(HelperId::TABLE_NEW) < at(HelperId::TABLE_INIT),
            "{profile:?}"
        );
        assert!(
            matches!(ir.body.last().map(|i| &i.op), Some(TraceOp::ReleaseStrings { keep }) if keep.is_empty()),
            "{profile:?}"
        );
        // The statistics count no table abort.
        assert_eq!(
            vm.jit_statistics()
                .trace
                .aborts(TraceAbortReason::VirtualEscape),
            0,
            "{profile:?}"
        );
    }
}

#[test]
fn a_table_inside_a_table_that_never_escapes_is_never_made() {
    for profile in PROFILES {
        let (vm, results) = record(profile, &nested_program(profile, 400));
        let ir = only_trace(&vm);
        ir.verify().unwrap();
        // Two constructors stay virtual, the inner one stored in the outer
        // and read back: charges, no allocation and no store call.
        assert_eq!(phantom_tables(ir), 2, "{profile:?}");
        assert_eq!(helper_calls(ir, HelperId::TABLE_NEW), 0, "{profile:?}");
        assert_eq!(helper_calls(ir, HelperId::TABLE_INIT), 0, "{profile:?}");
        assert!(!ir.makes_tables(), "{profile:?}");
        assert!(
            !ir.body
                .iter()
                .any(|i| matches!(i.op, TraceOp::ReleaseStrings { .. })),
            "{profile:?}: nothing was made, so nothing is released"
        );
        let expected: i64 = (1..=400).sum();
        assert_eq!(
            results.first().map(|value| match value {
                Value::Integer(n) => *n as f64,
                Value::Number(n) => *n,
                other => panic!("{other:?}"),
            }),
            Some(expected as f64),
            "{profile:?}"
        );
    }
}
