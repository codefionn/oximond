//! String-key lookups in native code ([`TraceOp::HashLoad`]).
//!
//! The tests play the VM. `keys` is a slice of `Value::String` the test owns
//! (the array a trace reads keys out of, borrowed by address), `values` is a
//! list of `(name, value)` pairs behind [`TraceTables::hash_get`]. The
//! compiled trace runs at both tiers, with and without a forced guard, the
//! reference evaluator runs over the same data, and the exit and the restored
//! registers must agree.

#![allow(clippy::too_many_lines)]

use super::*;
use crate::jit::{OptimizationTier, TRACE_TAG_BORROWED_STRING, TRACE_TAG_STRING};
use crate::jit_ir::trace::reference::{ForcedExit, Outcome, TraceHeap, evaluate_with_heap};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TableBinding, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp,
    TraceScalar, TraceSite, TraceTerminator, TraceType,
};
use crate::runtime::{NativeStringProbe, NativeStringProbeScratch, Runtime, TableHandle, Value};

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn snap(phase: SnapshotPhase, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: None,
            registers: registers.into(),
        }],
    )
}

/// What the lookup expects of the value it finds.
pub(super) fn trace(expected: TraceType) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value as Val};
    let mut b = TraceBuilder::new(
        TraceId(0),
        TraceFrameInfo {
            register_count: 5,
            code_len: 32,
            inlined: None,
            activation: None,
        },
    );
    for register in [0, 1, 4] {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Integer);
    let sum_in = b.input(1, TraceType::Integer);
    let keys = b.input(2, TraceType::Table);
    let values = b.input(3, TraceType::Table);
    b.set_bindings(vec![
        TableBinding {
            register: 2,
            kind: BindingKind::Array { first: None },
        },
        TableBinding {
            register: 3,
            kind: BindingKind::NoMetatable,
        },
    ]);
    let i = b.loop_param(TraceType::Integer, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Integer, LoopParamInit::Input(sum_in));
    let nil = Const(TraceScalar::Nil);
    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        vec![Val(i), Val(sum), Unchanged, Unchanged, nil],
    ));
    let g0 = b.guard(site(1), s0);
    let key = b
        .emit(
            site(1),
            TraceOp::ArrayLoad {
                binding: 0,
                table: keys,
                key: i,
                expected: TraceType::StringRef,
                past_end: false,
                guard: g0,
            },
            Some(TraceType::StringRef),
        )
        .unwrap();
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 2 },
        vec![Val(i), Val(sum), Unchanged, Unchanged, Val(key)],
    ));
    let g1 = b.guard(site(2), s1);
    let found = b
        .emit(
            site(2),
            TraceOp::HashLoad {
                binding: 1,
                table: values,
                key,
                expected,
                guard: g1,
            },
            Some(expected),
        )
        .unwrap();
    let one = b
        .emit(
            site(4),
            TraceOp::Constant(TraceScalar::Integer(1)),
            Some(TraceType::Integer),
        )
        .unwrap();
    // Another type is only looked at: the sum counts the lookups, so no loop
    // parameter passes through unchanged (which the MIR backend leaves to
    // Cranelift for a root with no side).
    let sum_next = b
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: sum,
                right: if expected == TraceType::Integer {
                    found
                } else {
                    one
                },
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let i_next = b
        .emit(
            site(4),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 4,
            successor: 1,
        },
        vec![Val(i_next), Val(sum_next), Unchanged, Unchanged, Val(key)],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![i_next, sum_next],
    })
}

#[derive(Clone)]
struct World {
    keys: Vec<Value>,
    values: Vec<(&'static str, Value)>,
}

impl World {
    fn new(names: &[&'static str], values: &[(&'static str, Value)]) -> Self {
        Self {
            keys: names.iter().map(|name| Value::from(*name)).collect(),
            values: values.to_vec(),
        }
    }

    /// The same world with every key string in an allocation of its own
    /// that nothing has hashed. A run of MIR's probe gives the slots the
    /// strings of the table it read (`trace_adopt_key`), so a world that
    /// has to meet a table for the first time is made again.
    fn fresh(&self) -> Self {
        Self {
            keys: self
                .keys
                .iter()
                .map(|key| match key {
                    Value::String(text) => Value::from(text.as_bytes().to_vec()),
                    other => other.clone(),
                })
                .collect(),
            values: self.values.clone(),
        }
    }

    /// The world with the hash of every key string computed, which a string
    /// has once any lookup used it.
    fn warmed(self) -> Self {
        for key in &self.keys {
            if let Value::String(text) = key {
                let _ = text.cached_hash();
            }
        }
        self
    }

    fn find(&self, name: &[u8]) -> Option<&Value> {
        self.values
            .iter()
            .find(|(held, _)| held.as_bytes() == name)
            .map(|(_, value)| value)
    }
}

fn scalar_of(value: Option<&Value>) -> Option<TraceScalar> {
    match value {
        None | Some(Value::Nil) => Some(TraceScalar::Nil),
        Some(Value::Integer(integer)) => Some(TraceScalar::Integer(*integer)),
        Some(Value::Number(number)) => Some(TraceScalar::Float(*number)),
        Some(Value::Boolean(flag)) => Some(TraceScalar::Boolean(*flag)),
        Some(_) => None,
    }
}

/// The reference's model: a borrow names its position in `keys`.
impl TraceHeap for World {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let TraceScalar::Integer(index) = key else {
            return None;
        };
        let position = usize::try_from(index)
            .ok()
            .filter(|p| (1..=self.keys.len()).contains(p))?;
        Some(TraceScalar::StringRef(position as u64))
    }
    fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
        unreachable!()
    }
    fn hash_get(&mut self, _: u32, key: TraceScalar) -> Option<TraceScalar> {
        let TraceScalar::StringRef(position) = key else {
            panic!("a borrowed key: {key:?}")
        };
        let Value::String(name) = &self.keys[position as usize - 1] else {
            panic!("keys hold strings")
        };
        scalar_of(self.find(name.as_bytes()))
    }
    fn field_get(&mut self, _: u32) -> TraceScalar {
        unreachable!()
    }
    fn field_set(&mut self, _: u32, _: TraceScalar) {
        unreachable!()
    }
    fn length(&mut self, _: u32) -> TraceScalar {
        unreachable!()
    }
}

/// The VM's side of a run.
struct Dispatcher<'a> {
    world: &'a World,
    lookups: usize,
}

impl TraceTables for Dispatcher<'_> {
    fn lane(&mut self, _: &Value) -> Option<usize> {
        None
    }
    fn row(&mut self, _: usize, _: &mut [usize]) {}
    #[allow(unsafe_code)]
    fn hash_get(&mut self, binding: usize, key_tag: i64, key_bits: i64) -> TraceInputSlot {
        assert_eq!(binding, 1);
        assert_eq!(key_tag, TRACE_TAG_BORROWED_STRING);
        assert_ne!(key_tag, TRACE_TAG_STRING);
        self.lookups += 1;
        // SAFETY: a borrowed key is the address of a slot of `world.keys`,
        // which the test keeps alive and untouched for the whole run.
        let Value::String(name) = (unsafe { &*(key_bits as usize as *const Value) }) else {
            panic!("a borrowed key is a string")
        };
        // The VM's lookup hashes the key, which is what lets native code
        // probe with it from then on.
        let _ = name.cached_hash();
        match scalar_of(self.world.find(name.as_bytes())) {
            Some(scalar) => TraceInputSlot::from_scalar(scalar),
            None => TraceInputSlot::other(),
        }
    }
}

#[derive(Debug, PartialEq)]
struct Observed {
    guard: Option<u32>,
    registers: Vec<String>,
    iterations: i64,
    lookups: usize,
}

fn show(scalar: TraceScalar, base: usize) -> String {
    match scalar {
        // Position of the slot in `keys`, counting from 1.
        TraceScalar::StringRef(address) => {
            format!("key{}", (address as usize - base) / size_of::<Value>() + 1)
        }
        other => format!("{other:?}"),
    }
}

fn registers() -> [TraceScalar; 5] {
    [
        TraceScalar::Integer(1),
        TraceScalar::Integer(0),
        TraceScalar::Table(0),
        TraceScalar::Table(1),
        TraceScalar::Nil,
    ]
}

fn run_native(
    ir: &TraceIr,
    tier: OptimizationTier,
    world: &World,
    forced: Option<usize>,
    index: Option<NativeStringProbe>,
) -> Observed {
    let trace = match forced {
        Some(guard) => compile_trace_forcing_guard(ir, tier, guard),
        None => compile_trace(ir, tier),
    }
    .unwrap();
    assert!(trace.needs_tables());
    let registers = registers();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let base = world.keys.as_ptr() as usize;
    let bindings = [
        TraceBinding::raw_array(world.keys.len(), base),
        index.map_or_else(TraceBinding::callee, |probe| {
            TraceBinding::probe(probe, true)
        }),
    ];
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(100_000);
    let mut space = TraceTableSpace::new();
    let mut dispatcher = Dispatcher { world, lookups: 0 };
    let outcome = trace.run_with_tables(
        &inputs,
        &bindings,
        &mut control,
        &mut area,
        &mut space,
        &mut dispatcher,
    );
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the trace refused entry");
    };
    let descriptor = &trace.exits()[index];
    let lowered = trace.lowered_ir();
    let snapshot = &lowered.snapshots[descriptor.snapshot.index()];
    let state = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| {
            let scalar = match entry {
                RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap(),
                RegisterEntry::Const(scalar) => *scalar,
                RegisterEntry::Unchanged => registers[register],
                RegisterEntry::Root(_) => unreachable!("no roots here"),
                RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                    unreachable!("no virtual tables here")
                }
            };
            show(scalar, base)
        })
        .collect();
    assert_eq!(
        control.hash_calls,
        i64::try_from(dispatcher.lookups).unwrap(),
        "the control block counts the calls the VM answered"
    );
    Observed {
        guard: descriptor.origin,
        registers: state,
        iterations: 100_000 - control.fuel,
        lookups: dispatcher.lookups,
    }
}

fn run_reference(ir: &TraceIr, world: &World, forced: Option<u32>) -> Observed {
    let mut model = world.clone();
    let forced = forced.map(|guard| ForcedExit {
        frame: FrameId(0),
        pc: if guard == 0 { 1 } else { 2 },
        guard: Some(crate::jit_ir::trace::GuardId(guard)),
        iteration: 0,
    });
    let evaluation = evaluate_with_heap(
        ir,
        &registers(),
        &[],
        100_000,
        forced,
        &mut model,
        |_, _| None,
    );
    let Outcome::GuardFailed { guard, restored } = evaluation.outcome else {
        panic!("unexpected reference outcome {:?}", evaluation.outcome)
    };
    Observed {
        guard: Some(guard.0),
        registers: restored
            .registers
            .iter()
            .map(|scalar| match scalar {
                TraceScalar::StringRef(position) => format!("key{position}"),
                other => format!("{other:?}"),
            })
            .collect(),
        iterations: 0,
        lookups: 0,
    }
}

/// How the table of `world.values` is built for the string index.
#[derive(Clone, Copy, PartialEq)]
enum Keys {
    /// The table's keys are strings of their own, as most lookups meet them.
    Separate,
    /// The table's keys are the very strings of `world.keys`.
    Shared,
}

/// A runtime holding the table of `world.values`, and its string index. The
/// fields other than `probe` own what the index points into.
#[allow(dead_code)]
struct Indexed {
    runtime: Runtime,
    table: TableHandle,
    scratch: NativeStringProbeScratch,
    probe: NativeStringProbe,
}

/// Indexes the table of `world.values`: with `extra` more keys the lookups
/// never ask for (the index fills up and the probes collide), a hashed table
/// when there are enough of them. `warm` computes the hash of the strings of
/// `world.keys`, which a string has once any lookup used it.
fn indexed(world: &World, keys: Keys, extra: usize, warm: bool) -> Indexed {
    let mut runtime = Runtime::new();
    let table = runtime.create_table().unwrap();
    for (name, value) in &world.values {
        let key = match keys {
            Keys::Shared => world
                .keys
                .iter()
                .find(|held| matches!(held, Value::String(text) if text.as_bytes() == name.as_bytes()))
                .cloned()
                .unwrap_or_else(|| Value::from(*name)),
            Keys::Separate => Value::from(*name),
        };
        runtime.table_set(table, &key, value.clone()).unwrap();
    }
    for index in 0..extra {
        runtime
            .table_set(
                table,
                &Value::from(format!("extra_{index}")),
                Value::Integer(0),
            )
            .unwrap();
    }
    if warm {
        for key in &world.keys {
            if let Value::String(text) = key {
                let _ = text.cached_hash();
            }
        }
    }
    let mut scratch = NativeStringProbeScratch::default();
    // A big hashed table builds its index on the second request.
    let _ = runtime.table_native_probe(table, &mut scratch);
    let probe = runtime
        .table_native_probe(table, &mut scratch)
        .expect("an index");
    Indexed {
        runtime,
        table,
        scratch,
        probe,
    }
}

/// Native and reference leave through the same guard with the same registers,
/// whether the VM answers the lookups or native code probes the table's index.
/// Returns what the run with the VM answering saw.
fn agree(ir: &TraceIr, world: &World, forced: Option<u32>) -> Observed {
    let reference = run_reference(ir, world, forced);
    let mut last = None;
    for tier in TIERS {
        let native = run_native(ir, tier, world, forced.map(|guard| guard as usize), None);
        assert_eq!(native.guard, reference.guard, "{tier:?}: guard");
        assert_eq!(native.registers, reference.registers, "{tier:?}: registers");
        last = Some(native);
        for (keys, extra) in [
            (Keys::Separate, 0),
            (Keys::Shared, 0),
            (Keys::Separate, 150),
        ] {
            let index = indexed(world, keys, extra, true);
            let probed = run_native(
                ir,
                tier,
                world,
                forced.map(|guard| guard as usize),
                Some(index.probe),
            );
            assert_eq!(probed.guard, reference.guard, "{tier:?}: guard (index)");
            assert_eq!(
                probed.registers, reference.registers,
                "{tier:?}: registers (index)"
            );
            assert_eq!(
                probed.iterations,
                last.as_ref().unwrap().iterations,
                "{tier:?}: iterations (index)"
            );
            if forced.is_none() {
                assert_eq!(
                    probed.lookups, 0,
                    "{tier:?}: the probe answers every lookup"
                );
            }
            drop(index);
        }
    }
    last.unwrap()
}

fn int(value: i64) -> Value {
    Value::Integer(value)
}

#[test]
fn a_loop_of_hits_runs_until_the_array_ends() {
    let ir = trace(TraceType::Integer);
    ir.verify().unwrap();
    let world = World::new(
        &["a", "b", "c"],
        &[
            ("a", int(5)),
            ("b", int(7)),
            ("c", int(11)),
            ("d", int(100)),
        ],
    );
    let observed = agree(&ir, &world, None);
    // Three lookups, and the fourth key read leaves before its instruction.
    assert_eq!(observed.guard, Some(0));
    assert_eq!(observed.iterations, 3);
    assert_eq!(observed.lookups, 3);
    assert_eq!(observed.registers[..2], ["Integer(4)", "Integer(23)"]);
}

#[test]
fn a_missing_key_or_another_type_leaves_at_the_lookup_with_the_key_restored() {
    let ir = trace(TraceType::Integer);
    for (label, other) in [
        ("missing", None),
        ("float", Some(Value::Number(1.5))),
        ("boolean", Some(Value::Boolean(true))),
        ("string", Some(Value::from("text"))),
    ] {
        let world = World::new(
            &["a", "b", "c"],
            &[("a", int(5))]
                .into_iter()
                .chain(other.map(|value| ("b", value)))
                .collect::<Vec<_>>(),
        );
        let observed = agree(&ir, &world, None);
        assert_eq!(observed.guard, Some(1), "{label}");
        assert_eq!(observed.iterations, 1, "{label}");
        assert_eq!(
            observed.registers,
            ["Integer(2)", "Integer(5)", "Table(0)", "Table(1)", "key2"],
            "{label}"
        );
    }
}

#[test]
fn a_nil_lookup_passes_for_a_missing_key_and_leaves_for_a_present_one() {
    let ir = trace(TraceType::Nil);
    ir.verify().unwrap();
    let world = World::new(&["x", "y"], &[("z", int(1))]);
    assert_eq!(agree(&ir, &world, None).guard, Some(0));
    let world = World::new(&["x", "y"], &[("y", int(1))]);
    let observed = agree(&ir, &world, None);
    assert_eq!(observed.guard, Some(1));
    assert_eq!(observed.iterations, 1);
}

#[test]
fn floats_and_booleans_are_looked_up_with_their_own_type() {
    for (expected, value) in [
        (TraceType::Float, Value::Number(2.5)),
        (TraceType::Boolean, Value::Boolean(false)),
    ] {
        let ir = trace(expected);
        ir.verify().unwrap();
        let world = World::new(&["a", "b"], &[("a", value.clone()), ("b", value)]);
        // Both lookups pass; the array ends.
        assert_eq!(agree(&ir, &world, None).guard, Some(0), "{expected:?}");
        // The other type leaves at the first lookup.
        let world = World::new(&["a"], &[("a", int(1))]);
        assert_eq!(agree(&ir, &world, None).guard, Some(1), "{expected:?}");
    }
}

#[test]
fn a_forced_guard_leaves_at_the_lookup_before_any_effect() {
    let ir = trace(TraceType::Integer);
    let world = World::new(&["a", "b"], &[("a", int(5)), ("b", int(6))]);
    let observed = agree(&ir, &world, Some(1));
    assert_eq!(observed.guard, Some(1));
    assert_eq!(observed.iterations, 0);
}

#[test]
fn a_root_trace_cannot_take_a_borrow_at_entry() {
    // Only a parent can hand a side trace the address of the slot it
    // borrows from. A root trace has no parent, so the compiler refuses.
    let mut b = TraceBuilder::new(
        TraceId(1),
        TraceFrameInfo {
            register_count: 2,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    let _borrowed = b.input(0, TraceType::StringRef);
    b.emit(
        site(0),
        TraceOp::Constant(TraceScalar::Integer(1)),
        Some(TraceType::Integer),
    );
    let poll = b.snapshot(snap(
        SnapshotPhase::After {
            pc: 0,
            successor: 0,
        },
        vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: Vec::new() });
    for tier in TIERS {
        ir.verify().unwrap();
        assert!(
            matches!(
                compile_trace(&ir, tier),
                Err(TraceCompileError::NativeBackendRejected)
            ),
            "{tier:?}"
        );
    }
}

/// Runs `ir` at both tiers over `world` with the table of `index` and returns
/// what each saw.
fn probed_runs(ir: &TraceIr, world: &World, index: &Indexed) -> Vec<Observed> {
    TIERS
        .into_iter()
        .map(|tier| run_native(ir, tier, world, None, Some(index.probe)))
        .collect()
}

#[test]
fn the_probe_replaces_the_call_on_every_lookup() {
    let ir = trace(TraceType::Integer);
    let names: Vec<&'static str> = ["a", "b", "c", "d", "e", "f", "g", "h"].to_vec();
    let values: Vec<(&'static str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(index as i64 + 1)))
        .collect();
    let world = World::new(&names, &values);
    // Asked of the VM, every iteration is a call.
    let called = agree(&ir, &world, None);
    assert_eq!(called.iterations, 8);
    assert_eq!(called.lookups, 8);
    // Probing the index, none is, at both tiers and for every layout of the
    // table (ordered, hashed with few keys, hashed with many).
    for extra in [0, 20, 500] {
        let index = indexed(&world, Keys::Separate, extra, true);
        for observed in probed_runs(&ir, &world, &index) {
            assert_eq!(observed.lookups, 0, "{extra} extra keys");
            assert_eq!(observed.iterations, 8);
            assert_eq!(observed.registers[..2], called.registers[..2]);
            assert_eq!(observed.registers[..2], ["Integer(9)", "Integer(36)"]);
        }
    }
}

#[test]
fn a_key_without_a_hash_goes_to_the_vm_once_and_the_answer_is_the_same() {
    let ir = trace(TraceType::Integer);
    let world = World::new(
        &["a", "b", "c"],
        &[("a", int(5)), ("b", int(7)), ("c", int(11))],
    );
    // The strings of `keys` have not been hashed: the probe cannot start, so
    // each lookup is the call it always was.
    let index = indexed(&world, Keys::Separate, 0, false);
    let first = run_native(&ir, TIERS[0], &world, None, Some(index.probe));
    assert_eq!(first.lookups, 3);
    assert_eq!(first.registers[..2], ["Integer(4)", "Integer(23)"]);
    // The calls hashed them: now the probe answers.
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.lookups, 0);
        assert_eq!(observed.registers[..2], ["Integer(4)", "Integer(23)"]);
    }
}

#[test]
fn long_keys_are_probed_by_identity_and_compared_by_the_vm_otherwise() {
    let ir = trace(TraceType::Integer);
    let long_a: &'static str = "a key that is longer than any inline string, a";
    let long_b: &'static str = "a key that is longer than any inline string, b";
    let world = World::new(&[long_a, long_b], &[(long_a, int(3)), (long_b, int(4))]);
    // The table's keys are the strings of `keys`: same allocation, so the
    // probe finds them with no byte comparison.
    let shared = indexed(&world, Keys::Shared, 0, true);
    for observed in probed_runs(&ir, &world, &shared) {
        assert_eq!(observed.lookups, 0);
        assert_eq!(observed.registers[..2], ["Integer(3)", "Integer(7)"]);
    }
    // The table's keys are other allocations with the same bytes: a byte
    // comparison, which the VM does.
    let separate = indexed(&world, Keys::Separate, 0, true);
    for observed in probed_runs(&ir, &world, &separate) {
        assert_eq!(observed.lookups, 2);
        assert_eq!(observed.registers[..2], ["Integer(3)", "Integer(7)"]);
    }
    // A missing long key is a miss the probe itself settles, or one the VM
    // does when the same hash is not found: either way the guard leaves.
    let missing = World::new(&[long_a], &[("other", int(1))]);
    let index = indexed(&missing, Keys::Separate, 0, true);
    for observed in probed_runs(&ir, &missing, &index) {
        assert_eq!(observed.guard, Some(1));
        assert_eq!(observed.registers[4], "key1");
    }
}

#[test]
fn numeric_looking_and_near_equal_keys_are_different_keys() {
    let ir = trace(TraceType::Integer);
    let names = ["1", "01", "1.0", "10", "a", "a\0", "ab"];
    let values: Vec<(&'static str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(1 << index)))
        .collect();
    let world = World::new(&names, &values);
    for extra in [0, 100] {
        let index = indexed(&world, Keys::Separate, extra, true);
        for observed in probed_runs(&ir, &world, &index) {
            assert_eq!(observed.lookups, 0);
            // Each key found its own value: the sum has every bit once.
            assert_eq!(observed.registers[1], "Integer(127)");
        }
    }
    // A key that differs from a stored one in the last byte only misses.
    let world = World::new(&["10"], &[("1", int(5)), ("100", int(6))]);
    let index = indexed(&world, Keys::Separate, 0, true);
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.guard, Some(1));
        assert_eq!(observed.iterations, 0);
    }
}

#[test]
fn a_deleted_key_reads_nil_through_the_probe() {
    let world = World::new(&["a", "b"], &[("a", int(5)), ("b", int(6))]);
    let mut index = indexed(&world, Keys::Separate, 200, true);
    let Value::String(deleted) = Value::from("b") else {
        unreachable!()
    };
    assert!(
        index
            .runtime
            .table_delete_string_in_place(index.table, &deleted)
            .unwrap()
    );
    // Deleting in place moves nothing, so the index stands.
    let again = index
        .runtime
        .table_native_probe(index.table, &mut index.scratch)
        .expect("the layout is the same");
    assert_eq!(again, index.probe);
    // A trace that expects an integer leaves at the deleted key; one that
    // expects nil passes it.
    let ir = trace(TraceType::Integer);
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.guard, Some(1));
        assert_eq!(observed.iterations, 1);
        assert_eq!(observed.registers[..2], ["Integer(2)", "Integer(5)"]);
    }
    let ir = trace(TraceType::Nil);
    let world_a = World::new(&["b"], &[]);
    for observed in probed_runs(&ir, &world_a, &index) {
        assert_eq!(observed.guard, Some(0));
        assert_eq!(observed.iterations, 1);
    }
}

#[test]
fn a_changed_value_is_read_through_the_index_the_run_holds() {
    let world = World::new(&["a", "b"], &[("a", int(5)), ("b", int(6))]);
    let mut index = indexed(&world, Keys::Separate, 100, true);
    let ir = trace(TraceType::Integer);
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.registers[1], "Integer(11)");
    }
    // An overwrite keeps the layout, so the index stands and sees the value.
    index
        .runtime
        .table_set(index.table, &Value::from("b"), int(60))
        .unwrap();
    let same = index
        .runtime
        .table_native_probe(index.table, &mut index.scratch)
        .unwrap();
    assert_eq!(same, index.probe);
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.registers[1], "Integer(65)");
    }
    // Another type under the key leaves at the lookup.
    index
        .runtime
        .table_set(index.table, &Value::from("b"), Value::Number(1.5))
        .unwrap();
    for observed in probed_runs(&ir, &world, &index) {
        assert_eq!(observed.guard, Some(1));
        assert_eq!(observed.registers[..2], ["Integer(2)", "Integer(5)"]);
    }
}

/// One run of `ir` with the backend `mir` asks for, with MIR's string-index
/// probe on (it is on by default, see `debug_env::mir_probe`). The MIR
/// backend must have made its code, which Cranelift's lookups never need to
/// be told apart by.
fn run_on(
    ir: &TraceIr,
    tier: OptimizationTier,
    world: &World,
    index: Option<NativeStringProbe>,
    mir: bool,
) -> Observed {
    use crate::jit::debug_env::FORCE_BACKEND;
    crate::jit::debug_env::mir_probe_for_this_test();
    FORCE_BACKEND.with(|force| force.set(Some(mir)));
    let compiled = compile_trace(ir, tier).unwrap();
    assert_eq!(compiled.mir_address().is_some(), mir, "{tier:?}");
    let observed = run_native(ir, tier, world, None, index);
    FORCE_BACKEND.with(|force| force.set(None));
    observed
}

/// MIR and Cranelift run `ir` over `world` with the VM answering and with the
/// index of every layout (hashed or not, long chains, keys shared with the
/// table), and leave through the same guard with the same registers, the same
/// iterations and the same number of calls to the VM.
fn backends_agree(ir: &TraceIr, world: &World) {
    for tier in TIERS {
        let cranelift = run_on(ir, tier, world, None, false);
        assert_eq!(run_on(ir, tier, world, None, true), cranelift, "{tier:?}");
        for (keys, extra, warm) in [
            (Keys::Separate, 0, true),
            (Keys::Shared, 0, true),
            (Keys::Separate, 20, true),
            (Keys::Separate, 150, true),
            (Keys::Shared, 500, true),
            (Keys::Separate, 0, false),
        ] {
            // Each backend meets keys no run has touched: the strings of
            // the table for `Shared`, strings of their own otherwise.
            let (first, second) = (world.fresh(), world.fresh());
            let second = if warm { second.warmed() } else { second };
            let index = indexed(&first, keys, extra, warm);
            let other = if keys == Keys::Shared {
                &first
            } else {
                &second
            };
            let expected = run_on(ir, tier, &first, Some(index.probe), false);
            if !warm && keys == Keys::Separate {
                // The VM hashed the first world's keys when Cranelift's run
                // asked it; MIR's run meets unhashed ones too.
                let mir = run_on(ir, tier, other, Some(index.probe), true);
                assert_eq!(mir, expected, "{tier:?} with unhashed keys");
                continue;
            }
            let mir = run_on(ir, tier, other, Some(index.probe), true);
            assert_eq!(mir, expected, "{tier:?} with {extra} extra keys");
        }
    }
}

/// The identities of the strings in the slots of `world.keys`.
fn identities(world: &World) -> Vec<usize> {
    world
        .keys
        .iter()
        .map(|key| match key {
            Value::String(text) => text.identity(),
            _ => 0,
        })
        .collect()
}

#[test]
fn a_slot_takes_the_string_of_the_table_it_is_looked_up_in() {
    use super::mir::adopted_keys;
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let values: Vec<(&str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(index as i64 + 1)))
        .collect();
    let ir = trace(TraceType::Integer);
    // An ordered table, a hashed one, and one whose probes collide.
    for extra in [0, 20, 500] {
        let world = World::new(&names, &values);
        let index = indexed(&world, Keys::Separate, extra, true);
        let before = identities(&world);
        let adopted = adopted_keys();
        let first = run_on(&ir, TIERS[0], &world, Some(index.probe), true);
        assert_eq!(first.lookups, 0);
        assert_eq!(first.registers[..2], ["Integer(9)", "Integer(36)"]);
        // Every slot holds another allocation of the same bytes now.
        assert_eq!(adopted_keys() - adopted, 8, "{extra} extra keys");
        let after = identities(&world);
        for (slot, name) in names.iter().enumerate() {
            assert_ne!(after[slot], before[slot], "{name}");
            assert_eq!(world.keys[slot], Value::from(*name));
        }
        // The next runs find each key at its first compare: no slot changes.
        for tier in TIERS {
            let again = run_on(&ir, tier, &world, Some(index.probe), true);
            assert_eq!(again, first, "{tier:?}");
            assert_eq!(identities(&world), after);
        }
        assert_eq!(adopted_keys() - adopted, 8);
        // The slots keep the strings alive when the table is gone.
        drop(index);
        for (slot, name) in names.iter().enumerate() {
            assert_eq!(world.keys[slot], Value::from(*name));
        }
    }
}

#[test]
fn a_slot_keeps_its_string_when_the_key_is_absent_long_or_the_same() {
    use super::mir::adopted_keys;
    let long: &'static str = "a key that is longer than any inline string";
    let ir = trace(TraceType::Integer);
    // The table's own strings: nothing to take.
    let world = World::new(
        &["a", "b", long],
        &[("a", int(1)), ("b", int(2)), (long, int(3))],
    );
    let index = indexed(&world, Keys::Shared, 0, true);
    let before = identities(&world);
    let adopted = adopted_keys();
    let observed = run_on(&ir, TIERS[0], &world, Some(index.probe), true);
    assert_eq!(observed.iterations, 3);
    assert_eq!(identities(&world), before);
    // A long key in another allocation is the VM's to compare, and a key
    // the table lacks has no string to take.
    let world = World::new(&[long, "a", "absent"], &[("a", int(1)), (long, int(3))]);
    let index = indexed(&world, Keys::Separate, 0, true);
    let before = identities(&world);
    let observed = run_on(&ir, TIERS[0], &world, Some(index.probe), true);
    assert_eq!(observed.iterations, 2);
    assert_eq!(observed.lookups, 1);
    let after = identities(&world);
    assert_eq!(after[0], before[0]);
    assert_ne!(after[1], before[1]);
    assert_eq!(after[2], before[2]);
    assert_eq!(adopted_keys() - adopted, 1);
}

#[test]
fn slots_looked_up_in_two_tables_in_turn_stop_changing_hands() {
    use super::mir::adopted_keys;
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let values: Vec<(&str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(index as i64 + 1)))
        .collect();
    let ir = trace(TraceType::Integer);
    let world = World::new(&names, &values).warmed();
    // Two tables with keys of their own, read in turn through one array.
    let one = indexed(&world.fresh(), Keys::Separate, 0, true);
    let two = indexed(&world.fresh(), Keys::Separate, 20, true);
    let adopted = adopted_keys();
    let expected = run_on(&ir, TIERS[0], &world, Some(one.probe), true);
    for turn in 0..200 {
        let index = if turn % 2 == 0 { &two } else { &one };
        let observed = run_on(&ir, TIERS[0], &world, Some(index.probe), true);
        assert_eq!(observed, expected, "turn {turn}");
    }
    // Every turn would swap all eight slots: the thread stopped long before.
    let swaps = adopted_keys() - adopted;
    assert!((8..200).contains(&swaps), "{swaps} swaps");
}

#[test]
fn with_the_probe_off_mir_leaves_the_lookup_to_cranelift() {
    use crate::jit::debug_env::FORCE_BACKEND;
    let ir = trace(TraceType::Integer);
    for tier in TIERS {
        FORCE_BACKEND.with(|force| force.set(Some(true)));
        crate::jit::debug_env::mir_probe_off_for_this_test();
        let off = compile_trace(&ir, tier).unwrap();
        crate::jit::debug_env::mir_probe_for_this_test();
        let on = compile_trace(&ir, tier).unwrap();
        FORCE_BACKEND.with(|force| force.set(None));
        assert!(off.mir_address().is_none(), "{tier:?}: Cranelift's tree");
        assert!(on.mir_address().is_some(), "{tier:?}: MIR's tree");
    }
}

#[test]
fn mir_probes_hits_and_misses_as_cranelift_does() {
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let hits: Vec<(&str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(index as i64 + 1)))
        .collect();
    let ir = trace(TraceType::Integer);
    backends_agree(&ir, &World::new(&names, &hits));
    // A miss, a value of another type, a deleted-looking nil: every way to
    // leave at the lookup.
    for other in [
        None,
        Some(Value::Number(1.5)),
        Some(Value::Boolean(true)),
        Some(Value::from("text")),
    ] {
        let values: Vec<(&str, Value)> = [("a", int(5))]
            .into_iter()
            .chain(other.map(|value| ("b", value)))
            .collect();
        backends_agree(&ir, &World::new(&["a", "b", "c"], &values));
    }
    for (expected, value) in [
        (TraceType::Float, Value::Number(2.5)),
        (TraceType::Boolean, Value::Boolean(false)),
    ] {
        let ir = trace(expected);
        backends_agree(
            &ir,
            &World::new(&["a", "b"], &[("a", value.clone()), ("b", value)]),
        );
        backends_agree(&ir, &World::new(&["a"], &[("a", int(1))]));
    }
    let ir = trace(TraceType::Nil);
    backends_agree(&ir, &World::new(&["x", "y"], &[("z", int(1))]));
    backends_agree(&ir, &World::new(&["x", "y"], &[("y", int(1))]));
}

/// A name that lives as long as the test, for [`World`].
fn leaked(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

#[test]
fn mir_walks_dense_indexes_past_its_steps_in_line_as_cranelift_does() {
    // An index of 4096 buckets that 3,300 or more keys fill: the probe
    // sequences of many keys are longer than the steps the code makes in
    // line, so the native walk behind them answers, and the lookups never
    // reach the VM.
    let names: Vec<&'static str> = (0..300).map(|n| leaked(format!("probe_{n}"))).collect();
    let values: Vec<(&'static str, Value)> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, int(index as i64 + 1)))
        .collect();
    let world = World::new(&names, &values);
    let ir = trace(TraceType::Integer);
    for tier in TIERS {
        for (keys, extra) in [
            (Keys::Separate, 3000),
            (Keys::Shared, 3000),
            (Keys::Separate, 1000),
        ] {
            let index = indexed(&world, keys, extra, true);
            let cranelift = run_on(&ir, tier, &world, Some(index.probe), false);
            let mir = run_on(&ir, tier, &world, Some(index.probe), true);
            assert_eq!(mir, cranelift, "{tier:?} with {extra} extra keys");
            assert_eq!(mir.lookups, 0, "{tier:?} with {extra} extra keys");
            assert_eq!(mir.iterations, 300);
        }
    }
    // Each key that the table lacks ends at its own empty bucket, after a
    // chain of its own: the lookup leaves through the guard in every case.
    let index = indexed(&world, Keys::Separate, 3000, true);
    for absent in 0..40 {
        let mut asked = names[..20].to_vec();
        asked.push(leaked(format!("absent_{absent}")));
        let world = World::new(&asked, &values);
        // The strings have their hashes before either backend meets them, or
        // the first to run would hash them for the second.
        for key in &world.keys {
            if let Value::String(text) = key {
                let _ = text.cached_hash();
            }
        }
        for tier in TIERS {
            let cranelift = run_on(&ir, tier, &world, Some(index.probe), false);
            let mir = run_on(&ir, tier, &world, Some(index.probe), true);
            assert_eq!(mir, cranelift, "{tier:?} absent_{absent}");
            assert_eq!(mir.guard, Some(1), "{tier:?} absent_{absent}");
            assert_eq!(mir.iterations, 20);
        }
    }
}

#[test]
fn mir_leaves_long_and_near_equal_keys_to_the_vm_as_cranelift_does() {
    let ir = trace(TraceType::Integer);
    let long_a: &'static str = "a key that is longer than any inline string, a";
    let long_b: &'static str = "a key that is longer than any inline string, b";
    backends_agree(
        &ir,
        &World::new(&[long_a, long_b], &[(long_a, int(3)), (long_b, int(4))]),
    );
    backends_agree(
        &ir,
        &World::new(&["1", "01", "1.0"], &[("1", int(1)), ("01", int(2))]),
    );
}

#[test]
fn mir_reads_a_deleted_key_and_a_probe_after_a_rebuild_as_cranelift_does() {
    let world = World::new(&["a", "b"], &[("a", int(5)), ("b", int(6))]);
    let mut index = indexed(&world, Keys::Separate, 200, true);
    let Value::String(deleted) = Value::from("b") else {
        unreachable!()
    };
    assert!(
        index
            .runtime
            .table_delete_string_in_place(index.table, &deleted)
            .unwrap()
    );
    for expected in [TraceType::Integer, TraceType::Nil] {
        let ir = trace(expected);
        for tier in TIERS {
            let cranelift = run_on(&ir, tier, &world, Some(index.probe), false);
            let mir = run_on(&ir, tier, &world, Some(index.probe), true);
            assert_eq!(mir, cranelift, "{expected:?} at {tier:?}");
        }
    }
    // Keys added past the index's room rebuild it: the words the run is given
    // are the new index's.
    for count in 0..400 {
        index
            .runtime
            .table_set(
                index.table,
                &Value::from(format!("more_{count}")),
                Value::Integer(0),
            )
            .unwrap();
    }
    // A big hashed table builds its index on the second request.
    let _ = index
        .runtime
        .table_native_probe(index.table, &mut index.scratch);
    let rebuilt = index
        .runtime
        .table_native_probe(index.table, &mut index.scratch)
        .expect("an index");
    assert_ne!(rebuilt, index.probe, "the table's hash part was rebuilt");
    let ir = trace(TraceType::Integer);
    for tier in TIERS {
        let cranelift = run_on(&ir, tier, &world, Some(rebuilt), false);
        let mir = run_on(&ir, tier, &world, Some(rebuilt), true);
        assert_eq!(mir, cranelift, "{tier:?}");
        assert_eq!(mir.lookups, 0);
    }
}
