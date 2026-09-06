//! String-key lookups in the trace IR ([`TraceOp::HashLoad`]): the verifier's
//! rules, the reference evaluator's semantics over a heap model, and the
//! optimizer's promise to keep a lookup's guard.
//!
//! The loop is `sum = sum + values[keys[i]]`: the key is a borrow of an array
//! slot of `keys`, and `values` is a table without a metatable.

use super::ir::*;
use super::optimize::{OptimizeLevel, optimize_trace};
use super::reference::{Evaluation, Outcome, TraceHeap, evaluate_with_heap};
use super::snapshot::*;
use super::verify::TraceVerifyError;
use crate::jit_ir::Comparison;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn snap(phase: SnapshotPhase, effects: u32, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        effects,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(0),
            registers: registers.into(),
        }],
    )
}

/// How the lookup is built.
#[derive(Clone, Copy)]
struct Shape {
    /// The type the lookup expects.
    expected: TraceType,
    /// Whether `values` has the `NoMetatable` binding.
    plain: bool,
    /// Whether the key is the borrow (otherwise an integer, which is wrong).
    string_key: bool,
}

const GOOD: Shape = Shape {
    expected: TraceType::Integer,
    plain: true,
    string_key: true,
};

/// r0 `i`, r1 `sum`, r2 `keys`, r3 `values`, r4 the key.
fn case(shape: Shape) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let mut b = TraceBuilder::new(
        TraceId(21),
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
    let mut bindings = vec![TableBinding {
        register: 2,
        kind: BindingKind::Array { first: None },
    }];
    if shape.plain {
        bindings.push(TableBinding {
            register: 3,
            kind: BindingKind::NoMetatable,
        });
    }
    b.set_bindings(bindings);
    let i = b.loop_param(TraceType::Integer, LoopParamInit::Input(i_in));
    let sum = b.loop_param(TraceType::Integer, LoopParamInit::Input(sum_in));
    let nil = Const(TraceScalar::Nil);

    let s0 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        vec![Value(i), Value(sum), Unchanged, Unchanged, nil],
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
    let operand = if shape.string_key { key } else { i };
    let s1 = b.snapshot(snap(
        SnapshotPhase::Before { pc: 2 },
        b.effect_position(),
        vec![Value(i), Value(sum), Unchanged, Unchanged, Value(key)],
    ));
    let g1 = b.guard(site(2), s1);
    let found = b
        .emit(
            site(2),
            TraceOp::HashLoad {
                binding: 1,
                table: values,
                key: operand,
                expected: shape.expected,
                guard: g1,
            },
            Some(shape.expected),
        )
        .unwrap();
    let (sum_next, one) = if shape.expected == TraceType::Integer {
        let sum_next = b
            .emit(
                site(3),
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left: sum,
                    right: found,
                },
                Some(TraceType::Integer),
            )
            .unwrap();
        let one = b
            .emit(
                site(4),
                TraceOp::Constant(TraceScalar::Integer(1)),
                Some(TraceType::Integer),
            )
            .unwrap();
        (sum_next, one)
    } else {
        let one = b
            .emit(
                site(4),
                TraceOp::Constant(TraceScalar::Integer(1)),
                Some(TraceType::Integer),
            )
            .unwrap();
        (sum, one)
    };
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
    let _ = Comparison::Equal;
    b.finish(TraceTerminator::LoopBack {
        args: vec![i_next, sum_next],
    })
}

/// `keys` as a list of names; a name is a borrow whose address is its
/// position (counting from 1). `values` maps a name to a scalar.
struct World {
    names: Vec<&'static str>,
    values: Vec<(&'static str, TraceScalar)>,
}

impl World {
    fn name(&self, key: TraceScalar) -> &'static str {
        let TraceScalar::StringRef(position) = key else {
            panic!("a lookup key is a borrow: {key:?}")
        };
        self.names[position as usize - 1]
    }
}

impl TraceHeap for World {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, _binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let TraceScalar::Integer(index) = key else {
            return None;
        };
        let position = usize::try_from(index)
            .ok()
            .filter(|p| (1..=self.names.len()).contains(p))?;
        Some(TraceScalar::StringRef(position as u64))
    }
    fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
        false
    }
    fn hash_get(&mut self, _binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let name = self.name(key);
        Some(
            self.values
                .iter()
                .find(|(held, _)| *held == name)
                .map_or(TraceScalar::Nil, |(_, value)| *value),
        )
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

fn run(world: &mut World, ir: &TraceIr, max: u32) -> Evaluation {
    let registers = [
        TraceScalar::Integer(1),
        TraceScalar::Integer(0),
        TraceScalar::Table(0),
        TraceScalar::Table(1),
        TraceScalar::Nil,
    ];
    evaluate_with_heap(ir, &registers, &[], max, None, world, |_, _| None)
}

fn int(value: i64) -> TraceScalar {
    TraceScalar::Integer(value)
}

#[test]
fn a_lookup_with_a_borrowed_key_verifies_and_survives_every_level() {
    let ir = case(GOOD);
    ir.verify().unwrap();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert_eq!(
            optimized
                .body
                .iter()
                .filter(|instruction| matches!(instruction.op, TraceOp::HashLoad { .. }))
                .count(),
            1,
            "{level:?} keeps the lookup"
        );
    }
}

#[test]
fn a_lookup_ends_no_borrow_and_reads_the_heap() {
    let lookup = TraceOp::HashLoad {
        binding: 0,
        table: ValueId(0),
        key: ValueId(0),
        expected: TraceType::Integer,
        guard: GuardId(0),
    };
    assert!(!lookup.ends_borrows());
    assert!(lookup.touches_heap());
    assert_eq!(lookup.guard(), Some(GuardId(0)));
    assert_eq!(lookup.committed_effect(), None);
}

#[test]
fn a_lookup_whose_result_nobody_reads_keeps_its_guard() {
    // The guard holds the speculation (`if value` on a lookup that found nil
    // folds to a constant branch), so the optimizer never drops the lookup.
    let ir = case(Shape {
        expected: TraceType::Nil,
        ..GOOD
    });
    ir.verify().unwrap();
    for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
        let (optimized, _) = optimize_trace(&ir, level);
        optimized.verify().unwrap();
        assert!(
            optimized
                .body
                .iter()
                .any(|instruction| matches!(instruction.op, TraceOp::HashLoad { .. })),
            "{level:?}"
        );
    }
}

#[test]
fn the_verifier_refuses_a_malformed_lookup() {
    let no_plain = case(Shape {
        plain: false,
        ..GOOD
    });
    // Without the binding the table operand names a missing one.
    assert!(matches!(
        no_plain.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    let integer_key = case(Shape {
        string_key: false,
        ..GOOD
    });
    // A number key is a raw lookup over both parts of the table.
    integer_key.verify().unwrap();
    for expected in [TraceType::Table, TraceType::StringRef, TraceType::String] {
        let ir = case(Shape { expected, ..GOOD });
        assert!(
            matches!(ir.verify(), Err(TraceVerifyError::TypeMismatch { .. })),
            "{expected:?}"
        );
    }
    // An array binding is no stand-in for the metatable proof.
    let mut ir = case(GOOD);
    ir.bindings[1].kind = BindingKind::Array { first: None };
    assert!(ir.verify().is_err());
}

#[test]
fn the_evaluator_finds_values_by_name_and_misses_as_nil() {
    let ir = case(GOOD);
    let mut world = World {
        names: vec!["a", "b", "c"],
        values: vec![("a", int(5)), ("b", int(7)), ("c", int(11))],
    };
    // Three lookups, then the array ends and the first guard leaves.
    let evaluation = run(&mut world, &ir, 10);
    let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
        panic!("{:?}", evaluation.outcome)
    };
    assert_eq!(*guard, GuardId(0));
    assert_eq!(restored.registers[..2], [int(4), int(23)]);
}

#[test]
fn a_missing_key_or_another_type_leaves_before_the_instruction() {
    let ir = case(GOOD);
    for (value, label) in [
        (None, "missing"),
        (Some(TraceScalar::Float(1.5)), "a float"),
        (Some(TraceScalar::Boolean(true)), "a boolean"),
    ] {
        let mut world = World {
            names: vec!["a", "b"],
            values: vec![("a", int(5))]
                .into_iter()
                .chain(value.map(|v| ("b", v)))
                .collect(),
        };
        let evaluation = run(&mut world, &ir, 10);
        let Outcome::GuardFailed { guard, restored } = &evaluation.outcome else {
            panic!("{label}: {:?}", evaluation.outcome)
        };
        assert_eq!(*guard, GuardId(1), "{label}");
        assert_eq!(restored.pc, 2, "{label}");
        // One lookup passed; the failing one changed nothing.
        assert_eq!(restored.registers[..2], [int(2), int(5)], "{label}");
        assert_eq!(restored.effect_position, 0);
    }
}

#[test]
fn a_nil_lookup_passes_exactly_when_the_key_is_missing() {
    let ir = case(Shape {
        expected: TraceType::Nil,
        ..GOOD
    });
    let mut world = World {
        names: vec!["x", "y"],
        values: vec![("z", int(1))],
    };
    let evaluation = run(&mut world, &ir, 10);
    let Outcome::GuardFailed { guard, .. } = &evaluation.outcome else {
        panic!("{:?}", evaluation.outcome)
    };
    // Both names are missing, so the lookups pass; the array ends.
    assert_eq!(*guard, GuardId(0));
    let mut world = World {
        names: vec!["x", "y"],
        values: vec![("y", int(1))],
    };
    let evaluation = run(&mut world, &ir, 10);
    let Outcome::GuardFailed { guard, .. } = &evaluation.outcome else {
        panic!("{:?}", evaluation.outcome)
    };
    assert_eq!(*guard, GuardId(1));
}
