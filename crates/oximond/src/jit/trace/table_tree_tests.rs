//! Trace trees with table operations, native against the reference
//! evaluator.
//!
//! The loop reads an array and branches on what it finds:
//!
//! ```text
//! pc0  r3 = t[r0]                 ArrayLoad, leaves before it
//! pc1  r4 = 0 < r3
//! pc2  JumpIfFalse r4 -> pc5      (recorded: not taken)
//! pc3  r1 = r1 + r3               (the root's path)
//! pc4  r0 = r0 + 1; Jump -> pc0
//! pc5  r3 = u[r0]                 (the side's path: another table)
//! pc6  u[r0] = r3 + 1             ArrayStore, a committed effect
//! pc7  r2 = r2 + r3
//! pc8  r0 = r0 + 1; Jump -> pc0
//! ```
//!
//! Registers: r0 the key, r1 and r2 the two sums, r3 and r4 temporaries, r5
//! the table `t` (which the root binds) and r6 the table `u` (which only the
//! side binds). The loop ends where a key leaves an array: the array guard
//! leaves before the load.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::wildcard_enum_match_arm
)]

use super::*;
use crate::jit::OptimizationTier;
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::reference::{Outcome, TraceHeap, TreeGuard, evaluate_tree_with_heap};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame,
    SnapshotPhase, TableBinding, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp,
    TraceScalar, TraceSide, TraceSite, TraceTerminator, TraceType, TreeError, ValueId,
};
use crate::runtime::Value;

const TIERS: [OptimizationTier; 2] = [OptimizationTier::Baseline, OptimizationTier::Optimized];

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 7,
        code_len: 16,
        inlined: None,
        activation: None,
    }
}

fn snapshot(
    phase: SnapshotPhase,
    effects: u32,
    last: usize,
    registers: Vec<RegisterEntry>,
) -> Snapshot {
    Snapshot::new(
        phase,
        effects,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(last),
            registers: registers.into(),
        }],
    )
}

fn f(value: f64) -> TraceScalar {
    TraceScalar::Float(value)
}

fn constant(b: &mut TraceBuilder, pc: usize, value: f64) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Constant(f(value)),
        Some(TraceType::Float),
    )
    .unwrap()
}

fn add(b: &mut TraceBuilder, pc: usize, left: ValueId, right: ValueId) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left,
            right,
        },
        Some(TraceType::Float),
    )
    .unwrap()
}

fn array(register: u16) -> TableBinding {
    TableBinding {
        register,
        kind: BindingKind::Array { first: None },
    }
}

use RegisterEntry::{Const, Unchanged, Value as Val};

const NIL: RegisterEntry = Const(TraceScalar::Nil);

/// The root: the path that adds a positive element to `pos`.
fn root_loop() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    for register in [0, 1, 3, 4] {
        b.mark_written(FrameId(0), register);
    }
    let i_in = b.input(0, TraceType::Float);
    let pos_in = b.input(1, TraceType::Float);
    let t = b.input(5, TraceType::Table);
    b.set_bindings(vec![array(5)]);
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let pos = b.loop_param(TraceType::Float, LoopParamInit::Input(pos_in));

    let before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 0 },
        0,
        4,
        vec![Val(i), Val(pos), Unchanged, NIL, NIL, Unchanged, Unchanged],
    ));
    let load_guard = b.guard(site(0), before);
    let x = b
        .emit(
            site(0),
            TraceOp::ArrayLoad {
                binding: 0,
                table: t,
                key: i,
                expected: TraceType::Float,
                past_end: false,
                guard: load_guard,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let zero = constant(&mut b, 1, 0.0);
    let cond = b
        .emit(
            site(1),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: zero,
                right: x,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    // The failing direction falls to the side's arm at pc5.
    let branch = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 2,
            successor: 5,
        },
        0,
        2,
        vec![
            Val(i),
            Val(pos),
            Unchanged,
            Val(x),
            Val(cond),
            Unchanged,
            Unchanged,
        ],
    ));
    let branch_guard = b.guard(site(2), branch);
    b.emit(
        site(2),
        TraceOp::GuardBranch {
            condition: cond,
            expected: true,
            guard: branch_guard,
        },
        None,
    );
    let pos2 = add(&mut b, 3, pos, x);
    let one = constant(&mut b, 4, 1.0);
    let i2 = add(&mut b, 4, i, one);
    let poll = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 4,
            successor: 0,
        },
        0,
        4,
        vec![
            Val(i2),
            Val(pos2),
            Unchanged,
            Val(x),
            Val(cond),
            Unchanged,
            Unchanged,
        ],
    ));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![i2, pos2],
    })
}

/// The side: the other arm, which reads and rewrites `u`.
fn other_side() -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame());
    for register in [0, 2, 3] {
        b.mark_written(FrameId(0), register);
    }
    let i = b.input(0, TraceType::Float);
    let pos = b.input(1, TraceType::Float);
    let neg = b.input(2, TraceType::Float);
    let _t = b.input(5, TraceType::Table);
    let u = b.input(6, TraceType::Table);
    // The root's binding first, then the side's own.
    b.set_bindings(vec![array(5), array(6)]);

    let load_before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 5 },
        0,
        2,
        vec![Val(i), Val(pos), Val(neg), NIL, NIL, Unchanged, Unchanged],
    ));
    let load_guard = b.guard(site(5), load_before);
    let y = b
        .emit(
            site(5),
            TraceOp::ArrayLoad {
                binding: 1,
                table: u,
                key: i,
                expected: TraceType::Float,
                past_end: false,
                guard: load_guard,
            },
            Some(TraceType::Float),
        )
        .unwrap();
    let one = constant(&mut b, 6, 1.0);
    let y1 = add(&mut b, 6, y, one);
    let store_before = b.snapshot(snapshot(
        SnapshotPhase::Before { pc: 6 },
        b.effect_position(),
        5,
        vec![
            Val(i),
            Val(pos),
            Val(neg),
            Val(y),
            NIL,
            Unchanged,
            Unchanged,
        ],
    ));
    let store_guard = b.guard(site(6), store_before);
    let effect = b.next_effect();
    b.emit(
        site(6),
        TraceOp::ArrayStore {
            binding: 1,
            table: u,
            key: i,
            value: y1,
            effect,
            guard: store_guard,
        },
        None,
    );
    // Past the store a guard can still fail: the interpreter resumes after
    // the store, which it must not replay.
    let three = constant(&mut b, 7, 3.0);
    let small = b
        .emit(
            site(7),
            TraceOp::Compare {
                comparison: Comparison::LessThan,
                left: y,
                right: three,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let after_store = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 7,
            successor: 9,
        },
        b.effect_position(),
        7,
        vec![
            Val(i),
            Val(pos),
            Val(neg),
            Val(y),
            Val(small),
            Unchanged,
            Unchanged,
        ],
    ));
    let small_guard = b.guard(site(7), after_store);
    b.emit(
        site(7),
        TraceOp::GuardBranch {
            condition: small,
            expected: true,
            guard: small_guard,
        },
        None,
    );
    let neg2 = add(&mut b, 7, neg, y);
    let i2 = add(&mut b, 8, i, one);
    let link = b.snapshot(snapshot(
        SnapshotPhase::After {
            pc: 8,
            successor: 0,
        },
        b.effect_position(),
        8,
        vec![
            Val(i2),
            Val(pos),
            Val(neg2),
            Val(y),
            NIL,
            Unchanged,
            Unchanged,
        ],
    ));
    b.finish(TraceTerminator::Link { snapshot: link })
}

fn tree() -> TraceIr {
    let mut ir = root_loop();
    ir.header_live = vec![0, 1, 2, 5, 6];
    ir.sides.push(TraceSide {
        parent: None,
        guard: 1,
        ir: Box::new(other_side()),
    });
    ir
}

// ------------------------------------------------------------ the model

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A float from a small set that includes both zeros, so every branch
    /// and every sum has its corner.
    fn element(&mut self) -> f64 {
        const SET: [f64; 8] = [-3.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.5, 7.0];
        SET[usize::try_from(self.next() % 8).unwrap()]
    }
}

/// The memory both runs work on: one array per binding of the tree.
#[derive(Clone)]
struct Tables {
    arrays: Vec<Vec<Value>>,
}

impl Tables {
    fn random(rng: &mut Rng, lengths: [usize; 2]) -> Self {
        Self {
            arrays: lengths
                .iter()
                .map(|length| (0..*length).map(|_| Value::Number(rng.element())).collect())
                .collect(),
        }
    }

    fn slot(&self, binding: u32, key: TraceScalar) -> Option<usize> {
        let index = match key {
            TraceScalar::Integer(index) => index,
            TraceScalar::Float(value) if value as i64 as f64 == value => value as i64,
            _ => return None,
        };
        let slot = usize::try_from(index.checked_sub(1)?).ok()?;
        (slot < self.arrays[binding as usize].len()).then_some(slot)
    }
}

impl TraceHeap for Tables {
    fn entry_accepts(&mut self, _: &[TableBinding]) -> bool {
        true
    }
    fn array_get(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let slot = self.slot(binding, key)?;
        match self.arrays[binding as usize][slot] {
            Value::Number(number) => Some(f(number)),
            _ => None,
        }
    }
    fn array_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool {
        let Some(slot) = self.slot(binding, key) else {
            return false;
        };
        let TraceScalar::Float(number) = value else {
            return false;
        };
        self.arrays[binding as usize][slot] = Value::Number(number);
        true
    }
    fn field_get(&mut self, _: u32) -> TraceScalar {
        unreachable!("no fields here")
    }
    fn field_set(&mut self, _: u32, _: TraceScalar) {
        unreachable!("no fields here")
    }
    fn length(&mut self, _: u32) -> TraceScalar {
        unreachable!("no lengths here")
    }
}

fn bits(values: &[Value]) -> Vec<u64> {
    values
        .iter()
        .map(|value| match value {
            Value::Number(number) => number.to_bits(),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn scalar_bits(scalars: &[TraceScalar]) -> Vec<String> {
    scalars
        .iter()
        .map(|scalar| match scalar {
            TraceScalar::Float(number) => format!("f{:016x}", number.to_bits()),
            other => format!("{other:?}"),
        })
        .collect()
}

fn registers(i: f64, pos: f64, neg: f64) -> [TraceScalar; 7] {
    [
        f(i),
        f(pos),
        f(neg),
        TraceScalar::Nil,
        TraceScalar::Nil,
        TraceScalar::Table(0),
        TraceScalar::Table(1),
    ]
}

/// What a native run of `trace` left: the exit's member, the restored
/// registers, the fuel left, and the memory.
struct Native {
    member: usize,
    pc: usize,
    effect_position: u32,
    state: Vec<TraceScalar>,
    fuel_left: i64,
    tables: Tables,
}

fn run_native(trace: &CompiledTrace, start: &[TraceScalar], tables: &Tables, fuel: i64) -> Native {
    let mut native = tables.clone();
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(start[usize::from(*register)]))
        .collect();
    let bindings: Vec<TraceBinding> = trace
        .bindings()
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            assert!(matches!(binding.kind, BindingKind::Array { .. }));
            TraceBinding::raw_array(
                native.arrays[index].len(),
                native.arrays[index].as_mut_ptr() as usize,
            )
        })
        .collect();
    let mut area = vec![0_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    let outcome = trace.run_bound(&inputs, &bindings, &mut control, &mut area);
    let TraceOutcome::Exit { index } = outcome else {
        panic!("the tree refused entry");
    };
    let descriptor = &trace.exits()[index];
    let member = trace.lowered_member(descriptor.member);
    let snapshot = &member.snapshots[descriptor.snapshot.index()];
    let state = snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .map(|(register, entry)| match entry {
            RegisterEntry::Value(value) => descriptor.scalar(&area, *value).unwrap(),
            RegisterEntry::Const(scalar) => *scalar,
            RegisterEntry::Unchanged => start[register],
            RegisterEntry::Root(_) => unreachable!("no roots here"),
            RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => {
                unreachable!("no virtual tables here")
            }
        })
        .collect();
    Native {
        member: descriptor.member,
        pc: snapshot.resume_point().pc,
        effect_position: snapshot.effect_position,
        state,
        fuel_left: control.fuel,
        tables: native,
    }
}

fn agree(
    ir: &TraceIr,
    trace: &CompiledTrace,
    start: &[TraceScalar],
    tables: &Tables,
    forced: Option<TreeGuard>,
    fuel: i64,
    label: &str,
) -> Native {
    let native = run_native(trace, start, tables, fuel);
    let mut model = tables.clone();
    let evaluation = evaluate_tree_with_heap(
        ir,
        start,
        &[],
        u32::try_from(fuel).unwrap_or(u32::MAX),
        forced,
        &mut model,
    );
    let restored = match evaluation.outcome {
        Outcome::GuardFailed { restored, .. } | Outcome::Exited(restored) => restored,
        Outcome::IterationLimit { .. } => {
            // Fuel ran out in the reference: native code leaves through a
            // poll exit. Nothing to compare but the memory so far.
            panic!("{label}: the reference ran out of iterations first");
        }
        other => panic!("{label}: unexpected reference outcome {other:?}"),
    };
    assert_eq!(native.pc, restored.pc, "{label}");
    assert_eq!(native.effect_position, restored.effect_position, "{label}");
    assert_eq!(
        scalar_bits(&native.state),
        scalar_bits(&restored.registers),
        "{label}"
    );
    for (index, (left, right)) in native.tables.arrays.iter().zip(&model.arrays).enumerate() {
        assert_eq!(bits(left), bits(right), "{label}: array {index} afterwards");
    }
    native
}

// ------------------------------------------------------------- the tests

#[test]
fn the_planned_tree_binds_every_table_and_gives_each_member_an_input_for_it() {
    let members = crate::jit_ir::trace::plan_tree(&tree()).unwrap();
    assert_eq!(members.len(), 2);
    for member in &members {
        assert_eq!(member.ir.bindings, vec![array(5), array(6)]);
        for register in [5, 6] {
            assert!(
                member
                    .ir
                    .entry
                    .inputs
                    .iter()
                    .any(|input| input.register == register && input.ty == TraceType::Table),
                "r{register} in member {:?}",
                member.parent
            );
        }
        member.ir.verify().unwrap();
    }
    // `neg` (r2) became a loop parameter of the root: only the side writes it.
    assert!(
        members[0]
            .ir
            .loop_params
            .iter()
            .any(|param| param.register == Some(2))
    );
}

#[test]
fn a_tree_over_tables_compiles_and_agrees_with_the_reference() {
    let ir = tree();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut left_by_side = 0;
    let mut left_by_root = 0;
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        assert_eq!(trace.member_count(), 2, "{tier:?}");
        assert_eq!(trace.bindings().len(), 2);
        for round in 0..400 {
            let tables = Tables::random(&mut rng, [1 + round % 17, 1 + round % 13]);
            let start = registers(
                f64::from(u32::try_from(1 + round % 5).unwrap()),
                rng.element(),
                rng.element(),
            );
            let native = agree(
                &ir,
                &trace,
                &start,
                &tables,
                None,
                1_000_000,
                &format!("{tier:?} round {round}"),
            );
            if native.member == 0 {
                left_by_root += 1;
            } else {
                left_by_side += 1;
            }
        }
    }
    // The loop ends where a key leaves an array: sometimes in the root and
    // sometimes after the side took over.
    assert!(left_by_root > 100, "{left_by_root}");
    assert!(left_by_side > 100, "{left_by_side}");
}

#[test]
fn forcing_any_guard_of_a_table_tree_exits_like_the_reference_forcing_it() {
    let ir = tree();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut forced_exits = 0;
    let mut stores_seen = 0;
    for tier in TIERS {
        let plain = compile_trace(&ir, tier).unwrap();
        let guards: Vec<(Option<usize>, u32, usize)> = ir
            .guards
            .iter()
            .map(|info| (None, info.origin, 0))
            .chain(
                ir.sides[0]
                    .ir
                    .guards
                    .iter()
                    .map(|info| (Some(0), info.origin, 0)),
            )
            .collect();
        assert_eq!(guards.len(), 5, "two in the root, three in the side");
        for (side, origin, _) in guards {
            let member = match side {
                None => Some(0),
                Some(index) => plain.side_members()[index],
            };
            let Some(member) = member else { continue };
            let Some(guard) = plain
                .lowered_member(member)
                .guards
                .iter()
                .position(|info| info.origin == origin)
            else {
                continue;
            };
            let forced = compile_trace_forcing_member_guard(&ir, tier, member, guard).unwrap();
            for round in 0..60 {
                let tables = Tables::random(&mut rng, [4 + round % 9, 4 + round % 7]);
                let start = registers(
                    f64::from(u32::try_from(1 + round % 3).unwrap()),
                    rng.element(),
                    rng.element(),
                );
                let native = agree(
                    &ir,
                    &forced,
                    &start,
                    &tables,
                    Some(TreeGuard { side, origin }),
                    1_000_000,
                    &format!("{tier:?} forcing {side:?}/{origin} round {round}"),
                );
                forced_exits += 1;
                if native.effect_position > 0 {
                    stores_seen += 1;
                }
            }
        }
    }
    assert!(forced_exits > 400, "{forced_exits}");
    // Some forced exits came after the side's store committed.
    assert!(stores_seen > 20, "{stores_seen}");
}

#[test]
fn a_poll_inside_the_side_publishes_the_state_the_loop_resumes_from() {
    // Poll every few loop-backs and links, restarting from what each poll
    // restored, and check that the last exit and the memory are the same as
    // one uninterrupted run.
    let ir = tree();
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    for tier in TIERS {
        let trace = compile_trace(&ir, tier).unwrap();
        for round in 0..40 {
            let tables = Tables::random(&mut rng, [12 + round % 7, 12 + round % 5]);
            let start = registers(1.0, 0.0, 0.0);
            let whole = run_native(&trace, &start, &tables, 1_000_000);
            let mut state = start.to_vec();
            let mut memory = tables.clone();
            let mut polls = 0;
            let last = loop {
                let run = run_native(&trace, &state, &memory, 3);
                memory = run.tables.clone();
                state = run.state.clone();
                if run.fuel_left > 0 {
                    break run;
                }
                polls += 1;
                assert!(polls < 1_000);
            };
            assert_eq!(last.pc, whole.pc, "{tier:?} round {round}");
            assert_eq!(
                scalar_bits(&last.state),
                scalar_bits(&whole.state),
                "{tier:?} round {round}"
            );
            for (left, right) in memory.arrays.iter().zip(&whole.tables.arrays) {
                assert_eq!(bits(left), bits(right), "{tier:?} round {round}");
            }
        }
    }
}

#[test]
fn the_tree_refuses_what_would_bind_the_wrong_table() {
    // A side that writes the register of a table the tree binds: the binding
    // resolved the table the register held at entry. The root only reads
    // `t` by `ArrayLoad`, so the binding demotes to lane reads, and the plan
    // stops at the side's link, which (in this made-up side) never names the
    // register it claims to write.
    let mut ir = tree();
    ir.sides[0].ir.written_registers.push((FrameId(0), 5));
    assert_eq!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(TreeError::LinkValue {
            side: 0,
            register: 5
        })
    );

    // A root that writes the register of a table its side binds.
    let mut ir = tree();
    ir.written_registers.push((FrameId(0), 6));
    assert_eq!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(TreeError::TableWritten { register: 6 })
    );

    // A side recorded against another list of bindings: it was recorded
    // against another version of the tree.
    let mut ir = tree();
    ir.sides[0].ir.bindings[0] = array(6);
    assert_eq!(
        crate::jit_ir::trace::plan_tree(&ir),
        Err(TreeError::BindingMismatch { side: 0 })
    );
}
