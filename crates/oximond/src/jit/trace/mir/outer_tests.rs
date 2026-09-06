//! Outer roots and loop sides whose parameters have no register, against
//! Cranelift.
//!
//! An outer root (see "Outer roots" in `vm::jit::trace`) ends in front of an
//! inner loop with a terminal exit instead of looping: the inner loop is a
//! loop side, and the side of its closing guard runs the rest of the outer
//! iteration and links back to the root's header (`TraceTerminator::Link`).
//! The tree is
//!
//! ```text
//! pc0   acc = acc + i                 (the root; leaves when acc >= cap)
//! pc1   j = 0
//! pc10  acc = acc + j                 (the loop side, the inner loop)
//! pc11  j = j + 1; j < m loops to pc10
//! pc12  i = i + 1; i <= n loops to pc0  (the side of the closing guard)
//! ```
//!
//! Registers: r0 `acc`, r1 `j`, r2 `i`, r3 nil. With `const_start` the loop
//! side's `j` starts from the constant 0 rather than from its entry input,
//! so that parameter has no register, as a loop side in a call's frames has
//! for the registers of those frames (see "Loops in calls").

use super::tests::{TIERS, compile, float, same_everywhere};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{
    ArithmeticOp, FrameId, LoopParamInit, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase,
    TERMINAL_GUARD, TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar,
    TraceSide, TraceSite, TraceTerminator, TraceType, ValueId,
};

const OUTER: usize = 0;
const HEADER: usize = 10;
const LATCH: usize = 11;
const AFTER: usize = 12;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn frame() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 4,
        code_len: 16,
        inlined: None,
        activation: None,
    }
}

fn snapshot(phase: SnapshotPhase, last: usize, registers: Vec<RegisterEntry>) -> Snapshot {
    Snapshot::new(
        phase,
        0,
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(last),
            registers: registers.into(),
        }],
    )
}

fn after(pc: usize, successor: usize) -> SnapshotPhase {
    SnapshotPhase::After { pc, successor }
}

fn arithmetic(
    b: &mut TraceBuilder,
    pc: usize,
    op: ArithmeticOp,
    left: ValueId,
    right: ValueId,
) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Arithmetic { op, left, right },
        Some(TraceType::Float),
    )
    .unwrap()
}

fn constant(b: &mut TraceBuilder, pc: usize, value: f64) -> ValueId {
    b.emit(
        site(pc),
        TraceOp::Constant(float(value)),
        Some(TraceType::Float),
    )
    .unwrap()
}

/// A guard at `pc` that holds while `left comparison right` is `expected`,
/// leaving with `registers`, resumed at `successor`.
#[allow(clippy::too_many_arguments)]
fn guard(
    b: &mut TraceBuilder,
    pc: usize,
    comparison: Comparison,
    left: ValueId,
    right: ValueId,
    successor: usize,
    registers: Vec<RegisterEntry>,
) {
    let condition = b
        .emit(
            site(pc),
            TraceOp::Compare {
                comparison,
                left,
                right,
            },
            Some(TraceType::Boolean),
        )
        .unwrap();
    let snap = b.snapshot(snapshot(after(pc, successor), pc, registers));
    let id = b.guard(site(pc), snap);
    b.emit(
        site(pc),
        TraceOp::GuardBranch {
            condition,
            expected: true,
            guard: id,
        },
        None,
    );
}

/// The root: `acc = acc + i` while `acc < cap`, then `j = 0` and a terminal
/// exit in front of the inner loop. It keeps the parameters of its header
/// and a poll snapshot of that header's shape, as an outer root does; no run
/// takes the poll.
fn outer_root(cap: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(0), frame());
    b.mark_written(FrameId(0), 0);
    let acc_in = b.input(0, TraceType::Float);
    let i_in = b.input(2, TraceType::Float);
    let acc = b.loop_param(TraceType::Float, LoopParamInit::Input(acc_in));
    let i = b.loop_param(TraceType::Float, LoopParamInit::Input(i_in));
    let acc2 = arithmetic(&mut b, OUTER, ArithmeticOp::Add, acc, i);
    let top = constant(&mut b, OUTER, cap);
    guard(
        &mut b,
        OUTER,
        Comparison::LessThan,
        acc2,
        top,
        OUTER + 1,
        vec![
            RegisterEntry::Value(acc2),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    );
    let poll = b.snapshot(snapshot(
        after(AFTER, OUTER),
        AFTER,
        vec![
            RegisterEntry::Value(acc2),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    b.set_poll_snapshot(poll);
    let exit = b.snapshot(snapshot(
        after(OUTER + 1, HEADER),
        OUTER + 1,
        vec![
            RegisterEntry::Value(acc2),
            RegisterEntry::Const(float(0.0)),
            RegisterEntry::Value(i),
            RegisterEntry::Const(TraceScalar::Nil),
        ],
    ));
    b.finish(TraceTerminator::Exit { snapshot: exit })
}

/// The inner loop: `acc = acc + j; j = j + 1` while `j < m`. With
/// `const_start`, `j` starts from the constant 0, a parameter without a
/// register.
fn inner_loop(m: f64, const_start: bool) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(1), frame());
    for register in [0, 1] {
        b.mark_written(FrameId(0), register);
    }
    let acc_in = b.input(0, TraceType::Float);
    let j_in = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let acc = b.loop_param(TraceType::Float, LoopParamInit::Input(acc_in));
    let j = b.loop_param(
        TraceType::Float,
        if const_start {
            LoopParamInit::Const(float(0.0))
        } else {
            LoopParamInit::Input(j_in)
        },
    );
    let acc2 = arithmetic(&mut b, HEADER, ArithmeticOp::Add, acc, j);
    let one = constant(&mut b, LATCH, 1.0);
    let j2 = arithmetic(&mut b, LATCH, ArithmeticOp::Add, j, one);
    let top = constant(&mut b, LATCH, m);
    let state = vec![
        RegisterEntry::Value(acc2),
        RegisterEntry::Value(j2),
        RegisterEntry::Value(i),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    guard(
        &mut b,
        LATCH,
        Comparison::LessThan,
        j2,
        top,
        AFTER,
        state.clone(),
    );
    let poll = b.snapshot(snapshot(after(LATCH, HEADER), LATCH, state));
    b.set_poll_snapshot(poll);
    b.finish(TraceTerminator::LoopBack {
        args: vec![acc2, j2],
    })
}

/// The side of the inner loop's closing guard: `i = i + 1`, and back to the
/// outer header while `i <= n`.
fn closing_side(n: f64) -> TraceIr {
    let mut b = TraceBuilder::new(TraceId(2), frame());
    b.mark_written(FrameId(0), 2);
    let acc = b.input(0, TraceType::Float);
    let j = b.input(1, TraceType::Float);
    let i = b.input(2, TraceType::Float);
    let one = constant(&mut b, AFTER, 1.0);
    let i2 = arithmetic(&mut b, AFTER, ArithmeticOp::Add, i, one);
    let top = constant(&mut b, AFTER, n);
    let state = vec![
        RegisterEntry::Value(acc),
        RegisterEntry::Value(j),
        RegisterEntry::Value(i2),
        RegisterEntry::Const(TraceScalar::Nil),
    ];
    guard(
        &mut b,
        AFTER,
        Comparison::LessEqual,
        i2,
        top,
        AFTER + 1,
        state.clone(),
    );
    let link = b.snapshot(snapshot(after(AFTER, OUTER), AFTER, state));
    b.finish(TraceTerminator::Link { snapshot: link })
}

/// The root and its loop side; with `links` also the side of the loop's
/// closing guard that links back to the root.
fn outer_tree(cap: f64, m: f64, n: f64, const_start: bool, links: bool) -> TraceIr {
    let mut ir = outer_root(cap);
    ir.header_live = vec![0, 2];
    ir.sides.push(TraceSide {
        parent: None,
        guard: TERMINAL_GUARD,
        ir: Box::new(inner_loop(m, const_start)),
    });
    if links {
        ir.sides.push(TraceSide {
            parent: Some(0),
            guard: 0,
            ir: Box::new(closing_side(n)),
        });
    }
    ir
}

fn states() -> Vec<Vec<TraceScalar>> {
    [
        (0.0, 1.0),
        (5.0, 3.0),
        (100.0, 0.0),
        (-40.0, 7.5),
        (1e6, 2.0),
    ]
    .iter()
    .map(|(acc, i)| vec![float(*acc), float(-1.0), float(*i), TraceScalar::Nil])
    .collect()
}

const CASES: [(f64, f64, f64); 4] = [
    // cap, m, n
    (1e9, 5.0, 40.0),
    (1e9, 1.0, 3.0),
    (300.0, 9.0, 1e6),
    (1e9, 30.0, 12.0),
];

#[test]
fn a_side_that_rejoins_a_root_that_does_not_loop_runs_through_mir_like_cranelift() {
    for const_start in [false, true] {
        for fuel in [1_000_000, 500, 41, 7, 2, 1] {
            for (cap, m, n) in CASES {
                let ir = outer_tree(cap, m, n, const_start, true);
                same_everywhere("outer root", &ir, &states(), fuel);
            }
        }
    }
}

#[test]
fn a_loop_side_with_a_parameter_without_a_register_runs_through_mir_like_cranelift() {
    // The optimized loop side still has the parameter, with no register.
    let trace = compile(&outer_tree(1e9, 5.0, 40.0, true, false), TIERS[1], true);
    let params = &trace.lowered_member(1).loop_params;
    assert!(
        params.iter().any(|param| param.register.is_none()),
        "{params:?}"
    );
    for fuel in [1_000_000, 500, 7, 1] {
        for (cap, m, n) in CASES {
            let ir = outer_tree(cap, m, n, true, false);
            same_everywhere("loop side with a constant start", &ir, &states(), fuel);
        }
    }
}

/// The outer iterations run in the tree: the run leaves through the closing
/// side's guard, at the end of the outer loop, with the sums the loops make
/// and the fuel of every inner trip and every link spent.
#[test]
fn the_outer_loop_runs_natively_through_its_links() {
    let (m, n) = (5.0, 40.0);
    for const_start in [false, true] {
        let ir = outer_tree(1e9, m, n, const_start, true);
        for tier in TIERS {
            let trace = compile(&ir, tier, true);
            assert!(trace.mir_address().is_some(), "{tier:?}");
            let registers = [float(0.0), float(-1.0), float(1.0), TraceScalar::Nil];
            let inputs: Vec<_> = trace
                .entry_contract()
                .iter()
                .map(|(register, _)| {
                    super::super::exit::TraceInputSlot::from_scalar(
                        registers[usize::from(*register)],
                    )
                })
                .collect();
            let mut area = vec![0_i64; trace.exit_area_len()];
            let fuel = 1_000_000;
            let mut control = super::super::exit::TraceControl::new(fuel);
            let outcome = trace.run(&inputs, &mut control, &mut area);
            let super::super::exit::TraceOutcome::Exit { index } = outcome else {
                panic!("the run leaves: {outcome:?}");
            };
            assert_eq!(trace.exits()[index].member, 2, "{tier:?}");
            // Each of the n outer iterations makes m - 1 inner loop-backs,
            // and each but the last a link.
            let spent = fuel - control.fuel;
            let (m, n) = (m as i64, n as i64);
            assert_eq!(spent, (m - 1) * n + n - 1, "{tier:?}");
        }
    }
}
