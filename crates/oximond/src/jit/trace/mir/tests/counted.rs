//! Counted trees (`TraceIr::counted`, a loop under a count hook) against
//! Cranelift: the budget is a second loop value beside the fuel, paid at each
//! loop-back and link by what the path ran, and every exit stores the budget
//! its iteration began with into the control block.

use super::*;

/// Marks `ir` and its sides counted, with a cost for each snapshot: the poll
/// exit of the root `poll`, the others `3 + 2 * index` in the root and
/// `link` plus the index in a side (the link's own snapshot is `link`).
pub(in super::super) fn counted(mut ir: TraceIr, poll: u32, link: u32) -> TraceIr {
    ir.counted = true;
    let polled = ir
        .poll_snapshot
        .map(crate::jit_ir::trace::SnapshotId::index);
    for (index, snapshot) in ir.snapshots.iter_mut().enumerate() {
        snapshot.counted = if Some(index) == polled {
            poll
        } else {
            3 + 2 * index as u32
        };
    }
    for side in &mut ir.sides {
        side.ir.counted = true;
        let linked = match side.ir.terminator {
            TraceTerminator::Link { snapshot } => Some(snapshot.index()),
            _ => None,
        };
        for (index, snapshot) in side.ir.snapshots.iter_mut().enumerate() {
            snapshot.counted = if Some(index) == linked {
                link
            } else {
                link + 1 + index as u32
            };
        }
    }
    ir
}

/// A run with the count hook's budget at `budget`, and the calls the host
/// answered (the tree of `tree_with_a_hoisted_call` makes some).
fn run_counted(
    trace: &CompiledTrace,
    registers: &[TraceScalar],
    fuel: i64,
    budget: i64,
) -> (Observed, u32) {
    let inputs: Vec<TraceInputSlot> = trace
        .entry_contract()
        .iter()
        .map(|(register, _)| TraceInputSlot::from_scalar(registers[usize::from(*register)]))
        .collect();
    let mut area = vec![0x5a5a_5a5a_i64; trace.exit_area_len()];
    let mut control = TraceControl::new(fuel);
    control.budget = budget;
    let mut space = super::super::super::exit::TraceTableSpace::new();
    let mut host = CountingHost::default();
    let outcome =
        trace.run_with_tables(&inputs, &[], &mut control, &mut area, &mut space, &mut host);
    control.scratch = 0;
    (
        Observed {
            outcome,
            control,
            area,
        },
        host.calls,
    )
}

/// Compiles the counted `ir` with both backends at both tiers, requires MIR
/// to have compiled it, and compares every run from `states` for each fuel
/// and budget.
fn counted_everywhere(name: &str, ir: &TraceIr, states: &[Vec<TraceScalar>], fuels: &[i64]) {
    ir.verify()
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    for tier in TIERS {
        FORCE_BACKEND.with(|force| force.set(Some(true)));
        LAST_FALLBACK.with(|last| last.set(None));
        let mir = compile_trace(ir, tier).unwrap();
        FORCE_BACKEND.with(|force| force.set(None));
        assert!(
            mir.mir_address().is_some(),
            "{name}: MIR compiled it ({:?})",
            LAST_FALLBACK.with(Cell::get)
        );
        let cranelift = compile(ir, tier, false);
        assert!(cranelift.mir_address().is_none());
        assert!(mir.counted() && cranelift.counted(), "{name}");
        assert_eq!(mir.max_counted(), cranelift.max_counted(), "{name}");
        assert_eq!(
            mir.exits(),
            cranelift.exits(),
            "{name}: same exit descriptors"
        );
        for state in states {
            for &fuel in fuels {
                for budget in [1_000_000, 77, 0] {
                    let (a, b) = (
                        run_counted(&mir, state, fuel, budget),
                        run_counted(&cranelift, state, fuel, budget),
                    );
                    assert_eq!(
                        a, b,
                        "{name} at {tier:?} from {state:?}, fuel {fuel}, budget {budget}"
                    );
                }
            }
        }
    }
}

/// The counting loop of `tests.rs`, counted: the exit of the loop's guard
/// has run 3 instructions of its iteration and the poll exit all 4.
#[test]
fn a_counted_loop_pays_for_each_iteration_like_cranelift() {
    let ir = counted(counting_loop(), 4, 0);
    let states: Vec<Vec<TraceScalar>> = [(0.0, 10.0), (0.0, 1.0e9), (5.0, 3.0), (f64::NAN, 4.0)]
        .iter()
        .map(|(x, limit)| vec![float(*x), float(*limit)])
        .collect();
    counted_everywhere(
        "counted loop",
        &ir,
        &states,
        &[1, 2, 3, 5, 11, 12, 64, 1_000],
    );
    // The numbers themselves: ten iterations complete and the eleventh leaves
    // through the guard; the poll leaves after `fuel` of them, reporting the
    // budget the last iteration began with.
    let mir = compile(&ir, OPTIMIZED, true);
    assert!(mir.mir_address().is_some());
    let (leaves, _) = run_counted(&mir, &[float(0.0), float(10.0)], 1_000, 1_000);
    assert!(matches!(leaves.outcome, TraceOutcome::Exit { .. }));
    assert_eq!(leaves.control.budget, 1_000 - 10 * 4);
    for fuel in [1_i64, 2, 5, 64] {
        let (polled, _) = run_counted(&mir, &[float(0.0), float(1.0e9)], fuel, 1_000_000);
        assert_eq!(
            polled.control.budget,
            1_000_000 - (fuel - 1) * 4,
            "fuel {fuel}"
        );
        assert_eq!(polled.control.fuel, 0);
    }
}

/// A counted tree whose side leaves through an exit of its own: the side's
/// exits report the budget of the iteration that entered it.
#[test]
fn a_counted_tree_with_a_leaving_side_runs_through_mir_like_cranelift() {
    let ir = counted(tree_with_a_leaving_side(), 6, 9);
    counted_everywhere(
        "counted tree with a leaving side",
        &ir,
        &side_states(),
        &[1, 2, 3, 10_000],
    );
}

/// A counted tree whose side links back to the loop, paying a cost of its own
/// (9 against the root's 6), so the budget after a run depends on which path
/// each iteration took.
#[test]
fn a_counted_tree_with_a_rejoining_side_runs_through_mir_like_cranelift() {
    for (poll, link) in [(6, 9), (5, 5), (1, 30)] {
        let ir = counted(tree_with_a_rejoining_side(), poll, link);
        counted_everywhere(
            &format!("counted tree with a rejoining side ({poll}, {link})"),
            &ir,
            &side_states(),
            &[1, 2, 3, 4, 7, 10, 10_000],
        );
    }
    // A root whose loop passes a parameter on unchanged, so its `Join` sits
    // above the preheader, and a root with a constant start.
    let ir = counted(tree_with_a_hoisted_call(true), 7, 4);
    counted_everywhere(
        "counted tree joined above its preheader",
        &ir,
        &side_states(),
        &[1, 2, 7, 10_000],
    );
    let ir = counted(tree_with_a_constant_start(), 7, 11);
    let states: Vec<Vec<TraceScalar>> = [20.0, 40.0, 99.0, f64::NAN]
        .iter()
        .flat_map(|limit| {
            [1.0, 2.0, 4.0, 50.5]
                .map(|start| vec![float(*limit), float(6.0), float(start), TraceScalar::Nil])
        })
        .collect();
    counted_everywhere(
        "counted tree with a constant start",
        &ir,
        &states,
        &[1, 2, 7, 10_000],
    );
}

fn side_states() -> Vec<Vec<TraceScalar>> {
    [1.0, 2.0, 3.0, 4.0, 8.0, 50.5, 99.0, 100.0, f64::NAN]
        .iter()
        .flat_map(|start| {
            [0.0, 5.0].map(|yes| vec![float(yes), float(6.0), float(*start), TraceScalar::Nil])
        })
        .collect()
}

/// A tree that is not counted still leaves the budget alone under MIR.
#[test]
fn a_tree_that_is_not_counted_leaves_the_budget_alone_under_mir() {
    let ir = tree_with_a_rejoining_side();
    let mir = compile(&ir, OPTIMIZED, true);
    assert!(mir.mir_address().is_some());
    assert!(!mir.counted());
    for state in side_states() {
        let (observed, _) = run_counted(&mir, &state, 10_000, 77);
        assert_eq!(observed.control.budget, 77);
    }
}
