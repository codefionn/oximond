//! Borrowed strings in the trace IR: the verifier's lifetime contract and
//! what the optimizer keeps of it.
//!
//! A borrowed string ([`TraceType::StringRef`]) is the address of the table
//! slot that holds a string. Any operation for which
//! [`TraceOp::ends_borrows`] holds may replace that slot, so no borrow
//! defined before one may be used after it.

use super::ir::*;
use super::optimize::{OptimizeLevel, optimize_trace};
use super::snapshot::*;
use super::verify::TraceVerifyError;

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

fn root() -> TraceFrameInfo {
    TraceFrameInfo {
        register_count: 3,
        code_len: 32,
        inlined: None,
        activation: None,
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

/// Where the table set sits relative to the borrow, and what uses the borrow
/// after it.
// The `Set` prefix names the store each case places; the cases differ only in
// where that store sits, so it belongs in every variant.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, PartialEq)]
enum Order {
    /// `k = t[i]; t[i] = i; guard (snapshot holds k)`.
    SetAfterLoadThenUse,
    /// `t[i] = i; k = t[i]; guard (snapshot holds k)`.
    SetBeforeLoad,
    /// `k = t[i]; t[i] = i` with no use of `k` after: only the set's own
    /// guard snapshot mentions it.
    SetOwnSnapshotOnly,
}

/// `r0` is an integer index, `r1` the table, `r2` the string.
// The single letters follow the names in the case listings (`k = t[i]`).
#[allow(clippy::many_single_char_names)]
fn case(order: Order) -> TraceIr {
    use RegisterEntry::{Const, Unchanged, Value};
    let mut b = TraceBuilder::new(TraceId(7), root());
    b.mark_written(FrameId(0), 0);
    b.mark_written(FrameId(0), 2);
    let i_in = b.input(0, TraceType::Integer);
    let table = b.input(1, TraceType::Table);
    b.set_bindings(vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }]);
    let i = b.loop_param(TraceType::Integer, LoopParamInit::Input(i_in));
    let nil = RegisterEntry::Const(TraceScalar::Nil);
    let _ = Const;

    let load = |b: &mut TraceBuilder, pc: usize| {
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            b.effect_position(),
            vec![Value(i), Unchanged, nil],
        ));
        let g = b.guard(site(pc), s);
        b.emit(
            site(pc),
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key: i,
                expected: TraceType::StringRef,
                past_end: false,
                guard: g,
            },
            Some(TraceType::StringRef),
        )
        .unwrap()
    };
    let set = |b: &mut TraceBuilder, pc: usize, k: Option<ValueId>| {
        let held = k.map_or(nil, Value);
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc },
            b.effect_position(),
            vec![Value(i), Unchanged, held],
        ));
        let g = b.guard(site(pc), s);
        let effect = b.next_effect();
        b.emit(
            site(pc),
            TraceOp::ArraySet {
                binding: 0,
                table,
                key: i,
                value: i,
                effect,
                guard: g,
            },
            None,
        );
    };
    let k = match order {
        Order::SetBeforeLoad => {
            set(&mut b, 1, None);
            load(&mut b, 2)
        }
        Order::SetAfterLoadThenUse | Order::SetOwnSnapshotOnly => {
            let k = load(&mut b, 1);
            set(&mut b, 2, Some(k));
            k
        }
    };
    let one = b
        .emit(
            site(3),
            TraceOp::Constant(TraceScalar::Integer(1)),
            Some(TraceType::Integer),
        )
        .unwrap();
    let i_next = b
        .emit(
            site(3),
            TraceOp::Arithmetic {
                op: ArithmeticOp::Add,
                left: i,
                right: one,
            },
            Some(TraceType::Integer),
        )
        .unwrap();
    if order != Order::SetOwnSnapshotOnly {
        let cond = b
            .emit(
                site(4),
                TraceOp::Compare {
                    comparison: crate::jit_ir::Comparison::LessThan,
                    left: i_next,
                    right: one,
                },
                Some(TraceType::Boolean),
            )
            .unwrap();
        // A guard after the set, whose snapshot restores the borrow.
        let s = b.snapshot(snap(
            SnapshotPhase::After {
                pc: 4,
                successor: 9,
            },
            b.effect_position(),
            vec![Value(i_next), Unchanged, Value(k)],
        ));
        let g = b.guard(site(4), s);
        b.emit(
            site(4),
            TraceOp::GuardBranch {
                condition: cond,
                expected: false,
                guard: g,
            },
            None,
        );
    }
    b.finish(TraceTerminator::LoopBack { args: vec![i_next] })
}

#[test]
fn a_borrow_used_after_a_table_set_is_refused() {
    let ir = case(Order::SetAfterLoadThenUse);
    let error = ir.verify().expect_err("the borrow outlives the set");
    assert!(
        matches!(error, TraceVerifyError::BorrowOutlivesEffect { .. }),
        "{error}"
    );
}

#[test]
fn a_borrow_taken_after_the_set_is_fine() {
    let ir = case(Order::SetBeforeLoad);
    ir.verify().unwrap();
    assert!(ir.body.iter().any(|instruction| matches!(
        instruction.op,
        TraceOp::ArrayLoad {
            expected: TraceType::StringRef,
            ..
        }
    )));
}

#[test]
fn the_ending_operations_own_guard_may_still_name_the_borrow() {
    // The set leaves through a `Before` snapshot having stored nothing, so
    // the slot is intact there.
    case(Order::SetOwnSnapshotOnly).verify().unwrap();
}

#[test]
fn the_optimizer_keeps_a_verified_borrow_trace_verified() {
    for order in [Order::SetBeforeLoad, Order::SetOwnSnapshotOnly] {
        let ir = case(order);
        for level in [OptimizeLevel::Baseline, OptimizeLevel::Optimized] {
            let (optimized, _) = optimize_trace(&ir, level);
            optimized.verify().unwrap();
            assert!(optimized.body.iter().any(|instruction| matches!(
                instruction.op,
                TraceOp::ArrayLoad {
                    expected: TraceType::StringRef,
                    ..
                }
            )));
        }
    }
}

#[test]
fn only_operations_that_may_move_a_slot_end_borrows() {
    let ir = case(Order::SetBeforeLoad);
    let ends: Vec<bool> = ir
        .body
        .iter()
        .map(|instruction| instruction.op.ends_borrows())
        .collect();
    // set (ends), load, constant, add, compare, branch guard.
    assert_eq!(ends, vec![true, false, false, false, false, false]);
    for opaque in [
        TraceOp::Read {
            label: 0,
            operands: Vec::new(),
            result: TraceType::Any,
        },
        TraceOp::Effect {
            effect: EffectId(0),
            label: 0,
            operands: Vec::new(),
            result: None,
        },
    ] {
        assert!(opaque.ends_borrows());
    }
}

#[test]
fn a_borrow_has_no_loop_parameter_or_constant() {
    // A borrowed entry input is the side trace's: its parent hands it the
    // address of the slot it holds. Use of it is checked like any borrow (see
    // `a_borrowed_entry_input_ends_at_the_first_operation_that_ends_borrows`).
    let mut b = TraceBuilder::new(TraceId(8), root());
    let borrowed = b.input(0, TraceType::StringRef);
    let ir = b.finish(TraceTerminator::LoopBack { args: Vec::new() });
    let _ = borrowed;
    ir.verify().unwrap();
    // A constant.
    let mut b = TraceBuilder::new(TraceId(9), root());
    b.emit(
        site(1),
        TraceOp::Constant(TraceScalar::StringRef(16)),
        Some(TraceType::StringRef),
    )
    .unwrap();
    let ir = b.finish(TraceTerminator::LoopBack { args: Vec::new() });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
    // A loop parameter.
    let mut b = TraceBuilder::new(TraceId(10), root());
    let init = b.input(0, TraceType::String);
    let param = b.loop_param(TraceType::StringRef, LoopParamInit::Input(init));
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![param] });
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::InvalidLoopParam { .. })
    ));
}

#[test]
fn a_borrow_takes_no_comparison() {
    let mut ir = case(Order::SetBeforeLoad);
    let k = ir.body[1].result.unwrap();
    let TraceOp::Compare { left, right, .. } = &mut ir.body[4].op else {
        panic!("the case compares at body 4")
    };
    (*left, *right) = (k, k);
    assert!(matches!(
        ir.verify(),
        Err(TraceVerifyError::TypeMismatch { .. })
    ));
}

#[test]
fn a_borrowed_entry_input_ends_at_the_first_operation_that_ends_borrows() {
    use RegisterEntry::{Unchanged, Value};
    // A side trace takes the key as a borrowed input, stores into a table
    // (which may replace the slot) and then names the key in a guard.
    for (use_after, ok) in [(false, true), (true, false)] {
        let mut b = TraceBuilder::new(TraceId(11), root());
        b.mark_written(FrameId(0), 0);
        let i = b.input(0, TraceType::Integer);
        let table = b.input(1, TraceType::Table);
        let key = b.input(2, TraceType::StringRef);
        b.set_bindings(vec![TableBinding {
            register: 1,
            kind: BindingKind::Array { first: None },
        }]);
        let s = b.snapshot(snap(
            SnapshotPhase::Before { pc: 1 },
            b.effect_position(),
            vec![Value(i), Unchanged, Value(key)],
        ));
        let g = b.guard(site(1), s);
        let effect = b.next_effect();
        b.emit(
            site(1),
            TraceOp::ArraySet {
                binding: 0,
                table,
                key: i,
                value: i,
                effect,
                guard: g,
            },
            None,
        );
        let held = if use_after { Value(key) } else { Unchanged };
        let after = b.snapshot(snap(
            SnapshotPhase::After {
                pc: 2,
                successor: 3,
            },
            b.effect_position(),
            vec![Value(i), Unchanged, held],
        ));
        let ir = b.finish(TraceTerminator::Exit { snapshot: after });
        match (ir.verify(), ok) {
            (Ok(()), true) => {}
            (Err(TraceVerifyError::BorrowOutlivesEffect { .. }), false) => {}
            (other, _) => panic!("use_after {use_after}: {other:?}"),
        }
    }
}
