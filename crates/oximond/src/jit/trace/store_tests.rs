//! Which traces need the VM's store entry, and so refuse to run without one.

use super::compile::member_stores_references;
use crate::jit_ir::trace::{
    BindingKind, FrameId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase, TableBinding,
    TraceBuilder, TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite,
    TraceTerminator, TraceType, verify_trace,
};

fn site(pc: usize) -> TraceSite {
    TraceSite {
        frame: FrameId(0),
        pc,
    }
}

/// `t[1] = <constant>` over a table with an array binding and, when `header`
/// is given, a header binding on the same register, looping for ever.
pub(in crate::jit::trace) fn set_case(header: Option<BindingKind>, value: TraceScalar) -> TraceIr {
    let mut b = TraceBuilder::new(
        TraceId(1),
        TraceFrameInfo {
            register_count: 2,
            code_len: 8,
            inlined: None,
            activation: None,
        },
    );
    let table = b.input(1, TraceType::Table);
    let mut bindings = vec![TableBinding {
        register: 1,
        kind: BindingKind::Array { first: None },
    }];
    bindings.extend(header.map(|kind| TableBinding { register: 1, kind }));
    b.set_bindings(bindings);
    let snapshot = b.snapshot(Snapshot::new(
        SnapshotPhase::Before { pc: 1 },
        b.effect_position(),
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(0),
            registers: vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged].into(),
        }],
    ));
    let guard = b.guard(site(1), snapshot);
    let key = b
        .emit(
            site(1),
            TraceOp::Constant(TraceScalar::Integer(1)),
            Some(TraceType::Integer),
        )
        .unwrap();
    let ty = value.ty();
    let value = b.emit(site(1), TraceOp::Constant(value), Some(ty)).unwrap();
    let effect = b.next_effect();
    b.emit(
        site(1),
        TraceOp::ArraySet {
            binding: 0,
            table,
            key,
            value,
            effect,
            guard,
        },
        None,
    );
    // The loop's poll exit, after the set.
    let poll = b.snapshot(Snapshot::new(
        SnapshotPhase::After {
            pc: 1,
            successor: 0,
        },
        b.effect_position(),
        vec![SnapshotFrame {
            frame: FrameId(0),
            last_interpreted_pc: Some(1),
            registers: vec![RegisterEntry::Unchanged, RegisterEntry::Unchanged].into(),
        }],
    ));
    b.set_poll_snapshot(poll);
    let ir = b.finish(TraceTerminator::LoopBack { args: vec![] });
    verify_trace(&ir).expect("the case is a shape the verifier accepts");
    ir
}

#[test]
fn a_nil_pop_of_a_header_bound_array_needs_the_store_entry() {
    let nil = TraceScalar::Nil;
    assert!(member_stores_references(&set_case(
        Some(BindingKind::Header { strict: false }),
        nil
    )));
    assert!(member_stores_references(&set_case(
        Some(BindingKind::Header { strict: true }),
        nil
    )));
    // An array binding alone stores a nil in place.
    assert!(!member_stores_references(&set_case(None, nil)));
    // A scalar into a header-bound array does not call the entry.
    assert!(!member_stores_references(&set_case(
        Some(BindingKind::Header { strict: false }),
        TraceScalar::Integer(3)
    )));
}
