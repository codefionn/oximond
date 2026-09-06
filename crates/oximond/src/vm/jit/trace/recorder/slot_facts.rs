//! What the recording knows about array slots it read or stored.
//!
//! Mineclonia keeps its 64-bit RNG state as `{ lo, hi }` pairs, and every
//! helper that `random.lua` builds for `LuaJIT` reads both halves of each
//! operand and stores both halves of its result (`a[1] = tonumber (band
//! (value, 0xffffffffull))`). A chain of those calls (`addull (lo1,
//! seedhi); rotlull (lo1, 17); addull (lo1, seedlo)`) reads back the halves
//! the call before it stored, and each read was a guard. The optimizer
//! forwarded most of those loads later, but the recording had paid for
//! their guards against `maxsnap` by then.
//!
//! A slot fact says that slot `key` of the table `table` (the operand the
//! access named: a bound window or a lane) holds `value`, of type `ty`, and
//! that an access in this iteration found it inside the array part with a
//! non-nil value of that type. The read or store that established it took
//! its guard. A later read of a slot with a fact reads nothing: it is a
//! `Move` of the value (store-to-load forwarding, and a second read of a
//! slot is the first), with no guard and no snapshot.
//!
//! A store into a slot with a fact of the stored type cannot fail its
//! guard either: the slot is inside the array and holds a non-nil value of
//! that type, so `__newindex` cannot run whatever metatable the table has.
//! It is an [`RecKind::ArrayOverwrite`] (`TraceOp::ArrayOverwrite` in the
//! IR), a payload store with no guard and no snapshot. The verifier checks
//! the proof on the IR itself (`verify::overwrite_proven`): an earlier
//! `ArrayLoad`, `ArrayStore` or overwrite of the same binding, table and key
//! with the stored type, and only operations that keep every array slot in
//! between (`TraceOp::keeps_array_slots`, the same kills as below). Native
//! code stores at the slot address an earlier access computed, or checks
//! the index and tag again and traps on a failure the proof rules out.
//!
//! Before the overwrites, on the 512-block Mineclonia emerge the recordings
//! of `get_structure_starts` (structures.lua:744) forwarded 144 reads and
//! still reached `maxsnap`, at 3360 guest instructions instead of 3331: most
//! of the reads they forwarded shared the snapshot before their inlined
//! call already (see `Recorder::read_snapshot`), and the stores' snapshots
//! were the ones left. With them, and with `maxsnap` counting only the
//! snapshots a guard leaves through, the root records 1782 guest
//! instructions with 132 such snapshots (72 reads forwarded, 76 stores
//! proven) and compiles, and its side 3421 (2026-11-01).
//!
//! A store kills the facts of every slot it may write: any table may be any
//! other (two windows or lanes can hold one table), so only a key known to
//! differ (two constants that are different numbers) keeps a fact. Anything
//! that may write a table or move an array ([`RecKind::ArrayStore`] that
//! sets, a helper that writes the heap, a host call, a self call) kills them
//! all. Facts hold for one recording, which is one path through one
//! iteration: native code runs the operations in that order or leaves.
//!
//! Facts are about plain values only: integers, floats and booleans. A table
//! or callable slot keeps its read, which feeds lanes and callee guards.

use super::{Operand, RecKind, Recorder, TraceScalar, TraceType};
use crate::jit_ir::trace::HelperEffects;

/// See the module docs.
#[derive(Clone, Copy, Debug)]
pub(super) struct SlotFact {
    pub(super) table: Operand,
    pub(super) key: Operand,
    pub(super) value: Operand,
    pub(super) ty: TraceType,
}

/// Whether the recording keeps slot facts for values of type `ty`.
const fn kept(ty: TraceType) -> bool {
    matches!(
        ty,
        TraceType::Integer | TraceType::Float | TraceType::Boolean
    )
}

impl Recorder {
    /// The number a key operand is, when it is a constant (past its moves).
    fn constant_key(&self, key: Operand) -> Option<f64> {
        let mut operand = key;
        loop {
            let Operand::Op(index) = operand else {
                return None;
            };
            match &self.ops.get(index as usize)?.kind {
                RecKind::Move(source) => operand = *source,
                #[allow(clippy::cast_precision_loss, reason = "keys compare as Lua numbers")]
                RecKind::Constant(TraceScalar::Integer(value)) => return Some(*value as f64),
                RecKind::Constant(TraceScalar::Float(value)) => return Some(*value),
                _ => return None,
            }
        }
    }

    /// Whether two key operands are the same key on every run (`Some(true)`),
    /// different keys on every run (`Some(false)`), or not known.
    fn same_key(&self, left: Operand, right: Operand) -> Option<bool> {
        if left == right {
            return Some(true);
        }
        match (self.constant_key(left), self.constant_key(right)) {
            // `==` on the numbers: `1` and `1.0` name one slot.
            #[allow(clippy::float_cmp, reason = "keys compare as Lua numbers")]
            (Some(left), Some(right)) => Some(left == right),
            _ => None,
        }
    }

    /// The fact about slot `key` of `table`, if the recording has one whose
    /// value has type `ty`.
    pub(super) fn slot_fact(
        &self,
        table: Operand,
        key: Operand,
        ty: TraceType,
    ) -> Option<SlotFact> {
        if !kept(ty) {
            return None;
        }
        self.slot_facts
            .iter()
            .rev()
            .find(|fact| {
                fact.table == table && fact.ty == ty && self.same_key(fact.key, key) == Some(true)
            })
            .copied()
    }

    /// The read of slot `key` of `table` that a fact answers: a `Move` of
    /// the value, with no guard. `None` when there is no fact of type `ty`.
    pub(super) fn forward_slot_read(
        &mut self,
        pc: usize,
        table: Operand,
        key: Operand,
        ty: TraceType,
    ) -> Option<u32> {
        let fact = self.slot_fact(table, key, ty)?;
        self.slot_reads_forwarded += 1;
        Some(self.emit(pc, RecKind::Move(fact.value), Some(ty)))
    }

    /// Notes that the guarded read `value` of slot `key` of `table` found a
    /// value of type `ty`. A read writes nothing, so every other fact stands.
    pub(super) fn note_slot_read(
        &mut self,
        table: Operand,
        key: Operand,
        value: u32,
        ty: TraceType,
    ) {
        if !kept(ty) {
            return;
        }
        self.slot_facts.push(SlotFact {
            table,
            key,
            value: Operand::Op(value),
            ty,
        });
    }

    /// What recording `kind` does to the facts: an in-place store kills
    /// those of every slot it may write, and anything that may write a table
    /// or move an array (a store that sets, a helper that writes the heap, a
    /// host call, a self call) kills them all. Called for every operation
    /// the recording emits, before it is pushed.
    pub(super) fn slot_facts_before(&mut self, kind: &RecKind) {
        if self.slot_facts.is_empty() {
            return;
        }
        let keeps = match kind {
            RecKind::ArrayStore {
                set: false, key, ..
            }
            | RecKind::ArrayOverwrite { key, .. }
            | RecKind::LaneArrayStore { key, .. } => {
                let key = *key;
                let facts = std::mem::take(&mut self.slot_facts);
                self.slot_facts = facts
                    .into_iter()
                    .filter(|fact| self.same_key(fact.key, key) == Some(false))
                    .collect();
                true
            }
            RecKind::ArrayStore { set: true, .. }
            | RecKind::HostCall { .. }
            | RecKind::SelfCall { .. } => false,
            RecKind::CallHelper { helper, .. } => {
                let effects = helper.effects();
                !effects.contains(HelperEffects::WRITES_HEAP)
                    && !effects.contains(HelperEffects::RESUMES)
            }
            _ => true,
        };
        if !keeps {
            self.slot_facts.clear();
        }
    }

    /// Notes the in-place store of `value`, of type `ty`, into slot `key` of
    /// `table`, after [`Self::slot_facts_before`] killed the facts it may
    /// have made stale: the slot's fact is the stored value.
    pub(super) fn note_slot_value(
        &mut self,
        table: Operand,
        key: Operand,
        value: Operand,
        ty: TraceType,
    ) {
        if kept(ty) {
            self.slot_facts.push(SlotFact {
                table,
                key,
                value,
                ty,
            });
        }
    }
}
