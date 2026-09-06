//! Write barriers and minor collections.
//!
//! # Invariant
//!
//! Outside a major cycle's mark phase, an old object points only to old
//! objects unless it is on the remembered set (`gc_remembered`, objects that
//! a barrier caught since the last minor collection) or the carry list
//! (`gc_carry`, objects the last minor collection traced or promoted). A
//! minor collection therefore marks from the roots, the remembered set, and
//! the carry list, treats every other old object as reachable, and sweeps
//! only the young and survival lists.
//!
//! Objects age from young to survival to old over two minor collections.
//! An object promoted to old may still point at a survivor, and a remembered
//! object's young children only became survivors, so both are traced once
//! more in the next minor collection through the carry list.
//!
//! # Barriers
//!
//! Every heap store of a reference goes through [`Runtime::gc_barrier`]:
//!
//! - During a major cycle's mark phase it is a Dijkstra insertion barrier:
//!   the stored object is marked and queued. VM registers are not barriered;
//!   the atomic step rescans them instead.
//! - At all times, storing a non-old object into an old container adds the
//!   container to the remembered set.
//! - After a major cycle's atomic step, storing a non-old object into any
//!   container remembers the container. The sweep promotes survivors arena
//!   by arena, so a still-young container may be promoted after it received
//!   an object from an arena that was already swept, and that object stays
//!   young.

use super::heap::{
    closure_accounted_bytes, file_accounted_bytes, gmatch_iterator_accounted_bytes,
    table_accounted_bytes, thread_accounted_bytes,
};
use super::mark::TableTraversal;
use super::{
    GC_BAD_MINOR_LIMIT, GC_BIG_TRIALS_PER_CYCLE, GC_FINAL_TRIAL_MIN_OLD_BYTES, GC_MAX_MINOR_PAUSE,
    GC_MIN_NURSERY_BYTES, GC_MINOR_VERDICT_OBJECTS, GC_TRIAL_BUDGET_DIVISOR,
    GC_TRIAL_BYTES_PER_OBJECT, GC_TRIAL_RECENT_LIVE_DIVISOR, GC_TRIAL_REMEMBERED_DIVISOR,
    GC_TRIAL_SAMPLES, GC_TRIAL_SPACING_DIVISOR, GcMode, GcReference, GcRoots, GcStats,
    IncrementalGcPhase, Marker,
};
use crate::runtime::closures::Cell;
use crate::runtime::runtime::Runtime;
use crate::runtime::table_storage::{ArraySlot, GcSlotRange};
use crate::runtime::userdata::UserdataFinalizerState;
use crate::runtime::value::Value;
use crate::runtime::value::{CellHandle, ClosureHandle, TableHandle};

impl Runtime {
    /// Records a store of `stored` into `container`. `container` is `None`
    /// for runtime roots, such as shared metatables, and for registrations
    /// that keep an object alive for the current cycle.
    #[inline]
    pub(crate) fn gc_barrier(&mut self, container: Option<GcReference>, stored: GcReference) {
        let after_atomic = self.incremental_gc.is_some() && self.gc_barrier_mark(stored);
        if let Some(container) = container
            && (after_atomic || self.gc_reference_is_old(container))
            && !self.gc_reference_is_old(stored)
        {
            self.gc_remember(container);
        }
    }

    /// [`Self::gc_barrier`] for a store of `stored` into one array slot of
    /// `table`. A remembered table then traces only the slots its stores
    /// touched, not the whole array (see [`Self::gc_minor_traversal`]). The
    /// caller guarantees that the store moved no other slot.
    ///
    /// Audit of the writers of `Table::array`, which decides who may call
    /// this. Only a store of one reference into one slot that moves nothing
    /// else does: `table_set_dense_reference`, the dense branch of
    /// `table_trace_set` and `run_array_store` (through
    /// `table_native_stored_slot`). Every other write notes the whole array.
    /// A store through `gc_barrier` (`set_value`, `set_array`, a site store,
    /// a hash store, and the write-back of `table.sort` and `table.move`,
    /// which go through `table_set`) makes `gc_remember` set `gc_dirty`
    /// whole. The shifting `trace_insert` and `trace_remove` call
    /// `Table::note_array_moved`. `grow_array_and_absorb` and
    /// `absorb_sparse_tail` move values out of the hash part, where a young
    /// value was stored with `gc_remember` (whole, and carried whole by the
    /// minor after). `prefill_array`, `set_dense_scalar` and the nil stores
    /// write no reference. The debug build checks all of it at every minor
    /// collection (`gc_unranged_young_child`).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn gc_barrier_array(
        &mut self,
        table: TableHandle,
        slot: ArraySlot,
        stored: GcReference,
    ) {
        let container = GcReference::Table(table);
        let after_atomic = self.incremental_gc.is_some() && self.gc_barrier_mark(stored);
        if (after_atomic || self.gc_reference_is_old(container))
            && !self.gc_reference_is_old(stored)
        {
            self.gc_remember_slot(table, slot);
        }
    }

    /// [`Self::gc_barrier_array`] for a table that native code stored into
    /// the array slot at `address` of `table`. Native code stores a table
    /// into the same few slots of one table over and over (a ring of recent
    /// objects, a cache), so this reads the container once, for its age, its
    /// remembered flag and the slots its minor collection traces, and
    /// leaves at the first answer that settles it. A container that is
    /// remembered with the slot already among those slots is left as it is:
    /// [`Self::gc_remember_slot`] would set the flag it has, widen its range
    /// by nothing and push nothing. A stale `table` is neither old nor
    /// remembered by [`Self::gc_remember_slot`], so it is left too. The
    /// callers that may use [`Self::gc_barrier_array`] may use this.
    ///
    /// The two answers a store in a loop gets nearly every time are in line
    /// at the caller, and read nothing but the container: with no major
    /// cycle running, a container that is not old needs no barrier, and one
    /// that is remembered with the slot among the slots its minor collection
    /// traces has had it. They are the first two of the three exits of
    /// [`Self::gc_barrier_array_table_checked`] with `after_atomic` false,
    /// which a runtime with no cycle gives. Every other store, and every
    /// store during a major cycle, takes that function whole, so the barrier
    /// decides what it decided before.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn gc_barrier_array_table(
        &mut self,
        table: TableHandle,
        address: usize,
        stored: TableHandle,
    ) {
        if self.incremental_gc.is_none()
            && let Some((container, old, remembered)) = self.tables.aged(table.0)
            && (!old || (remembered && container.gc_dirty_holds_address(address)))
        {
            return;
        }
        self.gc_barrier_array_table_checked(table, address, stored);
    }

    /// [`Self::gc_barrier_array_table`] in full, out of line: the caller
    /// keeps no register for it on the stores its two exits answer.
    #[cold]
    #[inline(never)]
    fn gc_barrier_array_table_checked(
        &mut self,
        table: TableHandle,
        address: usize,
        stored: TableHandle,
    ) {
        let after_atomic = self.incremental_gc.is_some() && self.gc_barrier_mark_table(stored);
        let Some((container, old, remembered)) = self.tables.aged(table.0) else {
            return;
        };
        if !(after_atomic || old)
            || (remembered && container.gc_dirty_holds_address(address))
            || self.tables.is_old(stored.0)
        {
            return;
        }
        self.gc_remember_slot_apart(table, address);
    }

    /// [`Self::gc_barrier_mark`] of a table, out of line.
    #[cold]
    #[inline(never)]
    fn gc_barrier_mark_table(&mut self, stored: TableHandle) -> bool {
        self.gc_barrier_mark(GcReference::Table(stored))
    }

    /// [`Self::gc_remember_slot`] for a slot named by address, out of line.
    #[cold]
    #[inline(never)]
    fn gc_remember_slot_apart(&mut self, table: TableHandle, address: usize) {
        self.gc_remember_slot(table, ArraySlot::Address(address));
    }

    /// Records that the array part of `table` moved values between slots
    /// (an insert or remove that shifted elements). Every slot may hold a
    /// young object now, so the next minor collection traces the whole array
    /// of a table it already traces by range. A table with both ranges empty
    /// is neither remembered nor carried, so it holds no young child and
    /// moving its slots changes nothing.
    ///
    /// Audit of the writers of `Table::array`, which decides what may use
    /// [`Self::gc_barrier_array`]. Only a store of one reference into one
    /// slot that moves nothing else does: `table_set_dense_reference`, the
    /// dense branch of `table_set_native_array` and `run_array_store`
    /// (through `table_native_stored_slot`). Everything else notes the whole
    /// array: `gc_remember` sets `gc_dirty` whole, so a store through
    /// `set_value`, `set_array`, a site store, a hash store or a
    /// `table.sort` write-back (all go through `gc_barrier`) traces the
    /// array in full; the shifting `trace_insert` and `trace_remove` call
    /// this function; `grow_array_and_absorb` and `absorb_sparse_tail` move
    /// values out of the hash part, where a young value was stored with
    /// `gc_remember` (whole, and carried whole for the minor after); and
    /// `prefill_array`, `set_dense_scalar` and the nil stores write no
    /// reference. The debug build checks the rest at every minor collection
    /// (`gc_unranged_young_child`).
    /// Objects allocated while a major cycle marks are born marked and are
    /// never traced, so whatever they hold from birth must be marked like a
    /// store. Without this, a cell created holding the only other reference
    /// to an unmarked object let the sweep free a reachable object.
    #[inline]
    pub(crate) fn gc_barrier_born_cell(&mut self, cell: CellHandle) {
        if self.incremental_gc.is_none() {
            return;
        }
        let Some(stored) = self
            .cells
            .get(cell.0)
            .and_then(|cell| super::heap::value_gc_reference(&cell.value))
        else {
            return;
        };
        self.gc_barrier(Some(GcReference::Cell(cell)), stored);
    }

    /// The closure counterpart of [`Self::gc_barrier_born_cell`]: its cells
    /// and environment.
    #[inline]
    pub(crate) fn gc_barrier_born_closure(&mut self, closure: ClosureHandle) {
        if self.incremental_gc.is_none() {
            return;
        }
        let Some(body) = self.closures.get(closure.0) else {
            return;
        };
        let environment = body.environment;
        let count = body.cells.len();
        let container = Some(GcReference::Closure(closure));
        self.gc_barrier(container, GcReference::Table(environment));
        // Index and re-fetch: copying the cells out would allocate per closure.
        for index in 0..count {
            let Some(cell) = self
                .closures
                .get(closure.0)
                .and_then(|body| body.cells.get(index).copied())
            else {
                break;
            };
            self.gc_barrier(container, GcReference::Cell(cell));
        }
    }

    #[inline]
    pub(crate) fn gc_barrier_value(&mut self, container: Option<GcReference>, value: &Value) {
        if let Some(stored) = super::heap::value_gc_reference(value) {
            self.gc_barrier(container, stored);
        }
    }

    /// The insertion barrier of an active major cycle. Returns whether the
    /// cycle is past its atomic step.
    #[cold]
    fn gc_barrier_mark(&mut self, stored: GcReference) -> bool {
        let Some(mut cycle) = self.incremental_gc.take() else {
            return false;
        };
        let marking = cycle.phase == IncrementalGcPhase::Mark;
        if marking {
            self.gc_mark_reference(stored, &mut cycle.marker);
        }
        self.incremental_gc = Some(cycle);
        !marking
    }

    /// Remembers `table` for a store into an array slot, and widens the slots
    /// its minor collection traces to cover it. One arena lookup does both.
    #[cold]
    fn gc_remember_slot(&mut self, table: TableHandle, slot: ArraySlot) {
        // The range is empty unless a minor collection has not consumed it
        // yet. A major cycle's sweep clears `touched` while the table stays
        // on the remembered list, so the table may be touched again with
        // its range still pending: widen it, never restart it.
        let newly_touched = self.tables.touch_with(table.0, |entry| {
            let range = match slot {
                ArraySlot::Index(index) => GcSlotRange::one(index),
                ArraySlot::Address(address) => entry.array_slot_range(address),
            };
            entry.gc_dirty = entry.gc_dirty.union(range);
        });
        if newly_touched {
            self.gc_remembered.push(GcReference::Table(table));
        }
    }

    #[cold]
    fn gc_remember(&mut self, container: GcReference) {
        let newly_touched = match container {
            // A store the barrier did not index may have written any slot.
            GcReference::Table(handle) => self
                .tables
                .touch_with(handle.0, |entry| entry.gc_dirty = GcSlotRange::WHOLE),
            GcReference::Closure(handle) => self.closures.touch(handle.0),
            GcReference::Cell(handle) => self.cells.touch(handle.0),
            GcReference::Userdata(handle) => self.userdatas.touch(handle.0),
            GcReference::Thread(handle) => self.threads.touch(handle.0),
            GcReference::File(_) | GcReference::GmatchIterator(_) => false,
        };
        if newly_touched {
            self.gc_remembered.push(container);
        }
    }

    fn gc_untouch(&mut self, reference: GcReference) {
        match reference {
            GcReference::Table(handle) => self.tables.untouch(handle.0),
            GcReference::Closure(handle) => self.closures.untouch(handle.0),
            GcReference::Cell(handle) => self.cells.untouch(handle.0),
            GcReference::Userdata(handle) => self.userdatas.untouch(handle.0),
            GcReference::Thread(handle) => self.threads.untouch(handle.0),
            GcReference::File(_) | GcReference::GmatchIterator(_) => {}
        }
    }

    /// Clears the remembered set and carry list. A major cycle does this at
    /// its atomic step: its sweep makes every survivor old at once.
    pub(crate) fn gc_forget_generations(&mut self) {
        for reference in std::mem::take(&mut self.gc_remembered) {
            self.gc_untouch(reference);
            self.gc_set_carry_range(reference, GcSlotRange::EMPTY);
            self.gc_set_dirty_range(reference, GcSlotRange::EMPTY);
        }
        for reference in std::mem::take(&mut self.gc_carry) {
            self.gc_set_carry_range(reference, GcSlotRange::EMPTY);
        }
    }

    /// The array slots stored into `reference` since the last minor
    /// collection, empty for anything but a table.
    fn gc_dirty_range(&self, reference: GcReference) -> GcSlotRange {
        match reference {
            GcReference::Table(handle) => self
                .tables
                .get(handle.0)
                .map_or(GcSlotRange::EMPTY, |table| table.gc_dirty),
            _ => GcSlotRange::EMPTY,
        }
    }

    /// Sets the array slots stored since the last minor collection for
    /// `reference`, a remembered table.
    fn gc_set_dirty_range(&mut self, reference: GcReference, range: GcSlotRange) {
        if let GcReference::Table(handle) = reference
            && let Some(table) = self.tables.get_mut(handle.0)
        {
            table.gc_dirty = range;
        }
    }

    /// Sets the array slots a minor collection traces for `reference` while
    /// it is on the carry list. Other objects have no such range.
    fn gc_set_carry_range(&mut self, reference: GcReference, range: GcSlotRange) {
        if let GcReference::Table(handle) = reference
            && let Some(table) = self.tables.get_mut(handle.0)
        {
            table.gc_carry = range;
        }
    }

    /// How a minor collection traces `reference`. A remembered table traces
    /// the array slots its young stores touched since the last minor
    /// collection (`gc_dirty`) and the slots the carry list covers
    /// (`gc_carry`). A carried table traces its carry slots. Anything else,
    /// and any table whose range is whole, is traced in full.
    pub(crate) fn gc_minor_traversal(
        &self,
        reference: GcReference,
        remembered: bool,
    ) -> TableTraversal {
        let GcReference::Table(handle) = reference else {
            return TableTraversal::Full;
        };
        let Some(table) = self.tables.get(handle.0) else {
            return TableTraversal::Full;
        };
        let range = if remembered {
            table.gc_carry.union(table.gc_dirty)
        } else {
            table.gc_carry
        };
        if range.is_whole() {
            TableTraversal::Full
        } else {
            TableTraversal::Dirty(range)
        }
    }

    /// The work a minor collection's traversal of `reference` costs, in the
    /// units of [`Marker::work`].
    fn gc_minor_work(&self, reference: GcReference, remembered: bool) -> usize {
        let GcReference::Table(handle) = reference else {
            return 0;
        };
        let Some(table) = self.tables.get(handle.0) else {
            return 0;
        };
        let range = if remembered {
            table.gc_carry.union(table.gc_dirty)
        } else {
            table.gc_carry
        };
        if range.is_whole() {
            table.array.len() + table.entries.len()
        } else {
            let end = (range.hi as usize).min(table.array.len());
            end.saturating_sub(range.lo as usize)
        }
    }

    /// The first young or survival-age child that an old table on the
    /// remembered or carry list holds where its next minor collection will
    /// not trace: an array slot outside its range, or the hash part. Returns
    /// the table's arena slot and the array index (`usize::MAX` for the hash
    /// part). A major cycle's sweep promotes arena by arena, so the check
    /// skips a cycle in progress (see `gc_generational_violation`).
    #[cfg(debug_assertions)]
    pub(crate) fn gc_unranged_young_child(&self) -> Option<(u32, usize)> {
        use super::heap::{key_gc_reference, value_gc_reference};
        if self.incremental_gc.is_some() {
            return None;
        }
        // A table on both lists is traced by the remembered pass, which
        // adds the slots stored since the last minor collection.
        let remembered_set: std::collections::BTreeSet<GcReference> =
            self.gc_remembered.iter().copied().collect();
        for reference in self.gc_remembered.iter().chain(&self.gc_carry) {
            let remembered = remembered_set.contains(reference);
            let GcReference::Table(handle) = reference else {
                continue;
            };
            let Some(table) = self.tables.get(handle.0) else {
                continue;
            };
            if !self.tables.is_old(handle.0) {
                continue;
            }
            let range = if remembered {
                table.gc_carry.union(table.gc_dirty)
            } else {
                table.gc_carry
            };
            if range.is_whole() {
                continue;
            }
            let young = |child: Option<GcReference>| {
                child.is_some_and(|child| !self.gc_reference_is_old(child))
            };
            for (index, value) in table.array.iter().enumerate() {
                let covered = (range.lo as usize..range.hi as usize).contains(&index);
                if !covered && young(value_gc_reference(value)) {
                    return Some((handle.0.slot, index));
                }
            }
            let mut in_hash = false;
            table.entries.for_each(|key, value| {
                in_hash |= young(key_gc_reference(key)) || young(value_gc_reference(value));
            });
            if in_hash {
                return Some((handle.0.slot, usize::MAX));
            }
        }
        None
    }

    #[inline]
    pub(crate) fn gc_reference_is_old(&self, reference: GcReference) -> bool {
        match reference {
            GcReference::Table(handle) => self.tables.is_old(handle.0),
            GcReference::Closure(handle) => self.closures.is_old(handle.0),
            GcReference::Cell(handle) => self.cells.is_old(handle.0),
            GcReference::File(handle) => self.files.is_old(handle.0),
            GcReference::Userdata(handle) => self.userdatas.is_old(handle.0),
            GcReference::GmatchIterator(handle) => self.gmatch_iterators.is_old(handle.0),
            GcReference::Thread(handle) => self.threads.is_old(handle.0),
        }
    }

    #[cfg(test)]
    fn gc_reference_is_young(&self, reference: GcReference) -> bool {
        match reference {
            GcReference::Table(handle) => self.tables.is_young(handle.0),
            GcReference::Closure(handle) => self.closures.is_young(handle.0),
            GcReference::Cell(handle) => self.cells.is_young(handle.0),
            GcReference::File(handle) => self.files.is_young(handle.0),
            GcReference::Userdata(handle) => self.userdatas.is_young(handle.0),
            GcReference::GmatchIterator(handle) => self.gmatch_iterators.is_young(handle.0),
            GcReference::Thread(handle) => self.threads.is_young(handle.0),
        }
    }

    /// Whether automatic collection runs minor collections now.
    ///
    /// A program that selected generational mode always gets them. The
    /// default policy, which reports incremental mode, pauses them for a
    /// few major cycles after minor collections stop paying: when most
    /// young objects survive, as while a program builds a large structure,
    /// a minor collection traces and promotes them only for a major cycle
    /// to free them later. Each run of unproductive minors after a pause
    /// doubles the next pause.
    pub(crate) fn gc_minors_active(&self) -> bool {
        self.gc_generational && self.gc_minor_pause == 0
    }

    /// Records whether a minor collection that swept `swept` young and
    /// surviving objects, freeing `freed` of them, paid for itself.
    fn gc_judge_minor(&mut self, swept: usize, freed: usize) {
        if self.gc_mode != GcMode::Incremental || swept < GC_MINOR_VERDICT_OBJECTS {
            return;
        }
        if freed.saturating_mul(2) >= swept {
            self.gc_bad_minors = 0;
            self.gc_minor_backoff = 1;
            return;
        }
        self.gc_bad_minors += 1;
        if self.gc_bad_minors >= GC_BAD_MINOR_LIMIT {
            self.gc_bad_minors = 0;
            // The first major cycle only collects what the minors promoted.
            self.gc_minor_pause = self.gc_minor_backoff + 1;
            self.gc_trial_fails = 0;
            self.gc_trial_prev_young = 0;
            self.gc_final_trial_done = false;
            self.gc_big_trials = 0;
            self.gc_minor_backoff = (self.gc_minor_backoff * 2).min(GC_MAX_MINOR_PAUSE);
        }
    }

    /// The threshold while the default policy pauses minor collections: a
    /// trial after a nursery of allocation, or the major cycle if its debt
    /// comes first. Without the trial, a program that stopped building would
    /// pay a whole-heap major cycle for every heap's worth of garbage until
    /// the pause ran out.
    pub(crate) fn gc_paused_threshold(&mut self, heap: usize) -> usize {
        self.gc_pause_major_debt = self.gc_major_debt(heap);
        if !self.gc_generational {
            return self.gc_pause_major_debt;
        }
        (self.gc_nursery_bytes() / GC_TRIAL_SPACING_DIVISOR).min(self.gc_pause_major_debt)
    }

    /// Whether the next automatic step is a trial during a pause rather than
    /// the start of a major cycle.
    pub(crate) fn gc_trial_minor_due(&self) -> bool {
        self.gc_generational
            && self.gc_minor_pause > 0
            && (self.gc_debt < self.gc_pause_major_debt
                || (!self.gc_final_trial_done && self.gc_old_bytes >= GC_FINAL_TRIAL_MIN_OLD_BYTES))
    }

    /// Ends the pause if a minor collection would pay now, and returns
    /// whether it did.
    ///
    /// The trial marks the young objects the roots and the remembered set
    /// reach and gives up once that work passes a small share of the young
    /// population. A program that still builds a structure keeps most of its
    /// young objects, so the trial stops early and the pause goes on. A
    /// program that stopped building loses nearly all of them, and a minor
    /// collection pays. The trial leaves no mark that a later collection
    /// reads, because each collection starts a new epoch.
    pub(crate) fn gc_trial_minor_pays(&mut self, roots: &GcRoots) -> bool {
        let young = self.tables.young.len()
            + self.tables.survival.len()
            + self.closures.young.len()
            + self.closures.survival.len()
            + self.cells.young.len()
            + self.cells.survival.len();
        if young < GC_MINOR_VERDICT_OBJECTS {
            return false;
        }
        // On a big heap, a few trials per major cycle (evenly spaced in debt,
        // and one right before the cycle) may do a share of the work the
        // cycle would do: marking the whole heap is the alternative.
        let share = self.gc_pause_major_debt / GC_BIG_TRIALS_PER_CYCLE;
        let last = self.gc_old_bytes >= GC_FINAL_TRIAL_MIN_OLD_BYTES
            && (self.gc_debt >= self.gc_pause_major_debt
                || self.gc_debt >= share.saturating_mul(usize::from(self.gc_big_trials) + 1));
        if last && self.gc_debt < self.gc_pause_major_debt {
            self.gc_big_trials += 1;
        }
        let recent_from = if self.gc_trial_prev_young <= self.tables.young.len() {
            self.gc_trial_prev_young
        } else {
            0
        };
        let recent = self.tables.young.len() - recent_from;
        self.gc_trial_prev_young = self.tables.young.len();
        if last && recent < GC_MINOR_VERDICT_OBJECTS {
            return false;
        }
        let old_objects = self.gc_old_bytes / GC_TRIAL_BYTES_PER_OBJECT;
        let mut budget = young / GC_TRIAL_BUDGET_DIVISOR;
        // A minor collection traces the remembered set whole, whatever it
        // frees. Before a major cycle that cost is bounded against the old
        // heap, not the young budget: a big old table with a young store is
        // still far cheaper to scan than a cycle that marks the whole heap.
        let remembered_budget = if last {
            self.gc_old_bytes / GC_TRIAL_REMEMBERED_DIVISOR
        } else {
            budget
        };
        let mut work = 0usize;
        let traced = self
            .gc_remembered
            .iter()
            .map(|reference| (reference, true))
            .chain(self.gc_carry.iter().map(|reference| (reference, false)));
        for (reference, remembered) in traced {
            work += 1 + self.gc_minor_work(*reference, remembered);
            if work > remembered_budget {
                return false;
            }
        }
        if last {
            // The young objects the pause held back are not all garbage even
            // after the program stops building: the tail of the build is
            // still alive. What tells the two apart is the objects allocated
            // since the last trial, so the trial marks all that is
            // reachable and judges those.
            budget += old_objects;
            work = 0;
        }
        self.begin_gc_epoch();
        let mut marker = Marker::new(self.gc_epoch, true);
        marker.work = work;
        self.gc_mark_roots(Some(roots), &mut marker);
        for reference in &self.gc_remembered {
            let traversal = self.gc_minor_traversal(*reference, true);
            self.gc_traverse(*reference, &mut marker, &traversal);
        }
        for reference in &self.gc_carry {
            let traversal = self.gc_minor_traversal(*reference, false);
            self.gc_traverse(*reference, &mut marker, &traversal);
        }
        if last {
            // From here the work is young objects only: old ones count as
            // marked.
            marker.work = 0;
        }
        while marker.work <= budget {
            let Some(reference) = marker.gray.pop() else {
                if last && !self.gc_recent_mostly_dead(recent_from, recent) {
                    return false;
                }
                self.gc_minor_pause = 0;
                self.gc_bad_minors = 0;
                self.gc_trial_fails = 0;
                return true;
            };
            self.gc_traverse(reference, &mut marker, &TableTraversal::Full);
        }
        false
    }

    /// Whether a sample of the tables allocated since `from` was left
    /// unmarked by the trial that just finished.
    fn gc_recent_mostly_dead(&self, from: usize, recent: usize) -> bool {
        let stride = (recent / GC_TRIAL_SAMPLES).max(1);
        let (mut seen, mut live) = (0usize, 0usize);
        for slot in self.tables.young[from..].iter().step_by(stride) {
            seen += 1;
            if self.tables.slots.get(*slot as usize).is_some_and(|entry| {
                entry
                    .marked_epoch
                    .load(std::sync::atomic::Ordering::Relaxed)
                    == self.gc_epoch
            }) {
                live += 1;
            }
        }
        live.saturating_mul(GC_TRIAL_RECENT_LIVE_DIVISOR) <= seen
    }

    /// Bytes to allocate before the next minor collection.
    pub(crate) fn gc_nursery_bytes(&self) -> usize {
        (((self.gc_major_base / 100) as f64 * self.gc_minor_multiplier.max(0.0)) as usize)
            .max(GC_MIN_NURSERY_BYTES)
    }

    /// Whether the old generation has outgrown its size after the last major
    /// collection enough to warrant another one.
    fn gc_old_generation_is_full(&self) -> bool {
        if self.gc_major_multiplier <= 0.0 {
            return false;
        }
        let base = self.gc_major_base.max(GC_MIN_NURSERY_BYTES);
        self.gc_old_bytes
            > base.saturating_add(((base as f64 / 100.0) * self.gc_major_multiplier) as usize)
    }

    /// Runs one complete minor collection. It is atomic: its cost is bounded
    /// by the roots, the nursery, and the remembered set rather than by the
    /// whole heap.
    pub(crate) fn gc_minor_collection(&mut self, roots: &GcRoots) -> GcStats {
        debug_assert!(self.incremental_gc.is_none());
        #[cfg(debug_assertions)]
        if let Some((slot, index)) = self.gc_unranged_young_child() {
            panic!("old table {slot} holds a young child at {index} outside its traced range");
        }
        self.begin_gc_epoch();
        let mut marker = Marker::new(self.gc_epoch, true);
        self.gc_mark_roots(Some(roots), &mut marker);
        let remembered = std::mem::take(&mut self.gc_remembered);
        let carry = std::mem::take(&mut self.gc_carry);
        for reference in &remembered {
            self.gc_untouch(*reference);
        }
        for reference in &remembered {
            let traversal = self.gc_minor_traversal(*reference, true);
            self.gc_traverse(*reference, &mut marker, &traversal);
        }
        for reference in &carry {
            let traversal = self.gc_minor_traversal(*reference, false);
            self.gc_traverse(*reference, &mut marker, &traversal);
        }
        self.gc_converge(&mut marker);
        if self.gc_mode == GcMode::Incremental {
            self.gc_defer_finalizers(&mut marker);
        }
        self.gc_clear_weak_tables(&marker, true);
        self.gc_separate_finalizers(&mut marker);
        self.gc_clear_weak_tables(&marker, false);

        // Remembered objects are traced once more: their young children have
        // only become survivors. The next minor collection traces the slots
        // this one saw written (`gc_dirty`), not the rest of the array.
        for reference in &carry {
            self.gc_set_carry_range(*reference, GcSlotRange::EMPTY);
        }
        // The pending ranges are read before any is cleared: a remembered
        // table may appear twice on the list.
        for reference in &remembered {
            let carried = if self.gc_reference_is_old(*reference) {
                self.gc_dirty_range(*reference)
            } else {
                GcSlotRange::EMPTY
            };
            self.gc_set_carry_range(*reference, carried);
        }
        for reference in &remembered {
            self.gc_set_dirty_range(*reference, GcSlotRange::EMPTY);
        }
        let mut next_carry = remembered;
        next_carry.retain(|reference| self.gc_reference_is_old(*reference));
        let carried = next_carry.len();
        let swept = self.tables.young.len()
            + self.tables.survival.len()
            + self.closures.young.len()
            + self.closures.survival.len()
            + self.cells.young.len()
            + self.cells.survival.len();
        let stats = self.gc_sweep_minor(marker.epoch, &mut next_carry);
        // A promoted table was traced whole, and its survivors need the next
        // minor collection too.
        for reference in &next_carry[carried..] {
            self.gc_set_carry_range(*reference, GcSlotRange::WHOLE);
        }
        self.gc_judge_minor(swept, stats.tables + stats.closures + stats.cells);
        self.gc_carry = next_carry;
        self.gc_cleanup_upvalue_identities();

        self.gc_debt = 0;
        self.elided_bytes_total += self.elided_table_bytes as u64;
        self.elided_table_bytes = 0;
        // Paused, the next major cycle starts once the heap has grown by
        // the pause, as in incremental mode.
        self.gc_threshold = if self.gc_minors_active() {
            self.gc_nursery_bytes()
        } else {
            self.gc_paused_threshold(self.gc_old_bytes)
        };
        if self.gc_old_generation_is_full() {
            self.gc_major_requested = true;
        }
        stats
    }

    /// Keeps unreachable finalizable objects alive and registered, and
    /// requests a major cycle to queue them.
    ///
    /// A program in incremental mode sees a finalizer run only after a
    /// full cycle, which has also collected every object that was garbage
    /// when the cycle began: Lua's `gc.lua` waits for a finalizer and then
    /// expects older garbage, such as ephemeron chains, to be gone. Minor
    /// collections only reclaim young objects, so they leave finalization
    /// to the major cycle they request here. Programs that select
    /// generational mode finalize in minor collections, as Lua 5.4 does.
    fn gc_defer_finalizers(&mut self, marker: &mut Marker) {
        let candidates = self
            .finalizer_registry
            .values()
            .map(|object| super::mark::finalizer_reference(*object))
            .filter(|reference| !self.gc_reference_is_reached(*reference, marker))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return;
        }
        self.gc_major_requested = true;
        for reference in candidates {
            self.gc_mark_reference(reference, marker);
        }
        self.gc_converge(marker);
    }

    fn gc_sweep_minor(&mut self, epoch: u64, carry: &mut Vec<GcReference>) -> GcStats {
        use crate::runtime::value::{
            CellHandle, ClosureHandle, FileHandle, GmatchIteratorHandle, TableHandle, ThreadHandle,
            UserdataHandle,
        };
        let mut stats = GcStats::default();
        let mut promoted = 0_usize;
        let detached = &mut self.gc_detached;
        self.tables.sweep_minor(
            epoch,
            |table| {
                stats.tables += 1;
                Self::gc_release_table(detached, table);
            },
            |key, table| {
                promoted += table_accounted_bytes(table);
                carry.push(GcReference::Table(TableHandle(key)));
            },
        );
        self.closures.sweep_minor(
            epoch,
            |_| stats.closures += 1,
            |key, closure| {
                promoted += closure_accounted_bytes(closure);
                carry.push(GcReference::Closure(ClosureHandle(key)));
            },
        );
        self.cells.sweep_minor(
            epoch,
            |_| stats.cells += 1,
            |key, _| {
                promoted += std::mem::size_of::<Cell>();
                carry.push(GcReference::Cell(CellHandle(key)));
            },
        );
        self.files.sweep_minor(
            epoch,
            |_| stats.files += 1,
            |key, file| {
                promoted += file_accounted_bytes(file);
                let _ = FileHandle(key);
            },
        );
        let registry = &mut self.finalizer_registry;
        let freed_userdata = &mut self.gc_freed_userdata;
        self.userdatas.sweep_minor(
            epoch,
            |userdata| {
                stats.userdatas += 1;
                *freed_userdata += 1;
                if let UserdataFinalizerState::Registered(order) = userdata.finalizer {
                    registry.remove(&order);
                }
            },
            |key, userdata| {
                promoted += userdata.accounted_bytes();
                carry.push(GcReference::Userdata(UserdataHandle(key)));
            },
        );
        self.gmatch_iterators.sweep_minor(
            epoch,
            |_| stats.gmatch_iterators += 1,
            |key, iterator| {
                promoted += gmatch_iterator_accounted_bytes(iterator);
                let _ = GmatchIteratorHandle(key);
            },
        );
        self.threads.sweep_minor(
            epoch,
            |_| stats.threads += 1,
            |key, thread| {
                promoted += thread_accounted_bytes(thread);
                carry.push(GcReference::Thread(ThreadHandle(key)));
            },
        );
        self.gc_old_bytes = self.gc_old_bytes.saturating_add(promoted);
        stats
    }

    /// Drops identity tokens whose cell or userdata no longer exists.
    pub(crate) fn gc_cleanup_upvalue_identities(&mut self) {
        if self.upvalue_identities.is_empty() {
            return;
        }
        let (cells, userdatas) = (&self.cells, &self.userdatas);
        self.upvalue_identities.retain(|cell, identity| {
            cells.get(cell.0).is_some() && userdatas.get(identity.0).is_some()
        });
    }

    /// Whether `reference` names a young object; used by tests.
    #[cfg(test)]
    pub(crate) fn gc_is_young(&self, reference: GcReference) -> bool {
        self.gc_reference_is_young(reference)
    }
}

#[cfg(test)]
impl Runtime {
    /// Direct references held by one object, for invariant checks.
    fn gc_children(&self, reference: GcReference) -> Vec<GcReference> {
        use super::heap::{key_gc_reference, value_gc_reference};
        let mut children = Vec::new();
        match reference {
            GcReference::Table(handle) => {
                if let Some(table) = self.tables.get(handle.0) {
                    children.extend(table.metatable.map(GcReference::Table));
                    children.extend(table.array.iter().filter_map(value_gc_reference));
                    table.entries.for_each(|key, value| {
                        children.extend(key_gc_reference(key));
                        children.extend(value_gc_reference(value));
                    });
                }
            }
            GcReference::Closure(handle) => {
                if let Some(closure) = self.closures.get(handle.0) {
                    children.push(GcReference::Table(closure.environment));
                    children.extend(closure.cells.iter().map(|cell| GcReference::Cell(*cell)));
                    children.extend(closure.retired_cells().map(|cell| GcReference::Cell(*cell)));
                }
            }
            GcReference::Cell(handle) => {
                if let Some(cell) = self.cells.get(handle.0) {
                    children.extend(value_gc_reference(cell.current()));
                }
            }
            GcReference::Userdata(handle) => {
                if let Some(userdata) = self.userdatas.get(handle.0) {
                    children.extend(userdata.metatable.map(GcReference::Table));
                    children.extend(userdata.environment.map(GcReference::Table));
                }
            }
            GcReference::Thread(handle) => {
                if let Some(thread) = self.threads.get(handle.0) {
                    children.extend(thread.roots.references.iter().copied());
                    children.extend(
                        thread
                            .roots
                            .cells
                            .iter()
                            .map(|cell| GcReference::Cell(*cell)),
                    );
                    children.extend(
                        thread
                            .roots
                            .closures
                            .iter()
                            .map(|closure| GcReference::Closure(*closure)),
                    );
                }
            }
            GcReference::File(_) | GcReference::GmatchIterator(_) => {}
        }
        children
    }

    /// Checks the generational invariant between collections: an old object
    /// points only to old objects unless a minor collection will trace it.
    /// Returns the first violating edge.
    pub(crate) fn gc_generational_violation(&self) -> Option<(GcReference, GcReference)> {
        use crate::runtime::value::{
            CellHandle, ClosureHandle, TableHandle, ThreadHandle, UserdataHandle,
        };
        if self.incremental_gc.is_some() {
            return None;
        }
        let traced: std::collections::BTreeSet<GcReference> = self
            .gc_remembered
            .iter()
            .chain(&self.gc_carry)
            .copied()
            .collect();
        let old = self
            .tables
            .live()
            .map(|(key, _)| GcReference::Table(TableHandle(key)))
            .chain(
                self.closures
                    .live()
                    .map(|(key, _)| GcReference::Closure(ClosureHandle(key))),
            )
            .chain(
                self.cells
                    .live()
                    .map(|(key, _)| GcReference::Cell(CellHandle(key))),
            )
            .chain(
                self.userdatas
                    .live()
                    .map(|(key, _)| GcReference::Userdata(UserdataHandle(key))),
            )
            .chain(
                self.threads
                    .live()
                    .map(|(key, _)| GcReference::Thread(ThreadHandle(key))),
            )
            .filter(|reference| self.gc_reference_is_old(*reference) && !traced.contains(reference))
            .collect::<Vec<_>>();
        old.into_iter().find_map(|parent| {
            self.gc_children(parent)
                .into_iter()
                .find(|child| self.gc_reference_is_young(*child))
                .map(|child| (parent, child))
        })
    }
}
