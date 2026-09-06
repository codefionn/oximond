//! Collector pacing, the public collection API, and incremental major cycles.
//!
//! # Scheduling
//!
//! Allocation adds to `gc_debt`. Once the debt reaches `gc_threshold`, the
//! VM calls [`Runtime::gc_automatic_step`] at its next safe point:
//!
//! - In generational mode with no major cycle due, that runs one atomic
//!   minor collection and sets the threshold to the nursery size.
//! - Otherwise it runs one bounded step of an incremental major cycle and
//!   sets the threshold one step interval further.
//!
//! # Major cycles
//!
//! A major cycle marks the runtime's roots, and the VM's roots if it has
//! them, when it starts. Marking then advances in bounded steps. Objects
//! allocated during the cycle are born marked, heap stores go through the
//! insertion barrier, and large tables are traced in resumable chunks.
//!
//! When no gray objects remain, the atomic step rescans a fresh VM root
//! snapshot, retraces tables that changed while partially traced, drains the
//! gray set, selects and resurrects finalizable userdata, and clears weak
//! tables. Because the mutator can no longer reach an unmarked object after
//! that point, the sweep stays incremental without needing roots again. The
//! sweep makes every survivor old.

use super::heap::{
    closure_accounted_bytes, file_accounted_bytes, gmatch_iterator_accounted_bytes, scale_gc_work,
    table_accounted_bytes, thread_accounted_bytes,
};
use super::mark::TableTraversal;
use super::{
    GC_CATCH_UP_NURSERY_DIVISOR, GC_DETACHED_DRAIN_CHUNK, GC_INCREMENTAL_UNIT_BYTES,
    GC_MAX_CATCH_UP, GC_MIN_NURSERY_BYTES, GC_MIN_THRESHOLD_BYTES, GC_SWEEP_SLOTS_PER_UNIT,
    GC_TABLE_SCAN_CHUNK, GC_TRIAL_SPACING_DIVISOR, GcMode, GcRoots, GcStats, GcStepResult,
    IncrementalGc, IncrementalGcPhase, Marker, SweepCursor,
};
use crate::runtime::closures::Cell;
use crate::runtime::error::RuntimeResult;
use crate::runtime::runtime::Runtime;
use crate::runtime::table_reclaim;
use crate::runtime::table_storage::Table;
use crate::runtime::table_storage::key::TableKey;
use crate::runtime::userdata::{
    Finalizer, FinalizerObject, TableFinalizerState, Userdata, UserdataFinalizerState,
};
use crate::runtime::value::Value;

impl Runtime {
    /// Queues every still registered finalizer for VM shutdown. Lua runs
    /// these even when the object remains reachable from globals.
    pub fn queue_close_finalizers(&mut self) {
        let registered = std::mem::take(&mut self.finalizer_registry);
        for (_, object) in registered.into_iter().rev() {
            self.queue_finalizer(object);
        }
    }

    /// Moves a registered object to the pending queue. Stale handles are
    /// dropped.
    pub(crate) fn queue_finalizer(&mut self, object: FinalizerObject) {
        match object {
            FinalizerObject::Userdata(handle) => {
                let Some(userdata) = self.userdatas.get_mut(handle.0) else {
                    return;
                };
                userdata.finalizer = UserdataFinalizerState::Pending;
            }
            FinalizerObject::Table(handle) => {
                if self.tables.get(handle.0).is_none() {
                    self.table_finalizers.remove(&handle);
                    return;
                }
                self.table_finalizers
                    .insert(handle, TableFinalizerState::Pending);
            }
        }
        self.pending_finalizers.push_back(object);
    }

    /// Marks and sweeps runtime-owned values from `roots`.
    ///
    /// The global environment is always rooted. A full collection abandons
    /// any active cycle and runs a fresh major cycle to completion.
    /// Unreachable finalizable userdata are moved to a VM-owned execution
    /// queue and survive this cycle.
    pub fn collect_full(&mut self, roots: GcRoots) -> GcStats {
        self.refresh_host_roots();
        // What the minor collections since the last cycle promoted dies
        // here, not in an automatic major cycle they would burden.
        self.gc_bad_minors = 0;
        self.incremental_gc = None;
        self.start_gc_cycle(Some(&roots));
        // Nothing runs until the cycle ends, so trace the roots' graph to
        // the end at once, on several threads if it is large, rather than
        // in incremental steps.
        if let Some(mut cycle) = self.incremental_gc.take() {
            self.gc_propagate_all(&mut cycle.marker);
            self.incremental_gc = Some(cycle);
        }
        let mut roots = Some(roots);
        let stats = loop {
            if let Some(stats) = self.gc_major_work(&mut roots, usize::MAX) {
                break stats;
            }
        };
        // Explicit full collection never yields to the mutator, so large
        // dead tables can use their normal destructor here.
        self.gc_detached.clear();
        // The array buffers and one-entry blocks of the tables that died wait
        // in lists.
        crate::runtime::table_storage::ordered_vec::trim_pool();
        crate::runtime::table_storage::array::trim_pool();
        stats
    }

    pub(crate) fn start_gc_cycle(&mut self, roots: Option<&GcRoots>) {
        // Only the generational major/minor policy reads the starting
        // size, and counting it walks every object.
        if self.gc_major_minor.is_some() && self.gc_mode == GcMode::Generational {
            self.gc_major_start_bytes = self.gc_count_bytes();
        }
        self.begin_gc_epoch();
        self.gc_partial_table = None;
        let mut marker = Marker::new(self.gc_epoch, false);
        self.gc_mark_roots(roots, &mut marker);
        self.incremental_gc = Some(Box::new(IncrementalGc {
            phase: IncrementalGcPhase::Mark,
            marker,
            partial: None,
            gray_again: Vec::new(),
            sweep: SweepCursor::default(),
            stats: GcStats::default(),
            upvalue_after: None,
            survivor_bytes: 0,
            debt_at_atomic: 0,
        }));
    }

    /// Removes the next queued finalizer in reverse registration order.
    ///
    /// The object is marked finalized before its handler runs, so a handler
    /// that raises an error cannot cause a second invocation.
    ///
    /// # Errors
    ///
    /// Returns an error if a queued userdata or its metatable became invalid.
    pub fn take_pending_finalizer(&mut self) -> RuntimeResult<Option<Finalizer>> {
        while let Some(object) = self.pending_finalizers.pop_front() {
            let (object, metatable) = match object {
                FinalizerObject::Userdata(handle) => {
                    let userdata = self.userdata_mut(handle)?;
                    userdata.finalizer = UserdataFinalizerState::Finalized;
                    (Value::Userdata(handle), userdata.metatable)
                }
                FinalizerObject::Table(handle) => {
                    // A finalized table may be registered again by a later
                    // `setmetatable`, as in Lua.
                    self.table_finalizers.remove(&handle);
                    let Some(table) = self.tables.get(handle.0) else {
                        continue;
                    };
                    (Value::Table(handle), table.metatable)
                }
            };
            let Some(metatable) = metatable else {
                continue;
            };
            let handler = self.table_get(metatable, &Value::from("__gc"))?;
            if matches!(handler, Value::Nil) {
                continue;
            }
            return Ok(Some(Finalizer { object, handler }));
        }
        Ok(None)
    }

    /// Stops allocation-triggered collection. Explicit `collect` and `step`
    /// requests still run while the collector is stopped.
    pub fn stop_gc(&mut self) {
        self.gc_running = false;
    }

    /// Enables allocation-triggered collection after a previous [`Self::stop_gc`].
    pub fn restart_gc(&mut self) {
        self.gc_running = true;
    }

    /// Makes weak-key tables ephemerons, as Lua 5.2 introduced: an entry
    /// keeps its value alive only while its key is reachable from outside
    /// the entry. Disabled, weak-key tables mark their values strongly, as
    /// in Lua 5.1 and LuaJIT.
    pub const fn set_ephemerons_enabled(&mut self, enabled: bool) {
        self.ephemerons_enabled = enabled;
    }

    /// Sets how many threads trace in the collector's stop-the-world
    /// phases, including the collecting thread. One traces on that thread
    /// alone. The default is `OXIMOND_GC_THREADS`, or the available
    /// parallelism up to four.
    pub fn set_gc_mark_threads(&mut self, threads: usize) {
        self.gc_parallel.threads = threads.max(1);
    }

    /// Moves tracing to `threads` threads at the first gray object and
    /// splits arrays into ranges of two, so small test graphs take the
    /// parallel paths.
    #[cfg(test)]
    pub(crate) fn force_parallel_marking(&mut self, threads: usize) {
        self.gc_parallel = super::ParallelMarking {
            threads,
            gray_threshold: 0,
            array_split: 2,
        };
    }

    /// Reports whether allocation-triggered collection is enabled.
    #[must_use]
    pub const fn gc_is_running(&self) -> bool {
        self.gc_running
    }

    /// Returns whether the VM should call [`Self::gc_automatic_step`] at its
    /// next safe point.
    #[must_use]
    #[inline]
    pub fn gc_needs_collection(&self) -> bool {
        self.gc_running && (self.gc_debt >= self.gc_threshold || self.gc_awaiting_roots)
    }

    /// Returns whether the next automatic step needs a VM root snapshot:
    /// it will run a minor collection, start a major cycle, or reach the
    /// major cycle's atomic step. Other steps ignore roots, so the VM can
    /// skip building them.
    #[must_use]
    pub fn gc_step_needs_roots(&self) -> bool {
        self.incremental_gc.as_ref().is_none_or(|cycle| {
            cycle.phase == IncrementalGcPhase::Mark
                && cycle.partial.is_none()
                && cycle.marker.gray.is_empty()
        })
    }

    /// Runs allocation-paced collector work: a minor collection, or one step
    /// of a major cycle that covers about `kilobytes` of allocation. Supply
    /// `roots` when [`Self::gc_step_needs_roots`] asks for them.
    pub fn gc_automatic_step(
        &mut self,
        mut roots: Option<GcRoots>,
        kilobytes: usize,
    ) -> GcStepResult {
        self.refresh_host_roots();
        if self.incremental_gc.is_none()
            && (self.gc_minors_active() || self.gc_trial_minor_due())
            && !self.gc_major_requested
        {
            let Some(held) = roots.take() else {
                self.gc_awaiting_roots = true;
                return GcStepResult::InProgress;
            };
            self.gc_awaiting_roots = false;
            let paused = !self.gc_minors_active();
            if paused && !self.gc_trial_minor_pays(&held) {
                if self.gc_debt < self.gc_pause_major_debt {
                    // Still building: look again after another nursery,
                    // unless the major cycle is due first.
                    let spacing =
                        (self.gc_nursery_bytes() / GC_TRIAL_SPACING_DIVISOR) << self.gc_trial_fails;
                    self.gc_trial_fails = (self.gc_trial_fails + 1).min(2);
                    self.gc_threshold = self
                        .gc_debt
                        .saturating_add(spacing)
                        .min(self.gc_pause_major_debt);
                    return GcStepResult::Complete;
                }
                // The last trial before the major cycle failed too: start
                // the cycle.
                self.gc_final_trial_done = true;
                roots = Some(held);
            } else if paused {
                // The paused major cycles left floating garbage in the old
                // generation (everything allocated while one marked), and
                // they sized the nursery for it. A major cycle clears it,
                // so the minor collections that resume start from the live
                // heap. It runs incrementally, and its end lifts the pause.
                // It runs fast enough to finish within a share of a
                // nursery, since what is allocated while it marks floats.
                self.gc_minor_pause = 1;
                let heap = self.gc_old_bytes.saturating_add(self.gc_debt);
                let window = (self.gc_nursery_bytes() / GC_CATCH_UP_NURSERY_DIVISOR).max(1);
                self.gc_catch_up = heap.div_ceil(window).clamp(1, GC_MAX_CATCH_UP);
                roots = Some(held);
            } else {
                self.gc_minor_collection(&held);
                self.gc_drain_detached(GC_DETACHED_DRAIN_CHUNK);
                return GcStepResult::Complete;
            }
        }
        let interval = self
            .gc_step_bytes
            .unwrap_or_else(|| kilobytes.max(1).saturating_mul(1024))
            .max(1);
        let units = (scale_gc_work(interval, self.gc_step_multiplier) / GC_INCREMENTAL_UNIT_BYTES)
            .max(1)
            .saturating_mul(self.gc_catch_up);
        let result = self.gc_step_units(roots, units);
        if result == GcStepResult::InProgress {
            self.gc_threshold = self.gc_debt.saturating_add(interval);
        }
        result
    }

    /// Does bounded major-cycle work, as `collectgarbage("step")` does.
    /// `roots` must describe the VM's current registers and stacks.
    /// The budget counts structural heap work, not root construction,
    /// allocator operations, or host resource cleanup such as buffered file
    /// flushing.
    pub fn gc_step_with_roots(&mut self, roots: GcRoots, kilobytes: usize) -> GcStepResult {
        self.refresh_host_roots();
        let requested = kilobytes.saturating_add(1).saturating_mul(1024);
        let units = (scale_gc_work(requested, self.gc_step_multiplier).max(1)
            / GC_INCREMENTAL_UNIT_BYTES)
            .max(1);
        self.gc_step_units(Some(roots), units)
    }

    fn gc_step_units(&mut self, mut roots: Option<GcRoots>, units: usize) -> GcStepResult {
        if self.incremental_gc.is_none() {
            self.start_gc_cycle(roots.as_ref());
        }
        let mut budget = units;
        // Large dead tables from earlier collections share the budget.
        budget -= self.gc_drain_detached(budget / 2);
        match self.gc_major_work(&mut roots, budget.max(1)) {
            Some(_) => GcStepResult::Complete,
            None => GcStepResult::InProgress,
        }
    }

    /// Advances the active major cycle by about `budget` units. Returns the
    /// cycle's statistics once it completes. Reaching the atomic step
    /// consumes `roots`; without them the step stops there and asks the VM
    /// for a snapshot.
    fn gc_major_work(&mut self, roots: &mut Option<GcRoots>, budget: usize) -> Option<GcStats> {
        let mut cycle = self.incremental_gc.take()?;
        let mut remaining = budget;
        while remaining > 0 {
            match cycle.phase {
                IncrementalGcPhase::Mark => {
                    cycle.marker.work = 0;
                    if !self.gc_mark_step(&mut cycle, remaining) {
                        let Some(roots) = roots.take() else {
                            self.gc_awaiting_roots = true;
                            break;
                        };
                        self.gc_awaiting_roots = false;
                        self.gc_atomic(&mut cycle, &roots);
                    }
                    remaining = remaining.saturating_sub(cycle.marker.work.max(1));
                }
                IncrementalGcPhase::Sweep => {
                    if !self.gc_sweep_slots(&mut cycle, GC_SWEEP_SLOTS_PER_UNIT) {
                        cycle.phase = IncrementalGcPhase::Cleanup;
                    }
                    remaining -= 1;
                }
                IncrementalGcPhase::Cleanup => {
                    if self.gc_cleanup_upvalue_one(&mut cycle) {
                        remaining -= 1;
                    } else if !self.gc_detached.is_empty() {
                        remaining -= self.gc_drain_detached(remaining).max(1).min(remaining);
                    } else {
                        let stats = cycle.stats;
                        let allocated = self.gc_debt.saturating_sub(cycle.debt_at_atomic);
                        self.finish_major_cycle(cycle.survivor_bytes, allocated);
                        return Some(stats);
                    }
                }
            }
        }
        self.incremental_gc = Some(cycle);
        None
    }

    /// Traces gray objects for about `budget` units. Returns `false` when
    /// no marking work is left before the atomic step.
    fn gc_mark_step(&mut self, cycle: &mut IncrementalGc, budget: usize) -> bool {
        while cycle.marker.work < budget {
            if let Some(mut scan) = cycle.partial.take() {
                if self.gc_partial_dirty {
                    // The mutator changed the table mid-trace; entries may
                    // have moved behind the cursor. Trace it whole later.
                    self.gc_partial_dirty = false;
                    self.gc_partial_table = None;
                    cycle.gray_again.push(scan.table);
                    continue;
                }
                let limit = (budget - cycle.marker.work).max(GC_TABLE_SCAN_CHUNK);
                if self.gc_continue_table_scan(&mut scan, &mut cycle.marker, limit) {
                    cycle.partial = Some(scan);
                } else {
                    self.gc_partial_table = None;
                }
                continue;
            }
            let Some(reference) = cycle.marker.gray.pop() else {
                return false;
            };
            if let Some(scan) =
                self.gc_traverse(reference, &mut cycle.marker, &TableTraversal::Chunked)
            {
                self.gc_partial_table = Some(scan.table);
                self.gc_partial_dirty = false;
                cycle.partial = Some(scan);
            }
        }
        true
    }

    /// The non-incremental end of marking. See the module documentation.
    fn gc_atomic(&mut self, cycle: &mut IncrementalGc, roots: &GcRoots) {
        let marker = &mut cycle.marker;
        self.gc_mark_roots(Some(roots), marker);
        for table in std::mem::take(&mut cycle.gray_again) {
            self.gc_traverse(
                crate::runtime::gc::GcReference::Table(table),
                marker,
                &TableTraversal::Full,
            );
        }
        self.gc_converge(marker);
        self.gc_clear_weak_tables(marker, true);
        self.gc_separate_finalizers(marker);
        self.gc_clear_weak_tables(marker, false);
        marker.weak_tables = Vec::new();
        self.gc_partial_table = None;
        self.gc_forget_generations();
        // Objects allocated from here on survive the sweep but stay young.
        self.gc_set_alloc_epoch(self.gc_epoch.wrapping_add(1));
        cycle.debt_at_atomic = self.gc_debt;
        cycle.phase = IncrementalGcPhase::Sweep;
    }

    /// Sweeps up to `slots` arena slots. Returns `false` once every arena is
    /// swept. The arena is chosen once per batch and `Arena::sweep_major_run`
    /// loops over its slots.
    fn gc_sweep_slots(&mut self, cycle: &mut IncrementalGc, mut slots: usize) -> bool {
        let epoch = self.gc_epoch;
        let stats = &mut cycle.stats;
        let survivors = &mut cycle.survivor_bytes;
        let cursor = &mut cycle.sweep.cursor;
        while slots > 0 {
            let done = match cycle.sweep.arena {
                0 => {
                    let detached = &mut self.gc_detached;
                    self.tables.sweep_major_run(
                        epoch,
                        cursor,
                        &mut slots,
                        |table| *survivors += table_accounted_bytes(table),
                        |table| {
                            stats.tables += 1;
                            Self::gc_release_table(detached, table);
                        },
                    )
                }
                1 => self.closures.sweep_major_run(
                    epoch,
                    cursor,
                    &mut slots,
                    |closure| *survivors += closure_accounted_bytes(closure),
                    |_| stats.closures += 1,
                ),
                2 => self.cells.sweep_major_run(
                    epoch,
                    cursor,
                    &mut slots,
                    |_| *survivors += std::mem::size_of::<Cell>(),
                    |_| stats.cells += 1,
                ),
                3 => self.files.sweep_major_run(
                    epoch,
                    cursor,
                    &mut slots,
                    |file| *survivors += file_accounted_bytes(file),
                    |_| stats.files += 1,
                ),
                4 => {
                    let registry = &mut self.finalizer_registry;
                    let freed_userdata = &mut self.gc_freed_userdata;
                    self.userdatas.sweep_major_run(
                        epoch,
                        cursor,
                        &mut slots,
                        |userdata| *survivors += userdata.accounted_bytes(),
                        |userdata| {
                            stats.userdatas += 1;
                            *freed_userdata += 1;
                            if let UserdataFinalizerState::Registered(order) = userdata.finalizer {
                                registry.remove(&order);
                            }
                        },
                    )
                }
                5 => self.gmatch_iterators.sweep_major_run(
                    epoch,
                    cursor,
                    &mut slots,
                    |iterator| *survivors += gmatch_iterator_accounted_bytes(iterator),
                    |_| stats.gmatch_iterators += 1,
                ),
                6 => self.threads.sweep_major_run(
                    epoch,
                    cursor,
                    &mut slots,
                    |thread| *survivors += thread_accounted_bytes(thread),
                    |_| stats.threads += 1,
                ),
                _ => return false,
            };
            if done {
                cycle.sweep.arena += 1;
                *cursor = crate::runtime::arena::MajorSweepCursor::default();
            }
        }
        true
    }

    /// Hands a dead table to the detached queue when it is large; a small
    /// one is dropped by the sweep where it lies.
    #[inline(always)]
    pub(super) fn gc_release_table(
        detached: &mut std::collections::VecDeque<Table>,
        table: &mut Table,
    ) {
        if table_reclaim::eligible(table) {
            Self::gc_detach_table(detached, table);
        }
    }

    /// The large-table half of `gc_release_table`, kept out of the sweep loop.
    #[cold]
    #[inline(never)]
    fn gc_detach_table(detached: &mut std::collections::VecDeque<Table>, table: &mut Table) {
        table_reclaim::prepare(table);
        detached.push_back(std::mem::take(table));
    }

    /// Releases up to about `budget` elements of large dead tables. Returns
    /// the units used.
    pub(crate) fn gc_drain_detached(&mut self, budget: usize) -> usize {
        let mut used = 0;
        while used < budget {
            let Some(table) = self.gc_detached.front_mut() else {
                break;
            };
            used += 1;
            // The array part goes a run at a time: truncating drops the run
            // in one tight loop, where popping each value costs several
            // times as much per value (a dead VoxelManip buffer holds
            // hundreds of thousands).
            let len = table.array.len();
            let mut drained = len.min(GC_DETACHED_DRAIN_CHUNK);
            table.array.truncate(len - drained);
            while drained < GC_DETACHED_DRAIN_CHUNK
                && (table.entries.pop_first().is_some()
                    || table
                        .deleted_array_keys
                        .pop_first(table.array.len())
                        .is_some()
                    || table.deleted_entries.pop_first().is_some())
            {
                drained += 1;
            }
            if drained == GC_DETACHED_DRAIN_CHUNK {
                continue;
            }
            let table = self.gc_detached.pop_front().expect("front table exists");
            if let Err(table) = table_reclaim::release(table) {
                // The reclaim worker is busy; retry on a later step.
                self.gc_detached.push_front(table);
                break;
            }
        }
        used
    }

    fn gc_cleanup_upvalue_one(&mut self, cycle: &mut IncrementalGc) -> bool {
        use std::ops::Bound::{Excluded, Unbounded};
        let next = self
            .upvalue_identities
            .range((
                cycle.upvalue_after.as_ref().map_or(Unbounded, Excluded),
                Unbounded,
            ))
            .next()
            .map(|(cell, identity)| (*cell, *identity));
        let Some((cell, identity)) = next else {
            return false;
        };
        cycle.upvalue_after = Some(cell);
        if self.cells.get(cell.0).is_none() || self.userdatas.get(identity.0).is_none() {
            self.upvalue_identities.remove(&cell);
        }
        true
    }

    /// Starts a new mark epoch. Epochs advance by two: a major cycle gives
    /// objects allocated after its atomic step the odd epoch that follows
    /// its own (see [`Self::gc_set_alloc_epoch`]).
    pub(crate) fn begin_gc_epoch(&mut self) {
        self.gc_epoch = self.gc_epoch.checked_add(2).unwrap_or(2);
        self.gc_set_alloc_epoch(self.gc_epoch);
    }

    /// Sets the mark epoch new objects are born with.
    fn gc_set_alloc_epoch(&mut self, epoch: u64) {
        self.tables.alloc_epoch = epoch;
        self.closures.alloc_epoch = epoch;
        self.cells.alloc_epoch = epoch;
        self.files.alloc_epoch = epoch;
        self.userdatas.alloc_epoch = epoch;
        self.gmatch_iterators.alloc_epoch = epoch;
        self.threads.alloc_epoch = epoch;
    }

    /// Ends a major cycle that kept `survivor_bytes` of the objects it
    /// swept, after `allocated` bytes of new objects since its atomic step.
    fn finish_major_cycle(&mut self, survivor_bytes: usize, allocated: usize) {
        if self.gc_mode == GcMode::Generational {
            if let Some(major_minor) = self.gc_major_minor {
                let added = self.gc_major_start_bytes.saturating_sub(self.gc_major_base);
                let collected = self.gc_major_start_bytes.saturating_sub(survivor_bytes);
                self.gc_generational = collected as f64 > added as f64 * major_minor / 100.0;
            }
        }
        self.gc_trial_prev_young = 0;
        self.gc_final_trial_done = false;
        self.gc_big_trials = 0;
        self.gc_debt = 0;
        self.gc_awaiting_roots = false;
        self.gc_major_requested = false;
        self.gc_minor_pause = self.gc_minor_pause.saturating_sub(1);
        self.gc_catch_up = 1;
        self.gc_old_bytes = survivor_bytes;
        self.gc_major_base = survivor_bytes;
        // Every elided table charged before this point is unreachable, so
        // the finished cycle retires its phantom bytes along with the real
        // garbage it swept.
        self.elided_bytes_total += self.elided_table_bytes as u64;
        self.elided_table_bytes = 0;
        self.gc_threshold = if self.gc_minors_active() {
            self.gc_nursery_bytes()
        } else {
            // Objects allocated after the atomic step survive it without
            // being counted by the sweep.
            self.gc_paused_threshold(survivor_bytes.saturating_add(allocated))
        };
    }

    /// The debt after which the next major cycle starts, for a heap of
    /// `heap` bytes. While the default policy pauses minor collections,
    /// major cycles run no more often than the minor collections did.
    pub(crate) fn gc_major_debt(&self, heap: usize) -> usize {
        let debt = self.gc_pause_debt(heap);
        if self.gc_generational {
            debt.max(GC_MIN_NURSERY_BYTES)
        } else {
            debt
        }
    }

    /// Returns the allocation debt after which the next major cycle starts:
    /// like Lua, it begins once the heap reaches `pause` percent of `heap`.
    /// Charging the whole scaled heap as debt would let every cycle start
    /// later than the one before, since objects allocated during a cycle
    /// survive it.
    fn gc_pause_debt(&self, heap: usize) -> usize {
        scale_gc_work(heap, self.gc_pause)
            .saturating_sub(heap)
            .max(GC_MIN_THRESHOLD_BYTES)
    }

    /// Returns deterministic accounted bytes for currently live GC objects.
    #[must_use]
    pub fn gc_count_bytes(&self) -> usize {
        self.tables.accounted_bytes(table_accounted_bytes)
            + self.closures.accounted_bytes(closure_accounted_bytes)
            + self.cells.accounted_bytes(|_| std::mem::size_of::<Cell>())
            + self.files.accounted_bytes(file_accounted_bytes)
            + self.userdatas.accounted_bytes(Userdata::accounted_bytes)
            + self
                .gmatch_iterators
                .accounted_bytes(gmatch_iterator_accounted_bytes)
            + self.threads.accounted_bytes(thread_accounted_bytes)
            + self.elided_table_bytes
    }

    /// Changes when collection starts, so native adapters can retire dead buffers.
    #[must_use]
    pub fn gc_collection_epoch(&self) -> u64 {
        self.gc_epoch
    }

    /// Counts the userdata collection has freed. A host that mirrors
    /// userdata (the C API) rescans its records only when this changes.
    #[must_use]
    pub const fn gc_freed_userdata(&self) -> u64 {
        self.gc_freed_userdata
    }

    /// Returns the configured full-GC pause value.
    #[must_use]
    pub const fn gc_pause(&self) -> f64 {
        self.gc_pause
    }

    /// Replaces the stored pause setting and returns its previous value.
    pub fn set_gc_pause(&mut self, value: f64) -> f64 {
        let previous = self.gc_pause;
        self.gc_pause = value;
        previous
    }

    /// Returns the configured incremental step multiplier.
    #[must_use]
    pub const fn gc_step_multiplier(&self) -> f64 {
        self.gc_step_multiplier
    }

    /// Replaces the stored step multiplier and returns its previous value.
    pub fn set_gc_step_multiplier(&mut self, value: f64) -> f64 {
        let previous = self.gc_step_multiplier;
        self.gc_step_multiplier = value;
        previous
    }

    /// Returns a Lua 5.5 collector parameter by its public parameter index.
    #[must_use]
    pub fn gc_parameter(&self, index: usize) -> Option<f64> {
        match index {
            0 => Some(self.gc_minor_multiplier),
            1 => Some(self.gc_major_minor.unwrap_or(50.0)),
            2 => Some(self.gc_major_multiplier),
            3 => Some(self.gc_pause),
            4 => Some(self.gc_step_multiplier),
            5 => Some(self.gc_step_bytes.unwrap_or(1024) as f64),
            _ => None,
        }
    }

    /// Changes a collector parameter and returns its previous value.
    /// Each setting controls collection pacing or the major/minor policy.
    pub fn set_gc_parameter(&mut self, index: usize, value: f64) -> Option<f64> {
        let previous = self.gc_parameter(index)?;
        let value = value.max(0.0);
        match index {
            0 => {
                self.gc_minor_multiplier = value;
                if self.gc_minors_active() && self.incremental_gc.is_none() {
                    self.gc_threshold = self.gc_nursery_bytes();
                }
            }
            1 => self.gc_major_minor = Some(value),
            2 => self.gc_major_multiplier = value,
            3 => {
                self.set_gc_pause(value);
            }
            4 => {
                self.set_gc_step_multiplier(value);
            }
            5 => self.gc_step_bytes = Some(value as usize),
            _ => unreachable!("parameter index checked above"),
        }
        Some(previous)
    }

    #[must_use]
    pub const fn gc_mode(&self) -> GcMode {
        self.gc_mode
    }

    /// Selects the collector policy. Generational mode runs minor
    /// collections between incremental major cycles; incremental mode runs
    /// only major cycles. The runtime starts generational while reporting
    /// Lua's default mode, `incremental`, until a program selects one.
    pub fn set_gc_mode(&mut self, mode: GcMode) -> GcMode {
        let previous = self.gc_mode;
        self.gc_mode = mode;
        self.gc_generational = mode == GcMode::Generational;
        self.gc_bad_minors = 0;
        self.gc_minor_pause = 0;
        self.gc_minor_backoff = 1;
        if !self.gc_generational && self.incremental_gc.is_none() {
            // Pace the next major cycle by the heap size, as Lua does.
            self.gc_threshold = self.gc_pause_debt(self.gc_major_base);
        }
        previous
    }

    #[inline]
    pub(crate) fn gc_charge(&mut self, bytes: usize) {
        self.gc_debt = self.gc_debt.saturating_add(bytes);
        if bytes != 0 {
            self.allocation_serial = self.allocation_serial.saturating_add(1);
        }
    }

    /// Charges heap use for one table whose allocation was elided.
    ///
    /// The bytes behave like a real table's for pacing and counting: they
    /// accrue collector debt, appear in `gcinfo`/`collectgarbage("count")`,
    /// and are released when a collection cycle finishes. Because the table
    /// never escapes, every charged byte is dead by construction, so cycle
    /// completion is the correct release point.
    pub fn charge_elided_table(&mut self, field_count: usize) {
        let bytes = Self::elided_footprint(field_count);
        self.elided_table_bytes = self.elided_table_bytes.saturating_add(bytes);
        self.elided_tables_total += 1;
        // Accounting only: unlike a real allocation, this charge does not
        // advance the allocation serial, because no arena work happened and
        // fragment-elided programs keep their real-allocation budget.
        self.gc_debt = self.gc_debt.saturating_add(bytes);
    }

    /// Charges `tables` elided tables at once, `bytes` in all, as the sum of
    /// [`Self::charge_elided_table`] calls would. A compiled trace counts the
    /// constructors it ran without calling the runtime, and charges the sums
    /// when its run ends.
    pub fn charge_elided_tables(&mut self, tables: u64, bytes: usize) {
        self.elided_table_bytes = self.elided_table_bytes.saturating_add(bytes);
        self.elided_tables_total += tables;
        self.gc_debt = self.gc_debt.saturating_add(bytes);
    }

    /// The bytes [`Self::charge_elided_table`] charges for a table of
    /// `field_count` entries.
    #[must_use]
    pub fn elided_footprint(field_count: usize) -> usize {
        let per_entry = std::mem::size_of::<(TableKey, Value)>();
        let capacity = match field_count {
            0 => 0,
            count => count.next_power_of_two().max(4),
        };
        crate::runtime::table_storage::TABLE_CHARGED_BYTES + capacity * per_entry
    }

    /// Takes back the charge [`Self::charge_elided_table`] made for a table of
    /// `field_count` entries that has now been allocated for real, so the
    /// table is counted once (by the allocation).
    ///
    /// A charge that a finished cycle already released has left the debt, so
    /// only the part still pending comes off the debt and the pending bytes.
    /// The running totals shrink by the whole charge.
    pub fn refund_elided_table(&mut self, field_count: usize) {
        let bytes = Self::elided_footprint(field_count);
        let pending = bytes.min(self.elided_table_bytes);
        self.elided_table_bytes -= pending;
        self.gc_debt = self.gc_debt.saturating_sub(pending);
        self.elided_bytes_total = self
            .elided_bytes_total
            .saturating_sub((bytes - pending) as u64);
        self.elided_tables_total = self.elided_tables_total.saturating_sub(1);
    }

    /// Heap bytes charged for elided tables and not yet released by a cycle.
    #[must_use]
    pub const fn elided_table_bytes(&self) -> usize {
        self.elided_table_bytes
    }

    /// Tables allocated in the arena since the runtime started.
    #[must_use]
    pub const fn tables_created(&self) -> u64 {
        self.tables_created
    }

    /// Elided tables charged since the runtime started, with their bytes.
    /// Unlike [`Self::elided_table_bytes`], a finished cycle does not reset
    /// these.
    #[must_use]
    pub const fn elided_tables_total(&self) -> (u64, u64) {
        (
            self.elided_tables_total,
            self.elided_bytes_total + self.elided_table_bytes as u64,
        )
    }

    #[must_use]
    pub const fn allocation_serial(&self) -> u64 {
        self.allocation_serial
    }
}
