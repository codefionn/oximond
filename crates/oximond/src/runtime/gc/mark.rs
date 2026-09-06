//! Tracing: marking references, traversing gray objects, converging
//! ephemeron tables, selecting finalizers, and clearing weak tables.
//!
//! Tracing reads the heap through [`Heap`], a view of the arenas that
//! several threads can share. Mark epochs are atomics, so a table can be
//! traversed in place while its children are marked, and the
//! stop-the-world phases can trace on several threads (see `parallel.rs`).

use super::heap::{key_gc_reference, value_gc_reference};
use super::{
    GC_TABLE_SCAN_CHUNK, GcReference, GcRoots, Marker, RuntimeThread, TableScan, WeakMode,
};
use crate::runtime::arena::Arena;
use crate::runtime::closures::{Cell, ClosureBody};
use crate::runtime::files::RuntimeFile;
use crate::runtime::iterators::GmatchIterator;
use crate::runtime::runtime::Runtime;
use crate::runtime::table_storage::GcSlotRange;
use crate::runtime::table_storage::Table;
use crate::runtime::table_storage::key::TableKey;
use crate::runtime::userdata::Userdata;
use crate::runtime::userdata::{FinalizerObject, TableFinalizerState, UserdataFinalizerState};
use crate::runtime::value::{TableHandle, UserdataHandle, Value};
use crate::string::LuaString;

/// How much of a table a traversal may cover.
pub(crate) enum TableTraversal {
    /// Trace the whole table now.
    Full,
    /// Trace small tables whole, and start a resumable scan for large ones.
    Chunked,
    /// Trace everything but a large array part, and return a scan of the
    /// array for the caller to split among threads.
    SplitArray,
    /// Trace the metatable and the array slots in the range, for a remembered
    /// table whose hash part no store touched since the last minor
    /// collection. A weak table, or a range that is whole, is traced in full.
    Dirty(GcSlotRange),
}

/// The runtime as tracing sees it: the arenas, `ephemerons_enabled`,
/// `mode_metamethod_key`, and `gc_parallel`. Marking threads share one.
///
/// It holds the whole runtime rather than a reference per arena, because
/// tracing reads the arenas on every edge and one pointer keeps those
/// reads as short as `&Runtime` methods had them. `Runtime` is not `Sync`
/// (it keeps host natives and lazy caches), so `Heap` is `Send` and `Sync`
/// by the promise that it reads nothing else: every access goes through the
/// accessors below.
#[derive(Clone, Copy)]
pub(crate) struct Heap<'a> {
    runtime: &'a Runtime,
}

const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<Arena<Table>>();
    assert_sync::<Arena<ClosureBody>>();
    assert_sync::<Arena<Cell>>();
    assert_sync::<Arena<RuntimeFile>>();
    assert_sync::<Arena<Userdata>>();
    assert_sync::<Arena<GmatchIterator>>();
    assert_sync::<Arena<RuntimeThread>>();
    assert_sync::<LuaString>();
};

// SAFETY: a `Heap` reads only the fields its accessors name, which are
// `Sync` (asserted above) or plain `Copy` data, and only through shared
// references; marking writes go to atomics in the arena slots. The
// collector creates a `Heap` from `&Runtime`, so nothing mutates the
// runtime while one lives.
#[allow(unsafe_code)]
unsafe impl Sync for Heap<'_> {}
// SAFETY: as for `Sync`; sending a `Heap` shares those same fields.
#[allow(unsafe_code)]
unsafe impl Send for Heap<'_> {}

impl<'a> Heap<'a> {
    #[inline]
    pub(crate) const fn tables(self) -> &'a Arena<Table> {
        &self.runtime.tables
    }

    #[inline]
    const fn closures(self) -> &'a Arena<ClosureBody> {
        &self.runtime.closures
    }

    #[inline]
    const fn cells(self) -> &'a Arena<Cell> {
        &self.runtime.cells
    }

    #[inline]
    const fn files(self) -> &'a Arena<RuntimeFile> {
        &self.runtime.files
    }

    #[inline]
    const fn userdatas(self) -> &'a Arena<Userdata> {
        &self.runtime.userdatas
    }

    #[inline]
    const fn gmatch_iterators(self) -> &'a Arena<GmatchIterator> {
        &self.runtime.gmatch_iterators
    }

    #[inline]
    const fn threads(self) -> &'a Arena<RuntimeThread> {
        &self.runtime.threads
    }

    #[inline]
    const fn ephemerons_enabled(self) -> bool {
        self.runtime.ephemerons_enabled
    }

    #[inline]
    const fn mode_key(self) -> &'a LuaString {
        &self.runtime.mode_metamethod_key
    }

    /// See [`super::parallel::ParallelMarking::array_split`].
    #[inline]
    pub(crate) const fn array_split(self) -> usize {
        self.runtime.gc_parallel.array_split
    }
}

impl Runtime {
    #[inline]
    pub(crate) const fn gc_heap(&self) -> Heap<'_> {
        Heap { runtime: self }
    }

    /// Marks one reference and queues it for traversal if it has children.
    #[inline]
    pub(crate) fn gc_mark_reference(&self, reference: GcReference, marker: &mut Marker) {
        self.gc_heap().mark_reference(reference, marker);
    }

    /// Marks the runtime's permanent roots and, if given, a VM snapshot.
    pub(crate) fn gc_mark_roots(&self, roots: Option<&GcRoots>, marker: &mut Marker) {
        self.gc_mark_reference(GcReference::Table(self.global_environment), marker);
        self.gc_mark_reference(GcReference::Table(self.registry), marker);
        if let Some(table) = self.package_require_table {
            self.gc_mark_reference(GcReference::Table(table), marker);
        }
        for table in self.shared_metatables.iter().flatten() {
            self.gc_mark_reference(GcReference::Table(*table), marker);
        }
        for file in self.default_input.iter().chain(&self.default_output) {
            self.gc_mark_reference(GcReference::File(*file), marker);
        }
        for object in &self.pending_finalizers {
            self.gc_mark_reference(finalizer_reference(*object), marker);
        }
        if let Some(roots) = roots {
            for reference in &roots.references {
                self.gc_mark_reference(*reference, marker);
            }
            for cell in &roots.cells {
                self.gc_mark_reference(GcReference::Cell(*cell), marker);
            }
            for closure in &roots.closures {
                self.gc_mark_reference(GcReference::Closure(*closure), marker);
            }
        }
    }

    /// Traces every gray object, on several threads once there is enough
    /// to trace (see `parallel.rs`).
    pub(crate) fn gc_propagate_all(&self, marker: &mut Marker) {
        super::parallel::propagate(self.gc_heap(), marker, self.gc_parallel);
    }

    /// Traces every gray object, then revisits the ephemeron tables traced
    /// so far until no entry's key becomes reached: a value an ephemeron
    /// holds is marked only once its key is. Tracing a revealed value can
    /// reach keys of other entries, including entries of other ephemeron
    /// tables, hence the fixpoint.
    pub(crate) fn gc_converge(&self, marker: &mut Marker) {
        self.gc_propagate_all(marker);
        if !self.ephemerons_enabled {
            return;
        }
        let heap = self.gc_heap();
        // Marking only queues objects. Weak tables that propagation traces
        // land in the marker's list and join the next round.
        let mut tables = std::mem::take(&mut marker.weak_tables);
        loop {
            // A table traced more than once in this collection, such as one
            // retraced at the atomic step or by two marking threads, needs
            // visiting once per round.
            tables.sort_unstable_by_key(|(handle, _)| *handle);
            tables.dedup_by_key(|(handle, _)| *handle);
            for (handle, mode) in &tables {
                if !heap.is_ephemeron(*mode) {
                    continue;
                }
                let Some(table) = heap.tables().get(handle.0) else {
                    continue;
                };
                marker.work += table.entries.len();
                table.entries.for_each(|key, value| {
                    if heap.key_is_reached(key, marker) {
                        heap.mark_value(value, marker);
                    }
                });
            }
            if marker.gray.is_empty() {
                break;
            }
            self.gc_propagate_all(marker);
            tables.append(&mut marker.weak_tables);
        }
        marker.weak_tables = tables;
    }

    /// Marks the children of one object. A large table under
    /// [`TableTraversal::Chunked`] returns a resumable scan instead of being
    /// traced whole.
    #[inline]
    pub(crate) fn gc_traverse(
        &self,
        reference: GcReference,
        marker: &mut Marker,
        traversal: &TableTraversal,
    ) -> Option<TableScan> {
        self.gc_heap().traverse(reference, marker, traversal)
    }

    /// Traces up to `limit` slots of a resumable table scan. Returns whether
    /// slots remain. The caller guarantees the table did not change since
    /// the scan started.
    pub(crate) fn gc_continue_table_scan(
        &self,
        scan: &mut TableScan,
        marker: &mut Marker,
        limit: usize,
    ) -> bool {
        self.gc_heap().continue_table_scan(scan, marker, limit)
    }

    pub(crate) fn gc_reference_is_reached(&self, reference: GcReference, marker: &Marker) -> bool {
        self.gc_heap().reference_is_reached(reference, marker)
    }

    /// Queues every unreachable object with a registered finalizer, in
    /// reverse registration order, and marks everything it reaches.
    ///
    /// Candidates are selected before any of them is marked, so a graph
    /// shared by several candidates cannot hide the later ones.
    pub(crate) fn gc_separate_finalizers(&mut self, marker: &mut Marker) {
        let candidates = self
            .finalizer_registry
            .iter()
            .rev()
            .filter(|(_, object)| {
                !self.gc_reference_is_reached(finalizer_reference(**object), marker)
            })
            .map(|(order, object)| (*order, *object))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return;
        }
        for (order, object) in &candidates {
            self.finalizer_registry.remove(order);
            self.queue_finalizer(*object);
        }
        for (_, object) in candidates {
            self.gc_mark_reference(finalizer_reference(object), marker);
        }
        self.gc_converge(marker);
    }

    /// Removes weak entries whose referents did not survive this collection.
    /// With `values_only`, only weak values are cleared: Lua does that
    /// before it resurrects objects for their finalizers, so a weak value
    /// reachable only from such an object is gone when the finalizer runs.
    pub(crate) fn gc_clear_weak_tables(&mut self, marker: &Marker, values_only: bool) {
        let mut cleared_indices = Vec::new();
        let mut cleared_keys = Vec::new();
        for (handle, mode) in &marker.weak_tables {
            if values_only && !mode.values {
                continue;
            }
            let Some(table) = self.tables.get(handle.0) else {
                continue;
            };
            if mode.values {
                for (index, value) in table.array.iter().enumerate() {
                    if self.gc_value_is_cleared(value, marker) {
                        cleared_indices.push(index);
                    }
                }
            }
            table.entries.for_each(|key, value| {
                if (!values_only && mode.keys && self.gc_key_is_cleared(key, marker))
                    || (mode.values && self.gc_value_is_cleared(value, marker))
                {
                    cleared_keys.push(key.clone());
                }
            });
            if cleared_indices.is_empty() && cleared_keys.is_empty() {
                continue;
            }
            let Some(table) = self.tables.get_mut(handle.0) else {
                continue;
            };
            if table.watched {
                self.inherit_epoch += 1;
            }
            for index in cleared_indices.drain(..) {
                table.array[index] = Value::Nil;
                if index < table.border {
                    table.border = index;
                }
            }
            for key in cleared_keys.drain(..) {
                table.entries.remove(&key);
            }
        }
    }

    fn gc_key_is_cleared(&self, key: &TableKey, marker: &Marker) -> bool {
        key_gc_reference(key)
            .is_some_and(|reference| !self.gc_reference_is_reached(reference, marker))
    }

    fn gc_value_is_cleared(&self, value: &Value, marker: &Marker) -> bool {
        match value {
            Value::Userdata(handle) => self.gc_userdata_is_cleared(*handle, marker),
            // Like userdata, tables queued for finalization leave weak
            // values at once.
            Value::Table(handle)
                if self.table_finalizers.get(handle) == Some(&TableFinalizerState::Pending) =>
            {
                true
            }
            value => value_gc_reference(value)
                .is_some_and(|reference| !self.gc_reference_is_reached(reference, marker)),
        }
    }

    /// Userdata queued for finalization leave weak values at once, but stay
    /// as weak keys until they are finalized.
    fn gc_userdata_is_cleared(&self, handle: UserdataHandle, marker: &Marker) -> bool {
        self.userdatas.get(handle.0).is_none_or(|userdata| {
            matches!(userdata.finalizer, UserdataFinalizerState::Pending)
                || !self
                    .userdatas
                    .is_reached(handle.0, marker.epoch, marker.minor)
        })
    }
}

impl Heap<'_> {
    /// Marks one reference and queues it for traversal if it has children.
    #[inline]
    pub(crate) fn mark_reference(self, reference: GcReference, marker: &mut Marker) {
        marker.work += 1;
        let (epoch, minor) = (marker.epoch, marker.minor);
        let newly_marked = match reference {
            GcReference::Table(handle) => self.tables().mark(handle.0, epoch, minor),
            GcReference::Closure(handle) => self.closures().mark(handle.0, epoch, minor),
            GcReference::Cell(handle) => self.cells().mark(handle.0, epoch, minor),
            GcReference::Userdata(handle) => self.userdatas().mark(handle.0, epoch, minor),
            GcReference::Thread(handle) => self.threads().mark(handle.0, epoch, minor),
            GcReference::File(handle) => self.files().mark(handle.0, epoch, minor),
            // Iterators have no children.
            GcReference::GmatchIterator(handle) => {
                self.gmatch_iterators().mark(handle.0, epoch, minor);
                false
            }
        };
        if newly_marked {
            marker.gray.push(reference);
        }
    }

    #[inline]
    pub(crate) fn mark_value(self, value: &Value, marker: &mut Marker) {
        if let Some(reference) = value_gc_reference(value) {
            self.mark_reference(reference, marker);
        }
    }

    #[inline]
    fn mark_key(self, key: &TableKey, marker: &mut Marker) {
        if let Some(reference) = key_gc_reference(key) {
            self.mark_reference(reference, marker);
        }
    }

    /// Traces every gray object on this thread.
    pub(crate) fn propagate_all(self, marker: &mut Marker) {
        while let Some(reference) = marker.gray.pop() {
            self.traverse(reference, marker, &TableTraversal::Full);
        }
    }

    /// Marks the children of one object. See [`Runtime::gc_traverse`].
    pub(crate) fn traverse(
        self,
        reference: GcReference,
        marker: &mut Marker,
        traversal: &TableTraversal,
    ) -> Option<TableScan> {
        marker.work += 1;
        match reference {
            GcReference::Table(handle) => return self.traverse_table(handle, marker, traversal),
            GcReference::Closure(handle) => {
                if let Some(closure) = self.closures().get(handle.0) {
                    self.mark_reference(GcReference::Table(closure.environment), marker);
                    for cell in closure.cells.iter() {
                        self.mark_reference(GcReference::Cell(*cell), marker);
                    }
                    for cell in closure.retired_cells() {
                        self.mark_reference(GcReference::Cell(*cell), marker);
                    }
                }
            }
            GcReference::Cell(handle) => {
                if let Some(cell) = self.cells().get(handle.0) {
                    self.mark_value(cell.current(), marker);
                }
            }
            GcReference::Userdata(handle) => {
                if let Some(userdata) = self.userdatas().get(handle.0) {
                    for table in [userdata.metatable, userdata.environment]
                        .into_iter()
                        .flatten()
                    {
                        self.mark_reference(GcReference::Table(table), marker);
                    }
                    for value in &userdata.uservalues {
                        self.mark_value(value, marker);
                    }
                }
            }
            GcReference::Thread(handle) => {
                if let Some(thread) = self.threads().get(handle.0) {
                    for reference in thread
                        .roots
                        .references
                        .iter()
                        .chain(&thread.host_roots.references)
                    {
                        self.mark_reference(*reference, marker);
                    }
                    for cell in thread.roots.cells.iter().chain(&thread.host_roots.cells) {
                        self.mark_reference(GcReference::Cell(*cell), marker);
                    }
                    for closure in thread
                        .roots
                        .closures
                        .iter()
                        .chain(&thread.host_roots.closures)
                    {
                        self.mark_reference(GcReference::Closure(*closure), marker);
                    }
                }
            }
            GcReference::File(handle) => {
                if let Some(file) = self.files().get(handle.0) {
                    self.mark_value(&file.uservalue, marker);
                }
            }
            GcReference::GmatchIterator(_) => {}
        }
        None
    }

    fn traverse_table(
        self,
        handle: TableHandle,
        marker: &mut Marker,
        traversal: &TableTraversal,
    ) -> Option<TableScan> {
        let table = self.tables().get(handle.0)?;
        let mode = self.weak_mode_of(table);
        if mode.is_weak() {
            marker.weak_tables.push((handle, mode));
        }
        if let Some(meta) = table.metatable {
            self.mark_reference(GcReference::Table(meta), marker);
        }
        if let TableTraversal::Dirty(range) = traversal
            && !range.is_whole()
            && !mode.is_weak()
        {
            let end = (range.hi as usize).min(table.array.len());
            let start = (range.lo as usize).min(end);
            marker.work += end - start;
            if !mode.values {
                for value in &table.array[start..end] {
                    self.mark_value(value, marker);
                }
            }
            return None;
        }
        let size = table.array.len() + table.entries.len();
        if matches!(traversal, TableTraversal::Chunked) && size > GC_TABLE_SCAN_CHUNK {
            return Some(TableScan {
                table: handle,
                mode,
                array_index: 0,
                after: None,
                array_done: false,
            });
        }
        let split_array = matches!(traversal, TableTraversal::SplitArray)
            && table.array.len() > self.array_split();
        if split_array {
            marker.work += table.entries.len();
        } else {
            marker.work += size;
            if !mode.values {
                for value in &table.array {
                    self.mark_value(value, marker);
                }
            }
        }
        match (mode.keys, mode.values) {
            (false, false) => {
                if let Some(values) = table.entries.shaped_values() {
                    // Shape keys are strings: only the values hold objects.
                    for value in values {
                        self.mark_value(value, marker);
                    }
                } else {
                    table.entries.for_each(|key, value| {
                        self.mark_key(key, marker);
                        self.mark_value(value, marker);
                    });
                }
            }
            (true, false) if self.ephemerons_enabled() => table.entries.for_each(|key, value| {
                if self.key_is_reached(key, marker) {
                    self.mark_value(value, marker);
                }
            }),
            (true, false) => table
                .entries
                .for_each(|_, value| self.mark_value(value, marker)),
            (false, true) => table.entries.for_each(|key, _| self.mark_key(key, marker)),
            (true, true) => {}
        }
        (split_array && !mode.values).then_some(TableScan {
            table: handle,
            mode,
            array_index: 0,
            after: None,
            array_done: false,
        })
    }

    /// Marks the array values of `table` in `range`, as a split traversal
    /// leaves them.
    pub(crate) fn mark_array_range(
        self,
        table: TableHandle,
        range: std::ops::Range<usize>,
        marker: &mut Marker,
    ) {
        let Some(table) = self.tables().get(table.0) else {
            return;
        };
        let end = range.end.min(table.array.len());
        let start = range.start.min(end);
        marker.work += end - start;
        for value in &table.array[start..end] {
            self.mark_value(value, marker);
        }
    }

    /// See [`Runtime::gc_continue_table_scan`].
    pub(crate) fn continue_table_scan(
        self,
        scan: &mut TableScan,
        marker: &mut Marker,
        limit: usize,
    ) -> bool {
        let Some(table) = self.tables().get(scan.table.0) else {
            return false;
        };
        let mut budget = limit.max(1);
        if !scan.array_done {
            let end = table
                .array
                .len()
                .min(scan.array_index.saturating_add(budget));
            if !scan.mode.values {
                for value in &table.array[scan.array_index.min(end)..end] {
                    self.mark_value(value, marker);
                }
            }
            let visited = end.saturating_sub(scan.array_index);
            marker.work += visited;
            budget = budget.saturating_sub(visited);
            scan.array_index = end;
            if end < table.array.len() {
                return true;
            }
            scan.array_done = true;
            if budget == 0 {
                return true;
            }
        }
        let mode = scan.mode;
        let ephemeron = self.is_ephemeron(mode);
        let mut visited = 0;
        let resume = table
            .entries
            .for_each_after(scan.after.as_ref(), budget, |key, value| {
                visited += 1;
                if !mode.keys {
                    self.mark_key(key, marker);
                }
                if !mode.values && (!ephemeron || self.key_is_reached(key, marker)) {
                    self.mark_value(value, marker);
                }
            });
        marker.work += visited;
        match resume {
            Some(key) => {
                scan.after = Some(key);
                true
            }
            None => false,
        }
    }

    pub(crate) fn reference_is_reached(self, reference: GcReference, marker: &Marker) -> bool {
        let (epoch, minor) = (marker.epoch, marker.minor);
        match reference {
            GcReference::Table(handle) => self.tables().is_reached(handle.0, epoch, minor),
            GcReference::Closure(handle) => self.closures().is_reached(handle.0, epoch, minor),
            GcReference::Cell(handle) => self.cells().is_reached(handle.0, epoch, minor),
            GcReference::File(handle) => self.files().is_reached(handle.0, epoch, minor),
            GcReference::Userdata(handle) => self.userdatas().is_reached(handle.0, epoch, minor),
            GcReference::GmatchIterator(handle) => {
                self.gmatch_iterators().is_reached(handle.0, epoch, minor)
            }
            GcReference::Thread(handle) => self.threads().is_reached(handle.0, epoch, minor),
        }
    }

    /// Whether a table with this mode holds its values only while their
    /// keys are reachable. Entries with unreached keys are traced by
    /// [`Runtime::gc_converge`] once their keys are reached.
    const fn is_ephemeron(self, mode: WeakMode) -> bool {
        self.ephemerons_enabled() && mode.keys && !mode.values
    }

    /// Non-collectable keys, such as numbers and strings, are always
    /// reached.
    fn key_is_reached(self, key: &TableKey, marker: &Marker) -> bool {
        key_gc_reference(key).is_none_or(|reference| self.reference_is_reached(reference, marker))
    }

    /// The weakness the metatable of `table` asks for.
    #[inline]
    fn weak_mode_of(self, table: &Table) -> WeakMode {
        let Some(metatable) = table.metatable else {
            return WeakMode::default();
        };
        let Some(Value::String(mode)) = self
            .tables()
            .get(metatable.0)
            .and_then(|metatable| metatable.entries.get_string(self.mode_key()))
        else {
            return WeakMode::default();
        };
        WeakMode {
            keys: mode.as_bytes().contains(&b'k'),
            values: mode.as_bytes().contains(&b'v'),
        }
    }
}

pub(super) const fn finalizer_reference(object: FinalizerObject) -> GcReference {
    match object {
        FinalizerObject::Userdata(handle) => GcReference::Userdata(handle),
        FinalizerObject::Table(handle) => GcReference::Table(handle),
    }
}
