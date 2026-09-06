//! Collector roots, statistics, modes, and resumable cycle state.
//!
//! The collector is a generational, incremental mark-and-sweep collector.
//! Minor collections trace only young objects plus the remembered set of old
//! objects that may point at them. Major collections trace the whole heap in
//! bounded incremental steps, with an atomic step that rescans the VM roots,
//! selects finalizers, and clears weak tables. See `collector.rs` for the
//! pacing and phase structure, `mark.rs` for tracing, and `generational.rs`
//! for barriers and minor collections.
//!
//! The phases that run while the mutator waits (a full collection, a minor
//! collection, and the atomic step) trace on several threads once their
//! gray set is large; see `parallel.rs`. `OXIMOND_GC_THREADS` sets how many.

mod collector;
mod generational;
pub(crate) mod heap;
mod mark;
mod parallel;

pub(crate) use parallel::ParallelMarking;

use crate::runtime::arena::MajorSweepCursor;
use crate::runtime::table_storage::key::TableKey;
use crate::runtime::value::{
    CellHandle, ClosureHandle, FileHandle, GmatchIteratorHandle, TableHandle, ThreadHandle,
    UserdataHandle, Value,
};
use heap::value_gc_reference;

#[derive(Clone, Copy, Default)]
pub(crate) struct WeakMode {
    pub(crate) keys: bool,
    pub(crate) values: bool,
}

impl WeakMode {
    pub(crate) const fn is_weak(self) -> bool {
        self.keys || self.values
    }
}

/// Opaque roots supplied by the VM at a collection boundary.
///
/// Values are narrowed to their arena references as the roots are built, so
/// a snapshot never allocates for numbers, booleans, or strings, and the
/// collector walks the compact reference vector directly.
#[derive(Clone)]
pub struct GcRoots {
    pub(crate) references: Vec<GcReference>,
    pub(crate) cells: Vec<CellHandle>,
    pub(crate) closures: Vec<ClosureHandle>,
}

impl GcRoots {
    /// An empty builder that reuses the buffers of these roots, for a
    /// caller that roots something new on every call.
    #[must_use]
    pub fn reuse(mut self) -> GcRootBuilder {
        self.references.clear();
        self.cells.clear();
        self.closures.clear();
        GcRootBuilder {
            references: self.references,
            cells: self.cells,
            closures: self.closures,
        }
    }

    /// The number of entries the buffers hold without growing.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.references.capacity() + self.cells.capacity() + self.closures.capacity()
    }
}

pub(crate) struct RuntimeThread {
    pub(crate) roots: GcRoots,
    pub(crate) host_roots: GcRoots,
}

/// Builds roots without exposing arena storage or mark bits.
#[derive(Default)]
pub struct GcRootBuilder {
    references: Vec<GcReference>,
    cells: Vec<CellHandle>,
    closures: Vec<ClosureHandle>,
}

impl GcRootBuilder {
    /// Adds a previously captured VM stack snapshot.
    pub fn extend(&mut self, roots: &GcRoots) {
        self.references.extend_from_slice(&roots.references);
        self.cells.extend_from_slice(&roots.cells);
        self.closures.extend_from_slice(&roots.closures);
    }
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(&mut self, value: &Value) {
        if let Some(reference) = value_gc_reference(value) {
            self.references.push(reference);
        }
    }

    pub fn values<'a>(&mut self, values: impl IntoIterator<Item = &'a Value>) {
        for value in values {
            self.value(value);
        }
    }

    pub fn cell(&mut self, cell: CellHandle) {
        self.cells.push(cell);
    }

    pub fn closure(&mut self, closure: ClosureHandle) {
        self.closures.push(closure);
    }

    #[must_use]
    pub fn finish(self) -> GcRoots {
        GcRoots {
            references: self.references,
            cells: self.cells,
            closures: self.closures,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcStats {
    pub tables: usize,
    pub closures: usize,
    pub cells: usize,
    pub files: usize,
    pub userdatas: usize,
    pub gmatch_iterators: usize,
    pub threads: usize,
}

/// The observable result of an incremental collector step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GcStepResult {
    /// More work remains in the current collection cycle.
    InProgress,
    /// This step completed a collection: a major cycle or a minor collection.
    Complete,
}

/// The collector scheduling policy selected through `collectgarbage`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GcMode {
    Incremental,
    Generational,
}

pub(crate) const GC_MIN_THRESHOLD_BYTES: usize = 4 * 1024;
// A collector unit is one reference edge, table entry, or arena slot. This is
// deliberately small: `stepmul` is byte-based while the implementation's
// resumable work is structural rather than a byte copy.
const GC_INCREMENTAL_UNIT_BYTES: usize = 64;
/// Arena slots swept per collector unit. Visiting a slot is far cheaper than
/// tracing an edge, and Lua likewise charges a swept object a fraction of
/// its size. At one slot per unit, a sweep over a heap of short-lived
/// objects fell behind the allocations made while it ran, so each
/// incremental cycle left more survivors than the last.
const GC_SWEEP_SLOTS_PER_UNIT: usize = 8;
/// Unproductive minor collections in a row after which the default policy
/// pauses minor collections. See [`Runtime::gc_minors_active`].
pub(crate) const GC_BAD_MINOR_LIMIT: u8 = 2;
/// The longest pause, in major cycles, the default policy takes from minor
/// collections.
pub(crate) const GC_MAX_MINOR_PAUSE: u32 = 16;
/// A trial of a paused minor collection gives up once it has done this
/// fraction of the young population in tracing work.
pub(crate) const GC_TRIAL_BUDGET_DIVISOR: usize = 16;
/// A heap this big or bigger gets a last trial before a major cycle starts
/// during a pause. A smaller heap's cycle is cheap enough not to try.
pub(crate) const GC_FINAL_TRIAL_MIN_OLD_BYTES: usize = 16 * 1024 * 1024;
/// Big trials run before the major cycle's own, evenly spaced in debt, so
/// the end of a build is noticed within a share of the cycle's debt.
pub(crate) const GC_BIG_TRIALS_PER_CYCLE: usize = 2;
/// The remembered set of a trial may cost up to this fraction of the old
/// heap's bytes in table slots.
pub(crate) const GC_TRIAL_REMEMBERED_DIVISOR: usize = 8;
/// A trial passes when no more than one in this many of the sampled objects
/// allocated since the previous trial is still reachable.
pub(crate) const GC_TRIAL_RECENT_LIVE_DIVISOR: usize = 8;
/// Objects a trial samples to judge the allocation since the previous one.
pub(crate) const GC_TRIAL_SAMPLES: usize = 1024;
/// Bytes an old object takes on average, to turn old heap bytes into a count.
pub(crate) const GC_TRIAL_BYTES_PER_OBJECT: usize = 64;
/// A paused default policy tries a minor collection after this fraction of
/// a nursery of allocation.
pub(crate) const GC_TRIAL_SPACING_DIVISOR: usize = 4;
/// Minor collections that sweep fewer young objects than this say nothing
/// about the generational hypothesis.
pub(crate) const GC_MINOR_VERDICT_OBJECTS: usize = 256;
/// The major cycle that ends a pause of minor collections finishes within
/// this fraction of a nursery of allocation. Everything allocated while it
/// marks survives it, so at the normal pace, which spreads the cycle over
/// about a heap's worth of allocation, it would leave the resumed minor
/// collections a heap of floating garbage they cannot free.
pub(crate) const GC_CATCH_UP_NURSERY_DIVISOR: usize = 16;
/// The fastest catch-up pace, in multiples of the normal one.
pub(crate) const GC_MAX_CATCH_UP: usize = 64;
/// The smallest nursery: bytes allocated between two minor collections.
///
/// A minor collection traces every young object that is still reachable, so
/// what it costs per byte allocated falls as the nursery grows past the
/// working set of the loop that allocates. At 256 KiB a tree of 4095 nodes
/// (about 400 KiB) was half built at each collection: most of the nursery
/// survived, the default policy gave up on minor collections, and major
/// cycles swept the trees. At 1 MiB `algorithm/binary_trees` runs 19% fewer
/// instructions, `allocation/gc_cycles` 18% and `opt/retained_slots` 5%;
/// 2 MiB and 4 MiB gain 3% more on the first and nothing on the others. The
/// price is up to 768 KiB more garbage held before a collection.
pub(crate) const GC_MIN_NURSERY_BYTES: usize = 1024 * 1024;
/// Tables with more slots than this are traced in resumable chunks during an
/// incremental major cycle.
const GC_TABLE_SCAN_CHUNK: usize = 1024;
/// Dead-table elements released per unit of drain work.
const GC_DETACHED_DRAIN_CHUNK: usize = 64;

impl GcStats {
    #[must_use]
    pub const fn reclaimed(self) -> usize {
        self.tables
            + self.closures
            + self.cells
            + self.files
            + self.userdatas
            + self.gmatch_iterators
            + self.threads
    }
}

/// Tracing state shared by minor collections and incremental major cycles.
pub(crate) struct Marker {
    pub(crate) epoch: u64,
    /// Minor collections treat old objects as marked.
    pub(crate) minor: bool,
    /// Marked objects whose children have not been traced yet.
    pub(crate) gray: Vec<GcReference>,
    /// Weak tables traced in this collection, cleared at its atomic step.
    pub(crate) weak_tables: Vec<(TableHandle, WeakMode)>,
    /// Units of tracing work done since the counter was last reset.
    pub(crate) work: usize,
}

impl Marker {
    pub(crate) const fn new(epoch: u64, minor: bool) -> Self {
        Self {
            epoch,
            minor,
            gray: Vec::new(),
            weak_tables: Vec::new(),
            work: 0,
        }
    }
}

/// A resumable major collection cycle.
pub(crate) struct IncrementalGc {
    pub(crate) phase: IncrementalGcPhase,
    pub(crate) marker: Marker,
    /// A large table whose tracing spans several steps.
    pub(crate) partial: Option<TableScan>,
    /// Partially traced tables that the mutator changed. The atomic step
    /// traces them again in full.
    pub(crate) gray_again: Vec<TableHandle>,
    sweep: SweepCursor,
    stats: GcStats,
    upvalue_after: Option<CellHandle>,
    survivor_bytes: usize,
    /// `gc_debt` at the atomic step. Objects allocated after it survive
    /// the sweep uncounted; the debt since then stands in for them.
    debt_at_atomic: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IncrementalGcPhase {
    /// Incremental tracing from the roots captured at the cycle start.
    Mark,
    /// Incremental sweep after the atomic step.
    Sweep,
    /// Upvalue-identity cleanup and release of large dead tables.
    Cleanup,
}

pub(crate) struct TableScan {
    pub(crate) table: TableHandle,
    pub(crate) mode: WeakMode,
    pub(crate) array_index: usize,
    /// The last traced entry key, or `None` before the entry part starts.
    pub(crate) after: Option<TableKey>,
    pub(crate) array_done: bool,
}

#[derive(Default)]
struct SweepCursor {
    arena: u8,
    cursor: MajorSweepCursor,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GcReference {
    Table(TableHandle),
    Closure(ClosureHandle),
    Cell(CellHandle),
    File(FileHandle),
    Userdata(UserdataHandle),
    GmatchIterator(GmatchIteratorHandle),
    Thread(ThreadHandle),
}
