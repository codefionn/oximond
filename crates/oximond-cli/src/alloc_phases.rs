//! Phase-split allocation counts, compiled only with the `alloc-phases`
//! feature.
//!
//! [`PhasedAllocator`] replaces the plain counting allocator. It keeps the
//! same global counters with the same definitions (see `alloc.rs`), and also
//! splits them by the allocating thread's current
//! [`oximond::runtime::alloc_phase::Phase`]:
//!
//! - An allocation is one successful `alloc`, `alloc_zeroed` or `realloc`
//!   call. It counts once, in the phase current on the thread that makes the
//!   call, with its requested size as cumulative bytes. A `realloc` counts as
//!   one allocation of the new size, as in the plain build.
//! - Each block carries a hidden header that records its origin phase: the
//!   phase of the call that last allocated or reallocated it. A release is
//!   a `dealloc`, or the old size of a successful `realloc`. It is counted
//!   under (origin phase, releasing phase), so the report shows what each
//!   phase frees and what another phase frees for it.
//! - Origin live bytes are the requested bytes of live blocks per origin
//!   phase, and their peak is a true per-origin peak, kept from the headers,
//!   not from frees seen in the current phase.
//! - The active peak of a phase is the highest global live byte total right
//!   after an allocation in that phase. It says how high memory stood while
//!   the phase allocated, not what the phase holds.
//!
//! Neither peak sums to the global peak. Every allocation lands in exactly
//! one phase, so at a quiescent point the phase counts and bytes sum to the
//! global ones, and released plus origin live bytes sum to cumulative bytes.
//!
//! The header is `max(16, align)` bytes in front of every block. It changes
//! block addresses and the system allocator's footprint, but no counted
//! size: the counters see the caller's layout only. Its bytes are reported
//! separately as instrumentation overhead.

#![allow(unsafe_code)]
#![allow(
    clippy::declare_interior_mutable_const,
    reason = "the ZERO constants only initialize the statics"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use oximond::runtime::alloc_phase::{self, Phase};

use crate::alloc;

/// The `schema` field of [`report_json`].
pub const SCHEMA: &str = "oximond-alloc-phases/1";

const PHASES: usize = Phase::COUNT;
/// Thread classes: the thread that called [`mark_vm_thread`], and others.
const THREADS: usize = 2;
const HEADER: usize = 16;

const ZERO: AtomicU64 = AtomicU64::new(0);
const ZERO_ROW: [AtomicU64; PHASES] = [ZERO; PHASES];
const ZERO_THREADS: [AtomicU64; THREADS] = [ZERO; THREADS];

static COUNT: [[AtomicU64; THREADS]; PHASES] = [ZERO_THREADS; PHASES];
static BYTES: [[AtomicU64; THREADS]; PHASES] = [ZERO_THREADS; PHASES];
static ACTIVE_PEAK: [AtomicU64; PHASES] = [ZERO; PHASES];
static ORIGIN_LIVE: [AtomicU64; PHASES] = [ZERO; PHASES];
static ORIGIN_PEAK: [AtomicU64; PHASES] = [ZERO; PHASES];
/// Indexed by origin phase, then releasing phase.
static RELEASED_COUNT: [[AtomicU64; PHASES]; PHASES] = [ZERO_ROW; PHASES];
static RELEASED_BYTES: [[AtomicU64; PHASES]; PHASES] = [ZERO_ROW; PHASES];
static HEADER_BYTES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // Const-initialized and `Copy`: reading it never allocates.
    static VM: Cell<bool> = const { Cell::new(false) };
}

/// Counts this thread's allocations as the VM thread's from now on.
pub fn mark_vm_thread() {
    VM.with(|vm| vm.set(true));
}

fn thread_class() -> usize {
    usize::from(!VM.try_with(Cell::get).unwrap_or(false))
}

fn header(align: usize) -> usize {
    HEADER.max(align)
}

/// The layout of a block with its header, or `None` on overflow.
fn outer(layout: Layout) -> Option<Layout> {
    let size = layout.size().checked_add(header(layout.align()))?;
    Layout::from_size_align(size, layout.align()).ok()
}

fn grow(phase: Phase, size: usize, header_bytes: usize) {
    let live = alloc::grow(size);
    let index = phase as usize;
    let size = size as u64;
    COUNT[index][thread_class()].fetch_add(1, Relaxed);
    BYTES[index][thread_class()].fetch_add(size, Relaxed);
    ACTIVE_PEAK[index].fetch_max(live, Relaxed);
    let origin_live = ORIGIN_LIVE[index].fetch_add(size, Relaxed) + size;
    ORIGIN_PEAK[index].fetch_max(origin_live, Relaxed);
    HEADER_BYTES.fetch_add(header_bytes as u64, Relaxed);
}

fn release(origin: Phase, size: usize) {
    alloc::shrink(size);
    let origin = origin as usize;
    let releaser = alloc_phase::current() as usize;
    let size = size as u64;
    ORIGIN_LIVE[origin].fetch_sub(size, Relaxed);
    RELEASED_COUNT[origin][releaser].fetch_add(1, Relaxed);
    RELEASED_BYTES[origin][releaser].fetch_add(size, Relaxed);
}

pub struct PhasedAllocator;

impl PhasedAllocator {
    /// Writes the current phase into a fresh block's header and returns the
    /// caller's pointer, or null.
    ///
    /// # Safety
    /// `base` is null or a block of `outer(layout)`.
    unsafe fn finish(base: *mut u8, layout: Layout) -> *mut u8 {
        if base.is_null() {
            return base;
        }
        let phase = alloc_phase::current();
        // SAFETY: the block is at least `header` bytes long.
        unsafe { base.write(phase as u8) };
        grow(phase, layout.size(), header(layout.align()));
        // SAFETY: as above; the caller's part starts after the header and
        // keeps the alignment, which divides the header size.
        unsafe { base.add(header(layout.align())) }
    }
}

// SAFETY: every method forwards to `System` with the caller's layout grown
// by a header of `max(16, align)` bytes, which keeps the alignment, and hands
// back the address just past the header. Deallocation and reallocation step
// back by the same header to recover the system block.
unsafe impl GlobalAlloc for PhasedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some(outer) = outer(layout) else {
            return std::ptr::null_mut();
        };
        // SAFETY: `outer` has a nonzero size and the caller's alignment.
        unsafe { Self::finish(System.alloc(outer), layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let Some(outer) = outer(layout) else {
            return std::ptr::null_mut();
        };
        // SAFETY: as in `alloc`.
        unsafe { Self::finish(System.alloc_zeroed(outer), layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let header = header(layout.align());
        // SAFETY: `ptr` came from `finish`, `header` bytes past the start of
        // a block allocated with `outer(layout)`, which therefore exists.
        unsafe {
            let base = ptr.sub(header);
            let origin = Phase::from_index(base.read());
            System.dealloc(base, outer(layout).unwrap_unchecked());
            release(origin, layout.size());
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let header = header(layout.align());
        let Some(new_outer) = new_size.checked_add(header) else {
            return std::ptr::null_mut();
        };
        if Layout::from_size_align(new_outer, layout.align()).is_err() {
            return std::ptr::null_mut();
        }
        // SAFETY: as in `dealloc`; `new_outer` is a valid size for the
        // alignment. The system copies the header with the contents.
        unsafe {
            let base = ptr.sub(header);
            let new = System.realloc(base, outer(layout).unwrap_unchecked(), new_outer);
            if new.is_null() {
                return new;
            }
            let origin = Phase::from_index(new.read());
            release(origin, layout.size());
            let phase = alloc_phase::current();
            new.write(phase as u8);
            grow(phase, new_size, header);
            new.add(header)
        }
    }
}

/// Every phase counter, copied at one point. Copying allocates nothing.
pub struct PhaseSnapshot {
    count: [[u64; THREADS]; PHASES],
    bytes: [[u64; THREADS]; PHASES],
    active_peak: [u64; PHASES],
    origin_live: [u64; PHASES],
    origin_peak: [u64; PHASES],
    released_count: [[u64; PHASES]; PHASES],
    released_bytes: [[u64; PHASES]; PHASES],
    header_bytes: u64,
    total_count: u64,
    total_bytes: u64,
    total_peak: u64,
    total_live: u64,
}

fn load<const N: usize>(row: &[AtomicU64; N]) -> [u64; N] {
    std::array::from_fn(|index| row[index].load(Relaxed))
}

/// Copies every counter. Call it when no other thread allocates, so the
/// totals reconcile.
pub fn snapshot() -> PhaseSnapshot {
    let (total_count, total_bytes, total_peak) = alloc::snapshot();
    PhaseSnapshot {
        count: std::array::from_fn(|phase| load(&COUNT[phase])),
        bytes: std::array::from_fn(|phase| load(&BYTES[phase])),
        active_peak: load(&ACTIVE_PEAK),
        origin_live: load(&ORIGIN_LIVE),
        origin_peak: load(&ORIGIN_PEAK),
        released_count: std::array::from_fn(|phase| load(&RELEASED_COUNT[phase])),
        released_bytes: std::array::from_fn(|phase| load(&RELEASED_BYTES[phase])),
        header_bytes: HEADER_BYTES.load(Relaxed),
        total_count,
        total_bytes,
        total_peak,
        total_live: alloc::live(),
    }
}

impl PhaseSnapshot {
    /// Whether the phase counters account for exactly the global ones.
    fn reconciles(&self) -> bool {
        let count: u64 = self.count.iter().flatten().sum();
        let bytes: u64 = self.bytes.iter().flatten().sum();
        let live: u64 = self.origin_live.iter().sum();
        let released: u64 = self.released_bytes.iter().flatten().sum();
        count == self.total_count
            && bytes == self.total_bytes
            && live == self.total_live
            && released + live == self.total_bytes
    }
}

/// Renders `snapshot` as one JSON document, schema [`SCHEMA`]. `build` names
/// the diagnostic features; `context` adds string fields such as the
/// execution mode.
pub fn report_json(snapshot: &PhaseSnapshot, build: &[&str], context: &[(&str, &str)]) -> String {
    let mut out = String::new();
    let _ = write!(out, "{{\"schema\":\"{SCHEMA}\",\"build\":[");
    for (index, feature) in build.iter().enumerate() {
        let separator = if index == 0 { "" } else { "," };
        let _ = write!(out, "{separator}\"{feature}\"");
    }
    out.push_str("],\"context\":{");
    for (index, (key, value)) in context.iter().enumerate() {
        let separator = if index == 0 { "" } else { "," };
        let _ = write!(out, "{separator}\"{key}\":\"{value}\"");
    }
    let _ = write!(
        out,
        "}},\"totals\":{{\"alloc_count\":{},\"alloc_bytes\":{},\"alloc_peak_bytes\":{},\"live_bytes\":{}}}",
        snapshot.total_count, snapshot.total_bytes, snapshot.total_peak, snapshot.total_live,
    );
    let _ = write!(
        out,
        ",\"reconciled\":{},\"instrumentation\":{{\"header_bytes\":{},\"header_bytes_min\":{HEADER}}}",
        snapshot.reconciles(),
        snapshot.header_bytes,
    );
    out.push_str(",\"phases\":{");
    for phase in Phase::ALL {
        let index = phase as usize;
        let [count_vm, count_other] = snapshot.count[index];
        let [bytes_vm, bytes_other] = snapshot.bytes[index];
        let separator = if index == 0 { "" } else { "," };
        let _ = write!(
            out,
            "{separator}\"{}\":{{\"alloc_count\":{},\"alloc_bytes\":{},\
             \"alloc_count_vm\":{count_vm},\"alloc_count_other\":{count_other},\
             \"alloc_bytes_vm\":{bytes_vm},\"alloc_bytes_other\":{bytes_other},\
             \"active_peak_bytes\":{},\"origin_live_bytes\":{},\"origin_peak_bytes\":{},\"released_by\":{{",
            phase.name(),
            count_vm + count_other,
            bytes_vm + bytes_other,
            snapshot.active_peak[index],
            snapshot.origin_live[index],
            snapshot.origin_peak[index],
        );
        for releaser in Phase::ALL {
            let separator = if releaser as usize == 0 { "" } else { "," };
            let _ = write!(
                out,
                "{separator}\"{}\":{{\"count\":{},\"bytes\":{}}}",
                releaser.name(),
                snapshot.released_count[index][releaser as usize],
                snapshot.released_bytes[index][releaser as usize],
            );
        }
        out.push_str("}}");
    }
    out.push_str("}}");
    out
}
