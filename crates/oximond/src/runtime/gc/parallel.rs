//! Tracing on several threads in the collector's stop-the-world phases.
//!
//! A full collection, a minor collection, and a major cycle's atomic step
//! trace the gray set to the end while the mutator waits. Once that gray
//! set is large, or a table with a large array part turns up, the tracing
//! moves to scoped worker threads that share the [`Heap`]. Each worker
//! traces from its own gray stack and gives half of it away while another
//! worker is idle. Large arrays are split into ranges any worker can take.
//! Tracing ends when every worker is idle with nothing left to share.
//!
//! Marking a slot is a relaxed load and store of its epoch, so two workers
//! can both mark one object and both trace it. That repeats work but marks
//! nothing wrongly. Weak tables traced by any worker join the caller's
//! marker, which [`Runtime::gc_converge`](oximond::runtime::Runtime) deduplicates.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, PoisonError};

use super::mark::{Heap, TableTraversal};
use super::{GcReference, Marker};
use crate::runtime::value::TableHandle;

/// When tracing moves to worker threads, and how many.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParallelMarking {
    /// Threads that trace, including the collecting one. One turns
    /// parallel tracing off. Zero is the process default.
    pub(crate) threads: usize,
    /// Gray objects queued before tracing moves to worker threads. Below
    /// it, starting the threads costs more than they save.
    pub(crate) gray_threshold: usize,
    /// A table with a longer array part starts parallel tracing on its
    /// own, and its array is split into ranges of this many values.
    pub(crate) array_split: usize,
}

impl ParallelMarking {
    /// Marking threads left to the default, which `resolved_threads` looks
    /// up at the first collection so an empty script never queries the OS.
    pub(crate) fn from_env() -> Self {
        Self {
            threads: 0,
            gray_threshold: 4096,
            array_split: 4096,
        }
    }

    /// The thread count to trace with. Zero stands for `OXIMOND_GC_THREADS`,
    /// or the available parallelism up to four, read once per process.
    fn resolved_threads(self) -> usize {
        static DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        if self.threads != 0 {
            return self.threads;
        }
        *DEFAULT.get_or_init(|| {
            std::env::var("OXIMOND_GC_THREADS")
                .ok()
                .and_then(|threads| threads.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map_or(1, std::num::NonZero::get)
                        .min(4)
                })
                .max(1)
        })
    }
}

/// Traces every gray object of `marker`, on worker threads once there is
/// enough to trace.
pub(crate) fn propagate(heap: Heap<'_>, marker: &mut Marker, parallel: ParallelMarking) {
    let threads = parallel.resolved_threads();
    if threads <= 1 {
        heap.propagate_all(marker);
        return;
    }
    while let Some(reference) = marker.gray.pop() {
        if marker.gray.len() >= parallel.gray_threshold || is_large_table(heap, reference) {
            marker.gray.push(reference);
            propagate_parallel(heap, marker, threads);
            return;
        }
        heap.traverse(reference, marker, &TableTraversal::Full);
    }
}

fn is_large_table(heap: Heap<'_>, reference: GcReference) -> bool {
    let GcReference::Table(handle) = reference else {
        return false;
    };
    heap.tables()
        .get(handle.0)
        .is_some_and(|table| table.array.len() > heap.array_split())
}

/// Work any worker can take.
#[derive(Default)]
struct Pool {
    /// Gray objects given away by busy workers.
    gray: Vec<Vec<GcReference>>,
    /// Array ranges of large tables: the table and the range's start.
    ranges: Vec<(TableHandle, usize)>,
    /// Workers waiting for work.
    idle: usize,
    /// Set once every worker is idle and the pool is empty.
    done: bool,
}

impl Pool {
    /// Pieces of work an idle worker can take.
    fn queued(&self) -> usize {
        self.gray.len() + self.ranges.len().min(1)
    }
}

struct Shared {
    pool: Mutex<Pool>,
    wake: Condvar,
    /// `pool.idle`, and the batches of gray objects and ranges in the pool,
    /// readable without the lock, so a busy worker can tell cheaply whether
    /// an idle worker lacks work.
    idle: AtomicUsize,
    queued: AtomicUsize,
    workers: usize,
}

impl Shared {
    /// Whether more workers wait than the pool has work for.
    fn starving(&self) -> bool {
        self.idle.load(Relaxed) > self.queued.load(Relaxed)
    }
}

fn propagate_parallel(heap: Heap<'_>, marker: &mut Marker, threads: usize) {
    let shared = Shared {
        pool: Mutex::new(Pool::default()),
        wake: Condvar::new(),
        idle: AtomicUsize::new(0),
        queued: AtomicUsize::new(0),
        workers: threads,
    };
    let (epoch, minor) = (marker.epoch, marker.minor);
    let results = std::thread::scope(|scope| {
        let helpers = (1..threads)
            .map(|_| {
                let shared = &shared;
                scope.spawn(move || {
                    let mut local = Marker::new(epoch, minor);
                    work(heap, shared, &mut local);
                    local
                })
            })
            .collect::<Vec<_>>();
        work(heap, &shared, marker);
        helpers
            .into_iter()
            .map(|helper| {
                helper
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect::<Vec<_>>()
    });
    for local in results {
        debug_assert_eq!(local.gray.len(), 0);
        marker.work += local.work;
        marker.weak_tables.extend(local.weak_tables);
    }
}

/// One worker's loop: trace its own gray stack, share when others wait,
/// and take shared work when its own runs out.
fn work(heap: Heap<'_>, shared: &Shared, marker: &mut Marker) {
    let mut ranges = Vec::new();
    loop {
        while let Some(reference) = marker.gray.pop() {
            if let Some(scan) = heap.traverse(reference, marker, &TableTraversal::SplitArray) {
                let length = heap
                    .tables()
                    .get(scan.table.0)
                    .map_or(0, |table| table.array.len());
                ranges.extend(
                    (0..length)
                        .step_by(heap.array_split())
                        .map(|start| (scan.table, start)),
                );
            }
            if shared.starving() {
                share(shared, marker, &mut ranges);
            }
        }
        if let Some((table, start)) = ranges.pop() {
            heap.mark_array_range(table, start..start + heap.array_split(), marker);
            if shared.starving() {
                share(shared, marker, &mut ranges);
            }
            continue;
        }
        if !take(shared, marker, &mut ranges) {
            return;
        }
    }
}

/// Gives the older half of the gray stack, and half of the array ranges,
/// to idle workers.
fn share(shared: &Shared, marker: &mut Marker, ranges: &mut Vec<(TableHandle, usize)>) {
    let gray = marker.gray.len() / 2;
    let given_ranges = ranges.len() / 2;
    if gray == 0 && given_ranges == 0 {
        return;
    }
    let mut pool = shared.pool.lock().unwrap_or_else(PoisonError::into_inner);
    if gray != 0 {
        pool.gray.push(marker.gray.drain(..gray).collect());
    }
    pool.ranges.extend(ranges.drain(..given_ranges));
    shared.queued.store(pool.queued(), Relaxed);
    drop(pool);
    shared.wake.notify_all();
}

/// Waits for shared work. Returns `false` once tracing is over.
fn take(shared: &Shared, marker: &mut Marker, ranges: &mut Vec<(TableHandle, usize)>) -> bool {
    let mut pool = shared.pool.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        if let Some(gray) = pool.gray.pop() {
            marker.gray = gray;
            shared.queued.store(pool.queued(), Relaxed);
            return true;
        }
        if !pool.ranges.is_empty() {
            let count = pool.ranges.len().div_ceil(shared.workers);
            let at = pool.ranges.len() - count;
            ranges.extend(pool.ranges.drain(at..));
            shared.queued.store(pool.queued(), Relaxed);
            return true;
        }
        if pool.done {
            return false;
        }
        pool.idle += 1;
        if pool.idle == shared.workers {
            pool.done = true;
            drop(pool);
            shared.wake.notify_all();
            return false;
        }
        shared.idle.store(pool.idle, Relaxed);
        pool = shared
            .wake
            .wait(pool)
            .unwrap_or_else(PoisonError::into_inner);
        pool.idle -= 1;
        shared.idle.store(pool.idle, Relaxed);
    }
}
