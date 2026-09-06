//! Where the time goes, by Lua source line, and why the tracer does not cover
//! it.
//!
//! A [`Collector`] owns a timer thread that ticks every period and raises a
//! flag in each attached VM. The flag is the one the fast loop already polls
//! at every backedge and call (the host interrupt flag's slot), so a VM with
//! no collector pays nothing: its poll points at a flag nobody sets, and the
//! dispatch loop's only extra work is one `Option::is_some` test per
//! iteration of the general loop, which is the slow path anyway. No debug
//! hook is involved, so traces record, compile and run as they do without
//! the sampler.
//!
//! What a sample sees:
//!
//! - An interpreter sample is taken when the dispatch loop next runs after a
//!   tick. The fast loop leaves at the first backedge or call it reaches, so
//!   the sampled line is the line of the next backedge or call after the tick
//!   in the running function, and the sampled loop is the loop that backedge
//!   closes. Time spent in a callee is charged to the callee's own next
//!   backedge or call, and to the caller's function and loops as inclusive
//!   time.
//! - A native sample is the ticks that passed while a trace ran. They are
//!   charged to the header the trace was entered at. The code ranges of the
//!   traces are named in a perf map (see `trace::perf_map`) for `perf report`.
//! - A tick still pending when the host calls into the VM was raised while
//!   the thread ran host code (or the VM's own call entry), not Lua: it is
//!   counted as a host sample, not charged to the Lua function the host
//!   enters. A host that converts values through small Lua functions (Luanti
//!   calls `read_vector` from C++ for every position) would otherwise see
//!   its C++ time charged to them.
//! - A tick that is observed more than a few periods late (the VM was idle,
//!   or sat in C code that did not return) is counted as stale and not
//!   attributed.
//!
//! For every sampled loop, the sample records the trace state of the loop
//! header ([`CoverageState`] and the last abort), so the report can say not
//! only that a loop is hot but why no trace covers it.
//!
//! The environment variable `OXIMOND_ATTRIBUTION=<path>` attaches a global
//! collector to every VM of the process and writes the report to `<path>`
//! (`%p` stands for the process id) about once a second, when a VM is dropped
//! and when [`flush`] is called. `OXIMOND_ATTRIBUTION_HZ` sets the tick rate
//! (default 1000). The name `OXIMOND_INTERP_PROFILE` is taken by the opcode
//! counters.

use std::cell::OnceCell;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::bytecode::Chunk;
use crate::vm::Vm;
use crate::vm::jit::trace::{
    Coverage, CoverageState, LoopSpan, RejectCause, TraceExitKind, TraceGuardKind, TraceStatistics,
    loops_of,
};
use crate::vm::stack::CallStack;

/// A tick seen later than this many periods after it was raised is stale.
const STALE_PERIODS: u32 = 4;
/// How often the timer thread writes the report file.
const WRITE_INTERVAL: Duration = Duration::from_secs(1);
/// Samples between refreshes of a VM's exit counters in the collector.
const EXIT_REFRESH_SAMPLES: u32 = 256;

/// The trace state of a header at the time of the last sample that saw it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// [`CoverageState::name`].
    pub state: &'static str,
    /// The last abort as `Reason at source:line`, or empty.
    pub reason: String,
    pub attempts: u32,
    pub exits: u64,
    pub entries: u64,
    pub iterations: u64,
    /// The most frequent interpreter exits of the header's trace as
    /// `Kind/Guard=count` separated by commas (at most [`TOP_EXITS`]).
    pub exit_kinds: String,
}

/// Exit kinds a snapshot names.
const TOP_EXITS: usize = 4;

/// Samples charged to one function (by source and definition line).
#[derive(Clone, Debug, Default)]
pub struct FunctionStat {
    /// Samples taken with the function on top.
    pub own: u64,
    /// Samples taken with the function anywhere on the stack.
    pub inclusive: u64,
    /// Ticks the function's traces ran natively.
    pub native: u64,
    /// The state of the function's entry site, when it had one.
    pub entry: Option<Snapshot>,
    /// Loops of the function that were sampled.
    pub loops: u32,
    /// For a function sampled outside any loop of its own: the innermost loop
    /// of the nearest caller that was inside one, as `(source, definition
    /// line, header pc)`, a key of [`ThreadData::loops`].
    pub via: Option<(String, u32, usize)>,
}

/// Samples charged to one loop header.
#[derive(Clone, Debug, Default)]
pub struct LoopStat {
    pub header_line: u32,
    /// Samples whose innermost loop this was.
    pub innermost: u64,
    /// Samples taken anywhere inside it, inner loops included.
    pub inclusive: u64,
    /// Samples whose nearest loop on the stack this was: its own code and
    /// the calls it makes that run in no loop of their own.
    pub nearest: u64,
    /// Ticks a trace entered at this header ran natively.
    pub native: u64,
    pub snapshot: Snapshot,
}

/// Everything one thread's VMs sampled.
#[derive(Clone, Debug, Default)]
pub struct ThreadData {
    pub interpreter: u64,
    pub native: u64,
    pub stale: u64,
    /// Ticks seen when the host called into the VM: time in host code.
    pub host: u64,
    /// `(source, line)` to samples with that line on top.
    pub lines: BTreeMap<(String, u32), u64>,
    /// `(source, line)` to the definition line and header pc of the innermost
    /// loop the line was last sampled in.
    pub line_loops: BTreeMap<(String, u32), (u32, usize)>,
    /// `(source, definition line)`.
    pub functions: BTreeMap<(String, u32), FunctionStat>,
    /// `(source, definition line, header pc)`.
    pub loops: BTreeMap<(String, u32, usize), LoopStat>,
}

/// Interpreter exits that stayed in the interpreter, by kind and guard.
pub type ExitCounts = Vec<(&'static str, &'static str, u64)>;

#[derive(Default)]
struct Data {
    threads: BTreeMap<String, ThreadData>,
    exits: HashMap<u64, (String, ExitCounts)>,
}

/// The flag a collector raises in one VM.
struct Shared {
    flag: Arc<AtomicBool>,
    /// Collector nanoseconds at which the flag was raised.
    raised_at: AtomicU64,
}

/// Gathers samples from the VMs attached to it. See the module docs.
pub struct Collector {
    period: Duration,
    path: Option<PathBuf>,
    epoch: Instant,
    /// Ticks so far, for the time a native run spent.
    ticks: AtomicU64,
    data: Mutex<Data>,
    handles: Mutex<Vec<Weak<Shared>>>,
    next_id: AtomicU64,
    dirty: AtomicBool,
    started: Once,
}

/// One frame of a sampled stack.
struct FrameSample {
    source: String,
    defined: u32,
    /// `(header pc, header line, state)` of the loops around the frame's pc,
    /// innermost first.
    loops: Vec<(usize, u32, Snapshot)>,
}

/// The loops of the chunks a VM has sampled, found once per chunk.
#[derive(Default)]
pub(crate) struct LoopCache {
    chunks: HashMap<usize, (Weak<Chunk>, Vec<LoopSpan>)>,
}

impl LoopCache {
    /// The loops of `chunk` around `pc`, innermost first.
    fn containing(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Vec<LoopSpan> {
        if self.chunks.len() > 4096 {
            self.chunks.clear();
        }
        let entry = self.chunks.entry(Arc::as_ptr(chunk) as usize).or_default();
        if entry.0.as_ptr() != Arc::as_ptr(chunk) {
            *entry = (Arc::downgrade(chunk), loops_of(chunk));
        }
        let mut found: Vec<LoopSpan> = entry
            .1
            .iter()
            .copied()
            .filter(|span| span.contains(pc))
            .collect();
        found.sort_by_key(|span| std::cmp::Reverse(span.header));
        found
    }
}

/// A VM's attachment to a collector.
pub(crate) struct SamplerHandle {
    spans: LoopCache,
    collector: Arc<Collector>,
    shared: Arc<Shared>,
    id: u64,
    label: OnceCell<String>,
    samples: u32,
}

/// What `begin_native_sample` remembers for the end of a native run.
pub(crate) struct NativeProbe {
    chunk: Arc<Chunk>,
    pc: usize,
    ticks: u64,
}

impl Collector {
    /// A collector that ticks every `period`. With a `path` it keeps a report
    /// file there.
    #[must_use]
    pub fn new(period: Duration, path: Option<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            period,
            path,
            epoch: Instant::now(),
            ticks: AtomicU64::new(0),
            data: Mutex::new(Data::default()),
            handles: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            dirty: AtomicBool::new(false),
            started: Once::new(),
        })
    }

    fn now(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn register(self: &Arc<Self>) -> SamplerHandle {
        let shared = Arc::new(Shared {
            flag: Arc::new(AtomicBool::new(false)),
            raised_at: AtomicU64::new(0),
        });
        self.handles
            .lock()
            .expect("handles")
            .push(Arc::downgrade(&shared));
        self.started.call_once(|| {
            let weak = Arc::downgrade(self);
            let period = self.period;
            let _ = std::thread::Builder::new()
                .name("oximond-sampler".into())
                .spawn(move || timer(&weak, period));
        });
        SamplerHandle {
            collector: Arc::clone(self),
            shared,
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            label: OnceCell::new(),
            samples: 0,
            spans: LoopCache::default(),
        }
    }

    fn tick(&self) {
        self.ticks.fetch_add(1, Ordering::Relaxed);
        let now = self.now();
        let mut handles = self.handles.lock().expect("handles");
        handles.retain(|weak| {
            let Some(shared) = weak.upgrade() else {
                return false;
            };
            if !shared.flag.load(Ordering::Relaxed) {
                shared.raised_at.store(now, Ordering::Relaxed);
                shared.flag.store(true, Ordering::Release);
            }
            true
        });
    }

    /// A copy of what was sampled, by thread name.
    ///
    /// # Panics
    ///
    /// Panics when the sampler's lock was poisoned by a panicking holder.
    #[must_use]
    pub fn snapshot(&self) -> BTreeMap<String, ThreadData> {
        self.data.lock().expect("data").threads.clone()
    }

    /// Samples on `line` of the source whose name ends with `source`, summed
    /// over all threads.
    ///
    /// # Panics
    ///
    /// Panics when the sampler's lock was poisoned by a panicking holder.
    #[must_use]
    pub fn line_samples(&self, source: &str, line: u32) -> u64 {
        self.data
            .lock()
            .expect("data")
            .threads
            .values()
            .flat_map(|thread| thread.lines.iter())
            .filter(|((name, at), _)| *at == line && name.ends_with(source))
            .map(|(_, count)| *count)
            .sum()
    }

    /// Writes the report file now, if the collector has a path.
    pub fn write(&self) {
        let Some(path) = &self.path else { return };
        self.dirty.store(false, Ordering::Relaxed);
        let text = self.render();
        let temporary = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&temporary, text).is_ok() {
            let _ = std::fs::rename(&temporary, path);
        }
    }

    /// The report in the line format `luanti.py bench --attribution` reads:
    /// tab separated records, one per line, led by their kind.
    ///
    /// # Panics
    ///
    /// Panics when the sampler's lock was poisoned by a panicking holder.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let data = self.data.lock().expect("data");
        let mut out = String::new();
        let hz = 1.0 / self.period.as_secs_f64();
        let _ = writeln!(out, "oximond-attribution\t1\t{hz:.0}");
        for (label, thread) in &data.threads {
            let label = clean(label);
            let _ = writeln!(
                out,
                "thread\t{label}\t{}\t{}\t{}\t{}",
                thread.interpreter, thread.native, thread.stale, thread.host
            );
            for ((source, defined), stat) in &thread.functions {
                let source = clean(source);
                let via = stat
                    .via
                    .as_ref()
                    .map_or_else(String::new, |(from, defined, pc)| {
                        format!("{}\t{defined}\t{pc}", clean(from))
                    });
                let (state, reason) = stat
                    .entry
                    .as_ref()
                    .map_or(("-", ""), |entry| (entry.state, entry.reason.as_str()));
                let _ = writeln!(
                    out,
                    "func\t{label}\t{source}\t{defined}\t{}\t{}\t{}\t{state}\t{}\t{}\t{via}",
                    stat.own,
                    stat.inclusive,
                    stat.native,
                    clean(reason),
                    stat.loops
                );
            }
            for ((source, line), count) in &thread.lines {
                let header = thread
                    .line_loops
                    .get(&(source.clone(), *line))
                    .map_or_else(String::new, |(defined, pc)| format!("{defined}:{pc}"));
                let _ = writeln!(
                    out,
                    "line\t{label}\t{}\t{line}\t{count}\t{header}",
                    clean(source)
                );
            }
            for ((source, defined, pc), stat) in &thread.loops {
                let snap = &stat.snapshot;
                let _ = writeln!(
                    out,
                    "loop\t{label}\t{}\t{defined}\t{pc}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    clean(source),
                    stat.header_line,
                    stat.innermost,
                    stat.inclusive,
                    stat.native,
                    snap.state,
                    clean(&snap.reason),
                    snap.attempts,
                    snap.exits,
                    snap.entries,
                    snap.iterations,
                    clean(&snap.exit_kinds),
                    stat.nearest
                );
            }
        }
        let mut totals: BTreeMap<(&str, &str), u64> = BTreeMap::new();
        for (_, counts) in data.exits.values() {
            for (kind, guard, count) in counts {
                *totals.entry((kind, guard)).or_default() += count;
            }
        }
        for ((kind, guard), count) in totals {
            if count > 0 {
                let _ = writeln!(out, "exits\t{kind}\t{guard}\t{count}");
            }
        }
        out
    }
}

fn clean(text: &str) -> String {
    text.replace(['\t', '\n', '\r'], " ")
}

fn timer(weak: &Weak<Collector>, period: Duration) {
    let mut last_write = Instant::now();
    loop {
        std::thread::sleep(period);
        let Some(collector) = weak.upgrade() else {
            return;
        };
        collector.tick();
        if collector.path.is_some()
            && last_write.elapsed() >= WRITE_INTERVAL
            && collector.dirty.load(Ordering::Relaxed)
        {
            collector.write();
            last_write = Instant::now();
        }
    }
}

/// The collector `OXIMOND_ATTRIBUTION` asks for, made on first use.
fn global() -> Option<&'static Arc<Collector>> {
    static GLOBAL: OnceLock<Option<Arc<Collector>>> = OnceLock::new();
    GLOBAL
        .get_or_init(|| {
            let path = std::env::var("OXIMOND_ATTRIBUTION").ok()?;
            if path.is_empty() {
                return None;
            }
            let path = path.replace("%p", &std::process::id().to_string());
            let hz = std::env::var("OXIMOND_ATTRIBUTION_HZ")
                .ok()
                .and_then(|text| text.parse::<f64>().ok())
                .filter(|hz| *hz > 0.0)
                .unwrap_or(1000.0);
            Some(Collector::new(
                Duration::from_secs_f64(1.0 / hz),
                Some(PathBuf::from(path)),
            ))
        })
        .as_ref()
}

/// Writes the global collector's report now. Hosts call this before they
/// exit; the file is otherwise at most a second old.
pub fn flush() {
    if let Some(collector) = global() {
        collector.write();
    }
}

impl SamplerHandle {
    /// Hands the VM's exit counters to the collector and writes the report.
    pub(crate) fn finish(self, statistics: &TraceStatistics) {
        self.collector
            .data
            .lock()
            .expect("data")
            .exits
            .insert(self.id, (self.label().to_owned(), exits_of(statistics)));
        self.collector.dirty.store(true, Ordering::Relaxed);
        self.collector.write();
    }

    /// The name of the thread that first sampled: the kernel's name for it
    /// (what a C host such as Luanti sets), else the thread id.
    fn label(&self) -> &str {
        self.label.get_or_init(|| {
            std::fs::read_to_string("/proc/thread-self/comm")
                .ok()
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| format!("{:?}", std::thread::current().id()))
        })
    }
}

fn exits_of(statistics: &TraceStatistics) -> ExitCounts {
    let mut counts = Vec::new();
    for kind in TraceExitKind::ALL {
        let name = exit_kind_name(kind);
        counts.push((name, "none", statistics.interpreter_exits_of(kind, None)));
        for guard in TraceGuardKind::ALL {
            counts.push((
                name,
                guard_name(guard),
                statistics.interpreter_exits_of(kind, Some(guard)),
            ));
        }
    }
    counts
}

const fn exit_kind_name(kind: TraceExitKind) -> &'static str {
    match kind {
        TraceExitKind::EntryMismatch => "EntryMismatch",
        TraceExitKind::GuardFail => "GuardFail",
        TraceExitKind::UnsupportedBefore => "UnsupportedBefore",
        TraceExitKind::CommittedEffect => "CommittedEffect",
        TraceExitKind::LoopComplete => "LoopComplete",
        TraceExitKind::Poll => "Poll",
        TraceExitKind::Invalidated => "Invalidated",
    }
}

const fn guard_name(guard: TraceGuardKind) -> &'static str {
    match guard {
        TraceGuardKind::Type => "Type",
        TraceGuardKind::Branch => "Branch",
        TraceGuardKind::ArrayBounds => "ArrayBounds",
        TraceGuardKind::TableShape => "TableShape",
        TraceGuardKind::Metatable => "Metatable",
        TraceGuardKind::Arithmetic => "Arithmetic",
        TraceGuardKind::Callee => "Callee",
        TraceGuardKind::Other => "Other",
    }
}

fn source_of(chunk: &Chunk) -> String {
    String::from_utf8_lossy(chunk.debug.source_name().as_bytes()).into_owned()
}

/// The reason column: the last abort as `Reason at source:line`, read from
/// the chunk the recording met it in (a callee's when it had inlined one),
/// and for a rejected header the path that gave it up when that was not the
/// abort policy (`Retired; last ...`).
fn reason_of(coverage: &Coverage) -> String {
    let abort = coverage.abort.as_ref().map(|abort| match &abort.at {
        Some((chunk, pc)) => format!(
            "{:?} at {}:{}",
            abort.reason,
            source_of(chunk),
            chunk.debug.line_at(*pc).unwrap_or(0)
        ),
        None => format!("{:?}", abort.reason),
    });
    match (coverage.rejected_by, abort) {
        (Some(cause), Some(abort)) if cause != RejectCause::Aborts => {
            format!("{cause:?}; last {abort}")
        }
        (Some(cause), None) => format!("{cause:?}"),
        (_, Some(abort)) => abort,
        (None, None) => String::new(),
    }
}

fn snapshot_of(coverage: &Coverage) -> Snapshot {
    let reason = reason_of(coverage);
    Snapshot {
        state: coverage.state.name(),
        reason,
        attempts: coverage.attempts,
        exits: coverage.exits,
        entries: coverage.entries,
        iterations: coverage.iterations,
        exit_kinds: exit_kinds_of(coverage),
    }
}

/// The header's most frequent interpreter exits, as `Kind/Guard=count`.
fn exit_kinds_of(coverage: &Coverage) -> String {
    let mut all = Vec::new();
    for kind in TraceExitKind::ALL {
        let row = &coverage.interpreter_exits[kind.index()];
        for (column, &count) in row.iter().enumerate() {
            if count > 0 {
                let guard = TraceGuardKind::ALL
                    .get(column)
                    .map_or("none", |&g| guard_name(g));
                all.push((count, format!("{}/{guard}", exit_kind_name(kind))));
            }
        }
    }
    all.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    all.iter()
        .take(TOP_EXITS)
        .map(|(count, name)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

impl Vm {
    /// Attaches `collector` to this VM, or detaches the current one. Only the
    /// VM's own thread samples it; tick rate and output belong to the
    /// collector.
    pub fn set_attribution(&mut self, collector: Option<Arc<Collector>>) {
        if let Some(old) = self.sampler.take() {
            old.finish(&self.jit.statistics.trace);
        }
        self.interrupt_poll = super::InterruptPoll::NEVER;
        if let Some(collector) = collector {
            let handle = collector.register();
            self.interrupt_poll = super::InterruptPoll::of(&handle.shared.flag);
            self.sampler = Some(handle);
        } else if let Some(flag) = &self.interrupt_flag {
            self.interrupt_poll = super::InterruptPoll::of(flag);
        }
    }

    /// Attaches the collector `OXIMOND_ATTRIBUTION` asks for, if it does.
    pub(super) fn attach_global_attribution(&mut self) {
        if let Some(collector) = global() {
            self.set_attribution(Some(Arc::clone(collector)));
        }
    }

    /// Takes the tick pending when the host calls into the VM, if any, as a
    /// host sample (see the module notes).
    #[cold]
    #[inline(never)]
    pub(super) fn sample_host_entry(&mut self) {
        let Some(handle) = self.sampler.as_ref() else {
            return;
        };
        if !handle.shared.flag.load(Ordering::Relaxed)
            || !handle.shared.flag.swap(false, Ordering::Acquire)
        {
            return;
        }
        let collector = Arc::clone(&handle.collector);
        let raised = handle.shared.raised_at.load(Ordering::Relaxed);
        let late = collector.now().saturating_sub(raised);
        let label = handle.label().to_owned();
        let stale = late
            > u64::try_from(collector.period.as_nanos()).unwrap_or(u64::MAX)
                * u64::from(STALE_PERIODS);
        let mut data = collector.data.lock().expect("data");
        let thread = data.threads.entry(label).or_default();
        if stale {
            thread.stale += 1;
        } else {
            thread.host += 1;
        }
    }

    /// Takes the sample the collector's tick asked for, if one is pending.
    #[cold]
    #[inline(never)]
    pub(super) fn sample_interpreter(&mut self, frames: &CallStack) {
        let Some(handle) = self.sampler.as_ref() else {
            return;
        };
        if !handle.shared.flag.load(Ordering::Relaxed)
            || !handle.shared.flag.swap(false, Ordering::Acquire)
        {
            return;
        }
        let collector = Arc::clone(&handle.collector);
        let raised = handle.shared.raised_at.load(Ordering::Relaxed);
        let late = collector.now().saturating_sub(raised);
        let label = handle.label().to_owned();
        if frames.is_empty() {
            return;
        }
        let stale = late
            > u64::try_from(collector.period.as_nanos()).unwrap_or(u64::MAX)
                * u64::from(STALE_PERIODS);
        if stale {
            let mut data = collector.data.lock().expect("data");
            data.threads.entry(label).or_default().stale += 1;
            return;
        }
        let top = frames.len() - 1;
        let chunk = Arc::clone(&frames[top].chunk.as_arc());
        let pc = frames[top].pc;
        let source = source_of(&chunk);
        let defined = chunk.debug.definition_line();
        let line = chunk.debug.line_at(pc).unwrap_or(0);
        let entry = snapshot_of(
            &self
                .jit
                .trace
                .coverage(&chunk, crate::vm::jit::trace::FUNCTION_ENTRY_PC),
        );
        // Every frame, top first: its function and the loops around its pc,
        // innermost first, with their trace state.
        let mut spans = self
            .sampler
            .as_mut()
            .map(|handle| std::mem::take(&mut handle.spans))
            .unwrap_or_default();
        let mut stack: Vec<FrameSample> = Vec::new();
        for index in (0..frames.len()).rev() {
            let frame_chunk = Arc::clone(&frames[index].chunk.as_arc());
            let frame_pc = frames[index].pc;
            let loops = spans
                .containing(&frame_chunk, frame_pc)
                .into_iter()
                .map(|span| {
                    let coverage = self.jit.trace.coverage(&frame_chunk, span.header);
                    (
                        span.header,
                        frame_chunk.debug.line_at(span.header).unwrap_or(0),
                        snapshot_of(&coverage),
                    )
                })
                .collect();
            stack.push(FrameSample {
                source: source_of(&frame_chunk),
                defined: frame_chunk.debug.definition_line(),
                loops,
            });
        }
        let refresh = {
            let handle = self.sampler.as_mut().expect("sampler");
            handle.spans = spans;
            handle.samples += 1;
            handle.samples.is_multiple_of(EXIT_REFRESH_SAMPLES)
        };
        let exits = refresh.then(|| exits_of(&self.jit.statistics.trace));
        let id = self.sampler.as_ref().map_or(0, |handle| handle.id);

        let mut data = collector.data.lock().expect("data");
        let thread = data.threads.entry(label.clone()).or_default();
        thread.interpreter += 1;
        *thread.lines.entry((source.clone(), line)).or_default() += 1;
        if let Some((header, _, _)) = stack[0].loops.first() {
            thread
                .line_loops
                .insert((source.clone(), line), (defined, *header));
        }
        // The first frame below the top that runs inside a loop: what the
        // top function was called from, when it has no loop of its own.
        let via = if stack[0].loops.is_empty() {
            stack[1..].iter().find_map(|frame| {
                frame
                    .loops
                    .first()
                    .map(|(header, _, _)| (frame.source.clone(), frame.defined, *header))
            })
        } else {
            None
        };
        // The frame, top first, whose code the nearest loop on the stack is.
        let nearest = stack.iter().position(|frame| !frame.loops.is_empty());
        let mut seen: Vec<(&str, u32)> = Vec::new();
        for (depth, frame) in stack.iter().enumerate() {
            let key = (frame.source.as_str(), frame.defined);
            if !seen.contains(&key) {
                seen.push(key);
                thread
                    .functions
                    .entry((frame.source.clone(), frame.defined))
                    .or_default()
                    .inclusive += 1;
            }
            let mut new_loops = 0;
            for (index, (header, header_line, snapshot)) in frame.loops.iter().enumerate() {
                let stat = thread
                    .loops
                    .entry((frame.source.clone(), frame.defined, *header))
                    .or_insert_with(|| {
                        new_loops += 1;
                        LoopStat::default()
                    });
                stat.header_line = *header_line;
                stat.inclusive += 1;
                if depth == 0 && index == 0 {
                    stat.innermost += 1;
                }
                if nearest == Some(depth) && index == 0 {
                    stat.nearest += 1;
                }
                stat.snapshot = snapshot.clone();
            }
            if new_loops > 0 {
                thread
                    .functions
                    .entry((frame.source.clone(), frame.defined))
                    .or_default()
                    .loops += new_loops;
            }
        }
        let function = thread.functions.entry((source, defined)).or_default();
        function.own += 1;
        if entry.state != CoverageState::NeverHot.name() || function.entry.is_none() {
            function.entry = Some(entry);
        }
        if via.is_some() {
            function.via = via;
        }
        if let Some(exits) = exits {
            data.exits.insert(id, (label, exits));
        }
        drop(data);
        collector.dirty.store(true, Ordering::Relaxed);
    }

    /// Notes the header a native run starts at and the tick count.
    pub(super) fn begin_native_sample(
        &self,
        frames: &CallStack,
        frame_index: usize,
    ) -> Option<NativeProbe> {
        let handle = self.sampler.as_ref()?;
        let frame = &frames[frame_index];
        Some(NativeProbe {
            chunk: Arc::clone(&frame.chunk.as_arc()),
            pc: frame.pc,
            ticks: handle.collector.ticks.load(Ordering::Relaxed),
        })
    }

    /// Charges the ticks that passed since `probe` to the header it names.
    pub(super) fn end_native_sample(&mut self, probe: NativeProbe) {
        let Some(handle) = self.sampler.as_ref() else {
            return;
        };
        let collector = Arc::clone(&handle.collector);
        let ticks = collector
            .ticks
            .load(Ordering::Relaxed)
            .saturating_sub(probe.ticks);
        if ticks == 0 {
            return;
        }
        // The flag raised during the run was spent natively.
        handle.shared.flag.store(false, Ordering::Release);
        let label = handle.label().to_owned();
        let chunk = probe.chunk;
        let mut coverage = self.jit.trace.coverage(&chunk, probe.pc);
        if coverage.state == CoverageState::NeverHot {
            coverage = self
                .jit
                .trace
                .coverage(&chunk, crate::vm::jit::trace::FUNCTION_ENTRY_PC);
        }
        let snapshot = snapshot_of(&coverage);
        let source = source_of(&chunk);
        let defined = chunk.debug.definition_line();
        let header_line = chunk.debug.line_at(probe.pc).unwrap_or(0);
        let mut data = collector.data.lock().expect("data");
        let thread = data.threads.entry(label).or_default();
        thread.native += ticks;
        let function = thread
            .functions
            .entry((source.clone(), defined))
            .or_default();
        function.native += ticks;
        let stat = thread.loops.entry((source, defined, probe.pc)).or_default();
        stat.header_line = header_line;
        stat.native += ticks;
        stat.snapshot = snapshot;
        drop(data);
        collector.dirty.store(true, Ordering::Relaxed);
    }
}
