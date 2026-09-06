//! Bounded compilation scheduling for callers that own the executable cache.

use crate::runtime::alloc_phase::{self, Phase};
use core::fmt;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(test)]
use std::sync::{Condvar, Mutex};

use crate::jit_ir::trace::{TraceIr, TreeMember};

use crate::jit::{
    CompiledSide, CompiledTrace, OptimizationTier, SideContext, TraceCompileError,
    TraceCompileOptions, compile_side_with_plan, compile_trace_with_plan,
};

/// Bytes of trace IR the broker accepts while requests are outstanding.
/// Admission counts the IR a request keeps alive, so a worker queue cannot
/// pin unbounded duplicate IR while the caller's code cache is full.
pub const DEFAULT_PENDING_TRACE_BYTES: usize = 8 * 1024 * 1024;

/// Opaque identity shared by duplicate outstanding submissions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompilationId(u64);

/// What a finished request made: a whole tree, or a side compiled apart.
pub enum CompileOutput {
    Tree(Box<CompiledTrace>),
    Side(Box<CompiledSide>),
}

/// A request to compile one side of an installed tree apart from it (see
/// [`compile_side`]).
#[derive(Clone, Debug)]
pub struct SideJob {
    /// The members of the tree planned with `apart` and `detached`, when the
    /// caller planned them already.
    pub planned: Option<Arc<Vec<TreeMember>>>,
    /// The sides of the tree that are compiled apart, and those of them that
    /// leave at the loop header where they would link back to the loop.
    pub apart: Vec<usize>,
    pub detached: Vec<usize>,
    /// The side to compile, as an index into the tree's sides.
    pub side: usize,
    /// Whether the side is one of `detached`.
    pub exit_style: bool,
    /// What the installed tree tells the compile about the exit it calls the
    /// side from.
    pub context: SideContext,
}

/// One compiled trace moved back to the caller by [`JitBroker::poll_trace`].
pub struct CompletedTrace {
    id: CompilationId,
    tier: OptimizationTier,
    epoch: u64,
    result: Result<CompileOutput, TraceCompileError>,
}

impl CompletedTrace {
    #[must_use]
    pub const fn id(&self) -> CompilationId {
        self.id
    }

    #[must_use]
    pub const fn tier(&self) -> OptimizationTier {
        self.tier
    }

    /// The caller's flush epoch the request was submitted under. A result
    /// from an older epoch must not be installed.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// # Errors
    /// Returns the compilation error produced for this request.
    ///
    /// # Panics
    /// Panics if the request was for a side compiled apart.
    pub fn into_result(self) -> Result<CompiledTrace, TraceCompileError> {
        self.result.map(|output| match output {
            CompileOutput::Tree(tree) => *tree,
            CompileOutput::Side(_) => panic!("the request was for a side compiled apart"),
        })
    }

    /// What the request made, whatever it was for.
    ///
    /// # Errors
    /// Returns the compilation error produced for this request.
    pub fn into_output(self) -> Result<CompileOutput, TraceCompileError> {
        self.result
    }
}

/// Why a broker did not accept a compilation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitError {
    /// The trace IR the broker already holds plus this request would exceed
    /// the byte budget.
    BytesExceeded,
    /// The configured number of outstanding requests is already reached.
    QueueFull,
    /// The background worker could not be started or has stopped.
    WorkerUnavailable,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BytesExceeded => formatter.write_str("JIT trace byte budget is exhausted"),
            Self::QueueFull => formatter.write_str("JIT compilation queue is full"),
            Self::WorkerUnavailable => formatter.write_str("JIT compilation worker is unavailable"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// A bounded compilation broker with one lazy background worker.
///
/// `submit_trace` never compiles or waits. Duplicate trace-and-tier requests
/// share an ID until their result is polled. `poll_trace` is the only point
/// where compiled code crosses back to the caller, so executable-cache
/// installation remains on the caller thread.
pub struct JitBroker {
    capacity: usize,
    next_id: u64,
    pending: Vec<PendingRequest>,
    adapter: Adapter,
    /// IR bytes held by outstanding trace requests.
    pending_trace_bytes: usize,
    trace_byte_budget: usize,
    /// What trace code compiled from now on counts (see
    /// [`TraceCompileOptions`]).
    trace_options: TraceCompileOptions,
    /// Compilation ids whose compile panics on purpose, for tests of the
    /// panic guard (see [`Self::panic_on_compiles`]).
    injected_panics: Vec<u64>,
    #[cfg(test)]
    test_gate: Option<Arc<BackgroundTestGate>>,
}

impl JitBroker {
    /// Create a broker that starts its sole worker on the first submission.
    ///
    /// # Panics
    /// Panics if `queue_capacity` is zero.
    #[must_use]
    pub fn new(queue_capacity: usize) -> Self {
        assert!(queue_capacity > 0, "JIT queue capacity must be nonzero");
        Self {
            capacity: queue_capacity,
            next_id: 0,
            pending: Vec::new(),
            pending_trace_bytes: 0,
            trace_byte_budget: DEFAULT_PENDING_TRACE_BYTES,
            trace_options: TraceCompileOptions::default(),
            injected_panics: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            adapter: Adapter::Background(None),
            #[cfg(target_arch = "wasm32")]
            adapter: Adapter::Synchronous {
                queue: VecDeque::new(),
                delay: 0,
                waited: 0,
            },
            #[cfg(test)]
            test_gate: None,
        }
    }

    /// Create a deterministic adapter that compiles one request in each poll.
    ///
    /// Submission remains nonblocking. Compilation runs on the thread calling
    /// `poll`, which makes ordering deterministic in tests.
    ///
    /// # Panics
    /// Panics if `queue_capacity` is zero.
    #[must_use]
    pub fn synchronous(queue_capacity: usize) -> Self {
        assert!(queue_capacity > 0, "JIT queue capacity must be nonzero");
        Self {
            capacity: queue_capacity,
            next_id: 0,
            pending: Vec::new(),
            pending_trace_bytes: 0,
            trace_byte_budget: DEFAULT_PENDING_TRACE_BYTES,
            trace_options: TraceCompileOptions::default(),
            injected_panics: Vec::new(),
            adapter: Adapter::Synchronous {
                queue: VecDeque::new(),
                delay: 0,
                waited: 0,
            },
            #[cfg(test)]
            test_gate: None,
        }
    }

    /// Like [`Self::synchronous`], but each request completes only after
    /// `delay_polls` empty polls. This reproduces the install points a
    /// background worker can produce, deterministically.
    ///
    /// # Panics
    /// Panics if `queue_capacity` is zero.
    #[must_use]
    pub fn synchronous_with_delay(queue_capacity: usize, delay_polls: u32) -> Self {
        let mut broker = Self::synchronous(queue_capacity);
        if let Adapter::Synchronous { delay, .. } = &mut broker.adapter {
            *delay = delay_polls;
        }
        broker
    }

    /// Queue a recorded trace without waiting for compilation.
    ///
    /// `epoch` is the caller's flush epoch. It is part of the request's
    /// identity, so a request made after a flush never shares an ID with a
    /// stale one, and it comes back with the result for the caller to check.
    /// An identical trace, tier and epoch that is still outstanding returns
    /// its existing ID and consumes no capacity. A canceled request never
    /// matches.
    ///
    /// # Errors
    /// Returns [`SubmitError::BytesExceeded`] when the IR bytes held by
    /// outstanding trace requests would pass the budget,
    /// [`SubmitError::QueueFull`] when the outstanding-work bound is reached,
    /// or [`SubmitError::WorkerUnavailable`] if the worker cannot run.
    pub fn submit_trace(
        &mut self,
        ir: Arc<TraceIr>,
        tier: OptimizationTier,
        epoch: u64,
    ) -> Result<CompilationId, SubmitError> {
        self.submit_trace_planned(ir, None, tier, epoch)
    }

    /// [`Self::submit_trace`] for a tree whose members `planned` holds (see
    /// `CompileRequest::planned`). A request folded into one with the same
    /// content keeps the planned members it has.
    ///
    /// # Errors
    /// As [`Self::submit_trace`].
    pub fn submit_trace_planned(
        &mut self,
        ir: Arc<TraceIr>,
        planned: Option<Arc<Vec<TreeMember>>>,
        tier: OptimizationTier,
        epoch: u64,
    ) -> Result<CompilationId, SubmitError> {
        if let Some(request) = self.pending.iter().find(|request| {
            request.tier == tier
                && !request.canceled
                && request.payload.side.is_none()
                && request.payload.epoch == epoch
                && (Arc::ptr_eq(&request.payload.ir, &ir) || request.payload.ir.same_content(&ir))
        }) {
            return Ok(request.id);
        }
        let bytes = ir.estimated_bytes();
        if self.pending_trace_bytes.saturating_add(bytes) > self.trace_byte_budget {
            return Err(SubmitError::BytesExceeded);
        }
        let options = self.trace_options;
        let id = self.enqueue(
            tier,
            Payload {
                ir,
                epoch,
                bytes,
                options,
                side: None,
                planned,
            },
        )?;
        self.pending_trace_bytes += bytes;
        Ok(id)
    }

    /// Queue the compile of one side of an installed tree, apart from the
    /// tree. It is never folded into another request: the side belongs to
    /// the tree whose exit asked.
    ///
    /// # Errors
    /// As [`Self::submit_trace`].
    pub fn submit_side(
        &mut self,
        ir: Arc<TraceIr>,
        tier: OptimizationTier,
        epoch: u64,
        job: SideJob,
    ) -> Result<CompilationId, SubmitError> {
        let bytes = ir.estimated_bytes();
        if self.pending_trace_bytes.saturating_add(bytes) > self.trace_byte_budget {
            return Err(SubmitError::BytesExceeded);
        }
        let options = self.trace_options;
        let id = self.enqueue(
            tier,
            Payload {
                ir,
                epoch,
                bytes,
                options,
                side: Some(job),
                planned: None,
            },
        )?;
        self.pending_trace_bytes += bytes;
        Ok(id)
    }

    /// Sets what trace code submitted from now on counts. Requests already
    /// queued keep the options they had.
    pub fn set_trace_options(&mut self, options: TraceCompileOptions) {
        self.trace_options = options;
    }

    /// Makes the compiles of the requests this broker queues as its
    /// `submissions`-th ones (counting from 0, a request folded into an
    /// outstanding one not counted) panic. Tests use it to check that a
    /// compile that panics rejects its trace only: the compile thread and
    /// the caller keep running, and later requests compile.
    #[doc(hidden)]
    pub fn panic_on_compiles(&mut self, submissions: &[u64]) {
        self.injected_panics.extend_from_slice(submissions);
    }

    /// Changes the IR byte budget for outstanding trace requests.
    pub fn set_trace_byte_budget(&mut self, bytes: usize) {
        self.trace_byte_budget = bytes;
    }

    /// IR bytes held by outstanding trace requests.
    #[must_use]
    pub const fn pending_trace_bytes(&self) -> usize {
        self.pending_trace_bytes
    }

    /// Marks every outstanding trace request canceled. Their workers still
    /// finish, because a running compile cannot be interrupted, but the
    /// results are dropped inside the broker and never reach
    /// [`Self::poll_trace`].
    pub fn cancel_trace_requests(&mut self) {
        for request in &mut self.pending {
            request.canceled = true;
        }
    }

    fn enqueue(
        &mut self,
        tier: OptimizationTier,
        payload: Payload,
    ) -> Result<CompilationId, SubmitError> {
        if self.pending.len() == self.capacity {
            return Err(SubmitError::QueueFull);
        }

        let id = CompilationId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let job = CompilationJob {
            id,
            tier,
            payload: payload.share(),
            inject_panic: self.injected_panics.contains(&id.0),
        };
        match &mut self.adapter {
            Adapter::Background(worker) => {
                if worker.is_none() {
                    *worker = Some(BackgroundWorker::start(
                        self.capacity,
                        #[cfg(test)]
                        self.test_gate.clone(),
                    )?);
                }
                let Some(worker) = worker.as_ref() else {
                    return Err(SubmitError::WorkerUnavailable);
                };
                worker.submit(job)?;
            }
            Adapter::Synchronous { queue, .. } => queue.push_back(job),
        }
        self.pending.push(PendingRequest {
            id,
            tier,
            payload,
            canceled: false,
        });
        Ok(id)
    }

    /// Return one finished trace request, if one is ready. Canceled requests
    /// never come back.
    ///
    /// The background adapter never waits. The synchronous adapter compiles
    /// its next request on this call.
    pub fn poll_trace(&mut self) -> Option<CompletedTrace> {
        self.step(None)
    }

    /// Whether requests compile on a worker thread, so that a result is not
    /// there when it is asked for.
    #[must_use]
    pub const fn is_background(&self) -> bool {
        matches!(self.adapter, Adapter::Background(_))
    }

    /// Like [`Self::poll_trace`], but a background worker that has nothing
    /// finished is waited for, up to `limit`. The caller does this when it
    /// can do nothing better than the code the result replaces: the
    /// interpreter's instructions in the meantime are the cost, and a
    /// blocked thread runs none. Returns `None` when the limit passes or
    /// nothing is outstanding. The synchronous adapter does not wait.
    pub fn wait_trace(&mut self, limit: Duration) -> Option<CompletedTrace> {
        if self.pending.is_empty() {
            return None;
        }
        self.step(Some(limit))
    }

    /// One adapter step: the next finished request. Releases the request's
    /// bookkeeping and swallows canceled results.
    fn step(&mut self, wait: Option<Duration>) -> Option<CompletedTrace> {
        loop {
            let finished = match &mut self.adapter {
                Adapter::Background(Some(worker)) => match wait {
                    Some(limit) => worker.poll_for(limit),
                    None => worker.poll(),
                },
                Adapter::Background(None) => None,
                Adapter::Synchronous {
                    queue,
                    delay,
                    waited,
                } => {
                    if queue.is_empty() {
                        None
                    } else if *waited < *delay {
                        *waited += 1;
                        None
                    } else {
                        *waited = 0;
                        queue.pop_front().map(CompilationJob::compile)
                    }
                }
            }?;
            let position = self
                .pending
                .iter()
                .position(|request| request.id == finished.id);
            let request = position.map(|position| self.pending.remove(position));
            if let Some(request) = &request {
                self.pending_trace_bytes = self
                    .pending_trace_bytes
                    .saturating_sub(request.payload.bytes);
            }
            if request.as_ref().is_some_and(|request| request.canceled) {
                // The caller flushed after submitting. Look for the next one.
                continue;
            }
            return Some(finished);
        }
    }
}

enum Adapter {
    Background(Option<BackgroundWorker>),
    Synchronous {
        queue: VecDeque<CompilationJob>,
        delay: u32,
        waited: u32,
    },
}

struct Payload {
    ir: Arc<TraceIr>,
    epoch: u64,
    bytes: usize,
    options: TraceCompileOptions,
    /// Set for a request to compile a side apart from its tree.
    side: Option<SideJob>,
    /// The members `ir` plans to, when the caller planned them already.
    planned: Option<Arc<Vec<TreeMember>>>,
}

impl Payload {
    fn share(&self) -> Self {
        Self {
            ir: Arc::clone(&self.ir),
            epoch: self.epoch,
            bytes: self.bytes,
            options: self.options,
            side: self.side.clone(),
            planned: self.planned.clone(),
        }
    }
}

struct PendingRequest {
    id: CompilationId,
    tier: OptimizationTier,
    payload: Payload,
    /// A request whose caller flushed. Its result is dropped.
    canceled: bool,
}

struct CompilationJob {
    id: CompilationId,
    tier: OptimizationTier,
    payload: Payload,
    /// Panic inside the compile (see [`JitBroker::panic_on_compiles`]).
    inject_panic: bool,
}

/// The message of a caught panic's payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a panic with no message".to_owned()
    }
}

impl CompilationJob {
    fn compile(self) -> CompletedTrace {
        // On whichever thread compiles. `compile_trace_with` moves on to
        // code generation itself.
        let _phase = alloc_phase::enter(Phase::Optimization);
        let started = std::time::Instant::now();
        let Payload {
            ir,
            epoch,
            options,
            side,
            planned,
            ..
        } = self.payload;
        let is_side = side.is_some();
        // A panic in the optimizer or in Cranelift (its register allocator
        // panics on malformed code, and release builds run no verifier to
        // reject that first) would end the worker thread, and every later
        // request with it. The panic stops at this one compile instead and
        // rejects its trace. The compile owns everything it builds (IR
        // copies, the Cranelift module and builder); state shared between
        // compiles is the shared stubs, which publish an address only once
        // its stub is complete and take a poisoned lock as it is. So the
        // thread is safe to compile the next request.
        let inject_panic = self.inject_panic;
        let tier = self.tier;
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert!(!inject_panic, "injected compile panic");
            let _icount = crate::jit::icount::scope("compile-job");
            crate::jit::icount::around(
                if is_side { "job side" } else { "job tree" },
                ir.body.len(),
                || {
                    Self::compile_payload(
                        &ir,
                        tier,
                        options,
                        side,
                        planned.as_deref().map(Vec::as_slice),
                    )
                },
                |_| "all",
            )
        }));
        let result = compiled.unwrap_or_else(|payload| {
            let message = panic_message(payload.as_ref());
            if crate::jit::debug_env::trace() {
                eprintln!(
                    "jit-trace: compile of {} tier={tier:?} body={} panicked, trace rejected: {message}",
                    if is_side { "side" } else { "trace" },
                    ir.body.len(),
                );
            }
            Err(TraceCompileError::Panicked(message))
        });
        if crate::jit::debug_env::trace() && !matches!(result, Err(TraceCompileError::Panicked(_)))
        {
            eprintln!(
                "jit-trace: compiled {} tier={:?} body={} in {:?} at {:?}",
                if matches!(result, Ok(CompileOutput::Side(_))) {
                    "side"
                } else {
                    "trace"
                },
                self.tier,
                ir.body.len(),
                started.elapsed(),
                started,
            );
        }
        CompletedTrace {
            id: self.id,
            tier: self.tier,
            epoch,
            result,
        }
    }

    fn compile_payload(
        ir: &TraceIr,
        tier: OptimizationTier,
        options: TraceCompileOptions,
        side: Option<SideJob>,
        planned: Option<&[TreeMember]>,
    ) -> Result<CompileOutput, TraceCompileError> {
        if side.is_none() {
            super::compile_dump::write(ir, tier, options, planned);
        }
        match side {
            Some(job) => compile_side_with_plan(
                ir,
                &job.detached,
                &job.apart,
                job.side,
                job.exit_style,
                &job.context,
                tier,
                options,
                job.planned.as_deref().map(Vec::as_slice),
            )
            .map(|side| CompileOutput::Side(Box::new(side))),
            None => compile_trace_with_plan(ir, tier, options, planned)
                .map(|tree| CompileOutput::Tree(Box::new(tree))),
        }
    }
}

struct BackgroundWorker {
    jobs: Option<SyncSender<CompilationJob>>,
    completed: Receiver<CompletedTrace>,
    shutting_down: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

#[cfg(test)]
struct BackgroundTestGate {
    started: Mutex<bool>,
    started_signal: Condvar,
    released: Mutex<bool>,
    release_signal: Condvar,
}

#[cfg(test)]
impl BackgroundTestGate {
    fn new() -> Self {
        Self {
            started: Mutex::new(false),
            started_signal: Condvar::new(),
            released: Mutex::new(false),
            release_signal: Condvar::new(),
        }
    }

    fn wait_for_worker(&self) {
        let mut started = self.started.lock().expect("test gate lock is available");
        *started = true;
        self.started_signal.notify_all();
        let released = self.released.lock().expect("test gate lock is available");
        drop(started);
        let _released = self
            .release_signal
            .wait_while(released, |released| !*released)
            .expect("test gate lock is available");
    }

    fn wait_until_blocked(&self) {
        let started = self.started.lock().expect("test gate lock is available");
        let _started = self
            .started_signal
            .wait_while(started, |started| !*started)
            .expect("test gate lock is available");
    }

    fn release(&self) {
        *self.released.lock().expect("test gate lock is available") = true;
        self.release_signal.notify_all();
    }
}

impl BackgroundWorker {
    fn start(
        capacity: usize,
        #[cfg(test)] test_gate: Option<Arc<BackgroundTestGate>>,
    ) -> Result<Self, SubmitError> {
        let (job_sender, job_receiver) = mpsc::sync_channel::<CompilationJob>(capacity);
        let (completed_sender, completed_receiver) = mpsc::channel();
        let shutting_down = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutting_down);
        let worker = thread::Builder::new()
            .name("oximond-jit".to_owned())
            .spawn(move || {
                // Baseline code makes a hot loop productive immediately;
                // optimized recompiles only improve it. Serving every queued
                // baseline before any deferred optimized job keeps a short
                // program from waiting behind a long optimized compile.
                let mut deferred_optimized: VecDeque<CompilationJob> = VecDeque::new();
                let compile_job = |job: CompilationJob| -> bool {
                    #[cfg(test)]
                    if let Some(gate) = &test_gate {
                        gate.wait_for_worker();
                    }
                    completed_sender.send(job.compile()).is_ok()
                };
                while !worker_shutdown.load(Ordering::Acquire) {
                    loop {
                        match job_receiver.try_recv() {
                            Ok(job) if job.tier == OptimizationTier::Baseline => {
                                if !compile_job(job) {
                                    return;
                                }
                            }
                            Ok(job) => deferred_optimized.push_back(job),
                            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                        }
                        if worker_shutdown.load(Ordering::Acquire) {
                            return;
                        }
                    }
                    if let Some(job) = deferred_optimized.pop_front() {
                        if !compile_job(job) {
                            break;
                        }
                        continue;
                    }
                    let Ok(job) = job_receiver.recv() else {
                        break;
                    };
                    if worker_shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    if !compile_job(job) {
                        break;
                    }
                }
            })
            .map_err(|_| SubmitError::WorkerUnavailable)?;
        Ok(Self {
            jobs: Some(job_sender),
            completed: completed_receiver,
            shutting_down,
            thread: Some(worker),
        })
    }

    fn submit(&self, job: CompilationJob) -> Result<(), SubmitError> {
        self.jobs
            .as_ref()
            .ok_or(SubmitError::WorkerUnavailable)?
            .try_send(job)
            .map_err(|error| match error {
                TrySendError::Full(_) => SubmitError::QueueFull,
                TrySendError::Disconnected(_) => SubmitError::WorkerUnavailable,
            })
    }

    fn poll(&self) -> Option<CompletedTrace> {
        self.completed.try_recv().ok()
    }

    fn poll_for(&self, limit: Duration) -> Option<CompletedTrace> {
        self.completed.recv_timeout(limit).ok()
    }
}

impl Drop for BackgroundWorker {
    fn drop(&mut self) {
        self.shutting_down.store(true, Ordering::Release);
        self.jobs.take();
        // A compile can be expensive. A VM may be dropped while the worker is
        // lowering a trace, and making that destructor wait would put the
        // compilation pause back on the caller thread. Dropping a JoinHandle
        // detaches the worker; it owns only the request and sender, so it
        // cannot observe freed VM or executable-cache state.
        self.thread.take();
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// `r2 = r0 + r1` at `pc`, then exit. Different pcs give different traces.
    fn trace_at(pc: usize) -> Arc<TraceIr> {
        use crate::jit_ir::trace::{
            ArithmeticOp, FrameId, RegisterEntry, Snapshot, SnapshotFrame, SnapshotPhase,
            TraceBuilder, TraceFrameInfo, TraceId, TraceOp, TraceSite, TraceTerminator, TraceType,
        };
        let mut b = TraceBuilder::new(
            TraceId(u32::try_from(pc).unwrap()),
            TraceFrameInfo {
                register_count: 3,
                code_len: 16,
                inlined: None,
                activation: None,
            },
        );
        b.mark_written(FrameId(0), 2);
        let left = b.input(0, TraceType::Float);
        let right = b.input(1, TraceType::Float);
        let result = b
            .emit(
                TraceSite {
                    frame: FrameId(0),
                    pc,
                },
                TraceOp::Arithmetic {
                    op: ArithmeticOp::Add,
                    left,
                    right,
                },
                Some(TraceType::Float),
            )
            .unwrap();
        let exit = b.snapshot(Snapshot::new(
            SnapshotPhase::After {
                pc,
                successor: pc + 1,
            },
            0,
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: Some(pc),
                registers: vec![
                    RegisterEntry::Value(left),
                    RegisterEntry::Value(right),
                    RegisterEntry::Value(result),
                ]
                .into(),
            }],
        ));
        Arc::new(b.finish(TraceTerminator::Exit { snapshot: exit }))
    }

    #[test]
    fn outstanding_work_is_bounded() {
        let mut broker = JitBroker::synchronous(1);
        broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 0)
            .expect("first request is accepted");
        assert_eq!(
            broker.submit_trace(trace_at(2), OptimizationTier::Baseline, 0),
            Err(SubmitError::QueueFull)
        );
    }

    #[test]
    fn dropping_background_broker_with_queued_work_is_safe() {
        let mut broker = JitBroker::new(2);
        broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 0)
            .expect("first request is accepted");
        broker
            .submit_trace(trace_at(1), OptimizationTier::Optimized, 0)
            .expect("second request is accepted");
        drop(broker);
    }

    #[test]
    fn submit_and_drop_do_not_wait_for_a_blocked_background_compile() {
        let gate = Arc::new(BackgroundTestGate::new());
        let mut broker = JitBroker::new(1);
        broker.test_gate = Some(Arc::clone(&gate));
        broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 0)
            .expect("submission returns before compilation finishes");
        gate.wait_until_blocked();
        assert!(
            broker.poll_trace().is_none(),
            "the blocked job cannot complete"
        );

        // This would deadlock if BackgroundWorker::drop joined the worker.
        drop(broker);
        gate.release();
    }

    #[test]
    fn trace_requests_deduplicate_by_content_tier_and_epoch() {
        let mut broker = JitBroker::synchronous(4);
        let first = broker
            .submit_trace(trace_at(3), OptimizationTier::Baseline, 0)
            .unwrap();
        // Same content, a different recording id: one request.
        let mut again = (*trace_at(3)).clone();
        again.id = crate::jit_ir::trace::TraceId(99);
        assert_eq!(
            broker
                .submit_trace(Arc::new(again), OptimizationTier::Baseline, 0)
                .unwrap(),
            first
        );
        let bytes = broker.pending_trace_bytes();
        assert!(bytes > 0);
        let optimized = broker
            .submit_trace(trace_at(3), OptimizationTier::Optimized, 0)
            .unwrap();
        let next_epoch = broker
            .submit_trace(trace_at(3), OptimizationTier::Baseline, 1)
            .unwrap();
        let other = broker
            .submit_trace(trace_at(4), OptimizationTier::Baseline, 0)
            .unwrap();
        let ids = [first, optimized, next_epoch, other];
        for (index, id) in ids.iter().enumerate() {
            assert!(ids[index + 1..].iter().all(|other| other != id));
        }
    }

    #[test]
    fn trace_requests_complete_in_submission_order_with_their_epoch() {
        let mut broker = JitBroker::synchronous_with_delay(8, 2);
        let first = broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 7)
            .unwrap();
        let second = broker
            .submit_trace(trace_at(2), OptimizationTier::Baseline, 8)
            .unwrap();
        let mut traces = Vec::new();
        for _ in 0..40 {
            if let Some(done) = broker.poll_trace() {
                traces.push((done.id(), done.epoch()));
                done.into_result().expect("the trace compiles");
            }
        }
        assert_eq!(traces, vec![(first, 7), (second, 8)]);
        assert_eq!(broker.pending_trace_bytes(), 0);
    }

    #[test]
    fn trace_byte_admission_is_released_when_a_result_is_polled() {
        let mut broker = JitBroker::synchronous(8);
        let one = trace_at(1).estimated_bytes();
        broker.set_trace_byte_budget(one * 2 + one / 2);
        broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 0)
            .unwrap();
        broker
            .submit_trace(trace_at(2), OptimizationTier::Baseline, 0)
            .unwrap();
        assert_eq!(
            broker.submit_trace(trace_at(3), OptimizationTier::Baseline, 0),
            Err(SubmitError::BytesExceeded)
        );
        // A duplicate of an outstanding request costs nothing.
        assert!(
            broker
                .submit_trace(trace_at(2), OptimizationTier::Baseline, 0)
                .is_ok()
        );
        assert!(broker.poll_trace().is_some());
        assert!(
            broker
                .submit_trace(trace_at(3), OptimizationTier::Baseline, 0)
                .is_ok()
        );
    }

    #[test]
    fn canceled_trace_requests_never_complete() {
        let mut broker = JitBroker::synchronous(8);
        broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 0)
            .unwrap();
        broker.cancel_trace_requests();
        // A canceled request does not absorb a later identical one.
        let fresh = broker
            .submit_trace(trace_at(1), OptimizationTier::Baseline, 1)
            .unwrap();
        let mut traces = Vec::new();
        while let Some(done) = broker.poll_trace() {
            traces.push(done.id());
        }
        assert_eq!(traces, vec![fresh]);
        assert_eq!(broker.pending_trace_bytes(), 0);
    }

    #[test]
    fn background_trace_compilation_delivers_runnable_code() {
        use crate::jit::{TraceControl, TraceInputSlot, TraceOutcome};
        let mut broker = JitBroker::new(2);
        let ir = trace_at(0);
        let id = broker
            .submit_trace(Arc::clone(&ir), OptimizationTier::Baseline, 0)
            .unwrap();
        let started = Instant::now();
        let done = loop {
            if let Some(done) = broker.poll_trace() {
                break done;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "worker stalled"
            );
            std::thread::yield_now();
        };
        assert_eq!(done.id(), id);
        let compiled = done.into_result().expect("the trace compiles");
        let inputs = [
            TraceInputSlot::from_scalar(crate::jit_ir::trace::TraceScalar::Float(1.0)),
            TraceInputSlot::from_scalar(crate::jit_ir::trace::TraceScalar::Float(2.0)),
        ];
        let mut area = vec![0_i64; compiled.exit_area_len()];
        let mut control = TraceControl::new(10);
        assert!(matches!(
            compiled.run(&inputs, &mut control, &mut area),
            TraceOutcome::Exit { .. }
        ));
    }

    /// Polls `broker` until a result arrives, for up to ten seconds.
    fn next_result(broker: &mut JitBroker) -> CompletedTrace {
        let started = Instant::now();
        loop {
            if let Some(done) = broker.poll_trace() {
                return done;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "worker stalled"
            );
            std::thread::yield_now();
        }
    }

    /// Submits three traces and makes the second compile panic: it alone is
    /// rejected with the panic's message, and the third still compiles.
    fn a_panicking_compile_rejects_only_its_trace(mut broker: JitBroker) {
        broker.panic_on_compiles(&[1]);
        let ids: Vec<_> = (1..=3)
            .map(|pc| {
                broker
                    .submit_trace(trace_at(pc), OptimizationTier::Baseline, 0)
                    .unwrap()
            })
            .collect();
        let mut results = Vec::new();
        for _ in 0..3 {
            let done = next_result(&mut broker);
            results.push((done.id(), done.into_result()));
        }
        results.sort_by_key(|(id, _)| *id);
        assert_eq!(results.iter().map(|(id, _)| *id).collect::<Vec<_>>(), ids);
        assert!(results[0].1.is_ok());
        assert_eq!(
            results[1].1.as_ref().err(),
            Some(&TraceCompileError::Panicked(
                "injected compile panic".to_owned()
            ))
        );
        assert!(results[2].1.is_ok());
        // The broker still takes and compiles work after the panic.
        let id = broker
            .submit_trace(trace_at(4), OptimizationTier::Optimized, 0)
            .unwrap();
        let done = next_result(&mut broker);
        assert_eq!(done.id(), id);
        assert!(done.into_result().is_ok());
    }

    #[test]
    fn a_panicking_background_compile_leaves_the_worker_running() {
        a_panicking_compile_rejects_only_its_trace(JitBroker::new(4));
    }

    #[test]
    fn a_panicking_synchronous_compile_returns_to_the_caller() {
        a_panicking_compile_rejects_only_its_trace(JitBroker::synchronous(4));
    }
}
