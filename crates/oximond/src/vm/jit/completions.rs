//! Installing finished JIT compilations.

use crate::jit::JitBroker;
use crate::runtime::alloc_phase::{self, Phase};
use crate::vm::Vm;
use crate::vm::jit::state::JIT_QUEUE_CAPACITY;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a background compile of a stepping stone is waited for. A
/// stone and its tree are a few dozen guest instructions; a worker that has
/// not answered by now is stuck behind something else, and the interpreter
/// runs on.
const STEPPING_STONE_WAIT: Duration = Duration::from_millis(100);

/// The broker a VM creates on its first compile request. With
/// `OXIMOND_JIT_SYNC` set to anything but `0` or the empty string, it
/// compiles on the recording thread and installs at the same safepoint, so
/// when a trace first runs natively does not depend on thread scheduling.
/// Unset, it is the background-worker broker, unless the process may run on
/// one CPU only (an affinity mask or a cgroup quota of one): the worker and
/// the recording thread then share it, so a request waits for the scheduler to
/// preempt the interpreter before its compile starts, and the interpreter
/// runs the loop it is waiting to replace for that long. Compiling inline
/// costs the same CPU time and installs at the next safepoint.
///
/// `OXIMOND_JIT_SYNC_DELAY=N` compiles inline but installs each result only
/// after `N` safepoint polls (see [`JitBroker::synchronous_with_delay`]). A
/// sweep over `N` replays the install points a background worker can hit,
/// so a program whose speed depends on compile timing can be measured at
/// each of them deterministically.
fn default_broker() -> JitBroker {
    let cpus = std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZero::get);
    let delay = std::env::var_os("OXIMOND_JIT_SYNC_DELAY")
        .and_then(|value| value.to_str().and_then(|value| value.parse().ok()));
    let mut broker = if let Some(delay) = delay {
        JitBroker::synchronous_with_delay(JIT_QUEUE_CAPACITY, delay)
    } else if compiles_inline(std::env::var_os("OXIMOND_JIT_SYNC").as_deref(), cpus) {
        JitBroker::synchronous(JIT_QUEUE_CAPACITY)
    } else {
        JitBroker::new(JIT_QUEUE_CAPACITY)
    };
    if let Some(list) = std::env::var_os("OXIMOND_JIT_INJECT_COMPILE_PANICS") {
        broker.panic_on_compiles(&injected_panics(&list.to_string_lossy()));
    }
    broker
}

/// The compile requests `OXIMOND_JIT_INJECT_COMPILE_PANICS` names: a comma
/// separated list of request numbers, counting from 0, whose compiles panic
/// on purpose (see [`JitBroker::panic_on_compiles`]). It is a test aid for
/// the panic guard in programs the tests cannot reach into, such as the CLI
/// or an embedder. Entries that are not numbers are ignored.
fn injected_panics(list: &str) -> Vec<u64> {
    list.split(',')
        .filter_map(|entry| entry.trim().parse().ok())
        .collect()
}

/// Whether the default broker compiles inline: `OXIMOND_JIT_SYNC` says so
/// (anything but `0` or the empty string), or says nothing and the process
/// may run on one CPU only.
fn compiles_inline(sync: Option<&std::ffi::OsStr>, cpus: Option<usize>) -> bool {
    match sync {
        Some(value) => !value.is_empty() && value != "0",
        None => cpus == Some(1),
    }
}

impl Vm {
    /// Compiles traces on this thread instead of a background worker.
    ///
    /// With background compilation, when a hot trace first runs natively
    /// depends on thread scheduling. Tests and embedders that need
    /// reproducible tiering, such as assertions on
    /// [`Self::jit_statistics`], opt into this. Pending background work is
    /// dropped, so call it before executing code.
    pub fn use_synchronous_jit_compilation(&mut self) {
        self.jit.broker = Some(JitBroker::synchronous(JIT_QUEUE_CAPACITY));
        self.forget_trace_compilations();
    }

    /// Compiles a side apart from its tree wherever the tracer allows it,
    /// also where a compile of the whole tree with the side in it would cost
    /// less (a small tree the MIR backend compiled, see
    /// `OXIMOND_JIT_JOIN`). Tests whose subject is a side compiled apart
    /// call this before executing code.
    pub fn keep_jit_sides_apart(&mut self) {
        self.jit.trace.keep_sides_apart();
    }

    /// Compiles a hot loop of a top-level chunk however little work the chunk
    /// has left for it. By default the tracer leaves such a loop to the
    /// interpreter when the rest of its run cannot repay a compile (a table
    /// filled once), and lets a loop whose work is pattern calls run a while
    /// in the interpreter before its trace compiles. Tests of what traces do
    /// run small loops in a main chunk and call this before executing code.
    #[doc(hidden)]
    pub fn ignore_remaining_work(&mut self) {
        self.jit.trace.set_judge_one_shot_loops(false);
    }

    /// Lets traces recorded from now on call host C functions themselves
    /// (the default), or makes their recordings abort at the calls.
    #[doc(hidden)]
    pub fn set_trace_vmcall(&mut self, enabled: bool) {
        self.jit.trace.set_vmcall(enabled);
    }

    /// Sets how many guest instructions a loop needs for its trace to call
    /// host C functions itself. Tests of what the calls do run small loops
    /// and set it to zero before executing code.
    #[doc(hidden)]
    pub fn set_vmcall_min_work(&mut self, work: u32) {
        self.jit.trace.set_vmcall_work(work);
    }

    /// Makes trace code compiled from now on count every jump into a side
    /// trace, which [`Self::jit_statistics`] reports as
    /// `trace.side_entries`. It costs a few instructions per jump, so it is
    /// off unless the `OXIMOND_JIT_STATS` variable is set. Tests and tools
    /// that assert on the counter call this before executing code.
    pub fn count_trace_links(&mut self) {
        self.jit.trace.set_count_links(true);
    }

    /// Keeps a guard on every branch of code recorded from now on, as before
    /// value selects: a pure two-arm branch then leaves or takes a side
    /// instead of running both arms. Tests whose subject is the sides of
    /// such branches call this before executing code.
    #[doc(hidden)]
    pub fn keep_trace_branch_guards(&mut self) {
        self.jit.trace.set_value_selects(false);
    }

    /// Lets a root take a register untyped once its loop's entries found it
    /// nil on some entries and a number on others, which is the default
    /// (`OXIMOND_JIT_UNTYPED=0` turns it off). Tests of those roots call
    /// this before executing code, so they hold with the variable set.
    #[doc(hidden)]
    pub fn take_untyped_trace_inputs(&mut self) {
        self.jit.trace.set_untyped_inputs(true);
    }

    /// Synchronous compilation where each request installs only after
    /// `delay_polls` safepoint polls. Tests sweep the delay to cover the
    /// install points a background worker can hit.
    #[doc(hidden)]
    pub fn use_delayed_synchronous_jit_compilation(&mut self, delay_polls: u32) {
        self.jit.broker = Some(JitBroker::synchronous_with_delay(
            JIT_QUEUE_CAPACITY,
            delay_polls,
        ));
        self.forget_trace_compilations();
    }

    /// Compiles on a background worker thread, whatever `OXIMOND_JIT_SYNC`
    /// and the CPU count say. Tests of the worker call this before executing
    /// code. Pending work is dropped, as for
    /// [`Self::use_synchronous_jit_compilation`].
    #[doc(hidden)]
    pub fn use_background_jit_compilation(&mut self) {
        self.jit.broker = Some(JitBroker::new(JIT_QUEUE_CAPACITY));
        self.forget_trace_compilations();
    }

    /// Makes the compiles of the VM's `submissions`-th compile requests
    /// panic (counting from 0 on the current broker; see
    /// [`JitBroker::panic_on_compiles`]). Tests use it to check that a
    /// panicking compile rejects only its own trace. Call it after choosing
    /// the broker: a replaced broker forgets it.
    #[doc(hidden)]
    pub fn panic_on_trace_compiles(&mut self, submissions: &[u64]) {
        self.jit
            .broker
            .get_or_insert_with(default_broker)
            .panic_on_compiles(submissions);
    }

    /// A replaced broker restarts its compilation ids and never finishes the
    /// old requests, so the trace policy drops what it waits for.
    fn forget_trace_compilations(&mut self) {
        if self.jit.trace.has_compile_work() {
            self.jit.trace.flush(&mut self.jit.statistics.trace);
        }
    }

    /// Hands the trace policy's compile requests to the broker and installs
    /// finished trace compilations. Runs at VM safepoints only: a loop
    /// header consult, or the end of a recording. Installation never happens
    /// while native trace code runs.
    pub(crate) fn service_trace_compilations(&mut self) {
        let submitting = alloc_phase::enter(Phase::Recording);
        let requests = self.jit.trace.take_outbox();
        let options = self.jit.trace.compile_options();
        let mut awaited = Vec::new();
        for request in requests {
            let broker = self.jit.broker.get_or_insert_with(default_broker);
            broker.set_trace_options(options);
            let awaits = self.jit.trace.awaits_result(&request);
            let submitted = match &request.side {
                Some(job) => broker.submit_side(
                    Arc::clone(&request.ir),
                    request.tier,
                    request.epoch,
                    job.clone(),
                ),
                None => broker.submit_trace_planned(
                    Arc::clone(&request.ir),
                    request.planned.clone(),
                    request.tier,
                    request.epoch,
                ),
            };
            match submitted {
                Ok(id) => {
                    if awaits {
                        awaited.push(id);
                    }
                    self.jit
                        .trace
                        .note_submitted(&mut self.jit.statistics.trace, request, id);
                }
                Err(error) => {
                    self.jit.trace.note_submit_failed(
                        &mut self.jit.statistics.trace,
                        request,
                        error,
                    );
                }
            }
        }
        submitting.end();
        // The code a stepping stone leads to is waited for (see
        // `TracePolicy::awaits_result`): the interpreter would spend the
        // time on instructions the compile exists to remove. The bound is
        // for all of them together: unrelated results that finish first do
        // not restart it.
        let deadline = (!awaited.is_empty()).then(|| Instant::now() + STEPPING_STONE_WAIT);
        while !awaited.is_empty()
            && let Some(deadline) = deadline
            && let Some(remaining) = deadline
                .checked_duration_since(Instant::now())
                .filter(|remaining| !remaining.is_zero())
            && let Some(broker) = self
                .jit
                .broker
                .as_mut()
                .filter(|broker| broker.is_background())
            && let Some(done) = broker.wait_trace(remaining)
        {
            awaited.retain(|id| *id != done.id());
            let (id, epoch) = (done.id(), done.epoch());
            self.jit.trace.complete_compile(
                &mut self.jit.statistics.trace,
                id,
                epoch,
                done.into_output(),
            );
        }
        // A synchronous broker compiles inside `poll_trace` and enters the
        // compiler phases itself; `complete_compile` enters installation.
        while let Some(done) = self.jit.broker.as_mut().and_then(JitBroker::poll_trace) {
            let (id, epoch) = (done.id(), done.epoch());
            self.jit.trace.complete_compile(
                &mut self.jit.statistics.trace,
                id,
                epoch,
                done.into_output(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{compiles_inline, injected_panics};
    use std::ffi::OsStr;

    #[test]
    fn a_process_with_one_cpu_compiles_inline_unless_told_otherwise() {
        assert!(compiles_inline(None, Some(1)));
        assert!(!compiles_inline(None, Some(2)));
        assert!(!compiles_inline(None, None));
        // The variable decides when it is set.
        assert!(compiles_inline(Some(OsStr::new("1")), Some(8)));
        assert!(!compiles_inline(Some(OsStr::new("0")), Some(1)));
        assert!(!compiles_inline(Some(OsStr::new("")), Some(1)));
    }

    #[test]
    fn injected_panics_parse_a_comma_separated_list() {
        assert_eq!(injected_panics("0"), [0]);
        assert_eq!(injected_panics("0, 3,x,,7"), [0, 3, 7]);
        assert_eq!(injected_panics(""), [] as [u64; 0]);
    }
}
