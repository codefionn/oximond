//! Native entry: runs a compiled trace for the active frame at a loop header.

use crate::bytecode::Chunk;
use crate::vm::jit::state::{
    NATIVE_GC_PENDING_BUDGET, NATIVE_STEADY_STATE_BUDGET, TRACING_REJECTED_BACKEDGE_SKIP,
    TRACING_REJECTED_BACKEDGE_SKIP_MAX,
};
use crate::vm::jit::trace::policy::FUNCTION_ENTRY_PC;
use crate::vm::jit::trace::{LoopHeader, TraceDispatch, TracePolicy, TraceRun};
use crate::vm::stack::CallStack;
use crate::vm::{Frame, Vm, VmResult};
use std::sync::Arc;

impl Vm {
    /// Drops compiled code and tiering decisions after all active native
    /// calls have returned. The broker may finish old work, but without its
    /// pending IDs those completions cannot reinstall flushed traces.
    pub(crate) fn service_jit_flush(&mut self) {
        if self.active_native_runs != 0 || !self.runtime.take_jit_flush_request() {
            return;
        }
        let mut broker = self.jit.broker.take();
        // Trace compiles in flight finish on their worker but never install.
        if let Some(broker) = broker.as_mut() {
            broker.cancel_trace_requests();
        }
        let mut statistics = self.jit.statistics;
        let placeholder = TracePolicy::new(self.execution_mode);
        let mut trace = std::mem::replace(&mut self.jit.trace, placeholder);
        trace.flush(&mut statistics.trace);
        if let Some(options) = self.runtime.take_jit_options() {
            trace.set_jit_options(options);
        }
        self.jit = crate::vm::jit::state::JitState::default();
        self.jit.broker = broker;
        self.jit.statistics = statistics;
        self.jit.trace = trace;
    }

    /// Reports whether this dispatch may enter native code. Interpreter mode
    /// never can.
    ///
    /// Only a backward step or a tail call (see `JitState::tail_landing`)
    /// reaches a loop header a trace may own, so this cheap check keeps the
    /// interpreter loop from calling the large entry routine on every forward
    /// instruction.
    #[inline(always)]
    pub(crate) fn native_entry_possible(&mut self, frame: &Frame) -> bool {
        self.service_jit_flush();
        if self.runtime.jit_flush_requested() || !self.runtime.jit_enabled() {
            // Only tracing mode raises them, so interpreter mode stores
            // nothing here.
            if self.jit.tail_landing || self.jit.call_landing {
                self.jit.tail_landing = false;
                self.jit.call_landing = false;
            }
            return false;
        }
        self.jit.tail_landing
            || self.jit.call_landing
            || self.jit.trace.has_loaded_profile()
            || frame
                .last_interpreted_pc
                .get()
                .is_some_and(|previous| frame.pc <= previous)
    }

    /// The header a tail call just landed on in the top frame, if the frame
    /// stands where one leaves it: pc 0 of the callee, with no previous pc.
    /// The span of such a loop in its chunk is the whole function.
    fn tail_landing_header(frame: &Frame) -> Option<(usize, usize)> {
        (frame.pc == 0 && frame.last_interpreted_pc.is_none())
            .then(|| (0, frame.chunk.code.len().saturating_sub(1)))
    }

    /// The fast loop's settlement of a tail call that landed at pc 0 of
    /// `chunk`, like [`Self::settle_fast_backedge`] for a backward branch: a
    /// header that only counts, waits for its compile or is rejected is
    /// settled without leaving the fast loop. A rejected one is remembered,
    /// so its later landings skip the lookup (see
    /// `TracePolicy::tail_header_rejected`). No backedge skip is armed or
    /// spent, since the skip counts backward branches. `false` changes
    /// nothing and leaves for the full lookup.
    #[inline(never)]
    pub(crate) fn settle_fast_tail_landing(&mut self, chunk: &Chunk) -> bool {
        if self.runtime.jit_flush_requested() || !self.runtime.jit_enabled() {
            return false;
        }
        // A hook keeps native code out, and `maybe_run_native` would only
        // drop the flag. The hooked fast loop goes on instead of leaving at
        // every landing.
        if !self.debug_hooks.is_empty() {
            self.jit.tail_landing = false;
            return true;
        }
        // A recording follows the call itself, and the consult would only
        // answer `Interpret`.
        if self.jit.trace.recording() {
            self.jit.tail_landing = false;
            return true;
        }
        if self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
        if self.jit.trace.rejected_header_dispatch(chunk, 0).is_some() {
            self.jit.trace.remember_rejected_tail(chunk);
            self.jit.tail_landing = false;
            return true;
        }
        let settled = self.jit.trace.count_backedge(chunk, 0).is_some();
        if settled {
            self.jit.tail_landing = false;
        }
        settled
    }

    /// Whether the frame `frames` just pushed at `depth` (counting frames)
    /// runs the chunk of the frame that called it: a direct self call, whose
    /// callee's entry the policy may own. Tells the dispatch loop to consult
    /// it (see `JitState::call_landing`). Only tracing mode looks.
    #[inline]
    pub(crate) fn note_call_landing(&mut self, frames: &CallStack, depth: usize) {
        // Interpreter mode pays this one test.
        if self.jit.trace.enabled() {
            self.note_tracing_call_landing(frames, depth);
        }
    }

    #[inline(never)]
    fn note_tracing_call_landing(&mut self, frames: &CallStack, depth: usize) {
        // Only a direct call (not `pcall`'s, say) returns to its caller the
        // way a function trace's return does.
        if frames.len() == depth + 1
            && depth > 0
            && frames[depth].pc == 0
            && matches!(
                frames[depth].caller,
                Some(crate::vm::ReturnTarget::Direct(_))
            )
            && !self
                .jit
                .trace
                .entry_rejected(frames[depth].chunk.address() as usize)
            && std::ptr::eq(
                frames[depth].chunk.address(),
                frames[depth - 1].chunk.address(),
            )
        {
            self.jit.call_landing = true;
        }
    }

    /// The fast loop's settlement of a self call that pushed a frame of
    /// `chunk`, like [`Self::settle_fast_tail_landing`]: a function entry
    /// that only counts, waits for its compile or is rejected is settled
    /// without leaving the fast loop. `false` changes nothing and leaves for
    /// the full lookup, which records or enters.
    #[inline(never)]
    pub(crate) fn settle_fast_call_landing(&mut self, chunk: &Chunk) -> bool {
        if self.runtime.jit_flush_requested() || !self.runtime.jit_enabled() {
            return false;
        }
        // As for a tail landing: under a hook there is nothing to enter.
        if !self.debug_hooks.is_empty() {
            self.jit.call_landing = false;
            return true;
        }
        // A recording sleeps through the callee or follows it, and the
        // consult would only answer `Interpret`.
        if self.jit.trace.recording() {
            self.jit.call_landing = false;
            return true;
        }
        if self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
        if self
            .jit
            .trace
            .rejected_header_dispatch(chunk, FUNCTION_ENTRY_PC)
            .is_some()
        {
            self.jit.trace.remember_rejected_entry(chunk);
            self.jit.call_landing = false;
            return true;
        }
        let settled = self
            .jit
            .trace
            .count_backedge(chunk, FUNCTION_ENTRY_PC)
            .is_some();
        if settled {
            self.jit.call_landing = false;
        }
        settled
    }

    /// Skips the next fast-path backedge lookups after the tracer rejected
    /// the loop at `pc` (`latch` is its backward branch). Returns the number
    /// of backedges to credit to the header's hotness, which is none when a
    /// recent native run cancelled the arming.
    ///
    /// While a trace is installed the skip is a window of the header: it
    /// counts the backedges of this loop only, so a trace that installs
    /// elsewhere, a loop inside this one that still counts, and the entry of a
    /// trace right after this loop all get their lookups. A loop that stays
    /// rejected earns longer windows, up to `cap`.
    ///
    /// With no trace installed the skip is global (`jit.backedge_skip`) and
    /// swallows the backedges of every loop, which are only counting then. A
    /// loop with another still counting or waiting inside it keeps a short
    /// one.
    fn arm_rejected_backedge_skip(
        &mut self,
        chunk: &Chunk,
        pc: usize,
        latch: usize,
        cap: u32,
    ) -> u32 {
        if self.jit.trace.traces_live() {
            return self
                .jit
                .rejected_skips
                .arm(chunk.code.as_ptr() as usize, pc, cap);
        }
        // The skip is global and swallows the backedges of the loops around
        // this one too, which have to count them to get hot. A short loop the
        // tracer turned away (it runs a few trips per outer iteration) would
        // keep the loop around it from ever recording: each window ends on
        // the inner loop's backedge, one of the several an outer iteration
        // has, so the outer loop counts a seventh of its backedges at best.
        // While a loop around still counts, the lookups are the lesser cost.
        if self.jit.trace.enclosing_loop_unsettled(chunk, pc, latch) {
            return 0;
        }
        // The skip is global. A rejected loop re-arms it on every iteration,
        // so a trace that ran since the last arming would find the skip
        // eating the entries of its own header: an inner loop the tracer owns
        // would run its first 64 iterations interpreted on every call. A
        // recent native run therefore cancels one arming.
        if self.jit.trace.take_recent_entry() {
            self.jit.backedge_skip_next = TRACING_REJECTED_BACKEDGE_SKIP;
            return 0;
        }
        let skip = self.jit.backedge_skip_next;
        self.jit.backedge_skip = skip;
        let cap = if self.jit.trace.inner_loops_rejected(chunk, pc, latch) {
            TRACING_REJECTED_BACKEDGE_SKIP_MAX
        } else {
            TRACING_REJECTED_BACKEDGE_SKIP
        };
        self.jit.backedge_skip_next = (skip * 2).min(cap);
        skip
    }

    /// The lookup of [`Self::maybe_run_native`] for a fast-loop backedge
    /// exit, in the cases that leave nothing to enter or record: a header
    /// the tracer recently rejected opens its next skip, and a header that
    /// only counts or waits for
    /// a compile is counted. The fast loop then resumes without a trip through
    /// the dispatch loop. Crediting the skipped backedges is left out, since
    /// a rejected site does not count them. `false` changes nothing and
    /// leaves for the full lookup.
    pub(crate) fn resume_fast_backedge(&mut self, frames: &CallStack) -> bool {
        let frame = &frames[frames.len() - 1];
        let pc = frame.pc;
        if self.jit.fast_backedge_pc != Some(pc) {
            return false;
        }
        // Other exits share this one (a tail call that changed the frame, a
        // flush request), so only the latch tells a backedge. A pc left from
        // an earlier backedge must not pass for one.
        if frame
            .last_interpreted_pc
            .get()
            .is_none_or(|previous| pc > previous)
        {
            self.jit.fast_backedge_pc = None;
            return false;
        }
        let latch = frame.last_interpreted_pc.unwrap_or(pc);
        let settled = self.settle_fast_backedge(&frame.chunk, pc, latch);
        if settled {
            self.jit.fast_backedge_pc = None;
        }
        settled
    }

    /// Settles the backedge to the header at `pc` of `chunk` without leaving
    /// the fast loop, in the cases [`Self::resume_fast_backedge`] describes.
    /// `false` changes nothing and leaves for the full lookup.
    #[inline(never)]
    pub(crate) fn settle_fast_backedge(&mut self, chunk: &Chunk, pc: usize, latch: usize) -> bool {
        // A count hook leaves loop headers to the trace engine, which counts
        // for it. Any other hook keeps native code out.
        if self.runtime.jit_flush_requested()
            || !self.runtime.jit_enabled()
            || !self.count_only_hooks()
        {
            return false;
        }
        self.jit.statistics.fast_backedge_resumes += 1;
        // As in the full lookup, a finished compile installs before the
        // consult, which then enters it there.
        if !self.jit.trace.recording() && self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
        match self.jit.trace.rejected_header_dispatch(chunk, pc) {
            Some(TraceDispatch::Skip) => {
                self.arm_rejected_backedge_skip(
                    chunk,
                    pc,
                    latch,
                    TRACING_REJECTED_BACKEDGE_SKIP_MAX,
                );
            }
            Some(_) => {}
            None => match self.jit.trace.count_backedge(chunk, pc) {
                // The header waits for its compile or out a cooldown, which a
                // poll at the end of the skip finds, so the loop runs on
                // without a consult per iteration.
                Some(skip) => self.jit.backedge_skip = skip,
                None => return false,
            },
        }
        true
    }

    /// Runs the compiled trace at `index` on the frame at its loop header.
    ///
    /// Returns `None` when nothing ran and the frame is untouched. Otherwise
    /// the frame holds the exit's restored state, and the answer says whether
    /// the dispatcher must return for a safepoint (a poll exit).
    pub(crate) fn enter_native_trace(
        &mut self,
        frames: &mut CallStack,
        frame_index: usize,
        index: u32,
    ) -> VmResult<Option<bool>> {
        // A pending collection takes the small slice so the safepoint comes
        // back quickly. A constructor native code keeps virtual is counted as
        // it runs and charged to the collector as phantom debt when the run
        // leaves, so the debt a slice adds is known only at its end, and this
        // budget bounds it. A table the run really makes misses once a
        // collection is owed, and the interpreter takes the safepoint.
        let fuel = if self.runtime.gc_needs_collection() {
            NATIVE_GC_PENDING_BUDGET
        } else {
            NATIVE_STEADY_STATE_BUDGET
        };
        self.enter_native_trace_with_fuel(frames, frame_index, index, fuel)
    }

    /// [`Self::enter_native_trace`] with an explicit loop-back budget.
    pub(crate) fn enter_native_trace_with_fuel(
        &mut self,
        frames: &mut CallStack,
        frame_index: usize,
        index: u32,
        fuel: i64,
    ) -> VmResult<Option<bool>> {
        // A count hook's live count goes into the run and comes back out
        // with the instructions the run counted subtracted.
        let budget = self.live_count_budget();
        self.jit.trace.set_hook_budget(budget);
        let probe = if self.sampler.is_some() {
            self.begin_native_sample(frames, frame_index)
        } else {
            None
        };
        // A trace that resumes a coroutine runs it through this VM (see
        // `TracePolicy::set_resume_host`), which the helper then borrows as a
        // whole while the run is in flight. So the run gets no reference
        // into the VM: the one pointer to it, its own policy, statistics and
        // runtime, and the stack are what the run is lent, and `self` is not
        // used again until the run is over.
        let vm = std::ptr::from_mut::<Self>(self);
        // SAFETY: `vm` and `frames` are unique references made into
        // pointers, used by the run and by the helper that resumes a
        // coroutine (`TracePolicy::enter_lent`). Every pointer below is
        // projected from them and no reference to the VM or the stack is
        // used until the run returns.
        #[allow(unsafe_code)]
        let run = unsafe {
            (*vm).jit.trace.set_resume_host(vm as usize);
            let run = TracePolicy::enter_lent(
                std::ptr::addr_of_mut!((*vm).jit.trace),
                std::ptr::addr_of_mut!((*vm).jit.statistics.trace),
                index,
                std::ptr::from_mut(frames),
                frame_index,
                fuel,
                std::ptr::addr_of_mut!((*vm).runtime),
            );
            (*vm).jit.trace.set_resume_host(0);
            run
        };
        if let Some(probe) = probe {
            self.end_native_sample(probe);
        }
        // An answer the trace kept for the interpreter's replay is good for
        // the replay of the call it left before, nothing else.
        if !matches!(run, TraceRun::Ran { .. }) {
            self.forget_resume();
        }
        if budget.is_some()
            && let Some(left) = self.jit.trace.hook_budget()
        {
            self.set_live_count_budget(left);
        }
        match run {
            TraceRun::NotEntered => Ok(None),
            TraceRun::Returned { value } => {
                // The entered activation returned: return from its frame as
                // the interpreter's `Return` does. The trace was entered only
                // with a direct caller, which takes the value.
                let finished = self.finish_single_return_frame(frames, value)?;
                debug_assert!(finished.is_none(), "a function trace's frame has a caller");
                Ok(Some(false))
            }
            TraceRun::Failed => {
                let restore = self.jit.trace.take_restore_error();
                // A host call the trace made raised an error.
                Err(self.vmcall_error.take().unwrap_or_else(|| restore.into()))
            }
            TraceRun::Ran { poll, .. } => {
                // No re-entry marker is needed after a guard or loop exit.
                // The caller (`maybe_run_native`) returns to the dispatcher,
                // which executes the instruction at the resume pc before it
                // consults the policy again, so the trace cannot re-enter
                // the state it just left. A marker would only suppress the
                // next legitimate entry at the following backedge.
                Ok(Some(poll))
            }
        }
    }

    pub(crate) fn maybe_run_native(&mut self, frames: &mut CallStack) -> VmResult<bool> {
        // Taken before any early return, so no later exit finds them stale.
        let fast_backedge_pc = self.jit.fast_backedge_pc.take();
        let tail_landing = std::mem::take(&mut self.jit.tail_landing);
        let call_landing = std::mem::take(&mut self.jit.call_landing);
        self.service_jit_flush();
        if self.runtime.jit_flush_requested()
            || !self.runtime.jit_enabled()
            || !self.jit.trace.enabled()
        {
            return Ok(false);
        }
        // A recording follows the interpreter through calls into their
        // callees. Native code would run them where the recorder cannot see.
        if self.jit.trace.recording() {
            return Ok(false);
        }
        let frame_index = frames.len() - 1;
        // A refusal below skips later fast-path backedges only when this
        // lookup came from one.
        let arm_backedge_skip = fast_backedge_pc == Some(frames[frame_index].pc);
        if arm_backedge_skip {
            self.jit.statistics.fast_backedge_lookups += 1;
        }
        // Native code does not expose instruction boundaries to a hook that
        // fires on them. Only count hooks, which a counted trace accounts
        // for in bulk, let it run. Any other hook on any coroutine would
        // abort every recording, spending the site's retry budget for
        // nothing: leave the header alone.
        let counting = !self.debug_hooks.is_empty();
        if counting {
            if !self.hooks_allow_traces(frames) {
                return Ok(false);
            }
            // The instruction the count hook just fired on runs uncounted,
            // which the trace would count. Counted traces are loop traces:
            // function entries and tail landings stay with the interpreter.
            let frame = &frames[frame_index];
            if frame
                .extras_ref()
                .is_some_and(|extras| extras.skip_hook_pc_once == Some(frame.pc))
                || tail_landing
                || call_landing
            {
                return Ok(false);
            }
        }
        if self.jit.trace.has_compile_work() {
            self.service_trace_compilations();
        }
        let frame = &frames[frame_index];
        if let Some(index) = self.jit.trace.enter_profile(
            &mut self.jit.statistics.trace,
            frames.at(frame_index),
            &self.runtime,
            self.compatibility,
        ) {
            return Ok(self
                .enter_native_trace(frames, frame_index, index)?
                .unwrap_or(false));
        }
        // A self call pushed this frame: its entry is a function-entry site.
        if call_landing && frame.pc == 0 && frame.last_interpreted_pc.is_none() {
            let chunk = frame.chunk.as_arc();
            let header = LoopHeader {
                chunk: &chunk,
                depth: frames.len(),
                pc: FUNCTION_ENTRY_PC,
                latch: chunk.code.len().saturating_sub(1),
                profile: self.compatibility,
                coroutine: self.current_coroutine,
                captured: frames.open_in(frame.base, frame.top),
                captured_base: frame.base,
                closure: frame.closure,
                function: true,
            };
            let dispatch = self
                .jit
                .trace
                .at_loop_header(&mut self.jit.statistics.trace, &header);
            if !matches!(dispatch, TraceDispatch::Enter(_)) {
                self.jit.trace.remember_rejected_entry(Arc::as_ptr(&chunk));
            }
            return match dispatch {
                TraceDispatch::Interpret | TraceDispatch::Skip => Ok(false),
                TraceDispatch::Enter(index) => Ok(self
                    .enter_native_trace(frames, frame_index, index)?
                    .unwrap_or(false)),
            };
        }
        let backedge = frame
            .last_interpreted_pc
            .get()
            .filter(|previous| frame.pc <= *previous)
            .map(|latch| (frame.pc, latch));
        let from_tail = backedge.is_none();
        let Some((pc, latch)) = backedge.or_else(|| {
            tail_landing
                .then(|| Self::tail_landing_header(frame))
                .flatten()
        }) else {
            return Ok(false);
        };
        let chunk = frame.chunk.as_arc();
        let header = LoopHeader {
            chunk: &chunk,
            depth: frames.len(),
            pc,
            latch,
            profile: self.compatibility,
            coroutine: self.current_coroutine,
            captured: frames.open_in(frame.base, frame.top),
            captured_base: frame.base,
            closure: frame.closure,
            function: false,
        };
        let dispatch = self
            .jit
            .trace
            .at_loop_header(&mut self.jit.statistics.trace, &header);
        // A tail call header the tracer gave up on is remembered, so the
        // later landings of its cycle skip this lookup.
        if from_tail && !matches!(dispatch, TraceDispatch::Enter(_)) {
            self.jit.trace.remember_rejected_tail(Arc::as_ptr(&chunk));
        }
        // No skip is armed for a header the tracer still counts: a skip is
        // global, and would starve the counters of every other header for
        // its duration.
        match dispatch {
            TraceDispatch::Interpret => Ok(false),
            TraceDispatch::Skip => {
                // The tracer gave this header up. Skip the next backedge
                // lookups, or the fast loop would leave at every iteration.
                if arm_backedge_skip {
                    let cap = self.jit.trace.skip_window_cap(&chunk, pc);
                    let skip = self.arm_rejected_backedge_skip(&chunk, pc, latch, cap);
                    self.jit.trace.credit_skipped_backedges(&chunk, pc, skip);
                }
                Ok(false)
            }
            // `None`: the trace refused the frame (entry types, captured
            // slots), so the interpreter runs the header.
            TraceDispatch::Enter(index) => Ok(self
                .enter_native_trace(frames, frame_index, index)?
                .unwrap_or(false)),
        }
    }
}
