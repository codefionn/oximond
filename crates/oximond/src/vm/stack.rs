//! The per-thread call stack: small frame records over one contiguous value stack.
//!
//! Every Lua activation owns the window `values[base..top]` of its thread's
//! value stack. A callee's window starts at the top of its caller's window,
//! so a call appends registers and a return truncates them. Nothing is
//! allocated per call once the stack has grown to its working depth.
//!
//! [`FrameMut`] and [`FrameRef`] pair one frame record with its register
//! window. Both dereference to [`Frame`], so frame helpers read
//! `frame.registers[..]` and `frame.chunk` exactly as they would on an
//! owning frame.

// These accessors run on every call and return; they are measured hot.
#![allow(clippy::inline_always)]

use crate::bytecode::{Chunk, Instruction, OpCode};
use crate::runtime::{
    CaptureCells, CellHandle, ClosureCells, ClosureHandle, Runtime, RuntimeError, StackAnchor,
    TableHandle, Value,
};
use crate::vm::frame::{FrameChunk, VARARG_PLAIN, VARARG_TRUNCATED};
use crate::vm::registers::{set_register, store_clone};
use crate::vm::{
    CALL_HINT_NUMERIC_PLAN, CALL_HINT_PLAIN, CALL_HINT_TABLE_PLAN, CALL_HINT_UNKNOWN,
    CALL_HINT_VARARG, CallReturn, Frame, FrameSite, MAX_LUA_CALL_FRAMES, OperatorMode,
    ReturnTarget, TailCallArguments, VmResult,
};

/// Rejects a Lua activation that would exceed the call-depth limit.
#[inline]
pub(crate) fn ensure_call_depth(frames: &CallStack) -> VmResult<()> {
    if frames.len() >= MAX_LUA_CALL_FRAMES {
        return Err(RuntimeError::message("stack overflow").into());
    }
    Ok(())
}
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Register operands are one byte wide, so an activation addresses at most
/// this many slots from its base. The fast loop views the top activation
/// as a window of exactly this size, which needs no bounds checks.
pub(crate) const REGISTER_WINDOW: usize = 256;

/// Converts a value-stack index to its frame-record width. The stack is
/// bounded by `MAX_LUA_CALL_FRAMES` windows of at most 256 registers.
#[inline(always)]
fn stack_slot(index: usize) -> u32 {
    u32::try_from(index).expect("value stack index fits u32")
}

/// A thread's activation records and their shared register storage.
///
/// `frames[..live]` are the activations. The fast interpreter loop retires
/// returning frames without dropping them: records above `live` and value
/// slots above the top activation's window are dead leftovers it overwrites
/// on the next call at that depth, which saves reference-count traffic when
/// the same function is called again. They are never read, rooted, or
/// exposed. Every general-purpose mutation drops retired records first and
/// keeps at most [`REGISTER_WINDOW`] dead slots, so the fast loop's
/// fixed-size window over the top activation rarely needs to grow the
/// stack. Mutations write windows in place rather than by length.
#[derive(Default)]
pub(crate) struct CallStack {
    frames: Vec<Frame>,
    live: usize,
    values: ValueStack,
    /// The captured variables that live in `values` right now, the slots
    /// they name ascending: PUC-Lua's open upvalues. A cell here reads and
    /// writes its stack slot until it is closed (see [`Self::pop_open`]),
    /// which every removal of the slots from the stack has to precede.
    open: Vec<OpenUpvalue>,
    /// One past the highest open slot, zero when none is open. Returns
    /// compare their window against it to know whether anything is to be
    /// closed.
    open_top: u32,
    /// Arguments in transit while a tail call rearranges the stack. Kept
    /// between calls, so a tail call allocates nothing once it has grown.
    scratch: Vec<Value>,
    /// A call site box a recycled record gave up, for the next tail call
    /// that names its callee: a chain of tail calls reuses one allocation
    /// instead of making and freeing a box per activation.
    spare_site: Option<Box<FrameSite>>,
    /// Whether this stack runs a host entry (`Vm::call_value`), whose
    /// bottom activation returns its values to the host. A coroutine's
    /// stack, whose bottom return finishes the thread, never sets it.
    host_entry: bool,
}

/// A captured variable that still lives in a register: the stack slot (an
/// absolute index into the thread's value stack) and the cell that stands for
/// it.
#[derive(Clone, Copy)]
pub(crate) struct OpenUpvalue {
    pub(crate) slot: u32,
    pub(crate) cell: CellHandle,
}

/// The value stack of one thread. It behaves as a slice of values and grows
/// like a vector; the base address of its buffer is published to the cells
/// that are open on it (see [`StackAnchor`]).
#[derive(Default)]
pub(crate) struct ValueStack {
    values: Vec<Value>,
    /// Created by the first capture. Boxed so that its address stays put
    /// while the stack moves between owners.
    anchor: Option<Box<StackAnchor>>,
}

impl ValueStack {
    /// Grows the stack to `end` slots, padding with nil. The buffer may
    /// move, so the anchor is told where it is now.
    #[cold]
    #[inline(never)]
    pub(crate) fn grow(&mut self, end: usize) {
        self.values.resize(end, Value::Nil);
        self.publish();
    }

    #[inline(always)]
    pub(crate) fn truncate(&mut self, end: usize) {
        self.values.truncate(end);
    }

    /// Moves the slots from `at` on into a stack of their own. Nothing may
    /// be open there: the new stack's buffer has another address.
    pub(crate) fn split_off(&mut self, at: usize) -> Self {
        Self {
            values: self.values.split_off(at),
            anchor: None,
        }
    }

    /// The anchor cells name this stack by, made on first use.
    pub(crate) fn anchor(&mut self) -> std::ptr::NonNull<StackAnchor> {
        let base = self.values.as_mut_ptr();
        let anchor = self.anchor.get_or_insert_with(Default::default);
        anchor.set_base(base);
        std::ptr::NonNull::from(&**anchor)
    }

    #[inline(always)]
    fn publish(&mut self) {
        if let Some(anchor) = &self.anchor {
            // The pointer is only ever used for the slots below `len`.
            anchor.set_base(self.values.as_mut_ptr());
        }
    }
}

impl Drop for CallStack {
    fn drop(&mut self) {
        if !self.open.is_empty() {
            // Cells that were left open still name this buffer and anchor.
            // Leaking both keeps those reads inside live memory. The owner
            // is supposed to have closed them first (`Vm::close_upvalues`),
            // so this is a safety net, not a code path.
            std::mem::forget(std::mem::take(&mut self.values));
        }
    }
}

impl std::ops::Deref for ValueStack {
    type Target = [Value];
    #[inline(always)]
    fn deref(&self) -> &[Value] {
        &self.values
    }
}

impl std::ops::DerefMut for ValueStack {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Value] {
        &mut self.values
    }
}

impl From<Vec<Value>> for ValueStack {
    fn from(values: Vec<Value>) -> Self {
        Self {
            values,
            anchor: None,
        }
    }
}

/// What a plain call of the fast loop did.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlainCall {
    Pushed,
    /// Nothing changed: the general routine decides the call.
    Declined,
    /// Nothing changed: the call is one only the dispatch loop makes.
    Leave,
    /// Nothing changed: the callee is better called through the general
    /// routine, which evaluates its inline plan.
    Plan,
}

/// What [`CallStack::push_closure_call`] did: the activation it pushed with
/// what the fast loop needs to run it, or what a [`PlainCall`] says.
///
/// The loop binds the new activation from these, which the push held in
/// registers. Reading them back from the stack loaded what the push had just
/// stored (the live count, the record's `pc`, `base` and chunk word), and each
/// of those loads waited for its store to forward.
#[derive(Clone, Copy)]
pub(crate) enum Entered {
    /// A plain activation, pushed: its record and the first register of its
    /// window. It starts at `pc` zero.
    Fast {
        frame: *mut Frame,
        window: *mut Value,
    },
    Other(PlainCall),
}

#[cfg(test)]
impl Entered {
    pub(crate) fn plain(self) -> PlainCall {
        match self {
            Self::Fast { .. } => PlainCall::Pushed,
            Self::Other(call) => call,
        }
    }
}

/// What a tail call of the fast loop did.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailStep {
    /// The running activation became the callee, and its window stayed
    /// where it was: the loop can go on in it with only the code, captures
    /// and environment bound again.
    Replaced,
    /// The activation became the callee, with a window elsewhere.
    Moved,
    /// Nothing changed: the callee has not been classified yet.
    Unclassified,
    /// Nothing changed: the general routine decides the call.
    Declined,
}

/// A return target that a pushed activation record stores.
trait ActivationCaller {
    /// The target a freshly built record starts with.
    fn into_caller(self) -> Option<ReturnTarget>;
    /// Stores the target in a reused record's slot.
    fn store(self, slot: &mut Option<ReturnTarget>);
}

impl ActivationCaller for CallReturn {
    #[inline(always)]
    fn into_caller(self) -> Option<ReturnTarget> {
        Some(ReturnTarget::Direct(self))
    }

    // A retired record usually already holds a direct caller, so only its
    // target is overwritten and the drop glue is skipped.
    #[inline(always)]
    fn store(self, slot: &mut Option<ReturnTarget>) {
        if let Some(ReturnTarget::Direct(target)) = slot {
            *target = self;
        } else {
            *slot = Some(ReturnTarget::Direct(self));
        }
    }
}

impl ActivationCaller for ReturnTarget {
    #[inline(always)]
    fn into_caller(self) -> Option<ReturnTarget> {
        Some(self)
    }

    #[inline(always)]
    fn store(self, slot: &mut Option<ReturnTarget>) {
        *slot = Some(self);
    }
}

/// The caller of the bottom activation of a host entry: none. Its return
/// ends the run and hands the values to the host.
struct HostEntry;

impl ActivationCaller for HostEntry {
    #[inline(always)]
    fn into_caller(self) -> Option<ReturnTarget> {
        None
    }

    #[inline(always)]
    fn store(self, slot: &mut Option<ReturnTarget>) {
        if slot.is_some() {
            *slot = None;
        }
    }
}

/// Where a new activation's parameters come from.
pub(crate) enum ActivationArguments {
    /// `count` stack slots from absolute slot `start`, below the new window.
    Stack {
        start: usize,
        count: usize,
    },
    One(Value),
    /// Owned arguments, as a call through `dispatch_call_value` holds them.
    /// The first `parameter_count` fill the parameters; the rest are dropped.
    Values(Vec<Value>),
    /// Two operands, as a binary metamethod receives them.
    Pair(Value, Value),
    /// Three operands, as a `__newindex` metamethod receives them.
    Triple(Value, Value, Value),
    None,
}

/// Where the key operand of an index metamethod call comes from: a register
/// of the caller, or a string constant of its chunk.
// Two pointer-sized words, passed by value through the hot handler path.
#[derive(Clone, Copy)]
pub(crate) enum HandlerKey<'a> {
    Register(*const Value),
    Constant(&'a crate::string::LuaString),
}

/// A frame record detached from any stack, together with its registers.
///
/// Frames are built this way before they are pushed, and frames removed
/// by error unwinding keep their registers here until their close
/// handlers have run.
pub(crate) struct OwnedFrame {
    pub(crate) frame: Frame,
    pub(crate) registers: Vec<Value>,
    /// The arguments past the parameters of a vararg function. Pushing the
    /// frame moves them onto the value stack, below its registers.
    pub(crate) varargs: Vec<Value>,
}

/// A mutable frame record and its register window.
pub(crate) struct FrameMut<'a> {
    pub(crate) frame: &'a mut Frame,
    pub(crate) registers: &'a mut [Value],
    /// The arguments past the parameters of a vararg function.
    pub(crate) varargs: &'a mut [Value],
}

/// A shared frame record and its register window.
#[derive(Clone, Copy)]
pub(crate) struct FrameRef<'a> {
    pub(crate) frame: &'a Frame,
    pub(crate) registers: &'a [Value],
    /// The arguments past the parameters of a vararg function.
    pub(crate) varargs: &'a [Value],
}

impl Deref for FrameMut<'_> {
    type Target = Frame;
    #[inline]
    fn deref(&self) -> &Frame {
        self.frame
    }
}

impl DerefMut for FrameMut<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Frame {
        self.frame
    }
}

impl Deref for FrameRef<'_> {
    type Target = Frame;
    #[inline]
    fn deref(&self) -> &Frame {
        self.frame
    }
}

/// Anything that can lend a shared frame view.
pub(crate) trait FrameView {
    fn frame_ref(&self) -> FrameRef<'_>;
}

impl FrameView for FrameMut<'_> {
    #[inline]
    fn frame_ref(&self) -> FrameRef<'_> {
        self.as_ref()
    }
}

impl FrameView for FrameRef<'_> {
    #[inline]
    fn frame_ref(&self) -> FrameRef<'_> {
        *self
    }
}

impl FrameView for OwnedFrame {
    fn frame_ref(&self) -> FrameRef<'_> {
        self.view_ref()
    }
}

impl FrameMut<'_> {
    #[inline(always)]
    pub(crate) fn as_ref(&self) -> FrameRef<'_> {
        FrameRef {
            frame: self.frame,
            registers: self.registers,
            varargs: self.varargs,
        }
    }
}

impl Deref for OwnedFrame {
    type Target = Frame;
    fn deref(&self) -> &Frame {
        &self.frame
    }
}

impl DerefMut for OwnedFrame {
    fn deref_mut(&mut self) -> &mut Frame {
        &mut self.frame
    }
}

impl OwnedFrame {
    pub(crate) fn view(&mut self) -> FrameMut<'_> {
        FrameMut {
            frame: &mut self.frame,
            registers: &mut self.registers,
            varargs: &mut self.varargs,
        }
    }

    pub(crate) fn view_ref(&self) -> FrameRef<'_> {
        FrameRef {
            frame: &self.frame,
            registers: &self.registers,
            varargs: &self.varargs,
        }
    }
}

impl Deref for CallStack {
    type Target = [Frame];
    #[inline(always)]
    fn deref(&self) -> &[Frame] {
        &self.frames[..self.live]
    }
}

impl DerefMut for CallStack {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Frame] {
        &mut self.frames[..self.live]
    }
}

impl CallStack {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Marks this stack as running a host entry (see [`Self::host_entry`]).
    pub(crate) fn set_host_entry(&mut self) {
        self.host_entry = true;
    }

    /// Whether a return from the bottom activation, which has no caller,
    /// hands its values to the host that called into the VM.
    #[inline(always)]
    pub(crate) fn host_entry(&self) -> bool {
        self.host_entry
    }

    pub(crate) fn with_frame(frame: OwnedFrame) -> Self {
        let mut stack = Self::new();
        stack.push(frame);
        stack
    }

    /// One past the top activation's last register.
    #[inline(always)]
    fn live_top(&self) -> usize {
        match self.live {
            0 => 0,
            live => self.frames[live - 1].top as usize,
        }
    }

    /// Drops retired frame records and dead value slots beyond one register
    /// window, restoring `frames.len() == live` and
    /// `values.len() <= live_top() + REGISTER_WINDOW`.
    #[inline(always)]
    fn drop_retired(&mut self) {
        if self.frames.len() != self.live || self.values.len() > self.live_top() + REGISTER_WINDOW {
            self.drop_retired_slow();
        }
    }

    #[cold]
    #[inline(never)]
    fn drop_retired_slow(&mut self) {
        let top = self.live_top();
        self.frames.truncate(self.live);
        self.values.truncate(top + REGISTER_WINDOW);
    }

    /// This stack ready for another host entry, once every activation on
    /// it has returned: retired records dropped and at most
    /// [`REGISTER_WINDOW`] dead slots kept, the state a stack is in between
    /// two calls at its bottom. `None` while an activation is live or a
    /// captured variable still names a slot, which only an unfinished run
    /// leaves.
    pub(crate) fn recycled(mut self) -> Option<Self> {
        if self.live != 0 || !self.open.is_empty() {
            return None;
        }
        self.drop_retired_slow();
        self.open_top = 0;
        Some(self)
    }

    /// Makes `values[..end]` addressable.
    #[inline(always)]
    fn reserve_slots(&mut self, end: usize) {
        if self.values.len() < end {
            self.values.grow(end);
        }
    }

    /// Every live register of every frame, bottom to top.
    #[inline(always)]
    pub(crate) fn values(&self) -> &[Value] {
        &self.values[..self.live_top()]
    }

    /// One past the highest open slot, zero when no captured variable lives
    /// in a register. Anything at or above this slot index needs no closing.
    #[inline(always)]
    pub(crate) fn open_top(&self) -> u32 {
        self.open_top
    }

    /// Every open upvalue, the lowest slot first.
    pub(crate) fn open_upvalues(&self) -> &[OpenUpvalue] {
        &self.open
    }

    /// The open upvalues whose slots lie in `first..end`.
    pub(crate) fn open_in(&self, first: u32, end: u32) -> &[OpenUpvalue] {
        let start = self.open.partition_point(|open| open.slot < first);
        let stop = self.open.partition_point(|open| open.slot < end);
        &self.open[start..stop.max(start)]
    }

    /// The cell standing for absolute stack slot `slot`, if one is open.
    pub(crate) fn open_cell_at(&self, slot: u32) -> Option<CellHandle> {
        // The newest capture is usually the last entry.
        if let Some(last) = self.open.last() {
            if last.slot == slot {
                return Some(last.cell);
            }
            if last.slot < slot {
                return None;
            }
        }
        let at = self
            .open
            .binary_search_by_key(&slot, |open| open.slot)
            .ok()?;
        Some(self.open[at].cell)
    }

    /// Records `cell` as standing for stack slot `slot`.
    pub(crate) fn push_open(&mut self, slot: u32, cell: CellHandle) {
        if self.open_top <= slot {
            // Above every open slot: the usual order of captures.
            self.open.push(OpenUpvalue { slot, cell });
            self.open_top = slot + 1;
            return;
        }
        let at = self.open.partition_point(|open| open.slot < slot);
        debug_assert!(self.open.get(at).is_none_or(|open| open.slot != slot));
        self.open.insert(at, OpenUpvalue { slot, cell });
    }

    /// Removes and returns the highest open upvalue at or above `level`,
    /// for the owner to close.
    #[inline(always)]
    pub(crate) fn pop_open(&mut self, level: u32) -> Option<OpenUpvalue> {
        let top = *self.open.last()?;
        if top.slot < level {
            return None;
        }
        self.open.pop();
        self.open_top = self.open.last().map_or(0, |open| open.slot + 1);
        Some(top)
    }

    /// Removes the open upvalue of exactly slot `slot` and returns it.
    pub(crate) fn remove_open(&mut self, slot: u32) -> Option<OpenUpvalue> {
        let at = self
            .open
            .binary_search_by_key(&slot, |open| open.slot)
            .ok()?;
        let removed = self.open.remove(at);
        self.open_top = self.open.last().map_or(0, |open| open.slot + 1);
        Some(removed)
    }

    /// The first slot of the top frame's registers.
    #[inline(always)]
    pub(crate) fn top_base(&self) -> u32 {
        self.frames[self.live - 1].base
    }

    /// The anchor that cells name this stack by.
    pub(crate) fn anchor(&mut self) -> std::ptr::NonNull<StackAnchor> {
        self.values.anchor()
    }

    /// Appends a new Lua activation unless the thread already holds
    /// `MAX_LUA_CALL_FRAMES`. Library callbacks and metamethods can recurse
    /// through native code without passing a call instruction, so every such
    /// entry needs this check.
    pub(crate) fn push_within_limit(&mut self, owned: OwnedFrame) -> VmResult<()> {
        ensure_call_depth(self)?;
        self.push(owned);
        Ok(())
    }

    /// Appends a detached frame, moving its registers onto the stack.
    pub(crate) fn push(&mut self, owned: OwnedFrame) {
        self.drop_retired();
        let OwnedFrame {
            mut frame,
            registers,
            varargs,
        } = owned;
        let start = self.live_top();
        let base = start + varargs.len();
        let top = base + registers.len();
        self.reserve_slots(top);
        let (extra, window) = self.values[start..top].split_at_mut(varargs.len());
        for (slot, value) in extra.iter_mut().zip(varargs) {
            *slot = value;
        }
        for (slot, value) in window.iter_mut().zip(registers) {
            *slot = value;
        }
        // A write of the count takes the plain mark off, here and below.
        frame.vararg_word = stack_slot(extra.len());
        frame.base = stack_slot(base);
        frame.top = stack_slot(top);
        self.frames.push(frame);
        self.live += 1;
    }

    /// Pushes `frame` over a window of `size` registers whose first
    /// `parameters` slots copy the stack slots starting at `arguments`
    /// (absolute), padding missing arguments with nil. `supplied` is the
    /// number of argument slots actually available.
    #[inline(always)]
    pub(crate) fn push_with_stack_arguments(
        &mut self,
        mut frame: Frame,
        size: usize,
        arguments: usize,
        supplied: usize,
        parameters: usize,
    ) {
        self.drop_retired();
        let start = self.live_top();
        // A vararg function keeps the arguments past its parameters on the
        // stack just below its window.
        let extra = if frame.chunk.is_vararg {
            supplied.saturating_sub(parameters)
        } else {
            0
        };
        let base = start + extra;
        let copied = supplied.min(parameters).min(size);
        self.reserve_slots(base + size);
        let (below, rest) = self.values.split_at_mut(start);
        let (extras, window) = rest.split_at_mut(extra);
        let window = &mut window[..size];
        #[cfg(feature = "op-stats")]
        {
            crate::vm::op_stats::stack_arguments(&below[arguments..arguments + copied]);
            crate::vm::op_stats::activation(copied, size - copied);
        }
        if extra != 0 {
            for (slot, argument) in extras
                .iter_mut()
                .zip(&below[arguments + parameters..arguments + supplied])
            {
                store_clone(slot, argument);
            }
        }
        for (slot, argument) in window.iter_mut().zip(&below[arguments..arguments + copied]) {
            store_clone(slot, argument);
        }
        for slot in &mut window[copied..] {
            set_register(slot, Value::Nil);
        }
        frame.vararg_word = stack_slot(extra);
        frame.base = stack_slot(base);
        frame.top = stack_slot(base + size);
        self.frames.push(frame);
        self.live += 1;
    }

    /// Pushes a plain Lua activation: a fixed-parameter `chunk` whose first
    /// parameters come from `arguments`.
    ///
    /// A retired record at this depth is overwritten in place. The record
    /// borrows `chunk` and `cells` from `closure` instead of counting them,
    /// so a call and its return do no atomic reference-count work.
    /// Returns the new frame's index.
    ///
    /// # Safety
    ///
    /// `chunk` and `cells` must be the chunk and cells of the live closure
    /// `closure`, as `Runtime::closure_call_parts` returns them. The record
    /// roots `closure` for as long as it is live (`Vm::root_frame`), which
    /// keeps both allocations alive.
    #[allow(clippy::too_many_arguments, unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn push_activation(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        closure: ClosureHandle,
        caller: CallReturn,
        arguments: ActivationArguments,
    ) -> usize {
        // SAFETY: forwarded from this function's contract.
        unsafe { self.push_activation_as(chunk, cells, environment, closure, caller, arguments) }
    }

    /// The body of [`Self::push_activation`] and [`Self::push_activation_to`],
    /// generic over the caller so each writes its own return target once.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`].
    #[allow(clippy::too_many_arguments, unsafe_code)]
    #[inline(always)]
    unsafe fn push_activation_as<C: ActivationCaller>(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        closure: ClosureHandle,
        caller: C,
        arguments: ActivationArguments,
    ) -> usize {
        let base = self.live_top();
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let end = base + size;
        self.reserve_slots(end);
        let (below, window) = self.values.split_at_mut(base);
        let window = &mut window[..size];
        let copied = match arguments {
            ActivationArguments::Stack { start, count } => {
                let copied = count.min(parameters);
                #[cfg(feature = "op-stats")]
                crate::vm::op_stats::stack_arguments(&below[start..start + copied]);
                for (slot, argument) in window.iter_mut().zip(&below[start..start + copied]) {
                    store_clone(slot, argument);
                }
                copied
            }
            ActivationArguments::One(value) if parameters != 0 => {
                #[cfg(feature = "op-stats")]
                crate::vm::op_stats::moved_argument(&value);
                set_register(&mut window[0], value);
                1
            }
            ActivationArguments::Values(values) => {
                let copied = values.len().min(parameters);
                for (slot, value) in window[..copied].iter_mut().zip(values) {
                    set_register(slot, value);
                }
                copied
            }
            ActivationArguments::Pair(left, right) if parameters != 0 => {
                #[cfg(feature = "op-stats")]
                crate::vm::op_stats::moved_argument(&left);
                set_register(&mut window[0], left);
                if parameters == 1 {
                    1
                } else {
                    #[cfg(feature = "op-stats")]
                    crate::vm::op_stats::moved_argument(&right);
                    set_register(&mut window[1], right);
                    2
                }
            }
            ActivationArguments::Triple(first, second, third) => {
                let copied = parameters.min(3);
                for (slot, value) in window[..copied].iter_mut().zip([first, second, third]) {
                    set_register(slot, value);
                }
                copied
            }
            ActivationArguments::One(_)
            | ActivationArguments::Pair(..)
            | ActivationArguments::None => 0,
        };
        #[cfg(feature = "op-stats")]
        crate::vm::op_stats::activation(copied, size - copied);
        clear_registers(&mut window[copied..]);
        let index = self.live;
        if let Some(record) = self.frames.get_mut(index) {
            // A retired record usually already borrows this very chunk, so
            // the common recursive or repeated call writes nothing here.
            if record.chunk.tag() != FrameChunk::borrowed_tag(chunk) {
                // SAFETY: forwarded from this function's contract.
                record.chunk = unsafe { FrameChunk::borrowed(chunk) };
            }
            record.closure = Some(closure);
            record.environment = environment;
            record.shape = crate::runtime::NO_SHAPE;
            record.base = stack_slot(base);
            record.top = stack_slot(end);
            // SAFETY: forwarded from this function's contract.
            record.upvalues = unsafe { CaptureCells::borrow_shared(cells) };
            record.pc = 0;
            record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
            record.tail_calls = 0;
            record.vararg_word = 0;
            // Retired records come from plain returns, so these usually
            // already hold their reset values; skip the drop glue then.
            caller.store(&mut record.caller);
            if record.call_site.is_some() {
                stash_site(&mut record.call_site, &mut self.spare_site);
            }
            if record.extras.is_some() {
                record.extras = None;
            }
        } else {
            let mut frame = Frame::activation(
                // SAFETY: forwarded from this function's contract.
                unsafe { FrameChunk::borrowed(chunk) },
                unsafe { CaptureCells::borrow_shared(cells) },
                caller.into_caller(),
                None,
                environment,
                closure,
            );
            frame.base = stack_slot(base);
            frame.top = stack_slot(end);
            self.frames.push(frame);
        }
        self.live = index + 1;
        index
    }

    /// [`Self::push_activation`] of a fixed-parameter `chunk` whose arguments
    /// are `count` stack slots from `start`, below the new window. A retired
    /// record at the new depth takes the activation through
    /// [`Self::push_plain_call`], which copies the arguments without the
    /// general clone, clears only the registers the chunk can read before it
    /// writes them and rewrites of the record only what changed. Without
    /// such a record the general push builds one. Returns the new frame's
    /// index, whose caller target is `ReturnTarget::Direct(caller)`.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`], with `shape` the shape of `closure`
    /// that its parts were read at.
    #[allow(clippy::too_many_arguments, unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn push_stack_activation(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller: CallReturn,
        start: usize,
        count: usize,
    ) -> usize {
        let live = self.live;
        let base = self.live_top();
        if live == 0
            || live >= self.frames.len()
            || live >= MAX_LUA_CALL_FRAMES
            || start + count > base
        {
            // SAFETY: forwarded from this function's contract.
            return unsafe {
                self.push_activation(
                    chunk,
                    cells,
                    environment,
                    closure,
                    caller,
                    ActivationArguments::Stack { start, count },
                )
            };
        }
        let window = self.values.as_mut_ptr();
        // SAFETY: the contract. The calling window is taken as the whole
        // stack below `base`, the live top, which holds the arguments
        // (checked above), and a record exists at depth `live` under the
        // depth limit.
        unsafe {
            self.push_plain_call(
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller,
                window,
                base,
                base,
                start,
                count,
            );
        }
        live
    }

    /// [`Self::push_activation`] for the fast loop's plain call, with its
    /// arguments in the calling window just below the new window, which
    /// begins at `base`, the top of the calling window.
    ///
    /// The caller has checked that `count` arguments are in the calling
    /// window from register `first`, that a record exists at the new depth
    /// (`live < frames.len()`) and the depth limit holds, and has to leave
    /// the caller's resume position stored.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`], `first + count <= calling_window`
    /// where the calling window starts at `window` and ends at `base`, the top
    /// of the live top frame, and `live < frames.len()`.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        unsafe_code
    )]
    #[inline(never)]
    unsafe fn push_plain_call(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller: CallReturn,
        window: *mut Value,
        calling_window: usize,
        base: usize,
        first: usize,
        count: usize,
    ) -> (*mut Frame, *mut Value) {
        debug_assert_eq!(base, self.live_top());
        debug_assert!(first + count <= base);
        let size = usize::from(chunk.register_count);
        let end = base + size;
        // The caller checked the depth limit, which keeps every window
        // inside the first `MAX_LUA_CALL_FRAMES * 256` slots: `u32` holds them.
        debug_assert!(u32::try_from(end).is_ok());
        // The fast loop views the whole window of the new top frame, so the
        // stack reaches `REGISTER_WINDOW` slots past its base.
        let mut window = window;
        if self.values.len() < base + REGISTER_WINDOW {
            // SAFETY: `window` points into the stack (the contract).
            let offset = unsafe { window.offset_from(self.values.as_ptr()) };
            self.values.grow(base + REGISTER_WINDOW);
            // The stack may have moved. SAFETY: the offset is the same and
            // the stack only grew.
            window = unsafe { self.values.as_mut_ptr().offset(offset) };
        }
        // Registers from `entry` up are written before any read (see
        // `Chunk::entry_register_count`), so they keep their dead contents:
        // no instruction, root scan or debug query looks at them first. The
        // parameters the caller left out start nil, as do the registers the
        // chunk can read before writing.
        let entry = chunk.entry_register_count();
        let copied = count.min(usize::from(chunk.parameter_count));
        debug_assert!(copied <= entry && entry <= size);
        // SAFETY: the stack reaches `base + REGISTER_WINDOW`, past `end`;
        // `window` is the calling window, which ends at `base`, so the new
        // window lies `base - window_base` slots on. The arguments lie in the
        // calling window (the contract), `entry <= size`, and the loops stay
        // inside `first..first + copied` of it and `0..entry` of the new one.
        unsafe {
            let (to, from) = (window.add(calling_window), window.add(first).cast_const());
            // One argument or two are the common call: no loop around them.
            if copied != 0 {
                copy_value(to, from);
                if copied > 1 {
                    copy_value(to.add(1), from.add(1));
                    for index in 2..copied {
                        copy_value(to.add(index), from.add(index));
                    }
                }
            }
            for index in copied..entry {
                clear_value(to.add(index));
            }
        }
        let index = self.live;
        debug_assert!(index < self.frames.len() && index < MAX_LUA_CALL_FRAMES);
        // SAFETY: the contract: a retired record exists at this depth.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        let record_ptr = std::ptr::from_mut(record);
        // A retired record usually has no interior state and no foreign
        // return target, which leaves its caller target and the record's
        // numbers to overwrite. Any other record is rewritten out of line,
        // which keeps the drops it needs (and their calls) off the hot path.
        let target = if holds_named_site(record) {
            None
        } else {
            recycled_target(record)
        };
        if let Some(target) = target {
            // It usually already runs this very closure at the shape it has
            // now, so its environment and captures are right too (see
            // `Frame::shape`), and so is its chunk: a closure's chunk never
            // changes, and every writer of a record's closure writes its
            // chunk with it. The common recursive or repeated call rewrites
            // none of them. Any other record is rewritten out of line, which
            // keeps the drops it needs (and their calls) off the hot path.
            //
            // The closure is fixed last, with nothing live after it, so the
            // call that does it saves none of this path's values.
            // SAFETY: `target` points into `record.caller`, which nothing
            // above replaced.
            unsafe { *target = caller };
            record.base = base as u32;
            record.top = end as u32;
            record.pc = 0;
            record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
            // Two four-byte stores, not the one eight-byte store the compiler
            // merges them into: the return reads `vararg_count` right away,
            // and a load from the upper half of a wider store does not forward.
            // SAFETY: both are fields of `record`.
            unsafe {
                std::ptr::write_volatile(&raw mut record.tail_calls, 0);
                // No plain mark: the hot push sets it, which checks what
                // it promises.
                std::ptr::write_volatile(&raw mut record.vararg_word, 0);
            }
            self.live = index + 1;
            if record.closure != Some(closure) || record.shape != shape {
                // SAFETY: the contract, and `recycled_target` said the record
                // owns neither its chunk nor its captures.
                unsafe {
                    overwrite_closure_cold(record, chunk, cells, environment, shape, closure);
                }
            }
            debug_assert_eq!(record.chunk.address(), Arc::as_ptr(chunk));
        } else {
            // SAFETY: forwarded from this function's contract.
            unsafe {
                rewrite_record(
                    record,
                    chunk,
                    cells,
                    environment,
                    shape,
                    closure,
                    caller,
                    base,
                    end,
                    0,
                    0,
                    &mut self.spare_site,
                );
            }
            self.live = index + 1;
        }
        // The new top record and its window, from what this function holds:
        // reading them back from the stack would load what it just stored.
        // SAFETY: the window starts `calling_window` slots on, inside the
        // stack (see above).
        (record_ptr, unsafe { window.add(calling_window) })
    }

    /// [`Self::push_plain_call`] for the call that needs nothing it would
    /// call a function for: the stack reaches the new window, no string is
    /// copied or overwritten (`COUNTED` takes their counts in line instead,
    /// and only freeing the last reference of one calls), the record at the
    /// new depth is retired, owns no
    /// chunk or captures and already runs this closure at this shape, and the
    /// chunk's entry count is known. Returns the new top record and its
    /// window, or `None` with the stack unchanged (apart
    /// from dead registers above the live windows), which the general routine
    /// then decides.
    ///
    /// It calls nothing, which keeps the call path's values in registers: a
    /// call in the middle spilled them and reloaded them a few instructions
    /// on, and each reload waited for its store.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_plain_call`], `live` is the number of live
    /// frames, `record` the retired record at index `live`, and `cells` and
    /// `environment` are `closure`'s own.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        unsafe_code
    )]
    #[inline(always)]
    unsafe fn push_plain_call_hot<const COUNTED: bool>(
        &mut self,
        runtime: &Runtime,
        chunk: &Arc<Chunk>,
        shape: u32,
        closure: ClosureHandle,
        caller: CallReturn,
        record: *mut Frame,
        window: *mut Value,
        calling_window: usize,
        base: usize,
        first: usize,
        count: usize,
        live: usize,
    ) -> Option<(*mut Frame, *mut Value)> {
        debug_assert_eq!(base, self.live_top());
        debug_assert_eq!(live, self.live);
        debug_assert!(first + count <= base);
        let size = usize::from(chunk.register_count);
        let end = base + size;
        debug_assert!(u32::try_from(end).is_ok());
        if self.values.len() < base + REGISTER_WINDOW {
            return None;
        }
        let entry = chunk.entry_register_count_cached()?;
        debug_assert!(std::ptr::eq(record, &raw const self.frames[live]));
        let record_ptr = record;
        // SAFETY: the contract: `record` is the retired record at this depth.
        let record = unsafe { &mut *record };
        // A record with the plain mark is a plain activation as the last
        // call here left it: one compare instead of the tests of its site,
        // its interior state, its chunk, its captures and its target.
        let target = if record.plain_marked() {
            debug_assert!(record.plain_mark_holds());
            // SAFETY: the mark says the target is direct.
            let Some(ReturnTarget::Direct(target)) = &mut record.caller else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            std::ptr::from_mut(target)
        } else {
            if holds_named_site(record) {
                return None;
            }
            recycled_target(record)?
        };
        // The record at the new depth runs this closure already, when the
        // same call repeats or recurses. If it ran another one, as it does
        // when callees take turns at one depth, it is pointed at this one
        // here, with a few stores and no call: the record owns neither its
        // chunk nor its captures (`recycled_target` said so), so nothing
        // needs a release. The closure's captures and environment are read
        // here, where they are needed: held from the caller's lookup, they
        // took two registers from the call that repeats. A push that
        // declines below leaves the retired record pointing at `closure`,
        // which the general push then finds.
        if record.closure != Some(closure) || record.shape != shape {
            // SAFETY: the contract: `closure` is live, so the parts are its
            // own, and the record owns neither its chunk nor its captures.
            unsafe {
                let parts = runtime.closure_call_parts_live(closure);
                overwrite_closure(
                    record,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
        }
        let copied = count.min(usize::from(chunk.parameter_count));
        debug_assert!(copied <= entry && entry <= size);
        // SAFETY: as for `push_plain_call`: the stack reaches `base +
        // REGISTER_WINDOW`, past `end`, the arguments lie in the calling
        // window, and the loops stay inside `first..first + copied` of it and
        // `0..entry` of the new one. Strings take their counts in line:
        // registration code passes string keys and values to every call it
        // makes, and the general routine cost such a call twice the push.
        unsafe {
            let (to, from) = (window.add(calling_window), window.add(first).cast_const());
            if COUNTED {
                for index in 0..copied {
                    copy_value_counted(to.add(index), from.add(index));
                }
                for index in copied..entry {
                    clear_value_counted(to.add(index));
                }
            } else {
                if copied != 0 {
                    if !copy_value_clean(to, from) {
                        return None;
                    }
                    if copied > 1 {
                        if !copy_value_clean(to.add(1), from.add(1)) {
                            return None;
                        }
                        for index in 2..copied {
                            if !copy_value_clean(to.add(index), from.add(index)) {
                                return None;
                            }
                        }
                    }
                }
                for index in copied..entry {
                    if !clear_value_clean(to.add(index)) {
                        return None;
                    }
                }
            }
            // The record already runs this closure at this shape, so its
            // environment, captures and chunk are right too
            // (see `Frame::shape`), and so is its chunk: a closure's chunk
            // never changes, and every writer of a record's closure writes
            // its chunk with it.
            *target = caller;
            record.base = base as u32;
            record.top = end as u32;
            record.pc = 0;
            record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
            // Two four-byte stores, not the one eight-byte store the compiler
            // merges them into: the return reads `vararg_count` right away,
            // and a load from the upper half of a wider store does not
            // forward.
            std::ptr::write_volatile(&raw mut record.tail_calls, 0);
            // No varargs, no cut caller, and the plain mark: everything it
            // promises was tested or written above, so the next call at
            // this depth and this activation's return test the mark alone.
            std::ptr::write_volatile(&raw mut record.vararg_word, VARARG_PLAIN);
            debug_assert!(record.plain_mark_holds());
            self.live = live + 1;
            Some((record_ptr, window.add(calling_window)))
        }
    }

    /// The call into a record that carries the plain mark: the one the last
    /// plain call at the new depth left there. The mark stands for every
    /// test of the record that [`Self::push_plain_call_hot`] makes. A record
    /// that ran this very closure at the shape it has now, as in a loop
    /// around a call or a recursion, also stands for the callee's kind (a
    /// marked record runs a plain closure, and a closure's classification
    /// never changes once made), its chunk, its captures and its
    /// environment; a record that ran another closure is pointed at this
    /// one first. What is left is to copy the arguments, place the window
    /// and write where the callee returns to.
    ///
    /// `caller` is the running record, `record` the one after it, `window`
    /// the first register of the running window and `pc` the call's. Returns
    /// the first register of the new window, or null for a record without
    /// the mark, a callee that is not to be called plainly or whose entry
    /// count is not known yet, a string among the values it would copy or
    /// overwrite, and a stack that does not reach the new window:
    /// [`Self::push_closure_call_known`] then decides. A push that declines
    /// has changed nothing but dead registers above the live windows and
    /// the retired record, which is then a whole marked record of this
    /// closure.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_closure_call`].
    #[allow(clippy::too_many_arguments, unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn push_marked_call_at(
        &mut self,
        runtime: &Runtime,
        caller: *mut Frame,
        window: *mut Value,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
    ) -> (*mut Value, *const Chunk) {
        if self.live >= self.frames.len() {
            return (std::ptr::null_mut(), std::ptr::null());
        }
        // SAFETY: the contract, and a record exists at the live depth: the
        // one after the caller's.
        unsafe {
            self.push_marked_call(
                runtime,
                caller,
                caller.add(1),
                window,
                closure,
                target,
                results,
                first,
                count,
                pc,
            )
        }
    }

    /// [`Self::push_marked_call_at`] once a record is known to exist at the
    /// live depth.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_closure_call`], and `record` is the retired
    /// record at the live depth.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        unsafe_code
    )]
    #[inline(always)]
    unsafe fn push_marked_call(
        &mut self,
        runtime: &Runtime,
        caller: *mut Frame,
        record: *mut Frame,
        window: *mut Value,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
    ) -> (*mut Value, *const Chunk) {
        debug_assert!(std::ptr::eq(record, &raw const self.frames[self.live]));
        // SAFETY: the contract.
        unsafe {
            if !(*record).plain_marked() {
                return (std::ptr::null_mut(), std::ptr::null());
            }
            debug_assert!((*record).plain_mark_holds());
            let parts = runtime.closure_call_parts_live(closure);
            // A marked record has a closure. Slot and generation compare as
            // one word. When it ran another closure, as it does when
            // callees take turns at one depth, or this one at another
            // shape, it is pointed at this one, with a few stores and no
            // call: a marked record owns neither its chunk nor its
            // captures, so nothing needs a release.
            if (*record).closure.unwrap_unchecked().native_identity() != closure.native_identity()
                || (*record).shape != parts.shape
            {
                // The parts through `opaque`: the compiler otherwise reads
                // the hint and the environment ahead of the compare above,
                // on every call.
                let parts = (*super::fast_path::opaque(std::ptr::from_ref(runtime).cast_mut())
                    .cast_const())
                .closure_call_parts_live(closure);
                if parts.call_hint & CALL_HINT_PLAIN == 0
                    || parts.chunk.entry_register_count_cached().is_none()
                {
                    return (std::ptr::null_mut(), std::ptr::null());
                }
                overwrite_closure(
                    &mut *record,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
            let (base, top) = ((*caller).base as usize, (*caller).top as usize);
            debug_assert!(first + count <= top - base);
            // A marked record borrows its chunk, whose entry count the push
            // that set the mark found computed.
            let chunk: &Chunk = &*(*record).chunk.borrowed_address();
            let entry = chunk.entry_register_count_cached().unwrap_unchecked();
            // The record usually sits where its last activation sat, under
            // the same caller's window. The stack reached that window when
            // the record was marked there and has not shrunk under it since:
            // what truncates the stack drops the retired records first, and
            // keeps a window's worth past the live top.
            if (*record).base as usize != top {
                if self.values.len() < top + REGISTER_WINDOW {
                    return (std::ptr::null_mut(), std::ptr::null());
                }
                (*record).base = top as u32;
            }
            debug_assert!(self.values.len() >= top + REGISTER_WINDOW);
            // Written whatever it holds: a caller cut short by a vararg
            // callee and removed without its window restored would
            // otherwise hand a short window on.
            (*record).top = (top + usize::from(chunk.register_count)) as u32;
            let copied = count.min(usize::from(chunk.parameter_count));
            debug_assert!(copied <= entry && entry <= usize::from(chunk.register_count));
            let (to, from) = (window.add(top - base), window.add(first).cast_const());
            // The first arguments each have their own copy: a loop keeps
            // its counters in memory here, at more than twice the cost for
            // each argument.
            macro_rules! copy_first {
                ($index:literal $(, $rest:literal)*) => {
                    if copied > $index {
                        if !copy_value_clean(to.add($index), from.add($index)) {
                            return (std::ptr::null_mut(), std::ptr::null());
                        }
                        copy_first!($($rest),*);
                    }
                };
                () => {
                    for index in 4..copied {
                        if !copy_value_clean(to.add(index), from.add(index)) {
                            return (std::ptr::null_mut(), std::ptr::null());
                        }
                    }
                };
            }
            copy_first!(0, 1, 2, 3);
            for index in copied..entry {
                if !clear_value_clean(to.add(index)) {
                    return (std::ptr::null_mut(), std::ptr::null());
                }
            }
            (*caller).pc = pc + 1;
            (*caller).last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
            // The mark says the target is direct.
            let Some(ReturnTarget::Direct(direct)) = &mut (*record).caller else {
                std::hint::unreachable_unchecked()
            };
            *direct = CallReturn {
                base: target,
                results,
            };
            (*record).pc = 0;
            (*record).last_interpreted_pc = crate::vm::frame::LastPc::NONE;
            debug_assert!((*record).plain_mark_holds());
            self.live += 1;
            (to, std::ptr::from_ref(chunk))
        }
    }

    /// The general form of the plain call, for what
    /// [`Self::push_plain_call_hot`] declined: it takes the closure's parts
    /// again and pushes with the calls the general routine may make. Returns
    /// the new top record and its window, in two registers: a result the
    /// size of [`Entered`] would come back through memory, and the hot path
    /// would then build its own in the same place.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_plain_call`], with `closure` the callee and its
    /// parts plain.
    #[allow(clippy::too_many_arguments, unsafe_code)]
    #[cold]
    #[inline(never)]
    unsafe fn push_plain_call_cold(
        &mut self,
        runtime: &Runtime,
        closure: ClosureHandle,
        caller: CallReturn,
        window: *mut Value,
        calling_window: usize,
        base: usize,
        first: usize,
        count: usize,
    ) -> (*mut Frame, *mut Value) {
        let parts = runtime
            .closure_call_parts(closure)
            .expect("the callee was just looked up");
        // Strings among the arguments, or in the registers they land on,
        // are the common reason the hot push declined: registration code
        // passes string keys and values to every call. The hot push again,
        // with the counts taken in line, does the copies it left undone
        // (the ones it made wrote no string). Inside the loop, those counts
        // cost every dispatch a register.
        let live = self.live;
        // SAFETY: the contract: a retired record exists at the live depth.
        let record = unsafe { self.frames.as_mut_ptr().add(live) };
        // SAFETY: the contract.
        if let Some(pushed) = unsafe {
            self.push_plain_call_hot::<true>(
                runtime,
                parts.chunk,
                parts.shape,
                closure,
                caller,
                record,
                window,
                calling_window,
                base,
                first,
                count,
                live,
            )
        } {
            return pushed;
        }
        // SAFETY: the contract, and `parts` are `closure`'s own.
        unsafe {
            self.push_plain_call(
                parts.chunk,
                parts.cells,
                parts.environment,
                parts.shape,
                closure,
                caller,
                window,
                calling_window,
                base,
                first,
                count,
            )
        }
    }

    /// [`Self::push_plain_call`] for any chunk without a named-vararg table
    /// and with the arguments in two runs: `count` registers of the calling
    /// window from `first`, then the `forwarded` varargs of the calling frame
    /// (a call of `...`). The new window starts at `calling_base +
    /// calling_window`, the top of the calling window. A vararg chunk keeps
    /// the arguments past its parameters just below its window, so the call
    /// allocates nothing.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_plain_call`], with the calling window starting at
    /// `calling_base` and the `forwarded` varargs just below it.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        unsafe_code
    )]
    #[inline(never)]
    pub(crate) unsafe fn push_call_spread(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller: CallReturn,
        calling_base: usize,
        calling_window: usize,
        first: usize,
        count: usize,
        forwarded: usize,
    ) {
        let start = calling_base + calling_window;
        debug_assert_eq!(start, self.live_top());
        debug_assert!(first + count <= calling_window && forwarded <= calling_base);
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let total = count + forwarded;
        let extra = if chunk.is_vararg {
            total.saturating_sub(parameters)
        } else {
            0
        };
        let new_base = start + extra;
        let end = new_base + size;
        debug_assert!(u32::try_from(end).is_ok());
        if self.values.len() < new_base + REGISTER_WINDOW {
            self.values.grow(new_base + REGISTER_WINDOW);
        }
        let entry = chunk.entry_register_count();
        let copied = total.min(parameters);
        debug_assert!(copied <= entry && entry <= size);
        // SAFETY: the stack reaches `new_base + REGISTER_WINDOW`, past `end`.
        // The registers `first..first + count` lie in the calling window and
        // the varargs just below it (the contract), all below `start`. The
        // extra arguments go to `start..new_base` and the parameters to
        // `new_base..new_base + copied`, both at or above `start`, so they
        // never overlap a source.
        unsafe {
            let values = self.values.as_mut_ptr();
            let registers = values.add(calling_base + first).cast_const();
            let extras = values.add(start);
            let to = values.add(new_base);
            if forwarded == 0 {
                // The arguments are one run of registers: the parameters,
                // then the extra arguments.
                copy_values(to, registers, copied);
                copy_values(extras, registers.add(parameters), extra);
            } else {
                let varargs = values.add(calling_base - forwarded).cast_const();
                let source = |index: usize| {
                    if index < count {
                        registers.add(index)
                    } else {
                        varargs.add(index - count)
                    }
                };
                for index in 0..copied {
                    copy_value(to.add(index), source(index));
                }
                for index in 0..extra {
                    copy_value(extras.add(index), source(parameters + index));
                }
            }
            clear_values(to.add(copied), entry - copied);
        }
        let index = self.live;
        debug_assert!(index < self.frames.len() && index < MAX_LUA_CALL_FRAMES);
        // SAFETY: the contract: a retired record exists at this depth.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        // SAFETY: forwarded from this function's contract.
        unsafe {
            install_record(
                record,
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller,
                new_base,
                end,
                extra,
                0,
                &mut self.spare_site,
                &mut self.live,
                index,
            );
        }
    }

    /// [`Self::push_closure_call_known`] for a closure with no plain form: a
    /// vararg body gets an activation, whatever plan it has (with its
    /// varargs on the stack, a call and its return cost no more than the
    /// evaluation of a plan). Closures nobody has classified yet, and
    /// bodies with only a plan, are the general loop's.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_closure_call_known`], with `parts` the parts of
    /// `closure`.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[inline(never)]
    unsafe fn push_nonplain_call(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        call_hint: u8,
        caller: *mut Frame,
        closure: ClosureHandle,
        caller_return: CallReturn,
        first: usize,
        count: usize,
        pc: usize,
    ) -> PlainCall {
        if call_hint & CALL_HINT_VARARG == 0 {
            // A plan of a vararg body is the general routine's to evaluate.
            return if call_hint & (CALL_HINT_NUMERIC_PLAN | CALL_HINT_TABLE_PLAN) == 0 {
                PlainCall::Leave
            } else {
                PlainCall::Plan
            };
        }
        // SAFETY: the contract.
        let frame = unsafe { &mut *caller };
        frame.pc = pc + 1;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        let (base, top) = (frame.base as usize, frame.top as usize);
        // SAFETY: the pushed record roots `closure`. The arguments lie in the
        // caller's window, which ends at `top`, and `parts.chunk` is a vararg
        // body (the hint says so). A record exists at the new depth.
        unsafe {
            self.push_vararg_call(
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller_return,
                top,
                base + first,
                count,
            );
        }
        PlainCall::Pushed
    }

    /// Pushes the activation of a vararg closure called by a `Call` with
    /// `count` arguments, more than its parameters, from the registers just
    /// above the callee's (`first` is the first of them in the caller's
    /// window, which `caller` records). The arguments past the parameters
    /// stay where they are and become the callee's varargs; the caller's
    /// window is cut short to end just before them, and the callee's window
    /// starts right after them. Only the parameters are copied.
    ///
    /// The registers the callee's window takes hold nothing the caller reads
    /// after the call: they are the call's dead arguments and dead
    /// temporaries. The activation carries [`VARARG_TRUNCATED`], and
    /// whatever removes it gives the caller its full window back
    /// ([`Self::restore_caller_top`]).
    ///
    /// # Safety
    ///
    /// As for [`Self::push_vararg_call`], with `parameters < count`;
    /// `caller` is the record of the live top frame, which called with
    /// `first + count` within its window, and the call is a `Call` of the
    /// register `caller_return.base`, after which the caller reads no
    /// register from there up but the `caller_return.results` ones the call
    /// writes (the chunk's live-register count at the next instruction is at
    /// most `base + results`).
    #[allow(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::too_many_arguments
    )]
    #[inline(never)]
    unsafe fn push_vararg_in_place_general(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller_return: CallReturn,
        caller: *mut Frame,
        first: usize,
        count: usize,
    ) {
        debug_assert!(chunk.is_vararg && chunk.arg_register.is_none());
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        debug_assert!(parameters < count);
        // SAFETY: the contract.
        let parent = unsafe { &mut *caller };
        let first = parent.base as usize + first;
        // The varargs are `first + parameters..first + count`.
        let cut = first + parameters;
        let base = first + count;
        let end = base + size;
        debug_assert!(u32::try_from(end).is_ok());
        if self.values.len() < base + REGISTER_WINDOW {
            self.values.grow(base + REGISTER_WINDOW);
        }
        let entry = chunk.entry_register_count();
        debug_assert!(parameters <= entry && entry <= size);
        // SAFETY: the stack reaches `base + REGISTER_WINDOW`, past `end`. The
        // parameters lie below `cut` and go to `base..base + parameters`, which
        // starts at or after `cut`, so no write reaches a source.
        let entry_pc = unsafe {
            let values = self.values.as_mut_ptr();
            copy_values(values.add(base), values.add(first).cast_const(), parameters);
            clear_values(values.add(base + parameters), entry - parameters);
            // `local a, b = ...` is the usual first statement of a vararg
            // body. The call runs that `Vararg` itself, from the extra
            // arguments that lie right here, and the callee starts at the
            // instruction after it: no dispatch, no second pass over the
            // varargs, no call.
            //
            // SAFETY: a validated chunk has code.
            let first_instruction = *chunk.code.get_unchecked(0);
            let (destination, wanted) = (
                usize::from(first_instruction.a()),
                usize::from(first_instruction.b()),
            );
            if first_instruction.raw().to_le_bytes()[0] == OpCode::Vararg as u8
                && destination + wanted <= size
            {
                let to = values.add(base + destination);
                let from = values.add(cut).cast_const();
                let extras = count - parameters;
                let copied = wanted.min(extras);
                copy_values(to, from, copied);
                clear_values(to.add(copied), wanted - copied);
                1
            } else {
                0
            }
        };
        parent.top = cut as u32;
        let index = self.live;
        debug_assert!(index < self.frames.len() && index < MAX_LUA_CALL_FRAMES);
        // SAFETY: the contract: a retired record exists at this depth.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        // SAFETY: forwarded from this function's contract.
        unsafe {
            install_record(
                record,
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller_return,
                base,
                end,
                (count - parameters) | VARARG_TRUNCATED as usize,
                entry_pc,
                &mut self.spare_site,
                &mut self.live,
                index,
            );
        }
    }
    /// [`Self::push_vararg_in_place`] with no call on its way, for a callee
    /// and a stack that need none: declines (`false`), with nothing changed
    /// but dead registers above the caller's, when a string stands on either
    /// side of a copy (a count to take or release), when the stack has to
    /// grow, and when the record at the new depth is not one to overwrite in
    /// place. The caller then goes the general way. Having no call before
    /// its end, the registers it keeps need no saving around one.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_vararg_in_place`].
    #[allow(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::too_many_arguments
    )]
    #[inline(always)]
    unsafe fn push_vararg_in_place_clean(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller_return: CallReturn,
        caller: *mut Frame,
        first: usize,
        count: usize,
    ) -> bool {
        debug_assert!(chunk.is_vararg && chunk.arg_register.is_none());
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        debug_assert!(parameters < count);
        // SAFETY: the contract.
        let parent = unsafe { &mut *caller };
        let first = parent.base as usize + first;
        let cut = first + parameters;
        let base = first + count;
        let end = base + size;
        debug_assert!(u32::try_from(end).is_ok());
        let index = self.live;
        debug_assert!(index < self.frames.len() && index < MAX_LUA_CALL_FRAMES);
        // SAFETY: the contract: a retired record exists at this depth.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        if self.values.len() < base + REGISTER_WINDOW || holds_named_site(record) {
            return false;
        }
        let Some(target) = recycled_target(record) else {
            return false;
        };
        let info = chunk.entry_info();
        let entry = info as u16 as usize;
        debug_assert!(parameters <= entry && entry <= size);
        let entry_pc = usize::from(info & crate::bytecode::ENTRY_VARARG_BIT != 0);
        let varargs = (count - parameters) | VARARG_TRUNCATED as usize;
        // SAFETY: `target` points into `record.caller`.
        unsafe { *target = caller_return };
        parent.top = cut as u32;
        record.base = base as u32;
        record.top = end as u32;
        record.pc = entry_pc;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = 0;
        record.vararg_word = varargs as u32;
        self.live = index + 1;
        if record.closure != Some(closure) || record.shape != shape {
            // SAFETY: the contract, and `recycled_target` said the record
            // owns neither its chunk nor its captures.
            unsafe { overwrite_closure_cold(record, chunk, cells, environment, shape, closure) };
        }
        debug_assert_eq!(record.chunk.address(), Arc::as_ptr(chunk));
        // SAFETY: the stack reaches `base + REGISTER_WINDOW`, past `end`. The
        // parameters lie below `cut` and go to `base..base + parameters`, which
        // starts at or after `cut`, so no write reaches a source. What is
        // written lies in the registers the callee's window takes, which the
        // caller reads no more.
        unsafe {
            let values = self.values.as_mut_ptr();
            fill_entry_registers(
                values.add(base),
                values.add(first).cast_const(),
                values.add(cut).cast_const(),
                parameters,
                entry,
                count - parameters,
                info,
            );
        }
        true
    }

    /// [`Self::push_vararg_in_place_general`], by a path without a call when
    /// the callee, its arguments and the stack allow one.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_vararg_in_place_general`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn push_vararg_in_place(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller_return: CallReturn,
        caller: *mut Frame,
        first: usize,
        count: usize,
    ) {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            if !self.push_vararg_in_place_clean(
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller_return,
                caller,
                first,
                count,
            ) {
                self.push_vararg_in_place_general(
                    chunk,
                    cells,
                    environment,
                    shape,
                    closure,
                    caller_return,
                    caller,
                    first,
                    count,
                );
            }
        }
    }

    /// Gives the frame below the top one its full window back, if the top
    /// frame cut it short ([`VARARG_TRUNCATED`]). Call it before the top
    /// frame is removed; the registers it regains hold dead values.
    #[inline(always)]
    fn restore_caller_top(&mut self) {
        let live = self.live;
        if live >= 2 && self.frames[live - 1].vararg_word & VARARG_TRUNCATED != 0 {
            let parent: *mut Frame = &raw mut self.frames[live - 2];
            // SAFETY: a record of `frames`, which nothing below touches.
            #[allow(unsafe_code)]
            let parent = unsafe { &mut *parent };
            restore_window(parent);
            let top = parent.top as usize;
            self.reserve_slots(top);
        }
    }

    /// [`Self::push_call_spread`] for a call of a vararg closure with all
    /// `count` arguments in one run of the value stack, starting at the
    /// absolute slot `first` in the calling window (a plain `Call`). The new
    /// window starts at `base`, the top of the calling window, plus the
    /// arguments past the callee's parameters, which go just below it.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_plain_call`]: `first + count` lies at or below
    /// `base`, which is the live top; `parts` are the parts of `closure`,
    /// whose chunk is a vararg body without a named-vararg table; a retired
    /// record exists at the new depth and the depth limit holds.
    #[allow(unsafe_code, clippy::cast_possible_truncation)]
    #[inline(never)]
    pub(crate) unsafe fn push_vararg_call(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        shape: u32,
        closure: ClosureHandle,
        caller: CallReturn,
        base: usize,
        first: usize,
        count: usize,
    ) {
        debug_assert_eq!(base, self.live_top());
        debug_assert!(first + count <= base);
        debug_assert!(chunk.is_vararg && chunk.arg_register.is_none());
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let extra = count.saturating_sub(parameters);
        let new_base = base + extra;
        let end = new_base + size;
        debug_assert!(u32::try_from(end).is_ok());
        if self.values.len() < new_base + REGISTER_WINDOW {
            self.values.grow(new_base + REGISTER_WINDOW);
        }
        let entry = chunk.entry_register_count();
        let copied = count.min(parameters);
        debug_assert!(copied <= entry && entry <= size);
        // SAFETY: the stack reaches `new_base + REGISTER_WINDOW`, past `end`.
        // The arguments lie below `base` (the contract); the extra ones are
        // written to `base..new_base` and the parameters to
        // `new_base..new_base + copied`, both at or above `base`, so no write
        // reaches a source. `entry <= size`.
        unsafe {
            let values = self.values.as_mut_ptr();
            let arguments = values.add(first).cast_const();
            copy_values(values.add(new_base), arguments, copied);
            copy_values(values.add(base), arguments.add(parameters), extra);
            clear_values(values.add(new_base + copied), entry - copied);
        }
        let index = self.live;
        debug_assert!(index < self.frames.len() && index < MAX_LUA_CALL_FRAMES);
        // SAFETY: the contract: a retired record exists at this depth.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        // SAFETY: forwarded from this function's contract.
        unsafe {
            install_record(
                record,
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller,
                new_base,
                end,
                extra,
                0,
                &mut self.spare_site,
                &mut self.live,
                index,
            );
        }
    }

    /// The fast loop's `CallVararg` of a plain or vararg Lua closure: the
    /// `count` registers from `first` of the top frame and then its varargs
    /// are the arguments. Declines, with nothing changed, for a frame with
    /// interior state, a named-vararg table, any other callee, and the depth
    /// limit; the general routine decides those.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_forward_call(
        &mut self,
        runtime: &Runtime,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
    ) -> PlainCall {
        let live = self.live;
        debug_assert!(live >= 1);
        let caller = &self.frames[live - 1];
        let (base, top) = (caller.base as usize, caller.top as usize);
        if results == 0
            || live >= MAX_LUA_CALL_FRAMES
            || caller.extras.is_some()
            || caller.chunk.arg_register.is_some()
            || first + count > top - base
        {
            return PlainCall::Declined;
        }
        let Some(parts) = runtime.closure_call_parts(closure) else {
            return PlainCall::Declined;
        };
        let hint = parts.call_hint;
        // Inline plans are the general routine's; a plain activation of the
        // callee is cheaper than evaluating one, as for any other call here.
        if hint & (CALL_HINT_PLAIN | CALL_HINT_VARARG) == 0 {
            return if hint == CALL_HINT_UNKNOWN {
                PlainCall::Leave
            } else {
                PlainCall::Declined
            };
        }
        if live >= self.frames.len() {
            // SAFETY: `parts` are `closure`'s own.
            #[allow(unsafe_code)]
            unsafe {
                push_blank_record(
                    &mut self.frames,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
        }
        let caller = &mut self.frames[live - 1];
        let forwarded = caller.vararg_count();
        caller.pc = pc + 1;
        caller.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        // SAFETY: `parts` are `closure`'s own, and the pushed record roots
        // `closure`. The registers lie in the caller's window (checked above)
        // and its varargs just below it; a record exists at the new depth.
        #[allow(unsafe_code)]
        unsafe {
            self.push_call_spread(
                parts.chunk,
                parts.cells,
                parts.environment,
                parts.shape,
                closure,
                CallReturn {
                    base: target,
                    results,
                },
                base,
                top - base,
                first,
                count,
                forwarded,
            );
        }
        PlainCall::Pushed
    }

    /// The fast loop's `TailCall`, `TailCallVararg` or `TailCallOpen` of a
    /// plain or vararg Lua closure: the top activation is replaced in place
    /// by the callee. The callee and `count` fixed arguments are the
    /// registers from `a`; `kind` adds the frame's varargs or open results.
    /// The activation keeps its record, return target and the start of its
    /// stack region, so a chain of tail calls uses no more stack than one
    /// activation. Declines, with nothing changed, for a frame with interior
    /// state (captured locals, to-be-closed slots, hook scratch), a callee
    /// with no plain or vararg form, and a named-vararg table.
    /// `pc` is the instruction's index.
    #[allow(clippy::too_many_lines)]
    #[inline(never)]
    fn tail_call_closure_general(
        &mut self,
        runtime: &Runtime,
        instruction: Instruction,
        callee: ClosureHandle,
        pc: usize,
    ) -> TailStep {
        debug_assert!(self.live >= 1);
        let index = self.live - 1;
        let (a, count) = (usize::from(instruction.a()), usize::from(instruction.b()));
        let kind = match instruction.opcode() {
            Ok(OpCode::TailCall) => TailCallArguments::Fixed,
            Ok(OpCode::TailCallVararg) => TailCallArguments::Vararg,
            _ => TailCallArguments::Open,
        };
        let record = &self.frames[index];
        let (base, top) = (record.base as usize, record.top as usize);
        if record.extras.is_some()
            || self.open_top as usize > base
            || a + 1 + count > top - base
            || match kind {
                TailCallArguments::Fixed => false,
                TailCallArguments::Vararg => record.chunk.arg_register.is_some(),
                TailCallArguments::Open => !record.has_open_results(),
            }
        {
            return TailStep::Declined;
        }
        let Some(parts) = runtime.closure_call_parts(callee) else {
            return TailStep::Declined;
        };
        let hint = parts.call_hint;
        let vararg_callee = hint & CALL_HINT_VARARG != 0;
        if hint & (CALL_HINT_PLAIN | CALL_HINT_VARARG) == 0 {
            // Nobody has classified the callee yet, or it has no plain form.
            return if hint == CALL_HINT_UNKNOWN {
                TailStep::Unclassified
            } else {
                TailStep::Declined
            };
        }
        let chunk = parts.chunk;
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let entry = chunk.entry_register_count();
        let old = record.vararg_count();
        let start = base - old;
        let extra;
        let new_base;
        if kind == TailCallArguments::Fixed && old == 0 && !vararg_callee {
            // The arguments move down over the registers of the activation.
            // Each swap puts an argument in its parameter and the dead value
            // there at a slot the later swaps or the clearing below reach.
            let copied = count.min(parameters);
            // SAFETY: the window `base..base + REGISTER_WINDOW` exists (the
            // running view), and the registers read lie inside the frame's
            // window (checked above). Each swap moves a source `a + 1 + i`
            // above its destination `i`, and a destination that is a later
            // source was moved from already, so every argument is read before
            // it is overwritten.
            #[allow(unsafe_code)]
            unsafe {
                let window = self.values.as_mut_ptr().add(base);
                for index in 0..copied {
                    std::ptr::swap(window.add(index), window.add(a + 1 + index));
                }
                for index in copied..entry {
                    clear_value(window.add(index));
                }
            }
            extra = 0;
            new_base = base;
        } else if kind == TailCallArguments::Vararg && count == 0 && !vararg_callee {
            // The arguments are the varargs, and they already stand where
            // the parameters go.
            let copied = old.min(parameters);
            // SAFETY: the stack reaches `start + entry`, as for the window of
            // any activation.
            #[allow(unsafe_code)]
            unsafe {
                let window = self.values.as_mut_ptr().add(start);
                for index in copied..entry {
                    clear_value(window.add(index));
                }
            }
            extra = 0;
            new_base = start;
        } else {
            let mut gathered = std::mem::take(&mut self.scratch);
            gathered.clear();
            for slot in &mut self.values[base + a + 1..base + a + 1 + count] {
                gathered.push(std::mem::replace(slot, Value::Nil));
            }
            match kind {
                TailCallArguments::Fixed => {}
                TailCallArguments::Vararg => {
                    for slot in &mut self.values[start..base] {
                        gathered.push(std::mem::replace(slot, Value::Nil));
                    }
                }
                TailCallArguments::Open => {
                    if let Some(open) = self.frames[index].open.as_deref_mut() {
                        open.active = false;
                        gathered.append(&mut open.values);
                    }
                }
            }
            let total = gathered.len();
            extra = if vararg_callee {
                total.saturating_sub(parameters)
            } else {
                0
            };
            new_base = start + extra;
            if self.values.len() < new_base + REGISTER_WINDOW {
                self.values.grow(new_base + REGISTER_WINDOW);
            }
            let copied = total.min(parameters);
            let mut arguments = gathered.drain(..);
            for slot in &mut self.values[new_base..new_base + copied] {
                *slot = arguments.next().unwrap_or(Value::Nil);
            }
            for slot in &mut self.values[start..new_base] {
                *slot = arguments.next().unwrap_or(Value::Nil);
            }
            drop(arguments);
            for slot in &mut self.values[new_base + copied..new_base + entry] {
                set_register(slot, Value::Nil);
            }
            self.scratch = gathered;
        }
        let record = &mut self.frames[index];
        // The replaced activation names the call, before its chunk goes.
        crate::vm::debug_info::set_call_site(
            &mut record.call_site,
            &mut self.spare_site,
            record.closure,
            &record.chunk,
            pc + 1,
        );
        if record.closure != Some(callee) || record.shape != parts.shape {
            // SAFETY: `parts` are `callee`'s own, and the record roots
            // `callee` from here on.
            #[allow(unsafe_code)]
            unsafe {
                rewrite_closure(
                    record,
                    chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    callee,
                );
            }
        }
        record.base = stack_slot(new_base);
        record.top = stack_slot(new_base + size);
        // A frame whose caller was cut short (`VARARG_TRUNCATED`) stays so.
        record.vararg_word = (record.vararg_word & VARARG_TRUNCATED) | stack_slot(extra);
        record.pc = 0;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = record.tail_calls.saturating_add(1);
        record.clear_plain_mark();
        if record.open.is_some() {
            record.clear_open_results();
        }
        TailStep::Moved
    }

    /// The tail call the fast loop meets in a steady state, done from the
    /// record and window it already holds: no index into the records, one
    /// test of the record's state, and the chunk's cached entry count. It
    /// replaces the activation in place and returns the chunk the record now
    /// runs, which starts at pc 0. A null result means nothing changed and
    /// [`Self::tail_call_closure`] decides.
    ///
    /// # Safety
    ///
    /// `frame` is the live top record, `window` the start of its window
    /// (`values[base..]`, which reaches `REGISTER_WINDOW` slots), and
    /// `instruction` a `TailCall` of the chunk it runs, at `pc`.
    #[allow(unsafe_code, clippy::cast_possible_truncation)]
    #[inline(never)]
    pub(crate) unsafe fn tail_call_closure_hot(
        &mut self,
        runtime: &Runtime,
        frame: *mut Frame,
        window: *mut Value,
        instruction: Instruction,
        pc: usize,
    ) -> *const Chunk {
        // SAFETY: the contract.
        let record = unsafe { &mut *frame };
        let (base, top) = (record.base as usize, record.top as usize);
        let (a, count) = (usize::from(instruction.a()), usize::from(instruction.b()));
        // The callee register lies in the window (the chunk is validated).
        debug_assert!(a < top - base);
        // SAFETY: as above: `a` lies inside the window.
        let Value::Closure(callee) = (unsafe { &*window.add(a) }) else {
            return std::ptr::null();
        };
        let callee = *callee;
        let Some(parts) = runtime.closure_call_parts(callee) else {
            return std::ptr::null();
        };
        // A frame whose registers hold captured variables, varargs, interior
        // state, an owned chunk or captures, or open results is the general
        // routine's. So is a callee that is not plain, or whose entry count
        // is not computed yet.
        let chunk = parts.chunk;
        let Some(entry) = chunk.entry_register_count_cached() else {
            return std::ptr::null();
        };
        let Some(origin) = record.closure else {
            return std::ptr::null();
        };
        if self.open_top as usize > base
            || parts.call_hint & CALL_HINT_PLAIN == 0
            || a + 1 + count > top - base
            || record.extras.is_some()
            || record.has_vararg_state()
            || record.chunk.is_owned()
            || record.upvalues.owns_slice()
            || record.open.is_some()
        {
            return std::ptr::null();
        }
        let copied = count.min(usize::from(chunk.parameter_count));
        debug_assert!(copied <= entry && entry <= usize::from(chunk.register_count));
        // SAFETY: the window reaches `REGISTER_WINDOW` slots, past `a + 1 +
        // count` (checked above) and past `entry`. Each argument moves from
        // `a + 1 + i` to the lower `i`, in order, so it is read before a
        // later copy overwrites it. The slots from `copied` to `entry` hold
        // dead values.
        unsafe {
            copy_values(window, window.add(a + 1).cast_const(), copied);
            clear_values(window.add(copied), entry - copied);
        }
        // The replaced activation names the call, before its chunk goes.
        // SAFETY: the record borrows its chunk (checked above), its closure
        // is set, and (validation) a pc fits `u32`.
        unsafe {
            crate::vm::debug_info::set_tail_site(
                &mut record.call_site,
                &mut self.spare_site,
                origin,
                &record.chunk,
                pc + 1,
            );
        }
        if record.closure != Some(callee) || record.shape != parts.shape {
            // SAFETY: `parts` are `callee`'s own, the record roots `callee`
            // from here on, and it owns neither its chunk nor its captures
            // (checked above).
            unsafe {
                overwrite_closure(
                    record,
                    chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    callee,
                );
            }
        }
        // The depth limit keeps every window inside `u32`.
        debug_assert!(u32::try_from(base + usize::from(chunk.register_count)).is_ok());
        record.top = (base + usize::from(chunk.register_count)) as u32;
        record.pc = 0;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = record.tail_calls.saturating_add(1);
        record.clear_plain_mark();
        debug_assert_eq!(record.chunk.address(), Arc::as_ptr(chunk));
        Arc::as_ptr(chunk)
    }

    /// The fast loop's `TailCall` and its variants: see
    /// [`Self::tail_call_closure_general`]. The common tail call, a fixed
    /// list of arguments from a frame without varargs or interior state to a
    /// callee with fixed parameters, is done here with no scratch space.
    #[allow(unsafe_code)]
    #[inline(never)]
    pub(crate) fn tail_call_closure(
        &mut self,
        runtime: &Runtime,
        instruction: Instruction,
        pc: usize,
    ) -> TailStep {
        debug_assert!(self.live >= 1 && self.live <= self.frames.len());
        debug_assert_eq!(instruction.opcode(), Ok(OpCode::TailCall));
        let index = self.live - 1;
        // SAFETY: a frame runs, so `live - 1 < frames.len()`.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        let (base, top) = (record.base as usize, record.top as usize);
        let open_top = self.open_top as usize;
        // A frame whose registers hold captured variables closes them
        // first, which needs the runtime: the general tail call does.
        if open_top > base {
            return TailStep::Declined;
        }
        let (a, count) = (usize::from(instruction.a()), usize::from(instruction.b()));
        // The callee register lies in the window: the chunk is validated.
        debug_assert!(a < top - base);
        let Some(&Value::Closure(callee)) = self.values.get(base + a) else {
            return TailStep::Declined;
        };
        let Some(parts) = runtime.closure_call_parts(callee) else {
            return TailStep::Declined;
        };
        if record.extras.is_some()
            || record.vararg_count() != 0
            || a + 1 + count > top - base
            || parts.call_hint & CALL_HINT_PLAIN == 0
            || record.chunk.is_owned()
            || record.upvalues.owns_slice()
        {
            return self.tail_call_closure_general(runtime, instruction, callee, pc);
        }
        let Some(origin) = record.closure else {
            return self.tail_call_closure_general(runtime, instruction, callee, pc);
        };
        let chunk = parts.chunk;
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let entry = chunk.entry_register_count();
        let copied = count.min(parameters);
        // SAFETY: the window `base..base + REGISTER_WINDOW` exists (the
        // running view), and the registers read lie inside the frame's
        // window (checked above). Each argument moves from `a + 1 + i` to the
        // lower `i`, in order, so it is read before a later copy overwrites
        // it. The slots from `copied` to `entry` hold dead values.
        unsafe {
            let window = self.values.as_mut_ptr().add(base);
            copy_values(window, window.add(a + 1).cast_const(), copied);
            clear_values(window.add(copied), entry - copied);
        }
        // The replaced activation names the call, before its chunk goes.
        // SAFETY: the check above found a closure, a chunk the record only
        // borrows, and (validation) a pc within `u32`.
        #[allow(unsafe_code)]
        unsafe {
            crate::vm::debug_info::set_tail_site(
                &mut record.call_site,
                &mut self.spare_site,
                origin,
                &record.chunk,
                pc + 1,
            );
        }
        if record.closure != Some(callee) || record.shape != parts.shape {
            // SAFETY: `parts` are `callee`'s own, the record roots `callee`
            // from here on, and it owns neither its chunk nor its captures
            // (checked above).
            unsafe {
                overwrite_closure(
                    record,
                    chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    callee,
                );
            }
        }
        // The depth limit keeps every window inside `u32`.
        debug_assert!(u32::try_from(base + size).is_ok());
        #[allow(clippy::cast_possible_truncation)]
        {
            record.top = (base + size) as u32;
        }
        record.pc = 0;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = record.tail_calls.saturating_add(1);
        record.clear_plain_mark();
        if record.open.is_some() {
            record.clear_open_results();
        }
        TailStep::Replaced
    }

    /// The fast loop's `TailCallVararg` and `TailCallOpen`. `return f(...)`
    /// of a callee with fixed parameters is done here: the arguments are the
    /// varargs of the frame, which already stand where the callee's
    /// parameters go, so the callee's window starts at the first of them.
    /// Anything else is [`Self::tail_call_closure_general`]'s.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) fn tail_call_closure_forward(
        &mut self,
        runtime: &Runtime,
        instruction: Instruction,
        pc: usize,
    ) -> TailStep {
        debug_assert!(self.live >= 1 && self.live <= self.frames.len());
        let index = self.live - 1;
        // SAFETY: a frame runs, so `live - 1 < frames.len()`.
        let record = unsafe { self.frames.get_unchecked_mut(index) };
        let (base, top) = (record.base as usize, record.top as usize);
        let open_top = self.open_top as usize;
        // A frame whose registers hold captured variables closes them
        // first, which needs the runtime: the general tail call does.
        if open_top > base {
            return TailStep::Declined;
        }
        let (a, count) = (usize::from(instruction.a()), usize::from(instruction.b()));
        debug_assert!(a < top - base);
        let Some(&Value::Closure(callee)) = self.values.get(base + a) else {
            return TailStep::Declined;
        };
        let Some(parts) = runtime.closure_call_parts(callee) else {
            return TailStep::Declined;
        };
        if instruction.raw().to_le_bytes()[0] != OpCode::TailCallVararg as u8
            || count != 0
            || record.extras.is_some()
            || record.chunk.arg_register.is_some()
            || a + 1 > top - base
            || parts.call_hint & CALL_HINT_PLAIN == 0
            || record.chunk.is_owned()
            || record.upvalues.owns_slice()
        {
            return self.tail_call_closure_general(runtime, instruction, callee, pc);
        }
        let Some(origin) = record.closure else {
            return self.tail_call_closure_general(runtime, instruction, callee, pc);
        };
        let chunk = parts.chunk;
        let size = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(size);
        let entry = chunk.entry_register_count();
        let start = base - record.vararg_count();
        let given = (base - start).min(parameters);
        // SAFETY: the stack reaches `start + REGISTER_WINDOW`, as for any
        // window below the running one. The slots from the last parameter
        // given to `entry` hold dead values.
        unsafe {
            clear_values(self.values.as_mut_ptr().add(start + given), entry - given);
        }
        // The replaced activation names the call, before its chunk goes.
        // SAFETY: the check above found a closure, a chunk the record only
        // borrows, and (validation) a pc within `u32`.
        #[allow(unsafe_code)]
        unsafe {
            crate::vm::debug_info::set_tail_site(
                &mut record.call_site,
                &mut self.spare_site,
                origin,
                &record.chunk,
                pc + 1,
            );
        }
        if record.closure != Some(callee) || record.shape != parts.shape {
            // SAFETY: `parts` are `callee`'s own, the record roots `callee`
            // from here on, and it owns neither its chunk nor its captures
            // (checked above).
            unsafe {
                overwrite_closure(
                    record,
                    chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    callee,
                );
            }
        }
        // The depth limit keeps every window inside `u32`.
        debug_assert!(u32::try_from(start + size).is_ok());
        #[allow(clippy::cast_possible_truncation)]
        {
            record.base = start as u32;
            record.top = (start + size) as u32;
        }
        record.vararg_word &= VARARG_TRUNCATED;
        record.pc = 0;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = record.tail_calls.saturating_add(1);
        record.clear_plain_mark();
        if record.open.is_some() {
            record.clear_open_results();
        }
        TailStep::Moved
    }

    /// The fast loop's `ReturnVararg`: the `fixed` registers from `a` of the
    /// live top frame and then its varargs go to the registers its direct
    /// caller expects, and the frame is retired. Declines, with nothing
    /// changed, in the cases [`Self::pop_plain_return`] does, and for a
    /// named-vararg table.
    #[inline(never)]
    pub(crate) fn pop_vararg_return(&mut self, a: usize, fixed: usize) -> bool {
        let live = self.live;
        if live < 2 {
            return false;
        }
        let (lower, upper) = self.frames[..live].split_at_mut(live - 1);
        let (parent, current) = (&mut lower[live - 2], &upper[0]);
        let cut = current.vararg_word & VARARG_TRUNCATED != 0;
        let Some(ReturnTarget::Direct(CallReturn {
            base: target,
            results,
        })) = &current.caller
        else {
            return false;
        };
        // Captured variables are closed first, by the general return.
        if self.open_top > current.base {
            return false;
        }
        let (target, results) = (usize::from(*target), usize::from(*results));
        let window = (current.top - current.base) as usize;
        // A caller cut short has its full window back below.
        let caller_window = if cut {
            usize::from(parent.chunk.register_count)
        } else {
            (parent.top - parent.base) as usize
        };
        if current.extras.is_some()
            || current.chunk.arg_register.is_some()
            || results == 0
            || a + fixed > window
            || target + results > caller_window
        {
            return false;
        }
        let varargs = current.vararg_count();
        let (base, destination) = (current.base as usize, parent.base as usize + target);
        #[allow(unsafe_code)]
        if cut {
            // The results land in registers that held the callee's varargs
            // (the caller's window grows back over them), and a result can
            // sit below the vararg it comes from or above it. So they go
            // through the scratch buffer first.
            let mut staged = std::mem::take(&mut self.scratch);
            staged.clear();
            // SAFETY: the returned values lie in the callee's registers
            // (checked above) and the varargs just below them, all inside
            // `values`.
            unsafe {
                let values = self.values.as_ptr();
                for index in 0..results {
                    let from = if index < fixed {
                        Some(values.add(base + a + index))
                    } else if index - fixed < varargs {
                        Some(values.add(base - varargs + index - fixed))
                    } else {
                        None
                    };
                    staged.push(from.map_or(Value::Nil, |from| (*from).clone()));
                }
            }
            restore_window(parent);
            for (slot, value) in self.values[destination..destination + results]
                .iter_mut()
                .zip(staged.drain(..))
            {
                set_register(slot, value);
            }
            self.scratch = staged;
        } else {
            // SAFETY: every frame's window lies inside `values`. The returned
            // values lie in the callee's registers (checked above) and the
            // varargs just below them, the results in the caller's window;
            // the callee's region starts at the caller's top, so they do not
            // overlap.
            unsafe {
                let values = self.values.as_mut_ptr();
                let to = values.add(destination);
                for index in 0..results {
                    let slot = to.add(index);
                    if index < fixed {
                        copy_value(slot, values.add(base + a + index).cast_const());
                    } else if index - fixed < varargs {
                        let from = values.add(base - varargs + index - fixed);
                        copy_value(slot, from.cast_const());
                    } else {
                        clear_value(slot);
                    }
                }
            }
        }
        if parent.open.is_some() {
            clear_open_results_cold(parent);
        }
        self.live = live - 1;
        true
    }

    /// The fast loop's call of a plain Lua closure: `function` is the
    /// register holding the callee, `count` arguments start at register
    /// `first`, and `results` values go back to the caller's registers from
    /// `target` on. The caller resumes after the call at `pc + 1`, and the
    /// call is its last executed instruction. Declines, with nothing
    /// changed, for any other call, which the general routine decides.
    ///
    /// `consecutive` says that the call is a `Call`: its arguments are the
    /// registers just above the callee's. A vararg callee can then keep its
    /// extra arguments where they are, if the caller reads nothing but the
    /// call's results above the callee's register afterwards (see
    /// [`Self::push_vararg_in_place`]).
    ///
    /// # Safety
    ///
    /// `caller` is the record of the live top frame, `window` the first
    /// register of its window, and `closure` the live closure in the
    /// callee's register.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) unsafe fn push_closure_call(
        &mut self,
        runtime: &Runtime,
        caller: *mut Frame,
        window: *mut Value,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
        consecutive: bool,
    ) -> Entered {
        let live = self.live;
        // The cases a steady state seldom meets share one branch to a cold
        // routine: no result wanted and a depth the thread has not reached
        // yet, which has no record to reuse.
        if results == 0 || live >= self.frames.len() {
            // SAFETY: the contract.
            return unsafe {
                self.push_closure_call_rare(
                    runtime,
                    caller,
                    window,
                    live,
                    closure,
                    target,
                    results,
                    first,
                    count,
                    pc,
                    consecutive,
                )
            };
        }
        // SAFETY: the contract, and a record exists at the new depth.
        unsafe {
            self.push_closure_call_known(
                runtime,
                caller,
                window,
                live,
                closure,
                target,
                results,
                first,
                count,
                pc,
                consecutive,
            )
        }
    }

    /// [`Self::push_closure_call`] for the cases it sends here.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_closure_call`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    unsafe fn push_closure_call_rare(
        &mut self,
        runtime: &Runtime,
        _caller: *mut Frame,
        window: *mut Value,
        live: usize,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
        consecutive: bool,
    ) -> Entered {
        // A call with an open result count takes the plain push when its
        // callee is plain: the record then holds a direct target with no
        // result count, which the plain return declines and the general
        // return turns into the caller's open results. A vararg callee keeps
        // the general routine: one pushed in place cuts the caller's window
        // short, which only the plain return restores.
        if results == 0
            && runtime
                .closure_call_parts(closure)
                .is_none_or(|parts| parts.call_hint & CALL_HINT_PLAIN == 0)
        {
            return Entered::Other(PlainCall::Declined);
        }
        if live >= self.frames.len() {
            if live >= MAX_LUA_CALL_FRAMES {
                return Entered::Other(PlainCall::Declined);
            }
            let Some(parts) = runtime.closure_call_parts(closure) else {
                return Entered::Other(PlainCall::Declined);
            };
            // SAFETY: `parts` are `closure`'s own.
            unsafe {
                push_blank_record(
                    &mut self.frames,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
        }
        // The records may have moved. SAFETY: `live - 1` is the caller's
        // index, and a record exists at the new depth now.
        unsafe {
            let caller = self.frames.as_mut_ptr().add(live - 1);
            self.push_closure_call_known(
                runtime,
                caller,
                window,
                live,
                closure,
                target,
                results,
                first,
                count,
                pc,
                consecutive,
            )
        }
    }

    /// [`Self::push_closure_call`] once its seldom cases are out of the way.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_closure_call`], `live < frames.len()`, and
    /// `results` is zero only for a plain callee.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn push_closure_call_known(
        &mut self,
        runtime: &Runtime,
        caller: *mut Frame,
        window: *mut Value,
        live: usize,
        closure: ClosureHandle,
        target: u8,
        results: u8,
        first: usize,
        count: usize,
        pc: usize,
        consecutive: bool,
    ) -> Entered {
        // SAFETY: the contract.
        let frame = unsafe { &mut *caller };
        let (base, top) = (frame.base as usize, frame.top as usize);
        // The callee register and the arguments lie in the caller's window:
        // the chunk is validated (`Call` and `CallOne` operands).
        debug_assert!(first + count <= (top - base));
        // SAFETY: the contract: the callee's register roots `closure`.
        let parts = unsafe { runtime.closure_call_parts_live(closure) };
        if parts.call_hint & CALL_HINT_PLAIN == 0 {
            // A vararg body called by a `Call` with arguments past its
            // parameters leaves them where they are, if the caller reads
            // nothing but the call's results from the callee's register up
            // after the call: the callee's window takes their place. The
            // chunk's liveness analysis, which the collector relies on in the
            // same way, says so: the highest register that matters after the
            // call is within the results.
            if consecutive
                && parts.call_hint & CALL_HINT_VARARG != 0
                && count > usize::from(parts.chunk.parameter_count)
                && usize::from(parts.chunk.parameter_count)
                    <= usize::from(parts.chunk.register_count)
                && (frame.chunk.calls_leave_registers_dead()
                    || frame.chunk.call_leaves_registers_dead(pc))
            {
                frame.pc = pc + 1;
                frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
                // SAFETY: the contract, and `parts` are `closure`'s own; the
                // callee is a vararg body (the hint says so) and has fewer
                // parameters than there are arguments.
                unsafe {
                    self.push_vararg_in_place(
                        parts.chunk,
                        parts.cells,
                        parts.environment,
                        parts.shape,
                        closure,
                        CallReturn {
                            base: target,
                            results,
                        },
                        caller,
                        first,
                        count,
                    );
                }
                return Entered::Other(PlainCall::Pushed);
            }
            // SAFETY: the contract, and `parts` are `closure`'s own.
            return unsafe {
                Entered::Other(self.push_nonplain_call(
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    parts.call_hint,
                    caller,
                    closure,
                    CallReturn {
                        base: target,
                        results,
                    },
                    first,
                    count,
                    pc,
                ))
            };
        }
        frame.pc = pc + 1;
        frame.last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
        // SAFETY: `parts` are `closure`'s own, and the pushed record roots
        // `closure`. The new window starts at the caller's top, above the
        // arguments checked against the caller's window above.
        let caller_return = CallReturn {
            base: target,
            results,
        };
        // SAFETY: the contract, and `live` is the live count: a record
        // exists at `live`. (Its address is taken from the records, not as
        // the one after `caller`: the loop keeps `caller` on its stack, and
        // every test of the record then loaded it again.)
        let record = unsafe { self.frames.as_mut_ptr().add(live) };
        let hot = unsafe {
            self.push_plain_call_hot::<false>(
                runtime,
                parts.chunk,
                parts.shape,
                closure,
                caller_return,
                record,
                window,
                top - base,
                top,
                first,
                count,
                live,
            )
        };
        let (frame, window) = match hot {
            Some(pushed) => pushed,
            // SAFETY: the contract.
            None => unsafe {
                self.push_plain_call_cold(
                    runtime,
                    closure,
                    caller_return,
                    window,
                    top - base,
                    top,
                    first,
                    count,
                )
            },
        };
        Entered::Fast { frame, window }
    }

    /// What the fast loop runs on: the top record and the register window
    /// of the top activation, which the stack is grown to hold.
    ///
    /// # Safety
    ///
    /// At least one frame is live.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn running_view(&mut self) -> (*mut Frame, *mut [Value; REGISTER_WINDOW]) {
        let live = self.live;
        debug_assert!(live >= 1 && live <= self.frames.len());
        // SAFETY: `live - 1 < frames.len()`.
        let frame = unsafe { self.frames.as_mut_ptr().add(live - 1) };
        // SAFETY: the record is live.
        let base = unsafe { (*frame).base } as usize;
        if self.values.len() < base + REGISTER_WINDOW {
            self.values.grow(base + REGISTER_WINDOW);
        }
        // SAFETY: the stack reaches `base + REGISTER_WINDOW`.
        let window =
            unsafe { self.values.as_mut_ptr().add(base) }.cast::<[Value; REGISTER_WINDOW]>();
        (frame, window)
    }

    #[cfg(test)]
    #[allow(unsafe_code)]
    pub(crate) fn top_raw_for_test(&mut self) -> (*mut Frame, *mut Value) {
        // SAFETY: tests call this with a frame live.
        unsafe { self.top_raw() }
    }

    /// The top record and the value stack, for the fast loop to switch
    /// activations without a lookup of its own.
    ///
    /// # Safety
    ///
    /// At least one frame is live.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn top_raw(&mut self) -> (*mut Frame, *mut Value) {
        debug_assert!(self.live >= 1 && self.live <= self.frames.len());
        // SAFETY: `live - 1 < frames.len()`.
        let frame = unsafe { self.frames.as_mut_ptr().add(self.live - 1) };
        (frame, self.values.as_mut_ptr())
    }

    /// The fast loop's plain return: `count` values from register `a` of the
    /// live top frame go to the registers its direct caller expects, and the
    /// frame is retired. Declines, with nothing changed, unless the frame
    /// has no interior state, returns to a direct caller and expects at least
    /// one result. The caller closes the frame's upvalues first. It returns
    /// the first register of the caller's window, or `None` if it declined.
    ///
    /// # Safety
    ///
    /// `frame` is the record of the live top frame and `window` the first
    /// register of its window.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn pop_plain_return(
        &mut self,
        frame: *mut Frame,
        window: *mut Value,
        a: usize,
        count: usize,
    ) -> Option<*mut Value> {
        // SAFETY: the contract.
        // A frame with the plain mark has a live caller below, a direct
        // target, no interior state and no cut caller: one compare instead
        // of those tests.
        let marked = unsafe { (*frame).plain_marked() };
        debug_assert!(std::ptr::eq(frame, &raw const self.frames[self.live - 1]));
        if !marked && self.live < 2 {
            return None;
        }
        // SAFETY: the contract, and a live frame below the top.
        let (current, parent) = unsafe { (&*frame, &mut *frame.sub(1)) };
        debug_assert!(!marked || (self.live >= 2 && current.plain_mark_holds()));
        let Some(ReturnTarget::Direct(CallReturn {
            base: target,
            results,
        })) = &current.caller
        else {
            if marked {
                // SAFETY: the mark says the target is direct.
                unsafe { std::hint::unreachable_unchecked() }
            }
            return None;
        };
        let (target, results) = (usize::from(*target), usize::from(*results));
        // The returned range lies in the callee's window and the result range
        // in the caller's: the chunks are validated (`Return` and `Call`
        // operands).
        debug_assert!(a + count <= (current.top - current.base) as usize);
        debug_assert!(target + results <= usize::from(parent.chunk.register_count));
        // A frame whose registers hold captured variables closes them
        // first, which needs the runtime: the general return does it.
        if (!marked && current.extras.is_some()) || results == 0 || self.open_top > current.base {
            return None;
        }
        // A caller the callee cut short ([`VARARG_TRUNCATED`]) gets its window
        // back (only now: every decline is above). The results then land in
        // registers the callee's window or its varargs covered. Each goes to
        // a lower address than the value it is copied from, since the
        // callee's window starts after the arguments, and the copies below
        // run upwards, so no source is overwritten before it is read.
        if !marked && current.vararg_word & VARARG_TRUNCATED != 0 {
            restore_window(parent);
        }
        let back = (current.base - parent.base) as usize;
        // SAFETY: every frame's window lies inside `values`, so the source
        // range `source..source + count` (checked against the callee's
        // window) and the destination range `destination..destination +
        // results` (checked against the caller's window) are addressable.
        // The callee's window starts at the caller's top, so the two ranges
        // do not overlap.
        unsafe {
            let (to, from) = (window.sub(back).add(target), window.add(a).cast_const());
            if results == 1 && count != 0 {
                // One value to one result is the common return.
                copy_value(to, from);
            } else {
                let copied = count.min(results);
                for index in 0..copied {
                    copy_value(to.add(index), from.add(index));
                }
                for index in copied..results {
                    clear_value(to.add(index));
                }
            }
        }
        // The values of a call with open results are stale now. Few frames
        // have any.
        if parent.open.is_some() {
            clear_open_results_cold(parent);
        }
        self.live -= 1;
        // The caller's window, which `back` already located.
        // SAFETY: as for the copies above.
        Some(unsafe { window.sub(back) })
    }

    /// The return of one value from a frame that carries the plain mark to
    /// a caller that wants one: the mark stands for every test of the
    /// record [`Self::pop_plain_return`] makes. Returns the first register
    /// of the caller's window, or null with nothing changed for a frame
    /// without the mark, a caller that wants another number of results, and
    /// a frame whose registers hold captured variables, which
    /// [`Self::pop_plain_return`] and the general return decide.
    ///
    /// # Safety
    ///
    /// As for [`Self::pop_plain_return`], and register `a` lies in the
    /// frame's window.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn pop_marked_return(
        &mut self,
        frame: *mut Frame,
        window: *mut Value,
        a: usize,
    ) -> *mut Value {
        // SAFETY: the contract; a marked frame has a live caller below.
        unsafe {
            let current = &*frame;
            if !current.plain_marked() {
                return std::ptr::null_mut();
            }
            debug_assert!(self.live >= 2 && current.plain_mark_holds());
            debug_assert!(std::ptr::eq(frame, &raw const self.frames[self.live - 1]));
            // The mark says the target is direct.
            let Some(ReturnTarget::Direct(CallReturn {
                base: target,
                results,
            })) = &current.caller
            else {
                std::hint::unreachable_unchecked()
            };
            if *results != 1 || self.open_top > current.base {
                return std::ptr::null_mut();
            }
            let parent = frame.sub(1);
            debug_assert!(a < (current.top - current.base) as usize);
            debug_assert!({
                let parent: &Frame = &*parent;
                usize::from(*target) < usize::from(parent.chunk.register_count)
            });
            // The callee's window starts at or above the caller's top, so
            // the result register and the returned one are distinct.
            let caller_window = window.sub((current.base - (*parent).base) as usize);
            copy_value(
                caller_window.add(usize::from(*target)),
                window.add(a).cast_const(),
            );
            // The values of a call with open results are stale now. Few
            // frames have any.
            if (*parent).open.is_some() {
                clear_open_results_cold(&mut *parent);
            }
            self.live -= 1;
            caller_window
        }
    }

    /// [`Self::pop_plain_return`] for the return of a plain `__index` or
    /// `__newindex` handler: the first of `count` values from register `a`
    /// goes to the register an [`ReturnTarget::IndexResult`] names, or is
    /// dropped for a [`ReturnTarget::SetMetamethod`], and the frame is
    /// retired. Declines, with nothing changed, for a frame with interior
    /// state or captured locals, a caller with interior state, or any other
    /// return target. The caller closes nothing: a frame with open captures
    /// declines. Returns the first register of the caller's window.
    ///
    /// # Safety
    ///
    /// As for [`Self::pop_plain_return`].
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) unsafe fn pop_handler_return(
        &mut self,
        frame: *mut Frame,
        window: *mut Value,
        a: usize,
        count: usize,
    ) -> Option<*mut Value> {
        let live = self.live;
        if live < 2 {
            return None;
        }
        // SAFETY: the contract, and a live frame below the top.
        let (current, parent) = unsafe { (&mut *frame, &*frame.sub(1)) };
        // The register the first result goes to, and how: as it is, or as the
        // truth of a comparison (negated if `.1`), or as a branch.
        let (destination, comparison, branch) = match current.caller {
            Some(ReturnTarget::IndexResult(destination)) => {
                (Some(usize::from(destination)), None, None)
            }
            Some(ReturnTarget::SetMetamethod) => (None, None, None),
            Some(ReturnTarget::Operator(operator)) => match operator.mode {
                OperatorMode::Value => (Some(usize::from(operator.destination)), None, None),
                OperatorMode::Compare { invert } => {
                    (Some(usize::from(operator.destination)), Some(invert), None)
                }
                OperatorMode::Branch { expected, invert } => (None, None, Some((expected, invert))),
            },
            _ => return None,
        };
        if current.extras.is_some()
            || parent.extras.is_some()
            || current.vararg_word & VARARG_TRUNCATED != 0
            || self.open_top > current.base
            || destination.is_some_and(|slot| slot >= (parent.top - parent.base) as usize)
        {
            return None;
        }
        debug_assert!(a + count <= (current.top - current.base) as usize);
        let back = (current.base - parent.base) as usize;
        if let Some((expected, invert)) = branch {
            // SAFETY: the value is in the callee's window (the chunk is
            // validated), and the caller is the live record below.
            let truth = count != 0 && unsafe { (*window.add(a)).is_truthy() };
            // A test is followed by its jump word (validated), so this
            // changes nothing only if it declines.
            if !crate::vm::instructions::try_branch_after_test(
                unsafe { &mut *frame.sub(1) },
                truth != invert,
                expected,
            ) {
                return None;
            }
        }
        if let Some(destination) = destination {
            // SAFETY: the result register lies in the caller's window (checked
            // above) and the value in the callee's (the chunk is validated).
            unsafe {
                let to = window.sub(back).add(destination);
                if let Some(invert) = comparison {
                    let truth = count != 0 && (*window.add(a)).is_truthy();
                    set_register(&mut *to, Value::Boolean(truth != invert));
                } else if count != 0 {
                    copy_value(to, window.add(a).cast_const());
                } else {
                    clear_value(to);
                }
                // The key the handler was given is a string, and the call
                // that follows (the method `C:m()` looked up) copies its
                // arguments over this register. A string there sends that
                // call down the general push, so release it here, where the
                // drop is the one the call would make anyway.
                let key = window.add(1);
                if matches!(*key, Value::String(_)) {
                    *key = Value::Nil;
                }
            }
        }
        // The retired record keeps a direct caller, which the next plain call
        // at this depth overwrites in place. The target it replaces owns
        // nothing, so it needs no drop.
        std::mem::forget(current.caller.replace(ReturnTarget::Direct(CallReturn {
            base: 0,
            results: 0,
        })));
        self.live = live - 1;
        // SAFETY: the caller's window starts `back` slots below.
        Some(unsafe { window.sub(back) })
    }

    /// Pushes the plain Lua `__index` or `__newindex` handler `closure` over
    /// the live top frame, returning to `target` (a plain target, which owns
    /// nothing), with the operands `receiver`, `key` and, for `__newindex`,
    /// `value` as its first parameters, copied as a call copies arguments.
    /// The record at the new depth is reused in place when it ran this
    /// closure last, made plain when it is not, and appended when the depth
    /// has none. Declines, with no live frame changed, at the depth limit,
    /// if the stack does not reach the new window, or if the handler is a
    /// vararg body: the general push decides. Returns the new top record and
    /// its window. The records may move, so `caller` is not to be used after
    /// the call.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`], `caller` is the record of the live
    /// top frame, the one at index `live - 1`, `window` the first register of
    /// its window, and `live` the number of live frames. `pc` is the pc of the
    /// instruction the handler answers. `receiver` and `value` address live
    /// values outside the new window, and `value` is null for a handler that
    /// takes two operands.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) unsafe fn push_handler(
        &mut self,
        runtime: &Runtime,
        closure: ClosureHandle,
        target: ReturnTarget,
        receiver: *const Value,
        key: HandlerKey<'_>,
        value: *const Value,
        caller: *mut Frame,
        window: *mut Value,
        live: usize,
        pc: usize,
    ) -> Option<(*mut Frame, *mut Value)> {
        debug_assert_eq!(live, self.live);
        debug_assert!(matches!(
            target,
            ReturnTarget::IndexResult(_) | ReturnTarget::SetMetamethod | ReturnTarget::Operator(_)
        ));
        if live >= MAX_LUA_CALL_FRAMES {
            return None;
        }
        // SAFETY: the contract.
        let (base, top) = unsafe { ((*caller).base as usize, (*caller).top as usize) };
        let parts = runtime.closure_call_parts(closure)?;
        let chunk = parts.chunk;
        if chunk.is_vararg || self.values.len() < top + REGISTER_WINDOW {
            return None;
        }
        let mut caller = caller;
        if live >= self.frames.len() {
            // No record at this depth: the thread has not reached it, or a
            // general push or pop since dropped the retired ones (a
            // `tostring` or a concatenation handler between two operator
            // handlers does). One is appended, as the plain call does
            // (`push_closure_call_rare`). Declining sent the handler the
            // general way, with a continuation and twice the instructions.
            // SAFETY: `parts` are `closure`'s own.
            unsafe {
                push_blank_record(
                    &mut self.frames,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
            // The records may have moved. SAFETY: `live - 1` is the caller's
            // index.
            caller = unsafe { self.frames.as_mut_ptr().add(live - 1) };
        }
        let entry = chunk.entry_register_count();
        let size = usize::from(chunk.register_count);
        let end = top + size;
        // SAFETY: a record exists at this depth (checked above).
        let record = unsafe { self.frames.get_unchecked_mut(live) };
        if holds_named_site(record) || recycled_target(record).is_none() {
            // The record last ran an activation the general push built: a
            // handler with a continuation, say, whose return left a target
            // that is not direct. It is retired, so it is made plain here,
            // out of line, with the releases that takes. Declining left it
            // as it was, and every handler entered at this depth after it
            // went the general way too, which built the same record again.
            // SAFETY: `parts` are `closure`'s own, and the record is retired.
            unsafe {
                rewrite_record(
                    record,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                    CallReturn {
                        base: 0,
                        results: 0,
                    },
                    0,
                    0,
                    0,
                    0,
                    &mut self.spare_site,
                );
            }
            debug_assert!(!holds_named_site(record) && recycled_target(record).is_some());
        }
        if record.closure != Some(closure) || record.shape != parts.shape {
            // A method call right after the handler runs another closure at
            // this depth, so this rewrite is the usual case. It calls
            // nothing, and the parts are the ones just looked up.
            // SAFETY: `recycled_target` said the record owns neither its
            // chunk nor its captures, and `parts` are `closure`'s own.
            unsafe {
                overwrite_closure(
                    record,
                    parts.chunk,
                    parts.cells,
                    parts.environment,
                    parts.shape,
                    closure,
                );
            }
        }
        let record_ptr = std::ptr::from_mut(record);
        let parameters = usize::from(chunk.parameter_count);
        // SAFETY: the stack reaches `top + REGISTER_WINDOW`, past `end`; the
        // new window starts at the caller's top, `top - base` slots on from
        // `window`. `entry <= size`, and the writes stay inside its first
        // `entry` registers.
        unsafe {
            let to = window.add(top - base);
            let copied = parameters.min(if value.is_null() { 2 } else { 3 });
            if copied != 0 {
                copy_value(to, receiver);
                if copied > 1 {
                    match key {
                        // A string key, the usual one, is cloned in line.
                        HandlerKey::Register(key) => {
                            if let Value::String(name) = &*key {
                                set_register(&mut *to.add(1), Value::String(name.clone()));
                            } else {
                                copy_value(to.add(1), key);
                            }
                        }
                        HandlerKey::Constant(name) => {
                            set_register(&mut *to.add(1), Value::String(name.clone()));
                        }
                    }
                    if copied > 2 {
                        copy_value(to.add(2), value);
                    }
                }
            }
            debug_assert!(copied <= entry && entry <= size);
            for index in copied..entry {
                clear_value(to.add(index));
            }
            // The caller resumes after the instruction at `pc`.
            (*caller).pc = pc + 1;
            (*caller).last_interpreted_pc = crate::vm::frame::LastPc::some(pc);
            // The record already runs this closure at this shape (see
            // `push_plain_call_hot`), so its environment, captures and chunk
            // are right too, and its target is direct, owning nothing.
            std::ptr::write(&raw mut (*record_ptr).caller, Some(target));
            (*record_ptr).base = top as u32;
            (*record_ptr).top = end as u32;
            (*record_ptr).pc = 0;
            (*record_ptr).last_interpreted_pc = crate::vm::frame::LastPc::NONE;
            std::ptr::write_volatile(&raw mut (*record_ptr).tail_calls, 0);
            std::ptr::write_volatile(&raw mut (*record_ptr).vararg_word, 0);
            self.live = live + 1;
            Some((record_ptr, to))
        }
    }

    /// Whether [`Self::pop_plain_return`] will take the running frame's
    /// return, whatever the values are: a caller to return to, a direct
    /// target that wants results, no interior state, no captured locals
    /// to close. A native tail call checks this before it runs, since it
    /// cannot be undone.
    #[inline(always)]
    pub(crate) fn can_pop_plain_return(&self) -> bool {
        let live = self.live;
        if live < 2 {
            return false;
        }
        // SAFETY: a frame runs, so `live - 1 < frames.len()`.
        #[allow(unsafe_code)]
        let current = unsafe { self.frames.get_unchecked(live - 1) };
        matches!(
            current.caller,
            Some(ReturnTarget::Direct(CallReturn { results, .. })) if results != 0
        ) && current.extras.is_none()
            && self.open_top <= current.base
    }

    /// [`Self::push_activation`] returning to any `caller` target.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`].
    #[allow(clippy::too_many_arguments, unsafe_code)]
    pub(crate) unsafe fn push_activation_to(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        closure: ClosureHandle,
        caller: ReturnTarget,
        arguments: ActivationArguments,
    ) -> usize {
        // SAFETY: forwarded from this function's contract.
        unsafe { self.push_activation_as(chunk, cells, environment, closure, caller, arguments) }
    }

    /// [`Self::push_activation`] for the bottom activation of a host entry,
    /// which has no caller: its return ends the run. The record borrows
    /// `chunk` and `cells` as a Lua call's does, so the entry neither builds
    /// a detached frame nor counts references.
    ///
    /// # Safety
    ///
    /// As for [`Self::push_activation`].
    #[allow(unsafe_code)]
    pub(crate) unsafe fn push_host_activation(
        &mut self,
        chunk: &Arc<Chunk>,
        cells: &ClosureCells,
        environment: TableHandle,
        closure: ClosureHandle,
        arguments: Vec<Value>,
    ) {
        self.drop_retired();
        // SAFETY: forwarded from this function's contract.
        unsafe {
            self.push_activation_as(
                chunk,
                cells,
                environment,
                closure,
                HostEntry,
                ActivationArguments::Values(arguments),
            );
        }
    }

    /// Ends the top activation for the fast loop, keeping its record and
    /// registers as dead leftovers. Only frames without interior state and
    /// with a direct caller are retired.
    #[inline(always)]
    pub(crate) fn retire_top(&mut self) {
        debug_assert!(self.frames[self.live - 1].extras.is_none());
        debug_assert!(
            self.open_top <= self.frames[self.live - 1].base,
            "a retired frame still has open upvalues"
        );
        self.restore_caller_top();
        self.live -= 1;
    }

    /// Removes the top frame, one that a trace pushed to stand for an inlined
    /// call while it made a host call (see `RunTables::restore_exit`) and that
    /// no code ran in. A frame with a direct caller, no interior state and no
    /// captured locals is retired, as a return retires it, so the restore of
    /// the next call reuses its record; any other is popped.
    pub(crate) fn take_off_top(&mut self) {
        let Some(current) = self.live.checked_sub(1).map(|top| &self.frames[top]) else {
            return;
        };
        if matches!(current.caller, Some(ReturnTarget::Direct(_)))
            && current.extras.is_none()
            && self.open_top <= current.base
        {
            self.retire_top();
        } else {
            drop(self.pop());
        }
    }

    /// Removes the top frame and returns it with its registers.
    pub(crate) fn pop(&mut self) -> Option<OwnedFrame> {
        self.drop_retired();
        self.restore_caller_top();
        let mut frame = self.frames.pop()?;
        frame.vararg_word &= !(VARARG_TRUNCATED | VARARG_PLAIN);
        self.live -= 1;
        debug_assert!(
            self.open_top <= frame.base,
            "a popped frame still has open upvalues"
        );
        let (base, top) = (frame.base as usize, frame.top as usize);
        let varargs = self.values[base - frame.vararg_count()..base]
            .iter_mut()
            .map(|slot| std::mem::replace(slot, Value::Nil))
            .collect();
        let registers = self.values[base..top]
            .iter_mut()
            .map(|slot| std::mem::replace(slot, Value::Nil))
            .collect();
        Some(OwnedFrame {
            frame,
            registers,
            varargs,
        })
    }

    /// Detaches every frame with its registers, bottom first.
    pub(crate) fn into_owned_frames(mut self) -> Vec<OwnedFrame> {
        let mut owned = Vec::with_capacity(self.live);
        while let Some(frame) = self.pop() {
            owned.push(frame);
        }
        owned.reverse();
        owned
    }

    /// Removes the top frame and drops its registers, both in place.
    #[inline(always)]
    pub(crate) fn discard_top(&mut self) {
        self.drop_retired();
        self.restore_caller_top();
        let Some(frame) = self.frames.last() else {
            return;
        };
        debug_assert!(
            self.open_top <= frame.base,
            "a discarded frame still has open upvalues"
        );
        let first = frame.base as usize - frame.vararg_count();
        for slot in &mut self.values[first..frame.top as usize] {
            set_register(slot, Value::Nil);
        }
        self.frames.truncate(self.frames.len() - 1);
        self.live -= 1;
    }

    /// Detaches frames `index..` into their own stack, rebased at zero.
    pub(crate) fn split_off(&mut self, index: usize) -> Self {
        self.drop_retired();
        if index >= self.frames.len() {
            return Self::new();
        }
        let base = self.frames[index].base - self.frames[index].vararg_count() as u32;
        debug_assert!(
            self.open_top <= base,
            "frames were split off with open upvalues"
        );
        let truncating = index > 0 && self.frames[index].vararg_word & VARARG_TRUNCATED != 0;
        let mut frames = self.frames.split_off(index);
        let values = self.values.split_off(base as usize);
        for frame in &mut frames {
            frame.base -= base;
            frame.top -= base;
            // The bottom one has no caller record below it any more, and
            // the others are cheap to mark again.
            frame.clear_plain_mark();
        }
        self.live = index;
        if truncating {
            // The frame below has no callee here any more: it regains its
            // window, whose slots the split took away.
            frames[0].vararg_word &= !VARARG_TRUNCATED;
            let parent: *mut Frame = &raw mut self.frames[index - 1];
            // SAFETY: a record of `frames`, which nothing below touches.
            #[allow(unsafe_code)]
            let parent = unsafe { &mut *parent };
            restore_window(parent);
            let top = parent.top as usize;
            self.reserve_slots(top);
        }
        let live = frames.len();
        Self {
            frames,
            live,
            values,
            open: Vec::new(),
            open_top: 0,
            scratch: Vec::new(),
            spare_site: None,
            host_entry: false,
        }
    }

    /// Resizes the top frame's window, clearing slots it gains or loses.
    pub(crate) fn resize_top(&mut self, size: usize) {
        self.drop_retired();
        let frame = self.frames.last_mut().expect("resized stack has a frame");
        let (old_top, top) = (frame.top as usize, frame.base as usize + size);
        frame.top = stack_slot(top);
        self.reserve_slots(top);
        for slot in &mut self.values[old_top.min(top)..old_top.max(top)] {
            set_register(slot, Value::Nil);
        }
    }

    #[inline(always)]
    pub(crate) fn last(&self) -> Option<FrameRef<'_>> {
        let index = self.live.checked_sub(1)?;
        Some(self.at(index))
    }

    /// The varargs of the frame at `index`: the slots just below its window.
    #[inline(always)]
    pub(crate) fn varargs(&self, index: usize) -> &[Value] {
        let frame = &self[index];
        &self.values[frame.base as usize - frame.vararg_count()..frame.base as usize]
    }

    /// The record of the top frame.
    #[inline(always)]
    pub(crate) fn last_frame(&self) -> Option<&Frame> {
        self.live.checked_sub(1).map(|index| &self.frames[index])
    }

    /// The registers of the top frame, without building a [`FrameRef`].
    #[inline(always)]
    pub(crate) fn last_registers(&self) -> Option<&[Value]> {
        let index = self.live.checked_sub(1)?;
        let frame = &self[index];
        Some(&self.values[frame.base as usize..frame.top as usize])
    }

    /// The record of the top frame and its registers, without the varargs
    /// a [`FrameMut`] carries: what a coroutine switch reads and writes.
    #[inline(always)]
    pub(crate) fn top_mut(&mut self) -> Option<(&mut Frame, &mut [Value])> {
        let index = self.live.checked_sub(1)?;
        let frame = self.frames.get_mut(index)?;
        let registers = self
            .values
            .get_mut(frame.base as usize..frame.top as usize)?;
        Some((frame, registers))
    }

    #[inline(always)]
    pub(crate) fn last_mut(&mut self) -> Option<FrameMut<'_>> {
        let index = self.live.checked_sub(1)?;
        Some(self.at_mut(index))
    }

    #[inline(always)]
    pub(crate) fn at(&self, index: usize) -> FrameRef<'_> {
        let frame = &self[index];
        let base = frame.base as usize;
        let registers = &self.values[base..frame.top as usize];
        // SAFETY: a frame's varargs lie in the `vararg_count` slots below its
        // `base`, inside the value stack (see `Frame::vararg_count`).
        #[allow(unsafe_code)]
        let varargs = unsafe {
            std::slice::from_raw_parts(
                self.values.as_ptr().add(base - frame.vararg_count()),
                frame.vararg_count(),
            )
        };
        FrameRef {
            frame,
            registers,
            varargs,
        }
    }

    #[inline(always)]
    pub(crate) fn at_mut(&mut self, index: usize) -> FrameMut<'_> {
        let frame = &mut self.frames[..self.live][index];
        let (base, top) = (frame.base as usize, frame.top as usize);
        let (below, window) = self.values.split_at_mut(base);
        // SAFETY: as for `at`.
        #[allow(unsafe_code)]
        let varargs = unsafe { below.get_unchecked_mut(base - frame.vararg_count()..) };
        FrameMut {
            registers: &mut window[..top - base],
            varargs,
            frame,
        }
    }

    /// The frame `depth` levels below the top, with its registers.
    pub(crate) fn nth_from_top(&self, depth: usize) -> Option<FrameRef<'_>> {
        let index = self.live.checked_sub(depth + 1)?;
        Some(self.at(index))
    }

    pub(crate) fn nth_from_top_mut(&mut self, depth: usize) -> Option<FrameMut<'_>> {
        let index = self.live.checked_sub(depth + 1)?;
        Some(self.at_mut(index))
    }

    /// Visits every frame with its register window, bottom to top.
    pub(crate) fn for_each_mut(&mut self, mut visit: impl FnMut(FrameMut<'_>)) {
        let values = &mut self.values;
        for frame in &mut self.frames[..self.live] {
            let (base, top) = (frame.base as usize, frame.top as usize);
            let (below, window) = values.split_at_mut(base);
            visit(FrameMut {
                registers: &mut window[..top - base],
                varargs: &mut below[base - frame.vararg_count()..],
                frame,
            });
        }
    }

    /// [`Self::for_each_mut`], also handing over each frame's open
    /// upvalues.
    pub(crate) fn for_each_mut_with_open(
        &mut self,
        mut visit: impl FnMut(FrameMut<'_>, &[OpenUpvalue]),
    ) {
        let values = &mut self.values;
        let open = &self.open;
        for frame in &mut self.frames[..self.live] {
            let (base, top) = (frame.base as usize, frame.top as usize);
            let pinned = {
                let start = open.partition_point(|entry| entry.slot < base as u32);
                let stop = open.partition_point(|entry| entry.slot < top as u32);
                &open[start..stop.max(start)]
            };
            let (below, window) = values.split_at_mut(base);
            visit(
                FrameMut {
                    registers: &mut window[..top - base],
                    varargs: &mut below[base - frame.vararg_count()..],
                    frame,
                },
                pinned,
            );
        }
    }

    /// The live frame records and the whole value stack, including the dead
    /// slots above the top window, for the fast loop.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub(crate) fn frames_and_values(&mut self) -> (&mut [Frame], &mut ValueStack) {
        debug_assert!(self.live <= self.frames.len());
        // SAFETY: every mutator keeps `live <= frames.len()`.
        let frames = unsafe { self.frames.get_unchecked_mut(..self.live) };
        (frames, &mut self.values)
    }
}

/// Sets every register of `slots` to nil. Only strings own anything to
/// release; the other variants take just the tag, four slots per loop pass.
#[inline(always)]
fn clear_registers(slots: &mut [Value]) {
    let (quads, remainder) = slots.as_chunks_mut::<4>();
    for quad in quads {
        for slot in quad {
            set_register(slot, Value::Nil);
        }
    }
    for slot in remainder {
        set_register(slot, Value::Nil);
    }
}

/// The fixed-size register window starting at `base`, for the fast loop.
///
/// Slots past the activation's `top` are dead: validated bytecode never
/// addresses them. Any register operand indexes the window without a bounds
/// check.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn register_window(
    values: &mut ValueStack,
    base: usize,
) -> &mut [Value; REGISTER_WINDOW] {
    let end = base + REGISTER_WINDOW;
    if values.len() < end {
        values.grow(end);
    }
    // SAFETY: `values[base..end]` exists, and its length is exactly
    // `REGISTER_WINDOW`.
    unsafe {
        &mut *values
            .as_mut_ptr()
            .add(base)
            .cast::<[Value; REGISTER_WINDOW]>()
    }
}

/// Makes the retired `record` the activation of `closure` that
/// [`CallStack::push_plain_call`] builds, with `varargs` arguments below its
/// window.
///
/// A retired record usually has no interior state and no foreign return
/// target, which leaves its caller target and the record's numbers to
/// overwrite. Any other record is rewritten out of line, which keeps the
/// drops it needs (and their calls) off the hot path.
///
/// # Safety
///
/// As for [`CallStack::push_plain_call`].
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    unsafe_code
)]
#[inline(always)]
unsafe fn install_record(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
    caller: CallReturn,
    base: usize,
    end: usize,
    varargs: usize,
    pc: usize,
    spare: &mut Option<Box<FrameSite>>,
    live: &mut usize,
    index: usize,
) {
    if let Some(target) = recycled_target(record) {
        // A site that named the last activation here is not this one's. A
        // tail site stays: it is read only while `tail_calls` is not zero
        // (see `Frame::current_call_site`), and the next tail call here
        // overwrites it in place.
        if holds_named_site(record) {
            stash_site(&mut record.call_site, spare);
        }
        // It usually already runs this very closure at the shape it has now,
        // so its environment and captures are right too (see
        // `Frame::shape`), and so is its chunk: a closure's chunk never
        // changes, and every writer of a record's closure writes its chunk
        // with it. The closure is fixed last, with nothing live after it, so
        // the call that does it saves none of this path's values.
        // SAFETY: `target` points into `record.caller`, which nothing above
        // replaced.
        unsafe { *target = caller };
        record.base = base as u32;
        record.top = end as u32;
        record.pc = pc;
        record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
        record.tail_calls = 0;
        record.vararg_word = varargs as u32;
        *live = index + 1;
        if record.closure != Some(closure) || record.shape != shape {
            // SAFETY: the contract, and `recycled_target` said the record
            // owns neither its chunk nor its captures.
            unsafe { overwrite_closure_cold(record, chunk, cells, environment, shape, closure) };
        }
        debug_assert_eq!(record.chunk.address(), Arc::as_ptr(chunk));
    } else {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            rewrite_record(
                record,
                chunk,
                cells,
                environment,
                shape,
                closure,
                caller,
                base,
                end,
                varargs,
                pc,
                spare,
            );
        }
        *live = index + 1;
    }
}

/// [`copy_value`] that declines (`false`, nothing copied) instead of calling
/// when a string is involved on either side.
///
/// # Safety
///
/// As for [`copy_value`].
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn copy_value_clean(to: *mut Value, from: *const Value) -> bool {
    // SAFETY: the contract; see `copy_value`.
    unsafe {
        if matches!(*to, Value::String(_)) || matches!(*from, Value::String(_)) {
            return false;
        }
        let (to, from) = (to.cast::<u8>(), from.cast::<u8>());
        to.write(from.read());
        to.add(8)
            .cast::<std::mem::MaybeUninit<u64>>()
            .write(from.add(8).cast::<std::mem::MaybeUninit<u64>>().read());
    }
    true
}

/// [`copy_value`] with a string's reference count taken in line: the only
/// call left is the one that frees an overwritten string whose last
/// reference this was, and it sits on a cold branch.
///
/// # Safety
///
/// As for [`copy_value`].
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn copy_value_counted(to: *mut Value, from: *const Value) {
    // SAFETY: the contract; see `copy_value`.
    unsafe {
        if let Value::String(old) = &mut *to {
            std::ptr::drop_in_place(old);
        }
        if let Value::String(text) = &*from {
            std::ptr::write(to, Value::String(text.clone()));
            return;
        }
        let (to, from) = (to.cast::<u8>(), from.cast::<u8>());
        to.write(from.read());
        to.add(8)
            .cast::<std::mem::MaybeUninit<u64>>()
            .write(from.add(8).cast::<std::mem::MaybeUninit<u64>>().read());
    }
}

/// [`clear_value`] with an old string released in line, as
/// [`copy_value_counted`] releases it.
///
/// # Safety
///
/// As for [`clear_value`].
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn clear_value_counted(to: *mut Value) {
    // SAFETY: the contract.
    unsafe {
        if let Value::String(old) = &mut *to {
            std::ptr::drop_in_place(old);
        }
        std::ptr::write(to, Value::Nil);
    }
}

/// [`clear_value`] that declines (`false`, nothing written) instead of
/// calling when the old value is a string.
///
/// # Safety
///
/// As for [`clear_value`].
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn clear_value_clean(to: *mut Value) -> bool {
    // SAFETY: the contract.
    unsafe {
        if matches!(*to, Value::String(_)) {
            return false;
        }
        std::ptr::write(to, Value::Nil);
    }
    true
}

/// Sets the registers of a vararg callee that starts at `window`: copies its
/// `parameters` from `arguments`, clears the others of its first `entry`
/// registers, and runs a leading `Vararg` that `info` describes
/// ([`Chunk::entry_info`]) from the `extras` values at `varargs`. A string
/// on either side of a write goes to a cold routine that takes the counts.
///
/// Kept in line: out of line, the call costs more than the loops save.
///
/// # Safety
///
/// `window` holds at least `entry` registers, plus the leading `Vararg`'s
/// when `info` has one; `arguments` holds `parameters` values and `varargs`
/// `extras`; none of these runs overlaps a written one.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn fill_entry_registers(
    window: *mut Value,
    arguments: *const Value,
    varargs: *const Value,
    parameters: usize,
    entry: usize,
    extras: usize,
    info: u64,
) {
    // SAFETY: the contract.
    unsafe {
        // A string, on either side of a write, sends the whole run to a cold
        // routine that takes the counts.
        let mut clean = true;
        for index in 0..parameters {
            if !copy_value_clean(window.add(index), arguments.add(index)) {
                clean = false;
                break;
            }
        }
        if !clean {
            copy_entry_varargs_slow(window, arguments, parameters, parameters);
        }
        clean = true;
        for index in parameters..entry {
            if !clear_value_clean(window.add(index)) {
                clean = false;
                break;
            }
        }
        if !clean {
            copy_entry_varargs_slow(window.add(parameters), arguments, 0, entry - parameters);
        }
        // `local a, b = ...` is the usual first statement of a vararg body.
        // The call runs that `Vararg` itself, from the extra arguments that
        // lie right here, and the callee starts at the instruction after it:
        // no dispatch, no second pass over the varargs.
        if info & crate::bytecode::ENTRY_VARARG_BIT != 0 {
            let (destination, wanted) = ((info >> 24) as u8 as usize, (info >> 32) as u8 as usize);
            let to = window.add(destination);
            let copied = wanted.min(extras);
            clean = true;
            for index in 0..copied {
                if !copy_value_clean(to.add(index), varargs.add(index)) {
                    clean = false;
                    break;
                }
            }
            if clean {
                for index in copied..wanted {
                    if !clear_value_clean(to.add(index)) {
                        clean = false;
                        break;
                    }
                }
            }
            if !clean {
                copy_entry_varargs_slow(to, varargs, copied, wanted);
            }
        }
    }
}

/// The entry `Vararg` of [`CallStack::push_vararg_in_place_clean`] where a
/// string stands on either side of a copy: copies the first `copied` values
/// at `from` to `to` with the reference counts that takes, and sets the rest
/// of the `wanted` values at `to` to nil. Repeats what the clean loop did
/// before the string turned up, which does no harm.
///
/// # Safety
///
/// As for [`copy_values`] for `copied` values, and `to` holds `wanted` values.
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
unsafe fn copy_entry_varargs_slow(
    to: *mut Value,
    from: *const Value,
    copied: usize,
    wanted: usize,
) {
    // SAFETY: the contract.
    unsafe {
        copy_values(to, from, copied);
        for index in copied..wanted {
            set_register(&mut *to.add(index), Value::Nil);
        }
    }
}

/// Whether `record` holds a site that names its last activation by a call
/// (`FrameSite::Named`), which a new activation must not inherit.
#[inline(always)]
fn holds_named_site(record: &Frame) -> bool {
    matches!(record.call_site.as_deref(), Some(FrameSite::Named(_)))
}

/// The return target of a retired `record` that has no interior state, a
/// direct caller, a chunk it only borrows and captures it does not count, so
/// that a new activation overwrites all of it in place with no drop glue.
/// One test of the target's tag, which the writes after it then need no
/// second match for.
#[inline(always)]
fn recycled_target(record: &mut Frame) -> Option<*mut CallReturn> {
    if record.extras.is_some() || record.chunk.is_owned() || record.upvalues.owns_slice() {
        return None;
    }
    match &mut record.caller {
        Some(ReturnTarget::Direct(target)) => Some(std::ptr::from_mut(target)),
        _ => None,
    }
}

/// Makes `record` run `closure`, whose parts are `chunk`, `cells`,
/// `environment` and `shape`. Calls nothing: the old chunk and captures
/// need no release (see [`recycled_target`]).
///
/// # Safety
///
/// The parts are `closure`'s own, the record roots `closure` from here on,
/// and it borrows its chunk and captures, owning no reference to either.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn overwrite_closure(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
) {
    // SAFETY: the old chunk and captures own nothing (the contract), so
    // skipping their destructors leaks nothing.
    unsafe {
        std::ptr::write(&raw mut record.chunk, FrameChunk::borrowed(chunk));
        std::ptr::write(&raw mut record.upvalues, CaptureCells::borrow_shared(cells));
    }
    record.closure = Some(closure);
    record.environment = environment;
    record.shape = shape;
}

/// [`overwrite_closure`] out of line, for a record that last ran another
/// closure.
///
/// # Safety
///
/// As for [`overwrite_closure`].
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn overwrite_closure_cold(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
) {
    // SAFETY: the contract.
    unsafe { overwrite_closure(record, chunk, cells, environment, shape, closure) };
}

/// Moves the call site box of a recycled record to `spare`, or frees it when
/// `spare` holds one already.
#[inline(always)]
fn stash_site(site: &mut Option<Box<FrameSite>>, spare: &mut Option<Box<FrameSite>>) {
    if spare.is_none() {
        // The spare slot owns nothing, and the record gives up its box.
        *spare = site.take();
    } else {
        free_site(site);
    }
}

#[cold]
#[inline(never)]
fn free_site(site: &mut Option<Box<FrameSite>>) {
    *site = None;
}

/// Appends a record for [`CallStack::push_plain_call`] to overwrite: the
/// first call at a new depth.
///
/// # Safety
///
/// `chunk` and `cells` are the chunk and cells of the live closure `closure`.
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
unsafe fn push_blank_record(
    frames: &mut Vec<Frame>,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
) {
    let mut frame = Frame::activation(
        // SAFETY: the contract.
        unsafe { FrameChunk::borrowed(chunk) },
        // SAFETY: the contract.
        unsafe { CaptureCells::borrow_shared(cells) },
        Some(ReturnTarget::Direct(CallReturn {
            base: 0,
            results: 1,
        })),
        None,
        environment,
        closure,
    );
    frame.shape = shape;
    frames.push(frame);
}

/// Makes `parent` span the registers of its chunk again, after a callee that
/// had cut its window short ([`VARARG_TRUNCATED`]) is gone.
#[inline(always)]
fn restore_window(parent: &mut Frame) {
    parent.top = parent.base + u32::from(parent.chunk.register_count);
}

/// `Vararg`: copies the first `wanted` of the `available` varargs just below
/// the window that starts at `window` to the registers from `a`, and nil past
/// the last.
///
/// # Safety
///
/// `window` is the first register of a window in the value stack, which holds
/// the `available` varargs of its frame just below it, and `a + wanted`
/// registers.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) unsafe fn copy_varargs(window: *mut Value, a: usize, wanted: u8, available: usize) {
    let count = usize::from(wanted);
    debug_assert!(a + count <= REGISTER_WINDOW);
    // SAFETY: the contract. The varargs and the registers never overlap.
    // `Value` is a `repr(C, u8)` enum: its tag, then a payload of at most one
    // word at offset 8 (see `copy_value`). A string, on either side, needs a
    // reference count taken or released: the loop leaves such a call to
    // `copy_varargs_with_strings` and ends with no call of its own, so the
    // usual call saves no registers.
    unsafe {
        // One value: false if a string is on either side (nothing copied).
        // The loops below run four at a time, which drops the loop's own
        // instructions from the few values a typical `local a, b = ...`
        // moves.
        #[inline(always)]
        unsafe fn one(target: *mut Value, source: *const Value) -> bool {
            // SAFETY: the contract of the caller.
            unsafe {
                if matches!(*target, Value::String(_)) || matches!(*source, Value::String(_)) {
                    return false;
                }
                let (target, source) = (target.cast::<u8>(), source.cast::<u8>());
                target.write(source.read());
                target
                    .add(8)
                    .cast::<std::mem::MaybeUninit<u64>>()
                    .write(source.add(8).cast::<std::mem::MaybeUninit<u64>>().read());
            }
            true
        }
        let to = window.add(a);
        let from = window.sub(available).cast_const();
        let copied = count.min(available);
        let mut index = 0;
        while index + 4 <= copied {
            let (target, source) = (to.add(index), from.add(index));
            if !(one(target, source)
                && one(target.add(1), source.add(1))
                && one(target.add(2), source.add(2))
                && one(target.add(3), source.add(3)))
            {
                return copy_varargs_with_strings(window, a, wanted, available);
            }
            index += 4;
        }
        while index < copied {
            if !one(to.add(index), from.add(index)) {
                return copy_varargs_with_strings(window, a, wanted, available);
            }
            index += 1;
        }
        while index < count {
            let target = to.add(index);
            if matches!(*target, Value::String(_)) {
                return copy_varargs_with_strings(window, a, wanted, available);
            }
            std::ptr::write(target, Value::Nil);
            index += 1;
        }
    }
}

/// [`copy_varargs`] where a string is involved. Repeats the copies made
/// before the string turned up, which does no harm.
///
/// # Safety
///
/// As for [`copy_varargs`].
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
unsafe fn copy_varargs_with_strings(window: *mut Value, a: usize, wanted: u8, available: usize) {
    let wanted = usize::from(wanted);
    // SAFETY: the contract.
    unsafe {
        let to = window.add(a);
        let from = window.sub(available).cast_const();
        let copied = wanted.min(available);
        copy_values(to, from, copied);
        for index in copied..wanted {
            set_register(&mut *to.add(index), Value::Nil);
        }
    }
}

/// [`Frame::clear_open_results`], out of line.
#[cold]
#[inline(never)]
fn clear_open_results_cold(frame: &mut Frame) {
    frame.clear_open_results();
}

/// The retired `record` runs another closure than it last did: its closure,
/// chunk, environment and captures are rewritten.
///
/// # Safety
///
/// As for [`CallStack::push_plain_call`].
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn rewrite_closure(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
) {
    // SAFETY: forwarded from this function's contract.
    unsafe { install_closure(record, chunk, cells, environment, shape, closure) };
}

/// [`rewrite_closure`] in line, for the tail call, which rewrites the
/// closure of its record on every hop of a chain of different callees.
///
/// # Safety
///
/// As for [`CallStack::push_plain_call`].
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn install_closure(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
) {
    // SAFETY: forwarded from this function's contract.
    unsafe { record.chunk.rebind_borrowed(chunk) };
    record.closure = Some(closure);
    record.environment = environment;
    record.shape = shape;
    // SAFETY: forwarded from this function's contract.
    record.upvalues = unsafe { CaptureCells::borrow_shared(cells) };
}

/// The retired `record` becomes the activation [`CallStack::push_plain_call`]
/// builds, whatever it held before.
///
/// # Safety
///
/// As for [`CallStack::push_plain_call`].
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    unsafe_code
)]
#[inline(never)]
unsafe fn rewrite_record(
    record: &mut Frame,
    chunk: &Arc<Chunk>,
    cells: &ClosureCells,
    environment: TableHandle,
    shape: u32,
    closure: ClosureHandle,
    caller: CallReturn,
    base: usize,
    end: usize,
    varargs: usize,
    pc: usize,
    spare: &mut Option<Box<FrameSite>>,
) {
    // SAFETY: forwarded from this function's contract.
    unsafe { rewrite_closure(record, chunk, cells, environment, shape, closure) };
    record.base = base as u32;
    record.top = end as u32;
    record.pc = pc;
    record.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
    record.tail_calls = 0;
    record.vararg_word = varargs as u32;
    // A retired record usually has no interior state and a direct caller
    // target already: skip the drop glue then.
    if let Some(ReturnTarget::Direct(target)) = &mut record.caller {
        *target = caller;
    } else {
        record.caller = Some(ReturnTarget::Direct(caller));
    }
    if record.call_site.is_some() {
        stash_site(&mut record.call_site, spare);
    }
    if record.extras.is_some() {
        record.extras = None;
    }
}

/// Copies the value at `from` to `to`. Neither holding a string is the
/// common case: the tag and the payload word are copied as two moves, each
/// as narrow as the stores that wrote them, which lets the load forward from
/// the store buffer where a 16-byte copy cannot. A string on either side (a
/// reference count to take, or an old string to release) goes out of line,
/// so the hot path makes no call and keeps its values in registers.
///
/// # Safety
///
/// Both pointers address distinct, initialised values.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) unsafe fn copy_value(to: *mut Value, from: *const Value) {
    // `Value` is a `repr(C, u8)` enum: its tag, then a payload of at most
    // one word at offset 8.
    const _: () = assert!(std::mem::size_of::<Value>() == 16 && std::mem::align_of::<Value>() == 8);
    // SAFETY: the contract. Every variant other than a string owns no heap
    // data, so a bitwise copy is a valid second value, and `to` held none
    // either. The payload word may be partly uninitialised.
    unsafe {
        if matches!(*to, Value::String(_)) || matches!(*from, Value::String(_)) {
            copy_value_string(to, from);
            return;
        }
        let (to, from) = (to.cast::<u8>(), from.cast::<u8>());
        to.write(from.read());
        to.add(8)
            .cast::<std::mem::MaybeUninit<u64>>()
            .write(from.add(8).cast::<std::mem::MaybeUninit<u64>>().read());
    }
}

/// [`copy_value`] where a string is involved.
///
/// # Safety
///
/// As for [`copy_value`].
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
unsafe fn copy_value_string(to: *mut Value, from: *const Value) {
    // SAFETY: the contract.
    unsafe { store_clone(&mut *to, &*from) }
}

/// [`copy_value`] for `count` values in a row.
///
/// # Safety
///
/// As for [`copy_value`], for every pair in the two runs.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) unsafe fn copy_values(mut to: *mut Value, mut from: *const Value, count: usize) {
    // SAFETY: the contract keeps both runs of `count` values addressable.
    unsafe {
        let end = to.add(count);
        while to != end {
            copy_value(to, from);
            to = to.add(1);
            from = from.add(1);
        }
    }
}

/// [`clear_value`] for `count` values in a row.
///
/// # Safety
///
/// As for [`clear_value`], for every value of the run.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn clear_values(mut to: *mut Value, count: usize) {
    // SAFETY: the contract keeps the run of `count` values addressable.
    unsafe {
        let end = to.add(count);
        while to != end {
            clear_value(to);
            to = to.add(1);
        }
    }
}

/// Sets the value at `to` to nil, releasing a string it holds out of line.
///
/// # Safety
///
/// `to` addresses an initialised value.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn clear_value(to: *mut Value) {
    // SAFETY: the contract.
    unsafe {
        if matches!(*to, Value::String(_)) {
            clear_value_string(to);
        } else {
            std::ptr::write(to, Value::Nil);
        }
    }
}

/// [`clear_value`] where the old value is a string.
///
/// # Safety
///
/// As for [`clear_value`].
#[allow(unsafe_code)]
#[cold]
#[inline(never)]
unsafe fn clear_value_string(to: *mut Value) {
    // SAFETY: the contract.
    unsafe { *to = Value::Nil };
}

impl<'a> IntoIterator for &'a CallStack {
    type Item = &'a Frame;
    type IntoIter = std::slice::Iter<'a, Frame>;
    fn into_iter(self) -> Self::IntoIter {
        self.frames[..self.live].iter()
    }
}

impl<'a> IntoIterator for &'a mut CallStack {
    type Item = &'a mut Frame;
    type IntoIter = std::slice::IterMut<'a, Frame>;
    fn into_iter(self) -> Self::IntoIter {
        self.frames[..self.live].iter_mut()
    }
}

impl From<Vec<OwnedFrame>> for CallStack {
    fn from(frames: Vec<OwnedFrame>) -> Self {
        let mut stack = Self::new();
        for frame in frames {
            stack.push(frame);
        }
        stack
    }
}

#[cfg(test)]
mod tests {
    //! The layout a `Call` of a vararg closure builds, and every way it is
    //! taken down again (see [`CallStack::push_vararg_in_place`]).

    use super::*;
    use crate::bytecode::{Chunk, Instruction};
    use crate::vm::Vm;

    /// The caller's registers: the callee in `CALLEE`, five arguments after.
    const CALLER_REGISTERS: u8 = 12;
    const CALLEE: usize = 2;
    const ARGUMENTS: usize = 5;

    struct Fixture {
        vm: Vm,
        stack: CallStack,
        closure: ClosureHandle,
    }

    /// A stack holding a caller that has called nothing yet, and a vararg
    /// closure with `parameters` parameters that returns its first register.
    /// The caller's call is its instruction 0, and it reads nothing but its
    /// result afterwards.
    fn fixture(parameters: u8) -> Fixture {
        fixture_reading(parameters, 0)
    }

    /// [`fixture`] with a caller that returns register `read` after the call.
    fn fixture_reading(parameters: u8, read: u8) -> Fixture {
        // It calls itself with r2 and r3 (see the nested test), then returns
        // its first register.
        let code = vec![Instruction::call(1, 2, 1), Instruction::return_values(0, 1)];
        fixture_with(parameters, read, code)
    }

    /// [`fixture_reading`] with a callee that runs `code`.
    fn fixture_with(parameters: u8, read: u8, code: Vec<Instruction>) -> Fixture {
        let mut vm = Vm::new();
        let mut callee_chunk =
            Chunk::with_parts(6, parameters, Vec::new(), code, Vec::new(), Vec::new());
        callee_chunk.is_vararg = true;
        let closure = vm
            .runtime_mut()
            .create_closure(Arc::new(callee_chunk), Vec::new())
            .unwrap();
        // What `Vm::classify_closure_call` records for a vararg body.
        vm.runtime_mut()
            .set_closure_call_hint(closure, CALL_HINT_VARARG);
        let caller = crate::vm::Frame::new(
            Arc::new(Chunk::new(
                CALLER_REGISTERS,
                Vec::new(),
                vec![
                    Instruction::call(CALLEE as u8, ARGUMENTS as u8, 1),
                    Instruction::return_(read),
                ],
            )),
            Vec::new(),
            Vec::new(),
            None,
            None,
            vm.runtime().global_environment(),
            None,
        );
        let mut stack = CallStack::new();
        stack.push(caller);
        {
            let top = stack.at_mut(0);
            top.registers[CALLEE] = Value::Closure(closure);
            for index in 0..ARGUMENTS {
                #[allow(clippy::cast_precision_loss)]
                {
                    top.registers[CALLEE + 1 + index] = Value::Number(10.0 + index as f64);
                }
            }
        }
        Fixture { vm, stack, closure }
    }

    impl Fixture {
        /// The fast loop's call of the closure with `count` arguments.
        fn call(&mut self, count: usize, consecutive: bool) -> PlainCall {
            let live = self.stack.live;
            let frames = self.stack.frames.as_mut_ptr();
            let window = self.stack.values.as_mut_ptr();
            // SAFETY: the record at `live - 1` is the running caller, whose
            // window starts at slot 0 and holds the arguments.
            #[allow(unsafe_code)]
            let entered = unsafe {
                self.stack.push_closure_call(
                    self.vm.runtime(),
                    frames.add(live - 1),
                    window,
                    self.closure,
                    CALLEE as u8,
                    1,
                    CALLEE + 1,
                    count,
                    0,
                    consecutive,
                )
            };
            if let Entered::Fast { frame, window } = entered {
                // What the loop binds from the push is what the stack holds.
                let (top, values) = self.stack.top_raw_for_test();
                assert_eq!(frame, top);
                // SAFETY: the new top record is live.
                #[allow(unsafe_code)]
                let top = unsafe { &*top };
                assert_eq!(top.pc, 0);
                assert_eq!(window, values.wrapping_add(top.base as usize));
            }
            entered.plain()
        }

        fn caller(&self) -> &Frame {
            &self.stack.frames[0]
        }

        fn callee(&self) -> &Frame {
            &self.stack.frames[1]
        }

        // A fixture accessor beside `caller()` and `callee()`; the value is
        // the constant window size, so no receiver is needed.
        #[allow(clippy::unused_self)]
        fn full_top(&self) -> u32 {
            u32::from(CALLER_REGISTERS)
        }
    }

    fn number(value: f64) -> Value {
        Value::Number(value)
    }

    #[test]
    fn a_call_leaves_the_extra_arguments_where_they_are() {
        let mut fixture = fixture(2);
        assert!(fixture.call(ARGUMENTS, true) == PlainCall::Pushed);
        let first = (CALLEE + 1) as u32;
        // The caller's window ends just before the extras (two parameters),
        // the callee's starts right after them, and its parameters are copies.
        assert_eq!(fixture.caller().top, first + 2);
        assert_eq!(fixture.callee().base, first + ARGUMENTS as u32);
        assert_eq!(fixture.callee().vararg_count(), 3);
        assert!(fixture.callee().vararg_word & VARARG_TRUNCATED != 0);
        let callee = fixture.stack.at(1);
        assert_eq!(callee.varargs, &[number(12.0), number(13.0), number(14.0)]);
        assert_eq!(callee.registers[..2], [number(10.0), number(11.0)]);
    }

    /// A callee that starts with `Vararg` has it run by the call: the locals
    /// hold the first extra arguments (nil past the last), the old contents
    /// of those registers are released, and the callee starts after it.
    #[test]
    fn a_call_runs_the_entry_vararg_and_starts_after_it() {
        // One parameter; the four extra arguments are 11..14.
        let code = vec![Instruction::vararg(1, 5), Instruction::return_values(0, 1)];
        let mut fixture = fixture_with(1, 0, code);
        // Where the callee's window will start, a string from an earlier call.
        let base = CALLEE + 1 + ARGUMENTS;
        fixture.stack.values.grow(base + REGISTER_WINDOW);
        fixture.stack.values[base + 2] = Value::from("old");
        fixture.stack.values[base + 5] = Value::from("older");
        assert!(fixture.call(ARGUMENTS, true) == PlainCall::Pushed);
        assert_eq!(fixture.callee().pc, 1);
        assert_eq!(fixture.callee().vararg_count(), 4);
        let callee = fixture.stack.at(1);
        assert_eq!(
            callee.registers[..6],
            [
                number(10.0),
                number(11.0),
                number(12.0),
                number(13.0),
                number(14.0),
                Value::Nil
            ]
        );
        // The extra arguments are still there for later uses of `...`.
        assert_eq!(
            callee.varargs,
            &[number(11.0), number(12.0), number(13.0), number(14.0)]
        );
    }

    /// Fewer locals than extra arguments takes the first ones only, and a
    /// callee whose first instruction is anything else starts at 0.
    #[test]
    fn the_entry_vararg_takes_the_first_extra_arguments_only() {
        let code = vec![Instruction::vararg(2, 2), Instruction::return_values(0, 1)];
        let mut fixture = fixture_with(0, 0, code);
        assert!(fixture.call(ARGUMENTS, true) == PlainCall::Pushed);
        assert_eq!(fixture.callee().pc, 1);
        let callee = fixture.stack.at(1);
        assert_eq!(callee.registers[2..4], [number(10.0), number(11.0)]);
        assert_eq!(callee.varargs.len(), ARGUMENTS);

        let mut fixture = fixture_reading(0, 0);
        assert!(fixture.call(ARGUMENTS, true) == PlainCall::Pushed);
        assert_eq!(fixture.callee().pc, 0);
    }

    #[test]
    fn a_plain_return_gives_the_caller_its_window_back() {
        let mut fixture = fixture(2);
        fixture.call(ARGUMENTS, true);
        let live = fixture.stack.live;
        // SAFETY: the callee's record is the top one, and its window starts
        // at its base.
        #[allow(unsafe_code)]
        let popped = unsafe {
            let frame = fixture.stack.frames.as_mut_ptr().add(live - 1);
            let window = fixture
                .stack
                .values
                .as_mut_ptr()
                .add((*frame).base as usize);
            fixture.stack.pop_plain_return(frame, window, 0, 1)
        };
        assert!(popped.is_some());
        assert_eq!(fixture.stack.live, 1);
        assert_eq!(fixture.caller().top, fixture.full_top());
        // The result went to the callee's register in the caller.
        assert_eq!(fixture.stack.at(0).registers[CALLEE], number(10.0));
    }

    #[test]
    fn a_vararg_return_stages_its_results_over_the_old_varargs() {
        let mut fixture = fixture(0);
        fixture.call(ARGUMENTS, true);
        assert_eq!(fixture.callee().vararg_count(), ARGUMENTS);
        // `return r0, ...` with four results wanted by the caller at CALLEE:
        // the results overlap the registers that held the varargs.
        fixture.stack.frames[1].caller = Some(ReturnTarget::Direct(CallReturn {
            base: CALLEE as u8,
            results: 4,
        }));
        fixture.stack.at_mut(1).registers[0] = number(99.0);
        assert!(fixture.stack.pop_vararg_return(0, 1));
        assert_eq!(fixture.stack.live, 1);
        assert_eq!(fixture.caller().top, fixture.full_top());
        let caller = fixture.stack.at(0);
        assert_eq!(
            caller.registers[CALLEE..CALLEE + 4],
            [number(99.0), number(10.0), number(11.0), number(12.0)]
        );
    }

    #[test]
    fn every_way_of_removing_the_callee_restores_the_caller() {
        for how in ["pop", "discard", "retire", "split"] {
            let mut fixture = fixture(1);
            fixture.call(ARGUMENTS, true);
            assert!(fixture.caller().top < fixture.full_top(), "{how}");
            match how {
                "pop" => {
                    let owned = fixture.stack.pop().unwrap();
                    // The detached frame forgets the cut.
                    assert_eq!(owned.varargs.len(), 4);
                    assert_eq!(owned.frame.vararg_word & VARARG_TRUNCATED, 0);
                }
                "discard" => fixture.stack.discard_top(),
                "retire" => fixture.stack.retire_top(),
                _ => {
                    let rest = fixture.stack.split_off(1);
                    assert_eq!(rest.live, 1);
                    assert_eq!(rest.frames[0].vararg_count(), 4);
                    assert_eq!(rest.frames[0].vararg_word & VARARG_TRUNCATED, 0);
                    // Its extras went with it.
                    assert_eq!(rest.at(0).varargs[0], number(11.0));
                }
            }
            assert_eq!(fixture.stack.live, 1, "{how}");
            assert_eq!(fixture.caller().top, fixture.full_top(), "{how}");
            // The caller's whole window is addressable again.
            assert!(
                fixture.stack.values.len() >= fixture.full_top() as usize,
                "{how}"
            );
            assert_eq!(
                fixture.stack.at(0).registers.len(),
                usize::from(CALLER_REGISTERS)
            );
        }
    }

    #[test]
    fn a_caller_that_reads_a_register_above_the_arguments_keeps_its_window() {
        // The caller reads r11 after the call, above the arguments (r3..r7):
        // the callee's window may not take it, so the extras are copied.
        let mut fixture = fixture_reading(0, 11);
        assert!(fixture.call(ARGUMENTS, true) == PlainCall::Pushed);
        assert_eq!(fixture.caller().top, fixture.full_top());
        assert_eq!(fixture.callee().vararg_word & VARARG_TRUNCATED, 0);
        assert_eq!(fixture.callee().vararg_count(), ARGUMENTS);
        assert!(fixture.callee().base >= fixture.full_top() + ARGUMENTS as u32);
        // Reading r7, the last argument's register, is as bad as reading r11.
        let mut fixture = fixture_reading(0, 7);
        fixture.call(ARGUMENTS, true);
        assert_eq!(fixture.caller().top, fixture.full_top());
        // Reading the call's own result register is what every caller does.
        let mut fixture = fixture_reading(0, CALLEE as u8);
        fixture.call(ARGUMENTS, true);
        assert!(fixture.caller().top < fixture.full_top());
    }

    #[test]
    fn calls_without_extra_arguments_or_registers_to_spare_keep_the_caller_whole() {
        // As many arguments as parameters: nothing to leave behind.
        let mut fixture = fixture(5);
        fixture.call(ARGUMENTS, true);
        assert_eq!(fixture.caller().top, fixture.full_top());
        assert_eq!(fixture.callee().vararg_count(), 0);

        // A `CallOne` names its argument registers one by one, and registers
        // above them may be live: the extras are copied.
        let mut fixture = self::fixture(0);
        fixture.call(1, false);
        assert_eq!(fixture.caller().top, fixture.full_top());
        assert_eq!(fixture.callee().vararg_count(), 1);
        assert_eq!(fixture.stack.at(1).varargs, &[number(10.0)]);
    }

    #[test]
    fn nested_calls_cut_and_restore_each_caller_in_turn() {
        let mut fixture = fixture(0);
        fixture.call(ARGUMENTS, true);
        // The callee calls itself with two of its registers as arguments.
        {
            let callee = fixture.stack.at_mut(1);
            callee.registers[1] = Value::Closure(fixture.closure);
            callee.registers[2] = number(7.0);
            callee.registers[3] = number(8.0);
        }
        let live = fixture.stack.live;
        let frames = fixture.stack.frames.as_mut_ptr();
        let window = fixture.stack.values.as_mut_ptr();
        let base = fixture.callee().base as usize;
        // SAFETY: as for `Fixture::call`, for the callee as the running frame.
        #[allow(unsafe_code)]
        let pushed = unsafe {
            fixture.stack.push_closure_call(
                fixture.vm.runtime(),
                frames.add(live - 1),
                window.add(base),
                fixture.closure,
                1,
                1,
                2,
                2,
                0,
                true,
            )
        };
        assert!(pushed.plain() == PlainCall::Pushed);
        assert_eq!(fixture.stack.live, 3);
        assert!(fixture.stack.frames[0].top < fixture.full_top());
        assert!(fixture.stack.frames[1].top < fixture.stack.frames[1].base + 6);
        fixture.stack.discard_top();
        assert_eq!(
            fixture.stack.frames[1].top,
            fixture.stack.frames[1].base + 6
        );
        assert!(fixture.stack.frames[0].top < fixture.full_top());
        fixture.stack.discard_top();
        assert_eq!(fixture.caller().top, fixture.full_top());
    }
}

#[cfg(test)]
mod plain_call_tests {
    //! The fast loop's plain call: the push that calls nothing and hands
    //! back the new record and window, the retarget of a record that ran
    //! another closure, the general push for what the first declines, and
    //! the return that hands back the caller's window.

    use super::*;
    use crate::bytecode::{Chunk, Instruction};
    use crate::vm::{ActiveCallSite, Vm};

    const CALLEE: usize = 2;

    struct Plain {
        vm: Vm,
        stack: CallStack,
        /// Two closures with different chunks, so a record that ran one
        /// differs from a call of the other.
        closures: [ClosureHandle; 2],
        /// How many calls the marked push took.
        marked_pushes: usize,
    }

    fn closure_with(vm: &mut Vm, parameters: u8, registers: u8) -> ClosureHandle {
        let chunk = Chunk::with_parts(
            registers,
            parameters,
            Vec::new(),
            vec![Instruction::return_values(0, 1)],
            Vec::new(),
            Vec::new(),
        );
        let closure = vm
            .runtime_mut()
            .create_closure(Arc::new(chunk), Vec::new())
            .unwrap();
        vm.runtime_mut()
            .set_closure_call_hint(closure, CALL_HINT_PLAIN);
        closure
    }

    fn plain_fixture() -> Plain {
        let mut vm = Vm::new();
        let closures = [closure_with(&mut vm, 2, 5), closure_with(&mut vm, 1, 7)];
        let caller = crate::vm::Frame::new(
            Arc::new(Chunk::new(
                12,
                Vec::new(),
                vec![
                    Instruction::call(CALLEE as u8, 2, 1),
                    Instruction::return_(0),
                ],
            )),
            Vec::new(),
            Vec::new(),
            None,
            None,
            vm.runtime().global_environment(),
            None,
        );
        let mut stack = CallStack::new();
        stack.push(caller);
        {
            let top = stack.at_mut(0);
            top.registers[CALLEE + 1] = Value::Number(10.0);
            top.registers[CALLEE + 2] = Value::Number(11.0);
        }
        Plain {
            vm,
            stack,
            closures,
            marked_pushes: 0,
        }
    }

    impl Plain {
        /// The fast loop's call of closure `which` with the two arguments
        /// after the callee register.
        fn call(&mut self, which: usize, count: usize) -> Entered {
            let live = self.stack.live;
            self.stack.at_mut(live - 1).registers[CALLEE] = Value::Closure(self.closures[which]);
            let frames = self.stack.frames.as_mut_ptr();
            let window = self.stack.values.as_mut_ptr();
            let base = self.stack.frames[live - 1].base as usize;
            // SAFETY: the record at `live - 1` is the running frame, `window`
            // the start of the stack and `base` where its window begins.
            // The marked push first and the general one for what it
            // declines, as the loop calls them.
            #[allow(unsafe_code)]
            unsafe {
                let caller = frames.add(live - 1);
                let entered = self.stack.push_marked_call_at(
                    self.vm.runtime(),
                    caller,
                    window.add(base),
                    self.closures[which],
                    CALLEE as u8,
                    1,
                    CALLEE + 1,
                    count,
                    0,
                );
                let (entered, chunk) = entered;
                assert_eq!(entered.is_null(), chunk.is_null());
                self.marked_pushes += usize::from(!entered.is_null());
                if !entered.is_null() {
                    assert_eq!(chunk, (*caller.add(1)).chunk.address());
                    return Entered::Fast {
                        frame: caller.add(1),
                        window: entered,
                    };
                }
                self.stack.push_closure_call(
                    self.vm.runtime(),
                    caller,
                    window.add(base),
                    self.closures[which],
                    CALLEE as u8,
                    1,
                    CALLEE + 1,
                    count,
                    0,
                    true,
                )
            }
        }

        /// Retires the top frame the way the loop's plain return does.
        fn retire(&mut self) -> Option<*mut Value> {
            let live = self.stack.live;
            // SAFETY: the top record is at `live - 1`, its window starts at
            // its base.
            #[allow(unsafe_code)]
            unsafe {
                let frame = self.stack.frames.as_mut_ptr().add(live - 1);
                let window = self.stack.values.as_mut_ptr().add((*frame).base as usize);
                self.stack.pop_plain_return(frame, window, 0, 1)
            }
        }

        fn top(&self) -> &Frame {
            &self.stack.frames[self.stack.live - 1]
        }
    }

    /// Where the new activation stands, as the loop will bind it.
    fn assert_bound(plain: &mut Plain, entered: Entered) {
        let Entered::Fast { frame, window } = entered else {
            panic!("the push should have been plain");
        };
        let (top, values) = plain.stack.top_raw_for_test();
        assert_eq!(frame, top);
        // SAFETY: the pushed record is the live top.
        #[allow(unsafe_code)]
        let record = unsafe { &*top };
        assert_eq!(record.pc, 0);
        assert!(record.last_interpreted_pc.is_none());
        assert_eq!(record.tail_calls, 0);
        assert_eq!(record.vararg_count(), 0);
        assert_eq!(window, values.wrapping_add(record.base as usize));
    }

    #[test]
    fn a_repeated_call_reuses_the_retired_record_and_reports_its_binding() {
        let mut plain = plain_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.stack.live, 2);
        let record = std::ptr::from_ref::<Frame>(plain.top());
        assert_eq!(
            plain.stack.at(1).registers[..2],
            [Value::Number(10.0), Value::Number(11.0)]
        );
        let window = plain.retire().expect("a plain return");
        assert_eq!(window, plain.stack.values.as_mut_ptr().wrapping_add(0));
        assert_eq!(plain.stack.live, 1);

        // The second call finds the record where the first left it, running
        // the same closure.
        plain.stack.at_mut(0).registers[CALLEE + 1] = Value::Number(20.0);
        let entered = plain.call(0, 1);
        assert_bound(&mut plain, entered);
        assert!(std::ptr::eq(plain.top(), record));
        // The parameter the call left out starts nil.
        assert_eq!(
            plain.stack.at(1).registers[..2],
            [Value::Number(20.0), Value::Nil]
        );
    }

    #[test]
    fn a_record_that_ran_another_closure_is_pointed_at_the_new_one() {
        let mut plain = plain_fixture();
        let first = plain.call(0, 2);
        assert_bound(&mut plain, first);
        let chunk_a = plain.top().chunk.address();
        plain.retire().unwrap();

        let second = plain.call(1, 1);
        assert_bound(&mut plain, second);
        assert_eq!(plain.top().closure, Some(plain.closures[1]));
        assert_ne!(plain.top().chunk.address(), chunk_a);
        assert!(!plain.top().chunk.is_owned());
        // The new chunk has seven registers.
        assert_eq!(plain.top().top - plain.top().base, 7);
        plain.retire().unwrap();

        // And back.
        let third = plain.call(0, 2);
        assert_bound(&mut plain, third);
        assert_eq!(plain.top().closure, Some(plain.closures[0]));
        assert_eq!(plain.top().chunk.address(), chunk_a);
        assert_eq!(plain.top().top - plain.top().base, 5);
    }

    #[test]
    fn a_string_argument_takes_the_general_push_with_the_same_result() {
        let mut plain = plain_fixture();
        plain.stack.at_mut(0).registers[CALLEE + 1] = Value::from("alpha");
        // A string left in the callee's window by an earlier activation, in
        // the slot the argument goes to.
        let first = plain.call(0, 2);
        assert_bound(&mut plain, first);
        plain.stack.at_mut(1).registers[1] = Value::from("stale");
        plain.retire().unwrap();

        let second = plain.call(0, 2);
        assert_bound(&mut plain, second);
        assert_eq!(plain.stack.at(1).registers[0], Value::from("alpha"));
        // The stale string was released, and the argument left in the caller
        // is still there.
        assert_eq!(plain.stack.at(1).registers[1], Value::Number(11.0));
        assert_eq!(
            plain.stack.at(0).registers[CALLEE + 1],
            Value::from("alpha")
        );
    }

    #[test]
    fn a_record_holding_a_named_site_is_rewritten_by_the_general_push() {
        let mut plain = plain_fixture();
        let first = plain.call(0, 2);
        assert_bound(&mut plain, first);
        plain.retire().unwrap();
        let index = plain.stack.live;
        plain.stack.frames[index].call_site = Some(Box::new(FrameSite::Named(ActiveCallSite {
            name: "old".into(),
            kind: crate::bytecode::CallKind::Global,
        })));
        let second = plain.call(0, 2);
        assert_bound(&mut plain, second);
        assert!(!holds_named_site(plain.top()));
    }

    #[test]
    fn the_return_hands_back_the_callers_window() {
        let mut plain = plain_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        let caller_base = plain.stack.frames[0].base as usize;
        let expected = plain.stack.values.as_mut_ptr().wrapping_add(caller_base);
        assert_eq!(plain.retire(), Some(expected));
        assert_eq!(plain.stack.live, 1);
    }

    impl Plain {
        /// Retires the top frame, which carries the plain mark.
        fn retire_marked(&mut self) -> Option<*mut Value> {
            assert!(self.top().plain_marked());
            let live = self.stack.live;
            // SAFETY: as for `retire`.
            #[allow(unsafe_code)]
            let window = unsafe {
                let frame = self.stack.frames.as_mut_ptr().add(live - 1);
                let window = self.stack.values.as_mut_ptr().add((*frame).base as usize);
                self.stack.pop_marked_return(frame, window, 0)
            };
            (!window.is_null()).then_some(window)
        }

        /// The retired record a call would take next.
        fn retired(&mut self) -> &mut Frame {
            let live = self.stack.live;
            &mut self.stack.frames[live]
        }
    }

    /// A fixture whose depth has run closure 0 once. The first call at a
    /// depth takes the general push, which computes the chunk's entry count
    /// and sets no mark; the calls after it are the ones under test.
    fn warm_fixture() -> Plain {
        let mut plain = plain_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert!(!plain.top().plain_marked());
        plain.retire().unwrap();
        plain
    }

    #[test]
    fn the_plain_push_marks_its_record_and_the_mark_survives_the_return() {
        let mut plain = warm_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert!(plain.top().plain_marked());
        assert!(plain.top().plain_mark_holds());
        // The mark is no vararg and no cut caller.
        assert_eq!(plain.top().vararg_count(), 0);
        assert!(!plain.top().has_vararg_state());
        let caller_base = plain.stack.frames[0].base as usize;
        let expected = plain.stack.values.as_mut_ptr().wrapping_add(caller_base);
        assert_eq!(plain.retire_marked(), Some(expected));
        assert_eq!(plain.stack.live, 1);
        assert!(plain.retired().plain_marked());
        // The value went to the register the call named.
        assert_eq!(plain.stack.at(0).registers[CALLEE], Value::Number(10.0));

        // The push into the marked record: same record, same binding, and
        // it writes what the push into an unmarked one writes.
        plain.stack.at_mut(0).registers[CALLEE + 1] = Value::Number(20.0);
        let entered = plain.call(0, 1);
        assert_bound(&mut plain, entered);
        assert!(plain.top().plain_mark_holds());
        assert_eq!(plain.top().top - plain.top().base, 5);
        assert_eq!(
            plain.stack.at(1).registers[..2],
            [Value::Number(20.0), Value::Nil]
        );
        assert!(matches!(
            plain.top().caller,
            Some(ReturnTarget::Direct(CallReturn {
                base: 2,
                results: 1
            }))
        ));
    }

    #[test]
    fn another_callee_takes_over_a_marked_record() {
        let mut plain = warm_fixture();
        // The other closure's first call computes its entry count.
        let first = plain.call(1, 1);
        assert_bound(&mut plain, first);
        plain.retire().unwrap();
        let second = plain.call(1, 1);
        assert_bound(&mut plain, second);
        assert!(plain.top().plain_mark_holds());
        assert_eq!(plain.top().top - plain.top().base, 7);
        plain.retire_marked().unwrap();

        // The record carries the mark and ran the one; the other is called.
        let third = plain.call(0, 2);
        assert_bound(&mut plain, third);
        assert!(plain.top().plain_mark_holds());
        assert_eq!(plain.top().closure, Some(plain.closures[0]));
        assert_eq!(plain.top().top - plain.top().base, 5);
        assert!(!plain.top().chunk.is_owned());
        assert_eq!(
            plain.stack.at(1).registers[..2],
            [Value::Number(10.0), Value::Number(11.0)]
        );
    }

    #[test]
    fn a_string_leaves_the_push_into_a_marked_record_with_the_same_result() {
        let mut plain = warm_fixture();
        let first = plain.call(0, 2);
        assert_bound(&mut plain, first);
        plain.stack.at_mut(1).registers[1] = Value::from("stale");
        plain.retire_marked().unwrap();
        plain.stack.at_mut(0).registers[CALLEE + 1] = Value::from("alpha");
        let second = plain.call(0, 2);
        assert_bound(&mut plain, second);
        assert_eq!(plain.stack.at(1).registers[0], Value::from("alpha"));
        assert_eq!(plain.stack.at(1).registers[1], Value::Number(11.0));
        assert!(plain.top().plain_mark_holds());
    }

    #[test]
    fn whatever_touches_a_marked_record_takes_the_mark_off() {
        fn marked() -> Plain {
            let mut plain = warm_fixture();
            let entered = plain.call(0, 2);
            assert_bound(&mut plain, entered);
            assert!(plain.top().plain_marked());
            plain
        }
        // Interior state.
        let mut plain = marked();
        let live = plain.stack.live;
        let _ = plain.stack.at_mut(live - 1).extras();
        assert!(!plain.top().plain_marked());

        // A frame taken off the stack and put back.
        let mut plain = marked();
        let owned = plain.stack.pop().unwrap();
        assert!(!owned.plain_marked());
        plain.stack.push(owned);
        assert!(!plain.top().plain_marked());

        // Frames split into a stack of their own.
        let mut plain = marked();
        let split = plain.stack.split_off(1);
        assert!(!split.frames[0].plain_marked());

        // A general push over a retired, marked record.
        let mut plain = marked();
        plain.retire_marked().unwrap();
        assert!(plain.retired().plain_marked());
        let closure = plain.closures[0];
        let parts = plain.vm.runtime().closure_call_parts(closure).unwrap();
        let (chunk, cells) = (parts.chunk.clone(), parts.cells);
        let environment = parts.environment;
        // SAFETY: the parts are the closure's own, which the fixture keeps.
        #[allow(unsafe_code)]
        unsafe {
            plain.stack.push_activation_to(
                &chunk,
                cells,
                environment,
                closure,
                ReturnTarget::SetMetamethod,
                ActivationArguments::None,
            );
        }
        assert!(!plain.top().plain_marked());

        // A tail call out of the marked activation.
        let mut plain = marked();
        let live = plain.stack.live;
        plain.stack.at_mut(live - 1).registers[0] = Value::Closure(plain.closures[1]);
        let step = plain.stack.tail_call_closure(
            plain.vm.runtime(),
            Instruction::abc(crate::bytecode::OpCode::TailCall, 0, 0, 0),
            0,
        );
        assert!(matches!(step, TailStep::Replaced | TailStep::Moved));
        assert!(!plain.top().plain_marked());
        assert_eq!(plain.top().tail_calls, 1);
    }

    #[test]
    fn a_record_that_lost_its_mark_is_checked_again_before_it_gets_one() {
        let mut plain = warm_fixture();
        let first = plain.call(0, 2);
        assert_bound(&mut plain, first);
        plain.retire_marked().unwrap();
        // A named site and no mark, as a tail call leaves a record.
        let retired = plain.retired();
        retired.clear_plain_mark();
        retired.call_site = Some(Box::new(FrameSite::Named(ActiveCallSite {
            name: "old".into(),
            kind: crate::bytecode::CallKind::Global,
        })));
        let second = plain.call(0, 2);
        assert_bound(&mut plain, second);
        assert!(!holds_named_site(plain.top()));
        // The general push rewrote it; the call after that marks it.
        assert!(!plain.top().plain_marked());
        plain.retire().unwrap();
        let third = plain.call(0, 2);
        assert_bound(&mut plain, third);
        assert!(plain.top().plain_mark_holds());
    }

    #[test]
    fn the_return_of_a_marked_frame_declines_what_it_cannot_do() {
        // A local of the callee is still captured.
        let mut plain = warm_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        let base = plain.top().base;
        plain.stack.open_top = base + 1;
        assert_eq!(plain.retire_marked(), None);
        assert_eq!(plain.stack.live, 2);
        plain.stack.open_top = 0;

        // A call that wants no fixed results.
        let Some(ReturnTarget::Direct(target)) = &mut plain.stack.frames[1].caller else {
            panic!("a marked record returns to a direct caller");
        };
        target.results = 0;
        assert_eq!(plain.retire_marked(), None);
        assert_eq!(plain.stack.live, 2);
    }

    #[test]
    fn the_marked_push_takes_the_calls_into_a_marked_record_and_no_other() {
        let mut plain = warm_fixture();
        // The record ran the closure but carries no mark yet: the general
        // push, which sets it.
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.marked_pushes, 0);
        plain.retire_marked().unwrap();
        // Marked: the same closure, then the other one once its entry
        // count is known, then the first again.
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.marked_pushes, 1);
        plain.retire_marked().unwrap();
        let entered = plain.call(1, 1);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.marked_pushes, 1, "no entry count is cached yet");
        plain.retire().unwrap();
        let entered = plain.call(1, 1);
        assert_bound(&mut plain, entered);
        plain.retire_marked().unwrap();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.top().closure, Some(plain.closures[0]));
        assert!(plain.top().plain_mark_holds());
        let pushes = plain.marked_pushes;
        assert!(pushes >= 2);
        plain.retire_marked().unwrap();

        // A retired record without the mark.
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        plain.retire().unwrap();
        plain.retired().clear_plain_mark();
        let pushes = plain.marked_pushes;
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert_eq!(plain.marked_pushes, pushes);
        plain.retire().unwrap();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        plain.retire_marked().unwrap();

        // A closure whose calls are not known to be plain, into a marked
        // record: the general push decides what it is.
        let other = plain.closures[1];
        plain
            .vm
            .runtime_mut()
            .set_closure_call_hint(other, CALL_HINT_UNKNOWN);
        let pushes = plain.marked_pushes;
        let _ = plain.call(1, 1);
        assert_eq!(plain.marked_pushes, pushes);
    }

    #[test]
    fn the_marked_push_declines_at_the_last_record() {
        let mut plain = warm_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        plain.retire_marked().unwrap();
        // No record after the caller's: the general push grows the stack.
        plain.stack.frames.truncate(1);
        let entered = plain.call(0, 2);
        assert_eq!(plain.marked_pushes, 0);
        assert_bound(&mut plain, entered);
    }

    #[test]
    fn the_marked_return_leaves_an_unmarked_frame_alone() {
        let mut plain = plain_fixture();
        let entered = plain.call(0, 2);
        assert_bound(&mut plain, entered);
        assert!(!plain.top().plain_marked());
        let live = plain.stack.live;
        // SAFETY: the top record and its window.
        #[allow(unsafe_code)]
        let window = unsafe {
            let frame = plain.stack.frames.as_mut_ptr().add(live - 1);
            let window = plain.stack.values.as_mut_ptr().add((*frame).base as usize);
            plain.stack.pop_marked_return(frame, window, 0)
        };
        assert!(window.is_null());
        assert_eq!(plain.stack.live, 2);
    }

    #[test]
    fn a_return_that_cannot_be_plain_hands_back_nothing() {
        let mut plain = plain_fixture();
        // The bottom frame has no direct caller to return to.
        assert_eq!(plain.retire(), None);
        assert_eq!(plain.stack.live, 1);
    }
}
