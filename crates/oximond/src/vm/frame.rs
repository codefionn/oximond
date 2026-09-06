//! Call-frame accessors: lazily allocated frame extras, varargs, open results, and captured slots.

use crate::bytecode::Chunk;
use crate::runtime::{CaptureCells, ClosureHandle, TableHandle, Value};
use crate::vm::continuations::reset_registers;
use crate::vm::stack::OwnedFrame;
use crate::vm::{ActiveCallSite, Frame, FrameExtras, ReturnTarget};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;

/// The chunk a frame runs, in one word so `Frame` keeps its size.
///
/// Frames pushed for a closure call borrow the chunk from that closure,
/// which the frame roots for as long as it is live, so a call and its return
/// do no atomic reference-count work. Frames without a closure own a
/// reference instead; the low pointer bit tells the two apart.
pub(crate) struct FrameChunk {
    tagged: NonNull<Chunk>,
}

/// Set in `Frame::vararg_count` of an activation whose direct caller's window
/// ends where this activation's varargs begin, short of the caller's own
/// `register_count` registers: a call of a vararg function leaves the
/// arguments past its parameters where the caller put them, in the registers
/// above the call that are dead anyway. Whatever removes the activation
/// gives the caller its full window back (`CallStack::restore_caller_top`).
pub(crate) const VARARG_TRUNCATED: u32 = 1 << 31;

/// The plain mark in the vararg word of a frame record, which then holds
/// nothing else. See the field for what it promises.
pub(crate) const VARARG_PLAIN: u32 = 1 << 30;

const OWNED: usize = 1;
const _: () = assert!(std::mem::align_of::<Chunk>() >= 2);

// SAFETY: a borrowed pointer is only dereferenced while the owning closure is
// rooted by the frame, and `Chunk` is `Send + Sync` behind its `Arc`.
#[allow(unsafe_code)]
unsafe impl Send for FrameChunk {}
// SAFETY: see the `Send` impl; shared access only reads the chunk.
#[allow(unsafe_code)]
unsafe impl Sync for FrameChunk {}

/// A non-counting `&Arc<Chunk>` view of a [`FrameChunk`].
pub(crate) struct ChunkArc(ManuallyDrop<Arc<Chunk>>);

impl std::ops::Deref for ChunkArc {
    type Target = Arc<Chunk>;

    #[inline(always)]
    fn deref(&self) -> &Arc<Chunk> {
        &self.0
    }
}

#[allow(unsafe_code)]
impl FrameChunk {
    /// Borrows `chunk` from the closure that owns it.
    ///
    /// # Safety
    ///
    /// The `Arc` behind `chunk` must stay alive for as long as this value is
    /// dereferenced. The VM roots the closure that owns it in the frame
    /// record holding this value, and a record is only read while live: the
    /// stale record above a stack's live depth keeps a pointer that is
    /// overwritten but never dereferenced. Coroutine stacks, unwinding and
    /// JIT entry and exit move whole records, closure root included, so they
    /// preserve this.
    #[inline(always)]
    pub(crate) unsafe fn borrowed(chunk: &Arc<Chunk>) -> Self {
        Self {
            tagged: NonNull::from(&**chunk),
        }
    }

    /// The tagged word of this value, to compare against
    /// [`Self::borrowed_tag`].
    #[inline(always)]
    pub(crate) fn tag(&self) -> usize {
        self.tagged.as_ptr() as usize
    }

    /// The word a value borrowing `chunk` holds.
    #[inline(always)]
    pub(crate) fn borrowed_tag(chunk: &Arc<Chunk>) -> usize {
        Arc::as_ptr(chunk) as usize
    }

    /// Whether this value holds its own reference to the chunk.
    #[inline(always)]
    pub(crate) fn is_owned(&self) -> bool {
        self.tagged.as_ptr() as usize & OWNED != 0
    }

    /// Makes this value borrow `chunk`, releasing the reference it owned, if
    /// any. [`Self::borrowed`] and an assignment, with the release in line.
    ///
    /// # Safety
    ///
    /// As for [`Self::borrowed`].
    #[inline(always)]
    pub(crate) unsafe fn rebind_borrowed(&mut self, chunk: &Arc<Chunk>) {
        if self.is_owned() {
            // SAFETY: an owned value holds the reference `Arc::into_raw`
            // leaked in `From<Arc<Chunk>>`, released once, here.
            unsafe { drop(Arc::from_raw(self.address())) };
        }
        self.tagged = NonNull::from(&**chunk);
    }

    /// A second borrowed view of the chunk, or `None` when this value owns a
    /// reference (the copy would outlive nothing that keeps the chunk alive).
    ///
    /// # Safety
    ///
    /// The caller keeps the closure that owns the chunk rooted for as long
    /// as the copy is dereferenced, as [`Self::borrowed`] requires.
    #[inline(always)]
    pub(crate) unsafe fn borrowed_copy(&self) -> Option<Self> {
        if self.is_owned() {
            None
        } else {
            Some(Self {
                tagged: self.tagged,
            })
        }
    }

    /// [`Self::borrowed_copy`] for a value known to borrow.
    ///
    /// # Safety
    ///
    /// As for [`Self::borrowed_copy`], and this value does not own a
    /// reference to the chunk ([`Self::is_owned`] is false).
    #[inline(always)]
    pub(crate) unsafe fn borrowed_copy_of_borrowed(&self) -> Self {
        debug_assert!(!self.is_owned());
        Self {
            tagged: self.tagged,
        }
    }

    /// [`Self::address`] of a value that borrows its chunk: the word itself,
    /// with no tag to take off.
    #[inline(always)]
    pub(crate) fn borrowed_address(&self) -> *const Chunk {
        debug_assert!(!self.is_owned());
        self.tagged.as_ptr().cast_const()
    }

    /// The chunk's address, for identity comparisons.
    #[inline(always)]
    pub(crate) fn address(&self) -> *const Chunk {
        (self.tagged.as_ptr() as usize & !OWNED) as *const Chunk
    }

    /// The chunk as a `&Arc<Chunk>` without touching the reference count.
    /// Cloning the result takes a counted reference as usual.
    #[inline(always)]
    pub(crate) fn as_arc(&self) -> ChunkArc {
        // SAFETY: the address came from `Arc::as_ptr` of a live `Arc<Chunk>`
        // (owned, or kept alive by the rooted closure), and the view never
        // drops, so the count stays balanced.
        ChunkArc(ManuallyDrop::new(unsafe { Arc::from_raw(self.address()) }))
    }
}

impl From<Arc<Chunk>> for FrameChunk {
    #[inline(always)]
    fn from(chunk: Arc<Chunk>) -> Self {
        let pointer = Arc::into_raw(chunk) as usize | OWNED;
        Self {
            // SAFETY: `Arc::into_raw` is never null; the tag only sets the
            // low bit of an aligned address.
            #[allow(unsafe_code)]
            tagged: unsafe { NonNull::new_unchecked(pointer as *mut Chunk) },
        }
    }
}

impl From<Chunk> for FrameChunk {
    fn from(chunk: Chunk) -> Self {
        Arc::new(chunk).into()
    }
}

impl std::ops::Deref for FrameChunk {
    type Target = Chunk;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn deref(&self) -> &Chunk {
        // SAFETY: owned values hold a reference; borrowed ones are kept alive
        // by the rooted closure (see `FrameChunk::borrowed`).
        unsafe { &*self.address() }
    }
}

impl Drop for FrameChunk {
    #[inline(always)]
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.is_owned() {
            // SAFETY: an owned value came from `Arc::into_raw` and is
            // released once, here.
            unsafe { drop(Arc::from_raw(self.address())) };
        }
    }
}

impl Frame {
    /// A fresh Lua activation record without interior state. Its register
    /// window is assigned when it is pushed.
    #[inline]
    pub(crate) fn activation(
        chunk: FrameChunk,
        upvalues: CaptureCells,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: ClosureHandle,
    ) -> Self {
        Self {
            chunk,
            closure: Some(closure),
            environment,
            base: 0,
            top: 0,
            upvalues,
            pc: 0,
            last_interpreted_pc: LastPc::NONE,
            caller,
            call_site: call_site.map(crate::vm::FrameSite::boxed),
            tail_calls: 0,
            vararg_word: 0,
            shape: crate::runtime::NO_SHAPE,
            open: None,
            extras: None,
        }
    }

    /// Interior state, creating it on first use.
    ///
    /// Frames that never touch varargs, captures, close slots, native-resume
    /// bookkeeping, or hook scratch state keep this allocation out of their
    /// create, move, and drop paths entirely.
    pub(crate) fn extras(&mut self) -> &mut FrameExtras {
        self.clear_plain_mark();
        self.extras.get_or_insert_with(Box::default)
    }

    /// Whether this record carries the plain mark (see the vararg word).
    #[inline(always)]
    pub(crate) fn plain_marked(&self) -> bool {
        self.vararg_word == VARARG_PLAIN
    }

    /// Takes the plain mark off this record: something is about to change
    /// that the fast loop's plain call or return would have to look at.
    #[inline(always)]
    pub(crate) fn clear_plain_mark(&mut self) {
        self.vararg_word &= !VARARG_PLAIN;
    }

    /// What the plain mark promises of this record, apart from the caller's
    /// record below it. For assertions.
    pub(crate) fn plain_mark_holds(&self) -> bool {
        matches!(self.caller, Some(crate::vm::ReturnTarget::Direct(_)))
            && self.extras.is_none()
            && !self.chunk.is_owned()
            && !self.upvalues.owns_slice()
            && !matches!(
                self.call_site.as_deref(),
                Some(crate::vm::FrameSite::Named(_))
            )
            && self.closure.is_some()
            && self.tail_calls == 0
            && self.vararg_word == VARARG_PLAIN
            && self.chunk.entry_register_count_cached().is_some()
    }

    /// The site that names this activation: a call site stored with it, or
    /// the one a tail call left. A `Tail` site belongs to the activation only
    /// while `tail_calls` is not zero: a new activation in a record that
    /// held one leaves it in place (the next tail call overwrites it), and
    /// zeroes `tail_calls`.
    #[inline(always)]
    pub(crate) fn current_call_site(&self) -> Option<&crate::vm::FrameSite> {
        match self.call_site.as_deref() {
            Some(crate::vm::FrameSite::Tail { .. }) if self.tail_calls == 0 => None,
            site => site,
        }
    }

    pub(crate) fn extras_ref(&self) -> Option<&FrameExtras> {
        self.extras.as_deref()
    }

    pub(crate) fn extras_mut_ref(&mut self) -> Option<&mut FrameExtras> {
        self.extras.as_deref_mut()
    }

    /// Whether this activation returns its values to the registers of a
    /// direct Lua caller ([`ReturnTarget::Direct`]).
    #[inline(always)]
    pub(crate) fn returns_direct(&self) -> bool {
        matches!(self.caller, Some(crate::vm::ReturnTarget::Direct(_)))
    }

    /// How many varargs this activation holds on the value stack, just below
    /// its window (see `CallStack::varargs`).
    #[inline(always)]
    pub(crate) fn vararg_count(&self) -> usize {
        (self.vararg_word & !(VARARG_TRUNCATED | VARARG_PLAIN)) as usize
    }

    /// Whether this activation holds varargs or cut its caller short: what
    /// a test of the vararg word against zero asked before the word held
    /// the plain mark too.
    #[inline(always)]
    pub(crate) fn has_vararg_state(&self) -> bool {
        self.vararg_word & !VARARG_PLAIN != 0
    }

    /// The open results, when a call left some for its consumer.
    #[inline(always)]
    pub(crate) fn open_values(&self) -> Option<&[Value]> {
        match self.open.as_deref() {
            Some(open) if open.active => Some(&open.values),
            _ => None,
        }
    }

    /// Whether a call left open results.
    #[inline(always)]
    pub(crate) fn has_open_results(&self) -> bool {
        self.open.as_deref().is_some_and(|open| open.active)
    }

    /// The buffer for new open results, emptied and marked as holding them.
    #[inline(always)]
    pub(crate) fn open_buffer(&mut self) -> &mut Vec<Value> {
        let open = self.open.get_or_insert_with(Box::default);
        open.active = true;
        open.values.clear();
        &mut open.values
    }

    /// Stores `values` as the open results and returns the buffer they
    /// replace, emptied, for reuse.
    pub(crate) fn replace_open_results(&mut self, mut values: Vec<Value>) -> Vec<Value> {
        let open = self.open.get_or_insert_with(Box::default);
        open.active = true;
        std::mem::swap(&mut open.values, &mut values);
        values.clear();
        values
    }

    /// Stores `values` as the open results.
    pub(crate) fn set_open_results(&mut self, values: Vec<Value>) {
        drop(self.replace_open_results(values));
    }

    /// Clears the open results.
    ///
    /// Fixed-arity results run this on every call return.
    #[inline(always)]
    pub(crate) fn clear_open_results(&mut self) {
        if let Some(open) = self.open.as_deref_mut()
            && open.active
        {
            open.active = false;
            open.values.clear();
        }
    }

    /// Takes the open results, empty when there are none.
    pub(crate) fn take_open_results(&mut self) -> Vec<Value> {
        match self.open.as_deref_mut() {
            Some(open) if open.active => {
                open.active = false;
                std::mem::take(&mut open.values)
            }
            _ => Vec::new(),
        }
    }

    /// Clears per-activation interior state in place.
    ///
    /// Callers resetting a frame for reuse keep their own fast path when no
    /// interior state exists: every cleared field already holds its reset
    /// value there.
    pub(crate) fn reset_extras(&mut self) {
        self.clear_open_results();
        if let Some(extras) = self.extras.as_deref_mut() {
            extras.close_slots.clear();
            extras.host_close_handler = None;
            extras.traceback = None;
            extras.last_hook_line = None;
            extras.last_hook_pc = None;
            extras.skip_hook_pc_once = None;
            extras.debug_hook_subject = None;
            extras.suppress_debug_hooks = false;
            extras.debug_hook_transfer = (0, 0);
            extras.suppress_return_hook = false;
        }
    }
    // A named constructor that builds the `OwnedFrame` wrapper around a frame.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(test)]
    pub(crate) fn new(
        chunk: Arc<Chunk>,
        arguments: Vec<Value>,
        upvalues: impl Into<CaptureCells>,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: Option<ClosureHandle>,
    ) -> OwnedFrame {
        Self::new_with_registers(
            chunk,
            arguments,
            upvalues.into(),
            caller,
            call_site,
            environment,
            closure,
            Vec::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_registers(
        chunk: impl Into<FrameChunk>,
        arguments: impl IntoIterator<Item = Value>,
        upvalues: impl Into<CaptureCells>,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: Option<ClosureHandle>,
        mut registers: Vec<Value>,
    ) -> OwnedFrame {
        let chunk: FrameChunk = chunk.into();
        reset_registers(&mut registers, usize::from(chunk.register_count));
        let mut arguments = arguments.into_iter();
        for slot in registers
            .iter_mut()
            .take(usize::from(chunk.parameter_count))
        {
            *slot = arguments.next().unwrap_or(Value::Nil);
        }
        let varargs: Vec<Value> = if chunk.is_vararg {
            arguments.collect()
        } else {
            Vec::new()
        };
        let frame = Self {
            chunk,
            closure,
            environment,
            base: 0,
            top: 0,
            upvalues: upvalues.into(),
            pc: 0,
            last_interpreted_pc: LastPc::NONE,
            caller,
            call_site: call_site.map(crate::vm::FrameSite::boxed),
            tail_calls: 0,
            vararg_word: 0,
            shape: crate::runtime::NO_SHAPE,
            open: None,
            extras: None,
        };
        OwnedFrame {
            frame,
            registers,
            varargs,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_fixed_one_argument(
        chunk: Arc<Chunk>,
        argument: Value,
        upvalues: impl Into<CaptureCells>,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: Option<ClosureHandle>,
    ) -> OwnedFrame {
        Self::new_fixed_one_argument_with_registers(
            chunk,
            argument,
            upvalues.into(),
            caller,
            call_site,
            environment,
            closure,
            Vec::new(),
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_fixed_one_argument_with_registers(
        chunk: Arc<Chunk>,
        argument: Value,
        upvalues: impl Into<CaptureCells>,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: Option<ClosureHandle>,
        mut registers: Vec<Value>,
    ) -> OwnedFrame {
        debug_assert!(!chunk.is_vararg);
        reset_registers(&mut registers, usize::from(chunk.register_count));
        if chunk.parameter_count != 0 {
            registers[0] = argument;
        }
        let frame = Self {
            chunk: chunk.into(),
            closure,
            environment,
            base: 0,
            top: 0,
            upvalues: upvalues.into(),
            pc: 0,
            last_interpreted_pc: LastPc::NONE,
            caller,
            call_site: call_site.map(crate::vm::FrameSite::boxed),
            tail_calls: 0,
            vararg_word: 0,
            shape: crate::runtime::NO_SHAPE,
            open: None,
            extras: None,
        };
        OwnedFrame {
            frame,
            registers,
            varargs: Vec::new(),
        }
    }
}

/// A frame's `last_interpreted_pc` as one machine word, `usize::MAX` meaning
/// none. An `Option<usize>` takes two words, written as two stores and read
/// back as one 16-byte load, which the store buffer cannot forward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub(crate) struct LastPc(usize);

impl LastPc {
    pub(crate) const NONE: Self = Self(usize::MAX);

    #[inline(always)]
    pub(crate) const fn some(pc: usize) -> Self {
        Self(pc)
    }

    #[inline(always)]
    pub(crate) const fn get(self) -> Option<usize> {
        if self.0 == usize::MAX {
            None
        } else {
            Some(self.0)
        }
    }

    #[inline(always)]
    pub(crate) const fn is_none(self) -> bool {
        self.0 == usize::MAX
    }

    #[inline(always)]
    pub(crate) const fn unwrap_or(self, default: usize) -> usize {
        if self.0 == usize::MAX {
            default
        } else {
            self.0
        }
    }
}

impl From<Option<usize>> for LastPc {
    #[inline(always)]
    fn from(pc: Option<usize>) -> Self {
        match pc {
            Some(pc) => Self(pc),
            None => Self::NONE,
        }
    }
}
