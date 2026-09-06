//! Closure bodies, captured cells, and their `Runtime` operations.

use crate::bytecode::Chunk;
use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::gc::GcReference;
use crate::runtime::gc::heap::value_gc_reference;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{CellHandle, ClosureHandle, TableHandle, Value};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

pub(crate) struct ClosureBody {
    chunk: Arc<Chunk>,
    pub(crate) cells: ClosureCells,
    pub(crate) environment: TableHandle,
    /// Cell slices replaced by `set_closure_cell` while another owner still
    /// held them. Frames that borrow this closure's cells without counting
    /// them may still read an old slice, so it lives as long as the closure.
    /// Boxed: almost no closure has one.
    #[allow(
        clippy::box_collection,
        reason = "the box keeps the closure small; almost none has retired cells"
    )]
    retired_cells: Option<Box<Vec<Arc<[CellHandle]>>>>,
    /// Opaque VM-owned classification of how calls to this closure may be
    /// dispatched. Zero until the VM records one.
    call_hint: u8,
    /// Counts the changes of `environment` and `cells`; see
    /// [`ClosureCallParts::shape`].
    shape: u32,
}

impl ClosureBody {
    /// Writes a closure into `slot` field by field, so that it is stored
    /// once, where it stays (see `Table::init_parts`).
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    fn init(
        slot: &mut std::mem::MaybeUninit<Self>,
        chunk: Arc<Chunk>,
        cells: ClosureCells,
        environment: TableHandle,
    ) {
        let body = slot.as_mut_ptr();
        // SAFETY: `body` points to writable memory for a `ClosureBody`, and
        // every field is written exactly once.
        unsafe {
            (&raw mut (*body).chunk).write(chunk);
            (&raw mut (*body).cells).write(cells);
            (&raw mut (*body).environment).write(environment);
            (&raw mut (*body).retired_cells).write(None);
            (&raw mut (*body).call_hint).write(0);
            (&raw mut (*body).shape).write(0);
        }
    }

    /// The cells of the slices `set_closure_cell` replaced.
    pub(crate) fn retired_cells(&self) -> impl Iterator<Item = &CellHandle> {
        self.retired_cells
            .iter()
            .flat_map(|retired| retired.iter())
            .flat_map(|cells| cells.iter())
    }
}

/// The captured cells of a closure. No cell and one cell are stored inline,
/// so a closure that captures at most one variable owns no allocation for
/// them; a frame copies those inline too (see [`CaptureCells::borrow_shared`]).
/// More cells share an allocation, which a call can borrow without a
/// reference count.
pub enum ClosureCells {
    Empty,
    One(CellHandle),
    Many(Arc<[CellHandle]>),
}

impl ClosureCells {
    /// The cells `cells`, in one allocation when there are two or more.
    #[must_use]
    #[inline]
    pub fn from_slice(cells: &[CellHandle]) -> Self {
        match cells {
            [] => Self::Empty,
            [cell] => Self::One(*cell),
            _ => Self::Many(Arc::from(cells)),
        }
    }
}

impl ClosureCells {
    /// [`Self::from_slice`] for cells given as [`CellHandle::native_identity`]
    /// words.
    #[must_use]
    #[inline]
    fn from_identities(cells: &[u64]) -> Self {
        match cells {
            [] => Self::Empty,
            [cell] => Self::One(CellHandle::from_native_identity(*cell)),
            _ => Self::Many(
                cells
                    .iter()
                    .map(|cell| CellHandle::from_native_identity(*cell))
                    .collect(),
            ),
        }
    }
}

impl std::ops::Deref for ClosureCells {
    type Target = [CellHandle];

    #[inline]
    fn deref(&self) -> &[CellHandle] {
        match self {
            Self::Empty => &[],
            Self::One(cell) => std::slice::from_ref(cell),
            Self::Many(cells) => cells,
        }
    }
}

/// Borrowed parts of a closure that a call needs, without reference-count
/// traffic until the caller decides to keep them.
pub struct ClosureCallParts<'a> {
    pub chunk: &'a Arc<Chunk>,
    pub cells: &'a ClosureCells,
    pub environment: TableHandle,
    pub call_hint: u8,
    /// Changes whenever the closure's environment or captured cells do, and
    /// never equals [`NO_SHAPE`]. A frame record that remembers the shape it
    /// was built at, for the same closure, still holds the closure's current
    /// environment and captures.
    pub shape: u32,
}

/// A shape no closure has: the mark of a frame record that was not built
/// from a closure's parts at any shape.
pub const NO_SHAPE: u32 = u32::MAX;

/// The shape after `shape`.
const fn next_shape(shape: u32) -> u32 {
    // Skips the reserved value on the wrap, which takes four billion changes
    // of one closure.
    match shape.wrapping_add(1) {
        NO_SHAPE => 0,
        next => next,
    }
}

/// Captured cell handles optimized for the common empty and single-cell cases.
///
/// Larger captures share an allocation until [`Self::replace`] needs to
/// change one handle. Empty and single-cell captures are stored inline, so
/// cloning them does not perform atomic reference-count operations. A
/// frame on the call path can instead borrow a live closure's cell slice
/// with [`Self::borrow_shared`], which touches no reference count at all.
pub struct CaptureCells {
    storage: CaptureCellsStorage,
}

enum CaptureCellsStorage {
    Empty,
    One(CellHandle),
    Shared(Arc<[CellHandle]>),
    /// The cell slice of a closure that outlives these captures; see
    /// [`CaptureCells::borrow_shared`].
    Borrowed(NonNull<[CellHandle]>),
}

// SAFETY: `Borrowed` points into an `Arc<[CellHandle]>` of immutable plain
// handles. The creator of the borrow guarantees the allocation outlives it,
// and nothing writes through the pointer, so it behaves like `&[CellHandle]`.
#[allow(unsafe_code)]
unsafe impl Send for CaptureCells {}
// SAFETY: see the `Send` impl; shared access only reads.
#[allow(unsafe_code)]
unsafe impl Sync for CaptureCells {}

impl CaptureCells {
    /// Builds captures from shared closure storage.
    ///
    /// Empty and single-cell storage is copied inline. Larger storage remains
    /// shared until it is replaced.
    #[must_use]
    #[inline]
    pub fn from_shared(cells: &ClosureCells) -> Self {
        match cells {
            ClosureCells::Empty => Self::default(),
            ClosureCells::One(cell) => Self {
                storage: CaptureCellsStorage::One(*cell),
            },
            ClosureCells::Many(cells) => Self {
                storage: CaptureCellsStorage::Shared(Arc::clone(cells)),
            },
        }
    }

    /// Borrows larger captures from shared closure storage without a
    /// reference-count operation. Empty and single-cell storage is copied
    /// inline, as in [`Self::from_shared`].
    ///
    /// # Safety
    ///
    /// The allocation behind `cells` must stay alive for as long as the
    /// returned value, or any value moved out of it, is used. A clone of
    /// the result owns its cells, so only the original carries this
    /// obligation. The VM meets it by rooting the closure that owns `cells`
    /// for as long as the frame holding the captures is live. Replacing the
    /// closure's cells (`set_closure_cell`) copies the allocation when other
    /// owners exist, so the replacer must keep the old one alive too.
    #[must_use]
    #[inline(always)]
    #[allow(unsafe_code)]
    pub unsafe fn borrow_shared(cells: &ClosureCells) -> Self {
        match cells {
            ClosureCells::Empty => Self::default(),
            ClosureCells::One(cell) => Self {
                storage: CaptureCellsStorage::One(*cell),
            },
            ClosureCells::Many(cells) => Self {
                storage: CaptureCellsStorage::Borrowed(NonNull::from(&**cells)),
            },
        }
    }

    /// The handle of capture `index`, if there is one: `self.get(index)`
    /// copied. A single inline cell and a borrowed slice, the two forms a
    /// running closure's frame holds, are tested one after the other. The
    /// match behind `Deref` compiles to a jump table, and `GetUpvalue` ran
    /// an indirect jump for it on every execution.
    #[must_use]
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub fn cell(&self, index: usize) -> Option<CellHandle> {
        if let CaptureCellsStorage::One(cell) = &self.storage {
            return if index == 0 { Some(*cell) } else { None };
        }
        if let CaptureCellsStorage::Borrowed(cells) = &self.storage {
            // SAFETY: `borrow_shared` obliges its caller to keep the
            // allocation alive while this value is used.
            return unsafe { cells.as_ref() }.get(index).copied();
        }
        self.cell_other(index)
    }

    #[inline(never)]
    fn cell_other(&self, index: usize) -> Option<CellHandle> {
        self.get(index).copied()
    }

    /// Whether these captures hold a reference count of their own, which
    /// overwriting them releases.
    #[must_use]
    #[inline(always)]
    pub fn owns_slice(&self) -> bool {
        matches!(self.storage, CaptureCellsStorage::Shared(_))
    }

    /// Whether these captures are what [`Self::borrow_shared`] builds from
    /// `cells`: the same inline cell, or a borrow of the same allocation.
    #[must_use]
    #[allow(clippy::inline_always, reason = "read on every call of a closure")]
    #[inline(always)]
    pub fn is_borrow_of(&self, cells: &ClosureCells) -> bool {
        match (&self.storage, cells) {
            (CaptureCellsStorage::Empty, ClosureCells::Empty) => true,
            (CaptureCellsStorage::One(cell), ClosureCells::One(other)) => cell == other,
            (CaptureCellsStorage::Borrowed(borrowed), ClosureCells::Many(cells)) => {
                std::ptr::addr_eq(borrowed.as_ptr(), Arc::as_ptr(cells))
            }
            _ => false,
        }
    }

    /// Replaces a captured cell handle.
    ///
    /// Returns `false` if `index` is outside the capture list. Shared storage
    /// is copied only when another snapshot still refers to it.
    pub fn replace(&mut self, index: usize, cell: CellHandle) -> bool {
        match &mut self.storage {
            CaptureCellsStorage::Empty => false,
            CaptureCellsStorage::One(destination) => {
                if index != 0 {
                    return false;
                }
                *destination = cell;
                true
            }
            CaptureCellsStorage::Shared(cells) => {
                if index >= cells.len() {
                    return false;
                }
                Arc::make_mut(cells)[index] = cell;
                true
            }
            CaptureCellsStorage::Borrowed(_) => {
                let mut owned: Arc<[CellHandle]> = Arc::from(&**self);
                if index >= owned.len() {
                    return false;
                }
                Arc::make_mut(&mut owned)[index] = cell;
                self.storage = CaptureCellsStorage::Shared(owned);
                true
            }
        }
    }
}

impl Clone for CaptureCells {
    fn clone(&self) -> Self {
        match &self.storage {
            CaptureCellsStorage::Empty => Self::default(),
            CaptureCellsStorage::One(cell) => Self {
                storage: CaptureCellsStorage::One(*cell),
            },
            CaptureCellsStorage::Shared(cells) => Self {
                storage: CaptureCellsStorage::Shared(Arc::clone(cells)),
            },
            // A clone may outlive the closure, so it takes its own copy.
            CaptureCellsStorage::Borrowed(_) => Self {
                storage: CaptureCellsStorage::Shared(Arc::from(&**self)),
            },
        }
    }
}

impl Default for CaptureCells {
    fn default() -> Self {
        Self {
            storage: CaptureCellsStorage::Empty,
        }
    }
}

impl std::ops::Deref for CaptureCells {
    type Target = [CellHandle];

    #[allow(unsafe_code)]
    fn deref(&self) -> &Self::Target {
        match &self.storage {
            CaptureCellsStorage::Empty => &[],
            CaptureCellsStorage::One(cell) => std::slice::from_ref(cell),
            CaptureCellsStorage::Shared(cells) => cells,
            // SAFETY: `borrow_shared` obliges its caller to keep the
            // allocation alive while this value is used.
            CaptureCellsStorage::Borrowed(cells) => unsafe { cells.as_ref() },
        }
    }
}

impl From<Vec<CellHandle>> for CaptureCells {
    fn from(cells: Vec<CellHandle>) -> Self {
        match cells.as_slice() {
            [] => Self::default(),
            [cell] => Self {
                storage: CaptureCellsStorage::One(*cell),
            },
            _ => Self {
                storage: CaptureCellsStorage::Shared(Arc::from(cells)),
            },
        }
    }
}

impl From<Arc<[CellHandle]>> for CaptureCells {
    fn from(cells: Arc<[CellHandle]>) -> Self {
        match cells.len() {
            0 => Self::default(),
            1 => Self {
                storage: CaptureCellsStorage::One(cells[0]),
            },
            _ => Self {
                storage: CaptureCellsStorage::Shared(cells),
            },
        }
    }
}

/// Immutable closure data copied out of the runtime without exposing storage.
#[derive(Clone)]
pub struct ClosureSnapshot {
    chunk: Arc<Chunk>,
    cells: CaptureCells,
    environment: TableHandle,
}

impl ClosureSnapshot {
    #[must_use]
    pub fn chunk(&self) -> Arc<Chunk> {
        Arc::clone(&self.chunk)
    }

    #[must_use]
    pub fn cells(&self) -> &[CellHandle] {
        &self.cells
    }

    /// Returns the cell handles for another execution frame.
    #[must_use]
    pub fn shared_cells(&self) -> CaptureCells {
        self.cells.clone()
    }

    /// Consumes the snapshot and returns its shared closure state.
    ///
    /// This lets an execution engine transfer the chunk ownership into a
    /// frame without incrementing and immediately decrementing the chunk's
    /// reference count.
    #[must_use]
    pub fn into_parts(self) -> (Arc<Chunk>, CaptureCells, TableHandle) {
        (self.chunk, self.cells, self.environment)
    }

    /// Returns the closure environment captured when this snapshot was made.
    #[must_use]
    pub const fn environment(&self) -> TableHandle {
        self.environment
    }
}

/// The base address of a thread's value stack, shared with the cells that
/// are open on it.
///
/// An open upvalue names its variable by stack slot, not by address, because
/// the stack's buffer moves when it grows. The owner of the stack keeps
/// [`Self::set_base`] current; readers add the slot to it.
pub struct StackAnchor {
    base: AtomicPtr<Value>,
}

impl StackAnchor {
    /// An anchor over no stack yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Records where slot 0 of the stack is now.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn set_base(&self, base: *mut Value) {
        self.base.store(base, Ordering::Relaxed);
    }

    /// Where slot 0 of the stack is.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn base(&self) -> *mut Value {
        self.base.load(Ordering::Relaxed)
    }
}

impl Default for StackAnchor {
    fn default() -> Self {
        Self::new()
    }
}

/// The stack slot an open cell stands for.
#[derive(Clone, Copy)]
struct OpenSlot {
    anchor: NonNull<StackAnchor>,
    slot: u32,
}

// SAFETY: the anchor is only read through atomics, and the value behind the
// slot is accessed under the same single-mutator discipline as every other
// heap object of a runtime: the collector reads it only while the mutator is
// stopped at a safe point.
#[allow(unsafe_code)]
unsafe impl Send for OpenSlot {}
// SAFETY: see the `Send` impl.
#[allow(unsafe_code)]
unsafe impl Sync for OpenSlot {}

/// A captured variable. While the function that declared it runs, the cell
/// is open: the variable lives in a register of the declaring thread's value
/// stack and the cell reads and writes that slot. When the function returns
/// or the variable's scope ends, the owner closes the cell, which moves the
/// value into the cell.
pub(crate) struct Cell {
    /// The value of a closed cell. Stale while the cell is open.
    pub(crate) value: Value,
    open: Option<OpenSlot>,
}

impl Cell {
    /// Writes a cell into `slot` field by field, so that it is stored once,
    /// where it stays (see `Table::init_parts`).
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    fn init(slot: &mut std::mem::MaybeUninit<Self>, value: Value, open: Option<OpenSlot>) {
        let cell = slot.as_mut_ptr();
        // SAFETY: `cell` points to writable memory for a `Cell`, and every
        // field is written exactly once.
        unsafe {
            (&raw mut (*cell).value).write(value);
            (&raw mut (*cell).open).write(open);
        }
    }

    /// The cell's current value, wherever it lives.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    pub(crate) fn current(&self) -> &Value {
        match self.open {
            None => &self.value,
            // SAFETY: an open cell names a slot inside its stack's buffer;
            // the owner keeps the anchor's base current and closes the cell
            // before the slot leaves the stack (see `create_open_cell`).
            Some(open) => unsafe { &*open.anchor.as_ref().base().add(open.slot as usize) },
        }
    }

    /// The cell's value slot, for a write.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    fn current_mut(&mut self) -> &mut Value {
        match self.open {
            None => &mut self.value,
            // SAFETY: as for `current`; the runtime has `&mut` access to
            // the cell, and nothing else reads the slot meanwhile.
            Some(open) => unsafe { &mut *open.anchor.as_ref().base().add(open.slot as usize) },
        }
    }
}

impl Runtime {
    /// Stores an immutable bytecode closure and its captured cells.
    ///
    /// The chunk is validated first: the VM runs validated chunks without
    /// bounds checks on instruction fetch.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk fails validation or the runtime has
    /// exhausted its closure-handle space.
    pub fn create_closure(
        &mut self,
        chunk: Arc<Chunk>,
        cells: Vec<CellHandle>,
    ) -> RuntimeResult<ClosureHandle> {
        self.create_closure_in_environment(chunk, cells, self.global_environment)
    }

    /// Stores an immutable bytecode closure with an explicit environment.
    ///
    /// Child closures should use the environment of their creating closure.
    /// The VM owns that inheritance rule and supplies the inherited table here.
    /// The chunk is validated first, as in [`Self::create_closure`].
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk fails validation, `environment` is stale,
    /// or closure-handle space is exhausted.
    #[allow(unsafe_code)]
    pub fn create_closure_in_environment(
        &mut self,
        chunk: Arc<Chunk>,
        cells: Vec<CellHandle>,
        environment: TableHandle,
    ) -> RuntimeResult<ClosureHandle> {
        chunk
            .validate_detached_prototype()
            .map_err(|error| RuntimeError::message(format!("invalid chunk: {error}")))?;
        // SAFETY: the chunk was just validated.
        unsafe { self.create_closure_in_environment_unchecked(chunk, cells, environment) }
    }

    /// Stores a closure without validating its chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if `environment` is stale or closure-handle space is
    /// exhausted.
    ///
    /// # Safety
    ///
    /// `chunk` and every prototype under it must have passed
    /// `Chunk::validate`, and must not change afterwards. The VM's fast loop
    /// fetches instructions without bounds checks and relies on validation
    /// to keep every program counter inside the code.
    #[allow(unsafe_code, clippy::needless_pass_by_value, reason = "public API")]
    pub unsafe fn create_closure_in_environment_unchecked(
        &mut self,
        chunk: Arc<Chunk>,
        cells: Vec<CellHandle>,
        environment: TableHandle,
    ) -> RuntimeResult<ClosureHandle> {
        // SAFETY: as the caller promises.
        unsafe { self.create_closure_from_slice_unchecked(chunk, &cells, environment) }
    }

    /// [`Self::create_closure_in_environment_unchecked`] for cells in a
    /// slice. A closure with at most one cell allocates nothing for them,
    /// and one with more allocates once.
    ///
    /// # Errors
    ///
    /// Returns an error if `environment` is stale or closure-handle space is
    /// exhausted.
    ///
    /// # Safety
    ///
    /// As for [`Self::create_closure_in_environment_unchecked`].
    #[allow(unsafe_code)]
    pub unsafe fn create_closure_from_slice_unchecked(
        &mut self,
        chunk: Arc<Chunk>,
        cells: &[CellHandle],
        environment: TableHandle,
    ) -> RuntimeResult<ClosureHandle> {
        // SAFETY: as the caller promises.
        let identity =
            unsafe { self.create_closure_from_slice_identity(chunk, cells, environment) };
        if identity == crate::runtime::runtime::NO_TABLE_IDENTITY {
            return Err(self.closure_creation_error(environment));
        }
        Ok(ClosureHandle::from_native_identity(identity))
    }

    /// [`Self::create_closure_from_slice_unchecked`] that returns the
    /// closure's [`ClosureHandle::native_identity`], or
    /// [`oximond::runtime::NO_TABLE_IDENTITY`] when `environment` is stale or closure
    /// handles ran out ([`Self::closure_creation_error`] says which). The
    /// result is one machine word and comes back in a register; a
    /// `RuntimeResult` comes back in memory, written with narrow stores and
    /// read with a wide load that cannot forward from them.
    ///
    /// # Safety
    ///
    /// As for [`Self::create_closure_in_environment_unchecked`].
    #[allow(unsafe_code)]
    #[inline]
    pub unsafe fn create_closure_from_slice_identity(
        &mut self,
        chunk: Arc<Chunk>,
        cells: &[CellHandle],
        environment: TableHandle,
    ) -> u64 {
        // SAFETY: as the caller promises.
        unsafe {
            self.create_closure_with_cells(chunk, || ClosureCells::from_slice(cells), environment)
        }
    }

    /// [`Self::create_closure_from_slice_identity`] for cells given as
    /// [`CellHandle::native_identity`] words, which the caller can gather in
    /// registers: a handle read back from the narrow stores that wrote it
    /// does not forward.
    ///
    /// # Safety
    ///
    /// As for [`Self::create_closure_in_environment_unchecked`]; every word
    /// must be the identity of a cell handle.
    #[allow(unsafe_code)]
    #[inline]
    pub unsafe fn create_closure_from_identities(
        &mut self,
        chunk: Arc<Chunk>,
        cells: &[u64],
        environment: TableHandle,
    ) -> u64 {
        // SAFETY: as the caller promises.
        unsafe {
            self.create_closure_with_cells(
                chunk,
                || ClosureCells::from_identities(cells),
                environment,
            )
        }
    }

    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    unsafe fn create_closure_with_cells(
        &mut self,
        chunk: Arc<Chunk>,
        cells: impl FnOnce() -> ClosureCells,
        environment: TableHandle,
    ) -> u64 {
        if self.tables.get(environment.0).is_none() {
            return crate::runtime::runtime::NO_TABLE_IDENTITY;
        }
        let Some(key) = self
            .closures
            .insert_init(|body| ClosureBody::init(body, chunk, cells(), environment))
        else {
            return crate::runtime::runtime::NO_TABLE_IDENTITY;
        };
        let handle = ClosureHandle(key);
        self.gc_charge(std::mem::size_of::<ClosureBody>());
        self.gc_barrier_born_closure(handle);
        handle.native_identity()
    }

    /// Why [`Self::create_closure_from_slice_identity`] made no closure.
    #[must_use]
    pub fn closure_creation_error(&self, environment: TableHandle) -> RuntimeError {
        match self.table(environment) {
            Err(error) => error,
            Ok(_) => RuntimeError::message("closure handle space exhausted"),
        }
    }

    /// Retrieves a closure's immutable VM-facing snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn closure_snapshot(&self, handle: ClosureHandle) -> RuntimeResult<ClosureSnapshot> {
        let body = self
            .closures
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidClosure(handle))?;
        Ok(ClosureSnapshot {
            chunk: Arc::clone(&body.chunk),
            cells: CaptureCells::from_shared(&body.cells),
            environment: body.environment,
        })
    }

    /// Borrows the parts of a closure that a call needs, or `None` for a
    /// stale handle.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn closure_call_parts(&self, handle: ClosureHandle) -> Option<ClosureCallParts<'_>> {
        let body = self.closures.get(handle.0)?;
        Some(ClosureCallParts {
            chunk: &body.chunk,
            cells: &body.cells,
            environment: body.environment,
            call_hint: body.call_hint,
            shape: body.shape,
        })
    }

    /// [`Self::closure_call_parts`] for a handle read from a live register:
    /// the register roots the closure, so the slot's bounds, generation and
    /// occupancy need no check.
    ///
    /// # Safety
    ///
    /// `handle` names a live closure of this runtime.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    #[must_use]
    pub(crate) unsafe fn closure_call_parts_live(
        &self,
        handle: ClosureHandle,
    ) -> ClosureCallParts<'_> {
        // SAFETY: the contract.
        let body = unsafe { self.closures.get_unchecked(handle.0) };
        ClosureCallParts {
            chunk: &body.chunk,
            cells: &body.cells,
            environment: body.environment,
            call_hint: body.call_hint,
            shape: body.shape,
        }
    }

    /// Records the VM's call classification for a live closure. Stale
    /// handles are ignored.
    pub fn set_closure_call_hint(&mut self, handle: ClosureHandle, hint: u8) {
        if let Some(body) = self.closures.get_mut(handle.0) {
            body.call_hint = hint;
        }
    }

    /// Returns one live captured cell without cloning the closure snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    pub fn closure_cell(
        &self,
        handle: ClosureHandle,
        index: usize,
    ) -> RuntimeResult<Option<CellHandle>> {
        let body = self
            .closures
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidClosure(handle))?;
        Ok(body.cells.get(index).copied())
    }

    /// Returns the table used for global lookups by a closure.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    pub fn closure_environment(&self, handle: ClosureHandle) -> RuntimeResult<TableHandle> {
        self.closures
            .get(handle.0)
            .map(|closure| closure.environment)
            .ok_or_else(|| RuntimeError::InvalidClosure(handle))
    }

    /// Replaces a closure's environment table.
    ///
    /// # Errors
    ///
    /// Returns an error if either handle is stale or invalid.
    pub fn set_closure_environment(
        &mut self,
        closure: ClosureHandle,
        environment: TableHandle,
    ) -> RuntimeResult<()> {
        self.table(environment)?;
        let closure_handle = closure;
        let closure = self
            .closures
            .get_mut(closure.0)
            .ok_or_else(|| RuntimeError::InvalidClosure(closure))?;
        closure.environment = environment;
        closure.shape = next_shape(closure.shape);
        self.gc_barrier(
            Some(GcReference::Closure(closure_handle)),
            GcReference::Table(environment),
        );
        Ok(())
    }

    /// Creates a closed captured value cell.
    ///
    /// # Errors
    ///
    /// Returns an error if the cell arena cannot allocate another handle.
    pub fn create_cell(&mut self, value: Value) -> RuntimeResult<CellHandle> {
        let handle = self
            .cells
            .insert_init(|cell| Cell::init(cell, value, None))
            .map(CellHandle)
            .ok_or_else(|| RuntimeError::message("upvalue cell space exhausted"))?;
        self.gc_charge(std::mem::size_of::<Cell>());
        self.gc_barrier_born_cell(handle);
        Ok(handle)
    }

    /// Creates a cell that stands for slot `slot` of the value stack behind
    /// `anchor`, which holds the variable until the cell is closed.
    ///
    /// # Errors
    ///
    /// Returns an error if the cell arena cannot allocate another handle.
    ///
    /// # Safety
    ///
    /// The slot must lie inside the stack's buffer, and the anchor must name
    /// that buffer from now on: its owner updates the base whenever the
    /// buffer moves. The anchor and the buffer must outlive the cell's open
    /// state, so the owner has to call [`Self::close_open_cell`] before the
    /// slot is reused, truncated away or its stack dropped.
    #[allow(unsafe_code)]
    pub unsafe fn create_open_cell(
        &mut self,
        anchor: NonNull<StackAnchor>,
        slot: u32,
    ) -> RuntimeResult<CellHandle> {
        let handle = self
            .cells
            .insert_init(|cell| Cell::init(cell, Value::Nil, Some(OpenSlot { anchor, slot })))
            .map(CellHandle)
            .ok_or_else(|| RuntimeError::message("upvalue cell space exhausted"))?;
        self.gc_charge(std::mem::size_of::<Cell>());
        Ok(handle)
    }

    /// Points a cell at slot `slot` of the stack behind `anchor`, whatever
    /// it stood for before, leaving the old stack and the cell's own value
    /// alone. A stale handle is ignored.
    ///
    /// # Safety
    ///
    /// As for [`Self::create_open_cell`].
    #[allow(unsafe_code)]
    pub unsafe fn reopen_cell(
        &mut self,
        handle: CellHandle,
        anchor: NonNull<StackAnchor>,
        slot: u32,
    ) {
        self.cell_writes = self.cell_writes.wrapping_add(1);
        self.cell_rebinds = self.cell_rebinds.wrapping_add(1);
        if let Some(cell) = self.cells.get_mut(handle.0) {
            cell.open = Some(OpenSlot { anchor, slot });
        }
    }

    /// A count that changes whenever a cell's value or a closure's cell
    /// is replaced: every write of a cell goes through this runtime. While
    /// it stands still, a closed cell holds the value it held. An open cell
    /// may still change, through a register of the thread that declared it.
    ///
    /// Native trace code stores to a callee's cells itself
    /// (`ClosureCellUse::Set`) and does not count. A host call that compares
    /// the count around it may still trust it: no trace is entered while a
    /// host call runs, so no native store happens between the two reads.
    #[must_use]
    pub(crate) const fn cell_writes(&self) -> u64 {
        self.cell_writes
    }

    /// A count that changes whenever a closure's cell is replaced
    /// ([`Self::set_closure_cell`], which `debug.upvaluejoin` and the frames'
    /// own copies follow) or a cell is reopened. Writes of a cell's value do
    /// not move it. While it stands still, every closure has the cells it
    /// had, so which of them are one cell is what it was.
    #[must_use]
    pub(crate) const fn cell_rebinds(&self) -> u64 {
        self.cell_rebinds
    }

    /// Reads a captured value cell.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    pub fn cell_get(&self, handle: CellHandle) -> RuntimeResult<Value> {
        self.cells
            .get(handle.0)
            .map(|cell| cell.current().clone())
            .ok_or_else(|| RuntimeError::InvalidCell(handle))
    }

    /// Borrows a captured value cell without cloning its [`Value`] payload.
    ///
    /// The caller clones only when it needs an owned copy, which keeps hot
    /// register fills from round-tripping an owned value and its error
    /// payload through a call boundary. The error is built lazily for the
    /// same reason.
    ///
    /// An open cell lends its stack slot, so the borrow must end before the
    /// stack moves or the slot is written.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn cell_ref(&self, handle: CellHandle) -> RuntimeResult<&Value> {
        self.cells
            .get(handle.0)
            .map(Cell::current)
            .ok_or_else(|| RuntimeError::InvalidCell(handle))
    }

    /// [`Self::cell_ref`] for a cell known to be alive, without the checks
    /// that turn a stale handle into an error.
    ///
    /// # Safety
    ///
    /// `handle` must name a live cell. The cells of a running closure
    /// qualify: its frame roots them, so a collection keeps them.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub unsafe fn cell_ref_live(&self, handle: CellHandle) -> &Value {
        // SAFETY: the caller promises the cell is live.
        unsafe { self.cells.get_unchecked(handle.0) }.current()
    }

    /// [`Self::cell_set`] for a cell known to be alive.
    ///
    /// # Safety
    ///
    /// As for [`Self::cell_ref_live`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub unsafe fn cell_set_live(&mut self, handle: CellHandle, value: Value) {
        let reference = value_gc_reference(&value);
        self.cell_writes = self.cell_writes.wrapping_add(1);
        // SAFETY: the caller promises the cell is live.
        *unsafe { self.cells.get_unchecked_mut(handle.0) }.current_mut() = value;
        if let Some(reference) = reference {
            self.gc_barrier(Some(GcReference::Cell(handle)), reference);
        }
    }

    /// [`Self::cell_set_number`] for a cell known to be alive.
    ///
    /// # Safety
    ///
    /// As for [`Self::cell_ref_live`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub unsafe fn cell_set_number_live(&mut self, handle: CellHandle, value: f64) -> bool {
        self.cell_writes = self.cell_writes.wrapping_add(1);
        // SAFETY: the caller promises the cell is live.
        let cell = unsafe { self.cells.get_unchecked_mut(handle.0) };
        let Value::Number(stored) = cell.current_mut() else {
            return false;
        };
        *stored = value;
        true
    }

    /// Reads a captured numeric cell without cloning its [`Value`] payload.
    ///
    /// Returns `Some` with the stored number, preserving signed zero and NaN
    /// payload bits. Returns `None` when the live cell contains any other
    /// value kind.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    pub fn cell_get_number(&self, handle: CellHandle) -> RuntimeResult<Option<f64>> {
        let cell = self
            .cells
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidCell(handle))?;
        Ok(match cell.current() {
            Value::Number(value) => Some(*value),
            _ => None,
        })
    }

    /// Replaces a captured value cell.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    pub fn cell_set(&mut self, handle: CellHandle, value: Value) -> RuntimeResult<()> {
        let reference = value_gc_reference(&value);
        self.cell_writes = self.cell_writes.wrapping_add(1);
        let cell = self
            .cells
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidCell(handle))?;
        *cell.current_mut() = value;
        if let Some(reference) = reference {
            self.gc_barrier(Some(GcReference::Cell(handle)), reference);
        }
        Ok(())
    }

    /// Replaces the payload of an existing captured numeric cell.
    ///
    /// Returns `true` after writing a live [`Value::Number`], preserving the
    /// supplied bits exactly. Returns `false` without changing the cell when
    /// the live cell has another value kind. Number-to-number writes cannot
    /// introduce a GC reference, so they need no write barrier.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or invalid.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn cell_set_number(&mut self, handle: CellHandle, value: f64) -> RuntimeResult<bool> {
        self.cell_writes = self.cell_writes.wrapping_add(1);
        let cell = self
            .cells
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidCell(handle))?;
        let Value::Number(stored) = cell.current_mut() else {
            return Ok(false);
        };
        *stored = value;
        Ok(true)
    }

    /// Returns an opaque userdata value that is stable for a live captured cell.
    ///
    /// The token does not keep the cell alive. If both the cell and an earlier
    /// token remain reachable, repeated calls return the same token.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or userdata allocation fails.
    pub fn cell_identity(&mut self, handle: CellHandle) -> RuntimeResult<Value> {
        self.cells
            .get(handle.0)
            .ok_or_else(|| RuntimeError::InvalidCell(handle))?;
        if let Some(identity) = self.upvalue_identities.get(&handle).copied()
            && self.userdatas.get(identity.0).is_some()
        {
            return Ok(Value::Userdata(identity));
        }
        let identity = self.create_userdata()?;
        self.upvalue_identities.insert(handle, identity);
        Ok(Value::Userdata(identity))
    }

    /// Replaces one closure upvalue with an existing captured cell.
    ///
    /// Returns `false` when `index` is outside the closure's upvalue list.
    ///
    /// # Errors
    ///
    /// Returns an error if either handle is stale.
    pub fn set_closure_cell(
        &mut self,
        closure: ClosureHandle,
        index: usize,
        cell: CellHandle,
    ) -> RuntimeResult<bool> {
        self.cells
            .get(cell.0)
            .ok_or_else(|| RuntimeError::InvalidCell(cell))?;
        let body = self
            .closures
            .get_mut(closure.0)
            .ok_or_else(|| RuntimeError::InvalidClosure(closure))?;
        if index >= body.cells.len() {
            return Ok(false);
        }
        self.cell_writes = self.cell_writes.wrapping_add(1);
        self.cell_rebinds = self.cell_rebinds.wrapping_add(1);
        body.shape = next_shape(body.shape);
        match &mut body.cells {
            ClosureCells::Empty => return Ok(false),
            // Frames copy a single cell, so there is no slice to retire.
            ClosureCells::One(destination) => *destination = cell,
            ClosureCells::Many(cells) => {
                if Arc::strong_count(cells) > 1 {
                    // `make_mut` is about to copy, leaving the old slice to
                    // its other owners. A frame that borrowed it
                    // (`CaptureCells::borrow_shared`) is not one of them, so
                    // keep it until the closure dies.
                    body.retired_cells
                        .get_or_insert_with(Box::default)
                        .push(Arc::clone(cells));
                }
                let Some(destination) = Arc::make_mut(cells).get_mut(index) else {
                    return Ok(false);
                };
                *destination = cell;
            }
        }
        self.gc_barrier(Some(GcReference::Closure(closure)), GcReference::Cell(cell));
        Ok(true)
    }

    /// The stack slot an open cell stands for, or `None` for a closed cell.
    /// A stale handle is closed.
    #[must_use]
    pub fn cell_open_slot(&self, handle: CellHandle) -> Option<(NonNull<StackAnchor>, u32)> {
        self.cells
            .get(handle.0)?
            .open
            .map(|open| (open.anchor, open.slot))
    }

    /// Closes an open cell: the variable's value moves from its stack slot
    /// into the cell, which is then independent of the stack. Returns whether
    /// the cell was open. A stale handle has nothing to close.
    ///
    /// # Safety
    ///
    /// The stack slot must still hold the variable, inside a buffer the
    /// anchor names.
    #[allow(unsafe_code)]
    pub unsafe fn close_open_cell(&mut self, handle: CellHandle) -> bool {
        let Some(cell) = self.cells.get_mut(handle.0) else {
            return false;
        };
        if cell.open.is_none() {
            return false;
        }
        let value = cell.current().clone();
        cell.open = None;
        let reference = value_gc_reference(&value);
        cell.value = value;
        if let Some(reference) = reference {
            self.gc_barrier(Some(GcReference::Cell(handle)), reference);
        }
        true
    }
}
