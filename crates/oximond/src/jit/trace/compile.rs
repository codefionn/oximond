//! Validation, exit planning and the [`CompiledTrace`] that owns the result.

use super::emit::{self, TraceCode};
use super::exit::{
    ExitDescriptor, ExitSlot, MAX_FUNCTION_TABLE_LANES, MAX_TRACE_TABLE_LANES, TRACE_TAG_BOOLEAN,
    TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT, TRACE_TAG_INTEGER, TRACE_TAG_TABLE, TableDispatch,
    TraceBinding, TraceControl, TraceExitClass, TraceGuardClass, TraceInputSlot, TraceOutcome,
    TraceTableSpace, TraceTables,
};
use crate::compat::TracingOptFlags;
use crate::jit::OptimizationTier;
use crate::jit_ir::trace::{
    HelperId, LoopParamInit, OptimizeLevel, OptimizeStats, RegisterEntry, Snapshot, SnapshotId,
    SnapshotPhase, TERMINAL_GUARD, TableBinding, TableField, TraceIr, TraceOp, TraceTerminator,
    TraceType, TraceVerifyError, TreeError, TreeMember, TreeWrites, ValueDef, ValueId,
    apart_link_writes, optimize_tree_member, plan_tree, plan_tree_with, untyped_link_writes,
};
use crate::runtime::alloc_phase::{self, Phase};
use std::borrow::Borrow;
use std::fmt;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Why a trace could not be compiled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceCompileError {
    /// The IR failed verification.
    Invalid(TraceVerifyError),
    /// The side traces do not fit their root.
    Tree(TreeError),
    /// The trace uses an operation the native backend does not lower yet.
    Unsupported {
        instruction: usize,
        what: &'static str,
    },
    /// A looping trace has no poll snapshot, so the VM could never stop it
    /// between iterations.
    MissingPollSnapshot,
    /// Cranelift or the executable-memory allocator refused the function.
    NativeBackendRejected,
    /// The compile panicked, with the panic's message. The broker catches
    /// the panic around the one compile, so only this trace is rejected and
    /// the compile thread keeps serving the others.
    Panicked(String),
}

impl fmt::Display for TraceCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => write!(f, "trace failed verification: {error}"),
            Self::Tree(error) => write!(f, "trace tree is malformed: {error}"),
            Self::Unsupported { instruction, what } => {
                write!(
                    f,
                    "instruction {instruction}: {what} is not supported natively"
                )
            }
            Self::MissingPollSnapshot => write!(f, "looping trace has no poll snapshot"),
            Self::NativeBackendRejected => write!(f, "the native backend rejected the trace"),
            Self::Panicked(message) => write!(f, "the compile panicked: {message}"),
        }
    }
}

impl std::error::Error for TraceCompileError {}

/// The exits of one member of a tree: where its descriptors start in the
/// plan, and its poll and terminator exits.
pub(super) struct MemberExits {
    /// Index of the descriptor of the member's first guard.
    pub(super) first: usize,
    pub(super) poll: Option<usize>,
    pub(super) terminal: Option<usize>,
    /// For each self call of a function trace: the index of its guard in the
    /// member, and of the unwind exit its activation leaves through when the
    /// callee left.
    pub(super) unwinds: Vec<(usize, usize)>,
}

/// The exits of a trace, in the order native code returns their indices. Each
/// member of the tree contributes one per guard, then its poll exit (the
/// state a link or loop-back publishes when the fuel runs out), then its
/// terminator exit. The root comes first, then the sides in order.
pub(super) struct ExitPlan {
    pub(super) descriptors: Vec<ExitDescriptor>,
    pub(super) members: Vec<MemberExits>,
    /// Slots the largest exit writes.
    pub(super) area_len: usize,
    /// For a function trace: slots per activation in the exit area, the
    /// largest exit's and one more for the index of the exit the activation
    /// left through (see [`CompiledTrace::activation_stride`]).
    pub(super) stride: Option<usize>,
    /// Whether the exits of each member share their stores (see
    /// [`share_exit_stores`]): not in a function trace, whose stubs store
    /// everything.
    shares: bool,
    /// The shared stores, made when Cranelift first asks for them
    /// ([`Self::shared_stores`]). The MIR backend shares the stores of its
    /// own stubs and never reads them.
    shared: std::sync::OnceLock<SharedStores>,
}

/// The stores the exits of a tree's members share, by Cranelift block.
#[derive(Debug, Default)]
pub(super) struct SharedStores {
    /// Stores that several exits of one member have in common. Each is
    /// lowered once, as a block the exits jump to.
    pub(super) tails: Vec<ExitTailPlan>,
    /// Per descriptor, the slots its own stub stores and the first tail of
    /// its chain, if any. Empty for a function trace.
    pub(super) stubs: Vec<ExitStubPlan>,
}

impl ExitPlan {
    /// The stores the exits of each member share (see
    /// [`share_exit_stores`]), made on the first call.
    pub(super) fn shared_stores(&self) -> &SharedStores {
        self.shared.get_or_init(|| {
            let mut shared = SharedStores::default();
            if !self.shares {
                return shared;
            }
            for (member, exits) in self.members.iter().enumerate() {
                let end = self
                    .members
                    .get(member + 1)
                    .map_or(self.descriptors.len(), |next| next.first);
                let first_tail = shared.tails.len();
                let (tails, stubs) = share_exit_stores(&self.descriptors[exits.first..end]);
                shared
                    .tails
                    .extend(tails.into_iter().map(|tail| ExitTailPlan {
                        next: tail.next.map(|next| next + first_tail),
                        ..tail
                    }));
                shared
                    .stubs
                    .extend(stubs.into_iter().map(|stub| ExitStubPlan {
                        tail: stub.tail.map(|tail| tail + first_tail),
                        ..stub
                    }));
            }
            shared
        })
    }
}

/// Stores shared by the exits of a range of one member's exits.
#[derive(Debug, PartialEq)]
pub(super) struct ExitTailPlan {
    /// The values every exit of the range names, sorted by value.
    pub(super) slots: Vec<ExitSlot>,
    /// The tail this one jumps to after its stores, or `None` to return.
    pub(super) next: Option<usize>,
}

/// What the stub of one exit stores before it jumps to its tails.
#[derive(Debug, PartialEq)]
pub(super) struct ExitStubPlan {
    pub(super) own: Vec<ExitSlot>,
    pub(super) tail: Option<usize>,
}

/// One trace of a tree as the backend lowers it.
pub(super) struct LoweredMember<'a> {
    pub(super) ir: &'a TraceIr,
    /// The side traces that continue this member's guards: the guard index
    /// (in this optimized member's numbering) and the member index.
    pub(super) attached: Vec<(usize, usize)>,
    /// The loop side that continues this member's terminal exit (see
    /// [`crate::jit_ir::trace::TERMINAL_GUARD`]): its member index.
    pub(super) terminal: Option<usize>,
    /// For a side that ends in [`TraceTerminator::LinkLoop`]: the member
    /// index of the loop side it rejoins.
    pub(super) link_loop: Option<usize>,
}

/// A trace lowered to machine code. It owns the executable memory, the entry
/// contract and the exit descriptors, and nothing that refers to a live VM.
// Deliberate shape: each bool is an independent trace fact the VM reads, not
// one state machine; packing them into enums would obscure the contract.
#[allow(clippy::struct_excessive_bools)]
pub struct CompiledTrace {
    code: TraceCode,
    tier: OptimizationTier,
    /// The root-frame register and type each input slot must hold.
    entry: Vec<(u16, TraceType)>,
    exits: Vec<ExitDescriptor>,
    exit_area_len: usize,
    /// The traces as lowered, after optimization: the root, then the side
    /// traces that were linked into the code. Exit descriptors name their
    /// snapshots, not the source traces'.
    lowered: Vec<Arc<TraceIr>>,
    /// For each source side trace (in the order of
    /// [`TraceIr::sides`]), its index in `lowered`, or `None` if it was left
    /// out because the optimizer removed or hoisted the guard it continues.
    side_members: Vec<Option<usize>>,
    /// Whether some member loads a table out of an array or reads a field of
    /// a table by lane, so a run needs [`Self::run_with_tables`].
    needs_tables: bool,
    /// [`super::exit::CLOSURE_ROW_WORDS`] when some member reads or writes a callee's cell
    /// natively (see [`emit::closure_cell_use`]), and zero otherwise.
    row_words: usize,
    /// Whether some operation makes a table ([`TraceIr::makes_tables`]).
    makes_tables: bool,
    /// Whether native code probes the string index of each table a hash
    /// lookup binds ([`Self::probes_hash`]).
    probes_hash: bool,
    /// Whether some member calls the VM for a library function.
    calls_host: bool,
    /// Whether some member calls a helper that runs a coroutine
    /// ([`TraceIr::resumes`]), so a run must give the VM a way to do it.
    resumes: bool,
    /// Whether some member sets a string or a table into an array, which
    /// native code leaves to the VM's store entry ([`HostEntry::store`]).
    /// A run without that entry refuses to start.
    stores_references: bool,
    /// Which bindings resolve an array part, by position.
    array_flags: Box<[bool]>,
    /// For each entry input, whether native code may read the row of the
    /// table it holds (see [`entry_row_needs`]). A run fills the rows of the
    /// other inputs only if the code asks for the table's lane later.
    entry_rows: Box<[bool]>,
    /// The entry inputs that hold a table, by position.
    table_inputs: Box<[u16]>,
    /// What the optimizer did, summed over the members of the tree, dropped
    /// sides included.
    optimize_stats: OptimizeStats,
    /// For a function trace, see [`Self::activation_stride`].
    stride: Option<usize>,
    /// Whether the code carries a count hook's budget (see
    /// [`Self::counted`]).
    counted: bool,
    /// The most counted instructions any path of the tree runs from the
    /// start of an iteration to an exit or a loop-back.
    max_counted: u32,
    /// One cell per exit compiled with the tree: empty, or the entry of the
    /// side compiled apart that the exit calls (see [`Self::attach`]). A run
    /// reads the cell of the exit the code left through, and calls what it
    /// finds there.
    cells: Box<[AtomicUsize]>,
    /// How many of [`Self::exits`] were compiled with the tree. The exits of
    /// the sides attached since follow.
    own_exits: usize,
    /// What the tree's code does that a side compiled apart must not
    /// contradict.
    facts: TreeFacts,
    /// Instructions of the function handed to Cranelift for the tree.
    clif_insts: usize,
    /// The table bindings the plan demoted to lane operations (see
    /// [`crate::jit_ir::trace::plan_tree`]).
    demoted_bindings: u32,
    /// The sides compiled apart and attached, in the order they were.
    attached: Vec<AttachedSide>,
    /// What [`Self::cancel_attached`] restores.
    undo: Option<AttachUndo>,
    /// Registers of the frame that sides compiled apart read and the tree
    /// does not, with the type each must hold. A run gives them input slots
    /// after the entry inputs ([`Self::run`]), unchecked: a side that finds
    /// another type declines to run.
    extras: Vec<Extra>,
    /// The input slots the sides compiled apart rewrite when they link back,
    /// with the registers the VM writes them to when a run that linked
    /// leaves. The exit area ends with the slots' values for it (see
    /// [`Self::writeback`]).
    writebacks: Vec<Writeback>,
    /// How many of the first [`Self::writebacks`] are the slots of registers
    /// the root takes untyped that in-tree links rewrite (see
    /// [`crate::jit_ir::trace::untyped_link_writes`]). Native code links
    /// without a trace of it in the run, so the run leaves them every time.
    untyped_writebacks: usize,
    /// For each loop parameter of the root, the entry input it starts from and
    /// its type, when every one starts from an entry input of its own and
    /// holds a number, a boolean, nil, a table or a closure: a side compiled apart that links back
    /// to the loop hands the run the parameters, which it puts in those
    /// input slots to enter the tree again. `None` for a tree that cannot take
    /// such a side.
    link_slots: Option<Box<[(usize, TraceType)]>>,
}

/// Where an input slot after the entry inputs comes from, and where the value
/// goes back to when a run leaves.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExtraSource {
    /// A register of the root frame the entry inputs do not name.
    Register(u16),
    /// A register of the frame that called the root frame (see
    /// [`oximond::jit_ir::trace::ExternalSlot::CallerRegister`]).
    Caller(u16),
    /// A global or an upvalue the side reads and never writes, which the run
    /// reads once at entry like the tree's own window slots: a global of the
    /// caller's chunk or of a pinned closure's, or an upvalue of a pinned
    /// closure.
    External(crate::jit_ir::trace::ExternalSlot),
}

/// A register a side compiled apart reads and the tree's entry does not, with
/// the type the side was recorded for.
pub type Extra = (ExtraSource, TraceType);

/// An input slot that a side compiled apart rewrites when it links back to the
/// tree's loop, and the register the VM must write it to when the run leaves:
/// the tree's own exits know nothing of the side's writes, so they leave the
/// register as the interpreter's frame holds it (see
/// [`CompiledTrace::writeback`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Writeback {
    /// The register of the frame the slot stands for.
    pub source: ExtraSource,
    /// The slot, as a position in the run's input slots (the entry inputs,
    /// then one slot per binding, then the extras).
    pub position: usize,
}

/// The state [`CompiledTrace::prepare_attach`] changes besides appending.
struct AttachUndo {
    exit_area_len: usize,
    optimize_stats: OptimizeStats,
    extras: usize,
    writebacks: usize,
}

/// What the code of a tree does that every part of it must agree on, because
/// the run is set up once for the tree (see [`CompiledTrace::attach`]).
// Deliberate shape: each bool is an independent fact about the tree's code.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeFacts {
    needs_tables: bool,
    makes_tables: bool,
    probes_hash: bool,
    calls_host: bool,
    resumes: bool,
    stores_references: bool,
    /// Some member sets tables, so array words are read again after a set.
    sets_tables: bool,
    /// Some member may move hash storage, so no lookup probes the string
    /// index the VM publishes.
    moves_hash: bool,
    /// Whether the root reads an array key that only the links of the
    /// members it was compiled with keep valid.
    trusted_key: bool,
    row_words: usize,
    /// Whether each row ends in a presence mask (see
    /// [`super::exit::row_stride_of`]). A side compiled apart takes its
    /// tree's.
    row_masked: bool,
}

impl TreeFacts {
    fn of<M: Borrow<TraceIr>>(
        members: &[M],
        row_words: usize,
        row_masked: bool,
        trusted_key: bool,
    ) -> Self {
        let any = |test: fn(&TraceIr) -> bool| members.iter().any(|member| test(member.borrow()));
        Self {
            needs_tables: any(member_needs_tables),
            makes_tables: any(TraceIr::makes_tables),
            probes_hash: any(TraceIr::uses_string_index) && !any(TraceIr::moves_hash_storage),
            calls_host: any(TraceIr::calls_host),
            resumes: any(TraceIr::resumes),
            stores_references: any(member_stores_references),
            sets_tables: any(TraceIr::sets_tables),
            moves_hash: any(TraceIr::moves_hash_storage),
            trusted_key,
            row_words,
            row_masked,
        }
    }

    /// Whether code with these facts, called from code with `tree`'s, asks
    /// the run for nothing the run was not set up for. A side that links
    /// back to the loop also leaves the tree's own assumptions about the
    /// loop true: what it sets and moves is what the tree allowed for.
    fn fits(&self, tree: &Self, links: bool) -> bool {
        let implied = |side: bool, tree: bool| !side || tree;
        implied(self.needs_tables, tree.needs_tables)
            && implied(self.makes_tables, tree.makes_tables)
            && implied(self.calls_host, tree.calls_host)
            && implied(self.resumes, tree.resumes)
            && implied(self.stores_references, tree.stores_references)
            && (self.row_words == 0 || self.row_words == tree.row_words)
            && (!links
                || (implied(self.sets_tables, tree.sets_tables)
                    && implied(self.moves_hash, tree.moves_hash)
                    && !(tree.probes_hash && self.moves_hash)
                    && !tree.trusted_key))
    }
}

/// A side compiled apart from its tree and attached to one of its exits.
struct AttachedSide {
    /// Names the side's exits in the codes the run returns.
    unit: u32,
    code: TraceCode,
    /// Where the side's exits start in [`CompiledTrace::exits`].
    exit_base: usize,
    exits: usize,
    /// One cell per exit of the side, for the sides attached to them.
    cells: Box<[AtomicUsize]>,
    /// The exit that calls the side.
    host: usize,
}

/// A side trace compiled as a function of its own, ready to be attached to
/// the exit of the compiled tree that continues it
/// ([`CompiledTrace::attach`]).
pub struct CompiledSide {
    unit: u32,
    code: TraceCode,
    exits: Vec<ExitDescriptor>,
    exit_area_len: usize,
    lowered: TraceIr,
    host: usize,
    exit_style: bool,
    cells: Box<[AtomicUsize]>,
    facts: TreeFacts,
    optimize_stats: OptimizeStats,
    /// The registers whose tables' rows the side reads, among its entry
    /// inputs (see [`entry_row_needs`]).
    row_registers: Vec<u16>,
    /// The tree's extra inputs after this side: those it had, then the ones
    /// the side needs.
    extras: Vec<Extra>,
    /// The input slots the side rewrites where it links back to the loop.
    writebacks: Vec<Writeback>,
}

impl CompiledSide {
    /// Instructions of the function handed to Cranelift.
    #[must_use]
    pub const fn clif_insts(&self) -> usize {
        self.code.clif_insts()
    }

    /// Bytes of machine code.
    #[must_use]
    pub const fn code_size(&self) -> usize {
        self.code.code_size()
    }

    /// The optimizer's counters for the side.
    #[must_use]
    pub const fn optimize_stats(&self) -> OptimizeStats {
        self.optimize_stats
    }

    /// Whether the side was compiled to leave at the loop header where it
    /// would link back to the loop.
    #[must_use]
    pub const fn exit_style(&self) -> bool {
        self.exit_style
    }
}

/// Why a compiled side could not be attached to its tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachError {
    /// The exit it continues has no cell, or already has a side.
    NoCell,
    /// The side does something the tree's run is not set up for.
    Facts,
    /// The side's tables or fields are not the tree's.
    Contract,
    /// The side is not the one the tree's list of sides expects next.
    Order,
}

/// What a worker needs to know about the compiled tree to compile a side
/// apart from it. Taken on the VM's thread by
/// [`CompiledTrace::side_context`].
#[derive(Clone, Debug)]
pub struct SideContext {
    root: Arc<TraceIr>,
    host: ExitDescriptor,
    host_snapshot: Snapshot,
    host_index: usize,
    root_entry: Vec<(u16, TraceType)>,
    extras: Vec<Extra>,
    tree_facts: TreeFacts,
    unit: u32,
}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<CompiledTrace>();
};

impl fmt::Debug for CompiledTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledTrace")
            .field("tier", &self.tier)
            .field("entry", &self.entry)
            .field("exits", &self.exits.len())
            .field("code_size", &self.code.code_size())
            .finish_non_exhaustive()
    }
}

impl CompiledTrace {
    /// A placeholder for a root whose tree is still being recorded: no code,
    /// no exits, the root's entry contract at the baseline tier. The policy
    /// installs it so that the sides recorded ahead of the tree's compile
    /// have a trace to join, and never runs it (see [`Self::is_staged`]).
    #[must_use]
    pub fn staged(ir: &TraceIr) -> Self {
        let array_flags: Box<[bool]> = ir.bindings.iter().map(TableBinding::is_array).collect();
        let entry: Vec<(u16, TraceType)> = ir
            .entry
            .inputs
            .iter()
            .map(|input| (input.register, input.ty))
            .collect();
        let table_inputs: Box<[u16]> = entry
            .iter()
            .enumerate()
            .filter(|(_, (_, ty))| *ty == TraceType::Table)
            .filter_map(|(input, _)| u16::try_from(input).ok())
            .collect();
        Self {
            code: TraceCode::staged(),
            tier: OptimizationTier::Baseline,
            entry,
            exits: Vec::new(),
            exit_area_len: 0,
            lowered: vec![Arc::new(ir.clone())],
            side_members: Vec::new(),
            needs_tables: false,
            row_words: 0,
            makes_tables: false,
            probes_hash: false,
            calls_host: false,
            resumes: false,
            stores_references: false,
            array_flags,
            entry_rows: vec![false; ir.entry.inputs.len()].into_boxed_slice(),
            table_inputs,
            optimize_stats: OptimizeStats::default(),
            stride: None,
            counted: false,
            max_counted: 0,
            cells: Box::new([]),
            own_exits: 0,
            facts: TreeFacts::default(),
            clif_insts: 0,
            demoted_bindings: 0,
            attached: Vec::new(),
            undo: None,
            extras: Vec::new(),
            writebacks: Vec::new(),
            untyped_writebacks: 0,
            link_slots: None,
        }
    }

    /// Whether this is the placeholder of [`Self::staged`], which holds no
    /// code to run.
    #[must_use]
    pub fn is_staged(&self) -> bool {
        self.exits.is_empty() && self.code.code_size() == 0
    }

    /// The optimizer's counters, summed over every member of the tree.
    #[must_use]
    pub const fn optimize_stats(&self) -> OptimizeStats {
        self.optimize_stats
    }

    /// The root trace this code was lowered from, after optimization. Its
    /// snapshots are the ones [`ExitDescriptor::snapshot`] names for exits
    /// with [`ExitDescriptor::member`] 0.
    #[must_use]
    pub fn lowered_ir(&self) -> &TraceIr {
        &self.lowered[0]
    }

    /// The optimized trace of tree member `member` (0 is the root), whose
    /// snapshots the exits of that member name.
    ///
    /// # Panics
    ///
    /// Panics if `member` is not an [`ExitDescriptor::member`] of this code.
    #[must_use]
    pub fn lowered_member(&self, member: usize) -> &TraceIr {
        &self.lowered[member]
    }

    /// Traces in the tree: the root and the sides linked into the code.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.lowered.len()
    }

    /// The tree member of each source side trace, in the order of
    /// [`TraceIr::sides`]. `None` marks a side the code does not contain,
    /// because the guard it continues is gone or runs before the loop.
    #[must_use]
    pub fn side_members(&self) -> &[Option<usize>] {
        &self.side_members
    }

    /// The optimization tier this code was compiled at.
    #[must_use]
    pub const fn tier(&self) -> OptimizationTier {
        self.tier
    }

    /// Whether the root's loop runs in the fast version its range plan makes.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn versioned_loop(&self) -> bool {
        self.code.versioned
    }

    /// Whether the root's exits call the sides compiled apart from native
    /// code and link back without leaving it.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn links_natively(&self) -> bool {
        self.code.native_link
    }

    /// The root-frame register and type of each entry input, in the order
    /// [`Self::run`] expects its input slots.
    #[must_use]
    pub fn entry_contract(&self) -> &[(u16, TraceType)] {
        &self.entry
    }

    /// Every way out of the trace.
    #[must_use]
    pub fn exits(&self) -> &[ExitDescriptor] {
        &self.exits
    }

    /// The table storage the caller resolves at entry. [`Self::run`] expects
    /// one input slot per binding after the entry inputs, in this order.
    #[must_use]
    pub fn bindings(&self) -> &[TableBinding] {
        &self.lowered[0].bindings
    }

    /// The fields of run-time tables the trace reads and writes, in the order
    /// of every row [`TraceTables::row`] fills.
    #[must_use]
    pub fn fields(&self) -> &[TableField] {
        &self.lowered[0].fields
    }

    /// Whether native code reads and writes the cells of closures through
    /// their rows ([`super::exit::CLOSURE_ROW_WORDS`]).
    #[must_use]
    pub const fn uses_closure_rows(&self) -> bool {
        self.row_words > 0
    }

    /// Whether the closure rows also name the lane of the table each cell
    /// holds, because the code reads table-valued cells itself
    /// ([`super::exit::CLOSURE_ROW_TABLE_WORDS`]).
    #[must_use]
    pub const fn uses_closure_table_cells(&self) -> bool {
        self.row_words >= super::exit::CLOSURE_ROW_TABLE_WORDS
    }

    /// The words of every row: a table's field slots, or a closure's
    /// prototype and cell addresses ([`super::exit::CLOSURE_ROW_WORDS`]) when the code
    /// reads and writes callee cells itself.
    #[must_use]
    pub fn row_stride(&self) -> usize {
        super::exit::row_stride_of(
            self.lowered[0].fields.len(),
            self.row_words,
            self.facts.row_masked,
        )
    }

    /// Whether a run needs [`Self::run_with_tables`]: some member loads a
    /// table out of an array or reads or writes a field of a table by lane.
    #[must_use]
    pub const fn needs_tables(&self) -> bool {
        self.needs_tables
    }

    /// Whether the trace makes tables in its run ([`TraceIr::makes_tables`]).
    /// The arena of tables must have room for them before the entry resolves
    /// any table address, since a table made when the arena is full moves
    /// every table in it.
    #[must_use]
    pub const fn makes_tables(&self) -> bool {
        self.makes_tables
    }

    /// Whether the code probes a string index of the table of each
    /// [`BindingKind::Hash`] and [`BindingKind::NoMetatable`] binding, so the
    /// bindings of a run should carry one ([`TraceBinding::probe`]) when the
    /// runtime has it. A binding without one is fine: the code asks the VM.
    #[must_use]
    pub const fn probes_hash(&self) -> bool {
        self.probes_hash
    }

    /// Whether some member calls the VM for a library function
    /// ([`oximond::jit_ir::trace::TraceOp::HostCall`]).
    #[must_use]
    pub const fn calls_host(&self) -> bool {
        self.calls_host
    }

    /// Whether some member calls a helper that runs a coroutine
    /// ([`oximond::jit_ir::trace::HelperEffects::RESUMES`]).
    #[must_use]
    pub const fn resumes(&self) -> bool {
        self.resumes
    }

    /// Which bindings resolve an array part, by position.
    #[must_use]
    pub fn array_flags(&self) -> &[bool] {
        &self.array_flags
    }

    /// Size of the exit area [`Self::run`] needs, in 8-byte slots.
    #[must_use]
    pub fn exit_area_len(&self) -> usize {
        self.exit_area_len + self.writeback_words()
    }

    /// For a function trace ([`TraceIr::function`]): how many slots of the
    /// exit area each activation owns. Activation `k` (0 for the one the VM
    /// entered) owns slots `k * stride ..`: its exit spills there, and its
    /// last slot holds the index of the exit it left through. When a run
    /// leaves, [`TraceControl::depth`] is the activation that left, whose
    /// exit is the one the run returns; every activation below it left
    /// through the unwind exit ([`super::TraceExitClass::Unwind`]) its last
    /// slot names. The exit area needs `stride * (depth_cap + 1)` slots.
    /// `None` for a loop trace.
    #[must_use]
    pub const fn activation_stride(&self) -> Option<usize> {
        self.stride
    }

    /// Whether the code accounts for a count hook. A counted trace takes the
    /// hook's budget from [`TraceControl::budget`] on entry, subtracts the
    /// counted instructions of each iteration it completes, and reports the
    /// budget of the iteration it left in the same field, so that the caller
    /// subtracts the exit's [`ExitDescriptor::counted`] to find the budget
    /// left. A run must start with a budget above [`Self::max_counted`] and
    /// give fuel for no more iterations than that budget covers, since the
    /// code never tests the budget itself.
    #[must_use]
    pub const fn counted(&self) -> bool {
        self.counted
    }

    /// The most counted instructions one iteration can run (zero unless
    /// [`Self::counted`]).
    #[must_use]
    pub const fn max_counted(&self) -> u32 {
        self.max_counted
    }

    /// Calls the sides compiled apart that the exit `result` names leads to,
    /// and the sides of their exits in turn, until the run leaves for good.
    ///
    /// A side that links back to the loop leaves the loop parameters in the
    /// exit area. They go to the entry input slots the parameters start from
    /// and the tree is entered again, which runs the loop on from there with
    /// the state of the run (its tables, its lanes, its fuel) as it is.
    /// `shared` says that `inputs` points into the caller's slots, which this
    /// leaves alone: the first link copies them.
    fn run_sides(
        &self,
        mut inputs: *mut TraceInputSlot,
        shared: bool,
        control: &mut TraceControl,
        exit_area: &mut [i64],
        mut result: i64,
    ) -> i64 {
        let mut owned: Vec<TraceInputSlot> = Vec::new();
        let mut linked = false;
        loop {
            if control.followed != 0 {
                // Native code followed the chain for this code already.
                control.followed = 0;
            } else {
                result = self.call_sides(inputs, control, exit_area.as_mut_ptr(), result);
            }
            if result != emit::LINK_CODE {
                // A link native code took already re-entered the loop
                // itself and left its mark in the control block.
                let native = control.linked != 0;
                self.leave_writebacks(inputs, exit_area, linked || native);
                return result;
            }
            linked = true;
            let slots = self
                .link_slots
                .as_ref()
                .expect("only a tree with link slots takes a side that links back");
            if shared && owned.is_empty() {
                owned = emit::copy_inputs(inputs, self.entry.len() + self.extras.len());
                inputs = owned.as_mut_ptr();
            }
            for (index, (position, ty)) in slots.iter().enumerate() {
                let bits = exit_area.get(index).copied().unwrap_or(0);
                let slot = match ty {
                    TraceType::Float => TraceInputSlot::from_words(TRACE_TAG_FLOAT, bits),
                    TraceType::Integer => TraceInputSlot::from_words(TRACE_TAG_INTEGER, bits),
                    TraceType::Boolean => TraceInputSlot::from_words(TRACE_TAG_BOOLEAN, bits),
                    // An object is a lane of the run's bank, which the side
                    // found or made and which the bank keeps for the run.
                    TraceType::Table => TraceInputSlot::table(bits.cast_unsigned()),
                    TraceType::Closure => TraceInputSlot::closure(bits.cast_unsigned()),
                    _ => TraceInputSlot::nil(),
                };
                emit::write_input(inputs, *position, slot);
            }
            if self.code.native_link {
                control.inputs = inputs as usize;
            }
            result = self.code.run(inputs, control, exit_area);
            if control.linked != 0 {
                // Native code linked and ran on in the loop: its result is
                // the code of the exit the run left through.
                linked = true;
            }
        }
    }

    /// Leaves for the VM, at the end of the exit area, the slots that links of
    /// the run may have rewritten (see [`Self::writeback`]): a flag that says
    /// whether a link happened, and for each of the slots its tag and bits.
    fn leave_writebacks(&self, inputs: *const TraceInputSlot, exit_area: &mut [i64], linked: bool) {
        if self.writebacks.is_empty() {
            return;
        }
        // A slot that no link rewrote holds the entry's value, which the
        // frame holds too: writing it back changes nothing.
        let linked = linked || self.untyped_writebacks > 0;
        let base = self.exit_area_len;
        exit_area[base] = i64::from(linked);
        if !linked {
            return;
        }
        for (index, writeback) in self.writebacks.iter().enumerate() {
            let slot = emit::read_input(inputs, writeback.position);
            exit_area[base + 1 + 2 * index] = slot.tag();
            exit_area[base + 2 + 2 * index] = slot.bits();
        }
    }

    /// Calls the side whose cell the exit `result` names holds, and the side
    /// of the exit that one left through, and so on. Returns what the last
    /// returned: the code of the exit the run leaves through, or
    /// [`emit::LINK_CODE`].
    fn call_sides(
        &self,
        inputs: *mut TraceInputSlot,
        control: &mut TraceControl,
        exit_area: *mut i64,
        mut result: i64,
    ) -> i64 {
        loop {
            let Ok(code) = usize::try_from(result) else {
                return result;
            };
            let Some(cell) = self.cell_for_code(code) else {
                return result;
            };
            let address = cell.load(Ordering::Acquire);
            if address == 0 {
                return result;
            }
            control.child_calls += 1;
            control.links += 1;
            let next = emit::call_side(address, inputs, control, exit_area);
            if next == emit::DECLINE_CODE {
                // The side did not run: the exit is the one the code left by.
                return result;
            }
            result = next;
        }
    }

    /// The cell of the exit a code names, if the code has one.
    fn cell_for_code(&self, code: usize) -> Option<&AtomicUsize> {
        if code >> emit::SIDE_CODE_SHIFT == 0 {
            return self.cells.get(code);
        }
        let unit = u32::try_from(code >> emit::SIDE_CODE_SHIFT).ok()?;
        let local = code & ((1 << emit::SIDE_CODE_SHIFT) - 1);
        self.attached
            .iter()
            .find(|side| side.unit == unit)
            .and_then(|side| side.cells.get(local))
    }

    /// The index in [`Self::exits`] of the exit a side compiled apart
    /// returned `code` for.
    fn side_exit(&self, code: usize) -> usize {
        let unit = u32::try_from(code >> emit::SIDE_CODE_SHIFT).unwrap_or(u32::MAX);
        let local = code & ((1 << emit::SIDE_CODE_SHIFT) - 1);
        self.attached
            .iter()
            .find(|side| side.unit == unit)
            .map_or(usize::MAX, |side| side.exit_base + local)
    }

    /// The cell of exit `index`, if the code has one.
    fn cell(&self, index: usize) -> Option<&AtomicUsize> {
        if index < self.own_exits {
            return self.cells.get(index);
        }
        self.attached
            .iter()
            .find(|side| index >= side.exit_base && index < side.exit_base + side.exits)
            .and_then(|side| side.cells.get(index - side.exit_base))
    }

    /// Whether exit `index` can call a side compiled apart: the code has a
    /// cell for it, and no side is attached to it yet.
    #[must_use]
    pub fn can_attach_at(&self, index: usize) -> bool {
        self.cell(index)
            .is_some_and(|cell| cell.load(Ordering::Acquire) == 0)
    }

    /// What a worker needs to compile a side that continues exit `index`,
    /// to be named `unit` in the codes it returns. `None` if the exit cannot
    /// call one.
    #[must_use]
    pub fn side_context(&self, index: usize, unit: u32) -> Option<SideContext> {
        if !self.can_attach_at(index) || unit == 0 || unit >= 1 << 30 {
            return None;
        }
        let host = self.exits.get(index)?.clone();
        let host_snapshot = self
            .lowered
            .get(host.member)?
            .snapshots
            .get(host.snapshot.index())?
            .clone();
        Some(SideContext {
            root: Arc::clone(&self.lowered[0]),
            host,
            host_snapshot,
            host_index: index,
            root_entry: self.entry.clone(),
            extras: self.extras.clone(),
            tree_facts: self.facts,
            unit,
        })
    }

    /// The units of the sides compiled apart that are attached.
    #[must_use]
    pub fn attached_sides(&self) -> usize {
        self.attached.len()
    }

    /// Adds `side`, a trace compiled apart, to the tree without making it
    /// run: [`Self::publish_attached`] does that, and
    /// [`Self::cancel_attached`] takes it out again. The caller checks what
    /// the grown tree needs (the restore plans of the new exits) in between.
    ///
    /// The exits of the side join [`Self::exits`] after the existing ones,
    /// its trace joins the lowered members, and the tree's exit area grows to
    /// the side's if it must. `source` is the side's index in the tree's list
    /// of sides, which `side_members` must be ready for.
    ///
    /// # Errors
    ///
    /// [`AttachError`], having changed nothing.
    pub fn prepare_attach(&mut self, side: CompiledSide, source: usize) -> Result<(), AttachError> {
        if self.side_members.len() != source {
            return Err(AttachError::Order);
        }
        if !self.can_attach_at(side.host) {
            return Err(AttachError::NoCell);
        }
        if !side.facts.fits(&self.facts, !side.exit_style) {
            return Err(AttachError::Facts);
        }
        let root = &self.lowered[0];
        // The side reads rows of the tables its entry inputs hold; the run
        // fills a row only for the tree's inputs that the tree itself reads
        // rows of, and a zero row sends every access out of the side.
        let rows_missing = side.row_registers.iter().any(|register| {
            self.entry
                .iter()
                .position(|(held, _)| held == register)
                .is_some_and(|position| !self.entry_rows.get(position).copied().unwrap_or(false))
        });
        if side.lowered.bindings != root.bindings
            || side.lowered.fields != root.fields
            || rows_missing
        {
            return Err(AttachError::Contract);
        }
        if !side.extras.starts_with(&self.extras) {
            return Err(AttachError::Contract);
        }
        // Slots another side rewrites hold the same register, in the same
        // place, for every side.
        let mut merged = self.writebacks.clone();
        // A link enters the root again with the loop parameters in the link
        // slots. An exit of the root that keeps a register as the entry gave
        // it (an exit before the iteration writes it) leaves the frame's
        // register alone, and the frame still holds the value of the run's
        // first entry: the slots go back too, before the exit's own writes.
        // A window slot of the root goes back to its upvalue cell, or to the
        // caller's register; the other window slots are read-only.
        let root_ir = &self.lowered[0];
        let guest = root_ir
            .frames
            .first()
            .map_or(0, |frame| usize::from(frame.register_count))
            .saturating_sub(root_ir.external.len());
        let link_writebacks = self
            .link_slots
            .iter()
            .flatten()
            .filter_map(|&(position, _)| {
                let &(register, _) = self.entry.get(position)?;
                let source = match usize::from(register).checked_sub(guest) {
                    None => ExtraSource::Register(register),
                    Some(window) => match *root_ir.external.get(window)? {
                        slot @ (crate::jit_ir::trace::ExternalSlot::Upvalue(_)
                        | crate::jit_ir::trace::ExternalSlot::ClosureUpvalue { .. }) => {
                            ExtraSource::External(slot)
                        }
                        crate::jit_ir::trace::ExternalSlot::CallerRegister(register) => {
                            ExtraSource::Caller(register)
                        }
                        _ => return None,
                    },
                };
                Some(Writeback { source, position })
            });
        for writeback in side.writebacks.iter().copied().chain(link_writebacks) {
            match merged.iter().find(|known| known.source == writeback.source) {
                Some(known) if known.position != writeback.position => {
                    return Err(AttachError::Contract);
                }
                Some(_) => {}
                None => merged.push(writeback),
            }
        }
        let member = self.lowered.len();
        let exit_base = self.exits.len();
        self.undo = Some(AttachUndo {
            exit_area_len: self.exit_area_len,
            optimize_stats: self.optimize_stats,
            extras: self.extras.len(),
            writebacks: self.writebacks.len(),
        });
        self.writebacks = merged;
        self.extras = side.extras;
        self.exit_area_len = self.exit_area_len.max(side.exit_area_len);
        self.optimize_stats.accumulate(side.optimize_stats);
        self.lowered.push(Arc::new(side.lowered));
        self.side_members.push(Some(member));
        let count = side.exits.len();
        for mut exit in side.exits {
            exit.member = member;
            self.exits.push(exit);
        }
        self.attached.push(AttachedSide {
            unit: side.unit,
            code: side.code,
            exit_base,
            exits: count,
            cells: side.cells,
            host: side.host,
        });
        Ok(())
    }

    /// Makes the side added by [`Self::prepare_attach`] run, from now on:
    /// the cell of the exit that calls it is set to the side's entry. That is
    /// one aligned store of a word, which the code reads once when it leaves
    /// through the exit, so it sees either no side or a whole one: the side's
    /// code was finalized and everything the call needs was written before
    /// the store (release), and the code is owned by this tree, which frees
    /// it with the cell. The VM installs between runs, so no run is in
    /// progress either.
    ///
    /// # Panics
    ///
    /// Panics if no side is waiting to be published.
    pub fn publish_attached(&mut self) {
        let side = self.attached.last().expect("a side was prepared");
        let entry = side.code.entry_address();
        let host = side.host;
        self.undo = None;
        let cell = self.cell(host).expect("the host's cell was checked");
        cell.store(entry, Ordering::Release);
    }

    /// Takes out the side [`Self::prepare_attach`] added and
    /// [`Self::publish_attached`] has not published.
    ///
    /// # Panics
    ///
    /// Panics if no side is waiting to be published.
    pub fn cancel_attached(&mut self) {
        let undo = self.undo.take().expect("a side was prepared");
        let side = self.attached.pop().expect("a side was prepared");
        self.exits.truncate(side.exit_base);
        self.lowered.pop();
        self.side_members.pop();
        self.exit_area_len = undo.exit_area_len;
        self.optimize_stats = undo.optimize_stats;
        self.extras.truncate(undo.extras);
        self.writebacks.truncate(undo.writebacks);
    }

    /// The registers a run gives input slots after the entry inputs, for the
    /// sides compiled apart that read them.
    #[must_use]
    pub fn extras(&self) -> &[Extra] {
        &self.extras
    }

    /// The input slots sides compiled apart rewrite when they link back to the
    /// loop, with the register of the frame each stands for. A run that
    /// linked leaves their values at the end of the exit area
    /// ([`Self::writeback`]), for the VM to write into the frames before it
    /// restores the exit: the exits of the tree leave a register a side
    /// wrote as the frame holds it.
    #[must_use]
    pub fn writebacks(&self) -> &[Writeback] {
        &self.writebacks
    }

    /// The values of the slots of [`Self::writebacks`] that the run which
    /// filled `exit_area` ended with, as a tag and bits for each, if any link
    /// of the run rewrote them.
    #[must_use]
    pub fn writeback<'a>(&self, exit_area: &'a [i64]) -> Option<&'a [i64]> {
        if self.writebacks.is_empty() || exit_area.get(self.exit_area_len).copied()? == 0 {
            return None;
        }
        exit_area.get(self.exit_area_len + 1..self.exit_area_len + 1 + 2 * self.writebacks.len())
    }

    /// The words the end of the exit area holds for [`Self::writeback`].
    fn writeback_words(&self) -> usize {
        if self.writebacks.is_empty() {
            0
        } else {
            1 + 2 * self.writebacks.len()
        }
    }

    /// [`Self::prepare_attach`] and [`Self::publish_attached`] in one.
    ///
    /// # Errors
    ///
    /// As [`Self::prepare_attach`].
    pub fn attach(&mut self, side: CompiledSide, source: usize) -> Result<(), AttachError> {
        self.prepare_attach(side, source)?;
        self.publish_attached();
        Ok(())
    }

    /// Whether a side that links back to the loop can be attached: the tree
    /// reads no array key that only the links it was compiled with keep valid.
    #[must_use]
    pub const fn allows_linked_sides(&self) -> bool {
        !self.facts.trusted_key && self.link_slots.is_some()
    }

    /// Bytes of machine code of each function the tree owns: its own, then
    /// one for each side compiled apart. Each is a module of its own, which
    /// occupies whole pages.
    pub fn code_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(self.code.code_size())
            .chain(self.attached.iter().map(|side| side.code.code_size()))
    }

    /// The machine code of the tree as `(address, size)` pairs: its own, then
    /// one for each side compiled apart.
    #[must_use]
    pub fn code_ranges(&self) -> Vec<(usize, usize)> {
        std::iter::once(&self.code)
            .chain(self.attached.iter().map(|side| &side.code))
            .map(|code| (code.entry_address(), code.code_size()))
            .collect()
    }

    /// Instructions handed to Cranelift for the tree and for every side
    /// attached to it.
    #[must_use]
    pub fn clif_insts_total(&self) -> usize {
        self.clif_insts
            + self
                .attached
                .iter()
                .map(|side| side.code.clif_insts())
                .sum::<usize>()
    }

    /// The side compiled apart that ended up last, for the counters: the
    /// instructions handed to Cranelift for it.
    #[must_use]
    pub fn attached_clif_insts(&self) -> usize {
        self.attached
            .last()
            .map_or(0, |side| side.code.clif_insts())
    }

    /// Whether attaching changed the exit area a run needs: its length now.
    /// The VM reads it before each run.
    #[must_use]
    pub const fn attached_exit_area(&self) -> usize {
        self.exit_area_len
    }

    /// The single-run patterns the tree's code scans itself, its own and
    /// those of the sides compiled apart: a diagnostic for tests.
    #[must_use]
    pub fn inline_runs(&self) -> usize {
        self.code.inline_runs()
            + self
                .attached
                .iter()
                .map(|side| side.code.inline_runs())
                .sum::<usize>()
    }

    /// Bytes of machine code, for the cache's byte accounting.
    #[must_use]
    pub fn code_size(&self) -> usize {
        self.code.code_size()
            + self
                .attached
                .iter()
                .map(|side| side.code.code_size())
                .sum::<usize>()
    }

    /// Instructions of the function handed to Cranelift for the tree's code,
    /// not counting the sides attached since.
    #[must_use]
    pub const fn clif_insts(&self) -> usize {
        self.clif_insts
    }

    /// The table bindings the tree's plan demoted to lane operations because
    /// a member writes their register.
    #[must_use]
    pub const fn demoted_bindings(&self) -> u32 {
        self.demoted_bindings
    }

    /// The [`TraceOp::LaneArrayLoad`]s of the tree's optimized members, the
    /// array reads of tables met by lane its code makes inline.
    #[must_use]
    pub fn lane_array_loads(&self) -> usize {
        self.lowered
            .iter()
            .flat_map(|member| member.preheader.iter().chain(&member.body))
            .filter(|instruction| matches!(instruction.op, TraceOp::LaneArrayLoad { .. }))
            .count()
    }

    /// The [`TraceOp::LaneArrayStore`]s of the tree's optimized members, the
    /// in-place array stores into tables met by lane its code makes inline.
    #[must_use]
    pub fn lane_array_stores(&self) -> usize {
        self.lowered
            .iter()
            .flat_map(|member| member.preheader.iter().chain(&member.body))
            .filter(|instruction| matches!(instruction.op, TraceOp::LaneArrayStore { .. }))
            .count()
    }

    /// The members of the tree that rejoin the loop of a loop side they
    /// descend from ([`TraceTerminator::LinkLoop`]).
    #[must_use]
    pub fn loop_links(&self) -> usize {
        self.lowered
            .iter()
            .filter(|member| matches!(member.terminator, TraceTerminator::LinkLoop { .. }))
            .count()
    }

    /// The [`TraceOp::Select`]s of the tree's optimized members: the pure
    /// two-arm branches its code runs without a guard.
    #[must_use]
    pub fn value_selects(&self) -> usize {
        self.lowered
            .iter()
            .flat_map(|member| member.preheader.iter().chain(&member.body))
            .filter(|instruction| matches!(instruction.op, TraceOp::Select { .. }))
            .count()
    }

    /// Runs a trace that binds no table storage from its loop header.
    ///
    /// `inputs` has one slot per entry input, in [`Self::entry_contract`]
    /// order, and `exit_area` at least [`Self::exit_area_len`] slots. On
    /// return `control.fuel` holds the fuel left. After
    /// [`TraceOutcome::Exit`], the descriptor and `exit_area` describe the
    /// values to restore.
    ///
    /// # Panics
    ///
    /// Panics if the slices are too short or the trace has table bindings
    /// (see [`Self::run_bound`]).
    pub fn run(
        &self,
        inputs: &[TraceInputSlot],
        control: &mut TraceControl,
        exit_area: &mut [i64],
    ) -> TraceOutcome {
        self.run_bound(inputs, &[], control, exit_area)
    }

    /// [`Self::run`] for a trace with table bindings. `bindings` has one
    /// entry per [`Self::bindings`] item, of the matching kind. Only the
    /// runtime can make a [`TraceBinding`], so safe callers cannot hand
    /// native code an address. Its views' lifetime contract holds for the
    /// call: no table mutation between building them and this return.
    ///
    /// # Panics
    ///
    /// Panics if the slices are too short, or `bindings` does not match
    /// [`Self::bindings`] in count and kind. Native code would read or write
    /// out of bounds.
    pub fn run_bound(
        &self,
        inputs: &[TraceInputSlot],
        bindings: &[TraceBinding],
        control: &mut TraceControl,
        exit_area: &mut [i64],
    ) -> TraceOutcome {
        assert!(
            !self.needs_tables,
            "a trace that reads tables by lane needs run_with_tables"
        );
        self.run_checked(inputs, bindings, control, exit_area)
    }

    /// [`Self::run_bound`] for a trace that loads tables out of arrays or
    /// reads fields of tables by lane ([`Self::needs_tables`]). `tables` gives
    /// out lanes and fills rows, and `space` holds the rows. Every table of
    /// an entry input gets its row before the code starts, so the VM only has
    /// to give lanes to tables native code meets on its own.
    ///
    /// # Panics
    ///
    /// As [`Self::run_bound`].
    pub fn run_with_tables(
        &self,
        inputs: &[TraceInputSlot],
        bindings: &[TraceBinding],
        control: &mut TraceControl,
        exit_area: &mut [i64],
        space: &mut TraceTableSpace,
        tables: &mut dyn TraceTables,
    ) -> TraceOutcome {
        let lane_cap = if self.lowered[0].function.is_some() {
            MAX_FUNCTION_TABLE_LANES
        } else {
            MAX_TRACE_TABLE_LANES
        };
        space.begin(self.row_stride(), lane_cap, self.facts.row_masked);
        if self.row_words > 0 {
            // A closure past the space has no row, and code indexing it would
            // read past the storage.
            for (input, (slot, (_, ty))) in inputs.iter().zip(&self.entry).enumerate() {
                if *ty == TraceType::Closure
                    && slot.tag == TRACE_TAG_CLOSURE
                    && !usize::try_from(slot.bits).is_ok_and(|lane| lane < lane_cap)
                {
                    return TraceOutcome::EntryMismatch { input };
                }
            }
        }
        for &input in &*self.table_inputs {
            let input = usize::from(input);
            let Some(slot) = inputs.get(input) else {
                break;
            };
            if slot.tag != TRACE_TAG_TABLE {
                continue;
            }
            // A table past the space has no row, and code indexing it would
            // read past the storage: it is not an input this run can take.
            match usize::try_from(slot.bits) {
                Ok(lane) if lane < lane_cap => {
                    if self.entry_rows[input] {
                        space.ensure_row(lane, tables);
                    } else {
                        space.clear_row(lane);
                    }
                }
                _ => return TraceOutcome::EntryMismatch { input },
            }
        }
        if self.row_words > 0 {
            // The closures of the bank (the pinned ones, and the entry's) get
            // their rows before the code reads one.
            let mut lanes = std::mem::take(&mut space.scratch);
            lanes.clear();
            tables.closure_lanes(&mut lanes);
            let mut too_far = false;
            for &lane in &lanes {
                if lane >= lane_cap {
                    too_far = true;
                    break;
                }
                space.ensure_row(lane, tables);
            }
            space.scratch = lanes;
            if too_far {
                return TraceOutcome::EntryMismatch { input: 0 };
            }
        }
        control.rows = space.rows_address();
        control.bank = tables.bank_address();
        (control.roots, control.roots_len) = tables.roots_address();
        let mut indexed = false;
        if let Some((index, ready)) = tables.lane_index() {
            control.lane_index = index;
            control.lane_ready = u64::from(ready);
            indexed = true;
        }
        // A store of a string into an appendable table goes to the VM's entry
        // too, so the entry is asked for whether or not the code calls the
        // host.
        let entry = tables.host_entry();
        if self.calls_host {
            tables.string_lengths(&mut space.lengths);
            tables.row_flags(&mut space.filled);
            tables.row_space(space);
            space.lengths[super::exit::TRACE_PENDING_STRINGS] = 0;
            control.lengths = space.lengths_address();
        }
        let fresh = tables.admits_fresh();
        let mut dispatch = TableDispatch {
            space,
            tables,
            arrays: &self.array_flags,
            bindings: bindings.len(),
            hash_calls: 0,
            fresh,
            indexed,
        };
        control.tables = std::ptr::from_mut(&mut dispatch) as usize;
        match entry {
            Some(entry) if entry.set != 0 => {
                control.host_entry = entry.function;
                control.host_context = entry.context;
                control.set_entry = entry.set;
                control.store_entry = entry.store;
            }
            _ => {
                control.host_entry = super::emit::trace_host_call as *const () as usize;
                control.host_context = control.tables;
                control.set_entry = super::emit::trace_table_set as *const () as usize;
                control.store_entry = 0;
            }
        }
        let outcome = self.run_checked(inputs, bindings, control, exit_area);
        #[cfg(debug_assertions)]
        dispatch.space.assert_sealed();
        control.hash_calls += dispatch.hash_calls;
        control.rows = 0;
        control.bank = 0;
        control.roots = 0;
        control.roots_len = 0;
        control.lane_index = 0;
        control.lane_ready = 0;
        control.lengths = 0;
        control.host_entry = 0;
        control.host_context = 0;
        control.set_entry = 0;
        control.store_entry = 0;
        control.tables = 0;
        outcome
    }

    fn run_checked(
        &self,
        inputs: &[TraceInputSlot],
        bindings: &[TraceBinding],
        control: &mut TraceControl,
        exit_area: &mut [i64],
    ) -> TraceOutcome {
        // The code calls the store entry to put a string or a table into an
        // array. A host without one (`control.store_entry` is zero) cannot
        // run it, and a call through zero is no answer. The host chooses its
        // entry per run, so the compile cannot refuse the trace for it.
        if self.stores_references && control.store_entry == 0 {
            return TraceOutcome::StoreEntryMissing;
        }
        assert!(
            inputs.len() >= self.entry.len() + self.extras.len(),
            "trace needs {} input slots, got {}",
            self.entry.len() + self.extras.len(),
            inputs.len()
        );
        assert!(
            bindings.len() == self.bindings().len()
                && bindings
                    .iter()
                    .zip(self.bindings())
                    .all(|(given, wanted)| given.matches(&wanted.kind)),
            "trace needs {} bindings of matching kinds, got {}",
            self.bindings().len(),
            bindings.len()
        );
        let area = match self.stride {
            Some(stride) => {
                assert!(control.depth == 0 && control.depth_cap >= 0);
                stride * (usize::try_from(control.depth_cap).expect("a depth cap fits usize") + 1)
            }
            None => self.exit_area_len + self.writeback_words(),
        };
        assert!(
            exit_area.len() >= area,
            "trace needs an exit area of {area} slots, got {}",
            exit_area.len()
        );
        let entry_inputs = inputs;
        let total = self.entry.len() + bindings.len() + self.extras.len();
        let mut stack = [std::mem::MaybeUninit::<TraceInputSlot>::uninit(); 32];
        let mut heap = Vec::new();
        // A table set rewrites the binding slots, so native code gets a
        // buffer of its own whenever the trace binds anything. Without
        // bindings it only reads, and the caller's slots serve.
        // A tree whose exits link natively writes the link slots into the
        // inputs when it enters the loop again (see
        // `Lowerer::finish_native_dispatch`), so it gets a buffer of its own
        // too. The others copy the inputs at their first link through the run.
        let shared = bindings.is_empty() && self.writebacks.is_empty() && !self.code.native_link;
        let inputs: *mut TraceInputSlot = if shared {
            inputs.as_ptr().cast_mut()
        } else {
            let entry_len = self.entry.len();
            let words: &mut [std::mem::MaybeUninit<TraceInputSlot>] = if total <= stack.len() {
                &mut stack[..total]
            } else {
                heap.resize(total, std::mem::MaybeUninit::uninit());
                &mut heap[..]
            };
            // Every word is written before native code reads one.
            let (entry, bound) = words.split_at_mut(entry_len);
            for (word, held) in entry.iter_mut().zip(&inputs[..entry_len]) {
                word.write(*held);
            }
            let (bound, extra) = bound.split_at_mut(bindings.len());
            for (word, binding) in bound.iter_mut().zip(bindings) {
                word.write(binding.word());
            }
            for (word, held) in extra
                .iter_mut()
                .zip(&inputs[entry_len..entry_len + self.extras.len()])
            {
                word.write(*held);
            }
            words.as_mut_ptr().cast::<TraceInputSlot>()
        };
        control.cells = self.cells.as_ptr() as usize;
        control.tree = std::ptr::from_ref(self) as usize;
        if self.code.native_link {
            control.inputs = inputs as usize;
        }
        control.linked = 0;
        control.followed = 0;
        let mut result = self.code.run(inputs, control, exit_area);
        if self.attached.is_empty() {
            self.leave_writebacks(inputs, exit_area, false);
        } else {
            result = self.run_sides(inputs, shared, control, exit_area, result);
        }
        control.cells = 0;
        control.tree = 0;
        if self.code.native_link {
            control.inputs = 0;
        }
        if result == super::emit::RETURNED_CODE {
            return TraceOutcome::Returned;
        }
        match usize::try_from(result) {
            Ok(code) => {
                // A side compiled apart names its exits with its unit.
                let index = if code >> emit::SIDE_CODE_SHIFT == 0 {
                    code
                } else {
                    self.side_exit(code)
                };
                debug_assert!(index < self.exits.len());
                TraceOutcome::Exit { index }
            }
            // A loop trace checks every input's tag in one branch and names
            // no input: the first that differs is the one.
            Err(_) if self.stride.is_none() => TraceOutcome::EntryMismatch {
                input: entry_inputs
                    .iter()
                    .zip(&self.entry)
                    .position(|(slot, (_, ty))| {
                        *ty != TraceType::Any && Some(slot.tag) != super::exit::entry_tag(*ty)
                    })
                    .unwrap_or(0),
            },
            Err(_) => TraceOutcome::EntryMismatch {
                input: usize::try_from(-(result + 1)).expect("mismatch code encodes an index"),
            },
        }
    }
}

/// The Rust half of a native link: follows the sides from the exit `result`
/// names, as [`CompiledTrace::call_sides`] does, for the root's native exit
/// that called a side and got a code back.
#[allow(unsafe_code)]
pub(super) extern "C" fn trace_call_sides(
    tree: *const CompiledTrace,
    inputs: *mut TraceInputSlot,
    control: *mut TraceControl,
    exit_area: *mut i64,
    result: i64,
) -> i64 {
    // SAFETY: native code passes the tree and the buffers of the run that
    // called it, which the run keeps alive and does not touch while native
    // code runs.
    unsafe { (*tree).call_sides(inputs, &mut *control, exit_area, result) }
}

/// For each entry input of the tree's root, whether native code is likely to
/// read the row of the table the input holds, which is so when an operation
/// that reads a row ([`TraceOp::TableFieldLoad`], [`TraceOp::TableFieldStore`],
/// [`TraceOp::MetaGuard`], [`TraceOp::LaneArrayLoad`] and
/// [`TraceOp::LaneArrayStore`]) takes the input's value as its table, directly,
/// through copies, or through a loop parameter that starts with it or is
/// handed it by a loop-back. A row is read nowhere else, so the run need not
/// fill the row of a table that is only bound (the array a loop indexes, the
/// library table it calls into).
///
/// The answer is a guess that costs time and never correctness: a run zeroes
/// the row of an input it skips, and a zero row makes every access leave
/// through its guard. A table a skipped input holds still gets its row when
/// native code asks for its lane later (an array read of the same table, say).
/// The activations of a function trace and the entry inputs of side traces
/// answer for every input.
fn entry_row_needs<M: Borrow<TraceIr>>(lowered: &[M]) -> Box<[bool]> {
    let root = lowered[0].borrow();
    let count = root.entry.inputs.len();
    let all = || -> Box<[bool]> { vec![true; count].into() };
    if root.function.is_some() {
        return all();
    }
    let mut needed = vec![false; count];
    for (number, member) in lowered.iter().enumerate() {
        let member = member.borrow();
        let mut seen_params = vec![false; member.loop_params.len()];
        let mut pending: Vec<ValueId> = Vec::new();
        for instruction in member.preheader.iter().chain(&member.body) {
            if let TraceOp::TableFieldLoad { table, .. }
            | TraceOp::TableFieldStore { table, .. }
            | TraceOp::MetaGuard { table, .. }
            | TraceOp::LaneArrayLoad { table, .. }
            | TraceOp::LaneArrayStore { table, .. } = &instruction.op
            {
                pending.push(*table);
            }
        }
        while let Some(value) = pending.pop() {
            // Follow the copies to where the lane comes from.
            let mut base = value;
            loop {
                let Some(info) = member.values.get(base.index()) else {
                    return all();
                };
                let origin = match info.def {
                    // A register of a seeded frame of a side: no root input.
                    ValueDef::FrameInput(_) => return all(),
                    ValueDef::Input(input) => {
                        if number != 0 || input >= count {
                            return all();
                        }
                        needed[input] = true;
                        break;
                    }
                    ValueDef::LoopParam(param) => {
                        if number != 0 || param >= seen_params.len() {
                            return all();
                        }
                        if !std::mem::replace(&mut seen_params[param], true) {
                            if let LoopParamInit::Input(start) = member.loop_params[param].init {
                                pending.push(start);
                            }
                            if let TraceTerminator::LoopBack { args } = &member.terminator
                                && let Some(next) = args.get(param)
                            {
                                pending.push(*next);
                            }
                        }
                        break;
                    }
                    ValueDef::Instruction(at) => member.body.get(at),
                    ValueDef::Preheader(at) => member.preheader.get(at),
                };
                match origin.map(|instruction| &instruction.op) {
                    Some(TraceOp::Move { source }) => base = *source,
                    // An array read, say: the lane is made when it is read.
                    _ => break,
                }
            }
        }
    }
    needed.into()
}

/// What a compilation may add to the code it generates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceCompileOptions {
    /// Count every native jump into a side trace in
    /// [`TraceControl::links`]. Costs a few instructions per link, so the VM
    /// asks for it only when it reports statistics.
    pub count_links: bool,
    /// The optimizations the trace optimizer may run, from the VM's
    /// [`oximond::compat::TracingOptLevel`].
    pub opt_flags: TracingOptFlags,
}

impl CompiledTrace {
    /// Call instructions in the lowered function.
    #[cfg(test)]
    pub(super) fn call_count(&self) -> usize {
        self.code.calls
    }

    /// `fma` instructions in the lowered function.
    #[cfg(test)]
    pub(super) fn fma_count(&self) -> usize {
        self.code.fmas
    }

    /// Whether the MIR backend made the code.
    #[must_use]
    pub fn made_by_mir(&self) -> bool {
        self.code.backend() == "mir"
    }

    /// Whether the MIR backend made the code, and where it is.
    #[cfg(test)]
    pub(super) fn mir_address(&self) -> Option<usize> {
        self.code.mir_address()
    }
}

/// Lowers a verified trace at `tier`.
///
/// # Errors
///
/// Returns [`TraceCompileError`] when the IR is invalid, uses something the
/// backend does not lower, or Cranelift rejects it.
pub fn compile_trace(
    ir: &TraceIr,
    tier: OptimizationTier,
) -> Result<CompiledTrace, TraceCompileError> {
    compile(ir, tier, None, TraceCompileOptions::default())
}

/// [`compile_trace`] with `options`.
///
/// # Errors
///
/// As [`compile_trace`].
pub fn compile_trace_with(
    ir: &TraceIr,
    tier: OptimizationTier,
    options: TraceCompileOptions,
) -> Result<CompiledTrace, TraceCompileError> {
    compile(ir, tier, None, options)
}

/// [`compile_trace_with`] for a tree `planned` was planned from (`plan_tree`
/// of `ir`), as the planner's caller has it. `None` plans it here.
///
/// # Errors
///
/// As [`compile_trace`].
pub fn compile_trace_with_plan(
    ir: &TraceIr,
    tier: OptimizationTier,
    options: TraceCompileOptions,
    planned: Option<&[TreeMember]>,
) -> Result<CompiledTrace, TraceCompileError> {
    compile_with_plan(ir, tier, None, options, planned)
}

/// Like [`compile_trace`], but the guard branch with index `guard` (in the
/// optimized trace's numbering, see [`CompiledTrace::lowered_ir`]) always
/// takes its exit, whatever its condition says. Tests use it to take the
/// native exit of every guard, including guards no input reaches.
///
/// # Errors
///
/// As [`compile_trace`].
#[doc(hidden)]
pub fn compile_trace_forcing_guard(
    ir: &TraceIr,
    tier: OptimizationTier,
    guard: usize,
) -> Result<CompiledTrace, TraceCompileError> {
    compile(ir, tier, Some((0, guard)), TraceCompileOptions::default())
}

/// Like [`compile_trace_forcing_guard`] for a guard of any member of a tree:
/// `member` is the index in [`CompiledTrace::lowered_member`] numbering and
/// `guard` the guard's index in that member.
///
/// # Errors
///
/// As [`compile_trace`].
#[doc(hidden)]
pub fn compile_trace_forcing_member_guard(
    ir: &TraceIr,
    tier: OptimizationTier,
    member: usize,
    guard: usize,
) -> Result<CompiledTrace, TraceCompileError> {
    compile(
        ir,
        tier,
        Some((member, guard)),
        TraceCompileOptions::default(),
    )
}

fn compile(
    ir: &TraceIr,
    tier: OptimizationTier,
    forced_guard: Option<(usize, usize)>,
    options: TraceCompileOptions,
) -> Result<CompiledTrace, TraceCompileError> {
    compile_with_plan(ir, tier, forced_guard, options, None)
}

/// [`compile`] with the members `planned` already holds, which the caller
/// planned from `ir` (`plan_tree`) and verified as it grew. Release builds
/// then skip the verify and the plan.
fn compile_with_plan(
    ir: &TraceIr,
    tier: OptimizationTier,
    forced_guard: Option<(usize, usize)>,
    options: TraceCompileOptions,
    planned: Option<&[TreeMember]>,
) -> Result<CompiledTrace, TraceCompileError> {
    let optimization = alloc_phase::enter(Phase::Optimization);
    // A tree that comes with its plan was verified as it grew: the root when
    // it was recorded and installed, each side when it was recorded, and the
    // tree-wide checks and the fit of every side in `verify_grown_tree`. Debug
    // builds verify it whole anyway, so the tests keep the full check.
    if planned.is_none() || cfg!(debug_assertions) {
        ir.verify().map_err(TraceCompileError::Invalid)?;
    }
    let level = match tier {
        OptimizationTier::Baseline => OptimizeLevel::Baseline,
        OptimizationTier::Optimized | OptimizationTier::Peak => OptimizeLevel::Optimized,
    };
    let owned;
    let planned: &[TreeMember] = if let Some(planned) = planned {
        planned
    } else {
        owned = plan_tree(ir).map_err(TraceCompileError::Tree)?;
        // A tree without sides plans as its root alone, unchanged, and the
        // root was verified just above.
        if !ir.sides.is_empty() {
            verify_fresh_plan(&owned)?;
        }
        &owned
    };
    // Each trace of the tree is optimized on its own, but all of them see
    // what the whole tree writes: a side's store reaches the root's loop
    // through a link, so the root may hoist a read only if no member writes
    // it. A side trace has no loop, so nothing moves into a preheader there.
    // A guard a side trace continues stays in the body of its trace.
    let writes = TreeWrites::of_members(planned.iter().map(|member| &member.ir))
        .with_continued_guards(planned.iter().filter_map(|member| member.parent));
    let demoted_bindings = planned.first().map_or(0, |root| root.demoted);
    let mut optimize_stats = OptimizeStats::default();
    let optimized: Vec<Arc<TraceIr>> = crate::jit::icount::around(
        "optimize",
        planned.len(),
        || {
            super::member_cache::optimize_tree(
                ir.id,
                planned,
                level,
                options.opt_flags,
                &writes,
                &mut optimize_stats,
            )
        },
        |_| "all",
    );
    if crate::jit::debug_env::trace_ir_opt() {
        eprintln!(
            "jit-trace: optimized tree of trace {:?}, {} member(s)",
            ir.id,
            optimized.len()
        );
        for (index, member) in optimized.iter().enumerate() {
            eprintln!("jit-trace: optimized member {index} {member:#?}");
        }
        eprintln!("jit-trace: optimizer {optimize_stats:?}");
    }
    for member in &optimized {
        check_supported(member)?;
    }
    // Only a parent can hand a side the address of a slot it borrows from.
    if optimized[0]
        .entry
        .inputs
        .iter()
        .any(|input| input.ty.is_borrowed())
    {
        return Err(TraceCompileError::NativeBackendRejected);
    }
    // A side links into the guard it continues, found by origin: the
    // optimizer renumbers guards, may remove one that cannot fail, and may
    // hoist one to the preheader, where it fails before the loop starts. A
    // side whose guard is gone, or whose parent was left out, is dropped.
    let mut included: Vec<bool> = vec![true; optimized.len()];
    let mut attach: Vec<Vec<(usize, usize)>> = vec![Vec::new(); optimized.len()];
    let mut terminal: Vec<Option<usize>> = vec![None; optimized.len()];
    for (index, member) in planned.iter().enumerate().skip(1) {
        let Some((parent, origin)) = member.parent else {
            continue;
        };
        // A loop side continues its parent's terminal exit, which is never
        // removed or moved by the optimizer.
        if origin == TERMINAL_GUARD {
            if included[parent]
                && matches!(optimized[parent].terminator, TraceTerminator::Exit { .. })
            {
                terminal[parent] = Some(index);
            } else {
                included[index] = false;
            }
            continue;
        }
        let found = included[parent]
            .then(|| continue_guard(&optimized[parent], origin, member.ir.seed_frames))
            .flatten();
        match found {
            Some(guard) => attach[parent].push((guard, index)),
            None => included[index] = false,
        }
    }
    // Member numbers in the compiled code skip the dropped sides.
    let mut renumber: Vec<Option<usize>> = Vec::with_capacity(optimized.len());
    let mut kept = 0;
    for &keep in &included {
        renumber.push(keep.then(|| {
            kept += 1;
            kept - 1
        }));
    }
    let lowered: Vec<Arc<TraceIr>> = optimized
        .into_iter()
        .zip(&included)
        .filter_map(|(member, keep)| keep.then_some(member))
        .collect();
    let members: Vec<LoweredMember<'_>> = lowered
        .iter()
        .enumerate()
        .map(|(position, member)| {
            let original = renumber
                .iter()
                .position(|number| *number == Some(position))
                .expect("every kept member has a source");
            LoweredMember {
                ir: member,
                attached: attach[original]
                    .iter()
                    .filter_map(|(guard, side)| renumber[*side].map(|side| (*guard, side)))
                    .collect(),
                terminal: terminal[original].and_then(|side| renumber[side]),
                link_loop: match member.terminator {
                    TraceTerminator::LinkLoop { side, .. } => {
                        renumber.get(side + 1).copied().flatten()
                    }
                    _ => None,
                },
            }
        })
        .collect();
    optimization.end();
    let _codegen = alloc_phase::enter(Phase::Codegen);
    let refs: Vec<&TraceIr> = lowered.iter().map(|member| &**member).collect();
    // The exit plans of the members the member cache kept, as the last
    // compile of the tree made them.
    let originals: Vec<usize> = (0..lowered.len())
        .map(|position| {
            renumber
                .iter()
                .position(|number| *number == Some(position))
                .expect("every kept member has a source")
        })
        .collect();
    let context = PlanContext::of(&lowered[0]);
    let held = held_registers(&refs);
    let held_by_original: Vec<&[u16]> = {
        let mut by_original: Vec<&[u16]> = vec![&[]; planned.len()];
        for (original, held) in originals.iter().zip(&held) {
            by_original[*original] = held;
        }
        by_original
    };
    let kept = super::member_cache::plans(ir.id, planned.len(), context, &held_by_original);
    let mut plans: Vec<Option<Arc<MemberPlan>>> = originals
        .iter()
        .map(|original| kept[*original].clone())
        .collect();
    let plan = crate::jit::icount::around(
        "plan",
        refs.len(),
        || plan_exits_with(&refs, 0, &held, &mut plans),
        |_| "all",
    )?;
    super::member_cache::keep_plans(
        ir.id,
        context,
        originals
            .iter()
            .zip(held)
            .zip(plans)
            .filter_map(|((original, held), plan)| Some((*original, held, plan?))),
    );
    let counted = lowered[0].counted;
    // A tree counts as a whole: one budget runs through the root and sides.
    if lowered.iter().any(|member| member.counted != counted)
        || (counted && (lowered[0].function.is_some() || lowered[0].loop_tail.is_some()))
    {
        return Err(TraceCompileError::NativeBackendRejected);
    }
    let max_counted = if counted {
        plan.descriptors
            .iter()
            .map(|descriptor| descriptor.counted)
            .max()
            .unwrap_or(0)
    } else {
        0
    };
    // Every exit of a loop trace can call a side compiled apart later, which
    // costs the exit a load and a branch. The exit reads its cell and calls
    // the side in `CompiledTrace::call_sides`, which the run loop
    // (`CompiledTrace::run_sides`) drives after the code returns the exit's
    // code; the code itself holds no call.
    let cells: Box<[AtomicUsize]> = if lowered[0].function.is_none() && !counted {
        (0..plan.descriptors.len())
            .map(|_| AtomicUsize::new(0))
            .collect()
    } else {
        Box::new([])
    };
    let trusted_key = emit::has_trusted_key(&members);
    let bridges = tree_bridges_virtuals(&members);
    let code = crate::jit::icount::around(
        "tree",
        members
            .iter()
            .map(|member| member.ir.preheader.len() + member.ir.body.len())
            .sum(),
        || emit::lower(&members, &plan, tier, forced_guard, options, None),
        |code| code.as_ref().map_or("rejected", emit::TraceCode::backend),
    )?;
    let exit_area_len = plan.area_len;
    let stride = plan.stride;
    #[cfg(feature = "op-stats")]
    count_static_opportunities(&members, plan.descriptors.len(), exit_area_len);
    let root = &lowered[0];
    let entry: Vec<(u16, TraceType)> = root
        .entry
        .inputs
        .iter()
        .map(|input| (input.register, input.ty))
        .collect();
    // The caller's sides, in the order it gave them.
    let side_members = renumber[1..].to_vec();
    drop(members);
    let calls_host = refs.iter().any(|member| member.calls_host());
    let resumes = refs.iter().any(|member| member.resumes());
    let makes_tables = refs.iter().any(|member| member.makes_tables()) || bridges;
    // Matches the emitter: a tree that looks up or stores through hashes and moves none probes
    // the string indexes of its tables in native code.
    let probes_hash = refs.iter().any(|member| member.uses_string_index())
        && !refs.iter().any(|member| member.moves_hash_storage());
    let stores_references = refs.iter().any(|member| member_stores_references(member));
    let closure_row_words = emit::closure_row_words(&refs);
    let row_masked = emit::rows_masked(&refs);
    let array_flags: Box<[bool]> = lowered[0]
        .bindings
        .iter()
        .map(TableBinding::is_array)
        .collect();
    let entry_rows = entry_row_needs(&lowered);
    let table_inputs: Box<[u16]> = entry
        .iter()
        .enumerate()
        .filter(|(_, (_, ty))| *ty == TraceType::Table)
        .filter_map(|(input, _)| u16::try_from(input).ok())
        .collect();
    let needs_tables = refs.iter().any(|member| member_needs_tables(member)) || bridges;
    let mut facts = TreeFacts::of(&lowered, closure_row_words, row_masked, trusted_key);
    facts.needs_tables |= bridges;
    facts.makes_tables |= bridges;
    let link_slots = link_slots_of(&lowered[0]);
    // The slots of registers the root takes untyped that links rewrite: the
    // run leaves them for the VM whether or not a link ran (see
    // `CompiledTrace::leave_writebacks`).
    let writebacks: Vec<Writeback> = untyped_link_writes(&lowered)
        .into_iter()
        .map(|(position, register)| Writeback {
            source: ExtraSource::Register(register),
            position,
        })
        .collect();
    let untyped_writebacks = writebacks.len();
    let own_exits = plan.descriptors.len();
    let clif_insts = code.clif_insts();
    Ok(CompiledTrace {
        code,
        tier,
        entry,
        exits: plan.descriptors,
        exit_area_len,
        lowered,
        side_members,
        needs_tables,
        row_words: closure_row_words,
        makes_tables,
        probes_hash,
        calls_host,
        resumes,
        stores_references,
        array_flags,
        entry_rows,
        table_inputs,
        optimize_stats,
        stride,
        counted,
        max_counted,
        cells,
        own_exits,
        facts,
        clif_insts,
        demoted_bindings,
        attached: Vec::new(),
        undo: None,
        extras: Vec::new(),
        writebacks,
        untyped_writebacks,
        link_slots,
    })
}

/// Where each loop parameter of `root` starts, when a side compiled apart can
/// hand the run new values for them (see [`CompiledTrace::link_slots`]).
pub(super) fn link_slots_of(root: &TraceIr) -> Option<Box<[(usize, TraceType)]>> {
    if root.function.is_some() || root.loop_params.is_empty() {
        return None;
    }
    let mut slots: Vec<(usize, TraceType)> = Vec::with_capacity(root.loop_params.len());
    for param in &root.loop_params {
        let LoopParamInit::Input(start) = param.init else {
            return None;
        };
        let position = root
            .entry
            .inputs
            .iter()
            .position(|input| input.value == start)?;
        let entry = &root.entry.inputs[position];
        if entry.ty != param.ty
            || param.register != Some(entry.register)
            || !matches!(
                param.ty,
                TraceType::Float
                    | TraceType::Integer
                    | TraceType::Boolean
                    | TraceType::Nil
                    | TraceType::Table
                    | TraceType::Closure
            )
            || slots.iter().any(|(held, _)| *held == position)
        {
            return None;
        }
        slots.push((position, param.ty));
    }
    Some(slots.into())
}

/// Compiles side number `side` of `tree` (an index into [`TraceIr::sides`])
/// as a function of its own, to be attached to the exit `context` names.
///
/// `apart` lists the sides of the tree that are compiled apart, and `detached`
/// those of them that leave at the loop header where they would link back:
/// they are planned as the tree was planned when its code was made (see
/// [`plan_tree_with`]). `exit_style` makes the side leave through its poll
/// exit where it would link back to the loop, so that the tree's code needs
/// nothing from it. Only the side is optimized and lowered: the cost of the
/// compile follows its size, not the tree's.
///
/// # Errors
///
/// As [`compile_trace`]. The VM then compiles the side with the tree.
pub fn compile_side(
    tree: &TraceIr,
    detached: &[usize],
    apart: &[usize],
    side: usize,
    exit_style: bool,
    context: &SideContext,
    tier: OptimizationTier,
    options: TraceCompileOptions,
) -> Result<CompiledSide, TraceCompileError> {
    compile_side_with_plan(
        tree, detached, apart, side, exit_style, context, tier, options, None,
    )
}

/// Verifies the members of a plan made on the compile thread.
///
/// `optimize_tree_member` checks its input only in debug builds. A plan the
/// VM made as the tree grew comes from IR that `verify_grown_tree` checked,
/// and a member-cache hit repeats a member that was optimized before. A plan
/// made here has passed through no such check, so a planner bug (root
/// widening, a detached or apart side) would reach the optimizer's `expect`s
/// and panic the compile instead of rejecting it.
fn verify_fresh_plan(members: &[TreeMember]) -> Result<(), TraceCompileError> {
    for member in members {
        member.ir.verify().map_err(TraceCompileError::Invalid)?;
    }
    Ok(())
}

/// [`compile_side`] with the members `planned` holds: the tree planned with
/// the same `detached` and `apart`. `None` plans it here.
///
/// # Errors
///
/// As [`compile_side`].
#[allow(clippy::too_many_arguments)]
pub fn compile_side_with_plan(
    tree: &TraceIr,
    detached: &[usize],
    apart: &[usize],
    side: usize,
    exit_style: bool,
    context: &SideContext,
    tier: OptimizationTier,
    options: TraceCompileOptions,
    planned: Option<&[TreeMember]>,
) -> Result<CompiledSide, TraceCompileError> {
    let optimization = alloc_phase::enter(Phase::Optimization);
    let level = match tier {
        OptimizationTier::Baseline => OptimizeLevel::Baseline,
        OptimizationTier::Optimized | OptimizationTier::Peak => OptimizeLevel::Optimized,
    };
    let owned;
    let planned: &[TreeMember] = if let Some(planned) = planned {
        planned
    } else {
        owned = plan_tree_with(tree, detached, apart).map_err(TraceCompileError::Tree)?;
        if let Some(member) = owned.get(side + 1) {
            member.ir.verify().map_err(TraceCompileError::Invalid)?;
        }
        &owned
    };
    let member = planned
        .get(side + 1)
        .ok_or(TraceCompileError::NativeBackendRejected)?;
    // The side sees what the whole tree writes, as its members do when they
    // are compiled together.
    let writes = TreeWrites::of_members(planned.iter().map(|member| &member.ir))
        .with_continued_guards(planned.iter().filter_map(|member| member.parent));
    let (child, optimize_stats) = optimize_tree_member(
        &member.ir,
        level,
        options.opt_flags,
        &writes.for_member(side + 1),
    );
    check_supported(&child)?;
    if child.function.is_some() || child.counted || child.loop_tail.is_some() {
        return Err(TraceCompileError::NativeBackendRejected);
    }
    optimization.end();
    let _codegen = alloc_phase::enter(Phase::Codegen);
    let root: &TraceIr = &context.root;
    let plan = plan_exits(&[root, &child], 1)?;
    let members = [
        LoweredMember {
            ir: root,
            attached: Vec::new(),
            terminal: None,
            link_loop: None,
        },
        LoweredMember {
            ir: &child,
            attached: Vec::new(),
            terminal: None,
            link_loop: None,
        },
    ];
    let cells: Box<[AtomicUsize]> = (0..plan.descriptors.len())
        .map(|_| AtomicUsize::new(0))
        .collect();
    // What the side writes where it links back, beyond the root's loop
    // parameters: the run writes it into the entry inputs and the extras.
    let link_writes = if exit_style {
        Vec::new()
    } else {
        apart_link_writes(tree, side).map_err(TraceCompileError::Tree)?
    };
    let spec = emit::DetachedSpec {
        link_writes: &link_writes,
        code_base: i64::from(context.unit) << emit::SIDE_CODE_SHIFT,
        host: &context.host,
        host_snapshot: &context.host_snapshot,
        root_entry: &context.root_entry,
        exit_style,
        native_probe: context.tree_facts.probes_hash,
        row_masked: context.tree_facts.row_masked,
        extras: &context.extras,
        guest_registers: usize::from(root.frames[0].register_count)
            .saturating_sub(root.external.len()),
    };
    let code = crate::jit::icount::around(
        "side",
        members[1].ir.preheader.len() + members[1].ir.body.len(),
        || emit::lower(&members, &plan, tier, None, options, Some(&spec)),
        |code| code.as_ref().map_or("rejected", emit::TraceCode::backend),
    )?;
    let extras = code.side_extras().to_vec();
    let writebacks = code.side_writebacks().to_vec();
    drop(members);
    let row_words = emit::closure_row_words(&[&child]);
    let facts = TreeFacts::of(
        std::slice::from_ref(&child),
        row_words,
        context.tree_facts.row_masked,
        false,
    );
    let rows = entry_row_needs(std::slice::from_ref(&child));
    let row_registers: Vec<u16> = child
        .entry
        .inputs
        .iter()
        .zip(rows.iter())
        .filter(|(_, needed)| **needed)
        .map(|(input, _)| input.register)
        .collect();
    Ok(CompiledSide {
        unit: context.unit,
        code,
        exit_area_len: plan.area_len,
        exits: plan.descriptors,
        lowered: child,
        host: context.host_index,
        exit_style,
        cells,
        facts,
        optimize_stats,
        row_registers,
        extras,
        writebacks,
    })
}

/// Whether some guard of the tree enters a side from a snapshot that holds a
/// virtual table: the run builds the table on the way
/// (`TraceTables::materialize_exit`), so the tree makes tables in its run
/// like one with a `TABLE_NEW` call, and needs the run's tables.
pub(super) fn tree_bridges_virtuals(members: &[LoweredMember<'_>]) -> bool {
    members.iter().any(|member| {
        member.attached.iter().any(|(guard, _)| {
            member
                .ir
                .guards
                .get(*guard)
                .and_then(|info| member.ir.snapshots.get(info.snapshot.index()))
                .is_some_and(crate::jit_ir::trace::Snapshot::has_virtual_entry)
        })
    })
}

/// Whether the member has an operation that reads or writes a table by lane,
/// or calls a helper that takes the VM's context.
fn member_needs_tables(member: &TraceIr) -> bool {
    member
        .preheader
        .iter()
        .chain(&member.body)
        .any(|instruction| {
            matches!(
                instruction.op,
                TraceOp::TableFieldLoad { .. }
                    | TraceOp::TableFieldStore { .. }
                    | TraceOp::MetaGuard { .. }
                    | TraceOp::LaneArrayLoad { .. }
                    | TraceOp::LaneArrayStore { .. }
                    | TraceOp::ArraySet { .. }
                    | TraceOp::HashLoad { .. }
                    | TraceOp::HostCall { .. }
                    | TraceOp::ReleaseStrings { .. }
            ) || matches!(
                &instruction.op,
                // A helper that takes the VM's context needs its run.
                TraceOp::CallHelper { helper, .. }
                    if helper.info().is_none_or(|info| info.context)
            ) || matches!(
                instruction.op,
                TraceOp::ArrayLoad {
                    expected: TraceType::Table | TraceType::Closure | TraceType::Thread,
                    ..
                }
            )
        })
}

/// Adds the static opportunity counts of a compiled tree to the diagnostic
/// counters.
#[cfg(feature = "op-stats")]
fn count_static_opportunities(members: &[LoweredMember<'_>], exits: usize, exit_area_len: usize) {
    use crate::jit_ir::trace::opportunity::member_opportunities;
    use crate::runtime::op_stats::{Counter, add_shared, max_shared};
    let resizes = members.iter().any(|member| member.ir.sets_tables());
    for member in members {
        let sides: Vec<usize> = member.attached.iter().map(|(guard, _)| *guard).collect();
        let counts = member_opportunities(member.ir, &sides, resizes);
        for (counter, value) in [
            (Counter::IrArraySites, counts.array_sites),
            (Counter::IrArraySitesBytes, counts.byte_sites),
            (Counter::IrArraySitesConstKey, counts.const_key_sites),
            (Counter::IrArraySitesFloatKey, counts.float_key_sites),
            (Counter::IrArraySitesInduction, counts.induction_sites),
            (Counter::IrArraySitesRangeOnce, counts.range_once_sites),
            (
                Counter::IrArraySitesFloatInvariantInduction,
                counts.float_invariant_induction_sites,
            ),
            (
                Counter::IrArraySitesFloatInvariantRangeOnce,
                counts.float_invariant_range_once_sites,
            ),
            (Counter::IrTypeGuards, counts.type_guards),
            (Counter::IrSnapshots, counts.snapshots),
            (Counter::IrSnapshotFrames, counts.snapshot_frames),
            (Counter::IrSnapshotEntries, counts.entries),
            (Counter::IrSnapshotValueEntries, counts.value_entries),
            (Counter::IrSnapshotConstEntries, counts.const_entries),
            (
                Counter::IrSnapshotUnchangedEntries,
                counts.unchanged_entries,
            ),
            (Counter::IrSnapshotRootEntries, counts.root_entries),
            (Counter::IrSnapshotVirtualEntries, counts.virtual_entries),
            (Counter::IrSnapshotOtherEntries, counts.other_entries),
            (Counter::IrSnapshotVirtuals, counts.snapshot_virtuals),
            (
                Counter::IrSnapshotRecipeEntries,
                counts.recipe_value_entries
                    + counts.recipe_const_entries
                    + counts.recipe_root_entries,
            ),
            (
                Counter::IrSnapshotRecipeValueEntries,
                counts.recipe_value_entries,
            ),
            (
                Counter::IrSnapshotRecipeConstEntries,
                counts.recipe_const_entries,
            ),
            (
                Counter::IrSnapshotRecipeRootEntries,
                counts.recipe_root_entries,
            ),
            (
                Counter::IrSnapshotRecipeValuesFeeding,
                counts.recipe_values_feeding,
            ),
            (
                Counter::IrSnapshotRecipeValuesRecovery,
                counts.recipe_values_recovery,
            ),
            (
                Counter::IrSnapshotRecipesUnknown,
                counts.snapshot_recipes_unknown,
            ),
            (Counter::IrSnapshotValues, counts.snapshot_values),
            (Counter::IrSnapshotOnlyValues, counts.snapshot_only_values),
            (
                Counter::IrSnapshotOnlyInstructions,
                counts.snapshot_only_instructions,
            ),
            (Counter::IrSnapshotOnlyCheap, counts.snapshot_only_cheap),
            (Counter::IrGuards, counts.guards),
            (Counter::IrPinnedClosures, counts.pinned_closures),
        ] {
            add_shared(counter, value);
        }
    }
    add_shared(Counter::IrMembers, members.len() as u64);
    add_shared(Counter::IrTreesCompiled, 1);
    add_shared(Counter::IrExitDescriptors, exits as u64);
    add_shared(Counter::IrExitAreaSlots, exit_area_len as u64);
    max_shared(Counter::IrExitAreaSlotsMax, exit_area_len as u64);
}

/// The index of the body guard of `ir` that came from source guard `origin`,
/// if it is still a guard that can fail inside the loop and resumes in the
/// root frame, or inside the `seeded` calls a side starts in (see
/// [`TraceIr::seed_frames`]).
fn continue_guard(ir: &TraceIr, origin: u32, seeded: u8) -> Option<usize> {
    let preheader_guards = ir
        .preheader
        .iter()
        .filter(|instruction| instruction.op.guard().is_some())
        .count();
    let (index, info) = ir
        .guards
        .iter()
        .enumerate()
        .skip(preheader_guards)
        .find(|(_, info)| info.origin == origin)?;
    let snapshot = &ir.snapshots[info.snapshot.index()];
    if snapshot.frames.len() != 1 + usize::from(seeded) {
        return None;
    }
    // A branch guard, an array read, a hash read and a search helper continue natively: a type guard
    // never fails, and the other heap accesses take no sides.
    let continues = ir.body.iter().any(|instruction| match instruction.op {
        TraceOp::GuardBranch { guard, .. }
        | TraceOp::ArrayLoad { guard, .. }
        | TraceOp::LaneArrayLoad { guard, .. }
        | TraceOp::NumberLoad { guard, .. }
        | TraceOp::ArrayNext { guard, .. }
        | TraceOp::HashLoad { guard, .. } => guard.index() == index,
        // A field that is absent where the recording found it present, or
        // the other way round, is a branch. A read that leaves for want of
        // a lane is not (the VM tells the two apart when it counts).
        // An object of another class is a branch too.
        TraceOp::TableFieldLoad { guard, .. } | TraceOp::MetaGuard { guard, .. } => {
            guard.index() == index
        }
        // A `find` or `match` that goes another way than the recording, or
        // a store that finds its key deleted or live where it was not.
        TraceOp::CallHelper { helper, guard, .. } => {
            guard.index() == index && helper.miss_takes_side()
        }
        // A register the root reads as nil or a number: the side runs the
        // iteration with the other type.
        TraceOp::GuardType { guard, .. } => {
            guard.index() == index && ir.guards_untyped_input(origin)
        }
        _ => false,
    });
    continues.then_some(index)
}

/// Rejects what the emitter does not lower: heap operations and `Any` values.
fn check_supported(ir: &TraceIr) -> Result<(), TraceCompileError> {
    for (index, instruction) in ir.preheader.iter().chain(&ir.body).enumerate() {
        match &instruction.op {
            TraceOp::Read { .. } => {
                return Err(TraceCompileError::Unsupported {
                    instruction: index,
                    what: "heap read",
                });
            }
            TraceOp::Effect { .. } => {
                return Err(TraceCompileError::Unsupported {
                    instruction: index,
                    what: "effect",
                });
            }
            // An entry input of either type is guarded where its slot is
            // (see `Lowerer::lower_entry`). Any other dynamic value is not.
            TraceOp::GuardType { source, .. }
                if ir.value_type(*source) == Some(TraceType::Any)
                    && !matches!(
                        ir.values.get(source.index()).map(|info| info.def),
                        Some(ValueDef::Input(_))
                    ) =>
            {
                return Err(TraceCompileError::Unsupported {
                    instruction: index,
                    what: "guard on a dynamically typed value",
                });
            }
            _ => {}
        }
    }
    if matches!(ir.terminator, TraceTerminator::LoopBack { .. }) && ir.poll_snapshot.is_none() {
        return Err(TraceCompileError::MissingPollSnapshot);
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn plan_exits(members: &[&TraceIr], first_member: usize) -> Result<ExitPlan, TraceCompileError> {
    let mut plans = vec![None; members.len()];
    plan_exits_with(members, first_member, &[], &mut plans)
}

/// What the exit plan of a member depends on besides the member itself and
/// its position in the tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlanContext {
    /// The loop span: the header to the closing instruction the root's poll
    /// snapshot names. A guard that resumes outside it ends the loop, in the
    /// root and in every side.
    span: Option<(usize, usize)>,
    /// Whether the exits share their stores ([`share_exit_stores`]): all but
    /// a function trace's, whose area repeats per activation and keeps each
    /// exit's slots packed.
    shares: bool,
}

impl PlanContext {
    pub(super) fn of(root: &TraceIr) -> Self {
        Self {
            span: root
                .poll_snapshot
                .and_then(|poll| match root.snapshots[poll.index()].phase {
                    SnapshotPhase::After { pc, successor } => Some((successor, pc)),
                    SnapshotPhase::Before { .. } => None,
                }),
            shares: root.function.is_none(),
        }
    }
}

/// For each member of the tree `members`, the registers of the root frame
/// where it takes an entry input that keeps, at every exit of the tree, the
/// value the frame held when the run began: sorted. An exit does not store a
/// value it names only in such registers, as the member's own input there
/// (see [`kept_by_frame`]); the restore leaves those registers alone.
///
/// Native code writes no register of the frame, so the frame keeps its entry
/// state until the exit's restore writes it, unless the tree is a function
/// trace, loops through tail calls, boxes its control or a value, or runs a
/// coroutine, the cases where the restore does not trust the frame either. A
/// guest register that no member writes, and that every snapshot of every
/// member names as it was entered (`Unchanged`, or the member's own input
/// for it), holds its entry value at every point of every member: a side's
/// input for it comes from the snapshot of the guard it continues, which
/// held it the same way. A register a guard commits in its snapshot alone is
/// not held. Registers past the guest window (upvalues, globals, the
/// caller's registers) are never held.
///
/// A member's list names only its own inputs, so a side that writes a new
/// register changes the lists, and the exit plans, of the members that take
/// that register only.
///
/// The bail-outs at the top must match the `skip_unchanged` conditions in
/// `restores_planned_from` (vm/jit/trace/native.rs): an exit leaves out a
/// held value only where the restore keeps the frame's register. Change the
/// two together.
fn held_registers(members: &[&TraceIr]) -> Vec<Box<[u16]>> {
    let root = members[0];
    let none = || members.iter().map(|_| Box::default()).collect();
    if root.function.is_some()
        || root.loop_tail.is_some()
        || root.boxed_control.is_some()
        || root.boxed_value
        || members.iter().any(|member| member.resumes())
    {
        return none();
    }
    let Some(frame) = root.frames.first() else {
        return none();
    };
    let guest = usize::from(frame.register_count).saturating_sub(root.external.len());
    let mut held = vec![true; guest];
    let mut input_of: Vec<Option<ValueId>> = vec![None; guest];
    for member in members {
        for (frame, register) in &member.written_registers {
            if frame.index() == 0
                && let Some(held) = held.get_mut(usize::from(*register))
            {
                *held = false;
            }
        }
        // A guard may commit a write in its snapshot alone (see
        // `guard_writes` in `jit_ir/trace/tree.rs`), and a side that starts
        // there takes that value as its input. So every snapshot must name
        // a held register as it was entered: `Unchanged`, or the member's
        // own input for it, never as a number of run-time subtype.
        input_of.fill(None);
        for input in &member.entry.inputs {
            if let Some(slot) = input_of.get_mut(usize::from(input.register)) {
                *slot = Some(input.value);
            }
        }
        for snapshot in &member.snapshots {
            let Some(frame) = snapshot
                .frames
                .first()
                .filter(|frame| frame.frame.index() == 0)
            else {
                return none();
            };
            for (register, entry) in frame.registers.iter().enumerate().take(guest) {
                let entered = match entry {
                    RegisterEntry::Unchanged => true,
                    RegisterEntry::Value(value) => input_of[register] == Some(*value),
                    _ => false,
                };
                if !entered {
                    held[register] = false;
                }
            }
            for number in &snapshot.numbers {
                if let Some(held) = usize::try_from(number.register)
                    .ok()
                    .and_then(|register| held.get_mut(register))
                {
                    *held = false;
                }
            }
        }
    }
    members
        .iter()
        .map(|member| {
            let mut registers: Vec<u16> = member
                .entry
                .inputs
                .iter()
                .map(|input| input.register)
                .filter(|register| held.get(usize::from(*register)) == Some(&true))
                .collect();
            registers.sort_unstable();
            registers.dedup();
            registers.into_boxed_slice()
        })
        .collect()
}

/// The exits of one member of a tree (see [`ExitPlan`]), every index in it
/// counted from the member's first exit and its first tail. It depends on
/// the member, its position (which only its descriptors name) and the
/// [`PlanContext`], so a tree compiled again keeps it for a member the
/// member cache kept.
#[derive(Debug, PartialEq)]
pub(super) struct MemberPlan {
    descriptors: Vec<ExitDescriptor>,
    poll: Option<usize>,
    terminal: Option<usize>,
    unwinds: Vec<(usize, usize)>,
}

impl MemberPlan {
    /// About the bytes the plan holds.
    pub(super) fn weight(&self) -> usize {
        size_of::<Self>()
            + self
                .descriptors
                .iter()
                .map(|descriptor| {
                    size_of::<ExitDescriptor>() + descriptor.slots.len() * size_of::<ExitSlot>()
                })
                .sum::<usize>()
    }
}

/// Plans the exits of member `member` of a tree in `context`.
fn plan_member(
    ir: &TraceIr,
    member: usize,
    context: PlanContext,
    held: &[u16],
) -> Result<MemberPlan, TraceCompileError> {
    let mut descriptors = Vec::new();
    // A branch guard whose condition compares closures, or asks whether
    // the callee runs the pinned closure's prototype, checks a callee.
    let compares_closures = |condition: ValueId| {
        let op = match ir.values.get(condition.index()).map(|info| info.def) {
            Some(ValueDef::Instruction(index)) => ir.body.get(index),
            Some(ValueDef::Preheader(index)) => ir.preheader.get(index),
            _ => None,
        };
        matches!(
            op.map(|instruction| &instruction.op),
            Some(TraceOp::Compare { left, .. })
                if ir.value_type(*left) == Some(TraceType::Closure)
        ) || matches!(
            op.map(|instruction| &instruction.op),
            Some(TraceOp::CallHelper { helper, .. })
                if *helper == HelperId::CLOSURE_SAME_PROTO
        )
    };
    let guard_kinds: Vec<TraceGuardClass> = ir
        .preheader
        .iter()
        .chain(&ir.body)
        .filter_map(|instruction| match &instruction.op {
            TraceOp::GuardType { .. } => Some(TraceGuardClass::Type),
            TraceOp::GuardBranch { condition, .. } if compares_closures(*condition) => {
                Some(TraceGuardClass::Callee)
            }
            TraceOp::GuardBranch { .. } => Some(TraceGuardClass::Branch),
            TraceOp::ArrayLoad { .. }
            | TraceOp::LaneArrayLoad { .. }
            | TraceOp::NumberLoad { .. }
            | TraceOp::ArrayNext { .. }
            | TraceOp::ArrayStore { .. }
            | TraceOp::ArraySet { .. }
            | TraceOp::HashLoad { .. }
            | TraceOp::StringByte { .. }
            | TraceOp::TableFieldLoad { .. }
            | TraceOp::TableFieldStore { .. }
            | TraceOp::LaneArrayStore { .. } => Some(TraceGuardClass::Array),
            TraceOp::MetaGuard { .. } => Some(TraceGuardClass::Class),
            TraceOp::HostCall { .. } | TraceOp::CallHelper { .. } => Some(TraceGuardClass::Host),
            TraceOp::SelfCall { .. } => Some(TraceGuardClass::Depth),
            _ => None,
        })
        .collect();
    for (guard, info) in ir.guards.iter().enumerate() {
        let snapshot = &ir.snapshots[info.snapshot.index()];
        let class = match snapshot.phase {
            // The loop's span is in the root chunk's pcs. A snapshot
            // inside an inlined call resumes in another chunk.
            SnapshotPhase::After { successor, .. }
                if snapshot.frames.len() == 1
                    && context
                        .span
                        .is_some_and(|(start, end)| !(start..=end).contains(&successor)) =>
            {
                TraceExitClass::LoopComplete
            }
            _ if snapshot.effect_position > 0 => TraceExitClass::CommittedEffect,
            _ => TraceExitClass::GuardFail,
        };
        descriptors.push(describe(
            ir,
            member,
            snapshot,
            class,
            Some((guard_kinds[guard], info.origin)),
            held,
        )?);
    }
    // A self call's activation unwinds through its call's snapshot.
    let mut unwinds = Vec::new();
    for instruction in &ir.body {
        let TraceOp::SelfCall { guard, .. } = instruction.op else {
            continue;
        };
        let snapshot = &ir.snapshots[ir.guards[guard.index()].snapshot.index()];
        descriptors.push(describe(
            ir,
            member,
            snapshot,
            TraceExitClass::Unwind,
            None,
            held,
        )?);
        unwinds.push((guard.index(), descriptors.len() - 1));
    }
    // A loop trace polls at its loop-back, a side trace at its link.
    let poll_snapshot = match &ir.terminator {
        TraceTerminator::Link { snapshot } | TraceTerminator::LinkLoop { snapshot, .. } => {
            Some(*snapshot)
        }
        _ => ir.poll_snapshot,
    };
    let poll = match poll_snapshot {
        Some(id) => {
            descriptors.push(describe(
                ir,
                member,
                &ir.snapshots[id.index()],
                TraceExitClass::Poll,
                None,
                held,
            )?);
            Some(descriptors.len() - 1)
        }
        None => None,
    };
    let terminal = match &ir.terminator {
        TraceTerminator::Exit { snapshot } => {
            descriptors.push(describe(
                ir,
                member,
                &ir.snapshots[snapshot.index()],
                TraceExitClass::LoopComplete,
                None,
                held,
            )?);
            Some(descriptors.len() - 1)
        }
        TraceTerminator::LoopBack { .. }
        | TraceTerminator::Link { .. }
        | TraceTerminator::LinkLoop { .. }
        | TraceTerminator::Return { .. }
        | TraceTerminator::TailCall { .. } => None,
    };
    if context.shares {
        rank_exit_slots(&mut descriptors);
    }
    Ok(MemberPlan {
        descriptors,
        poll,
        terminal,
        unwinds,
    })
}

/// [`plan_exits`] with the member plans in `plans` that an earlier compile
/// made for members equal to these, in the same [`PlanContext`]. Plans the
/// rest and fills them in.
fn plan_exits_with(
    members: &[&TraceIr],
    first_member: usize,
    held: &[Box<[u16]>],
    plans: &mut [Option<Arc<MemberPlan>>],
) -> Result<ExitPlan, TraceCompileError> {
    let root = members[0];
    let context = PlanContext::of(root);
    let held_of = |member: usize| held.get(member).map_or(&[][..], |held| &held[..]);
    let mut descriptors = Vec::new();
    let mut exits = Vec::with_capacity(members.len());
    for (member, ir) in members.iter().enumerate() {
        let first = descriptors.len();
        // A side compiled apart plans its own exits only.
        if member < first_member {
            exits.push(MemberExits {
                first,
                poll: None,
                terminal: None,
                unwinds: Vec::new(),
            });
            continue;
        }
        let plan = if let Some(plan) = &plans[member] {
            // Debug builds, the tests among them, check every reuse
            // against what planning the member makes now.
            debug_assert!(
                plan_member(ir, member, context, held_of(member)).is_ok_and(|mut fresh| {
                    for (descriptor, kept) in fresh.descriptors.iter_mut().zip(&plan.descriptors) {
                        descriptor.member = kept.member;
                    }
                    fresh == **plan
                }),
                "the kept exit plan of member {member} differs from a fresh one"
            );
            Arc::clone(plan)
        } else {
            let plan = Arc::new(plan_member(ir, member, context, held_of(member))?);
            plans[member] = Some(Arc::clone(&plan));
            plan
        };
        descriptors.extend(plan.descriptors.iter().map(|descriptor| ExitDescriptor {
            member,
            ..descriptor.clone()
        }));
        exits.push(MemberExits {
            first,
            poll: plan.poll.map(|poll| poll + first),
            terminal: plan.terminal.map(|terminal| terminal + first),
            unwinds: plan
                .unwinds
                .iter()
                .map(|(guard, exit)| (*guard, exit + first))
                .collect(),
        });
    }
    let area_len = descriptors
        .iter()
        .map(ExitDescriptor::area_len)
        .max()
        .unwrap_or(0);
    // A function's result goes to slot 0 of the entered activation's area.
    let stride = root.function.as_ref().map(|_| area_len.max(1) + 1);
    Ok(ExitPlan {
        descriptors,
        members: exits,
        area_len,
        stride,
        shares: context.shares,
        shared: std::sync::OnceLock::new(),
    })
}

/// Gives the exits of each member of a tree common tails.
///
/// The stubs of a trace's exits store mostly the same values: the registers
/// a guard's snapshot names change little from one guard to the next. Each
/// store is an instruction for Cranelift to lower, and each value a stub
/// names is a use the register allocator places, so the stubs made most of
/// a trace's code and of its compile time.
///
/// Here every value a member's exits name gets one slot of the exit area,
/// the same in all of them, and the stores are shared along a segment tree
/// over the member's exits in order. A value that a run of consecutive exits
/// names is stored by the few tree nodes that cover the run exactly, each a
/// tail block. An exit's stub stores the values only it names, then jumps
/// up the chain of its tails, each of which stores its values and jumps on;
/// the last returns the exit's index.
///
/// A tail stores only values each exit below it names, so it keeps no value
/// live further than a stub would, and each value it stores is defined where
/// every one of those exits branches off, so on every path into the tail.
///
/// `descriptors` are one member's, their slots given by [`rank_exit_slots`]:
/// a stored value's slot is its rank among the values the member's exits
/// store.
fn share_exit_stores(descriptors: &[ExitDescriptor]) -> (Vec<ExitTailPlan>, Vec<ExitStubPlan>) {
    let stored = |slot: &ExitSlot| !matches!(slot.ty, TraceType::Nil | TraceType::Any);
    let mut tails: Vec<ExitTailPlan> = Vec::new();
    let mut stubs: Vec<ExitStubPlan> = (0..descriptors.len())
        .map(|_| ExitStubPlan {
            own: Vec::new(),
            tail: None,
        })
        .collect();
    // The exits of a member share their stores: each is one member's.
    let indices: Vec<usize> = (0..descriptors.len()).collect();
    {
        let named: Vec<Vec<ExitSlot>> = indices
            .iter()
            .map(|index| {
                descriptors[*index]
                    .slots
                    .iter()
                    .filter(|slot| stored(slot))
                    .copied()
                    .collect()
            })
            .collect();
        if named.len() < 2 {
            for (&index, own) in indices.iter().zip(named) {
                stubs[index].own = own;
            }
            return (tails, stubs);
        }
        let values = named
            .iter()
            .flatten()
            .map(|slot| slot.slot as usize + 1)
            .max()
            .unwrap_or(0);
        let mut nodes = cover_runs(&named, values, |slot| slot.slot as usize);
        let levels = nodes.len() - 1;
        push_down_small_nodes(&mut nodes);
        // Tails for the non-empty nodes above the leaves, topmost first, so
        // a node knows its next tail when it is made.
        let mut tail_at: Vec<Vec<Option<usize>>> =
            nodes.iter().map(|level| vec![None; level.len()]).collect();
        for level in (1..=levels).rev() {
            for node in 0..nodes[level].len() {
                if nodes[level][node].is_empty() {
                    continue;
                }
                let next =
                    (level + 1..=levels).find_map(|above| tail_at[above][node >> (above - level)]);
                let mut slots = mem::take(&mut nodes[level][node]);
                slots.sort_unstable_by_key(|slot| slot.value);
                tail_at[level][node] = Some(tails.len());
                tails.push(ExitTailPlan { slots, next });
            }
        }
        for (position, &index) in indices.iter().enumerate() {
            let mut own = mem::take(&mut nodes[0][position]);
            own.sort_unstable_by_key(|slot| slot.value);
            stubs[index] = ExitStubPlan {
                own,
                tail: (1..=levels).find_map(|level| tail_at[level][position >> level]),
            };
        }
    }
    (tails, stubs)
}

/// Gives every value that one member's exits (`descriptors`) store one slot
/// of the exit area, the same in all of them: its rank among those values.
fn rank_exit_slots(descriptors: &mut [ExitDescriptor]) {
    let stored = |slot: &ExitSlot| !matches!(slot.ty, TraceType::Nil | TraceType::Any);
    let Some(last) = descriptors
        .iter()
        .flat_map(|descriptor| &descriptor.slots)
        .filter(|slot| stored(slot))
        .map(|slot| slot.value.index())
        .max()
    else {
        return;
    };
    // By value: whether an exit stores it, then its rank.
    let mut rank = vec![0_u32; last + 1];
    for slot in descriptors.iter().flat_map(|descriptor| &descriptor.slots) {
        if stored(slot) {
            rank[slot.value.index()] = 1;
        }
    }
    let mut next = 0_u32;
    for entry in &mut rank {
        let named = *entry;
        *entry = next;
        next += named;
    }
    for slot in descriptors
        .iter_mut()
        .flat_map(|descriptor| &mut descriptor.slots)
    {
        if stored(slot) {
            slot.slot = rank[slot.value.index()];
        }
    }
}

/// The segment tree over a member's exits for [`share_exit_stores`]: per
/// level, per node, the slots it stores. Node `(level, i)` covers the exits
/// at positions `i << level` up to `((i + 1) << level) - 1`. `named` holds
/// the stored slots of each exit in order, and `rank` gives a slot's index
/// among the `values` values they name.
fn cover_runs(
    named: &[Vec<ExitSlot>],
    values: usize,
    rank: impl Fn(&ExitSlot) -> usize,
) -> Vec<Vec<Vec<ExitSlot>>> {
    let count = named.len();
    let levels = usize::BITS as usize - (count - 1).leading_zeros() as usize;
    let mut nodes: Vec<Vec<Vec<ExitSlot>>> = (0..=levels)
        .map(|level| vec![Vec::new(); count.div_ceil(1 << level)])
        .collect();
    // The canonical cover of the run `start..=end` by tree nodes.
    let mut place = |start: usize, end: usize, slot: ExitSlot| {
        let (mut low, mut high, mut level) = (start, end + 1, 0);
        while low < high {
            if low & 1 == 1 {
                nodes[level][low].push(slot);
                low += 1;
            }
            if high & 1 == 1 {
                high -= 1;
                nodes[level][high].push(slot);
            }
            low >>= 1;
            high >>= 1;
            level += 1;
        }
    };
    // Each value's current run of consecutive exits that name it, the values
    // with a run, and the last exit that named each value. A run ends at the
    // first exit that does not name its value, so each exit looks only at the
    // values the exit before it named: the work is the slots, not exits
    // times values.
    let mut open: Vec<Option<(usize, ExitSlot)>> = vec![None; values];
    let mut running: Vec<usize> = Vec::new();
    let mut named_at: Vec<usize> = vec![usize::MAX; values];
    for (position, slots) in named.iter().enumerate() {
        for slot in slots {
            let value = rank(slot);
            named_at[value] = position;
            if open[value].is_none() {
                open[value] = Some((position, *slot));
                running.push(value);
            }
        }
        running.retain(|&value| {
            if named_at[value] == position {
                return true;
            }
            if let Some((start, slot)) = open[value].take() {
                place(start, position - 1, slot);
            }
            false
        });
    }
    for value in running {
        if let Some((start, slot)) = open[value].take() {
            place(start, count - 1, slot);
        }
    }
    // The slots of a node are sorted where they are used.
    nodes
}

/// Moves the stores of each node above the leaves that saves fewer than
/// [`LEAST_TAIL_SAVING`] stores down to its children: such a node costs more
/// as a block than it saves.
fn push_down_small_nodes(nodes: &mut [Vec<Vec<ExitSlot>>]) {
    for level in (1..nodes.len()).rev() {
        let (lower, upper) = nodes.split_at_mut(level);
        let children = &mut lower[level - 1];
        let width = children.len();
        for (node, slots) in upper[0].iter_mut().enumerate() {
            let below = &mut children[node * 2..(node * 2 + 2).min(width)];
            if !slots.is_empty() && slots.len() * (below.len() - 1) < LEAST_TAIL_SAVING {
                let moved = mem::take(slots);
                for child in below {
                    child.extend_from_slice(&moved);
                }
            }
        }
    }
}

/// Fewest stores (beyond the first exit's) a tail must save to repay its
/// block and the jumps into it.
const LEAST_TAIL_SAVING: usize = 4;

/// Lays out the values `snapshot` needs: one 8-byte slot each, nil excepted.
fn describe(
    ir: &TraceIr,
    member: usize,
    snapshot: &Snapshot,
    class: TraceExitClass,
    guard: Option<(TraceGuardClass, u32)>,
    held: &[u16],
) -> Result<ExitDescriptor, TraceCompileError> {
    let mut values: Vec<ValueId> = Vec::new();
    snapshot.for_each_value(|value| values.push(value));
    values.sort_unstable();
    let kept = kept_by_frame(ir, snapshot, held);
    let mut slots = Vec::with_capacity(values.len());
    let mut next = 0_u32;
    let mut at = 0;
    while at < values.len() {
        let value = values[at];
        let mentions = values[at..].partition_point(|other| *other == value);
        at += mentions;
        // A value the snapshot names only where the frame keeps it needs no
        // store: the restore leaves those registers alone.
        let in_frame = kept.partition_point(|other| *other < value);
        if kept[in_frame..].partition_point(|other| *other == value) == mentions {
            continue;
        }
        let ty = ir
            .value_type(value)
            .filter(|ty| *ty != TraceType::Any)
            .ok_or(TraceCompileError::NativeBackendRejected)?;
        slots.push(ExitSlot {
            value,
            ty,
            slot: next,
        });
        if ty != TraceType::Nil {
            next += 1;
        }
    }
    Ok(ExitDescriptor {
        class,
        member,
        origin: guard.map(|(_, origin)| origin),
        snapshot: SnapshotId(snapshot.id.0),
        guard: guard.map(|(class, _)| class),
        slots,
        counted: snapshot.counted,
    })
}

/// The mentions in `snapshot` of `ir` that the frame keeps, sorted: each
/// register of the root frame in `held` (see [`PlanContext::held`]) that
/// holds the member's own entry input for it. A register whose number has a
/// run-time subtype is never one (the restore converts it).
fn kept_by_frame(ir: &TraceIr, snapshot: &Snapshot, held: &[u16]) -> Vec<ValueId> {
    let mut kept = Vec::new();
    let Some(frame) = snapshot.frames.first() else {
        return kept;
    };
    if held.is_empty()
        || !ir
            .frames
            .get(frame.frame.index())
            .is_some_and(|info| info.is_root() && info.activation.is_none())
    {
        return kept;
    }
    for &register in held {
        let Some(RegisterEntry::Value(value)) = frame.registers.get(usize::from(register)) else {
            continue;
        };
        let own_input = matches!(
            ir.values.get(value.index()).map(|info| info.def),
            Some(ValueDef::Input(index))
                if ir.entry.inputs.get(index).is_some_and(|input| input.register == register)
        );
        if own_input && snapshot.number_tag(usize::from(register)).is_none() {
            kept.push(*value);
        }
    }
    kept.sort_unstable();
    kept
}

/// Whether `member` has an array set that can reach the VM's store entry: a
/// string or a table stored, or a nil pop of a header-bound array, which
/// releases the reference the emptied slot held.
pub(super) fn member_stores_references(member: &TraceIr) -> bool {
    member.body.iter().any(|instruction| {
        matches!(
            &instruction.op,
            TraceOp::ArraySet { binding, value, .. }
                if match member.value_type(*value) {
                    Some(TraceType::String | TraceType::Table) => true,
                    // The header is its own binding on the array's register.
                    Some(TraceType::Nil) => {
                        let register = member.bindings[*binding as usize].register;
                        member.bindings.iter().any(|other| {
                            other.register == register
                                && matches!(
                                    other.kind,
                                    crate::jit_ir::trace::BindingKind::Header { .. }
                                )
                        })
                    }
                    _ => false,
                }
        )
    })
}

#[cfg(test)]
mod cover_tests {
    use super::{ExitSlot, cover_runs};
    use crate::jit_ir::trace::{TraceType, ValueId};

    /// `cover_runs` as it was written first: every exit looks at every value.
    fn cover_runs_by_value(named: &[Vec<ExitSlot>], values: usize) -> Vec<Vec<Vec<ExitSlot>>> {
        let count = named.len();
        let levels = usize::BITS as usize - (count - 1).leading_zeros() as usize;
        let mut nodes: Vec<Vec<Vec<ExitSlot>>> = (0..=levels)
            .map(|level| vec![Vec::new(); count.div_ceil(1 << level)])
            .collect();
        let mut place = |start: usize, end: usize, slot: ExitSlot| {
            let (mut low, mut high, mut level) = (start, end + 1, 0);
            while low < high {
                if low & 1 == 1 {
                    nodes[level][low].push(slot);
                    low += 1;
                }
                if high & 1 == 1 {
                    high -= 1;
                    nodes[level][high].push(slot);
                }
                low >>= 1;
                high >>= 1;
                level += 1;
            }
        };
        let mut open: Vec<Option<(usize, ExitSlot)>> = vec![None; values];
        for (position, slots) in named.iter().enumerate() {
            let mut here = vec![false; values];
            for slot in slots {
                let value = slot.value.index();
                here[value] = true;
                if open[value].is_none() {
                    open[value] = Some((position, *slot));
                }
            }
            for (value, here) in here.iter().enumerate() {
                if !here && let Some((start, slot)) = open[value].take() {
                    place(start, position - 1, slot);
                }
            }
        }
        for run in &mut open {
            if let Some((start, slot)) = run.take() {
                place(start, count - 1, slot);
            }
        }
        nodes
    }

    /// Runs of values over many exits cover the same nodes as the first
    /// version of the cover found.
    #[test]
    fn the_cover_of_each_run_is_the_same_as_by_value() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let exits = 2 + (next() % 40) as usize;
            let values = 1 + (next() % 30) as usize;
            let named: Vec<Vec<ExitSlot>> = (0..exits)
                .map(|_| {
                    (0..values)
                        .filter(|_| next() % 3 != 0)
                        .map(|value| ExitSlot {
                            value: ValueId(u32::try_from(value).unwrap()),
                            ty: TraceType::Float,
                            slot: 0,
                        })
                        .collect()
                })
                .collect();
            let sorted = |mut nodes: Vec<Vec<Vec<ExitSlot>>>| {
                for level in &mut nodes {
                    for node in level {
                        node.sort_unstable_by_key(|slot| slot.value);
                    }
                }
                nodes
            };
            assert_eq!(
                sorted(cover_runs(&named, values, |slot| slot.value.index())),
                sorted(cover_runs_by_value(&named, values)),
            );
        }
    }
}
