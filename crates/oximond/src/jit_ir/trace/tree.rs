//! Trace trees: a root trace and the side traces recorded from its guards.
//!
//! A side trace continues a failing guard natively instead of leaving for the
//! interpreter. The backend compiles the whole family as one function, so a
//! side is a block the guard's failure edge jumps to. Its entry values are the
//! guard snapshot's values, and its end jumps back to the root loop header.
//! This module prepares the family: it makes every register that crosses a
//! link a value both ends agree on, and checks that each link is well typed.
//!
//! # Why the root needs widening
//!
//! The recorder treats a register the iteration does not write as
//! `Unchanged`: the interpreter register still holds it. That is wrong once a
//! side trace writes it. Take
//!
//! ```text
//! for i = 1, n do
//!     if i % 7 < 3 then yes = yes + 1 else no = no + 1 end
//! end
//! ```
//!
//! recorded along the `else` path. The root never touches `yes`, so every one
//! of its snapshots says `yes` is unchanged. The side trace (recorded along the
//! `then` path) writes it. After the side has run and linked back, a later
//! guard failure of the root would restore the stale `yes`.
//!
//! [`plan_tree`] therefore *widens* the root before it is optimized: every
//! register a side writes that is live at the loop header becomes a loop
//! parameter of the root (initialised from a new entry input, passed through
//! unchanged by the root's own iteration), and the root's snapshots name that
//! parameter instead of `Unchanged`. A register a side reads that the root
//! never reads becomes an invariant entry input. The cost is one more typed
//! entry check and one more value in a register, and it makes the root enter
//! only for the types the side saw, like any other version.
//!
//! Sides need no such rewrite: the recorder makes every live register a side
//! reads or leaves alone an entry input, so a side never relies on the
//! interpreter register of a live value.
//!
//! # The window grows
//!
//! A side may name an upvalue cell or a global the root never touched, so the
//! window beyond the guest registers can be longer in a side than in the root.
//! Each side starts from the tree's window as it was when it began recording
//! and extends it, so the window of the tree is the longest of them and the
//! others are prefixes. Every member is given that window. A cell is live
//! everywhere, so a side that did not know it takes its value as an input like
//! any other live register. A global is only read, at entry, and never
//! restored, so every member lists it as `Unchanged` and a side that reads one
//! its parent does not takes the root's entry input for it.

use super::ir::{
    BindingKind, EntryInput, ExternalSlot, FrameId, GuardId, GuardInfo, LoopParam, LoopParamInit,
    MAX_HOST_ARGS, TABLE_ADDRESS_KEY, TERMINAL_GUARD, TableBinding, TableField, TraceInstruction,
    TraceIr, TraceOp, TraceScalar, TraceSide, TraceSite, TraceTerminator, TraceType, ValueDef,
    ValueId, ValueInfo,
};
use super::register_map::RegisterMap;
use super::snapshot::{
    RegisterEntry, Snapshot, SnapshotFrame, replace_root_entry, rewrite_entries,
};
use crate::jit_ir::Comparison;

/// Whether `side` takes every register of the root frame that `snapshot`
/// restores as a virtual table as a table input: its own exits then write
/// the table the guard built, and none relies on the interpreter register.
fn takes_virtuals(side: &TraceIr, snapshot: &Snapshot) -> bool {
    snapshot.frames[0]
        .registers
        .iter()
        .enumerate()
        .filter(|(_, entry)| matches!(entry, RegisterEntry::Virtual(_)))
        .all(|(register, _)| {
            side.entry.inputs.iter().any(|input| {
                usize::from(input.register) == register && input.ty == TraceType::Table
            })
        })
}

/// Why a trace and its sides cannot become a tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TreeError {
    /// The root already has a preheader, so it was optimized before.
    NotSource,
    /// Two members give one register different types.
    TypeConflict { register: u16 },
    /// A side continues a guard its parent has no usable record of: it must be
    /// a branch guard of the parent's body that resumes in the root frame.
    NoSuchGuard { side: usize },
    /// A side's final state has no typed value of the right type for a loop
    /// parameter of the root, so it cannot link back.
    LinkValue { side: usize, register: u16 },
    /// A side input cannot take its value from the parent's snapshot.
    Unbound { side: usize, register: u16 },
    /// A side names the cells and globals of the window in another order than
    /// the root or an earlier side.
    WindowMismatch { side: usize },
    /// A side lists table bindings that disagree with the root's or an earlier
    /// side's: it was recorded against another version of the tree.
    BindingMismatch { side: usize },
    /// A side pins callables the tree pins otherwise: it was recorded against
    /// another version of the tree.
    PinMismatch { side: usize },
    /// Some member of the tree writes the register of a bound table. The
    /// binding resolved the table the register held at entry, so the register
    /// has to hold the same table in every iteration. A binding with a lane
    /// form is demoted instead (see [`plan_tree`]); this is the answer for the
    /// rest.
    TableWritten { register: u16 },
    /// The members of a function trace give their returns different types.
    ResultConflict,
    /// A side changes a string or an iterator the loop carries, and the
    /// root's release at the end of its body has no room to keep it.
    ReleaseFull { register: u16 },
    /// The guard a side continues restores a virtual table that the run
    /// cannot build on the way to the side (see
    /// [`Snapshot::bridges_virtuals`]). The side would start from state the
    /// exit materializes, which a native jump into the side skips.
    VirtualCarried { side: usize },
}

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSource => write!(f, "the root trace is already optimized"),
            Self::TypeConflict { register } => {
                write!(f, "register r{register} has different types in the tree")
            }
            Self::NoSuchGuard { side } => {
                write!(
                    f,
                    "side trace {side} continues no branch guard of its parent"
                )
            }
            Self::LinkValue { side, register } => write!(
                f,
                "side trace {side} cannot give the loop a typed r{register}"
            ),
            Self::Unbound { side, register } => write!(
                f,
                "side trace {side} cannot take r{register} from its parent's snapshot"
            ),
            Self::WindowMismatch { side } => {
                write!(f, "side trace {side} orders the window slots differently")
            }
            Self::BindingMismatch { side } => {
                write!(f, "side trace {side} binds tables the tree binds otherwise")
            }
            Self::PinMismatch { side } => {
                write!(
                    f,
                    "side trace {side} pins callables the tree pins otherwise"
                )
            }
            Self::TableWritten { register } => {
                write!(f, "register r{register} holds a bound table and is written")
            }
            Self::ResultConflict => write!(f, "the function returns different types in the tree"),
            Self::ReleaseFull { register } => write!(
                f,
                "the release that ends the root cannot also keep r{register}"
            ),
            Self::VirtualCarried { side } => write!(
                f,
                "side trace {side} continues a guard whose snapshot holds a virtual table"
            ),
        }
    }
}

impl std::error::Error for TreeError {}

/// One trace of a tree, ready to optimize and lower.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeMember {
    /// The trace, without sides. For the root, widened.
    pub ir: TraceIr,
    /// The member this one continues and the [`super::GuardInfo::origin`] of
    /// the guard. `None` for the root, which is always member 0.
    pub parent: Option<(usize, u32)>,
    /// For the root, how many table bindings the plan demoted to lane
    /// operations because a member writes their register (see
    /// [`plan_tree`]). Zero for the sides.
    pub demoted: u32,
}

/// The type a snapshot entry of `ir` has, when it has a typed value.
fn entry_type(ir: &TraceIr, entry: &RegisterEntry) -> Option<TraceType> {
    match entry {
        RegisterEntry::Value(value) => ir.value_type(*value),
        RegisterEntry::Const(scalar) => Some(scalar.ty()),
        // A root constant is a string.
        RegisterEntry::Root(_) => Some(TraceType::String),
        RegisterEntry::Unchanged | RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => None,
    }
}

/// The snapshot of the branch guard with `origin` in the body of `ir`, when
/// it resumes in the root frame, or for a side that seeds `seeded` inlined
/// frames (see [`TraceIr::seed_frames`]), inside that many calls.
fn guard_snapshot(ir: &TraceIr, origin: u32, seeded: u8) -> Option<&Snapshot> {
    // The terminal exit of a trace that ends where a loop side begins, in
    // the root frame or in the frames of calls (see "Loops in calls" in
    // `vm::jit::trace`).
    if origin == TERMINAL_GUARD {
        let TraceTerminator::Exit { snapshot } = ir.terminator else {
            return None;
        };
        let snapshot = ir.snapshots.get(snapshot.index())?;
        return (snapshot.frames.len() == 1 + usize::from(seeded)).then_some(snapshot);
    }
    let preheader_guards = ir
        .preheader
        .iter()
        .filter(|instruction| instruction.op.guard().is_some())
        .count();
    let info = ir
        .guards
        .iter()
        .skip(preheader_guards)
        .find(|info| info.origin == origin)?;
    let snapshot = ir.snapshots.get(info.snapshot.index())?;
    (snapshot.frames.len() == 1 + usize::from(seeded)).then_some(snapshot)
}

/// Adds to `written` the registers of the root frame that a side starting
/// at the guard `origin` of `parent` (inside `seeded` calls) finds changed
/// from the state `parent` entered with, beyond the ones `parent` writes:
/// those the guard's snapshot names with a value `parent` made or a
/// constant. `written_registers` lists only what the recorded
/// path writes. A guard behind an instruction that writes a register on the
/// other path only commits that write in its snapshot. The closing `ForLoop`
/// of a numeric `for` loop recorded as the loop ended is one: the side that
/// goes on with the loop starts with the loop's next variable, which its
/// parent never writes. A link below that side carried the new variable to
/// the root's header, where the root went on with its entry value (the
/// region nest of `trace_return_nests.rs` read the cells of the first `x`
/// on every outer iteration). A `Const(Nil)` is a register the snapshot
/// clears as dead, not a write, and so is the entry input of the register
/// itself: the value `parent` came in with. The entry input of another
/// register is a copy (a `TFORLOOP` sets the control to the first
/// variable), except in a boxed `pairs` loop: there the key is the control
/// at the header, and a snapshot names it by the control's input (see
/// [`TraceIr::boxed_control`]). Counting that made the key (boxed, as the
/// control is) a typed loop parameter that no entry matched.
pub(crate) fn guard_writes(
    parent: &TraceIr,
    origin: u32,
    seeded: u8,
    written: &mut Vec<(FrameId, u16)>,
) {
    if let Some(frame) =
        guard_snapshot(parent, origin, seeded).and_then(|snapshot| snapshot.frames.first())
    {
        frame_writes(parent, frame, written);
    }
}

/// [`guard_writes`] for the root frame `frame` of a snapshot of `parent`.
pub(crate) fn frame_writes(
    parent: &TraceIr,
    frame: &SnapshotFrame,
    written: &mut Vec<(FrameId, u16)>,
) {
    if frame.frame != FrameId(0) {
        return;
    }
    // The register whose entry input names `register` without a write.
    let source = |register: usize| match parent.boxed_control {
        Some(control) if register == usize::from(control) + 1 => usize::from(control),
        _ => register,
    };
    for (register, entry) in frame.registers.iter().enumerate() {
        let changed = match entry {
            RegisterEntry::Unchanged | RegisterEntry::Const(TraceScalar::Nil) => false,
            RegisterEntry::Value(value) => !matches!(
                parent.values.get(value.index()).map(|info| info.def),
                Some(ValueDef::Input(input)) if parent
                    .entry
                    .inputs
                    .get(input)
                    .is_some_and(|input| {
                        let held = usize::from(input.register);
                        held == register || held == source(register)
                    })
            ),
            RegisterEntry::Const(_)
            | RegisterEntry::Root(_)
            | RegisterEntry::Virtual(_)
            | RegisterEntry::Slice(_) => true,
        };
        let Ok(register) = u16::try_from(register) else {
            continue;
        };
        if changed && !written.contains(&(FrameId(0), register)) {
            written.push((FrameId(0), register));
        }
    }
}

/// What the guard each side of the tree starts at commits for it
/// ([`guard_writes`]), over all the sides of `root`.
pub(crate) fn start_writes(root: &TraceIr, sides: &[TraceSide]) -> Vec<(FrameId, u16)> {
    let mut written = Vec::new();
    for side in sides {
        let parent = side.parent.and_then(|parent| sides.get(parent));
        let parent_ir = parent.map_or(root, |parent| &*parent.ir);
        guard_writes(parent_ir, side.guard, side.ir.seed_frames, &mut written);
    }
    written
}

/// The type of `register` as side `index` of `root` starts with it: what
/// the snapshot of the guard it starts at names, or for a register that
/// snapshot leaves `Unchanged`, what its parent started with in turn. `None`
/// when that is the root's entry value, a borrow, a root constant or a number
/// of run-time subtype, none of which a side takes as a typed input here.
fn started_type(
    root: &TraceIr,
    sides: &[TraceSide],
    index: usize,
    register: u16,
) -> Option<TraceType> {
    let mut child = sides.get(index)?;
    loop {
        let parent_ir = match child.parent {
            None => root,
            Some(parent) => &*sides.get(parent)?.ir,
        };
        let snapshot = guard_snapshot(parent_ir, child.guard, child.ir.seed_frames)?;
        match snapshot
            .frames
            .first()?
            .registers
            .get(usize::from(register))?
        {
            RegisterEntry::Unchanged => child = sides.get(child.parent?)?,
            entry @ (RegisterEntry::Value(_) | RegisterEntry::Const(_))
                if snapshot.number_tag(usize::from(register)).is_none() =>
            {
                return entry_type(parent_ir, entry).filter(|ty| !ty.is_borrowed());
            }
            _ => return None,
        }
    }
}

fn new_value(ir: &mut TraceIr, ty: TraceType, def: ValueDef) -> ValueId {
    ir.values.push(ValueInfo { ty, def });
    ValueId(u32::try_from(ir.values.len() - 1).expect("value count fits u32"))
}

/// Gives `ir` the window `external` (which extends its own): the extra slots
/// hold what the trace does not touch, so every snapshot lists them as
/// `Unchanged`. A member that never heard of a slot (a side recorded before
/// another side named it) has to be given the slot's value as an input and
/// name it in its snapshots, as every side does for a live register.
fn extend_window(
    ir: &mut TraceIr,
    external: &[ExternalSlot],
    types: &[Option<TraceType>],
    is_side: bool,
) -> Result<(), TreeError> {
    let old = ir.external.len();
    let guest = usize::from(ir.frames[0].register_count) - old;
    if old == external.len() {
        return Ok(());
    }
    ir.frames[0].register_count = u16::try_from(guest + external.len()).expect("window fits u16");
    ir.external = external.to_vec();
    // The map the snapshot before became, which shares the new runs.
    let mut previous: Option<RegisterMap> = None;
    for snapshot in &mut ir.snapshots {
        if let Some(frame) = snapshot.frames.first_mut() {
            frame.registers = frame.registers.resized(
                guest + external.len(),
                RegisterEntry::Unchanged,
                previous.as_ref(),
            );
            previous = Some(frame.registers.clone());
        }
    }
    for (slot, ty) in types.iter().enumerate().skip(old) {
        // A function trace's tail call count changes in every activation
        // that tail calls: the root names it too.
        let count = ir.function.is_some() && external[slot] == ExternalSlot::TailCalls;
        if !is_side && !count {
            continue;
        }
        let register = u16::try_from(guest + slot).expect("window fits u16");
        // A slot that is a cell or global is live everywhere.
        let ty = ty.ok_or(TreeError::TypeConflict { register })?;
        let input = ensure_input(ir, register, ty)?;
        // A global is never written, so its snapshot entries stay `Unchanged`
        // (the restore plan refuses a global slot with a value).
        if matches!(
            external[slot],
            ExternalSlot::Global(_)
                | ExternalSlot::ClosureGlobal { .. }
                | ExternalSlot::CallerGlobal(_)
                | ExternalSlot::UpvalueField { .. }
                | ExternalSlot::ClosureUpvalueField { .. }
                | ExternalSlot::Environment
        ) {
            continue;
        }
        replace_root_entry(
            &mut ir.snapshots,
            usize::from(register),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(input),
        );
    }
    Ok(())
}

/// Gives every member of the tree the pinned callables of all of them.
///
/// A side lists the root's pinned closures first and may pin natives after
/// them, so the longest list is the tree's and every other list has to be a
/// prefix of it. A lane then names the same callable in every member.
///
/// A side compiled apart (`apart`) may pin more than the rest of the tree
/// does, a closure of the caller it returns into, say. The tree's code names
/// no lane beyond its own, so the extra pins stay with the side: they do not
/// widen the root, and the run pins them at entry instead (see
/// [`TraceIr::pinned_closures`]). The run pins one list for all of them, so
/// the extra pins of every side compiled apart must agree lane by lane too.
fn share_pins(
    root: &mut TraceIr,
    sides: &mut [TraceSide],
    apart: &[usize],
) -> Result<(), TreeError> {
    let agree = |held: &[u64], side: &[u64]| {
        let shared = held.len().min(side.len());
        held[..shared] == side[..shared]
    };
    let mut pinned = root.pinned_closures.clone();
    let mut run_pinned = pinned.clone();
    for (index, side) in sides.iter().enumerate() {
        let side_pins = &side.ir.pinned_closures;
        // `pinned` is a prefix of `run_pinned`, so this checks both.
        if !agree(&run_pinned, side_pins) {
            return Err(TreeError::PinMismatch { side: index });
        }
        if side_pins.len() > pinned.len() && !apart.contains(&index) {
            pinned.clone_from(side_pins);
        }
        if side_pins.len() > run_pinned.len() {
            run_pinned.clone_from(side_pins);
        }
    }
    root.pinned_closures.clone_from(&pinned);
    for (index, side) in sides.iter_mut().enumerate() {
        if !(apart.contains(&index) && side.ir.pinned_closures.len() > pinned.len()) {
            side.ir.pinned_closures.clone_from(&pinned);
        }
    }
    Ok(())
}

/// Gives every member of the tree the table bindings of all of them.
///
/// A side lists the bindings of the root first and may add its own after
/// them, so the longest list is the tree's, and every other list has to be a
/// prefix of it: a side recorded before another side added a binding is
/// stale (the same case as a window that grew). Operations name a binding by
/// its position, which is then the same in every member. The entry resolves
/// them all, from the table in the entry input of the binding's register, so
/// every member gets an input for each of those registers. That is only
/// sound when nothing in the tree writes such a register, since then every
/// snapshot leaves it `Unchanged` and every side's input is the root's. A
/// binding whose register is written is demoted to lane operations where it
/// has a lane form ([`demote_written_bindings`]), and refused otherwise.
///
/// Returns how many bindings were demoted to lane operations, and the
/// registers whose bindings stay behind a link guard ([`guard_identity`]):
/// the root does not carry those, since every link hands back the entry
/// table.
fn share_bindings(
    root: &mut TraceIr,
    sides: &mut [TraceSide],
) -> Result<(u32, Vec<u16>), TreeError> {
    let mut bindings = union_bindings(root, sides)?;
    if bindings.is_empty() {
        return Ok((0, Vec::new()));
    }
    let demoted = demote_written_bindings(root, sides, &mut bindings)?;
    for member in std::iter::once(&mut *root).chain(sides.iter_mut().map(|side| &mut *side.ir)) {
        member.bindings.clone_from(&bindings);
        for binding in &bindings {
            let ty = if matches!(binding.kind, BindingKind::Bytes { .. }) {
                TraceType::String
            } else {
                TraceType::Table
            };
            ensure_input(member, binding.register, ty)?;
        }
    }
    Ok(demoted)
}

/// The table bindings of the tree `ir` and its sides as [`plan_tree`] shares
/// them before it demotes any: the list a side recording starts from, so
/// that the side names a binding of the tree by the position the plan gives
/// it, whether or not the plan then demotes it.
///
/// # Errors
///
/// [`TreeError::BindingMismatch`] for a side recorded against another
/// version of the tree.
pub fn tree_bindings(ir: &TraceIr) -> Result<Vec<TableBinding>, TreeError> {
    let mut root = ir.clone();
    let mut sides = std::mem::take(&mut root.sides);
    union_bindings(&root, &mut sides)
}

/// The union of the bindings of `root` and `sides` (see [`share_bindings`]),
/// with each side's operations renumbered to the union's positions.
fn union_bindings(root: &TraceIr, sides: &mut [TraceSide]) -> Result<Vec<TableBinding>, TreeError> {
    let mut bindings = root.bindings.clone();
    for (index, side) in sides.iter_mut().enumerate() {
        // Sides recorded apart each add their own bindings after the root's.
        // The tree takes the union, and a side's operations are renumbered
        // to the union's positions. A side that disagrees with a binding of
        // the tree on the same table register and kind is stale.
        let mut map = Vec::with_capacity(side.ir.bindings.len());
        for (position, wanted) in side.ir.bindings.iter().enumerate() {
            if bindings.get(position) == Some(wanted) {
                map.push(position);
                continue;
            }
            if let Some(known) = bindings.iter().position(|known| known == wanted) {
                map.push(known);
                continue;
            }
            // One table register has one array binding and one header binding
            // (their hints differ without making them two bindings).
            let single = matches!(
                wanted.kind,
                BindingKind::Array { .. } | BindingKind::Header { .. }
            );
            let conflict = single
                && bindings.iter().any(|known| {
                    known.register == wanted.register
                        && std::mem::discriminant(&known.kind)
                            == std::mem::discriminant(&wanted.kind)
                });
            if conflict || position < root.bindings.len() {
                return Err(TreeError::BindingMismatch { side: index });
            }
            bindings.push(wanted.clone());
            map.push(bindings.len() - 1);
        }
        if map.iter().enumerate().any(|(from, to)| from != *to) {
            let renumber = |instruction: &mut TraceInstruction| {
                if let Some(binding) = instruction.op.binding_mut() {
                    *binding = u32::try_from(map[*binding as usize]).expect("bindings fit u32");
                }
            };
            side.ir.preheader.iter_mut().for_each(renumber);
            side.ir.body.iter_mut().for_each(renumber);
        }
    }
    Ok(bindings)
}

/// Demotes the bindings of `bindings` (the tree's shared list) whose
/// register some member writes to operations on the table's lane, and
/// removes them from the list. `LuaJIT` has no entry bindings at all: every
/// trace loads the array words from the table object. A binding resolves the
/// table its register held at entry, which a member that writes the register
/// (an argument register the root fills every trip, a parameter a fresh
/// activation rewrites) makes wrong; the operations themselves already name
/// the table they access as a value, which is right in every member.
///
/// An [`BindingKind::Array`] binding has a lane form for reads of a present
/// value of a type a lane read gives (an [`TraceOp::ArrayLoad`] of an
/// integer, float, boolean or table, not past the end): each becomes a
/// [`TraceOp::LaneArrayLoad`] on the same table value, which reads the array
/// words through the table's address in its row ([`TABLE_ADDRESS_KEY`],
/// added to every member's fields). Its guard and snapshot are the load's,
/// so it leaves where the load left.
///
/// A binding with no lane form ([`BindingKind::Field`], `Length`, `Header`,
/// `NoMetatable`, or an `Array` binding with a store, a set, a number load,
/// a nil read or an array step) stays when only sides write its register,
/// each ends in a [`TraceTerminator::Link`] or an exit, and none of them has
/// sides of its own: [`guard_identity`] then makes each linking writer leave
/// through its link snapshot unless the register holds the entry table
/// again, so every path through the tree still sees the table the binding
/// resolved. A `FieldLoad` has no guard and no snapshot to leave by, so a
/// rewrite into a `TableFieldLoad` is not possible at plan time. Any other
/// binding on a written register (a `Callee`, `Bytes`, `StringField` or
/// `MetaClass` one, or one the root writes) keeps
/// [`TreeError::TableWritten`]. A function trace demotes nothing: its runs
/// may give one table two lanes.
///
/// The rest of the list keeps its order, and every operation is renumbered
/// to it. Returns how many bindings were demoted to lane operations, and the
/// registers of the guarded ones.
fn demote_written_bindings(
    root: &mut TraceIr,
    sides: &mut [TraceSide],
    bindings: &mut Vec<TableBinding>,
) -> Result<(u32, Vec<u16>), TreeError> {
    let started = start_writes(root, sides);
    let written = |register: u16| {
        let writes = |ir: &TraceIr| ir.written_registers.contains(&(FrameId(0), register));
        writes(root)
            || sides.iter().any(|side| writes(&side.ir))
            || started.contains(&(FrameId(0), register))
    };
    let mut demoted: Vec<bool> = bindings
        .iter()
        .map(|binding| written(binding.register))
        .collect();
    let Some(first) = demoted.iter().position(|&demote| demote) else {
        return Ok((0, Vec::new()));
    };
    if root.function.is_some() {
        return Err(TreeError::TableWritten {
            register: bindings[first].register,
        });
    }
    // Every operation that names a lane-demoted binding must have a lane
    // form.
    let lane_form = |op: &TraceOp| match op {
        TraceOp::ArrayLoad {
            expected,
            past_end: false,
            ..
        } => matches!(
            expected,
            TraceType::Integer | TraceType::Float | TraceType::Boolean | TraceType::Table
        ),
        _ => false,
    };
    let mut lane = vec![true; bindings.len()];
    for (index, binding) in bindings.iter().enumerate() {
        lane[index] = matches!(binding.kind, BindingKind::Array { .. });
    }
    for member in std::iter::once(&*root).chain(sides.iter().map(|side| &*side.ir)) {
        // The native steps of a boxed `next` loop read the array binding.
        if let Some(binding) = member.boxed_steps.and_then(|steps| steps.array) {
            lane[binding as usize] = false;
        }
        for instruction in member.preheader.iter().chain(&member.body) {
            let mut op = instruction.op.clone();
            if let Some(binding) = op.binding_mut()
                && !lane_form(&instruction.op)
            {
                lane[*binding as usize] = false;
            }
        }
    }
    let mut guarded: Vec<u16> = Vec::new();
    for (index, binding) in bindings.iter().enumerate() {
        if !demoted[index] || lane[index] {
            continue;
        }
        let keeps = matches!(
            binding.kind,
            BindingKind::Array { .. }
                | BindingKind::Field { .. }
                | BindingKind::Length { .. }
                | BindingKind::Header { .. }
                | BindingKind::NoMetatable
        );
        if !keeps || !identity_guardable(root, sides, &started, binding.register) {
            return Err(TreeError::TableWritten {
                register: binding.register,
            });
        }
        demoted[index] = false;
        if !guarded.contains(&binding.register) {
            guarded.push(binding.register);
        }
    }
    for &register in &guarded {
        for side in sides.iter_mut() {
            if side.ir.written_registers.contains(&(FrameId(0), register)) {
                guard_identity(&mut side.ir, register)?;
            }
        }
    }
    let lane_count = demoted.iter().filter(|&&demote| demote).count();
    let count = u32::try_from(lane_count).expect("bindings fit u32");
    if lane_count == 0 {
        return Ok((count, guarded));
    }
    // The address word, at the same position in every member: the fields
    // are shared already.
    let known = root
        .fields
        .iter()
        .position(|field| field.key.as_ref() == TABLE_ADDRESS_KEY);
    let field = if let Some(position) = known {
        position
    } else {
        let address = TableField {
            key: TABLE_ADDRESS_KEY.into(),
            ty: TraceType::Integer,
        };
        for member in std::iter::once(&mut *root).chain(sides.iter_mut().map(|side| &mut *side.ir))
        {
            member.fields.push(address.clone());
        }
        root.fields.len() - 1
    };
    let field = u32::try_from(field).expect("fields fit u32");
    let mut map: Vec<Option<u32>> = Vec::with_capacity(bindings.len());
    let mut kept = 0_u32;
    for &demote in &demoted {
        map.push((!demote).then_some(kept));
        kept += u32::from(!demote);
    }
    let rewrite = |instruction: &mut TraceInstruction| {
        if let TraceOp::ArrayLoad {
            binding,
            table,
            key,
            expected,
            guard,
            ..
        } = instruction.op
            && map[binding as usize].is_none()
        {
            instruction.op = TraceOp::LaneArrayLoad {
                table,
                field,
                key,
                expected,
                guard,
            };
        }
        if let Some(binding) = instruction.op.binding_mut() {
            *binding = map[*binding as usize].expect("a kept binding has a position");
        }
    };
    for member in std::iter::once(&mut *root).chain(sides.iter_mut().map(|side| &mut *side.ir)) {
        member.preheader.iter_mut().for_each(rewrite);
        member.body.iter_mut().for_each(rewrite);
        if let Some(steps) = &mut member.boxed_steps
            && let Some(binding) = &mut steps.array
        {
            *binding = map[*binding as usize].expect("a kept binding has a position");
        }
    }
    let mut position = 0;
    bindings.retain(|_| {
        position += 1;
        !demoted[position - 1]
    });
    Ok((count, guarded))
}

/// Whether the tables bound on `register` can stay bound although sides
/// write it (see [`demote_written_bindings`]): the root does not write it,
/// and every side that does ends in a link or an exit and has no sides of its
/// own, whose entry would see the written value. A side that links with the
/// entry table of another local of the root frame in it (`a, b = b, a`,
/// `t = B`) would leave at its link guard on every pass, so it is refused
/// instead, as before demotion. So is a register that a guard's snapshot
/// alone writes ([`start_writes`]): the side it starts already sees the
/// written value.
fn identity_guardable(
    root: &TraceIr,
    sides: &[TraceSide],
    started: &[(FrameId, u16)],
    register: u16,
) -> bool {
    let slot = (FrameId(0), register);
    if root.written_registers.contains(&slot) || started.contains(&slot) {
        return false;
    }
    let swaps = |ir: &TraceIr| {
        let TraceTerminator::Link { snapshot } = ir.terminator else {
            return false;
        };
        let Some(RegisterEntry::Value(value)) = ir.snapshots[snapshot.index()].frames[0]
            .registers
            .get(usize::from(register))
        else {
            return false;
        };
        // Through moves, to where the value came from.
        let mut value = *value;
        while let ValueDef::Instruction(index) = ir.values[value.index()].def
            && let TraceOp::Move { source } = ir.body[index].op
        {
            value = source;
        }
        // Another local of the root frame. A caller's register (a window
        // slot past the guest registers) is the argument the caller passes
        // again, usually the same table.
        let guest = usize::from(ir.frames[0].register_count).saturating_sub(ir.external.len());
        matches!(
            ir.values[value.index()].def,
            ValueDef::Input(input)
                if ir.entry.inputs[input].register != register
                    && usize::from(ir.entry.inputs[input].register) < guest
        )
    };
    sides.iter().enumerate().all(|(index, side)| {
        !side.ir.written_registers.contains(&slot)
            || (matches!(
                side.ir.terminator,
                TraceTerminator::Link { .. } | TraceTerminator::Exit { .. }
            ) && !sides.iter().any(|other| other.parent == Some(index))
                && !swaps(&side.ir))
    })
}

/// Makes the side `ir`, which writes the root register `register`, leave
/// through its link snapshot unless the table it links with in `register` is
/// the one it entered with: a `Compare` and a `GuardBranch` after the body.
/// The guard is the last in body order, with an origin past every other, so
/// the snapshot it leaves by is the state the loop header would see and the
/// interpreter resumes there. A side that exits instead leaves the tree, and
/// a link that leaves the register `Unchanged` needs no guard.
fn guard_identity(ir: &mut TraceIr, register: u16) -> Result<(), TreeError> {
    let refuse = TreeError::TableWritten { register };
    let snapshot_id = match ir.terminator {
        TraceTerminator::Link { snapshot } => snapshot,
        TraceTerminator::Exit { .. } => return Ok(()),
        _ => return Err(refuse),
    };
    let snapshot = &ir.snapshots[snapshot_id.index()];
    let linked = match snapshot.frames[0].registers.get(usize::from(register)) {
        Some(RegisterEntry::Value(value)) => *value,
        Some(RegisterEntry::Unchanged) => return Ok(()),
        _ => return Err(refuse),
    };
    if ir.values[linked.index()].ty != TraceType::Table {
        return Err(refuse);
    }
    let innermost = snapshot.frames[snapshot.frames.len() - 1].frame;
    let site = TraceSite {
        frame: innermost,
        pc: snapshot.phase_pc(),
    };
    let entered = ensure_input(ir, register, TraceType::Table)?;
    let same = new_value(ir, TraceType::Boolean, ValueDef::Instruction(ir.body.len()));
    ir.body.push(TraceInstruction {
        site,
        result: Some(same),
        op: TraceOp::Compare {
            comparison: Comparison::Equal,
            left: linked,
            right: entered,
        },
    });
    let guard = GuardId(u32::try_from(ir.guards.len()).expect("guard count fits u32"));
    let origin = ir
        .guards
        .iter()
        .map(|info| info.origin)
        .filter(|&origin| origin != TERMINAL_GUARD)
        .max()
        .map_or(0, |origin| origin + 1);
    ir.guards.push(GuardInfo {
        snapshot: snapshot_id,
        site,
        origin,
    });
    ir.body.push(TraceInstruction {
        site,
        result: None,
        op: TraceOp::GuardBranch {
            condition: same,
            expected: true,
            guard,
        },
    });
    Ok(())
}

/// Gives every member of the tree the table fields of all of them, as
/// [`share_bindings`] does for bindings: the longest list is the tree's and
/// every other list has to be a prefix of it. Fields name no register, so
/// nothing else needs to hold.
fn share_fields(root: &mut TraceIr, sides: &mut [TraceSide]) -> Result<(), TreeError> {
    let mut fields = root.fields.clone();
    for (index, side) in sides.iter().enumerate() {
        let shared = fields.len().min(side.ir.fields.len());
        if fields[..shared] != side.ir.fields[..shared] {
            return Err(TreeError::BindingMismatch { side: index });
        }
        if side.ir.fields.len() > fields.len() {
            fields.clone_from(&side.ir.fields);
        }
    }
    for member in std::iter::once(&mut *root).chain(sides.iter_mut().map(|side| &mut *side.ir)) {
        member.fields.clone_from(&fields);
    }
    Ok(())
}

/// Gives a side's entry input the type its parent hands over, when the side
/// only passes the register on: no operation of the side reads the value,
/// only its snapshots and its link name it. Such a register is a local that
/// is in scope where the side runs (a `local a, b` of the loop around the
/// root, which every exit restores for the debug library) but that the side
/// neither reads nor writes. Its recording took the type the register held
/// in the run it watched, which a side recorded from the interpreter (a
/// scout's, see "Nested loops") may have seen under other types than the
/// root's entry: `ly_dom` of `mcl_util/shape.lua`'s `region_evaluate` is
/// nil on one call and a number on the next. The value the side restores is
/// the parent's, so it has the parent's type, and the side binds.
///
/// Only slots where one of the two types is a number are retyped that way:
/// that is the nil-or-number local of the case above. A table or string slot
/// the side disagrees on keeps the old way out (the tree retires and records
/// again), which in a nest of `while` loops whose callee returns `nil, nil`
/// or two indices (`n1.lua` of the Luanti repros) grew a tree that ran 11%
/// fewer instructions than the one the retyped side joined.
///
/// An input of any type that no snapshot of the side names either is
/// retyped too. A side takes an input for the register of every table
/// binding of the tree it was recorded against, and a binding the plan
/// demotes sits in a temporary (`l.z_edges` in `l.z_edges[lz]`), which is
/// dead where the side starts: nil where the parent's exit cleared it, a
/// table where the interpreter ran last. Two sides that disagree on such a
/// register would otherwise never share a tree.
fn retype_passed_inputs(root: &TraceIr, sides: &mut [TraceSide], apart: &[usize]) {
    for index in 0..sides.len() {
        if apart.contains(&index) {
            continue;
        }
        let parent_ir = match sides[index].parent {
            None => root,
            Some(parent) if parent < index => &sides[parent].ir,
            Some(_) => continue,
        };
        let Some(snapshot) =
            guard_snapshot(parent_ir, sides[index].guard, sides[index].ir.seed_frames)
        else {
            continue;
        };
        let mut retyped: Vec<(usize, TraceType, bool)> = Vec::new();
        for (position, input) in sides[index].ir.entry.inputs.iter().enumerate() {
            let register = usize::from(input.register);
            // A loop side's parameter starts as the input of its register.
            if snapshot.number_tag(register).is_some()
                || sides[index]
                    .ir
                    .loop_params
                    .iter()
                    .any(|param| param.register == Some(input.register))
            {
                continue;
            }
            let handed = match snapshot.frames[0].registers.get(register) {
                // A parent that passes the register on without an input of
                // its own started with what its own parent handed it.
                Some(RegisterEntry::Unchanged) => parent_ir
                    .entry
                    .inputs
                    .iter()
                    .find(|held| held.register == input.register)
                    .map(|held| held.ty)
                    .or_else(|| started_type(root, sides, index, input.register)),
                Some(entry) => entry_type(parent_ir, entry),
                None => None,
            };
            let Some(handed) = handed.filter(|handed| *handed != input.ty) else {
                continue;
            };
            let number = |ty: TraceType| matches!(ty, TraceType::Integer | TraceType::Float);
            if handed.is_borrowed() || handed == TraceType::Any {
                continue;
            }
            retyped.push((position, handed, number(handed) || number(input.ty)));
        }
        let side = &mut sides[index].ir;
        for (position, handed, numeric) in retyped {
            let value = side.entry.inputs[position].value;
            let mut read = false;
            for instruction in side.preheader.iter().chain(&side.body) {
                instruction.op.operands(|operand| read |= operand == value);
            }
            // An input no snapshot names either is one the recording took
            // for a register it never touched: the temporary a table binding
            // of the tree sits in, which held whatever the interpreter left
            // there. Its type says nothing, whatever it is.
            if !numeric {
                for snapshot in &side.snapshots {
                    snapshot.for_each_value(|named| read |= named == value);
                }
            }
            if read {
                continue;
            }
            side.entry.inputs[position].ty = handed;
            side.values[value.index()].ty = handed;
        }
    }
}

/// Gives `ir` an entry input for `register`, or checks the one it has.
fn ensure_input(ir: &mut TraceIr, register: u16, ty: TraceType) -> Result<ValueId, TreeError> {
    if let Some(input) = ir
        .entry
        .inputs
        .iter()
        .find(|input| input.register == register)
    {
        return if input.ty == ty {
            Ok(input.value)
        } else {
            Err(TreeError::TypeConflict { register })
        };
    }
    let value = new_value(ir, ty, ValueDef::Input(ir.entry.inputs.len()));
    ir.entry.inputs.push(EntryInput {
        register,
        ty,
        value,
    });
    Ok(value)
}

/// Makes each register of `carry` a loop parameter of the root that every
/// iteration passes on unchanged unless a side writes it, and rewrites the
/// root's snapshots to name them.
///
/// The registers are carried together: the body and the snapshots are walked
/// once for all of them, where carrying one at a time walked them once per
/// register. Debug builds carry them one at a time as well and compare.
fn carry_registers(root: &mut TraceIr, carry: &[(u16, TraceType)]) -> Result<(), TreeError> {
    if carry.is_empty() {
        return Ok(());
    }
    #[cfg(debug_assertions)]
    let reference = {
        let mut reference = root.clone();
        carry
            .iter()
            .try_for_each(|&(register, ty)| carry_register(&mut reference, register, ty))
            .map(|()| reference)
    };
    let carried = carry_together(root, carry);
    #[cfg(debug_assertions)]
    match (&carried, &reference) {
        (Ok(()), Ok(reference)) => assert!(
            root == reference,
            "registers carried together differ from the same carried one at a time"
        ),
        (Err(error), Err(reference)) => assert_eq!(error, reference),
        _ => panic!("carrying registers together fails where one at a time does not"),
    }
    carried
}

fn carry_together(root: &mut TraceIr, carry: &[(u16, TraceType)]) -> Result<(), TreeError> {
    // A closure is only an iterator when the root makes or steps one; a
    // plain closure needs no keeping.
    let makes_iterators = root.body.iter().any(|item| {
        matches!(
            &item.op,
            TraceOp::CallHelper { helper, .. } if helper.uses_iterator()
        )
    });
    // The entry inputs that were invariants, each with the parameter that
    // takes its place. The values are renamed once, after the loop: what the
    // loop reads of the body and the snapshots until then still names the
    // inputs, and is renamed with the rest.
    let mut renamed: Vec<(ValueId, ValueId)> = Vec::new();
    let mut params: Vec<(u16, ValueId)> = Vec::with_capacity(carry.len());
    for &(register, ty) in carry {
        let existing = root
            .entry
            .inputs
            .iter()
            .any(|input| input.register == register);
        let input = ensure_input(root, register, ty)?;
        let param = new_value(root, ty, ValueDef::LoopParam(root.loop_params.len()));
        // The release that ends the body drops the lanes the next iteration
        // cannot name, so a carried string or iterator is one it keeps.
        if (ty == TraceType::String || (ty == TraceType::Closure && makes_iterators))
            && let Some(TraceOp::ReleaseStrings { keep }) =
                root.body.last_mut().map(|item| &mut item.op)
        {
            if existing && keep.contains(&input) {
                // An invariant the release kept is the parameter once the
                // values are renamed.
            } else if keep.len() >= MAX_HOST_ARGS {
                return Err(TreeError::ReleaseFull { register });
            } else {
                keep.push(param);
            }
        }
        if existing && !renamed.iter().any(|(known, _)| *known == input) {
            // The register was an invariant. It now changes across
            // iterations, so the operations that read the entry value must
            // read the parameter.
            renamed.push((input, param));
        }
        root.loop_params.push(LoopParam {
            value: param,
            ty,
            init: LoopParamInit::Input(input),
            register: Some(register),
        });
        // A register the root's own iteration writes without reading it (the
        // key variable of a `next` loop whose body never reads it) is named by
        // the poll snapshot: the parameter takes that value at the loop-back,
        // and passes on unchanged only when the root leaves the register alone.
        let written = root
            .poll_snapshot
            .and_then(|poll| root.snapshots[poll.index()].frames.first())
            .and_then(|frame| frame.registers.get(usize::from(register)))
            .and_then(|entry| match entry {
                RegisterEntry::Value(value) if root.value_type(*value) == Some(ty) => Some(*value),
                _ => None,
            });
        if let TraceTerminator::LoopBack { args } = &mut root.terminator {
            args.push(written.unwrap_or(param));
        }
        params.push((register, param));
        if !root.written_registers.contains(&(FrameId(0), register)) {
            root.written_registers.push((FrameId(0), register));
        }
    }
    if !renamed.is_empty() {
        let replace = |value: &mut ValueId| {
            if let Some((_, param)) = renamed.iter().find(|(input, _)| input == value) {
                *value = *param;
            }
        };
        for instruction in &mut root.body {
            instruction.op.operands_mut(replace);
        }
        rewrite_entries(&mut root.snapshots, |entry| {
            if let RegisterEntry::Value(value) = entry {
                replace(value);
            }
        });
        if let TraceTerminator::LoopBack { args } = &mut root.terminator {
            args.iter_mut().for_each(replace);
        }
    }
    for (register, param) in params {
        replace_root_entry(
            &mut root.snapshots,
            usize::from(register),
            RegisterEntry::Unchanged,
            RegisterEntry::Value(param),
        );
    }
    Ok(())
}

/// [`carry_registers`] for one register, as it was first written: the
/// reference debug builds check the registers carried together against.
#[cfg(debug_assertions)]
fn carry_register(root: &mut TraceIr, register: u16, ty: TraceType) -> Result<(), TreeError> {
    let existing = root
        .entry
        .inputs
        .iter()
        .find(|input| input.register == register)
        .map(|input| input.value);
    let input = ensure_input(root, register, ty)?;
    let param = new_value(root, ty, ValueDef::LoopParam(root.loop_params.len()));
    if existing.is_some() {
        // The register was an invariant. It now changes across iterations, so
        // the operations that read the entry value must read the parameter.
        let replace = |value: &mut ValueId| {
            if *value == input {
                *value = param;
            }
        };
        for instruction in &mut root.body {
            instruction.op.operands_mut(replace);
        }
        rewrite_entries(&mut root.snapshots, |entry| {
            if let RegisterEntry::Value(value) = entry {
                replace(value);
            }
        });
        if let TraceTerminator::LoopBack { args } = &mut root.terminator {
            args.iter_mut().for_each(replace);
        }
    }
    // The release that ends the body drops the lanes the next iteration
    // cannot name, so a carried string or iterator is one it keeps.
    // A closure is only an iterator when the root makes or steps one; a
    // plain closure needs no keeping.
    let makes_iterators = root.body.iter().any(|item| {
        matches!(
            &item.op,
            TraceOp::CallHelper { helper, .. } if helper.uses_iterator()
        )
    });
    if (ty == TraceType::String || (ty == TraceType::Closure && makes_iterators))
        && let Some(TraceOp::ReleaseStrings { keep }) =
            root.body.last_mut().map(|item| &mut item.op)
    {
        if keep.contains(&param) {
            // An invariant the release kept is the parameter now.
        } else if keep.len() >= MAX_HOST_ARGS {
            return Err(TreeError::ReleaseFull { register });
        } else {
            keep.push(param);
        }
    }
    root.loop_params.push(LoopParam {
        value: param,
        ty,
        init: LoopParamInit::Input(input),
        register: Some(register),
    });
    // A register the root's own iteration writes without reading it (the
    // key variable of a `next` loop whose body never reads it) is named by
    // the poll snapshot: the parameter takes that value at the loop-back,
    // and passes on unchanged only when the root leaves the register alone.
    let written = root
        .poll_snapshot
        .and_then(|poll| root.snapshots[poll.index()].frames.first())
        .and_then(|frame| frame.registers.get(usize::from(register)))
        .and_then(|entry| match entry {
            RegisterEntry::Value(value) if root.value_type(*value) == Some(ty) => Some(*value),
            _ => None,
        });
    if let TraceTerminator::LoopBack { args } = &mut root.terminator {
        args.push(written.unwrap_or(param));
    }
    replace_root_entry(
        &mut root.snapshots,
        usize::from(register),
        RegisterEntry::Unchanged,
        RegisterEntry::Value(param),
    );
    if !root.written_registers.contains(&(FrameId(0), register)) {
        root.written_registers.push((FrameId(0), register));
    }
    Ok(())
}

/// Checks that `side` (source index `index`), which rejoins the loop of the
/// loop side `target` ([`TraceTerminator::LinkLoop`]) through its snapshot
/// `link`, can bind: the target is an ancestor loop side, the final state
/// names a value of the right type for each of its loop parameters, and no
/// register the side or the sides between it and the target wrote is one the
/// loop side reads at its entry without carrying it. Such a register would
/// keep the value the loop side entered with, so the plan refuses the link
/// (the recorder ends such a side in front of the loop instead). Carrying is
/// not extended to loop sides.
fn link_loop_binds(
    members: &[TreeMember],
    side: &TraceSide,
    index: usize,
    target: usize,
    link: super::ir::SnapshotId,
) -> Result<(), TreeError> {
    let refuse = |register: u16| TreeError::LinkValue {
        side: index,
        register,
    };
    let Some(looping) = members.get(target + 1) else {
        return Err(refuse(u16::MAX));
    };
    if !matches!(looping.ir.terminator, TraceTerminator::LoopBack { .. })
        || looping
            .parent
            .is_none_or(|(_, guard)| guard != TERMINAL_GUARD)
    {
        return Err(refuse(u16::MAX));
    }
    let snapshot = &side.ir.snapshots[link.index()];
    if !snapshot.numbers.is_empty() || snapshot.frames.len() != 1 {
        return Err(refuse(u16::MAX));
    }
    let frame = &snapshot.frames[0];
    for param in &looping.ir.loop_params {
        let register = param.register.ok_or(refuse(u16::MAX))?;
        let ty = frame
            .registers
            .get(usize::from(register))
            .and_then(|entry| entry_type(&side.ir, entry));
        if ty != Some(param.ty) || param.ty.is_borrowed() {
            return Err(refuse(register));
        }
    }
    // The writes of the side and of its ancestors below the target, with
    // what each guard on the way commits for the side it starts
    // ([`guard_writes`]). Of the target's guard only what the snapshot alone
    // holds counts: a register the loop side writes itself is a parameter
    // or one it sets before it reads it.
    let mut written: Vec<(FrameId, u16)> = side.ir.written_registers.clone();
    let (mut origin, mut seeded) = (side.guard, side.ir.seed_frames);
    let mut at = side.parent;
    loop {
        match at {
            Some(ancestor) if ancestor == target => {
                let mut committed = Vec::new();
                guard_writes(&looping.ir, origin, seeded, &mut committed);
                written.extend(
                    committed
                        .into_iter()
                        .filter(|slot| !looping.ir.written_registers.contains(slot)),
                );
                break;
            }
            Some(ancestor) => {
                let member = members.get(ancestor + 1).ok_or(refuse(u16::MAX))?;
                guard_writes(&member.ir, origin, seeded, &mut written);
                written.extend(member.ir.written_registers.iter().copied());
                let (parent, guard) = member.parent.ok_or(refuse(u16::MAX))?;
                (origin, seeded) = (guard, member.ir.seed_frames);
                at = parent.checked_sub(1);
            }
            // The target is no ancestor.
            None => return Err(refuse(u16::MAX)),
        }
    }
    for (frame, register) in written {
        if frame != FrameId(0) {
            continue;
        }
        let read = looping
            .ir
            .entry
            .inputs
            .iter()
            .any(|input| input.register == register);
        let carried = looping
            .ir
            .loop_params
            .iter()
            .any(|param| param.register == Some(register));
        if read && !carried {
            return Err(refuse(register));
        }
    }
    Ok(())
}

/// Whether a side compiled apart that links back to the root's loop may write
/// `register` of the root frame's window (a guest register, a slot of the
/// root's `window_len` slots, or a slot the side names beyond them) without
/// the root carrying it: a guest register, or a slot of the caller the side
/// returns into. A cell or a global of the root's window is shared with the
/// rest of the program, so a side that writes one changes the root.
fn apart_writable(register: u16, guest: usize, window_len: usize, side: &TraceIr) -> bool {
    let register = usize::from(register);
    if register < guest {
        return true;
    }
    let slot = register - guest;
    slot >= window_len
        && matches!(
            side.external.get(slot),
            Some(ExternalSlot::CallerRegister(_))
        )
}

/// A register a side compiled apart gives the run when it links back to the
/// root's loop, beyond the root's loop parameters (see [`apart_link_writes`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkWrite {
    /// The register of the root frame's window the side writes, which is a
    /// guest register or, beyond the root's window, a slot of the side's own.
    pub register: u16,
    /// The type of the value the side links with.
    pub ty: TraceType,
    /// The register of the caller frame the slot stands for, when it is one.
    pub caller: Option<u16>,
}

/// Whether `root` takes `register` as an entry input of type `Any`: a
/// register that holds nil on some entries and a number on others, which
/// only type guards read (see `Recorder::with_lazy`).
fn takes_untyped(root: &TraceIr, register: u16) -> bool {
    root.entry
        .inputs
        .iter()
        .any(|input| input.register == register && input.ty == TraceType::Any)
}

/// The entry inputs of type `Any` of `members[0]`, the root of a planned
/// tree, that some side of it writes and hands to the root's loop at its
/// [`TraceTerminator::Link`]: as positions among the root's entry inputs,
/// with their registers. Native code stores the link's value into the
/// input's slot, where the root's type guard reads it, and the run leaves
/// those slots for the VM to write into the frame before it restores an
/// exit, whose snapshot keeps the register `Unchanged` (see
/// [`plan_tree_with`]).
#[must_use]
pub fn untyped_link_writes<M: std::borrow::Borrow<TraceIr>>(members: &[M]) -> Vec<(usize, u16)> {
    let Some(root) = members.first().map(M::borrow) else {
        return Vec::new();
    };
    root.entry
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| input.ty == TraceType::Any)
        .filter(|(_, input)| {
            members[1..].iter().any(|side| {
                let side = side.borrow();
                let TraceTerminator::Link { snapshot } = side.terminator else {
                    return false;
                };
                side.snapshots
                    .get(snapshot.index())
                    .and_then(|snapshot| snapshot.frames.first())
                    .and_then(|frame| frame.registers.get(usize::from(input.register)))
                    .is_some_and(|entry| {
                        entry_type(side, entry).is_some_and(|ty| ty != TraceType::Any)
                    })
            })
        })
        .map(|(position, input)| (position, input.register))
        .collect()
}

/// The registers side `source` of `tree`, compiled apart, writes that are live
/// where it links back and that the root does not carry as loop parameters:
/// what the run takes from the side's link and writes into the interpreter's
/// frames when it leaves (see [`plan_tree_with`], which leaves the root as it
/// is for them).
///
/// # Errors
///
/// [`TreeError::LinkValue`] for a register the side cannot hand over without
/// the root carrying it, or a value with no type.
pub fn apart_link_writes(tree: &TraceIr, source: usize) -> Result<Vec<LinkWrite>, TreeError> {
    let root = tree;
    let side = tree
        .sides
        .get(source)
        .ok_or(TreeError::NoSuchGuard { side: source })?;
    let TraceTerminator::Link { snapshot } = side.ir.terminator else {
        return Ok(Vec::new());
    };
    let guest = usize::from(root.frames[0].register_count).saturating_sub(root.external.len());
    let live =
        |register: u16| usize::from(register) >= guest || root.header_live.contains(&register);
    let params: Vec<u16> = root
        .loop_params
        .iter()
        .filter_map(|param| param.register)
        .collect();
    let link = &side.ir.snapshots[snapshot.index()].frames[0];
    let mut written: Vec<(FrameId, u16)> = side.ir.written_registers.clone();
    let mut child = side;
    let mut at = side.parent;
    loop {
        // So does what each guard on the way commits for the side it starts
        // ([`guard_writes`]), the root's included.
        let parent_ir = at.map_or(root, |parent| &*tree.sides[parent].ir);
        guard_writes(parent_ir, child.guard, child.ir.seed_frames, &mut written);
        let Some(parent) = at else {
            break;
        };
        // Every ancestor counts: one that links hands the run its writes at
        // its own link, which a side below it does not pass.
        written.extend(tree.sides[parent].ir.written_registers.iter().copied());
        child = &tree.sides[parent];
        at = child.parent;
    }
    let mut writes: Vec<LinkWrite> = Vec::new();
    for &(frame, register) in &written {
        if frame != FrameId(0)
            || !live(register)
            || params.contains(&register)
            || writes.iter().any(|known| known.register == register)
        {
            continue;
        }
        // A register the root takes untyped goes to the root's own slot at an
        // in-tree link only (see [`untyped_link_writes`]).
        if !apart_writable(register, guest, root.external.len(), &side.ir)
            || takes_untyped(root, register)
        {
            return Err(TreeError::LinkValue {
                side: source,
                register,
            });
        }
        let ty = link
            .registers
            .get(usize::from(register))
            .and_then(|entry| entry_type(&side.ir, entry))
            .ok_or(TreeError::LinkValue {
                side: source,
                register,
            })?;
        let caller = match usize::from(register)
            .checked_sub(guest)
            .and_then(|slot| side.ir.external.get(slot))
        {
            Some(ExternalSlot::CallerRegister(caller)) => Some(*caller),
            _ => None,
        };
        writes.push(LinkWrite {
            register,
            ty,
            caller,
        });
    }
    Ok(writes)
}

/// Prepares `ir` (a recorded root trace, not yet optimized) and its sides to
/// be compiled together. Member 0 is the widened root, followed by the sides
/// in order with their parents as member indexes.
///
/// # Errors
///
/// A side that cannot link: the first problem found, which names the side.
/// Callers that accept sides one at a time (the VM) check each candidate with
/// this before adding it.
pub fn plan_tree(ir: &TraceIr) -> Result<Vec<TreeMember>, TreeError> {
    plan_tree_with(ir, &[], &[])
}

/// [`plan_tree`] for a tree some of whose sides are compiled apart from the
/// root, each as a function of its own that the parent's exit calls (`apart`,
/// as indexes into [`TraceIr::sides`]), and the root is already compiled.
///
/// A side compiled apart takes the registers it reads and the parent leaves
/// alone from the root's entry state or from the run's extra inputs, so it
/// gives the root no entry input. The slots it names in its window beyond the
/// tree's (the registers of the caller it returns into) and the callables it
/// pins beyond the root's stay with the side too: they widen neither the root
/// nor the other members, and the side reads them where the run keeps them. If
/// it links back to the root's loop, the registers it writes are the root's
/// loop parameters, which the link hands the run, and the others, which the
/// run stores in the entry input or extra input slots (see
/// [`apart_link_writes`]); none of them is carried by the root. Those listed in
/// `detached` (a subset of `apart`) are planned as sides that
/// leave through their poll exit instead of linking, which resumes at the
/// loop header in the interpreter with the state their link names. They
/// widen nothing and their link needs no typed value for each loop
/// parameter. A side that does not link is planned as in [`plan_tree`].
///
/// # Errors
///
/// As [`plan_tree`].
pub fn plan_tree_with(
    ir: &TraceIr,
    detached: &[usize],
    apart: &[usize],
) -> Result<Vec<TreeMember>, TreeError> {
    let _icount = crate::jit::icount::scope("plan-tree");
    if !ir.preheader.is_empty() {
        return Err(TreeError::NotSource);
    }
    let mut root = ir.clone();
    let mut sides = std::mem::take(&mut root.sides);
    // A function's result is the one type every member that returns or
    // calls itself gives; a member that only tail calls takes it.
    if let Some(function) = &mut root.function {
        let mut result = function.result;
        for side in &sides {
            match (
                result,
                side.ir.function.as_ref().and_then(|side| side.result),
            ) {
                (Some(known), Some(other)) if known != other => {
                    return Err(TreeError::ResultConflict);
                }
                (None, other) => result = other,
                _ => {}
            }
        }
        function.result = result;
        for side in &mut sides {
            if let Some(function) = &mut side.ir.function {
                function.result = result;
            }
        }
    }
    if sides.is_empty() {
        return Ok(vec![TreeMember {
            ir: root,
            parent: None,
            demoted: 0,
        }]);
    }
    // A virtual table exists only in the exit that materializes it. A guard
    // of a loop trace that resumes in the root frame can have its exit's
    // tables built in the run on its way to the side
    // ([`Snapshot::bridges_virtuals`]), which then takes each as the table
    // its register holds. Any other side would start from state a native
    // jump skips.
    for (index, side) in sides.iter().enumerate() {
        let parent = side.parent.map_or(&root, |parent| &*sides[parent].ir);
        // Any frame: a seeded frame's register is a frame input.
        if guard_snapshot(parent, side.guard, side.ir.seed_frames).is_some_and(|snapshot| {
            snapshot.has_virtual_entry()
                && (root.function.is_some()
                    || side.ir.seed_frames > 0
                    || !snapshot.bridges_virtuals()
                    || !takes_virtuals(&side.ir, snapshot))
        }) {
            return Err(TreeError::VirtualCarried { side: index });
        }
    }
    let guest = usize::from(root.frames[0].register_count).saturating_sub(root.external.len());

    // The window. A side may name cells and globals the root does not, each
    // side extending the list its recording started from. The tree's window is
    // the longest of them, which every member is given.
    //
    // A side compiled apart keeps the slots it names beyond that window to
    // itself: they widen neither the root nor the other sides, and the side
    // reads them where the run keeps them (the caller's registers it returns
    // into, see `ExternalSlot::CallerRegister`).
    let mut window = root.external.clone();
    for (index, side) in sides.iter().enumerate() {
        let shared = window.len().min(side.ir.external.len());
        if window[..shared] != side.ir.external[..shared] {
            return Err(TreeError::WindowMismatch { side: index });
        }
        if side.ir.external.len() > window.len() && !apart.contains(&index) {
            window.clone_from(&side.ir.external);
        }
    }
    let register_of = |slot: usize| u16::try_from(guest + slot).expect("window fits u16");
    let mut slot_types: Vec<Option<TraceType>> = vec![None; window.len()];
    for (slot, known) in slot_types.iter_mut().enumerate() {
        let register = register_of(slot);
        for member in std::iter::once(&root).chain(sides.iter().map(|side| &*side.ir)) {
            let Some(input) = member
                .entry
                .inputs
                .iter()
                .find(|input| input.register == register)
            else {
                continue;
            };
            match known {
                Some(held) if *held != input.ty => {
                    return Err(TreeError::TypeConflict { register });
                }
                _ => *known = Some(input.ty),
            }
        }
    }
    extend_window(&mut root, &window, &slot_types, false)?;
    for (index, side) in sides.iter_mut().enumerate() {
        // The slots beyond the tree's window are the side's own.
        if apart.contains(&index) && side.ir.external.len() > window.len() {
            continue;
        }
        extend_window(&mut side.ir, &window, &slot_types, true)?;
    }
    share_pins(&mut root, &mut sides, apart)?;
    // Fields first: a demoted binding adds the address word to the shared
    // field list.
    share_fields(&mut root, &mut sides)?;
    let (demoted, guarded) = share_bindings(&mut root, &mut sides)?;
    // A cell or global is visible to the rest of the program, so it counts as
    // live everywhere.
    let live = |register: u16| usize::from(register) >= guest || ir.header_live.contains(&register);
    let params: Vec<u16> = root
        .loop_params
        .iter()
        .filter_map(|param| param.register)
        .collect();

    retype_passed_inputs(&root, &mut sides, apart);

    // The registers the tree takes untyped: the root's inputs of type `Any`,
    // and those a side in the tree takes so where the root reads nothing (a
    // root recorded with `lx` nil never reads `ly` behind `lx and ly`).
    let untyped: Vec<u16> = root
        .entry
        .inputs
        .iter()
        .chain(
            sides
                .iter()
                .enumerate()
                .filter(|(index, _)| !apart.contains(index))
                .flat_map(|(_, side)| side.ir.entry.inputs.iter()),
        )
        .filter(|input| input.ty == TraceType::Any)
        .map(|input| input.register)
        .collect();
    let mut untyped_links: Vec<u16> = Vec::new();
    // The registers a linking side writes and the loop still needs.
    let mut carry: Vec<(u16, TraceType)> = Vec::new();
    // The registers a linking side hands over as it started with them (side,
    // register, type).
    let mut unchanged_links: Vec<(usize, u16, TraceType)> = Vec::new();
    for (index, side) in sides.iter().enumerate() {
        let TraceTerminator::Link { snapshot } = side.ir.terminator else {
            continue;
        };
        if detached.contains(&index) {
            continue;
        }
        let link = &side.ir.snapshots[snapshot.index()].frames[0];
        // The registers its ancestors wrote before it took over count too: a
        // loop side or a side that ends in an exit never links itself, but
        // the side below it brings its registers to the header. So do the
        // writes each guard on the way commits for the side it starts
        // ([`guard_writes`]), the root's included: no link of an
        // ancestor carries those.
        let mut written: Vec<(FrameId, u16)> = side.ir.written_registers.clone();
        let mut child = side;
        let mut at = side.parent;
        loop {
            let parent_ir = at.map_or(&root, |parent| &*sides[parent].ir);
            guard_writes(parent_ir, child.guard, child.ir.seed_frames, &mut written);
            let Some(parent) = at else {
                break;
            };
            let ancestor = &sides[parent];
            // A side compiled apart that links hands the run its own writes at
            // its link, which the side below it does not pass: the registers
            // the ancestor wrote before the guard count for the side that
            // links in its place.
            if !matches!(ancestor.ir.terminator, TraceTerminator::Link { .. })
                || apart.contains(&parent)
            {
                written.extend(ancestor.ir.written_registers.iter().copied());
            }
            child = ancestor;
            at = ancestor.parent;
        }
        for &(frame, register) in &written {
            // A register whose bindings a link guard keeps holds the entry
            // table at every link, so the root goes on reading its input.
            if frame != FrameId(0)
                || !live(register)
                || params.contains(&register)
                || guarded.contains(&register)
            {
                continue;
            }
            // A side compiled apart hands the registers it writes beyond the
            // root's loop parameters to the run (see [`apart_link_writes`]),
            // which writes them into the entry inputs, so the root stays as
            // it is.
            let beyond_params = apart.contains(&index);
            if beyond_params && !apart_writable(register, guest, window.len(), &side.ir) {
                return Err(TreeError::LinkValue {
                    side: index,
                    register,
                });
            }
            let entry = link.registers.get(usize::from(register));
            let ty = match entry.and_then(|entry| entry_type(&side.ir, entry)) {
                Some(ty) => ty,
                // A register the side never touches, which a guard on the
                // way committed: the side links with the value it started
                // with, so it takes that value as an input.
                None if !beyond_params && entry == Some(&RegisterEntry::Unchanged) => {
                    let ty = started_type(&root, &sides, index, register).ok_or(
                        TreeError::LinkValue {
                            side: index,
                            register,
                        },
                    )?;
                    if !unchanged_links.contains(&(index, register, ty)) {
                        unchanged_links.push((index, register, ty));
                    }
                    ty
                }
                None => {
                    return Err(TreeError::LinkValue {
                        side: index,
                        register,
                    });
                }
            };
            // A borrow cannot become a root input or loop parameter.
            if ty.is_borrowed() {
                return Err(TreeError::LinkValue {
                    side: index,
                    register,
                });
            }
            // A register the root takes untyped (nil or a number, see
            // `Recorder::with_lazy`) stays an entry input of type `Any`: the
            // link writes its value into the input's slot, which the root's
            // type guard reads at the header, and a run that leaves writes
            // the slot back into the frame (see [`untyped_link_writes`]). A
            // side compiled apart has no such store, and a value without a
            // tag of its own (a table, a string) has no slot form.
            if untyped.contains(&register) {
                if beyond_params
                    || !matches!(
                        ty,
                        TraceType::Nil | TraceType::Float | TraceType::Integer | TraceType::Boolean
                    )
                {
                    return Err(TreeError::LinkValue {
                        side: index,
                        register,
                    });
                }
                if !untyped_links.contains(&register) {
                    untyped_links.push(register);
                }
                continue;
            }
            if beyond_params {
                continue;
            }
            match carry.iter().find(|(known, _)| *known == register) {
                Some((_, known)) if *known != ty => {
                    return Err(TreeError::TypeConflict { register });
                }
                Some(_) => {}
                None => carry.push((register, ty)),
            }
        }
    }
    for &(index, register, ty) in &unchanged_links {
        let side = &mut sides[index].ir;
        let value = ensure_input(side, register, ty)?;
        let TraceTerminator::Link { snapshot } = side.terminator else {
            continue;
        };
        side.snapshots[snapshot.index()].frames[0]
            .registers
            .set(usize::from(register), RegisterEntry::Value(value));
    }
    carry_registers(&mut root, &carry)?;
    // The root reads the slot of each such register at its header, through
    // a type guard when it reads the register at all.
    for &register in &untyped_links {
        ensure_input(&mut root, register, TraceType::Any)?;
    }

    // Registers a side reads that its parent left as `Unchanged` come from
    // the parent's own entry state: an invariant input. A side parent takes
    // the input too, from its own parent in turn. Children go first, so an
    // input a side takes on is seen by the sides before it.
    for index in (0..sides.len()).rev() {
        let inputs = sides[index].ir.entry.inputs.clone();
        let guard = sides[index].guard;
        let seeded = sides[index].ir.seed_frames;
        let parent = sides[index].parent;
        if apart.contains(&index) {
            // Its code reads them where the run keeps them.
            continue;
        }
        let snapshot = match parent {
            None => guard_snapshot(&root, guard, seeded),
            Some(parent) => guard_snapshot(&sides[parent].ir, guard, seeded),
        }
        .ok_or(TreeError::NoSuchGuard { side: index })?
        .clone();
        for input in &inputs {
            let entry = snapshot.frames[0]
                .registers
                .get(usize::from(input.register));
            if entry == Some(&RegisterEntry::Unchanged) {
                match parent {
                    None => ensure_input(&mut root, input.register, input.ty)?,
                    Some(parent) => ensure_input(&mut sides[parent].ir, input.register, input.ty)?,
                };
            }
        }
    }

    // Every link has to bind: each side input from the parent's snapshot, and
    // the final state to each root loop parameter.
    let mut members = vec![TreeMember {
        ir: root,
        parent: None,
        demoted,
    }];
    for (index, side) in sides.into_iter().enumerate() {
        let parent_member = side.parent.map_or(0, |parent| parent + 1);
        let parent = &members[parent_member].ir;
        let snapshot = guard_snapshot(parent, side.guard, side.ir.seed_frames)
            .ok_or(TreeError::NoSuchGuard { side: index })?;
        // The registers of the calls the side starts inside: the frames do
        // not exist in the interpreter, so each must be a typed value of
        // the parent's state there (a register of run-time subtype is only
        // ever in the root frame, see `Snapshot::numbers`).
        for input in &side.ir.frame_inputs {
            let bound = snapshot
                .frames
                .get(input.frame.index())
                .filter(|_| input.frame.index() > 0)
                .and_then(|frame| frame.registers.get(usize::from(input.register)))
                .and_then(|entry| entry_type(parent, entry))
                .is_some_and(|ty| ty == input.ty && !ty.is_borrowed());
            if !bound {
                return Err(TreeError::Unbound {
                    side: index,
                    register: input.register,
                });
            }
        }
        for input in &side.ir.entry.inputs {
            let register = input.register;
            let bound = match snapshot.frames[0].registers.get(usize::from(register)) {
                Some(RegisterEntry::Unchanged) => {
                    match parent
                        .entry
                        .inputs
                        .iter()
                        .find(|held| held.register == register)
                    {
                        Some(held) => held.ty == input.ty,
                        // A side compiled apart reads it where the run
                        // keeps it.
                        None => apart.contains(&index),
                    }
                }
                // The table the guard builds on its way to the side.
                Some(RegisterEntry::Virtual(_)) => input.ty == TraceType::Table,
                Some(entry) => match snapshot.number_tag(usize::from(register)) {
                    // A side recorded for one subtype starts behind a check
                    // of it.
                    Some(_) => {
                        entry_type(parent, entry) == Some(TraceType::Float)
                            && matches!(input.ty, TraceType::Integer | TraceType::Float)
                    }
                    None => entry_type(parent, entry) == Some(input.ty),
                },
                // A slot of the side's own window, beyond the parent's: a side
                // compiled apart reads it where the run keeps it.
                None => apart.contains(&index) && usize::from(register) >= guest,
            };
            if !bound {
                return Err(TreeError::Unbound {
                    side: index,
                    register,
                });
            }
        }
        if let TraceTerminator::Link { snapshot: link } = side.ir.terminator
            && !detached.contains(&index)
        {
            if !side.ir.snapshots[link.index()].numbers.is_empty() {
                return Err(TreeError::LinkValue {
                    side: index,
                    register: u16::MAX,
                });
            }
            let link = &side.ir.snapshots[link.index()].frames[0];
            for param in &members[0].ir.loop_params {
                let register = param.register.ok_or(TreeError::LinkValue {
                    side: index,
                    register: u16::MAX,
                })?;
                let ty = link
                    .registers
                    .get(usize::from(register))
                    .and_then(|entry| entry_type(&side.ir, entry));
                if ty != Some(param.ty) {
                    return Err(TreeError::LinkValue {
                        side: index,
                        register,
                    });
                }
            }
        }
        if let TraceTerminator::LinkLoop {
            side: target,
            snapshot: link,
        } = side.ir.terminator
        {
            link_loop_binds(&members, &side, index, target, link)?;
        }
        members.push(TreeMember {
            ir: *side.ir,
            parent: Some((parent_member, side.guard)),
            demoted: 0,
        });
    }
    Ok(members)
}
