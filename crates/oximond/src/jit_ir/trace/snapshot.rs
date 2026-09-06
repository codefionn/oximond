//! Recovery snapshots: the exact guest state a guard restores on failure.

use super::ir::{FrameId, SnapshotId, TraceScalar, ValueId};
use super::register_map::RegisterMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The most string keys a shaped virtual table names. It equals the runtime's
/// `MAX_SHAPE_KEYS`; a VM test pins the two together.
pub const MAX_VIRTUAL_SHAPE_KEYS: usize = 8;

/// Names a virtual table of one snapshot: an index into
/// [`Snapshot::virtuals`]. Ids are local to their snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct VirtualId(pub u32);

impl VirtualId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Names a slice recipe of one snapshot: an index into [`Snapshot::slices`].
/// Ids are local to their snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SliceId(pub u32);

impl SliceId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Which side of a guest instruction a snapshot describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SnapshotPhase {
    /// Immediately before the instruction at `pc`: resume by executing it.
    Before { pc: usize },
    /// Immediately after the instruction at `pc` committed exactly once:
    /// resume at `successor`. A branch that changes no registers uses its
    /// post-state with the alternative successor.
    After { pc: usize, successor: usize },
}

/// What one register holds in the restored frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RegisterEntry {
    /// A typed SSA value.
    Value(ValueId),
    Const(TraceScalar),
    /// The interpreter register still holds the right value. Legal only for a
    /// register the trace never writes.
    Unchanged,
    /// An opaque value in the trace root bank.
    Root(u32),
    /// A table the trace never allocated. The exit builds it from the
    /// snapshot's recipe [`Snapshot::virtuals`]`[id]`. Registers that name one
    /// id hold the same table. Legal only in a register of a frame (not in a
    /// cell or global slot of the window), and in the array or fields of
    /// another recipe, where it is a table built inside a table.
    Virtual(VirtualId),
    /// A string the trace never made: `string.sub (source, first, last)`,
    /// which the exit makes from the snapshot's recipe
    /// [`Snapshot::slices`]`[id]`. Legal only in a register of a frame, not in
    /// a cell or global slot of the window and not in a recipe.
    Slice(SliceId),
}

// Written out so that it is inlined: register maps are compared entry by
// entry wherever snapshots share their runs.
impl PartialEq for RegisterEntry {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unchanged, Self::Unchanged) => true,
            (Self::Value(ours), Self::Value(theirs)) => ours == theirs,
            (Self::Const(ours), Self::Const(theirs)) => ours == theirs,
            (Self::Root(ours), Self::Root(theirs)) => ours == theirs,
            (Self::Virtual(ours), Self::Virtual(theirs)) => ours == theirs,
            (Self::Slice(ours), Self::Slice(theirs)) => ours == theirs,
            _ => false,
        }
    }
}

/// The recipe of a substring an exit makes: `string.sub (string, first,
/// last)`, which the trace skipped because only a recovery path reads it.
/// The entries are `Value` or `Const`: `string` a string, `first` and `last`
/// numbers of the profile. The trace checked that both indices are numbers
/// strictly between -2^31 and 2^31 (so every profile converts them alike, by
/// truncation toward zero).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceRecipe {
    pub string: RegisterEntry,
    pub first: RegisterEntry,
    pub last: RegisterEntry,
}

impl SliceRecipe {
    /// The entries of the recipe.
    pub fn entries(&self) -> impl Iterator<Item = &RegisterEntry> {
        [&self.string, &self.first, &self.last].into_iter()
    }
}

/// How an exit allocates a virtual table, so that the restored table has the
/// storage the interpreter's own constructor would have made.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum VirtualAlloc {
    /// An empty table, as `NewTable` makes at a site with no shape.
    Plain,
    /// A table whose string keys are `keys` (one to
    /// [`MAX_VIRTUAL_SHAPE_KEYS`] distinct ones), all absent, as `NewTable`
    /// makes at a constructor site with a shape.
    Shaped { keys: Vec<Arc<[u8]>> },
}

/// A key of a field of a virtual table. Numbers are normalized: an integral
/// float key is an `Integer`, and a key in `1..=array.len()` is an array
/// position, not a field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VirtualKey {
    Integer(i64),
    /// A float key that is neither integral nor NaN.
    Float(f64),
    String(Arc<[u8]>),
}

/// The recipe of one virtual table: what an exit stores into it. A nil value
/// stores nothing (a hole, or a deleted field).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VirtualTable {
    pub alloc: VirtualAlloc,
    /// The values at keys 1, 2, ... Entries are `Value`, `Const`, `Root` or
    /// `Virtual` (a table of another recipe, which the exit builds before it
    /// fills any).
    pub array: Vec<RegisterEntry>,
    /// The other fields, stored in this order after the array part. Values
    /// are as for `array`.
    pub fields: Vec<(VirtualKey, RegisterEntry)>,
    /// The entry count the trace charged this table's footprint for, as
    /// phantom debt (`Runtime::charge_elided_table`), before it leaves, or
    /// `None` when it charged nothing. The count is the table's peak size,
    /// which can exceed the entries of this recipe: a table that grows after
    /// the exit's snapshot was charged for the growth too. Materializing
    /// refunds exactly that charge, so the real allocation is counted once.
    pub phantom_entries: Option<usize>,
    /// The metatable `setmetatable` gave the table, a `Value` entry of type
    /// `Table`, which an exit sets after it has stored the contents. The
    /// trace checked that the metatable has no `__gc` and no `__mode`
    /// (otherwise the call would have registered a finalizer or a weak table),
    /// and every access of the table that the metatable could change.
    pub metatable: Option<RegisterEntry>,
}

impl VirtualTable {
    /// The entries the recipe stores, in the order an exit stores them, and
    /// then the metatable it sets.
    pub fn entries(&self) -> impl Iterator<Item = &RegisterEntry> {
        self.array
            .iter()
            .chain(self.fields.iter().map(|(_, entry)| entry))
            .chain(self.metatable.iter())
    }

    /// The `charge_elided_table` field count for this table: every entry a
    /// restore may store.
    #[must_use]
    pub fn footprint_entries(&self) -> usize {
        self.array.len() + self.fields.len()
    }
}

/// One logical frame of a snapshot, with a complete register map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotFrame {
    pub frame: FrameId,
    /// The frame's `last_interpreted_pc` for a `Before` snapshot: the
    /// previously executed guest pc, as recorded. `After` snapshots ignore it
    /// for the innermost frame. Outer frames keep it as recorded.
    pub last_interpreted_pc: Option<usize>,
    /// One entry per register in the frame's window. No liveness compression:
    /// a register unused on the hot path may be live on the cold successor.
    /// Snapshots in a row share the runs of the map they do not change.
    pub registers: RegisterMap,
}

/// Where the interpreter continues after a restore.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResumePoint {
    pub pc: usize,
    pub last_interpreted_pc: Option<usize>,
}

/// A register of the root frame that holds a number of either subtype: the
/// entry names a float (the number's value, exact because the trace keeps
/// integers within 2^53 in magnitude), and `is_integer` says at run time
/// whether the restored value is the integer of that float.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumberRegister {
    pub register: u32,
    /// A boolean: `Value` of a boolean value, or a `Const` boolean.
    pub is_integer: RegisterEntry,
}

/// The recovery description a guard or exit carries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub phase: SnapshotPhase,
    /// How many trace effects have committed. A failed guard resumes with
    /// exactly these committed and never replays them.
    pub effect_position: u32,
    /// Under a count hook: how many counted guest instructions the trace has
    /// executed since the iteration began (the loop header, or for a side
    /// trace the same header, through its parent's path), when the exit
    /// restores this snapshot. The instruction at the resume point is not
    /// included. Zero in a trace without count accounting
    /// ([`super::TraceIr::counted`]).
    #[serde(default)]
    pub counted: u32,
    /// Frames from outermost to innermost. The first is the root frame.
    pub frames: Vec<SnapshotFrame>,
    /// The recipes of the virtual tables the frames' `Virtual` entries name,
    /// by [`VirtualId`]. Empty for a trace that elides no table.
    pub virtuals: Vec<VirtualTable>,
    /// The recipes of the substrings the frames' `Slice` entries name, by
    /// [`SliceId`]. Empty for a trace that elides no substring.
    #[serde(default)]
    pub slices: Vec<SliceRecipe>,
    /// The registers of the root frame (frame 0) that hold a number whose
    /// subtype is known only at run time (see [`NumberRegister`]). Their
    /// entries in `frames` are `Value` or `Const` floats.
    #[serde(default)]
    pub numbers: Vec<NumberRegister>,
}

/// Rewrites the entries of `snapshots`, a trace's in order, by `rewrite`,
/// which must make of an entry the same whatever snapshot, frame or
/// register holds it (a substitution of values). A run of a register map
/// that the snapshot before shares is rewritten once, and stays shared.
pub fn rewrite_entries(snapshots: &mut [Snapshot], mut rewrite: impl FnMut(&mut RegisterEntry)) {
    // By frame position: the map of the snapshot before, and what it became.
    let mut previous: Vec<(RegisterMap, RegisterMap)> = Vec::new();
    for snapshot in snapshots {
        for (position, frame) in snapshot.frames.iter_mut().enumerate() {
            let made = frame.registers.rewritten(
                previous.get(position).map(|(old, new)| (old, new)),
                |_, _, run| run.iter_mut().for_each(&mut rewrite),
            );
            #[cfg(debug_assertions)]
            {
                let mut whole = frame.registers.to_vec();
                whole.iter_mut().for_each(&mut rewrite);
                assert!(made == whole, "a shared run was rewritten to other entries");
            }
            let old = std::mem::replace(&mut frame.registers, made.clone());
            if position < previous.len() {
                previous[position] = (old, made);
            } else {
                previous.push((old, made));
            }
        }
        snapshot.recipe_entries_mut(&mut rewrite);
    }
}

/// In the root frame of each of `snapshots`, makes the entry of `register`
/// `to` where it is `from`.
pub(crate) fn replace_root_entry(
    snapshots: &mut [Snapshot],
    register: usize,
    from: RegisterEntry,
    to: RegisterEntry,
) {
    let mut memo = super::register_map::RunMemo::default();
    for snapshot in snapshots {
        if let Some(frame) = snapshot.frames.first_mut() {
            #[cfg(debug_assertions)]
            let whole = {
                let mut whole = frame.registers.to_vec();
                if whole.get(register) == Some(&from) {
                    whole[register] = to;
                }
                whole
            };
            frame.registers.replace(register, from, to, &mut memo);
            #[cfg(debug_assertions)]
            assert!(
                frame.registers == whole,
                "a shared run was replaced by other entries"
            );
        }
    }
}

impl Snapshot {
    /// A snapshot with the given phase and frames. The builder assigns `id`.
    #[must_use]
    pub const fn new(
        phase: SnapshotPhase,
        effect_position: u32,
        frames: Vec<SnapshotFrame>,
    ) -> Self {
        Self {
            id: SnapshotId(0),
            phase,
            effect_position,
            counted: 0,
            frames,
            virtuals: Vec::new(),
            slices: Vec::new(),
            numbers: Vec::new(),
        }
    }

    /// This snapshot with the registers `numbers` restored as a float or an
    /// integer by their `is_integer` entries.
    #[must_use]
    pub fn with_numbers(mut self, numbers: Vec<NumberRegister>) -> Self {
        self.numbers = numbers;
        self
    }

    /// The entry that says whether `register` of the root frame restores as
    /// an integer, if the register holds a number of run-time subtype.
    #[must_use]
    pub fn number_tag(&self, register: usize) -> Option<RegisterEntry> {
        self.numbers
            .iter()
            .find(|number| number.register as usize == register)
            .map(|number| number.is_integer)
    }

    /// This snapshot with `counted` instructions executed since the
    /// iteration began.
    #[must_use]
    pub const fn with_counted(mut self, counted: u32) -> Self {
        self.counted = counted;
        self
    }

    /// This snapshot with the recipes `virtuals`.
    #[must_use]
    pub fn with_virtuals(mut self, virtuals: Vec<VirtualTable>) -> Self {
        self.virtuals = virtuals;
        self
    }

    /// Whether any register of any frame holds a virtual table.
    #[must_use]
    pub fn has_virtual_entry(&self) -> bool {
        self.frames
            .iter()
            .flat_map(|frame| &frame.registers)
            .any(|entry| matches!(entry, RegisterEntry::Virtual(_)))
    }

    /// Whether a guard with this snapshot can build the snapshot's virtual
    /// tables in the run before it enters a side: the snapshot resumes in
    /// the root frame alone and lists no more recipes than a helper's result
    /// words name lanes.
    #[must_use]
    pub fn bridges_virtuals(&self) -> bool {
        self.frames.len() == 1 && self.virtuals.len() <= super::helper::MAX_HELPER_RESULTS
    }

    /// The place of virtual table `id` among the tables an exit builds for
    /// this snapshot: they are numbered in the order the registers of the
    /// frames first name them (the tables those hold follow). `None` for a
    /// table no register names.
    #[must_use]
    pub fn virtual_place(&self, id: VirtualId) -> Option<usize> {
        let mut seen: Vec<VirtualId> = Vec::new();
        for entry in self.frames.iter().flat_map(|frame| &frame.registers) {
            if let RegisterEntry::Virtual(named) = entry {
                if *named == id {
                    return Some(seen.len());
                }
                if !seen.contains(named) {
                    seen.push(*named);
                }
            }
        }
        None
    }

    /// Calls `visit` for every entry the snapshot restores: the registers of
    /// each frame, then the values of each recipe.
    pub fn entries_mut(&mut self, mut visit: impl FnMut(&mut RegisterEntry)) {
        for frame in &mut self.frames {
            frame.registers.update(|_, entry| {
                let mut entry = *entry;
                visit(&mut entry);
                entry
            });
        }
        self.recipe_entries_mut(visit);
    }

    /// Calls `visit` for every entry outside the frames' register maps: the
    /// values of each recipe and the subtype tags.
    fn recipe_entries_mut(&mut self, mut visit: impl FnMut(&mut RegisterEntry)) {
        for table in &mut self.virtuals {
            table.array.iter_mut().for_each(&mut visit);
            table.fields.iter_mut().for_each(|(_, entry)| visit(entry));
            table.metatable.iter_mut().for_each(&mut visit);
        }
        for slice in &mut self.slices {
            visit(&mut slice.string);
            visit(&mut slice.first);
            visit(&mut slice.last);
        }
        for number in &mut self.numbers {
            visit(&mut number.is_integer);
        }
    }

    /// Calls `visit` for every SSA value the snapshot names, in frames and in
    /// recipes. A recipe nothing refers to still counts: it keeps its values
    /// alive, which is conservative.
    pub fn for_each_value(&self, mut visit: impl FnMut(ValueId)) {
        let mut note = |entry: &RegisterEntry| {
            if let RegisterEntry::Value(value) = entry {
                visit(*value);
            }
        };
        for frame in &self.frames {
            frame.registers.iter().for_each(&mut note);
        }
        for table in &self.virtuals {
            table.entries().for_each(&mut note);
        }
        for slice in &self.slices {
            slice.entries().for_each(&mut note);
        }
        for number in &self.numbers {
            note(&number.is_integer);
        }
    }

    /// [`Self::for_each_value`] for a visitor that has seen `previous`, a
    /// snapshot of the same trace, and gains nothing from seeing a value
    /// again in the place it had there: the runs of a register map that
    /// `previous` shares at the same frame position are left out. The
    /// recipes and subtype tags are visited whole.
    pub fn for_each_value_since(&self, previous: Option<&Self>, mut visit: impl FnMut(ValueId)) {
        let mut note = |entry: &RegisterEntry| {
            if let RegisterEntry::Value(value) = entry {
                visit(*value);
            }
        };
        for (position, frame) in self.frames.iter().enumerate() {
            let before = previous
                .and_then(|previous| previous.frames.get(position))
                .map(|frame| &frame.registers);
            for (_, run) in frame.registers.changed_from(before) {
                run.iter().for_each(&mut note);
            }
        }
        for table in &self.virtuals {
            table.entries().for_each(&mut note);
        }
        for slice in &self.slices {
            slice.entries().for_each(&mut note);
        }
        for number in &self.numbers {
            note(&number.is_integer);
        }
    }

    /// The program counter and `last_interpreted_pc` of the innermost frame
    /// after restore. `Before` keeps the recorded previous pc. `After` sets
    /// it to the committed instruction's pc, which tracing backedge detection
    /// reads.
    #[must_use]
    pub fn resume_point(&self) -> ResumePoint {
        match self.phase {
            SnapshotPhase::Before { pc } => ResumePoint {
                pc,
                last_interpreted_pc: self
                    .frames
                    .last()
                    .and_then(|frame| frame.last_interpreted_pc),
            },
            SnapshotPhase::After { pc, successor } => ResumePoint {
                pc: successor,
                last_interpreted_pc: Some(pc),
            },
        }
    }

    /// The pc of the instruction the phase is about.
    #[must_use]
    pub const fn phase_pc(&self) -> usize {
        match self.phase {
            SnapshotPhase::Before { pc } | SnapshotPhase::After { pc, .. } => pc,
        }
    }
}
