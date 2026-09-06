//! Generational slot arenas backing every runtime heap object.
//!
//! Every live slot sits on exactly one age list. New objects start on the
//! young list; a minor collection moves its survivors to the survival list,
//! and survivors of a second minor collection to the old list. A major
//! collection moves every survivor to the old list. Minor collections walk
//! only the young and survival lists, so their sweep cost is proportional to
//! the nursery rather than to the whole heap.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// `repr(C)`: native code reads a handle as one word, the slot then the
// generation (see `TableHandle::native_identity`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(C)]
pub(crate) struct ArenaKey {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

/// The generation an object belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Age {
    /// Allocated since the last collection.
    Young,
    /// Survived one minor collection.
    Survival,
    /// Survived two minor collections or a major collection. Minor
    /// collections treat old objects as reachable and do not traverse them
    /// unless a barrier or a recent promotion put them on the remembered set.
    Old,
}

pub(crate) struct Arena<T> {
    pub(crate) slots: Vec<ArenaSlot<T>>,
    pub(crate) young: Vec<u32>,
    pub(crate) survival: Vec<u32>,
    pub(crate) old: Vec<u32>,
    free: Vec<u32>,
    /// Mark epoch given to new objects. The collector sets it to the active
    /// cycle's epoch, so objects allocated during a cycle are born marked.
    pub(crate) alloc_epoch: u64,
    /// The first slot index never handed out: the end of the handle space,
    /// unless a test brings it closer ([`Arena::limit_growth`]).
    slot_limit: u32,
}

pub(crate) struct ArenaSlot<T> {
    pub(crate) generation: u32,
    pub(crate) age: Age,
    pub(crate) retired: bool,
    /// Set while the object is on the collector's remembered set.
    pub(crate) touched: bool,
    /// The last collection that proved this slot reachable. An epoch rather
    /// than a bit lets a new cycle start without clearing every slot. It is
    /// atomic so tracing can mark children while borrowing their parent, and
    /// so several threads can mark at once. Marking is a relaxed load and
    /// store, not a read-modify-write: two threads that mark one object at
    /// the same moment both trace it, which is harmless.
    pub(crate) marked_epoch: AtomicU64,
    /// Whether `value` holds an object.
    occupied: bool,
    /// The object, when `occupied`. A bare `MaybeUninit` rather than an
    /// `Option`, so a new object's fields can be written straight into the
    /// slot without building the object on the stack first (see
    /// [`Arena::insert_init`]).
    value: MaybeUninit<T>,
}

#[allow(unsafe_code)]
impl<T> ArenaSlot<T> {
    #[inline]
    fn get(&self) -> Option<&T> {
        // SAFETY: `occupied` says `value` is initialized.
        self.occupied
            .then(|| unsafe { self.value.assume_init_ref() })
    }

    #[inline]
    fn get_mut(&mut self) -> Option<&mut T> {
        // SAFETY: as in `get`.
        self.occupied
            .then(|| unsafe { self.value.assume_init_mut() })
    }

    /// Drops the object in place and marks the slot empty.
    #[inline]
    fn clear(&mut self) {
        if self.occupied {
            self.occupied = false;
            // SAFETY: the slot held an initialized object, now forgotten.
            unsafe { self.value.assume_init_drop() };
        }
    }
}

impl<T> Drop for ArenaSlot<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Where a major sweep stands within one arena.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MajorSweepCursor {
    stage: u8,
    index: usize,
}

/// What one step of a major sweep did.
pub(crate) enum SweepStep {
    Kept,
    Freed,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            young: Vec::new(),
            survival: Vec::new(),
            old: Vec::new(),
            free: Vec::new(),
            alloc_epoch: 0,
            slot_limit: u32::MAX - 1,
        }
    }
}

impl<T> Arena<T> {
    #[inline]
    pub(crate) fn insert(&mut self, value: T) -> Option<ArenaKey> {
        self.insert_init(move |slot| {
            slot.write(value);
        })
    }

    /// Finds a slot for a new object and lets `init` initialize it there.
    /// `init` must write the whole object. Table creation writes the fields
    /// one by one: an object built elsewhere and copied in is stored with
    /// narrow writes and read back with wide ones, which cannot forward.
    #[allow(clippy::inline_always, reason = "see `insert_in_place`")]
    #[inline(always)]
    pub(crate) fn insert_init(
        &mut self,
        init: impl FnOnce(&mut MaybeUninit<T>),
    ) -> Option<ArenaKey> {
        let epoch = self.alloc_epoch;
        // `init` appears once, so it is inlined once.
        let slot = loop {
            let Some(slot) = self.free.pop() else {
                break self.push_empty_slot(epoch)?;
            };
            if let Some(entry) = self.slots.get(slot as usize)
                && !entry.retired
                && !entry.occupied
            {
                break slot;
            }
        };
        // The next allocation takes the slot on top of the free list; start
        // loading it now.
        if let Some(&next) = self.free.last() {
            crate::runtime::prefetch::soon(self.slots.as_ptr().wrapping_add(next as usize));
        }
        let entry = &mut self.slots[slot as usize];
        init(&mut entry.value);
        entry.occupied = true;
        entry.age = Age::Young;
        entry.touched = false;
        *entry.marked_epoch.get_mut() = epoch;
        self.young.push(slot);
        Some(ArenaKey {
            slot,
            generation: entry.generation,
        })
    }

    /// Makes room for `additional` more slots, so that the objects the next
    /// allocations make do not move the ones that exist.
    pub(crate) fn reserve_slots(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Lets the arena grow by at most `additional` more slots, as if its
    /// handle space ran out there. Slots a collection freed are still reused.
    pub(crate) fn limit_growth(&mut self, additional: u32) {
        let len = u32::try_from(self.slots.len()).unwrap_or(u32::MAX);
        self.slot_limit = self.slot_limit.min(len.saturating_add(additional));
    }

    /// Whether another slot fits without moving the slots: native code holds
    /// the address of an object, which a full `Vec` would move. Only the
    /// room past the last slot counts, so a free slot is no promise.
    pub(crate) fn has_spare_slot(&self) -> bool {
        self.slots.len() < self.slots.capacity()
    }

    /// [`Self::has_spare_slot`] for `count` more slots.
    pub(crate) fn has_spare_slots(&self, count: usize) -> bool {
        self.slots.capacity() - self.slots.len() >= count
    }

    /// Appends an empty slot, for `insert_init` to fill.
    #[cold]
    #[inline(never)]
    fn push_empty_slot(&mut self, epoch: u64) -> Option<u32> {
        // The last two slots are never handed out: native identities use
        // them to mark a native function or a builtin (see
        // `crate::jit_ir::trace::pinned_native_identity`).
        let slot = u32::try_from(self.slots.len())
            .ok()
            .filter(|slot| *slot < self.slot_limit)?;
        self.slots.push(ArenaSlot {
            generation: 0,
            age: Age::Young,
            retired: false,
            touched: false,
            marked_epoch: AtomicU64::new(epoch),
            occupied: false,
            value: MaybeUninit::uninit(),
        });
        Some(slot)
    }

    #[inline]
    fn entry(&self, key: ArenaKey) -> Option<&ArenaSlot<T>> {
        self.slots
            .get(key.slot as usize)
            .filter(|entry| entry.generation == key.generation && entry.occupied)
    }

    #[inline]
    pub(crate) fn get(&self, key: ArenaKey) -> Option<&T> {
        let entry = self.slots.get(key.slot as usize)?;
        (entry.generation == key.generation)
            .then(|| entry.get())
            .flatten()
    }

    /// [`Self::get`] for a key known to be live.
    ///
    /// # Safety
    ///
    /// `key` must name a live object of this arena: its slot exists, holds
    /// a value and has the key's generation.
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, key: ArenaKey) -> &T {
        debug_assert!(self.get(key).is_some());
        // SAFETY: the caller promises the slot exists and holds a value.
        unsafe {
            self.slots
                .get_unchecked(key.slot as usize)
                .value
                .assume_init_ref()
        }
    }

    /// [`Self::get_mut`] for a key known to be live; see
    /// [`Self::get_unchecked`].
    ///
    /// # Safety
    ///
    /// As for [`Self::get_unchecked`].
    #[allow(unsafe_code, clippy::inline_always)]
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked_mut(&mut self, key: ArenaKey) -> &mut T {
        debug_assert!(self.get(key).is_some());
        // SAFETY: the caller promises the slot exists and holds a value.
        unsafe {
            self.slots
                .get_unchecked_mut(key.slot as usize)
                .value
                .assume_init_mut()
        }
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, key: ArenaKey) -> Option<&mut T> {
        let entry = self.slots.get_mut(key.slot as usize)?;
        (entry.generation == key.generation)
            .then(|| entry.get_mut())
            .flatten()
    }

    #[cfg(test)]
    pub(crate) fn remove(&mut self, key: ArenaKey) -> Option<T> {
        let entry = self.slots.get(key.slot as usize)?;
        if entry.generation != key.generation || entry.retired || !entry.occupied {
            return None;
        }
        for list in [&mut self.young, &mut self.survival, &mut self.old] {
            if let Some(index) = list.iter().position(|slot| *slot == key.slot) {
                list.swap_remove(index);
                break;
            }
        }
        Some(self.free_slot(key.slot))
    }

    /// Marks a live object. Returns whether it was unmarked before. With
    /// `minor`, old objects count as already marked.
    #[inline]
    pub(crate) fn mark(&self, key: ArenaKey, epoch: u64, minor: bool) -> bool {
        let Some(entry) = self.entry(key) else {
            return false;
        };
        if (minor && entry.age == Age::Old) || entry.marked_epoch.load(Relaxed) == epoch {
            return false;
        }
        entry.marked_epoch.store(epoch, Relaxed);
        true
    }

    /// Whether a live object is reachable in the current collection.
    #[inline]
    pub(crate) fn is_reached(&self, key: ArenaKey, epoch: u64, minor: bool) -> bool {
        self.entry(key).is_some_and(|entry| {
            (minor && entry.age == Age::Old) || entry.marked_epoch.load(Relaxed) == epoch
        })
    }

    /// Whether a live object is old. Stale handles are not.
    #[inline]
    pub(crate) fn is_old(&self, key: ArenaKey) -> bool {
        self.entry(key).is_some_and(|entry| entry.age == Age::Old)
    }

    /// Whether a live object is young or a survivor. Stale handles are not.
    #[cfg(test)]
    pub(crate) fn is_young(&self, key: ArenaKey) -> bool {
        self.entry(key).is_some_and(|entry| entry.age != Age::Old)
    }

    /// Sets the remembered flag of a live object. Returns whether it was
    /// clear before.
    #[inline]
    pub(crate) fn touch(&mut self, key: ArenaKey) -> bool {
        match self.slots.get_mut(key.slot as usize) {
            Some(entry)
                if entry.generation == key.generation && entry.occupied && !entry.touched =>
            {
                entry.touched = true;
                true
            }
            _ => false,
        }
    }

    /// A live object with whether it is old and whether it is on the
    /// remembered set, in one lookup. `None` for a stale handle.
    #[inline]
    pub(crate) fn aged(&self, key: ArenaKey) -> Option<(&T, bool, bool)> {
        let entry = self.entry(key)?;
        Some((entry.get()?, entry.age == Age::Old, entry.touched))
    }

    /// [`Self::touch`] that also lets `update` change the object of a live
    /// slot, with one lookup. Returns whether the remembered flag was clear
    /// before, or `false` for a stale handle (and then runs nothing).
    #[inline]
    pub(crate) fn touch_with(&mut self, key: ArenaKey, update: impl FnOnce(&mut T)) -> bool {
        match self.slots.get_mut(key.slot as usize) {
            Some(entry) if entry.generation == key.generation && entry.occupied => {
                let newly = !entry.touched;
                entry.touched = true;
                if let Some(value) = entry.get_mut() {
                    update(value);
                }
                newly
            }
            _ => false,
        }
    }

    #[inline]
    pub(crate) fn untouch(&mut self, key: ArenaKey) {
        if let Some(entry) = self.slots.get_mut(key.slot as usize)
            && entry.generation == key.generation
        {
            entry.touched = false;
        }
    }

    /// Takes the object out of a listed slot and frees the slot.
    #[cfg(test)]
    #[allow(unsafe_code)]
    fn free_slot(&mut self, slot: u32) -> T {
        let entry = &mut self.slots[slot as usize];
        assert!(entry.occupied, "listed slot has a value");
        entry.occupied = false;
        // SAFETY: the slot held an initialized object, now marked empty.
        let value = unsafe { entry.value.assume_init_read() };
        self.release_slot(slot);
        value
    }

    /// Drops the object of a listed slot where it lies and frees the slot.
    #[inline]
    fn free_slot_in_place(&mut self, slot: u32) {
        self.slots[slot as usize].clear();
        self.release_slot(slot);
    }

    /// Retires an emptied slot, or puts it on the free list.
    #[inline]
    fn release_slot(&mut self, slot: u32) {
        let entry = &mut self.slots[slot as usize];
        debug_assert!(!entry.occupied);
        entry.touched = false;
        if entry.generation == u32::MAX {
            entry.retired = true;
        } else {
            entry.generation += 1;
            self.free.push(slot);
        }
    }

    /// Sweeps the young and survival lists after a minor collection.
    /// Unmarked objects are lent to `freed` and then dropped where they lie,
    /// so a dead object is never copied out of its slot; `freed` may swap
    /// its contents away to keep them. Marked survivors are promoted to old
    /// and passed to `promoted`; marked young objects become survivors.
    pub(crate) fn sweep_minor(
        &mut self,
        epoch: u64,
        mut freed: impl FnMut(&mut T),
        mut promoted: impl FnMut(ArenaKey, &T),
    ) {
        let survival = std::mem::take(&mut self.survival);
        for slot in survival {
            let entry = &mut self.slots[slot as usize];
            if *entry.marked_epoch.get_mut() == epoch {
                entry.age = Age::Old;
                self.old.push(slot);
                let entry = &self.slots[slot as usize];
                if let Some(value) = entry.get() {
                    promoted(
                        ArenaKey {
                            slot,
                            generation: entry.generation,
                        },
                        value,
                    );
                }
            } else {
                self.free_dying(slot, &mut freed);
            }
        }
        let mut young = std::mem::take(&mut self.young);
        for slot in &young {
            let entry = &mut self.slots[*slot as usize];
            if *entry.marked_epoch.get_mut() == epoch {
                entry.age = Age::Survival;
                self.survival.push(*slot);
            } else {
                self.free_dying(*slot, &mut freed);
            }
        }
        // Reuse the young list's allocation for the next nursery.
        young.clear();
        self.young = young;
    }

    /// Lends a dying object to `freed`, then drops it and frees its slot.
    #[inline(always)]
    fn free_dying(&mut self, slot: u32, freed: &mut impl FnMut(&mut T)) {
        let entry = &mut self.slots[slot as usize];
        if let Some(value) = entry.get_mut() {
            freed(value);
        }
        self.free_slot_in_place(slot);
    }

    /// Sweeps at most one slot of a major collection. Objects marked with
    /// `epoch` survive and become old; `kept` sees each of them. Objects
    /// marked with `epoch + 1` were allocated after the atomic step: they
    /// survive but stay young, because their fields were never traced and
    /// may point at young objects in arenas that were already swept. Other
    /// objects are lent to `freed` and dropped where they lie. Returns
    /// `None` once every list has been swept. The young list is swept last
    /// and in place, so objects allocated during the sweep are visited too.
    #[inline(always)]
    pub(crate) fn sweep_major_one(
        &mut self,
        epoch: u64,
        cursor: &mut MajorSweepCursor,
        kept: &mut impl FnMut(&T),
        freed: &mut impl FnMut(&mut T),
    ) -> Option<SweepStep> {
        loop {
            let slot = match cursor.stage {
                0 => {
                    let Some(slot) = self.old.get(cursor.index) else {
                        cursor.stage = 1;
                        continue;
                    };
                    *slot
                }
                1 => {
                    let Some(slot) = self.survival.pop() else {
                        cursor.stage = 2;
                        cursor.index = 0;
                        continue;
                    };
                    slot
                }
                2 => *self.young.get(cursor.index)?,
                _ => return None,
            };
            let entry = &mut self.slots[slot as usize];
            entry.touched = false;
            let marked = *entry.marked_epoch.get_mut();
            if marked == epoch {
                entry.age = Age::Old;
                match cursor.stage {
                    0 => cursor.index += 1,
                    1 => self.old.push(slot),
                    _ => {
                        self.young.swap_remove(cursor.index);
                        self.old.push(slot);
                    }
                }
                if let Some(value) = self.slots[slot as usize].get() {
                    kept(value);
                }
                return Some(SweepStep::Kept);
            }
            if cursor.stage == 2 && marked == epoch.wrapping_add(1) {
                cursor.index += 1;
                return Some(SweepStep::Kept);
            }
            match cursor.stage {
                0 => {
                    self.old.swap_remove(cursor.index);
                }
                2 => {
                    self.young.swap_remove(cursor.index);
                }
                _ => {}
            }
            if let Some(value) = self.slots[slot as usize].get_mut() {
                freed(value);
            }
            self.free_slot_in_place(slot);
            return Some(SweepStep::Freed);
        }
    }

    /// Sweeps up to `*slots` slots of a major collection in one loop, one
    /// slot per unit and the step that finds the arena swept included, and
    /// counts them down. Returns `true` once the arena is swept. One call
    /// replaces a call of `sweep_major_one` per slot, so the dispatch on the
    /// arena and the cursor reload happen once per batch, not per object.
    /// It is never inlined: how much of the sweep the caller absorbs must
    /// not depend on what else the crate links in.
    #[inline(never)]
    pub(crate) fn sweep_major_run(
        &mut self,
        epoch: u64,
        cursor: &mut MajorSweepCursor,
        slots: &mut usize,
        mut kept: impl FnMut(&T),
        mut freed: impl FnMut(&mut T),
    ) -> bool {
        while *slots > 0 {
            *slots -= 1;
            if self
                .sweep_major_one(epoch, cursor, &mut kept, &mut freed)
                .is_none()
            {
                return true;
            }
        }
        false
    }

    /// Frees every unmarked object and makes survivors old at once.
    #[cfg(test)]
    pub(crate) fn sweep(&mut self, epoch: u64) -> usize {
        let mut cursor = MajorSweepCursor::default();
        let mut reclaimed = 0;
        while let Some(step) = self.sweep_major_one(epoch, &mut cursor, &mut |_| {}, &mut |_| {}) {
            reclaimed += usize::from(matches!(step, SweepStep::Freed));
        }
        reclaimed
    }

    pub(crate) fn live_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.old
            .iter()
            .chain(&self.survival)
            .chain(&self.young)
            .copied()
    }

    pub(crate) fn accounted_bytes(&self, value_bytes: impl Fn(&T) -> usize) -> usize {
        self.live_slots()
            .filter_map(|slot| self.slots[slot as usize].get())
            .map(value_bytes)
            .sum()
    }

    pub(crate) fn live(&self) -> impl Iterator<Item = (ArenaKey, &T)> {
        self.live_slots().filter_map(move |slot| {
            let entry = &self.slots[slot as usize];
            entry.get().map(|value| {
                (
                    ArenaKey {
                        slot,
                        generation: entry.generation,
                    },
                    value,
                )
            })
        })
    }
}
