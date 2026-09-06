//! The register map of a snapshot frame, stored in shared runs.
//!
//! A trace takes a snapshot at every guard, and two snapshots in a row differ
//! in a few registers. A map is a list of runs of [`RUN`] entries, each behind
//! an `Arc`, so a snapshot that changes one register shares every other run
//! with the one before it. Cloning a map costs one reference count. Whoever
//! walks the snapshots of a trace in order can skip a run it has already seen
//! by comparing pointers ([`RegisterMap::shares_run`],
//! [`RegisterMap::changed_from`]), which makes the cost of a pass follow what
//! changed between snapshots and not snapshots times registers.
//!
//! Sharing is an optimisation only. Two maps are equal when their entries are
//! equal, however their runs are shared, and the serialized form is the plain
//! list of entries.

use super::snapshot::RegisterEntry;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

/// Entries per run.
pub const RUN: usize = 8;

type Run = [RegisterEntry; RUN];

/// What the entries past the end of the last run hold. They are never read
/// as registers, and one fixed value keeps equal maps equal run by run.
const PAD: RegisterEntry = RegisterEntry::Unchanged;

/// One entry per register of a frame's window.
#[derive(Clone)]
pub struct RegisterMap {
    len: u32,
    runs: Arc<[Arc<Run>]>,
}

impl RegisterMap {
    /// A map of no registers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            len: 0,
            runs: Arc::from(Vec::new()),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    #[inline]
    pub fn get(&self, register: usize) -> Option<&RegisterEntry> {
        if register < self.len() {
            Some(&self.runs[register / RUN][register % RUN])
        } else {
            None
        }
    }

    #[must_use]
    pub fn last(&self) -> Option<&RegisterEntry> {
        self.len().checked_sub(1).and_then(|last| self.get(last))
    }

    /// The entries in register order.
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            runs: self.runs.iter(),
            run: [].iter(),
            left: self.len(),
        }
    }

    /// The entries of the registers `from..to`, clamped to the map.
    #[inline]
    pub fn range(&self, from: usize, to: usize) -> Iter<'_> {
        let end = to.min(self.len());
        let from = from.min(end);
        let mut runs = self.runs[(from / RUN).min(self.runs.len())..].iter();
        let run = match runs.next() {
            Some(run) => run[from % RUN..].iter(),
            None => [].iter(),
        };
        Iter {
            runs,
            run,
            left: end - from,
        }
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<RegisterEntry> {
        self.iter().copied().collect()
    }

    /// Sets the entry of `register`. A run that other maps share is copied
    /// first, and an entry that already holds `entry` leaves the map as it
    /// is, shared runs included.
    ///
    /// # Panics
    ///
    /// When the map has no such register.
    pub fn set(&mut self, register: usize, entry: RegisterEntry) {
        assert!(
            register < self.len(),
            "register {register} is outside the map"
        );
        if self.runs[register / RUN][register % RUN] == entry {
            return;
        }
        let runs = Arc::make_mut(&mut self.runs);
        Arc::make_mut(&mut runs[register / RUN])[register % RUN] = entry;
    }

    /// Replaces every entry by what `change` makes of it. Runs that `change`
    /// leaves alone stay shared.
    pub fn update(&mut self, mut change: impl FnMut(usize, &RegisterEntry) -> RegisterEntry) {
        for register in 0..self.len() {
            let entry = change(register, &self.runs[register / RUN][register % RUN]);
            self.set(register, entry);
        }
    }

    /// Adds a register at the end.
    pub fn push(&mut self, entry: RegisterEntry) {
        let register = self.len();
        if register % RUN == 0 {
            let mut runs: Vec<Arc<Run>> = self.runs.iter().cloned().collect();
            runs.push(Arc::new([PAD; RUN]));
            self.runs = Arc::from(runs);
        }
        self.len += 1;
        self.set(register, entry);
    }

    /// This map with `len` registers: cut short, or extended with `entry`.
    /// The runs it keeps whole stay shared with this map, and each of the
    /// others is shared with `previous` where that holds the same entries (a
    /// pass that resizes the snapshots of a trace in order hands in the map
    /// it made of the snapshot before).
    #[must_use]
    pub fn resized(&self, len: usize, entry: RegisterEntry, previous: Option<&Self>) -> Self {
        if len == self.len() {
            return self.clone();
        }
        let previous = previous.filter(|previous| previous.len() == len);
        let kept = self.len().min(len) / RUN;
        let mut runs: Vec<Arc<Run>> = self.runs[..kept].to_vec();
        let mut at = kept * RUN;
        while at < len {
            let mut run = [PAD; RUN];
            for (offset, slot) in run.iter_mut().enumerate().take(len - at) {
                *slot = self.get(at + offset).copied().unwrap_or(entry);
            }
            runs.push(match previous {
                Some(previous) if same_run(&previous.runs[at / RUN], &run) => {
                    Arc::clone(&previous.runs[at / RUN])
                }
                _ => Arc::new(run),
            });
            at += RUN;
        }
        Self {
            len: u32::try_from(len).expect("register count fits u32"),
            runs: Arc::from(runs),
        }
    }

    /// How many runs the map has.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// The entries of run `run`: the registers `run * RUN ..`, of which the
    /// last run may hold fewer than [`RUN`].
    #[must_use]
    pub fn run(&self, run: usize) -> &[RegisterEntry] {
        let from = run * RUN;
        &self.runs[run][..(self.len() - from).min(RUN)]
    }

    /// Whether the two maps are one: the same runs behind one pointer. Equal
    /// maps need not be.
    #[must_use]
    #[inline]
    pub fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.runs, &other.runs)
    }

    /// Whether run `run` of this map is the very run `run` of `other`, so
    /// that whatever was found out about the one holds for the other.
    #[must_use]
    #[inline]
    pub fn shares_run(&self, other: &Self, run: usize) -> bool {
        self.len == other.len
            && other
                .runs
                .get(run)
                .is_some_and(|theirs| Arc::ptr_eq(&self.runs[run], theirs))
    }

    /// The runs of this map that `previous` does not share, as (first
    /// register, entries). With no `previous`, or one of another length,
    /// every run.
    pub fn changed_from<'a>(
        &'a self,
        previous: Option<&'a Self>,
    ) -> impl Iterator<Item = (usize, &'a [RegisterEntry])> + 'a {
        let previous = previous.filter(|previous| previous.len == self.len);
        let whole = previous.is_some_and(|previous| self.is(previous));
        (0..if whole { 0 } else { self.runs.len() })
            .filter(move |run| {
                !previous
                    .is_some_and(|previous| Arc::ptr_eq(&self.runs[*run], &previous.runs[*run]))
            })
            .map(|run| (run * RUN, self.run(run)))
    }

    /// The map `entries` make, sharing with `previous` every run that holds
    /// the same entries there.
    #[must_use]
    pub fn sharing(previous: &Self, entries: &[RegisterEntry]) -> Self {
        if previous.len() != entries.len() {
            return entries.iter().copied().collect();
        }
        let mut same = true;
        let runs: Arc<[Arc<Run>]> = entries
            .chunks(RUN)
            .zip(previous.runs.iter())
            .map(|(chunk, known)| {
                if known[..chunk.len()] == *chunk {
                    Arc::clone(known)
                } else {
                    same = false;
                    Arc::new(padded(chunk))
                }
            })
            .collect();
        if same {
            return previous.clone();
        }
        Self {
            len: previous.len,
            runs,
        }
    }

    /// A map of `previous`'s length made run by run: `fill` gets a run's
    /// number and the entries to write, and returns whether to take
    /// `previous`'s run instead (it then need not write any). A run it fills
    /// with what `previous` holds there is shared all the same.
    ///
    /// # Errors
    ///
    /// The first error `fill` returns.
    pub fn made_like<E>(
        previous: &Self,
        mut fill: impl FnMut(usize, &mut [RegisterEntry]) -> Result<bool, E>,
    ) -> Result<Self, E> {
        let mut same = true;
        let mut runs: Vec<Arc<Run>> = Vec::with_capacity(previous.runs.len());
        for (run, known) in previous.runs.iter().enumerate() {
            let width = (previous.len() - run * RUN).min(RUN);
            let mut made = [PAD; RUN];
            if fill(run, &mut made[..width])? || same_run(&made, known) {
                runs.push(Arc::clone(known));
            } else {
                same = false;
                runs.push(Arc::new(made));
            }
        }
        if same {
            return Ok(previous.clone());
        }
        Ok(Self {
            len: previous.len,
            runs: Arc::from(runs),
        })
    }

    /// This map with every run rebuilt by `rewrite` (which gets the first
    /// register and the entries of a run), except the runs it shares with
    /// `old_previous`: those take the run of `new_previous`, which must be
    /// what the same `rewrite` made of `old_previous`. So a pass that
    /// rewrites the snapshots of a trace in order keeps their sharing and
    /// does the work of a run once.
    #[must_use]
    pub fn rewritten(
        &self,
        previous: Option<(&Self, &Self)>,
        mut rewrite: impl FnMut(usize, &[RegisterEntry], &mut [RegisterEntry]),
    ) -> Self {
        let previous = previous.filter(|(old, new)| old.len == self.len && new.len == self.len);
        if let Some((old, new)) = previous
            && self.is(old)
        {
            return new.clone();
        }
        // The runs are copied when the first of them changes: a rewrite that
        // leaves the map as it is allocates nothing.
        let mut runs: Option<Vec<Arc<Run>>> = None;
        for (run, current) in self.runs.iter().enumerate() {
            let made = if let Some((old, new)) = previous
                && Arc::ptr_eq(current, &old.runs[run])
            {
                if Arc::ptr_eq(&new.runs[run], current) {
                    None
                } else {
                    Some(Arc::clone(&new.runs[run]))
                }
            } else {
                let width = (self.len() - run * RUN).min(RUN);
                let mut made = **current;
                rewrite(run * RUN, &current[..width], &mut made[..width]);
                if same_run(&made, current) {
                    None
                } else {
                    Some(Arc::new(made))
                }
            };
            match (made, &mut runs) {
                (None, None) => {}
                (None, Some(runs)) => runs.push(Arc::clone(current)),
                (Some(made), Some(runs)) => runs.push(made),
                (Some(made), None) => {
                    let mut copied = Vec::with_capacity(self.runs.len());
                    copied.extend_from_slice(&self.runs[..run]);
                    copied.push(made);
                    runs = Some(copied);
                }
            }
        }
        match runs {
            None => self.clone(),
            Some(runs) => Self {
                len: self.len,
                runs: Arc::from(runs),
            },
        }
    }

    /// Makes the entry of `register` `to` where it is `from`, for a pass
    /// that does so to the snapshots of a trace in order: `memo` holds the
    /// run the pass changed last and what it became, so the snapshots that
    /// shared that run share the new one. A map without the register, or
    /// with another entry there, is left as it is at the cost of one look.
    pub fn replace(
        &mut self,
        register: usize,
        from: RegisterEntry,
        to: RegisterEntry,
        memo: &mut RunMemo,
    ) {
        if self.get(register) != Some(&from) {
            return;
        }
        // The snapshot before held these very runs: it became this map too.
        if let Some((old, new)) = &memo.whole
            && Arc::ptr_eq(old, &self.runs)
        {
            self.runs = Arc::clone(new);
            return;
        }
        let (run, offset) = (register / RUN, register % RUN);
        let current = &self.runs[run];
        let made = match &memo.run {
            Some((old, new)) if Arc::ptr_eq(old, current) => Arc::clone(new),
            _ => {
                let mut made = **current;
                made[offset] = to;
                let made = Arc::new(made);
                memo.run = Some((Arc::clone(current), Arc::clone(&made)));
                made
            }
        };
        // Runs no other map holds are changed in place, and no snapshot
        // after this one can hold them either.
        if let Some(runs) = Arc::get_mut(&mut self.runs) {
            runs[run] = made;
            memo.whole = None;
            return;
        }
        let before = Arc::clone(&self.runs);
        Arc::make_mut(&mut self.runs)[run] = made;
        memo.whole = Some((before, Arc::clone(&self.runs)));
    }

    /// About the bytes the map holds that no other map shares with it, for
    /// the size estimates of the trace caches: a run with other owners is
    /// charged to none of them, so the estimate stays a lower bound.
    #[must_use]
    pub fn owned_bytes(&self) -> usize {
        self.runs
            .iter()
            .filter(|run| Arc::strong_count(run) == 1)
            .count()
            * size_of::<Run>()
            + self.runs.len() * size_of::<Arc<Run>>()
    }
}

/// What [`RegisterMap::replace`] remembers between the snapshots of one
/// pass: the run it changed last, and the run that became.
#[derive(Default)]
pub struct RunMemo {
    run: Option<(Arc<Run>, Arc<Run>)>,
    /// The runs of the map the pass changed last, and what they became.
    whole: Option<(Arc<[Arc<Run>]>, Arc<[Arc<Run>]>)>,
}

/// Whether two runs hold the same entries, compared in place.
#[inline]
fn same_run(ours: &Run, theirs: &Run) -> bool {
    ours.iter().zip(theirs).all(|(ours, theirs)| ours == theirs)
}

fn padded(chunk: &[RegisterEntry]) -> Run {
    let mut run = [PAD; RUN];
    run[..chunk.len()].copy_from_slice(chunk);
    run
}

impl Default for RegisterMap {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<RegisterEntry> for RegisterMap {
    fn from_iter<I: IntoIterator<Item = RegisterEntry>>(entries: I) -> Self {
        let mut runs: Vec<Arc<Run>> = Vec::new();
        let mut run = [PAD; RUN];
        let mut len = 0_usize;
        for entry in entries {
            run[len % RUN] = entry;
            len += 1;
            if len % RUN == 0 {
                runs.push(Arc::new(run));
                run = [PAD; RUN];
            }
        }
        if len % RUN != 0 {
            runs.push(Arc::new(run));
        }
        Self {
            len: u32::try_from(len).expect("register count fits u32"),
            runs: Arc::from(runs),
        }
    }
}

impl From<Vec<RegisterEntry>> for RegisterMap {
    fn from(entries: Vec<RegisterEntry>) -> Self {
        entries.into_iter().collect()
    }
}

impl From<&[RegisterEntry]> for RegisterMap {
    fn from(entries: &[RegisterEntry]) -> Self {
        entries.iter().copied().collect()
    }
}

impl<const N: usize> From<[RegisterEntry; N]> for RegisterMap {
    fn from(entries: [RegisterEntry; N]) -> Self {
        entries.into_iter().collect()
    }
}

impl std::ops::Index<usize> for RegisterMap {
    type Output = RegisterEntry;

    #[inline]
    fn index(&self, register: usize) -> &RegisterEntry {
        self.get(register).expect("a register of the map")
    }
}

impl PartialEq for RegisterMap {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && (self.is(other)
                || self
                    .runs
                    .iter()
                    .zip(other.runs.iter())
                    .all(|(ours, theirs)| Arc::ptr_eq(ours, theirs) || same_run(ours, theirs)))
    }
}

impl PartialEq<Vec<RegisterEntry>> for RegisterMap {
    fn eq(&self, other: &Vec<RegisterEntry>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl PartialEq<[RegisterEntry]> for RegisterMap {
    fn eq(&self, other: &[RegisterEntry]) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl std::fmt::Debug for RegisterMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Serialize for RegisterMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for RegisterMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<RegisterEntry>::deserialize(deserializer).map(Self::from)
    }
}

/// The entries of a [`RegisterMap`] in register order.
#[derive(Clone)]
pub struct Iter<'a> {
    /// The runs after the one being walked.
    runs: std::slice::Iter<'a, Arc<Run>>,
    run: std::slice::Iter<'a, RegisterEntry>,
    /// How many entries are still to come: the last run's padding is none.
    left: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a RegisterEntry;

    #[inline]
    fn next(&mut self) -> Option<&'a RegisterEntry> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        loop {
            if let Some(entry) = self.run.next() {
                return Some(entry);
            }
            self.run = self.runs.next()?.iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Iter<'_> {}

impl<'a> IntoIterator for &'a RegisterMap {
    type Item = &'a RegisterEntry;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{RUN, RegisterMap};
    use crate::jit_ir::trace::{RegisterEntry, TraceScalar, ValueId};
    use std::sync::Arc;

    fn entries(count: usize) -> Vec<RegisterEntry> {
        (0..count)
            .map(|index| match index % 3 {
                0 => RegisterEntry::Unchanged,
                1 => RegisterEntry::Value(ValueId(u32::try_from(index).unwrap())),
                _ => RegisterEntry::Const(TraceScalar::Integer(i64::try_from(index).unwrap())),
            })
            .collect()
    }

    #[test]
    fn a_map_holds_the_entries_it_was_made_from() {
        for count in [0, 1, RUN - 1, RUN, RUN + 1, 3 * RUN, 3 * RUN + 5] {
            let list = entries(count);
            let map = RegisterMap::from(list.clone());
            assert_eq!(map.len(), count);
            assert_eq!(map.to_vec(), list);
            assert_eq!(map.iter().len(), count);
            assert_eq!(map.last(), list.last());
            assert_eq!(map.get(count), None);
            for (register, entry) in list.iter().enumerate() {
                assert_eq!(map[register], *entry);
            }
            let runs: Vec<RegisterEntry> = (0..map.run_count())
                .flat_map(|run| map.run(run).to_vec())
                .collect();
            assert_eq!(runs, list);
            assert_eq!(format!("{map:?}"), format!("{list:?}"));
            let bytes = bincode::serialize(&map).unwrap();
            assert_eq!(bytes, bincode::serialize(&list).unwrap());
            let back: RegisterMap = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back, map);
        }
    }

    #[test]
    fn a_write_copies_one_run_and_leaves_the_other_map_alone() {
        let list = entries(3 * RUN + 2);
        let first = RegisterMap::from(list.clone());
        let mut second = first.clone();
        assert!(second.is(&first));
        // Writing what is there already changes nothing.
        second.set(RUN + 1, list[RUN + 1]);
        assert!(second.is(&first));
        second.set(RUN + 1, RegisterEntry::Root(7));
        assert_eq!(first.to_vec(), list);
        assert_eq!(second[RUN + 1], RegisterEntry::Root(7));
        assert_ne!(first, second);
        let shared: Vec<bool> = (0..second.run_count())
            .map(|run| second.shares_run(&first, run))
            .collect();
        assert_eq!(shared, [true, false, true, true]);
        let changed: Vec<usize> = second
            .changed_from(Some(&first))
            .map(|(register, _)| register)
            .collect();
        assert_eq!(changed, [RUN]);
        assert_eq!(second.changed_from(None).count(), 4);
        assert_eq!(second.changed_from(Some(&second.clone())).count(), 0);
    }

    #[test]
    fn equal_maps_are_equal_however_they_share() {
        let list = entries(2 * RUN + 3);
        let one = RegisterMap::from(list.clone());
        let other = RegisterMap::from(list.clone());
        assert!(!one.is(&other));
        assert_eq!(one, other);
        assert_eq!(one, list);
        let mut longer = list.clone();
        longer.push(RegisterEntry::Unchanged);
        // The padding of the last run is not a register.
        assert_ne!(one, RegisterMap::from(longer));
        let again = RegisterMap::sharing(&one, &list);
        assert!(again.is(&one));
        let mut changed = list;
        changed[0] = RegisterEntry::Root(1);
        let third = RegisterMap::sharing(&one, &changed);
        assert_eq!(third.to_vec(), changed);
        assert!(!third.shares_run(&one, 0));
        assert!(third.shares_run(&one, 1) && third.shares_run(&one, 2));
    }

    #[test]
    fn a_rewrite_in_order_keeps_the_sharing() {
        let list = entries(3 * RUN);
        let first = RegisterMap::from(list);
        let mut second = first.clone();
        second.set(2 * RUN, RegisterEntry::Root(3));
        let mut calls = 0;
        let mut bump = |_: usize, _: &[RegisterEntry], run: &mut [RegisterEntry]| {
            calls += 1;
            for entry in run {
                if let RegisterEntry::Value(value) = entry {
                    value.0 += 100;
                }
            }
        };
        let new_first = first.rewritten(None, &mut bump);
        let new_second = second.rewritten(Some((&first, &new_first)), &mut bump);
        assert_eq!(calls, 4);
        assert!(new_second.shares_run(&new_first, 0) && new_second.shares_run(&new_first, 1));
        let expected = |map: &RegisterMap| -> Vec<RegisterEntry> {
            map.iter()
                .map(|entry| match entry {
                    RegisterEntry::Value(value) => RegisterEntry::Value(ValueId(value.0 + 100)),
                    other => *other,
                })
                .collect()
        };
        assert_eq!(new_first, expected(&first));
        assert_eq!(new_second, expected(&second));
    }

    #[test]
    fn a_resize_keeps_whole_runs_and_shares_the_rest_with_the_map_before() {
        let list = entries(RUN + 3);
        let map = RegisterMap::from(list.clone());
        let longer = map.resized(3 * RUN + 1, RegisterEntry::Unchanged, None);
        let mut expected = list.clone();
        expected.resize(3 * RUN + 1, RegisterEntry::Unchanged);
        assert_eq!(longer, expected);
        assert!(Arc::ptr_eq(&longer.runs[0], &map.runs[0]));
        let again = map.resized(3 * RUN + 1, RegisterEntry::Unchanged, Some(&longer));
        assert_eq!(again, expected);
        assert!((0..again.run_count()).all(|run| again.shares_run(&longer, run)));
        let shorter = longer.resized(RUN + 3, RegisterEntry::Unchanged, None);
        assert_eq!(shorter, list);
        assert_eq!(shorter, map);
        assert!(
            map.resized(map.len(), RegisterEntry::Root(0), None)
                .is(&map)
        );
    }

    #[test]
    fn a_replace_in_order_keeps_the_sharing_and_skips_other_entries() {
        let list = entries(3 * RUN);
        let first = RegisterMap::from(list.clone());
        let mut second = first.clone();
        second.set(0, RegisterEntry::Root(1));
        let mut third = second.clone();
        third.set(RUN, RegisterEntry::Root(2));
        // Register RUN holds `Value(RUN)` in the first two and a root slot
        // in the third.
        let from = list[RUN];
        let to = RegisterEntry::Const(TraceScalar::Nil);
        let mut maps = [first.clone(), second.clone(), third.clone()];
        let mut memo = super::RunMemo::default();
        for map in &mut maps {
            map.replace(RUN, from, to, &mut memo);
            // Out of range: nothing happens.
            map.replace(3 * RUN, from, to, &mut memo);
        }
        assert_eq!(maps[0][RUN], to);
        assert_eq!(maps[1][RUN], to);
        assert!(maps[1].shares_run(&maps[0], 1));
        assert!(maps[2].is(&third));
        assert_eq!(first[RUN], from);
        let mut expected = list;
        expected[RUN] = to;
        assert_eq!(maps[0], expected);
        assert_eq!(first.range(RUN - 1, RUN + 2).count(), 3);
        assert_eq!(first.range(RUN, usize::MAX).count(), 2 * RUN);
        assert_eq!(first.range(5 * RUN, usize::MAX).count(), 0);
        assert_eq!(
            first.range(3, 2 * RUN + 1).copied().collect::<Vec<_>>(),
            first.to_vec()[3..2 * RUN + 1]
        );
    }

    #[test]
    fn a_map_made_like_another_takes_the_runs_it_is_told_to() {
        let list = entries(2 * RUN + 3);
        let first = RegisterMap::from(list.clone());
        let mut asked = Vec::new();
        let made = RegisterMap::made_like(&first, |run, out| {
            asked.push((run, out.len()));
            match run {
                // Taken as it is, written as it is, and changed.
                0 => Ok::<_, ()>(true),
                1 => {
                    out.copy_from_slice(&list[RUN..2 * RUN]);
                    Ok(false)
                }
                _ => {
                    out.copy_from_slice(&list[2 * RUN..]);
                    out[0] = RegisterEntry::Root(9);
                    Ok(false)
                }
            }
        })
        .unwrap();
        assert_eq!(asked, [(0, RUN), (1, RUN), (2, 3)]);
        let mut expected = list;
        expected[2 * RUN] = RegisterEntry::Root(9);
        assert_eq!(made, expected);
        assert!(made.shares_run(&first, 0) && made.shares_run(&first, 1));
        assert!(!made.shares_run(&first, 2));
        let again = RegisterMap::made_like(&first, |_, _| Ok::<_, ()>(true)).unwrap();
        assert!(again.is(&first));
        assert_eq!(
            RegisterMap::made_like(&first, |_, _| Err::<bool, _>(7)),
            Err(7)
        );
    }

    #[test]
    fn push_and_update_keep_the_entries() {
        let mut map = RegisterMap::new();
        let list = entries(RUN + 3);
        for entry in &list {
            map.push(*entry);
        }
        assert_eq!(map, list);
        let before = map.clone();
        map.update(|_, entry| *entry);
        assert!(map.is(&before));
        map.update(|register, entry| {
            if register == 1 {
                RegisterEntry::Unchanged
            } else {
                *entry
            }
        });
        assert_eq!(map[1], RegisterEntry::Unchanged);
        assert!(map.shares_run(&before, 1));
    }
}
