//! Per-instruction inline caches for string-keyed table accesses.
//!
//! A [`Chunk`] owns one [`FieldSite`] per instruction, allocated the first
//! time the VM installs a cache in it. The VM keys a site on the `GetGlobal`,
//! `GetField` or `SetField` at that pc. The chunk only stores two words per
//! site and knows nothing about tables: the runtime defines what they mean.
//!
//! One chunk may run in several VMs on different threads, so the sites of a
//! chunk are written under one lock, and a reader accepts a site only when
//! the guard word reads the same before and after the data word (a seqlock,
//! see [`FieldSite::load`]). The guard is not a version count, so a site
//! never takes back a guard it gave up: see [`Sites::install`].

use crate::bytecode::Chunk;
use crate::string::LuaString;
use core::fmt;
use core::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Refills after which a site stops caching. A site that keeps seeing
/// different tables costs more to refill than to look up, and one that
/// misses because its field is not there yet refills the same entry.
const CHURN_LIMIT: u8 = 8;

/// Fills skipped after one is refused by a closed site.
const FILL_BACKOFF: u8 = 31;

/// The churn count of a site that gave up: it was cleared, so a read of it
/// stops at its empty guard, and it is never filled again.
const CLOSED: u8 = u8::MAX;

/// What one access site remembers about the table it last found its key in.
/// A guard of zero means empty.
#[derive(Default)]
pub struct FieldSite {
    guard: AtomicU64,
    data: AtomicUsize,
}

/// How many empty sites a chunk with no cache yet can index into: the
/// longest chunk that starts without sites of its own (see
/// [`FieldSites::for_code_len`]).
const PLACEHOLDER_SITES: usize = 1 << 16;

/// Sites that stay empty for ever: no fill writes to them. A chunk with no
/// sites yet points here ([`Chunk::read_sites_first`]), so a hit path indexes
/// by pc without asking whether the chunk has sites. All zeros, so it takes
/// address space but no memory until a page is read.
static PLACEHOLDER: [FieldSite; PLACEHOLDER_SITES] =
    [const { FieldSite::empty() }; PLACEHOLDER_SITES];

impl FieldSite {
    /// An empty site: a guard of zero.
    const fn empty() -> Self {
        Self {
            guard: AtomicU64::new(0),
            data: AtomicUsize::new(0),
        }
    }

    /// The `(guard, data)` pair of the site, or a guard of zero when the site
    /// is empty or a writer was at it. This is a seqlock read: the guard is
    /// read, then the data, then the guard again, and the pair is returned only
    /// when both guard reads agree. A writer clears the guard, writes the
    /// data, then writes the new guard. So the data a reader sees was written
    /// by a fill no older than the one its first guard read saw, and no newer
    /// than the one its second guard read saw. Both reads seeing the same
    /// guard proves the data belongs to it only because the site never goes
    /// back to a guard it gave up ([`Sites::install`]): fills `A`, `B`, `A`
    /// would let a reader stalled across them pair `A` with `B`'s data. A
    /// guard names one hashed layout or one shape; an ordered table's guard
    /// names no layout, and its reader checks the key at the position it
    /// finds.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn load(&self) -> (u64, usize) {
        let before = self.guard.load(Ordering::Acquire);
        let data = self.data.load(Ordering::Relaxed);
        core::sync::atomic::fence(Ordering::Acquire);
        let after = self.guard.load(Ordering::Relaxed);
        (if before == after { before } else { 0 }, data)
    }

    /// [`Self::load`] as a branch: `None` when the two guard reads disagree,
    /// so a hit path jumps once on a torn read instead of selecting a zero
    /// guard. An empty site still gives `Some` with a guard of zero, which no
    /// table layout carries.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn load_pair(&self) -> Option<(u64, usize)> {
        let before = self.guard.load(Ordering::Acquire);
        let data = self.data.load(Ordering::Relaxed);
        core::sync::atomic::fence(Ordering::Acquire);
        let after = self.guard.load(Ordering::Relaxed);
        (before == after).then_some((before, data))
    }

    /// Fills the site. Writes to one site must not overlap.
    fn store(&self, guard: u64, data: usize) {
        self.guard.store(0, Ordering::Relaxed);
        core::sync::atomic::fence(Ordering::Release);
        self.data.store(data, Ordering::Relaxed);
        self.guard.store(guard, Ordering::Release);
    }

    /// The guard the site holds, or zero when it is empty. A writer reads it
    /// back as it stood after the last write.
    fn guard(&self) -> u64 {
        self.guard.load(Ordering::Relaxed)
    }

    /// Empties the site. Like [`Self::store`], it must not overlap a write.
    fn clear(&self) {
        self.guard.store(0, Ordering::Relaxed);
    }
}

struct Sites {
    slots: Box<[FieldSite]>,
    churn: Box<[AtomicU8]>,
    /// Held across every write to `slots` and `churn`.
    writer: Mutex<Writer>,
}

/// What the writers of one [`Sites`] keep. Each site is refilled only a few
/// times, which bounds both collections.
#[derive(Default)]
struct Writer {
    /// Strings that sites identify by address. A site holds the address of a
    /// key string of an ordered table; keeping the string alive keeps the
    /// address from naming any other string.
    pins: Vec<LuaString>,
    /// The `(pc, guard)` of every guard a site gave up for another one.
    retired: HashSet<(usize, u64)>,
}

impl Sites {
    fn new(len: usize) -> Self {
        Self {
            slots: (0..len).map(|_| FieldSite::default()).collect(),
            churn: (0..len).map(|_| AtomicU8::new(0)).collect(),
            writer: Mutex::new(Writer::default()),
        }
    }

    fn writer(&self) -> MutexGuard<'_, Writer> {
        self.writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn open(&self, pc: usize) -> bool {
        match (self.slots.get(pc), self.churn.get(pc)) {
            (Some(slot), Some(churn)) => {
                let replaced = churn.load(Ordering::Relaxed);
                replaced != CLOSED && (slot.guard() == 0 || replaced < CHURN_LIMIT)
            }
            _ => false,
        }
    }

    /// Fills the site at `pc`, unless it is closed. A site replaced too
    /// often closes, and so does a site asked to take back a guard it gave
    /// up: after fills `A` then `B`, a fill of `A` could let a reader that
    /// read `A`, stalled, then read `A` again, keep the data `B` wrote (see
    /// [`FieldSite::load`]). A closed site is cleared. Refilling the guard
    /// the site holds is fine: every fill under one guard writes data for it.
    fn install(&self, pc: usize, guard: u64, data: usize, pin: Option<LuaString>) -> bool {
        let (Some(slot), Some(churn)) = (self.slots.get(pc), self.churn.get(pc)) else {
            return false;
        };
        let mut writer = self.writer();
        let replaced = churn.load(Ordering::Relaxed);
        if replaced == CLOSED {
            return false;
        }
        let held = slot.guard();
        if held != 0 {
            if replaced >= CHURN_LIMIT || writer.retired.contains(&(pc, guard)) {
                churn.store(CLOSED, Ordering::Relaxed);
                slot.clear();
                return false;
            }
            churn.store(replaced + 1, Ordering::Relaxed);
            if held != guard {
                writer.retired.insert((pc, held));
            }
        }
        if let Some(pin) = pin {
            writer.pins.push(pin);
        }
        slot.store(guard, data);
        true
    }
}

/// Which instructions a site serves. A chunk allocates the sites of each
/// kind when it first fills one, so a chunk whose stores never find a field
/// to overwrite checks for its store sites at the cost of one load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SiteKind {
    /// `GetGlobal` and `GetField`.
    Read,
    /// `SetGlobal` and `SetField`.
    Write,
}

/// The lazily allocated sites of a [`Chunk`].
///
/// It is a cache: clones start empty and two chunks compare equal whatever
/// they hold.
pub struct FieldSites {
    reads: OnceLock<Sites>,
    writes: OnceLock<Sites>,
    /// The first read site, for a reader that has the chunk and nothing else:
    /// the shared placeholder until `reads` is allocated, then its slots. One
    /// load, where `reads` takes a state check, a bounds check and a load.
    read_first: AtomicPtr<FieldSite>,
    /// [`Self::read_first`] for `writes`.
    write_first: AtomicPtr<FieldSite>,
    /// Read fills to skip: set when a site refuses one, so a run of reads
    /// that all find closed sites costs a counter step each, not a lookup.
    read_backoff: AtomicU8,
    /// [`Self::read_backoff`] for write fills.
    write_backoff: AtomicU8,
}

impl Default for FieldSites {
    fn default() -> Self {
        Self {
            reads: OnceLock::new(),
            writes: OnceLock::new(),
            read_first: AtomicPtr::new(PLACEHOLDER.as_ptr().cast_mut()),
            write_first: AtomicPtr::new(PLACEHOLDER.as_ptr().cast_mut()),
            read_backoff: AtomicU8::new(0),
            write_backoff: AtomicU8::new(0),
        }
    }
}

impl FieldSites {
    /// The sites of a chunk with `len` instructions. A chunk longer than the
    /// placeholder owns its (empty) sites from the start, so every chunk's
    /// first-site pointer covers its code.
    #[must_use]
    pub fn for_code_len(len: usize) -> Self {
        let sites = Self::default();
        if len > PLACEHOLDER_SITES {
            sites.sites_of(SiteKind::Read, len);
            sites.sites_of(SiteKind::Write, len);
        }
        sites
    }

    /// The sites of `kind`, allocated for `len` instructions on first use,
    /// and published through the first-site pointer.
    fn sites_of(&self, kind: SiteKind, len: usize) -> &Sites {
        let sites = self.of(kind).get_or_init(|| Sites::new(len));
        let first = match kind {
            SiteKind::Read => &self.read_first,
            SiteKind::Write => &self.write_first,
        };
        first.store(sites.slots.as_ptr().cast_mut(), Ordering::Release);
        sites
    }

    fn backoff_of(&self, kind: SiteKind) -> &AtomicU8 {
        match kind {
            SiteKind::Read => &self.read_backoff,
            SiteKind::Write => &self.write_backoff,
        }
    }

    fn of(&self, kind: SiteKind) -> &OnceLock<Sites> {
        match kind {
            SiteKind::Read => &self.reads,
            SiteKind::Write => &self.writes,
        }
    }
}

impl Clone for FieldSites {
    fn clone(&self) -> Self {
        // A cache starts empty, but a long chunk's must still cover its code.
        let clone = Self::default();
        for kind in [SiteKind::Read, SiteKind::Write] {
            if let Some(sites) = self.of(kind).get()
                && sites.slots.len() > PLACEHOLDER_SITES
            {
                clone.sites_of(kind, sites.slots.len());
            }
        }
        clone
    }
}

impl PartialEq for FieldSites {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl fmt::Debug for FieldSites {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FieldSites")
    }
}

impl Chunk {
    /// The read site of the `GetGlobal` or `GetField` at `pc`, or `None`
    /// until a cache has been installed for any read in this chunk.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn field_site(&self, pc: usize) -> Option<&FieldSite> {
        self.field_sites.reads.get()?.slots.get(pc)
    }

    /// The first of the read sites of this chunk: one site per instruction
    /// of [`Chunk::code`], valid for as long as the chunk lives, as long as
    /// the code keeps the length it had when the chunk was built. Until a
    /// read installs a cache these are empty placeholder sites shared by every
    /// chunk. A site pointer taken here stays right across fills: the
    /// pointer is only ever the placeholder or the chunk's own sites, and a
    /// fill moves a chunk from the first to the second, so ask again after
    /// one. This is [`Chunk::field_site`] without a state check or a bounds
    /// check, for a loop that runs the code it borrows, whose `pc` is below
    /// the code's length.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn read_sites_first(&self) -> *const FieldSite {
        self.field_sites.read_first.load(Ordering::Acquire)
    }

    /// [`Chunk::read_sites_first`] for the write sites of `SetField`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn write_sites_first(&self) -> *const FieldSite {
        self.field_sites.write_first.load(Ordering::Acquire)
    }

    /// The write site of the `SetField` at `pc`, or `None` until a cache has
    /// been installed for any write in this chunk. [`Chunk::store_site_at`]
    /// for a caller that has the index already.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn store_site(&self, pc: usize) -> Option<&FieldSite> {
        self.field_sites.writes.get()?.slots.get(pc)
    }

    /// The write site of the `SetGlobal` or `SetField` at address `at` in this
    /// chunk's code, or `None` until a cache has been installed for any write
    /// in the chunk. The index is worked out only when the chunk has write
    /// sites.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn store_site_at(&self, at: *const crate::bytecode::Instruction) -> Option<&FieldSite> {
        let sites = self.field_sites.writes.get()?;
        let pc = (at as usize).wrapping_sub(self.code.as_ptr() as usize)
            / core::mem::size_of::<crate::bytecode::Instruction>();
        sites.slots.get(pc)
    }

    /// Whether this fill of a site of `kind` is skipped: a closed site
    /// started a run of skipped fills (see [`Chunk::refuse_fills`]), and this
    /// is one of them.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn skip_fill(&self, kind: SiteKind) -> bool {
        let backoff = self.field_sites.backoff_of(kind);
        let left = backoff.load(Ordering::Relaxed);
        if left == 0 {
            return false;
        }
        backoff.store(left - 1, Ordering::Relaxed);
        true
    }

    /// Skips the next few fills of sites of `kind` in this chunk: a site
    /// refused one.
    pub fn refuse_fills(&self, kind: SiteKind) {
        self.field_sites
            .backoff_of(kind)
            .store(FILL_BACKOFF, Ordering::Relaxed);
    }

    /// Whether a cache may still be installed at `pc`: the chunk has no sites
    /// yet, the site is empty, or it has been replaced fewer than a handful
    /// of times. The VM asks before it looks the field up again.
    #[must_use]
    #[inline]
    pub fn field_site_open(&self, kind: SiteKind, pc: usize) -> bool {
        self.field_sites
            .of(kind)
            .get()
            .is_none_or(|sites| sites.open(pc))
    }

    /// Stores `guard` and `data` in the site of `pc`, allocating the chunk's
    /// sites of that kind on first use. `guard` must not be zero. `pin` is
    /// kept alive for as long as the chunk lives. A site that has been
    /// replaced too often is left alone. Returns whether the site was written.
    #[cold]
    #[inline(never)]
    pub fn install_field_site(
        &self,
        kind: SiteKind,
        pc: usize,
        guard: u64,
        data: usize,
        pin: Option<LuaString>,
    ) -> bool {
        debug_assert!(guard != 0);
        self.field_sites
            .sites_of(kind, self.code.len())
            .install(pc, guard, data, pin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Instruction;

    #[test]
    fn a_reader_never_pairs_a_guard_with_the_data_of_another_fill() {
        // Each fill brings a new guard, as `Sites::install` guarantees: a
        // guard that came back would let a stalled reader pair it with the
        // data of the fill in between (see the next test).
        let rounds: u64 = if cfg!(miri) { 300 } else { 100_000 };
        let site = FieldSite::default();
        site.store(1, 10);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for guard in 2..rounds {
                    site.store(guard, usize::try_from(guard * 10).unwrap());
                }
            });
            for _ in 0..rounds {
                let (guard, data) = site.load();
                if guard != 0 {
                    assert_eq!(data, usize::try_from(guard * 10).unwrap());
                }
            }
        });
    }

    #[test]
    fn a_site_never_takes_back_a_guard_it_gave_up() {
        let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt(); 2]);
        let guard = |pc| chunk.field_site(pc).unwrap().load().0;
        assert!(chunk.install_field_site(SiteKind::Read, 1, 1, 10, None));
        // The same guard again, with other data, is a refill like any other.
        assert!(chunk.install_field_site(SiteKind::Read, 1, 1, 11, None));
        assert!(chunk.install_field_site(SiteKind::Read, 1, 2, 20, None));
        assert_eq!(guard(1), 2);
        // Going back to guard 1 closes the site instead.
        assert!(!chunk.install_field_site(SiteKind::Read, 1, 1, 10, None));
        assert_eq!(guard(1), 0);
        assert!(!chunk.field_site_open(SiteKind::Read, 1));
        assert!(!chunk.install_field_site(SiteKind::Read, 1, 3, 30, None));
        // Guards are remembered per site.
        assert!(chunk.install_field_site(SiteKind::Read, 0, 2, 20, None));
        assert!(chunk.install_field_site(SiteKind::Read, 0, 1, 10, None));
        assert_eq!(guard(0), 1);
    }

    #[test]
    fn a_site_replaced_too_often_is_cleared_and_stays_closed() {
        let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt(); 3]);
        for round in 0..=u64::from(CHURN_LIMIT) {
            assert!(chunk.field_site_open(SiteKind::Read, 1));
            assert!(chunk.install_field_site(SiteKind::Read, 1, 1 + round, 0, None));
        }
        assert_eq!(chunk.field_site(1).unwrap().load().0, 9);
        // The next fill finds the site worn out: it is cleared, not refilled.
        assert!(!chunk.install_field_site(SiteKind::Read, 1, 99, 0, None));
        assert_eq!(chunk.field_site(1).unwrap().load().0, 0);
        assert!(!chunk.field_site_open(SiteKind::Read, 1));
        assert!(!chunk.install_field_site(SiteKind::Read, 1, 100, 0, None));
        assert_eq!(chunk.field_site(1).unwrap().load().0, 0);
        // Other sites of the chunk are unaffected.
        assert!(chunk.field_site_open(SiteKind::Read, 0));
    }

    #[test]
    fn a_refused_fill_makes_the_chunk_skip_the_next_few() {
        let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt(); 2]);
        assert!(!chunk.skip_fill(SiteKind::Read));
        chunk.refuse_fills(SiteKind::Read);
        for _ in 0..FILL_BACKOFF {
            assert!(chunk.skip_fill(SiteKind::Read));
        }
        assert!(!chunk.skip_fill(SiteKind::Read));
        // Writes keep their own count.
        chunk.refuse_fills(SiteKind::Write);
        assert!(!chunk.skip_fill(SiteKind::Read));
        assert!(chunk.skip_fill(SiteKind::Write));
    }

    #[test]
    fn the_first_site_is_a_placeholder_until_a_fill_and_matches_field_site() {
        let placeholder = PLACEHOLDER.as_ptr();
        let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt(); 3]);
        assert!(std::ptr::eq(chunk.read_sites_first(), placeholder));
        assert!(std::ptr::eq(chunk.write_sites_first(), placeholder));
        assert!(chunk.field_site(1).is_none());
        assert!(
            PLACEHOLDER
                .iter()
                .all(|site| site.load_pair() == Some((0, 0)))
        );
        assert!(chunk.install_field_site(SiteKind::Read, 1, 7, 3, None));
        let first = chunk.read_sites_first();
        assert!(!std::ptr::eq(first, placeholder));
        assert!(std::ptr::eq(first, chunk.field_site(0).unwrap()));
        assert!(std::ptr::eq(
            first.wrapping_add(2),
            chunk.field_site(2).unwrap()
        ));
        assert_eq!(chunk.field_site(1).unwrap().load(), (7, 3));
        assert_eq!(chunk.field_site(0).unwrap().load().0, 0);
        // Write sites are separate from read sites.
        assert!(std::ptr::eq(chunk.write_sites_first(), placeholder));
        let other = Chunk::new(1, Vec::new(), vec![Instruction::halt(); 3]);
        assert!(other.install_field_site(SiteKind::Write, 1, 7, 3, None));
        assert!(std::ptr::eq(other.read_sites_first(), placeholder));
        assert!(std::ptr::eq(
            other.write_sites_first(),
            other.store_site(0).unwrap()
        ));
        assert!(std::ptr::eq(
            other.store_site(2).unwrap(),
            other
                .store_site_at(other.code.as_ptr().wrapping_add(2))
                .unwrap()
        ));
        assert!(chunk.store_site(0).is_none());
    }

    #[test]
    fn a_chunk_longer_than_the_placeholder_owns_empty_sites_from_the_start() {
        let len = PLACEHOLDER_SITES + 5;
        let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt(); len]);
        let first = chunk.read_sites_first();
        assert!(!std::ptr::eq(first, PLACEHOLDER.as_ptr()));
        assert!(std::ptr::eq(first, chunk.field_site(0).unwrap()));
        assert!(std::ptr::eq(
            first.wrapping_add(len - 1),
            chunk.field_site(len - 1).unwrap()
        ));
        assert!(std::ptr::eq(
            chunk.write_sites_first(),
            chunk.store_site(0).unwrap()
        ));
        assert!(chunk.install_field_site(SiteKind::Read, len - 1, 7, 3, None));
        assert!(std::ptr::eq(first, chunk.read_sites_first()));
        // A clone starts empty and still covers its code.
        let clone = chunk.clone();
        let cloned = clone.read_sites_first();
        assert!(!std::ptr::eq(cloned, PLACEHOLDER.as_ptr()));
        assert!(!std::ptr::eq(cloned, first));
        assert_eq!(clone.field_site(len - 1).unwrap().load().0, 0);
    }

    #[test]
    fn load_pair_agrees_with_load_and_gives_an_empty_site_a_zero_guard() {
        let site = FieldSite::default();
        assert_eq!(site.load_pair(), Some((0, 0)));
        site.store(5, 9);
        assert_eq!(site.load_pair(), Some(site.load()));
        assert_eq!(site.load_pair(), Some((5, 9)));
    }
}
