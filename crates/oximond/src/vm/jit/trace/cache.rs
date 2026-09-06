//! The per-VM cache of compiled traces.
//!
//! The cache owns every installed trace and keeps three kinds of bound:
//!
//! - Executable bytes ([`MAX_TRACE_CODE_BYTES`]): the machine code of the
//!   installed traces, each charged in whole pages.
//! - Metadata bytes ([`MAX_TRACE_META_BYTES`]): the IR and exit plans an
//!   installed trace keeps for recompiling and restoring, plus the IR of
//!   requests still waiting for the broker. A request reserves its bytes when
//!   it is accepted, before any work starts, so a full cache stops new
//!   requests instead of letting the worker queue pin more IR.
//! - Counts: traces in the table, versions per loop header, and rejected
//!   sites (the negative cache, see [`super::policy`]).
//!
//! At a byte cap the cache evicts the least recently entered trace that is
//! not running, and keeps going until the new trace fits. If it still does
//! not fit, the new trace is dropped and its site backs off. Evicted work
//! costs only a later recording.
//!
//! # Lifetime
//!
//! Native trace code never calls back into the VM, so nothing can flush or
//! evict while one runs. The cache still states the rule instead of relying
//! on it: [`TraceCache::begin_run`] pins the running trace, eviction skips
//! it, and a flush that arrives meanwhile is deferred to
//! [`TraceCache::end_run`]. Code is freed only when the outermost run has
//! returned.

use super::native::InstalledTrace;

/// Executable bytes the cache may hold.
pub(crate) const MAX_TRACE_CODE_BYTES: usize = 32 * 1024 * 1024;
/// IR and plan bytes the cache may hold, installed and in flight. A tree of
/// Mineclonia's map generation holds 3.5 to 6 MB; at 16 MB the emerge
/// evicted productive trees for room (see "Mineshaft corridors" in the
/// trace module documentation).
pub(crate) const MAX_TRACE_META_BYTES: usize = 48 * 1024 * 1024;
/// Compiled versions one loop header may keep, one per entry type signature.
pub(crate) const MAX_TRACE_VERSIONS: usize = 4;
/// Most compiled traces one VM keeps.
pub(crate) const MAX_INSTALLED_TRACES: usize = 1_000;
/// The most traces `jit.opt.start('maxtrace=..')` may allow.
pub(crate) const MAX_CONFIGURED_TRACES: usize = 1 << 20;
/// The least and most executable bytes `jit.opt.start('maxmcode=..')` may allow.
pub(crate) const MIN_CONFIGURED_CODE_BYTES: usize = 64 * 1024;
pub(crate) const MAX_CONFIGURED_CODE_BYTES: usize = 1 << 30;
/// Rejected sites kept as the negative cache.
pub(crate) const MAX_NEGATIVE_SITES: usize = 256;
/// Executable memory is allocated in pages.
const CODE_PAGE: usize = 4096;

/// The caps of a cache. Tests lower them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheLimits {
    pub(crate) code_bytes: usize,
    pub(crate) meta_bytes: usize,
    pub(crate) traces: usize,
    pub(crate) versions: usize,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            code_bytes: MAX_TRACE_CODE_BYTES,
            meta_bytes: MAX_TRACE_META_BYTES,
            traces: MAX_INSTALLED_TRACES,
            versions: MAX_TRACE_VERSIONS,
        }
    }
}

/// Bytes of executable memory a trace of `code_size` machine-code bytes
/// occupies.
pub(crate) const fn code_charge(code_size: usize) -> usize {
    code_size.div_ceil(CODE_PAGE) * CODE_PAGE
}

/// Owns the installed traces and their byte accounting.
#[derive(Debug, Default)]
pub(crate) struct TraceCache {
    entries: Vec<Option<Box<InstalledTrace>>>,
    live: usize,
    code_bytes: usize,
    meta_bytes: usize,
    /// IR bytes of requests the broker holds, charged against the metadata
    /// cap until they finish.
    reserved_meta: usize,
    /// Orders entries by last use.
    tick: u64,
    /// Identifies a trace across slot reuse.
    next_serial: u64,
    /// The trace native code is running, if any.
    running: Option<u32>,
    /// A flush arrived during a run.
    clear_deferred: bool,
    /// Traces a flush took out of the table during a run. Their code stays
    /// alive until the run returns, and nothing installed later joins them.
    // Deliberate boxing: a flush retires whole traces without moving them.
    #[allow(clippy::vec_box)]
    retired: Vec<Box<InstalledTrace>>,
    limits: CacheLimits,
}

impl TraceCache {
    pub(super) fn profile_entries(&self) -> impl Iterator<Item = &InstalledTrace> {
        self.entries.iter().filter_map(|entry| entry.as_deref())
    }
    pub(crate) const fn limits(&self) -> CacheLimits {
        self.limits
    }

    /// The most versions of one header the cache keeps.
    #[inline]
    pub(crate) const fn version_limit(&self) -> usize {
        self.limits.versions
    }

    pub(crate) const fn set_limits(&mut self, limits: CacheLimits) {
        self.limits = limits;
    }

    /// Installed traces.
    pub(crate) const fn live(&self) -> usize {
        self.live
    }

    #[cfg(test)]
    pub(crate) const fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    pub(crate) const fn meta_bytes(&self) -> usize {
        self.meta_bytes
    }

    #[cfg(test)]
    pub(crate) const fn reserved_meta(&self) -> usize {
        self.reserved_meta
    }

    pub(crate) fn get(&self, index: u32) -> Option<&InstalledTrace> {
        self.entries.get(index as usize)?.as_deref()
    }

    pub(crate) fn get_mut(&mut self, index: u32) -> Option<&mut InstalledTrace> {
        self.entries.get_mut(index as usize)?.as_deref_mut()
    }

    /// Marks a trace as used now, for eviction order.
    pub(crate) fn touch(&mut self, index: u32) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(entry) = self.get_mut(index) {
            entry.last_used = tick;
        }
    }

    /// Charges `bytes` even past the metadata cap. The caller evicts the
    /// traces [`Self::plan_eviction`] named once the request is accepted, and
    /// releases the charge when the request ends either way.
    pub(crate) fn reserve_meta_over_cap(&mut self, bytes: usize) {
        self.reserved_meta += bytes;
    }

    pub(crate) fn release_reserved(&mut self, bytes: usize) {
        self.reserved_meta = self.reserved_meta.saturating_sub(bytes);
    }

    /// The traces to evict so that one charging `code` and `meta` bytes
    /// fits, least recently used first. `new_slot` says the trace also needs
    /// a table slot. Skips the running trace and every index in `keep`.
    /// `Some(vec![])` means it fits already, `None` that eviction cannot
    /// make room.
    pub(crate) fn plan_eviction(
        &self,
        code: usize,
        meta: usize,
        new_slot: bool,
        keep: &[u32],
    ) -> Option<Vec<u32>> {
        if code > self.limits.code_bytes || meta > self.limits.meta_bytes {
            return None;
        }
        let mut victims = Vec::new();
        let (mut freed_code, mut freed_meta) = (0, 0);
        let mut live = self.live;
        let mut candidates: Vec<(u64, u32)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let index = u32::try_from(index).ok()?;
                let entry = entry.as_deref()?;
                (self.running != Some(index) && !keep.contains(&index))
                    .then_some((entry.last_used, index))
            })
            .collect();
        candidates.sort_unstable();
        let mut candidates = candidates.into_iter();
        loop {
            let slots_ok = !new_slot || live < self.limits.traces;
            if slots_ok
                && self.code_bytes - freed_code + code <= self.limits.code_bytes
                && self.meta_bytes - freed_meta + self.reserved_meta + meta
                    <= self.limits.meta_bytes
            {
                return Some(victims);
            }
            let (_, index) = candidates.next()?;
            let entry = self.get(index)?;
            freed_code += entry.code_charge();
            freed_meta += entry.meta_charge();
            live -= 1;
            victims.push(index);
        }
    }

    /// Stores a trace and charges it. The caller made room first.
    pub(crate) fn insert(&mut self, mut trace: Box<InstalledTrace>) -> Option<u32> {
        let slot = match self.entries.iter().position(Option::is_none) {
            Some(free) => free,
            None if self.entries.len() < self.limits.traces.max(1) => {
                self.entries.push(None);
                self.entries.len() - 1
            }
            None => return None,
        };
        self.tick += 1;
        self.next_serial += 1;
        trace.last_used = self.tick;
        trace.serial = self.next_serial;
        self.code_bytes += trace.code_charge();
        self.meta_bytes += trace.meta_charge();
        self.live += 1;
        self.entries[slot] = Some(trace);
        u32::try_from(slot).ok()
    }

    /// Removes a trace and gives back its charge.
    pub(crate) fn remove(&mut self, index: u32) -> Option<Box<InstalledTrace>> {
        if self.running == Some(index) {
            return None;
        }
        let trace = self.entries.get_mut(index as usize)?.take()?;
        self.code_bytes -= trace.code_charge();
        self.meta_bytes -= trace.meta_charge();
        self.live -= 1;
        Some(trace)
    }

    /// Replaces the code of a trace, adjusting the code charge. The caller
    /// checked the new charge fits.
    pub(crate) fn recharge_code(&mut self, old: usize, new: usize) {
        self.code_bytes = self.code_bytes - old + new;
    }

    /// Replaces the metadata of a trace, adjusting the metadata charge. The
    /// caller checked the new charge fits.
    pub(crate) fn recharge_meta(&mut self, old: usize, new: usize) {
        self.meta_bytes = self.meta_bytes - old + new;
    }

    /// Whether native code of the trace at `index` is running.
    pub(crate) fn is_running(&self, index: u32) -> bool {
        self.running == Some(index)
    }

    /// Pins `index` as the running trace.
    pub(crate) fn begin_run(&mut self, index: u32) {
        self.running = Some(index);
    }

    /// Ends the run. Returns the traces a deferred flush now drops, so the
    /// caller frees code only after the run returned.
    #[allow(
        clippy::vec_box,
        reason = "the traces already live in boxes; moving them out keeps their code in place"
    )]
    pub(crate) fn end_run(&mut self) -> Vec<Box<InstalledTrace>> {
        self.running = None;
        if self.clear_deferred {
            self.clear_deferred = false;
            return std::mem::take(&mut self.retired);
        }
        Vec::new()
    }

    /// Drops every trace, or defers that to [`Self::end_run`] while one
    /// runs. The caller drops what this returns.
    #[allow(
        clippy::vec_box,
        reason = "the traces already live in boxes; moving them out keeps their code in place"
    )]
    pub(crate) fn clear(&mut self) -> Vec<Box<InstalledTrace>> {
        self.reserved_meta = 0;
        let taken = self.take_all();
        if self.running.is_some() {
            // Out of the table now, so a later install can never land in the
            // batch that is about to be dropped. The running trace's code
            // lives until `end_run`.
            self.clear_deferred = true;
            self.retired.extend(taken);
            return Vec::new();
        }
        taken
    }

    /// Traces a deferred flush holds until the run returns.
    #[cfg(test)]
    pub(crate) const fn retired_len(&self) -> usize {
        self.retired.len()
    }

    /// Whether a flush waits for the running trace to return.
    #[cfg(test)]
    pub(crate) const fn clear_deferred(&self) -> bool {
        self.clear_deferred
    }

    #[allow(
        clippy::vec_box,
        reason = "the traces already live in boxes; moving them out keeps their code in place"
    )]
    fn take_all(&mut self) -> Vec<Box<InstalledTrace>> {
        self.code_bytes = 0;
        self.meta_bytes = 0;
        self.live = 0;
        self.entries.drain(..).flatten().collect()
    }

    /// Indexes of the installed traces, for tests.
    #[cfg(test)]
    pub(crate) fn installed_indexes(&self) -> Vec<u32> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.as_ref().and_then(|_| u32::try_from(index).ok()))
            .collect()
    }

    /// Indexes of the traces whose chunk is gone.
    pub(crate) fn dead_indexes(&self) -> Vec<u32> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry
                    .as_deref()
                    .filter(|trace| !trace.chunk_alive())
                    .and_then(|_| u32::try_from(index).ok())
            })
            .collect()
    }
}
