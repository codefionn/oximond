//! Cache for string-keyed reads that an `__index` chain answers.
//!
//! `object.method` with `method` stored in the class reaches the class through
//! the object's metatable. The cache is keyed on the metatable and the key
//! string, not on the site or the receiver, so every object of a class and every
//! site reading the same name share one entry. An entry says where the
//! chain below that metatable ends: the slot of the value in the table that
//! holds the key, or that no table does.
//!
//! The receiver's own lack of the key is the caller's check (the raw read that
//! precedes every inherited read), and the receiver's metatable is compared
//! on every use. Everything below the metatable is covered by one epoch.
//! Each table the walk passes through (every metatable and every table on the
//! way, but not the table holding the value) is marked `watched`. A mutable
//! borrow of a watched table, a store over its table-valued fields, and the
//! collector clearing its weak entries all advance [`Runtime::inherit_epoch`], which
//! drops every entry at once. The table holding the value is not watched: its
//! slot is validated on each use by the same guard the per-site caches use, so
//! overwriting a field of a class stays cheap.

use crate::runtime::runtime::Runtime;
use crate::runtime::value::{ClosureHandle, TableHandle, Value};
use crate::string::LuaString;

/// Entries in the cache. A power of two.
const ENTRIES: usize = 256;

/// Tables a recorded walk may pass through. A longer chain is not cached.
const MAX_CACHED_HOPS: usize = 16;

/// The hop budget of [`Runtime::table_index_walk`].
const MAX_WALK_HOPS: usize = 100;

#[derive(Clone)]
enum Target {
    /// The chain cannot be answered by a table slot (a handler function, a
    /// loop, a long chain): the full walk decides.
    Walk,
    /// The metatable's own `__index` is this plain closure, which the miss
    /// calls with the receiver. A closure further down the chain is called
    /// with another table, so it is `Walk`.
    Handler(ClosureHandle),
    /// No table of the chain holds the key.
    Absent,
    /// The key is in `table`; `guard` and `data` come from
    /// `TableEntries::site_fill`.
    Slot {
        table: TableHandle,
        guard: u64,
        data: usize,
    },
}

/// One cache entry. An `epoch` of zero is an empty entry.
#[derive(Clone)]
pub(crate) struct InheritEntry {
    epoch: u64,
    metatable: Option<TableHandle>,
    /// The key string. Held so that its address names no other string while
    /// the entry lives.
    key: Option<LuaString>,
    /// For an ordered table, the key string of the table holding the value,
    /// whose address `Target::Slot::data` is.
    #[allow(dead_code, reason = "held to keep the key string alive, never read")]
    pin: Option<LuaString>,
    target: Target,
}

impl Default for InheritEntry {
    fn default() -> Self {
        Self {
            epoch: 0,
            metatable: None,
            key: None,
            pin: None,
            target: Target::Walk,
        }
    }
}

#[inline(always)]
fn entry_index(metatable: TableHandle, key: &LuaString) -> usize {
    let mixed = (key.identity() >> 4) ^ (metatable.0.slot as usize).wrapping_mul(0x9e37);
    mixed & (ENTRIES - 1)
}

impl Runtime {
    /// The value that `table[key]` reads through `table`'s metatable chain,
    /// when a valid cache entry knows it. `None` means the entry is absent,
    /// stale, or does not apply, and the caller walks the chain.
    ///
    /// The caller has just read `key` raw from `table` and found it absent
    /// (or nil), and `table` has a metatable. This does not repeat that read.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_inherited_get<'a>(
        &'a self,
        table: TableHandle,
        key: &LuaString,
    ) -> Option<&'a Value> {
        static NIL: Value = Value::Nil;
        let metatable = self.tables.get(table.0)?.metatable?;
        let entry = self.inherit_cache.get(entry_index(metatable, key))?;
        if entry.epoch != self.inherit_epoch
            || entry.metatable != Some(metatable)
            || !entry.key.as_ref().is_some_and(|stored| stored.ptr_eq(key))
        {
            return None;
        }
        match entry.target {
            Target::Absent => Some(&NIL),
            Target::Slot { table, guard, data } => {
                self.tables.get(table.0)?.entries.site_value(guard, data)
            }
            Target::Walk | Target::Handler(_) => None,
        }
    }

    /// The `__index` closure that `table[key]` calls after a raw miss, when a
    /// valid cache entry knows that the receiver's metatable has it
    /// directly. `None` means the entry is absent, stale or says something
    /// else, and the caller looks the handler up itself.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub fn table_inherited_handler(
        &self,
        table: TableHandle,
        key: &LuaString,
    ) -> Option<ClosureHandle> {
        let metatable = self.tables.get(table.0)?.metatable?;
        let entry = self.inherit_cache.get(entry_index(metatable, key))?;
        if entry.epoch != self.inherit_epoch
            || entry.metatable != Some(metatable)
            || !entry.key.as_ref().is_some_and(|stored| stored.ptr_eq(key))
        {
            return None;
        }
        match entry.target {
            Target::Handler(closure) => Some(closure),
            _ => None,
        }
    }

    /// [`Self::table_inherited_get`] that records the chain on an entry that
    /// does not answer. Returns the value if the chain is cacheable and
    /// found one, or a nil for a key found nowhere. `None` sends the caller
    /// to the full walk: handler functions, loops, long chains, and a slot
    /// that holds nil.
    pub fn table_inherited_resolve(
        &mut self,
        table: TableHandle,
        key: &LuaString,
    ) -> Option<&Value> {
        let metatable = self.tables.get(table.0)?.metatable?;
        let index = entry_index(metatable, key);
        if self.inherit_cache.is_empty() {
            self.inherit_cache = vec![InheritEntry::default(); ENTRIES];
        }
        let entry = &self.inherit_cache[index];
        let current = entry.epoch == self.inherit_epoch
            && entry.metatable == Some(metatable)
            && entry.key.as_ref().is_some_and(|stored| stored.ptr_eq(key));
        if current && matches!(entry.target, Target::Walk | Target::Handler(_)) {
            return None;
        }
        let (target, pin, watch, count) = self.record_chain(metatable, key);
        for handle in &watch[..count] {
            if let Some(table) = self.tables.get_mut(handle.0) {
                table.watched = true;
            }
        }
        self.inherit_cache[index] = InheritEntry {
            epoch: self.inherit_epoch,
            metatable: Some(metatable),
            key: Some(key.clone()),
            pin,
            target,
        };
        self.table_inherited_get(table, key)
    }

    /// Walks the chain below `metatable` as [`Runtime::table_index_walk`]
    /// does after a miss on the receiver, noting the tables it depends on.
    #[allow(clippy::type_complexity)]
    fn record_chain(
        &self,
        metatable: TableHandle,
        key: &LuaString,
    ) -> (
        Target,
        Option<LuaString>,
        [TableHandle; MAX_CACHED_HOPS],
        usize,
    ) {
        let mut watch = [metatable; MAX_CACHED_HOPS];
        let mut count = 0;
        let mut meta = Some(metatable);
        // The receiver was the first hop and is known absent.
        for _ in 0..MAX_WALK_HOPS {
            let Some(handle) = meta else {
                // The previous table has no metatable.
                return (Target::Absent, None, watch, count);
            };
            let Some(current) = self.tables.get(handle.0) else {
                return (Target::Walk, None, watch, count);
            };
            if count == MAX_CACHED_HOPS {
                return (Target::Walk, None, watch, count);
            }
            watch[count] = handle;
            count += 1;
            let next = match current.entries.get_string(&self.index_metamethod_key) {
                None | Some(Value::Nil) => return (Target::Absent, None, watch, count),
                Some(Value::Table(next)) => *next,
                Some(Value::Closure(handler)) if count == 1 => {
                    return (Target::Handler(*handler), None, watch, count);
                }
                Some(_) => return (Target::Walk, None, watch, count),
            };
            let Some(holder) = self.tables.get(next.0) else {
                return (Target::Walk, None, watch, count);
            };
            if holder.entries.get_string_live(key).is_some() {
                // The holder is validated by its slot guard, not watched.
                return match holder.entries.site_fill(key) {
                    Some(fill) => (
                        Target::Slot {
                            table: next,
                            guard: fill.guard,
                            data: fill.data,
                        },
                        fill.pin,
                        watch,
                        count,
                    ),
                    None => (Target::Walk, None, watch, count),
                };
            }
            meta = holder.metatable;
            // `holder` lacks the key, so it is part of the chain: setting the
            // key in it, or a new metatable, changes the answer.
            if count == MAX_CACHED_HOPS {
                return (Target::Walk, None, watch, count);
            }
            watch[count] = next;
            count += 1;
        }
        (Target::Walk, None, watch, count)
    }
}
