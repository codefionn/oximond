//! Shared key lists for tables built by one constructor site.

use crate::runtime::table_storage::key::TableKey;
use crate::string::LuaString;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Most shapes that [`TableShape::pin`] keeps alive for good.
const MAX_PINNED_SHAPES: usize = 1024;
static PINNED_SHAPES: AtomicUsize = AtomicUsize::new(0);

/// Most keys a shape holds, matching the small ordered table limit.
pub const MAX_SHAPE_KEYS: usize = 8;

/// An immutable, sorted list of distinct string keys with their cached
/// hashes. Tables created from one constructor site share a single `Arc` of
/// it and keep only a value slot per key, so filling a record field neither
/// clones the key's string nor frees it again with the table.
///
/// The keys sort in [`TableKey`] order, so a shaped table iterates exactly as
/// the ordered representation would.
#[derive(Debug)]
pub struct TableShape {
    len: usize,
    /// Names this shape for the interpreter's inline caches. It comes from
    /// the counter that numbers hashed layouts, so no layout and no other
    /// shape has the same number, and unlike the shape's address it is never
    /// reused after the shape is freed.
    pub(crate) id: u64,
    // Inline, not boxed: a field read on a shaped table loads the shape
    // through the table, and every extra pointer there is a dependent load.
    // Slots past `len` are padding that lookups never reach.
    hashes: [u64; MAX_SHAPE_KEYS],
    keys: [TableKey; MAX_SHAPE_KEYS],
    /// Set once by [`Self::pin`], before any table holds the shape.
    pinned: std::sync::atomic::AtomicBool,
}

impl TableShape {
    /// Builds a shape from `keys`, sorting them and dropping duplicates.
    /// Returns `None` when no key is given or more than [`MAX_SHAPE_KEYS`]
    /// distinct keys are.
    #[must_use]
    pub fn new(keys: impl IntoIterator<Item = LuaString>) -> Option<Arc<Self>> {
        let mut keys: Vec<TableKey> = keys.into_iter().map(|key| TableKey::string(&key)).collect();
        keys.sort();
        keys.dedup();
        if keys.is_empty() || keys.len() > MAX_SHAPE_KEYS {
            return None;
        }
        let len = keys.len();
        let mut hashes = [0; MAX_SHAPE_KEYS];
        for (slot, key) in hashes.iter_mut().zip(&keys) {
            *slot = match key {
                TableKey::String(string) => string.cached_hash(),
                _ => unreachable!("shape keys are strings"),
            };
        }
        keys.resize(MAX_SHAPE_KEYS, TableKey::Boolean(false));
        let keys = <[TableKey; MAX_SHAPE_KEYS]>::try_from(keys)
            .unwrap_or_else(|_| unreachable!("resized to the maximum"));
        Some(Arc::new(Self {
            len,
            id: crate::runtime::table_storage::entries::next_layout_epoch(),
            hashes,
            keys,
            pinned: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    /// [`Self::new`] for keys given as bytes.
    #[must_use]
    pub fn from_bytes<'a>(keys: impl IntoIterator<Item = &'a [u8]>) -> Option<Arc<Self>> {
        Self::new(keys.into_iter().map(LuaString::from))
    }

    /// Keeps the shape alive for the rest of the process, so that the tables
    /// built over it need not count themselves in its reference count: an
    /// atomic increment per table made and another per table freed. Only the
    /// first [`MAX_PINNED_SHAPES`] shapes are pinned, which bounds what a
    /// program that makes shapes without end can leave behind. Call it
    /// before the first table is made over the shape; returns whether it
    /// pinned.
    pub fn pin(self: &Arc<Self>) -> bool {
        if self.is_pinned()
            || PINNED_SHAPES
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                    (count < MAX_PINNED_SHAPES).then_some(count + 1)
                })
                .is_err()
        {
            return false;
        }
        // The leaked count is the one every table over this shape borrows.
        std::mem::forget(Arc::clone(self));
        self.pinned.store(true, Ordering::Release);
        true
    }

    /// Whether [`Self::pin`] pinned this shape.
    #[inline]
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Relaxed)
    }

    /// The slot of the string `key` in tables of this shape.
    #[must_use]
    pub fn slot_of(&self, key: &[u8]) -> Option<usize> {
        self.keys().iter().position(
            |stored| matches!(stored, TableKey::String(stored) if stored.as_bytes() == key),
        )
    }

    /// Number of keys.
    #[must_use]
    #[allow(clippy::len_without_is_empty, reason = "a shape is never empty")]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn keys(&self) -> &[TableKey] {
        &self.keys[..self.len]
    }

    #[inline]
    pub(crate) fn hashes(&self) -> &[u64] {
        &self.hashes[..self.len]
    }

    /// Slot of the string `key`, whose cached hash is `hash`. Compares the
    /// hashes inline and confirms by pointer identity; a hit that needs a
    /// byte comparison continues out of line, so the inlined copies in the
    /// VM's field reads stay small.
    #[allow(clippy::inline_always, unsafe_code)]
    #[inline(always)]
    pub(crate) fn find_string(&self, key: &LuaString, hash: u64) -> Option<usize> {
        for index in 0..self.len {
            // SAFETY: `index < len <= MAX_SHAPE_KEYS`.
            let (stored, key_slot) = unsafe {
                (
                    *self.hashes.get_unchecked(index),
                    self.keys.get_unchecked(index),
                )
            };
            if stored == hash {
                if let TableKey::String(stored) = key_slot
                    && stored.ptr_eq(key)
                {
                    return Some(index);
                }
                return self.find_string_confirm(index, key, hash);
            }
        }
        None
    }

    /// Continues [`Self::find_string`] from a hash match at `start` whose key
    /// is not the same allocation: compares bytes, then goes on in case the
    /// hash is shared by another key.
    #[inline(never)]
    fn find_string_confirm(&self, start: usize, key: &LuaString, hash: u64) -> Option<usize> {
        (start..self.len).find(|&index| {
            self.hashes[index] == hash
                && matches!(&self.keys[index], TableKey::String(stored) if stored == key)
        })
    }

    #[inline]
    pub(crate) fn position(&self, key: &TableKey) -> Result<usize, usize> {
        self.keys[..self.len].binary_search_by(|stored| stored.cmp(key))
    }
}
