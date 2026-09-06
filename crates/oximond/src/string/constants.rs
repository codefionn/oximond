//! One shared allocation per distinct short string constant, process-wide.
//!
//! Strings are not interned, so two equal strings are usually two
//! allocations, and a table lookup whose key is not the stored key's
//! allocation confirms the match by comparing bytes. Most keys a program
//! looks up are constants, though: field names in compiled code and the C
//! literals a host passes to `lua_getfield`. The compiler already shares one
//! string per constant within a compilation unit; this pool extends that
//! across units, runtimes and the C interface, so a field written by one
//! file or by the host and read by another file matches by identity.
//!
//! The pool holds weak references, so a pooled string dies with its last
//! user. Dead entries are swept once the pool has doubled since its last
//! sweep. Only strings stored inline (at most
//! [`INLINE_STRING_BYTES`](super::INLINE_STRING_BYTES) bytes) are pooled:
//! field names are short, and the pool's memory stays small.

use super::{INLINE_STRING_BYTES, LuaString, WeakLuaString};
use std::sync::{Mutex, PoisonError};

struct Pool {
    entries: hashbrown::HashTable<(u64, WeakLuaString)>,
    /// The entry count after the last sweep of dead entries.
    swept: usize,
}

/// The fewest entries a sweep waits for.
const SWEEP_FLOOR: usize = 4096;

/// The pool's first allocation, in entries.
const INITIAL_CAPACITY: usize = 1024;

static POOL: Mutex<Pool> = Mutex::new(Pool {
    entries: hashbrown::HashTable::new(),
    swept: 0,
});

/// The shared string with `bytes`, whose cached hash is `hash`, built and
/// pooled only when the pool has no live string with these bytes. Longer
/// strings are built unshared.
pub(crate) fn shared_bytes(bytes: &[u8], hash: u64) -> LuaString {
    debug_assert_eq!(hash, super::cached_hash_bytes(bytes));
    if bytes.len() > INLINE_STRING_BYTES {
        return LuaString::new(bytes);
    }
    let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = pool.find(hash, |string| string.as_bytes() == bytes) {
        return found;
    }
    let string = LuaString::with_hash(bytes, hash);
    pool.insert(hash, &string);
    string
}

/// [`shared_bytes`] for a host that names a key by its bytes, such as a C
/// literal passed to `lua_getfield`.
#[must_use]
pub fn shared_key(bytes: &[u8]) -> LuaString {
    shared_bytes(bytes, super::cached_hash_bytes(bytes))
}

/// The shared string equal to `string`: a pooled one if the pool holds one,
/// else `string` itself, which is pooled from now on. Longer strings come
/// back as they are.
pub(crate) fn shared(string: &LuaString) -> LuaString {
    if string.len() > INLINE_STRING_BYTES {
        return string.clone();
    }
    let hash = string.cached_hash();
    let mut pool = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = pool.find(hash, |pooled| pooled == string) {
        return found;
    }
    pool.insert(hash, string);
    string.clone()
}

impl Pool {
    fn find(&self, hash: u64, mut same: impl FnMut(&LuaString) -> bool) -> Option<LuaString> {
        let mut found = None;
        // Upgrading inside the probe keeps a dead entry from matching: its
        // bytes are gone with its last strong reference.
        let _ = self.entries.find(hash, |(stored, weak)| {
            if *stored != hash {
                return false;
            }
            match weak.upgrade() {
                Some(string) if same(&string) => {
                    found = Some(string);
                    true
                }
                _ => false,
            }
        });
        found
    }

    fn insert(&mut self, hash: u64, string: &LuaString) {
        if self.entries.capacity() == 0 {
            // A runtime pools its library names at startup, several hundred
            // of them; growing there from empty would rehash them eight times.
            self.entries.reserve(INITIAL_CAPACITY, |(hash, _)| *hash);
        }
        // A string pooled again after its last user dropped it takes over a
        // dead entry with its hash. Without this, a key that dies and comes
        // back on every host cache miss would pile up dead entries that
        // every later lookup of that hash upgrades and rejects.
        if let Some(entry) = self
            .entries
            .find_mut(hash, |(stored, weak)| *stored == hash && !weak.is_alive())
        {
            entry.1 = string.downgrade();
            return;
        }
        if self.entries.len() >= (2 * self.swept).max(SWEEP_FLOOR) {
            self.entries.retain(|(_, weak)| weak.is_alive());
            self.swept = self.entries.len();
        }
        self.entries
            .insert_unique(hash, (hash, string.downgrade()), |(hash, _)| *hash);
    }
}

#[cfg(test)]
mod tests {
    use super::{shared, shared_bytes};
    use crate::string::{LuaString, cached_hash_bytes};

    #[test]
    fn equal_constants_share_one_allocation() {
        let name = b"shared_constant_test_a";
        let first = shared_bytes(name, cached_hash_bytes(name));
        let second = shared_bytes(name, cached_hash_bytes(name));
        assert!(first.ptr_eq(&second));
        let built = LuaString::new(name);
        assert!(shared(&built).ptr_eq(&first));
        assert!(!built.ptr_eq(&first));
    }

    #[test]
    fn a_dead_constant_is_not_returned() {
        let name = b"shared_constant_test_b";
        let first = shared_bytes(name, cached_hash_bytes(name));
        let identity = first.identity();
        let weak = first.downgrade();
        drop(first);
        assert!(!weak.is_alive());
        let second = shared_bytes(name, cached_hash_bytes(name));
        assert_eq!(second.as_bytes(), name);
        // The weak entry still holds the old allocation, so the new string
        // cannot reuse its address.
        assert_ne!(second.identity(), identity);
        assert!(shared(&LuaString::new(name)).ptr_eq(&second));
    }

    #[test]
    fn a_constant_pooled_again_reuses_its_dead_entry() {
        let name = b"shared_constant_test_c";
        let hash = cached_hash_bytes(name);
        let entries = || {
            let pool = super::POOL.lock().unwrap();
            pool.entries
                .iter()
                .filter(|(stored, _)| *stored == hash)
                .count()
        };
        drop(shared_bytes(name, hash));
        let after_first = entries();
        for _ in 0..100 {
            drop(shared_bytes(name, hash));
        }
        assert_eq!(entries(), after_first);
    }

    #[test]
    fn long_strings_are_not_pooled() {
        let text = [b'x'; 64];
        let first = shared_bytes(&text, cached_hash_bytes(&text));
        let second = shared_bytes(&text, cached_hash_bytes(&text));
        assert!(!first.ptr_eq(&second));
        assert_eq!(first, second);
    }
}
