#![forbid(unsafe_code)]

//! Immutable byte strings used for Lua strings and string constants.

use std::borrow::Cow;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

pub(crate) mod constants;

pub use constants::shared_key;

/// An immutable, reference-counted Lua byte string.
#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct LuaString(Arc<StringStorage>);

/// Observes a string allocation without keeping its bytes alive.
#[derive(Clone)]
pub struct WeakLuaString(Weak<StringStorage>);

impl WeakLuaString {
    #[must_use]
    pub fn upgrade(&self) -> Option<LuaString> {
        self.0.upgrade().map(LuaString)
    }

    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.0.strong_count() != 0
    }

    #[must_use]
    pub fn identity(&self) -> usize {
        self.0.as_ptr() as usize
    }
}

// A sized allocation keeps the shared handle one word wide. In particular,
// embedding a string in a VM value no longer needs a slice's length word.
//
// `repr(C)` fixes where the bytes and the hash sit, because native trace code
// reads them (see `NATIVE_STRING_HASH_OFFSET`).
#[repr(C)]
struct StringStorage {
    bytes: StringBytes,
    /// A hash of the bytes, computed once. Table lookups and other
    /// hash-based consumers reuse it instead of rehashing the bytes on every
    /// operation. Equal bytes always produce an equal cached hash. A string
    /// hashes on first use, not at construction: most strings that programs
    /// build (concatenation results, formatted text) are never table keys,
    /// and hashing each one costs a pass over its bytes. Zero means not
    /// computed yet.
    hash: AtomicU64,
}

/// The longest string stored inside its shared allocation.
const INLINE_CAPACITY: usize = 22;

/// The longest string stored inside its allocation instead of beside it.
pub const INLINE_STRING_BYTES: usize = INLINE_CAPACITY;

/// String bytes, stored inline when short. Most Lua strings are short
/// (field names, formatted numbers, separators), and keeping them inline
/// halves the allocations made per created string.
///
/// `repr(u8)` fixes the layout as a union of `repr(C)` structs that start with
/// the one-byte variant tag: an inline string is the tag (0), its length and
/// its bytes padded with zeros, in 24 bytes; a heap string is the tag (1) and
/// then the box at offset 8. Native trace code compares inline strings word by
/// word (see `NATIVE_STRING_INLINE_WORDS`).
#[repr(u8)]
enum StringBytes {
    Inline {
        length: u8,
        bytes: [u8; INLINE_CAPACITY],
    },
    Heap(Box<[u8]>),
}

/// Where native code finds the parts of a string, as byte offsets from the
/// address [`LuaString::identity`] returns (the string's storage). The cached
/// hash is the word at [`NATIVE_STRING_HASH_OFFSET`], zero until computed. A
/// string is inline when the byte at offset 0 is [`NATIVE_STRING_INLINE_TAG`];
/// two inline strings are equal exactly when their first
/// [`NATIVE_STRING_INLINE_WORDS`] words are (the length and the zero-padded
/// bytes follow the tag).
pub const NATIVE_STRING_HASH_OFFSET: usize = std::mem::offset_of!(StringStorage, hash);
pub const NATIVE_STRING_INLINE_TAG: u8 = 0;
pub const NATIVE_STRING_INLINE_WORDS: usize = 3;
const _: () = {
    assert!(std::mem::size_of::<StringBytes>() == 8 * NATIVE_STRING_INLINE_WORDS);
    assert!(std::mem::offset_of!(StringStorage, bytes) == 0);
    assert!(NATIVE_STRING_HASH_OFFSET == 8 * NATIVE_STRING_INLINE_WORDS);
    assert!(INLINE_CAPACITY + 2 == 8 * NATIVE_STRING_INLINE_WORDS);
};

/// The [`NATIVE_STRING_INLINE_WORDS`] words at the start of the storage of
/// a string of `bytes`, as native code reads them, when the string is inline:
/// the tag, the length and the bytes padded with zeros. Every string of these
/// bytes holds these words wherever it was made, and every other string
/// differs in one of them: other bytes, another length, or the tag of a heap
/// string (bytes too long to be inline). `None` for such bytes.
#[must_use]
pub fn native_inline_words(bytes: &[u8]) -> Option<[u64; NATIVE_STRING_INLINE_WORDS]> {
    let (length, packed) = pack_inline(bytes)?;
    let mut raw = [0_u8; 8 * NATIVE_STRING_INLINE_WORDS];
    raw[0] = NATIVE_STRING_INLINE_TAG;
    raw[1] = length;
    raw[2..].copy_from_slice(&packed);
    Some(std::array::from_fn(|word| {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&raw[8 * word..8 * word + 8]);
        u64::from_ne_bytes(bytes)
    }))
}

impl StringBytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Inline { length, bytes } => &bytes[..usize::from(*length)],
            Self::Heap(bytes) => bytes,
        }
    }
}

/// Packs up to eight bytes into a little-endian word without a `memcpy`.
///
/// Byte copies through a stack buffer left the storage's later wide moves
/// waiting on store forwarding; words assembled in registers store whole.
#[inline]
fn pack_word(bytes: &[u8]) -> u64 {
    let length = bytes.len();
    match length {
        0 => 0,
        1..=3 => {
            u64::from(bytes[0])
                | (u64::from(bytes[length / 2]) << (8 * (length / 2)))
                | (u64::from(bytes[length - 1]) << (8 * (length - 1)))
        }
        4..=7 => {
            let low = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let tail = &bytes[length - 4..];
            let high = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
            u64::from(low) | (u64::from(high) << (8 * (length - 4)))
        }
        _ => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

/// The inline form of `bytes`, if they fit.
#[inline]
fn pack_inline(bytes: &[u8]) -> Option<(u8, [u8; INLINE_CAPACITY])> {
    let length = u8::try_from(bytes.len())
        .ok()
        .filter(|&length| usize::from(length) <= INLINE_CAPACITY)?;
    let (first, rest) = bytes.split_at(bytes.len().min(8));
    let (second, third) = rest.split_at(rest.len().min(8));
    let [a0, a1, a2, a3, a4, a5, a6, a7] = pack_word(first).to_le_bytes();
    let [b0, b1, b2, b3, b4, b5, b6, b7] = pack_word(second).to_le_bytes();
    let [c0, c1, c2, c3, c4, c5, _, _] = pack_word(third).to_le_bytes();
    Some((
        length,
        [
            a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5,
        ],
    ))
}

impl StringStorage {
    #[inline]
    fn from_slice(bytes: &[u8]) -> Self {
        match pack_inline(bytes) {
            Some((length, bytes)) => Self::inline(length, bytes),
            None => Self {
                bytes: StringBytes::Heap(Box::from(bytes)),
                hash: AtomicU64::new(0),
            },
        }
    }

    #[inline]
    fn from_vec(bytes: Vec<u8>) -> Self {
        match pack_inline(&bytes) {
            Some((length, bytes)) => Self::inline(length, bytes),
            None => Self {
                bytes: StringBytes::Heap(bytes.into_boxed_slice()),
                hash: AtomicU64::new(0),
            },
        }
    }

    #[inline]
    fn inline(length: u8, bytes: [u8; INLINE_CAPACITY]) -> Self {
        Self {
            bytes: StringBytes::Inline { length, bytes },
            hash: AtomicU64::new(0),
        }
    }

    #[inline]
    fn hash(&self) -> u64 {
        match self.hash.load(Ordering::Relaxed) {
            0 => self.compute_hash(),
            hash => hash,
        }
    }

    /// Hashes a string on first use. A hash of zero is never stored, so that
    /// string rehashes on every call and still agrees with equal bytes.
    #[inline(never)]
    fn compute_hash(&self) -> u64 {
        let hash = cached_hash_bytes(self.bytes.as_slice());
        self.hash.store(hash, Ordering::Relaxed);
        hash
    }
}

impl PartialEq for StringStorage {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // Callers with a hash in hand pre-filter on it (see `known_hash`);
        // comparing the bytes here is cheaper than hashing a long string
        // for one comparison.
        match (&self.bytes, &other.bytes) {
            // Zero padding makes the fixed-size arrays compare exactly like
            // the strings, without a variable-length `memcmp` call.
            (
                StringBytes::Inline { length, bytes },
                StringBytes::Inline {
                    length: other_length,
                    bytes: other_bytes,
                },
            ) => length == other_length && bytes == other_bytes,
            (left, right) => left.as_slice() == right.as_slice(),
        }
    }
}

impl Eq for StringStorage {}

impl PartialOrd for StringStorage {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StringStorage {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (&self.bytes, &other.bytes) {
            (
                StringBytes::Inline { length, bytes },
                StringBytes::Inline {
                    length: other_length,
                    bytes: other_bytes,
                },
            ) => inline_order(bytes, other_bytes).then(length.cmp(other_length)),
            (left, right) => left.as_slice().cmp(right.as_slice()),
        }
    }
}

/// The byte order of two zero-padded inline buffers, as three big-endian
/// words instead of a `memcmp` call. Where the buffers first differ inside
/// both strings, that byte decides, as for the strings. Where they first
/// differ past the end of one string, that string has a padding zero and the
/// other a nonzero byte, and the shorter string is a prefix of the longer
/// one, which orders first; equal buffers leave the order to the lengths.
#[inline]
fn inline_order(left: &[u8; INLINE_CAPACITY], right: &[u8; INLINE_CAPACITY]) -> std::cmp::Ordering {
    let word = |bytes: &[u8; INLINE_CAPACITY], start: usize| {
        let mut word = [0; 8];
        let end = (start + 8).min(INLINE_CAPACITY);
        word[..end - start].copy_from_slice(&bytes[start..end]);
        u64::from_be_bytes(word)
    };
    word(left, 0)
        .cmp(&word(right, 0))
        .then_with(|| word(left, 8).cmp(&word(right, 8)))
        .then_with(|| word(left, 16).cmp(&word(right, 16)))
}

/// Mixes a byte fold step the way `finish` does, so incremental writes and
/// whole-slice hashing agree.
#[inline]
const fn fold(mut state: u64, bytes: &[u8]) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        state = (state ^ bytes[index] as u64).wrapping_mul(0x100_0000_01b3);
        index += 1;
    }
    state
}

/// The last step of the cached hash: folds the high half of the FNV state into
/// the low half and multiplies, so that the high 32 bits of the result depend
/// on every bit of the state. The string index native code probes starts at
/// those 32 bits (`native_probe_start`), read from the string with one load.
/// An FNV state leaves them weak for keys that differ in a number at their
/// end, which then land in runs of the index. The step is a bijection that
/// keeps zero at zero.
#[inline]
const fn finish_hash(state: u64) -> u64 {
    (state ^ (state >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Computes the cached hash for string bytes.
///
/// This is a fixed-seed non-cryptographic hash chosen for speed on the short
/// field-name strings that dominate table traffic. Determinism keeps equal
/// strings interchangeable across runtimes in one process. Long strings mix
/// one FNV round per eight-byte word instead of per byte; short strings keep
/// the classic byte-at-a-time mixing so small keys retain full avalanche.
/// Both end with [`finish_hash`].
#[must_use]
pub const fn cached_hash_bytes(bytes: &[u8]) -> u64 {
    // Sixteen bytes keep the byte loop unrolled-tight for tiny keys while
    // longer concatenation buffers and sources use word mixing.
    if bytes.len() <= 16 {
        return finish_hash(fold(0xcbf2_9ce4_8422_2325, bytes));
    }
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    let mut chunk = 0;
    let words = bytes.len() / 8;
    while chunk < words {
        let base = chunk * 8;
        // Little-endian word assembly keeps the hash identical on every
        // target without unaligned loads, which const contexts forbid.
        let word = (bytes[base] as u64)
            | ((bytes[base + 1] as u64) << 8)
            | ((bytes[base + 2] as u64) << 16)
            | ((bytes[base + 3] as u64) << 24)
            | ((bytes[base + 4] as u64) << 32)
            | ((bytes[base + 5] as u64) << 40)
            | ((bytes[base + 6] as u64) << 48)
            | ((bytes[base + 7] as u64) << 56);
        state = (state ^ word).wrapping_mul(0x100_0000_01b3);
        chunk += 1;
    }
    finish_hash(fold_from(state, bytes, words * 8))
}

/// Folds bytes starting at `start`, for hashes that already mixed a prefix.
#[inline]
const fn fold_from(mut state: u64, bytes: &[u8], mut index: usize) -> u64 {
    while index < bytes.len() {
        state = (state ^ bytes[index] as u64).wrapping_mul(0x100_0000_01b3);
        index += 1;
    }
    state
}

impl LuaString {
    /// Creates a byte string from any byte slice.
    #[must_use]
    pub fn new(bytes: impl AsRef<[u8]>) -> Self {
        Self(Arc::new(StringStorage::from_slice(bytes.as_ref())))
    }

    /// [`Self::new`] for a caller that already holds
    /// [`cached_hash_bytes`]`(bytes)`, which the string keeps instead of
    /// hashing its bytes again on first use.
    pub(crate) fn with_hash(bytes: &[u8], hash: u64) -> Self {
        debug_assert_eq!(hash, cached_hash_bytes(bytes));
        let storage = StringStorage::from_slice(bytes);
        storage.hash.store(hash, Ordering::Relaxed);
        Self(Arc::new(storage))
    }

    /// Creates a string of at most sixteen bytes packed little-endian into
    /// `word` (its first byte is the low byte); bytes past `length` must be
    /// zero.
    ///
    /// Formatters that assemble short text in a register, such as decimal
    /// numbers, avoid staging it in memory: the storage is built from the
    /// word directly.
    ///
    /// # Panics
    ///
    /// Panics if `length` exceeds sixteen.
    #[must_use]
    #[inline]
    pub fn from_packed(word: u128, length: usize) -> Self {
        assert!(length <= 16, "packed strings hold at most sixteen bytes");
        debug_assert!(length == 16 || word >> (8 * length) == 0);
        let [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
        ] = word.to_le_bytes();
        #[allow(clippy::cast_possible_truncation, reason = "at most sixteen")]
        let length = length as u8;
        Self(Arc::new(StringStorage {
            bytes: StringBytes::Inline {
                length,
                bytes: [
                    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, 0, 0, 0,
                    0, 0, 0,
                ],
            },
            hash: AtomicU64::new(0),
        }))
    }

    /// Creates a string from the first `length` bytes of a buffer of the
    /// inline size, with no staging copy. Bytes past `length` must be zero.
    ///
    /// # Panics
    ///
    /// Panics if `length` exceeds [`INLINE_STRING_BYTES`].
    #[must_use]
    #[inline]
    pub fn from_inline_buffer(bytes: [u8; INLINE_STRING_BYTES], length: usize) -> Self {
        assert!(length <= INLINE_STRING_BYTES, "inline strings are short");
        debug_assert!(bytes[length..].iter().all(|&byte| byte == 0));
        #[allow(clippy::cast_possible_truncation, reason = "at most 22")]
        Self(Arc::new(StringStorage::inline(length as u8, bytes)))
    }

    /// Returns the underlying Lua bytes.
    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.bytes.as_slice()
    }

    /// The bytes of a short string followed by the zero byte its inline
    /// storage pads them with, for C callers that need a terminated string
    /// (`lua_tolstring`). The bytes stay where they are for as long as this
    /// string's allocation lives. `None` for a string kept on the heap and
    /// for an inline string that fills its whole buffer.
    #[must_use]
    #[inline]
    pub fn inline_bytes_with_nul(&self) -> Option<&[u8]> {
        match &self.0.bytes {
            StringBytes::Inline { length, bytes } => bytes.get(..=usize::from(*length)),
            StringBytes::Heap(_) => None,
        }
    }

    /// The whole inline buffer of a short string and its length: the bytes,
    /// then zeros up to [`INLINE_STRING_BYTES`]. A caller with room for the
    /// buffer copies it at its fixed size, with no copy loop for a length
    /// only the string knows. `None` for a string kept on the heap.
    #[must_use]
    #[inline]
    pub(crate) fn inline_buffer(&self) -> Option<(&[u8; INLINE_STRING_BYTES], usize)> {
        match &self.0.bytes {
            StringBytes::Inline { length, bytes } => Some((bytes, usize::from(*length))),
            StringBytes::Heap(_) => None,
        }
    }

    /// Returns the number of bytes in this string.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns whether this string has no bytes.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }

    /// Returns an identity token for this string allocation.
    #[must_use]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    #[must_use]
    pub fn downgrade(&self) -> WeakLuaString {
        WeakLuaString(Arc::downgrade(&self.0))
    }

    /// Returns the hash of the bytes, computed on first use for long strings.
    ///
    /// Equal strings have equal cached hashes, so callers may use it as a
    /// fast equality pre-filter and as precomputed hash input.
    #[must_use]
    #[inline]
    pub fn cached_hash(&self) -> u64 {
        self.0.hash()
    }

    /// Compares by value without ever calling out of line: `None` when both
    /// strings are heap strings, whose bytes only a `memcmp` call can
    /// compare. Strings are packed inline up to a fixed length, so an inline
    /// and a heap string are never equal. Hash probes in hot loops use it to
    /// stay free of calls and hand the rare long-string case to a cold path.
    #[must_use]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn eq_without_call(&self, other: &Self) -> Option<bool> {
        match (&self.0.bytes, &other.0.bytes) {
            (
                StringBytes::Inline { length, bytes },
                StringBytes::Inline {
                    length: other_length,
                    bytes: other_bytes,
                },
            ) => Some(length == other_length && bytes == other_bytes),
            (StringBytes::Heap(_), StringBytes::Heap(_)) => None,
            _ => Some(false),
        }
    }

    /// Whether both handles point at the same allocation.
    #[must_use]
    #[inline(always)]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Returns the hash if some caller already computed it, else zero.
    ///
    /// For comparing against a stored key, whose owner called
    /// [`Self::cached_hash`] when it stored the key: the load needs no
    /// fallback path, which keeps scans over small tables tight. A zero
    /// result says nothing about equality.
    #[must_use]
    #[inline]
    pub fn known_hash(&self) -> u64 {
        self.0.hash.load(Ordering::Relaxed)
    }

    /// Returns allocated storage bytes, excluding allocator bookkeeping.
    #[must_use]
    pub fn allocation_size(&self) -> usize {
        let heap = match &self.0.bytes {
            StringBytes::Inline { .. } => 0,
            StringBytes::Heap(bytes) => bytes.len(),
        };
        heap.saturating_add(std::mem::size_of::<StringStorage>())
            .saturating_add(2 * std::mem::size_of::<usize>())
    }

    /// What [`Self::allocation_size`] answers for a string of `length` bytes,
    /// without making one: a run that skips a string it would have made
    /// still owes the collector the same charge.
    #[must_use]
    pub const fn allocation_size_of_length(length: usize) -> usize {
        let heap = if length <= INLINE_CAPACITY { 0 } else { length };
        heap.saturating_add(std::mem::size_of::<StringStorage>())
            .saturating_add(2 * std::mem::size_of::<usize>())
    }

    /// Converts this value to UTF-8 for diagnostics, replacing invalid bytes.
    #[must_use]
    pub fn to_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.as_bytes())
    }
}

impl AsRef<[u8]> for LuaString {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::hash::Hash for LuaString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.cached_hash());
    }
}

impl From<&str> for LuaString {
    fn from(value: &str) -> Self {
        Self::new(value.as_bytes())
    }
}
impl From<String> for LuaString {
    fn from(value: String) -> Self {
        Self::from(value.into_bytes())
    }
}
impl From<Vec<u8>> for LuaString {
    fn from(value: Vec<u8>) -> Self {
        Self(Arc::new(StringStorage::from_vec(value)))
    }
}
impl From<&[u8]> for LuaString {
    fn from(value: &[u8]) -> Self {
        Self::new(value)
    }
}
impl<const LENGTH: usize> From<&[u8; LENGTH]> for LuaString {
    fn from(value: &[u8; LENGTH]) -> Self {
        Self::new(value)
    }
}
impl From<Arc<[u8]>> for LuaString {
    fn from(value: Arc<[u8]>) -> Self {
        Self::new(value.as_ref())
    }
}

impl fmt::Debug for LuaString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("b\"")?;
        for byte in self.as_bytes() {
            for escaped in std::ascii::escape_default(*byte) {
                write!(formatter, "{}", char::from(escaped))?;
            }
        }
        formatter.write_str("\"")
    }
}

impl fmt::Display for LuaString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_lossy())
    }
}

#[cfg(test)]
mod tests {
    use super::LuaString;

    #[test]
    fn comparison_without_a_call_defers_only_long_strings() {
        let short = LuaString::from("short");
        assert_eq!(short.eq_without_call(&LuaString::from("short")), Some(true));
        assert_eq!(
            short.eq_without_call(&LuaString::from("shorT")),
            Some(false)
        );
        let long = LuaString::from("a string past the inline capacity");
        // An inline and a heap string never hold the same bytes.
        assert_eq!(short.eq_without_call(&long), Some(false));
        assert_eq!(long.eq_without_call(&short), Some(false));
        assert_eq!(long.eq_without_call(&long.clone()), None);
        assert_eq!(
            long.eq_without_call(&LuaString::from(long.as_bytes())),
            None
        );
    }

    #[test]
    fn weak_strings_observe_shared_allocation_lifetime() {
        let string = LuaString::from("external string");
        let weak = string.downgrade();
        assert_eq!(weak.identity(), string.identity());
        let shared = weak.upgrade().unwrap();
        drop(string);
        assert!(weak.is_alive());
        drop(shared);
        assert!(!weak.is_alive());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn string_handle_fits_one_machine_word() {
        assert_eq!(
            std::mem::size_of::<LuaString>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn sharing_preserves_identity_but_equality_uses_bytes() {
        use std::hash::BuildHasher;
        let first = LuaString::from(vec![0, 255, b'x']);
        let shared = first.clone();
        let separate = LuaString::new([0, 255, b'x']);
        assert_eq!(first.identity(), shared.identity());
        assert_ne!(first.identity(), separate.identity());
        assert_eq!(first, separate);
        let state = std::collections::hash_map::RandomState::new();
        assert_eq!(state.hash_one(&first), state.hash_one(&separate));
        let empty = LuaString::from("");
        let other_empty = LuaString::from("");
        assert_ne!(empty.identity(), other_empty.identity());
        assert_eq!(empty, other_empty);
    }

    #[test]
    fn inline_and_heap_storage_agree_at_the_boundary() {
        for length in (0..=30).chain([64]) {
            let bytes: Vec<u8> = (0..length)
                .map(|index| (index * 37 % 251 + 1) as u8)
                .collect();
            let from_slice = LuaString::new(&bytes);
            let from_vec = LuaString::from(bytes.clone());
            assert_eq!(from_slice.as_bytes(), bytes.as_slice());
            assert_eq!(from_vec.as_bytes(), bytes.as_slice());
            assert_eq!(from_slice.len(), length);
            assert_eq!(from_slice, from_vec);
            assert_eq!(from_slice.cached_hash(), from_vec.cached_hash());
        }
        let short = LuaString::from("abc\0");
        let long = LuaString::from("abc\0defghijklmnopqrstuvwxyz");
        assert!(short < long);
        assert_ne!(short, long);
        assert!(LuaString::from("b") > LuaString::from("abcdefghijklmnopqrstuvwxyz"));
    }

    #[test]
    fn strings_hash_on_first_use_and_still_compare_equal() {
        let bytes: Vec<u8> = (0..200).map(|index| (index % 251 + 1) as u8).collect();
        let first = LuaString::from(bytes.clone());
        let second = LuaString::new(&bytes);
        // Neither string has been hashed, and comparing does not hash them.
        assert_eq!(first.known_hash(), 0);
        assert_eq!(second.known_hash(), 0);
        assert_eq!(first, second);
        assert_eq!(first.known_hash(), 0);
        let hash = first.cached_hash();
        assert_eq!(hash, super::cached_hash_bytes(&bytes));
        assert_eq!(first.known_hash(), hash);
        // One side hashed, the other not: still equal, and equal hashes.
        assert_eq!(first, second);
        assert_eq!(second.cached_hash(), hash);
        let mut other = bytes.clone();
        other[199] ^= 1;
        let other = LuaString::from(other);
        assert_ne!(first, other);
        assert!(first < other || other < first);
        // Short strings wait for their first use too.
        let short = LuaString::from("short");
        assert_eq!(short.known_hash(), 0);
        assert_eq!(short.cached_hash(), super::cached_hash_bytes(b"short"));
        assert_eq!(short.known_hash(), short.cached_hash());
    }

    #[test]
    fn packed_strings_match_byte_strings() {
        for text in ["", "7", "-0", "100", "-12345", "1234567890123456"] {
            let mut word = 0_u128;
            for (index, byte) in text.bytes().enumerate() {
                word |= u128::from(byte) << (8 * index);
            }
            let packed = LuaString::from_packed(word, text.len());
            let plain = LuaString::from(text);
            assert_eq!(packed.as_bytes(), text.as_bytes());
            assert_eq!(packed.cached_hash(), plain.cached_hash());
            assert_eq!(packed, plain);
        }
    }

    #[test]
    fn stores_arbitrary_bytes_and_orders_them_unsigned() {
        let invalid = LuaString::from(vec![b'a', 0xff]);
        assert_eq!(invalid.as_bytes(), b"a\xff");
        assert_eq!(invalid.len(), 2);
        assert_eq!(format!("{invalid:?}"), "b\"a\\xff\"");
        assert_eq!(invalid.to_lossy(), "a\u{fffd}");
        assert!(LuaString::from(b"\x7f" as &[u8]) < LuaString::from(b"\x80" as &[u8]));
    }
}

/// The UTF-8 decoding of the `utf8` library, shared by the library's natives
/// and the helpers a trace calls in their place.
pub mod utf8 {
    /// How strict a decoder is.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Utf8Mode {
        /// Lua 5.3: surrogates and code points up to 0x10FFFF.
        Lua53,
        /// Lua 5.4 and later: no surrogates, nothing past 0x10FFFF.
        Strict,
        /// The lax flag: sequences of up to six bytes, up to 0x7FFFFFFF.
        Lax,
    }

    /// Lua 5.3 permits surrogates; 5.4 adds strict Unicode validation and an
    /// explicit lax mode accepting sequences up to six bytes. Gives the
    /// position after the character at `position` and its code point, or
    /// `None` for an invalid sequence (or none: the end of `source`).
    ///
    /// Inlined, so that a loop over a string (`utf8.len`, a `utf8.codes`
    /// step) reads an ASCII byte without a call.
    #[must_use]
    #[inline]
    pub fn decode(source: &[u8], position: usize, mode: Utf8Mode) -> Option<(usize, u32)> {
        let first = *source.get(position)?;
        if first < 0x80 {
            return Some((position + 1, u32::from(first)));
        }
        decode_wide(source, position, first, mode)
    }

    /// [`decode`] of a character that starts with the non-ASCII byte `first`.
    #[inline(never)]
    fn decode_wide(
        source: &[u8],
        position: usize,
        first: u8,
        mode: Utf8Mode,
    ) -> Option<(usize, u32)> {
        let width = first.leading_ones() as usize;
        if !(2..=6).contains(&width) {
            return None;
        }
        let mut code = u32::from(first & (0x7f >> width));
        for index in 1..width {
            let byte = *source.get(position + index)?;
            if byte & 0xc0 != 0x80 {
                return None;
            }
            code = (code << 6) | u32::from(byte & 0x3f);
        }
        let minimum = [0, 0, 0x80, 0x800, 0x10000, 0x20_0000, 0x400_0000][width];
        if code < minimum || code > 0x7fff_ffff {
            return None;
        }
        if mode != Utf8Mode::Lax && code > 0x10_ffff {
            return None;
        }
        if mode == Utf8Mode::Strict && (0xd800..=0xdfff).contains(&code) {
            return None;
        }
        Some((position + width, code))
    }

    /// Whether the byte at `position` is a continuation byte.
    #[must_use]
    #[inline]
    pub fn is_continuation(source: &[u8], position: usize) -> bool {
        source.get(position).is_some_and(|byte| byte & 0xc0 == 0x80)
    }

    /// What a step of a `utf8.codes` loop gives.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CodesStep {
        /// The loop ends.
        End,
        /// The next character: its one-based position and its code point.
        Character { position: usize, code: u32 },
        /// An invalid sequence, which the step raises an error for.
        Invalid,
    }

    /// The step of a `utf8.codes` loop over `source` after the character at
    /// `index` (the previous position, zero at the start).
    #[must_use]
    pub fn codes_step(source: &[u8], index: isize, mode: Utf8Mode) -> CodesStep {
        let mut position = if index < 0 {
            if mode == Utf8Mode::Lua53 {
                0
            } else {
                return CodesStep::End;
            }
        } else {
            usize::try_from(index).unwrap_or(usize::MAX)
        };
        while (mode != Utf8Mode::Lua53 || index > 0)
            && position < source.len()
            && is_continuation(source, position)
        {
            position += 1;
        }
        if position >= source.len() {
            return CodesStep::End;
        }
        let Some((next, code)) = decode(source, position, mode) else {
            return CodesStep::Invalid;
        };
        if is_continuation(source, next) {
            return CodesStep::Invalid;
        }
        CodesStep::Character {
            position: position + 1,
            code,
        }
    }

    /// The index an integer argument names, as `isize`.
    fn index(value: Option<i64>, default: isize) -> isize {
        value.map_or(default, |value| {
            isize::try_from(value).unwrap_or(if value < 0 { isize::MIN } else { isize::MAX })
        })
    }

    /// A negative position counts from the end of a string of `length` bytes.
    fn relative(position: isize, length: isize) -> isize {
        if position >= 0 {
            position
        } else {
            length.saturating_add(position).saturating_add(1).max(0)
        }
    }

    /// What `utf8.len` gives.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Len {
        /// The number of characters.
        Count(i64),
        /// The one-based position of the first invalid byte, with nil.
        Invalid(usize),
    }

    /// `utf8.len (s, i, j)` for integer positions (`None` for an absent
    /// one), or `None` for a call that raises an error.
    #[must_use]
    pub fn len(source: &[u8], start: Option<i64>, end: Option<i64>, mode: Utf8Mode) -> Option<Len> {
        let length = isize::try_from(source.len()).ok()?;
        let start = relative(index(start, 1), length);
        let end = relative(index(end, -1), length);
        if start < 1 || start - 1 > length || end > length {
            return None;
        }
        let mut position = usize::try_from(start - 1).ok()?;
        let mut count = 0;
        while end > 0 && position < usize::try_from(end).ok()? {
            let Some((next, _)) = decode(source, position, mode) else {
                return Some(Len::Invalid(position + 1));
            };
            position = next;
            count += 1;
        }
        Some(Len::Count(count))
    }

    /// `utf8.codepoint (s, i, j)` for integer positions: the code points in
    /// `out`, and how many. `None` for a call that raises an error or gives
    /// more than `out` holds.
    #[must_use]
    pub fn codepoints(
        source: &[u8],
        start: Option<i64>,
        end: Option<i64>,
        mode: Utf8Mode,
        out: &mut [u32],
    ) -> Option<usize> {
        let length = isize::try_from(source.len()).ok()?;
        let start = relative(index(start, 1), length);
        let end = relative(index(end, start), length);
        if start < 1 || end > length {
            return None;
        }
        if start > end {
            return Some(0);
        }
        let mut position = usize::try_from(start - 1).ok()?;
        let end = usize::try_from(end).ok()?;
        let mut count = 0;
        while position < end {
            let (next, code) = decode(source, position, mode)?;
            *out.get_mut(count)? = code;
            count += 1;
            position = next;
        }
        Some(count)
    }

    /// What `utf8.offset` gives.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Offset {
        /// The one-based position of the character, and of the last byte of
        /// it (which Lua 5.5 gives too).
        Found { start: usize, end: usize },
        /// There is no such character: nil.
        Absent,
    }

    /// `utf8.offset (s, n, i)` for integer arguments, or `None` for a call
    /// that raises an error. The end is the Lua 5.5 one.
    #[must_use]
    pub fn offset(source: &[u8], shift: i64, position: Option<i64>) -> Option<Offset> {
        let mut shift = index(Some(shift), 0);
        let length = isize::try_from(source.len()).ok()?;
        let position = relative(
            index(position, if shift >= 0 { 1 } else { length + 1 }),
            length,
        );
        if position < 1 || position - 1 > length {
            return None;
        }
        let mut position = usize::try_from(position - 1).ok()?;
        if shift == 0 {
            while position > 0 && is_continuation(source, position) {
                position -= 1;
            }
        } else {
            if is_continuation(source, position) {
                return None;
            }
            if shift < 0 {
                while shift < 0 && position > 0 {
                    position -= 1;
                    while position > 0 && is_continuation(source, position) {
                        position -= 1;
                    }
                    shift += 1;
                }
            } else {
                shift -= 1;
                while shift > 0 && position < source.len() {
                    position += 1;
                    while is_continuation(source, position) {
                        position += 1;
                    }
                    shift -= 1;
                }
            }
        }
        if shift != 0 {
            return Some(Offset::Absent);
        }
        let start = position + 1;
        // The end of the character, as Lua 5.5 gives it. A continuation
        // byte there is an error in that profile.
        if is_continuation(source, position) {
            return None;
        }
        if source.get(position).is_some_and(|byte| byte & 0x80 != 0) {
            while is_continuation(source, position + 1) {
                position += 1;
            }
        }
        Some(Offset::Found {
            start,
            end: position + 1,
        })
    }

    /// The bytes of the character `code` (`utf8.char`), or `None` for a code
    /// past `maximum` or below zero.
    #[must_use]
    pub fn encode(code: i64, maximum: i64, out: &mut [u8; 8]) -> Option<usize> {
        let (word, count) = encode_word(code, maximum)?;
        *out = word.to_le_bytes();
        Some(count)
    }

    /// [`encode`] as a word: the bytes of the character from the low byte
    /// up, zeros above them, and how many there are. A caller stores the
    /// word whole and advances by the count, with no loop over the bytes.
    #[must_use]
    #[inline]
    pub fn encode_word(code: i64, maximum: i64) -> Option<(u64, usize)> {
        if !(0..=maximum).contains(&code) {
            return None;
        }
        let code = u64::from(u32::try_from(code).ok()?);
        if code < 0x80 {
            return Some((code, 1));
        }
        // The continuation byte that holds the six bits of `code` from bit
        // `shift` up, in its place in the word.
        let tail = |shift: u32, place: u32| (0x80 | (code >> shift & 0x3f)) << (8 * place);
        Some(if code < 0x800 {
            (0xc0 | code >> 6 | tail(0, 1), 2)
        } else if code < 0x1_0000 {
            (0xe0 | code >> 12 | tail(6, 1) | tail(0, 2), 3)
        } else if code < 0x20_0000 {
            (0xf0 | code >> 18 | tail(12, 1) | tail(6, 2) | tail(0, 3), 4)
        } else if code < 0x400_0000 {
            (
                0xf8 | code >> 24 | tail(18, 1) | tail(12, 2) | tail(6, 3) | tail(0, 4),
                5,
            )
        } else if code < 0x8000_0000 {
            (
                0xfc | code >> 30
                    | tail(24, 1)
                    | tail(18, 2)
                    | tail(12, 3)
                    | tail(6, 4)
                    | tail(0, 5),
                6,
            )
        } else {
            return None;
        })
    }
}
