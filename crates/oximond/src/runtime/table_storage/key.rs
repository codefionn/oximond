//! Table keys, their hashing, and Lua number key normalization.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::value::{
    BuiltinFunction, ClosureHandle, FileHandle, GmatchIteratorHandle, NativeFunctionHandle,
    TableHandle, ThreadHandle, UserdataHandle, Value,
};
use crate::string::LuaString;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum TableKey<S = LuaString> {
    Boolean(bool),
    Integer(i64),
    Number(u64),
    /// Build with [`TableKey::string`] so the string is hashed before it
    /// is stored; ordered entries copy that hash inline for scans.
    String(S),
    Table(TableHandle),
    File(FileHandle),
    Userdata(UserdataHandle),
    GmatchIterator(GmatchIteratorHandle),
    Thread(ThreadHandle),
    CoroutineWrapper(ThreadHandle),
    NativeFunction(NativeFunctionHandle),
    Closure(ClosureHandle),
    BuiltinFunction(BuiltinFunction),
    /// 64-bit integer cdata key by value, distinct from equal numbers.
    Int64(i64),
    UInt64(u64),
}

impl TableKey {
    /// A string key. Hashes the string now, so that scans over a stored key
    /// can copy its hash into an ordered entry.
    #[inline]
    pub(crate) fn string(value: &LuaString) -> Self {
        let _ = value.cached_hash();
        Self::String(value.clone())
    }
}

/// Borrows string bytes while retaining the owned key's discriminant and hash.
/// Reads therefore do not change the string's reference count.
pub(crate) struct StringTableKey<'a>(pub(crate) &'a LuaString);

impl std::hash::Hash for StringTableKey<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&TableKey::<&LuaString>::String(self.0), state);
    }
}

impl hashbrown::Equivalent<TableKey> for StringTableKey<'_> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn equivalent(&self, key: &TableKey) -> bool {
        matches!(key, TableKey::String(string) if self.0 == string)
    }
}

/// A string lookup that never calls out of line. Two heap strings that are
/// not the same allocation compare unequal here and set `heap_compare`, so
/// that the caller redoes the lookup with [`StringTableKey`] if it found
/// nothing.
pub(crate) struct StringProbe<'a> {
    pub(crate) key: &'a LuaString,
    /// The key's cached hash, already computed: hashing here never calls
    /// out of line, which keeps the probe free of saved registers.
    pub(crate) hash: u64,
    pub(crate) heap_compare: std::cell::Cell<bool>,
}

impl std::hash::Hash for StringProbe<'_> {
    // The words a `TableKey::String` hashes: its variant, then the string's
    // cached hash. Spelled out so that the hashing inlines into the probe.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(
            &std::mem::discriminant(&TableKey::<&LuaString>::String(self.key)),
            state,
        );
        state.write_u64(self.hash);
    }
}

impl hashbrown::Equivalent<TableKey> for StringProbe<'_> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn equivalent(&self, key: &TableKey) -> bool {
        let TableKey::String(string) = key else {
            return false;
        };
        if self.key.ptr_eq(string) {
            return true;
        }
        self.key.eq_without_call(string).unwrap_or_else(|| {
            self.heap_compare.set(true);
            false
        })
    }
}

/// A string lookup for a mutable access to a hashed table. It hashes the
/// same words as [`StringProbe`] and [`StringTableKey`], so it finds the
/// same entries. Unlike [`StringProbe`] it never needs a second lookup: two
/// heap strings that are not one allocation compare with `==` inside the
/// comparator, which keeps a single `get_mut` call sufficient.
pub(crate) struct StringSlotProbe<'a>(pub(crate) &'a LuaString);

impl std::hash::Hash for StringSlotProbe<'_> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(
            &std::mem::discriminant(&TableKey::<&LuaString>::String(self.0)),
            state,
        );
        state.write_u64(self.0.cached_hash());
    }
}

impl hashbrown::Equivalent<TableKey> for StringSlotProbe<'_> {
    #[inline]
    fn equivalent(&self, key: &TableKey) -> bool {
        let TableKey::String(string) = key else {
            return false;
        };
        self.0.ptr_eq(string)
            || self
                .0
                .eq_without_call(string)
                .unwrap_or_else(|| self.0 == string)
    }
}

/// A fixed-seed folding hasher for table keys.
///
/// String keys reach this as a single cached-hash word, so hot lookups avoid
/// rehashing key bytes. Non-string keys fold one word per derived write. The
/// final mixing step keeps bucket spread healthy for the power-of-two
/// tables hashbrown allocates.
#[derive(Default)]
pub(crate) struct TableKeyHasher {
    state: u64,
}

impl std::hash::Hasher for TableKeyHasher {
    fn finish(&self) -> u64 {
        // One multiply spreads every input bit into the high bits, which
        // hashbrown reads for its tag; folding them back down feeds the
        // low bits that pick the bucket.
        let mixed = (self.state ^ (self.state >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        mixed ^ (mixed >> 29)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state = (self.state ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.state = self.state.wrapping_mul(0x100_0000_01b3) ^ value;
    }

    // The derived hashes of keys write these: an enum's discriminant as a
    // `usize`, a handle as two `u32`s, a boolean as a `u8`. The defaults
    // would fold them a byte at a time; one word mix each is as good, since
    // `finish` spreads the bits. Every key hashes through this one hasher,
    // so the lookups agree.
    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.write_u64(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

pub(crate) type TableKeyBuildHasher = std::hash::BuildHasherDefault<TableKeyHasher>;

/// The cached hash of the `"__gc"` metatable field name.
pub(crate) const GC_FIELD_HASH: u64 = crate::string::cached_hash_bytes(b"__gc");

impl TryFrom<&Value> for TableKey {
    type Error = RuntimeError;
    #[inline]
    fn try_from(value: &Value) -> RuntimeResult<Self> {
        match value {
            Value::Nil => Err(RuntimeError::InvalidTableKey("nil")),
            Value::Boolean(value) => Ok(Self::Boolean(*value)),
            Value::Integer(value) => Ok(Self::Integer(*value)),
            Value::Number(value) if value.is_nan() => Err(RuntimeError::InvalidTableKey("number")),
            Value::Number(value) => Ok(number_table_key(*value)),
            Value::String(value) => Ok(Self::string(value)),
            Value::Table(value) => Ok(Self::Table(*value)),
            Value::File(value) => Ok(Self::File(*value)),
            Value::Userdata(value) => Ok(Self::Userdata(*value)),
            Value::GmatchIterator(value) => Ok(Self::GmatchIterator(*value)),
            Value::Thread(value) => Ok(Self::Thread(*value)),
            Value::CoroutineWrapper(value) => Ok(Self::CoroutineWrapper(*value)),
            Value::Closure(value) => Ok(Self::Closure(*value)),
            Value::NativeFunction(value) => Ok(Self::NativeFunction(*value)),
            Value::BuiltinFunction(value) => Ok(Self::BuiltinFunction(*value)),
            Value::Int64(value) => Ok(Self::Int64(*value)),
            Value::UInt64(value) => Ok(Self::UInt64(*value)),
        }
    }
}

pub(crate) fn normalize_number_key(value: f64) -> u64 {
    // Lua numbers are IEEE-754 doubles in this runtime. Apart from signed
    // zero, equal finite doubles have the same representation, so their bits
    // are a lossless table-key identity. Do not route large integral doubles
    // through an integer conversion here: `1e31` is a valid Lua key even
    // though it cannot name an array slot on this host.
    debug_assert!(!value.is_nan());
    if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

const I64_MIN_AS_F64: f64 = -9_223_372_036_854_775_808.0;
pub(crate) const I64_EXCLUSIVE_UPPER_AS_F64: f64 = 9_223_372_036_854_775_808.0;

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]
// In range, the cast truncates exactly, and the round trip is equal only for
// integral values. This avoids a libm `trunc`/`fract` call per numeric key.
#[inline]
pub(crate) fn integer_key_from_number(value: f64) -> Option<i64> {
    if !(I64_MIN_AS_F64..I64_EXCLUSIVE_UPPER_AS_F64).contains(&value) {
        return None;
    }
    let integer = truncate_to_i64(value);
    (integer as f64 == value).then_some(integer)
}

/// Truncates toward zero without the fix-up code of a saturating `as i64`.
///
/// On x86-64 this is the raw `cvttsd2si`, which yields `i64::MIN` for NaN and
/// for values outside the `i64` range. Every caller range-checks the result
/// or round-trips it through `f64`, and `i64::MIN` fails both when the input
/// was not already `-2^63`, so the saturating cast would only add instructions.
/// Other targets use the plain cast, which agrees on every input the callers
/// accept.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn truncate_to_i64(value: f64) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[allow(unsafe_code)]
        // SAFETY: SSE2 is part of the x86-64 baseline, and the intrinsics only
        // convert registers; they read no memory.
        unsafe {
            use core::arch::x86_64::{_mm_cvttsd_si64, _mm_set_sd};
            _mm_cvttsd_si64(_mm_set_sd(value))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        value as i64
    }
}

#[allow(clippy::cast_precision_loss)] // Lua 5.1 and 5.2 represent all integers as doubles.
pub(crate) const fn integer_to_number(value: i64) -> f64 {
    value as f64
}

fn number_table_key(value: f64) -> TableKey {
    if let Some(value) = integer_key_from_number(value) {
        return TableKey::Integer(value);
    }
    TableKey::Number(normalize_number_key(value))
}

#[cfg(test)]
mod truncation_tests {
    use super::{integer_key_from_number, truncate_to_i64};

    const TWO_63: f64 = 9_223_372_036_854_775_808.0;
    const TWO_53: f64 = 9_007_199_254_740_992.0;

    #[test]
    fn truncation_agrees_with_the_cast_inside_the_i64_range() {
        for value in [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.9,
            -1.9,
            TWO_53,
            TWO_53.next_up(),
            TWO_53.next_down(),
            -TWO_53,
            TWO_63.next_down(),
            -TWO_63,
            (-TWO_63).next_up(),
        ] {
            assert_eq!(truncate_to_i64(value), value as i64, "{value}");
        }
    }

    #[test]
    fn out_of_range_truncation_never_looks_like_a_valid_key() {
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            TWO_63,
            TWO_63.next_up(),
            (-TWO_63).next_down(),
            f64::MAX,
            f64::MIN,
        ] {
            assert_eq!(integer_key_from_number(value), None, "{value}");
        }
    }

    #[test]
    fn integer_keys_cover_the_boundaries() {
        assert_eq!(integer_key_from_number(0.0), Some(0));
        assert_eq!(integer_key_from_number(-0.0), Some(0));
        assert_eq!(integer_key_from_number(0.5), None);
        assert_eq!(integer_key_from_number(1.0), Some(1));
        assert_eq!(integer_key_from_number(TWO_53), Some(1 << 53));
        assert_eq!(integer_key_from_number(-TWO_63), Some(i64::MIN));
        assert_eq!(
            integer_key_from_number(TWO_63.next_down()),
            Some(i64::MAX - 1023)
        );
        assert_eq!(
            integer_key_from_number((-TWO_63).next_up()),
            Some(i64::MIN + 1024)
        );
    }
}
