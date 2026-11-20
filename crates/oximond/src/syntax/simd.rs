//! SIMD byte-scanning and block-classification primitives for Lua tokenization.
//!
//! All `unsafe` in `syntax` stays confined to this module: pointer-based vector
//! loads (strictly bounds-checked against slice lengths) and runtime-dispatched
//! `#[target_feature]` calls. Every `pub(crate)` function exposes a safe slice
//! interface and produces identical results across every backend:
//!
//! - `x86_64`: baseline `SSE2` (16-byte vectors) plus runtime-detected `AVX2`
//!   (32-byte vectors) for 64-byte block classification, line-start indexing,
//!   and longer runs.
//! - `aarch64`: baseline `NEON` (16-byte vectors, 64-byte pairwise movemask).
//! - `wasm32` with `simd128`: `v128` 16-byte vectors.
//! - Portable scalar fallback on all other targets (including `wasm32` without
//!   `simd128`).

/// Pre-classified byte-class stop bitmasks for a window of up to 64 source bytes.
///
/// Bit `i` corresponds to byte `i` of the window. Bits at or past the window's
/// valid length are set to `1` in every mask so `trailing_zeros()` stops
/// naturally at the end of the source without an extra clamp.
#[allow(clippy::struct_field_names)] // Every mask encodes where its token-class run stops.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BlockMasks {
    /// Bit `i` is `1` iff byte `i` is not Lua whitespace (or is past the window end).
    pub(crate) ws_stops: u64,
    /// Bit `i` is `1` iff byte `i` is not an identifier-continue byte (or is past the window end).
    pub(crate) ident_stops: u64,
    /// Bit `i` is `1` iff byte `i` is not an ASCII decimal digit (or is past the window end).
    pub(crate) digit_stops: u64,
}

/// Returns whether `AVX2` is available on the current CPU.
#[inline]
pub(crate) fn has_avx2() -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        true
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
const fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c')
}

#[inline(always)]
const fn is_ident_continue(byte: u8) -> bool {
    matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

#[inline(always)]
const fn is_hex_digit(byte: u8) -> bool {
    byte.is_ascii_hexdigit()
}

#[inline(always)]
const fn is_short_string_special(byte: u8, quote: u8) -> bool {
    byte == quote || matches!(byte, b'\\' | b'\n' | b'\r')
}

/// Classifies up to 64 leading bytes of `bytes` into whitespace, identifier,
/// and decimal-digit stop bitmasks, returning `(masks, valid_len)`.
#[inline]
pub(crate) fn classify_block64(bytes: &[u8], use_avx2: bool) -> (BlockMasks, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        x86::classify_block64(bytes, use_avx2)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = use_avx2;
        neon::classify_block64(bytes)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let _ = use_avx2;
        wasm::classify_block64(bytes)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let _ = use_avx2;
        scalar::classify_block64(bytes)
    }
}

/// Returns the index of the first non-whitespace byte in `bytes`, or
/// `bytes.len()` when all bytes are Lua whitespace.
#[inline]
pub(crate) fn skip_whitespace(bytes: &[u8]) -> usize {
    // Most token boundaries have zero or one space; avoid vector setup there.
    let Some(&first) = bytes.first() else {
        return 0;
    };
    if !is_whitespace(first) {
        return 0;
    }
    let Some(&second) = bytes.get(1) else {
        return 1;
    };
    if !is_whitespace(second) {
        return 1;
    }
    2 + skip_whitespace_long(&bytes[2..])
}

#[inline]
fn skip_whitespace_long(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        x86::skip_whitespace(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::skip_whitespace(bytes)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::skip_whitespace(bytes)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar::skip_whitespace(bytes)
    }
}

/// Returns the length of the leading ASCII identifier-continue run
/// (`a..=z | A..=Z | 0..=9 | _`) in `bytes`.
#[inline]
pub(crate) fn scan_ident(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        x86::scan_ident(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::scan_ident(bytes)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::scan_ident(bytes)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar::scan_ident(bytes)
    }
}

/// Returns the length of the leading ASCII decimal-digit run (`0..=9`).
#[inline]
pub(crate) fn scan_digits(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        x86::scan_digits(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::scan_digits(bytes)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::scan_digits(bytes)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar::scan_digits(bytes)
    }
}

/// Returns the length of the leading ASCII hexadecimal-digit run
/// (`0..=9 | a..=f | A..=F`).
#[inline]
pub(crate) fn scan_hex_digits(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        x86::scan_hex_digits(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::scan_hex_digits(bytes)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::scan_hex_digits(bytes)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar::scan_hex_digits(bytes)
    }
}

/// Returns the index of the first byte in `bytes` matching `quote`, `b'\\'`,
/// `b'\n'`, or `b'\r'`.
#[inline]
pub(crate) fn find_short_string_special(bytes: &[u8], quote: u8) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        x86::find_short_string_special(bytes, quote)
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::find_short_string_special(bytes, quote)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::find_short_string_special(bytes, quote)
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar::find_short_string_special(bytes, quote)
    }
}

/// Returns the index of the first `b'\n'` or `b'\r'` in `bytes`.
#[inline]
pub(crate) fn find_line_break(bytes: &[u8]) -> Option<usize> {
    memchr::memchr2(b'\n', b'\r', bytes)
}

/// Returns the index of the closing `]==...==]` delimiter of `level` equals
/// inside `bytes`, using `memchr` to jump between `b']'` candidates.
#[inline]
pub(crate) fn find_long_bracket_close(bytes: &[u8], level: usize) -> Option<usize> {
    let mut cursor = 0;
    while let Some(offset) = memchr::memchr(b']', &bytes[cursor..]) {
        let pos = cursor + offset;
        let tail = pos + 1 + level;
        if bytes.get(tail) == Some(&b']')
            && (level == 0 || bytes[pos + 1..tail].iter().all(|byte| *byte == b'='))
        {
            return Some(pos);
        }
        cursor = pos + 1;
    }
    None
}

/// Collects the byte offset of every physical line start in `source`.
///
/// `\r\n` and `\n\r` end one line; any other `\r` or `\n` ends one each.
/// The returned vector always begins with `0`.
pub(crate) fn collect_line_starts(source: &[u8]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(source.len() / 32 + 1);
    starts.push(0);
    #[cfg(target_arch = "x86_64")]
    {
        x86::collect_line_starts(source, &mut starts);
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::collect_line_starts(source, &mut starts);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::collect_line_starts(source, &mut starts);
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        collect_line_starts_tail(source, 0, None, &mut starts);
    }
    starts
}

/// Processes one 64-byte block's `\n` and `\r` masks into `starts`, updating
/// `trailing` when byte 63 was an unpaired line-break byte that may pair with
/// byte 0 of the next block.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
fn push_line_breaks_block64(
    block: &[u8],
    base: usize,
    mut nl_mask: u64,
    mut cr_mask: u64,
    trailing: &mut Option<u8>,
    starts: &mut Vec<usize>,
) {
    if cr_mask == 0 {
        if let Some(prev) = *trailing {
            if prev == b'\r' && (nl_mask & 1) != 0 {
                if let Some(last) = starts.last_mut() {
                    *last += 1;
                }
                nl_mask &= !1;
            }
            *trailing = None;
        }
        if nl_mask == 0 {
            return;
        }
        if (nl_mask >> 63) != 0 {
            *trailing = Some(b'\n');
        }
        while nl_mask != 0 {
            let tz = nl_mask.trailing_zeros() as usize;
            starts.push(base + tz + 1);
            nl_mask &= nl_mask - 1;
        }
        return;
    }

    if let Some(prev) = *trailing {
        let first = block[0];
        if first != prev && matches!(first, b'\n' | b'\r') {
            if let Some(last) = starts.last_mut() {
                *last += 1;
            }
            nl_mask &= !1;
            cr_mask &= !1;
        }
        *trailing = None;
    }

    let mut any = nl_mask | cr_mask;
    while any != 0 {
        let tz = any.trailing_zeros() as usize;
        any &= any - 1;
        let byte = block[tz];
        if tz + 1 < 64 {
            let next = block[tz + 1];
            if next != byte && matches!(next, b'\n' | b'\r') {
                any &= !(1_u64 << (tz + 1));
                starts.push(base + tz + 2);
            } else {
                starts.push(base + tz + 1);
            }
        } else {
            starts.push(base + 64);
            *trailing = Some(byte);
        }
    }
}

/// Scalar/memchr tail for `collect_line_starts` from `base` onward.
#[inline]
fn collect_line_starts_tail(
    source: &[u8],
    mut offset: usize,
    trailing: Option<u8>,
    starts: &mut Vec<usize>,
) {
    if let Some(prev) = trailing
        && source
            .get(offset)
            .is_some_and(|next| *next != prev && matches!(*next, b'\n' | b'\r'))
    {
        if let Some(last) = starts.last_mut() {
            *last += 1;
        }
        offset += 1;
    }
    while let Some(rel) = find_line_break(&source[offset..]) {
        let found = offset + rel;
        let pair = source
            .get(found + 1)
            .is_some_and(|next| *next != source[found] && matches!(*next, b'\r' | b'\n'));
        offset = found + 1 + usize::from(pair);
        starts.push(offset);
    }
}

mod scalar {
    use super::{
        BlockMasks, is_hex_digit, is_ident_continue, is_short_string_special, is_whitespace,
    };

    #[inline]
    pub(crate) fn classify_tail(
        bytes: &[u8],
        start: usize,
        mut ws_bits: u64,
        mut ident_bits: u64,
        mut digit_bits: u64,
    ) -> BlockMasks {
        for (i, &byte) in bytes.iter().enumerate() {
            let bit = 1_u64 << (start + i);
            if is_whitespace(byte) {
                ws_bits |= bit;
            } else if byte.is_ascii_digit() {
                digit_bits |= bit;
                ident_bits |= bit;
            } else if is_ident_continue(byte) {
                ident_bits |= bit;
            }
        }
        BlockMasks {
            ws_stops: !ws_bits,
            ident_stops: !ident_bits,
            digit_stops: !digit_bits,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn classify_block64(bytes: &[u8]) -> (BlockMasks, usize) {
        let len = bytes.len().min(64);
        (classify_tail(&bytes[..len], 0, 0, 0, 0), len)
    }

    #[inline]
    pub(crate) fn skip_whitespace(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .position(|&byte| !is_whitespace(byte))
            .unwrap_or(bytes.len())
    }

    #[inline]
    pub(crate) fn scan_ident(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .position(|&byte| !is_ident_continue(byte))
            .unwrap_or(bytes.len())
    }

    #[inline]
    pub(crate) fn scan_digits(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .unwrap_or(bytes.len())
    }

    #[inline]
    pub(crate) fn scan_hex_digits(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .position(|&byte| !is_hex_digit(byte))
            .unwrap_or(bytes.len())
    }

    #[inline]
    pub(crate) fn find_short_string_special(bytes: &[u8], quote: u8) -> Option<usize> {
        bytes
            .iter()
            .position(|&byte| is_short_string_special(byte, quote))
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_add_epi8, _mm_cmpeq_epi8, _mm_cmpgt_epi8, _mm_loadu_si128,
        _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm256_add_epi8, _mm256_cmpeq_epi8,
        _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_set1_epi8,
    };

    use super::{BlockMasks, collect_line_starts_tail, has_avx2, push_line_breaks_block64, scalar};

    // Signed-range offsets for SSE2/AVX2 unsigned interval checks:
    // `(x + (0x80 - lo)) < (-128 + (hi - lo + 1))` (signed `i8`).
    //   0x09..=0x0D (len 5):  add 0x77, cmpgt(-123, .)
    //   0x30..=0x39 (len 10): add 0x50, cmpgt(-118, .)
    //   0x61..=0x7A (len 26): add 0x1F, cmpgt(-102, .)
    //   0x61..=0x66 (len 6):  add 0x1F, cmpgt(-122, .)

    #[target_feature(enable = "sse2")]
    #[inline]
    fn ws_mask_sse2(v: __m128i) -> u16 {
        let space = _mm_cmpeq_epi8(v, _mm_set1_epi8(b' ' as i8));
        let ctrl = _mm_cmpgt_epi8(_mm_set1_epi8(-123), _mm_add_epi8(v, _mm_set1_epi8(0x77)));
        _mm_movemask_epi8(_mm_or_si128(space, ctrl)) as u16
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn ident_mask_sse2(v: __m128i) -> u16 {
        let lower = _mm_or_si128(v, _mm_set1_epi8(0x20));
        let alpha = _mm_cmpgt_epi8(
            _mm_set1_epi8(-102),
            _mm_add_epi8(lower, _mm_set1_epi8(0x1F)),
        );
        let digit = _mm_cmpgt_epi8(_mm_set1_epi8(-118), _mm_add_epi8(v, _mm_set1_epi8(0x50)));
        let under = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'_' as i8));
        let ident = _mm_or_si128(_mm_or_si128(alpha, digit), under);
        _mm_movemask_epi8(ident) as u16
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn digit_mask_sse2(v: __m128i) -> u16 {
        let digit = _mm_cmpgt_epi8(_mm_set1_epi8(-118), _mm_add_epi8(v, _mm_set1_epi8(0x50)));
        _mm_movemask_epi8(digit) as u16
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn hex_mask_sse2(v: __m128i) -> u16 {
        let digit = _mm_cmpgt_epi8(_mm_set1_epi8(-118), _mm_add_epi8(v, _mm_set1_epi8(0x50)));
        let lower = _mm_or_si128(v, _mm_set1_epi8(0x20));
        let hex_alpha = _mm_cmpgt_epi8(
            _mm_set1_epi8(-122),
            _mm_add_epi8(lower, _mm_set1_epi8(0x1F)),
        );
        _mm_movemask_epi8(_mm_or_si128(digit, hex_alpha)) as u16
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn string_special_mask_sse2(v: __m128i, quote: __m128i) -> u16 {
        let eq_q = _mm_cmpeq_epi8(v, quote);
        let eq_bs = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\\' as i8));
        let eq_nl = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\n' as i8));
        let eq_cr = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\r' as i8));
        let any = _mm_or_si128(_mm_or_si128(eq_q, eq_bs), _mm_or_si128(eq_nl, eq_cr));
        _mm_movemask_epi8(any) as u16
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn classify_16_sse2(v: __m128i) -> (u16, u16, u16) {
        let space = _mm_cmpeq_epi8(v, _mm_set1_epi8(b' ' as i8));
        let ctrl = _mm_cmpgt_epi8(_mm_set1_epi8(-123), _mm_add_epi8(v, _mm_set1_epi8(0x77)));
        let ws = _mm_movemask_epi8(_mm_or_si128(space, ctrl)) as u16;

        let digit_v = _mm_cmpgt_epi8(_mm_set1_epi8(-118), _mm_add_epi8(v, _mm_set1_epi8(0x50)));
        let digit = _mm_movemask_epi8(digit_v) as u16;

        let lower = _mm_or_si128(v, _mm_set1_epi8(0x20));
        let alpha = _mm_cmpgt_epi8(
            _mm_set1_epi8(-102),
            _mm_add_epi8(lower, _mm_set1_epi8(0x1F)),
        );
        let under = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'_' as i8));
        let ident = _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(alpha, digit_v), under)) as u16;

        (ws, ident, digit)
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    fn classify_32_avx2(v: __m256i) -> (u32, u32, u32) {
        let space = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b' ' as i8));
        let ctrl = _mm256_cmpgt_epi8(
            _mm256_set1_epi8(-123),
            _mm256_add_epi8(v, _mm256_set1_epi8(0x77)),
        );
        let ws = _mm256_movemask_epi8(_mm256_or_si256(space, ctrl)) as u32;

        let digit_v = _mm256_cmpgt_epi8(
            _mm256_set1_epi8(-118),
            _mm256_add_epi8(v, _mm256_set1_epi8(0x50)),
        );
        let digit = _mm256_movemask_epi8(digit_v) as u32;

        let lower = _mm256_or_si256(v, _mm256_set1_epi8(0x20));
        let alpha = _mm256_cmpgt_epi8(
            _mm256_set1_epi8(-102),
            _mm256_add_epi8(lower, _mm256_set1_epi8(0x1F)),
        );
        let under = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'_' as i8));
        let ident =
            _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(alpha, digit_v), under)) as u32;

        (ws, ident, digit)
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn classify_block64_avx2(ptr: *const u8) -> BlockMasks {
        // SAFETY: Caller guarantees `ptr` is valid for reading 64 bytes.
        let v0 = unsafe { _mm256_loadu_si256(ptr.cast()) };
        // SAFETY: `ptr.add(32)` is within the 64-byte slice.
        let v1 = unsafe { _mm256_loadu_si256(ptr.add(32).cast()) };
        let (ws0, id0, dg0) = classify_32_avx2(v0);
        let (ws1, id1, dg1) = classify_32_avx2(v1);
        BlockMasks {
            ws_stops: !(u64::from(ws0) | (u64::from(ws1) << 32)),
            ident_stops: !(u64::from(id0) | (u64::from(id1) << 32)),
            digit_stops: !(u64::from(dg0) | (u64::from(dg1) << 32)),
        }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn classify_block64_sse2(bytes: &[u8]) -> (BlockMasks, usize) {
        if bytes.len() >= 64 {
            let ptr = bytes.as_ptr();
            // SAFETY: `bytes.len() >= 64`, so all four 16-byte loads are in bounds.
            let (ws0, id0, dg0) = classify_16_sse2(unsafe { _mm_loadu_si128(ptr.cast()) });
            let (ws1, id1, dg1) = classify_16_sse2(unsafe { _mm_loadu_si128(ptr.add(16).cast()) });
            let (ws2, id2, dg2) = classify_16_sse2(unsafe { _mm_loadu_si128(ptr.add(32).cast()) });
            let (ws3, id3, dg3) = classify_16_sse2(unsafe { _mm_loadu_si128(ptr.add(48).cast()) });
            let ws = u64::from(ws0)
                | (u64::from(ws1) << 16)
                | (u64::from(ws2) << 32)
                | (u64::from(ws3) << 48);
            let id = u64::from(id0)
                | (u64::from(id1) << 16)
                | (u64::from(id2) << 32)
                | (u64::from(id3) << 48);
            let dg = u64::from(dg0)
                | (u64::from(dg1) << 16)
                | (u64::from(dg2) << 32)
                | (u64::from(dg3) << 48);
            return (
                BlockMasks {
                    ws_stops: !ws,
                    ident_stops: !id,
                    digit_stops: !dg,
                },
                64,
            );
        }
        let len = bytes.len();
        let mut ws = 0_u64;
        let mut id = 0_u64;
        let mut dg = 0_u64;
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len` keeps the 16-byte load in bounds.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let (w, i, d) = classify_16_sse2(v);
            ws |= u64::from(w) << offset;
            id |= u64::from(i) << offset;
            dg |= u64::from(d) << offset;
            offset += 16;
        }
        (
            scalar::classify_tail(&bytes[offset..], offset, ws, id, dg),
            len,
        )
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn classify_block64(bytes: &[u8], use_avx2: bool) -> (BlockMasks, usize) {
        if bytes.len() >= 64 && use_avx2 {
            // SAFETY: `bytes.len() >= 64` and `use_avx2` confirms AVX2 support.
            return (unsafe { classify_block64_avx2(bytes.as_ptr()) }, 64);
        }
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { classify_block64_sse2(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn skip_whitespace_avx2(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let sp = _mm256_set1_epi8(b' ' as i8);
        let add_c = _mm256_set1_epi8(0x77);
        let lim_c = _mm256_set1_epi8(-123);
        while offset + 32 <= bytes.len() {
            // SAFETY: `offset + 32 <= bytes.len()`.
            let v = unsafe { _mm256_loadu_si256(ptr.add(offset).cast()) };
            let space = _mm256_cmpeq_epi8(v, sp);
            let ctrl = _mm256_cmpgt_epi8(lim_c, _mm256_add_epi8(v, add_c));
            let stops = !(_mm256_movemask_epi8(_mm256_or_si256(space, ctrl)) as u32);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 32;
        }
        offset + skip_whitespace_sse2(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn skip_whitespace_sse2(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let stops = !ws_mask_sse2(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::skip_whitespace(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[cold]
    #[inline(never)]
    fn skip_whitespace_rest(rest: &[u8]) -> usize {
        if rest.len() >= 32 && has_avx2() {
            // SAFETY: `has_avx2()` checked above.
            unsafe { skip_whitespace_avx2(rest) }
        } else {
            // SAFETY: `sse2` is baseline on `x86_64`.
            unsafe { skip_whitespace_sse2(rest) }
        }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn skip_whitespace_first16(bytes: &[u8]) -> usize {
        // SAFETY: Caller verified `bytes.len() >= 16`.
        let v = unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) };
        let stops = !ws_mask_sse2(v);
        if stops != 0 {
            stops.trailing_zeros() as usize
        } else {
            16 + skip_whitespace_rest(&bytes[16..])
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn skip_whitespace(bytes: &[u8]) -> usize {
        if bytes.len() < 16 {
            return scalar::skip_whitespace(bytes);
        }
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { skip_whitespace_first16(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_ident_avx2(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let bit20 = _mm256_set1_epi8(0x20);
        let add_a = _mm256_set1_epi8(0x1F);
        let lim_a = _mm256_set1_epi8(-102);
        let add_d = _mm256_set1_epi8(0x50);
        let lim_d = _mm256_set1_epi8(-118);
        let under_v = _mm256_set1_epi8(b'_' as i8);
        while offset + 32 <= bytes.len() {
            // SAFETY: `offset + 32 <= bytes.len()`.
            let v = unsafe { _mm256_loadu_si256(ptr.add(offset).cast()) };
            let lower = _mm256_or_si256(v, bit20);
            let alpha = _mm256_cmpgt_epi8(lim_a, _mm256_add_epi8(lower, add_a));
            let digit = _mm256_cmpgt_epi8(lim_d, _mm256_add_epi8(v, add_d));
            let under = _mm256_cmpeq_epi8(v, under_v);
            let ident = _mm256_or_si256(_mm256_or_si256(alpha, digit), under);
            let stops = !(_mm256_movemask_epi8(ident) as u32);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 32;
        }
        offset + scan_ident_sse2(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scan_ident_sse2(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let stops = !ident_mask_sse2(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_ident(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[cold]
    #[inline(never)]
    fn scan_ident_rest(rest: &[u8]) -> usize {
        if rest.len() >= 32 && has_avx2() {
            // SAFETY: `has_avx2()` checked above.
            unsafe { scan_ident_avx2(rest) }
        } else {
            // SAFETY: `sse2` is baseline on `x86_64`.
            unsafe { scan_ident_sse2(rest) }
        }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scan_ident_first16(bytes: &[u8]) -> usize {
        // SAFETY: Caller verified `bytes.len() >= 16`.
        let v = unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) };
        let stops = !ident_mask_sse2(v);
        if stops != 0 {
            stops.trailing_zeros() as usize
        } else {
            16 + scan_ident_rest(&bytes[16..])
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_ident(bytes: &[u8]) -> usize {
        if bytes.len() < 16 {
            return scalar::scan_ident(bytes);
        }
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { scan_ident_first16(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scan_digits_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let stops = !digit_mask_sse2(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_digits(bytes: &[u8]) -> usize {
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { scan_digits_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn scan_hex_digits_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let stops = !hex_mask_sse2(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_hex_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_hex_digits(bytes: &[u8]) -> usize {
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { scan_hex_digits_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn find_short_string_special_avx2(bytes: &[u8], quote: u8) -> Option<usize> {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let q_v = _mm256_set1_epi8(quote as i8);
        let bs_v = _mm256_set1_epi8(b'\\' as i8);
        let nl_v = _mm256_set1_epi8(b'\n' as i8);
        let cr_v = _mm256_set1_epi8(b'\r' as i8);
        while offset + 32 <= bytes.len() {
            // SAFETY: `offset + 32 <= bytes.len()`.
            let v = unsafe { _mm256_loadu_si256(ptr.add(offset).cast()) };
            let any = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(v, q_v), _mm256_cmpeq_epi8(v, bs_v)),
                _mm256_or_si256(_mm256_cmpeq_epi8(v, nl_v), _mm256_cmpeq_epi8(v, cr_v)),
            );
            let mask = _mm256_movemask_epi8(any) as u32;
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }
            offset += 32;
        }
        find_short_string_special_sse2(&bytes[offset..], quote).map(|found| offset + found)
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn find_short_string_special_sse2(bytes: &[u8], quote: u8) -> Option<usize> {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let q_v = _mm_set1_epi8(quote as i8);
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let mask = string_special_mask_sse2(v, q_v);
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }
            offset += 16;
        }
        scalar::find_short_string_special(&bytes[offset..], quote).map(|found| offset + found)
    }

    #[allow(unsafe_code)]
    #[inline(never)]
    fn find_short_string_special_rest(rest: &[u8], quote: u8) -> Option<usize> {
        if rest.len() >= 32 && has_avx2() {
            // SAFETY: `has_avx2()` checked above.
            unsafe { find_short_string_special_avx2(rest, quote) }
        } else {
            // SAFETY: `sse2` is baseline on `x86_64`.
            unsafe { find_short_string_special_sse2(rest, quote) }
        }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    #[inline]
    fn find_short_string_special_first16(bytes: &[u8], quote: u8) -> Option<usize> {
        let q_v = _mm_set1_epi8(quote as i8);
        // SAFETY: Caller verified `bytes.len() >= 16`.
        let v = unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) };
        let mask = string_special_mask_sse2(v, q_v);
        if mask != 0 {
            Some(mask.trailing_zeros() as usize)
        } else {
            find_short_string_special_rest(&bytes[16..], quote).map(|found| 16 + found)
        }
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn find_short_string_special(bytes: &[u8], quote: u8) -> Option<usize> {
        if bytes.len() < 16 {
            return scalar::find_short_string_special(bytes, quote);
        }
        // SAFETY: `sse2` is baseline on `x86_64`.
        unsafe { find_short_string_special_first16(bytes, quote) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "avx2")]
    unsafe fn collect_line_starts_avx2(source: &[u8], starts: &mut Vec<usize>) {
        let mut offset = 0;
        let mut trailing = None;
        let ptr = source.as_ptr();
        let nl_v = _mm256_set1_epi8(b'\n' as i8);
        let cr_v = _mm256_set1_epi8(b'\r' as i8);
        while offset + 64 <= source.len() {
            // SAFETY: `offset + 64 <= source.len()`.
            let v0 = unsafe { _mm256_loadu_si256(ptr.add(offset).cast()) };
            let v1 = unsafe { _mm256_loadu_si256(ptr.add(offset + 32).cast()) };
            let nl0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, nl_v)) as u32;
            let nl1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, nl_v)) as u32;
            let cr0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, cr_v)) as u32;
            let cr1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, cr_v)) as u32;
            let nl_mask = u64::from(nl0) | (u64::from(nl1) << 32);
            let cr_mask = u64::from(cr0) | (u64::from(cr1) << 32);
            push_line_breaks_block64(
                &source[offset..offset + 64],
                offset,
                nl_mask,
                cr_mask,
                &mut trailing,
                starts,
            );
            offset += 64;
        }
        collect_line_starts_tail(source, offset, trailing, starts);
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "sse2")]
    fn collect_line_starts_sse2(source: &[u8], starts: &mut Vec<usize>) {
        let mut offset = 0;
        let mut trailing = None;
        let ptr = source.as_ptr();
        let nl_v = _mm_set1_epi8(b'\n' as i8);
        let cr_v = _mm_set1_epi8(b'\r' as i8);
        while offset + 64 <= source.len() {
            // SAFETY: `offset + 64 <= source.len()`.
            let v0 = unsafe { _mm_loadu_si128(ptr.add(offset).cast()) };
            let v1 = unsafe { _mm_loadu_si128(ptr.add(offset + 16).cast()) };
            let v2 = unsafe { _mm_loadu_si128(ptr.add(offset + 32).cast()) };
            let v3 = unsafe { _mm_loadu_si128(ptr.add(offset + 48).cast()) };
            let nl_mask = u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v0, nl_v)) as u16)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v1, nl_v)) as u16) << 16)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v2, nl_v)) as u16) << 32)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v3, nl_v)) as u16) << 48);
            let cr_mask = u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v0, cr_v)) as u16)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v1, cr_v)) as u16) << 16)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v2, cr_v)) as u16) << 32)
                | (u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v3, cr_v)) as u16) << 48);
            push_line_breaks_block64(
                &source[offset..offset + 64],
                offset,
                nl_mask,
                cr_mask,
                &mut trailing,
                starts,
            );
            offset += 64;
        }
        collect_line_starts_tail(source, offset, trailing, starts);
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn collect_line_starts(source: &[u8], starts: &mut Vec<usize>) {
        if source.len() >= 64 && has_avx2() {
            // SAFETY: `has_avx2()` checked above.
            unsafe { collect_line_starts_avx2(source, starts) };
        } else {
            // SAFETY: `sse2` is baseline on `x86_64`.
            unsafe { collect_line_starts_sse2(source, starts) };
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        uint8x16_t, vaddv_u8, vandq_u8, vceqq_u8, vcleq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8,
        vgetq_lane_u64, vld1q_u8, vorrq_u8, vpaddq_u8, vreinterpretq_u64_u8, vsubq_u8,
    };

    use super::{BlockMasks, collect_line_starts_tail, push_line_breaks_block64, scalar};

    const BIT_WEIGHTS: [u8; 16] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80,
    ];

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn movemask_16(v: uint8x16_t) -> u16 {
        // SAFETY: `BIT_WEIGHTS` is 16 bytes.
        let weights = unsafe { vld1q_u8(BIT_WEIGHTS.as_ptr()) };
        let masked = vandq_u8(v, weights);
        let lo = vaddv_u8(vget_low_u8(masked));
        let hi = vaddv_u8(vget_high_u8(masked));
        u16::from(lo) | (u16::from(hi) << 8)
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn movemask_64(m0: uint8x16_t, m1: uint8x16_t, m2: uint8x16_t, m3: uint8x16_t) -> u64 {
        // SAFETY: `BIT_WEIGHTS` is 16 bytes.
        let weights = unsafe { vld1q_u8(BIT_WEIGHTS.as_ptr()) };
        let t0 = vpaddq_u8(vandq_u8(m0, weights), vandq_u8(m1, weights));
        let t1 = vpaddq_u8(vandq_u8(m2, weights), vandq_u8(m3, weights));
        let t2 = vpaddq_u8(t0, t1);
        let t3 = vpaddq_u8(t2, t2);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(t3)).to_le()
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn ws_vec(v: uint8x16_t) -> uint8x16_t {
        let space = vceqq_u8(v, vdupq_n_u8(b' '));
        let ctrl = vcleq_u8(vsubq_u8(v, vdupq_n_u8(0x09)), vdupq_n_u8(4));
        vorrq_u8(space, ctrl)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn digit_vec(v: uint8x16_t) -> uint8x16_t {
        vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn ident_vec(v: uint8x16_t, digit: uint8x16_t) -> uint8x16_t {
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let alpha = vcleq_u8(vsubq_u8(lower, vdupq_n_u8(b'a')), vdupq_n_u8(25));
        let under = vceqq_u8(v, vdupq_n_u8(b'_'));
        vorrq_u8(vorrq_u8(alpha, digit), under)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn hex_vec(v: uint8x16_t) -> uint8x16_t {
        let digit = digit_vec(v);
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let hex_alpha = vcleq_u8(vsubq_u8(lower, vdupq_n_u8(b'a')), vdupq_n_u8(5));
        vorrq_u8(digit, hex_alpha)
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn classify_block64_impl(bytes: &[u8]) -> (BlockMasks, usize) {
        if bytes.len() >= 64 {
            let ptr = bytes.as_ptr();
            // SAFETY: `bytes.len() >= 64`, so all four 16-byte loads are in bounds.
            let v0 = unsafe { vld1q_u8(ptr) };
            let v1 = unsafe { vld1q_u8(ptr.add(16)) };
            let v2 = unsafe { vld1q_u8(ptr.add(32)) };
            let v3 = unsafe { vld1q_u8(ptr.add(48)) };

            let w0 = ws_vec(v0);
            let w1 = ws_vec(v1);
            let w2 = ws_vec(v2);
            let w3 = ws_vec(v3);

            let d0 = digit_vec(v0);
            let d1 = digit_vec(v1);
            let d2 = digit_vec(v2);
            let d3 = digit_vec(v3);

            let i0 = ident_vec(v0, d0);
            let i1 = ident_vec(v1, d1);
            let i2 = ident_vec(v2, d2);
            let i3 = ident_vec(v3, d3);

            return (
                BlockMasks {
                    ws_stops: !movemask_64(w0, w1, w2, w3),
                    ident_stops: !movemask_64(i0, i1, i2, i3),
                    digit_stops: !movemask_64(d0, d1, d2, d3),
                },
                64,
            );
        }
        let len = bytes.len();
        let mut ws = 0_u64;
        let mut id = 0_u64;
        let mut dg = 0_u64;
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let d = digit_vec(v);
            ws |= u64::from(movemask_16(ws_vec(v))) << offset;
            dg |= u64::from(movemask_16(d)) << offset;
            id |= u64::from(movemask_16(ident_vec(v, d))) << offset;
            offset += 16;
        }
        (
            scalar::classify_tail(&bytes[offset..], offset, ws, id, dg),
            len,
        )
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn classify_block64(bytes: &[u8]) -> (BlockMasks, usize) {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { classify_block64_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn skip_whitespace_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let stops = !movemask_16(ws_vec(v));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::skip_whitespace(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn skip_whitespace(bytes: &[u8]) -> usize {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { skip_whitespace_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn scan_ident_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let stops = !movemask_16(ident_vec(v, digit_vec(v)));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_ident(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_ident(bytes: &[u8]) -> usize {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { scan_ident_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn scan_digits_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let stops = !movemask_16(digit_vec(v));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_digits(bytes: &[u8]) -> usize {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { scan_digits_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn scan_hex_digits_impl(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let stops = !movemask_16(hex_vec(v));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_hex_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_hex_digits(bytes: &[u8]) -> usize {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { scan_hex_digits_impl(bytes) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    #[inline]
    fn find_short_string_special_impl(bytes: &[u8], quote: u8) -> Option<usize> {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let q_v = vdupq_n_u8(quote);
        let bs_v = vdupq_n_u8(b'\\');
        let nl_v = vdupq_n_u8(b'\n');
        let cr_v = vdupq_n_u8(b'\r');
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { vld1q_u8(ptr.add(offset)) };
            let any = vorrq_u8(
                vorrq_u8(vceqq_u8(v, q_v), vceqq_u8(v, bs_v)),
                vorrq_u8(vceqq_u8(v, nl_v), vceqq_u8(v, cr_v)),
            );
            let mask = movemask_16(any);
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }
            offset += 16;
        }
        scalar::find_short_string_special(&bytes[offset..], quote).map(|found| offset + found)
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn find_short_string_special(bytes: &[u8], quote: u8) -> Option<usize> {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { find_short_string_special_impl(bytes, quote) }
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "neon")]
    fn collect_line_starts_impl(source: &[u8], starts: &mut Vec<usize>) {
        let mut offset = 0;
        let mut trailing = None;
        let ptr = source.as_ptr();
        let nl_v = vdupq_n_u8(b'\n');
        let cr_v = vdupq_n_u8(b'\r');
        while offset + 64 <= source.len() {
            // SAFETY: `offset + 64 <= source.len()`.
            let v0 = unsafe { vld1q_u8(ptr.add(offset)) };
            let v1 = unsafe { vld1q_u8(ptr.add(offset + 16)) };
            let v2 = unsafe { vld1q_u8(ptr.add(offset + 32)) };
            let v3 = unsafe { vld1q_u8(ptr.add(offset + 48)) };
            let nl_mask = movemask_64(
                vceqq_u8(v0, nl_v),
                vceqq_u8(v1, nl_v),
                vceqq_u8(v2, nl_v),
                vceqq_u8(v3, nl_v),
            );
            let cr_mask = movemask_64(
                vceqq_u8(v0, cr_v),
                vceqq_u8(v1, cr_v),
                vceqq_u8(v2, cr_v),
                vceqq_u8(v3, cr_v),
            );
            push_line_breaks_block64(
                &source[offset..offset + 64],
                offset,
                nl_mask,
                cr_mask,
                &mut trailing,
                starts,
            );
            offset += 64;
        }
        collect_line_starts_tail(source, offset, trailing, starts);
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn collect_line_starts(source: &[u8], starts: &mut Vec<usize>) {
        // SAFETY: `neon` is baseline on `aarch64`.
        unsafe { collect_line_starts_impl(source, starts) };
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        u8x16_bitmask, u8x16_eq, u8x16_le, u8x16_splat, u8x16_sub, v128, v128_load, v128_or,
    };

    use super::{BlockMasks, collect_line_starts_tail, push_line_breaks_block64, scalar};

    #[inline(always)]
    fn ws_mask(v: v128) -> u16 {
        let space = u8x16_eq(v, u8x16_splat(b' '));
        let ctrl = u8x16_le(u8x16_sub(v, u8x16_splat(0x09)), u8x16_splat(4));
        u8x16_bitmask(v128_or(space, ctrl))
    }

    #[inline(always)]
    fn digit_vec(v: v128) -> v128 {
        u8x16_le(u8x16_sub(v, u8x16_splat(b'0')), u8x16_splat(9))
    }

    #[inline(always)]
    fn ident_mask(v: v128, digit: v128) -> u16 {
        let lower = v128_or(v, u8x16_splat(0x20));
        let alpha = u8x16_le(u8x16_sub(lower, u8x16_splat(b'a')), u8x16_splat(25));
        let under = u8x16_eq(v, u8x16_splat(b'_'));
        u8x16_bitmask(v128_or(v128_or(alpha, digit), under))
    }

    #[inline(always)]
    fn hex_mask(v: v128) -> u16 {
        let digit = digit_vec(v);
        let lower = v128_or(v, u8x16_splat(0x20));
        let hex_alpha = u8x16_le(u8x16_sub(lower, u8x16_splat(b'a')), u8x16_splat(5));
        u8x16_bitmask(v128_or(digit, hex_alpha))
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn classify_block64(bytes: &[u8]) -> (BlockMasks, usize) {
        let len = bytes.len().min(64);
        let mut ws = 0_u64;
        let mut id = 0_u64;
        let mut dg = 0_u64;
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let d = digit_vec(v);
            ws |= u64::from(ws_mask(v)) << offset;
            dg |= u64::from(u8x16_bitmask(d)) << offset;
            id |= u64::from(ident_mask(v, d)) << offset;
            offset += 16;
        }
        (
            scalar::classify_tail(&bytes[offset..len], offset, ws, id, dg),
            len,
        )
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn skip_whitespace(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let stops = !ws_mask(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::skip_whitespace(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_ident(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let stops = !ident_mask(v, digit_vec(v));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_ident(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_digits(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let stops = !u8x16_bitmask(digit_vec(v));
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn scan_hex_digits(bytes: &[u8]) -> usize {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let stops = !hex_mask(v);
            if stops != 0 {
                return offset + stops.trailing_zeros() as usize;
            }
            offset += 16;
        }
        offset + scalar::scan_hex_digits(&bytes[offset..])
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn find_short_string_special(bytes: &[u8], quote: u8) -> Option<usize> {
        let mut offset = 0;
        let ptr = bytes.as_ptr();
        let q_v = u8x16_splat(quote);
        let bs_v = u8x16_splat(b'\\');
        let nl_v = u8x16_splat(b'\n');
        let cr_v = u8x16_splat(b'\r');
        while offset + 16 <= bytes.len() {
            // SAFETY: `offset + 16 <= bytes.len()`.
            let v = unsafe { v128_load(ptr.add(offset).cast()) };
            let any = v128_or(
                v128_or(u8x16_eq(v, q_v), u8x16_eq(v, bs_v)),
                v128_or(u8x16_eq(v, nl_v), u8x16_eq(v, cr_v)),
            );
            let mask = u8x16_bitmask(any);
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }
            offset += 16;
        }
        scalar::find_short_string_special(&bytes[offset..], quote).map(|found| offset + found)
    }

    #[allow(unsafe_code)]
    #[inline]
    pub(crate) fn collect_line_starts(source: &[u8], starts: &mut Vec<usize>) {
        let mut offset = 0;
        let mut trailing = None;
        let ptr = source.as_ptr();
        let nl_v = u8x16_splat(b'\n');
        let cr_v = u8x16_splat(b'\r');
        while offset + 64 <= source.len() {
            // SAFETY: `offset + 64 <= source.len()`.
            let v0 = unsafe { v128_load(ptr.add(offset).cast()) };
            let v1 = unsafe { v128_load(ptr.add(offset + 16).cast()) };
            let v2 = unsafe { v128_load(ptr.add(offset + 32).cast()) };
            let v3 = unsafe { v128_load(ptr.add(offset + 48).cast()) };
            let nl_mask = u64::from(u8x16_bitmask(u8x16_eq(v0, nl_v)))
                | (u64::from(u8x16_bitmask(u8x16_eq(v1, nl_v))) << 16)
                | (u64::from(u8x16_bitmask(u8x16_eq(v2, nl_v))) << 32)
                | (u64::from(u8x16_bitmask(u8x16_eq(v3, nl_v))) << 48);
            let cr_mask = u64::from(u8x16_bitmask(u8x16_eq(v0, cr_v)))
                | (u64::from(u8x16_bitmask(u8x16_eq(v1, cr_v))) << 16)
                | (u64::from(u8x16_bitmask(u8x16_eq(v2, cr_v))) << 32)
                | (u64::from(u8x16_bitmask(u8x16_eq(v3, cr_v))) << 48);
            push_line_breaks_block64(
                &source[offset..offset + 64],
                offset,
                nl_mask,
                cr_mask,
                &mut trailing,
                starts,
            );
            offset += 64;
        }
        collect_line_starts_tail(source, offset, trailing, starts);
    }
}
