//! Exhaustive correctness tests for SIMD byte classification, scanning,
//! line-start indexing, and multi-block lexer boundaries.

use super::*;
use crate::syntax::simd;

fn is_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b'\x0b' | b'\x0c')
}

fn is_ident(byte: u8) -> bool {
    matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

fn naive_line_starts(source: &[u8]) -> Vec<usize> {
    let mut starts = vec![0];
    let mut i = 0;
    while i < source.len() {
        let byte = source[i];
        if matches!(byte, b'\n' | b'\r') {
            let pair = source
                .get(i + 1)
                .is_some_and(|next| *next != byte && matches!(*next, b'\n' | b'\r'));
            i += 1 + usize::from(pair);
            starts.push(i);
        } else {
            i += 1;
        }
    }
    starts
}

#[test]
fn classify_block64_matches_scalar_for_all_256_bytes_and_every_length() {
    let all_bytes: Vec<u8> = (0..=255_u8).collect();
    let avx2_modes: &[bool] = if simd::has_avx2() {
        &[false, true]
    } else {
        &[false]
    };

    for &use_avx2 in avx2_modes {
        for start in 0..all_bytes.len() {
            for len in 0..=80.min(all_bytes.len() - start) {
                let slice = &all_bytes[start..start + len];
                let (masks, valid_len) = simd::classify_block64(slice, use_avx2);
                assert_eq!(
                    valid_len,
                    len.min(64),
                    "avx2={use_avx2} start={start} len={len}"
                );
                for bit in 0..64 {
                    let (exp_ws_stop, exp_id_stop, exp_dg_stop) =
                        slice.get(bit).copied().map_or((true, true, true), |byte| {
                            (!is_ws(byte), !is_ident(byte), !byte.is_ascii_digit())
                        });
                    assert_eq!(
                        ((masks.ws_stops >> bit) & 1) != 0,
                        exp_ws_stop,
                        "ws_stops mismatch avx2={use_avx2} start={start} len={len} bit={bit}"
                    );
                    assert_eq!(
                        ((masks.ident_stops >> bit) & 1) != 0,
                        exp_id_stop,
                        "ident_stops mismatch avx2={use_avx2} start={start} len={len} bit={bit}"
                    );
                    assert_eq!(
                        ((masks.digit_stops >> bit) & 1) != 0,
                        exp_dg_stop,
                        "digit_stops mismatch avx2={use_avx2} start={start} len={len} bit={bit}"
                    );
                }
            }
        }
    }
}

#[test]
fn simd_scanners_handle_every_byte_value_at_every_vector_boundary() {
    let boundaries = [
        0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 96,
    ];

    for &prefix_len in &boundaries {
        // Test every byte value 0..=255 placed right after `prefix_len` matching bytes.
        for byte in 0..=255_u8 {
            let mut ws_buf = vec![b' '; prefix_len];
            ws_buf.push(byte);
            ws_buf.push(b'x');
            let expected_ws = if is_ws(byte) {
                prefix_len + 1
            } else {
                prefix_len
            };
            assert_eq!(
                simd::skip_whitespace(&ws_buf),
                expected_ws,
                "skip_whitespace prefix={prefix_len} byte={byte:#04x}"
            );

            let mut id_buf = vec![b'a'; prefix_len];
            id_buf.push(byte);
            id_buf.push(b'!');
            let expected_id = if is_ident(byte) {
                prefix_len + 1
            } else {
                prefix_len
            };
            assert_eq!(
                simd::scan_ident(&id_buf),
                expected_id,
                "scan_ident prefix={prefix_len} byte={byte:#04x}"
            );

            let mut dg_buf = vec![b'5'; prefix_len];
            dg_buf.push(byte);
            dg_buf.push(b'z');
            let expected_dg = if byte.is_ascii_digit() {
                prefix_len + 1
            } else {
                prefix_len
            };
            assert_eq!(
                simd::scan_digits(&dg_buf),
                expected_dg,
                "scan_digits prefix={prefix_len} byte={byte:#04x}"
            );

            let mut hx_buf = vec![b'F'; prefix_len];
            hx_buf.push(byte);
            hx_buf.push(b'g');
            let expected_hx = if byte.is_ascii_hexdigit() {
                prefix_len + 1
            } else {
                prefix_len
            };
            assert_eq!(
                simd::scan_hex_digits(&hx_buf),
                expected_hx,
                "scan_hex_digits prefix={prefix_len} byte={byte:#04x}"
            );

            for quote in *b"\"'" {
                let mut str_buf = vec![b'a'; prefix_len];
                str_buf.push(byte);
                str_buf.push(b'b');
                let expected_str = if byte == quote || matches!(byte, b'\\' | b'\n' | b'\r') {
                    Some(prefix_len)
                } else {
                    None
                };
                assert_eq!(
                    simd::find_short_string_special(&str_buf, quote),
                    expected_str,
                    "find_short_string_special quote={} prefix={prefix_len} byte={byte:#04x}",
                    char::from(quote)
                );
            }
        }
    }
}

#[test]
fn simd_scanners_handle_all_matching_buffers_of_every_length() {
    for len in 0..=140 {
        assert_eq!(simd::skip_whitespace(&vec![b'\t'; len]), len);
        assert_eq!(simd::scan_ident(&vec![b'_'; len]), len);
        assert_eq!(simd::scan_digits(&vec![b'9'; len]), len);
        assert_eq!(simd::scan_hex_digits(&vec![b'a'; len]), len);
        assert_eq!(
            simd::find_short_string_special(&vec![b'x'; len], b'"'),
            None
        );
        assert_eq!(
            simd::find_short_string_special(&vec![b'x'; len], b'\''),
            None
        );
    }
}

#[test]
fn find_long_bracket_close_handles_false_positives_and_levels() {
    assert_eq!(simd::find_long_bracket_close(b"abc]def]]ghi", 0), Some(7));
    assert_eq!(simd::find_long_bracket_close(b"abc]]def]=]ghi", 1), Some(8));
    assert_eq!(simd::find_long_bracket_close(b"]]=]=]==]==]", 2), Some(5));
    assert_eq!(simd::find_long_bracket_close(b"]===]", 4), None);
    assert_eq!(simd::find_long_bracket_close(b"no bracket at all", 0), None);
    let long = format!("{}]==]", "x".repeat(200));
    assert_eq!(simd::find_long_bracket_close(long.as_bytes(), 2), Some(200));
}

#[test]
fn collect_line_starts_matches_naive_across_64_byte_boundaries() {
    let endings: &[&[u8]] = &[b"\n", b"\r", b"\r\n", b"\n\r", b"\r\r", b"\n\n", b"\n\r\n"];
    // Place every ending at offsets straddling 16, 32, 64, and 128 byte boundaries.
    for &prefix in &[0, 14, 15, 16, 30, 31, 32, 62, 63, 64, 65, 126, 127, 128] {
        for &ending in endings {
            for &suffix in &[0, 1, 15, 64] {
                let mut buf = vec![b'a'; prefix];
                buf.extend_from_slice(ending);
                buf.extend(core::iter::repeat_n(b'b', suffix));
                assert_eq!(
                    simd::collect_line_starts(&buf),
                    naive_line_starts(&buf),
                    "prefix={prefix} ending={ending:?} suffix={suffix}"
                );
            }
        }
    }

    // Dense random-like mix of bytes, CR, and LF across several 64-byte blocks.
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut buf = Vec::with_capacity(512);
    for _ in 0..512 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let byte = match state & 7 {
            0 => b'\n',
            1 => b'\r',
            _ => b'x',
        };
        buf.push(byte);
    }
    for len in 0..=buf.len() {
        assert_eq!(
            simd::collect_line_starts(&buf[..len]),
            naive_line_starts(&buf[..len]),
            "dense mix len={len}"
        );
    }
}

#[test]
fn fast_and_streaming_parsers_agree_on_tokens_crossing_64_byte_windows() {
    let long_ws = " \t\x0b\x0c".repeat(25);
    let long_ident = format!("ident_{}", "a1_Z".repeat(25));
    let long_string = "hello_world_".repeat(12);
    let long_escaped = format!("{}\\n\\z{}tail", "part_".repeat(16), " \t\n".repeat(12));
    let long_digits = "1234567890".repeat(3);
    let long_hex = format!("0x{}", "deadBEEF01234567".repeat(2));
    let long_comment = format!("--[==[\n{}\r\n{}\n\r]==]", "c".repeat(80), "d".repeat(80));
    let long_bracket_str = format!(
        "[=[\r\n{}\n\r{}\r{}]=]",
        "p".repeat(70),
        "q".repeat(70),
        "r".repeat(70)
    );

    let source = format!(
        "#!/usr/bin/env lua{}\n{long_ws}local {long_ident} = \"{long_string}\" .. \"{long_escaped}\" .. {long_bracket_str}\n{long_comment}\nlocal n = {long_digits} + {long_hex}\nreturn {long_ident}, n\n",
        "_".repeat(90),
    );
    let config = ParserConfig::new(CompatibilityProfile::Lua54);
    let fast = crate::syntax::fast::parse_complete(source.as_bytes(), config)
        .expect("fast parser should accept long-token chunk");
    let mut streaming = StreamingParser::with_config(config);
    for piece in source.as_bytes().chunks(19) {
        assert!(matches!(streaming.push(piece), SourceParsePoll::NeedInput));
    }
    let streamed = streaming
        .end_owned()
        .expect("streaming parser should succeed");
    assert_eq!(fast, streamed);
}
