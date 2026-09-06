//! The recursive-descent fast path must build exactly the streaming parser's
//! tree for every chunk it accepts.

use super::*;

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// Programs that use every construct the fast path handles.
const SAMPLES: &[&str] = &[
    "#!/usr/bin/env lua\nlocal a, b = 1, 2.5e3 return a",
    "local t = { 1, 2; x = 3, ['k' .. 1] = 4, [f()] = g, nested = { {}, {1} }, }",
    "local s = 'a' .. \"b\\n\\65\" .. [[long\nstring]] .. [==[x]]==] -- comment\n--[[ long\ncomment ]] x = s",
    "x = 0x10 + 0xA.8p1 + 1e-3 + .5 + 3. + 9223372036854775807 + -9223372036854775808",
    "y = 9223372036854775808",
    "x = -2 ^ -3 ^ 2 + not a == b and c or d .. e .. f < g",
    "x = a + b * c - d / e % f // g",
    "x = 1 | 2 ~ 3 & 4 << 5 >> 6 .. 7 | ~8",
    "x = #t + -#t - - -1",
    "f() f'a' f\"b\" f{} f[[c]] a.b.c:d(1, 2):e'x'.f[g]{}",
    "a.b, c[d], e = 1, 2",
    "local function f(a, b, ...) return ... end\nfunction g.h.i:j(...) return self, ... end\nfunction k() end",
    "local f = function() end; local g = function(...) local t = {...} return #t end",
    "if a then b() elseif c then d() elseif e then else f() end",
    "if a then end if b then else end",
    "while a do break end repeat local x = 1 until x == 1 for i = 1, 10 do end for i = 1, 10, 2 do break end",
    "for k, v in pairs(t) do print(k, v) end for k in next, t do end",
    "do local x <const> = 1; local y <close> = nil end",
    "goto done; ::done:: return",
    "return;",
    "return 1, 2, 3",
    "local x; ;; x = (a)(b); (c).d = 1",
    "local goto = 1 global = goto",
    "x = a\n(f)()",
    "x = {}\n(f)()",
    "x = 1LL + 2ULL + 0x10ull",
    "do do do return end end end",
    "repeat if a then break end until b",
    "local a = {f = function(x) return x end, [1] = 2}",
    "return function(...) return select('#', ...) end",
    "x = (((((1)))))",
    "x = a.b.c.d.e.f",
    "t[i], t[j] = t[j], t[i]",
];

fn streaming(source: &[u8], config: ParserConfig) -> Result<Chunk, SyntaxError> {
    let mut parser = StreamingParser::with_config(config);
    match parser.push(source) {
        SourceParsePoll::Parsed(chunk) => Ok(chunk),
        SourceParsePoll::Failed(error) => Err(error),
        SourceParsePoll::NeedInput => parser.end_owned(),
    }
}

/// Returns whether the fast path accepted `source`.
fn check(source: &[u8], profile: CompatibilityProfile) -> bool {
    let config = ParserConfig::new(profile);
    let Some((fast, depth)) = crate::syntax::fast::parse_complete_with_depth(source, config) else {
        return false;
    };
    assert_eq!(
        depth,
        fast.nesting_depth(),
        "{profile:?}: depth of {}",
        String::from_utf8_lossy(source)
    );
    assert_eq!(
        Ok(fast),
        streaming(source, config),
        "{profile:?}: {}",
        String::from_utf8_lossy(source)
    );
    true
}

#[test]
fn fast_parser_matches_streaming_parser() {
    for sample in SAMPLES {
        let mut accepted = false;
        for profile in PROFILES {
            accepted |= check(sample.as_bytes(), profile);
        }
        assert!(accepted, "no profile used the fast path for {sample}");
    }
}

#[test]
fn fast_parser_handles_the_lua54_samples() {
    for sample in SAMPLES {
        let config = ParserConfig::new(CompatibilityProfile::Lua54);
        if streaming(sample.as_bytes(), config).is_ok() && !sample.contains("LL") {
            assert!(
                crate::syntax::fast::parse_complete(sample.as_bytes(), config).is_some(),
                "fast path gave up on valid {sample}"
            );
        }
    }
}

#[test]
fn fast_parser_leaves_deep_nesting_to_the_streaming_parser() {
    let source = format!("x = {}1{}", "(".repeat(150), ")".repeat(150));
    let config = ParserConfig::default();
    assert!(crate::syntax::fast::parse_complete(source.as_bytes(), config).is_none());
    assert_eq!(
        parse_bytes_with_config(source.as_bytes(), config),
        streaming(source.as_bytes(), config)
    );
}

#[test]
fn fast_parser_records_line_starts_like_the_streaming_parser() {
    for source in [
        &b"x = 1\r\ny = 2\n\rz = 3"[..],
        b"x = 1\r\ry = 2\n\nz = 3\r",
        b"x = 1\n\r\n\ry = 2\r\n",
        b"\n\nx = 1 -- comment\r\ny = 'a'\n",
        b"x = 1",
    ] {
        assert!(check(source, CompatibilityProfile::Lua54), "{source:?}");
    }
}

#[test]
fn fast_parser_counts_the_depth_of_long_chains() {
    let config = ParserConfig::new(CompatibilityProfile::LuaJit21);
    for source in [
        format!("x = a{}", ".b".repeat(500)),
        format!("x = 1{}", " + 1".repeat(500)),
        format!("x = f{}", "(1)".repeat(300)),
        format!("x = a{}", ":m({y = {z = 1}})".repeat(100)),
        format!("x = {}1", "- ".repeat(30)),
        format!(
            "local t = {{ [a.b.c] = function() return {} end }}",
            "f(1) + ".repeat(80) + "2"
        ),
    ] {
        let (chunk, depth) =
            crate::syntax::fast::parse_complete_with_depth(source.as_bytes(), config)
                .expect("the fast parser accepts the chain");
        assert_eq!(depth, chunk.nesting_depth(), "{source}");
        assert!(depth > 30);
    }
}

/// Token-level mutants of the samples. Many are invalid and fall back; the
/// valid ones must agree with the streaming parser.
#[test]
fn fast_parser_matches_streaming_parser_on_mutants() {
    const PIECES: &[&str] = &[
        "(", ")", "end", "=", ",", "[", "]", "{", "}", "...", "::", "local", "function", ";",
        "<const>", "goto", ".", ":", "x", "1", "'s'", "return", "break", "do", "then", "if",
        "else", "elseif", "for", "in", "while", "repeat", "until", "and", "or", "not", "-", "^",
        "..", "#", "~", "//", "&", "<", "global", "\n(", "f", "a.b", "a:b()", "{}",
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut random = move |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        usize::try_from(state % bound as u64).expect("bounded")
    };
    let mut accepted = 0;
    for sample in SAMPLES {
        let source = sample.as_bytes();
        let Ok(tokens) = Lexer::new(source).lex() else {
            continue;
        };
        for _ in 0..200 {
            let span = tokens[random(tokens.len())].span;
            let mut mutant = source.to_vec();
            let piece = PIECES[random(PIECES.len())].as_bytes();
            match random(4) {
                0 => drop(mutant.drain(span.start..span.end)),
                1 => drop(mutant.splice(span.start..span.start, piece.iter().copied())),
                2 => drop(mutant.splice(span.start..span.end, piece.iter().copied())),
                _ => {
                    let copy = source[span.start..span.end].to_vec();
                    drop(mutant.splice(span.end..span.end, copy));
                }
            }
            for profile in PROFILES {
                accepted += usize::from(check(&mutant, profile));
            }
        }
    }
    assert!(
        accepted > 1000,
        "only {accepted} mutants took the fast path"
    );
}
