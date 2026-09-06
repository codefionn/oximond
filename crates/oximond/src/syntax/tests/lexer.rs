use super::*;

#[test]
fn lua51_streaming_parser_remembers_semicolons_across_fragments() {
    let mut parser = StreamingParser::with_profile(CompatibilityProfile::Lua51);
    assert!(matches!(parser.push(b"a = 1;"), SourceParsePoll::NeedInput));
    let SourceParsePoll::Failed(error) = parser.push(b"; ") else {
        panic!("a second semicolon must fail as soon as it is lexed");
    };
    assert_eq!(error.message, "unexpected ';'");
}

#[test]
fn ambiguous_call_newline_is_found_across_stream_fragments() {
    let mut parser = StreamingParser::with_profile(CompatibilityProfile::LuaJit21);
    assert!(matches!(
        parser.push(b"value = target\r"),
        SourceParsePoll::NeedInput
    ));
    let SourceParsePoll::Failed(error) = parser.push(b"\n(3) ") else {
        panic!("split CRLF call must fail once the parenthesis is lexed");
    };
    assert_eq!(error.span.start, 16);
}

#[test]
fn chunk_machine_fails_as_soon_as_the_lexer_releases_an_impossible_start() {
    let tokens = stream_tokens(&[b"*", b"a = 123"]).unwrap();
    let mut machine = ChunkMachine::new(8);
    assert!(matches!(
        machine.resume(PrattInput::Token(tokens[0].clone())),
        ChunkPoll::Failed(_)
    ));
    assert_eq!(machine.transition_count(), 1);
}

#[test]
fn streaming_parser_matches_legacy_at_every_byte_split() {
    for source in [
        b"local a = { function(x) return x end }; if a then for i = 1, 2 do a[i] = i end end"
            as &[u8],
        b"function a.b:call(x, ...) return x, ... end; a.b:call(1, 2)",
        b"repeat local value = 0x1.8p+1 until value == 3",
    ] {
        let mut expected = parse_with_legacy_parser(source).unwrap();
        expected.line_starts = vec![0];
        for split in 1..=source.len() {
            let mut parser = StreamingParser::new();
            for bytes in source.chunks(split) {
                assert!(matches!(parser.push(bytes), SourceParsePoll::NeedInput));
            }
            assert_eq!(parser.end(), SourceParsePoll::Parsed(expected.clone()));

            let mut owned = StreamingParser::new();
            for bytes in source.chunks(split) {
                assert!(matches!(owned.push(bytes), SourceParsePoll::NeedInput));
            }
            assert_eq!(owned.end_owned().unwrap(), expected);
        }
    }
}

#[test]
fn streaming_parser_is_terminal_and_rejects_invalid_reader_on_second_byte() {
    let mut parser = StreamingParser::new();
    assert!(matches!(parser.push(b"*"), SourceParsePoll::NeedInput));
    let SourceParsePoll::Failed(error) = parser.push(b"a") else {
        panic!("second byte must make the invalid start fail");
    };
    assert_eq!(error.message, "expected expression");
    assert_eq!(
        parser.push(b" = 123"),
        SourceParsePoll::Failed(error.clone())
    );
    assert_eq!(parser.end(), SourceParsePoll::Failed(error));

    let mut complete = StreamingParser::new();
    assert!(matches!(
        complete.push(b"return 1"),
        SourceParsePoll::NeedInput
    ));
    let SourceParsePoll::Parsed(chunk) = complete.end() else {
        panic!("valid source must complete at end");
    };
    assert_eq!(complete.end(), SourceParsePoll::Parsed(chunk.clone()));
    assert_eq!(complete.push(b" ignored"), SourceParsePoll::Parsed(chunk));
}

#[test]
fn pratt_machine_uses_one_transition_per_token() {
    let source = b"1 + 2 + 3 + 4 + 5 + 6 + 7 + 8";
    let mut tokens = Lexer::new(source).lex().unwrap();
    tokens.pop();
    let mut machine = PrattMachine::new(source.len());
    for token in tokens {
        assert!(matches!(
            machine.resume(PrattInput::Token(token)),
            PrattPoll::NeedToken
        ));
    }
    assert!(matches!(
        machine.resume(PrattInput::End),
        PrattPoll::Parsed(_)
    ));
    assert_eq!(machine.transition_count(), 16);
}

#[test]
fn incremental_lexer_waits_for_star_lookahead() {
    let mut lexer = IncrementalLexer::new();
    lexer.feed(b"*");
    assert!(matches!(lexer.poll(), Ok(LexPoll::NeedInput)));
    lexer.feed(b"a");
    let LexPoll::Token(token) = lexer.poll().unwrap() else {
        panic!("second reader call must make star available");
    };
    assert!(matches!(token.kind, TokenKind::Star));
    assert_eq!(token.span, Span { start: 0, end: 1 });
    assert_eq!(lexer.unconsumed(), b"a");
    lexer.end();
    let LexPoll::Token(token) = lexer.poll().unwrap() else {
        panic!("end must release the final token");
    };
    assert!(matches!(token.kind, TokenKind::Identifier(ref name) if name == "a"));
    assert!(matches!(lexer.poll(), Ok(LexPoll::Eof)));
}

#[test]
fn incremental_lexer_matches_one_shot_at_every_split() {
    let cases: &[&[u8]] = &[
        b"local a = 0x7f + 1.5e-2",
        b"local a = 0x1.8p+1 + 0X.EP-1; local b = 0xe+9",
        b"#! /usr/bin/lua\r\nvalue = [[\r\na\r\nb\n\rc]]",
        b"--[=[ comment\n]=]\nfunction t:m(x, ...) return x..'\xff' end",
        b"a = { [1] = 'x\\n', key = 2; 3 }",
        b"local a = 1.+.1 .. 2",
    ];
    for source in cases {
        let mut expected = Lexer::new(source).lex().unwrap();
        expected.pop();
        for split in 0..=source.len() {
            assert_eq!(
                stream_tokens(&[&source[..split], &source[split..]]).unwrap(),
                expected,
                "split at byte {split}"
            );
        }
        let pieces: Vec<&[u8]> = source.iter().map(core::slice::from_ref).collect();
        assert_eq!(
            stream_tokens(&pieces).unwrap(),
            expected,
            "one-byte partition"
        );
    }
}

#[test]
fn incremental_lexer_reports_unterminated_input_only_after_end() {
    let mut lexer = IncrementalLexer::new();
    lexer.feed(b"'unterminated");
    assert!(matches!(lexer.poll(), Ok(LexPoll::NeedInput)));
    lexer.end();
    assert!(matches!(
        lexer.poll(),
        Err(SyntaxError { message, .. }) if message == "unterminated string"
    ));
}

#[test]
fn incremental_lexer_matches_one_shot_for_long_fragmented_literal() {
    let body = "x".repeat(32 * 1024);
    let source = format!("local value = [[{body}]]");
    let expected = Lexer::new(source.as_bytes()).lex().unwrap();
    let pieces: Vec<&[u8]> = source.as_bytes().chunks(1).collect();
    let mut actual = stream_tokens(&pieces).unwrap();
    actual.push(Token {
        kind: TokenKind::Eof,
        span: Span {
            start: source.len(),
            end: source.len(),
        },
    });
    assert_eq!(actual, expected);
}

#[test]
fn incremental_lexer_advances_long_literal_and_comment_boundaries_once() {
    fn assert_monotonic_boundary(prefix: &[u8]) {
        let mut lexer = IncrementalLexer::new();
        let mut previous = 0;
        for byte in prefix {
            lexer.feed(core::slice::from_ref(byte));
            assert!(matches!(lexer.poll(), Ok(LexPoll::NeedInput)));
            let progress = lexer
                .pending_boundary_progress()
                .expect("long boundary scanner retains its cursor");
            assert!(progress >= previous, "scanner revisited a consumed prefix");
            previous = progress;
        }
    }

    let body = vec![b'x'; 32 * 1024];
    let mut literal = b"[[".to_vec();
    literal.extend_from_slice(&body);
    assert_monotonic_boundary(&literal);

    let mut comment = b"--[[".to_vec();
    comment.extend_from_slice(&body);
    assert_monotonic_boundary(&comment);
}

#[test]
fn incremental_lexer_keeps_stateful_boundaries_across_every_byte_split() {
    for source in [
        b"return [=[text]=]" as &[u8],
        b"return [==[text]==]",
        b"return '\\z \r\nvalue'",
        b"return '\\\r\nvalue'",
        b"return '\\u{1f600}\xff'",
    ] {
        let mut expected = Lexer::new(source).lex().unwrap();
        expected.pop();
        for split in 0..=source.len() {
            assert_eq!(
                stream_tokens(&[&source[..split], &source[split..]]).unwrap(),
                expected,
                "split={split}, source={source:?}"
            );
        }
    }

    let source = b"return '\\u{110000000}'";
    let expected = Lexer::new(source).lex().unwrap_err();
    for split in 0..=source.len() {
        assert_eq!(
            stream_tokens(&[&source[..split], &source[split..]]).unwrap_err(),
            expected,
            "invalid Unicode split={split}"
        );
    }
}

#[test]
fn incremental_lexer_releases_arithmetic_numbers_without_buffering_the_expression() {
    let source = b"1+1+1+1+1+1+1+1+1+1+1+1";
    let mut lexer = IncrementalLexer::new();
    let mut tokens = Vec::new();
    for byte in source {
        lexer.feed(core::slice::from_ref(byte));
        while let LexPoll::Token(token) = lexer.poll().unwrap() {
            tokens.push(token);
        }
        assert!(lexer.unconsumed().len() <= 2);
    }
    lexer.end();
    loop {
        match lexer.poll().unwrap() {
            LexPoll::Token(token) => tokens.push(token),
            LexPoll::Eof => break,
            LexPoll::NeedInput => panic!("ended lexer requested input"),
        }
    }
    let mut expected = Lexer::new(source).lex().unwrap();
    expected.pop();
    assert_eq!(tokens, expected);
}

#[test]
fn incremental_lexer_releases_hex_numbers_and_concatenation_before_end() {
    for source in [b"0xface+0xface+0xface" as &[u8], b"1..1..1"] {
        let mut lexer = IncrementalLexer::new();
        lexer.feed(source);
        assert!(matches!(lexer.poll(), Ok(LexPoll::Token(_))), "{source:?}");
    }

    let source = b"0xface+0xface+0xface+0xface";
    let mut lexer = IncrementalLexer::new();
    let mut tokens = Vec::new();
    for byte in source {
        lexer.feed(core::slice::from_ref(byte));
        while let LexPoll::Token(token) = lexer.poll().unwrap() {
            tokens.push(token);
        }
        assert!(lexer.unconsumed().len() <= 7);
    }
    lexer.end();
    while let LexPoll::Token(token) = lexer.poll().unwrap() {
        tokens.push(token);
    }
    let mut expected = Lexer::new(source).lex().unwrap();
    expected.pop();
    assert_eq!(tokens, expected);
}

#[test]
fn lua_diagnostics_include_chunk_line_and_near_token() {
    let source = b"local value = 1\r\nlocal broken =";
    let error = parse_bytes(source)
        .unwrap_err()
        .with_source(&LuaString::from("=named chunk"), source);
    assert_eq!(
        error.diagnostic().as_bytes(),
        b"named chunk:2: expected expression near '<eof>'"
    );
}

#[test]
fn lua_diagnostics_keep_invalid_near_token_bytes() {
    let source = b"\xffvalue = 1";
    let error = parse_bytes(source)
        .unwrap_err()
        .with_source(&LuaString::from(source), source);
    assert_eq!(
        error.diagnostic().as_bytes(),
        b"[string \"\xffvalue = 1\"]:1: unexpected character near '\xff'"
    );
}

#[test]
fn records_physical_line_starts_across_streaming_boundaries() {
    let mut parser = StreamingParser::new();
    assert!(matches!(
        parser.push(b"a = 1\r"),
        SourceParsePoll::NeedInput
    ));
    assert!(matches!(
        parser.push(b"\nb = 2\rc = 3\n"),
        SourceParsePoll::NeedInput
    ));
    let SourceParsePoll::Parsed(chunk) = parser.end() else {
        panic!("valid source should parse at end");
    };
    assert_eq!(chunk.line_starts, vec![0, 7, 13, 19]);
}

#[test]
fn luajit_lexes_64_bit_integer_suffixes() {
    let kinds = |source: &[u8]| {
        Lexer::with_escapes(source, true, true, false, true)
            .lex()
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| token.kind)
                    .collect::<Vec<_>>()
            })
    };
    assert_eq!(
        kinds(b"3ll 3ULL 3llu 0xffffffffffffffffll 9223372036854775808ll").unwrap(),
        vec![
            TokenKind::Int64(3),
            TokenKind::UInt64(3),
            TokenKind::UInt64(3),
            TokenKind::Int64(-1),
            TokenKind::Int64(i64::MIN),
            TokenKind::Eof,
        ]
    );
    for malformed in [
        &b"1.5ll"[..],
        b"1e3ll",
        b"0x1p4ll",
        b"1u",
        b"1lul",
        b"18446744073709551616ull",
        b"0x10000000000000000ull",
    ] {
        assert!(
            kinds(malformed).is_err(),
            "{}",
            String::from_utf8_lossy(malformed)
        );
    }
    // Without the LuaJIT profile the suffix is a separate identifier.
    let other = Lexer::with_escapes(b"1ll", true, true, false, false)
        .lex()
        .unwrap();
    assert!(matches!(other[1].kind, TokenKind::Identifier(_)));
}

#[test]
fn completed_statements_are_available_before_eof_in_every_profile() {
    for profile in CompatibilityProfile::ALL {
        let mut parser = StreamingParser::with_profile(profile);
        assert!(matches!(
            parser.push(b"local first = "),
            SourceParsePoll::NeedInput
        ));
        assert_eq!(parser.completed_statements(), []);
        assert!(matches!(
            parser.push(b"1;\nlocal second = "),
            SourceParsePoll::NeedInput
        ));
        assert_eq!(parser.completed_statements().len(), 1, "{profile:?}");
        let first = parser.completed_statements()[0].clone();
        assert_eq!(parser.line_starts(), &[0, 17]);
        assert!(matches!(
            parser.push(b"2;\nreturn first + second"),
            SourceParsePoll::NeedInput
        ));
        assert_eq!(parser.completed_statements().len(), 2, "{profile:?}");
        assert_eq!(parser.completed_statements()[0], first);
        let SourceParsePoll::Parsed(chunk) = parser.end() else {
            panic!("complete source")
        };
        assert_eq!(chunk.statements.len(), 3);
        assert_eq!(parser.completed_statements(), chunk.statements);
        assert_eq!(parser.line_starts(), chunk.line_starts);
        assert_eq!(parser.end(), SourceParsePoll::Parsed(chunk));
    }
}

#[test]
fn completed_statements_exclude_an_open_nested_function() {
    for profile in CompatibilityProfile::ALL {
        let mut parser = StreamingParser::with_profile(profile);
        assert!(matches!(
            parser.push(b"local ready = 1;\nlocal function large()\n"),
            SourceParsePoll::NeedInput
        ));
        assert_eq!(parser.completed_statements().len(), 1);
        for _ in 0..1000 {
            assert!(matches!(
                parser.push(b"ready = ready + 1;\n"),
                SourceParsePoll::NeedInput
            ));
            assert_eq!(parser.completed_statements().len(), 1);
        }
        assert!(matches!(
            parser.push(b"return ready\nend;\n"),
            SourceParsePoll::NeedInput
        ));
        assert_eq!(parser.completed_statements().len(), 2, "{profile:?}");
        let Statement::LocalFunction { body, .. } = &parser.completed_statements()[1] else {
            panic!("completed function")
        };
        assert_eq!(body.statements.len(), 1001);
        let chunk = parser.end_owned().unwrap();
        assert_eq!(chunk.statements.len(), 2);
        assert_eq!(chunk.line_starts.len(), 1005);
    }
}

#[test]
fn completed_statements_survive_a_later_syntax_failure() {
    let mut parser = StreamingParser::new();
    assert!(matches!(
        parser.push(b"local valid = 1; "),
        SourceParsePoll::NeedInput
    ));
    let completed = parser.completed_statements().to_vec();
    assert_eq!(completed.len(), 1);
    assert!(matches!(
        parser.push(b"* invalid "),
        SourceParsePoll::Failed(_)
    ));
    assert_eq!(parser.completed_statements(), completed);
    assert!(matches!(parser.end(), SourceParsePoll::Failed(_)));
    assert_eq!(parser.completed_statements(), completed);
}

#[test]
fn streaming_line_metadata_updates_split_line_endings() {
    let mut parser = StreamingParser::new();
    assert!(matches!(
        parser.push(b"local a = 1;\r"),
        SourceParsePoll::NeedInput
    ));
    assert_eq!(parser.line_starts(), &[0, 13]);
    assert!(matches!(
        parser.push(b"\nlocal b = 2;\n"),
        SourceParsePoll::NeedInput
    ));
    assert_eq!(parser.line_starts(), &[0, 14, 27]);
    let chunk = parser.end_owned().unwrap();
    assert_eq!(chunk.line_starts, vec![0, 14, 27]);
}

#[test]
fn completed_statement_depth_matches_the_final_chunk() {
    let mut parser = StreamingParser::new();
    assert!(matches!(
        parser.push(b"local a = {1, {2, {3}}}; "),
        SourceParsePoll::NeedInput
    ));
    let statement_depth = parser.completed_statements()[0].nesting_depth();
    assert_eq!(statement_depth, 5);
    let chunk = parser.end_owned().unwrap();
    assert_eq!(chunk.nesting_depth(), statement_depth);
}

#[test]
fn transferred_statement_batches_reconstruct_the_complete_tree() {
    let source = b"local x=1; local function f(y) return x+y end; x=f(2); return x";
    for profile in CompatibilityProfile::ALL {
        let config = ParserConfig::new(profile);
        let expected = parse_bytes_with_config(source, config).unwrap();
        for size in [1, 2, 7, source.len()] {
            let mut parser = StreamingParser::with_config(config);
            let mut statements = Vec::new();
            for fragment in source.chunks(size) {
                assert!(matches!(parser.push(fragment), SourceParsePoll::NeedInput));
                statements.extend(parser.take_completed_statements().unwrap());
                assert_eq!(parser.completed_statements(), []);
            }
            let mut tail = parser.end_owned().unwrap();
            statements.append(&mut tail.statements);
            tail.statements = statements;
            assert_eq!(tail, expected, "{profile:?} fragment {size}");
        }
    }
}

#[test]
fn transferred_statements_keep_named_vararg_validation() {
    let source = b"local function f(...args) return args end; next=1; ";
    for profile in CompatibilityProfile::ALL {
        let mut parser = StreamingParser::with_profile(profile);
        let poll = parser.push(source);
        if profile == CompatibilityProfile::Lua55 {
            assert!(matches!(poll, SourceParsePoll::NeedInput));
            assert_ne!(parser.take_completed_statements().unwrap(), []);
            assert!(parser.end_owned().is_ok());
        } else {
            match poll {
                SourceParsePoll::Failed(_) => {}
                SourceParsePoll::NeedInput => assert!(parser.take_completed_statements().is_err()),
                SourceParsePoll::Parsed(_) => panic!("parser ended without EOF"),
            }
        }
    }
}

#[test]
fn transferring_return_does_not_allow_later_statements() {
    let mut parser = StreamingParser::default();
    assert!(matches!(
        parser.push(b"return 1; "),
        SourceParsePoll::NeedInput
    ));
    parser.take_completed_statements().unwrap();
    assert!(matches!(parser.push(b"x=2; "), SourceParsePoll::Failed(_)));
}

#[test]
fn transferred_terminal_statements_match_complete_parsing_in_every_profile() {
    for (source, modern_only, valid) in [
        (b"return 1; x=2; ".as_slice(), false, false),
        (b"return; return 2; ", false, false),
        (b"return 1; ", false, true),
        (b"return 1;; ", false, false),
        (b"do return 1; x=2; end ", false, false),
        (b"do return 1; end; x=2; ", false, true),
        (b"while true do break; x=1; end ", true, true),
        (b"while true do do break; x=1; end end ", true, true),
        (b"while true do break; end; x=1; ", false, true),
        (
            b"while true do local function f() return 1; x=2; end; break end ",
            false,
            false,
        ),
    ] {
        for profile in CompatibilityProfile::ALL {
            let config = ParserConfig::new(profile);
            let expected = parse_bytes_with_config(source, config);
            let valid_for_profile = if source == b"return 1;; " {
                profile == CompatibilityProfile::LuaJit21
            } else {
                valid && !(modern_only && profile.is_lua51_family())
            };
            assert_eq!(
                expected.is_ok(),
                valid_for_profile,
                "{profile:?} {source:?}"
            );
            for size in [1, 2, 7, source.len()] {
                let mut parser = StreamingParser::with_config(config);
                let mut transferred = Vec::new();
                let mut failure = None;
                for fragment in source.chunks(size) {
                    match parser.push(fragment) {
                        SourceParsePoll::NeedInput => {
                            transferred.extend(parser.take_completed_statements().unwrap());
                        }
                        SourceParsePoll::Failed(error) => {
                            failure = Some(error);
                            break;
                        }
                        SourceParsePoll::Parsed(_) => panic!("source completed without EOF"),
                    }
                }
                let actual = match failure {
                    Some(error) => Err(error),
                    None => parser.end_owned().map(|mut chunk| {
                        transferred.append(&mut chunk.statements);
                        chunk.statements = transferred;
                        chunk
                    }),
                };
                match (&expected, actual) {
                    (Ok(expected), Ok(actual)) => assert_eq!(&actual, expected),
                    (Err(expected), Err(actual)) => {
                        assert_eq!(actual.message, expected.message, "{profile:?}, size={size}");
                        assert_eq!(actual.span, expected.span, "{profile:?}, size={size}");
                    }
                    _ => panic!("drained result differs for {profile:?}, size={size}, {source:?}"),
                }
            }
        }
    }
}
