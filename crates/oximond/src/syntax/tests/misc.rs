use super::*;

#[test]
fn parses_assignment_slice() {
    let chunk = parse(
        "-- comment\ndo local a, b = 1, 2 + 3; t[a], t[b] = t[b], nil; check((a == b), {}) end",
    )
    .unwrap();
    assert_eq!(chunk.statements.len(), 1);
    let Statement::Do(block) = &chunk.statements[0] else {
        panic!()
    };
    assert_eq!(block.statements.len(), 3);
}

#[test]
fn long_comment_and_number() {
    let chunk = parse("--[[ ignored ]] local a = 1.5e2").unwrap();
    let Statement::Local { values, .. } = &chunk.statements[0] else {
        panic!()
    };
    assert!(matches!(values[0], Expr::Number(value, _) if (value - 150.0).abs() < f64::EPSILON));
}

#[test]
fn leading_shebang_is_a_single_line_comment() {
    for source in [
        "#!/usr/bin/env lua\nvalue = 1",
        "#!/usr/bin/env lua\rvalue = 1",
        "#!/usr/bin/env lua\r\nvalue = 1",
        "#!/usr/bin/env lua\n\rvalue = 1",
    ] {
        let chunk = parse(source).unwrap();
        assert_eq!(chunk.statements.len(), 1, "{source:?}");
    }
    assert_eq!(
        parse("#!/usr/bin/env lua").unwrap().statements,
        [] as [Statement; 0]
    );
    assert!(parse("value = 1\n#! not a shebang").is_err());
}

#[test]
fn parses_lua_hexadecimal_integers() {
    let chunk = parse("value = 0x7fffffff").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected assignment");
    };
    assert!(
        matches!(values[0], Expr::Number(value, _) if (value - 2_147_483_647.0).abs() < f64::EPSILON)
    );
    assert_eq!(
        parse("value = 0xnope").unwrap_err().message,
        "invalid hexadecimal number"
    );
}

#[test]
fn parses_luajit_hexadecimal_floating_constants() {
    let chunk =
        parse("a = 0xep9; b = 0xep+9; c = 0xep-9; d = 0X1.8P+1; e = 0x.8p+1; f = 0x1p0").unwrap();
    let values: Vec<f64> = chunk
        .statements
        .iter()
        .map(|statement| match statement {
            Statement::Assign { values, .. } => match values.as_slice() {
                [Expr::Number(value, _)] => *value,
                _ => panic!("expected one numeric value"),
            },
            _ => panic!("expected assignment"),
        })
        .collect();
    assert_eq!(values, [7168.0, 7168.0, 0.027_343_75, 3.0, 1.0, 1.0]);
}

#[test]
fn hexadecimal_float_conversion_has_f64_overflow_and_underflow_behavior() {
    let chunk = parse("a = 0x1p1024; b = 0x1p-1074; c = 0x1p-1075").unwrap();
    let values: Vec<f64> = chunk
        .statements
        .iter()
        .map(|statement| match statement {
            Statement::Assign { values, .. } => match values.as_slice() {
                [Expr::Number(value, _)] => *value,
                _ => panic!("expected one numeric value"),
            },
            _ => panic!("expected assignment"),
        })
        .collect();
    assert!(values[0].is_infinite());
    assert_eq!(values[1].to_bits(), 1);
    assert_eq!(values[2].to_bits(), 0);
}

#[test]
fn hexadecimal_float_conversion_rounds_to_nearest_even() {
    let chunk = parse(
        "a = 0x1.00000000000008p0; b = 0x1.000000000000080000000001p0; c = 0x1.fffffffffffff8p1023",
    )
    .unwrap();
    let values: Vec<f64> = chunk
        .statements
        .iter()
        .map(|statement| match statement {
            Statement::Assign { values, .. } => match values.as_slice() {
                [Expr::Number(value, _)] => *value,
                _ => panic!("expected one numeric value"),
            },
            _ => panic!("expected assignment"),
        })
        .collect();
    assert_eq!(values[0].to_bits(), 0x3ff0_0000_0000_0000);
    assert_eq!(values[1].to_bits(), 0x3ff0_0000_0000_0001);
    assert!(values[2].is_infinite());
}

#[test]
fn rejects_invalid_hexadecimal_float_constants() {
    for source in [
        "value = 0x",
        "value = 0x.p1",
        "value = 0x1p",
        "value = 0x1p+",
    ] {
        assert!(parse(source).is_err(), "{source}");
    }
}

#[test]
fn parses_lua_leading_and_trailing_decimal_points_without_eating_concat() {
    let chunk = parse("value = 1.+.1; joined = 1..'x'").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected numeric assignment");
    };
    assert!(matches!(
        values[0],
        Expr::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let Statement::Assign { values, .. } = &chunk.statements[1] else {
        panic!("expected concatenation assignment");
    };
    assert!(matches!(
        values[0],
        Expr::Binary {
            op: BinaryOp::Concat,
            ..
        }
    ));
}

#[test]
fn string_literals_preserve_escaped_non_utf8_bytes() {
    let chunk = parse("value = '\\255\\0'").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == [255, 0]
    ));
}

#[test]
fn escaped_newlines_accept_both_two_byte_line_ending_orders() {
    for source in [b"value = '\\\r\nnext'" as &[u8], b"value = '\\\n\rnext'"] {
        let chunk = parse_bytes(source).unwrap();
        let Statement::Assign { values, .. } = &chunk.statements[0] else {
            panic!("expected assignment");
        };
        assert!(matches!(
            &values[0],
            Expr::String(value, _) if value.as_bytes() == b"\nnext"
        ));
    }
}

#[test]
fn byte_oriented_source_preserves_non_utf8_string_bytes() {
    let long = parse_bytes(b"value = [[\xff\x80]]").unwrap();
    let Statement::Assign { values, .. } = &long.statements[0] else {
        panic!("expected long-string assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == [255, 128]
    ));

    let short = parse_bytes(&[
        b'v', b'a', b'l', b'u', b'e', b' ', b'=', b' ', b'\'', 255, b'\'',
    ])
    .unwrap();
    let Statement::Assign { values, .. } = &short.statements[0] else {
        panic!("expected short-string assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == [255]
    ));
}

#[test]
fn invalid_non_utf8_bytes_outside_literals_are_syntax_errors() {
    let error = parse_bytes(&[0xff]).unwrap_err();
    assert_eq!(error.message, "unexpected character");
    assert_eq!(error.span, Span { start: 0, end: 1 });
}

#[test]
fn long_bracket_strings_keep_literal_bytes_and_delimiter_levels() {
    let chunk = parse(
        "first = [[\nline\\ntext]]; second = [=[a ] not a close ]=]; third = [==[h\u{e9}]==]",
    )
    .unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected first assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == b"line\\ntext"
    ));
    let Statement::Assign { values, .. } = &chunk.statements[1] else {
        panic!("expected second assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == b"a ] not a close "
    ));
    let Statement::Assign { values, .. } = &chunk.statements[2] else {
        panic!("expected third assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _) if value.as_bytes() == "h\u{e9}".as_bytes()
    ));
}

#[test]
fn long_brackets_normalize_every_line_ending() {
    let chunk = parse("value = [[\r\nfirst\rsecond\nthird\r\nfourth\n\rfifth]]").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::String(value, _)
            if value.as_bytes() == b"first\nsecond\nthird\nfourth\nfifth"
    ));
    // Comments use the same delimiter scanner, so mixed line endings do
    // not affect the next token after their closing bracket.
    let chunk = parse("--[=[\r\nignored\r\nbody\n\r]=]\r\nvalue = 4").unwrap();
    assert_eq!(chunk.statements.len(), 1);
}

#[test]
fn rejects_non_target() {
    assert_eq!(
        parse("1 = nil").unwrap_err().message,
        "expected assignment target"
    );
}

#[test]
fn parses_modern_string_escapes() {
    let parsed = parse_with_config(
        "return '\\x41\\u{1f642}\\z  x'",
        ParserConfig::new(CompatibilityProfile::Lua55),
    )
    .unwrap();
    let Statement::Return { values, .. } = &parsed.statements[0] else {
        panic!("expected return");
    };
    let Expr::String(value, _) = &values[0] else {
        panic!("expected string");
    };
    assert_eq!(value.as_bytes(), "A🙂x".as_bytes());
    assert!(
        parse_with_config(
            "return '\\x00\\xff'",
            ParserConfig::new(CompatibilityProfile::Lua52),
        )
        .is_ok()
    );
    assert!(
        parse_with_config(
            "return '\\x41'",
            ParserConfig::new(CompatibilityProfile::Lua51),
        )
        .is_err()
    );
    assert!(
        parse_with_config(
            "return '\\x4z'",
            ParserConfig::new(CompatibilityProfile::Lua52),
        )
        .is_err()
    );
}
