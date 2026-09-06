use super::*;

#[test]
fn lua51_and_luajit_reject_ambiguous_parenthesized_call_newlines() {
    let source = "value = target\n(3)";
    let open = source.rfind('(').unwrap();
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let error = parse_with_config(source, ParserConfig::new(profile)).unwrap_err();
        assert_eq!(
            error.span,
            Span {
                start: open,
                end: open + 1
            }
        );
        assert_eq!(
            error.message,
            "ambiguous syntax (function call x new statement)"
        );
    }
    assert!(
        parse(source).is_err(),
        "the default LuaJIT profile must reject it"
    );
}

#[test]
fn later_profiles_and_unambiguous_lua51_calls_cross_newlines() {
    let lua52 = ParserConfig::new(CompatibilityProfile::Lua52);
    assert!(parse_with_config("value = target\n(3)", lua52).is_ok());

    let lua51 = ParserConfig::new(CompatibilityProfile::Lua51);
    assert!(parse_with_config("value = target(\n3)", lua51).is_ok());
    assert!(parse_with_config("value = target\n{3}", lua51).is_ok());
    // Lua 5.1's table-field name lookahead updates its last-token line.
    assert!(parse_with_config("return {target\n(3)}", lua51).is_ok());
}

#[test]
fn pratt_machine_matches_legacy_expression_parser() {
    for source in [
        b"1 + 2 * 3 - 4 / 2" as &[u8],
        b"-2^2",
        b"2^-2",
        b"2^3^2",
        b"not false and true or nil",
        b"identifier .. 'value'",
        b"f(1, t.x)[key] .. {label = 'x', [2] = 3}",
        b"receiver:method { nested = 1 }",
        b"(identifier).field('value')",
    ] {
        let mut tokens = Lexer::new(source).lex().unwrap();
        tokens.pop();
        let mut legacy_tokens = tokens.clone();
        legacy_tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span {
                start: source.len(),
                end: source.len(),
            },
        });
        let mut legacy = Parser::new(legacy_tokens, source.len());
        let expected = legacy.expression().unwrap();

        let mut machine = PrattMachine::new(source.len());
        for token in tokens {
            match machine.resume(PrattInput::Token(token)) {
                PrattPoll::NeedToken => {}
                PrattPoll::Parsed(_) | PrattPoll::Boundary(_, _) => {
                    panic!("{source:?}: expression parsed before end")
                }
                PrattPoll::Failed(error) => panic!("{source:?}: {error}"),
            }
        }
        match machine.resume(PrattInput::End) {
            PrattPoll::Parsed(actual) => assert_eq!(actual, expected),
            PrattPoll::NeedToken | PrattPoll::Boundary(_, _) | PrattPoll::Failed(_) => {
                panic!("complete expression failed")
            }
        }
    }
}

#[test]
fn pratt_machine_keeps_terminal_failures_terminal() {
    let tokens = stream_tokens(&[b"*", b"a"]).unwrap();
    let mut machine = PrattMachine::new(2);
    assert!(matches!(
        machine.resume(PrattInput::Token(tokens[0].clone())),
        PrattPoll::Failed(_)
    ));
    assert!(matches!(
        machine.resume(PrattInput::Token(tokens[1].clone())),
        PrattPoll::Failed(_)
    ));
}

#[test]
fn pratt_machine_rejects_broken_table_fields_at_the_delimiter() {
    let source = b"{ [key] value }";
    let mut tokens = Lexer::new(source).lex().unwrap();
    tokens.pop();
    let mut machine = PrattMachine::new(source.len());
    for token in tokens {
        let poll = machine.resume(PrattInput::Token(token));
        if matches!(poll, PrattPoll::Failed(_)) {
            assert_eq!(machine.transition_count(), 5);
            return;
        }
    }
    panic!("broken field should fail before end");
}

#[test]
fn hexadecimal_float_exponents_do_not_eat_operators_or_concatenation() {
    let chunk = parse("sum = 0xe+9; joined = 0x1..'x'").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected assignment");
    };
    assert!(matches!(
        values.as_slice(),
        [Expr::Binary {
            op: BinaryOp::Add,
            ..
        }]
    ));
    let Statement::Assign { values, .. } = &chunk.statements[1] else {
        panic!("expected assignment");
    };
    assert!(matches!(
        values.as_slice(),
        [Expr::Binary {
            op: BinaryOp::Concat,
            ..
        }]
    ));
}

#[test]
fn precedence() {
    let chunk = parse("a = 1 + 2 == 3").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!()
    };
    assert!(matches!(
        &values[0],
        Expr::Binary {
            op: BinaryOp::Equal,
            ..
        }
    ));
}

#[test]
fn parses_lua_conditional_precedence_and_unary_operators() {
    let chunk = parse("result = a or b and not #t < 3").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!();
    };
    let Expr::Binary {
        op: BinaryOp::Or,
        right,
        ..
    } = &values[0]
    else {
        panic!("expected top-level or");
    };
    let Expr::Binary {
        op: BinaryOp::And,
        right,
        ..
    } = right.as_ref()
    else {
        panic!("expected and to bind more tightly than or");
    };
    let Expr::Binary {
        op: BinaryOp::LessThan,
        left,
        ..
    } = right.as_ref()
    else {
        panic!("expected relation to bind more tightly than and");
    };
    assert!(matches!(
        left.as_ref(),
        Expr::Unary {
            op: UnaryOp::Not,
            expression,
            ..
        } if matches!(expression.as_ref(), Expr::Unary { op: UnaryOp::Length, .. })
    ));
}

#[test]
fn parses_power_precedence_and_call_shorthand() {
    let chunk = parse("a = -2^2; b = 2^-2; c = 2^3^2; f'x'; g{1}; s:send't'").unwrap();
    let Statement::Assign { values, .. } = &chunk.statements[0] else {
        panic!("expected first assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::Unary {
            op: UnaryOp::Negate,
            expression,
            ..
        } if matches!(expression.as_ref(), Expr::Binary { op: BinaryOp::Power, .. })
    ));
    let Statement::Assign { values, .. } = &chunk.statements[1] else {
        panic!("expected second assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::Binary {
            op: BinaryOp::Power,
            right,
            ..
        } if matches!(right.as_ref(), Expr::Unary { op: UnaryOp::Negate, .. })
    ));
    let Statement::Assign { values, .. } = &chunk.statements[2] else {
        panic!("expected third assignment");
    };
    assert!(matches!(
        &values[0],
        Expr::Binary {
            op: BinaryOp::Power,
            right,
            ..
        } if matches!(right.as_ref(), Expr::Binary { op: BinaryOp::Power, .. })
    ));
    assert!(matches!(
        chunk.statements[3],
        Statement::Call(Expr::Call { .. })
    ));
    assert!(matches!(
        chunk.statements[4],
        Statement::Call(Expr::Call { .. })
    ));
    assert!(matches!(
        chunk.statements[5],
        Statement::Call(Expr::MethodCall { .. })
    ));
}

#[test]
fn parses_anonymous_function_expression() {
    let chunk = parse("local handler = function(value) return value end").unwrap();
    let Statement::Local { values, .. } = &chunk.statements[0] else {
        panic!("expected local declaration");
    };
    assert!(matches!(&values[0], Expr::Function { parameters, .. } if parameters.len() == 1));
}

#[test]
fn postfix_suffixes_require_prefix_expressions() {
    for source in [
        "a = {}\n(function (x) a=x end)(23)",
        "a = {} (function (x) a=x end)(23)",
    ] {
        let chunk = parse(source).unwrap();
        assert_eq!(chunk.statements.len(), 2, "{source:?}");
        assert!(matches!(
            &chunk.statements[0],
            Statement::Assign { values, .. } if matches!(values[0], Expr::Table { .. })
        ));
        assert!(matches!(
            &chunk.statements[1],
            Statement::Call(Expr::Call { callee, .. })
                if matches!(callee.as_ref(), Expr::Parenthesized { expression, .. }
                    if matches!(expression.as_ref(), Expr::Function { .. }))
        ));
    }
    assert!(parse("value = {}[1]").is_err());
    assert!(parse("function () end()").is_err());
}

#[test]
fn parses_vararg_parameters_and_expressions() {
    let chunk = parse(
            "function fixed(first, ...) return first, ... end; local function only(...) return {...} end; local callback = function(value, ...) return value end",
        )
        .unwrap();
    let Statement::Function {
        parameters,
        vararg,
        body,
        ..
    } = &chunk.statements[0]
    else {
        panic!("expected function declaration");
    };
    assert_eq!(parameters.len(), 1);
    assert!(*vararg);
    let Statement::Return { values, .. } = &body.statements[0] else {
        panic!("expected return");
    };
    assert!(matches!(values[1], Expr::Vararg(_)));
    assert!(matches!(
        &chunk.statements[1],
        Statement::LocalFunction { vararg: true, .. }
    ));
    let Statement::Local { values, .. } = &chunk.statements[2] else {
        panic!("expected anonymous function");
    };
    assert!(matches!(values[0], Expr::Function { vararg: true, .. }));
}

#[test]
fn lua53_bitwise_and_floor_division_have_lua_precedence_and_profile_gates() {
    let parsed = parse_with_config(
        "return 1 | 2 ~ 3 & 4 << 1 // 2",
        ParserConfig::new(CompatibilityProfile::Lua53),
    )
    .unwrap();
    let Statement::Return { values, .. } = &parsed.statements[0] else {
        panic!("expected return");
    };
    assert!(matches!(
        values[0],
        Expr::Binary {
            op: BinaryOp::BitOr,
            ..
        }
    ));
    assert!(
        parse_with_config(
            "return 4 // 2",
            ParserConfig::new(CompatibilityProfile::Lua52),
        )
        .is_err()
    );
    assert!(
        parse_with_config(
            "return 1 & 1",
            ParserConfig::new(CompatibilityProfile::LuaJit21),
        )
        .is_err()
    );
}
