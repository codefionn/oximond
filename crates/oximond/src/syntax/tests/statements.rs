use super::*;

#[test]
fn lua51_rejects_empty_statements_but_keeps_semicolon_terminators() {
    let lua51 = ParserConfig::new(CompatibilityProfile::Lua51);
    for source in [";", "a = 1;;", "return;;", "do ; end", "if true then ; end"] {
        let error = parse_with_config(source, lua51).unwrap_err();
        assert_eq!(error.message, "unexpected ';'", "source={source:?}");
    }
    for source in [
        "a = 1; b = 2",
        "return;",
        "do a = 1; end;",
        "local t = {1; 2; key = 3}",
    ] {
        assert!(
            parse_with_config(source, lua51).is_ok(),
            "source={source:?}"
        );
    }
}

#[test]
fn luajit_profile_keeps_empty_statement_extension() {
    let luajit = ParserConfig::new(CompatibilityProfile::LuaJit21);
    for source in [";", "a = 1;;", "return;;", "do ; end"] {
        assert!(
            parse_with_config(source, luajit).is_ok(),
            "source={source:?}"
        );
    }
    assert!(parse(";").is_ok(), "the convenience API must remain LuaJIT");
}

#[test]
fn chunk_machine_matches_legacy_for_root_statement_core() {
    for source in [
        b"local a, b = 1 + 2, f { key = 3 }; a, t.x = b, a" as &[u8],
        b"f(1, 2); receiver:method 'value'; return a, b * 3",
        b"local value; value = (1 + 2) * 3",
        b"do local value = 1; f(value) end",
        b"while ready do f(); break end",
        b"repeat local value = 1; break until value == 1",
        b"if ready then f() elseif pending then return else g() end",
        b"if a then if b then return end elseif c then do return end else return end",
        b"for i = 1, 10, 2 do break end; f(i)",
        b"for key, value in pairs(t), next, state do if key then break end end; f()",
    ] {
        let expected = parse_with_legacy_parser(source).unwrap();
        assert_eq!(
            parse_with_chunk_machine(source).unwrap(),
            expected,
            "{source:?}"
        );
    }
}

#[test]
fn chunk_machine_if_matches_legacy_under_every_byte_split() {
    for source in [
        b"if ready then f() elseif pending then return else g() end" as &[u8],
        b"if a then if b then return end elseif c then do return end else return end",
        b"while a do if b then repeat return until c elseif d then break else return end end",
    ] {
        let expected = parse_with_legacy_parser(source).unwrap();
        for split in 1..=source.len() {
            assert_eq!(
                parse_streaming_with_chunk_machine(source, split).unwrap(),
                expected,
                "split={split}, source={source:?}"
            );
        }
    }
}

#[test]
fn chunk_machine_if_consumes_terminators_without_refeeding() {
    let source = b"if a then return elseif b then return else return end";
    let expected = parse_with_legacy_parser(source).unwrap();
    let actual = parse_streaming_with_chunk_machine(source, 1).unwrap();
    assert_eq!(actual, expected);
    let Statement::If {
        body,
        else_ifs,
        else_body,
        ..
    } = &actual.statements[0]
    else {
        panic!("expected if statement");
    };
    assert!(
        matches!(body.statements.as_slice(), [Statement::Return { values, .. }] if values.is_empty())
    );
    assert!(
        matches!(else_ifs[0].body.statements.as_slice(), [Statement::Return { values, .. }] if values.is_empty())
    );
    assert!(
        matches!(else_body.as_ref().unwrap().statements.as_slice(), [Statement::Return { values, .. }] if values.is_empty())
    );
}

#[test]
fn chunk_machine_reports_unclosed_and_mismatched_block_delimiters() {
    for (source, message) in [
        (
            b"if true then local a = 1" as &[u8],
            "expected 'end', 'elseif', or 'else'",
        ),
        (b"while true do until true" as &[u8], "expected 'end'"),
        (b"repeat end" as &[u8], "expected 'until'"),
        (b"end" as &[u8], "unexpected block terminator"),
    ] {
        let error = parse_with_chunk_machine(source).unwrap_err();
        assert_eq!(error.message, message, "{source:?}");
    }
}

#[test]
fn chunk_machine_for_matches_legacy_under_every_byte_split() {
    for source in [
        b"for i = -2, 8, 2 do break end; f(i)" as &[u8],
        b"for key, value in pairs(t), next, state do if key then break end end; f()",
        b"for i = 1, 2 do for _, value in ipairs(items) do repeat break until value end end",
        b"for i = 1, 0 do end; for key in pairs(t) do end; trailing()",
    ] {
        let expected = parse_with_legacy_parser(source).unwrap();
        for split in 1..=source.len() {
            assert_eq!(
                parse_streaming_with_chunk_machine(source, split).unwrap(),
                expected,
                "split={split}, source={source:?}"
            );
        }
    }
}

#[test]
fn chunk_machine_for_enforces_header_and_loop_rules() {
    for (source, message) in [
        (b"for = 1, 2 do end" as &[u8], "expected name"),
        (
            b"for i 1, 2 do end" as &[u8],
            "expected 'in' after loop names",
        ),
        (
            b"for i = 1 do end" as &[u8],
            "expected ',' after loop start",
        ),
        (b"for i = 1, 2" as &[u8], "expected 'do' after numeric for"),
        (
            b"for key, value pairs(t) do end" as &[u8],
            "expected 'in' after loop names",
        ),
        (
            b"for key in pairs(t)" as &[u8],
            "expected 'do' after iterator expressions",
        ),
        (b"for i = 1, 2 do" as &[u8], "expected 'end'"),
        (b"for key in pairs(t) do" as &[u8], "expected 'end'"),
    ] {
        let error = parse_with_chunk_machine(source).unwrap_err();
        assert_eq!(error.message, message, "{source:?}");
    }
    assert!(parse_with_chunk_machine(b"for i = 1, 2 do break end").is_ok());
    assert!(parse_with_chunk_machine(b"for key in pairs(t) do break end").is_ok());
}

#[test]
fn chunk_machine_functions_match_legacy_under_every_byte_split() {
    for source in [
            b"function count(n) if n <= 0 then return 0 end return count(n - 1) end; count(2)"
                as &[u8],
            b"function a.b.c:send(value, ...) return self, value, ... end; a.b.c:send('x')",
            b"local function outer(value) local nested = function(...) return value, ... end; return nested end",
            b"local callback = function(value) for i = 1, value do if i == 2 then break end end return value end; callback(3)",
            b"local function empty() end; function outer() local function inner(...) return ... end return inner end",
            b"function outer() consume(function(...) return ... end, { handler = function(...) return ... end }) end",
        ] {
            let expected = parse_with_legacy_parser(source).unwrap();
            for split in 1..=source.len() {
                assert_eq!(
                    parse_streaming_with_chunk_machine(source, split).unwrap(),
                    expected,
                    "split={split}, source={source:?}"
                );
            }
        }
}

#[test]
fn chunk_machine_functions_enforce_parameters_scopes_and_closers() {
    for (source, message) in [
        (b"function () end" as &[u8], "expected local name"),
        (b"function f(x, ) end" as &[u8], "expected local name"),
        (
            b"function f(..., x) end" as &[u8],
            "expected ')' after parameters",
        ),
        (b"function f(x)" as &[u8], "expected 'end'"),
        (b"local function f(...)" as &[u8], "expected 'end'"),
        (b"function f() break end" as &[u8], "break outside a loop"),
        (
            b"while true do function inner() break end end" as &[u8],
            "break outside a loop",
        ),
        (
            b"function f() return ... end" as &[u8],
            "cannot use '...' outside a vararg function",
        ),
        (
            b"function f(...) local nested = function() return ... end end" as &[u8],
            "cannot use '...' outside a vararg function",
        ),
    ] {
        let error = parse_with_chunk_machine(source).unwrap_err();
        assert_eq!(error.message, message, "{source:?}");
    }
    assert!(parse_with_chunk_machine(b"function f(...) return ... end").is_ok());
}

#[test]
fn nested_assignment_function_varargs_remain_in_their_function_scope() {
    let source = b"function outer() if safe then return else env.loadstring = function(str, ...) local func, err = loadstring(str, ...) return func end end end";
    let parsed = parse_with_chunk_machine(source);
    assert!(parsed.is_ok(), "{parsed:?}");
    assert!(crate::syntax::parse_bytes(source).is_ok());
    let invalid = b"function outer() if safe then return else env.loadstring = function(str) return ... end end end";
    assert_eq!(
        parse_with_chunk_machine(invalid).unwrap_err().message,
        "cannot use '...' outside a vararg function",
    );
}

#[test]
fn lua_diagnostics_truncate_literal_chunk_names_at_the_first_line() {
    let source = b"local value =";
    let source_name = LuaString::from("local value =\nsecond line");
    let error = parse_bytes(source)
        .unwrap_err()
        .with_source(&source_name, source);
    assert!(
        error
            .diagnostic()
            .as_bytes()
            .starts_with(b"[string \"local value =...\"]:1:")
    );
}

#[test]
fn local_limit_diagnostic_names_the_enclosing_function_line() {
    let mut source = b"\nfunction overloaded()\nlocal ".to_vec();
    for index in 0..=200 {
        source.extend_from_slice(format!("value{index},").as_bytes());
    }
    source.extend_from_slice(b"tail\nend");
    let error = parse_bytes(&source)
        .unwrap_err()
        .with_source(&LuaString::from("=limit"), &source);
    assert!(error.diagnostic().to_lossy().contains("function at line 2"));
}

#[test]
fn rejects_excessive_syntax_nesting_before_growing_parser_stacks() {
    for (initial, repeated) in [
        ("a=", "{"),
        ("a=", "("),
        ("", "a("),
        ("", "do "),
        ("", "while a do "),
        ("", "if a then else "),
        ("a=", "a.."),
        ("a=", "a^"),
    ] {
        let source = format!("local a; {initial}{}", repeated.repeat(400));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, "chunk has too many syntax levels");
    }
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = format!("local a; {}", "function foo () ".repeat(400));
            let error = parse(&source).unwrap_err();
            assert!(error.message.contains("syntax levels"));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn parses_nested_blocks_up_to_the_syntax_limit_on_a_small_stack() {
    // Block machines recurse once per nesting level. Debug builds give every
    // binding in a function its own stack slot, so the recursive path must
    // stay small enough for embedders' threads at Lua's 200-level limit.
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(|| {
            let source = format!("local a; {}{}", "do ".repeat(199), "end ".repeat(199));
            let chunk = parse(&source).unwrap();
            assert_eq!(chunk.statements.len(), 2);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn long_bracket_comments_accept_levels_and_unterminated_forms_report_cleanly() {
    let chunk = parse("--[=[\nlocal ignored = [==[not syntax]==]\n]=]\nvalue = 4").unwrap();
    assert_eq!(chunk.statements.len(), 1);
    assert_eq!(
        parse("value = [=[unfinished").unwrap_err().message,
        "unterminated long string"
    );
    assert_eq!(
        parse("--[==[unfinished").unwrap_err().message,
        "unterminated long comment"
    );
}

#[test]
fn parses_local_function_else_and_right_associative_concatenation() {
    let chunk = parse(
            "local function choose(value) if value then return 'yes' else return 'no' end end; out = 'a' .. 'b' .. 'c'",
        )
        .unwrap();
    let Statement::LocalFunction { name, body, .. } = &chunk.statements[0] else {
        panic!("expected a local function");
    };
    assert_eq!(&*name.text, "choose");
    let Statement::If { else_body, .. } = &body.statements[0] else {
        panic!("expected function body to contain an if");
    };
    assert!(else_body.is_some());
    let Statement::Assign { values, .. } = &chunk.statements[1] else {
        panic!("expected concatenation assignment");
    };
    let Expr::Binary {
        op: BinaryOp::Concat,
        right,
        ..
    } = &values[0]
    else {
        panic!("expected concatenation");
    };
    assert!(matches!(
        right.as_ref(),
        Expr::Binary {
            op: BinaryOp::Concat,
            ..
        }
    ));
}

#[test]
fn parses_multiple_elseif_arms_and_a_final_else() {
    let chunk = parse(
            "if first then selected = 1 elseif second then selected = 2 elseif third then selected = 3 else selected = 4 end",
        )
        .unwrap();
    let Statement::If {
        else_ifs,
        else_body,
        ..
    } = &chunk.statements[0]
    else {
        panic!("expected if statement");
    };
    assert_eq!(else_ifs.len(), 2);
    assert!(matches!(
        else_ifs[0].body.statements[0],
        Statement::Assign { .. }
    ));
    assert!(else_body.is_some());
}

#[test]
fn parses_plain_function_declaration() {
    let chunk =
        parse("function count(n) if n <= 0 then return 0 end return count(n - 1) end").unwrap();
    let Statement::Function {
        name, parameters, ..
    } = &chunk.statements[0]
    else {
        panic!("expected a function declaration");
    };
    assert_eq!(name.segments.len(), 1);
    assert_eq!(&*name.terminal().text, "count");
    assert_eq!(parameters.len(), 1);
}

#[test]
fn parses_table_fields_and_generic_for() {
    let chunk = parse(
            "local t = {10; name = 20, [3] = 30, 40}; for key, value in pairs(t), t do total = value end",
        )
        .unwrap();
    let Statement::Local { values, .. } = &chunk.statements[0] else {
        panic!("expected table local");
    };
    let Expr::Table { fields, .. } = &values[0] else {
        panic!("expected table constructor");
    };
    assert!(matches!(fields[0], TableField::List(_)));
    assert!(matches!(fields[1], TableField::Name { .. }));
    assert!(matches!(fields[2], TableField::Key { .. }));
    assert!(matches!(fields[3], TableField::List(_)));
    let Statement::GenericFor {
        names, expressions, ..
    } = &chunk.statements[1]
    else {
        panic!("expected generic for");
    };
    assert_eq!(names.len(), 2);
    assert_eq!(expressions.len(), 2);
}

#[test]
fn rejects_invalid_vararg_parameter_and_scope_forms() {
    for source in [
        "function f(..., value) end",
        "function f(value, ) end",
        "function f() return ... end",
        "function f(...) local nested = function() return ... end end",
    ] {
        assert!(parse(source).is_err(), "{source}");
    }
    // The outer chunk is a vararg function in Lua 5.1.
    assert!(matches!(
        parse("value = ...").unwrap().statements[0],
        Statement::Assign { .. }
    ));
}

#[test]
fn lua55_parses_named_varargs_and_const_global_forms() {
    let lua55 = ParserConfig::new(CompatibilityProfile::Lua55);
    let chunk = parse_with_config(
            "global <const> *; global<const> answer; global later<const>; function f(head, ... extras) return extras end",
            lua55,
        )
        .unwrap();
    assert!(matches!(
        &chunk.statements[0],
        Statement::Global {
            mode: GlobalDeclarationMode::All,
            all_const: true,
            ..
        }
    ));
    assert!(matches!(
        &chunk.statements[1],
        Statement::Global { const_names, .. } if const_names.as_slice() == [true]
    ));
    assert!(matches!(
        &chunk.statements[2],
        Statement::Global { const_names, .. } if const_names.as_slice() == [true]
    ));
    assert!(matches!(
        &chunk.statements[3],
        Statement::Function { vararg_name: Some(name), .. } if &*name.text == "extras"
    ));
    assert!(
        parse_with_config(
            "function f(... extras) end",
            ParserConfig::new(CompatibilityProfile::Lua54)
        )
        .is_err()
    );
}

#[test]
fn lua52_gotos_and_labels_parse_while_lua51_keeps_goto_as_a_name() {
    let lua52 = ParserConfig::new(CompatibilityProfile::Lua52);
    let chunk = parse_with_config("goto done; ::done::", lua52).unwrap();
    assert!(matches!(chunk.statements[0], Statement::Goto { .. }));
    assert!(matches!(chunk.statements[1], Statement::Label { .. }));
    assert!(parse_with_config("goto = 1", ParserConfig::new(CompatibilityProfile::Lua51)).is_ok());
    assert!(parse_with_config("::done::", ParserConfig::new(CompatibilityProfile::Lua51)).is_err());
}

#[test]
fn local_attributes_are_retained_and_gated() {
    let parsed = parse_with_config(
        "local x<const>, y<close> = 1, resource()",
        ParserConfig::new(CompatibilityProfile::Lua54),
    )
    .unwrap();
    let Statement::Local { attributes, .. } = &parsed.statements[0] else {
        panic!("expected local declaration");
    };
    assert_eq!(attributes, &[LocalAttribute::Const, LocalAttribute::Close]);
    let parsed = parse_with_config(
        "local<const> x, y = 1, 2",
        ParserConfig::new(CompatibilityProfile::Lua55),
    )
    .unwrap();
    let Statement::Local { attributes, .. } = &parsed.statements[0] else {
        panic!("expected prefixed local declaration");
    };
    assert_eq!(attributes, &[LocalAttribute::Const, LocalAttribute::Const]);
    assert!(
        parse_with_config(
            "local<const> x = 1",
            ParserConfig::new(CompatibilityProfile::Lua54),
        )
        .is_err()
    );
    assert!(
        parse_with_config(
            "local x<const> = 1",
            ParserConfig::new(CompatibilityProfile::Lua53),
        )
        .is_err()
    );
}

#[test]
fn lua55_global_declarations_parse_and_are_gated() {
    let parsed = parse_with_config(
        "global answer, other = 42, 7; global *; global none",
        ParserConfig::new(CompatibilityProfile::Lua55),
    )
    .unwrap();
    assert!(matches!(
        parsed.statements[0],
        Statement::Global {
            mode: GlobalDeclarationMode::Names,
            ..
        }
    ));
    assert!(matches!(
        parsed.statements[1],
        Statement::Global {
            mode: GlobalDeclarationMode::All,
            ..
        }
    ));
    assert!(matches!(
        parsed.statements[2],
        Statement::Global {
            mode: GlobalDeclarationMode::None,
            ..
        }
    ));
    let legacy_global =
        parse_with_config("global = 1", ParserConfig::new(CompatibilityProfile::Lua54));
    assert!(legacy_global.is_ok(), "{legacy_global:?}");
    for source in ["global X<close>", "global <close> *"] {
        let error =
            parse_with_config(source, ParserConfig::new(CompatibilityProfile::Lua55)).unwrap_err();
        assert!(error.message.contains("cannot be"), "{error:?}");
    }
}

#[test]
fn parses_methods_functions_returns_and_strings() {
    let chunk = parse("function t:set(x) self.a = x end; t:set(\"foo\\n\")").unwrap();
    let Statement::Function {
        name,
        parameters,
        body,
        ..
    } = &chunk.statements[0]
    else {
        panic!()
    };
    assert!(name.method);
    assert_eq!(parameters.len(), 1);
    assert!(matches!(body.statements[0], Statement::Assign { .. }));
    assert!(matches!(
        chunk.statements[1],
        Statement::Call(Expr::MethodCall { .. })
    ));
}

#[test]
fn parses_deep_dot_and_method_function_paths() {
    let chunk =
        parse("function a.b.c.f(x) return x end; function a.b.c:m(x) return self end").unwrap();
    let Statement::Function { name, .. } = &chunk.statements[0] else {
        panic!("expected dotted function");
    };
    assert!(!name.method);
    assert_eq!(
        name.segments
            .iter()
            .map(|segment| &*segment.text)
            .collect::<Vec<_>>(),
        ["a", "b", "c", "f"]
    );
    let Statement::Function { name, .. } = &chunk.statements[1] else {
        panic!("expected method function");
    };
    assert!(name.method);
    assert_eq!(&*name.terminal().text, "m");
}

#[test]
fn parses_numeric_for_if_and_arithmetic() {
    let chunk = parse("for x = -2, 2, 0.5 do if x ~= 0 then y = x%2 - x/2*3 end end").unwrap();
    let Statement::NumericFor {
        name,
        step: Some(_),
        body,
        ..
    } = &chunk.statements[0]
    else {
        panic!()
    };
    assert_eq!(&*name.text, "x");
    let Statement::If {
        condition, body, ..
    } = &body.statements[0]
    else {
        panic!()
    };
    assert!(matches!(
        condition,
        Expr::Binary {
            op: BinaryOp::NotEqual,
            ..
        }
    ));
    assert!(matches!(body.statements[0], Statement::Assign { .. }));
}

#[test]
fn parses_while_repeat_and_nested_breaks() {
    let chunk = parse(
            "while ready do repeat local value = 1; if value then break end until value == 2; break end",
        )
        .unwrap();
    let Statement::While { body, .. } = &chunk.statements[0] else {
        panic!("expected while statement");
    };
    let Statement::Repeat {
        body: repeat_body,
        condition,
        ..
    } = &body.statements[0]
    else {
        panic!("expected repeat statement");
    };
    assert!(matches!(repeat_body.statements[1], Statement::If { .. }));
    assert!(matches!(
        condition,
        Expr::Binary {
            op: BinaryOp::Equal,
            ..
        }
    ));
    assert!(matches!(body.statements[1], Statement::Break { .. }));
}

#[test]
fn rejects_break_outside_its_own_function_loop() {
    assert_eq!(parse("break").unwrap_err().message, "break outside a loop");
    assert_eq!(
        parse("while true do function inner() break end end")
            .unwrap_err()
            .message,
        "break outside a loop"
    );
    assert!(parse("for i = 1, 2 do break end").is_ok());
}
