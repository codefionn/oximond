//! Large real-world sources (for example Luanti games such as Mineclonia)
//! must compile wherever stock Lua 5.1 and `LuaJIT` 2.1 accept them: long
//! concatenation chains, deep call nesting, wide multiple assignments, deeply
//! nested constructors and blocks, and `LuaJIT`'s Lua 5.2 syntax extensions.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source)
        .unwrap_or_else(|error| panic!("{profile}: {error}"))
}

fn run_both(source: &str) -> Vec<Value> {
    let lua51 = run(CompatibilityProfile::Lua51, source);
    assert_eq!(run(CompatibilityProfile::LuaJit21, source), lua51);
    lua51
}

fn nested(depth: usize, open: &str, inner: &str, close: &str) -> String {
    format!("{}{inner}{}", open.repeat(depth), close.repeat(depth))
}

#[test]
fn long_concatenation_of_calls_reuses_scratch_registers() {
    // One register per operand, as in Lua; each call's scratch registers are
    // free again before the next operand.
    let chain = vec![r#"S("x").."\n""#; 95].join("..");
    let source = format!("local S = tostring\nlocal t = {{ text = {chain} }}\nreturn #t.text");
    assert_eq!(run_both(&source), vec![Value::Number(190.0)]);
}

#[test]
fn deeply_nested_call_arguments_compile() {
    let mut expression = String::from("1");
    for _ in 0..80 {
        expression = format!("f(2, {expression})");
    }
    let source = format!("local function f(a, b) return a + b end\nreturn {expression}");
    assert_eq!(run_both(&source), vec![Value::Number(161.0)]);
}

#[test]
fn wide_multiple_assignment_to_indexed_targets_compiles() {
    let targets = (1..=80)
        .map(|i| format!("t[{i}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let values = (1..=80)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!("local t = {{}}\n{targets} = {values}\nreturn #t, t[80]");
    assert_eq!(
        run_both(&source),
        vec![Value::Number(80.0), Value::Number(80.0)]
    );
}

#[test]
fn multiple_assignment_reads_target_operands_before_the_values() {
    let source = r"
        local t, i = {}, 1
        t[i], i = 10, 2
        local u = {}
        local k = 'a'
        u[k], k = (function() k = 'b' return 1 end)(), 'c'
        return t[1], t[2], i, u.a, u.b
    ";
    assert_eq!(
        run_both(source),
        vec![
            Value::Number(10.0),
            Value::Nil,
            Value::Number(2.0),
            Value::Number(1.0),
            Value::Nil
        ]
    );
}

#[test]
fn deep_constructor_paren_and_block_nesting_is_allowed() {
    let table = format!("local t = {}\nreturn type(t)", nested(150, "{", "1", "}"));
    assert_eq!(run_both(&table), vec![Value::from("table")]);
    let parens = format!("return {}", nested(150, "(", "1", ")"));
    assert_eq!(run_both(&parens), vec![Value::Number(1.0)]);
    let blocks = format!(
        "local x = 0\n{}\nreturn x",
        nested(150, "if true then\n", "x = 1\n", "end\n")
    );
    assert_eq!(run_both(&blocks), vec![Value::Number(1.0)]);
}

#[test]
fn nesting_beyond_lua_limit_is_a_syntax_error() {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .build()
        .unwrap();
    let error = lua
        .exec(format!("return {}", nested(250, "(", "1", ")")))
        .unwrap_err()
        .to_string();
    assert!(error.contains("too many syntax levels"), "{error}");
}

#[test]
fn luajit_accepts_lua52_escapes_and_goto() {
    let source = r#"
        local total = 0
        for i = 1, 4 do
            if i == 2 then goto continue end
            total = total + i
            ::continue::
        end
        local goto = 5
        return "\x41\z
                \u{20AC}", total, goto
    "#;
    assert_eq!(
        run(CompatibilityProfile::LuaJit21, source),
        vec![
            Value::from("A\u{20AC}"),
            Value::Number(8.0),
            Value::Number(5.0)
        ]
    );
}

#[test]
fn stock_lua51_keeps_rejecting_lua52_escapes() {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua51)
        .build()
        .unwrap();
    assert!(lua.exec(r#"return "\x41""#).is_err());
    assert_eq!(
        lua.exec("local goto = 1 return goto").unwrap(),
        vec![Value::Number(1.0)]
    );
}

#[test]
fn long_left_associative_chains_compile_without_overflowing_the_stack() {
    // Lua does not limit these chains; each link is one more tree level.
    let calls = format!(
        "local function f() return f end\nlocal g = f{}\nreturn g == f",
        "()".repeat(20_000)
    );
    assert_eq!(run_both(&calls), vec![Value::Boolean(true)]);
}
