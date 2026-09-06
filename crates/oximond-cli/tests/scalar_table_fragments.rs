//! Non-escaping table constructors may run without allocating a table. The
//! results must match what the table would have held, whether the fragment
//! needs a few scratch numbers or enough to spill past the inline buffer.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::fmt::Write;

fn run(source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua51)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source).unwrap_or_else(|error| panic!("{error}"))
}

/// Builds a loop whose table has `fields` numeric fields, each computed from
/// the loop variable, and sums every field it reads back.
fn field_loop(fields: usize) -> (String, f64) {
    let mut constructor = String::new();
    let mut reads = String::new();
    for field in 0..fields {
        write!(constructor, "f{field} = i * {field} + {field}.5, ")
            .expect("formatting into a String cannot fail");
        write!(reads, " + t.f{field}").expect("formatting into a String cannot fail");
    }
    let source = format!(
        "local total = 0\nfor i = 1, 40 do local t = {{{constructor}}} total = total{reads} end\nreturn total"
    );
    let mut expected = 0.0;
    for i in 1..=40 {
        for field in 0..fields {
            expected += f64::from(i) * field as f64 + field as f64 + 0.5;
        }
    }
    (source, expected)
}

#[test]
fn short_and_long_numeric_fragments_match_table_results() {
    for fields in [1, 4, 9, 12, 20, 40] {
        let (source, expected) = field_loop(fields);
        assert_eq!(
            run(&source),
            vec![Value::Number(expected)],
            "{fields} fields"
        );
    }
}

#[test]
fn empty_constructor_fragments_leave_registers_usable() {
    let source = r"
        local count = 0
        for i = 1, 1000 do
            local t = {}
            if t then count = count + 1 end
        end
        return count
    ";
    assert_eq!(run(source), vec![Value::Number(1000.0)]);
}
