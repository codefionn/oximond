//! Constructors run inside the fast loop. Collector pacing, error positions
//! after the constructor, and the scalar fragment's decline path must match
//! what the general loop produced.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

fn run(profile: CompatibilityProfile, source: &str) -> Result<Vec<Value>, String> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source).map_err(|error| error.to_string())
}

#[test]
fn churned_tables_stay_collectable() {
    let source = "
        local keep
        for i = 1, 200000 do
            local t = {}
            t.next = keep and {} or nil
            if i % 50000 == 0 then keep = t end
        end
        collectgarbage()
        return collectgarbage('count') < 20000
    ";
    for profile in PROFILES {
        assert_eq!(
            run(profile, source),
            Ok(vec![Value::Boolean(true)]),
            "{profile:?}"
        );
    }
}

#[test]
fn error_after_constructor_reports_its_line() {
    let source = "local t = {}\nlocal u = {}\nlocal x = t.a.b\nreturn x";
    for profile in PROFILES {
        let error = run(profile, source).unwrap_err();
        assert!(error.contains(":3:"), "{profile:?}: {error}");
    }
}

#[test]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn alternating_fragment_sites_still_evaluate() {
    // Three scalar-fragment sites exceed the two hot plan entries, so some
    // visits decline the in-place path.
    let source = "
        local total = 0
        for i = 1, 100 do
            local a = {x = i + 1}
            local b = {x = i * 2}
            local c = {x = i - 1}
            total = total + a.x + b.x + c.x
        end
        return total
    ";
    for profile in PROFILES {
        let result = run(profile, source).unwrap();
        assert!(
            matches!(result[..], [Value::Number(n)] if n == 20200.0)
                || matches!(result[..], [Value::Integer(20200)]),
            "{profile:?}: {result:?}"
        );
    }
}
