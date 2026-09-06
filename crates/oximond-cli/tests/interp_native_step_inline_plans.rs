//! The interpreter's ipairs step (written without staging `Value`s in
//! memory) and the inline numeric call plans (scratch sized to the plan)
//! must keep their results and fallbacks.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

/// Numbers as `f64` (integers on profiles that have them), other values
/// unchanged.
fn run(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    run_raw(profile, source)
        .into_iter()
        .map(|value| match value {
            Value::Integer(integer) => Value::Number(integer as f64),
            other => other,
        })
        .collect()
}

fn run_raw(profile: CompatibilityProfile, source: &str) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    lua.exec(source).unwrap_or_else(|error| panic!("{error}"))
}

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

#[test]
fn ipairs_steps_cover_hits_holes_padding_and_metatables() {
    let source = r#"
        local t = {10, 20, 30, nil, 50}
        local sum, count = 0, 0
        for i, v in ipairs(t) do sum = sum + i * v; count = count + 1 end
        local proxied = setmetatable({}, {__index = function(_, k)
            if k <= 3 then return k * 100 end
        end})
        local seen = 0
        for i, v in ipairs(proxied) do seen = seen + v end
        local only_index = 0
        for i in ipairs({7, 8, 9}) do only_index = only_index + i end
        local padded
        for i, v, extra in ipairs({4}) do padded = extra end
        local strings = {"a", "b", "c"}
        local joined = ""
        for _, s in ipairs(strings) do joined = joined .. s end
        return sum, count, seen, only_index, padded == nil, joined
    "#;
    for profile in PROFILES {
        // Only 5.3 and later ipairs honor `__index`.
        let seen = if profile == CompatibilityProfile::Lua54 {
            600.0
        } else {
            0.0
        };
        assert_eq!(
            run(profile, source),
            vec![
                Value::Number(140.0),
                Value::Number(3.0),
                Value::Number(seen),
                Value::Number(6.0),
                Value::Boolean(true),
                Value::String("abc".into()),
            ],
            "{profile:?}"
        );
    }
}

#[test]
fn ipairs_after_the_array_part_still_finds_hash_entries() {
    let source = r"
        local t = {}
        t[1] = 1
        t[2] = 2
        t[4] = 4
        t[3] = 3
        local n = 0
        for _, v in ipairs(t) do n = n + v end
        return n
    ";
    assert_eq!(
        run(CompatibilityProfile::Lua54, source),
        vec![Value::Number(10.0)]
    );
}

#[test]
fn inline_plans_agree_with_plain_calls_across_register_tiers() {
    // Plans with few, some and many registers, and more than four arguments.
    let source = r"
        local function small(x) return x * 2 + 1 end
        local function mid(x)
            local a, b, c, d, e, f, g, h, i = x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7, x + 8
            return a + b * c - d / e + f * g - h + i
        end
        local function many(x)
            local r1 = x + 1
            local r2 = r1 * 2
            local r3 = r2 + r1
            local r4 = r3 * r2
            local r5 = r4 - r3
            local r6 = r5 + r4
            local r7 = r6 * 0.5
            local r8 = r7 + r6
            local r9 = r8 - r1
            local r10 = r9 + r2
            local r11 = r10 * r3
            local r12 = r11 - r4
            local r13 = r12 + r5
            local r14 = r13 * 0.25
            local r15 = r14 + r6
            local r16 = r15 - r7
            local r17 = r16 + r8
            return r17 + r9 + r10 + r11 + r12 + r13 + r14 + r15
        end
        local function five(a, b, c, d, e) return a + b * c - d + e end
        local function five_slow(a, b, c, d, e)
            local t = {a, b, c, d, e}
            return t[1] + t[2] * t[3] - t[4] + t[5]
        end
        local function many_slow(x)
            local t = {x}
            return t[1] and many(x)
        end
        local up = 0
        local function bump(x) up = up + x return up end
        local function branchy(x)
            if x % 2 == 0 then return x * 3 else return x - 1 end
        end
        local s1, s2, s3, s4, s5, up_total, branch_total = 0, 0, 0, 0, 0, 0, 0
        for i = 1, 300 do
            up_total = up_total + bump(i)
            branch_total = branch_total + branchy(i)
            s1 = s1 + small(i)
            s2 = s2 + mid(i)
            s3 = s3 + many(i)
            s4 = s4 + five(i, 2, 3, 4, 5)
            s5 = s5 + five_slow(i, 2, 3, 4, 5) + many_slow(i) * 0
        end
        return s1, s2, s3, s4, s5, up_total, branch_total
    ";
    let result = run(CompatibilityProfile::Lua51, source);
    assert_eq!(result[0], Value::Number(90_300.0 + 300.0));
    assert_eq!(result.len(), 7);
    assert_eq!(result[3], result[4]);
    assert_eq!(
        result[5],
        Value::Number((1..=300).map(|k| f64::from(k * (k + 1) / 2)).sum::<f64>())
    );
    // The same program with plans refused must give the same sums.
    let hooked = format!("debug.sethook(function() end, '', 1000000)\n{source}");
    assert_eq!(run(CompatibilityProfile::Lua51, &hooked), result);
}

#[test]
fn inline_plans_decline_non_number_arguments() {
    let source = r#"
        local function f(x) return x * 2 end
        local total = 0
        for i = 1, 100 do total = total + f(i) end
        local ok = pcall(f, {})
        local coerced = f("21")
        local mt = setmetatable({}, {__mul = function() return 5 end})
        return total, ok, coerced, f(mt)
    "#;
    assert_eq!(
        run(CompatibilityProfile::Lua54, source),
        vec![
            Value::Number(10_100.0),
            Value::Boolean(false),
            Value::Number(42.0),
            Value::Number(5.0),
        ]
    );
}
