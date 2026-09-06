//! Array stores of strings, tables and functions take a direct slot path in
//! the dispatch loop. The collector must still see what they store: a young
//! object put into an old table survives collections, a weak table still
//! clears, and tables that need the general path keep their semantics.

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

/// Ages `old` with full collections, then stores young values into it while
/// garbage is allocated and collected in between.
fn old_table_program(mode: &str) -> String {
    format!(
        r#"
        {mode}
        local function churn()
          for i = 1, 3000 do local _ = {{i, tostring(i)}} end
        end
        local old = {{}}
        for i = 1, 50 do old[i] = "seed" .. i end
        collectgarbage(); collectgarbage()
        -- append young tables and strings after the table is old
        for i = 51, 250 do
          if i % 2 == 0 then old[i] = {{id = i}} else old[i] = "s" .. i end
          if i % 20 == 0 then churn(); collectgarbage("step", 50) end
        end
        churn(); collectgarbage()
        -- overwrite old slots with young closures
        for i = 1, 250, 5 do
          old[i] = function() return i end
          if i % 25 == 1 then churn(); collectgarbage("step", 50) end
        end
        churn(); collectgarbage(); collectgarbage()
        local bad = 0
        for i = 1, 250 do
          local v = old[i]
          if i % 5 == 1 then
            if type(v) ~= "function" or v() ~= i then bad = bad + 1 end
          elseif i <= 50 then
            if v ~= "seed" .. i then bad = bad + 1 end
          elseif i % 2 == 0 then
            if type(v) ~= "table" or v.id ~= i then bad = bad + 1 end
          else
            if v ~= "s" .. i then bad = bad + 1 end
          end
        end
        return bad, #old
        "#
    )
}

#[test]
fn young_values_stored_into_old_tables_survive_collections() {
    let expected = vec![Value::Integer(0), Value::Integer(250)];
    let cases = [
        (CompatibilityProfile::Lua51, ""),
        (CompatibilityProfile::Lua54, "collectgarbage('incremental')"),
        (
            CompatibilityProfile::Lua54,
            "collectgarbage('generational')",
        ),
        (
            CompatibilityProfile::Lua55,
            "collectgarbage('generational')",
        ),
    ];
    for (profile, mode) in cases {
        let values = run(profile, &old_table_program(mode));
        let numbers: Vec<_> = values
            .iter()
            .map(|value| match value {
                Value::Integer(n) => Value::Integer(*n),
                Value::Number(n) => Value::Integer(*n as i64),
                other => other.clone(),
            })
            .collect();
        assert_eq!(numbers, expected, "{profile} {mode}");
    }
}

#[test]
fn weak_tables_still_clear_values_stored_by_the_direct_path() {
    let source = r#"
        local weak = setmetatable({}, {__mode = "v"})
        local keep = {}
        -- filled in a callee, so no stale register of this chunk holds one
        local function fill() for i = 1, 100 do weak[i] = {} end end
        fill()
        weak[101] = keep
        local kept = weak[101]
        collectgarbage(); collectgarbage()
        local live = 0
        for _, v in pairs(weak) do live = live + 1 end
        return live, weak[101] == keep, kept == keep
    "#;
    for (profile, mode) in [
        (CompatibilityProfile::Lua51, ""),
        (CompatibilityProfile::Lua54, ""),
        (
            CompatibilityProfile::Lua54,
            "collectgarbage('generational')",
        ),
    ] {
        let values = run(profile, &format!("{mode}\n{source}"));
        assert_eq!(values.len(), 3, "{profile} {mode}");
        assert!(
            matches!(values[0], Value::Integer(1) | Value::Number(1.0)),
            "{profile} {mode}: {values:?}"
        );
        assert_eq!(values[1], Value::Boolean(true), "{profile} {mode}");
        assert_eq!(values[2], Value::Boolean(true), "{profile} {mode}");
    }
}

#[test]
fn stores_that_need_the_general_path_keep_their_semantics() {
    let source = r#"
        local log = {}
        -- a metatable with __newindex sees an append of a new key
        local watched = setmetatable({}, {__newindex = function(t, k, v)
          log[#log + 1] = k .. "=" .. tostring(v)
          rawset(t, k, v)
        end})
        watched[1] = "a"
        watched[2] = "b"
        watched[2] = "c" -- present key: raw overwrite, no call
        -- a gap is a hash store and does not extend the border
        local gap = {}
        gap[1] = "x"
        gap[3] = "z"
        -- hash entries beside the array part
        local mixed = {name = "n"}
        mixed[1] = {}
        mixed[2] = "two"
        -- overwrite releases the old string
        local over = {}
        for i = 1, 10 do over[i] = "old" .. i end
        for i = 1, 10 do over[i] = "new" .. i end
        -- storing nil, then a string, at the border
        local border = {"a", "b", "c"}
        border[3] = nil
        border[3] = "again"
        -- functions of every kind
        local fns = {}
        fns[1] = print
        fns[2] = tostring
        fns[3] = function() return 3 end
        return table.concat(log, ","), #gap, mixed.name .. mixed[2],
          over[1] .. over[10], #border, fns[1] == print, fns[2] == tostring,
          fns[3]()
    "#;
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let values = run(profile, source);
        assert_eq!(values.len(), 8, "{profile}");
        assert_eq!(values[0], Value::from("1=a,2=b"), "{profile}");
        assert_eq!(values[2], Value::from("ntwo"), "{profile}");
        assert_eq!(values[3], Value::from("new1new10"), "{profile}");
        assert_eq!(values[5], Value::Boolean(true), "{profile}");
        assert_eq!(values[6], Value::Boolean(true), "{profile}");
    }
}
