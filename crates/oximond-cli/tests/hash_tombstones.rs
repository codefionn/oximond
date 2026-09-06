//! A string key deleted from a hashed table stays behind as a tombstone (a
//! nil value) until a new key compacts the table. To Lua the key is gone:
//! reads fall to `__index`, stores go through `__newindex`, `pairs` skips it,
//! and `next` still accepts it as the key just cleared. Each program runs
//! under the interpreter and under tracing execution, and both must give the
//! expected values.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(mode: ExecutionMode, source: &str) -> String {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua51)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let outcome: Result<Vec<Value>, _> = lua.exec(source);
    match outcome {
        Ok(results) => format!("{results:?}"),
        Err(error) => format!("error: {error}"),
    }
}

fn check(source: &str, expected: &str) {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        assert_eq!(run(mode, source), expected, "{mode:?}\n{source}");
    }
}

#[test]
fn a_cleared_field_reads_through_index_and_stores_through_newindex() {
    check(
        r#"
        local stores = 0
        local t = setmetatable({}, {
          __index = function(_, k) return "missing " .. k end,
          __newindex = function(t, k, v) stores = stores + 1; rawset(t, k, v) end,
        })
        for i = 1, 200 do rawset(t, "f" .. i, i) end
        t.f5 = nil
        local read = t.f5
        t.f5 = 55
        local total = 0
        for i = 1, 2000 do
          t.f6 = nil
          total = total + (rawget(t, "f6") or 0)
          t.f6 = i
          total = total + t.f6
        end
        return read, t.f5, stores, total
        "#,
        r#"[String(b"missing f5"), Number(55.0), Number(2001.0), Number(2001000.0)]"#,
    );
}

#[test]
fn pairs_skips_cleared_keys_and_survives_clearing_while_it_walks() {
    check(
        r#"
        local t = {}
        for i = 1, 300 do t["k" .. i] = i end
        local visited, sum = 0, 0
        for k, v in pairs(t) do
          visited = visited + 1
          if v % 3 == 0 then t[k] = nil end
        end
        for _, v in pairs(t) do sum = sum + v end
        local cleared = next(t)
        t[cleared] = nil
        local ok = pcall(next, t, cleared)
        return visited, sum, ok, (pcall(next, t, "never"))
        "#,
        "[Number(300.0), Number(30000.0), Boolean(true), Boolean(false)]",
    );
}

#[test]
fn storing_nil_clears_present_keys_and_leaves_absent_ones_alone() {
    check(
        r#"
        local t = {}
        for i = 1, 100 do t["k" .. i] = i end
        t.k1 = nil
        t.k1 = nil
        t.absent = nil
        local n = 0
        for _ in pairs(t) do n = n + 1 end
        t.k1 = 7

        -- A table small enough to keep its fields in a shape.
        local s = {a = 1, b = 2}
        s.a = nil
        s.c = nil
        local sn = 0
        for _ in pairs(s) do sn = sn + 1 end

        -- A present key is cleared raw; an absent or cleared one asks
        -- __newindex.
        local stores = 0
        local p = setmetatable({}, {__newindex = function() stores = stores + 1 end})
        for i = 1, 40 do rawset(p, "f" .. i, i) end
        p.f1 = nil
        p.f1 = nil
        p.other = nil

        -- Keys cleared and set again, round after round.
        local c = {}
        for round = 1, 50 do
          for i = 1, 40 do c["r" .. i] = round end
          for i = 1, 40, 2 do c["r" .. i] = nil end
        end
        local live, sum = 0, 0
        for _, v in pairs(c) do live = live + 1; sum = sum + v end

        -- Values that own memory are released the same way.
        local owners = {}
        for i = 1, 60 do owners["o" .. i] = {i} end
        for i = 1, 60 do owners["o" .. i] = nil end
        collectgarbage()
        return n, t.k1, sn, stores, rawget(p, "f1"), live, sum, next(owners)
        "#,
        "[Number(99.0), Number(7.0), Number(1.0), Number(2.0), Nil, Number(20.0), Number(1000.0), Nil]",
    );
}
