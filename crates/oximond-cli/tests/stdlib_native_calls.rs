use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

fn run(source: &str, profile: CompatibilityProfile, mode: ExecutionMode) -> Vec<Value> {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.exec(source).unwrap()
}

fn compare_modes(source: &str) {
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua55] {
        let expected = run(source, profile, ExecutionMode::Interpreter);
        let actual = run(source, profile, ExecutionMode::Tracing);
        assert_eq!(actual, expected, "{profile}");
    }
}

#[test]
fn hot_math_string_and_table_natives_match_the_interpreter() {
    let cases = [
        (
            "math.max",
            r"
            local total = 0
            for i = 1, 20000 do total = total + math.max(i, 10) end
            assert(total == 200010045)
            return total
        ",
        ),
        (
            "string.find",
            r#"
            local total = 0
            for i = 1, 20000 do
                local start = string.find("hello", "ell", 1, true)
                total = total + start
            end
            assert(total == 40000)
            return total
        "#,
        ),
        (
            "string.upper",
            r#"
            local total = 0
            for i = 1, 20000 do total = total + #string.upper("abc") end
            assert(total == 60000)
            return total
        "#,
        ),
        (
            "table.concat",
            r#"
            local parts = {"a", "b", "c"}
            local total = 0
            for i = 1, 20000 do total = total + #table.concat(parts, ":") end
            assert(total == 100000)
            return total
        "#,
        ),
        (
            "table.insert",
            r"
            local values = {}
            for i = 1, 20000 do table.insert(values, i) end
            assert(#values == 20000 and values[1] == 1 and values[20000] == 20000)
            return #values
        ",
        ),
        (
            "next",
            r"
            local values = {7}
            local total = 0
            for i = 1, 20000 do
                local key, value = next(values, nil)
                total = total + key + value
            end
            assert(total == 160000)
            return total
        ",
        ),
        (
            "ipairs",
            r"
            local values = {7}
            local total = 0
            for i = 1, 20000 do
                local iterator, state, index = ipairs(values)
                local key, value = iterator(state, index)
                total = total + key + value
            end
            assert(total == 160000)
            return total
        ",
        ),
    ];
    for (name, source) in cases {
        let expected = run(
            source,
            CompatibilityProfile::Lua51,
            ExecutionMode::Interpreter,
        );
        let actual = run(source, CompatibilityProfile::Lua51, ExecutionMode::Tracing);
        assert_eq!(actual, expected, "{name}");
        // With enough numeric work around the call the loop is worth
        // compiling.
        let source = with_numeric_work(source);
        let expected = run(
            &source,
            CompatibilityProfile::Lua51,
            ExecutionMode::Interpreter,
        );
        let actual = run(&source, CompatibilityProfile::Lua51, ExecutionMode::Tracing);
        assert_eq!(actual, expected, "{name}");
    }
}

/// Adds arithmetic to the hot loop so native code is worth entering.
fn with_numeric_work(source: &str) -> String {
    let loop_head = "for i = 1, 20000 do";
    assert!(source.contains(loop_head));
    source.replacen(
        loop_head,
        &format!(
            "{loop_head}\n local p = i * 3 + 1 local q = p * p - i local r = q % 7 + p * 2 \
             local s = r * 5 - q % 3 local t = s + r * p - q local u = t % 11 + s * 2 \
             local v = u * 3 - t % 5 local w = v % 9 + u * r local x = w - v * 2 + s \
             local y = x % 13 + w * 2 - p local z = y * 3 + x % 7 - q"
        ),
        1,
    )
}

#[test]
fn native_string_results_feed_lengths_like_the_interpreter() {
    let source = r#"
        local parts = {"a", "b", "c"}
        local total = 0
        for i = 1, 20000 do
            total = total + #table.concat(parts, ":")
        end
        return total
    "#;
    let expected = run(
        source,
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
    );
    let actual = run(source, CompatibilityProfile::Lua51, ExecutionMode::Tracing);
    assert_eq!(actual, expected);
    let source = with_numeric_work(source);
    let expected = run(
        &source,
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
    );
    let actual = run(&source, CompatibilityProfile::Lua51, ExecutionMode::Tracing);
    assert_eq!(actual, expected);

    compare_modes(
        r"
        local metatable = {__len = function() return 7 end}
        local table_value = {}
        local total = 0
        for i = 1, 1000 do
            local value = setmetatable(table_value, metatable)
            total = total + #value
        end
        return total
        ",
    );
    compare_modes(
        r#"
        local lookups = 0
        local parts = setmetatable({"a", nil, "c"}, {
            __index = function(_, key)
                lookups = lookups + 1
                return key == 2 and "b" or nil
            end
        })
        local total = 0
        for i = 1, 2000 do total = total + #table.concat(parts, ":") end
        return total, lookups
        "#,
    );
}

#[test]
fn native_calls_observe_rebinding_hooks_and_protected_errors() {
    compare_modes(
        r#"
        local original_max, original_find, original_concat = math.max, string.find, table.concat
        local parts = {"a", "b"}
        local function hot(n)
            local total = 0
            for i = 1, n do
                total = total + math.max(i, 2)
                total = total + (string.find("abc", "b", 1, true) or 0)
                total = total + #table.concat(parts, ".")
            end
            return total
        end
        local warm = hot(20000)
        assert(warm > 0)
        math.max = function(a, b) return a + b end
        string.find = function() return 9 end
        table.concat = function() return "long" end
        assert(hot(3) == 51)
        math.max, string.find, table.concat = original_max, original_find, original_concat
        local events = 0
        debug.sethook(function() events = events + 1 end, "c")
        local hooked = hot(4)
        debug.sethook()
        assert(hooked > 0 and events > 0)
        local ok, message = pcall(function() return math.max({}) end)
        assert(not ok and type(message) == "string")
        ok, message = pcall(function() return table.concat({"a", {}}, ",") end)
        assert(not ok and type(message) == "string")
        return warm, hooked, events
    "#,
    );
    compare_modes(
        r#"
        local minimum = math.min
        local total = 0
        for i = 1, 10000 do total = total + minimum(i, 2) end
        assert(minimum("8", 2) == 2)
        assert(math.max("8", 2) == 8)
        local ok, message = pcall(function() return minimum({}) end)
        assert(not ok and type(message) == "string")
        return total
        "#,
    );
}

#[test]
fn callback_metamethod_and_coroutine_yield_remain_observable() {
    compare_modes(
        r#"
        local calls = 0
        local item = setmetatable({}, { __tostring = function()
            calls = calls + 1
            return "item"
        end })
        local function invoke()
            return tostring(item)
        end
        for i = 1, 2000 do assert(invoke() == "item") end
        assert(calls == 2000)

        local sorted = {3, 1, 2}
        table.sort(sorted, function(a, b)
            calls = calls + 1
            return a < b
        end)
        assert(table.concat(sorted, ",") == "1,2,3")

        local thread = coroutine.create(function()
            local value = coroutine.yield("pause")
            return math.max(value, 5)
        end)
        local ok, first = coroutine.resume(thread)
        assert(ok and first == "pause")
        local resumed, second = coroutine.resume(thread, 7)
        assert(resumed and second == 7)
        return calls, first, second
    "#,
    );
}

#[test]
fn pairs_keeps_its_profiled_metamethod_dispatch() {
    let source = r#"
        local values = {10}
        for i = 1, 20000 do
            local iterator, state, key = pairs(values)
            assert(type(iterator) == "function" and state == values and key == nil)
        end
        local calls = 0
        setmetatable(values, { __pairs = function()
            calls = calls + 1
            local function iterator(_, key)
                if key == nil then return "special", 42 end
            end
            return iterator, values, nil
        end })
        local key, value
        for k, v in pairs(values) do key, value = k, v end
        assert(calls == 1 and key == "special" and value == 42)
        return key, value, calls
    "#;
    let expected = run(
        source,
        CompatibilityProfile::Lua55,
        ExecutionMode::Interpreter,
    );
    let actual = run(source, CompatibilityProfile::Lua55, ExecutionMode::Tracing);
    assert_eq!(actual, expected);
}
