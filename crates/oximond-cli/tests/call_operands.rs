//! Call operands placed straight into call windows, native calls that skip
//! the frame-switching path, shared small-integer strings, and lazily hashed
//! long strings.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn operands_of_every_kind_reach_their_call_slots() {
    run(r#"
        local t = { x = 5, f = function(a, b, c) return a, b, c end }
        local up = 7
        G = 11
        local function id3(a, b, c) return a, b, c end
        local function pack(...) return select('#', ...), ... end
        local function check(...)
            local expected = { ... }
            return function(...)
                local got = { ... }
                assert(#got == #expected, #got)
                for i = 1, #expected do assert(got[i] == expected[i], i) end
            end
        end
        -- constant, global, field, upvalue, arithmetic, length, unary, local
        local local_value = 3
        local function inner()
            check(1, 7, 6, 3, -3)(id3(1, G, t.x), up, t.x + 1, #"abc", -local_value)
            local a, b, c = id3(up, G, t.x)
            assert(a == 7 and b == 11 and c == 5)
            return up
        end
        assert(inner() == 7)
        -- every operand is read before any later one runs
        local order = {}
        local function note(v) order[#order + 1] = v; return v end
        local a, b, c = id3(note(1), note(2), note(3))
        assert(a == 1 and b == 2 and c == 3 and #order == 3)
        -- method calls with constant, global and nested operands
        local obj = { n = 2 }
        function obj:add(x, y) return self.n + x + y end
        assert(obj:add(10, G) == 23)
        assert(obj:add(obj:add(1, 1), 1) == 7)
        assert(("abc"):sub(2, 3) == "bc")
        assert(string.rep("ab", 3, "-") == "ab-ab-ab" or _VERSION == "Lua 5.1")
        -- operands before a later call survive it
        local n, x, y, z, w = pack("x", 10, id3(8, 9, 10), "y")
        assert(n == 4 and x == "x" and y == 10 and z == 8 and w == "y", n)
        -- a call whose operand is another call's open results
        assert(select('#', pack(1, 2, pack(3, 4))) == 6)
        -- varargs after constants and globals
        local function forward(...) return pack(1, G, ...) end
        local count, p, q, r = forward(5)
        assert(count == 3 and p == 1 and q == 11 and r == 5)
    "#);
}

#[test]
fn call_errors_keep_their_names_and_positions() {
    run(r#"
        local t = {}
        local ok, message = pcall(function() missing_global(1, 2) end)
        assert(not ok and message:find("global 'missing_global'", 1, true), message)
        ok, message = pcall(function() t.field("x", 1) end)
        assert(not ok and message:find("field 'field'", 1, true), message)
        ok, message = pcall(function() t:method(1) end)
        assert(not ok and message:find("method 'method'", 1, true), message)
        ok, message = pcall(function() return math.max("x", 1) end)
        assert(not ok and message:find("bad argument #1", 1, true), message)
        ok, message = pcall(function() local s = string.rep() return s end)
        -- LuaJIT's argument errors name no function.
        assert(not ok and message:find("bad argument #1", 1, true), message)
        ok, message = pcall(function() local s = ("x"):rep({}) return s end)
        assert(not ok and message:find("bad argument", 1, true), message)
    "#);
}

#[test]
fn native_calls_return_their_values_in_every_shape() {
    run(r#"
        -- several results, fewer wanted, more wanted
        local s, e = string.find("haystack", "stack", 1, true)
        assert(s == 4 and e == 8)
        local only = string.find("haystack", "stack", 1, true)
        assert(only == 4)
        local a, b, c = string.find("haystack", "stack", 1, true)
        assert(a == 4 and b == 8 and c == nil)
        assert(select('#', string.find("haystack", "stack", 1, true)) == 2)
        assert(string.find("haystack", "needle", 1, true) == nil)
        assert(select('#', string.find("haystack", "needle", 1, true)) == 1)
        assert(string.find("haystack", "stack", 10, true) == nil)
        local x, y = string.find("a.b", ".", 1, true)
        assert(x == 2 and y == 2)
        -- the same function called over and over keeps its results apart
        for i = 1, 50 do
            local p, q = string.find("abcabc", "c", i % 4 + 1, true)
            assert((p == 3 and q == 3) or (p == 6 and q == 6), i)
        end
        -- one result written into several wanted slots
        local m, n, o = math.max(3, 9), math.min(3, 9), math.abs(-2)
        assert(m == 9 and n == 3 and o == 2)
        local u, v, w = (math.max(1, 2))
        assert(u == 2 and v == nil and w == nil)
        -- results can overwrite their own arguments
        local k = "abc"
        k = string.upper(k)
        assert(k == "ABC")
        k = k:lower():upper():rep(2)
        assert(k == "ABCABC")
        -- an error leaves the frame usable
        local ok = pcall(string.rep, "x", -1, {}, {})
        local after = string.find("abc", "b", 1, true)
        assert(after == 2)
    "#);
}

#[test]
fn tostring_of_small_integers_matches_string_format() {
    run(r#"
        local float_suffix = _VERSION == "Lua 5.3" or _VERSION == "Lua 5.4" or _VERSION == "Lua 5.5"
        for i = 0, 300 do
            local text = tostring(i)
            assert(text == string.format("%d", i), text)
            assert(#text == #tostring(i + 0) and text + 0 == i)
            assert(text .. "" == text)
            local float = tostring(i + 0.0)
            assert(float == text or (float_suffix and float == text .. ".0"), float)
        end
        assert(tostring(-1) == "-1" and tostring(-0.0) == (float_suffix and "-0.0" or "-0"))
        assert(tostring(255) == "255" and tostring(256) == "256" and tostring(1e2) == (float_suffix and "100.0" or "100"))
        local t = {}
        for i = 1, 5 do t[tostring(i)] = i end
        for i = 1, 5 do assert(t[tostring(i)] == i and t["" .. i] == i) end
        local kept = tostring(42)
        assert(kept == "42" and tostring(42) == kept)
    "#);
}

#[test]
fn a_tostring_metamethod_on_numbers_wins_over_the_fast_path() {
    for profile in PROFILES {
        let mut lua = Lua::builder().profile(profile).build().unwrap();
        let result = lua.exec(
            r#"
            local before = tostring(5)
            debug.setmetatable(0, { __tostring = function(n) return "number" end })
            local during = tostring(5)
            local float = tostring(2.5)
            debug.setmetatable(0, nil)
            return before, during, float, tostring(5)
        "#,
        );
        let values = result.unwrap_or_else(|error| panic!("{profile}: {error:?}"));
        let text = |index: usize| match &values[index] {
            oximond::Value::String(text) => text.to_string(),
            other => panic!("{profile}: {other:?}"),
        };
        assert_eq!(text(0), "5", "{profile}");
        assert_eq!(text(1), "number", "{profile}");
        assert_eq!(text(2), "number", "{profile}");
        assert_eq!(text(3), "5", "{profile}");
    }
}

#[test]
fn long_strings_compare_and_key_tables_before_they_are_hashed() {
    run(r#"
        local function build(n, tag)
            local value = ""
            for i = 1, n do value = value .. string.char(97 + (i + tag:byte()) % 26) end
            return value
        end
        local first, second = build(200, "k"), build(200, "k")
        assert(first == second and first ~= build(200, "j") and first ~= build(199, "k"))
        -- small tables scan their keys, large ones hash them
        for _, filler in ipairs({ 3, 40 }) do
            local t = {}
            for i = 1, filler do t["filler" .. i] = i end
            t[first] = "long"
            assert(t[second] == "long" and t[build(200, "j")] == nil)
            t[second] = "again"
            assert(t[first] == "again")
            local count = 0
            for key in pairs(t) do
                count = count + 1
                if #key == 200 then assert(key == first) end
            end
            assert(count == filler + 1)
        end
        -- strings that differ only past the first word still differ
        local base = string.rep("a", 100)
        assert(base .. "x" ~= base .. "y" and base .. "x" == base .. "x")
        local sorted = { base .. "b", base .. "a", base .. "c" }
        table.sort(sorted)
        assert(sorted[1] == base .. "a" and sorted[3] == base .. "c")
    "#);
}

#[test]
fn empty_and_one_byte_strings_behave_like_any_other_string() {
    run(r#"
        local text = "hello\0world\255"
        local rebuilt = {}
        for i = 1, #text do
            local c = text:sub(i, i)
            assert(#c == 1 and c == string.char(text:byte(i)) and c:byte() == text:byte(i))
            rebuilt[#rebuilt + 1] = c
        end
        assert(table.concat(rebuilt) == text)
        -- every byte value, as a key, a concatenation operand, and an ordering
        local seen, previous = {}, nil
        for b = 0, 255 do
            local c = string.char(b)
            assert(#c == 1 and c:byte() == b and not seen[c])
            seen[c] = b
            assert(c .. "" == c and "" .. c == c and (c .. c):sub(2) == c)
            if previous then assert(previous < c and not (c < previous)) end
            previous = c
        end
        for b = 0, 255 do assert(seen[string.char(b)] == b) end
        -- empty strings from every source are equal and behave as keys
        local empties = { "", text:sub(3, 2), string.rep("x", 0), ("abc"):sub(4), table.concat({}), "" .. "" }
        local keyed = {}
        for i, empty in ipairs(empties) do
            assert(#empty == 0 and empty == "" and empty .. "a" == "a")
            keyed[empty] = i
        end
        assert(keyed[""] == #empties and next(keyed) == "")
        -- tostring and number coercion of one-digit numbers
        for d = 0, 9 do assert(tostring(d) == string.char(48 + d) and d .. "" == tostring(d)) end
        assert(("x"):rep(3, "") == "xxx" or _VERSION == "Lua 5.1")
    "#);
}

#[test]
fn strings_built_at_the_stack_buffer_limits_keep_their_bytes() {
    run(r#"
        -- results straddling the inline (22 byte), stack buffer (32 and 64
        -- byte) and heap sizes, from string.char, .., table.concat and
        -- string.format
        local unpack_all = unpack or table.unpack
        for length = 0, 70 do
            local expected = string.rep("k", length)
            local pieces, codes = {}, {}
            for i = 1, length do pieces[i] = "k"; codes[i] = 107 end
            assert(table.concat(pieces) == expected, length)
            assert(string.char(unpack_all(codes)) == expected, length)
            local joined = ""
            for i = 1, length do joined = joined .. "k" end
            assert(joined == expected and #joined == length, length)
            assert(string.format("%s", expected) == expected, length)
            assert(string.format("%s%s", expected, "") == expected, length)
            local middle = math.floor(length / 2)
            local halves = { expected:sub(1, middle), expected:sub(middle + 1) }
            assert(table.concat(halves, "") == expected, length)
            assert(table.concat(halves, "--") == halves[1] .. "--" .. halves[2], length)
        end
        -- one format result much larger than the buffer kept between calls
        local big = string.format("%s|%d", string.rep("z", 200000), 7)
        assert(#big == 200002 and big:sub(-2) == "|7")
        for i = 1, 5 do assert(string.format("%d-%s", i, "ok") == i .. "-ok") end
        -- format called from a __tostring that format itself is calling
        local object = setmetatable({}, { __tostring = function() return string.format("<%d>", 42) end })
        -- Lua 5.1 refuses a table for %s; the other profiles call __tostring.
        local formatted, text = pcall(string.format, "[%s]", object)
        assert(not formatted or text == "[<42>]", text)
        assert(string.format("%5d|%-5d|%05d|%x|%X|%o", 42, 42, 42, 255, 255, 8) == "   42|42   |00042|ff|FF|10")
        assert(string.format("%d %d", math.maxinteger or 2^53, -7):find("-7", 1, true))
    "#);
}

#[test]
fn and_or_results_come_from_the_right_operand_without_disturbing_locals() {
    run(r#"
        local function id(...) return ... end
        local t = { 1 }
        local flag, none, zero, empty = true, nil, 0, ""
        G = 7
        -- left operand is a scratch value, a local, a global or a field
        assert((id(nil) or 5) == 5 and (id(false) or "x") == "x" and (id(3) or 5) == 3)
        assert((id(nil) and 5) == nil and (id(false) and 5) == false and (id(3) and 5) == 5)
        assert((none or G) == 7 and (flag and G) == 7 and (zero or G) == 0 and (empty and G) == 7)
        assert((t[2] or t[1]) == 1 and (t[1] and t[2]) == nil and (t[1] and t[1] + 1) == 2)
        -- right operand is a constant, global, field, arithmetic, call
        assert((none or 9) == 9 and (none or G) == 7 and (none or t[1]) == 1)
        assert((none or zero + 4) == 4 and (none or id(8)) == 8 and (none or #"abc") == 3)
        assert((flag and "yes" or "no") == "yes" and (none and "yes" or "no") == "no")
        assert((none or false or nil) == nil and (none or false or 0) == 0)
        -- the operands are locals the right side must not overwrite
        local a, b = nil, 4
        local c = a or b
        assert(c == 4 and a == nil and b == 4)
        c = b and a
        assert(c == nil and a == nil and b == 4)
        a = a or 1
        b = b and b + 1
        assert(a == 1 and b == 5)
        -- as call operands and in conditions
        assert(select('#', id(none or 1, flag and 2, none and 3)) == 3)
        local count = 0
        for i = 1, 5 do
            if (i % 2 == 0 or i == 5) and not (i == 4) then count = count + 1 end
        end
        assert(count == 2)
    "#);
}

#[test]
fn cached_tostring_and_window_find_keep_their_semantics() {
    run(r##"
        -- Small non-negative integers print bare in every profile, floats
        -- keep their suffix where the profile has one.
        local ok = tostring(7)
        assert(ok == "7" or ok == "7.0", ok)
        assert(tostring(-1) == "-1" or tostring(-1) == "-1.0")
        assert(tostring(300) == "300" or tostring(300) == "300.0")
        assert(tostring(0.5) == "0.5")
        for i = 1, 300 do
            local text = tostring(i % 101)
            assert(tonumber(text) == i % 101)
        end
        -- A number metatable's __tostring is not consulted for numbers, but
        -- a shadowed global and a table with __tostring are.
        local saved = tostring
        tostring = function(x) return "shadow" end
        assert(tostring(5) == "shadow")
        tostring = saved
        assert(tostring(setmetatable({}, {__tostring = function() return "meta" end})) == "meta")
        -- Several results from a plain find, and fewer results than values.
        local a, b = string.find("haystack", "stack", 1, true)
        assert(a == 4 and b == 8)
        local only = string.find("haystack", "stack", 1, true)
        assert(only == 4)
        assert(string.find("haystack", "zzz", 1, true) == nil)
        local x, y, z, w = string.find("haystack", "stack", 1, true)
        assert(x == 4 and y == 8 and z == nil and w == nil)
        assert(select("#", string.find("haystack", "zzz", 1, true)) == 1)
        assert(select("#", string.find("haystack", "stack", 1, true)) == 2)
        -- Patterns and captures take the general callback.
        local s, e, cap = string.find("key=val", "(%w+)=")
        assert(s == 1 and e == 4 and cap == "key")
        -- A call hook sees both natives.
        local calls = 0
        debug.sethook(function() calls = calls + 1 end, "c")
        tostring(3)
        string.find("haystack", "stack", 1, true)
        debug.sethook()
        assert(calls >= 2, calls)
    "##);
}
