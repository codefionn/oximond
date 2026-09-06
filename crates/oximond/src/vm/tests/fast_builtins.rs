//! `pairs`, `ipairs` and `next` run inside the fast loop's register window,
//! and equality and string ordering decided before the numeric ladder. Each
//! script runs plainly and again under a call hook, which keeps the in-window
//! calls out (`Vm::call_native_in_window` declines when hooks observe calls);
//! the two must agree.

use super::*;

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

/// Runs `body` (a function body that receives the mode and returns a value)
/// in the interpreter, without a hook (`"none"`) or under a call hook
/// (`"c"`).
fn run(profile: CompatibilityProfile, body: &str, mode: &str) -> Vec<Value> {
    let mut lua = crate::Lua::builder()
        .profile(profile)
        .execution_mode(ExecutionMode::Interpreter)
        .build()
        .unwrap();
    let chunk = format!(
        "return (function(mode)
           if mode == 'c' then debug.sethook(function() end, 'c') end
           local result = {{(function() {body} end)()}}
           debug.sethook()
           return (table.unpack or unpack)(result)
         end)('{mode}')"
    );
    lua.exec(&chunk)
        .unwrap_or_else(|error| panic!("{profile:?} {mode}: {error}"))
}

fn text(values: &[Value]) -> String {
    match values {
        [Value::String(text)] => text.to_lossy().into_owned(),
        other => panic!("{other:?}"),
    }
}

const TRAVERSALS: &str = r"
local out = {}
local function add(...)
  for i = 1, select('#', ...) do out[#out + 1] = tostring((select(i, ...))) end
end
local function keys(t)
  local list = {}
  for k, v in pairs(t) do list[#list + 1] = tostring(k) .. '=' .. tostring(v) end
  table.sort(list)
  return table.concat(list, ',')
end
-- A constructor's shaped table, a small ordered one, a hashed one, and a
-- table with an array part and other keys.
add(keys({a = 1, b = 2, c = 3}))
local small = {} for i = 1, 12 do small['k' .. i] = i end
add(keys(small))
local big = {} for i = 1, 40 do big['k' .. i] = i end
add(keys(big))
local mixed = {10, 20, 30, x = 'y', [2.5] = true}
add(keys(mixed))
-- `next` called directly, at the start, in the middle and past the end.
add((next({})))
add(next({7}))
add((next({a = 1}, 'a')))
local first = next(big)
add(type(first), select('#', next(big, first)))
-- The end of a table is one nil, as in every reference Lua.
add('ends', select('#', next({})), select('#', next({a = 1}, 'a')))
local last = 0
for _, t in ipairs({{}, {a = 1}}) do
  local k = next(t)
  if k ~= nil then last = select('#', next(t, k)) end
end
add('ends', last)
-- Clearing every field while traversing.
local cleared = {} for i = 1, 30 do cleared['d' .. i] = i end
local count = 0
for k in pairs(cleared) do cleared[k] = nil count = count + 1 end
add(count, (next(cleared)))
local shaped = {a = 1, b = 2, c = 3, d = 4}
count = 0
for k in pairs(shaped) do shaped[k] = nil count = count + 1 end
add(count, (next(shaped)))
-- `ipairs` stops at the first nil.
local sum = 0
for _, v in ipairs({1, 2, 3, nil, 5}) do sum = sum + v end
add(sum)
-- The triples `pairs` and `ipairs` return.
local f, t, c = pairs(mixed)
add(f == next, t == mixed, c)
local g, u, d = ipairs(mixed)
add(u == mixed, d, g(mixed, 0))
-- `__pairs`: honoured from 5.2 on, ignored by 5.1 and LuaJIT.
local proxy = setmetatable({real = true}, {__pairs = function(self)
  return function(_, k) if k == nil then return 1, 'one' end end, self, nil
end})
add(keys(proxy))
-- An invalid key raises the same error either way.
local ok, message = pcall(function() return next({}, 'missing') end)
add(ok, message)
local ok2, message2 = pcall(function() return pairs(nil) end)
add(ok2, message2)
return table.concat(out, ' ')
";

#[test]
fn iteration_builtins_in_the_window_match_the_general_path() {
    for profile in PROFILES {
        let plain = text(&run(profile, TRAVERSALS, "none"));
        let hooked = text(&run(profile, TRAVERSALS, "c"));
        assert_eq!(plain, hooked, "{profile:?}");
        let expected_pairs = if matches!(
            profile,
            CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
        ) {
            "real=true"
        } else {
            "1=one"
        };
        assert!(plain.contains(expected_pairs), "{profile:?}: {plain}");
        assert!(plain.contains("30 nil 4 nil 6"), "{profile:?}: {plain}");
        assert!(
            plain.contains(" nil 1 7 nil string 2 "),
            "{profile:?}: {plain}"
        );
        assert!(plain.contains("a=1,b=2,c=3 "), "{profile:?}: {plain}");
        assert!(plain.contains(" ends 1 1 ends 1 "), "{profile:?}: {plain}");
        assert!(plain.contains("false"), "{profile:?}: {plain}");
    }
}

const COMPARISONS: &str = r"
local mt = {__eq = function() return true end}
local e1, e2 = setmetatable({}, mt), setmetatable({}, mt)
local p1, p2 = {}, {}
local m1, m2 = setmetatable({}, {}), setmetatable({}, {})
local checks = {
  e1 == e2, not (e1 ~= e2), p1 ~= p2, p1 == p1, m1 ~= m2, m1 ~= p1,
  nil == nil, nil ~= false, false == false, true ~= false,
  'ab' == 'a' .. 'b', 'ab' ~= 'abc', ('x'):rep(30) == ('x'):rep(30),
  'ab' ~= nil, p1 ~= 'p1', e1 ~= nil,
  'a' < 'b', 'a' < 'a\0', not ('a\0' < 'a'), 'ab' < 'abc', 'abc' <= 'abc',
  ('z'):rep(22) < ('z'):rep(23), 'a\0b' < 'a\0c', '' < '\0',
  ('y'):rep(21) .. 'a' < ('y'):rep(21) .. 'b', not ('b' <= 'a'),
  'a\255' > 'a\1', ('w'):rep(40) .. '1' < ('w'):rep(40) .. '2',
}
for i, check in ipairs(checks) do
  if not check then return 'check ' .. i .. ' failed' end
end
if #checks ~= 28 then return 'missing checks' end
local words = {'b', 'a\0', 'a', '', 'a\0\0', 'ab', ('q'):rep(25), ('q'):rep(22), 'a\1'}
table.sort(words)
local spelled = {}
for i, word in ipairs(words) do spelled[i] = #word .. ':' .. (word:gsub('%c', string.byte)) end
return table.concat(spelled, ' ')
";

#[test]
fn equality_and_string_order_match_lua() {
    for profile in PROFILES {
        let plain = text(&run(profile, COMPARISONS, "none"));
        let expected = format!(
            "0: 1:a 2:a0 3:a00 2:a1 2:ab 1:b 22:{} 25:{}",
            "q".repeat(22),
            "q".repeat(25)
        );
        assert_eq!(plain, expected, "{profile:?}");
        assert_eq!(plain, text(&run(profile, COMPARISONS, "c")), "{profile:?}");
    }
}

#[test]
fn integer_cdata_equality_keeps_the_numeric_ladder() {
    let body = r"
local a, b = 1ull, 1ull
local checks = {a == b, 1ull == 1, 2ll ~= 3ll, 0x10ull == 16ull, not (a ~= b)}
local out = {}
for i, check in ipairs(checks) do out[i] = tostring(check) end
return table.concat(out, ' ')
";
    let plain = text(&run(CompatibilityProfile::LuaJit21, body, "none"));
    assert_eq!(plain, "true true true true true");
    assert_eq!(plain, text(&run(CompatibilityProfile::LuaJit21, body, "c")));
}

/// The `bit` operations on 64-bit integer cdata run their pure forms in the
/// window (`NativeWideBinary`), alone and as the inner call of
/// `tonumber (band (...))`; a call hook keeps them on the callbacks. Both
/// must agree, on every operand mix and count, with the results `LuaJIT`
/// prints.
#[test]
fn wide_bit_operations_match_the_callbacks() {
    let body = r"
local out = {}
local function add(v) out[#out + 1] = tostring(v) end
local values = {0x8000000000000001ull, -5ll, 3ll, 0xffffffffull, 7, -1.5, 2^40}
local counts = {0, 1, 31, 32, 63, 64, 65, -1, 4ll, 33ull}
local ops = {bit.band, bit.bor, bit.bxor}
for _, op in ipairs(ops) do
  for _, x in ipairs(values) do
    for _, y in ipairs(values) do add(op(x, y)) end
  end
end
local shifts = {bit.lshift, bit.rshift, bit.arshift, bit.rol, bit.ror}
for _, op in ipairs(shifts) do
  for _, x in ipairs(values) do
    for _, n in ipairs(counts) do add(op(x, n)) end
  end
end
local band, rshift, tonumber = bit.band, bit.rshift, tonumber
local v = 0x123456789abcdefull
add(tonumber(band(v, 0xffffffffull)))
add(tonumber(rshift(v, 32)))
add(tonumber(band(-2ll, 0xffffull)))
add(select('#', band(v, 1ull)))
add(pcall(bit.band, 1ull, {}))
add(pcall(bit.lshift, 1ull, 'x'))
add(bit.lshift(1ull, '3'))
return table.concat(out, ' ')
";
    let plain = text(&run(CompatibilityProfile::LuaJit21, body, "none"));
    assert_eq!(plain, text(&run(CompatibilityProfile::LuaJit21, body, "c")));
    // Spot checks against LuaJIT 2.1: band of the first value with itself,
    // of -5ll with itself, and the tail of fused and erroring calls.
    assert!(plain.starts_with("9223372036854775809ULL "), "{plain}");
    assert!(plain.contains(" -5LL "), "{plain}");
    assert!(
        plain.ends_with(" 2309737967 19088743 65534 1 false false 8ULL"),
        "{plain}"
    );
}

/// `LuaJIT` truncates a float string index into the 32-bit range and takes
/// anything outside it, NaN included, as the minimum integer.
#[test]
fn luajit_string_indices_truncate_to_32_bits() {
    let body = r#"
local s, out = "hello", {}
local indices = {1.9, 2.9, -1.5, 2147483647.9, -2147483648.5, -2147483649.5, 2^31, 0/0, 2^53}
for _, x in ipairs(indices) do out[#out + 1] = "[" .. s:sub(x) .. "]" end
return table.concat(out, " ")
"#;
    assert_eq!(
        text(&run(CompatibilityProfile::LuaJit21, body, "none")),
        "[hello] [ello] [o] [] [hello] [hello] [hello] [hello] [hello]"
    );
}

/// `s:method (...)` reads the string library raw in the fast loop
/// (`Vm::string_self_field`); a replaced `__index` and a missing method keep
/// the general path's answers.
#[test]
fn string_methods_resolve_through_the_string_metatable() {
    let body = r"
local s, out = 'Hello', {}
for i = 1, 3 do out[#out + 1] = s:sub(i, i) .. s:upper() .. s:len() end
local mt = getmetatable('')
local index = mt.__index
mt.__index = function(_, key) return function() return key end end
out[#out + 1] = s:anything()
mt.__index = index
out[#out + 1] = s:rep(2)
local ok = pcall(function() return s:nosuch() end)
out[#out + 1] = tostring(ok)
return table.concat(out, ' ')
";
    for profile in PROFILES {
        let plain = text(&run(profile, body, "none"));
        assert_eq!(
            plain, "HHELLO5 eHELLO5 lHELLO5 anything HelloHello false",
            "{profile:?}"
        );
        assert_eq!(plain, text(&run(profile, body, "c")), "{profile:?}");
    }
}

/// A trace that records the end of a `next` traversal passes on one nil,
/// as the interpreter does, to a call that takes its open results.
#[test]
fn traced_next_ends_with_one_value() {
    let source = "
local t, total, last = {[10] = 1}, 0, 10
for i = 1, 3000 do total = total + select('#', next(t, last)) end
return tostring(total == 3000)
";
    let chunk = crate::compiler::compile(source).unwrap();
    for profile in PROFILES {
        let mut vm = super::trace_native::native_vm(profile);
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let result = vm.execute(&chunk).unwrap();
        assert_eq!(text(&result), "true", "{profile:?}");
        let stats = super::trace_native::stats_of(&vm);
        assert!(stats.traces_compiled > 0, "{profile:?}: {stats:?}");
    }
}

/// Tracing mode runs Lua tail calls and `return native(...)` inside the fast
/// loop, noting each landing for the tracer (`Vm::tail_landing_leaves`). A
/// self tail recursion still becomes a trace, a mutual one and a chain that
/// passes through give the interpreter's results, and so does a tail call
/// of a native.
#[test]
fn traced_tail_calls_match_the_interpreter() {
    let source = "
local function count(n, acc) if n == 0 then return acc end return count(n - 1, acc + n) end
local odd
local function even(n) if n == 0 then return 1 end return odd(n - 1) end
function odd(n) if n == 0 then return 0 end return even(n - 1) end
local function inner(x) return x * 2 end
local function outer(x) return inner(x + 1) end
local abs = math.abs
local function magnitude(x) return abs(x) end
local a, b, c, d = 0, 0, 0, 0
for i = 1, 300 do
  a = a + count(40, 0)
  b = b + even(i % 7)
  c = c + outer(i)
  d = d + magnitude(-i)
end
return tostring(a == 246000 and b == 171 and c == 90900 and d == 45150)
";
    let chunk = crate::compiler::compile(source).unwrap();
    for profile in PROFILES {
        let mut interpreter =
            Vm::with_profile_and_execution_mode(profile, ExecutionMode::Interpreter);
        install_base_with_profile(interpreter.runtime_mut(), profile).unwrap();
        let expected = interpreter.execute(&chunk).unwrap();
        let mut vm = super::trace_native::native_vm(profile);
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let result = vm.execute(&chunk).unwrap();
        assert_eq!(text(&result), text(&expected), "{profile:?}");
        assert_eq!(text(&result), "true", "{profile:?}");
        let stats = super::trace_native::stats_of(&vm);
        assert!(stats.traces_compiled > 0, "{profile:?}: {stats:?}");
    }
}

/// Run-once registration shapes the fast loop takes shortcuts on: `type()`
/// answered without its callback (`Vm::call_native_in_window`), string
/// arguments pushed over registers that hold strings by the counted hot push
/// (`CallStack::push_plain_call_cold`), and `next` steps over small tables
/// that start at the position the last step returned (`Vm::next_hint`, one
/// for every table, so interleaved traversals keep moving it elsewhere).
const REGISTRATION: &str = r"
local out = {}
local function add(...)
  for i = 1, select('#', ...) do out[#out + 1] = tostring((select(i, ...))) end
end
local function keys(t)
  local list = {}
  for k, v in pairs(t) do list[#list + 1] = tostring(k) .. '=' .. tostring(v) end
  table.sort(list)
  return table.concat(list, ',')
end
-- Interleaved traversals of tables that share their key strings, in
-- different orders: a small ordered one, a constructor's shaped one, and a
-- hashed one.
local ordered = {} for _, k in ipairs({'c', 'a', 'd', 'b', 'e'}) do ordered[k] = k end
local shaped = {e = 1, d = 2, c = 3, b = 4, a = 5}
local hashed = {} for i = 1, 40 do hashed['k' .. i] = i end
hashed.a, hashed.b = 'x', 'y'
local pairs_seen = 0
for k1 in pairs(ordered) do
  for k2 in pairs(shaped) do
    for k3 in pairs(hashed) do pairs_seen = pairs_seen + 1 end
    pairs_seen = pairs_seen + 1
  end
end
add('nest', pairs_seen)
-- Overwrites and deletes of the current key while traversing.
local copy = {}
for k, v in pairs(shaped) do shaped[k] = v * 10 copy[k] = v end
add(keys(shaped), keys(copy))
local removed = 0
for k in pairs(ordered) do ordered[k] = nil removed = removed + 1 end
add('removed', removed, next(ordered))
-- `next` by hand, from a key another traversal last returned.
local t1, t2 = {x = 1, y = 2, z = 3}, {z = 1, y = 2, x = 3}
local k1 = next(t1)
local k2 = next(t2, k1)
local count = 0
for _ in pairs(t2) do count = count + 1 end
add('hand', type(k2), count)
-- `type`: one result, none asked, several asked, open results, a missing
-- argument, extra arguments, and names that are not the builtin.
local values = {nil, true, 1, 2.5, 'text', {}, print, function() end, coroutine.create(function() end)}
local names = {}
for i = 1, 9 do names[#names + 1] = type(values[i]) end
add(table.concat(names, ','))
type(1)
local r1, r2 = type('x')
add(r1, r2, select('#', type(1)), select('#', type(1, 2, 3)))
add(pcall(type))
do local type = function() return 'mine' end add(type(1)) end
local saved = type
type = function(v) return 'global ' .. saved(v) end
add(type(1))
type = saved
-- Calls with string arguments into registers that held strings, with
-- fewer and more arguments than parameters.
local function join(a, b, c) return tostring(a) .. tostring(b) .. tostring(c) end
local acc = {}
for i = 1, 4 do
  local s = 'p' .. i
  acc[#acc + 1] = join(s, s .. 'q')
  acc[#acc + 1] = join(s, 'r', s, 'extra')
  acc[#acc + 1] = join()
end
add(table.concat(acc, ' '))
return table.concat(out, ' ')
";

const REGISTRATION_EXPECTED: &str = "nest 1075 \
a=50,b=40,c=30,d=20,e=10 a=5,b=4,c=3,d=2,e=1 removed 5 nil hand string 3 \
nil,boolean,number,number,string,table,function,function,thread \
string nil 1 1 false";

#[test]
fn registration_shapes_match_the_general_path() {
    for profile in PROFILES {
        let plain = text(&run(profile, REGISTRATION, "none"));
        let hooked = text(&run(profile, REGISTRATION, "c"));
        assert_eq!(plain, hooked, "{profile:?}");
        assert!(
            plain.starts_with(REGISTRATION_EXPECTED),
            "{profile:?}: {plain}"
        );
        assert!(
            plain.contains(" mine global number "),
            "{profile:?}: {plain}"
        );
        assert!(
            plain.ends_with("p1p1qnil p1rp1 nilnilnil p2p2qnil p2rp2 nilnilnil p3p3qnil p3rp3 nilnilnil p4p4qnil p4rp4 nilnilnil"),
            "{profile:?}: {plain}"
        );
        // The tracing loop takes the same shortcuts.
        let mut lua = crate::Lua::builder()
            .profile(profile)
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        let traced = text(
            &lua.exec(format!("return (function() {REGISTRATION} end)()"))
                .unwrap(),
        );
        assert_eq!(traced, plain, "{profile:?}");
    }
}
