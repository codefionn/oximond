//! Collections clear the registers they do not root.
//!
//! The collector roots a Lua frame only at the registers live at its pc and
//! sets the others to nil, as stock Lua clears the dead part of a stack.
//! These programs check that objects held only by dead temporaries are
//! collected, that `debug.getlocal` never shows a collected object
//! afterwards, and that everything a frame still needs survives: values a
//! suspended statement still reads, open results, varargs and long table
//! constructors. Stock Lua 5.1 and 5.4 print the expected text.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROGRAM: &str = r##"
local out = {}
local function note(s) out[#out + 1] = s end

-- An object held only by a dead temporary is collected, and every slot
-- debug.getlocal shows afterwards holds a usable value.
local weak = setmetatable({}, {__mode = "v"})
local function mark(t) weak[1] = t end
local function inspect(level)
  local shown = {}
  for i = 1, 40 do
    local name, value = debug.getlocal(level + 1, i)
    if not name then break end
    local kind = type(value)
    if kind == "table" then local _ = value[1] end
    shown[#shown + 1] = kind == "nil" and "nil" or tostring(value) and kind
  end
  return #shown
end
local function dead_temporary()
  local a = 1
  mark({})
  collectgarbage()
  collectgarbage()
  local count = inspect(2)
  return weak[1] == nil and count > 0
end
note("dead " .. tostring(dead_temporary()))

-- The same inside a suspended coroutine, checked from outside.
local co = coroutine.create(function()
  local b = 2
  mark({})
  coroutine.yield()
  return b
end)
assert(coroutine.resume(co))
collectgarbage()
collectgarbage()
local shown = 0
for i = 1, 40 do
  local name, value = debug.getlocal(co, 1, i)
  if not name then break end
  if type(value) == "table" then local _ = value[1] end
  shown = shown + 1
end
note("suspended " .. tostring(weak[1] == nil) .. " " .. tostring(select(2, coroutine.resume(co))))

-- Coroutines suspended mid-statement keep the temporaries they still need.
local function fresh(tag) return {tag = tag .. ("!"):rep(3)} end
local mids = {
  coroutine.wrap(function()
    local t = {fresh("a").tag, coroutine.yield(), fresh("c").tag}
    return table.concat(t, ",")
  end),
  coroutine.wrap(function()
    local s = fresh("x").tag .. coroutine.yield() .. fresh("z").tag
    return s
  end),
  coroutine.wrap(function()
    local t = fresh("m")
    local n = #t.tag + #coroutine.yield() + #t.tag
    return n
  end),
  coroutine.wrap(function(...)
    local r = select("#", ...) .. coroutine.yield() .. table.concat({...}, "+")
    return r
  end),
}
for i, f in ipairs(mids) do f("p", "q") end
for _ = 1, 3 do
  local garbage = {}
  for i = 1, 200 do garbage[i] = {("g"):rep(i)} end
  collectgarbage()
end
for i, f in ipairs(mids) do note("mid" .. i .. " " .. tostring(f("Y"))) end

-- Open results, varargs and table constructors across a collection.
local function gen(...)
  collectgarbage()
  return ...
end
local function count(...)
  collectgarbage()
  return select("#", ...), ...
end
local t = {fresh("o").tag, gen(fresh("p").tag, fresh("q").tag)}
note("ctor " .. table.concat(t, ","))
note("open " .. table.concat({count(fresh("r").tag, gen(fresh("s").tag, fresh("t").tag))}, ","))
local function va(...)
  local first = fresh("v").tag
  local packed = {first, ...}
  collectgarbage()
  return #packed, gen(...)
end
note("vararg " .. table.concat({va(fresh("w").tag, fresh("x").tag)}, ","))
local big = {gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, fresh("end").tag)}
note("setlist " .. #big .. " " .. big[53])
return table.concat(out, "\n")
"##;

const EXPECTED: &str = "\
dead true
suspended true 2
mid1 a!!!,Y,c!!!
mid2 x!!!Yz!!!
mid3 9
mid4 2Yp+q
ctor o!!!,p!!!,q!!!
open 3,r!!!,s!!!,t!!!
vararg 3,w!!!,x!!!
setlist 53 end!!!";

#[test]
fn collections_clear_dead_registers_and_keep_live_ones() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua51] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let results = lua
                .exec(PROGRAM)
                .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}"));
            let [Value::String(text)] = results.as_slice() else {
                panic!("{profile:?} {mode:?}: unexpected results {results:?}");
            };
            assert_eq!(
                String::from_utf8_lossy(text.as_bytes()),
                EXPECTED,
                "{profile:?} {mode:?}"
            );
        }
    }
}

/// A traced loop whose constructors charge the collector enough to start
/// collections. A native run with one due leaves at its next poll, and the
/// collection roots the frame its exit restored. The tables the loop reads
/// and the ones each round keeps must survive every collection.
const TRACED_ALLOCATION: &str = r"
local acc = {step = 0, name = 'acc'}
local kept = {}
local total = 0
for round = 1, 20 do
  kept[round] = {round, tag = 'r' .. round}
  for i = 1, 20000 do
    local a = {x = i + 1}
    local b = {x = i * 2}
    total = total + a.x + b.x + acc.step
  end
end
collectgarbage()
for round, t in ipairs(kept) do
  assert(t[1] == round and t.tag == 'r' .. round)
end
return total, acc.name, #kept
";

#[test]
// Exact comparison is required here (NaN semantics).
#[allow(clippy::float_cmp)]
fn collections_between_native_runs_keep_what_a_traced_loop_holds() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua51] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let results = lua
                .exec(TRACED_ALLOCATION)
                .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}"));
            let [total, Value::String(name), kept] = results.as_slice() else {
                panic!("{profile:?} {mode:?}: unexpected results {results:?}");
            };
            assert!(
                matches!(total, Value::Integer(12_001_000_000))
                    || matches!(total, Value::Number(n) if *n == 12_001_000_000.0),
                "{profile:?} {mode:?}: {total:?}"
            );
            assert_eq!(name.as_bytes(), b"acc", "{profile:?} {mode:?}");
            assert!(
                matches!(kept, Value::Integer(20))
                    || matches!(kept, Value::Number(n) if *n == 20.0),
                "{profile:?} {mode:?}: {kept:?}"
            );
        }
    }
}
