//! Statement boundaries no longer clear dead temporaries, so a named local
//! declared without a value after one must still read nil through
//! `debug.getlocal`, as it does in stock Lua.

use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const PROGRAM: &str = r#"
local function dump()
  local out = {}
  for i = 1, 60 do
    local name, value = debug.getlocal(3, i)
    if not name then break end
    if name:sub(1, 1) ~= "(" then
      out[#out + 1] = name .. "=" .. (type(value) == "table" and "tbl" or tostring(value))
    end
  end
  return table.concat(out, " ")
end
local lines = {}
local function note(tag) lines[#lines + 1] = tag .. " " .. dump() end

local function outside()
  local t = {1}
  local u = (t[1] + 1) .. "x" .. #t
  local x
  note("outside")
  local w = {t[1] + 2, #u .. "y"}
  local y, z
  note("outside2")
end
outside()

local function in_loops()
  for i = 1, 2 do
    local t = {i}
    local q = #t .. "k" .. i
    local x
    local y
    note("for")
    x = i
    y = q
  end
  local n = 0
  while n < 2 do
    n = n + 1
    local a = n .. "q" .. n
    local b
    note("while")
    b = a
  end
  repeat
    local c = #lines .. "q"
    local d
    note("repeat")
  until true
  for k, v in pairs({10, 20}) do
    local w = k .. v .. "e"
    local e1
    note("pairs")
  end
end
in_loops()

local function after_calls()
  function many() return {}, {}, "a", "b", "c", "d" end
  many()
  local x
  local t = {many()}
  local y
  local s = #t .. tostring(#t)
  local z
  note("calls")
  do
    local p = tostring({}) .. "p"
  end
  do local aa, bb; note("block") end
end
after_calls()

return table.concat(lines, "\n")
"#;

const EXPECTED: &str = "\
outside t=tbl u=2x1 x=nil
outside2 t=tbl u=2x1 x=nil w=tbl y=nil z=nil
for i=1 t=tbl q=1k1 x=nil y=nil
for i=2 t=tbl q=1k2 x=nil y=nil
while n=1 a=1q1 b=nil
while n=2 a=2q2 b=nil
repeat n=2 c=6q d=nil
pairs n=2 k=1 v=10 w=110e e1=nil
pairs n=2 k=2 v=20 w=220e e1=nil
calls x=nil t=tbl y=nil s=66 z=nil
block x=nil t=tbl y=nil s=66 z=nil aa=nil bb=nil";

#[test]
fn uninitialized_locals_read_nil_after_dead_temporaries() {
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

/// Runs `program` in every profile and mode this file checks and returns
/// each run's label with its single string result.
fn run_each(program: &str) -> Vec<(String, String)> {
    let mut outputs = Vec::new();
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua51] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            let results = lua
                .exec(program)
                .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}"));
            let [Value::String(text)] = results.as_slice() else {
                panic!("{profile:?} {mode:?}: unexpected results {results:?}");
            };
            outputs.push((
                format!("{profile:?} {mode:?}"),
                String::from_utf8_lossy(text.as_bytes()).into_owned(),
            ));
        }
    }
    outputs
}

/// A debugger can write any `(temporary)` slot of a suspended coroutine,
/// including one the frame's pc no longer reads. The value it wrote must
/// stay alive and read back through a full collection, as in stock Lua 5.1
/// and 5.4. Each shape reports `written/alive/wrong`: the temporaries it
/// wrote, how many of those objects a weak table still holds, and how many
/// read back as something else.
#[test]
fn a_temporary_set_in_a_suspended_coroutine_survives_a_collection() {
    const SETLOCAL: &str = r#"
local y = coroutine.yield
local shapes = {
  function() local a = 1 local t = {a, y(), 3} return t end,
  function(c) local a = 1 local v = {a, (c and y() or 5), 3} return v end,
  function(c) local a = 1 local v = c and {a, y()} or {a} return v end,
  function() local a = 1 local t = {} t[a], t[a + 1] = y() return t end,
}
local out = {}
for shape, f in ipairs(shapes) do
  local co = coroutine.create(f)
  assert(coroutine.resume(co, true))
  local weak = setmetatable({}, {__mode = "k"})
  local written = {}
  local index = 1
  while true do
    local name = debug.getlocal(co, 1, index)
    if not name then break end
    if name:sub(1, 1) == "(" then
      local object = {tag = shape * 100 + index}
      assert(debug.setlocal(co, 1, index, object) == name)
      weak[object] = true
      written[#written + 1] = index
    end
    index = index + 1
  end
  collectgarbage()
  collectgarbage()
  local alive, wrong = 0, 0
  for _ in pairs(weak) do alive = alive + 1 end
  for _, index in ipairs(written) do
    local _, value = debug.getlocal(co, 1, index)
    if type(value) ~= "table" or value.tag ~= shape * 100 + index then
      wrong = wrong + 1
    end
  end
  assert(coroutine.resume(co, 1))
  out[#out + 1] = #written .. "/" .. alive .. "/" .. wrong
end
return table.concat(out, " ")
"#;
    for (label, output) in run_each(SETLOCAL) {
        // Stock Lua 5.1 and 5.4 print `2/2/0 2/2/0 2/2/0 1/1/0`. Oximond
        // shows one temporary fewer for the third shape; the count of
        // temporaries is not what this test checks.
        for shape in output.split(' ') {
            let counts = shape
                .split('/')
                .map(|count| count.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            let [written, alive, wrong] = counts[..] else {
                panic!("{label}: {output}");
            };
            assert!(written > 0, "{label}: {output}");
            assert_eq!(alive, written, "{label}: {output}");
            assert_eq!(wrong, 0, "{label}: {output}");
        }
    }
}
