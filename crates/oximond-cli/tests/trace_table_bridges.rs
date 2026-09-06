//! A guard whose exit builds virtual tables can still enter a side trace: the
//! run builds the tables on the way and the side takes each as the table its
//! register holds. Also the two smaller rules of the same round: `tonumber`
//! of a number is the number, and a constructor that stays virtual does not
//! stop the loop from hoisting a check or a read.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match. The accounting tests check
//! that each constructor ran once: real tables plus net phantom charges are
//! the tables the interpreter allocated.

use oximond::runtime::Runtime;
use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

struct Outcome {
    result: String,
    stats: TraceStatistics,
    created: u64,
    elided: (u64, u64),
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, source: &str) -> Outcome {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    let runtime: &Runtime = lua.vm().runtime();
    Outcome {
        result: format!("{results:?}"),
        stats: lua.vm().jit_statistics().trace,
        created: runtime.tables_created(),
        elided: runtime.elided_tables_total(),
    }
}

fn agree(source: &str) -> Vec<(CompatibilityProfile, Outcome, Outcome)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let interpreter = run(profile, ExecutionMode::Interpreter, source);
            let tracing = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(
                tracing.result, interpreter.result,
                "{profile:?}\n{source}\n{:?}",
                tracing.stats
            );
            (profile, interpreter, tracing)
        })
        .collect()
}

fn assert_each_constructor_ran_once(
    profile: CompatibilityProfile,
    interpreter: &Outcome,
    tracing: &Outcome,
) {
    assert_eq!(
        tracing.created + tracing.elided.0,
        interpreter.created + interpreter.elided.0,
        "{profile:?}: real {} + phantom {} against the interpreter's {} + {}\n{:?}",
        tracing.created,
        tracing.elided.0,
        interpreter.created,
        interpreter.elided.0,
        tracing.stats
    );
}

/// Lua source that prints a table's contents in a fixed order, and one that
/// prints them in `pairs` order.
const DUMP: &str = "
local function dump(t)
  local keys = {}
  for k in pairs(t) do keys[#keys + 1] = k end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  local out = {}
  for _, k in ipairs(keys) do out[#out + 1] = tostring(k) .. '=' .. tostring(t[k]) end
  return table.concat(out, ',')
end
local function order(t)
  local out = {}
  for k, v in pairs(t) do out[#out + 1] = tostring(k) .. '=' .. tostring(v) end
  return table.concat(out, ',')
end
";

// ---- a side entered from an exit that builds tables ----

#[test]
fn rare_branch_keeps_the_table() {
    let source = "
local kept, count, total = {}, 0, 0
for i = 1, 30000 do
  local v = {i, i + 1, tag = i % 5}
  if i % 50 == 0 then
    count = count + 1
    kept[count] = v
  end
  total = total + v[1] + v.tag
end
local sum = 0
for j = 1, count do sum = sum + kept[j][1] * 3 + kept[j][2] + kept[j].tag end
return total, count, sum, #kept[7], kept[600][1]
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
        assert!(
            tracing.stats.side_installs >= 1,
            "{profile:?}: no side joined the tree\n{:?}",
            tracing.stats
        );
        // The loop runs in native code through the rare branch too: far
        // fewer native entries than rare iterations.
        assert!(
            tracing.stats.native_entries < 200,
            "{profile:?}: {} native entries\n{:?}",
            tracing.stats.native_entries,
            tracing.stats
        );
    }
}

#[test]
fn kept_tables_have_their_contents_and_order() {
    let source = format!(
        "{DUMP}
local kept, n = {{}}, 0
local acc = 0
for i = 1, 12000 do
  local v = {{x = i, y = i * 2, [1] = i % 3, [2] = 'k' .. (i % 4), flag = i % 2 == 0}}
  if i % 37 == 0 then
    n = n + 1
    kept[n] = v
  end
  acc = acc + v.x + v[1]
end
local out = {{}}
for j = 1, n, 29 do out[#out + 1] = dump(kept[j]) .. '|' .. order(kept[j]) end
return acc, n, table.concat(out, ';')
"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn every_kept_table_is_its_own_table() {
    // More rare iterations in one run than the run has lanes for tables:
    // the guard leaves through its exit when the lanes are used up, and each
    // kept table is still a table of its own.
    let source = "
local kept, n = {}, 0
local acc = 0
for i = 1, 20000 do
  local v = {i, tag = i}
  if i % 3 == 0 then
    n = n + 1
    kept[n] = v
  end
  acc = acc + v[1]
end
local distinct, seen = 0, {}
for j = 1, n do
  if not seen[kept[j]] then distinct = distinct + 1 seen[kept[j]] = true end
  kept[j].tag = kept[j].tag + j
end
local sum = 0
for j = 1, n do sum = (sum + kept[j].tag * 7 + kept[j][1]) % 1000003 end
return acc, n, distinct, sum
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn the_side_writes_the_table_it_was_given() {
    let source = format!(
        "{DUMP}
local kept, n = {{}}, 0
local acc = 0
for i = 1, 12000 do
  local v = {{a = i, b = 2}}
  if i % 41 == 0 then
    v.b = v.a + 1
    v.c = 'new'
    v[1] = i
    n = n + 1
    kept[n] = v
  end
  acc = acc + v.a + v.b
end
return acc, n, dump(kept[1]), dump(kept[n]), order(kept[3])
"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_table_inside_a_table_crosses_into_the_side() {
    let source = format!(
        "{DUMP}
local kept, n = {{}}, 0
local acc = 0
for i = 1, 12000 do
  local inner = {{i, i + 1}}
  local outer = {{inner = inner, id = i}}
  if i % 43 == 0 then
    n = n + 1
    kept[n] = outer
  end
  acc = acc + outer.id + inner[2]
end
local same = 0
for j = 1, n do
  if kept[j].inner[1] == kept[j].id then same = same + 1 end
end
return acc, n, same, dump(kept[5].inner), kept[n].inner[2]
"
    );
    for (profile, interpreter, tracing) in agree(&source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn two_tables_cross_into_the_side() {
    let source = "
local left, right, n = {}, {}, 0
local acc = 0
for i = 1, 12000 do
  local a = {i, kind = 'a'}
  local b = {i * 2, kind = 'b'}
  if i % 47 == 0 then
    n = n + 1
    left[n] = a
    right[n] = b
  end
  acc = acc + a[1] + b[1]
end
local sum, mixed = 0, 0
for j = 1, n do
  sum = sum + left[j][1] + right[j][1]
  if left[j] == right[j] or left[j].kind ~= 'a' or right[j].kind ~= 'b' then mixed = mixed + 1 end
end
return acc, n, sum, mixed
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_table_with_a_metatable_crosses_into_the_side() {
    let source = "
local mt = {__index = function(_, k) return k == 'missing' and 99 or nil end}
local kept, n = {}, 0
local acc = 0
for i = 1, 12000 do
  local v = setmetatable({x = i}, mt)
  if i % 53 == 0 then
    n = n + 1
    kept[n] = v
  end
  acc = acc + v.x
end
local sum = 0
for j = 1, n do
  if getmetatable(kept[j]) == mt then sum = sum + kept[j].x + kept[j].missing end
end
return acc, n, sum
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn the_side_leaves_with_the_table_it_was_given() {
    // The side itself has a guard that fails later (the type of `extra`
    // changes), and the interpreter then reads the table the guard built.
    let source = "
local kept, n = {}, 0
local acc = 0
local extra = 1
for i = 1, 16000 do
  local v = {i, w = i % 9}
  if i % 31 == 0 then
    if i > 9000 then extra = 'x' end
    n = n + 1
    kept[n] = v
    if type(extra) == 'number' then acc = acc + extra else acc = acc + #extra + v.w end
  end
  acc = acc + v[1] + v.w
end
local sum = 0
for j = 1, n do sum = sum + kept[j][1] + kept[j].w end
return acc, n, sum
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn numbers_of_either_subtype_beside_the_built_table() {
    // Where the profile has integers, `x` is an integer until `step` turns
    // into a float inside the rare branch.
    let source = "
local kept, n, acc = {}, 0, 0
local step = 1
for i = 1, 16000 do
  local x = i * step
  local v = {x, tag = i % 5}
  if i % 40 == 0 then
    n = n + 1
    kept[n] = v
    if i == 8000 then step = 1.5 end
  end
  acc = acc + v[1] + x
end
local sum = 0
for j = 1, n do sum = sum + kept[j][1] + kept[j].tag end
return acc, n, sum, kept[1][1], kept[n][1]
";
    for (profile, interpreter, tracing) in agree(source) {
        assert_each_constructor_ran_once(profile, &interpreter, &tracing);
    }
}

#[test]
fn a_table_carried_by_the_loop_after_the_side() {
    // The side leaves its table in a register the loop carries, while the
    // loop makes strings that it releases at the end of each iteration.
    let source = "
local last = {0, name = 'none'}
local acc, len = 0, 0
for i = 1, 16000 do
  local v = {i, name = 'n'}
  local s = tostring(i % 10) .. 'x'
  len = len + #s
  if i % 29 == 0 then last = v end
  acc = acc + last[1] + v[1]
end
return acc, len, last[1], last.name
";
    agree(source);
}

#[test]
fn kept_tables_survive_collections() {
    let source = "
local kept, n = {}, 0
local acc = 0
local junk
for i = 1, 30000 do
  local v = {i, i * 2, name = i % 13}
  junk = {i, {i}, tostring(i)}
  if i % 17 == 0 then
    n = n + 1
    kept[n] = v
  end
  acc = acc + v[2] + v.name
end
collectgarbage()
local sum = 0
for j = 1, n do sum = sum + kept[j][1] + kept[j][2] + kept[j].name end
return acc, n, sum, junk[1]
";
    agree(source);
}

#[test]
fn kept_tables_in_a_weak_table_are_collected() {
    let source = "
local weak = setmetatable({}, {__mode = 'v'})
local strong, n = {}, 0
local acc = 0
for i = 1, 16000 do
  local v = {i, tag = i % 7}
  if i % 23 == 0 then
    n = n + 1
    weak[n] = v
    if n % 2 == 0 then strong[n] = v end
  end
  acc = acc + v[1] + v.tag
end
collectgarbage()
collectgarbage()
local alive, sum = 0, 0
for j = 1, n do
  if weak[j] then alive = alive + 1 sum = sum + weak[j][1] end
end
return acc, n, alive, sum
";
    agree(source);
}

#[test]
fn an_error_in_the_side_sees_the_table() {
    let source = "
local kept, n = {}, 0
local function body(limit)
  local acc = 0
  for i = 1, limit do
    local v = {i, tag = i % 5}
    if i % 19 == 0 then
      n = n + 1
      kept[n] = v
      if i > 9000 then error({at = i, table = v}) end
    end
    acc = acc + v[1] + v.tag
  end
  return acc
end
local ok, err = pcall(body, 12000)
return ok, type(err), err.at, err.table[1], err.table.tag, err.table == kept[n], n
";
    agree(source);
}

#[test]
fn memory_count_stays_close_to_the_interpreters() {
    // The tables the guard builds are charged like the interpreter's: the
    // heap after a full collection holds the kept tables and nothing else.
    let source = "
collectgarbage()
local before = collectgarbage('count')
local kept, n = {}, 0
local acc = 0
for i = 1, 40000 do
  local v = {i, i, i, tag = i}
  if i % 20 == 0 then
    n = n + 1
    kept[n] = v
  end
  acc = acc + v[1]
end
collectgarbage()
collectgarbage()
local grown = collectgarbage('count') - before
return acc, n, grown > 50, grown < 2000
";
    agree(source);
}

// ---- tonumber of a number ----

#[test]
fn tonumber_of_a_number_is_the_number() {
    let source = "
local values = {1.5, -0.0, 1e308 * 10, -(1e308 * 10), 0, 3, -7, 2^53, 0.1}
local acc, text = 0, {}
for i = 1, 9000 do
  local x = values[i % #values + 1]
  local y = tonumber(x)
  if y == y and y ~= 1e308 * 10 and y ~= -(1e308 * 10) then acc = acc + y end
  if i <= 18 then text[#text + 1] = tostring(y) .. ':' .. tostring(1 / y) end
end
local nan = 0 / 0
local nans = 0
for i = 1, 3000 do
  local y = tonumber(nan)
  if y ~= y then nans = nans + 1 end
end
return acc, nans, table.concat(text, ' ')
";
    agree(source);
}

#[test]
fn tonumber_leaves_the_move_for_other_arguments() {
    let source = "
local acc, nils = 0, 0
local mixed = {5, '7', ' 0x10 ', 'x', 2.5, '1e2', true, '', '10'}
for i = 1, 9000 do
  local x = mixed[i % #mixed + 1]
  local y = tonumber(x)
  if y then acc = acc + y else nils = nils + 1 end
end
local based = 0
for i = 1, 6000 do
  based = based + tonumber('ff', 16) + tonumber('11', 2) + (tonumber(i % 7 .. '', 8) or 0)
end
local late = 0
for i = 1, 9000 do
  local x = i
  if i > 7000 then x = tostring(i) end
  late = late + tonumber(x)
end
return acc, nils, based, late
";
    agree(source);
}

#[test]
fn tonumber_replaced_by_the_program() {
    let source = "
local real = tonumber
local acc = 0
for i = 1, 9000 do
  if i == 6000 then tonumber = function(x) return x * 2 end end
  acc = acc + tonumber(i + 0.5)
end
tonumber = real
return acc
";
    agree(source);
}

// ---- a virtual table blocks no hoist ----

#[test]
fn a_metatable_that_changes_beside_virtual_tables() {
    let source = "
local mt = {}
local acc = 0
local function count(t) acc = acc + 1 return 5 end
for i = 1, 12000 do
  local v = setmetatable({i, k = i % 3}, mt)
  if i == 4000 then mt.__index = function(_, key) return 100 end end
  if i == 8000 then mt.__newindex = function(t, key, value) rawset(t, key, value * 2) end end
  acc = acc + v[1] + v.k + (v.other or 0)
  v.fresh = 1
  acc = acc + v.fresh
end
return acc
";
    agree(source);
}

#[test]
fn a_metatable_changed_through_a_call_beside_virtual_tables() {
    let source = "
local mt = {}
local function arm(i)
  if i == 5000 then mt.__index = function() return 7 end end
  if i == 9000 then mt.__index = nil end
end
local acc = 0
for i = 1, 12000 do
  local v = setmetatable({a = i % 11}, mt)
  arm(i)
  acc = acc + v.a + (v.b or 1)
end
return acc
";
    agree(source);
}

#[test]
fn a_metatable_changed_between_loops_beside_virtual_tables() {
    let source = "
local mt = {}
local function pass(n)
  local acc = 0
  for i = 1, n do
    local v = setmetatable({i % 5, tag = 2}, mt)
    acc = acc + v[1] + v.tag + (v.none or 0)
  end
  return acc
end
local a = pass(6000)
mt.__index = function() return 3 end
local b = pass(6000)
mt.__index = {none = 4}
local c = pass(6000)
mt.__mode = 'k'
local d = pass(6000)
return a, b, c, d
";
    agree(source);
}

#[test]
fn a_finalizer_metatable_beside_virtual_tables() {
    let source = "
local finalized = 0
local mt = {__gc = function() finalized = finalized + 1 end}
local acc = 0
for i = 1, 12000 do
  local v = setmetatable({i % 4}, mt)
  acc = acc + v[1]
end
collectgarbage()
collectgarbage()
return acc, finalized <= 12000
";
    agree(source);
}

#[test]
fn reads_of_another_table_beside_virtual_tables() {
    let source = "
local config = {scale = 2, offset = 1, list = {3, 4, 5}}
local acc = 0
for i = 1, 12000 do
  local v = {i % 7, w = config.scale}
  if i == 5000 then config.scale = 3 end
  if i == 9000 then config.list[2] = 40 end
  acc = acc + v[1] * v.w + config.offset + config.list[2]
end
return acc
";
    agree(source);
}
