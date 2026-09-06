//! Stores into the array part of a table from a trace.
//!
//! A trace stores a number or boolean over a slot that holds one of its own
//! type by writing the payload, fills a nil hole of a table with no hash
//! part, appends to a dense list, pops its last slot, and grows a list of
//! booleans a few slots past its end. Everything else is the VM's store.
//! Each program here gives the interpreter's result under every profile, and
//! most push inputs off the short way: a slot of another type, a slot that
//! holds a string or a table, a nil, a key past the end, below 1, fractional
//! or NaN, the slot after the border, a table with a hash part or
//! `__newindex`, a table that changes its representation, and tables and
//! strings stored while the collector runs.
//!
//! The loops are long, and those in functions run for thousands of
//! iterations a call: a loop with a few hundred iterations left is not
//! compiled.

use oximond::runtime::Value;
use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// What every program starts with: `sig(t, n)` folds the keys 1 to `n` of
/// `t` into one number that tells nil, false, true, a number, a string and a
/// table apart, `count(t)` counts its keys with `next`, and `order(t)` folds
/// its number keys in the order `next` gives them, which tells a key in the
/// array part from the same key in the hash part.
const PRELUDE: &str = r"
local function sig(t, n)
  local h = 7
  for i = 1, n do
    local v = rawget(t, i)
    local c
    if v == nil then c = 1
    elseif v == false then c = 2
    elseif v == true then c = 3
    elseif type(v) == 'number' then c = 5 + math.floor(v * 4) % 9973
    elseif type(v) == 'string' then c = 10007 + #v
    else c = 20011 + (rawget(v, 'id') or 0) end
    h = (h * 31 + c) % 1000003
  end
  return h
end
local function count(t)
  local n = 0
  for _ in next, t do n = n + 1 end
  return n
end
local function order(t)
  local h = 3
  for k in next, t do
    if type(k) == 'number' then h = (h * 31 + math.floor(k) % 9973) % 1000003 end
  end
  return h
end
";

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let source = format!("{PRELUDE}{source}");
    let results = lua.exec(&source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in both modes of every profile; the results agree.
fn run_each(source: &str) -> Vec<TraceStatistics> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?}"
        );
        all.push(lua.vm().jit_statistics().trace);
    }
    all
}

/// Asserts that every profile ran native iterations.
fn assert_native(all: &[TraceStatistics]) {
    for stats in all {
        assert!(stats.iterations > 0, "no native iteration: {stats:?}");
    }
}

#[test]
fn a_sieve_marks_booleans_past_the_end_into_holes_and_over_booleans() {
    // The pass of 2 grows the list two slots at a time, the later passes
    // fill holes and write over booleans, and each pass ends past the limit.
    assert_native(&run_each(
        r"
local total, marks = 0, 0
for round = 1, 3 do
  local limit = 20000 + round
  local composite = {}
  for i = 2, limit do
    if not composite[i] then
      total = total + 1
      local multiple = i * i
      while multiple <= limit do
        composite[multiple] = true
        multiple = multiple + i
      end
    end
  end
  marks = (marks + sig(composite, limit + 4) + #composite + order(composite)) % 1000003
end
return total, marks
",
    ));
}

#[test]
fn growth_with_holes_takes_every_step_and_value() {
    // One loop, entered with each step. A step of 1 appends, steps up to the
    // reach leave holes (one, then several), longer ones go to the hash
    // part, and false is stored as true is. A number is not grown with
    // holes.
    assert_native(&run_each(
        r"
local function grow(step, n)
  local flags, numbers = { false }, { 0 }
  local key = 1
  for i = 1, n do
    key = key + step
    flags[key] = (i % 3 == 0)
    numbers[key] = i
  end
  local out = sig(flags, key + 2) + sig(numbers, key + 2) * 3 + count(flags) * 5 + count(numbers)
  return (out + order(flags) * 7 + order(numbers) + #flags) % 1000003
end
local out = 0
for _, step in ipairs({ 1, 2, 3, 2, 15, 16, 17, 16, 24, 2, 1 }) do
  out = (out * 3 + grow(step, 9000)) % 1000003
end
-- A list with holes whose last two steps are longer: the longest step the
-- array part takes, and the first that goes to the hash part. The step is
-- arithmetic, so the trace that made the short steps makes the long ones.
local function leap(far, n)
  local flags = { false }
  local key = 1
  for i = 1, n do
    local late = (i - i % (n - 1)) / (n - 1)
    key = key + 2 + (far - 2) * late
    flags[key] = (i % 3 == 0)
  end
  return (sig(flags, key + 2) + count(flags) * 5 + order(flags) * 7 + #flags) % 1000003
end
for _, far in ipairs({ 14, 15, 16, 17, 18, 40, 16, 17 }) do
  out = (out * 3 + leap(far, 9000)) % 1000003
end
-- A first store far into an empty table, then stores below it.
local late = {}
for i = 1, 20000 do
  late[9 + (i % 7) * 2] = true
  late[i % 5 + 1] = false
end
return out, sig(late, 40), count(late), order(late)
",
    ));
}

#[test]
fn overwrites_keep_the_type_or_change_it() {
    // The same slots take numbers, then booleans, then numbers again, in a
    // list (dense) and in one with a hole, so a store meets its own type and
    // every other scalar.
    assert_native(&run_each(
        r"
local dense, holey, sum = {}, {}, 0
for i = 1, 64 do dense[i] = i end
for i = 1, 64 do if i % 9 ~= 0 then holey[i] = i * 0.5 end end
for i = 1, 30000 do
  local k = i % 64 + 1
  local v
  if i % 1000 < 400 then v = i
  elseif i % 1000 < 700 then v = (i % 2 == 0)
  else v = i + 0.25 end
  dense[k] = v
  holey[k] = v
  if i % 5000 == 0 then sum = (sum + sig(dense, 66) + sig(holey, 66) + #dense) % 1000003 end
end
return sum, sig(dense, 66), sig(holey, 66), count(holey)
",
    ));
}

#[test]
fn a_slot_that_holds_a_string_or_a_table_is_the_vms_to_overwrite() {
    // The slots hold numbers, booleans, strings and tables in turn. A
    // scalar over a string or a table releases the reference, which is the
    // VM's store, and the trace goes on with the other slots.
    assert_native(&run_each(
        r"
local names, pool = {}, {}
for j = 1, 16 do names[j] = 'name' .. j; pool[j] = { id = j } end
local t, sum = {}, 0
for i = 1, 200 do t[i] = i end
for i = 1, 40000 do
  local k = i % 200 + 1
  if i % 7 == 0 then t[k] = names[i % 16 + 1]
  elseif i % 11 == 0 then t[k] = pool[i % 16 + 1]
  else t[k] = (i % 2 == 0) end
  t[(i * 7) % 200 + 1] = i
  if i % 8000 == 0 then
    sum = (sum + sig(t, 201)) % 1000003
    collectgarbage()
  end
end
return sum, #t, sig(t, 201)
",
    ));
}

#[test]
fn nil_pops_the_last_slot_and_punches_holes_elsewhere() {
    assert_native(&run_each(
        r"
local t, sum = {}, 0
for i = 1, 20000 do
  t[#t + 1] = i
  t[#t + 1] = i * 2
  -- The last slot: a pop.
  t[#t] = nil
  -- One past the end: nothing to store.
  t[#t + 1] = nil
  if i % 4000 == 0 then
    -- A slot in the middle: a hole, then the list is filled again.
    local at = math.floor(#t / 2)
    t[at] = nil
    sum = (sum + sig(t, #t + 2) + count(t)) % 1000003
    t[at] = -i
  end
end
return sum, #t, sig(t, #t + 2), count(t)
",
    ));
}

#[test]
fn keys_that_name_no_slot_go_to_the_hash_part() {
    // Zero, negative, fractional and huge keys, in the same loop as keys
    // inside the array, as integers and as floats.
    assert_native(&run_each(
        r"
local t, sum = {}, 0
for i = 1, 100 do t[i] = 0 end
local keys = {
  0, -1, -7, 0.5, 2.5, 100.25, -0.5, 2^53, 2^63, 1e300, -1e300, 2^31, 101, 103,
  5e-324, 1 + 2^-52, 4 - 2^-50, 1 / 0, -1 / 0,
}
local function store(k, n)
  for i = 1, n do
    t[k] = (i % 2 == 0)
    t[k] = i
    t[i % 100 + 1] = (i % 3 == 0)
    t[(i % 100 + 1) + 0.0] = i + 1
    t[i % 100 + 0.5] = i
  end
end
for n = 1, #keys do store(keys[n], 6000) end
for _, k in ipairs(keys) do sum = (sum + (t[k] or -1) % 9973) % 1000003 end
return sum, sig(t, 104), count(t), order(t)
",
    ));
    // A NaN key is an error, which leaves the table as it was.
    assert_native(&run_each(
        r"
local t, errors = {}, 0
for i = 1, 50 do t[i] = i end
local nan = 0 / 0
local function fill(n, bad)
  for i = 1, n do
    local k = i % 50 + 1
    if i == bad then k = nan end
    t[k] = (i % 3 == 0)
    t[(i * 7) % 50 + 1] = i
  end
end
for round = 1, 6 do
  local ok = pcall(fill, 12000, round % 2 == 0 and 9000 + round or 0)
  if not ok then errors = errors + 1 end
end
return errors, sig(t, 52), count(t)
",
    ));
}

#[test]
fn a_table_with_a_hash_part_overwrites_in_place_and_appends_through_the_vm() {
    assert_native(&run_each(
        r"
local t, sum = { name = 'x', n = 0 }, 0
for i = 1, 30000 do
  local k = i % 300 + 1
  t[k] = i
  t.n = t.n + 1
  if i % 2000 == 0 then
    t[#t + 1] = true
    t['k' .. i] = i
    sum = (sum + sig(t, 320) + count(t)) % 1000003
  end
end
return sum, t.n, #t, sig(t, 320), count(t)
",
    ));
}

#[test]
fn newindex_sees_absent_keys_only() {
    // The loop fills the holes it punches. Once the table has a metatable
    // with `__newindex`, a store into a hole calls it, with a nil too, and
    // the function stores some and drops others. A present key is still
    // overwritten raw.
    assert_native(&run_each(
        r"
local calls = 0
local function churn(t, n)
  for i = 1, n do
    t[i % 64 + 1] = i
    t[(i * 7) % 64 + 1] = nil
    t[(i * 11) % 64 + 1] = (i % 2 == 0)
  end
end
local t = {}
for i = 1, 64 do t[i] = i end
churn(t, 12000)
local before = sig(t, 66) + count(t)
setmetatable(t, { __newindex = function(self, k, v)
  calls = calls + 1
  if calls % 3 == 0 then rawset(self, k, v) end
end })
churn(t, 12000)
local with = sig(t, 66) + count(t)
setmetatable(t, nil)
churn(t, 12000)
return calls, before, with, sig(t, 66), count(t)
",
    ));
}

#[test]
fn a_table_changes_its_representation_under_the_store() {
    // The array part is emptied and refilled, turned into hash keys by
    // sparse stores, and brought back; the stores of the loop go on.
    assert_native(&run_each(
        r"
local t, sum = {}, 0
for i = 1, 40000 do
  local k = i % 128 + 1
  t[k] = (i % 2 == 0)
  if i % 6000 == 0 then
    for j = 128, 1, -1 do t[j] = nil end
    sum = (sum + count(t)) % 1000003
  elseif i % 6000 == 2000 then
    for j = 1, 128, 3 do t[j] = nil end
    t[100000 + i] = i
    sum = (sum + sig(t, 130) + count(t)) % 1000003
  elseif i % 6000 == 4000 then
    for j = 1, 128 do t[j] = j end
    sum = (sum + #t) % 1000003
  end
end
return sum, sig(t, 130), count(t)
",
    ));
}

#[test]
fn tables_and_strings_stored_in_a_list_survive_collections() {
    // An old list takes tables and strings the interpreter just made, by
    // append and over what it held, with steps and full collections between
    // the runs of the loop. What it holds at the end is alive and is what
    // was stored. A profile that has no collector mode by that name goes on
    // in the mode it has.
    for mode in ["incremental", "generational"] {
        let source = format!(
            r"
pcall(collectgarbage, '{mode}')
local keep, sum = {{}}, 0
for i = 1, 256 do keep[i] = {{ id = i }} end
collectgarbage()
collectgarbage()
local function store(fresh, names, n, base)
  for i = 1, n do
    local k = (i + base) % 256 + 1
    if i % 3 == 0 then keep[k] = fresh[i % 32 + 1]
    elseif i % 3 == 1 then keep[k] = names[i % 32 + 1]
    else keep[k] = i end
    keep[#keep + 1] = fresh[(i + 5) % 32 + 1]
    keep[#keep] = nil
  end
end
for round = 1, 6 do
  local fresh, names = {{}}, {{}}
  for j = 1, 32 do
    fresh[j] = {{ id = round * 1000 + j }}
    names[j] = 'v' .. (round * 100 + j)
  end
  store(fresh, names, 9000, round * 37)
  fresh, names = nil, nil
  if round % 2 == 0 then collectgarbage() else collectgarbage('step') end
end
collectgarbage()
for i = 1, 256 do
  local v = keep[i]
  if type(v) == 'table' then sum = (sum + v.id) % 1000003
  elseif type(v) == 'string' then sum = (sum + #v + tonumber(string.sub(v, 2))) % 1000003
  else sum = (sum + v) % 1000003 end
end
return sum, #keep, sig(keep, 258)
"
        );
        assert_native(&run_each(&source));
    }
}

#[test]
fn insert_and_remove_share_a_list_with_keyed_stores() {
    // `table.insert` and `table.remove` take the list in place (a strict
    // header). The keyed stores between them overwrite a slot of their own
    // type, change a slot's type, and append. The second list has a named
    // field, so only the overwrites of a slot's own type stay in place.
    assert_native(&run_each(
        r"
local plain, named, sum = {}, { tag = 'n' }, 0
for i = 1, 64 do plain[i] = i; named[i] = i end
for i = 1, 30000 do
  table.insert(plain, i)
  table.insert(named, i)
  plain[i % 64 + 1] = i
  named[i % 64 + 1] = i
  if i % 16 == 0 then
    plain[i % 64 + 1] = (i % 32 == 0)
    named[i % 64 + 1] = (i % 32 == 0)
  end
  sum = (sum + table.remove(plain) + table.remove(named)) % 1000003
  if i % 5000 == 0 then
    plain[#plain + 1] = i
    named[#named + 1] = i
  end
end
return sum, #plain, #named, sig(plain, 80), sig(named, 80), count(named)
",
    ));
}

#[test]
fn the_slot_after_the_border_is_filled_by_the_vm() {
    // Every other slot is a hole the growth of the list left, so the table
    // has no record of a deleted key. Filled from the back, each store lands in
    // a hole away from the border and the length stays. Filled from the
    // front, each lands in the slot after the border, whose fill moves the
    // border over the slot that follows, which the length shows.
    assert_native(&run_each(
        r"
local function holes(n)
  local t = { true }
  for i = 3, n, 2 do t[i] = true end
  return t
end
local sum = 0
local back, front = holes(24000), holes(24000)
for i = 24000, 2, -2 do
  back[i] = false
  if i % 250 == 0 then sum = (sum + #back) % 1000003 end
end
for i = 2, 24000, 2 do
  front[i] = false
  if i % 250 == 0 then sum = (sum + #front) % 1000003 end
end
return sum, #back, #front, sig(back, 24002), sig(front, 24002), order(front)
",
    ));
}

#[test]
fn an_odd_key_meets_a_list_that_is_still_dense() {
    // Each odd key arrives once, in compiled code, at a fresh dense list
    // with no hash part: the loop warmed on whole float keys first.
    assert_native(&run_each(
        r"
local keys = {
  0, -1, 0.5, 2.5, 60.25, -0.5, 2^53, 2^63, -2^63, 1e300, -1e300, 2^31,
  5e-324, 1 + 2^-52, 4 - 2^-50, 1 / 0, -1 / 0, 41, 42, 0.999999,
}
local function store(t, odd, n)
  for i = 1, n do
    local k = (i % 40 + 1) + 0.0
    if i == n then k = odd end
    t[k] = i
    if i == n then t[odd] = nil end
  end
end
local total = 0
for round = 1, 3 do
  for n = 1, #keys do
    local t = {}
    for i = 1, 41 do t[i] = 0 end
    store(t, keys[n], 4000)
    total = (total * 31 + sig(t, 44) + count(t) * 7 + order(t) + #t) % 1000003
  end
end
return total
",
    ));
}

// ---- sets with a key the range plan proves ------------------------------------
//
// A counted loop whose set is keyed on its induction runs a version that
// stores without bounds checks while the keys stay inside the array, and the
// guarded version otherwise. The programs below move between the two.

#[test]
fn counted_loops_store_over_every_kind_of_slot() {
    // Each loop is compiled on a list of booleans and then meets lists of
    // numbers, floats, holes, strings, tables and one with a hash part: a
    // slot of the value's type, one of another scalar type, a hole, a
    // reference (the VM's to release) and a table that is not solid.
    assert_native(&run_each(
        r"
local function make(kind, n)
  local t = {}
  for i = 1, n do
    if kind == 1 then t[i] = false
    elseif kind == 2 then t[i] = i
    elseif kind == 3 then t[i] = i + 0.5
    elseif kind == 4 then t[i] = true
    elseif kind == 5 then if i % 5 == 0 then t[i] = 's' .. i else t[i] = i end
    elseif kind == 6 then if i % 7 == 0 then t[i] = { id = i } else t[i] = true end
    else t[i] = i end
  end
  if kind == 4 then for i = 3, n - 1, 3 do t[i] = nil end end
  if kind == 7 then t.name = 'hash'; for i = 4, n - 1, 4 do t[i] = nil end end
  return t
end
local function flags(t, n)
  for i = 1, n do t[i] = (i % 2 == 0) end
end
local function numbers(t, n, step)
  local i = 1
  while i <= n do
    t[i] = i * step
    i = i + step
  end
end
local function shifted(t, n)
  for i = 2, n - 1 do t[i - 1] = i + 0.25 end
end
local function downward(t, n)
  for i = n, 1, -1 do t[i] = i end
end
local function fold(t, n)
  return sig(t, n + 4) + count(t) * 7 + order(t) + #t
end
local out, n = 0, 600
for round = 1, 3 do
  for kind = 1, 7 do
    local t = make(kind, n)
    flags(t, n)
    out = (out * 31 + fold(t, n)) % 1000003
    t = make(kind, n)
    numbers(t, n, round)
    out = (out * 31 + fold(t, n)) % 1000003
    t = make(kind, n)
    shifted(t, n)
    out = (out * 31 + fold(t, n)) % 1000003
    t = make(kind, n)
    downward(t, n)
    out = (out * 31 + fold(t, n)) % 1000003
  end
end
return out
",
    ));
}

#[test]
fn counted_stores_run_every_trip_count_inside_and_past_the_array() {
    // A loop that tests first and one that tests its next value, called with
    // every start and trip count around a list of sixteen slots: runs that
    // stay inside, start below 1, end past the last slot (the list grows or
    // the key goes to the hash part) and make no trip at all.
    assert_native(&run_each(
        r"
local function mark(t, from, to, step, v)
  local i = from
  while i <= to do
    t[i] = v
    i = i + step
  end
end
local function fill(t, from, to, step, v)
  for i = from, to, step do t[i] = v end
end
local function back(t, from, to, v)
  for i = from, to, -1 do t[i] = v end
end
local starts = { 1, 2, 5, 15, 16, 17, 0, -1, 30 }
local out, calls = 0, 0
local t = {}
for round = 1, 40 do
  for s = 1, #starts do
    for trips = 0, 12 do
      for step = 1, 3 do
        calls = calls + 1
        if calls % 97 == 1 then
          out = (out * 31 + sig(t, 60) + count(t) * 7 + order(t) + #t) % 1000003
          t = {}
          for i = 1, 16 do t[i] = (i % 2 == 0) end
        end
        local from = starts[s]
        local to = from + (trips - 1) * step
        local v = (calls % 3 == 0)
        if calls % 5 == 0 then v = calls end
        if round % 3 == 0 then mark(t, from, to, step, v)
        elseif round % 3 == 1 then fill(t, from, to, step, v)
        else back(t, to, from, v) end
      end
    end
  end
end
return out, sig(t, 60), count(t), order(t), #t
",
    ));
}

#[test]
fn a_table_that_changes_under_a_counted_store_leaves_the_fast_loop() {
    // The loop stores into a list while, at one iteration, the list gains a
    // hash part, grows, shrinks, gets a `__newindex` over its holes, or has
    // a string, a table or a hole put in front of the store.
    assert_native(&run_each(
        r"
local log = 0
local meta = { __newindex = function(t, k, v) log = log + k; rawset(t, k, v) end }
local function change(t, mode, i, n)
  if mode == 1 then t.extra = i
  elseif mode == 2 then t[n + 1] = true
  elseif mode == 3 then t[#t] = nil
  elseif mode == 4 then setmetatable(t, meta)
  elseif mode == 5 then t[i + 3] = 'text'
  elseif mode == 6 then t[i + 3] = { id = i }
  elseif mode == 7 then t[i + 2] = nil
  elseif mode == 8 then for k = #t, i + 2, -1 do t[k] = nil end
  elseif mode == 9 then for k = n + 1, n + 40 do t[k] = k end end
end
local function sweep(t, n, at, mode)
  for i = 1, n do
    if i == at then change(t, mode, i, n) end
    t[i] = (i % 3 == 0)
  end
end
local function crawl(t, n, at, mode)
  local i = 1
  while i <= n do
    if i == at then change(t, mode, i, n) end
    t[i] = i
    i = i + 1
  end
end
local out, n = 0, 300
for round = 1, 6 do
  for mode = 0, 9 do
    for _, at in ipairs({ 1, 2, 150, 296, 299, 300 }) do
      local t = {}
      for i = 1, n do if mode ~= 4 or i % 6 ~= 0 then t[i] = false end end
      if round % 2 == 0 then sweep(t, n, at, mode) else crawl(t, n, at, mode) end
      out = (out * 31 + sig(t, n + 44) + count(t) * 7 + order(t) + #t + log) % 1000003
    end
  end
end
return out, log
",
    ));
}

#[test]
fn a_counted_store_through_two_names_of_one_table() {
    // The loop reads through one register and stores through another, which
    // hold the same table or two, and the stores change what the reads of
    // later iterations find.
    assert_native(&run_each(
        r"
local function spread(a, b, n)
  for i = 2, n do
    local v = a[i - 1]
    if v == true then b[i] = false elseif v == false then b[i] = true else b[i] = i end
  end
end
local function creep(a, b, n)
  local i = 1
  while i <= n - 2 do
    b[i + 2] = a[i] ~= true
    i = i + 1
  end
end
local out, n = 0, 400
for round = 1, 12 do
  local a, b = {}, {}
  for i = 1, n do
    a[i] = (i % 3 == 0)
    if round % 4 == 0 and i % 9 == 0 then b[i] = i else b[i] = (i % 2 == 0) end
  end
  if round % 2 == 0 then spread(a, a, n) else spread(a, b, n) end
  if round % 3 == 0 then creep(b, b, n) else creep(a, b, n) end
  out = (out * 31 + sig(a, n + 2) + sig(b, n + 2) * 3 + count(a) + count(b) + #a + #b) % 1000003
end
return out
",
    ));
}

#[test]
fn a_while_loop_with_odd_starts_bounds_and_steps_stores_like_the_interpreter() {
    // The start, the bound and the step of a loop that tests first, as
    // fractions, large numbers and values past the array: the proof fails
    // where the loop starts and the guarded loop stores what the
    // interpreter stores.
    assert_native(&run_each(
        r"
local function mark(t, from, to, step)
  local i = from
  while i <= to do
    t[i] = true
    i = i + step
  end
end
local cases = {
  { 1, 40, 1 }, { 1.5, 9, 1 }, { 1, 9.5, 1 }, { 1, 40, 1.5 }, { 2, 40, 2 },
  { 39, 44, 1 }, { 0, 3, 1 }, { -3, 2, 1 }, { 1e15, 1e15 + 3, 1 }, { 41, 41, 1 },
  { 1, 40, 2^31 }, { 1, 40, 2^31 + 1 }, { 40, 1, 1 }, { 1, 1 / 0 - 1 / 0, 1 },
  { 3, 40, 37 }, { 1, 2^40, 2^39 }, { 7, 7, 1 }, { 1, 40, 39 },
}
local out = 0
for round = 1, 300 do
  for n = 1, #cases do
    local t = {}
    for i = 1, 40 do t[i] = (i % 2 == 0) end
    local case = cases[n]
    mark(t, case[1], case[2], case[3])
    if round % 50 == 0 then
      out = (out * 31 + sig(t, 46) + count(t) * 7 + order(t) + #t) % 1000003
    end
  end
end
return out
",
    ));
}

#[test]
fn an_index_the_loop_does_not_carry_is_right_wherever_the_loop_leaves() {
    // The loops read and store with their index and nothing else reads it,
    // so the fast loop counts in a scaled copy alone. Each way out hands the
    // index to the code after: the end of the array, the bound, a slot of
    // another type in the middle (a string, a hole, a float among integers),
    // a `break` on a value, and the index read after the loop. Steps of 1, 2
    // and 3, upward and downward, with integer and float indexes by profile.
    assert_native(&run_each(
        r"
local function scan(t, from, to, step)
  local last, total = -1, 0
  for i = from, to, step do
    local v = t[i]
    if v == nil then break end
    t[i] = v % 7
    total = total + v
    last = i
  end
  return last, total
end
local function crawl(t, from, to, step)
  local i, total = from, 0
  while i <= to do
    total = total + t[i]
    t[i] = i
    i = i + step
  end
  return i, total
end
local function down(t, from, to)
  local seen = 0
  for i = from, to, -1 do
    t[i] = t[i] + 1
    seen = i
  end
  return seen
end
local out, n = 0, 400
for round = 1, 24 do
  local t = {}
  for i = 1, n do t[i] = i % 13 end
  local mode = round % 6
  if mode == 1 then t[250] = 'text'
  elseif mode == 2 then t[251] = nil
  elseif mode == 3 then t[249] = 2.5
  elseif mode == 4 then t.extra = round end
  local step = round % 3 + 1
  local ok, a, b = pcall(scan, t, 1, n + (round % 2) * 5, step)
  out = (out * 31 + (ok and a + b or 77)) % 1000003
  ok, a, b = pcall(crawl, t, 1 + round % 2, n - (round % 4) * 3, step)
  out = (out * 31 + (ok and a + b or 78)) % 1000003
  ok, a = pcall(down, t, n - round % 3, 2 + round % 5)
  out = (out * 31 + (ok and a or 79)) % 1000003
  out = (out * 31 + sig(t, n + 6) + count(t)) % 1000003
end
return out
",
    ));
}
