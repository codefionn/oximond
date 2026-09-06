//! Counted loops over arrays, whose keys the backend checks once where the
//! loop starts (see `oximond::jit_ir::trace::range`), against the
//! interpreter.
//!
//! Each loop runs natively in every profile and must give the interpreter's
//! results: loops that stay inside their arrays, and loops that run off an
//! end, start outside, step by fractions or backwards, count near 2^53, grow
//! the array they read, or come back to the loop from a side trace with
//! another start.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const ALL: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

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
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in both modes of every profile: the results agree, and
/// the tracing run entered native code.
fn assert_agrees(source: &str) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
    }
}

#[test]
fn dense_reads_and_writes_agree() {
    assert_agrees(
        r"
        local n = 300
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total = 0
        for round = 1, 40 do
            for i = 1, n do
                local v = values[i] + round
                values[i] = v % 101
                total = total + v
            end
        end
        return total, values[1], values[n]
        ",
    );
}

#[test]
fn a_stencil_with_offsets_on_two_arrays_agrees() {
    assert_agrees(
        r"
        local n = 200
        local a, b = {}, {}
        for i = 1, n do a[i] = i % 89; b[i] = 0 end
        local total = 0
        for _ = 1, 30 do
            for i = 2, n - 1 do
                local v = (a[i - 1] + a[i] * 2 + a[i + 1]) % 1009
                b[i] = v
                total = total + v
            end
            a, b = b, a
        end
        return total, a[2], a[n - 1]
        ",
    );
}

#[test]
fn loops_that_run_off_the_end_agree() {
    // Keys past the end read nil, every round a different share of them.
    assert_agrees(
        r"
        local n = 200
        local values = {}
        for i = 1, n do values[i] = i % 97 end
        local total, missing = 0, 0
        for round = 1, 40 do
            local offset = (round % 4) * 30
            for i = 1, n do
                local v = values[i + offset]
                if v then total = total + v else missing = missing + 1 end
            end
        end
        return total, missing
        ",
    );
    // A loop whose limit lies past the array in some rounds only.
    assert_agrees(
        r"
        local n = 100
        local t = {}
        for i = 1, n do t[i] = i end
        local total = 0
        for round = 1, 60 do
            local m = (round % 3 == 0) and n + 5 or n
            for i = 1, m do
                local v = t[i]
                if v == nil then break end
                total = total + v
            end
        end
        return total
        ",
    );
    // Keys below the array: the hash part answers for zero and less.
    assert_agrees(
        r"
        local t = {}
        for i = 1, 100 do t[i] = i end
        t[0], t[-1] = 1000, 2000
        local total = 0
        for round = 1, 40 do
            local first = (round % 2 == 0) and -1 or 1
            for i = first, 100 do total = total + t[i] end
        end
        return total
        ",
    );
}

#[test]
fn backward_and_fractional_steps_agree() {
    assert_agrees(
        r"
        local t = {}
        for i = 1, 150 do t[i] = i * 3 end
        local total = 0
        for round = 1, 40 do
            for i = 150, 1, -1 do total = total + t[i] end
            for i = 150, 2, -2 do t[i] = t[i] + 1 end
            for i = 149, 0, -1 do
                local v = t[i]
                if v then total = total + v end
            end
        end
        return total
        ",
    );
    // A step of a half reaches every key twice, half of them fractional.
    assert_agrees(
        r"
        local t = {}
        for i = 1, 120 do t[i] = i end
        local total, holes = 0, 0
        for _ = 1, 30 do
            for i = 1, 120, 0.5 do
                local v = t[i]
                if v then total = total + v else holes = holes + 1 end
            end
        end
        return total, holes
        ",
    );
    // A step that is a variable, sometimes fractional.
    assert_agrees(
        r"
        local t = {}
        for i = 1, 120 do t[i] = i end
        local total, holes = 0, 0
        for round = 1, 40 do
            local step = (round % 4 == 0) and 1.5 or 1
            for i = 1, 120, step do
                local v = t[i]
                if v then total = total + v else holes = holes + 1 end
            end
        end
        return total, holes
        ",
    );
}

#[test]
fn bounds_near_two_to_the_fifty_third_agree() {
    assert_agrees(
        r"
        local t = {}
        for i = 1, 50 do t[i] = i end
        local big = 2^53
        t[big - 2] = 7
        local total, holes = 0, 0
        for _ = 1, 40 do
            for i = big - 40, big - 1 do
                local v = t[i]
                if v then total = total + v else holes = holes + 1 end
            end
            for i = 1, big do
                local v = t[i]
                if v == nil then break end
                total = total + v
            end
        end
        return total, holes
        ",
    );
}

#[test]
fn loops_that_grow_their_array_agree() {
    assert_agrees(
        r"
        local total = 0
        for _ = 1, 20 do
            local t = {}
            for i = 1, 40 do t[i] = i end
            for i = 1, 3000 do
                t[i + 40] = t[i] + 1
                total = total + t[i]
            end
            total = total + #t
        end
        return total
        ",
    );
}

#[test]
fn side_traces_that_restart_the_loop_agree() {
    // The inner loop starts at another key each round, and a branch that
    // goes both ways takes a side trace that links back into the loop.
    assert_agrees(
        r"
        local n = 120
        local t = {}
        for i = 1, n do t[i] = i % 13 end
        local total = 0
        for round = 1, 60 do
            for i = round % 30 - 5, n - round % 7 do
                local v = t[i]
                if v == nil then
                    total = total - 1
                elseif i % 3 == 0 then
                    total = total + v * 2
                else
                    total = total + v
                end
            end
        end
        return total
        ",
    );
    // A `while` loop is counted too. The branch that steps by two takes a
    // side trace whose link does not continue the root's induction, and the
    // one that steps by one past the end of the array reads nil.
    assert_agrees(
        r"
        local n = 150
        local t = {}
        for i = 1, n do t[i] = i % 11 end
        local total = 0
        for round = 1, 40 do
            local i = 1
            while i <= n + 2 do
                local v = t[i]
                if v == nil then
                    total = total - 1
                    i = i + 1
                elseif v % 4 == 0 then
                    total = total + v
                    i = i + 2
                else
                    total = total + 1
                    i = i + 1
                end
            end
        end
        return total
        ",
    );
    // A side every few iterations that keeps the root's step: its link goes
    // straight back into the version without checks.
    assert_agrees(
        r"
        local n = 200
        local t = {}
        for i = 1, n do t[i] = i % 9 end
        local total, odd = 0, 0
        for round = 1, 40 do
            for i = 1, n - round % 3 do
                local v = t[i]
                if v % 3 == 0 then odd = odd + 1 else total = total + v end
            end
        end
        return total, odd
        ",
    );
    // Keys read from an array index a hash, and the side that adds a found
    // value links back every few iterations with the accumulator it changed.
    // An early build passed the link's arguments in the wrong order here.
    assert_agrees(
        r#"
        local values, probes = {}, {}
        for i = 1, 300 do values["item_" .. i] = i % 101 end
        for i = 1, 400 do
            if i % 8 == 0 then
                probes[i] = "absent_" .. i
            else
                probes[i] = "item_" .. ((i * 7) % 300 + 1)
            end
        end
        local total, missing = 0, 0
        for _ = 1, 40 do
            for i = 1, 400 do
                local value = values[probes[i]]
                if value then total = total + value else missing = missing + 1 end
            end
        end
        return total, missing
        "#,
    );
    // A side trace's link brings a limit past the array.
    assert_agrees(
        r"
        local n = 100
        local t = {}
        for i = 1, n do t[i] = i end
        local total = 0
        for round = 1, 80 do
            local limit = n - (round % 5)
            if round % 9 == 0 then limit = n + 3 end
            for i = 1, limit do
                local v = t[i]
                if v then total = total + v end
            end
        end
        return total
        ",
    );
}

#[test]
fn a_stencil_that_reaches_the_peak_tier_agrees() {
    // 2.4 million native iterations: the trace recompiles at the peak tier
    // and the array loop must still agree with the interpreter there.
    assert_agrees(
        r"
        local n = 20000
        local a, b = {}, {}
        for i = 1, n do a[i] = i % 97; b[i] = 0 end
        local total = 0
        for pass = 1, 120 do
            for i = 2, n - 1 do
                b[i] = (a[i - 1] + a[i] + a[i + 1]) % 1009
            end
            a, b = b, a
            total = total + a[10000]
        end
        return total, a[1], a[n]
        ",
    );
}

#[test]
fn an_append_between_proven_reads_of_one_index_agrees() {
    // The append may move `a`, so the row of `a[i]` read before it cannot
    // serve `a[i + 1]` and `a[i - 1]` after it. The negative values take the
    // append now and then, in every pass.
    assert_agrees(
        r"
        local n = 3000
        local a, b = {}, {}
        for i = 1, n do
            a[i] = (i % 7 == 0) and -i or i
            b[i] = 0
        end
        local total = 0
        for pass = 1, 6 do
            for i = 2, n - 1 do
                local x = a[i]
                if x < 0 then a[#a + 1] = x end
                b[i] = x + a[i + 1] + a[i - 1]
            end
            total = total + b[2] + b[n - 1] + #a
        end
        return total, #a, b[7], b[1500]
        ",
    );
}

#[test]
fn a_far_offset_keeps_the_checked_path() {
    // `a[i + 200000000]` is a displacement of 3.2 GB, too big for an address
    // mode: the access takes the checked path, leaves the array's end and
    // reads nil, as in the interpreter. A modest offset beside it still reads
    // its row.
    assert_agrees(
        r"
        local n = 2000
        local a = {}
        for i = 1, n do a[i] = i end
        local near, far = 0, 0
        for pass = 1, 4 do
            for i = 1, n - 5 do
                near = near + a[i + 3] + a[i]
                local v = a[i + 200000000]
                if v then far = far + v end
                local w = a[i - 200000000]
                if w then far = far - w end
            end
        end
        return near, far
        ",
    );
}

/// The iterations the tracing run of `source` made natively under `profile`,
/// after checking its results against the interpreter's.
fn native_iterations(profile: CompatibilityProfile, source: &str) -> u64 {
    let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
    let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
    assert!(expected.is_ok(), "{profile:?}: {expected:?}");
    assert_eq!(
        format!("{actual:?}"),
        format!("{expected:?}"),
        "{profile:?}"
    );
    lua.vm().jit_statistics().trace.iterations
}

/// Loops with no bound, which end at the load that finds no element
/// (`ipairs`, `while t[i] do`): the fast loop compares each key with the
/// array's length and proves the rest once. Arrays with holes, with keys in
/// the hash part past the border, with a float among integers, starts at 0,
/// -1 and a fraction, steps of 1, 2 and 3 as integers and floats, a second
/// array read by the index, overwrites from the body, and tables that
/// shrink, grow and move while they are traversed.
#[test]
fn loops_that_end_where_the_array_does_agree() {
    assert_agrees(
        r#"
-- ipairs / open inductions over many shapes
local out = {}
local function emit(...) out[#out + 1] = table.concat({...}, ",") end
local function sum_ipairs(t)
  local s, c = 0, 0
  for i, v in ipairs(t) do s = s + v * i; c = c + 1 end
  return s, c
end
for round = 1, 300 do
  local t = {}
  local n = (round * 7) % 41
  for i = 1, n do t[i] = i + round end
  if round % 5 == 0 and n > 3 then t[n - 2] = nil end      -- hole
  if round % 7 == 0 then t[n + 1] = 5; t[n + 2] = 6 end    -- appended
  if round % 11 == 0 then t[n + 5] = 9 end                 -- hash key past border
  if round % 13 == 0 and n > 2 then t[2] = 2.5 end         -- float among ints
  emit(sum_ipairs(t))
end
-- while loops ending at nil, steps 1, 2, 3, starting at 0, 1, 2, -1
local function walk(t, start, step)
  local i, s, c = start, 0, 0
  while t[i] do s = s + t[i]; i = i + step; c = c + 1 end
  return s, c, i
end
for round = 1, 200 do
  local t = {}
  local n = 30 + round % 17
  for i = 1, n do t[i] = i * 2 end
  if round % 3 == 0 then t[0] = 100 end
  if round % 4 == 0 then t[-1] = 7 end
  if round % 9 == 0 then t[n + 1] = 1; t[n + 3] = 2 end
  emit(walk(t, 1, 1)); emit(walk(t, 0, 1)); emit(walk(t, 2, 2)); emit(walk(t, 1, 3)); emit(walk(t, -1, 1))
  emit(walk(t, 1.0, 1)); emit(walk(t, 1, 1.0)); emit(walk(t, 0.5, 1))
end
-- ipairs with body that reads another array by the index, and overwrites
local a, b = {}, {}
for i = 1, 100 do a[i] = i; b[i] = i * 3 end
b[101] = nil
for round = 1, 60 do
  local s = 0
  for i, v in ipairs(a) do s = s + v + (b[i] or -1000); a[i] = v + 1 end
  if round == 30 then b[50] = nil end
  if round == 40 then a[60] = "7" end
  if round == 45 then a[60] = 7 end
  emit(s)
end
-- table modified during ipairs: shrink and grow from the body
for round = 1, 50 do
  local t = {}
  for i = 1, 64 do t[i] = i end
  local s = 0
  for i, v in ipairs(t) do
    s = s + v
    if i == 20 and round % 2 == 0 then t[40] = nil end
    if i == 30 and round % 3 == 0 then t[#t + 1] = 1000 end
    if i == 10 and round % 5 == 0 then for k = 65, 200 do t[k] = k end end
  end
  emit(s, #t)
end
-- a long ipairs whose table grows from the body: the array moves
do
  local t = {}
  for i = 1, 3000 do t[i] = i end
  local s = 0
  for i, v in ipairs(t) do
    s = s + v
    if i % 500 == 0 and #t < 20000 then for k = #t + 1, #t + 2000 do t[k] = 1 end end
    if i == 9000 then t[9500] = nil end
  end
  emit(s, #t)
end
-- ipairs through __index
local m = setmetatable({}, { __index = function(_, k) if k <= 150 then return k end end })
local s = 0
for round = 1, 20 do for i, v in ipairs(m) do s = s + v end end
emit(s)
local h = 0
for _, line in ipairs(out) do for k = 1, #line do h = (h * 31 + line:byte(k)) % 1000000007 end end
return #out, h
"#,
    );
}

/// A long `ipairs` loop runs natively, one entry for the whole array.
#[test]
fn an_ipairs_loop_stays_native_to_the_end_of_its_array() {
    let source = r"
        local values = {}
        for i = 1, 20000 do values[i] = i % 113 end
        local total = 0
        for _ = 1, 4 do
            for _, value in ipairs(values) do total = total + value end
        end
        return total
    ";
    for profile in ALL {
        let iterations = native_iterations(profile, source);
        assert!(iterations > 60_000, "{profile:?}: {iterations}");
    }
}

/// Arrays indexed from 0 or from a negative key (the keys below 1 live in
/// the hash part), read by an inner counted loop whose outer loop stores into
/// another table, or into the same one under another name, and grows it. The
/// key that misses the array checks the plan again, so the rest of the loop
/// runs without the checks; an absent key 0 raises the interpreter's error,
/// and a string at key 0 is converted as the interpreter converts it.
#[test]
fn arrays_that_start_at_zero_or_below_agree() {
    assert_agrees(
        r#"
local out = {}
local function emit(...) out[#out + 1] = table.concat({...}, ",") end
local function mul(input, output, n, first)
  for i = first, n - 1 do
    local total = 0
    for j = first, n - 1 do total = total + (i + j + 1) * input[j] end
    output[i] = total
  end
end
for round = 1, 40 do
  local n = 20 + round % 9
  for _, first in ipairs({0, 1, -3, -1}) do
    local u, v = {}, {}
    for i = first, n - 1 do u[i] = (i % 7) + 1 end
    mul(u, v, n, first); mul(v, u, n, first)
    local s = 0
    for i = first, n - 1 do s = s + u[i] % 1000 end
    emit(s)
    -- aliasing: output is the input, growing while read
    local w = {}
    for i = first, 5 do w[i] = i + 2 end
    for i = first, n - 1 do
      local total = 0
      for j = first, 5 do total = total + w[j] end
      w[i + 6] = total % 97
    end
    local s2 = 0
    for i = first, n + 5 do s2 = s2 + (w[i] or 0) end
    emit(s2)
  end
end
-- one table under two names: the inner loop reads what the outer store grows
local function grow(input, output, n)
  for i = 0, n - 1 do
    local total = 0
    for j = 0, #input do total = total + (input[j] or 0) end
    output[#output + 1] = total % 1009
  end
end
for round = 1, 20 do
  local t = { [0] = 1, 2, 3 }
  grow(t, t, 150 + round)
  local s = 0
  for i = 0, #t do s = (s + t[i] * i) % 1000003 end
  emit(s, #t)
end
-- absent key 0: errors must match
local function bad(t, n) local s = 0 for j = 0, n do s = s + t[j] end return s end
for round = 1, 30 do
  local t = {}
  for i = 1, 50 do t[i] = i end
  if round % 2 == 0 then t[0] = 1 end
  local ok, err = pcall(bad, t, 50)
  emit(tostring(ok), ok and err or (tostring(err):gsub("^.-:%d+: ", "")))
end
-- key 0 holding other types, start float, downward to 0
local function down(t, n) local s = 0 for j = n, 0, -1 do s = s + t[j] end return s end
for round = 1, 60 do
  local t = {}
  for i = 0, 40 do t[i] = i * (round % 3 == 0 and 1.5 or 1) end
  emit(down(t, 40), down(t, 40.0))
  if round % 10 == 0 then t[0] = "3" emit(down(t, 40)) end
end
local h = 0
for _, line in ipairs(out) do for k = 1, #line do h = (h * 31 + line:byte(k)) % 1000000007 end end
return #out, h
"#,
    );
}

/// A loop over `t[0]` to `t[n]` whose store grows another table runs its
/// inner iterations natively.
#[test]
fn a_zero_based_product_stays_native() {
    let source = r"
        local n = 100
        local function multiply(input, output)
            for i = 0, n - 1 do
                local total = 0
                for j = 0, n - 1 do total = total + (i + j + 1) * input[j] end
                output[i] = total % 1009
            end
        end
        local u = {}
        for i = 0, n - 1 do u[i] = i % 7 + 1 end
        local sum = 0
        for _ = 1, 12 do
            local v = {}
            multiply(u, v)
            multiply(v, u)
            sum = (sum + u[0] + u[n - 1]) % 1000003
        end
        return sum
    ";
    for profile in ALL {
        let iterations = native_iterations(profile, source);
        assert!(iterations > 200_000, "{profile:?}: {iterations}");
    }
}

/// Counted stores into the nil slots after a border, where the fill of the
/// slot after the border is the VM's: over tables with a hash part, with a
/// string or a table in the way, with `__newindex`, and with the length read
/// while the border moves.
#[test]
fn counted_stores_that_fill_the_slots_after_a_border_agree() {
    assert_agrees(
        r#"
-- Counted stores that fill the nil slots after a border, over slots of every kind.
local out = {}
local function emit(...) out[#out + 1] = table.concat({...}, ",") end
local function fill(t, from, to, v)
  for i = from, to do t[i] = v + i end
end
local function fillb(t, from, to)
  for i = from, to do t[i] = (i % 3 == 0) end
end
local function sum(t, n)
  local s, c = 0, 0
  for i = 1, n do
    local v = t[i]
    if v == true then s = s + 1; c = c + 1 elseif v then s = s + v; c = c + 1 end
  end
  return s, c, #t
end
for round = 1, 40 do
  local t = {}
  for i = 1, 400 do t[i] = i end
  -- holes after a border: the array keeps its slots
  local cut = 100 + round * 3
  for i = cut, 400 do t[i] = nil end
  if round % 4 == 0 then t.name = "x" end               -- a hash part
  if round % 5 == 0 then t[cut + 20] = "s" end           -- a reference in the way
  if round % 7 == 0 then t[cut + 30] = {} end
  if round % 6 == 0 then setmetatable(t, { __newindex = function(self, k, v) rawset(self, k, type(v) == 'number' and v * 2 or v) end }) end
  fill(t, cut, 380, round % 2 == 0 and 0.5 or 0)
  emit(sum(t, 400))
  fillb(t, cut - 10, 399)
  emit(sum(t, 400))
  -- the length read in the loop while the border moves
  local u = {}
  for i = 1, 300 do u[i] = i end
  for i = 150, 300 do u[i] = nil end
  local lens = 0
  for i = 150, 290 do u[i] = i; lens = lens + #u end
  emit(lens, #u)
end
local h = 0
for _, line in ipairs(out) do for k = 1, #line do h = (h * 31 + line:byte(k)) % 1000000007 end end
return #out, h
"#,
    );
}

/// The fill of the slots after a border stays in the trace: the VM stores
/// the slot the border moves over and the loop goes on.
#[test]
fn a_fill_after_the_border_stays_native() {
    let source = r"
        local function fill(t, from, to, v)
            for i = from, to do t[i] = v + i end
        end
        local total = 0
        for round = 1, 200 do
            local t = {}
            for i = 1, 400 do t[i] = i end
            for i = 200, 399 do t[i] = nil end
            fill(t, 200, 380, 0)
            total = total + #t
        end
        return total
    ";
    for profile in ALL {
        let iterations = native_iterations(profile, source);
        assert!(iterations > 30_000, "{profile:?}: {iterations}");
    }
}
