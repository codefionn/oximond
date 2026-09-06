//! `%` at one site that sees integers, floats and both mixed, in native
//! traces. The script prints every result with its subtype and 17 digits, so
//! a wrong sign, a lost fraction or an integer that turned float shows. The
//! outputs of the interpreter and of tracing execution must agree, and with
//! the stock Lua binary of the profile when it is installed.

use oximond::vm::TraceStatistics;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Operands of both subtypes: negatives, fractions, zero (float only, since
/// an integer zero divisor raises), infinities, NaN, values beyond 2^53 and
/// the ends of the integer range. It returns one text for the whole run.
const SCRIPT: &str = r#"
local function show(r)
  local kind = math.type and math.type(r) or ""
  if r ~= r then return kind .. ":nan" end
  if kind == "integer" then return "i:" .. tostring(r) end
  return kind .. ":" .. string.format("%.17g", r)
end
local operands = {
  7, -7, 3, -3, 13, 1, -1, 2.5, -2.5, 0.25, -0.25, 7.0, -7.0, 0.5, 1e3,
  -1e3, 0.0, -0.0, 1 / 0, -1 / 0, 0 / 0, BIG, -BIG,
  BIGF, HUGE, -HUGE, 5e-324, 4294967296, -64, 3.0, 1e9 + 0.5,
}
local count = #operands
local out = {}
local function run(rounds, offset)
  for i = 1, rounds do
    local v = operands[(i * 3 + offset) % count + 1]
    local w = operands[(i * 7 + 2 * offset + math.floor(i / count)) % count + 1]
    out[#out + 1] = show(v % w)
  end
end
run(4000, 0)
run(3000, 5)
-- The site's types change after the trace is hot: every operand a float,
-- then every operand an integer.
local function modulo(v, w) return v % w end
local lines = {}
for round = 1, 3 do
  for i = 1, 1500 do
    local v, w
    if round == 1 then v, w = i - 700, (i % 11) + 0.5
    elseif round == 2 then v, w = (i - 700) + 0.25, -((i % 5) + 1)
    else v, w = i - 700, (i % 13) - 6.5 end
    if w ~= 0 then lines[#lines + 1] = show(modulo(v, w)) end
  end
end
-- Integer zero divisors raise, at the iteration that brings one.
local zero_at
local ok, err = pcall(function()
  for i = 1, 3000 do
    local w = 3000 - i
    if w >= 1000 then w = w - 999 end
    local r = (i * 5) % w
    zero_at = i
  end
end)
local message = tostring(err):gsub("^.-:%d+: ", "")
return table.concat(out, ",") .. "|" .. table.concat(lines, ",") .. "|"
  .. tostring(ok) .. "," .. message .. "," .. tostring(zero_at)
"#;

/// The source for a profile. Lua 5.1 has only doubles, so its `BIG` is the
/// double 2^53; Oximond keeps an integer literal beyond 2^53 exact in every
/// profile, which is a difference this test is not about. Its `a - floor(a /
/// b) * b` also rounds differently where a build fuses the multiply and the
/// subtract (the distribution's does), so `BIGF` and `HUGE` stay below the doubles for
/// which the product is inexact.
fn source(profile: CompatibilityProfile) -> String {
    let big = if profile == CompatibilityProfile::Lua54 {
        "9007199254740993"
    } else {
        "1099511627776"
    };
    let float = if profile == CompatibilityProfile::Lua54 {
        "9007199254740992.0"
    } else {
        "1099511627776.5"
    };
    let huge = if profile == CompatibilityProfile::Lua54 {
        "1e300"
    } else {
        "1e10"
    };
    SCRIPT
        .replace("BIGF", float)
        .replace("BIG", big)
        .replace("HUGE", huge)
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source(profile))
        .unwrap_or_else(|error| panic!("{mode:?}: {error}"));
    let Some(Value::String(text)) = results.first() else {
        panic!("a string result, got {results:?}");
    };
    let text = String::from_utf8_lossy(text.as_bytes()).into_owned();
    (text, lua.vm().jit_statistics().trace)
}

fn stock(binary: &str, source: &str) -> Option<String> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = Path::new("/usr/bin").join(binary);
    if !path.exists() {
        return None;
    }
    let directory = std::env::temp_dir().join(format!(
        "oximond-mixed-modulo-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let script = directory.join("script.lua");
    std::fs::write(&script, format!("io.write((function() {source} end)())")).unwrap();
    let output = Command::new(path).arg(&script).output().unwrap();
    let _ = std::fs::remove_dir_all(&directory);
    assert!(
        output.status.success(),
        "{binary} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(String::from_utf8(output.stdout).unwrap())
}

fn check(profile: CompatibilityProfile, binary: &str) {
    let (interpreted, _) = run(profile, ExecutionMode::Interpreter);
    let (traced, stats) = run(profile, ExecutionMode::Tracing);
    assert_eq!(interpreted.split(',').count(), traced.split(',').count());
    for (index, (want, got)) in interpreted.split(',').zip(traced.split(',')).enumerate() {
        assert_eq!(want, got, "{profile:?}: result {index}");
    }
    if let Some(live) = stock(binary, &source(profile)) {
        for (index, (want, got)) in live.split(',').zip(traced.split(',')).enumerate() {
            assert_eq!(want, got, "{profile:?} against {binary}: result {index}");
        }
        assert_eq!(live.split(',').count(), traced.split(',').count());
    }
    assert!(stats.traces_compiled > 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(stats.iterations > 1000, "native code ran: {stats:?}");
}

#[test]
fn polymorphic_modulo_matches_stock_lua_54_in_traces() {
    check(CompatibilityProfile::Lua54, "lua5.4");
}

#[test]
fn polymorphic_modulo_matches_stock_lua_51_in_traces() {
    check(CompatibilityProfile::Lua51, "lua5.1");
}

/// Numbers read from an array that alternates integers and floats, so every
/// site sees the subtype change from one iteration to the next. The results
/// are stored and compared too, and an integer beyond 2^53 and an integer
/// zero divisor sit in the data.
const ALTERNATING: &str = r#"
local function show(r)
  local kind = math.type(r) or type(r)
  if r ~= r then return kind .. ":nan" end
  if kind == "integer" then return "i:" .. tostring(r) end
  if kind == "float" then return "f:" .. string.format("%.17g", r) end
  return kind .. ":" .. tostring(r)
end
local operands = {
  7, 2.5, -7, -2.5, 13, 0.25, 3, 1e3, -1, 0.5, 4294967296, -64, 3.0, 1.5,
  9007199254740992, -9007199254740992, 5, 0.0, -0.0, 1 / 0, 8, -1 / 0, 0 / 0, 2,
}
local count = #operands
local rounds = 40000
local mod, add, sub, mul, div, less, same, other = {}, {}, {}, {}, {}, {}, {}, {}
for i = 1, rounds do
  mod[i], add[i], sub[i], mul[i], div[i], less[i], same[i], other[i] = 0, 0, 0, 0, 0, false, false, 0
end
local total = 0.0
for i = 1, rounds do
  local v = operands[i % count + 1]
  local w = operands[(i * 5 + i // count) % count + 1]
  if w == 0 then w = 3 end
  mod[i] = v % w
  add[i] = v + w
  sub[i] = v - w
  mul[i] = v * w
  div[i] = v / w
  less[i] = v < w
  same[i] = v == w
  other[i] = (v + 1) % 4
  total = total + v % w
end
local lines = {}
for i = 1, rounds do
  lines[i] = show(mod[i]) .. " " .. show(add[i]) .. " " .. show(sub[i]) .. " " .. show(mul[i])
    .. " " .. show(div[i]) .. " " .. tostring(less[i]) .. " " .. tostring(same[i]) .. " "
    .. show(other[i])
end
-- An integer zero divisor raises when the iteration brings it.
local zero_at
local ok, err = pcall(function()
  local fixed = {1, 2.5, 0, 4, 0.0, 3}
  for i = 1, 3000 do
    local v = operands[i % count + 1]
    local w = fixed[i % 6 + 1]
    zero_at = i
    local r = v % w
  end
end)
return table.concat(lines, "\n") .. "\n" .. show(total) .. "\n" .. tostring(ok)
  .. "," .. tostring(err):gsub("^.-:%d+: ", "") .. "," .. tostring(zero_at)
"#;

fn run_alternating(mode: ExecutionMode) -> (String, TraceStatistics) {
    run_source(ALTERNATING, mode)
}

fn run_source(source: &str, mode: ExecutionMode) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{mode:?}: {error}"));
    let Some(Value::String(text)) = results.first() else {
        panic!("a string result, got {results:?}");
    };
    let text = String::from_utf8_lossy(text.as_bytes()).into_owned();
    (text, lua.vm().jit_statistics().trace)
}

#[test]
fn numbers_that_alternate_between_integers_and_floats_match_stock_lua_54_in_traces() {
    let (interpreted, _) = run_alternating(ExecutionMode::Interpreter);
    let (traced, stats) = run_alternating(ExecutionMode::Tracing);
    assert_eq!(interpreted.lines().count(), traced.lines().count());
    for (index, (want, got)) in interpreted.lines().zip(traced.lines()).enumerate() {
        assert_eq!(want, got, "line {index}");
    }
    if let Some(live) = stock("lua5.4", ALTERNATING) {
        assert_eq!(live.lines().count(), traced.lines().count());
        for (index, (want, got)) in live.lines().zip(traced.lines()).enumerate() {
            assert_eq!(want, got, "against lua5.4: line {index}");
        }
    }
    assert!(stats.traces_compiled > 0, "{stats:?}");
    assert!(stats.iterations > 1000, "native code ran: {stats:?}");
    // The root takes a number of either subtype: only the branch on a zero
    // divisor and the integers beyond 2^53 go through side traces.
    assert!(
        stats.side_entries * 10 < stats.iterations,
        "the subtype of a number still sends iterations through side traces: {stats:?}"
    );
}

/// One modulo site whose operands alternate between integers and floats
/// every iteration, with no branch on the data and no zero or huge operand,
/// so nothing but the subtype of a number could send an iteration anywhere
/// else than the root trace.
const ALTERNATING_STRAIGHT: &str = r#"
local function show(r)
  local kind = math.type(r) or type(r)
  if r ~= r then return kind .. ":nan" end
  if kind == "integer" then return "i:" .. tostring(r) end
  return kind .. ":" .. string.format("%.17g", r)
end
local operands = {7, 2.5, -7, -2.5, 13, 0.25, 3, 1.5, -1, 0.5, 5, -3.0}
local count = #operands
local rounds = 60000
local out = {}
for i = 1, rounds do out[i] = 0 end
local acc = 0.0
for i = 1, rounds do
  local v = operands[i % count + 1]
  local w = operands[(i * 5 + i // count) % count + 1]
  out[i] = v % w
  acc = acc + v % w
end
local lines = {}
for i = 1, rounds do lines[i] = show(out[i]) end
return table.concat(lines, "\n") .. "\n" .. show(acc)
"#;

#[test]
fn a_site_that_alternates_integers_and_floats_stays_in_the_root_trace() {
    let (interpreted, _) = run_source(ALTERNATING_STRAIGHT, ExecutionMode::Interpreter);
    let (traced, stats) = run_source(ALTERNATING_STRAIGHT, ExecutionMode::Tracing);
    assert_eq!(interpreted, traced);
    if let Some(live) = stock("lua5.4", ALTERNATING_STRAIGHT) {
        assert_eq!(live, traced, "against lua5.4");
    }
    assert!(stats.iterations > 50_000, "native code ran: {stats:?}");
    assert!(
        stats.side_entries * 100 < stats.iterations,
        "the subtype of a number still sends iterations through side traces: {stats:?}"
    );
}

/// Numbers of either subtype that reach many kinds of instruction: a float
/// operand, a branch on a comparison, a copy, a value that is stored, called,
/// divided by and floor-divided, and a modulo whose result feeds more
/// arithmetic. An integer beyond 2^53 leaves the trace at its load. The first instructions that read them take a
/// number of either subtype, the others narrow it to the subtype it has.
const MIXED_STATEMENTS: &str = r#"
local function show(r)
  local kind = math.type(r) or type(r)
  if r ~= r then return kind .. ":nan" end
  if kind == "integer" then return "i" .. tostring(r) end
  if kind == "float" then return "f" .. string.format("%.17g", r) end
  return kind .. ":" .. tostring(r)
end
local operands = {
  7, -2.5, 3, 1e15, -64, 0.25, 9007199254740993, 5, 1.5, 2, -1, 0.0, 4294967296,
  -0.0, 13, 123456789.75, 9007199254740992, -3.0, 1000, 0.5, 1e300, 8, 5e-324,
}
local n = #operands
local rounds = 40000
local r1, r2, r3, r4, r5, r6, r7, r8 = {}, {}, {}, {}, {}, {}, {}, {}
for i = 1, rounds do
  r1[i], r2[i], r3[i], r4[i], r5[i], r6[i], r7[i], r8[i] = 0, 0, 0, 0, 0, 0, 0, 0
end
local total = 0.0
local function run()
  for i = 1, rounds do
    local v = operands[(i * 7 + 3) % n + 1]
    local w = operands[(i * 5 + 2 + i // n) % n + 1]
    if w == 0 then w = 3 end
    local other = v
    if other == 0 then other = 5 end
    r1[i] = v * 1.0 % w
    if v < w then r2[i] = v % w else r2[i] = w % other end
    r3[i] = (v % w) * 2
    local x = v % w
    r4[i] = x * 3 - w
    r5[i] = v // w
    r6[i] = tostring(v % w)
    r7[i] = math.max(v, w)
    r8[i] = v / w
    total = total + v % w
  end
end
local ok, err = pcall(run)
local out = {tostring(ok), (tostring(err):gsub("^.-:%d+: ", "")), show(total)}
for _, column in ipairs({r1, r2, r3, r4, r5, r6, r7, r8}) do
  for i = 1, rounds do out[#out + 1] = show(column[i]) end
end
return table.concat(out, "\n")
"#;

#[test]
fn numbers_of_either_subtype_meet_every_kind_of_instruction_in_traces() {
    let (interpreted, _) = run_source(MIXED_STATEMENTS, ExecutionMode::Interpreter);
    let (traced, stats) = run_source(MIXED_STATEMENTS, ExecutionMode::Tracing);
    assert_eq!(interpreted.lines().count(), traced.lines().count());
    for (index, (want, got)) in interpreted.lines().zip(traced.lines()).enumerate() {
        assert_eq!(want, got, "line {index}");
    }
    if let Some(live) = stock("lua5.4", MIXED_STATEMENTS) {
        assert_eq!(live.lines().count(), traced.lines().count());
        for (index, (want, got)) in live.lines().zip(traced.lines()).enumerate() {
            assert_eq!(want, got, "against lua5.4: line {index}");
        }
    }
    assert!(stats.iterations > 1000, "native code ran: {stats:?}");
}
