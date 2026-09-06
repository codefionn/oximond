//! `x == "name"` in a trace, where `x` is a string borrowed from a table:
//! native code compares the words of the string's storage against those of
//! the constant (`string_equal_constant` in `jit/trace/emit.rs`) instead of
//! calling the helper, under both backends. The answers must be the
//! interpreter's for every kind of pair: equal bytes in another allocation,
//! the same length with other bytes, other lengths, zero bytes inside the
//! string, the empty string, a string of exactly the longest inline length,
//! long strings that share a short constant's start, and long constants,
//! which the helper still compares.

use std::process::Command;

/// The strings the loop reads, built at run time so that none is the
/// constant's own allocation, and the constants it compares them with.
const SOURCE: &str = r#"
local function make(text) return (text .. "!"):sub(1, -2) end
local long22 = "abcdefghijklmnopqrstuv"
local long23 = "abcdefghijklmnopqrstuvw"
local t = {
  make("stone"), make("stona"), make("ston"), make("stones"), make("air"),
  make("a\0"), make("a"), make(""), make(long22), make("abcdefghijklmnopqrstuX"),
  make(long23), make("abcdefghijklmnopqrstuvX"), make("stone" .. string.rep("x", 30)),
  make("mcl_core:stone_with_coal"), make("mcl_core:stone_with_iron"),
}
local counts = {}
for k = 1, 12 do counts[k] = 0 end
for i = 1, 60000 do
  local s = t[i % #t + 1]
  if s == "stone" then counts[1] = counts[1] + 1 end
  if s ~= "air" then counts[2] = counts[2] + 1 end
  if s == "a" then counts[3] = counts[3] + 1 end
  if s == "a\0" then counts[4] = counts[4] + 1 end
  if s == "" then counts[5] = counts[5] + 1 end
  if s == "abcdefghijklmnopqrstuv" then counts[6] = counts[6] + 1 end
  if s == "abcdefghijklmnopqrstuvw" then counts[7] = counts[7] + 1 end
  if s == "mcl_core:stone_with_coal" then counts[8] = counts[8] + 1 end
  if "stone" == s then counts[9] = counts[9] + 1 end
  if s == "ston" then counts[10] = counts[10] + 1 end
  if s == "stones" then counts[11] = counts[11] + 1 end
  if s == "abcdefghijklmnopqrstuX" then counts[12] = counts[12] + 1 end
end
print(table.concat(counts, " "))
"#;

/// Comparisons where neither string is a constant the compile knows: the
/// elements of an array against a string argument the loop holds (a scan),
/// two strings of table fields, two results of `string.sub`, and a string
/// a loop carries from one constant to the other. Native code reads the
/// lanes of the run's bank and of the trace's roots and compares the words,
/// and the helper compares two long strings in two allocations.
const LANES_SOURCE: &str = r#"
local function make(text) return (text .. "!"):sub(1, -2) end
local long = string.rep("x", 30)
local t = {
  make("stone"), make("air"), make(long), make("a\0"), make(""),
  make("abcdefghijklmnopqrstuv"), make(long .. "y"), long, make("a"),
}
local wanted = {
  make("stone"), make("air"), make(long), make("abcdefghijklmnopqrstuv"), make(""),
  long, make("ston"), make("a"), make("a\0"),
}
local function scan(val)
  local n = 0
  for i = 1, #t do if t[i] == val then n = n + 1 end end
  return n
end
local counts = {}
for round = 1, 2000 do
  for w = 1, #wanted do counts[w] = (counts[w] or 0) + scan(wanted[w]) end
end
local records = {}
for i = 1, 13 do
  records[i] = { name = t[i % #t + 1], other = wanted[(i * 7) % #wanted + 1] }
end
local same, prefix = 0, 0
for i = 1, 60000 do
  local r = records[i % #records + 1]
  if r.name == r.other then same = same + 1 end
  if r.name:sub(1, 3) == r.other:sub(1, 3) then prefix = prefix + 1 end
end
local key, hits = "stone", 0
for i = 1, 60000 do
  if t[i % #t + 1] == key then hits = hits + 1 end
  if i % 1000 == 0 then key = (key == "stone") and "air" or "stone" end
end
print(table.concat(counts, " "), same, prefix, hits)
"#;

fn run(compat: &str, execution: &str, backend: Option<&str>) -> String {
    run_source(SOURCE, compat, execution, backend)
}

fn run_source(source: &str, compat: &str, execution: &str, backend: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .args([
            &format!("--compat={compat}"),
            &format!("--execution={execution}"),
            "-e",
            source,
        ])
        .env("OXIMOND_JIT_SYNC", "1")
        .env_remove("OXIMOND_JIT_BACKEND");
    if let Some(backend) = backend {
        command.env("OXIMOND_JIT_BACKEND", backend);
    }
    let output = command.output().expect("run oximond");
    assert!(
        output.status.success(),
        "{compat} {execution} {backend:?}: {output:?}"
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn borrowed_strings_compare_with_constants_as_the_interpreter_compares_them() {
    for compat in ["lua51", "lua54", "luajit21"] {
        let interpreted = run(compat, "interpreter", None);
        assert_eq!(
            interpreted.trim(),
            "4000 56000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000",
            "{compat}: the interpreter's counts"
        );
        for backend in ["cranelift", "mir"] {
            assert_eq!(
                run(compat, "tracing", Some(backend)),
                interpreted,
                "{compat} under {backend}"
            );
        }
    }
}

#[test]
fn strings_of_lanes_and_roots_compare_as_the_interpreter_compares_them() {
    for compat in ["lua51", "lua54", "luajit21"] {
        let interpreted = run_source(LANES_SOURCE, compat, "interpreter", None);
        assert!(!interpreted.trim().is_empty(), "{compat}: it prints");
        for backend in ["cranelift", "mir"] {
            assert_eq!(
                run_source(LANES_SOURCE, compat, "tracing", Some(backend)),
                interpreted,
                "{compat} under {backend}"
            );
        }
    }
}
