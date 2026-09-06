//! Groups of `string.byte` reads of one string in a trace: positions that
//! are one number plus small constants (`string.byte(s, i)` next to
//! `string.byte(s, i + 3)`, the bytes of a `string.sub` the trace never
//! makes, a byte read twice).
//!
//! The compiled code checks the number once and each further position with
//! one compare or none (`mir/lower/bytes.rs`). Every source runs in both
//! modes of every profile and has to give the same results, for positions
//! inside the string and for every position that has to leave the compiled
//! read: past either end, negative, fractional, not a number, too big for an
//! integer, and a number just below a whole one whose sum with a constant
//! rounds up to it.

use std::fmt::Write as _;

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

/// Both modes agree, and at least `iterations` loop iterations completed in
/// compiled code.
fn check(source: &str, iterations: u64) {
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
        assert!(
            stats.iterations >= iterations,
            "{profile:?} {source}: {} iterations {stats:?}",
            stats.iterations
        );
    }
}

/// A program that runs `group` (statements over the strings `s` and `short`
/// and the position `i` that set the number `v`) for every position of a
/// list in one loop: first hot over positions inside both strings, then once
/// for each position of `outside` among positions that are inside. A run
/// that raises an error gives the error's text. With `float` every position
/// is a float, also in a profile that has integers, where a whole float is
/// a position too.
fn over_positions(group: &str, outside: &[&str], float: bool) -> String {
    let plus = if float { "+ 0.0" } else { "" };
    let mut source = format!(
        "local byte, sub = string.byte, string.sub
         local s = string.rep('0123456789abcdef', 4)
         local short = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
         local function run(positions)
             local total = 0
             for j = 1, #positions do
                 local i = positions[j]
                 {group}
                 total = total + v
             end
             return total
         end
         local inside = {{}}
         for j = 1, 40 do inside[j] = j + 4 {plus} end
         local results = {{}}
         for _ = 1, 300 do results[1] = run(inside) end
         local function try(positions)
             local ok, value = pcall(run, positions)
             results[#results + 1] = ok and value or tostring(value)
         end
         "
    );
    for position in outside {
        source.push_str("try({ ");
        for at in 5..13 {
            write!(source, "{at} {plus}, ").unwrap();
        }
        writeln!(source, "({position}) {plus}, 13 {plus}, 14 {plus} }})").unwrap();
    }
    source.push_str("return table.concat(results, '|')");
    source
}

/// Runs `group` over integer positions and over float ones.
fn check_group(group: &str) {
    for float in [false, true] {
        check(&over_positions(group, &OUTSIDE, float), 10_000);
    }
}

/// Positions that leave the compiled read.
const OUTSIDE: [&str; 20] = [
    "49",
    "50",
    "52",
    "53",
    "61",
    "62",
    "64",
    "65",
    "1",
    "0",
    "-1",
    "-64",
    "-70",
    "5.5",
    "0.5",
    "1 - 2^-53",
    "2^53",
    "-2^63",
    "1e300",
    "0/0",
];

#[test]
fn a_group_of_reads_at_a_position_and_the_ones_after_it() {
    let group = "local v = (byte(s, i) or -1) + (byte(s, i + 1) or -2) + (byte(s, i + 2) or -3)
                     + (byte(s, i + 3) or -4) + (byte(s, i) or -5)";
    check_group(group);
}

#[test]
fn a_group_of_reads_on_both_sides_of_a_position() {
    // The highest and the lowest are checked first, so the ones between
    // them need no check of their own.
    let group = "local v = (byte(s, i + 2) or -1) + (byte(s, i - 2) or -2) + (byte(s, i) or -3)
                     + (byte(s, i + 1) or -4) + (byte(s, i - 1) or -5) + (byte(s, i - 3) or -6)";
    check_group(group);
}

#[test]
fn the_bytes_of_a_substring_and_a_read_of_its_source() {
    // `string/byte_sub`: four bytes of a substring the trace does not make,
    // and the first of them again from the source.
    let group = "local slice = sub(s, i, i + 3)
                 local a, b, c, d = byte(slice, 1, 4)
                 local v = (a or -1) + (b or -2) + (c or -3) + (d or -4) + (byte(s, i) or -5)";
    check_group(group);
    // The recorder takes a range of `string.byte` only with positions of
    // the profile's number type, so the float form of a profile with
    // integers runs in the interpreter: its answers still have to agree.
    let group = "local a, b, c, d = byte(s, i, i + 3)
                 local v = (a or -1) + (b or -2) + (c or -3) + (d or -4)";
    check(&over_positions(group, &OUTSIDE, false), 10_000);
    check(&over_positions(group, &OUTSIDE, true), 0);
}

#[test]
fn positions_that_are_floats_and_sums_of_sums() {
    // A float position in every profile, and a position two sums away from
    // the number the group shares.
    let group = "local f = i * 1.0
                 local after = f + 1
                 local v = (byte(s, f) or -1) + (byte(s, after + 1) or -2) + (byte(s, after) or -3)";
    check_group(group);
}

#[test]
fn two_strings_read_at_one_position() {
    // What the checks of one string prove says nothing about a shorter one.
    let group = "local v = (byte(s, i) or -1) + (byte(s, i + 2) or -2)
                     + (byte(short, i + 1) or -3) + (byte(short, i) or -4) + (byte(short, i + 2) or -5)
                     + (byte(s, i + 1) or -6) + (byte(short, i + 1) or -7)";
    check_group(group);
}
