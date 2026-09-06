//! `#(a .. b)` of a concatenation nothing else reads is the sum of the
//! operands' lengths: the trace makes no string. The length must be what the
//! interpreter's concatenation gives, for every spelling a number can have in
//! each profile, and a concatenation read in any other way still makes its
//! string.
//!
//! Every program runs under the interpreter and under tracing execution in
//! every profile, and the results must match.

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

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (String, TraceStatistics) {
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
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` in every profile under both engines, checks the results
/// agree, and returns the interpreter's result per profile.
fn agree(source: &str) -> Vec<(CompatibilityProfile, String)> {
    ALL_PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
            (profile, expected)
        })
        .collect()
}

/// [`agree`] for a program whose hot loop must run compiled code.
fn agree_traced(source: &str) -> Vec<(CompatibilityProfile, String)> {
    for &profile in &ALL_PROFILES {
        let stats = run(profile, ExecutionMode::Tracing, source).1;
        assert!(stats.iterations > 0, "{profile:?} ran no trace\n{stats:?}");
    }
    agree(source)
}

/// The lengths of `s .. v` for every value of `values` (a Lua table
/// constructor), summed over many rounds so that the loop compiles, with the
/// separate lengths folded into a checksum that tells every case apart.
fn length_source(values: &str) -> String {
    format!(
        "local values = {values}
         local total = 0
         for round = 1, 40 do
           for i = 1, #values do
             local v = values[i]
             total = total + #('ab' .. v) * i + #(v .. 'xyz' .. v) * 7
           end
         end
         return total"
    )
}

#[test]
fn the_length_of_a_number_concatenation_matches_the_interpreter() {
    for values in [
        "{0, 1, 7, 10, 99, 100, 12345, 1000000}",
        "{-1, -10, -123456, -0.5, -1e100}",
        "{0.5, 1.25, 3.14159265358979, 1/3, 2/3, 1e15, 1e14, 123456789012345}",
        "{1e100, 1e300, 5e-324, 2^53, 2^63, -2^63, 1e-5, 123.456e10}",
        "{1/0, -1/0, 0/0}",
        "{'', 'a', 'hello', 'a longer string of more than twenty-two bytes'}",
    ] {
        agree(&length_source(values));
    }
}

#[test]
fn the_length_of_an_integer_concatenation_matches_the_interpreter() {
    // Integers apart from floats in the profiles that have both.
    for values in [
        "{0, 1, -1, 9, 10, 2^31, 1 << 40, math.maxinteger, math.mininteger}",
        "{3, 3.0, -3.0, 1e15, 2^53}",
    ] {
        let source = length_source(values);
        for &profile in &[
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ] {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (traced, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(traced, expected, "{profile:?}\n{source}\n{stats:?}");
        }
    }
}

#[test]
fn a_concatenation_of_a_loop_counter_matches_the_interpreter() {
    agree_traced(
        "local total = 0
         for i = -300, 3000 do
           total = total + #('n' .. i) + #(i .. i) + #(i / 7 .. '|') + #(-i * 1.5 .. 'x' .. i)
         end
         return total",
    );
}

#[test]
fn a_concatenation_used_besides_its_length_still_makes_its_string() {
    let results = agree(
        "local keep, total = nil, 0
         for i = 1, 500 do
           local s = 'k' .. i
           total = total + #s
           keep = s
         end
         local pair = {}
         for i = 1, 500 do
           local s = 'p' .. (i / 4)
           pair[i % 3 + 1] = s
           total = total + #s
         end
         return total, keep, pair[1], pair[2], pair[3]",
    );
    for (profile, result) in results {
        assert!(result.contains("k500"), "{profile:?}: {result}");
    }
}

#[test]
fn a_concatenation_whose_length_a_metamethod_handler_takes_matches_the_interpreter() {
    // The handler's concatenation is measured where the handler returns it.
    agree_traced(
        "local V = {}
         V.__concat = function(a, b)
           if type(a) == 'table' then return a.x .. ',' .. a.y .. b end
           return a .. b.x .. ',' .. b.y
         end
         V.__tostring = function(a) return '(' .. a.x .. ',' .. a.y .. ')' end
         local pool = {}
         for i = 1, 16 do pool[i] = setmetatable({x = i / 3, y = -i}, V) end
         local total = 0
         for i = 1, 2000 do
           local a = pool[i % 16 + 1]
           total = total + #('v=' .. a) + #(a .. ';') + #tostring(a)
         end
         return total",
    );
}

#[test]
fn the_collector_is_charged_for_a_concatenation_whose_string_is_never_made() {
    // A trace that sizes strings it never makes must still pace the
    // collector as the interpreter does: the count after the same work is
    // about the same, and a long enough run collects.
    let source = "local total = 0
         for i = 1, 200000 do total = total + #('a long prefix of text ' .. i .. ' and more') end
         collectgarbage()
         return total, collectgarbage('count') < 4096";
    for (profile, result) in agree(source) {
        assert!(result.contains("true"), "{profile:?}: {result}");
    }
}

#[test]
fn a_result_too_long_for_the_native_buffers_matches_the_interpreter() {
    agree(
        "local s = string.rep('x', 70000)
         local total = 0
         for i = 1, 300 do
           total = total + #(s .. s .. i) + #(i .. s)
         end
         return total",
    );
}

#[test]
fn a_concatenation_that_is_made_matches_the_interpreter_for_mixed_operands() {
    // Each result is kept, so the trace makes the string: integers, negative
    // numbers, fractions, huge, infinite and NaN floats, empty and long
    // strings, in every position and mixed in one chain.
    let results = agree_traced(
        "local out = {}
         local long = string.rep('ab', 40)
         local floats = {0.5, -2.25, 1e15, 1e100, 1/0, -1/0, 0/0, 3.0, -0.0}
         for i = 1, 600 do
           local f = floats[i % #floats + 1]
           out[#out + 1] = i .. '|' .. f
           out[#out + 1] = (-i) .. ''
           out[#out + 1] = f .. long .. i
           out[#out + 1] = '' .. i .. '' .. (i / 8)
         end
         local last = {}
         for i = 1, 9 do last[i] = out[#out - i] end
         return #out, table.concat(last, ';')",
    );
    for (profile, result) in results {
        assert!(result.contains("2400"), "{profile:?}: {result}");
    }
}
