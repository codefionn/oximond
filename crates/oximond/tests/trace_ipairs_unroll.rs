//! Short generic for loops inside a call the recording inlines. Mineclonia's
//! `place_structures_in_chunk` (`structures.lua:977`) walks the structures
//! of a generation step with `ipairs`, and the starts of a chunk inside
//! that, for every chunk `finish_structures` (`structures.lua:1014`) visits.
//! A loop around such a call was turned away at the latch of the generic
//! for, and the `ipairs` step could not read a table the trace had read out
//! of another (`ipairs (lists[i])`). Now the recording that unrolls short
//! loops follows the latch like the jump of a `while` loop, a table met by
//! lane is read through its lane, and the step that ends the loop is a
//! guard (`TABLE_ABSENT_INDEX` for a table met by lane): a list of another
//! length leaves there and takes a side trace. Every program runs traced
//! and interpreted in every profile and must give the same results: lists
//! of changing length, an empty list, a `break`, a hole, a metatable that
//! appears, a list that outgrows the unroll limit, an element of another
//! type, an error inside the loop.

use oximond::runtime::Value;
use oximond::vm::TraceAbortReason;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
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

struct Seen {
    profile: CompatibilityProfile,
    compiled: u64,
    iterations: u64,
    inner_loop_aborts: u64,
    ok: bool,
    stats: String,
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. Returns what each traced run did.
fn run_all(source: &str) -> Vec<Seen> {
    PROFILES
        .into_iter()
        .map(|profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{profile:?} {source}"
            );
            let stats = lua.vm().jit_statistics().trace;
            Seen {
                profile,
                compiled: stats.traces_compiled,
                iterations: stats.iterations,
                inner_loop_aborts: stats.recording_aborts[TraceAbortReason::InnerLoop.index()],
                ok: actual.is_ok(),
                stats: format!("{stats:?}"),
            }
        })
        .collect()
}

/// [`run_all`], and the loop around the call must have run native code for
/// at least `native` of its iterations.
fn assert_traced(source: &str, native: u64) {
    for seen in run_all(source) {
        let profile = seen.profile;
        assert!(seen.ok, "{profile:?} {}", seen.stats);
        assert!(seen.compiled >= 1, "{profile:?} {}", seen.stats);
        assert!(
            seen.iterations >= native,
            "{profile:?}: {} iterations ran native: {}",
            seen.iterations,
            seen.stats
        );
    }
}

/// A function that sums a list with `ipairs`, and lists of the lengths
/// `lengths`, which a loop of 20000 iterations hands it in turn.
fn sums(lengths: &[usize], body: &str) -> String {
    let lists = lengths
        .iter()
        .map(|&n| {
            let items = (1..=n).map(|k| k.to_string()).collect::<Vec<_>>();
            format!("{{{}}}", items.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "local function sum(t)
             local s = 0
             for _, v in ipairs(t) do {body} end
             return s
         end
         local lists = {{ {lists} }}
         local total = 0
         for i = 1, 20000 do
             total = total + sum(lists[i % #lists + 1])
         end
         return total"
    )
}

#[test]
fn a_short_ipairs_loop_in_a_call_is_unrolled_into_the_loop_around_it() {
    assert_traced(&sums(&[3], "s = s + v"), 18_000);
}

#[test]
fn ipairs_loops_of_other_lengths_take_side_traces() {
    // Lists of two, one, three and no elements, a list read out of another
    // table on each iteration.
    let source = sums(&[2, 1, 3, 0], "s = s + v");
    for seen in run_all(&source) {
        let profile = seen.profile;
        assert!(seen.compiled >= 1, "{profile:?} {}", seen.stats);
        assert!(seen.iterations >= 15_000, "{profile:?} {}", seen.stats);
        // The loop around the call is no longer turned away for meeting it.
        assert!(seen.inner_loop_aborts <= 2, "{profile:?} {}", seen.stats);
    }
}

#[test]
fn an_ipairs_loop_left_by_break_is_unrolled() {
    assert_traced(&sums(&[4, 2], "if v > 2 then break end s = s + v"), 15_000);
}

#[test]
fn an_ipairs_loop_past_the_unroll_limit_agrees() {
    // Too many trips to unroll: the loop keeps a trace of its own.
    for seen in run_all(&sums(&[40, 3], "s = s + v")) {
        assert!(seen.compiled >= 1, "{:?} {}", seen.profile, seen.stats);
    }
}

#[test]
fn nested_ipairs_loops_in_a_call_agree_and_run_natively() {
    // The shape of `place_structures_in_chunk`: the structures of a step,
    // and for each the starts of the chunk that belong to it.
    let source = "local function place(structures, starts)
             local n = 0
             for j, sid in ipairs(structures) do
                 for _, start in ipairs(starts) do
                     if start.structure == sid then n = n + j * start.size end
                 end
             end
             return n
         end
         local structures = { 'a', 'b' }
         local refs = {}
         for k = 0, 4 do
             refs[k] = {}
             if k % 2 == 0 then refs[k][1] = { structure = 'b', size = k } end
         end
         local t = 0
         for i = 1, 20000 do t = t + place(structures, refs[i % 5]) end
         return t";
    assert_traced(source, 5_000);
}

#[test]
fn a_hole_ends_the_unrolled_loop_as_the_interpreter_does() {
    let source = "local function sum(t)
             local s = 0
             for k, v in ipairs(t) do s = s + v * k end
             return s
         end
         local lists = { {1, 2, nil, 4}, {5, nil, 7} }
         local t = 0
         for i = 1, 20000 do t = t + sum(lists[i % 2 + 1]) end
         return t";
    assert_traced(source, 15_000);
}

#[test]
fn a_metatable_that_appears_on_the_list_leaves_the_trace() {
    // From iteration 10000 on, the list has an `__index` that answers past
    // its end (Lua 5.3 and later ask it; 5.1 and LuaJIT read raw).
    let source = "local function sum(t)
             local s = 0
             for _, v in ipairs(t) do s = s + v end
             return s
         end
         local lists = { {1, 2}, {3} }
         local t = 0
         for i = 1, 20000 do
             if i == 10000 then
                 setmetatable(lists[2], { __index = function(_, k)
                     if k < 4 then return k * 10 end
                 end })
             end
             t = t + sum(lists[i % 2 + 1])
         end
         return t";
    for seen in run_all(source) {
        assert!(seen.ok, "{:?} {}", seen.profile, seen.stats);
        assert!(
            seen.iterations >= 5_000,
            "{:?} {}",
            seen.profile,
            seen.stats
        );
    }
}

#[test]
fn an_element_of_another_type_leaves_the_trace() {
    let source = "local function sum(t)
             local s = 0
             for _, v in ipairs(t) do s = s + v end
             return s
         end
         local lists = { {1, 2}, {3, 4} }
         local t = 0
         for i = 1, 20000 do
             if i == 12000 then lists[2][2] = 4.5 end
             t = t + sum(lists[i % 2 + 1])
         end
         return t";
    for seen in run_all(source) {
        assert!(seen.ok, "{:?} {}", seen.profile, seen.stats);
        assert!(
            seen.iterations >= 5_000,
            "{:?} {}",
            seen.profile,
            seen.stats
        );
    }
}

#[test]
fn an_error_inside_the_unrolled_loop_is_the_interpreters() {
    let source = "local function sum(t)
             local s = 0
             for _, v in ipairs(t) do s = s + v end
             return s
         end
         local lists = { {1, 2}, {3} }
         local t = 0
         for i = 1, 20000 do
             if i == 15000 then lists[1][2] = {} end
             t = t + sum(lists[i % 2 + 1])
         end
         return t";
    for seen in run_all(source) {
        assert!(!seen.ok, "{:?} {}", seen.profile, seen.stats);
    }
}
