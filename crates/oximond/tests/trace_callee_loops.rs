//! Loops in calls. An outer loop whose iteration calls a function with a
//! loop of its own (`indexof (list, value)`, an `ipairs` scan, reached
//! through a predicate the caller was handed) is recorded from the outer
//! header up to the callee loop's first backward branch, where it ends with
//! a terminal exit that rebuilds the call frames. The callee loop runs as a
//! loop side in those frames, and the sides of its exits return out of the
//! calls and link back to the outer header. The results are the
//! interpreter's in every profile, also where the callee loop leaves through
//! its guards: a hole, a value of another type, a metatable, an error.

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

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree, and returns the traced run's statistics by profile.
fn run_all(source: &str) -> Vec<(CompatibilityProfile, TraceStatistics)> {
    PROFILES
        .iter()
        .map(|&profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            (profile, lua.vm().jit_statistics().trace)
        })
        .collect()
}

/// The scan of `indexof` over `list`, called through `pred` from the `dx`
/// loop of `locate`, as Mineclonia's `locate_biome_in_area` calls
/// `is_preferred_biome` (biomes.lua:6980, structures.lua:30). `extra` runs
/// at the top of each outer iteration.
fn scan(extra: &str, list: &str, outer: u32) -> String {
    format!(
        "local function indexof(list, val)
             for i, v in ipairs(list) do
                 if v == val then return i end
             end
             return -1
         end
         local list = {list}
         local names = {{}}
         for i = 1, 64 do names[i] = 'biome_' .. (i * 7 % 90) end
         local function pred(name) return indexof(list, name) ~= -1 end
         local function locate(n, predicate)
             local hits, acc = 0, 0
             for k = 1, n do
                 {extra}
                 for dx = 1, 64 do
                     local name = names[dx]
                     acc = (acc * 31 + dx) % 1000003
                     if predicate(name) then hits = hits + 1 end
                 end
             end
             return hits, acc
         end
         return locate({outer}, pred)"
    )
}

/// Forty strings, in the array part.
const BIOMES: &str = "{} for i = 1, 40 do list[i] = 'biome_' .. i end";

#[test]
fn an_ipairs_scan_in_a_called_predicate_runs_as_a_loop_side() {
    let source = scan("", BIOMES, 600);
    for (profile, stats) in run_all(&source) {
        assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
        // The trips of the scan run natively: some 25 per call.
        assert!(stats.iterations > 600 * 64 * 10, "{profile:?} {stats:?}");
        // A match returns natively too: the side of the scan's `v == val`
        // starts at `return i`, where the borrowed `v` is dead, so the
        // tree is not entered again from the interpreter on every match
        // (some 17000 of the 38400 calls match).
        assert!(stats.native_entries < 600 * 64 / 8, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_scan_whose_list_changes_agrees_with_the_interpreter() {
    // A hole in the list, then a number in it, then a metatable that fills
    // the hole (`ipairs` runs `__index` from Lua 5.3 on), then a list of
    // other length, each for a while.
    let extra = "
        if k == 200 then list[10] = nil end
        if k == 260 then list[10] = 10 end
        if k == 320 then
            list[10] = nil
            setmetatable(list, {__index = function(_, i)
                if i == 10 then return 'biome_10' end
            end})
        end
        if k == 380 then setmetatable(list, nil) list[10] = 'x' list[41] = 'biome_41' end
        if k == 440 then for i = 20, 41 do list[i] = nil end end
    ";
    let source = scan(extra, BIOMES, 600);
    for (profile, stats) in run_all(&source) {
        assert!(stats.native_entries >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_scan_that_meets_an_error_raises_it_as_the_interpreter_does() {
    // From outer iteration 300 on the list holds a table, and the
    // comparison `v == val` is fine, but the caller then indexes the result.
    let source = scan(
        "if k == 300 then list[5] = {} end
         if k == 301 then local _ = list[5] .. 'x' end",
        BIOMES,
        400,
    );
    for (profile, _) in run_all(&source) {
        let (traced, _) = outcome(profile, ExecutionMode::Tracing, &source);
        assert!(traced.is_err(), "{profile:?} {traced:?}");
    }
}

#[test]
fn a_while_loop_in_a_called_function_runs_as_a_loop_side() {
    // A `while` scan in a call, which the outer root cannot unroll.
    let source = "
        local function count_below(t, limit)
            local i = 1
            while t[i] and t[i] < limit do i = i + 1 end
            return i - 1
        end
        local t = {}
        for i = 1, 60 do t[i] = i * 3 end
        local function run(n)
            local acc = 0
            for k = 1, n do
                acc = acc + count_below(t, 30 + k % 120)
            end
            return acc
        end
        return run(30000)
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
        assert!(stats.iterations > 30000, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_scan_that_writes_its_own_list_agrees_with_the_interpreter() {
    // The scan's body changes the list it walks while the outer loop runs
    // natively: a string slot becomes a number, then nil (the end of the
    // walk), then `table.remove` shifts the rest down under the walk. The
    // list is rebuilt with fresh strings at the top of each outer trip.
    let source = "
        local mode = 0
        local function indexof(list, val)
            for i, v in ipairs(list) do
                if v == val then return i end
                if mode == 1 and i == 12 then list[12] = 12 end
                if mode == 2 and i == 15 then list[15] = nil end
                if mode == 3 and i == 20 then table.remove(list, 20) end
            end
            return -1
        end
        local list = {}
        local function fill()
            for j = 1, 40 do list[j] = 'biome_' .. j end
            for j = 41, 45 do list[j] = nil end
        end
        fill()
        local names = {}
        for i = 1, 64 do names[i] = 'biome_' .. (i * 7 % 90) end
        local function pred(name) return indexof(list, name) ~= -1 end
        local hits, acc = 0, 0
        for k = 1, 500 do
            if k == 150 then mode = 1 elseif k == 230 then mode = 2
            elseif k == 310 then mode = 3 elseif k == 390 then mode = 0 end
            if mode ~= 0 then fill() end
            for dx = 1, 64 do
                local name = names[dx]
                acc = (acc * 31 + dx) % 1000003
                if pred(name) then hits = hits + 1 end
            end
        end
        local shape = {}
        for j = 1, 41 do shape[j] = type(list[j]) end
        return hits, acc, #list, table.concat(shape, ',')
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_scan_element_captured_by_a_closure_survives_a_collection() {
    // On some outer trips a match hands the borrowed element `v` to
    // `remember`, which captures it in a closure kept outside the loop. The
    // recorder takes neither `Closure` nor a loop local captured in its
    // body, so `remember` makes the closure and runs in the interpreter,
    // from the state the native scan's exit left. The list then drops its
    // strings and the collector runs: the closures alone hold them, and
    // must still give them back.
    let source = "
        local keep, kept, capture = {}, 0, false
        local function remember(x)
            kept = kept % 16 + 1
            keep[kept] = function() return x end
        end
        local function indexof(list, val)
            for i, v in ipairs(list) do
                if v == val then
                    if capture then remember(v) end
                    return i
                end
            end
            return -1
        end
        local list = {}
        local function fill()
            for j = 1, 40 do list[j] = 'biome_' .. j end
        end
        fill()
        local names = {}
        for i = 1, 64 do names[i] = 'biome_' .. (i * 7 % 90) end
        local function pred(name) return indexof(list, name) ~= -1 end
        local hits = 0
        for k = 1, 400 do
            capture = k % 50 == 49
            if k % 50 == 0 then
                for j = 1, 40 do list[j] = 'other_' .. j end
                collectgarbage('collect')
                fill()
            end
            for dx = 1, 64 do
                if pred(names[dx]) then hits = hits + 1 end
            end
        end
        for j = 1, 40 do list[j] = j end
        for i = 1, 64 do names[i] = i end
        collectgarbage('collect')
        collectgarbage('collect')
        local seen = {}
        for j = 1, 16 do seen[j] = keep[j] and keep[j]() or '-' end
        return hits, table.concat(seen, ',')
    ";
    for (profile, stats) in run_all(source) {
        assert!(stats.loop_sides >= 1, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_close_handler_sees_the_locals_of_a_scan_that_matched() {
    // `indexof` holds a to-be-closed local whose `__close` runs in the
    // `return i` of a match and reads the returning frame's `v` with
    // `debug.getlocal`. A side that starts at that return must not hold nil
    // in `v` (`SeedRegister::Dead`): the handler sees the matched string.
    // Today the recorder stops at `MarkClose` and `CloseLocal`, so these
    // calls stay interpreted; `seed_frames_of` also keeps every register of
    // a function with a to-be-closed local, for a recorder that takes them.
    let source = "
        local hits, last, sum = 0, nil, 0
        local closer = setmetatable({}, {__close = function()
            local i = 1
            while true do
                local name, value = debug.getlocal(2, i)
                if not name then break end
                if name == 'v' then
                    hits = hits + 1
                    last = value
                    sum = sum + #value
                end
                i = i + 1
            end
        end})
        local function indexof(list, val)
            local guard <close> = closer
            for i, v in ipairs(list) do
                if v == val then return i end
            end
            return -1
        end
        local list = {}
        for i = 1, 40 do list[i] = 'biome_' .. i end
        local names = {}
        for i = 1, 64 do names[i] = 'biome_' .. (i * 7 % 90) end
        local function pred(name) return indexof(list, name) ~= -1 end
        local found = 0
        for k = 1, 300 do
            for dx = 1, 64 do
                if pred(names[dx]) then found = found + 1 end
            end
        end
        return found, hits, last, sum
    ";
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        assert_eq!(actual, expected, "{profile:?}");
    }
}
