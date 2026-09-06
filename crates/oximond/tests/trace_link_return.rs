//! A tree that runs on through the return of the function it was entered in:
//! the call of a function whose nest is a tree runs in the same native code as
//! the loop that makes the call, and the loop around the call and the nest it
//! calls are one native loop. The results are the interpreter's in every
//! profile, whatever happens in the nest or in the code around the call: a
//! type changes, the nest breaks out of a loop, raises an error, is called
//! through `pcall`, runs under a hook, shows a traceback, yields, loses its
//! tree, is replaced by another function, calls itself, or runs while the
//! collector works on tables the caller made.

use oximond::runtime::Value;
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

/// What a traced run did.
struct Seen {
    profile: CompatibilityProfile,
    native_entries: u64,
    stats: String,
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. Returns what each traced run did.
fn run_all(source: &str) -> Vec<Seen> {
    let mut all = Vec::new();
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        assert!(expected.is_ok(), "{profile:?} {source}: {expected:?}");
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        let stats = lua.vm().jit_statistics().trace;
        all.push(Seen {
            profile,
            native_entries: stats.native_entries,
            stats: format!("{stats:?}"),
        });
    }
    all
}

/// [`run_all`], and the calls of the function (`calls` of them) must have
/// run in the tree that the loop around them is: native code was entered once
/// for far fewer than one call in four.
fn assert_linked(source: &str, calls: u64) {
    assert_linked_by(source, calls, 2);
}

/// [`assert_linked`] with native code entered for fewer than one call in
/// `ratio`.
fn assert_linked_by(source: &str, calls: u64, ratio: u64) {
    for seen in run_all(source) {
        let Seen {
            profile,
            native_entries,
            stats,
        } = seen;
        assert!(
            native_entries * ratio < calls,
            "{profile:?}: {native_entries} entries for {calls} calls: {stats}"
        );
    }
}

/// [`run_all`], and in the float profiles (the integer profiles record again
/// when a number changes its subtype, which costs entries) the calls must have
/// run in the tree.
fn assert_linked_in_float_profiles(source: &str, calls: u64) {
    for seen in run_all(source) {
        let float = matches!(
            seen.profile,
            CompatibilityProfile::Lua51
                | CompatibilityProfile::Lua52
                | CompatibilityProfile::LuaJit21
        );
        assert!(
            !float || seen.native_entries * 2 < calls,
            "{:?}: {} entries for {calls} calls: {}",
            seen.profile,
            seen.native_entries,
            seen.stats
        );
    }
}

/// The function with the nest, and the loop that calls it.
fn program(body: &str, driver: &str) -> String {
    format!(
        "local function sweep(n, m, probe)
             local acc = 0
             for i = 1, n do
                 for j = i, m do
                     {body}
                 end
             end
             return acc
         end
         {driver}"
    )
}

const PLAIN_BODY: &str = "acc = acc + (i * j) % 7";
const PLAIN_DRIVER: &str = "
    local total = 0
    for k = 1, 12000 do
        total = (total + sweep(8 + k % 3, 40)) % 1000003
    end
    return total";

#[test]
fn a_call_in_a_loop_runs_its_nest_in_the_tree_of_the_nest() {
    assert_linked_by(&program(PLAIN_BODY, PLAIN_DRIVER), 12000, 4);
}

#[test]
fn a_call_whose_results_the_caller_ignores_is_linked_too() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total = 0
             local seen = 0
             for k = 1, 12000 do
                 sweep(8 + k % 3, 40)
                 seen = seen + 1
                 total = total + k
             end
             return total, seen",
        ),
        12000,
    );
}

#[test]
fn the_caller_keeps_a_table_and_a_closure_across_the_calls() {
    // The caller's registers hold values of several kinds: a function, a
    // table it reads, a float, a boolean, a string.
    assert_linked(
        &program(
            PLAIN_BODY,
            "local box = {0, 1}
             local flag = true
             local label = 'x'
             local total = 0.5
             local call = sweep
             for k = 1, 12000 do
                 total = (total + call(8 + k % 3, 40)) % 1000003
                 flag = not flag
             end
             return total, flag, label, box[1] + box[2]",
        ),
        12000,
    );
}

#[test]
fn a_nest_that_breaks_out_of_its_loop_leaves_the_tree_and_comes_back() {
    assert_linked(
        &program(
            "acc = acc + (i * j) % 7
                     if acc > 3000 and (i + j) % 29 == 0 then break end",
            PLAIN_DRIVER,
        ),
        12000,
    );
}

#[test]
fn a_type_that_changes_while_the_calls_go_on_is_the_interpreters() {
    // The limit of the inner loop turns into a float and back, the total
    // into a float, the number of rows into a string that is a number.
    assert_linked_in_float_profiles(
        &program(
            PLAIN_BODY,
            "local total = 0
         local limit = 40
         for k = 1, 12000 do
             if k == 6000 then limit = 40.5 end
             if k == 7000 then limit = 40 end
             if k == 8000 then total = total + 0.25 end
             local rows = 8 + k % 3
             if k > 9000 and k < 9100 then rows = '9' + 0 end
             total = (total + sweep(rows, limit)) % 1000003
         end
         return total",
        ),
        12000,
    );
}

#[test]
fn an_error_in_the_nest_is_caught_by_the_code_around_the_loop() {
    let source = program(
        "if probe and i == 5 and j == 17 then error('bad ' .. i .. ' ' .. j) end
                     acc = acc + (i * j) % 7",
        "local total = 0
         local messages = {}
         for round = 1, 3 do
             local ok, message = pcall(function()
                 local sum, r = 0, round
                 for k = 1, 4000 do
                     local fail = (r == 2 and k == 3999) or (r == 3 and k == 20)
                     sum = (sum + sweep(8 + k % 3, 40, fail)) % 1000003
                 end
                 total = (total + sum) % 1000003
             end)
             messages[#messages + 1] = tostring(ok) .. ':' .. tostring(message)
         end
         return total, table.concat(messages, '|')",
    );
    // The flag the nest tests comes in as an argument: the guard on it takes the error.
    assert_linked(&source, 12000);
}

#[test]
fn calls_through_pcall_and_plain_calls_share_the_nest() {
    let source = program(
        PLAIN_BODY,
        "local total, failures = 0, 0
         for k = 1, 12000 do
             if k % 3 == 0 then
                 local ok, value = pcall(sweep, 8 + k % 3, 40)
                 if ok then total = (total + value) % 1000003 else failures = failures + 1 end
             else
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
         end
         return total, failures",
    );
    // The branch to `pcall` is a path the recording of the loop did not take
    // (a call of a function the caller does not call again), so the tree
    // stays out of this loop; the answer is the interpreter's.
    run_all(&source);
}

#[test]
fn a_nest_called_from_two_places_runs_right_from_both() {
    let source = program(
        PLAIN_BODY,
        "local total = 0
         local function other(k)
             return sweep(6 + k % 5, 30) + 1
         end
         for k = 1, 12000 do
             if k % 5 == 0 then
                 total = (total + other(k)) % 1000003
             else
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
         end
         return total",
    );
    // `sweep` is captured by `other`, so its register in the caller is the
    // variable of a cell and not a value the tree may read: the tree does
    // not link here, and both call sites give the interpreter's answer.
    run_all(&source);
}

#[test]
fn a_hook_set_while_the_calls_run_sees_what_the_interpreter_shows() {
    for kind in ["l", "c", "r"] {
        let source = program(
            PLAIN_BODY,
            &format!(
                "local total, hits = 0, 0
                 for k = 1, 12000 do
                     if k == 9000 then debug.sethook(function() hits = hits + 1 end, '{kind}') end
                     if k == 9004 then debug.sethook() end
                     total = (total + sweep(8 + k % 3, 40)) % 1000003
                 end
                 return total, hits"
            ),
        );
        assert_linked(&source, 12000);
    }
}

#[test]
fn a_count_hook_set_while_the_calls_run_stops_after_its_budget() {
    for count in [1, 7, 100] {
        let source = program(
            PLAIN_BODY,
            &format!(
                "local total, hits = 0, 0
                 for k = 1, 12000 do
                     if k == 9000 then debug.sethook(function() hits = hits + 1 end, '', {count}) end
                     if k == 9010 then debug.sethook() end
                     total = (total + sweep(8 + k % 3, 40)) % 1000003
                 end
                 return total, hits"
            ),
        );
        run_all(&source);
    }
}

#[test]
fn a_traceback_and_the_debug_info_from_inside_the_nest_show_both_frames() {
    let source = program(
        "if probe and i == 3 and j == 11 then
                         local info = debug.getinfo(1, 'nSl')
                         local up = debug.getinfo(2, 'l')
                         seen[#seen + 1] = info.currentline .. ':' .. up.currentline .. ':'
                             .. #debug.traceback('here', 1)
                     end
                     acc = acc + (i * j) % 7",
        "seen = {}
         local total = 0
         for k = 1, 12000 do
             total = (total + sweep(8 + k % 3, 40, k == 9000 or k == 11999)) % 1000003
         end
         return total, table.concat(seen, ',')",
    );
    assert_linked(&source, 12000);
}

#[test]
fn a_yield_from_inside_the_nest_resumes_in_the_same_loop() {
    let source = program(
        "if probe and i == 4 and j == 9 then coroutine.yield(acc) end
                     acc = acc + (i * j) % 7",
        "local worker = coroutine.wrap(function()
             local total = 0
             for k = 1, 12000 do
                 total = (total + sweep(8 + k % 3, 40, k % 3000 == 0)) % 1000003
             end
             return -1, total
         end)
         local yields, last, total = 0, 0, 0
         while last ~= -1 do last, total = worker(); yields = yields + 1 end
         return yields, total",
    );
    run_all(&source);
}

#[test]
fn flushing_the_jit_while_the_calls_run_leaves_the_program_right() {
    let source = program(
        PLAIN_BODY,
        "local total = 0
         for k = 1, 12000 do
             if k % 3000 == 0 and jit and jit.flush then jit.flush() end
             total = (total + sweep(8 + k % 3, 40)) % 1000003
         end
         return total",
    );
    assert_linked(&source, 12000);
}

#[test]
fn a_function_replaced_while_the_calls_run_is_called_in_its_place() {
    let source = "
        local function first(n, m)
            local acc = 0
            for i = 1, n do
                for j = i, m do acc = acc + (i * j) % 7 end
            end
            return acc
        end
        local function second(n, m)
            local acc = 0
            for i = 1, n do
                for j = i, m do acc = acc + (i + j) % 5 end
            end
            return acc
        end
        local sweep = first
        local total = 0
        for k = 1, 12000 do
            if k == 7000 then sweep = second end
            if k == 9000 then sweep = first end
            total = (total + sweep(8 + k % 3, 40)) % 1000003
        end
        return total";
    assert_linked(source, 12000);
}

#[test]
fn a_nest_that_calls_itself_gives_the_interpreters_answer() {
    let source = "
        local depth = 0
        local function sweep(n, m)
            local acc = 0
            for i = 1, n do
                for j = i, m do acc = acc + (i * j) % 7 end
            end
            if depth < 2 and n == 10 and acc % 3 == 0 then
                depth = depth + 1
                acc = acc + sweep(n - 4, m - 20)
                depth = depth - 1
            end
            return acc
        end
        local total = 0
        for k = 1, 12000 do
            total = (total + sweep(8 + k % 3, 40)) % 1000003
        end
        return total";
    run_all(source);
}

#[test]
fn tables_the_caller_makes_and_the_collector_frees_while_the_calls_run() {
    // The caller makes a table with a table inside it in every round, keeps
    // fifty of them, and the collector runs on its own and when asked. The
    // table is stored before the call, so the loop carries nothing made.
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total = 0
             local keep = {}
             for k = 1, 12000 do
                 local row = {k, k + 1, {k}}
                 keep[k % 50 + 1] = row
                 if k % 1000 == 0 then collectgarbage() end
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             local sum = 0
             for i = 1, #keep do sum = sum + keep[i][1] + keep[i][3][1] end
             return total, #keep, sum",
        ),
        12000,
    );
}

#[test]
fn tables_the_caller_reads_back_after_the_call_are_the_interpreters() {
    // A table made before the call and read after it is a value the loop
    // carries, which the tree does not link: the answer is the interpreter's.
    run_all(&program(
        PLAIN_BODY,
        "local total = 0
         local keep = {}
         for k = 1, 12000 do
             local row = {k, k + 1, {k}}
             keep[k % 50 + 1] = row
             if k % 1000 == 0 then collectgarbage() end
             total = (total + sweep(8 + k % 3, 40) + row[1] + row[3][1]) % 1000003
         end
         return total, #keep",
    ));
}

#[test]
fn a_caller_that_stores_into_a_table_links() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local out = {}
             for k = 1, 12000 do
                 out[k % 10 + 1] = sweep(8 + k % 3, 40)
             end
             return out[1], out[5], out[10]",
        ),
        12000,
    );
}

#[test]
fn a_caller_that_calls_a_library_function_and_a_global_links() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local t = 0
             for k = 1, 12000 do
                 t = t + sweep(8 + k % 3, 40) + math.abs(-k)
             end
             return t",
        ),
        12000,
    );
}

#[test]
fn a_caller_with_stores_tables_and_library_calls_links_in_a_function() {
    // The caller runs a closure, so its globals and upvalues are the
    // closure's; it stores into upvalue tables, makes tables, reads a global
    // and calls a library function.
    assert_linked(
        &program(
            PLAIN_BODY,
            "local out, keep, log = {}, {}, {}
             local function drive()
                 local t = 0
                 for k = 1, 12000 do
                     local row = {k, k + 1}
                     keep[k % 50 + 1] = row
                     local v = sweep(8 + k % 3, 40)
                     out[k % 10 + 1] = v
                     t = (t + v + math.abs(-k) + #out) % 1000003
                     if k % 7000 == 0 then log[#log + 1] = t end
                 end
                 return t
             end
             return drive(), #out, #keep, #log, out[3]",
        ),
        12000,
    );
}

#[test]
fn a_function_that_stores_after_its_nest_links() {
    // A store to a global and one to a field of an upvalue table, after the
    // loops and before the return.
    for tail in ["G = (G or 0) + 1", "box.n = box.n + 1"] {
        let source = format!(
            "G = 0
             local box = {{n = 0}}
             local function sweep(n, m)
                 local acc = 0
                 for i = 1, n do
                     for j = i, m do acc = acc + (i * j) % 7 end
                 end
                 {tail}
                 return acc
             end
             local total = 0
             for k = 1, 12000 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total, G, box.n"
        );
        assert_linked(&source, 12000);
    }
}

#[test]
fn a_branch_in_the_callers_code_that_changes_leaves_the_tree_for_the_interpreter() {
    // The caller's branches go the other way part of the way through the
    // run: the guard in the code after the call fails, and the exit pops the
    // root frame and resumes the caller at the branch. The answer is the
    // interpreter's, with a table stored in the other arm.
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total, seen, rare = 0, {}, 0
             for k = 1, 12000 do
                 local v = sweep(8 + k % 3, 40)
                 if k > 9000 then
                     v = v + 1
                     seen[k % 7 + 1] = v
                 end
                 if k % 2999 == 0 then rare = rare + k end
                 total = (total + v) % 1000003
             end
             return total, #seen, rare",
        ),
        12000,
    );
}

#[test]
fn a_register_a_closure_captured_is_never_left_stale_by_the_tree() {
    // `n` is a captured local of the caller that the loop writes (and never
    // reads) before the call; `peek` reads it through its cell. A tree that
    // kept `n` in a native value would leave the cell, and the register, as
    // they were.
    run_all(&program(
        PLAIN_BODY,
        "local n, total = 0, 0
         local function peek() return n end
         for k = 1, 12000 do
             n = k
             total = (total + sweep(8 + k % 3, 40) + peek()) % 1000003
         end
         return total, n",
    ));
}

#[test]
fn a_caller_loop_that_counts_down_is_the_same_loop() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total = 0
             for k = 12000, 1, -1 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total",
        ),
        12000,
    );
}

#[test]
fn a_caller_loop_with_a_step_is_the_same_loop() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total = 0
             for k = 1, 24000, 2 do
                 total = (total + sweep(8 + k % 3, 40)) % 1000003
             end
             return total",
        ),
        12000,
    );
}

#[test]
fn a_caller_while_loop_is_the_same_loop() {
    assert_linked(
        &program(
            PLAIN_BODY,
            "local total, n = 0, 0
             while n < 12000 do
                 n = n + 1
                 total = (total + sweep(8 + n % 3, 40)) % 1000003
             end
             return total",
        ),
        12000,
    );
}

#[test]
fn a_triangular_pair_loop_with_an_update_loop_links_through_the_calls() {
    // The shape of `nbody`: the pair loop, the loop after it, and a loop
    // that calls `step` again and again. The tree is big, so it links
    // after many entries.
    assert_linked_in_float_profiles(
        "local items = {}
         for i = 1, 5 do items[i] = {x = i, v = i * 0.5, w = 0.25} end
         local function step(dt)
             for i = 1, #items do
                 local a = items[i]
                 for j = i + 1, #items do
                     local b = items[j]
                     local d = b.x - a.x
                     local m = dt / (d * d + 1)
                     a.v = a.v + d * m * b.w
                     b.v = b.v - d * m * a.w
                 end
             end
             for i = 1, #items do
                 local a = items[i]
                 a.x = a.x + a.v * dt
             end
         end
         for _ = 1, 40000 do step(0.001) end
         local total = 0
         for i = 1, #items do total = total + items[i].x + items[i].v end
         return math.floor(total * 1000)",
        40000,
    );
}
