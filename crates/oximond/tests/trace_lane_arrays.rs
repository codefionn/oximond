//! Lane array views: a root trace reads `row[j]` from a row table it loaded
//! inside the loop (`local row = m[i]`) through a `LaneArrayLoad`, with no
//! table lookup helper. A nil hole, a string element, a `table.insert` that
//! grows the row and a coroutine resume beside the read all give the
//! interpreter's results in every profile.

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

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree and that native code ran, and hands the tracing statistics
/// to `check`.
fn assert_agrees(
    source: &str,
    check: impl Fn(CompatibilityProfile, &oximond::vm::TraceStatistics),
) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        assert!(expected.is_ok(), "{profile:?} {expected:?}");
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        check(profile, &stats);
    }
}

#[test]
fn row_sums_read_the_lane_array_without_helpers() {
    assert_agrees(
        "local m = {}
         for i = 1, 8 do
             local row = {}
             for j = 1, 6 do row[j] = i * 10 + j end
             m[i] = row
         end
         local total = 0
         for k = 1, 30000 do
             local row = m[k % 8 + 1]
             for j = 1, 6 do total = total + row[j] end
         end
         return total",
        |profile, stats| {
            assert_eq!(stats.table_lookup_helper_calls, 0, "{profile:?} {stats:?}");
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

#[test]
fn float_rows_read_at_a_varying_index() {
    assert_agrees(
        "local m = {}
         for i = 1, 8 do
             local row = {}
             for j = 1, 6 do row[j] = i * 0.5 + j end
             m[i] = row
         end
         local total = 0
         for k = 1, 60000 do
             local row = m[k % 8 + 1]
             total = total + row[k % 6 + 1] + row[1]
         end
         return total",
        |profile, stats| {
            assert_eq!(stats.table_lookup_helper_calls, 0, "{profile:?} {stats:?}");
        },
    );
}

#[test]
fn a_nil_hole_and_a_string_element_leave_the_lane_read() {
    assert_agrees(
        "local m = {}
         for i = 1, 8 do
             local row = {}
             for j = 1, 6 do row[j] = i * 10 + j end
             m[i] = row
         end
         local total, holes, words = 0, 0, 0
         for k = 1, 30000 do
             if k == 20000 then m[3][4] = nil end
             if k == 25000 then m[5][2] = 'word' end
             local row = m[k % 8 + 1]
             for j = 1, 6 do
                 local v = row[j]
                 if v == nil then holes = holes + 1
                 elseif type(v) == 'string' then words = words + 1
                 else total = total + v end
             end
         end
         return total, holes, words",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

#[test]
fn an_insert_into_the_row_grows_what_later_reads_see() {
    assert_agrees(
        "local m = {}
         for i = 1, 4 do
             local row = {}
             for j = 1, 4 do row[j] = i + j end
             m[i] = row
         end
         local total = 0
         for k = 1, 20000 do
             local row = m[k % 4 + 1]
             if k % 1000 == 0 then table.insert(row, k) end
             total = total + row[1] + row[#row] + row[4]
         end
         return total, #m[1], #m[4]",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

#[test]
fn a_coroutine_resume_beside_the_lane_read_is_not_recorded() {
    assert_agrees(
        "local m = {}
         for i = 1, 8 do
             local row = {}
             for j = 1, 6 do row[j] = i * 10 + j end
             m[i] = row
         end
         local co = coroutine.wrap(function()
             local n = 0
             while true do
                 n = n + 1
                 m[n % 8 + 1][n % 6 + 1] = n
                 coroutine.yield(n)
             end
         end)
         local total = 0
         for k = 1, 20000 do
             local row = m[k % 8 + 1]
             total = total + row[k % 6 + 1]
             if k % 7 == 0 then total = total + co() end
             total = total + row[k % 6 + 1]
         end
         return total",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

/// `o.out[1] = u` is a number store into a table met by lane, so the
/// recording of the second loop is rowless and `o.arr` is a lookup call. The
/// lane that call gives has no row until native code asks for one, and the
/// rows are shared by the thread's traces and not cleared between runs: the
/// read of `a[2]` through the lane's row found the address of a table the
/// first loop's run had left there, and summed its slots.
#[test]
fn a_table_a_lookup_call_gives_reads_its_own_array() {
    assert_agrees(
        "local n = 300
         local list = {}
         for i = 1, n do list[i] = {i * 100, i * 100 + 1, i * 100 + 2} end
         local objs = {}
         for i = 1, n do objs[i] = {arr = {i, i + 1, i + 2}, out = {0}} end
         local function first()
             local s = 0
             for i = 1, n do
                 local t = list[i]
                 s = s + t[2]
             end
             return s
         end
         local function second()
             local u = 0
             for i = 1, n do
                 local o = objs[i]
                 local a = o.arr
                 u = u + a[2]
                 o.out[1] = u
             end
             return u
         end
         local total = 0
         for rep = 1, 300 do
             total = total + first() + second()
         end
         return total",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

/// A side trace attached at the guard of a lane array read runs inline in
/// that guard's exit, which the check of the row's address word also takes.
/// The array words read past that check are no values the side may use: the
/// side's own read of `a[2]` once took them from the parent's cache, and in
/// a release build Cranelift's register allocator panicked (`EntryLivein`),
/// which killed the compile thread for the rest of the process.
#[test]
fn a_side_at_a_lane_array_read_loads_the_words_itself() {
    assert_agrees(
        "local n = 400
         local list = {}
         for i = 1, n do list[i] = {x = i + 0.5, i * 100, i * 100 + 1} end
         local objs = {}
         for i = 1, n do objs[i] = {arr = {i, i + 1, i + 2}, out = {0}} end
         local total = 0
         for rep = 1, 200 do
             local s = 0
             for i = 1, n do
                 local t = list[i]
                 s = s + t.x + t[1]
             end
             local u = 0
             for i = 1, n do
                 local o = objs[i]
                 local a = o.arr
                 u = u + a[2]
                 o.out[1] = u
             end
             total = total + s + u
         end
         return total",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
            // Debug builds run Cranelift's verifier, which refused the tree
            // with the side instead.
            assert_eq!(stats.side_rejections, 0, "{profile:?} {stats:?}");
        },
    );
}

/// The value of a `pairs` step is the second result of the `next` helper,
/// and a table there gets a lane of the run's bank. The loop reads `box[5]`
/// through the lane's row, so the row must be this run's. Rows are shared by
/// every trace of the thread and not cleared between runs: the other loop
/// fills the same lanes with the rows of `m`'s tables, and before the row was
/// filled after the step, the sum read `m`'s elements (or, once a row's table
/// was freed, faulted, as Mineclonia's `gwp_compute_fixed_ground_height` did
/// at startup).
#[test]
fn a_table_the_next_step_gives_reads_its_own_array() {
    assert_agrees(
        "local m = {}
         for i = 1, 10 do m[i] = {0, 0, 0, 0, 1000 + i, 0} end
         local total = 0
         local function sum(fixed)
             local t = 0
             for _, box in pairs(fixed) do t = t + box[5] end
             return t
         end
         local bad = 0
         for d = 1, 400 do
             local fixed = {}
             for b = 1, 4 do fixed[b] = {0, 0, 0, 0, b, 0} end
             if sum(fixed) ~= 10 then bad = bad + 1 end
             if d % 50 == 0 then
                 for k = 1, 3000 do total = total + m[k % 10 + 1][5] end
             end
         end
         return bad, total",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

/// `pairs` over a list with no hash part steps along its array part with an
/// `ArrayNext` (the loop never reads the key) and reads each slot. Lists of
/// other lengths, a run of holes between values and a slot cleared during
/// the traversal must all leave the compiled step where the interpreter
/// moves on.
#[test]
fn array_steps_walk_lists_of_every_length_and_pattern_of_holes() {
    assert_agrees(
        "local total = 0
         local function sum(list)
             local t = 0
             for _, v in pairs(list) do t = t + v end
             return t
         end
         local function keyed(list)
             local t = 0
             for k, v in pairs(list) do t = t + k * v end
             return t
         end
         for r = 1, 600 do
             local dense, holey, sparse = {}, {}, {}
             local n = r % 37 + 1
             for i = 1, n do
                 dense[i] = i * 0.5
                 if i % 3 ~= 0 then holey[i] = i * 2 end
             end
             sparse[1] = 1
             sparse[n + 9] = n
             total = total + sum(dense) + sum(holey) + keyed(dense) + keyed(holey)
             if r % 5 == 0 then total = total + sum(sparse) end
             if r % 7 == 0 then
                 for k, v in pairs(holey) do holey[k] = nil end
                 total = total + sum(holey)
             end
         end
         return total",
        |_, _| {},
    );
}

/// A list of tables walked with `pairs`: the step's key reads the slot the
/// step found, and the table it holds is read through the lane's row. The
/// second loop fills the same lanes with other rows between the runs.
#[test]
fn array_steps_over_a_list_of_tables_read_each_one_through_its_own_row() {
    assert_agrees(
        "local m = {}
         for i = 1, 10 do m[i] = {1, 2, 3, 4, 1000 + i, 6} end
         local function sum(list)
             local t = 0
             for _, box in pairs(list) do t = t + box[5] + box[1] end
             return t
         end
         local bad, total = 0, 0
         for d = 1, 400 do
             local list = {}
             for b = 1, d % 6 + 1 do list[b] = {0, 0, 0, 0, b, 0} end
             local want = 0
             for b = 1, d % 6 + 1 do want = want + b end
             if sum(list) ~= want then bad = bad + 1 end
             if d % 40 == 0 then
                 for k = 1, 3000 do total = total + m[k % 10 + 1][5] end
             end
         end
         return bad, total",
        |profile, stats| {
            assert!(stats.lane_array_loads_compiled > 0, "{profile:?} {stats:?}");
        },
    );
}

/// `next` over a list in a loop that keeps its key (`for k, v in next, t`):
/// the same step with the key read, through the helper form when a value is
/// not a number.
#[test]
fn next_loops_over_lists_of_numbers_and_mixed_values_agree() {
    assert_agrees(
        "local total = 0
         local function walk(t)
             local s = 0
             for k, v in next, t do
                 if type(v) == 'number' then s = s + k + v else s = s + k end
             end
             return s
         end
         for r = 1, 500 do
             local t = {}
             for i = 1, 12 do t[i] = i end
             if r % 50 == 0 then t[5] = 'five' end
             if r % 70 == 0 then t[7] = nil end
             total = total + walk(t)
         end
         return total",
        |_, _| {},
    );
}

/// A loop that starts from a key the caller gives takes its steps without
/// checking the key (the entry checks the one it starts with). A start key
/// that names a slot that was cleared, one past the end of the array and a
/// float must be declined at entry or give the interpreter's sums.
#[test]
fn a_traversal_started_at_any_key_gives_the_interpreters_sums() {
    assert_agrees(
        "local t = {}
         for i = 1, 40 do t[i] = i end
         local function walk(start)
             local s = 0
             for k, v in next, t, start do s = s + k * v end
             return s
         end
         local total = 0
         for r = 1, 400 do
             total = total + walk(nil)
             if r % 3 == 0 then total = total + walk(r % 40 + 1) end
             if r == 100 then t[7] = nil end
             if r == 200 then t[7] = 7 t[40] = nil end
             if r % 17 == 0 then
                 local ok, err = pcall(walk, 41 + r % 5)
                 total = total + (ok and err or -1)
             end
             if r % 23 == 0 then
                 local ok, err = pcall(walk, 7)
                 total = total + (ok and err or -2)
             end
         end
         return total",
        |_, _| {},
    );
}
