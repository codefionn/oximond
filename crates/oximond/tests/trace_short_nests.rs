//! Nests of short `while` and generic-for loops in one frame. The innermost
//! loop is the root, runs a few trips per entry, and is kept as a stepping
//! stone while outer sides carry it through the loops around it (the outer
//! part may read tables out of tables by number and measure them). The
//! results are the interpreter's in every profile, and the nest runs natively.

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

/// Runs `source` traced and interpreted in every profile, checks that the
/// outcomes agree and are not errors, and returns the traced run's statistics
/// by profile.
fn run_all(source: &str) -> Vec<(CompatibilityProfile, oximond::vm::TraceStatistics)> {
    PROFILES
        .into_iter()
        .map(|profile| {
            let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
            let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?} {source}");
            assert!(expected.is_ok(), "{profile:?} {expected:?}");
            (profile, lua.vm().jit_statistics().trace)
        })
        .collect()
}

/// The first value `source` returns when the interpreter runs it in `profile`,
/// as a whole number.
fn interpreted_count(profile: CompatibilityProfile, source: &str) -> u64 {
    let (results, _) = outcome(profile, ExecutionMode::Interpreter, source);
    match results.unwrap().first() {
        Some(Value::Integer(count)) => u64::try_from(*count).unwrap(),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(Value::Number(count)) => *count as u64,
        other => panic!("not a count: {other:?}"),
    }
}

/// `n1`: three `while` loops with compound conditions, a few trips each, that
/// walk the edge lists of two boxes; the innermost calls `dom`, which indexes
/// the lists it is passed, and the outer parts index `l.y`/`l.x` by number
/// and measure `#l.z`. `count` puts a counter of the innermost trips in the
/// loop and returns it.
fn n1(count: bool) -> String {
    let (declare, step, result) = if count {
        ("local trips = 0", "trips = trips + 1", "return trips")
    } else {
        ("", "", "return t")
    };
    format!(
        "{declare}
         local function dom(i, j, l, r, lm, rm)
           local a, b
           if i == lm then a = nil; b = j ~= rm and j or nil
           elseif j == rm then a = i ~= lm and i or nil; b = nil
           elseif l[i] < r[j] then a = i; b = j ~= 1 and j - 1 or nil
           elseif r[j] < l[i] then a = i ~= 1 and i - 1 or nil; b = j
           else a = i; b = j end
           return a, b
         end
         local L = {{ x = {{1,2,3,4}}, y = {{1,2,3}}, z = {{1,3,4,5}} }}
         local R = {{ x = {{1,2,4}}, y = {{1,2,3,4}}, z = {{2,3,4}} }}
         local function run(l, r)
           local n = 0
           local lx, rx = 1, 1
           local lxm, rxm = #l.x + 1, #r.x + 1
           while lx ~= lxm or rx ~= rxm do
             local lx_old, rx_old = lx, rx
             local ly, ry = 1, 1
             local lym, rym = #l.y + 1, #r.y + 1
             while ly ~= lym or ry ~= rym do
               local ly_old, ry_old = ly, ry
               local lz, rz = 1, 1
               local lzm, rzm = #l.z + 1, #r.z + 1
               while lz ~= lzm or rz ~= rzm do
                 {step}
                 local lz_old, rz_old = lz, rz
                 local a, b = dom(lz, rz, l.z, r.z, lzm, rzm)
                 if a and b then n = n + 1 end
                 if lz ~= lzm and (rz_old == rzm or l.z[lz_old] <= r.z[rz_old]) then lz = lz + 1 end
                 if rz ~= rzm and (lz_old == lzm or r.z[rz_old] <= l.z[lz_old]) then rz = rz + 1 end
               end
               if ly ~= lym and (ry_old == rym or l.y[ly_old] <= r.y[ry_old]) then ly = ly + 1 end
               if ry ~= rym and (ly_old == lym or r.y[ry_old] <= l.y[ly_old]) then ry = ry + 1 end
             end
             if lx ~= lxm and (rx_old == rxm or l.x[lx_old] <= r.x[rx_old]) then lx = lx + 1 end
             if rx ~= rxm and (lx_old == lxm or r.x[rx_old] <= l.x[lx_old]) then rx = rx + 1 end
           end
           return n
         end
         local t = 0
         for i = 1, 20000 do t = t + run(L, R) end
         {result}"
    )
}

#[test]
fn three_short_while_loops_run_as_one_tree_of_the_innermost() {
    let source = n1(false);
    let trips = interpreted_count(CompatibilityProfile::Lua51, &n1(true));
    for (profile, stats) in run_all(&source) {
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        if profile == CompatibilityProfile::Lua51 {
            // The middle and the outer loop are outer sides of the innermost
            // loop's root, which runs nearly every trip natively.
            assert!(stats.outer_sides >= 2, "{profile:?} {stats:?}");
            assert!(stats.covered_loops >= 2, "{profile:?} {stats:?}");
            assert!(
                stats.iterations * 10 >= trips * 9,
                "{profile:?}: {} iterations for {trips} inner trips: {stats:?}",
                stats.iterations
            );
            // Each call is about one entry, not one per middle iteration.
            assert!(stats.native_entries < 40_000, "{profile:?} {stats:?}");
        }
    }
}

#[test]
fn a_generic_for_inside_a_while_inside_a_for_runs_natively() {
    let source = "local t = 0
         local items = {1, 2, 3}
         for i = 1, 100000 do
           local j = 0
           while j < i % 4 do
             for _, v in ipairs(items) do t = t + v * j end
             j = j + 1
           end
         end
         return t";
    for (profile, stats) in run_all(source) {
        // The `ipairs` loop's end takes an outer side through the `while`
        // loop, whose end takes one through the `for` loop.
        assert!(stats.side_installs >= 2, "{profile:?} {stats:?}");
        assert!(stats.outer_sides >= 2, "{profile:?} {stats:?}");
        assert!(stats.native_entries < 1000, "{profile:?} {stats:?}");
        assert!(stats.iterations >= 300_000, "{profile:?} {stats:?}");
    }
}

#[test]
fn an_ipairs_nest_agrees_when_the_list_changes_under_it() {
    // A nil hole appears, the list grows and shrinks, and a metatable with
    // `__index` appears: the end of the `ipairs` loop moves, and the outer
    // side's read past the end leaves for the interpreter wherever it can no
    // longer prove nil.
    let source = "local t = 0
         local items = {1, 2, 3}
         for i = 1, 60000 do
           if i == 20000 then items[4] = 4 end
           if i == 30000 then items[2] = nil end
           if i == 40000 then items[2] = 2; items[4] = nil end
           if i == 50000 then
             setmetatable(items, {__index = function(_, k) if k == 4 then return 10 end end})
           end
           local j = 0
           while j < i % 4 do
             for k, v in ipairs(items) do t = t + v * j + k end
             j = j + 1
           end
         end
         return t";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_short_while_nest_agrees_when_types_and_tables_change() {
    // The middle loop's table is swapped for one of another length, an
    // element turns into a float and then a string, so the trees' guards
    // fail at places they never failed while they grew.
    let source = "local rows = { {1, 2, 3}, {4, 5}, {6, 7, 8, 9} }
         local other = { {1}, {2, 3, 4, 5, 6} }
         local function run(m, k)
           local s, i = 0, 1
           while i <= #m do
             local row = m[i]
             local j = 1
             while j <= #row do
               s = s + row[j] * k
               j = j + 1
             end
             i = i + 1
           end
           return s
         end
         local t = 0
         for n = 1, 30000 do
           if n == 15000 then rows = other end
           if n == 20000 then other[2][3] = 3.5 end
           if n == 25000 then other[2][4] = '7' end
           t = t + run(rows, n % 3)
         end
         return t";
    for (profile, stats) in run_all(source) {
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
        if profile == CompatibilityProfile::Lua51 {
            assert!(stats.outer_sides >= 1, "{profile:?} {stats:?}");
        }
    }
}

/// A `while` loop of one to three trips inside a `while` loop whose body
/// builds a string no side can hold (a concatenation has no root slot to
/// restore it from), so no outer side can record the outer part.
fn uncoverable_nest(outer: u32) -> String {
    format!(
        "local t, i = 0, 0
         local limit = {outer}
         local last
         while i ~= limit or t < 0 do
           local j, m = 0, 1 + i % 3
           while j ~= m or t < 0 do
             t = t + i * j + 1
             j = j + 1
           end
           last = 'k' .. i
           i = i + 1
         end
         return t, last"
    )
}

#[test]
fn a_short_while_root_whose_outer_side_never_records_is_retired_and_unrolled() {
    // The inner root is retired as before, which marks the loop short (or
    // gives its header up), and the outer loop's trace unrolls it: most
    // outer iterations run natively, with no outer side.
    let outer = 100_000;
    for (profile, stats) in run_all(&uncoverable_nest(outer)) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert_eq!(stats.outer_sides, 0, "{profile:?} {stats:?}");
        assert!(
            stats.iterations * 10 >= u64::from(outer) * 7,
            "{profile:?} {stats:?}"
        );
        assert!(stats.native_entries < 10_000, "{profile:?} {stats:?}");
    }
}

#[test]
fn a_short_while_root_whose_entries_mostly_mismatch_gives_up_its_stone() {
    // `a`, `b` and `c` are nil or a number by outer iteration, in eight
    // mixes, more than a root keeps versions for (as `lx_dom` and its
    // siblings in mcl_util's `region_evaluate`). Most entries of the inner
    // root find types no version takes, so it is not kept as a stepping
    // stone (or its stone gives up) while those entries go to the
    // interpreter: it is judged as any root, and the outer sides that do
    // not fit the tree are not recorded over and over.
    //
    // With untyped inputs on (the default), once the loop's
    // entries have shown `a`, `b` and `c` with both types, the root takes
    // them untyped (`Recorder::with_lazy`) and the outer
    // sides that set them again link through the root's input slots, so
    // the tree carries the `i` loop: no entry mismatches any more, and an
    // entry or two per call. The bound is on that: no recording over and
    // over, and no entry per `j` trip. (Each call is 24 `j` trips, short
    // enough that the entries cost more than the native trips save: about
    // 0.61 G instructions traced against 0.34 G when the stone gave up and
    // 0.26 G interpreted, see the notes in `vm/jit/trace/mod.rs`.)
    let source = "local xs = {}
         for k = 1, 64 do xs[k] = (k * 7) % 5 ~= 0 and k or nil end
         local function run(o)
           local s, i = 0, 1
           while i <= 8 do
             local a, b, c = xs[i + o], xs[i + o + 1], xs[i + o + 3]
             local j = 1
             while j <= 3 do
               if a then s = s + a * j end
               if b then s = s + b end
               if c then s = s - c end
               j = j + 1
             end
             i = i + 1
           end
           return s
         end
         local t = 0
         for n = 1, 20000 do t = t + run(n % 50) end
         return t";
    // 20 000 calls run the inner loop 160 000 times.
    for (profile, stats) in run_all(source) {
        assert!(stats.traces_compiled >= 1, "{profile:?} {stats:?}");
        assert!(stats.native_entries < 3 * 20_000, "{profile:?} {stats:?}");
        assert!(stats.traces_compiled <= 12, "{profile:?} {stats:?}");
    }
}

/// A `while` loop of two trips in a function that a loop calls: the hot
/// count is taken at the backedge, and the second backedge of a call is the
/// pass whose test leaves the loop.
const TWO_TRIPS: &str = "local function run(n)
       local s, j = 0, 0
       while j ~= 2 or s < 0 do
         s = s + j + n
         j = j + 1
       end
       return s
     end
     local t, i = 0, 0
     while i ~= 100000 or t < 0 do
       t = t + run(i % 7)
       i = i + 1
     end
     return t";

#[test]
fn a_root_recording_that_starts_on_the_leaving_pass_spends_no_attempt() {
    // Every recording of the inner loop that starts on the second backedge
    // of a call leaves at once. It is dropped, uncounted, and the next
    // backedge (the first of the next call) records the loop.
    for (profile, stats) in run_all(TWO_TRIPS) {
        assert_eq!(
            stats.recording_aborts[TraceAbortReason::LoopNotClosed.index()],
            0,
            "{profile:?} {stats:?}"
        );
        assert_eq!(stats.recording_starts, 2, "{profile:?} {stats:?}");
        assert_eq!(stats.traces_compiled, 2, "{profile:?} {stats:?}");
    }
}

/// `n1` with what `region_evaluate` of `mcl_util` adds: the values `dom` gives
/// the `x` and `y` loops live into the `z` loop, nil on some trips and a
/// number on others, and the `z` loop calls `op`, one of three closures by
/// caller. Returns the `z` trips and what `op` counted.
const EVALUATE: &str = "local trips = 0
     local function dom(i, j, l, r, lm, rm)
       local a, b
       if i == lm then a = nil; b = j ~= rm and j or nil
       elseif j == rm then a = i ~= lm and i or nil; b = nil
       elseif l[i] < r[j] then a = i; b = j ~= 1 and j - 1 or nil
       elseif r[j] < l[i] then a = i ~= 1 and i - 1 or nil; b = j
       else a = i; b = j end
       return a, b
     end
     local L = { x = {1,2,3,4}, y = {1,2,3}, z = {1,3,4,5} }
     local R = { x = {1,2,4}, y = {1,2,3,4}, z = {2,3,4} }
     local ops = {
       function(l, r) return l ~= r end,
       function(l, r) return r and not l end,
       function(l, r) return l and r end,
     }
     local function run(l, r, op)
       local n = 0
       local lx, rx = 1, 1
       local lxm, rxm = #l.x + 1, #r.x + 1
       while lx ~= lxm or rx ~= rxm do
         local lx_old, rx_old = lx, rx
         local ly, ry = 1, 1
         local lym, rym = #l.y + 1, #r.y + 1
         local xa, xb = dom(lx, rx, l.x, r.x, lxm, rxm)
         while ly ~= lym or ry ~= rym do
           local ly_old, ry_old = ly, ry
           local lz, rz = 1, 1
           local lzm, rzm = #l.z + 1, #r.z + 1
           local ya, yb = dom(ly, ry, l.y, r.y, lym, rym)
           while lz ~= lzm or rz ~= rzm do
             trips = trips + 1
             local lz_old, rz_old = lz, rz
             local za, zb = dom(lz, rz, l.z, r.z, lzm, rzm)
             local l_on = xa and ya and za and (xa + ya + za) % 2 == 0
             local r_on = xb and yb and zb and (xb + yb + zb) % 3 ~= 0
             if op(l_on or false, r_on or false) then n = n + 1 end
             if lz ~= lzm and (rz_old == rzm or l.z[lz_old] <= r.z[rz_old]) then lz = lz + 1 end
             if rz ~= rzm and (lz_old == lzm or r.z[rz_old] <= l.z[lz_old]) then rz = rz + 1 end
           end
           if ly ~= lym and (ry_old == rym or l.y[ly_old] <= r.y[ry_old]) then ly = ly + 1 end
           if ry ~= rym and (ly_old == lym or r.y[ry_old] <= l.y[ly_old]) then ry = ry + 1 end
         end
         if lx ~= lxm and (rx_old == rxm or l.x[lx_old] <= r.x[rx_old]) then lx = lx + 1 end
         if rx ~= rxm and (lx_old == lxm or r.x[rx_old] <= l.x[lx_old]) then rx = rx + 1 end
       end
       return n
     end
     local t = 0
     for k = 1, 4000 do
       t = t + run(L, R, ops[k % 3 + 1]) + run(R, L, ops[(k + 1) % 3 + 1])
     end
     return trips, t";

#[test]
fn a_nest_with_nil_or_number_locals_and_a_callee_argument_runs_natively() {
    // The first root of the `z` loop types `ya` and `yb`, and outlives the
    // versions that take them untyped (those retire on the guard of `op`,
    // hoisted in front of the loop). As a stepping stone it gives up over
    // entries of the other type. The next root takes both untyped, guards
    // `op` on its prototype in the loop, and is a stepping stone again:
    // every `z` trip runs natively, a few entries per call of `run`.
    for (profile, stats) in run_all(EVALUATE) {
        let trips = interpreted_count(profile, EVALUATE);
        assert!(stats.iterations >= trips, "{profile:?} {trips} {stats:?}");
        // 8000 calls of `run`, 14 runs of the `z` loop each.
        assert!(stats.native_entries < 8 * 8_000, "{profile:?} {stats:?}");
        assert!(stats.traces_compiled <= 6, "{profile:?} {stats:?}");
    }
}
