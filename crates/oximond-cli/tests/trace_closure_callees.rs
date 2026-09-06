//! Calls of closures that take turns at one call site, through compiled Lua
//! source.
//!
//! A call site whose callee is not one closure (a set of locals that swap
//! places, an array of closures) is guarded on the callee's prototype, and
//! the callee's upvalue cells are read and written through the live closure.
//! One trace then serves every closure of a prototype, and a side trace
//! serves each other prototype. What these programs hold the engine to:
//!
//! - The shapes of `opt/closure_family`, `opt/closure_protos` and
//!   `opt/retained_slots` run natively: no recording aborts at the
//!   `SetUpvalue` or at the load of a closure from an array, the trees stay
//!   installed, and nearly every iteration runs in native code.
//! - Cells keep their semantics: two closures sharing a cell see each other's
//!   writes within one iteration, a cell still open in the enclosing frame
//!   holds what the trace wrote, closures of one prototype keep their own
//!   values, and a table stored into a cell reaches the collector's barrier.
//! - Everything else that can change under the loop is the interpreter's
//!   business: `debug.setupvalue` and `debug.upvaluejoin`, a callee that
//!   becomes a native function or a table with `__call`, an array slot that
//!   stops holding a function, and an error inside the callee with the
//!   traceback it makes. Results, error messages and tracebacks match the
//!   interpreter exactly.

use oximond::vm::{TraceAbortReason, TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::fmt::Write;

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles with `debug.upvaluejoin`.
const HAS_UPVALUEJOIN: [CompatibilityProfile; 4] = [
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
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
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` under the interpreter and under tracing execution, checks
/// the results agree, and returns them with the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// `agree` for every profile of `profiles`, requiring native iterations.
fn agree_natively(profiles: &[CompatibilityProfile], source: &str) -> Vec<TraceStatistics> {
    profiles
        .iter()
        .map(|profile| {
            let (_, stats) = agree(*profile, source);
            assert!(
                stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            stats
        })
        .collect()
}

// ------------------------------------------------------------ the shapes

/// `opt/closure_family`: eight closures of one prototype, each with a cell of
/// its own, rotated through one call site.
fn family(limit: u32) -> String {
    format!(
        "
        local function make(k)
            local count = k * 10
            return function(x)
                count = count + 1
                return (x * k + count) % 1009
            end
        end
        local g1, g2, g3, g4 = make(1), make(2), make(3), make(4)
        local g5, g6, g7, g8 = make(5), make(6), make(7), make(8)
        local total = 0
        for i = 1, {limit} do
            total = (total + g1(i)) % 1000003
            g1, g2, g3, g4, g5, g6, g7, g8 = g2, g3, g4, g5, g6, g7, g8, g1
        end
        return total"
    )
}

/// `opt/closure_protos`: eight prototypes, each over a local of the chunk,
/// rotated through one call site.
fn protos(limit: u32) -> String {
    let mut source = String::new();
    for k in 1..=8 {
        write!(
            source,
            "local count{k} = 0
             local function p{k}(x)
                 count{k} = count{k} + 1
                 return (x * {k} + count{k}) % 1009
             end
            "
        )
        .expect("formatting into a String cannot fail");
    }
    write!(
        source,
        "local g1, g2, g3, g4, g5, g6, g7, g8 = p1, p2, p3, p4, p5, p6, p7, p8
         local total = 0
         for i = 1, {limit} do
             total = (total + g1(i)) % 1000003
             g1, g2, g3, g4, g5, g6, g7, g8 = g2, g3, g4, g5, g6, g7, g8, g1
         end
         return total, count1, count8"
    )
    .expect("formatting into a String cannot fail");
    source
}

/// `opt/retained_slots`: closures loaded out of an array, each over a table
/// of its own, and every table made kept in a ring.
fn retained(limit: u32, ring: u32) -> String {
    format!(
        "
        local function make(i)
            local object = {{v = i}}
            return function()
                object.v = (object.v + 1) % 1009
                return object.v
            end
        end
        local closures = {{}}
        for i = 1, 64 do closures[i] = make(i) end
        local ring = {{}}
        local first, second = {{1}}, {{v = 2}}
        local total = 0
        for i = 1, {limit} do
            local pair = {{i % 17, i % 23}}
            ring[i % {ring} + 1] = pair
            total = (total + closures[i % 64 + 1]() + first[1] + second.v + pair[1] + pair[2]) % 1000003
            first[1] = (first[1] + 1) % 7
            second.v = (second.v + pair[2]) % 11
        end
        return total, #ring"
    )
}

#[test]
fn closures_of_one_prototype_run_in_one_trace_in_every_profile() {
    for stats in agree_natively(&ALL_PROFILES, &family(60_000)) {
        assert_eq!(stats.traces_recorded, 1, "{stats:?}");
        // The prototype guard never fails: no side trace has anything to do.
        assert_eq!(stats.side_installs, 0, "{stats:?}");
        assert_eq!(stats.side_recordings, 0, "{stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{stats:?}"
        );
        assert!(stats.iterations > 59_000, "{stats:?}");
        // The cells hold numbers: native code reads and writes them itself.
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
    }
}

#[test]
fn prototypes_that_take_turns_get_a_side_trace_each_and_stay_installed() {
    for stats in agree_natively(&ALL_PROFILES, &protos(60_000)) {
        // The root and a side for each of the seven other prototypes.
        assert_eq!(stats.side_installs, 7, "{stats:?}");
        assert_eq!(stats.side_aborts, 0, "{stats:?}");
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{stats:?}"
        );
        assert!(stats.iterations > 57_000, "{stats:?}");
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
        // No trace was retired and recorded again.
        assert_eq!(stats.traces_recorded, 1, "{stats:?}");
    }
}

#[test]
fn closures_loaded_out_of_an_array_record_and_run_natively() {
    let source = "
        local function make(i)
            local object = {v = i}
            return function()
                object.v = (object.v + 1) % 1009
                return object.v
            end
        end
        local closures = {}
        for i = 1, 64 do closures[i] = make(i) end
        local total = 0
        for i = 1, 60000 do
            total = (total + closures[i % 64 + 1]()) % 1000003
        end
        return total";
    for stats in agree_natively(&ALL_PROFILES, source) {
        // The load of a closure from the array is no unsupported operand.
        assert_eq!(
            stats.aborts(TraceAbortReason::UnsupportedOperand),
            0,
            "{stats:?}"
        );
        assert_eq!(stats.traces_recorded, 1, "{stats:?}");
        assert!(stats.iterations > 59_000, "{stats:?}");
    }
}

#[test]
fn a_ring_that_is_still_filling_is_recorded_once_it_is_full() {
    // The store into the ring is a store the VM does while the ring fills
    // (the table is no dense list yet). The recording that meets it is
    // retried, not rejected, and the one after the ring is full keeps the
    // store in the trace.
    for stats in agree_natively(&ALL_PROFILES, &retained(200_000, 256)) {
        assert!(stats.traces_recorded >= 1, "{stats:?}");
        assert!(stats.iterations > 170_000, "{stats:?}");
    }
}

#[test]
fn closures_over_tables_read_their_cells_natively() {
    // `opt/retained_slots`: the cell of each closure holds a table. Native
    // code reads it through the closure's row, so no cell helper runs, and
    // the array constructor that escapes into the ring fills its slots in
    // place.
    for stats in agree_natively(&ALL_PROFILES, &retained(300_000, 256)) {
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
        assert!(stats.iterations > 290_000, "{stats:?}");
        assert!(stats.table_new_helper_calls > 290_000, "{stats:?}");
    }
}

#[test]
fn a_table_cell_the_enclosing_frame_reads_and_a_closure_changes_stays_right() {
    // `box` is a local of `run` that both closures capture, open while the
    // loop runs. The loop reads a field through the closures and the frame
    // reads the local itself; the first closure replaces the table in the
    // cell (a write of a table into a cell, which the VM stores), the second
    // only reads it.
    let source = "
        local function run(n)
            local box = {v = 0}
            local fs = {
                function() box = {v = box.v + 1} return box.v end,
                function() return box.v * 2 end,
            }
            local t = 0
            for i = 1, n do
                t = (t + fs[i % 2 + 1]() + box.v) % 1000003
            end
            return t, box.v
        end
        local a, b = run(30000)
        local c, d = run(30000)
        return a, b, c, d";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_cell_the_enclosing_frame_reads_between_traced_loops_holds_what_the_trace_wrote() {
    // Reading an open captured local inside the traced loop itself aborts
    // recording (UnsupportedOperand), so that shape stays in the interpreter
    // (see a_loop_that_reads_a_captured_local_the_callees_write_stays_right).
    // The traceable shape: an inner loop calls closures that write a number
    // cell and a field of a table cell, and the enclosing frame reads both
    // locals after every inner loop, then writes them back for the next one.
    let source = "
        local function run(n)
            local c = 0
            local box = {v = 0}
            local f1 = function(x) c = c + x box.v = box.v + 1 return c end
            local f2 = function(x) c = c - 1 box.v = box.v + 2 return box.v end
            local t = 0
            for round = 1, n do
                for i = 1, 300 do
                    t = (t + f1(i)) % 1000003
                    f1, f2 = f2, f1
                end
                t = (t + c + box.v) % 1000003
                c = c % 97
                box.v = box.v % 89
            end
            return t, c, box.v
        end
        local a, b, e = run(100)
        local d, f, g = run(100)
        return a, b, e, d, f, g";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_table_cell_the_enclosing_frame_replaces_is_read_afresh() {
    // The reverse: the loop of the frame stores a new table into the local
    // the closures read, now and then. Every read must see the table the cell
    // holds, not the one an earlier read found.
    let source = "
        local function run(n)
            local box = {v = 0}
            local fs = {
                function() box.v = box.v + 1 return box.v end,
                function() return box.v * 2 end,
            }
            local t = 0
            for i = 1, n do
                t = (t + fs[i % 2 + 1]()) % 1000003
                if i % 997 == 0 then box = {v = i} end
            end
            return t, box.v
        end
        local a, b = run(40000)
        local c, d = run(40000)
        return a, b, c, d";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_table_cell_closed_after_the_trace_was_made_reads_the_same() {
    // The first drive runs with the cells open in the frame of `run`, which
    // returns afterwards and closes them. The second drive runs the trace
    // again over closed cells.
    let source = "
        local function run(n)
            local box = {v = 0}
            local fs = {
                function() box.v = box.v + 1 return box.v end,
                function() return box.v * 3 end,
            }
            local function drive(m)
                local t = 0
                for i = 1, m do t = (t + fs[i % 2 + 1]()) % 1000003 end
                return t
            end
            local first = drive(n)
            return drive, first, function() return box.v end
        end
        local drive, first, read = run(20000)
        local second = drive(20000)
        local third = drive(20000)
        return first, second, third, read()";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn several_closures_sharing_one_table_cell_see_one_table() {
    let source = "
        local function pair(k)
            local o = {v = k}
            return function() o.v = (o.v + 1) % 1009 return o.v end,
                   function() return o.v end
        end
        local inc = {}
        local get = {}
        for i = 1, 3 do inc[i], get[i] = pair(i) end
        local total = 0
        for i = 1, 30000 do
            total = (total + inc[i % 3 + 1]() * 3 + get[(i + 1) % 3 + 1]() + get[i % 3 + 1]()) % 1000003
        end
        return total, get[1](), get[2](), get[3]()";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 0, "{profile:?} {stats:?}");
    }
}

#[test]
fn closures_made_inside_the_loop_over_a_table_agree() {
    // Making a closure is not recorded (the recorder aborts at the closure
    // op), so the loop that makes them stays in the interpreter. The traced
    // shape is an inner loop that calls fresh closures over fresh tables
    // many times; the outer loop checks results agree.
    let source = "
        local total = 0
        for i = 1, 200 do
            local o = {v = i}
            local f = function() o.v = o.v + 1 return o.v end
            for j = 1, 300 do
                total = (total + f() + o.v) % 1000003
            end
        end
        return total";
    agree_natively(&ALL_PROFILES, source);
}

/// The guard exits (a failed guard of the root or of a side trace) a run took.
fn guard_exits(stats: &TraceStatistics) -> u64 {
    stats.exits_of(TraceExitKind::GuardFail) + stats.side_exits.iter().sum::<u64>()
}

#[test]
fn a_cell_that_stops_holding_a_table_leaves_the_trace() {
    // Once the cell of one closure holds a number, the read of it as a table
    // must not run natively. The check for a number happens in the callee
    // itself, which errors the way the interpreter does. The same program
    // without the flip is the baseline for the exits: the trace must leave
    // more often when a cell stops holding a table.
    let program = |flip_at: i32| {
        format!(
            "
        local function make(i)
            local o = {{v = i}}
            return function(flip)
                if flip then o = i end
                if type(o) == 'table' then
                    o.v = (o.v + 1) % 1009
                    return o.v
                end
                return o
            end
        end
        local fs = {{}}
        for i = 1, 8 do fs[i] = make(i) end
        local total = 0
        for i = 1, 40000 do
            total = (total + fs[i % 8 + 1](i == {flip_at})) % 1000003
        end
        return total"
        )
    };
    // The profiles whose loop traces. Lua 5.1 and LuaJIT abort every
    // recording of it with UnsupportedOperand, so they only agree.
    let tracing = [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ];
    let flipping = program(30001);
    let steady = program(0);
    for profile in ALL_PROFILES {
        agree(profile, &flipping);
    }
    for profile in tracing {
        let (_, with_flip) = agree(profile, &flipping);
        let (_, without) = agree(profile, &steady);
        assert!(
            with_flip.iterations > 0 && without.iterations > 0,
            "{profile:?}: the loop never ran natively\n{with_flip:?}"
        );
        assert!(
            guard_exits(&with_flip) > guard_exits(&without),
            "{profile:?}: the cell stopped holding a table and no guard left the trace\n\
             {} guard exits with the flip, {} without",
            guard_exits(&with_flip),
            guard_exits(&without)
        );
    }
}

#[test]
fn a_table_an_earlier_run_made_and_stored_into_a_cell_gets_one_lane() {
    // Between two runs of the loop the interpreter stores a table the last
    // run made, kept in a ring, into a cell. The getters read that cell as a
    // table through its closure row while the trace makes tables of its own,
    // and the lane of the cell's table must be the one it has as a ring
    // entry or a made table, or a write through one lane would be missed
    // through the other. The getters add to the field they find, and the loop
    // reads the table back through the ring.
    //
    // A table made in the run itself can never sit in a cell whose row is
    // filled natively. Helpers never call Lua (only a coroutine resume does,
    // and a trace that resumes reads cells through helpers), and a table
    // stored into a cell by the trace makes the whole tree read cells through
    // helpers (see `closure_row_words`). So the scan of the made tables in
    // `cell_table_lane` is a guard no script reaches.
    let source = "
        local function make(i)
            local o = {v = i}
            return function() o.v = o.v + 1 return o.v end,
                   function(t) o = t end
        end
        local gets, sets = {}, {}
        for k = 1, 24 do gets[k], sets[k] = make(k) end
        local ring = {}
        for k = 1, 24 do ring[k] = {v = 0} end
        local total = 0
        for round = 1, 40 do
            for i = 1, 3000 do
                local k = i % 24 + 1
                local pair = {v = i}
                ring[k] = pair
                total = (total + gets[k]() + ring[k].v) % 1000003
            end
            sets[round % 24 + 1](ring[round % 24 + 1])
        end
        return total, gets[1](), gets[2]()";
    for stats in agree_natively(&ALL_PROFILES, source) {
        assert!(stats.table_new_helper_calls > 0, "{stats:?}");
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
    }
}

#[test]
fn a_fresh_table_stored_into_an_old_one_while_cells_are_read_natively_is_kept() {
    // Every iteration stores a new table into the ring, an old table, while
    // the collector runs often. The ring is all that keeps the last tables
    // alive: the collector must hear of each store, and the arrays the trace
    // filled in place must hold their values after a full collection.
    let source = "
        local function make(i)
            local object = {v = i}
            return function()
                object.v = (object.v + 1) % 1009
                return object.v
            end
        end
        local closures = {}
        for i = 1, 16 do closures[i] = make(i) end
        local ring = {}
        for i = 1, 64 do ring[i] = {0, 0} end
        collectgarbage()
        if pcall(collectgarbage, 'setpause', 100) then
            collectgarbage('setstepmul', 1000)
        else
            collectgarbage('incremental', 100, 1000)
        end
        local weak = setmetatable({}, {__mode = 'v'})
        local total = 0
        for i = 1, 60000 do
            local pair = {i % 17, i % 23}
            ring[i % 64 + 1] = pair
            weak[i % 32 + 1] = pair
            total = (total + closures[i % 16 + 1]() + pair[1] + pair[2]) % 1000003
        end
        collectgarbage()
        collectgarbage()
        local sum, alive = 0, 0
        for i = 1, 64 do sum = sum + ring[i][1] * 31 + ring[i][2] + #ring[i] end
        for _, v in pairs(weak) do alive = alive + 1 end
        return total, sum, alive";
    for stats in agree_natively(&ALL_PROFILES, source) {
        assert_eq!(stats.closure_helper_calls, 0, "{stats:?}");
    }
}

#[test]
fn array_constructors_that_escape_keep_their_values_length_and_growth() {
    // Short constructors of numbers and booleans fill their slots in place
    // when they escape into the ring. The loop only builds and stores; growth,
    // table.insert/remove, a metatable and the reads happen after it. The
    // tables must read back as the dense lists the interpreter makes.
    let source = "
        local ring = {}
        local mt = {__index = function() return -1 end}
        for i = 1, 40 do ring[i] = {0} end
        for i = 1, 30000 do
            local k = i % 8
            ring[k + 1] = {i}
            ring[k + 9] = {i % 3 == 0, i % 5 == 0}
            ring[k + 17] = {i, i * 0.5, i % 2 == 0, i + 1, i + 2, i + 3, i + 4, i + 5}
        end
        for k = 1, 8 do
            local a, b, c = ring[k], ring[k + 8], ring[k + 16]
            a[2] = k
            b[#b + 1] = k
            c[#c + 1] = k
            if k % 2 == 0 then setmetatable(a, mt) end
            if k % 3 == 0 then table.insert(c, 1, 5) end
            if k % 4 == 0 then table.remove(c) end
        end
        local sum = 0
        for i = 1, 24 do
            local t = ring[i]
            sum = (sum + #t * 7 + (t[1] and 1 or 0)) % 1000003
            for j = 1, #t do
                local v = t[j]
                if type(v) == 'number' then sum = (sum + v) % 1000003
                elseif v then sum = sum + 3 end
            end
        end
        local n = 0
        for _ in pairs(ring[17]) do n = n + 1 end
        return sum, #ring[1], #ring[9], #ring[17], ring[1][3], ring[2][-1], n, select('#', (table.unpack or unpack)(ring[18]))";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_constructor_with_a_late_nil_exits_between_allocation_and_fill() {
    // The element is a number for most iterations and nil once in a while, so
    // the trace leaves after the table is allocated and before it is filled.
    let source = "
        local ring = {}
        for i = 1, 16 do ring[i] = {0} end
        for i = 1, 40000 do
            local v = i
            if i % 4999 == 0 then v = nil end
            ring[i % 16 + 1] = {i, v, 2}
        end
        local sum = 0
        for i = 1, 16 do
            local t = ring[i]
            sum = (sum + (t[1] or 0) * 3 + (t[2] or 1000) + t[3]) % 1000003
        end
        return sum, ring[1][2], ring[2][2], ring[3][2]";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn nine_element_constructors_that_escape_agree_in_a_trace() {
    // Past the fill-in-place limit the constructor takes the ordinary path.
    let source = "
        local ring = {}
        for i = 1, 8 do ring[i] = {0} end
        for i = 1, 30000 do
            ring[i % 8 + 1] = {i, i, i, i, i, i, i, i, i}
        end
        local sum = 0
        for i = 1, 8 do sum = (sum + #ring[i] + ring[i][9]) % 1000003 end
        return sum";
    agree_natively(&ALL_PROFILES, source);
}

// ------------------------------------------------------------- semantics

#[test]
fn closures_sharing_one_cell_see_each_others_writes_within_one_iteration() {
    let source = "
        local function pair()
            local c = 0
            return function() c = c + 1 return c end, function() return c end
        end
        local inc1, get1 = pair()
        local inc2, get2 = pair()
        local total = 0
        for i = 1, 20000 do
            total = (total + inc1() * 3 + get1() + inc2() * 5 + get2()) % 1000003
            inc1, inc2 = inc2, inc1
            get1, get2 = get2, get1
        end
        return total";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_cell_open_in_the_enclosing_frame_holds_what_the_trace_wrote() {
    // `c1` and `c2` are locals of `run` that its two closures capture. The
    // loop writes them through the closures, which take turns at one call
    // site, and `run` reads them after the loop.
    let source = "
        local function run(n)
            local c1, c2 = 0, 100
            local f1 = function(x) c1 = c1 + x return c1 end
            local f2 = function(x) c2 = c2 + 2 * x return c2 end
            local t = 0
            for i = 1, n do
                t = (t + f1(i)) % 1000003
                f1, f2 = f2, f1
            end
            return t, c1, c2
        end
        local a, b, c = run(20000)
        local d, e, f = run(20000)
        return a, b, c, d, e, f";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_loop_that_reads_a_captured_local_the_callees_write_stays_right() {
    // The loop reads `c1` itself while the closures that take turns write it
    // through their cells. A trace may hold the variable in a register or
    // write it through the closure, never both.
    let source = "
        local function run(n)
            local c1, c2 = 0, 0
            local f1 = function(x) c1 = c1 + x return c1 end
            local f2 = function(x) c1 = c1 - 1 c2 = c2 + 1 return c2 end
            local t = 0
            for i = 1, n do
                t = (t + f1(i) + c1) % 1000003
                f1, f2 = f2, f1
            end
            return t, c1, c2
        end
        local a, b, c = run(20000)
        local d, e, f = run(20000)
        return a, b, c, d, e, f";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_cell_the_loop_holds_in_a_window_slot_and_the_callees_share_is_one_variable() {
    // `shared` is an upvalue of `loop`, which the trace keeps in a register,
    // and the cell of every closure that takes turns at the call site. A run
    // that wrote it through the closures would leave the register stale, so
    // such closures stay out of the prototype guard.
    let source = "
        local shared = 0
        local function make(k)
            return function(x) shared = shared + k return x + shared end
        end
        local g1, g2 = make(1), make(2)
        local function loop()
            local t = 0
            for i = 1, 20000 do
                t = (t + g1(i) + shared) % 1000003
                g1, g2 = g2, g1
            end
            return t, shared
        end
        local a, b = loop()
        local c, d = loop()
        return a, b, c, d";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        // The closures stay behind the identity guard: no side trace is
        // recorded for a prototype guard that no run could pass.
        assert!(stats.side_recordings <= 2, "{profile:?} {stats:?}");
        assert!(stats.traces_recorded <= 6, "{profile:?} {stats:?}");
        assert!(
            stats.iterations > 0,
            "{profile:?}: no native iteration ran\n{stats:?}"
        );
    }
}

#[test]
fn two_closures_of_one_prototype_writing_a_cell_the_loop_reads_stay_cheap() {
    let source = "
        local M = {x = 1}
        local function make(k)
            return function(i) M.x = M.x + k return i end
        end
        local g1, g2 = make(1), make(2)
        local s = 0
        for i = 1, 20000 do
            g1(i)
            if i % 2 == 0 then s = (s + M.x) % 1000003 end
            g1, g2 = g2, g1
        end
        return s, M.x";
    for profile in ALL_PROFILES {
        let (_, stats) = agree(profile, source);
        assert!(stats.side_recordings <= 2, "{profile:?} {stats:?}");
        assert!(stats.traces_recorded <= 6, "{profile:?} {stats:?}");
    }
}

#[test]
fn closures_with_environments_of_their_own_are_never_taken_for_one_another() {
    // Two closures of one prototype read a global each through its own
    // environment. The prototype guard cannot tell them apart, so the
    // recording refuses a callee that reads a global.
    let source = "
        local function make()
            return function(x) return x + (G or 0) end
        end
        local f1, f2 = make(), make()
        setfenv(f2, {G = 1000})
        G = 1
        local total = 0
        for i = 1, 20000 do
            total = (total + f1(i)) % 1000003
            f1, f2 = f2, f1
        end
        return total";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        agree(profile, source);
    }
}

#[test]
fn closures_of_one_prototype_keep_the_values_of_their_own_cells() {
    // Each closure starts from another count and another multiplier, and
    // the loop reads the counts back from the closures afterwards.
    let source = "
        local function make(k, start)
            local count = start
            return function(x)
                count = count + x
                return count * k
            end, function() return count end
        end
        local fs, gets = {}, {}
        for i = 1, 6 do fs[i], gets[i] = make(i, i * 1000) end
        local a, b, c, d, e, f = fs[1], fs[2], fs[3], fs[4], fs[5], fs[6]
        local total = 0
        for i = 1, 30000 do
            total = (total + a(i)) % 1000003
            a, b, c, d, e, f = b, c, d, e, f, a
        end
        return total, gets[1](), gets[2](), gets[3](), gets[4](), gets[5](), gets[6]()";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn a_table_stored_into_a_cell_reaches_the_collectors_barrier() {
    // Each call stores the table it is given, a new one every iteration,
    // into the cell of its closure, which is older than the table. The
    // collector must hear of every such store: it frees the tables nothing
    // reaches, and the cells are all that keeps the last ones. The collector
    // is set to run often, so it works while the trace runs (a call of
    // `collectgarbage` would end the recording), and a weak table sees
    // which tables the cells kept.
    let source = "
        local function make()
            local cur = {n = 0}
            return function(t)
                cur = t
                return cur.n + 1
            end, function() return cur.n end
        end
        local weak = setmetatable({}, {__mode = 'v'})
        local fa, ga = make()
        local fb, gb = make()
        local fc, gc = make()
        if pcall(collectgarbage, 'setpause', 100) then
            collectgarbage('setstepmul', 1000)
        else
            collectgarbage('incremental', 100, 1000)
        end
        local total = 0
        for i = 1, 40000 do
            local t = {n = i % 7}
            weak[i % 64 + 1] = t
            total = (total + fa(t)) % 1000003
            fa, fb, fc = fb, fc, fa
            ga, gb, gc = gb, gc, ga
        end
        collectgarbage()
        collectgarbage()
        local alive = 0
        for k, v in pairs(weak) do alive = alive + 1 end
        return total, ga(), gb(), gc(), alive, collectgarbage('count') < 5000";
    let stats = agree_natively(&ALL_PROFILES, source);
    // A table in a cell is the VM's to store, with the barrier.
    assert!(
        stats.iter().any(|stats| stats.closure_helper_calls > 0),
        "{stats:?}"
    );
}

#[test]
fn a_fresh_table_beside_a_looked_up_one_keeps_both_rows() {
    // The loop reads a record out of a sparse number key of a bound table
    // (`TABLE_GET_TABLE`, the shape of Mineclonia's `starts[hash].starts`)
    // and a field of it through its row, which the call's own fill makes
    // this run's. It also builds a fresh table every iteration, reads its
    // field, keeps it in a ring so it is a real table, and calls closures
    // whose cells hold tables. Only the lookup's result gets its row filled
    // after the call: the constructor's and the cell's tables keep the rows
    // they had, and the loop runs natively.
    let source = "
        local function make(i)
            local o = {v = i}
            return function() o.v = (o.v + 1) % 1009 return o.v end
        end
        local gets = {}
        for k = 1, 8 do gets[k] = make(k) end
        local starts = {}
        for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, starts = {k}} end
        local ring = {}
        for k = 1, 16 do ring[k] = {n = 0} end
        local total = 0
        for i = 1, 30000 do
            local key = (i % 64 + 1) * 8191 + 4096
            local fresh = {n = i % 7}
            ring[i % 16 + 1] = fresh
            local record = starts[key]
            total = (total + record.n + record.starts[1] + fresh.n
                + gets[i % 8 + 1]()) % 1000003
        end
        starts[2 * 8191 + 4096] = {n = -5, starts = {-5}}
        for i = 1, 3000 do
            local key = (i % 64 + 1) * 8191 + 4096
            local fresh = {n = i % 5}
            total = (total + starts[key].n + fresh.n) % 1000003
        end
        return total, gets[1](), gets[8](), ring[3].n";
    for stats in agree_natively(&ALL_PROFILES, source) {
        assert!(stats.table_new_helper_calls > 0, "{stats:?}");
        assert!(stats.table_lookup_helper_calls > 0, "{stats:?}");
        // A lookup row left unfilled leaves at the field read on every entry.
        assert!(stats.iterations >= 20_000, "{stats:?}");
    }
}

#[test]
fn a_lookup_row_is_this_runs_whether_the_key_hits_misses_or_is_replaced() {
    // The shapes of the lookup's row fill, under the default backend (MIR
    // fills it as Cranelift does): a record read by two fields, a key that
    // is absent on some entries, a record replaced by another shape under
    // the compiled loop, and a lookup of a lookup.
    let hit = "
        local starts = {}
        for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end
        local s = 0
        for i = 1, 100000 do
            local record = starts[(i % 64 + 1) * 8191 + 4096]
            s = (s + record.n + record.m) % 1000003
        end
        return s";
    let miss = "
        local starts = {}
        for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end
        local s = 0
        for i = 1, 100000 do
            local record = starts[(i % 70 + 1) * 8191 + 4096]
            if record then s = (s + record.n + record.m) % 1000003 else s = s + 1 end
        end
        return s";
    let replaced = "
        local starts = {}
        for k = 1, 64 do starts[k * 8191 + 4096] = {n = k, m = k * 2} end
        local s = 0
        for i = 1, 100000 do
            if i % 20000 == 0 then
                starts[(i % 64 + 1) * 8191 + 4096] = {m = i, n = -i, extra = 1}
            end
            local record = starts[(i % 64 + 1) * 8191 + 4096]
            s = (s + record.n + record.m) % 1000003
        end
        return s";
    let nested = "
        local grid = {}
        for a = 1, 16 do
            local row = {}
            for b = 1, 16 do row[b * 4099 + 17] = {n = a + b, m = a * b} end
            grid[a * 8191 + 4096] = row
        end
        local s = 0
        for i = 1, 100000 do
            local row = grid[(i % 16 + 1) * 8191 + 4096]
            local cell = row[(math.floor(i / 16) % 16 + 1) * 4099 + 17]
            s = (s + cell.n + cell.m) % 1000003
        end
        return s";
    for source in [hit, miss, replaced, nested] {
        for stats in agree_natively(&ALL_PROFILES, source) {
            assert!(stats.table_lookup_helper_calls > 0, "{stats:?}");
            // A row left unfilled leaves at the field read on every entry.
            assert!(stats.iterations >= 50_000, "{stats:?}\n{source}");
        }
    }
}

#[test]
fn a_string_in_a_cell_is_the_interpreters_business_and_stays_right() {
    let source = "
        local function make(s)
            local acc = s
            return function(x)
                if x % 1000 == 0 then acc = acc .. 'x' end
                return #acc + x
            end
        end
        local a, b = make('a'), make('bb')
        local total = 0
        for i = 1, 20000 do
            total = (total + a(i)) % 1000003
            a, b = b, a
        end
        return total";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn debug_setupvalue_between_calls_is_seen_by_the_next_run() {
    let source = "
        local function make(k)
            local c = 0
            return function(x) c = c + 1 return x * k + c end
        end
        local f1, f2 = make(1), make(2)
        local total = 0
        for i = 1, 8000 do
            if i == 3000 then debug.setupvalue(f1, 1, 1000) end
            total = (total + f1(i)) % 1000003
            f1, f2 = f2, f1
        end
        return total";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn debug_upvaluejoin_between_calls_is_seen_by_the_next_run() {
    // After the join the two closures share one cell: the trace that ran
    // before it holds an address for each, and a run after it must hold the
    // new ones.
    let source = "
        local function make(k)
            local c = 0
            return function(x) c = c + 1 return x * k + c end
        end
        local f1, f2 = make(1), make(2)
        local total = 0
        for i = 1, 8000 do
            if i == 3000 then debug.upvaluejoin(f1, 1, f2, 1) end
            total = (total + f1(i)) % 1000003
            f1, f2 = f2, f1
        end
        return total";
    agree_natively(&HAS_UPVALUEJOIN, source);
}

/// Entry remembers the cells it last checked for aliasing and passes a frame
/// that names the same ones without checking again. A join between two
/// entries makes two upvalues of the loop's own function one variable: the
/// next entry must see it, or the trace would keep the two in separate
/// native values and lose a write.
#[test]
fn a_join_between_entries_of_the_loops_function_is_seen_by_the_next_entry() {
    let source = "
        local a, b, c = 0, 0, 0
        local function inner(n)
            for i = 1, n do
                a = a + 1
                b = b + a
                c = c + b % 7
            end
            return a + b + c
        end
        local total = 0
        for round = 1, 300 do
            if round == 150 then debug.upvaluejoin(inner, 2, inner, 1) end
            total = (total + inner(40)) % 1000003
        end
        return total, a, b, c";
    agree_natively(&HAS_UPVALUEJOIN, source);
}

/// The cells of a callee the trace pinned are no cells of the frame's own: a
/// join of two of them between entries must reach the next entry too, which
/// then keeps them as one variable or leaves them to the interpreter.
#[test]
fn a_join_of_a_pinned_callees_upvalues_between_entries_is_seen() {
    let source = "
        local x, y, z = 0, 0, 0
        local function step(i)
            x = x + i
            y = y + x
            z = z + y % 5
        end
        local function inner(n)
            for i = 1, n do step(i) end
            return x + y + z
        end
        local total = 0
        for round = 1, 300 do
            if round == 150 then debug.upvaluejoin(step, 2, step, 1) end
            total = (total + inner(40)) % 1000003
        end
        return total, x, y, z";
    agree_natively(&HAS_UPVALUEJOIN, source);
}

/// Two closures of one prototype take turns at entering the loop's trace,
/// and one of them has its two upvalues joined: entry remembers the cells of
/// one closure and must check the other's, every time they alternate.
#[test]
fn closures_of_one_prototype_with_and_without_a_join_alternate_at_entry() {
    let source = "
        local function make()
            local a, b = 0, 0
            return function(n)
                for i = 1, n do
                    a = a + 1
                    b = b + a
                end
                return a * 1000 + b
            end
        end
        local f, g = make(), make()
        debug.upvaluejoin(g, 2, g, 1)
        local total = 0
        for round = 1, 300 do
            total = (total + f(30) + g(30)) % 1000003
        end
        return total";
    agree_natively(&HAS_UPVALUEJOIN, source);
}

#[test]
fn a_callee_that_becomes_a_native_function_or_a_callable_table_leaves_the_trace() {
    let source = "
        local function make(k)
            local c = 0
            return function(x) c = c + 1 return x * k + c end
        end
        local g1, g2, g3 = make(1), make(2), make(3)
        local callable = setmetatable({}, {__call = function(self, x) return x + 7 end})
        local total = 0
        for i = 1, 12000 do
            if i == 4000 then g2 = math.abs end
            if i == 8000 then g3 = callable end
            total = (total + g1(i)) % 1000003
            g1, g2, g3 = g2, g3, g1
        end
        return total";
    agree_natively(&ALL_PROFILES, source);
}

#[test]
fn an_array_slot_that_stops_holding_a_function_raises_the_interpreters_error() {
    let source = "
        local function make(k)
            local c = 0
            return function(x) c = c + 1 return x * k + c end
        end
        local closures = {}
        for i = 1, 8 do closures[i] = make(i) end
        local ok, err = pcall(function()
            local total = 0
            for i = 1, 20000 do
                if i == 15000 then closures[i % 8 + 1] = 42 end
                total = (total + closures[i % 8 + 1](i)) % 1000003
            end
            return total
        end)
        return ok, err";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("attempt to call"), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn an_error_inside_a_callee_makes_the_traceback_the_interpreter_makes() {
    let source = "
        local function make(k)
            local c = 0
            return function(x)
                c = c + 1
                if x == 15000 then error('boom ' .. c) end
                return x * k + c
            end
        end
        local g1, g2, g3, g4 = make(1), make(2), make(3), make(4)
        local function loop()
            local total = 0
            for i = 1, 20000 do
                total = (total + g1(i)) % 1000003
                g1, g2, g3, g4 = g2, g3, g4, g1
            end
            return total
        end
        local ok, trace = xpcall(loop, debug.traceback)
        return ok, trace";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("boom"), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn collected_memory_stays_bounded_while_closures_rotate() {
    // The trace keeps no object alive between runs: the closures a loop
    // drops are collected.
    let source = "
        local function make(k)
            local c = {n = 0}
            return function(x) c.n = c.n + 1 return x * k + c.n end
        end
        local total = 0
        for round = 1, 30 do
            local a, b, c = make(1), make(2), make(3)
            for i = 1, 2000 do
                total = (total + a(i)) % 1000003
                a, b, c = b, c, a
            end
        end
        collectgarbage()
        return total, collectgarbage('count') < 3000";
    for profile in ALL_PROFILES {
        let (result, _) = agree(profile, source);
        assert!(result.ends_with("Boolean(true)]"), "{profile:?}: {result}");
    }
}
