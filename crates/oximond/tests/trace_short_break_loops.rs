//! Numeric for loops with a long range that a `break` leaves after a trip or
//! two. Every recording of such a loop starts on a trip that leaves it, so
//! its own root aborts with `LoopNotClosed` until the policy gives it up as
//! short. The loop around it then unrolls the trips each iteration takes, as
//! it does for a short `while` loop, whatever the range has left: the
//! `break` is a guard like any other. A side of that tree that meets a run
//! of the loop with more trips than unrolling takes leaves the rest to a
//! loop side. The shape is the column scan of Mineclonia's `build_heightmap`
//! (`mg_ersatz.lua:122`), which looks down from the top of each column for
//! the first solid node and finds it on the first trip under the ground.
//!
//! Every program runs traced and interpreted in every profile and must give
//! the same results, also when the data changes under the trace: columns of
//! other heights, nil holes, a metatable that appears, values of another
//! type, another table, an error, a count hook.

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

/// What a traced run did.
struct Seen {
    profile: CompatibilityProfile,
    compiled: u64,
    iterations: u64,
    loop_sides: u64,
    not_closed: u64,
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
            assert_eq!(actual, expected, "{profile:?} {source}");
            let stats = lua.vm().jit_statistics().trace;
            Seen {
                profile,
                compiled: stats.traces_compiled,
                iterations: stats.iterations,
                loop_sides: stats.loop_sides,
                not_closed: stats.aborts(TraceAbortReason::LoopNotClosed),
                stats: format!("{stats:?}"),
            }
        })
        .collect()
}

/// The column scan and its driver. `columns` columns of `HEIGHT` cells each
/// sit in `cells` (column `x` at `x * HEIGHT + 1 ..`); `scan` looks down
/// each column from the top for the first non-air cell and the first
/// walkable one, as `build_heightmap` does, and `rounds` calls of `scan` run
/// with `before(round)` ahead of each. Air is 126, stone 230, water 412 and
/// dirt 231; water is not walkable.
fn scan_program(before: &str, rounds: u32) -> String {
    format!(
        "local HEIGHT, COLUMNS = 40, 64
         local air, stone, water, dirt = 126, 230, 412, 231
         local walkable = {{}}
         for cid = 0, 500 do walkable[cid] = cid ~= air and cid ~= water end
         local function walkable_p (cid)
             return walkable[cid]
         end
         local mathmax = math.max
         -- Column x: `top` cells of air, then stone.
         local function column (cells, x, top)
             for y = 1, HEIGHT do
                 cells[x * HEIGHT + y] = y > HEIGHT - top and air or stone
             end
         end
         local function ground (top_of)
             local cells = {{}}
             for x = 0, COLUMNS - 1 do column (cells, x, top_of (x)) end
             return cells
         end
         local cells = ground (function (x) return 0 end)
         local function scan (cells, ytop, ybot)
             local total = 0
             for x = 0, COLUMNS - 1 do
                 local base = x * HEIGHT
                 local motion_blocking = -1
                 local world_surface = -1
                 for y = ytop, ybot, -1 do
                     local cid = cells[base + y]
                     if cid ~= air then
                         world_surface = mathmax (world_surface, y)
                     end
                     if walkable_p (cid) then
                         motion_blocking = mathmax (motion_blocking, y)
                     end
                     if world_surface >= 0 and motion_blocking >= 0 then
                         break
                     end
                 end
                 total = total + world_surface * 64 + motion_blocking
             end
             return total
         end
         local sum = 0
         for round = 1, {rounds} do
             {before}
             sum = (sum + scan (cells, HEIGHT, 1)) % 1000003
         end
         return sum, rawget (_G, 'count')"
    )
}

#[test]
fn a_scan_that_breaks_on_its_first_trip_runs_in_the_loop_around_it() {
    // Every column is solid to the top: the scan breaks on its first trip
    // with 39 more in its range. Where its recordings start on that trip
    // (the profiles whose numeric for loops count their entry as a
    // backedge), its own root never closes and is given up; the column loop
    // unrolls the one trip. Either way nearly every column runs native.
    let rounds = 400;
    let mut given_up = 0;
    for seen in run_all(&scan_program("", rounds)) {
        let Seen {
            profile,
            compiled,
            iterations,
            not_closed,
            stats,
            ..
        } = seen;
        let columns = u64::from(rounds) * 64;
        given_up += u32::from(not_closed >= 1);
        assert!(compiled >= 1, "{profile:?} {stats}");
        assert!(
            iterations * 10 >= columns * 9,
            "{profile:?}: {iterations} of {columns} columns ran native: {stats}"
        );
    }
    assert!(given_up >= 1, "no profile gave the scan up");
}

#[test]
fn a_cave_under_the_top_makes_a_few_columns_run_more_trips() {
    // One column in seven has two or three cells of air on top: the trips
    // of the scan change from column to column, and the break leaves on the
    // first, the third or the fourth.
    let source = scan_program(
        "if round == 1 then
             cells = ground (function (x) return x % 7 == 0 and 2 + x % 2 or 0 end)
         end",
        400,
    );
    for seen in run_all(&source) {
        let columns = 400 * 64;
        assert!(seen.compiled >= 1, "{:?} {}", seen.profile, seen.stats);
        assert!(
            seen.iterations * 10 >= columns * 9,
            "{:?}: {} of {columns} columns ran native: {}",
            seen.profile,
            seen.iterations,
            seen.stats
        );
    }
}

#[test]
fn columns_that_run_long_later_take_a_loop_side() {
    // Solid columns first, so the scan is given up as short and unrolled
    // into the column loop. Then the ground sinks: each column runs 20 to 35
    // trips of air, more than unrolling takes, and the side of the break's
    // guard leaves them to a loop side. Then all air: the scan runs out its
    // range without a break.
    let source = scan_program(
        "if round == 200 then
             cells = ground (function (x) return 20 + x % 16 end)
         elseif round == 400 then
             cells = ground (function (x) return HEIGHT end)
         end",
        600,
    );
    let mut given_up = 0;
    for seen in run_all(&source) {
        assert!(seen.compiled >= 1, "{:?} {}", seen.profile, seen.stats);
        // Where the scan's own root closes (it starts on a trip that goes
        // on), it is a tree of its own and needs no loop side.
        if seen.not_closed >= 1 {
            given_up += 1;
            assert!(
                seen.loop_sides >= 1,
                "{:?}: no loop side for the long columns: {}",
                seen.profile,
                seen.stats
            );
        }
        // The long columns run hundreds of thousands of trips; most of them
        // must be native.
        assert!(
            seen.iterations >= 200_000,
            "{:?}: {} native iterations: {}",
            seen.profile,
            seen.iterations,
            seen.stats
        );
    }
    assert!(given_up >= 1, "no profile gave the scan up");
}

#[test]
fn water_on_top_is_not_walkable_and_the_scan_goes_on_under_it() {
    // The first trip finds water: the surface is set and the scan goes on
    // to the stone under it, two to four trips down.
    run_all(&scan_program(
        "if round == 100 then
             for x = 0, COLUMNS - 1 do
                 for y = HEIGHT - x % 3, HEIGHT do cells[x * HEIGHT + y] = water end
             end
         end",
        400,
    ));
}

#[test]
fn nil_holes_in_the_column() {
    // A nil is not air and not walkable: the scan sets the surface and
    // goes on.
    run_all(&scan_program(
        "if round == 150 then
             for x = 0, COLUMNS - 1, 3 do cells[x * HEIGHT + HEIGHT] = nil end
         end",
        400,
    ));
}

#[test]
fn a_metatable_appears_on_the_cells_while_the_scan_runs() {
    // Holes read through `__index` once the metatable is there: the trace
    // read the table raw and must leave before it.
    run_all(&scan_program(
        "if round == 150 then
             for x = 0, COLUMNS - 1, 2 do cells[x * HEIGHT + HEIGHT] = nil end
             setmetatable (cells, {__index = function (t, k) return water end})
         end",
        400,
    ));
}

#[test]
fn cells_of_another_type_appear() {
    // A float, a string and a boolean where the trace saw integers.
    run_all(&scan_program(
        "if round == 120 then cells[HEIGHT] = 230.5 end
         if round == 200 then cells[2 * HEIGHT] = 'stone' end
         if round == 280 then cells[3 * HEIGHT] = true end",
        400,
    ));
}

#[test]
fn the_scan_reads_another_table_every_round() {
    // Two tables of other shapes, swapped on every round.
    run_all(&scan_program(
        "if round == 1 then
             solid = cells
             caves = ground (function (x) return x % 5 end)
         end
         if round % 2 == 0 then cells = caves else cells = solid end",
        400,
    ));
}

#[test]
fn an_error_in_the_scan_is_caught() {
    run_all(
        "local HEIGHT = 40
         local walkable = setmetatable ({}, {__index = function (t, cid)
             if cid == 999 then error ('bad cell ' .. cid) end
             return cid ~= 126
         end})
         local cells = {}
         for i = 1, 64 * HEIGHT do cells[i] = 230 end
         local function scan ()
             local n = 0
             for x = 0, 63 do
                 for y = HEIGHT, 1, -1 do
                     local cid = cells[x * HEIGHT + y]
                     if walkable[cid] then n = n + y; break end
                 end
             end
             return n
         end
         local sum, errors = 0, 0
         for round = 1, 400 do
             if round == 300 then cells[5 * HEIGHT + HEIGHT] = 999 end
             local ok, n = pcall (scan)
             if ok then sum = sum + n else errors = errors + 1 end
         end
         return sum, errors",
    );
}

#[test]
fn a_count_hook_counts_the_scan_as_the_interpreter_does() {
    run_all(&scan_program(
        "if round == 1 then
             count = 0
             debug.sethook (function () count = count + 1 end, '', 97)
         end
         if round == 300 then debug.sethook () end",
        400,
    ));
}
