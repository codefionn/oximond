//! The shapes of Mineclonia's mineshaft corridor loops (`mcl_levelgen`'s
//! mineshaft.lua:299 and :1045): reads keyed by a table, a stateful iterator
//! whose rows end inside the loop body, and a branch on an argument that
//! alternates between calls.

use oximond::runtime::Value;
use oximond::vm::{TraceAbortReason, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, LuaJitExtension};

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
        .luajit_extension(LuaJitExtension::Bit)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` in both modes in every profile, checks that they agree, and
/// gives the tracing run's statistics to `check`.
fn agree(source: &str, check: impl Fn(CompatibilityProfile, &TraceStatistics)) {
    for profile in ALL {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert!(expected.is_ok(), "{profile:?}: {expected:?}");
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "{profile:?} {source}"
        );
        let stats = lua.vm().jit_statistics().trace;
        if std::env::var_os("MINESHAFT_PRINT").is_some() {
            eprintln!(
                "{profile:?} compiled={} entries={} iterations={} unsupported={}",
                stats.traces_compiled,
                stats.native_entries,
                stats.iterations,
                stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()],
            );
        }
        check(profile, &stats);
    }
}

fn unsupported(stats: &TraceStatistics) -> u64 {
    stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()]
}

/// `ordinal_sturdy[shape]` (nodeprops.lua:705): a cache keyed by shape
/// tables, read in a loop through a call. The read records as a lookup call,
/// and the loop runs natively.
#[test]
fn a_read_keyed_by_a_table_runs_natively() {
    let source = r"
        local shapes = {}
        for i = 1, 4 do shapes[i] = { id = i } end
        local cache = {}
        for i = 1, 4 do cache[shapes[i]] = i * 16 end
        local function get (shape) return cache[shape] end
        local s = 0
        for i = 1, 40000 do s = s + get (shapes[i % 4 + 1]) end
        return s
    ";
    agree(source, |profile, stats| {
        assert_eq!(unsupported(stats), 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 30_000, "{profile:?} {stats:?}");
    });
}

/// A function as the key, and a table read out of an array as the
/// receiver.
#[test]
fn a_read_keyed_by_a_function_from_a_table_met_by_lane_runs_natively() {
    let source = r"
        local fns = {}
        for i = 1, 3 do fns[i] = function () return i end end
        local holders = {}
        for i = 1, 2 do
            local t = {}
            for j = 1, 3 do t[fns[j]] = i * 10 + j end
            holders[i] = t
        end
        local s = 0
        for i = 1, 30000 do
            s = s + holders[i % 2 + 1][fns[i % 3 + 1]]
        end
        return s
    ";
    agree(source, |profile, stats| {
        assert_eq!(unsupported(stats), 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 20_000, "{profile:?} {stats:?}");
    });
}

/// The exits of a read keyed by a table: a key that goes missing (nil, then
/// an `__index` function), a value of another type, and a string key and a
/// number key at the same instruction. Each leaves before the read and the
/// interpreter gives the result.
#[test]
fn a_read_keyed_by_a_table_leaves_when_the_key_or_value_changes() {
    let source = r#"
        local shapes = {}
        for i = 1, 4 do shapes[i] = { id = i } end
        local cache = {}
        for i = 1, 4 do cache[shapes[i]] = i end
        local keys = { shapes[1], shapes[2], shapes[3], shapes[4] }
        local s = 0
        for i = 1, 6000 do
            if i == 2000 then cache[shapes[2]] = nil end
            if i == 3000 then cache[shapes[3]] = "abc" end
            if i == 4000 then
                setmetatable (cache, { __index = function (t, k) return 100 end })
            end
            if i == 5000 then
                keys[1] = "x"
                cache.x = 5
                keys[4] = 7
                cache[7] = 9
            end
            local v = cache[keys[i % 4 + 1]]
            if type (v) == "number" then
                s = s + v
            elseif type (v) == "string" then
                s = s + #v
            else
                s = s + 1000
            end
        end
        return s
    "#;
    agree(source, |profile, stats| {
        assert!(stats.iterations > 1000, "{profile:?} {stats:?}");
    });
}

/// A store under a table key is still the interpreter's, and a read after it
/// sees it.
#[test]
fn a_store_under_a_table_key_stays_correct() {
    let source = r"
        local shapes = {}
        for i = 1, 8 do shapes[i] = {} end
        local cache = {}
        local s = 0
        for i = 1, 3000 do
            local shape = shapes[i % 8 + 1]
            local v = cache[shape]
            if v then
                s = s + v
            else
                cache[shape] = i
            end
            if i % 500 == 0 then cache[shapes[math.floor (i / 500) % 8 + 1]] = nil end
        end
        return s
    ";
    agree(source, |_, _| {});
}

/// `ipos3` (mineshaft.lua:293): a stateful iterator over the cells of a box.
/// The row advance is a branch inside the iterator, and the last row ends
/// the loop at the same branch. Its side records on a row that goes on.
#[test]
fn a_box_iterator_runs_its_rows_natively() {
    let source = r"
        local function ipos3 (x1, y1, z1, x2, y2, z2)
            local x, y, z = x1 - 1, y1, z1
            return function ()
                x = x + 1
                if x > x2 then
                    x = x1
                    z = z + 1
                    if z > z2 then
                        z = z1
                        y = y + 1
                        if y > y2 then return nil end
                    end
                end
                return x, y, z
            end
        end
        local function fill (chance, length)
            local s = 0
            for x, y, z in ipos3 (0, 0, 0, 2, 1, length) do
                if chance == 1.0 or (x + y + z) % 5 ~= 0 then
                    s = s + x * 3 + y * 5 + z
                end
            end
            return s
        end
        local total = 0
        for r = 1, 30 do
            total = total + fill (1.0, 9 + r % 7)
            total = total + fill (0.8, 4 + r % 5)
        end
        return total
    ";
    agree(source, |profile, stats| {
        assert!(stats.iterations > 2_000, "{profile:?} {stats:?}");
    });
}

/// `fill_box_rotated` (mineshaft.lua:288): corridors pass an rng table and
/// staircases pass nil, in bursts. The loop keeps a version for each, and
/// the sums agree with the interpreter when an rng appears, disappears, or
/// is swapped for another table between calls.
#[test]
fn a_box_loop_called_with_an_rng_table_and_with_nil_runs_natively() {
    let source = r"
        local function make_rng (seed)
            local rng = { seed = seed }
            function rng:next_float ()
                self.seed = (self.seed * 1103515245 + 12345) % 2147483648
                return self.seed / 2147483648
            end
            return rng
        end
        local cells = {}
        local function fill (length, chance, rng)
            for z = 0, length * 3 do
                if chance == 1.0 or (rng:next_float () <= chance) then
                    cells[z] = (cells[z] or 0) + 1
                end
            end
        end
        local rngs = { make_rng (1), make_rng (7) }
        for r = 1, 400 do
            if r % 9 < 4 then
                -- A staircase: no rng.
                fill (6 + r % 3, 1.0, nil)
            else
                local rng = rngs[r % 2 + 1]
                fill (14 + r % 5, 1.0, rng)
                fill (14 + r % 5, 0.8, rng)
            end
        end
        local s = 0
        for key, count in pairs (cells) do s = s + key * count end
        return s, rngs[1].seed, rngs[2].seed
    ";
    agree(source, |profile, stats| {
        assert!(stats.traces_compiled >= 2, "{profile:?} {stats:?}");
        assert!(stats.iterations > 10_000, "{profile:?} {stats:?}");
    });
}
