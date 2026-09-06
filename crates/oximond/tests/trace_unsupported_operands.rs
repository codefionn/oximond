//! The operand shapes that made Mineclonia's structure loops abort with
//! `UnsupportedOperand` on the emerge (`mcl_levelgen`'s structures.lua):
//! a table read out of the hash part under a sparse number key
//! (`starts[hash].starts` at :856), `math.max` or `math.min` of more than
//! two numbers (`mathmax (0, a, b)` at :1271), and `t[nil]`
//! (`static_node_shapes[cid]` at nodeprops.lua:410).

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
        if std::env::var_os("UNSUPPORTED_PRINT").is_some() {
            eprintln!(
                "{profile:?} compiled={} entries={} iterations={} unsupported={}",
                stats.traces_compiled,
                stats.native_entries,
                stats.iterations,
                unsupported(&stats),
            );
        }
        check(profile, &stats);
    }
}

fn unsupported(stats: &TraceStatistics) -> u64 {
    stats.recording_aborts[TraceAbortReason::UnsupportedOperand.index()]
}

fn native(profile: CompatibilityProfile, stats: &TraceStatistics, trips: u64) {
    assert_eq!(unsupported(stats), 0, "{profile:?} {stats:?}");
    assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
    assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    assert!(stats.iterations > trips, "{profile:?} {stats:?}");
}

/// `starts[hash].starts` (structures.lua:856): records of a cache keyed by
/// a sparse number (`struct_hash`, a chunk position shifted into one
/// number) sit in the hash part. The read gives the record's lane through a
/// lookup call, the fields of the record read through its row, and the loop
/// runs natively.
#[test]
fn a_table_under_a_sparse_number_key_is_read_natively() {
    let source = r"
        local function hash (x, z) return (x + 4096) * 8192 + (z + 4096) end
        local starts = {}
        for x = -8, 8 do
            for z = -8, 8 do
                starts[hash (x, z)] = { n = x * 3 + z, starts = { x, z } }
            end
        end
        local s = 0
        for i = 1, 40000 do
            local record = starts[hash (i % 17 - 8, i % 5 - 2)]
            s = s + record.n + record.starts[2]
        end
        return s
    ";
    agree(source, |profile, stats| native(profile, stats, 30_000));
}

/// A float key in every profile (`struct_hash` under `bit.bor` gives `LuaJIT`
/// and Lua 5.1 a float; a division gives one in Lua 5.3 and later), and a
/// function and a string under such keys.
#[test]
fn a_function_or_string_under_a_float_hash_key_is_read_natively() {
    let source = r"
        local fns, names = {}, {}
        for i = 1, 8 do
            fns[i * 1000 + 0.5] = function () return i end
            names[i * 1000 + 0.5] = 'n' .. i
        end
        local s = 0
        for i = 1, 20000 do
            local k = (i % 8 + 1) * 1000 + 0.5
            local f = fns[k]
            local name = names[k]
            if name then s = s + f () end
        end
        return s
    ";
    agree(source, |profile, stats| native(profile, stats, 15_000));
}

/// The exits of the read: a key that goes missing (nil, then an `__index`
/// function), a value of another type, the table swapped for another one
/// mid-loop, and a metatable that appears while the key is present. Each
/// leaves before the read, and the interpreter gives the result.
#[test]
fn a_table_under_a_sparse_number_key_leaves_when_the_slot_changes() {
    let source = r#"
        local function hash (x) return (x + 4096) * 8192 + 7 end
        local starts = {}
        for x = 0, 7 do starts[hash (x)] = { n = x } end
        local other = {}
        for x = 0, 7 do other[hash (x)] = { n = 100 + x } end
        local s = 0
        for i = 1, 8000 do
            if i == 1500 then starts[hash (2)] = nil end
            if i == 2500 then starts[hash (3)] = 42 end
            if i == 3500 then
                setmetatable (starts, { __index = function (t, k) return { n = 1000 } end })
            end
            if i == 5000 then starts = other end
            if i == 6500 then
                setmetatable (starts, { __index = function () return { n = -1 } end })
                starts[hash (5)] = "abc"
            end
            local record = starts[hash (i % 8)]
            if type (record) == "table" then
                s = s + record.n
            elseif type (record) == "number" then
                s = s + record * 2
            elseif type (record) == "string" then
                s = s + #record
            else
                s = s + 7
            end
        end
        return s
    "#;
    // The `__index` function's fresh table is no lane to read a field of,
    // so a recording past the metatable may refuse; the runs before it
    // leave at the read.
    agree(source, |profile, stats| {
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 1000, "{profile:?} {stats:?}");
    });
}

/// `mathmax (0, piece_bbox[1] - x, x - piece_bbox[4])` (structures.lua:1271):
/// three and four numbers fold the binary compare from the left.
#[test]
fn math_max_and_min_of_more_than_two_numbers_run_natively() {
    let source = r"
        local mathmax, mathmin = math.max, math.min
        local bbox = { 3, 0, 5, 9, 4, 12 }
        local s = 0
        for i = 1, 30000 do
            local x = i % 17
            local dx = mathmax (0, bbox[1] - x, x - bbox[4])
            local dz = mathmin (bbox[3] - x, x - bbox[6], 4, x - 2)
            local df = mathmax (x * 0.5, 1.5, x * 0.25)
            s = s + dx * 3 + dz + df
        end
        return s
    ";
    agree(source, |profile, stats| native(profile, stats, 25_000));
}

/// Integers in Lua 5.3 and later fold the integer compare and stay
/// integers (a mix of integers and floats still refuses: the result's
/// subtype is the chosen argument's); floats keep their order rules: NaN in any position, and `0`
/// against `-0`, where `LuaJIT` takes the later of equal values and the
/// others the earlier.
#[test]
fn math_max_and_min_of_many_numbers_keep_the_natives_order_rules() {
    let source = r"
        local nan = 0 / 0
        local z, nz = 0.0, -0.0
        local rows = {
            { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 }, { 2.0, 3.0, 1.0 },
            { nan, 1.0, 2.0 }, { 1.0, nan, 2.0 }, { 1.0, 2.0, nan },
            { z, nz, z }, { nz, z, nz }, { z, z, nz },
        }
        local out = {}
        for i = 1, 36000 do
            local r = rows[i % 9 + 1]
            local a, b, c = r[1], r[2], r[3]
            local hi, lo = math.max (a, b, c), math.min (a, b, c)
            local hi4, lo4 = math.max (a, b, c, a), math.min (c, b, a, c)
            if i > 35991 then
                out[#out + 1] = tostring (hi) .. ' ' .. tostring (1 / hi)
                out[#out + 1] = tostring (lo) .. ' ' .. tostring (1 / lo)
                out[#out + 1] = tostring (hi4) .. ' ' .. tostring (lo4)
            end
        end
        local ints = 0
        for i = 1, 36000 do
            ints = ints + math.max (i % 5, 7 - i % 11, 2) - math.min (i % 3, i % 7, 5)
        end
        out[#out + 1] = tostring (ints)
        return table.concat (out, ',')
    ";
    agree(source, |profile, stats| {
        assert_eq!(unsupported(stats), 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 60_000, "{profile:?} {stats:?}");
    });
}

/// An argument that turns into a string or a table mid-loop: the guards of
/// the call leave, and the interpreter converts the string or raises the
/// error with its message.
#[test]
fn math_max_of_many_numbers_leaves_when_an_argument_changes_type() {
    let source = r"
        local args = { 1, 2, 3 }
        local s = 0
        for i = 1, 40000 do
            if i == 20000 then args[2] = '10' end
            s = s + math.max (args[1], args[2], args[3] + i % 3)
        end
        local ok, err = pcall (function ()
            local t = { 1, 2, 3 }
            local u = 0
            for i = 1, 40000 do
                if i == 20000 then t[3] = {} end
                u = u + math.min (t[1], t[2], t[3])
            end
            return u
        end)
        return s, ok, tostring (err):gsub ('^.-:%d+: ', '')
    ";
    // The recordings after the string appears refuse the call: a string is
    // no number the native compare takes.
    agree(source, |profile, stats| {
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 30_000, "{profile:?} {stats:?}");
    });
}

/// `for _, start in ipairs (local_starts)` (structures.lua:857) inside the
/// `z` loop of `collect_structure_references`: most chunks hold no start,
/// so the inner loop's first step ends it. The recording of the outer loop
/// follows that end as it follows an `if`, behind a check that the key is
/// still absent from a table without a metatable, and the outer loop runs
/// natively. A list that holds starts leaves at that check.
#[test]
fn an_outer_loop_runs_natively_past_an_empty_inner_ipairs() {
    let source = r"
        local function hash (x, z) return (x + 4096) * 8192 + (z + 4096) end
        local starts = {}
        for x = -12, 12 do
            for z = -12, 12 do
                local list = {}
                if (x * 7 + z * 3) % 11 == 0 then list[1] = { n = x + z } end
                starts[hash (x, z)] = { starts = list }
            end
        end
        local function collect (cx, cz)
            local s = 0
            for x = cx - 3, cx + 3 do
                for z = cz - 3, cz + 3 do
                    local local_starts = starts[hash (x, z)].starts
                    for _, start in ipairs (local_starts) do
                        s = s + start.n
                    end
                end
            end
            return s
        end
        local total = 0
        for i = 1, 3000 do total = total + collect (i % 9 - 4, i % 7 - 3) end
        return total
    ";
    agree(source, |profile, stats| native(profile, stats, 60_000));
}

/// The exits of the skipped inner loop: an empty list that gains an element
/// (the check leaves before the step), and one that gains a metatable
/// whose `__index` gives an element in Lua 5.3 and later (`ipairs` honours
/// it there) or none before.
#[test]
fn a_skipped_inner_ipairs_leaves_when_its_list_changes() {
    let source = r"
        local lists = {}
        for i = 1, 8 do lists[i] = {} end
        local s = 0
        for i = 1, 30000 do
            if i == 10000 then lists[3][1] = 5 end
            if i == 20000 then
                setmetatable (lists[5], { __index = function (t, k)
                    if k == 1 then return 7 end
                end })
            end
            for _, v in ipairs (lists[i % 8 + 1]) do s = s + v end
            s = s + 1
        end
        return s
    ";
    agree(source, |profile, stats| {
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.iterations > 9_000, "{profile:?} {stats:?}");
    });
}

/// `static_node_shapes[cid]` (Mineclonia's nodeprops.lua:410), where
/// `get_block` gives no content id outside the chunk: `t[nil]` of a table
/// without a metatable is nil. The loop runs natively, both for a bound
/// table and for one read out of another table (met by lane).
#[test]
fn a_read_under_a_nil_key_runs_natively() {
    let source = r"
        local shapes = { [1] = 'full', [3] = 'slab' }
        local holder = { shapes = { [2] = 'stair' } }
        local function get_block (x)
            if x < 1000 then return nil end
            return x % 4
        end
        local hits, misses = 0, 0
        for i = 1, 40000 do
            local cid = get_block (i % 900)
            if shapes[cid] or holder.shapes[cid] then
                hits = hits + 1
            else
                misses = misses + 1
            end
        end
        return hits, misses
    ";
    agree(source, |profile, stats| native(profile, stats, 30_000));
}

/// The key turns from nil to a number and back, and the table later gains a
/// metatable whose `__index` answers the nil key: the trace leaves at each
/// change and the results stay the interpreter's.
#[test]
fn a_read_under_a_nil_key_leaves_when_the_key_or_the_metatable_changes() {
    let source = r"
        local shapes = { [1] = 10, [3] = 30 }
        local holder = { shapes = { [2] = 20 } }
        local function get_block (i)
            if i % 3000 < 2000 then return nil end
            return i % 4
        end
        local s = 0
        for i = 1, 40000 do
            if i == 30000 then
                setmetatable (shapes, { __index = function (_, k) return k == nil and 5 or 1 end })
                setmetatable (holder.shapes, { __index = function () return 7 end })
            end
            local cid = get_block (i)
            s = s + (shapes[cid] or 0) + (holder.shapes[cid] or 0)
        end
        return s
    ";
    agree(source, |profile, stats| {
        assert!(stats.traces_compiled > 0, "{profile:?} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {stats:?}");
    });
}
