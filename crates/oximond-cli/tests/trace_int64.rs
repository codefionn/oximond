//! 64-bit integer cdata (`LuaJIT`'s `int64_t` and `uint64_t`) in native traces.
//!
//! `fixtures/trace_int64_ops.lua` evaluates every arithmetic, comparison,
//! `bit` library and `tonumber` form over a grid of awkward operands (zero,
//! the extremes, signed against unsigned, numbers with a fraction, 2^53, 2^63
//! and NaN) inside hot loops, and returns what the last iteration computed.
//! It also changes a loop's operand types between runs and lets an
//! accumulator wrap. The script runs under the interpreter, under tracing
//! execution with synchronous compilation, and under a live `LuaJIT` when one
//! is installed; the three outputs must be identical. Counters must show that
//! native code ran and that guards left it, because equal output alone cannot
//! tell a trace from a slow path.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run(source: &str, mode: ExecutionMode) -> (String, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::LuaJit21)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{mode:?}: {error}\n{source}"));
    let text = results
        .iter()
        .map(|value| match value {
            Value::String(text) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\t");
    let stats = lua.vm().jit_statistics().trace;
    (text, stats)
}

fn live_luajit(source: &str) -> Option<String> {
    // Tests run in parallel threads of one process, so each call needs a
    // directory of its own.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let luajit = Path::new("/usr/bin/luajit");
    if !luajit.exists() {
        return None;
    }
    let directory = std::env::temp_dir().join(format!(
        "oximond-trace-int64-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let script = directory.join("script.lua");
    std::fs::write(
        &script,
        format!("io.write(assert(loadstring([=====[{source}]=====]))())"),
    )
    .unwrap();
    let output = Command::new(luajit).arg(&script).output().unwrap();
    let _ = std::fs::remove_dir_all(&directory);
    assert!(
        output.status.success(),
        "luajit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(String::from_utf8(output.stdout).unwrap())
}

fn assert_same(reference: &str, expected: &str, actual: &str, context: &str) {
    let mismatches: Vec<String> = expected
        .lines()
        .zip(actual.lines())
        .filter(|(expected, actual)| expected != actual)
        .take(20)
        .map(|(expected, actual)| format!("  {reference}: {expected}\n  {context}: {actual}"))
        .collect();
    assert!(
        mismatches.is_empty() && expected.lines().count() == actual.lines().count(),
        "{context} differs from {reference}:\n{}",
        mismatches.join("\n")
    );
}

/// Runs `source` in every mode and checks the outputs agree. Returns the
/// tracing statistics.
fn agree(source: &str) -> TraceStatistics {
    let (expected, _) = run(source, ExecutionMode::Interpreter);
    let (tracing, stats) = run(source, ExecutionMode::Tracing);
    assert_same("interpreter", &expected, &tracing, "tracing");
    if let Some(live) = live_luajit(source) {
        assert_same("luajit", &live, &tracing, "tracing");
    }
    stats
}

fn exits(stats: &TraceStatistics, kind: TraceExitKind) -> u64 {
    stats.exits[kind.index()]
}

#[test]
fn cdata_operations_in_traced_loops_match_the_interpreter_and_luajit() {
    let source = std::fs::read_to_string(fixture_path("trace_int64_ops.lua")).unwrap();
    let stats = agree(&source);
    assert!(
        stats.traces_compiled > 100,
        "the grid compiles a trace per loop: {stats:?}"
    );
    assert!(
        stats.iterations > 10_000,
        "native code runs most iterations: {stats:?}"
    );
    assert!(stats.native_entries > 100, "{stats:?}");
}

/// Every operand of a cdata operation changes type between the runs of one
/// loop. A trace is specialised on the types it saw at entry; the others
/// must enter another trace or the interpreter, with the same answer.
#[test]
fn operand_types_that_change_between_runs_leave_the_trace_correctly() {
    let source = r#"
        local function make()
            return assert(loadstring([[
                local band, rshift = bit.band, bit.rshift
                return function(n, acc, step)
                    for i = 1, n do
                        acc = (acc + step) * 3ull - rshift(acc, 3)
                        acc = band(acc, 0xffffffffffffull) + 1
                    end
                    return acc
                end
            ]]))()
        end
        local f = make()
        local out = {}
        local cases = {
            { 0ull, 1 }, { 5ull, 2ull }, { 7ll, 3ll }, { 9, 4 }, { 11ll, 0.5 },
            { 13ull, -1ll }, { 15, 2ull }, { 0x7fffffffffffffffll, 1 },
        }
        for round = 1, 3 do
            for k, case in ipairs(cases) do
                local r = f(150, case[1], case[2])
                out[#out + 1] = round .. ":" .. k .. " " .. type(r) .. " " .. tostring(r)
            end
        end
        return table.concat(out, "\n")
    "#;
    let stats = agree(source);
    assert!(stats.iterations > 1_000, "{stats:?}");
    assert!(
        exits(&stats, TraceExitKind::EntryMismatch) > 5,
        "entry types differ: {stats:?}"
    );
}

/// A table element that was a number when the trace recorded and holds a
/// cdata later fails the array load's type guard in the middle of a loop.
#[test]
fn an_array_element_that_turns_into_cdata_exits_the_trace() {
    let source = r#"
        local data = {}
        for i = 1, 400 do data[i] = i % 7 end
        local f = assert(loadstring([[
            local data = ...
            return function(n)
                local acc = 0ull
                for i = 1, n do
                    acc = acc * 3ull + data[i] % 5
                    acc = acc % 1000003ull
                end
                return acc
            end
        ]]))(data)
        local out = {}
        for round = 1, 6 do
            if round == 3 then data[250] = 7ull end
            if round == 5 then data[250] = 2ll; data[90] = 4ull end
            local r = f(400)
            out[#out + 1] = round .. " " .. type(r) .. " " .. tostring(r)
        end
        return table.concat(out, "\n")
    "#;
    let stats = agree(source);
    assert!(stats.iterations > 500, "{stats:?}");
    assert!(
        exits(&stats, TraceExitKind::GuardFail) > 0,
        "the array load guard failed in native code: {stats:?}"
    );
}

/// Zero divisors, `MIN / -1`, shifts by 64 or more and by negative counts,
/// and conversions of NaN and out-of-range numbers, with the operands
/// reaching the trace as loop-carried values rather than constants.
#[test]
fn division_shift_and_conversion_edges_agree_with_luajit() {
    let source = r#"
        local out = {}
        local src = [[
            local band, bor, bxor, lshift, rshift, arshift, rol, ror, bnot =
                bit.band, bit.bor, bit.bxor, bit.lshift, bit.rshift, bit.arshift, bit.rol, bit.ror, bit.bnot
            return function(n, a, b, c)
                local r1, r2, r3, r4, r5, r6, r7, r8, r9, r10
                for i = 1, n do
                    r1 = a / b
                    r2 = a % b
                    r3 = lshift(a, c)
                    r4 = rshift(a, c)
                    r5 = arshift(a, c)
                    r6 = rol(a, c)
                    r7 = ror(a, c)
                    r8 = a + c
                    r9 = bnot(a)
                    r10 = a < c
                end
                return r1, r2, r3, r4, r5, r6, r7, r8, r9, r10
            end
        ]]
        local f = assert(loadstring(src))()
        local minll = -0x7fffffffffffffffll - 1
        local operands = {
            { 7ll, 0ll, 64 }, { minll, -1ll, 63 }, { minll, 0ll, -1 },
            { 7ull, 0ull, 65 }, { 0xffffffffffffffffull, 3ull, 1 },
            { -9ll, 2ll, 100 }, { 0x8000000000000000ull, 0x8000000000000000ull, 31 },
            { 5ll, 3ll, 0.5 }, { 5ll, 3ll, 2.5 }, { 5ll, 3ll, 3.5 }, { 5ll, 3ll, -1.5 },
            { 5ull, 3ll, 4294967297 }, { 5ll, 1e19, 2 }, { 5ull, -2.7, 2 }, { 5ll, 0/0, 2 },
            { 1ll, 2^63, 2^64 }, { 1ull, 2^63, 2^64 }, { 6ll, 4ull, 1ull },
        }
        for k, case in ipairs(operands) do
            local r = { f(120, case[1], case[2], case[3]) }
            for i = 1, 10 do
                out[#out + 1] = k .. "." .. i .. " " .. type(r[i]) .. " " .. tostring(r[i])
            end
        end
        return table.concat(out, "\n")
    "#;
    let stats = agree(source);
    assert!(stats.iterations > 1_000, "{stats:?}");
}

/// Mineclonia's 64-bit helpers: the chunk `mcl_levelgen/random.lua` builds
/// with `loadstring` when the runtime has `LuaJIT`'s `1ull`, verbatim, and
/// the locals it fills.
const MINECLONIA_ULL: &str = r"
        local mulull, addull, addkull, andull, xorull, rotlull, shrull, ashrull, shlull, mul2ull
        	local str = [[
        	local bxor = bit.bxor
        	local band = bit.band
        	local bnot = bit.bnot
        	local bor = bit.bor
        	local rshift = bit.rshift
        	local lshift = bit.lshift
        	local rol = bit.rol
        	local arshift = bit.arshift
        	local UINT_MAX = 4294967295
        	local tonumber = tonumber
        
        	local function mulull (a, m)
        		local a_lo, a_hi = a[1], a[2]
        
        		-- Long multiplication of two 32 bit operands into a
        		-- 64 bit product.
        		local m1 = m * 1ull
        		local lo = a_lo * m1
        		local hi = a_hi * m1
        		local excess = rshift (hi, 32)
        		local lohi = rshift (lo, 32)
        		local carry
        
        		hi = band (hi, 0xffffffffull)
        		carry = rshift ((UINT_MAX - lohi) - hi, 63)
        		excess = excess + carry
        		hi = band (hi + lohi, 0xffffffffull)
        		a[1] = tonumber (band (lo, 0xffffffffull))
        		a[2] = tonumber (hi)
        		return tonumber (excess)
        	end
        
        	local function addull (a, b)
        		local along = 0x100000000ull * a[2] + a[1]
        		local blong = 0x100000000ull * b[2] + b[1]
        		local value = along + blong
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	local function addkull (a, k)
        		local along = 0x100000000ull * a[2] + a[1]
        		local value = along + k
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	-- Avoid expensive normalization by performing unsigned
        	-- arithmetic.
        	local function andull (a, b)
        		a[1] = tonumber (band (a[1] * 1ull, b[1] * 1ull))
        		a[2] = tonumber (band (a[2] * 1ull, b[2] * 1ull))
        	end
        
        	local function xorull (a, b)
        		a[1] = tonumber (bxor (a[1] * 1ull, b[1] * 1ull))
        		a[2] = tonumber (bxor (a[2] * 1ull, b[2] * 1ull))
        	end
        
        	local function rotlull (a, k)
        		local along = 0x100000000ull * a[2] + a[1]
        		local value = rol (along, k)
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	local function shrull (a, k)
        		local along = 0x100000000ull * a[2] + a[1]
        		local value = rshift (along, k)
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	local function ashrull (a, k)
        		local along = 0x100000000ull * a[2] + a[1]
        		local value = arshift (along, k)
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	local function shlull (a, k)
        		local along = 0x100000000ull * a[2] + a[1]
        		local value = lshift (along, k)
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        
        	local function mul2ull (a, b)
        		local along = 0x100000000ull * a[2] + a[1]
        		local blong = 0x100000000ull * b[2] + b[1]
        		local value = along * blong
        		a[1] = tonumber (band (value, 0xffffffffull))
        		a[2] = tonumber (rshift (value, 32))
        	end
        	return mulull, addull, addkull, andull, xorull,
        		rotlull, shrull, ashrull, shlull, mul2ull
        ]]
        local fn = loadstring (str)
        mulull, addull, addkull, andull, xorull,
            rotlull, shrull, ashrull, shlull, mul2ull = fn ()
";

/// Mineclonia's 64-bit helpers (`mcl_levelgen/random.lua`, the chunk it
/// builds with `loadstring` when the runtime has `LuaJIT`'s `1ull`, verbatim)
/// and its xoroshiro128++ step, which calls them nine times on `{ lo, hi }`
/// pairs. The loop that draws numbers runs natively, with no abort, and two
/// generators taking turns every 500 draws leave the trace at its callee
/// guard and come back.
#[test]
fn mineclonia_ull_helpers_run_natively() {
    let source = format!(
        "{MINECLONIA_ULL}{}",
        r#"
        local function xoroshiro (seedlo, seedhi)
            local lo1, hi1 = { nil, nil, }, { nil, nil, }
            local function getlong ()
                lo1[1] = seedlo[1]
                lo1[2] = seedlo[2]
                addull (lo1, seedhi)
                rotlull (lo1, 17)
                addull (lo1, seedlo)
                hi1[1] = seedhi[1]
                hi1[2] = seedhi[2]
                xorull (hi1, seedlo)
                rotlull (seedlo, 49)
                xorull (seedlo, hi1)
                local h1l, h1h = hi1[1], hi1[2]
                shlull (hi1, 21)
                xorull (seedlo, hi1)
                seedhi[1] = h1l
                seedhi[2] = h1h
                rotlull (seedhi, 28)
                return lo1
            end
            return getlong
        end

        local out = {}
        local one = xoroshiro ({ 1000, 0 }, { 1000, 0 })
        local other = xoroshiro ({ 7, 3 }, { 99, 12345 })
        local acc = 0
        for round = 1, 8 do
            local draw = round % 2 == 0 and one or other
            for i = 1, 500 do
                local value = draw ()
                acc = (acc + value[1] + value[2]) % 4294967296
            end
            out[#out + 1] = tostring (acc)
        end
        local a, b = { 12345, 678 }, { 99, 4242 }
        local mask = { 0xffffffff, 0x7fffffff }
        for i = 1, 2000 do
            mulull (a, 42317861)
            addkull (b, 77)
            andull (b, mask)
            shrull (b, 1)
            ashrull (a, 3)
            mul2ull (a, b)
        end
        out[#out + 1] = a[1] .. " " .. a[2] .. " " .. b[1] .. " " .. b[2]
        return table.concat (out, "\n")
    "#
    );
    let stats = agree(&source);
    assert_eq!(stats.total_aborts(), 0, "{stats:?}");
    assert!(stats.traces_compiled >= 2, "{stats:?}");
    assert!(stats.iterations > 2_000, "{stats:?}");
    assert!(
        exits(&stats, TraceExitKind::GuardFail) > 0,
        "the other generator leaves at a guard: {stats:?}"
    );
}

/// A loop body of 80 calls of the helpers, about 2700 guest instructions:
/// before the stores of a pair half that the iteration already read or
/// stored took no snapshot, the recording passed the 500-snapshot cap
/// (`maxsnap`) at about one snapshot per three guest instructions, and the
/// loop stayed in the interpreter (`get_structure_starts` on the Mineclonia
/// emerge). Now it compiles with no abort and runs natively, and the stores
/// into pairs the call before stored take no guard.
#[test]
fn a_loop_of_many_ull_helper_calls_fits_the_snapshot_cap() {
    let calls = [
        "addull (a, b)",
        "rotlull (a, 17)",
        "xorull (b, a)",
        "shrull (c, 3)",
        "addkull (c, 77)",
        "andull (c, mask)",
        "mul2ull (b, c)",
        "shlull (a, 5)",
    ];
    let mut body = String::new();
    for index in 0..80 {
        let _ = writeln!(body, "            {}", calls[index % calls.len()]);
    }
    let source = format!(
        "{MINECLONIA_ULL}
        local a, b, c = {{ 12345, 678 }}, {{ 99, 4242 }}, {{ 7, 3 }}
        local mask = {{ 0xffffffff, 0x7fffffff }}
        local acc = 0
        for i = 1, 300 do
{body}            acc = (acc + a[1] + b[2] + c[1]) % 4294967296
        end
        return a[1] .. ' ' .. a[2] .. ' ' .. b[1] .. ' ' .. b[2] .. ' ' .. c[1] .. ' ' .. c[2] .. ' ' .. acc
"
    );
    let stats = agree(&source);
    assert_eq!(stats.total_aborts(), 0, "{stats:?}");
    assert!(stats.traces_compiled >= 1, "{stats:?}");
    assert!(stats.iterations > 200, "{stats:?}");
    assert!(stats.array_overwrites > 100, "{stats:?}");
}
