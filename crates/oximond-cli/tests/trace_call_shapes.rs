//! Native traces that follow the call shapes of library-heavy code: calls
//! with open results, tail calls inside inlined callees, strings, and
//! closures reached through upvalues and table fields.
//!
//! Every program runs under the interpreter and under tracing execution. The
//! results must match exactly, and the counters must show that a trace really
//! ran the shape, because equal output alone cannot tell a trace from a slow
//! path. Each shape also has a program that breaks the trace's assumption in
//! the middle of the loop: a callee rebound, a string replaced by a number, a
//! metamethod appearing on a table. The guard that catches it must leave the
//! trace with a frame the interpreter can finish.

use oximond::vm::{TraceGuardKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles whose numbers are all floats, where `math.floor` and the
/// string natives have a numeric trace form.
const FLOAT_PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::LuaJit21,
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
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    (format!("{results:?}"), lua.vm().jit_statistics().trace)
}

/// Runs `source` under both modes, checks the results agree, and returns the
/// result text and the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`] over `profiles`, requiring that a trace ran native
/// iterations and that no recording aborted.
fn agree_traced(profiles: &[CompatibilityProfile], source: &str) -> Vec<(String, TraceStatistics)> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.traces_compiled > 0 && stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            (result, stats)
        })
        .collect()
}

fn aborts(stats: &TraceStatistics) -> u64 {
    stats.recording_aborts.iter().sum()
}

// ------------------------------------------------------ open results

const OPEN_NATIVES: &str = "
    local floor, tonumber = math.floor, tonumber
    local function low(x) return tonumber(floor(x / 3)) end
    local s = 0
    for i = 1, 20000 do s = s + low(i) end
    return s";

#[test]
fn a_native_call_that_feeds_another_runs_inside_the_trace() {
    for (_, stats) in agree_traced(&FLOAT_PROFILES, OPEN_NATIVES) {
        assert!(stats.iterations > 19_000, "{stats:?}");
        assert_eq!(aborts(&stats), 0, "{stats:?}");
    }
    // The integer profiles floor to integers, which has no float form. The
    // answer stays the same.
    for profile in [CompatibilityProfile::Lua53, CompatibilityProfile::Lua54] {
        agree(profile, OPEN_NATIVES);
    }
}

#[test]
fn rebinding_either_native_of_an_open_chain_leaves_the_trace() {
    // The consumer's callee is guarded before the call that makes the open
    // results, so a rebound `tonumber` leaves before `floor` runs.
    let source = "
        local floor, tonumber = math.floor, tonumber
        local function low(x) return tonumber(floor(x / 3)) end
        local s = 0
        for i = 1, 20000 do
          if i == 12000 then tonumber = function(v) return -v end end
          s = s + low(i)
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
    let source = "
        local floor, tonumber = math.floor, tonumber
        local function low(x) return tonumber(floor(x / 3)) end
        local s = 0
        for i = 1, 20000 do
          if i == 12000 then floor = function(v) return v - 0.5 end end
          s = s + low(i)
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
}

#[test]
fn a_value_that_is_not_a_float_leaves_an_open_chain() {
    // `x / 3` is a float until the loop hands `low` a string, which `floor`
    // converts in the interpreter.
    let source = "
        local floor, tonumber = math.floor, tonumber
        local function low(x) return tonumber(floor(x)) end
        local s = 0
        for i = 1, 20000 do
          local v = i / 4
          if i == 15000 then v = '77.5' end
          s = s + low(v)
        end
        return s";
    for profile in FLOAT_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_closure_that_returns_several_values_feeds_the_call_after_it() {
    let source = "
        local function two(x) return x, x + 1 end
        local function add(a, b) return a * b end
        local function wrap(x) return add(two(x)) end
        local function more(x) return x, two(x) end
        local s = 0
        for i = 1, 20000 do
          local a, b, c = more(i)
          s = (s + wrap(i) + a + b + c) % 1000003
        end
        return s";
    for (_, stats) in agree_traced(&ALL_PROFILES, source) {
        assert!(stats.inlined_calls >= 4, "{stats:?}");
        assert!(stats.iterations > 19_000, "{stats:?}");
    }
}

#[test]
fn rebinding_the_closures_of_an_open_chain_leaves_the_trace() {
    let source = "
        local function two(x) return x, x + 1 end
        local function add(a, b) return a * b end
        local function wrap(x) return add(two(x)) end
        local s = 0
        for i = 1, 20000 do
          if i == 9000 then add = function(a, b) return a - b end end
          if i == 14000 then two = function(x) return x * 2, x end end
          s = (s + wrap(i)) % 1000003
        end
        return s";
    for (_, stats) in agree_traced(&ALL_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
}

// --------------------------------------------------------- tail calls

#[test]
fn a_tail_call_inside_an_inlined_callee_runs_in_the_trace() {
    let source = "
        local function inner(x) return x * 2 + 1 end
        local function middle(x) return inner(x + 1) end
        local function outer(x) return middle(x) end
        local s = 0
        for i = 1, 20000 do s = (s + outer(i)) % 1000003 end
        return s";
    for (_, stats) in agree_traced(&ALL_PROFILES, source) {
        assert!(stats.inlined_calls >= 3, "{stats:?}");
        assert!(stats.iterations > 19_000, "{stats:?}");
        assert_eq!(aborts(&stats), 0, "{stats:?}");
    }
}

#[test]
fn a_tail_called_callee_that_is_replaced_leaves_the_trace() {
    let source = "
        local function inner(x) return x * 2 + 1 end
        local function outer(x) return inner(x + 1) end
        local s = 0
        for i = 1, 20000 do
          if i == 11000 then inner = function(x) return x - 7 end end
          s = (s + outer(i)) % 1000003
        end
        return s";
    for (_, stats) in agree_traced(&ALL_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
}

#[test]
fn an_exit_after_a_tail_call_rebuilds_a_frame_that_counts_it() {
    // `inner` replaced `outer`'s frame, and a store before the call keeps the
    // exit from simply starting the call over. When `x` passes 15000 `inner`
    // asks for a traceback and the stack depth, which show the tail call.
    let source = "
        local tb, depth
        local cells = {0, 0}
        local function inner(x)
          if x > 15000 then
            tb = debug.traceback('tb')
            local level = 1
            while debug.getinfo(level, 'l') do level = level + 1 end
            depth = level
            return 0
          end
          return x + 1
        end
        local function outer(x)
          cells[1] = cells[1] + 1
          return inner(x)
        end
        local v = 1
        for i = 1, 20000 do v = outer(v) end
        return tb, depth, v, cells[1]";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(
            stats.inlined_calls > 0 && stats.iterations > 10_000,
            "{profile:?}: {stats:?}"
        );
        assert!(result.contains("tb"), "{profile:?}: {result}");
    }
}

#[test]
fn a_tail_call_of_a_native_function_runs_in_the_trace() {
    let source = "
        local floor = math.floor
        local function half(x) return floor(x / 2) end
        local s = 0
        for i = 1, 20000 do s = (s + half(i)) % 1000003 end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.iterations > 19_000, "{stats:?}");
        assert_eq!(aborts(&stats), 0, "{stats:?}");
    }
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        agree(profile, source);
    }
    // `floor` rebound between two runs and in the middle of one.
    let source = "
        local floor = math.floor
        local function half(x) return floor(x / 2) end
        local s = 0
        for i = 1, 20000 do
          if i == 10000 then floor = function(v) return v end end
          s = (s + half(i)) % 1000003
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
}

// -------------------------------------------- closures behind variables

#[test]
fn a_closure_called_through_an_upvalue_and_a_field_runs_in_the_trace() {
    let source = "
        local floor = math.floor
        local function counter(seed)
          local state = {seed, 0}
          local function advance(bits)
            state[1] = (state[1] * 69069 + 12345) % 4294967296
            state[2] = state[2] + 1
            return floor(state[1] / 2 ^ (32 - bits))
          end
          return {
            first = function(self) return advance(32) end,
            second = function(self, bound) return (advance(31)) % bound end,
          }
        end
        local rng, other = counter(5), counter(5)
        local s = 0
        for i = 1, 20000 do
          s = (s * 3 + rng:first() + other:second(5000)) % 1000000007
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.iterations > 19_000, "{stats:?}");
        assert_eq!(aborts(&stats), 0, "{stats:?}");
    }
}

#[test]
fn a_method_replaced_mid_loop_leaves_the_trace() {
    let source = "
        local obj = {}
        function obj.step(self, x) return x + 1 end
        local s = 0
        for i = 1, 20000 do
          if i == 10000 then obj.step = function(self, x) return x * 3 end end
          s = (s + obj:step(i)) % 1000003
        end
        return s";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn a_metamethod_appearing_on_the_table_operand_leaves_the_trace() {
    // `a[2] = nil` and a `__newindex` make the next store run the metamethod,
    // which the bound array store refuses.
    let source = "
        local a = {1, 2}
        local log = 0
        local s = 0
        for i = 1, 20000 do
          if i == 12000 then
            a[2] = nil
            setmetatable(a, {__newindex = function(t, k, v) log = log + 1; rawset(t, k, v * 2) end})
          end
          a[1] = a[1] + 1
          a[2] = i
          s = s + a[1] + (a[2] or 0)
        end
        return s, log, a[1], a[2]";
    for (_, stats) in agree_traced(&ALL_PROFILES, source) {
        assert!(stats.iterations > 10_000, "{stats:?}");
    }
}

// ------------------------------------------------------------ strings

const DIGITS: &str = "
    local max, min, byte, sub = math.max, math.min, string.byte, string.sub
    local function digits(str)
      local total = 0
      for i = 1, #str do
        total = total * 10 + max(min(byte(sub(str, i, i)) - 48, 9), 0)
      end
      return total
    end
    local text = string.rep('3948575739', 8)
    local s = 0
    for round = 1, 400 do s = (s + digits(text)) % 1000000007 end
    return s";

#[test]
fn a_loop_over_the_bytes_of_a_string_runs_natively() {
    // The loop over the rounds holds a loop in a call. It records again in
    // front of that loop (see `trace_callee_loops`), but the loop side of
    // the byte loop aborts, so that root retires after a few dozen entries
    // and the rounds stay interpreted. The loop over the bytes runs
    // natively, entered once a round.
    for (_, stats) in agree_traced(&FLOAT_PROFILES, DIGITS) {
        assert!(stats.iterations > 30_000, "{stats:?}");
        assert!(
            (400..=400 + 2 * 64).contains(&stats.native_entries),
            "{stats:?}"
        );
    }
    for profile in [CompatibilityProfile::Lua53, CompatibilityProfile::Lua54] {
        agree(profile, DIGITS);
    }
}

#[test]
fn string_bytes_and_lengths_come_from_the_bound_string() {
    let source = "
        local byte, rep = string.byte, string.rep
        local function checksum(str)
          local total = 0
          for i = 1, #str do total = (total * 31 + byte(str, i)) % 1000003 end
          return total, #str
        end
        local a, b = rep('abcde', 40), rep('xyz', 70)
        local s = 0
        for round = 1, 300 do
          local c, n = checksum(round % 2 == 0 and a or b)
          s = (s + c + n) % 1000003
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.iterations > 10_000, "{stats:?}");
    }
    for profile in [CompatibilityProfile::Lua53, CompatibilityProfile::Lua55] {
        agree(profile, source);
    }
}

#[test]
fn an_index_outside_the_string_leaves_the_trace() {
    // The loop reads one byte past the end. `string.byte` returns nothing
    // there, which only the interpreter handles.
    let source = "
        local byte = string.byte
        local text = string.rep('hello world', 20)
        local s = 0
        for i = 1, #text + 3 do s = s + (byte(text, i) or 1000) end
        return s";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    // A negative and a fractional index.
    let source = "
        local byte = string.byte
        local text = string.rep('hello world', 20)
        local s = 0
        for i = -2, 400 do s = s + (byte(text, i) or 1000) end
        for i = 0.5, 300, 1 do s = s + (byte(text, i) or 7) end
        return s";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        agree(profile, source);
    }
}

#[test]
fn a_string_replaced_by_a_number_or_another_string_stays_correct() {
    let source = "
        local function digits(str)
          local total = 0
          for i = 1, #str do
            total = total * 10 + (string.byte(string.sub(str, i, i)) - 48)
          end
          return total
        end
        local words = {'12345', '2718', '31415926', '1'}
        local s = 0
        for round = 1, 300 do s = (s + digits(words[round % 4 + 1])) % 1000003 end
        local ok, err = pcall(digits, 12345)
        return s, ok, err";
    for profile in FLOAT_PROFILES {
        let (result, _) = agree(profile, source);
        assert!(result.contains("false"), "{profile:?}: {result}");
    }
    // A number that becomes a string in the middle of a loop.
    let source = "
        local v = 1.5
        local s = 0
        for i = 1, 20000 do
          if i == 10000 then v = '2.5' end
          s = s + v
        end
        return s";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
    // A string carried through an inlined call.
    let source = "
        local function pass(x) return x end
        local text = 'abcdef'
        local s = 0
        for i = 1, 20000 do
          local t = pass(text)
          s = s + #t
          if i == 10000 then text = 'xy' end
        end
        return s, text";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

#[test]
fn string_natives_that_were_rebound_leave_the_trace() {
    let source = "
        local text = string.rep('0123456789', 30)
        local byte, sub = string.byte, string.sub
        local s = 0
        for i = 1, 20000 do
          local k = i % 300 + 1
          if i == 12000 then byte = function(x) return 7 end end
          s = s + byte(sub(text, k, k))
        end
        return s";
    for (_, stats) in agree_traced(&FLOAT_PROFILES, source) {
        assert!(stats.side_exits_of(TraceGuardKind::Callee) > 0, "{stats:?}");
    }
}

#[test]
fn a_substring_read_by_byte_is_never_made_and_rebinding_either_function_leaves() {
    // `sub` then `byte` of the result, both bound to locals. The trace reads
    // the source string; rebinding either function in the middle of the loop
    // leaves through the callee guard, and the interpreter makes the string.
    let program = |rebind: &str| {
        format!(
            "
        local text = string.rep('0123456789abcdef', 16)
        local byte, sub = string.byte, string.sub
        local s = 0
        for i = 1, 20000 do
          local k = i % 250 + 1
          {rebind}
          local piece = sub(text, k, k + 3)
          local a, b, c, d = byte(piece, 1, 4)
          s = s + (a or 1) + (b or 2) + (c or 3) + (d or 1000)
        end
        return s"
        )
    };
    for (_, stats) in agree_traced(&FLOAT_PROFILES, &program("")) {
        assert_eq!(aborts(&stats), 0, "{stats:?}");
        assert!(stats.iterations > 15_000, "{stats:?}");
    }
    for rebind in [
        "if i == 12000 then sub = function(x, y, z) return 'abcd' end end",
        "if i == 12000 then byte = function(x, y, z) return 1, 2, 3, 4 end end",
        "if i == 12000 then sub = string.rep end",
        "if i == 12000 then text = 'short' end",
        "if i == 12000 then text = 12345678 end",
    ] {
        let source = program(rebind);
        for profile in ALL_PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (tracing, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(tracing, expected, "{profile:?} {rebind}\n{stats:?}");
        }
    }
}

#[test]
fn strings_never_compare_through_a_lane() {
    // Two strings with equal bytes in two allocations are equal.
    let source = "
        local a = string.rep('ab', 10)
        local b = string.rep('ab', 10)
        local n = 0
        for i = 1, 20000 do
          if a == b then n = n + 1 end
          if i == 10000 then b = 'zz' end
        end
        return n";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}
