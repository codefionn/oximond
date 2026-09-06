//! Native traces over captured locals, upvalue cells, globals, opaque values
//! and string constants, through compiled Lua source.
//!
//! Every program runs under the interpreter and under tracing execution, in
//! every profile it makes sense for. The results must match exactly, and
//! counters must show that native trace code really ran, because equal output
//! alone cannot tell a trace from a slow path.
//!
//! What these programs hold the engine to:
//!
//! - A local that a closure captured keeps one truth in its cell. A trace
//!   that changes it must leave the cell right at every way out of the loop:
//!   a failed guard, the end of the loop, an error, and a poll.
//! - Two closures with the same bytecode and different cells share one
//!   trace, and each run uses its own cell.
//! - A global is read when the trace is entered. A rebound global is a
//!   different value, never a stale constant.
//! - Tables and closures travel through a trace by identity only, a table
//!   with `__eq` is never compared natively, and no trace keeps an object
//!   alive between runs.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

const ALL_PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// The profiles whose bytecode reads a global with one instruction.
const HAS_GLOBAL_OPCODE: [CompatibilityProfile; 2] =
    [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21];

const INTEGER_PROFILES: [CompatibilityProfile; 3] = [
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
/// the results agree, and returns the tracing statistics.
fn agree(profile: CompatibilityProfile, source: &str) -> (String, TraceStatistics) {
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (tracing, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(
        tracing, expected,
        "{profile:?} tracing\n{source}\n{stats:?}"
    );
    (expected, stats)
}

/// Like [`agree`] for every profile in `profiles`, requiring native
/// iterations in each.
fn agree_natively(profiles: &[CompatibilityProfile], source: &str) -> Vec<String> {
    profiles
        .iter()
        .map(|profile| {
            let (result, stats) = agree(*profile, source);
            assert!(
                stats.iterations > 0,
                "{profile:?}: no native iteration ran\n{source}\n{stats:?}"
            );
            result
        })
        .collect()
}

#[test]
fn a_loop_over_a_captured_local_runs_natively_and_leaves_its_cell_right() {
    // `total` is captured by `f`, so its register mirrors a cell. The trace
    // adds to the register, and the cell must hold the sum when `f` reads it.
    let source = "
        local function run(n)
          local total = 0.0
          local f = function() return total end
          for i = 1, n do total = total + i * 0.5 end
          return f()
        end
        local r = 0.0
        for k = 1, 30 do r = r + run(2000) end
        return r";
    let results = agree_natively(&ALL_PROFILES, source);
    assert!(
        results.iter().all(|result| result == &results[0]),
        "{results:?}"
    );
}

#[test]
fn upvalue_reads_and_writes_run_natively() {
    let source = "
        local count = 0.0
        local function bump(n) for i = 1, n do count = count + 1.0 end end
        for k = 1, 40 do bump(500) end
        return count";
    let results = agree_natively(&ALL_PROFILES, source);
    assert!(
        results.iter().all(|result| result == &results[0]),
        "{results:?}"
    );
}

#[test]
fn two_names_for_one_cell_reconcile_at_every_guard_exit() {
    // `n` is the outer local, captured by `peek` and by `run`, which sees it
    // as an upvalue. Every hundredth iteration the guard on `i % 100` fails
    // and the interpreter calls `peek`, which must see the iterations the
    // trace ran natively, not the value the cell held at entry.
    let source = "
        local n = 0.0
        local function peek() return n end
        local seen = {}
        local function run(limit)
          for i = 1, limit do
            n = n + 1.0
            if i % 100 == 0 then seen[#seen + 1] = string.format('%.0f', peek()) end
          end
        end
        run(1000) run(1000) run(777)
        return table.concat(seen, ',') .. ';' .. string.format('%.0f', n)";
    let results = agree_natively(&ALL_PROFILES, source);
    assert!(
        results.iter().all(|result| result == &results[0]),
        "{results:?}"
    );
    assert!(results[0].contains("100,200,300"), "{}", results[0]);
    assert!(results[0].contains(";2777"), "{}", results[0]);
}

#[test]
fn an_error_leaves_the_cell_as_the_interpreter_would() {
    // The divisor reaches zero at the last iteration. A guard before the
    // division leaves the trace, the interpreter raises the error, and the
    // caller of `pcall` reads the cell. Integer division by zero is an error
    // in Lua 5.3 and later.
    let source = "
        local total = 0
        local function peek() return total end
        local function work(limit)
          for i = 1, limit do total = total + 1000 // (limit - i) end
        end
        local ok, err = pcall(work, 500)
        return tostring(ok) .. ':' .. peek() .. ':' .. tostring(err)";
    let results = agree_natively(&INTEGER_PROFILES, source);
    assert!(results[0].contains("false:"), "{}", results[0]);
}

#[test]
fn a_long_loop_over_a_cell_polls_and_still_finishes_exactly() {
    let source = "
        local count = 0.0
        local function bump(n) for i = 1, n do count = count + 0.5 end end
        bump(900000)
        return count";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (_, stats) = agree(profile, source);
        assert!(stats.exits_of(TraceExitKind::Poll) >= 2, "{stats:?}");
        assert!(stats.iterations > 890_000, "{stats:?}");
    }
}

#[test]
fn a_cell_that_changes_type_between_entries_mismatches_and_the_interpreter_runs() {
    // The trace takes floats from `acc`. A numeric string there fails the
    // entry check and the interpreter does the arithmetic.
    let source = "
        local acc = 0.0
        local function add(n) for i = 1, n do acc = acc + 1.0 end end
        for k = 1, 30 do add(300) end
        acc = '5'
        add(300)
        for k = 1, 30 do add(300) end
        return acc";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (_, stats) = agree(profile, source);
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn closures_with_one_bytecode_and_different_cells_keep_their_own_cells() {
    let source = "
        local function make()
          local n = 0.0
          return function(k) for i = 1, k do n = n + 1.0 end return n end
        end
        local a, b = make(), make()
        local ra, rb = 0.0, 0.0
        for round = 1, 20 do ra = a(500) rb = b(250) end
        return string.format('%.0f,%.0f', ra, rb)";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("10000,5000"), "{profile:?}: {result}");
        assert!(stats.iterations > 10_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_closure_made_per_call_shares_one_trace_and_keeps_its_own_cell() {
    // Every call of `work` makes a fresh closure over a fresh cell. The
    // closures share one prototype, so they must share recording and
    // compilation too, and each run must use its own cell.
    let source = "
        local total = 0.0
        local function work(n)
          local acc = 0.0
          local f = function(m) for i = 1, m do acc = acc + i * 0.5 end end
          f(n)
          return acc
        end
        for k = 1, 3000 do total = total + work(100 + k % 7) end
        return string.format('%.1f', total)";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (_, stats) = agree(profile, source);
        assert!(
            stats.traces_compiled <= 8 && stats.traces_recorded <= 8,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn a_global_function_is_read_at_entry_and_a_rebound_one_is_a_new_value() {
    // The loop compares a global function with an upvalue function. The
    // trace names no constant: the global is read each time it is entered.
    let source = "
        local f1 = function() end
        gf = f1
        local uf = f1
        local function run(n)
          local c = 0.0
          for i = 1, n do
            if gf == uf then c = c + 1.0 else c = c - 1.0 end
          end
          return c
        end
        local r = 0.0
        for k = 1, 30 do r = r + run(300) end
        local middle = r
        gf = function() end
        for k = 1, 30 do r = r + run(300) end
        return string.format('%.0f,%.0f', middle, r)";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("9000,0"), "{profile:?}: {result}");
        // Lua 5.2 and later reach globals through the `_ENV` table, which
        // takes table operations. Only the profiles with a global opcode
        // trace the read.
        if HAS_GLOBAL_OPCODE.contains(&profile) {
            assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
        }
    }
}

#[test]
fn tables_are_compared_by_identity_and_never_by_a_native_eq() {
    // `T` and `ut` are different tables whose `__eq` says they are equal.
    // The interpreter must run that comparison, so the trace has to leave
    // before it. Counting the true results proves the metamethod ran.
    let source = "
        local mt = { __eq = function() return true end }
        local a, b = setmetatable({}, mt), setmetatable({}, mt)
        T = a
        local ut = a
        local function run(n)
          local c = 0.0
          for i = 1, n do
            if T == ut then c = c + 1.0 end
          end
          return c
        end
        local r = 0.0
        for k = 1, 30 do r = r + run(200) end
        local same = r
        T = b
        for k = 1, 30 do r = r + run(200) end
        return string.format('%.0f,%.0f', same, r)";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("6000,12000"), "{profile:?}: {result}");
        if HAS_GLOBAL_OPCODE.contains(&profile) {
            assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
        }
    }
}

#[test]
fn a_string_constant_rides_to_the_exit_through_the_root_bank() {
    let source = "
        local function run(n)
          local s = 'start'
          local x = 0.0
          for i = 1, n do s = 'tag' x = x + 1.0 end
          return s .. string.format('%.0f', x)
        end
        local r
        for k = 1, 30 do r = run(300) end
        return r";
    let results = agree_natively(&ALL_PROFILES, source);
    assert!(results[0].contains("tag300"), "{}", results[0]);
}

#[test]
fn a_trace_never_keeps_an_object_alive_between_runs() {
    // `t` rides through the trace as a table lane. Once the program drops it,
    // the weak table must lose its entry, and no trace may still hold it.
    let source = "
        local weak = setmetatable({}, { __mode = 'v' })
        local function run(t, n)
          local c = 0.0
          local alias = t
          for i = 1, n do if alias == t then c = c + 1.0 end end
          return c
        end
        local function setup()
          local t = {}
          weak[1] = t
          for k = 1, 40 do run(t, 300) end
        end
        setup()
        collectgarbage() collectgarbage()
        return tostring(weak[1] == nil)";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(result.contains("true"), "{profile:?}: {result}");
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
    }
}

#[test]
fn swapping_two_upvalue_tables_in_a_trace_keeps_both_alive() {
    // The loop swaps two table cells every iteration. The tables ride
    // natively as lanes, and the collector runs at a poll, so both must come
    // back from the exit into their cells.
    let source = "
        local a, b = { name = 'a' }, { name = 'b' }
        local weak = setmetatable({ a, b }, { __mode = 'v' })
        local function churn(n)
          for i = 1, n do a, b = b, a end
        end
        churn(300000)
        collectgarbage() collectgarbage()
        return a.name .. b.name .. tostring(weak[1] ~= nil) .. tostring(weak[2] ~= nil)";
    for profile in ALL_PROFILES {
        let (result, stats) = agree(profile, source);
        assert!(
            result.contains("abtruetrue") || result.contains("batruetrue"),
            "{profile:?}: {result}"
        );
        assert!(
            stats.exits_of(TraceExitKind::Poll) >= 1,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn recording_a_loop_that_makes_a_closure_leaves_results_alone() {
    // Closure creation is not traceable. The loop still computes the same.
    let source = "
        local fns = {}
        for i = 1, 200 do fns[#fns + 1] = function() return i end end
        local s = 0
        for i = 1, #fns do s = s + fns[i]() end
        return s";
    for profile in ALL_PROFILES {
        agree(profile, source);
    }
}

// ------------------------------------------------------ seeded programs

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).unwrap()
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

const FLOATS: &[&str] = &[
    "0.5", "-0.5", "1.5", "2.0", "-3.25", "0.1", "7.0", "-0.0", "100.0",
];
/// Plain locals, upvalues of `f`, and a local that a closure captured.
const VARIABLES: &[&str] = &["a", "b", "u1", "u2", "u3", "cap"];
const COMPARISONS: &[&str] = &["<", "<=", "==", "~=", ">", ">="];

struct CellGenerator {
    rng: Rng,
}

impl CellGenerator {
    fn atom(&mut self) -> String {
        match self.rng.below(6) {
            0 => self.rng.pick(FLOATS).to_string(),
            1 => "i".to_string(),
            _ => self.rng.pick(VARIABLES).to_string(),
        }
    }

    fn expression(&mut self) -> String {
        let (left, right) = (self.atom(), self.atom());
        match self.rng.below(6) {
            0 => format!("-({left})"),
            1 => format!("({left} + {right}) * 0.5"),
            _ => format!("{left} {} {right}", self.rng.pick(&["+", "-", "*"])),
        }
    }

    fn condition(&mut self) -> String {
        format!(
            "{} {} {}",
            self.atom(),
            self.rng.pick(COMPARISONS),
            self.atom()
        )
    }

    fn assignment(&mut self) -> String {
        format!("{} = {}", self.rng.pick(VARIABLES), self.expression())
    }

    fn statement(&mut self, depth: usize) -> String {
        match self.rng.below(6) {
            0 if depth < 2 => format!(
                "if {} then {} else {} end",
                self.condition(),
                self.statement(depth + 1),
                self.statement(depth + 1)
            ),
            1 => {
                // Keep every variable finite, so branches stay interesting.
                let variable = self.rng.pick(VARIABLES);
                format!(
                    "if {variable} ~= {variable} or {variable} > 1e30 or {variable} < -1e30 then {variable} = 1.0 end"
                )
            }
            _ => self.assignment(),
        }
    }

    fn program(&mut self) -> String {
        let body: Vec<String> = (0..=self.rng.below(5)).map(|_| self.statement(0)).collect();
        let body = body.join("\n      ");
        // Every few iterations the interpreter calls a closure that reads and
        // changes the shared variables, so every exit must leave the cells
        // exactly as they would be.
        let touch = match self.rng.below(3) {
            0 => "",
            _ => "if i % 53 == 0 then touch() end",
        };
        let calls: Vec<String> = (0..=self.rng.below(3))
            .map(|_| {
                format!(
                    "ra, rb = run({})",
                    [70, 90, 150, 300, 1_200][self.rng.below(5)]
                )
            })
            .collect();
        let calls = calls.join("\n             ");
        let (u1, u2, u3, cap, a, b) = (
            self.rng.pick(FLOATS),
            self.rng.pick(FLOATS),
            self.rng.pick(FLOATS),
            self.rng.pick(FLOATS),
            self.rng.pick(FLOATS),
            self.rng.pick(FLOATS),
        );
        format!(
            "local u1, u2, u3 = {u1}, {u2}, {u3}
             local cap = {cap}
             local function touch() cap = cap + 1.0 u1 = u1 + 2.0 end
             local function run(n)
               local a, b = {a}, {b}
               for i = 1, n do
                 {body}
                 {touch}
               end
               return a, b
             end
             local ra, rb = 0.0, 0.0
             {calls}
             return ra, rb, u1, u2, u3, cap"
        )
    }
}

fn number_bits(
    source: &str,
    profile: CompatibilityProfile,
    mode: ExecutionMode,
) -> (Vec<u64>, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{profile:?} {mode:?}: {error}\n{source}"));
    let bits = results
        .iter()
        .map(|value| match value {
            Value::Number(number) => number.to_bits(),
            // Lua 5.3 and later keep `i + i` an integer. Keep the two
            // subtypes apart so the comparison sees a wrong one.
            Value::Integer(integer) => integer.cast_unsigned() ^ 0xa5a5_a5a5_a5a5_a5a5,
            other => panic!("unexpected result {other:?}\n{source}"),
        })
        .collect();
    (bits, lua.vm().jit_statistics().trace)
}

#[test]
fn generated_loops_over_cells_and_captured_locals_match_the_interpreter_bit_for_bit() {
    let mut total = TraceStatistics::default();
    let mut native_programs = 0;
    let count = 120;
    for seed in 1..=count {
        let mut generator = CellGenerator {
            rng: Rng(0x9e37_79b9_7f4a_7c15 ^ (seed * 0x1000_0000_01b3)),
        };
        let source = generator.program();
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::LuaJit21,
        ] {
            let (expected, _) = number_bits(&source, profile, ExecutionMode::Interpreter);
            let (tracing, stats) = number_bits(&source, profile, ExecutionMode::Tracing);
            assert_eq!(
                tracing, expected,
                "seed {seed} {profile:?} tracing\n{source}\n{stats:?}"
            );
            if profile == CompatibilityProfile::Lua51 {
                if stats.iterations > 0 {
                    native_programs += 1;
                }
                total.native_entries += stats.native_entries;
                total.iterations += stats.iterations;
                for kind in TraceExitKind::ALL {
                    total.exits[kind.index()] += stats.exits_of(kind);
                }
            }
        }
    }
    assert!(
        native_programs * 2 > count,
        "{native_programs} of {count} programs ran natively: {total:?}"
    );
    assert!(total.iterations > 50_000, "{total:?}");
    assert!(total.exits_of(TraceExitKind::LoopComplete) > 0, "{total:?}");
    assert!(total.exits_of(TraceExitKind::GuardFail) > 0, "{total:?}");
}

#[test]
fn a_rejected_outer_loop_does_not_starve_an_inner_trace_of_its_entries() {
    // The outer loop calls a function, which the tracer cannot follow. The
    // rejected loop arms a global backedge skip on
    // every iteration. The inner loop's trace must still be entered at its
    // first backedge on every call, not 64 iterations in.
    let source = "
        local count = 0.0
        local function bump(n) for i = 1, n do count = count + 1.0 end end
        for k = 1, 5000 do bump(80) end
        return count";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
        let (result, stats) = run(profile, ExecutionMode::Tracing, source);
        assert!(result.contains("400000"), "{profile:?}: {result}");
        assert!(
            stats.iterations > 380_000,
            "{profile:?}: only {} of 400000 iterations ran natively",
            stats.iterations
        );
    }
}
