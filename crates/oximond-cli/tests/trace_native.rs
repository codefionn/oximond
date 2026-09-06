//! Native traces through the whole stack: generated Lua programs.
//!
//! Seeded programs mix float arithmetic, comparisons, branches and the three
//! loop forms over awkward values (NaN, infinities, negative zero, negative
//! modulo, negative and fractional steps). Each runs under the interpreter
//! and under tracing execution. Results must match bit for bit, and counters
//! must show that native trace code ran across the corpus, because equal
//! output alone cannot tell a trace from a slow path.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};

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

const CONSTANTS: &[&str] = &[
    "0", "1", "-1", "2", "0.5", "-0.5", "3", "-7", "7.5", "100", "1e300", "1e-300", "0.1", "1/0",
    "-1/0", "0/0", "-0",
];
const VARIABLES: &[&str] = &["a", "b", "c", "d"];
const COMPARISONS: &[&str] = &["<", "<=", "==", "~=", ">", ">="];
const OPERATORS: &[&str] = &["+", "-", "*", "/", "%", "^"];

struct Generator {
    rng: Rng,
    /// Whether the loop index is in scope.
    index: bool,
}

impl Generator {
    fn atom(&mut self) -> String {
        match self.rng.below(5) {
            0 => self.rng.pick(CONSTANTS).to_string(),
            1 if self.index => "i".to_string(),
            _ => self.rng.pick(VARIABLES).to_string(),
        }
    }

    fn expression(&mut self) -> String {
        match self.rng.below(8) {
            0 => format!("-({})", self.atom()),
            1 => format!(
                "({} {} {}) {} {}",
                self.atom(),
                self.rng.pick(OPERATORS),
                self.atom(),
                self.rng.pick(&["+", "*", "-"]),
                self.atom()
            ),
            _ => format!(
                "{} {} {}",
                self.atom(),
                self.rng.pick(OPERATORS),
                self.atom()
            ),
        }
    }

    fn condition(&mut self) -> String {
        let condition = format!(
            "{} {} {}",
            self.atom(),
            self.rng.pick(COMPARISONS),
            self.atom()
        );
        match self.rng.below(5) {
            0 => format!("not ({condition})"),
            1 => format!("{condition} and {} < {}", self.atom(), self.atom()),
            _ => condition,
        }
    }

    fn statement(&mut self, depth: usize) -> String {
        match self.rng.below(7) {
            0 if depth < 2 => format!(
                "if {} then {} else {} end",
                self.condition(),
                self.statement(depth + 1),
                self.statement(depth + 1)
            ),
            1 if depth < 2 => format!(
                "if {} then {} end",
                self.condition(),
                self.statement(depth + 1)
            ),
            // Keep values finite often enough that branches stay interesting.
            2 => format!(
                "if {0} ~= {0} or {0} > 1e30 or {0} < -1e30 then {0} = 1 end",
                self.rng.pick(VARIABLES)
            ),
            _ => format!("{} = {}", self.rng.pick(VARIABLES), self.expression()),
        }
    }

    fn body(&mut self) -> String {
        let count = 1 + self.rng.below(5);
        (0..count)
            .map(|_| self.statement(0))
            .collect::<Vec<_>>()
            .join("\n    ")
    }

    fn loop_source(&mut self) -> String {
        let count = [70, 90, 150, 200, 400, 5_000][self.rng.below(6)];
        match self.rng.below(5) {
            0 => {
                self.index = true;
                let body = self.body();
                self.index = false;
                let (start, stop, stride) = match self.rng.below(6) {
                    0 => (1.0, f64::from(count), 1.0),
                    1 => (f64::from(count), 1.0, -1.0),
                    2 => (1.0, f64::from(count) * 3.0, 3.0),
                    3 => (0.0, f64::from(count) / 2.0, 0.5),
                    4 => (f64::from(count), 1.0, -2.5),
                    _ => (-f64::from(count), f64::from(count), 7.0),
                };
                format!("for i = {start}, {stop}, {stride} do\n    {body}\nend")
            }
            1 => {
                let body = self.body();
                format!("local k = 0\nwhile k < {count} do\n    {body}\n    k = k + 1\nend")
            }
            2 => {
                let body = self.body();
                format!("local k = 0\nrepeat\n    {body}\n    k = k + 1\nuntil k >= {count}")
            }
            3 => {
                // A loop that exits early through a condition.
                let body = self.body();
                let cut = 20 + self.rng.below(60);
                format!(
                    "local k = 0\nwhile true do\n    {body}\n    k = k + 1\n    if k > {cut} + {count} then break end\nend"
                )
            }
            _ => {
                // Zero and negative steps against ranges that never run, plus
                // an exact-zero accumulating step.
                self.index = true;
                let body = self.body();
                self.index = false;
                format!(
                    "for i = 1, 100, 0 * a do\n    {body}\n    break\nend\nfor i = 5, 1 do\n    {body}\nend"
                )
            }
        }
    }

    fn program(&mut self) -> String {
        let mut source = format!(
            "local a, b, c, d = {}, {}, {}, {}\n",
            self.rng.pick(CONSTANTS),
            self.rng.pick(CONSTANTS),
            self.rng.pick(CONSTANTS),
            self.rng.pick(CONSTANTS)
        );
        for _ in 0..=self.rng.below(2) {
            source.push_str(&self.loop_source());
            source.push('\n');
        }
        source.push_str("return a, b, c, d\n");
        source
    }
}

/// The bit patterns of a program's results, so signed zero counts. Every NaN
/// maps to one pattern: when both operands of an arithmetic operation are
/// NaN, x86 returns the first operand's payload, and Cranelift's optimizer,
/// like LLVM's, may swap the operands of a commutative operation. Rust leaves
/// the sign and payload of a NaN result unspecified for the same reason.
fn bits(values: &[Value]) -> Vec<u64> {
    values
        .iter()
        .map(|value| match value {
            Value::Number(number) if number.is_nan() => f64::NAN.to_bits(),
            Value::Number(number) => number.to_bits(),
            other => panic!("unexpected result {other:?}"),
        })
        .collect()
}

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Vec<u64>, TraceStatistics) {
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
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    (bits(&results), lua.vm().jit_statistics().trace)
}

#[test]
fn generated_programs_match_the_interpreter_bit_for_bit_and_run_natively() {
    let mut total = TraceStatistics::default();
    let mut programs_with_native_iterations = 0;
    let program_count = 140;
    for seed in 1..=program_count {
        let mut generator = Generator {
            rng: Rng(0x9e37_79b9_7f4a_7c15 ^ (seed * 0x1000_0000_01b3)),
            index: false,
        };
        let source = generator.program();
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ] {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (tracing, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                tracing, expected,
                "seed {seed} {profile:?} tracing\n{source}\n{stats:?}"
            );
            if profile == CompatibilityProfile::Lua51 {
                if stats.iterations > 0 {
                    programs_with_native_iterations += 1;
                }
                total.traces_compiled += stats.traces_compiled;
                total.native_entries += stats.native_entries;
                total.iterations += stats.iterations;
                for kind in TraceExitKind::ALL {
                    total.exits[kind.index()] += stats.exits_of(kind);
                }
                for (sum, one) in total.side_exits.iter_mut().zip(stats.side_exits) {
                    *sum += one;
                }
                // Every entry that ran code ended in exactly one exit.
                assert_eq!(
                    stats.exits.iter().sum::<u64>(),
                    stats.native_entries,
                    "seed {seed}: {stats:?}"
                );
            }
        }
    }
    // Most programs have a loop the recorder follows, so the corpus has to
    // show real native work and every common way of leaving a trace.
    assert!(
        programs_with_native_iterations * 2 > program_count,
        "{programs_with_native_iterations} of {program_count} programs ran natively: {total:?}"
    );
    assert!(total.iterations > 100_000, "{total:?}");
    assert!(total.exits_of(TraceExitKind::LoopComplete) > 0, "{total:?}");
    assert!(total.exits_of(TraceExitKind::GuardFail) > 0, "{total:?}");
    assert!(total.side_exits.iter().sum::<u64>() > 0, "{total:?}");
}

#[test]
fn awkward_values_survive_native_loops() {
    // Each program is hot enough to record and runs a native trace. The
    // interpreter computes the expected bits.
    let programs = [
        "local x = 0 for i = 1, 500 do x = x + (-i) % 7 - i % -3 end return x, 0, 0, 0",
        "local x = 1 for i = 1, 500 do x = x * 1.5 end return x, -x, x - x, x / x",
        "local x = -0 for i = 1, 300 do x = x * -1 end return x, 1 / x, 0, 0",
        "local x, y = 0, 0 for i = 1, 300 do x = i % 0 y = x ~= x and 1 or 2 end return y, 0, 0, 0",
        "local x = 1 for i = 1, 400 do x = x ^ 0.5 + i ^ -1 end return x, 0, 0, 0",
        "local s = 0 for i = 300, 1, -3 do s = s + i / 3 end return s, 0, 0, 0",
        "local s = 0 for i = 0, 100, 0.1 do s = s + i end return s, 0, 0, 0",
        "local a, b = 1, 2 for i = 1, 501 do a, b = b, a end return a, b, 0, 0",
        "local a, b, c = 1, 2, 3 for i = 1, 300 do a, b, c = c, a, b end return a, b, c, 0",
        "local x = 1e308 for i = 1, 300 do x = x * 10 end return x, -x, x - x, 0",
        "local n = 0 for i = 1, 400 do if i % 2 == 0 then n = n + 1 elseif i % 3 == 0 then n = n + 2 else n = n - 1 end end return n, 0, 0, 0",
    ];
    for source in programs {
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ] {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?}: {source}");
            assert!(stats.iterations > 0, "{profile:?}: {source}\n{stats:?}");
        }
    }
}

#[test]
fn entries_with_other_types_mismatch_and_run_in_the_interpreter() {
    // The loop compiles for a number in `a`. A string there fails the entry
    // check, and the interpreter runs those calls and gets the same sums.
    let source = "local function f(a, n) local s = a for i = 1, n do s = s + i end return s end \
                  local t = 0 \
                  for round = 1, 200 do t = t + f(1, 50) end \
                  for round = 1, 200 do t = t + f('2', 50) end \
                  return t";
    let (expected, _) = run(
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
        source,
    );
    let (actual, stats) = run(CompatibilityProfile::Lua51, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected);
    assert!(
        stats.exits_of(TraceExitKind::EntryMismatch) > 0,
        "{stats:?}"
    );
}

#[test]
fn a_long_native_loop_polls_and_still_finishes_exactly() {
    let source = "local s = 0 for i = 1, 600000 do s = s + i * 0.5 end return s, 0, 0, 0";
    let (expected, _) = run(
        CompatibilityProfile::Lua51,
        ExecutionMode::Interpreter,
        source,
    );
    let (actual, stats) = run(CompatibilityProfile::Lua51, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected);
    assert!(stats.exits_of(TraceExitKind::Poll) >= 2, "{stats:?}");
    assert!(stats.iterations > 599_000, "{stats:?}");
}

#[test]
fn jit_off_and_flush_control_the_trace_engine() {
    let off = "jit.off() local s = 0 for i = 1, 2000 do s = s + i end return s, 0, 0, 0";
    let (expected, _) = run(
        CompatibilityProfile::LuaJit21,
        ExecutionMode::Interpreter,
        off,
    );
    let (actual, stats) = run(CompatibilityProfile::LuaJit21, ExecutionMode::Tracing, off);
    assert_eq!(actual, expected);
    assert_eq!(stats, TraceStatistics::default(), "jit.off() stops tracing");

    let flush = "local s = 0 for i = 1, 400 do s = s + i end jit.flush() \
                 for i = 1, 400 do s = s + i end return s, 0, 0, 0";
    let (expected, _) = run(
        CompatibilityProfile::LuaJit21,
        ExecutionMode::Interpreter,
        flush,
    );
    let (actual, stats) = run(
        CompatibilityProfile::LuaJit21,
        ExecutionMode::Tracing,
        flush,
    );
    assert_eq!(actual, expected);
    assert!(
        stats.traces_compiled >= 2,
        "each loop compiles after the flush: {stats:?}"
    );
    assert!(stats.iterations > 0, "{stats:?}");
}
