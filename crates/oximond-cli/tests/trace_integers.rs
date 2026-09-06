//! Profile-correct integer traces through the whole stack: generated Lua
//! programs over integers and floats, in every profile that has them.
//!
//! Seeded programs mix integer and float arithmetic, floor division and
//! modulo, bitwise operations and shifts, mixed integer and float
//! comparisons, and the three loop forms, over the awkward values: the
//! integer extremes, numbers around 2^53 and 2^63, minus one and zero
//! divisors. Each runs under the interpreter and under tracing execution, and
//! the results must match exactly, integer against float included. Counters
//! must show that native trace code ran across the corpus, because equal
//! output alone cannot tell a trace from a slow path. Where stock `lua5.4` or `lua5.5` binaries exist, the
//! same programs also run there, since the interpreter is no authority on
//! its own semantics.

use oximond::vm::{TraceExitKind, TraceStatistics};
use oximond::{CompatibilityProfile, ExecutionMode, Lua, Value};
use std::fmt::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const INTEGER_PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
];

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

const INTEGER_CONSTANTS: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "3",
    "7",
    "-7",
    "100",
    "63",
    "64",
    "math.maxinteger",
    "math.mininteger",
    "9007199254740992",
    "9007199254740993",
    "-9007199254740993",
    "0x7fffffff",
    "(1 << 62)",
];
const FLOAT_CONSTANTS: &[&str] = &[
    "0.0",
    "-0.0",
    "0.5",
    "-2.5",
    "1.5",
    "0.1",
    "1e15",
    "9007199254740992.0",
    "9007199254740994.0",
    "9223372036854775808.0",
    "-9223372036854775808.0",
    "1e300",
    "(1/0)",
    "(-1/0)",
];
const COMPARISONS: &[&str] = &["<", "<=", "==", "~=", ">", ">="];

/// Which variables hold integers (`a`, `b`) and which floats (`c`, `d`), and
/// whether the loop index is in scope and of what kind. Every assignment
/// keeps a variable's kind, so loops stay type-stable for the recorder while
/// the operations still cover every integer and float combination.
struct Generator {
    rng: Rng,
    index: Option<bool>,
    /// Restrict to operations whose results stock Lua and Oximond agree on
    /// bit for bit (no float `//` and `%`, which stock Lua spells with fmod).
    stock: bool,
}

impl Generator {
    fn integer_atom(&mut self) -> String {
        match self.rng.below(6) {
            0 | 1 => self.rng.pick(INTEGER_CONSTANTS).to_string(),
            2 if self.index == Some(false) => "i".to_string(),
            _ => self.rng.pick(&["a", "b"]).to_string(),
        }
    }

    fn float_atom(&mut self) -> String {
        match self.rng.below(6) {
            0 | 1 => self.rng.pick(FLOAT_CONSTANTS).to_string(),
            2 if self.index == Some(true) => "i".to_string(),
            _ => self.rng.pick(&["c", "d"]).to_string(),
        }
    }

    fn number_atom(&mut self) -> String {
        if self.rng.below(2) == 0 {
            self.integer_atom()
        } else {
            self.float_atom()
        }
    }

    /// A divisor that is never zero (an odd number), or occasionally raw.
    fn divisor(&mut self) -> String {
        let atom = self.integer_atom();
        if self.rng.below(12) == 0 {
            atom
        } else {
            format!("({atom} | 1)")
        }
    }

    fn integer_expression(&mut self) -> String {
        let left = self.integer_atom();
        match self.rng.below(12) {
            0 => format!("-({left})"),
            1 => format!("~({left})"),
            2 => format!("({left} // {})", self.divisor()),
            3 => format!("({left} % {})", self.divisor()),
            4 => format!("({left} << ({} % 70 - 3))", self.integer_atom()),
            5 => format!("({left} >> ({} % 70 - 3))", self.integer_atom()),
            6 => {
                let op = self.rng.pick(&["&", "|", "~"]);
                format!("({left} {op} {})", self.integer_atom())
            }
            7 => {
                let op = self.rng.pick(&["+", "-", "*"]);
                format!(
                    "(({left} {op} {}) {} {})",
                    self.integer_atom(),
                    self.rng.pick(&["+", "-", "*"]),
                    self.integer_atom()
                )
            }
            _ => {
                let op = self.rng.pick(&["+", "-", "*"]);
                format!("({left} {op} {})", self.integer_atom())
            }
        }
    }

    fn float_expression(&mut self) -> String {
        let left = self.float_atom();
        // `+ - *` need a float operand to stay float, which `left` is.
        let operators: &[&str] = if self.stock {
            &["+", "-", "*", "/", "^"]
        } else {
            &["+", "-", "*", "/", "^", "//", "%"]
        };
        match self.rng.below(8) {
            0 => format!("-({left})"),
            1 => {
                let right = self.number_atom();
                format!("({right} / {left})")
            }
            2 => {
                // Integers divide and raise to powers into floats.
                let (a, b) = (self.integer_atom(), self.integer_atom());
                format!("({a} {} {b})", self.rng.pick(&["/", "^"]))
            }
            _ => {
                let op = self.rng.pick(operators);
                let right = self.number_atom();
                if self.rng.below(2) == 0 {
                    format!("({left} {op} {right})")
                } else {
                    format!("({right} {op} {left})")
                }
            }
        }
    }

    fn condition(&mut self) -> String {
        let condition = format!(
            "{} {} {}",
            self.number_atom(),
            self.rng.pick(COMPARISONS),
            self.number_atom()
        );
        match self.rng.below(5) {
            0 => format!("not ({condition})"),
            1 => format!(
                "{condition} and {} < {}",
                self.number_atom(),
                self.number_atom()
            ),
            _ => condition,
        }
    }

    fn statement(&mut self, depth: usize) -> String {
        match self.rng.below(8) {
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
            // Keep floats finite often enough that branches stay interesting.
            2 => format!(
                "if {0} ~= {0} or {0} > 1e30 or {0} < -1e30 then {0} = 1.5 end",
                self.rng.pick(&["c", "d"])
            ),
            3 | 4 => format!(
                "{} = {}",
                self.rng.pick(&["c", "d"]),
                self.float_expression()
            ),
            _ => format!(
                "{} = {}",
                self.rng.pick(&["a", "b"]),
                self.integer_expression()
            ),
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
        match self.rng.below(7) {
            0 => {
                self.index = Some(false);
                let body = self.body();
                self.index = None;
                // Ranges that end on the integer limits stop without wrapping.
                let range = match self.rng.below(12) {
                    0 => format!("1, {count}"),
                    1 => format!("{count}, 1, -1"),
                    2 => format!("1, {}, 3", count * 3),
                    3 => format!("-{count}, {count}, 7"),
                    4 => format!("math.maxinteger - {count}, math.maxinteger"),
                    5 => format!("math.maxinteger - {}, math.maxinteger, 7", count * 7 + 4),
                    6 => format!("math.mininteger + {count}, math.mininteger, -1"),
                    7 => format!("math.mininteger + {}, math.mininteger, -7", count * 7 + 4),
                    8 => format!("math.mininteger, math.mininteger + {count}"),
                    9 => format!("math.maxinteger, math.maxinteger - {count}, -1"),
                    10 => format!(
                        "math.mininteger, math.maxinteger, math.maxinteger // {}",
                        count / 2
                    ),
                    _ => format!("0, {count}, {}", self.rng.pick(&["1", "2", "5"])),
                };
                format!("for i = {range} do\n    {body}\nend")
            }
            1 => {
                // A float loop: any float operand makes the whole loop float.
                self.index = Some(true);
                let body = self.body();
                self.index = None;
                let range = match self.rng.below(4) {
                    0 => format!("1.0, {count}"),
                    1 => format!("{count}.0, 1, -1"),
                    2 => format!("1, {count}, 0.5"),
                    _ => format!("0, {count}, 1.5"),
                };
                format!("for i = {range} do\n    {body}\nend")
            }
            2 => {
                let body = self.body();
                format!("local k = 0\nwhile k < {count} do\n    {body}\n    k = k + 1\nend")
            }
            3 => {
                let body = self.body();
                format!("local k = 0\nrepeat\n    {body}\n    k = k + 1\nuntil k >= {count}")
            }
            4 => {
                let body = self.body();
                let cut = 20 + self.rng.below(60);
                format!(
                    "local k = 0\nwhile true do\n    {body}\n    k = k + 1\n    if k > {cut} + {count} then break end\nend"
                )
            }
            5 => {
                // Countdown loops that reach a zero divisor mid-trace.
                let divisor = self.rng.pick(&["a", "b"]);
                self.index = Some(false);
                let body = self.body();
                self.index = None;
                format!(
                    "local z = {count}\nfor i = 1, {count} do\n    {body}\n    {divisor} = {divisor} + 1000 // (z - i) % 5000\nend"
                )
            }
            _ => {
                self.index = Some(false);
                let body = self.body();
                self.index = None;
                format!(
                    "for i = 1, 100, 0 * a do\n    {body}\n    break\nend\nfor i = 5, 1 do\n    {body}\nend"
                )
            }
        }
    }

    /// A function so the loops see registers, not upvalues. The result is
    /// `pcall`'s: a flag, then the four variables or the error.
    fn program(&mut self) -> String {
        let mut source = String::from("local function run(a, b, c, d)\n");
        for _ in 0..=self.rng.below(2) {
            source.push_str(&self.loop_source());
            source.push('\n');
        }
        source.push_str("return a, b, c, d\nend\n");
        writeln!(
            source,
            "return pcall(run, {}, {}, {}, {})",
            self.rng.pick(INTEGER_CONSTANTS),
            self.rng.pick(INTEGER_CONSTANTS),
            self.rng.pick(FLOAT_CONSTANTS),
            self.rng.pick(FLOAT_CONSTANTS)
        )
        .expect("formatting into a String cannot fail");
        source
    }
}

/// A canonical, bit-exact rendering of a result, so NaN payloads, signed
/// zero and the integer/float distinction all count.
fn canon(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .map(|value| match value {
            Value::Integer(integer) => format!("I{integer}"),
            Value::Number(number) => format!("F{:016x}", number.to_bits()),
            Value::Boolean(flag) => format!("B{flag}"),
            // An error message: the text is the interpreter's own.
            other => format!("{other:?}"),
        })
        .collect()
}

fn run(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Vec<String>, TraceStatistics) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    lua.vm_mut().count_trace_links();
    let results: Vec<Value> = lua
        .exec(source)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    (canon(&results), lua.vm().jit_statistics().trace)
}

fn add(total: &mut TraceStatistics, stats: &TraceStatistics) {
    total.traces_compiled += stats.traces_compiled;
    total.native_entries += stats.native_entries;
    total.iterations += stats.iterations;
    for kind in TraceExitKind::ALL {
        total.exits[kind.index()] += stats.exits_of(kind);
    }
}

#[test]
fn generated_programs_match_the_interpreter_bit_for_bit_and_run_natively() {
    let mut total = TraceStatistics::default();
    let mut programs_with_native_iterations = 0;
    let mut runs = 0;
    let mut completed = 0;
    let program_count = 90;
    for seed in 1..=program_count {
        let mut generator = Generator {
            rng: Rng(0x9e37_79b9_7f4a_7c15 ^ (seed * 0x1000_0000_01b3)),
            index: None,
            stock: false,
        };
        let source = generator.program();
        for profile in INTEGER_PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &source);
            let (tracing, stats) = run(profile, ExecutionMode::Tracing, &source);
            assert_eq!(
                tracing, expected,
                "seed {seed} {profile:?} tracing\n{source}\n{stats:?}"
            );
            runs += 1;
            completed += usize::from(expected[0] == "Btrue");
            if stats.iterations > 0 {
                programs_with_native_iterations += 1;
            }
            add(&mut total, &stats);
            // Every entry that ran code ended in exactly one exit.
            assert_eq!(
                stats.exits.iter().sum::<u64>(),
                stats.native_entries,
                "seed {seed}: {stats:?}"
            );
        }
    }
    assert!(
        programs_with_native_iterations * 2 > runs,
        "{programs_with_native_iterations} of {runs} runs ran natively: {total:?}"
    );
    // Most programs run to the end, so the comparison covers real results
    // and not just the same early error three times.
    assert!(completed * 2 > runs, "{completed} of {runs} completed");
    assert!(total.iterations > 100_000, "{total:?}");
    assert!(total.exits_of(TraceExitKind::LoopComplete) > 0, "{total:?}");
    assert!(total.exits_of(TraceExitKind::GuardFail) > 0, "{total:?}");
}

#[test]
fn float_only_profiles_still_trace_integer_looking_programs_as_floats() {
    // Lua 5.1, 5.2 and LuaJIT have no integers: `1 // 2` does not parse, but
    // whole numbers are floats, and the same traces run.
    let programs = [
        "local s = 0 for i = 1, 500 do s = s + i % 7 * 3 - i / 4 end return s",
        "local s, x = 0, 1 for i = 1, 400 do x = (x * 3 + i) % 1000 s = s + x end return s, x",
        "local n = 0 for i = 1, 500 do if i % 3 == 0 then n = n + 2 else n = n - 1 end end return n",
    ];
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::LuaJit21,
    ] {
        for source in programs {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?}: {source}");
            assert!(stats.iterations > 0, "{profile:?}: {source}\n{stats:?}");
            assert!(expected.iter().all(|value| value.starts_with('F')));
        }
    }
}

/// Programs that each target one corner of integer semantics. Each is hot
/// enough to record and compile, and the interpreter and the stock binaries
/// are the oracles.
const CORNERS: &[(&str, &str)] = &[
    (
        "wrapping arithmetic",
        "local x = 1 for i = 1, 400 do x = x * 3 + i end return x",
    ),
    (
        "wrapping negation and subtraction",
        "local x = math.mininteger for i = 1, 300 do x = -x - 1 end return x",
    ),
    (
        "floor division by minus one",
        "local x = math.mininteger local n = 0 for i = 1, 300 do x = x // -1 n = n + 1 end return x, n",
    ),
    (
        "modulo by minus one",
        "local x = math.mininteger local s = 0 for i = 1, 300 do s = s + x % -1 end return s",
    ),
    (
        "floor division and modulo signs",
        "local s = 0 for i = -150, 150 do s = s + (i // 7) * 3 + (i % -7) - (-i // 5) end return s",
    ),
    (
        "division by zero mid loop",
        "local s = 0 for i = 1, 400 do s = s + 1000 // (200 - i) end return s",
    ),
    (
        "modulo by zero mid loop",
        "local s = 0 for i = 1, 400 do s = s + 1000 % (250 - i) end return s",
    ),
    (
        "shifts across the word size",
        "local x, s = 0x123456789abcdef1, 0 for i = 1, 300 do s = s ~ (x << (i - 150)) ~ (x >> (i - 150)) end return s",
    ),
    (
        "bitwise operations",
        "local s = 0 for i = 1, 300 do s = (s ~ i) | (i << 3) & ~(i >> 1) end return s",
    ),
    (
        "loop to the largest integer",
        "local n = 0 for i = math.maxinteger - 300, math.maxinteger do n = n + 1 end return n",
    ),
    (
        "loop past the largest integer by steps",
        "local n, last = 0, 0 for i = math.maxinteger - 704, math.maxinteger, 7 do n = n + 1 last = i end return n, last",
    ),
    (
        "loop to the smallest integer",
        "local n = 0 for i = math.mininteger + 300, math.mininteger, -1 do n = n + 1 end return n",
    ),
    (
        "loop down past the smallest integer by steps",
        "local n, last = 0, 0 for i = math.mininteger + 704, math.mininteger, -7 do n = n + 1 last = i end return n, last",
    ),
    (
        "loop with a huge step",
        "local n = 0 for i = math.mininteger, math.maxinteger, math.maxinteger // 150 do n = n + 1 end return n",
    ),
    (
        "mixed comparisons near 2^53",
        "local f, n = 9007199254740992.0 for i = 0, 300 do local k = 9007199254740992 + i if f < k then n = (n or 0) + 1 end if f == k then n = (n or 0) + 10 end if k <= f then n = (n or 0) + 100 end end return n",
    ),
    (
        "mixed comparisons near 2^63",
        "local n, hi, lo = 0, math.maxinteger, math.mininteger for i = 0, 300 do local k = hi - i if k < 9223372036854775808.0 then n = n + 1 end if k == 9223372036854775808.0 then n = n + 1000 end if 9223372036854775808.0 <= k then n = n + 1000000 end if -9223372036854775808.0 <= lo + i then n = n + 5 end end return n",
    ),
    (
        "integer and float stay apart",
        "local a, b = 0, 0.0 for i = 1, 300 do a = a + 1 b = b + 1 end return a, b, a == b, math.type(a), math.type(b)",
    ),
    (
        "integer division and power make floats",
        "local s = 0.0 for i = 1, 300 do s = s + i / 4 + i ^ 2 end return s, math.type(s)",
    ),
    (
        "float floor division and modulo",
        "local s = 0.0 for i = 1, 300 do s = s + (i + 0.5) // 2 + (i - 0.25) % 3 end return s",
    ),
    (
        "float loop from integers and floats",
        "local s = 0 for i = 1.0, 300 do s = s + i end local t = 0 for i = 1, 300, 0.5 do t = t + i end return s, t, math.type(s), math.type(t)",
    ),
    (
        "unary minus on integers and floats",
        "local a, b = 5, 5.0 for i = 1, 301 do a = -a b = -b end return a, b",
    ),
];

#[test]
fn corner_programs_match_the_interpreter_in_every_integer_profile() {
    for (name, source) in CORNERS {
        let wrapped = format!("return pcall(function() {source} end)");
        for profile in INTEGER_PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &wrapped);
            let (actual, stats) = run(profile, ExecutionMode::Tracing, &wrapped);
            assert_eq!(actual, expected, "{name} {profile:?}");
            // Native code ran, and every program here has a hot loop
            // the recorder can follow. A trace-free run proves nothing.
            assert!(stats.iterations > 0, "{name} {profile:?}: {stats:?}");
        }
    }
}

#[test]
fn zero_divisors_raise_the_interpreters_error_from_inside_a_native_loop() {
    for (name, source) in [
        (
            "division",
            "local s = 0 for i = 1, 400 do s = s + 1000 // (200 - i) end return s",
        ),
        (
            "modulo",
            "local s = 0 for i = 1, 400 do s = s + 1000 % (250 - i) end return s",
        ),
    ] {
        let wrapped = format!("return pcall(function() {source} end)");
        for profile in INTEGER_PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, &wrapped);
            assert_eq!(expected[0], "Bfalse", "{name}: the program errors");
            let (actual, stats) = run(profile, ExecutionMode::Tracing, &wrapped);
            assert_eq!(actual, expected, "{name} {profile:?}");
            // The loop ran natively until the divisor reached zero, then left
            // through its guard so the interpreter could raise the error.
            assert!(stats.iterations > 50, "{name} {profile:?}: {stats:?}");
            assert!(
                stats.exits_of(TraceExitKind::GuardFail) > 0,
                "{name} {profile:?}: {stats:?}"
            );
        }
    }
}

#[test]
fn integer_and_float_registers_get_separate_entry_guards() {
    // The same function runs hot with an integer, then a float, then an
    // integer again. Each call must compute in its own number type.
    let source = "local function f(x) for i = 1, 300 do x = x + 1 end return x end \
                  local a = f(0) local b = f(0.5) local c = f(7) local d = f(0.25) \
                  return a, b, c, d, math.type(a), math.type(b), math.type(c), math.type(d)";
    for profile in INTEGER_PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        assert_eq!(expected[0], "I300");
        assert!(expected[1].starts_with('F'));
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.iterations > 0, "{profile:?}: {stats:?}");
        // The float call meets an integer trace at entry.
        assert!(
            stats.exits_of(TraceExitKind::EntryMismatch) > 0,
            "{profile:?}: {stats:?}"
        );
    }
}

#[test]
fn numeric_strings_take_part_through_a_conversion_helper() {
    // Arithmetic on numeric strings converts them as the interpreter does,
    // by a helper call of the trace, so these loops compile and give the
    // interpreter's results, integer or float (see `trace_string_arith`).
    let programs = [
        "local s, x = '10', 0 for i = 1, 300 do x = x + s end return x",
        "local x = 0 for i = 1, 300 do x = x + (i .. '') end return x",
    ];
    for source in programs {
        for profile in INTEGER_PROFILES {
            let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
            let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
            assert_eq!(actual, expected, "{profile:?}: {source}");
            assert!(stats.iterations > 0, "{profile:?}: {source}: {stats:?}");
            assert!(stats.traces_compiled > 0, "{profile:?}: {source}");
        }
    }
    // A string constant is a root of the trace, which no operation reads.
    let source = "local x = 0 for i = 1, 300 do x = x + ('3' * i) end return x";
    for profile in INTEGER_PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}: {source}");
        assert_eq!(stats.traces_compiled, 0, "{profile:?}: {source}");
    }
}

#[test]
fn a_branch_that_flips_every_few_iterations_gets_a_side_trace_and_stays_native() {
    // The branch is taken three times in seven. Without a side trace no trace
    // survives long: temporaries the two arms leave typed or cleared
    // differently also gave the header many entry signatures, and every
    // retired version used to be replaced by a newly recorded one for the
    // whole run. The guard that fails now continues in a side trace, the tree
    // runs both arms natively, and the loop keeps its trace.
    let source = "local yes, no = 0, 0 \
                  for i = 1, 300000 do \
                      if i % 7 < 3 then yes = yes + i % 101 else no = no + i % 89 end \
                  end return yes, no";
    for profile in INTEGER_PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_compiled <= 12, "{profile:?}: {stats:?}");
        assert!(stats.side_installs >= 1, "{profile:?}: {stats:?}");
        assert!(stats.iterations > 290_000, "{profile:?}: {stats:?}");
        assert!(stats.side_entries > 100_000, "{profile:?}: {stats:?}");
    }
}

#[test]
fn a_branch_with_an_arm_no_trace_can_hold_stops_costing_recompiles() {
    // Same loop, but the `else` arm calls a native function, which no trace
    // contains. A side trace cannot be recorded for it, so the root keeps
    // failing its guard, and the header gives up after a handful of compiles.
    let source = "local yes, no = 0, 0 \
                  for i = 1, 300000 do \
                      if i % 7 < 3 then yes = yes + i % 101 else no = no + (os.clock() < 0 and 0 or 1) end \
                  end return yes, no";
    for profile in INTEGER_PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(stats.traces_compiled <= 12, "{profile:?}: {stats:?}");
        assert_eq!(stats.side_installs, 0, "{profile:?}: {stats:?}");
    }
}

// ---------------------------------------------------------- stock Lua

/// Renders results in a form stock Lua and Oximond print identically.
const SHOW: &str = r#"
local function show(v)
  if v ~= v then return "nan" end
  local t = math.type(v)
  if t == "integer" then return string.format("%d", v) end
  if t == "float" then return string.format("%.17g", v) .. "f" end
  return tostring(v)
end
"#;

/// The script that prints a program's results, one per line. An error prints
/// only `error`, because stock Lua words its messages differently.
fn printing_script(source: &str) -> String {
    format!(
        "{SHOW}local r = table.pack((function()\n{source}\nend)())\n\
         if r[1] == false then print('error') else\n\
         for i = 1, r.n do print(show(r[i])) end end\n"
    )
}

fn stock(binary: &str, script: &str) -> Option<String> {
    // Tests run in parallel, so every script gets its own file.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory =
        std::env::temp_dir().join(format!("oximond-trace-integers-{}", std::process::id()));
    std::fs::create_dir_all(&directory).ok()?;
    let path = directory.join(format!(
        "{binary}-{}.lua",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, script).ok()?;
    let output = Command::new(binary).arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    assert!(output.status.success(), "{binary}: {output:?}\n{script}");
    Some(String::from_utf8(output.stdout).unwrap())
}

fn oximond_output(profile: CompatibilityProfile, mode: ExecutionMode, script: &str) -> String {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    // These programs test what traces do with small loops of a main chunk.
    lua.vm_mut().ignore_remaining_work();
    // The script prints through a collector instead of stdout.
    let collected = format!(
        "local out = {{}}\nlocal print = function(...) out[#out + 1] = table.concat({{...}}, '\\t') end\n{script}\nreturn table.concat(out, '\\n') .. '\\n'"
    );
    let results: Vec<Value> = lua
        .exec(&collected)
        .unwrap_or_else(|error| panic!("{error}\n{script}"));
    match &results[0] {
        Value::String(text) => String::from_utf8_lossy(text.as_bytes()).into_owned(),
        other => panic!("unexpected {other:?}"),
    }
}

fn stock_binaries() -> Vec<(&'static str, CompatibilityProfile)> {
    [
        ("lua5.4", CompatibilityProfile::Lua54),
        ("lua5.5", CompatibilityProfile::Lua55),
    ]
    .into_iter()
    .filter(|(binary, _)| Command::new(binary).arg("-v").output().is_ok())
    .collect()
}

#[test]
fn corner_programs_agree_with_stock_lua() {
    let binaries = stock_binaries();
    if binaries.is_empty() {
        eprintln!("no stock lua5.4 or lua5.5 binary: skipped");
        return;
    }
    for (name, source) in CORNERS {
        let script = printing_script(&format!("return pcall(function() {source} end)"));
        for &(binary, profile) in &binaries {
            let expected = stock(binary, &script).expect("run stock lua");
            for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
                let actual = oximond_output(profile, mode, &script);
                assert_eq!(actual, expected, "{name}: {binary} against {mode:?}");
            }
        }
    }
}

#[test]
fn generated_programs_agree_with_stock_lua() {
    let binaries = stock_binaries();
    if binaries.is_empty() {
        eprintln!("no stock lua5.4 or lua5.5 binary: skipped");
        return;
    }
    let mut compared = 0;
    for seed in 1..=40_u64 {
        let mut generator = Generator {
            rng: Rng(0xd1b5_4a32_d192_ed03 ^ (seed * 0x1000_0000_01b3)),
            index: None,
            stock: true,
        };
        let source = generator.program();
        // Float loops over an integer start and step with a float limit
        // clip the limit in stock Lua, which Oximond's interpreter does not
        // do. The generator never writes one.
        let script = printing_script(&source);
        for &(binary, profile) in &binaries {
            let expected = stock(binary, &script).expect("run stock lua");
            let actual = oximond_output(profile, ExecutionMode::Tracing, &script);
            assert_eq!(actual, expected, "seed {seed}: {binary}\n{source}");
            compared += 1;
        }
    }
    assert!(compared >= 40, "{compared} comparisons");
}

#[test]
fn a_productive_version_survives_an_unproductive_sibling_signature() {
    // The integer signature is hot. The float signature records a sibling
    // version that never gets enough loop-backs per entry and keeps
    // retiring. The retirements must not take the integer version down.
    let source = "local function f(a, n) local s = a; for i = 1, n do s = s + i end return s end \
                  local t = 0 \
                  for round = 1, 3000 do \
                    t = t + f(1, 2000) \
                    for k = 1, 6 do t = t + f(0.5, 7) end \
                  end return t";
    let profile = CompatibilityProfile::Lua54;
    let (expected, _) = run(profile, ExecutionMode::Interpreter, source);
    let (actual, stats) = run(profile, ExecutionMode::Tracing, source);
    assert_eq!(actual, expected);
    assert!(
        stats.iterations > 5_000_000,
        "lost the hot integer version: {stats:?}"
    );
}

/// The alternating-branch loop of `benchmarks/lua/control/branching.lua`.
/// Its temporaries hold nil on one path and integers on the other when the
/// loop header runs, so every type they take used to open a trace version.
const ALTERNATING_BRANCH: &str = "local yes, no = 0, 0 \
     for i = 1, 200000 do \
       if i % 7 < 3 then yes = yes + i % 101 else no = no + i % 89 end \
     end return yes * 1000003 + no";

#[test]
fn dead_temporaries_do_not_multiply_trace_versions() {
    for profile in INTEGER_PROFILES {
        let (expected, _) = run(profile, ExecutionMode::Interpreter, ALTERNATING_BRANCH);
        let (actual, stats) = run(profile, ExecutionMode::Tracing, ALTERNATING_BRANCH);
        assert_eq!(actual, expected, "{profile:?}");
        assert!(
            stats.version_installs == 0,
            "{profile:?} opened versions for dead temporaries: {stats:?}"
        );
    }
}
