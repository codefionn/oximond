//! Tail-call benchmark probe: instruction mix and frame-replacement breakdown.
//!
//! ```text
//! cargo run --release -p oximond-cli --example tailcall_probe -- benchmarks/lua/calls/tail_call.lua 4
//! cargo run --release -p oximond-cli --example tailcall_probe -- benchmarks/lua/calls/tail_call.lua 4 --mode tracing
//! ```
//!
//! Compiles one Lua file with the Lua 5.1 profile, exposes it as `arg[1]`
//! scaling like the benchmark harness does, and runs it twice: once with
//! interpreter profiling off for an undistorted elapsed time, and once
//! with it on for the instruction mix. Profiling stays off by default in
//! the VM so hot loops skip histogram accounting.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

use oximond::Lua;
use oximond::bytecode::{Chunk, OpCode};
use oximond::compat::CompatibilityProfile;
use oximond::compat::ExecutionMode;
use oximond::compiler::{CompilerConfig, compile_named};
use oximond::runtime::Value;
use oximond::vm::InterpreterCounters;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse(env::args_os().skip(1))?;
    let source = fs::read(&options.source)?;
    let source_name = options.source.to_string_lossy().into_owned();
    let compatibility = CompatibilityProfile::Lua51;
    let chunk = compile_named(source_name, &source, CompilerConfig::new(compatibility))?;

    let (elapsed, _, _, _) = run(&chunk, compatibility, options.mode, options.scale, false)?;
    let (_, counters, values, jit) = run(&chunk, compatibility, options.mode, options.scale, true)?;
    let steps: u64 = counters.opcode_steps().iter().sum();

    println!("result: {values:?}");
    println!("mode: {} scale: {}", options.mode, options.scale);
    println!("jit: {jit:?}");
    println!(
        "elapsed: {elapsed:.3?} steps: {steps} ns/step: {:.1}",
        ns_per_step(elapsed, steps)
    );
    println!(
        "tail_fixed: attempts={} fast={} miss_observed={} miss_callee={} miss_vararg={} generic_entries={}",
        counters.tail_fixed_attempts.get(),
        counters.tail_fixed_fast.get(),
        counters.tail_fixed_miss_observed.get(),
        counters.tail_fixed_miss_callee.get(),
        counters.tail_fixed_miss_vararg.get(),
        counters.tail_generic_entries.get(),
    );
    println!(
        "lookups: get_global={} get_upvalue={} closure_body={} scalar_fast={} scalar_slow={}",
        counters.get_global_lookups.get(),
        counters.get_upvalue_reads.get(),
        counters.closure_body_calls.get(),
        counters.scalar_fast.get(),
        counters.scalar_slow.get(),
    );
    let mut histogram: Vec<(u8, u64)> = counters
        .opcode_steps()
        .iter()
        .enumerate()
        .filter_map(|(index, count)| {
            u8::try_from(index)
                .ok()
                .filter(|_| *count > 0)
                .map(|opcode| (opcode, *count))
        })
        .collect();
    histogram.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    println!("opcode mix:");
    for (opcode, count) in histogram {
        let name = opcode_name(opcode);
        println!(
            "  {name:16} {count:>12} {:>5.1}%",
            share_percent(count, steps)
        );
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "probe display math tolerates float rounding for nanosecond ratios"
)]
fn ns_per_step(elapsed: Duration, steps: u64) -> f64 {
    elapsed.as_nanos() as f64 / steps.max(1) as f64
}

#[allow(
    clippy::cast_precision_loss,
    reason = "probe display math tolerates float rounding for share ratios"
)]
fn share_percent(count: u64, steps: u64) -> f64 {
    100.0 * count as f64 / steps.max(1) as f64
}

fn opcode_name(opcode: u8) -> String {
    OpCode::try_from(opcode).map_or_else(
        |_| format!("unknown({opcode})"),
        |opcode| format!("{opcode:?}"),
    )
}

// The four probe outputs stay one tuple; a type alias would hide what they are.
#[allow(clippy::type_complexity)]
fn run(
    chunk: &Chunk,
    compatibility: CompatibilityProfile,
    mode: ExecutionMode,
    scale: u64,
    step_profiling: bool,
) -> Result<
    (
        Duration,
        InterpreterCounters,
        Vec<Value>,
        oximond::vm::JitStatistics,
    ),
    Box<dyn Error>,
> {
    let mut lua = Lua::builder()
        .profile(compatibility)
        .execution_mode(mode)
        .build()?;
    let vm = lua.vm_mut();
    let arguments = vm.runtime_mut().create_table()?;
    vm.runtime_mut().table_set(
        arguments,
        &Value::Number(1.0),
        Value::from(scale.to_string()),
    )?;
    vm.runtime_mut().set_global("arg", Value::Table(arguments));
    vm.set_interpreter_profiling(step_profiling);
    let started = Instant::now();
    let values = vm.execute(chunk)?;
    let elapsed = started.elapsed();
    let counters = vm.interpreter_counters().clone();
    let jit = vm.jit_statistics();
    Ok((elapsed, counters, values, jit))
}

struct Options {
    source: PathBuf,
    scale: u64,
    mode: ExecutionMode,
}

impl Options {
    fn parse(
        arguments: impl IntoIterator<Item = std::ffi::OsString>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut source = None;
        let mut scale = 1_u64;
        let mut mode = ExecutionMode::Interpreter;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.to_string_lossy().as_ref() {
                "--mode" => {
                    let value = arguments
                        .next()
                        .ok_or("--mode requires interpreter, jit, or tracing")?;
                    mode = ExecutionMode::from_str(&value.to_string_lossy())?;
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: tailcall_probe <source.lua> [scale] [--mode interpreter|jit|tracing]"
                            .into(),
                    );
                }
                value if value.starts_with('-') => {
                    return Err(format!("unknown option: {value}").into());
                }
                value if source.is_none() => source = Some(PathBuf::from(value)),
                value => {
                    scale = value.parse()?;
                }
            }
        }
        Ok(Self {
            source: source.ok_or(
                "usage: tailcall_probe <source.lua> [scale] [--mode interpreter|jit|tracing]",
            )?,
            scale,
            mode,
        })
    }
}
