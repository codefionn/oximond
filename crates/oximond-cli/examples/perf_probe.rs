//! Compile and run a Lua fixture while exposing the VM's tracing-execution counters.
//!
//! ```text
//! cargo run -p oximond-cli --example perf_probe -- benchmarks/lua/control/branching.lua
//! cargo run -p oximond-cli --example perf_probe -- benchmarks/lua/algorithm/mandelbrot.lua --dump-bytecode
//! cargo run -p oximond-cli --example perf_probe -- probe.lua --mode interpreter
//! ```

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use oximond::Lua;
use oximond::bytecode::Chunk;
use oximond::compat::CompatibilityProfile;
use oximond::compat::ExecutionMode;
use oximond::compiler::{CompilerConfig, compile_named};

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse(env::args_os().skip(1))?;
    let source = fs::read(&options.source)?;
    let source_name = options.source.to_string_lossy().into_owned();
    let compatibility = CompatibilityProfile::Lua51;
    let chunk = compile_named(source_name, &source, CompilerConfig::new(compatibility))?;

    if options.dump_bytecode {
        dump_chunk(&chunk, 0);
    }

    let mut lua = Lua::builder()
        .profile(compatibility)
        .execution_mode(options.mode)
        .build()?;
    let vm = lua.vm_mut();

    // Deliberately pass no script arguments. The benchmark fixtures use
    // `tonumber(arg and arg[1]) or 1`, so this selects their scale-one path.
    let started = Instant::now();
    let values = vm.execute(&chunk)?;
    let elapsed = started.elapsed();

    let mut stdout = io::stdout().lock();
    stdout.write_all(&vm.runtime_mut().take_output())?;
    stdout.flush()?;
    let mut stderr = io::stderr().lock();
    stderr.write_all(&vm.runtime_mut().take_error_output())?;
    writeln!(
        stderr,
        "perf probe: mode={} elapsed={elapsed:.3?}",
        options.mode
    )?;
    writeln!(stderr, "perf probe: results={values:?}")?;
    writeln!(stderr, "perf probe: jit={:?}", vm.jit_statistics())?;
    Ok(())
}

struct Options {
    source: PathBuf,
    mode: ExecutionMode,
    dump_bytecode: bool,
}

impl Options {
    fn parse(
        arguments: impl IntoIterator<Item = std::ffi::OsString>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut source = None;
        let mut mode = ExecutionMode::Tracing;
        let mut dump_bytecode = false;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.to_string_lossy().as_ref() {
                "--mode" => {
                    let value = arguments
                        .next()
                        .ok_or("--mode requires interpreter or tracing")?;
                    mode = ExecutionMode::from_str(&value.to_string_lossy())?;
                }
                "--dump-bytecode" => dump_bytecode = true,
                "--help" | "-h" => return Err(
                    "usage: perf_probe <source.lua> [--mode interpreter|tracing] [--dump-bytecode]"
                        .into(),
                ),
                value if value.starts_with('-') => {
                    return Err(format!("unknown option: {value}").into());
                }
                _ if source.is_none() => source = Some(PathBuf::from(argument)),
                _ => return Err("perf_probe accepts one Lua source file".into()),
            }
        }
        Ok(Self {
            source: source.ok_or(
                "usage: perf_probe <source.lua> [--mode interpreter|tracing] [--dump-bytecode]",
            )?,
            mode,
            dump_bytecode,
        })
    }
}

fn dump_chunk(chunk: &Chunk, depth: usize) {
    let indent = "  ".repeat(depth);
    eprintln!(
        "{indent}chunk: registers={} parameters={} vararg={} constants={} instructions={}",
        chunk.register_count,
        chunk.parameter_count,
        chunk.is_vararg,
        chunk.constants.len(),
        chunk.code.len()
    );
    for (pc, instruction) in chunk.code.iter().enumerate() {
        match instruction.opcode() {
            Ok(opcode) => eprintln!(
                "{indent}  {pc:04}: {opcode:?} a={} b={} c={} d={} line={:?}",
                instruction.a(),
                instruction.b(),
                instruction.c(),
                instruction.d(),
                chunk.debug.line_at(pc)
            ),
            Err(_) => eprintln!("{indent}  {pc:04}: {instruction:?}"),
        }
    }
    for (index, prototype) in chunk.prototypes.iter().enumerate() {
        eprintln!("{indent}  prototype {index}:");
        dump_chunk(prototype, depth + 2);
    }
}
