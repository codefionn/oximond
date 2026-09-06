//! Compiles Lua files repeatedly to measure the source-to-bytecode pipeline.
//!
//! `cargo run --release -p oximond --example compile_bench -- MODE ROUNDS FILE...`
//!
//! A single FILE ending in `.txt` is a list of paths, one per line.
//!
//! MODE is one of:
//!
//! - `parse`: the syntax tree only.
//! - `compile`: `compile_named`, tree and bytecode.
//! - `load`: `CompilerLoader::try_load_complete`, the whole-source fast path
//!   `luaL_loadbuffer` takes in the C API.
//! - `capi`: `load` plus what `Vm::load_chunk` does before it installs the
//!   closure (a second validation and a clone of the chunk).
//! - `analyze`: `capi` plus the analyses a first call and a first table
//!   constructor run, forced on every prototype.
//! - `fingerprint`: one line per file with an FNV hash of the binary dump of
//!   the `load` result. ROUNDS is ignored. Diff the output of two builds to
//!   show the bytecode is unchanged.
//!
//! The files compile under the `LuaJIT` 2.1 profile, the dialect the Luanti C
//! API build speaks. Compare builds by `perf stat -e instructions:u`.

use oximond::bytecode::Chunk;
use oximond::compat::CompatibilityProfile;
use oximond::compiler::{CompilerConfig, CompilerLoader, compile_named};
use oximond::string::LuaString;
use oximond::syntax::{ParserConfig, parse_bytes_with_config};

/// Forces the analyses a first call would run, on a chunk and its prototypes.
fn force_analyses(chunk: &Chunk) {
    chunk.compute_live_registers();
    std::hint::black_box(chunk.entry_register_count());
    std::hint::black_box(chunk.constructor_shape(0));
    for prototype in &chunk.prototypes {
        force_analyses(prototype);
    }
}

/// An FNV hash of the binary dump.
fn fingerprint(chunk: &Chunk) -> (u64, usize) {
    let bytes = oximond::bytecode::binary::dump(chunk).expect("dump");
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    });
    (hash, bytes.len())
}

fn main() {
    let usage = "usage: compile_bench MODE ROUNDS FILE...";
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect(usage);
    let rounds: usize = args
        .next()
        .and_then(|rounds| rounds.parse().ok())
        .expect(usage);
    let mut paths: Vec<String> = args.collect();
    if paths.len() == 1
        && std::path::Path::new(&paths[0])
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("txt"))
    {
        let list = std::fs::read_to_string(&paths[0]).expect("file list");
        paths = list.lines().map(str::to_owned).collect();
    }
    let sources: Vec<(LuaString, Vec<u8>)> = paths
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
            (LuaString::from(format!("@{path}")), bytes)
        })
        .collect();
    let profile = CompatibilityProfile::LuaJit21;
    let loader = CompilerLoader::with_profile(profile);
    if mode == "fingerprint" {
        for (name, source) in &sources {
            match loader.try_load_complete(name, source) {
                Some(chunk) => {
                    let (hash, length) = fingerprint(&chunk);
                    println!("{hash:016x} {length} {}", name.to_lossy());
                }
                None => println!("failed {}", name.to_lossy()),
            }
        }
        return;
    }
    let mut failures = 0;
    for _ in 0..rounds {
        for (name, source) in &sources {
            let source = std::hint::black_box(source.as_slice());
            let ok = match mode.as_str() {
                "parse" => parse_bytes_with_config(source, ParserConfig::new(profile)).is_ok(),
                "compile" => {
                    compile_named(name.clone(), source, CompilerConfig::new(profile)).is_ok()
                }
                "load" => loader.try_load_complete(name, source).is_some(),
                "capi" | "analyze" => match loader.try_load_complete(name, source) {
                    Some(chunk) => {
                        let valid = chunk.validate().is_ok();
                        let installed = std::sync::Arc::new(chunk);
                        if mode == "analyze" {
                            force_analyses(&installed);
                        }
                        valid
                    }
                    None => false,
                },
                _ => panic!("unknown mode {mode}; {usage}"),
            };
            failures += usize::from(!ok);
        }
    }
    println!(
        "{rounds} rounds, {} files, {failures} failures",
        sources.len()
    );
}
