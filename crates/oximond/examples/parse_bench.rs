//! Parses Lua files repeatedly to measure lexer and parser throughput.
//!
//! `cargo run --release -p oximond --example parse_bench -- 20 a.lua b.lua`
//!
//! Set `LEAK` to skip dropping each tree. Compare builds by
//! `perf stat -e instructions:u` rather than the printed rate.

use oximond::compat::CompatibilityProfile;
use oximond::syntax::{ParserConfig, parse_bytes_with_config};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args
        .next()
        .and_then(|rounds| rounds.parse().ok())
        .expect("usage: parse_bench ROUNDS FILE...");
    let sources: Vec<Vec<u8>> = args
        .map(|path| std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}")))
        .collect();
    let bytes: usize = sources.iter().map(Vec::len).sum();
    let config = ParserConfig::new(CompatibilityProfile::Lua54);
    let leak = std::env::var_os("LEAK").is_some();
    let mut failures = 0;
    let started = Instant::now();
    for _ in 0..rounds {
        for source in &sources {
            match parse_bytes_with_config(std::hint::black_box(source), config) {
                Ok(chunk) if leak => core::mem::forget(std::hint::black_box(chunk)),
                Ok(chunk) => drop(std::hint::black_box(chunk)),
                Err(_) => failures += 1,
            }
        }
    }
    let elapsed = started.elapsed();
    #[allow(clippy::cast_precision_loss)] // A throughput estimate.
    let megabytes = (bytes * rounds) as f64 / 1e6;
    println!(
        "{rounds} rounds, {bytes} bytes, {failures} failures, {:.1} MB/s",
        megabytes / elapsed.as_secs_f64()
    );
}
