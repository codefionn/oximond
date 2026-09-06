//! Compiles the tree jobs of an `OXIMOND_JIT_DUMP` file again, in order, on
//! one thread, to measure the compile thread offline.
//!
//! `cargo run --release -p oximond --example compile_replay -- DUMP [ROUNDS]`
//!
//! Each round compiles every job once; the member cache carries over from one
//! job to the next as it did on the compile thread. Compare builds by
//! `perf stat -e instructions:u`; `OXIMOND_JIT_ICOUNT=1` prints the phases of
//! each compile. The last line counts the compiles, the errors and the exits,
//! which two builds that compile alike print alike.

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: compile_replay DUMP [ROUNDS]");
    let rounds: usize = args
        .next()
        .map_or(1, |rounds| rounds.parse().expect("ROUNDS"));
    let jobs = oximond::jit::compile_dump::read(Path::new(&path)).expect("a readable dump");
    let (mut compiled, mut failed, mut exits) = (0_usize, 0_usize, 0_usize);
    for _ in 0..rounds {
        for job in &jobs {
            match job.compile() {
                Ok(count) => {
                    compiled += 1;
                    exits += count;
                }
                Err(error) => {
                    failed += 1;
                    if std::env::var_os("REPLAY_ERRORS").is_some() {
                        eprintln!("replay: {error}");
                    }
                }
            }
        }
    }
    println!(
        "jobs {} compiled {compiled} failed {failed} exit slots {exits}",
        jobs.len()
    );
}
