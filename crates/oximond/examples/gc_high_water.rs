//! Measures allocation throughput and empty full-GC latency after a high-water mark.
//!
//! Run with `cargo run -p oximond --release --example gc_high_water --
//! [peak_tables] [collections]`.

use std::time::Instant;

use oximond::runtime::{GcRootBuilder, Runtime, Value};

fn argument(index: usize, default: usize) -> usize {
    std::env::args()
        .nth(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn roots_for(value: &Value) -> GcRootBuilder {
    let mut roots = GcRootBuilder::new();
    roots.value(value);
    roots
}

fn main() {
    let peak_tables = argument(1, 500_000);
    let collections = argument(2, 100);
    let mut runtime = Runtime::new();
    runtime.stop_gc();

    let allocation_started = Instant::now();
    for _ in 0..peak_tables {
        runtime.create_table().expect("table handle space");
    }
    let allocation_elapsed = allocation_started.elapsed();

    // Drop the peak and retain one small object, matching a runtime which has
    // returned to a tiny live heap after a temporary allocation burst.
    runtime.collect_full(GcRootBuilder::new().finish());
    let tiny = runtime.create_table().expect("table handle space");
    let tiny_value = Value::Table(tiny);
    runtime.collect_full(roots_for(&tiny_value).finish());

    let collection_started = Instant::now();
    for _ in 0..collections {
        runtime.collect_full(roots_for(&tiny_value).finish());
    }
    let collection_elapsed = collection_started.elapsed();

    println!(
        "peak_tables={peak_tables} allocation_ms={:.3} collections={collections} tiny_heap_full_gc_ms={:.3}",
        allocation_elapsed.as_secs_f64() * 1_000.0,
        collection_elapsed.as_secs_f64() * 1_000.0,
    );
}
