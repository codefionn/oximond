//! Samples explicit GC-step latency against a retained graph and a large table.
//!
//! This is a diagnostic, not a promise of a hard wall-clock bound.  Allocation,
//! root construction, allocator deallocation, and host resource cleanup such as
//! file flushing are outside the structural work bound.

use std::time::{Duration, Instant};

use oximond::runtime::{GcRootBuilder, GcStepResult, Runtime, Value};

const LIVE_NODES: u32 = 20_000;
const TABLE_ENTRIES: u32 = 100_000;
const DEAD_TABLE_ENTRIES: u32 = 1_000_000;

fn main() {
    let dead_table = std::env::args().any(|argument| argument == "--dead-table");
    let step_kb = std::env::args()
        .find_map(|argument| {
            argument
                .strip_prefix("--step-kb=")
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(1);
    let mut runtime = Runtime::new();
    let root = runtime.create_table().expect("root table");
    let wide = runtime.create_table().expect("wide table");
    runtime
        .table_set(root, &Value::from("wide"), Value::Table(wide))
        .expect("root link");

    let mut tail = root;
    for index in 0..LIVE_NODES {
        let node = runtime.create_table().expect("node");
        runtime
            .table_set(tail, &Value::from("next"), Value::Table(node))
            .expect("node link");
        runtime
            .table_set(node, &Value::from("id"), Value::from(f64::from(index)))
            .expect("node payload");
        tail = node;
    }
    for index in 1..=TABLE_ENTRIES {
        runtime
            .table_set(
                wide,
                &Value::from(f64::from(index)),
                Value::from(f64::from(index)),
            )
            .expect("wide entry");
    }

    let mut roots = GcRootBuilder::new();
    roots.value(&Value::Table(root));
    runtime.collect_full(roots.finish());
    runtime.stop_gc();

    if dead_table {
        // Do not retain this table. A correct incremental sweep must dismantle
        // its array storage over slices rather than dropping a million Values
        // when it removes one arena slot.
        let dead = runtime.create_table().expect("dead table");
        for index in 1..=DEAD_TABLE_ENTRIES {
            runtime
                .table_set(
                    dead,
                    &Value::from(f64::from(index)),
                    Value::from(f64::from(index)),
                )
                .expect("dead entry");
        }
    } else {
        // Feed enough allocation debt to force a new explicit cycle while
        // keeping the retained graph alive.
        for _ in 0..LIVE_NODES {
            runtime.create_table().expect("garbage");
        }
    }

    let mut step = 0_usize;
    let mut maximum = Duration::ZERO;
    let mut checksum = 0_u128;
    let mut samples = Vec::new();
    loop {
        let started = Instant::now();
        let mut step_roots = GcRootBuilder::new();
        step_roots.value(&Value::Table(root));
        let result = runtime.gc_step_with_roots(step_roots.finish(), step_kb);
        let elapsed = started.elapsed();
        maximum = maximum.max(elapsed);
        samples.push(elapsed);
        checksum = checksum.wrapping_add(elapsed.as_nanos());
        step += 1;
        if step == 1 || step.is_multiple_of(1_000) || matches!(result, GcStepResult::Complete) {
            println!(
                "step={step} requested_kb={step_kb} latency_us={} complete={}",
                elapsed.as_micros(),
                matches!(result, GcStepResult::Complete)
            );
        }
        if matches!(result, GcStepResult::Complete) {
            break;
        }
    }
    samples.sort_unstable();
    let percentile = |percent: usize| samples[(samples.len() - 1) * percent / 100];
    println!(
        "cycles_complete=1 steps={step} dead_table={dead_table} p50_us={} p95_us={} p99_us={} max_latency_us={} checksum={checksum}",
        percentile(50).as_micros(),
        percentile(95).as_micros(),
        percentile(99).as_micros(),
        maximum.as_micros()
    );
}
