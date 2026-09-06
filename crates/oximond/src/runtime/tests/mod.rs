// helpers shared by the split test modules
#[allow(unused_imports)]
use super::*;
use crate::bytecode::{Chunk, Instruction};
use crate::runtime::arena::{Arena, ArenaKey};
use crate::runtime::table_storage::array_index_number;
use crate::runtime::table_storage::entries::{
    HASHED_TABLE_ENTRY_THRESHOLD, OtherRef, TableEntries,
};
use crate::runtime::table_storage::key::I64_EXCLUSIVE_UPPER_AS_F64;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

static NAMED_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn named_file_path() -> PathBuf {
    let sequence = NAMED_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oximond-runtime-{}-{sequence}", std::process::id()))
}

fn incremental_roots(value: &Value) -> crate::runtime::GcRoots {
    let mut roots = GcRootBuilder::new();
    roots.value(value);
    roots.finish()
}

fn advance_to_incremental_sweep(runtime: &mut Runtime, root: &Value) {
    runtime.set_gc_step_multiplier(0.01);
    for _ in 0..10_000 {
        assert_eq!(
            runtime.gc_step_with_roots(incremental_roots(root), 0),
            GcStepResult::InProgress,
        );
        if runtime.incremental_gc.as_ref().is_some_and(|cycle| {
            matches!(cycle.phase, crate::runtime::gc::IncrementalGcPhase::Sweep)
        }) {
            return;
        }
    }
    panic!("collector never reached sweep");
}

fn finish_incremental_cycle(runtime: &mut Runtime, root: &Value) {
    for _ in 0..10_000 {
        if runtime.gc_step_with_roots(incremental_roots(root), 0) == GcStepResult::Complete {
            return;
        }
    }
    panic!("collector did not finish after root refresh");
}

#[cfg(target_os = "linux")]
mod clock;
mod closures;
mod core;
mod files;
mod gc;
mod probe;
mod shaped;
mod tables;
mod values;
