//! Tree compile jobs written to a file and compiled again offline, to measure
//! the compile thread without the program that made its trees.
//!
//! With `OXIMOND_JIT_DUMP=<path>` set, the compile thread appends each tree
//! job it runs (the tree, its tier, its options and the members its caller
//! planned) to `<path>`, one length-prefixed bincode record per job, in the
//! order it ran them. [`read`] gives them back and [`DumpedJob::compile`]
//! runs the same compile, so a replay in order on one thread meets the member
//! cache as the compile thread did. Sides compiled apart are not written.
//! `examples/compile_replay.rs` drives it.

use super::{OptimizationTier, TraceCompileError, TraceCompileOptions};
use crate::compat::TracingOptFlags;
use crate::jit_ir::trace::{TraceIr, TreeMember};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// One member of a planned tree, as [`TreeMember`] holds it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DumpedMember {
    pub ir: TraceIr,
    pub parent: Option<(usize, u32)>,
    pub demoted: u32,
}

/// One tree compile job of the compile thread.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DumpedJob {
    pub ir: TraceIr,
    /// 0 Baseline, 1 Optimized, 2 Peak.
    pub tier: u8,
    pub count_links: bool,
    pub opt_flags: u64,
    pub planned: Option<Vec<DumpedMember>>,
}

impl DumpedJob {
    /// Compiles the job as the compile thread did. `Ok` holds the number of
    /// values its exits store, over all exits of the compiled tree.
    ///
    /// # Errors
    ///
    /// The compile's own error.
    pub fn compile(&self) -> Result<usize, TraceCompileError> {
        crate::vm::install_trace_helpers();
        let tier = match self.tier {
            0 => OptimizationTier::Baseline,
            1 => OptimizationTier::Optimized,
            _ => OptimizationTier::Peak,
        };
        let options = TraceCompileOptions {
            count_links: self.count_links,
            opt_flags: TracingOptFlags::from_bits(self.opt_flags),
        };
        let planned: Option<Vec<TreeMember>> = self.planned.as_ref().map(|members| {
            members
                .iter()
                .map(|member| TreeMember {
                    ir: member.ir.clone(),
                    parent: member.parent,
                    demoted: member.demoted,
                })
                .collect()
        });
        super::compile_trace_with_plan(&self.ir, tier, options, planned.as_deref())
            .map(|tree| tree.exits().iter().map(|exit| exit.slots.len()).sum())
    }
}

/// The file `OXIMOND_JIT_DUMP` names, opened on first use.
fn sink() -> Option<&'static Mutex<File>> {
    static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var_os("OXIMOND_JIT_DUMP")?;
        File::create(path).ok().map(Mutex::new)
    })
    .as_ref()
}

/// Appends a tree job to the dump when `OXIMOND_JIT_DUMP` is set.
pub(crate) fn write(
    ir: &TraceIr,
    tier: OptimizationTier,
    options: TraceCompileOptions,
    planned: Option<&[TreeMember]>,
) {
    let Some(sink) = sink() else {
        return;
    };
    let job = DumpedJob {
        ir: ir.clone(),
        tier: match tier {
            OptimizationTier::Baseline => 0,
            OptimizationTier::Optimized => 1,
            OptimizationTier::Peak => 2,
        },
        count_links: options.count_links,
        opt_flags: options.opt_flags.bits(),
        planned: planned.map(|members| {
            members
                .iter()
                .map(|member| DumpedMember {
                    ir: member.ir.clone(),
                    parent: member.parent,
                    demoted: member.demoted,
                })
                .collect()
        }),
    };
    let Ok(bytes) = bincode::serialize(&job) else {
        return;
    };
    if let Ok(mut file) = sink.lock() {
        let _ = file.write_all(&(bytes.len() as u64).to_le_bytes());
        let _ = file.write_all(&bytes);
    }
}

/// The jobs of a dump, in the order they ran.
///
/// # Errors
///
/// A file that cannot be read or holds a record that does not decode.
pub fn read(path: &Path) -> std::io::Result<Vec<DumpedJob>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut jobs = Vec::new();
    let mut length = [0_u8; 8];
    loop {
        match reader.read_exact(&mut length) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let mut bytes = vec![0_u8; u64::from_le_bytes(length) as usize];
        reader.read_exact(&mut bytes)?;
        jobs.push(
            bincode::deserialize(&bytes)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?,
        );
    }
    Ok(jobs)
}
