//! The backend's entry point: arenas, the executable region, counters and the
//! log lines. The lowering itself is in `lower.rs`.

use super::super::compile::{ExitPlan, LoweredMember, TraceCompileOptions};
use super::lower::{Lowerer, Reject, Scratch};
use crate::jit::OptimizationTier;
use oximond_mir::exec::{CodeBlock, CodeRegion};
use oximond_mir::ir::Func;
use oximond_mir::ra::Compiler;
use oximond_mir::x64::X64;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Code the backend made: a piece of the process's executable region, which
/// goes back to it when the code is dropped.
pub(in super::super) struct Block {
    block: CodeBlock,
}

impl Block {
    pub(in super::super) fn address(&self) -> usize {
        self.block.address()
    }
}

/// A tree the backend compiled.
pub(in super::super) struct Compiled {
    pub(in super::super) block: Block,
    /// Bytes of machine code and constants.
    pub(in super::super) code_size: usize,
    /// MIR instructions the lowering made.
    pub(in super::super) insts: usize,
    /// The root's loop runs in a version without the bounds checks its range
    /// plan proves.
    pub(in super::super) versioned: bool,
    /// Float operations the exit stubs compute (see `oximond_mir::ir::COLD`),
    /// counted once for each stub that does.
    pub(in super::super) cold_ops: u32,
    /// Where the entry starts in the block: 0, or for a function trace the
    /// entry function's offset past the activation.
    pub(in super::super) entry: usize,
}

/// What the backend has accepted and refused so far in this process.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Counters {
    pub accepted_trees: u64,
    /// Trace IR instructions (preheader and body, every member) of the
    /// accepted trees.
    pub accepted_ir: u64,
    pub fallback_trees: u64,
    pub fallback_ir: u64,
    /// Fallbacks and the IR instructions they held, by reason.
    pub reasons: BTreeMap<&'static str, (u64, u64)>,
}

static ACCEPTED_TREES: AtomicU64 = AtomicU64::new(0);
static ACCEPTED_IR: AtomicU64 = AtomicU64::new(0);
static FALLBACK_TREES: AtomicU64 = AtomicU64::new(0);
static FALLBACK_IR: AtomicU64 = AtomicU64::new(0);
static REASONS: Mutex<BTreeMap<&'static str, (u64, u64)>> = Mutex::new(BTreeMap::new());

#[must_use]
pub fn counters() -> Counters {
    Counters {
        accepted_trees: ACCEPTED_TREES.load(Relaxed),
        accepted_ir: ACCEPTED_IR.load(Relaxed),
        fallback_trees: FALLBACK_TREES.load(Relaxed),
        fallback_ir: FALLBACK_IR.load(Relaxed),
        reasons: REASONS.lock().map(|r| r.clone()).unwrap_or_default(),
    }
}

/// Everything one compile thread reuses from tree to tree.
struct Arena {
    func: Func,
    target: X64,
    compiler: Compiler<X64>,
    scratch: Scratch,
    /// A function trace's entry function, and the activation's machine code
    /// while the entry is compiled.
    wrapper: Func,
    image: Vec<u8>,
}

/// The target the code is made for: what the CPU has, or in a test the one it
/// asked for (see [`with_target`]).
fn target() -> X64 {
    #[cfg(test)]
    if let Some(target) = TARGET.with(std::cell::Cell::get) {
        return target;
    }
    X64::detect()
}

#[cfg(test)]
thread_local! {
    static TARGET: std::cell::Cell<Option<X64>> = const { std::cell::Cell::new(None) };
}

/// Runs `body` with this thread's compiles made for `target`, which may have
/// fewer features than the CPU.
#[cfg(test)]
pub(super) fn with_target<T>(target: X64, body: impl FnOnce() -> T) -> T {
    let before = TARGET.with(|cell| cell.replace(Some(target)));
    let result = body();
    TARGET.with(|cell| cell.set(before));
    result
}

thread_local! {
    static ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
}

fn ir_size(members: &[LoweredMember<'_>]) -> u64 {
    members
        .iter()
        .map(|member| (member.ir.preheader.len() + member.ir.body.len()) as u64)
        .sum()
}

/// Compiles the tree with the MIR backend. `None` means Cranelift must do it;
/// the reason has been logged and counted.
pub(in super::super) fn lower(
    members: &[LoweredMember<'_>],
    plan: &ExitPlan,
    tier: OptimizationTier,
    forced_guard: Option<(usize, usize)>,
    options: TraceCompileOptions,
) -> Option<Compiled> {
    let ir = ir_size(members);
    let result = compile(members, plan, tier, forced_guard, options);
    match result {
        Ok(compiled) => {
            #[cfg(test)]
            if members.iter().any(|member| member.ir.makes_vm_calls()) {
                super::tests::VM_CALL_TREES.with(|trees| trees.set(trees.get() + 1));
            }
            ACCEPTED_TREES.fetch_add(1, Relaxed);
            ACCEPTED_IR.fetch_add(ir, Relaxed);
            if crate::jit::debug_env::trace() {
                eprintln!(
                    "jit-trace: mir backend compiled tree: {ir} IR instructions, {} MIR instructions, {} bytes{}{}",
                    compiled.insts,
                    compiled.code_size,
                    if compiled.versioned {
                        ", range-proven loop version"
                    } else {
                        ""
                    },
                    if compiled.cold_ops > 0 {
                        format!(", {} operations in exit stubs", compiled.cold_ops)
                    } else {
                        String::new()
                    }
                );
            }
            Some(compiled)
        }
        Err(reject) => {
            #[cfg(test)]
            super::tests::LAST_FALLBACK.with(|last| last.set(Some(reject.what)));
            FALLBACK_TREES.fetch_add(1, Relaxed);
            FALLBACK_IR.fetch_add(ir, Relaxed);
            if let Ok(mut reasons) = REASONS.lock() {
                let entry = reasons.entry(reject.what).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += ir;
            }
            if crate::jit::debug_env::trace() {
                if reject.at == usize::MAX {
                    eprintln!(
                        "jit-trace: mir backend fallback to cranelift ({ir} IR instructions): {}",
                        reject.what
                    );
                } else {
                    eprintln!(
                        "jit-trace: mir backend fallback to cranelift ({ir} IR instructions): instruction {}: {}",
                        reject.at, reject.what
                    );
                }
            }
            None
        }
    }
}

fn compile(
    members: &[LoweredMember<'_>],
    plan: &ExitPlan,
    tier: OptimizationTier,
    forced_guard: Option<(usize, usize)>,
    options: TraceCompileOptions,
) -> Result<Compiled, Reject> {
    if tier == OptimizationTier::Peak {
        return Err(Reject::new("peak tier compiles with cranelift"));
    }
    if forced_guard.is_some() {
        return Err(Reject::new(
            "a forced guard is a test of the cranelift lowering",
        ));
    }
    if !std::arch::is_x86_feature_detected!("sse4.1") {
        return Err(Reject::new("the cpu has no sse4.1"));
    }
    let Some(region) = CodeRegion::global() else {
        return Err(Reject::new("no executable code region"));
    };
    let target = target();
    // A panic in the allocator (it asserts on states it does not handle)
    // must not reject the tree: Cranelift takes it. The arena may be half
    // updated then, so the next compile starts a new one.
    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ARENA.with(|cell| {
            let mut slot = cell.borrow_mut();
            let arena = slot.get_or_insert_with(|| Arena {
                func: Func::new(),
                target,
                compiler: Compiler::new(target),
                scratch: Scratch::default(),
                wrapper: Func::new(),
                image: Vec::new(),
            });
            if (arena.target.avx, arena.target.fma) != (target.avx, target.fma) {
                // A test compiled for another target.
                arena.target = target;
                arena.compiler = Compiler::new(target);
            }
            build(arena, region, members, plan, options, tier)
        })
    }));
    match compiled {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            if crate::jit::debug_env::trace() {
                eprintln!("jit-trace: mir backend panicked: {message}");
            }
            ARENA.with(|cell| {
                if let Ok(mut slot) = cell.try_borrow_mut() {
                    *slot = None;
                }
            });
            Err(Reject::new("the MIR backend panicked"))
        }
    }
}

fn build(
    arena: &mut Arena,
    region: &'static CodeRegion,
    members: &[LoweredMember<'_>],
    plan: &ExitPlan,
    options: TraceCompileOptions,
    tier: OptimizationTier,
) -> Result<Compiled, Reject> {
    let Arena {
        func,
        target,
        compiler,
        scratch,
        wrapper,
        image,
    } = arena;
    #[cfg(test)]
    let allocations_before = super::tests::thread_allocations();
    let versioned = Lowerer::run(
        members,
        plan,
        options,
        tier,
        (target.fma, target.bmi2),
        func,
        scratch,
    )?;
    #[cfg(test)]
    let lowering_allocations = super::tests::thread_allocations() - allocations_before;
    let function = members[0].ir.function.is_some();
    if function {
        // The activation takes every argument in a register (`Op::Entry`).
        let entry = func.insts[0];
        let args: Vec<_> = (entry.a..entry.a + entry.b)
            .map(|val| func.val_ty[val as usize])
            .collect();
        if !compiler.entry_fits(&args) {
            return Err(Reject::new(
                "function trace with more arguments than argument registers",
            ));
        }
    }
    if crate::jit::debug_env::disasm() {
        eprintln!("jit-disasm: mir\n{}", func.dump());
    }
    if cfg!(debug_assertions)
        && let Err(error) = func.validate()
    {
        eprintln!("jit-trace: mir backend produced an invalid function: {error}");
        return Err(Reject::new("the lowering made an invalid MIR function"));
    }
    #[cfg(test)]
    let allocations_before = super::tests::thread_allocations();
    let stats = compiler.compile(func);
    #[cfg(test)]
    super::tests::LAST_STATS.with(|last| last.set(Some(stats)));
    let mut len = compiler.code_len();
    let mut entry = 0;
    if function {
        // The entry function follows the activation, which it calls at a
        // negative offset from its own first byte.
        entry = len.next_multiple_of(16);
        image.clear();
        image.resize(entry, 0xCC);
        compiler.write_code(&mut image[..len]);
        super::lower::function::wrapper(
            members[0].ir,
            i64::try_from(entry).expect("code fits i64"),
            wrapper,
        )?;
        if crate::jit::debug_env::disasm() {
            eprintln!("jit-disasm: mir entry\n{}", wrapper.dump());
        }
        if cfg!(debug_assertions)
            && let Err(error) = wrapper.validate()
        {
            eprintln!("jit-trace: mir backend produced an invalid entry: {error}");
            return Err(Reject::new("the lowering made an invalid MIR function"));
        }
        compiler.compile(wrapper);
        len = entry + compiler.code_len();
    }
    let block = region
        .carve(len, |bytes| {
            if function {
                bytes[..entry].copy_from_slice(image);
            }
            compiler.write_code(&mut bytes[entry..]);
        })
        .ok_or(Reject::new("the executable code region is full"))?;
    if crate::jit::debug_env::disasm() {
        // The bytes as hex, for `objdump -D -b binary -mi386:x86-64 -M intel`
        // (`benchmarks/mir_disasm.py` feeds them).
        let mut bytes = vec![0_u8; len];
        bytes[..entry].copy_from_slice(image);
        compiler.write_code(&mut bytes[entry..]);
        let mut hex = String::new();
        for byte in &bytes {
            let _ = write!(hex, "{byte:02x}");
        }
        eprintln!("jit-disasm: mir machine code {len} bytes\n{hex}");
    }
    #[cfg(test)]
    super::tests::LAST_COMPILE_ALLOCATIONS.with(|last| {
        last.set(lowering_allocations + super::tests::thread_allocations() - allocations_before);
    });
    Ok(Compiled {
        block: Block { block },
        code_size: len,
        insts: func.insts.len(),
        versioned,
        cold_ops: stats.cold_ops,
        entry,
    })
}
