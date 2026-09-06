//! Seeded random MIR programs, native against the evaluator.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::compile_with;
use oximond_mir::ir::Ty;
use oximond_mir::ir::{Func, Op};
use oximond_mir::kernels::check_native;
use oximond_mir::ra::{Compiler, Stats};
use oximond_mir::randgen::generate_with;
use oximond_mir::rng::Rng;
use oximond_mir::x64::X64;

/// The fixed seed list: output `i` of the SplitMix64 stream for this seed.
const LIST_SEED: u64 = 0x4d49_5230_5345_4544;

fn seed(i: u64) -> u64 {
    Rng::nth(LIST_SEED, i)
}

fn count() -> u64 {
    std::env::var("OXIMOND_MIR_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500)
}

#[test]
fn random_programs_match_the_evaluator_sse2() {
    random_programs(X64::sse2());
}

#[test]
fn random_programs_match_the_evaluator_avx() {
    if X64::detect().avx {
        random_programs(X64::detect());
    }
}

/// AVX without the fused multiply-add: the programs have no `FNma`.
#[test]
fn random_programs_match_the_evaluator_avx_without_fma() {
    if X64::detect().avx {
        random_programs(X64 {
            avx: true,
            fma: false,
            bmi2: false,
        });
    }
}

fn random_programs(target: X64) {
    let mut f = Func::new();
    let mut c = Compiler::new(target);
    let mut total = Stats::default();
    let mut exits = std::collections::BTreeMap::new();
    let mut failures = Vec::new();
    let mut side_runs = 0u32;
    let mut calls = 0usize;
    let (mut fnmas, mut float_call_ifs, mut int_call_ifs) = (0usize, 0usize, 0usize);
    let (mut joins, mut rejoins, mut second_rejoins) = (0usize, 0usize, 0usize);
    let mut register_shifts = 0usize;
    for i in 0..count() {
        let s = seed(i);
        let (mut sc, _) = generate_with(s, &mut f, target.fma, target.bmi2);
        if let Err(e) = f.validate() {
            panic!("seed {s:#x} (#{i}) built an invalid program: {e}");
        }
        if std::env::var_os("OXIMOND_MIR_VERBOSE").is_some() {
            eprintln!("seed #{i} {s:#x}");
        }
        calls += f.insts.iter().filter(|i| i.op == Op::Call).count();
        fnmas += f.insts.iter().filter(|i| i.op == Op::FNma).count();
        float_call_ifs += f
            .insts
            .iter()
            .filter(|i| i.op == Op::CallIf && i.ty == Ty::F64)
            .count();
        int_call_ifs += f
            .insts
            .iter()
            .filter(|i| i.op == Op::CallIf && i.ty == Ty::I64)
            .count();
        register_shifts += f
            .insts
            .iter()
            .filter(|i| {
                matches!(i.op, Op::Shl | Op::Shr | Op::Sar)
                    && f.insts
                        .iter()
                        .all(|k| k.dst != i.b || !matches!(k.op, Op::Const))
            })
            .count();
        joins += f.insts.iter().filter(|i| i.op == Op::Join).count();
        rejoins += f.insts.iter().filter(|i| i.op == Op::Rejoin).count();
        second_rejoins += f
            .insts
            .iter()
            .filter(|i| i.op == Op::Rejoin && i.e != oximond_mir::ir::NONE)
            .count();
        let (code, st) = compile_with(&mut c, &f);
        match check_native(&f, &code, &mut sc) {
            Ok(r) => {
                *exits.entry(r).or_insert(0u32) += 1;
                // Did the run end inside a side? A side's own exit follows
                // its guard, so look the snapshot up among the sides' `Exit`s.
                let snap = (-1 - r) as u32;
                for k in 0..f.num_sides() {
                    let (s0, e0) = f.region(1 + k);
                    if f.insts[s0..e0]
                        .iter()
                        .any(|i| i.op == Op::Exit && i.c == snap)
                    {
                        side_runs += 1;
                    }
                }
            }
            Err(e) => failures.push(format!("seed {s:#x} (#{i}): {e}")),
        }
        total.evictions += st.evictions;
        total.reloads += st.reloads;
        total.def_spill_stores += st.def_spill_stores;
        total.remats += st.remats;
        total.weak_slots += st.weak_slots;
        total.cycles_broken += st.cycles_broken;
        total.trampolines += st.trampolines;
        total.side_direct += st.side_direct;
        total.header_moves += st.header_moves;
        total.backedge_moves += st.backedge_moves;
        total.dead_skipped += st.dead_skipped;
        total.ifs += st.ifs;
        total.if_stubs += st.if_stubs;
        total.if_moves += st.if_moves;
        total.if_rebound += st.if_rebound;
        total.if_entry_moves += st.if_entry_moves;
        total.cold_stubs += st.cold_stubs;
        total.cold_ops += st.cold_ops;
        total.cold_hot += st.cold_hot;
        total.cell_homes += st.cell_homes;
    }
    assert!(
        failures.is_empty(),
        "{} of {} seeds differ:\n{}",
        failures.len(),
        count(),
        failures[..failures.len().min(10)].join("\n")
    );
    eprintln!(
        "coverage over {} seeds: {total:?}\nexits: {exits:?}\nruns that ended in a side: {side_runs}",
        count()
    );
    assert!(side_runs > 20, "sides must actually run: {side_runs}");
    assert!(calls > 200, "programs call functions: {calls}");
    assert!(
        total.cell_homes > 20,
        "values of a join stay in their cells: {}",
        total.cell_homes
    );
    assert!(
        joins > 100 && rejoins > 50,
        "programs have a join under what the loads made, and sides that rejoin: {joins} {rejoins}"
    );
    assert!(
        second_rejoins > 20,
        "some sides rejoin the second join of a region: {second_rejoins}"
    );
    assert!(
        float_call_ifs > 100 && int_call_ifs > 30,
        "programs call out of line on float and integer compares: {float_call_ifs} {int_call_ifs}"
    );
    assert_eq!(
        fnmas > 100,
        target.fma,
        "programs have a fused multiply-add exactly where the target has one: {fnmas}"
    );
    assert_eq!(
        register_shifts > 50,
        target.bmi2,
        "programs shift by a register exactly where the target can: {register_shifts}"
    );
    assert!(
        total.evictions > 0 && total.weak_slots > 0 && total.remats > 0,
        "{total:?}"
    );
    assert!(total.trampolines + total.side_direct > 0, "{total:?}");
    assert!(
        total.ifs > 100 && total.if_stubs > 20 && total.if_moves > 20,
        "skipped blocks are generated, and some need moves on the skip path: {total:?}"
    );
    assert!(
        total.if_rebound > 10 && total.if_entry_moves > 10,
        "some blocks take registers that the skip path keeps: {total:?}"
    );
    assert!(
        total.cold_stubs > 50 && total.cold_ops > 100,
        "some exits compute cold values in their stub: {total:?}"
    );
    assert!(
        exits.len() > 5,
        "programs leave through many different exits: {exits:?}"
    );
}

#[test]
fn the_callee_saved_check_catches_a_clobber() {
    // mov rbx, 1; mov rax, 0; ret
    let code = oximond_mir::exec::ExecMem::new(&[&[
        0x48, 0xC7, 0xC3, 1, 0, 0, 0, 0x48, 0xC7, 0xC0, 0, 0, 0, 0, 0xC3,
    ]]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        code.call(&mut [0; 2], &mut [0; 2], &mut [0; 2])
    }));
    assert!(r.is_err(), "a function that clobbers rbx must be reported");
}
