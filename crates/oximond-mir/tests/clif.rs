//! The CLIF translation of every kernel (and some random programs) must agree
//! with the evaluator, so the bench compares one program built two ways.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
#![allow(unsafe_code)]

mod common;

use oximond_mir::ir::Func;
use oximond_mir::kernels::{self, Scenario, same_bits};

#[path = "common/clif.rs"]
mod clif;

type Entry = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) -> i64;

fn run_clif(code: &clif::ClifCode, sc: &mut Scenario) -> i64 {
    // SAFETY: the function was declared as (ptr, ptr, ptr) -> i64.
    let f: Entry = unsafe { std::mem::transmute::<*const u8, Entry>(code.entry) };
    // SAFETY: the buffers outlive the call and the code touches only them.
    unsafe {
        f(
            sc.inputs.as_mut_ptr(),
            sc.control.as_mut_ptr(),
            sc.area.as_mut_ptr(),
        )
    }
}

fn compare(f: &Func, sc: &mut Scenario, what: &str) {
    let (want, regions) = sc.eval(f);
    let code = clif::compile(f, false);
    let got = run_clif(&code, sc);
    assert_eq!(got, want, "{what}: result");
    for (k, r) in sc.regions().into_iter().enumerate() {
        for (w, (&a, &b)) in r.iter().zip(&regions[k]).enumerate() {
            assert!(
                same_bits(a, b),
                "{what}: region {k} word {w}: cranelift {a:#x}, evaluator {b:#x}"
            );
        }
    }
}

#[test]
fn cranelift_agrees_on_every_kernel() {
    for k in kernels::ALL {
        let mut f = Func::new();
        (k.build)(&mut f);
        for (fuel, data) in (k.scenarios)() {
            let mut sc = k.scenario(&f, fuel, data);
            compare(&f, &mut sc, k.name);
        }
    }
}

#[test]
fn cranelift_agrees_on_random_programs() {
    let mut f = Func::new();
    for i in 0..60 {
        let seed = oximond_mir::rng::Rng::nth(0x4d49_5230_5345_4544, i);
        let (mut sc, _) = oximond_mir::randgen::generate(seed, &mut f);
        compare(&f, &mut sc, &format!("seed {seed:#x}"));
    }
}

/// The fused multiply-add and the conditional calls on float compares, where
/// the CPU has the instruction.
#[test]
fn cranelift_agrees_on_random_programs_with_a_fused_multiply_add() {
    if !oximond_mir::x64::X64::detect().fma {
        return;
    }
    let mut f = Func::new();
    for i in 0..60 {
        let seed = oximond_mir::rng::Rng::nth(0x4d49_5230_5345_4544, i);
        let (mut sc, _) = oximond_mir::randgen::generate_for(seed, &mut f, true);
        compare(&f, &mut sc, &format!("seed {seed:#x}"));
    }
}
