//! Hand-written kernels: native result, exit index and exit area must match
//! the reference evaluator, and the allocator features must show up.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile, compile_on, targets};
use oximond_mir::exit_status;
use oximond_mir::ir::Func;
use oximond_mir::kernels::{self, Kernel, check_native};

fn run_kernel(k: &Kernel) -> Vec<i64> {
    let mut f = Func::new();
    (k.build)(&mut f);
    f.validate().unwrap();
    let mut results = Vec::new();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        eprintln!(
            "{} [{}]: {stats:?}",
            k.name,
            if t.avx { "avx" } else { "sse2" }
        );
        results.clear();
        for (fuel, data) in (k.scenarios)() {
            let mut sc = k.scenario(&f, fuel, data);
            match check_native(&f, &code, &mut sc) {
                Ok(r) => results.push(r),
                Err(e) => panic!("{} (fuel {fuel}, avx {}): {e}", k.name, t.avx),
            }
        }
    }
    results
}

#[test]
fn every_kernel_matches_the_evaluator() {
    for k in kernels::ALL {
        run_kernel(k);
    }
}

#[test]
fn exit_indices() {
    // nbody: the loop guard is exit 0; the pair guards are 1..=10.
    let r = run_kernel(&kernels::NBODY);
    assert_eq!(r[0], exit_status(0));
    assert_eq!(
        r[1],
        exit_status(1),
        "coincident bodies fail the first d2 guard"
    );
    // int_side: fuel exit is 0, the guard that goes to the side is 1, and the
    // side's own exit comes after the root's.
    let r = run_kernel(&kernels::INT_SIDE);
    assert_eq!(r[0], exit_status(0));
    assert_eq!(r[1], exit_status(2));
    // guard_restore: first scenario runs out of fuel (exit 1), second fails
    // the data check at once (exit 0).
    let r = run_kernel(&kernels::GUARD_RESTORE);
    assert_eq!(r, vec![exit_status(1), exit_status(0)]);
}

#[test]
fn exit_area_holds_spilled_remat_and_weak_values() {
    let k = &kernels::GUARD_RESTORE;
    let mut f = Func::new();
    (k.build)(&mut f);
    let (code, stats) = compile(&f);
    assert!(
        stats.slots > 0 && stats.reloads + stats.def_spill_stores > 0,
        "{stats:?}"
    );
    assert!(stats.weak_uses > 0, "{stats:?}");
    let (fuel, data) = (k.scenarios)().remove(1);
    let mut sc = k.scenario(&f, fuel, data.clone());
    let r = check_native(&f, &code, &mut sc).unwrap();
    assert_eq!(r, exit_status(0));
    // slots: 0 = v3, 1 = v17, 2 = 77, 3 = 2.5, 4 = v5 + v6, 5 = 1.5 * 2.5, 6 = acc, 7 = cnt
    assert_eq!(sc.area[0], data[3]);
    assert_eq!(sc.area[1], data[17]);
    assert_eq!(sc.area[2], 77);
    assert_eq!(f64::from_bits(sc.area[3]), 2.5);
    assert_eq!(sc.area[4], data[5] + data[6]);
    assert_eq!(f64::from_bits(sc.area[5]), 1.5 * 2.5);
    assert_eq!(sc.area[6], 0);
    assert_eq!(sc.area[7], 5);
}

// ---- one test per allocator feature -----------------------------------------

fn stats_of(k: &Kernel) -> oximond_mir::ra::Stats {
    let mut f = Func::new();
    (k.build)(&mut f);
    compile(&f).1
}

#[test]
fn weak_snapshot_uses_never_evict_and_take_a_register_only_with_room() {
    // `weak` and `wf` in guard_restore are used by the snapshot only. One
    // finds a register with room to spare, the other gets a slot written at
    // its definition (the file is full around it): neither evicts a value.
    let s = stats_of(&kernels::GUARD_RESTORE);
    assert!(s.weak_slots >= 1 && s.weak_in_regs >= 1, "{s:?}");
    assert!(s.weak_slots + s.weak_in_regs >= 2, "{s:?}");
    // The nbody snapshots name loop values that are bound anyway: no slot.
    let s = stats_of(&kernels::NBODY);
    assert_eq!(s.weak_slots, 0, "{s:?}");
    assert_eq!(s.slots, 0, "nbody fits in registers: {s:?}");
}

#[test]
fn loop_parameters_are_hinted_so_the_back_edge_moves_nothing() {
    let s = stats_of(&kernels::NBODY);
    assert!(s.hint_hits > 0 && s.hints >= s.hint_hits, "{s:?}");
    assert_eq!(s.backedge_moves, 0, "{s:?}");
    assert_eq!(s.header_moves, 0, "{s:?}");
    let s = stats_of(&kernels::TINY);
    assert_eq!(s.backedge_moves, 0, "{s:?}");
}

#[test]
fn constants_are_rematerialized_not_spilled() {
    let s = stats_of(&kernels::NBODY);
    assert!(
        s.remats + s.pool_operands > 20,
        "dt and zero are rematerialized at each use: {s:?}"
    );
    assert!(
        s.pool_operands > 0,
        "float constants fold into memory operands: {s:?}"
    );
    assert_eq!(s.slots, 0, "constants never take a slot: {s:?}");
    assert!(
        s.remats > 0,
        "dt as the dividend of fdiv needs a register: {s:?}"
    );
}

#[test]
fn spill_is_stored_once_at_the_definition_and_reloaded_often() {
    let s = stats_of(&kernels::PRESSURE);
    assert!(
        s.evictions > 0,
        "the pairwise products exceed the registers: {s:?}"
    );
    assert!(s.def_spill_stores > 0, "{s:?}");
    assert!(
        s.def_spill_stores <= s.slots + 1,
        "one store per slot at most: {s:?}"
    );
    assert!(s.reloads >= s.evictions, "{s:?}");
}

#[test]
fn back_edge_swap_is_a_cycle_and_is_broken() {
    let s = stats_of(&kernels::SWAP);
    assert!(s.cycles_broken >= 1, "{s:?}");
    // And the result is right for both an odd and an even iteration count.
    run_kernel(&kernels::SWAP);
}

#[test]
fn side_entry_registers_hint_the_parents_values() {
    let s = stats_of(&kernels::INT_SIDE);
    assert_eq!(
        s.trampolines, 0,
        "values already sit where the side wants them: {s:?}"
    );
    assert_eq!(s.side_direct, 1, "{s:?}");
}
