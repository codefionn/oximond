//! A loop inside a side: the root enters it with `Goto`, the side has its own
//! `Join` and `Header`, and a nested side jumps back to that `Join`
//! (`Rejoin` to a region other than the root). What the nested side cannot
//! hand back lives in frame cells. The native code matches the evaluator on
//! every scenario, with and without register pressure. An outer root, a root
//! with a `Join` and no loop, is rejoined by a side nested in its loop side.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};

extern "C" fn bump(a: i64) -> i64 {
    a.wrapping_mul(5).wrapping_add(3)
}

/// The root loads `limit`, `step` and `extra` values and enters side 0 (the
/// loop side) with them. The loop side keeps them in cells and joins on
/// `(x, fuel)`: each iteration adds `step` to `x`, and leaves through the
/// poll exit when the fuel is gone. While `x < limit` the loop goes on. Past
/// it, side 1 (nested) takes over: an even `x` leaves, an odd one goes back
/// to the loop side's join with `x - 50`, which the loop climbs again.
fn loop_side(extra: usize, with_call: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let limit0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let step0 = f.load(Ty::I64, inputs, NONE, 1, 24);
    let x0 = f.load(Ty::I64, inputs, NONE, 1, 40);
    let more: Vec<Val> = (0..extra)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * (3 + k as i32) + 8))
        .collect();
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let mut args = vec![limit0, step0, x0, fuel0];
    args.extend(&more);
    f.goto_side(0, &args);

    // Side 0, the loop side.
    let mut tys = vec![Ty::I64; 4 + extra];
    tys.rotate_left(0);
    let (_, p) = f.begin_side(&tys);
    let (limit, step, x_in, fuel_in) = (p, p + 1, p + 2, p + 3);
    let extras: Vec<Val> = (0..extra as u32).map(|k| p + 4 + k).collect();
    let cells: Vec<u32> = (0..2 + extra).map(|_| f.new_cell()).collect();
    f.cell_store(cells[0], limit);
    f.cell_store(cells[1], step);
    for (k, v) in extras.iter().enumerate() {
        f.cell_store(cells[2 + k], *v);
    }
    let j = f.join(&[x_in, fuel_in]);
    let (xj, fuelj) = (j, j + 1);
    let limit = f.cell_load(Ty::I64, cells[0]);
    let step = f.cell_load(Ty::I64, cells[1]);
    let extras: Vec<Val> = (0..extra)
        .map(|k| f.cell_load(Ty::I64, cells[2 + k]))
        .collect();
    let h = f.header(&[xj, fuelj]);
    let (x, fuel) = (h, h + 1);
    let one = f.iconst(1);
    let zero = f.iconst(0);
    let mut x2 = f.bin(Op::Add, x, step);
    if with_call {
        let c = f.call(bump as *const () as usize, &[x2], Some(Ty::I64));
        let low = f.iconst(0xff);
        let masked = f.bin(Op::And, c, low);
        x2 = f.bin(Op::Add, x2, masked);
    }
    let fuel2 = f.bin(Op::Sub, fuel, one);
    let mut snap = vec![(x2, 0), (limit, 1), (step, 2)];
    for (k, v) in extras.iter().enumerate() {
        snap.push((*v, 3 + k as u32));
    }
    snap.push((fuel2, CONTROL_SLOT));
    f.guard(Cc::Gt, fuel2, zero, &snap);
    f.guard_side(Cc::Lt, x2, limit, 1, &[x2, fuel2]);
    f.back(&[x2, fuel2]);

    // Side 1, nested in the loop side.
    let (_, p) = f.begin_side(&[Ty::I64; 2]);
    let (sx, sf) = (p, p + 1);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let low = f.bin(Op::And, sx, one);
    let area = (4 + extra) as u32;
    f.guard(Cc::Ne, low, zero, &[(sx, area), (sf, CONTROL_SLOT)]);
    let fifty = f.iconst(50);
    let back = f.bin(Op::Sub, sx, fifty);
    f.rejoin_to(1, &[back, sf]);
    f
}

fn run(f: &Func, extra: usize, limit: i64, step: i64, x: i64, fuel: i64) {
    f.validate().unwrap();
    let mut kinds = vec![Input::Int(limit), Input::Int(step), Input::Int(x)];
    for k in 0..extra {
        kinds.push(Input::Int(11 * (k as i64 + 1)));
    }
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
        if let Err(e) = check_native(f, &code, &mut sc) {
            panic!(
                "limit={limit} step={step} x={x} fuel={fuel} extra={extra} avx={}: {e}; {stats:?}",
                t.avx
            );
        }
    }
}

fn scenarios(f: &Func, extra: usize) {
    for fuel in [1, 2, 3, 7, 40, 500] {
        for (limit, step, x) in [
            (100, 7, 0),
            (100, 7, 1),
            (100, 8, 0),
            (100, 9, 4),
            (30, 3, 29),
            (1000, 17, -400),
            (5, 100, 0),
        ] {
            run(f, extra, limit, step, x, fuel);
        }
    }
}

#[test]
fn a_loop_in_a_side_with_a_nested_side_that_rejoins_it_matches_the_evaluator() {
    scenarios(&loop_side(0, false), 0);
}

#[test]
fn the_loop_side_works_under_register_pressure_and_with_a_call() {
    for extra in [3, 9, 14] {
        scenarios(&loop_side(extra, false), extra);
        scenarios(&loop_side(extra, true), extra);
    }
}

/// An outer root: the root has no loop of its own. It loads `n`, `m`, `acc`
/// and `i`, keeps `n` and `m` in cells, joins on `(acc, i, fuel)`, adds `i`
/// to `acc` (leaving when the sum is past a bound), and enters side 0, the
/// inner loop, with `Goto`. The loop adds `j` to `acc` for `j` in `0..m` and
/// spends a unit of fuel per loop-back. Its closing guard enters side 1,
/// nested in it, which steps `i`, leaves past `n` and otherwise spends a unit
/// of fuel and jumps back to the root's `Join` (`Rejoin` to region 0 from a
/// side of a loop side): the next outer iteration runs the root's code below
/// the `Join` again.
fn outer_root(with_call: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let n0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let m0 = f.load(Ty::I64, inputs, NONE, 1, 24);
    let acc0 = f.load(Ty::I64, inputs, NONE, 1, 40);
    let i0 = f.load(Ty::I64, inputs, NONE, 1, 56);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let (cell_n, cell_m) = (f.new_cell(), f.new_cell());
    f.cell_store(cell_n, n0);
    f.cell_store(cell_m, m0);
    let j = f.join(&[acc0, i0, fuel0]);
    let (acc, i, fuel) = (j, j + 1, j + 2);
    let n = f.cell_load(Ty::I64, cell_n);
    let m = f.cell_load(Ty::I64, cell_m);
    let mut acc2 = f.bin(Op::Add, acc, i);
    if with_call {
        let c = f.call(bump as *const () as usize, &[i], Some(Ty::I64));
        let low = f.iconst(0xf);
        let masked = f.bin(Op::And, c, low);
        acc2 = f.bin(Op::Add, acc2, masked);
    }
    let cap = f.iconst(1 << 20);
    f.guard(
        Cc::Lt,
        acc2,
        cap,
        &[(acc2, 0), (i, 1), (n, 2), (fuel, CONTROL_SLOT)],
    );
    f.goto_side(0, &[acc2, i, n, m, fuel]);

    // Side 0, the inner loop.
    let (_, p) = f.begin_side(&[Ty::I64; 5]);
    let (a_in, si, sn, sm, sfuel) = (p, p + 1, p + 2, p + 3, p + 4);
    let zero = f.iconst(0);
    let h = f.header(&[a_in, zero, sfuel]);
    let (a, jv, lfuel) = (h, h + 1, h + 2);
    let one = f.iconst(1);
    let zero = f.iconst(0);
    let a2 = f.bin(Op::Add, a, jv);
    let j2 = f.bin(Op::Add, jv, one);
    let fuel2 = f.bin(Op::Sub, lfuel, one);
    f.guard(
        Cc::Gt,
        fuel2,
        zero,
        &[(a2, 0), (si, 1), (j2, 3), (fuel2, CONTROL_SLOT)],
    );
    f.guard_side(Cc::Lt, j2, sm, 1, &[a2, si, sn, fuel2]);
    f.back(&[a2, j2, fuel2]);

    // Side 1, nested in the loop: the rest of the outer iteration.
    let (_, p) = f.begin_side(&[Ty::I64; 4]);
    let (oa, oi, on, ofuel) = (p, p + 1, p + 2, p + 3);
    let one = f.iconst(1);
    let zero = f.iconst(0);
    let i2 = f.bin(Op::Add, oi, one);
    f.guard(Cc::Le, i2, on, &[(oa, 0), (i2, 1), (ofuel, CONTROL_SLOT)]);
    let fuel3 = f.bin(Op::Sub, ofuel, one);
    f.guard(
        Cc::Gt,
        fuel3,
        zero,
        &[(oa, 0), (i2, 1), (fuel3, CONTROL_SLOT)],
    );
    f.rejoin(&[oa, i2, fuel3]);
    f
}

#[test]
fn a_side_of_a_loop_side_rejoins_a_root_without_a_loop_like_the_evaluator() {
    for with_call in [false, true] {
        let f = outer_root(with_call);
        f.validate().unwrap();
        for fuel in [1, 2, 3, 7, 40, 500, 100_000] {
            for (n, m, acc, i) in [
                (10, 5, 0, 1),
                (3, 1, 0, 1),
                (40, 9, -100, 3),
                (5, 30, 7, 5),
                (1000, 4, 0, 1),
                (100, 3, (1 << 20) - 10, 1),
            ] {
                let kinds = [Input::Int(n), Input::Int(m), Input::Int(acc), Input::Int(i)];
                for t in targets() {
                    let (code, stats) = compile_on(t, &f);
                    let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
                    if let Err(e) = check_native(&f, &code, &mut sc) {
                        panic!(
                            "n={n} m={m} acc={acc} i={i} fuel={fuel} call={with_call} avx={}: {e}; {stats:?}",
                            t.avx
                        );
                    }
                }
            }
        }
    }
}
