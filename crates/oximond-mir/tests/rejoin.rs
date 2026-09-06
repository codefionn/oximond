//! Sides that jump back to the root's outer header (`Join` and `Rejoin`):
//! a swap on the back edge is a cycle of the parallel move, nested sides
//! rejoin too, calls and a loop inside the root work with them, and the
//! native code matches the evaluator on every scenario.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};

extern "C" fn bump(a: i64) -> i64 {
    a.wrapping_mul(3).wrapping_add(1)
}

/// The root joins on `(a, b, fuel)`. One iteration: spend a unit of fuel
/// (leave through exit 0 when it is gone), and when `a + b` is below the
/// limit leave through exit 1, otherwise go to a side that swaps the two and
/// rejoins. The side's parallel move is a two-register cycle. A second,
/// nested side (taken when `a` is odd) rotates three values instead.
fn swap_func(with_call: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let b0 = f.load(Ty::I64, inputs, NONE, 1, 24);
    let c0 = f.load(Ty::I64, inputs, NONE, 1, 40);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let first = f.join(&[a0, b0, c0, fuel0]);
    let (a, b, c, fuel) = (first, first + 1, first + 2, first + 3);
    let one = f.iconst(1);
    let zero = f.iconst(0);
    let fuel1 = f.bin(Op::Sub, fuel, one);
    f.guard(
        Cc::Gt,
        fuel1,
        zero,
        &[(a, 0), (b, 1), (c, 2), (fuel1, CONTROL_SLOT)],
    );
    let fuel_now = fuel1;
    let limit = f.iconst(1000);
    let sum = f.bin(Op::Add, a, b);
    // Taken (continues) when the sum is below the limit.
    f.guard_side(Cc::Lt, sum, limit, 0, &[a, b, c, fuel_now]);
    let c = if with_call {
        f.call(bump as *const () as usize, &[c], Some(Ty::I64))
    } else {
        c
    };
    f.exit(&[(a, 0), (b, 1), (c, 2), (sum, 3), (fuel_now, CONTROL_SLOT)]);
    // Side 0: swap a and b; an odd `a` goes on to side 1.
    let (_, p) = f.begin_side(&[Ty::I64; 4]);
    let (sa, sb, sc, sf) = (p, p + 1, p + 2, p + 3);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let low = f.bin(Op::And, sa, one);
    f.guard_side(Cc::Eq, low, zero, 1, &[sa, sb, sc, sf]);
    let sc2 = f.bin(Op::Add, sc, one);
    f.rejoin(&[sb, sa, sc2, sf]);
    // Side 1 (nested): rotate a, b, c.
    let (_, p) = f.begin_side(&[Ty::I64; 4]);
    let (ra, rb, rc, rf) = (p, p + 1, p + 2, p + 3);
    f.rejoin(&[rb, rc, ra, rf]);
    f
}

fn run(f: &Func, a: i64, b: i64, c: i64, fuel: i64) {
    f.validate().unwrap();
    let kinds = [Input::Int(a), Input::Int(b), Input::Int(c)];
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
        if let Err(e) = check_native(f, &code, &mut sc) {
            panic!(
                "a={a} b={b} c={c} fuel={fuel} avx={}: {e}; {stats:?}",
                t.avx
            );
        }
    }
}

#[test]
fn a_swap_on_the_back_edge_of_a_rejoining_side_matches_the_evaluator() {
    let f = swap_func(false);
    for fuel in [1, 2, 3, 4, 5, 6, 7, 8, 31, 100] {
        for (a, b, c) in [
            (600, 700, 0),
            (700, 600, 5),
            (601, 900, 7),
            (1, 2, 3),
            (-5, 2000, 9),
        ] {
            run(&f, a, b, c, fuel);
        }
    }
}

#[test]
fn nested_sides_rejoin_and_a_call_runs_beside_them() {
    let f = swap_func(true);
    for fuel in [1, 2, 5, 9, 40] {
        for (a, b, c) in [(600, 700, 0), (601, 900, 7), (4, 5, 6), (999, 1, 2)] {
            run(&f, a, b, c, fuel);
        }
    }
}

/// The side's swap really is a cycle the planner breaks, and the stats say a
/// side reached the join.
#[test]
fn the_rejoin_move_is_a_cycle() {
    let f = swap_func(false);
    let (_, stats) = compile_on(targets()[0], &f);
    assert!(stats.cycles_broken >= 1, "{stats:?}");
}

/// A root with an inner loop and a join: the inner loop's own header keeps its
/// back edge, the join is a second label.
#[test]
fn a_join_and_a_loop_in_one_root() {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let b0 = f.load(Ty::I64, inputs, NONE, 1, 24);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let j = f.join(&[a0, b0, fuel0]);
    let (a, b, fuel) = (j, j + 1, j + 2);
    let one = f.iconst(1);
    let zero = f.iconst(0);
    let ten = f.iconst(10);
    let h = f.header(&[a, fuel]);
    let (x, fl) = (h, h + 1);
    let x2 = f.bin(Op::Add, x, b);
    let fl2 = f.bin(Op::Sub, fl, one);
    f.guard(Cc::Gt, fl2, zero, &[(x2, 0), (b, 1), (fl2, CONTROL_SLOT)]);
    f.guard_side(Cc::Lt, x2, ten, 0, &[x2, b, fl2]);
    f.back(&[x2, fl2]);
    // Side: reset x and rejoin with b doubled.
    let (_, p) = f.begin_side(&[Ty::I64; 3]);
    let (sx, sb, sf) = (p, p + 1, p + 2);
    let ten = f.iconst(10);
    let nb = f.bin(Op::Add, sb, sb);
    let sx2 = f.bin(Op::Sub, sx, ten);
    f.rejoin(&[sx2, nb, sf]);
    f.validate().unwrap();
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        for (a, b, fuel) in [(1i64, 1i64, 50i64), (0, 3, 30), (2, 2, 7), (5, 4, 100)] {
            let kinds = [Input::Int(a), Input::Int(b)];
            let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
            check_native(&f, &code, &mut sc).unwrap_or_else(|e| panic!("{a} {b} {fuel}: {e}"));
        }
    }
}

/// Join parameters that the inner loop reads as invariants (so the loop pins
/// them or keeps them in slots under pressure) still arrive where the sides
/// that rejoin put them.
#[test]
fn join_parameters_that_are_invariants_of_the_inner_loop_under_pressure() {
    for count in [2usize, 6, 11, 14] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let init: Vec<u32> = (0..count)
            .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * k as i32 + 8))
            .collect();
        let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
        let mut all = init.clone();
        all.push(fuel0);
        let j = f.join(&all);
        let params: Vec<u32> = (0..count).map(|k| j + k as u32).collect();
        let fuel = j + count as u32;
        let one = f.iconst(1);
        let zero = f.iconst(0);
        let limit = f.iconst(500);
        let h = f.header(&[params[0], fuel]);
        let (x, fl) = (h, h + 1);
        let mut x2 = x;
        for p in &params {
            x2 = f.bin(Op::Add, x2, *p);
        }
        let fl2 = f.bin(Op::Sub, fl, one);
        f.guard(
            Cc::Gt,
            fl2,
            zero,
            &[
                (x2, 0),
                (params[0], 1),
                (params[count - 1], 2),
                (fl2, CONTROL_SLOT),
            ],
        );
        let mut args = params.clone();
        args.push(fl2);
        f.guard_side(Cc::Lt, x2, limit, 0, &args);
        f.back(&[x2, fl2]);
        let mut tys = vec![Ty::I64; count];
        tys.push(Ty::I64);
        let (_, p) = f.begin_side(&tys);
        let sone = f.iconst(1);
        let first = f.bin(Op::Add, p, sone);
        let mut next: Vec<u32> = (0..count).map(|k| p + k as u32).collect();
        next[0] = first;
        // Rotate the rest so the move is a long cycle.
        next.rotate_left(1);
        next.push(p + count as u32);
        f.rejoin(&next);
        f.validate().unwrap();
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            for fuel in [3i64, 8, 40] {
                let kinds: Vec<Input> = (0..count).map(|k| Input::Int(k as i64 * 3 - 2)).collect();
                let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
                check_native(&f, &code, &mut sc)
                    .unwrap_or_else(|e| panic!("{count} params, fuel {fuel}: {e}"));
            }
        }
    }
}

/// A join below the work a preheader did: what the loop reads of it (`k`, a
/// value computed from it, a constant) goes through cells, and the side that
/// rejoins hands over only the new `a` and the fuel. A cell nobody loads
/// costs no store.
#[test]
fn a_join_after_the_preheader_reads_what_its_cells_kept() {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let k = f.load(Ty::I64, inputs, NONE, 1, 24);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let three = f.iconst(3);
    let kk = f.bin(Op::Mul, k, three);
    let seven = f.iconst(7);
    let spare = f.bin(Op::Add, k, seven);
    let (cell_k, cell_kk, cell_spare) = (f.new_cell(), f.new_cell(), f.new_cell());
    f.cell_store(cell_k, k);
    f.cell_store(cell_kk, kk);
    f.cell_store(cell_spare, spare);
    let j = f.join(&[a0, fuel0]);
    let (a, fuel) = (j, j + 1);
    let k = f.cell_load(Ty::I64, cell_k);
    let kk = f.cell_load(Ty::I64, cell_kk);
    let (one, zero, limit) = (f.iconst(1), f.iconst(0), f.iconst(900));
    let h = f.header(&[a, fuel]);
    let (x, fl) = (h, h + 1);
    let step = f.bin(Op::Add, kk, seven);
    let x2 = f.bin(Op::Add, x, step);
    let fl2 = f.bin(Op::Sub, fl, one);
    f.guard(
        Cc::Gt,
        fl2,
        zero,
        &[(x2, 0), (k, 1), (kk, 2), (fl2, CONTROL_SLOT)],
    );
    f.guard_side(Cc::Lt, x2, limit, 0, &[x2, fl2]);
    f.back(&[x2, fl2]);
    // Side: start again from a fifth of where the loop got to.
    let (_, p) = f.begin_side(&[Ty::I64; 2]);
    let five = f.iconst(5);
    let low = f.bin(Op::UDiv, p, five);
    f.rejoin(&[low, p + 1]);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        assert!(stats.dead_skipped >= 1, "{stats:?}");
        for (a, k, fuel) in [(1i64, 2i64, 40i64), (500, 10, 9), (0, 0, 100), (-3, 4, 7)] {
            let kinds = [Input::Int(a), Input::Int(k)];
            let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
            check_native(&f, &code, &mut sc).unwrap_or_else(|e| panic!("{a} {k} {fuel}: {e}"));
        }
    }
}

/// The values a loop reads from cells outnumber the registers: those without
/// one live in their cells, so a link of the side reloads none of them (it
/// used to copy each from its cell to a spill slot after the join).
#[test]
fn a_cell_value_without_a_register_stays_in_its_cell() {
    const KEPT: usize = 24;
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 0);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let cells: Vec<u32> = (0..KEPT).map(|_| f.new_cell()).collect();
    for (k, cell) in cells.iter().enumerate() {
        let v = f.load(Ty::I64, inputs, NONE, 1, 8 * (k as i32 + 1));
        f.cell_store(*cell, v);
    }
    let j = f.join(&[a0, fuel0]);
    let (a, fuel) = (j, j + 1);
    let kept: Vec<u32> = cells.iter().map(|c| f.cell_load(Ty::I64, *c)).collect();
    let (one, zero, limit) = (f.iconst(1), f.iconst(0), f.iconst(100_000));
    let h = f.header(&[a, fuel]);
    let (x, fl) = (h, h + 1);
    // Every kept value is read on every pass, so all of them are live at once.
    let mut sum = x;
    for k in &kept {
        sum = f.bin(Op::Add, sum, *k);
    }
    let fl2 = f.bin(Op::Sub, fl, one);
    f.guard(Cc::Gt, fl2, zero, &[(sum, 0), (fl2, CONTROL_SLOT)]);
    f.guard_side(Cc::Lt, sum, limit, 0, &[sum, fl2]);
    f.back(&[sum, fl2]);
    let (_, p) = f.begin_side(&[Ty::I64; 2]);
    let four = f.iconst(4);
    let low = f.bin(Op::UDiv, p, four);
    f.rejoin(&[low, p + 1]);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        assert!(stats.cell_homes > 0, "{stats:?}");
        for (a, step, fuel) in [(1i64, 1i64, 40i64), (500, 3, 9), (0, 0, 100), (-3, 2, 30)] {
            let mut kinds = vec![Input::Int(a)];
            kinds.extend((0..KEPT).map(|k| Input::Int(step + k as i64)));
            let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
            check_native(&f, &code, &mut sc).unwrap_or_else(|e| panic!("{a} {step} {fuel}: {e}"));
        }
    }
}

/// A load above the only store of its cell, inside a loop: the value it
/// defines is the store of the iteration before, and is read after this
/// iteration's store. With the registers taken, the values that have none
/// must not live in their cells (a later load of each cell keeps the store).
#[test]
fn a_load_above_the_only_store_reads_the_store_of_the_iteration_before() {
    const KEPT: usize = 24;
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let limit = f.load(Ty::I64, inputs, NONE, 1, 8);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let (zero, one) = (f.iconst(0), f.iconst(1));
    let cells: Vec<u32> = (0..KEPT).map(|_| f.new_cell()).collect();
    let h = f.header(&[zero, zero, fuel0]);
    let (i, acc, fuel) = (h, h + 1, h + 2);
    // What the cells held at the end of the iteration before.
    let before: Vec<Val> = cells.iter().map(|c| f.cell_load(Ty::I64, *c)).collect();
    for (k, cell) in cells.iter().enumerate() {
        let k = f.iconst(100 + k as i64);
        let w = f.bin(Op::Mul, i, k);
        f.cell_store(*cell, w);
    }
    let mut now = acc;
    for cell in &cells {
        let z = f.cell_load(Ty::I64, *cell);
        now = f.bin(Op::Add, now, z);
    }
    // The first iteration has nothing before it.
    let open = f.begin_if(Cc::Gt, i, zero);
    let mut with = now;
    for y in &before {
        with = f.bin(Op::Add, with, *y);
    }
    let acc2 = f.end_if(open, &[(with, now)]);
    let i2 = f.bin(Op::Add, i, one);
    let fuel2 = f.bin(Op::Sub, fuel, one);
    f.guard(
        Cc::Gt,
        fuel2,
        zero,
        &[(acc2, 0), (i2, 1), (fuel2, CONTROL_SLOT)],
    );
    f.guard(Cc::Lt, i2, limit, &[(acc2, 0), (i2, 1)]);
    f.back(&[i2, acc2, fuel2]);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);

        for (limit, fuel) in [(1i64, 40i64), (2, 40), (7, 40), (30, 5), (30, 100)] {
            let kinds = [Input::Int(limit)];
            let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
            check_native(&f, &code, &mut sc)
                .unwrap_or_else(|e| panic!("{limit} {fuel}: {e}; {stats:?}"));
        }
    }
}

/// Two joins in the root, one above the other: the first takes `(a, b, fuel)`
/// and is followed by a check, the second takes `(a, fuel)` right before the
/// loop and reads `b` back from a cell. Side 0 rejoins the second join (the
/// loop is entered again without the check), a side nested in it rejoins the
/// first with a new `b`. The native code and the evaluator agree on which
/// join each side reached.
fn two_joins_func(count: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 8);
    let b0 = f.load(Ty::I64, inputs, NONE, 1, 24);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let first = f.join(&[a0, b0, fuel0]);
    let (a, b, fuel) = (first, first + 1, first + 2);
    let (zero, one, limit) = (f.iconst(0), f.iconst(1), f.iconst(500));
    // The check, which only a rejoin of the first join runs again: a
    // negative `b` leaves.
    f.guard(Cc::Ge, b, zero, &[(a, 0), (b, 1), (fuel, CONTROL_SLOT)]);
    let cell = f.new_cell();
    f.cell_store(cell, b);
    let second = f.join(&[a, fuel]);
    let (a2, fuel2) = (second, second + 1);
    let b2 = f.cell_load(Ty::I64, cell);
    let h = f.header(&[a2, fuel2]);
    let (x, fl) = (h, h + 1);
    let x2 = f.bin(Op::Add, x, b2);
    let fl2 = f.bin(Op::Sub, fl, one);
    f.guard(Cc::Gt, fl2, zero, &[(x2, 0), (b2, 1), (fl2, CONTROL_SLOT)]);
    f.guard_side(Cc::Lt, x2, limit, 0, &[x2, fl2]);
    f.back(&[x2, fl2]);
    // Side 0: halve `x`. An odd result goes on to side 1, an even one
    // rejoins the second join.
    let (_, p) = f.begin_side(&[Ty::I64; 2]);
    let (sx, sf) = (p, p + 1);
    let two = f.iconst(2);
    let half = f.bin(Op::UDiv, sx, two);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let low = f.bin(Op::And, half, one);
    f.guard_side(Cc::Eq, low, zero, 1, &[half, sf]);
    f.rejoin_nth(0, 1, &[half, sf]);
    // Side 1: the first join again, with a `b` made from `x` (and, when
    // `count` is set, a negative one for some inputs, so the check leaves).
    let (_, p) = f.begin_side(&[Ty::I64; 2]);
    let (rx, rf) = (p, p + 1);
    let three = f.iconst(3);
    let nb = f.bin(Op::And, rx, three);
    let nb = if count {
        let five = f.iconst(5);
        f.bin(Op::Sub, nb, five)
    } else {
        nb
    };
    f.rejoin_nth(0, 0, &[rx, nb, rf]);
    f
}

#[test]
fn a_root_with_two_joins_reaches_the_one_each_side_names() {
    for count in [false, true] {
        let f = two_joins_func(count);
        f.validate().unwrap();
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            for fuel in [1i64, 2, 3, 5, 9, 20, 64] {
                for (a, b) in [(1i64, 1i64), (0, 3), (7, 2), (100, 0), (300, 9), (-4, 6)] {
                    let kinds = [Input::Int(a), Input::Int(b)];
                    let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
                    check_native(&f, &code, &mut sc)
                        .unwrap_or_else(|e| panic!("count={count} a={a} b={b} fuel={fuel}: {e}"));
                }
            }
        }
    }
}

/// A chain of sides that take what the root keeps in cells, for their exits
/// alone: each parameter lives in the cell every entry hands it from, so an
/// entry moves only what the loop computed (the sum and the fuel), however
/// many kept values the sides name. The parameters of the second side are
/// handed the first side's parameters, which live in the same cells. A
/// parameter that one entry hands a cell's value and another the sum (the
/// last one of side 1, entered from the root and from side 0) keeps a place
/// of its own.
#[test]
fn a_side_parameter_handed_a_cell_from_every_entry_lives_in_the_cell() {
    const KEPT: usize = 20;
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let a0 = f.load(Ty::I64, inputs, NONE, 1, 0);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let cells: Vec<u32> = (0..KEPT).map(|_| f.new_cell()).collect();
    for (k, cell) in cells.iter().enumerate() {
        let v = f.load(Ty::I64, inputs, NONE, 1, 8 * (k as i32 + 1));
        f.cell_store(*cell, v);
    }
    let j = f.join(&[a0, fuel0]);
    let (a, fuel) = (j, j + 1);
    let kept: Vec<Val> = cells.iter().map(|c| f.cell_load(Ty::I64, *c)).collect();
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let h = f.header(&[a, fuel]);
    let (x, fl) = (h, h + 1);
    let sum = f.bin(Op::Add, x, kept[0]);
    let fl2 = f.bin(Op::Sub, fl, one);
    f.guard(Cc::Gt, fl2, zero, &[(sum, 0), (fl2, CONTROL_SLOT)]);
    // Side 0 when the sum reaches 1000, side 1 straight from the root when
    // it is a multiple of 7 (its last parameter is then a kept value).
    let limit = f.iconst(1000);
    let mut to_first = vec![sum, fl2];
    to_first.extend(&kept);
    f.guard_side(Cc::Lt, sum, limit, 0, &to_first);
    let seven = f.iconst(7);
    let rest = f.bin(Op::URem, sum, seven);
    let mut to_second = vec![sum, fl2];
    to_second.extend(&kept);
    to_second.push(kept[1]);
    f.guard_side(Cc::Ne, rest, zero, 1, &to_second);
    f.back(&[sum, fl2]);
    let exit_of = |p: Val, extra: Option<Val>| {
        let mut out = vec![(p, 0), (p + 1, CONTROL_SLOT)];
        out.extend((0..KEPT as u32).map(|k| (p + 2 + k, 1 + k)));
        if let Some(extra) = extra {
            out.push((extra, 1 + KEPT as u32));
        }
        out
    };
    // Side 0: leave on an odd sum, go on to side 1 on a multiple of four
    // (handing the sum as the last parameter), otherwise halve and rejoin.
    let mut tys = vec![Ty::I64; 2 + KEPT];
    let (_, p) = f.begin_side(&tys);
    let (one, zero, three) = (f.iconst(1), f.iconst(0), f.iconst(3));
    let low = f.bin(Op::And, p, one);
    f.guard(Cc::Eq, low, zero, &exit_of(p, None));
    let quarter = f.bin(Op::And, p, three);
    let mut onward: Vec<Val> = (0..2 + KEPT as u32).map(|k| p + k).collect();
    onward.push(p);
    f.guard_side(Cc::Ne, quarter, zero, 1, &onward);
    let two = f.iconst(2);
    let half = f.bin(Op::UDiv, p, two);
    f.rejoin(&[half, p + 1]);
    // Side 1: leave when the last parameter is above 40, otherwise rejoin
    // from a third of the sum plus a kept value it reads itself.
    tys.push(Ty::I64);
    let (_, p) = f.begin_side(&tys);
    let last = p + 2 + KEPT as u32;
    let forty = f.iconst(40);
    f.guard(Cc::Le, last, forty, &exit_of(p, Some(last)));
    let three = f.iconst(3);
    let third = f.bin(Op::UDiv, p, three);
    let next = f.bin(Op::Add, third, p + 2 + 5);
    f.rejoin(&[next, p + 1]);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        // Three entries, each moving the sum and the fuel at most, and the
        // mixed parameter: none of the kept values.
        assert!(stats.trampoline_moves <= 8, "{stats:?}");
        for (a, step, fuel) in [
            (1i64, 1i64, 400i64),
            (990, 3, 90),
            (0, 0, 100),
            (-3, 2, 300),
            (5, 7, 2000),
            (996, 4, 50),
            (2, 14, 500),
            (1, 50, 500),
        ] {
            let mut kinds = vec![Input::Int(a)];
            kinds.extend((0..KEPT).map(|k| Input::Int(step + k as i64)));
            let mut sc = Scenario::new(&kinds, fuel, f.exit_area_len(), vec![0; 4]);
            check_native(&f, &code, &mut sc).unwrap_or_else(|e| panic!("{a} {step} {fuel}: {e}"));
        }
    }
}
