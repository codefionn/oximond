//! The call op: native result, exit area and callee-saved registers match
//! the reference evaluator when values are live across a call, whatever the
//! register they sit in, for integer and float arguments and results, with
//! spills across calls, in a loop and next to a division.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
#![allow(unsafe_code)]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};

extern "C" fn mix(a: i64, b: i64) -> i64 {
    a.wrapping_mul(31).wrapping_add(b) ^ (a >> 3)
}

extern "C" fn fscale(x: f64, k: i64) -> f64 {
    x * k as f64 + 0.5
}

extern "C" fn fsum4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a * 1.5 - b + c * d
}

extern "C" fn six(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> i64 {
    a.wrapping_mul(3)
        .wrapping_add(b.wrapping_mul(5))
        .wrapping_sub(c)
        .wrapping_add(d << 2)
        .wrapping_add(e ^ f)
}

extern "C" fn mixed(a: i64, x: f64, b: i64, y: f64, c: i64, z: f64) -> f64 {
    (a as f64) * x + (b as f64) * y - (c as f64) * z
}

extern "C" fn eight_floats(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> f64 {
    a + 2.0 * b + 3.0 * c + 4.0 * d + 5.0 * e + 6.0 * f + 7.0 * g + 8.0 * h
}

// Returns 1 when rsp + 8 (the stack pointer before the call pushed the return
// address) is a multiple of 16, as the ABI wants at a call site.
std::arch::global_asm!(
    ".globl oximond_mir_test_aligned",
    "oximond_mir_test_aligned:",
    "lea rax, [rsp + 8]",
    "and eax, 15",
    "cmp eax, 0",
    "sete al",
    "movzx eax, al",
    "ret",
);

unsafe extern "C" {
    fn oximond_mir_test_aligned(a: i64) -> i64;
}

fn addr1(f: extern "C" fn(i64, i64) -> i64) -> usize {
    f as *const () as usize
}

fn inputs_of(ints: usize, floats: usize) -> Vec<Input> {
    let mut v = Vec::new();
    for k in 0..ints {
        v.push(Input::Int(1000 + 37 * k as i64 * (k as i64 - 3)));
    }
    for k in 0..floats {
        v.push(Input::Float(
            0.25 * (k as f64 + 1.5) - 0.125 * (k * k) as f64,
        ));
    }
    v
}

fn load_inputs(f: &mut Func, inputs: Val, ints: usize, floats: usize) -> (Vec<Val>, Vec<Val>) {
    let is = (0..ints)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * k as i32 + 8))
        .collect();
    let fs = (0..floats)
        .map(|k| f.load(Ty::F64, inputs, NONE, 1, 16 * (ints + k) as i32 + 8))
        .collect();
    (is, fs)
}

fn exit_all(f: &mut Func, vals: &[Val]) {
    let snap: Vec<(Val, u32)> = vals
        .iter()
        .enumerate()
        .map(|(k, v)| (*v, k as u32))
        .collect();
    f.exit(&snap);
}

fn check(f: &Func, kinds: &[Input], what: &str) {
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        let mut sc = Scenario::new(kinds, 10, f.exit_area_len(), vec![0; 8]);
        if let Err(e) = check_native(f, &code, &mut sc) {
            panic!("{what} (avx {}): {e}; {stats:?}", t.avx);
        }
    }
}

/// Integers and floats loaded before a call and read after it, at every
/// amount of pressure: values in caller-saved registers survive, the rest
/// stay where they are.
#[test]
fn values_live_across_a_call_survive() {
    for (ints, floats) in [(2, 1), (5, 3), (9, 6), (13, 13), (14, 14)] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, ints, floats);
        let r = f.call(addr1(mix), &[is[0], is[1]], Some(Ty::I64));
        let g = f.call(fscale as *const () as usize, &[fs[0], r], Some(Ty::F64));
        let mut all = vec![r, g];
        let mut sum = r;
        for v in &is {
            sum = f.bin(Op::Add, sum, *v);
        }
        let mut fsum = g;
        for v in &fs {
            fsum = f.bin(Op::FAdd, fsum, *v);
        }
        all.push(sum);
        all.push(fsum);
        all.extend(&is);
        all.extend(&fs);
        exit_all(&mut f, &all);
        check(
            &f,
            &inputs_of(ints, floats),
            &format!("pressure {ints}+{floats}"),
        );
    }
}

/// Six integer arguments, eight float arguments, mixed classes, float and
/// integer results, a constant argument, and one value passed twice.
#[test]
fn arguments_and_results_of_both_classes_are_placed() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 8, 8);
    let k = f.iconst(77);
    let a = f.call(
        six as *const () as usize,
        &[is[5], is[4], is[3], is[2], is[1], k],
        Some(Ty::I64),
    );
    let b = f.call(
        mixed as *const () as usize,
        &[is[0], fs[0], a, fs[1], is[0], fs[2]],
        Some(Ty::F64),
    );
    let c = f.call(
        eight_floats as *const () as usize,
        &[fs[7], fs[6], fs[5], fs[4], fs[3], fs[2], b, fs[0]],
        Some(Ty::F64),
    );
    let c2 = f.fconst(2.5);
    let d = f.call(
        fsum4 as *const () as usize,
        &[c, c, c2, fs[1]],
        Some(Ty::F64),
    );
    let e = f.call(addr1(mix), &[a, a], Some(Ty::I64));
    let mut all = vec![a, b, c, d, e];
    all.extend(&is);
    all.extend(&fs);
    exit_all(&mut f, &all);
    check(&f, &inputs_of(8, 8), "arguments");
}

/// More float values live across a call than the register file holds
/// (there is no callee-saved float register), so each is stored once at its
/// definition and reloaded after the call, calls in a row.
#[test]
fn spills_across_calls_are_stored_once_and_reloaded() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 10, 14);
    let mut acc = is[0];
    let mut facc = fs[0];
    for step in 0..4 {
        acc = f.call(addr1(mix), &[acc, is[1 + step]], Some(Ty::I64));
        facc = f.call(fscale as *const () as usize, &[facc, acc], Some(Ty::F64));
        // Use the loaded values after each call.
        for v in &fs {
            facc = f.bin(Op::FAdd, facc, *v);
        }
    }
    let mut all = vec![acc, facc];
    all.extend(&is);
    all.extend(&fs);
    exit_all(&mut f, &all);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        assert!(stats.reloads > 0, "reloads {stats:?}");
        let mut sc = Scenario::new(&inputs_of(10, 14), 10, f.exit_area_len(), vec![0; 8]);
        check_native(&f, &code, &mut sc).unwrap();
    }
}

/// A call inside a loop with a loop-carried integer and float, a loop
/// invariant live across the call and a guard that leaves.
#[test]
fn a_call_in_a_loop_keeps_the_carried_and_invariant_values() {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 6, 3);
    let fuel = f.load(Ty::I64, control, NONE, 1, 0);
    let first = f.header(&[fuel, is[0], fs[0]]);
    let (n, acc, facc) = (first, first + 1, first + 2);
    let zero = f.iconst(0);
    f.guard(Cc::Gt, n, zero, &[(n, 0), (acc, 1), (facc, 2)]);
    let acc2 = f.call(addr1(mix), &[acc, is[1]], Some(Ty::I64));
    let acc3 = f.bin(Op::Add, acc2, is[2]);
    let facc2 = f.call(fscale as *const () as usize, &[facc, is[3]], Some(Ty::F64));
    let facc3 = f.bin(Op::FAdd, facc2, fs[1]);
    let facc4 = f.bin(Op::FMul, facc3, fs[2]);
    let one = f.iconst(1);
    let n2 = f.bin(Op::Sub, n, one);
    f.back(&[n2, acc3, facc4]);
    check(&f, &inputs_of(6, 3), "loop");
}

/// A division in the same function keeps its register rules next to calls.
#[test]
fn a_call_and_a_division_share_a_function() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 10, 0);
    let seven = f.iconst(7);
    let q = f.bin(Op::SDiv, is[0], seven);
    let c = f.call(addr1(mix), &[q, is[1]], Some(Ty::I64));
    let r = f.bin(Op::SRem, c, seven);
    let mut all = vec![q, c, r];
    all.extend(&is);
    exit_all(&mut f, &all);
    check(&f, &inputs_of(10, 0), "division");
}

/// The callee sees an aligned stack at every call, whatever the number of
/// callee-saved registers the function pushes.
#[test]
fn the_stack_is_aligned_at_the_call() {
    for ints in [1usize, 3, 6, 9, 12, 14] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let (is, _) = load_inputs(&mut f, inputs, ints, 0);
        let a = f.call(
            oximond_mir_test_aligned as *const () as usize,
            &[is[0]],
            Some(Ty::I64),
        );
        let mut sum = a;
        for v in &is {
            sum = f.bin(Op::Add, sum, *v);
        }
        exit_all(&mut f, &[a, sum]);
        f.validate().unwrap();
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            let mut sc = Scenario::new(&inputs_of(ints, 0), 10, 2, vec![0; 8]);
            check_native(&f, &code, &mut sc).unwrap();
            assert_eq!(sc.area[0], 1, "aligned with {ints} live values");
        }
    }
}

extern "C" fn ten(
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
    i: i64,
    j: i64,
) -> i64 {
    // Each argument has its own weight, so a swapped pair shows.
    a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i + 10 * j
}

extern "C" fn nine_and_ten(
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
    i: i64,
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
    x8: f64,
    x9: f64,
) -> f64 {
    let ints = a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i;
    let floats = x0
        + 2.0 * x1
        + 3.0 * x2
        + 4.0 * x3
        + 5.0 * x4
        + 6.0 * x5
        + 7.0 * x6
        + 8.0 * x7
        + 9.0 * x8
        + 10.0 * x9;
    ints as f64 + floats
}

/// Integer arguments past the sixth go to the stack in order, at every amount
/// of pressure, with values live across the call.
#[test]
fn integer_arguments_past_the_sixth_go_to_the_stack() {
    for live in [10usize, 12, 14] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let (is, _) = load_inputs(&mut f, inputs, live, 0);
        let mut args: Vec<Val> = is[..10].to_vec();
        // A constant and a repeated value among the stack arguments.
        args[7] = f.iconst(77);
        args[9] = is[0];
        let r = f.call(ten as *const () as usize, &args, Some(Ty::I64));
        let mut sum = r;
        for v in &is {
            sum = f.bin(Op::Add, sum, *v);
        }
        let mut all = vec![r, sum];
        all.extend(&is);
        exit_all(&mut f, &all);
        check(
            &f,
            &inputs_of(live, 0),
            &format!("ten arguments, {live} live"),
        );
    }
}

/// More than eight floats and more than six integers at once: both classes
/// spill to the stack, in argument order. The evaluator takes at most eight
/// floats, so the result is compared with the function run by Rust.
#[test]
fn float_arguments_past_the_eighth_go_to_the_stack() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 9, 10);
    let mut args: Vec<Val> = is.clone();
    args.extend(&fs);
    let r = f.call(nine_and_ten as *const () as usize, &args, Some(Ty::F64));
    let mut sum = r;
    for v in &fs {
        sum = f.bin(Op::FAdd, sum, *v);
    }
    exit_all(&mut f, &[r, sum]);
    f.validate().unwrap();
    let kinds = inputs_of(9, 10);
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        let mut sc = Scenario::new(&kinds, 10, 2, vec![0; 8]);
        code.call(&mut sc.inputs, &mut sc.control, &mut sc.area);
        let ints: Vec<i64> = (0..9).map(|k| sc.inputs[2 * k + 1] as i64).collect();
        let fl: Vec<f64> = (0..10)
            .map(|k| f64::from_bits(sc.inputs[2 * (9 + k) + 1]))
            .collect();
        let want = nine_and_ten(
            ints[0], ints[1], ints[2], ints[3], ints[4], ints[5], ints[6], ints[7], ints[8], fl[0],
            fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7], fl[8], fl[9],
        );
        assert_eq!(f64::from_bits(sc.area[0]), want, "avx {}", t.avx);
    }
}

extern "C" fn through_control(a: i64, b: i64) -> i64 {
    a * 1000 + b
}

/// The function pointer is read from the control block when the call is made.
#[test]
fn a_call_through_the_control_block_reaches_its_function() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 8, 0);
    let r = f.call_ctl(8, &[is[0], is[1]], Some(Ty::I64));
    let mut sum = r;
    for v in &is {
        sum = f.bin(Op::Add, sum, *v);
    }
    exit_all(&mut f, &[r, sum]);
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        let mut sc = Scenario::new(&inputs_of(8, 0), 10, 2, vec![0; 8]);
        sc.control[1] = through_control as *const () as u64;
        if let Err(e) = check_native(&f, &code, &mut sc) {
            panic!("avx {}: {e}; {stats:?}", t.avx);
        }
        let a = sc.inputs[1] as i64;
        let b = sc.inputs[3] as i64;
        assert_eq!(sc.area[0] as i64, through_control(a, b));
    }
}

/// Frame cells keep a value across a call and a division, and a float too.
#[test]
fn cells_keep_values_across_calls() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 6, 2);
    let (c0, c1) = (f.new_cell(), f.new_cell());
    f.cell_store(c0, is[2]);
    f.cell_store(c1, fs[1]);
    let seven = f.iconst(7);
    let r = f.call(addr1(mix), &[is[0], seven], Some(Ty::I64));
    let a = f.cell_load(Ty::I64, c0);
    let b = f.cell_load(Ty::F64, c1);
    let s = f.bin(Op::Add, r, a);
    let g = f.bin(Op::FAdd, b, fs[0]);
    exit_all(&mut f, &[r, a, b, s, g]);
    check(&f, &inputs_of(6, 2), "cells");
}

static MARKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static MARKED: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

extern "C" fn mark(_unused: i64, a: i64, x: f64, b: i64) {
    use std::sync::atomic::Ordering::SeqCst;
    MARKS.fetch_add(1, SeqCst);
    MARKED.store(
        a.wrapping_mul(7)
            .wrapping_add(b)
            .wrapping_add((x * 4.0) as i64),
        SeqCst,
    );
}

/// A call made only when a condition holds, out of line: the registers that
/// hold values survive it whether it is made or not, at every pressure, and
/// the callee's effect is seen exactly when it is made.
#[test]
fn a_conditional_call_keeps_every_live_value() {
    for (ints, floats) in [(2usize, 1usize), (6, 4), (10, 9), (13, 13)] {
        for ctl in [false, true] {
            let mut f = Func::new();
            let (inputs, _control) = f.entry();
            let ptr = f.load(Ty::I64, inputs, NONE, 1, 8);
            let n_ints = ints + 1;
            let is: Vec<Val> = (0..n_ints)
                .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * (1 + k as i32) + 8))
                .collect();
            let fs: Vec<Val> = (0..floats)
                .map(|k| {
                    f.load(
                        Ty::F64,
                        inputs,
                        NONE,
                        1,
                        16 * (1 + n_ints as i32 + k as i32) + 8,
                    )
                })
                .collect();
            let target = mark as *const () as usize;
            let (target, ctl_off) = if ctl { (8usize, true) } else { (target, false) };
            f.call_if(
                Cc::Lt,
                (is[0], is[1]),
                target,
                ctl_off,
                &[ptr, is[1], fs[0], is[2.min(ints)]],
            );
            let mut sum = is[0];
            for v in &is {
                sum = f.bin(Op::Add, sum, *v);
            }
            let mut fsum = fs[0];
            for v in &fs {
                fsum = f.bin(Op::FAdd, fsum, *v);
            }
            let mut all = vec![sum, fsum];
            all.extend(&is);
            all.extend(&fs);
            exit_all(&mut f, &all);
            f.validate().unwrap();
            let mut kinds = vec![Input::DataPtr];
            kinds.extend(inputs_of(n_ints, floats));
            for swap in [false, true] {
                for t in targets() {
                    let (code, stats) = compile_on(t, &f);
                    let mut sc = Scenario::new(&kinds, 10, f.exit_area_len(), vec![0; 8]);
                    sc.control[1] = mark as *const () as u64;
                    MARKS.store(0, std::sync::atomic::Ordering::SeqCst);
                    if swap {
                        // The condition fails.
                        sc.inputs[3] = 5000;
                        sc.inputs[2 * 2 + 1] = 3;
                    } else {
                        sc.inputs[3] = 3;
                        sc.inputs[2 * 2 + 1] = 5000;
                    }
                    if let Err(e) = check_native(&f, &code, &mut sc) {
                        panic!(
                            "{ints}+{floats} ctl {ctl} swap {swap} avx {}: {e}; {stats:?}",
                            t.avx
                        );
                    }
                    // The evaluator and the native code each made the call, or
                    // neither did.
                    let calls = MARKS.load(std::sync::atomic::Ordering::SeqCst);
                    assert_eq!(calls, if swap { 0 } else { 2 }, "the call ran when asked");
                }
            }
        }
    }
}

extern "C" fn answer(a: i64, b: i64) -> i64 {
    a.wrapping_mul(1000).wrapping_add(b)
}

/// A conditional call that stores its result in the control block: the code
/// after it reads the answer when the call was made and the old word when it
/// was not, with values live across it.
#[test]
fn a_conditional_call_stores_its_result_where_it_is_asked_to() {
    for (ints, ctl) in [(3usize, false), (12, false), (12, true), (13, true)] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let is: Vec<Val> = (0..ints)
            .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * k as i32 + 8))
            .collect();
        // The function pointer is the word at byte 8 (the control block has
        // two words); the result goes to byte 16, a word the control block of
        // the scenario has.
        let target = if ctl {
            8usize
        } else {
            answer as *const () as usize
        };
        f.call_if_storing(
            Cc::Lt,
            (is[0], is[1]),
            target,
            ctl,
            &[is[0], is[1]],
            Some(16),
        );
        let after = f.load(Ty::I64, control, NONE, 1, 16);
        let mut sum = after;
        for v in &is {
            sum = f.bin(Op::Add, sum, *v);
        }
        let mut all = vec![after, sum];
        all.extend(&is);
        exit_all(&mut f, &all);
        f.validate().unwrap();
        for swap in [false, true] {
            for t in targets() {
                let (code, stats) = compile_on(t, &f);
                let mut sc = Scenario::new(&inputs_of(ints, 0), 10, f.exit_area_len(), vec![0; 8]);
                sc.control = vec![10, answer as *const () as u64, 0x77, 0];
                if swap {
                    sc.inputs[1] = 5000;
                    sc.inputs[3] = 3;
                } else {
                    sc.inputs[1] = 3;
                    sc.inputs[3] = 5000;
                }
                if let Err(e) = check_native(&f, &code, &mut sc) {
                    panic!("{ints} ctl {ctl} swap {swap} avx {}: {e}; {stats:?}", t.avx);
                }
                let want = if swap { 0x77 } else { answer(3, 5000) };
                assert_eq!(sc.area[0] as i64, want, "the stored result");
            }
        }
    }
}
