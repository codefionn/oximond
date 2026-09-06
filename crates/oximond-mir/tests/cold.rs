//! Cold values (`COLD`): float operations only snapshots read are computed in
//! the exit stubs, from operands the snapshot reads, and not where they stand.
//! The native code matches the evaluator, which ignores the mark, on every
//! scenario, with both encodings.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};
use oximond_mir::ra::Stats;

fn check(f: &Func, kinds: &[Input], what: &str, tweaks: &[&dyn Fn(&mut Scenario)]) -> Stats {
    f.validate().unwrap();
    let mut last = Stats::default();
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        last = stats;
        for (n, tweak) in tweaks.iter().enumerate() {
            let mut sc = Scenario::new(kinds, 10, f.exit_area_len(), vec![0; 8]);
            tweak(&mut sc);
            if let Err(e) = check_native(f, &code, &mut sc) {
                panic!("{what}, scenario {n} (avx {}): {e}; {stats:?}", t.avx);
            }
        }
    }
    last
}

fn cold<R>(f: &mut Func, make: impl FnOnce(&mut Func) -> R) -> R {
    f.cold = true;
    let r = make(f);
    f.cold = false;
    r
}

/// Input 0 is an integer count, the rest floats.
fn kinds(floats: usize) -> Vec<Input> {
    let mut v = vec![Input::Int(3)];
    for k in 0..floats {
        v.push(Input::Float(
            0.75 * (k as f64 + 1.0) - 0.125 * (k * k) as f64,
        ));
    }
    v
}

fn float_input(f: &mut Func, inputs: Val, k: usize) -> Val {
    f.load(Ty::F64, inputs, NONE, 1, 16 * (1 + k) as i32 + 8)
}

fn count_input(f: &mut Func, inputs: Val) -> Val {
    f.load(Ty::I64, inputs, NONE, 1, 8)
}

/// A loop whose guard leaves through a snapshot of values only it reads.
fn loop_with_cold_chain(unary: bool) -> Func {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let n = count_input(&mut f, inputs);
    let k = float_input(&mut f, inputs, 0);
    let seed = float_input(&mut f, inputs, 1);
    let first = f.header(&[n, seed]);
    let (counter, acc) = (first, first + 1);
    let step = f.bin(Op::FMul, acc, k);
    let next = f.bin(Op::FAdd, step, k);
    let one = f.fconst(1.5);
    let (a, b, c, d) = cold(&mut f, |f| {
        let a = f.bin(Op::FAdd, step, next);
        let b = f.bin(Op::FMul, a, one);
        let c = if unary {
            f.funary(Op::FNeg, b)
        } else {
            f.bin(Op::FSub, b, step)
        };
        let q = f.bin(Op::FDiv, c, a);
        let d = f.funary(Op::FFloor, q);
        (a, b, c, d)
    });
    let zero = f.iconst(0);
    f.guard(
        Cc::Gt,
        counter,
        zero,
        &[(counter, 0), (next, 1), (a, 2), (b, 3), (c, 4), (d, 5)],
    );
    let one_i = f.iconst(1);
    let left = f.bin(Op::Sub, counter, one_i);
    f.back(&[left, next]);
    f
}

#[test]
fn a_loop_does_not_compute_what_only_its_exit_reads() {
    for unary in [false, true] {
        let f = loop_with_cold_chain(unary);
        let st = check(
            &f,
            &kinds(2),
            "cold chain",
            &[
                &|_| {},
                &|sc| sc.inputs[1] = 1,
                &|sc| sc.inputs[1] = 0,
                &|sc| sc.inputs[1] = 40,
            ],
        );
        assert_eq!((st.cold_stubs, st.cold_ops), (1, 5), "{st:?}");
        assert!(st.dead_skipped >= 5, "the chain is not in the loop: {st:?}");
        assert_eq!(st.cold_hot, 0, "{st:?}");
    }
}

/// A loop that counts in a scaled copy: the index is a cold shift of the
/// copy, and a float made of the index is cold too. With `read` the loop
/// adds the index up, so it is computed where it stands.
fn loop_with_cold_shift(read: bool) -> Func {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let n = count_input(&mut f, inputs);
    let scaled = f.load(Ty::I64, inputs, NONE, 1, 24);
    let first = f.header(&[n, scaled, n]);
    let (counter, copy, sum) = (first, first + 1, first + 2);
    let step = f.iconst(48);
    let (index, float, wide) = cold(&mut f, |f| {
        let index = f.sar_cold(copy, 4);
        let float = f.i2f(index);
        let wide = f.sar_cold(index, 63);
        (index, float, wide)
    });
    let stepped = f.bin(Op::Add, copy, step);
    let total = if read {
        f.bin(Op::Add, sum, index)
    } else {
        f.bin(Op::Add, sum, copy)
    };
    let zero = f.iconst(0);
    f.guard(
        Cc::Gt,
        counter,
        zero,
        &[
            (counter, 0),
            (index, 1),
            (float, 2),
            (wide, 3),
            (total, 4),
            (copy, 5),
        ],
    );
    let one = f.iconst(1);
    let left = f.bin(Op::Sub, counter, one);
    f.back(&[left, stepped, total]);
    f
}

#[test]
fn a_shift_only_an_exit_reads_is_made_by_the_exit() {
    for read in [false, true] {
        let f = loop_with_cold_shift(read);
        let st = check(
            &f,
            &[Input::Int(3), Input::Int(-1000)],
            "cold shift",
            &[
                &|_| {},
                &|sc| sc.inputs[1] = 0,
                &|sc| sc.inputs[1] = 40,
                &|sc| sc.inputs[3] = 16 * 77 + 5,
                &|sc| sc.inputs[3] = i64::MIN as u64,
                &|sc| sc.inputs[3] = (i64::MAX - 5000) as u64,
            ],
        );
        // The guard stands after the sum, so its stub makes the index even
        // where the loop computes it for the sum.
        assert_eq!((st.cold_stubs, st.cold_ops), (1, 3), "{st:?}");
        if !read {
            assert!(
                st.dead_skipped >= 3,
                "the shifts are not in the loop: {st:?}"
            );
        }
        assert_eq!(st.cold_hot, 0, "{st:?}");
    }
}

/// Values the loop reads too are computed where they stand, and the stub
/// reads them from where they are.
#[test]
fn a_cold_value_something_else_reads_is_computed_in_place() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let n = count_input(&mut f, inputs);
    let k = float_input(&mut f, inputs, 0);
    let seed = float_input(&mut f, inputs, 1);
    let first = f.header(&[n, seed]);
    let (counter, acc) = (first, first + 1);
    let (shared, only) = cold(&mut f, |f| {
        let shared = f.bin(Op::FMul, acc, k);
        let only = f.bin(Op::FSub, shared, k);
        (shared, only)
    });
    // The back edge reads `shared`.
    let next = f.bin(Op::FAdd, shared, k);
    let zero = f.iconst(0);
    f.guard(
        Cc::Gt,
        counter,
        zero,
        &[(counter, 0), (shared, 1), (only, 2), (next, 3)],
    );
    let one = f.iconst(1);
    let left = f.bin(Op::Sub, counter, one);
    f.back(&[left, shared]);
    let st = check(&f, &kinds(2), "shared", &[&|_| {}, &|sc| sc.inputs[1] = 0]);
    assert_eq!(st.cold_ops, 1, "only `only` is left to the stub: {st:?}");
}

/// A guard that has a side passes a cold value as a parameter: it is computed
/// in place there, and the other exits still compute it in their stub.
#[test]
fn a_cold_value_a_side_takes_is_computed_in_place() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let n = count_input(&mut f, inputs);
    let k = float_input(&mut f, inputs, 0);
    let seed = float_input(&mut f, inputs, 1);
    let first = f.header(&[n, seed]);
    let (counter, acc) = (first, first + 1);
    let next = f.bin(Op::FMul, acc, k);
    let (a, b) = cold(&mut f, |f| {
        let a = f.bin(Op::FAdd, next, k);
        let b = f.bin(Op::FMul, a, a);
        (a, b)
    });
    let zero = f.iconst(0);
    let limit = f.iconst(2);
    f.guard_side(Cc::Gt, counter, limit, 0, &[counter, a, b]);
    f.guard(Cc::Gt, counter, zero, &[(counter, 0), (a, 1), (b, 2)]);
    let one = f.iconst(1);
    let left = f.bin(Op::Sub, counter, one);
    f.back(&[left, next]);
    let (_, p) = f.begin_side(&[Ty::I64, Ty::F64, Ty::F64]);
    let sum = f.bin(Op::FAdd, p + 1, p + 2);
    f.exit(&[(p, 0), (sum, 1), (p + 1, 2)]);
    check(
        &f,
        &kinds(2),
        "side",
        &[&|_| {}, &|sc| sc.inputs[1] = 2, &|sc| sc.inputs[1] = 8],
    );
}

/// A snapshot that holds every float register leaves none for the stub's
/// temporaries: the values are computed where they stand then.
#[test]
fn a_stub_with_no_registers_to_spare_reads_the_values_where_they_stand() {
    let floats = 15;
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let n = count_input(&mut f, inputs);
    let fs: Vec<Val> = (0..floats)
        .map(|k| float_input(&mut f, inputs, k))
        .collect();
    let (c0, c1) = cold(&mut f, |f| {
        (f.bin(Op::FAdd, fs[0], fs[1]), f.bin(Op::FMul, fs[2], fs[3]))
    });
    let zero = f.iconst(0);
    let mut snap: Vec<(Val, u32)> = vec![(c0, 0), (c1, 1)];
    for (k, v) in fs.iter().enumerate() {
        snap.push((*v, 2 + k as u32));
    }
    f.guard(Cc::Gt, n, zero, &snap);
    // Every float is read again after the guard, so each holds a register.
    let mut total = fs[0];
    for v in &fs[1..] {
        total = f.bin(Op::FAdd, total, *v);
    }
    f.exit(&[(total, 0), (c0, 1)]);
    let st = check(
        &f,
        &kinds(floats),
        "pressure",
        &[&|_| {}, &|sc| sc.inputs[1] = 0],
    );
    assert!(st.cold_hot > 0, "{st:?}");
}

/// A constant operand and an operand that is a cold value itself.
#[test]
fn cold_operands_may_be_constants_or_cold() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let x = float_input(&mut f, inputs, 0);
    let y = float_input(&mut f, inputs, 1);
    let (a, b, c, d) = cold(&mut f, |f| {
        let two = f.fconst(2.0);
        let a = f.bin(Op::FMul, x, two);
        let b = f.bin(Op::FSub, two, a);
        let zero = f.fconst(0.0);
        let c = f.bin(Op::FDiv, y, zero);
        let d = f.funary(Op::FAbs, b);
        (a, b, c, d)
    });
    let one = f.iconst(1);
    let n = count_input(&mut f, inputs);
    f.guard(Cc::Gt, n, one, &[(a, 0), (b, 1), (c, 2), (d, 3), (x, 4)]);
    f.exit(&[(a, 0), (b, 1), (d, 2)]);
    let st = check(
        &f,
        &kinds(2),
        "constants",
        &[&|_| {}, &|sc| sc.inputs[1] = 0, &|sc| sc.inputs[1] = 9],
    );
    assert!(st.cold_stubs >= 2 && st.cold_ops >= 6, "{st:?}");
}

/// A long chain in which each value is read twice by the next (as a modulo
/// reads its dividend) is followed once, not once for each path, and a chain
/// longer than a stub takes is computed where it stands.
#[test]
fn a_long_chain_of_shared_cold_values_compiles_at_once_and_runs_right() {
    for len in [10, 40, 200] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let n = count_input(&mut f, inputs);
        let x = float_input(&mut f, inputs, 0);
        let k = float_input(&mut f, inputs, 1);
        let last = cold(&mut f, |f| {
            let mut v = x;
            for _ in 0..len {
                let product = f.bin(Op::FMul, v, k);
                v = f.bin(Op::FSub, v, product);
            }
            v
        });
        let zero = f.iconst(0);
        f.guard(Cc::Gt, n, zero, &[(last, 0), (x, 1)]);
        f.exit(&[(last, 0), (x, 1)]);
        let started = std::time::Instant::now();
        let st = check(
            &f,
            &kinds(2),
            "long chain",
            &[&|_| {}, &|sc| sc.inputs[1] = 0],
        );
        assert!(started.elapsed().as_secs() < 5, "len {len}");
        if len == 200 {
            // The exit that ends the region is met first and computes the
            // chain where it stands, so the guard before it finds it held.
            assert_eq!((st.cold_hot, st.cold_ops), (1, 0), "{st:?}");
        }
    }
}

/// A cold chain defined before the loop and read by a guard inside it: the
/// loop holds the value like any invariant (computed once, before the loop),
/// whether a stub could have computed it or not.
#[test]
fn a_cold_chain_before_the_header_is_an_invariant_of_the_loop() {
    for len in [3, 70] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let n = count_input(&mut f, inputs);
        let x = float_input(&mut f, inputs, 0);
        let k = float_input(&mut f, inputs, 1);
        let last = cold(&mut f, |f| {
            let mut v = x;
            for _ in 0..len {
                let product = f.bin(Op::FMul, v, k);
                v = f.bin(Op::FSub, v, product);
            }
            v
        });
        let first = f.header(&[n, x]);
        let (counter, acc) = (first, first + 1);
        let zero = f.iconst(0);
        f.guard(Cc::Gt, counter, zero, &[(counter, 0), (last, 1), (acc, 2)]);
        let one = f.iconst(1);
        let left = f.bin(Op::Sub, counter, one);
        f.back(&[left, acc]);
        check(
            &f,
            &kinds(2),
            "chain before the header",
            &[&|_| {}, &|sc| sc.inputs[1] = 0, &|sc| sc.inputs[1] = 40],
        );
    }
}

/// A cold conversion of an integer: the loop counts in an integer and only
/// the exit needs the float, so the stub converts the integer it finds in a
/// register, in a spill slot (`spilled` integers crowd the registers) or as a
/// constant, and the chain goes on in floats.
#[test]
fn a_cold_conversion_reads_its_integer_in_the_stub() {
    for spilled in [0usize, 14] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let n = count_input(&mut f, inputs);
        let k = float_input(&mut f, inputs, 0);
        let sixteen = f.iconst(16);
        let start = f.bin(Op::Mul, n, sixteen);
        let fixed = f.iconst(48);
        let crowd: Vec<Val> = (0..spilled)
            .map(|at| {
                let c = f.iconst(3 + at as i64);
                f.bin(Op::Add, n, c)
            })
            .collect();
        let first = f.header(&[n, start, k]);
        let (counter, scaled, acc) = (first, first + 1, first + 2);
        let sixteenth = f.fconst(0.0625);
        let (float, next, constant) = cold(&mut f, |f| {
            let converted = f.i2f(scaled);
            let float = f.bin(Op::FMul, converted, sixteenth);
            let next = f.bin(Op::FAdd, float, k);
            (float, next, f.i2f(fixed))
        });
        let sum = f.bin(Op::FAdd, acc, k);
        let zero = f.iconst(0);
        let mut snap: Vec<(Val, u32)> =
            vec![(counter, 0), (float, 1), (next, 2), (constant, 3), (sum, 4)];
        for (at, v) in crowd.iter().enumerate() {
            snap.push((*v, 5 + at as u32));
        }
        f.guard(Cc::Gt, counter, zero, &snap);
        let one = f.iconst(1);
        let left = f.bin(Op::Sub, counter, one);
        let step = f.iconst(16);
        let stepped = f.bin(Op::Add, scaled, step);
        f.back(&[left, stepped, sum]);
        let st = check(
            &f,
            &kinds(1),
            "cold conversion",
            &[
                &|_| {},
                &|sc| sc.inputs[1] = 0,
                &|sc| sc.inputs[1] = 1,
                &|sc| sc.inputs[1] = 33,
                &|sc| sc.inputs[1] = 7,
            ],
        );
        assert_eq!((st.cold_stubs, st.cold_ops), (1, 4), "{st:?}");
        assert_eq!(st.cold_hot, 0, "{st:?}");
    }
}
