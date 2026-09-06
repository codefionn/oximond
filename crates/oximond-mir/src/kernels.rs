//! Hand-written kernels and the buffers they run on. The same `Func` feeds the
//! evaluator, the x86-64 compiler and (in tests and the bench) a CLIF
//! translator, so every comparison is between one program in three forms.
//!
//! Entry convention, as in the current ABI: `inputs` is an array of
//! `TraceInputSlot { tag, bits }` (16 bytes, so input `i` is at word `2i + 1`),
//! `control[0]` is the fuel counter, `exit_area` receives the snapshot.

use crate::ir::{Cc, Func, NONE, Op, Ty, Val};

/// Buffers for one run. Regions are `Vec<u64>` so their addresses are stable.
#[derive(Clone)]
pub struct Scenario {
    pub inputs: Vec<u64>,
    pub control: Vec<u64>,
    pub area: Vec<u64>,
    pub data: Vec<u64>,
}

/// One input slot.
#[derive(Clone, Copy)]
pub enum Input {
    /// The address of `data`, filled in by [`Scenario::fix_pointers`].
    DataPtr,
    Int(i64),
    Float(f64),
}

impl Scenario {
    #[must_use]
    pub fn new(inputs: &[Input], fuel: i64, area_len: usize, data: Vec<u64>) -> Scenario {
        let mut words = Vec::new();
        for i in inputs {
            words.push(0);
            words.push(match *i {
                Input::DataPtr => 0,
                Input::Int(v) => v as u64,
                Input::Float(v) => v.to_bits(),
            });
        }
        let mut s = Scenario {
            inputs: words,
            control: vec![fuel as u64, 0],
            area: vec![0; area_len.max(1)],
            data,
        };
        s.fix_pointers(inputs);
        s
    }

    /// Rewrites `DataPtr` inputs with the current address of `data`. Call it
    /// again after cloning.
    pub fn fix_pointers(&mut self, kinds: &[Input]) {
        for (k, i) in kinds.iter().enumerate() {
            if matches!(i, Input::DataPtr) {
                self.inputs[2 * k + 1] = self.data.as_ptr() as u64;
            }
        }
    }

    #[must_use]
    pub fn regions(&self) -> [&Vec<u64>; 4] {
        [&self.inputs, &self.control, &self.area, &self.data]
    }

    /// Runs the evaluator on copies placed at this scenario's addresses.
    /// Returns its result and its final copy of the four regions.
    pub fn eval(&self, f: &Func) -> (i64, [Vec<u64>; 4]) {
        let mut mem = crate::eval::Mem::default();
        for r in self.regions() {
            mem.add_words(r.as_ptr() as u64, r);
        }
        let args = [
            self.inputs.as_ptr() as u64,
            self.control.as_ptr() as u64,
            self.area.as_ptr() as u64,
        ];
        let res = crate::eval::run(f, &mut mem, args, 50_000_000);
        (
            res,
            [mem.words(0), mem.words(1), mem.words(2), mem.words(3)],
        )
    }
}

/// Compares two f64 bit patterns, treating any two NaNs as equal.
#[must_use]
pub fn same_bits(a: u64, b: u64) -> bool {
    a == b || (f64::from_bits(a).is_nan() && f64::from_bits(b).is_nan())
}

/// Runs `code` natively on `sc` and the evaluator on a copy, and compares the
/// result, exit index and every region.
///
/// # Errors
///
/// Describes the first difference.
#[cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
pub fn check_native(
    f: &Func,
    code: &crate::exec::ExecMem,
    sc: &mut Scenario,
) -> Result<i64, String> {
    let (want, regions) = sc.eval(f);
    let got = code.call(&mut sc.inputs, &mut sc.control, &mut sc.area);
    if got != want {
        return Err(format!("result: native {got}, evaluator {want}"));
    }
    let names = ["inputs", "control", "exit area", "data"];
    for (k, r) in sc.regions().into_iter().enumerate() {
        for (w, (&a, &b)) in r.iter().zip(&regions[k]).enumerate() {
            if !same_bits(a, b) {
                return Err(format!(
                    "{} word {w}: native {a:#x}, evaluator {b:#x}",
                    names[k]
                ));
            }
        }
    }
    Ok(got)
}

pub struct Kernel {
    pub name: &'static str,
    pub build: fn(&mut Func),
    pub inputs: &'static [Input],
    /// Scenarios: `(fuel, data)` pairs.
    pub scenarios: fn() -> Vec<(i64, Vec<u64>)>,
}

impl Kernel {
    #[must_use]
    pub fn scenario(&self, f: &Func, fuel: i64, data: Vec<u64>) -> Scenario {
        Scenario::new(self.inputs, fuel, f.exit_area_len(), data)
    }
}

pub const ALL: &[Kernel] = &[NBODY, INT_SIDE, SWAP, PRESSURE, GUARD_RESTORE, TINY, SKIP];

fn fwords(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---- nbody-shaped f64 loop with guards ----------------------------------

const NB: usize = 5;

pub const NBODY: Kernel = Kernel {
    name: "nbody",
    build: build_nbody,
    inputs: &[Input::DataPtr],
    scenarios: nbody_scenarios,
};

fn nbody_scenarios() -> Vec<(i64, Vec<u64>)> {
    let mut d = Vec::new();
    for b in 0..NB {
        let b = b as f64;
        d.extend_from_slice(&fwords(&[
            1.0 + b * 1.7,
            0.5 - b * 0.9,
            b * 0.31 - 0.4,
            0.01 * b,
            -0.02 + 0.005 * b,
            0.003 * b,
            0.8 + 0.2 * b,
        ]));
    }
    // A second scenario where two bodies coincide, so the d2 > 0 guard exits.
    let mut same = d.clone();
    for k in 0..7 {
        same[7 + k] = same[k];
    }
    vec![(30, d), (30, same)]
}

/// The advance step of nbody over five bodies stored as 7 doubles each
/// (x y z vx vy vz mass), with the loop-carried values in registers: the
/// remaining step count and a float accumulator. Guards: step count above 0,
/// and `d2 > 0` for every pair.
pub fn build_nbody(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let bodies = f.load(Ty::I64, inp, NONE, 1, 8);
    let fuel = f.load(Ty::I64, ctl, NONE, 1, 0);
    let acc0 = f.fconst(0.0);
    let first = f.header(&[fuel, acc0]);
    let (cnt, acc) = (first, first + 1);
    let dt = f.fconst(0.01);
    let zero = f.fconst(0.0);
    let izero = f.iconst(0);
    f.guard(Cc::Gt, cnt, izero, &[(cnt, 0), (acc, 1)]);
    let mut acc_cur = acc;
    let off = |i: usize, k: usize| ((i * 7 + k) * 8) as i32;
    for i in 0..NB {
        let xi = f.load(Ty::F64, bodies, NONE, 1, off(i, 0));
        let yi = f.load(Ty::F64, bodies, NONE, 1, off(i, 1));
        let zi = f.load(Ty::F64, bodies, NONE, 1, off(i, 2));
        let mut vxi = f.load(Ty::F64, bodies, NONE, 1, off(i, 3));
        let mut vyi = f.load(Ty::F64, bodies, NONE, 1, off(i, 4));
        let mut vzi = f.load(Ty::F64, bodies, NONE, 1, off(i, 5));
        let mi = f.load(Ty::F64, bodies, NONE, 1, off(i, 6));
        for j in i + 1..NB {
            let xj = f.load(Ty::F64, bodies, NONE, 1, off(j, 0));
            let yj = f.load(Ty::F64, bodies, NONE, 1, off(j, 1));
            let zj = f.load(Ty::F64, bodies, NONE, 1, off(j, 2));
            let mj = f.load(Ty::F64, bodies, NONE, 1, off(j, 6));
            let dx = f.bin(Op::FSub, xi, xj);
            let dy = f.bin(Op::FSub, yi, yj);
            let dz = f.bin(Op::FSub, zi, zj);
            let dx2 = f.bin(Op::FMul, dx, dx);
            let dy2 = f.bin(Op::FMul, dy, dy);
            let dz2 = f.bin(Op::FMul, dz, dz);
            let s1 = f.bin(Op::FAdd, dx2, dy2);
            let d2 = f.bin(Op::FAdd, s1, dz2);
            f.guard(Cc::Gt, d2, zero, &[(cnt, 0), (acc_cur, 1)]);
            let sq = f.fsqrt(d2);
            let d3 = f.bin(Op::FMul, d2, sq);
            let mag = f.bin(Op::FDiv, dt, d3);
            let mjm = f.bin(Op::FMul, mj, mag);
            let mim = f.bin(Op::FMul, mi, mag);
            let t = f.bin(Op::FMul, dx, mjm);
            vxi = f.bin(Op::FSub, vxi, t);
            let t = f.bin(Op::FMul, dy, mjm);
            vyi = f.bin(Op::FSub, vyi, t);
            let t = f.bin(Op::FMul, dz, mjm);
            vzi = f.bin(Op::FSub, vzi, t);
            for (k, d) in [(3, dx), (4, dy), (5, dz)] {
                let v = f.load(Ty::F64, bodies, NONE, 1, off(j, k));
                let t = f.bin(Op::FMul, d, mim);
                let nv = f.bin(Op::FAdd, v, t);
                f.store(bodies, NONE, 1, off(j, k), nv);
            }
            acc_cur = f.bin(Op::FAdd, acc_cur, mag);
        }
        f.store(bodies, NONE, 1, off(i, 3), vxi);
        f.store(bodies, NONE, 1, off(i, 4), vyi);
        f.store(bodies, NONE, 1, off(i, 5), vzi);
    }
    for i in 0..NB {
        for k in 0..3 {
            let p = f.load(Ty::F64, bodies, NONE, 1, off(i, k));
            let v = f.load(Ty::F64, bodies, NONE, 1, off(i, k + 3));
            let t = f.bin(Op::FMul, dt, v);
            let np = f.bin(Op::FAdd, p, t);
            f.store(bodies, NONE, 1, off(i, k), np);
        }
    }
    let one = f.iconst(1);
    let cnt2 = f.bin(Op::Sub, cnt, one);
    f.back(&[cnt2, acc_cur]);
}

// ---- integer loop with a side --------------------------------------------

pub const INT_SIDE: Kernel = Kernel {
    name: "int_side",
    build: build_int_side,
    inputs: &[Input::DataPtr],
    scenarios: int_side_scenarios,
};

fn int_side_scenarios() -> Vec<(i64, Vec<u64>)> {
    let mut quiet: Vec<u64> = (0..128).map(|k| (k * 37 % 900) as u64).collect();
    let fuel_quiet = quiet.len() as i64 / 4;
    let mut loud = quiet.clone();
    loud[20] = 1500; // fuel 40 counts down to 20: the side runs at i = 20
    quiet[0] = 3;
    vec![(fuel_quiet, quiet), (40, loud)]
}

/// Counts `i` down from the fuel, reads `data[i]`, and when it is 1000 or
/// more leaves into a side that takes (i, sum, v, data). The loop keeps a running
/// sum in a register and stores it.
pub fn build_int_side(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let data = f.load(Ty::I64, inp, NONE, 1, 8);
    let n = f.load(Ty::I64, ctl, NONE, 1, 0);
    let sum0 = f.iconst(0);
    let first = f.header(&[n, sum0]);
    let (i, sum) = (first, first + 1);
    let zero = f.iconst(0);
    f.guard(Cc::Gt, i, zero, &[(i, 0), (sum, 1)]);
    let v = f.load(Ty::I64, data, i, 8, 0);
    let limit = f.iconst(1000);
    f.guard_side(Cc::Lt, v, limit, 0, &[i, sum, v, data]);
    let three = f.iconst(3);
    let t = f.bin(Op::Mul, v, three);
    let two = f.iconst(2);
    let sh = f.bin(Op::Shl, i, two);
    let t2 = f.bin(Op::Xor, t, sh);
    let sum2 = f.bin(Op::Add, sum, t2);
    f.store(data, i, 8, 512, sum2);
    let one = f.iconst(1);
    let i2 = f.bin(Op::Sub, i, one);
    f.back(&[i2, sum2]);
    let (_, p) = f.begin_side(&[Ty::I64, Ty::I64, Ty::I64, Ty::I64]);
    let (si, ssum, sv, data) = (p, p + 1, p + 2, p + 3);
    let lim = f.iconst(1000);
    let w = f.bin(Op::Sub, sv, lim);
    let m = f.bin(Op::Mul, w, si);
    let s2 = f.bin(Op::Add, ssum, m);
    f.store(data, NONE, 1, 0, s2);
    let five = f.iconst(5);
    let r = f.select(Cc::Lt, w, five, w, five);
    f.exit(&[(s2, 0), (r, 1), (si, 2), (sv, 3)]);
}

// ---- loop-carried swap ----------------------------------------------------

pub const SWAP: Kernel = Kernel {
    name: "swap",
    build: build_swap,
    inputs: &[Input::Int(7), Input::Int(100)],
    scenarios: || vec![(9, vec![0; 4]), (8, vec![0; 4])],
};

/// `(p0, p1) -> (p1, p0)` every iteration, with an accumulator that depends on
/// which is which, so the swap cannot be dropped. Also a rotation of three.
pub fn build_swap(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let a = f.load(Ty::I64, inp, NONE, 1, 8);
    let b = f.load(Ty::I64, inp, NONE, 1, 24);
    let n = f.load(Ty::I64, ctl, NONE, 1, 0);
    let zero = f.iconst(0);
    let c3 = f.iconst(11);
    let first = f.header(&[n, a, b, c3, zero]);
    let (cnt, p0, p1, p2, acc) = (first, first + 1, first + 2, first + 3, first + 4);
    f.guard(
        Cc::Gt,
        cnt,
        zero,
        &[(cnt, 0), (p0, 1), (p1, 2), (p2, 3), (acc, 4)],
    );
    let two = f.iconst(2);
    let t = f.bin(Op::Mul, p1, two);
    let d = f.bin(Op::Sub, p0, t);
    let e = f.bin(Op::Xor, d, p2);
    let acc2 = f.bin(Op::Add, acc, e);
    let one = f.iconst(1);
    let cnt2 = f.bin(Op::Sub, cnt, one);
    // p0 <-> p1 swap, and p2 takes the new accumulator's low bits.
    let m = f.iconst(0xff);
    let p2n = f.bin(Op::And, acc2, m);
    f.back(&[cnt2, p1, p0, p2n, acc2]);
}

// ---- register pressure -------------------------------------------------------

pub const PRESSURE: Kernel = Kernel {
    name: "pressure",
    build: build_pressure,
    inputs: &[Input::DataPtr],
    scenarios: || {
        let mut d: Vec<u64> = (0..24).map(|k| (k * 7 + 3) as u64).collect();
        d.extend(fwords(
            &(0..24)
                .map(|k| 0.5 + f64::from(k) * 0.25)
                .collect::<Vec<_>>(),
        ));
        vec![(6, d)]
    },
};

/// 20 integer and 18 float loop invariants, all live through a body that first
/// forms every pairwise product and then sums them, so there are far more
/// live values than registers.
pub fn build_pressure(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let data = f.load(Ty::I64, inp, NONE, 1, 8);
    let n = f.load(Ty::I64, ctl, NONE, 1, 0);
    let v: Vec<Val> = (0..20)
        .map(|k| f.load(Ty::I64, data, NONE, 1, 8 * k))
        .collect();
    let g: Vec<Val> = (0..18)
        .map(|k| f.load(Ty::F64, data, NONE, 1, 8 * (24 + k)))
        .collect();
    let zero = f.iconst(0);
    let fz = f.fconst(0.0);
    let first = f.header(&[n, zero, fz]);
    let (cnt, ai, af) = (first, first + 1, first + 2);
    f.guard(
        Cc::Gt,
        cnt,
        zero,
        &[(cnt, 0), (ai, 1), (af, 2), (v[3], 3), (g[5], 4)],
    );
    let mut prods = Vec::new();
    for k in 0..20 {
        prods.push(f.bin(Op::Mul, v[k], v[(k + 1) % 20]));
    }
    let mut fprods = Vec::new();
    for k in 0..18 {
        fprods.push(f.bin(Op::FMul, g[k], g[(k + 5) % 18]));
    }
    let mut ai2 = ai;
    for (k, p) in prods.iter().enumerate() {
        let c = f.iconst(k as i64 + 1);
        let t = f.bin(Op::Xor, *p, c);
        ai2 = f.bin(Op::Add, ai2, t);
    }
    let mut af2 = af;
    for p in &fprods {
        af2 = f.bin(Op::FAdd, af2, *p);
    }
    let one = f.iconst(1);
    let cnt2 = f.bin(Op::Sub, cnt, one);
    f.back(&[cnt2, ai2, af2]);
}

// ---- guard exit restoring spilled and rematerialized values ----------------------

pub const GUARD_RESTORE: Kernel = Kernel {
    name: "guard_restore",
    build: build_guard_restore,
    inputs: &[Input::DataPtr],
    scenarios: || {
        let mut d: Vec<u64> = (0..24).map(|k| (k * 5 + 1) as u64).collect();
        d.extend(fwords(&[1.5, 2.5, 3.5, 4.5]));
        // Fuel 5 leaves through the loop guard; a sentinel in data[23] makes
        // the second guard fail on the first iteration.
        let mut early = d.clone();
        early[23] = 9999;
        vec![(5, d), (5, early)]
    },
};

/// Two exits: one after the loop counter hits 0 and one on a data check. Both
/// snapshots hold values that are bound, spilled, rematerialized constants,
/// and a value only the snapshot uses.
pub fn build_guard_restore(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let data = f.load(Ty::I64, inp, NONE, 1, 8);
    let n = f.load(Ty::I64, ctl, NONE, 1, 0);
    let v: Vec<Val> = (0..24)
        .map(|k| f.load(Ty::I64, data, NONE, 1, 8 * k))
        .collect();
    let fl = f.load(Ty::F64, data, NONE, 1, 8 * 24);
    let zero = f.iconst(0);
    let first = f.header(&[n, zero]);
    let (cnt, acc) = (first, first + 1);
    let c77 = f.iconst(77);
    let c25 = f.fconst(2.5);
    let weak = f.bin(Op::Add, v[5], v[6]);
    let wf = f.bin(Op::FMul, fl, c25);
    let big = f.iconst(9000);
    f.guard(
        Cc::Lt,
        v[23],
        big,
        &[
            (v[3], 0),
            (v[17], 1),
            (c77, 2),
            (c25, 3),
            (weak, 4),
            (wf, 5),
            (acc, 6),
            (cnt, 7),
        ],
    );
    f.guard(
        Cc::Gt,
        cnt,
        zero,
        &[
            (v[3], 0),
            (v[17], 1),
            (c77, 2),
            (c25, 3),
            (weak, 4),
            (wf, 5),
            (acc, 6),
            (cnt, 7),
        ],
    );
    let mut prods = Vec::new();
    for k in 0..22 {
        prods.push(f.bin(Op::Mul, v[k], v[k + 1]));
    }
    let mut a2 = acc;
    for p in &prods {
        a2 = f.bin(Op::Add, a2, *p);
    }
    let one = f.iconst(1);
    let cnt2 = f.bin(Op::Sub, cnt, one);
    f.back(&[cnt2, a2]);
}

// ---- smallest function, for the fixed cost of a compile ----------------------------

pub const TINY: Kernel = Kernel {
    name: "tiny",
    build: build_tiny,
    inputs: &[Input::Int(5)],
    scenarios: || vec![(3, vec![0; 2])],
};

pub fn build_tiny(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let a = f.load(Ty::I64, inp, NONE, 1, 8);
    let n = f.load(Ty::I64, ctl, NONE, 1, 0);
    let zero = f.iconst(0);
    let first = f.header(&[n, a]);
    f.guard(Cc::Gt, first, zero, &[(first + 1, 0)]);
    let one = f.iconst(1);
    let n2 = f.bin(Op::Sub, first, one);
    let a2 = f.bin(Op::Add, first + 1, one);
    f.back(&[n2, a2]);
}

// ---- a lookup that walks two steps, in line, with nested skips --------------------

pub const SKIP: Kernel = Kernel {
    name: "skip",
    build: build_skip,
    inputs: &[Input::DataPtr, Input::Int(7)],
    scenarios: skip_scenarios,
};

fn skip_scenarios() -> Vec<(i64, Vec<u64>)> {
    let words = |f: fn(u64) -> u64| (0..16).map(f).collect::<Vec<u64>>();
    vec![
        // The key is in the first word of most steps.
        (24, words(|k| if k % 2 == 0 { 7 } else { k * 3 })),
        // Never the key: every step walks on, to the second and third word.
        (24, words(|k| k * 5 + 100)),
        // Only the second step finds it.
        (24, words(|k| if k % 2 == 1 { 7 } else { k + 40 })),
    ]
}

/// Each iteration reads three words of a table, one after the other, and goes
/// on to the next only when the one it read is not the key (the first
/// steps of a probe walk): a skip inside a skip, with a join value on each.
pub fn build_skip(f: &mut Func) {
    f.clear();
    let (inp, ctl) = f.entry();
    let data = f.load(Ty::I64, inp, NONE, 1, 8);
    let key = f.load(Ty::I64, inp, NONE, 1, 24);
    let fuel = f.load(Ty::I64, ctl, NONE, 1, 0);
    let zero = f.iconst(0);
    let one = f.iconst(1);
    let two = f.iconst(2);
    let mask = f.iconst(15);
    let first = f.header(&[fuel, zero]);
    let (cnt, acc) = (first, first + 1);
    f.guard(Cc::Gt, cnt, zero, &[(cnt, 0), (acc, 1)]);
    let at0 = f.bin(Op::And, cnt, mask);
    let w0 = f.load(Ty::I64, data, at0, 8, 0);
    let outer = f.begin_if(Cc::Ne, w0, key);
    let next1 = f.bin(Op::Add, cnt, one);
    let at1 = f.bin(Op::And, next1, mask);
    let w1 = f.load(Ty::I64, data, at1, 8, 0);
    let inner = f.begin_if(Cc::Ne, w1, key);
    let next2 = f.bin(Op::Add, cnt, two);
    let at2 = f.bin(Op::And, next2, mask);
    let w2 = f.load(Ty::I64, data, at2, 8, 0);
    let mixed = f.bin(Op::Xor, w1, w2);
    let step2 = f.end_if(inner, &[(mixed, w1)]);
    let step1 = f.bin(Op::Add, step2, one);
    let found = f.end_if(outer, &[(step1, w0)]);
    let acc2 = f.bin(Op::Add, acc, found);
    let cnt2 = f.bin(Op::Sub, cnt, one);
    f.back(&[cnt2, acc2]);
}
