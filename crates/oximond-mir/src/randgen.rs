//! Seeded random MIR programs for the differential test.

use crate::ir::{Cc, Func, NONE, Op, Ty, Val};
use crate::kernels::{Input, Scenario};
use crate::rng::Rng;

extern "C" fn rg_mix(a: i64, b: i64) -> i64 {
    a.wrapping_mul(31).wrapping_add(b) ^ (a >> 3)
}

extern "C" fn rg_fscale(x: f64, k: i64) -> f64 {
    x * (k & 0xffff) as f64 + 0.5
}

extern "C" fn rg_mixed(a: i64, x: f64, y: f64) -> f64 {
    (a & 0xff) as f64 * x - y
}

/// The target of a conditional call on a compare: the bits of what it was
/// given, mixed, so a wrong argument or a wrong register shows in the result.
/// Every NaN counts as one, as the comparison of the regions does: the sign of
/// the NaN that an operation of two NaNs of different signs gives is the
/// machine's pick of operand, which the register allocator may swap.
extern "C" fn rg_compared(x: f64, y: f64, a: i64, b: i64) -> i64 {
    let bits = |v: f64| {
        if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        }
    };
    (bits(x) ^ bits(y).rotate_left(7)).cast_signed() ^ a.wrapping_mul(13) ^ b
}

#[allow(clippy::too_many_arguments)]
extern "C" fn rg_nine(
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
    i: i64,
) -> i64 {
    a ^ b.wrapping_mul(3)
        ^ c.wrapping_mul(5)
        ^ d.wrapping_mul(7)
        ^ e
        ^ f.wrapping_mul(11)
        ^ g.wrapping_mul(13)
        ^ h.wrapping_mul(17)
        ^ i.wrapping_mul(19)
}

struct Gen<'a> {
    r: &'a mut Rng,
    /// The stream that decides about the `Join`, apart from `r` so that the
    /// programs without one stay what they were.
    j: Rng,
    /// The stream that decides which float operations are cold, for the same
    /// reason.
    c: Rng,
    /// The stream that decides about the fused multiply-add and the
    /// conditional calls, for the same reason.
    n: Rng,
    /// The target has a fused multiply-add, so `FNma` is generated.
    fused: bool,
    /// The target shifts by a register, so shifts by a value are generated.
    shifts: bool,
    /// The stream that decides about those, apart from `n` so that the
    /// programs of a target without them stay what they were.
    s: Rng,
    /// The join's second parameter, the budget of rejoins, or `NONE` for a
    /// program with no `Join`.
    budget: Val,
    ints: Vec<Val>,
    flts: Vec<Val>,
    data: Val,
    next_slot: u32,
    sides: Vec<Vec<Ty>>,
    /// How many skipped blocks enclose the code being generated.
    depth: u32,
}

impl Gen<'_> {
    fn pick(&mut self, ty: Ty) -> Val {
        let n = match ty {
            Ty::I64 => self.ints.len(),
            Ty::F64 => self.flts.len(),
        };
        // Bias toward recent values, but reach far back sometimes.
        let k = if self.r.chance(60) {
            self.r.below(n.min(6) as u64) as usize
        } else {
            self.r.below(n as u64) as usize
        };
        match ty {
            Ty::I64 => self.ints[n - 1 - k],
            Ty::F64 => self.flts[n - 1 - k],
        }
    }

    fn operand(&mut self, f: &mut Func, ty: Ty) -> Val {
        if self.r.chance(12) {
            match ty {
                Ty::I64 => {
                    let c = match self.r.below(4) {
                        0 => self.r.below(8) as i64,
                        1 => -(self.r.below(100) as i64),
                        2 => i64::from(self.r.next_u64() as i32),
                        _ => self.r.next_u64() as i64,
                    };
                    f.iconst(c)
                }
                Ty::F64 => {
                    let c = [0.0, 1.0, -1.5, 2.5, 0.125, 1e10][self.r.below(6) as usize];
                    f.fconst(c)
                }
            }
        } else {
            self.pick(ty)
        }
    }

    fn snapshot(&mut self, f: &mut Func) -> Vec<(Val, u32)> {
        let n = self.r.below(6);
        let mut v = Vec::new();
        for _ in 0..n {
            let ty = if self.r.chance(65) { Ty::I64 } else { Ty::F64 };
            let val = if self.r.chance(15) {
                self.operand(f, ty)
            } else {
                self.pick(ty)
            };
            v.push((val, self.next_slot));
            self.next_slot += 1;
        }
        v
    }

    /// A forward skip: a few operations that run when a random condition
    /// holds, and join values taken from the block or from before it.
    fn skipped_block(&mut self, f: &mut Func, in_main: bool) {
        let isf = self.r.chance(25);
        let ty = if isf { Ty::F64 } else { Ty::I64 };
        let ccs: &[Cc] = if isf {
            &[Cc::Lt, Cc::Le, Cc::Gt, Cc::Ge, Cc::Eq, Cc::Ne]
        } else {
            &[
                Cc::Lt,
                Cc::Le,
                Cc::Gt,
                Cc::Ge,
                Cc::Eq,
                Cc::Ne,
                Cc::Ult,
                Cc::Uge,
            ]
        };
        let cc = ccs[self.r.below(ccs.len() as u64) as usize];
        let (a, b) = (self.operand(f, ty), self.operand(f, ty));
        // The skipped side reads only what stands before the block, constants
        // included, so those are chosen (and made) here.
        let mut skipped = Vec::new();
        for _ in 0..self.r.below(4) {
            let t = if self.r.chance(60) { Ty::I64 } else { Ty::F64 };
            let e = if self.r.chance(15) {
                self.operand(f, t)
            } else {
                self.pick(t)
            };
            skipped.push((t, e));
        }
        let (ni, nf) = (self.ints.len(), self.flts.len());
        let open = f.begin_if(cc, a, b);
        self.depth += 1;
        for _ in 0..=self.r.below(6) {
            self.op(f, in_main);
        }
        self.depth -= 1;
        // The block's values stand until its end.
        let joins: Vec<(Val, Val)> = skipped.iter().map(|&(t, e)| (self.pick(t), e)).collect();
        self.ints.truncate(ni);
        self.flts.truncate(nf);
        let then: Vec<Ty> = skipped.iter().map(|&(t, _)| t).collect();
        let first = f.end_if(open, &joins);
        for (k, &t) in then.iter().enumerate() {
            match t {
                Ty::I64 => self.ints.push(first + k as u32),
                Ty::F64 => self.flts.push(first + k as u32),
            }
        }
    }

    /// One operation of the body, and sometimes one more of those that the
    /// streams of `r` do not decide: a fused multiply-add, a conditional call.
    fn op(&mut self, f: &mut Func, in_main: bool) {
        self.plain_op(f, in_main);
        if self.shifts && self.s.chance(8) {
            // Counts of every size: the machine takes the low six bits, as
            // the evaluator does.
            let op = [Op::Shl, Op::Shr, Op::Sar][self.s.below(3) as usize];
            let a = self.operand(f, Ty::I64);
            let count = self.operand(f, Ty::I64);
            let v = f.bin(op, a, count);
            self.ints.push(v);
        }
        if self.fused && self.n.chance(6) {
            let (a, b, c) = (
                self.operand(f, Ty::F64),
                self.operand(f, Ty::F64),
                self.operand(f, Ty::F64),
            );
            let v = f.fnma(a, b, c);
            self.flts.push(v);
        }
        if self.n.chance(5) {
            self.conditional_call(f);
        }
    }

    /// A call out of line on an integer compare or, mostly, a float compare
    /// (the unordered conditions among them), storing its result in the
    /// control block.
    fn conditional_call(&mut self, f: &mut Func) {
        let float = self.n.chance(70);
        let ty = if float { Ty::F64 } else { Ty::I64 };
        let ccs: &[Cc] = if float {
            &[
                Cc::Eq,
                Cc::Ne,
                Cc::Lt,
                Cc::Le,
                Cc::Gt,
                Cc::Ge,
                Cc::Nlt,
                Cc::Nle,
                Cc::Ngt,
                Cc::Nge,
            ]
        } else {
            &[Cc::Eq, Cc::Ne, Cc::Lt, Cc::Ge, Cc::Ult, Cc::Uge]
        };
        let cc = ccs[self.n.below(ccs.len() as u64) as usize];
        let (a, b) = (self.operand(f, ty), self.operand(f, ty));
        let (x, y) = (self.operand(f, Ty::F64), self.operand(f, Ty::F64));
        let (i, j) = (self.operand(f, Ty::I64), self.operand(f, Ty::I64));
        // The second word of the control block.
        let at = if self.n.chance(50) { Some(8) } else { None };
        f.call_if_storing(
            cc,
            (a, b),
            rg_compared as *const () as usize,
            false,
            &[x, y, i, j],
            at,
        );
    }

    fn plain_op(&mut self, f: &mut Func, in_main: bool) {
        match self.r.below(100) {
            0..=29 => {
                let op =
                    [Op::Add, Op::Sub, Op::Mul, Op::And, Op::Or, Op::Xor][self.r.below(6) as usize];
                let a = self.operand(f, Ty::I64);
                let b = self.operand(f, Ty::I64);
                let v = f.bin(op, a, b);
                self.ints.push(v);
            }
            30..=34 => {
                let op = [Op::Shl, Op::Shr, Op::Sar][self.r.below(3) as usize];
                let a = self.pick(Ty::I64);
                let count = self.r.below(64);
                // A shift right that only snapshots may end up reading is
                // made by the stubs that read it.
                let v = if op == Op::Sar && self.c.chance(50) {
                    f.cold = true;
                    let v = f.sar_cold(a, count as u8);
                    f.cold = false;
                    v
                } else {
                    let c = f.iconst(count as i64);
                    f.bin(op, a, c)
                };
                self.ints.push(v);
            }
            35..=54 => {
                let op = [Op::FAdd, Op::FSub, Op::FMul, Op::FDiv][self.r.below(4) as usize];
                let a = self.operand(f, Ty::F64);
                let b = self.operand(f, Ty::F64);
                // A cold operation is read where a stub computes it, unless
                // something else reads it too.
                f.cold = self.c.chance(35);
                let op = if self.c.chance(8) { Op::FCopySign } else { op };
                let v = f.bin(op, a, b);
                f.cold = false;
                self.flts.push(v);
            }
            55..=56 => {
                let a = self.pick(Ty::F64);
                let v = if self.c.chance(50) {
                    let op = [Op::FNeg, Op::FAbs, Op::FFloor, Op::FCeil, Op::FTrunc]
                        [self.c.below(5) as usize];
                    f.cold = self.c.chance(60);
                    let v = f.funary(op, a);
                    f.cold = false;
                    v
                } else {
                    f.fsqrt(a)
                };
                self.flts.push(v);
            }
            57 => {
                // A call: live values of both classes sit across it.
                match self.r.below(5) {
                    3 => {
                        // Arguments past the sixth go to the stack.
                        let args: Vec<Val> = (0..9).map(|_| self.operand(f, Ty::I64)).collect();
                        let v = f.call(rg_nine as *const () as usize, &args, Some(Ty::I64));
                        self.ints.push(v);
                    }
                    4 => {
                        // A value kept in a frame cell across a call.
                        let kept = self.pick(Ty::I64);
                        let cell = f.new_cell();
                        f.cell_store(cell, kept);
                        let (a, b) = (self.operand(f, Ty::I64), self.operand(f, Ty::I64));
                        let v = f.call(rg_mix as *const () as usize, &[a, b], Some(Ty::I64));
                        let back = f.cell_load(Ty::I64, cell);
                        self.ints.push(v);
                        self.ints.push(back);
                    }
                    0 => {
                        let (a, b) = (self.operand(f, Ty::I64), self.operand(f, Ty::I64));
                        let v = f.call(rg_mix as *const () as usize, &[a, b], Some(Ty::I64));
                        self.ints.push(v);
                    }
                    1 => {
                        let (x, k) = (self.operand(f, Ty::F64), self.operand(f, Ty::I64));
                        let v = f.call(rg_fscale as *const () as usize, &[x, k], Some(Ty::F64));
                        self.flts.push(v);
                    }
                    _ => {
                        let (x, a, y) = (
                            self.operand(f, Ty::F64),
                            self.operand(f, Ty::I64),
                            self.operand(f, Ty::F64),
                        );
                        let v = f.call(rg_mixed as *const () as usize, &[a, x, y], Some(Ty::F64));
                        self.flts.push(v);
                    }
                }
            }
            58..=61 => {
                let a = self.pick(Ty::I64);
                let m = f.iconst(0xff);
                let small = f.bin(Op::And, a, m);
                // A cold conversion reads an integer in the stub.
                f.cold = self.c.chance(40);
                let v = f.i2f(small);
                f.cold = false;
                self.flts.push(v);
            }
            62..=69 => {
                let cc = [
                    Cc::Eq,
                    Cc::Ne,
                    Cc::Lt,
                    Cc::Le,
                    Cc::Gt,
                    Cc::Ge,
                    Cc::Ult,
                    Cc::Ule,
                    Cc::Ugt,
                    Cc::Uge,
                ][self.r.below(10) as usize];
                let a = self.operand(f, Ty::I64);
                let b = self.operand(f, Ty::I64);
                let t = self.operand(f, Ty::I64);
                let e = self.operand(f, Ty::I64);
                let v = f.select(cc, a, b, t, e);
                self.ints.push(v);
            }
            70..=79 => {
                // Load: ints live in words 0..16 of data, floats in 16..32.
                let isf = self.r.chance(40);
                let (base, ty) = if isf { (128, Ty::F64) } else { (0, Ty::I64) };
                let v = if self.r.chance(50) {
                    f.load(ty, self.data, NONE, 1, base + 8 * self.r.below(16) as i32)
                } else {
                    let a = self.pick(Ty::I64);
                    let m = f.iconst(15);
                    let idx = f.bin(Op::And, a, m);
                    f.load(ty, self.data, idx, 8, base)
                };
                if isf {
                    self.flts.push(v)
                } else {
                    self.ints.push(v)
                }
            }
            80..=86 => {
                let isf = self.r.chance(40);
                let (base, ty) = if isf { (128, Ty::F64) } else { (0, Ty::I64) };
                let val = self.operand(f, ty);
                if self.r.chance(50) {
                    f.store(self.data, NONE, 1, base + 8 * self.r.below(16) as i32, val);
                } else {
                    let a = self.pick(Ty::I64);
                    let m = f.iconst(15);
                    let idx = f.bin(Op::And, a, m);
                    f.store(self.data, idx, 8, base, val);
                }
            }
            87..=91 if self.depth < 3 => self.skipped_block(f, in_main),
            _ => {
                // A guard: mostly true, sometimes random. Sides only from the root.
                let isf = self.r.chance(35);
                let ty = if isf { Ty::F64 } else { Ty::I64 };
                let (a, b, cc) = if self.r.chance(70) {
                    let a = self.pick(ty);
                    (
                        a,
                        a,
                        if isf {
                            Cc::Ge
                        } else {
                            [Cc::Le, Cc::Ge, Cc::Eq, Cc::Uge][self.r.below(4) as usize]
                        },
                    )
                } else {
                    let a = self.operand(f, ty);
                    let b = self.operand(f, ty);
                    let ccs: &[Cc] = if isf {
                        &[Cc::Lt, Cc::Le, Cc::Gt, Cc::Ge, Cc::Eq, Cc::Ne]
                    } else {
                        &[
                            Cc::Lt,
                            Cc::Le,
                            Cc::Gt,
                            Cc::Ge,
                            Cc::Eq,
                            Cc::Ne,
                            Cc::Ult,
                            Cc::Uge,
                        ]
                    };
                    (a, b, ccs[self.r.below(ccs.len() as u64) as usize])
                };
                if in_main && self.r.chance(30) && self.sides.len() < 2 {
                    let n = 1 + self.r.below(4) as usize;
                    // The data pointer travels as the side's first parameter.
                    let mut args = vec![self.data];
                    let mut tys = vec![Ty::I64];
                    if self.budget != NONE {
                        args.push(self.budget);
                        tys.push(Ty::I64);
                    }
                    for _ in 0..n {
                        let t = if self.r.chance(60) { Ty::I64 } else { Ty::F64 };
                        args.push(self.pick(t));
                        tys.push(t);
                    }
                    let idx = self.sides.len() as u32;
                    self.sides.push(tys);
                    f.guard_side(cc, a, b, idx, &args);
                } else {
                    let snap = self.snapshot(f);
                    f.guard(cc, a, b, &snap);
                }
            }
        }
    }
}

/// Builds a random program into `f` and returns a scenario to run it on.
pub fn generate(seed: u64, f: &mut Func) -> (Scenario, &'static [Input]) {
    generate_for(seed, f, false)
}

/// [`generate`] for a target that has a fused multiply-add when `fused`: the
/// program has `FNma` too.
pub fn generate_for(seed: u64, f: &mut Func, fused: bool) -> (Scenario, &'static [Input]) {
    generate_with(seed, f, fused, false)
}

/// [`generate_for`] for a target that also shifts by a register when
/// `shifts`.
pub fn generate_with(
    seed: u64,
    f: &mut Func,
    fused: bool,
    shifts: bool,
) -> (Scenario, &'static [Input]) {
    const INPUTS: &[Input] = &[Input::DataPtr, Input::Int(5), Input::Float(1.25)];
    let mut rng = Rng::new(seed);
    f.clear();
    let mut g = Gen {
        r: &mut rng,
        j: Rng::new(seed ^ 0x4a4f_494e_5f31),
        c: Rng::new(seed ^ 0x434f_4c44_5f31),
        n: Rng::new(seed ^ 0x4e4d_415f_5f31),
        fused,
        shifts,
        s: Rng::new(seed ^ 0x5348_4946_5f31),
        budget: NONE,
        ints: Vec::new(),
        flts: Vec::new(),
        data: 0,
        next_slot: 0,
        sides: Vec::new(),
        depth: 0,
    };
    let (inp, ctl) = f.entry();
    g.data = f.load(Ty::I64, inp, NONE, 1, 8);
    g.ints.push(f.load(Ty::I64, inp, NONE, 1, 24));
    g.flts.push(f.load(Ty::F64, inp, NONE, 1, 40));
    g.ints.push(f.load(Ty::I64, ctl, NONE, 1, 0));
    for k in 0..(3 + g.r.below(5)) {
        let v = f.load(Ty::I64, g.data, NONE, 1, 8 * k as i32);
        g.ints.push(v);
    }
    for k in 0..(2 + g.r.below(4)) {
        let v = f.load(Ty::F64, g.data, NONE, 1, 128 + 8 * k as i32);
        g.flts.push(v);
    }
    let looped = g.r.chance(75);
    let nbody = 4 + g.r.below(50) as usize;
    let joined = looped && g.j.chance(30);
    let mut joined_counter = NONE;
    let mut second_join = false;
    // The values the join keeps beyond the usual ones: the first guard of the
    // loop leaves with them, so each is read there, from its cell when it
    // has no register.
    let mut joined_extra: Vec<Val> = Vec::new();
    if joined {
        // A `Join` between the loads and the loop: whatever the code below
        // reads of what the loads made goes through frame cells, and the
        // sides rejoin with a new counter and one rejoin less to spend.
        let budget = g.ints[1];
        let counter = f.iconst(1 + g.j.below(5) as i64);
        // More values than registers, some of the time: those that get none
        // live in their cells. Their own stream keeps the other programs as
        // they were.
        let mut more = Rng::new(seed ^ 0x4345_4c4c_5f48);
        let mut extra = (0, 0);
        if more.chance(40) {
            extra = (8 + more.below(10) as usize, 3 + more.below(6) as usize);
            for _ in 0..extra.0 {
                let v = f.load(Ty::I64, g.data, NONE, 1, 8 * more.below(3) as i32);
                g.ints.push(v);
            }
            for _ in 0..extra.1 {
                let v = f.load(Ty::F64, g.data, NONE, 1, 128 + 8 * more.below(2) as i32);
                g.flts.push(v);
            }
        }
        let mut kept: Vec<(Val, Ty, u32)> = Vec::new();
        let (ints, flts, data) = (g.ints.clone(), g.flts.clone(), g.data);
        for (val, ty) in ints
            .iter()
            .map(|v| (*v, Ty::I64))
            .chain(flts.iter().map(|v| (*v, Ty::F64)))
            .chain([(data, Ty::I64)])
        {
            if !kept.iter().any(|(held, _, _)| *held == val) {
                let cell = f.new_cell();
                f.cell_store(cell, val);
                kept.push((val, ty, cell));
            }
        }
        let first = f.join(&[counter, budget]);
        joined_counter = first;
        g.budget = first + 1;
        for (old, ty, cell) in kept {
            let new = f.cell_load(ty, cell);
            for pool in [&mut g.ints, &mut g.flts] {
                for held in pool.iter_mut().filter(|held| **held == old) {
                    *held = new;
                }
            }
            if g.data == old {
                g.data = new;
            }
        }
        let zero = f.iconst(0);
        f.guard(Cc::Gt, g.budget, zero, &[]);
        let (ints, flts) = (g.ints.len(), g.flts.len());
        joined_extra.extend_from_slice(&g.ints[ints - extra.0..]);
        joined_extra.extend_from_slice(&g.flts[flts - extra.1..]);
        // A second `Join` right below the first one's work, as a root that
        // versions its loop has (the first before the checks, the second
        // before the loop): what the code below reads goes through cells
        // again, and some sides rejoin this one. Its own stream keeps the
        // other programs as they were.
        let mut again = Rng::new(seed ^ 0x4a4f_494e_5f32);
        if again.chance(50) {
            let mut kept: Vec<(Val, Ty, u32)> = Vec::new();
            let (ints, flts, data) = (g.ints.clone(), g.flts.clone(), g.data);
            for (val, ty) in ints
                .iter()
                .map(|v| (*v, Ty::I64))
                .chain(flts.iter().map(|v| (*v, Ty::F64)))
                .chain([(data, Ty::I64)])
            {
                if !kept.iter().any(|(held, _, _)| *held == val) {
                    let cell = f.new_cell();
                    f.cell_store(cell, val);
                    kept.push((val, ty, cell));
                }
            }
            let first = f.join(&[joined_counter, g.budget]);
            joined_counter = first;
            g.budget = first + 1;
            for (old, ty, cell) in kept {
                let new = f.cell_load(ty, cell);
                for pool in [&mut g.ints, &mut g.flts, &mut joined_extra] {
                    for held in pool.iter_mut().filter(|held| **held == old) {
                        *held = new;
                    }
                }
                if g.data == old {
                    g.data = new;
                }
            }
            // A rejoin of this join spends its unit like the other's.
            let zero = f.iconst(0);
            f.guard(Cc::Gt, g.budget, zero, &[]);
            second_join = true;
        }
    }
    if looped {
        // Cold operations before the loop, which snapshots inside it may
        // read: a short run or one longer than a stub computes. Their own
        // stream keeps the other programs as they were.
        let mut pre = Rng::new(seed ^ 0x434f_4c44_5f32);
        if pre.chance(20) {
            let len = if pre.chance(15) { 70 } else { 1 + pre.below(4) };
            let mut v = g.flts[pre.below(g.flts.len() as u64) as usize];
            f.cold = true;
            for _ in 0..len {
                let op = [Op::FAdd, Op::FSub, Op::FMul, Op::FDiv][pre.below(4) as usize];
                let b = g.flts[pre.below(g.flts.len() as u64) as usize];
                v = f.bin(op, v, b);
                if pre.chance(30) {
                    g.flts.push(v);
                }
            }
            f.cold = false;
            g.flts.push(v);
        }
        let counter = if joined {
            joined_counter
        } else {
            f.iconst(1 + g.r.below(5) as i64)
        };
        let mut init = vec![counter];
        let mut tys = vec![Ty::I64];
        for _ in 0..g.r.below(7) {
            let t = if g.r.chance(60) { Ty::I64 } else { Ty::F64 };
            init.push(g.pick(t));
            tys.push(t);
        }
        let first = f.header(&init);
        let zero = f.iconst(0);
        // Parameters join the pools; the counter does not (it is checked here).
        for (k, t) in tys.iter().enumerate().skip(1) {
            match t {
                Ty::I64 => g.ints.push(first + k as u32),
                Ty::F64 => g.flts.push(first + k as u32),
            }
        }
        let snap: Vec<(Val, u32)> = vec![(first, 0), (g.pick(Ty::I64), 1), (g.pick(Ty::F64), 2)];
        g.next_slot = 3;
        let mut snap = snap;
        for v in &joined_extra {
            snap.push((*v, g.next_slot));
            g.next_slot += 1;
        }
        f.guard(Cc::Gt, first, zero, &snap);
        for _ in 0..nbody {
            g.op(f, true);
        }
        let one = f.iconst(1);
        let c2 = f.bin(Op::Sub, first, one);
        let mut args = vec![c2];
        for t in tys.iter().skip(1) {
            args.push(g.pick(*t));
        }
        f.back(&args);
    } else {
        for _ in 0..nbody {
            g.op(f, true);
        }
        let snap = g.snapshot(f);
        f.exit(&snap);
    }
    let sides = std::mem::take(&mut g.sides);
    let mut which = Rng::new(seed ^ 0x5349_4445_5f32);
    for tys in sides {
        let (_, first) = f.begin_side(&tys);
        g.ints.clear();
        g.flts.clear();
        g.data = first;
        let budget = if joined { first + 1 } else { NONE };
        for (k, t) in tys.iter().enumerate().skip(1) {
            match t {
                Ty::I64 => g.ints.push(first + k as u32),
                Ty::F64 => g.flts.push(first + k as u32),
            }
        }
        // A side needs one value of each type to pick from.
        if g.ints.is_empty() {
            g.ints.push(f.iconst(3));
        }
        if g.flts.is_empty() {
            g.flts.push(f.fconst(0.5));
        }
        for _ in 0..(2 + g.r.below(12)) {
            g.op(f, false);
        }
        if joined && g.j.chance(70) {
            // Back to the loop with a counter of 0 to 3, spending one rejoin.
            let three = f.iconst(3);
            let any = g.pick(Ty::I64);
            let counter = f.bin(Op::And, any, three);
            let one = f.iconst(1);
            let left = f.bin(Op::Sub, budget, one);
            if second_join && which.chance(50) {
                f.rejoin_nth(0, 1, &[counter, left]);
            } else {
                f.rejoin(&[counter, left]);
            }
            continue;
        }
        let snap = g.snapshot(f);
        f.exit(&snap);
    }
    // Data: ints in words 0..16, floats in 16..32 (kept moderate).
    let mut d = Vec::with_capacity(32);
    for _ in 0..16 {
        d.push(match rng.below(3) {
            0 => rng.below(20),
            1 => rng.next_u64(),
            _ => (-(rng.below(50) as i64)) as u64,
        });
    }
    for _ in 0..16 {
        let x = (rng.below(2000) as f64 - 1000.0) / 250.0;
        d.push(x.to_bits());
    }
    let area = f.exit_area_len();
    (Scenario::new(INPUTS, 3, area, d), INPUTS)
}
