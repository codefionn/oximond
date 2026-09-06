//! The operations the oximond backend added: float unary ops, `F2I`, byte
//! loads and stores, negated float conditions, control-block exit slots and
//! exit codes. Each is run natively on both encodings and compared with the
//! value Rust computes.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, Inst, NONE, Op, SnapVal, Ty};

/// `inputs[0]` is a float, `inputs[1]` an integer. Every result is stored to
/// the exit area by an exit.
fn unary_func() -> Func {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let x = f.load(Ty::F64, inputs, NONE, 1, 0);
    let mut results = Vec::new();
    for op in [Op::FNeg, Op::FAbs, Op::FFloor, Op::FCeil, Op::FTrunc] {
        let dst = f.val(Ty::F64);
        let mut i = Inst::new(op, Ty::F64);
        i.dst = dst;
        i.a = x;
        f.push(i);
        results.push(dst);
    }
    let dst = f.val(Ty::I64);
    let mut i = Inst::new(Op::F2I, Ty::I64);
    i.dst = dst;
    i.a = x;
    f.push(i);
    results.push(dst);
    let vals: Vec<(u32, u32)> = results
        .iter()
        .enumerate()
        .map(|(k, v)| (*v, k as u32))
        .collect();
    f.exit(&vals);
    f
}

#[test]
fn float_unary_ops_and_f2i_match_rust() {
    let f = unary_func();
    f.validate().unwrap();
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        for x in [
            0.0f64, -0.0, 1.5, -1.5, 2.0, -2.5, 1e10, -1e10, 0.49999, 7.0,
        ] {
            let mut inputs = [x.to_bits(), 0];
            let mut control = [0u64; 4];
            let mut area = [0u64; 8];
            let r = code.call(&mut inputs, &mut control, &mut area);
            assert_eq!(r, -1);
            assert_eq!(f64::from_bits(area[0]).to_bits(), (-x).to_bits(), "neg {x}");
            assert_eq!(
                f64::from_bits(area[1]).to_bits(),
                x.abs().to_bits(),
                "abs {x}"
            );
            assert_eq!(
                f64::from_bits(area[2]).to_bits(),
                x.floor().to_bits(),
                "floor {x}"
            );
            assert_eq!(
                f64::from_bits(area[3]).to_bits(),
                x.ceil().to_bits(),
                "ceil {x}"
            );
            assert_eq!(
                f64::from_bits(area[4]).to_bits(),
                x.trunc().to_bits(),
                "trunc {x}"
            );
            assert_eq!(area[5] as i64, x as i64, "f2i {x} (avx {})", t.avx);
        }
    }
}

/// A byte load reads one byte, a byte store writes one and keeps the rest.
#[test]
fn byte_loads_and_stores_touch_one_byte() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let idx = f.load(Ty::I64, inputs, NONE, 1, 0);
    let b = f.val(Ty::I64);
    let mut i = Inst::new(Op::Load8, Ty::I64);
    i.dst = b;
    i.a = inputs;
    i.b = idx;
    i.scale = 1;
    i.imm = 8;
    f.push(i);
    let seven = f.iconst(0x1_07);
    let mut st = Inst::new(Op::Store8, Ty::I64);
    st.a = inputs;
    st.b = idx;
    st.scale = 1;
    st.imm = 16;
    st.c = seven;
    f.push(st);
    f.exit(&[(b, 0)]);
    f.validate().unwrap();
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        // Bytes of inputs[1] are 0x80, 0x70, ... and inputs[2] is all ones.
        let mut inputs = [3u64, 0x1122_3344_5566_7788, u64::MAX];
        let mut control = [0u64; 4];
        let mut area = [0u64; 2];
        code.call(&mut inputs, &mut control, &mut area);
        assert_eq!(area[0], 0x55, "byte 3 of the second word");
        assert_eq!(
            inputs[2], 0xffff_ffff_07ff_ffff,
            "byte 3 of the third word replaced"
        );
    }
}

/// A 32-bit load with a scaled index reads a zero-extended word of a table
/// of words, as the length of a string lane is read, in a skip that jumps over
/// an index that is out of the table.
#[test]
fn a_word_load_with_a_scaled_index_zero_extends_and_a_skip_guards_the_index() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let table = f.load(Ty::I64, inputs, NONE, 1, 0);
    let lane = f.load(Ty::I64, inputs, NONE, 1, 8);
    let limit = f.iconst(4);
    let none = f.iconst(-1);
    let open = f.begin_if(Cc::Ult, lane, limit);
    let word = f.val(Ty::I64);
    let mut i = Inst::new(Op::Load32, Ty::I64);
    i.dst = word;
    i.a = table;
    i.b = lane;
    i.scale = 4;
    f.push(i);
    let joined = f.end_if(open, &[(word, none)]);
    f.exit(&[(joined, 0)]);
    f.validate().unwrap();
    // Four words, the high bit set in the second, and a poison word after.
    let words: [u32; 5] = [7, 0x8000_0001, 0xffff_ffff, 0, 0xdead_beef];
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        for lane in 0..8u64 {
            let mut inputs = [words.as_ptr() as u64, lane];
            let mut control = [0u64; 4];
            let mut area = [0u64; 1];
            code.call(&mut inputs, &mut control, &mut area);
            let want = if lane < 4 {
                u64::from(words[lane as usize])
            } else {
                u64::MAX
            };
            assert_eq!(area[0], want, "lane {lane}");
        }
    }
}

/// `Guard` with a negated float condition continues on NaN.
#[test]
fn negated_float_conditions_continue_on_nan() {
    for (cc, name) in [
        (Cc::Nlt, "nlt"),
        (Cc::Nle, "nle"),
        (Cc::Ngt, "ngt"),
        (Cc::Nge, "nge"),
        (Cc::Lt, "lt"),
        (Cc::Ge, "ge"),
        (Cc::Ne, "ne"),
        (Cc::Eq, "eq"),
    ] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let a = f.load(Ty::F64, inputs, NONE, 1, 0);
        let b = f.load(Ty::F64, inputs, NONE, 1, 8);
        let one = f.iconst(1);
        f.guard(cc, a, b, &[(one, 0)]);
        let two = f.iconst(2);
        f.exit(&[(two, 0)]);
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            for (x, y) in [
                (1.0f64, 2.0f64),
                (2.0, 1.0),
                (1.0, 1.0),
                (f64::NAN, 1.0),
                (1.0, f64::NAN),
                (f64::NAN, f64::NAN),
            ] {
                let mut inputs = [x.to_bits(), y.to_bits()];
                let mut control = [0u64; 2];
                let mut area = [0u64; 1];
                let r = code.call(&mut inputs, &mut control, &mut area);
                let holds = cc.eval_f(x, y);
                // The guard continues (exit 1, the second snapshot) when the
                // condition holds and leaves through exit 0 otherwise.
                let (want_code, want_area) = if holds { (-2, 2) } else { (-1, 1) };
                assert_eq!(
                    (r, area[0]),
                    (want_code, want_area),
                    "{name} {x} {y} avx {}",
                    t.avx
                );
            }
        }
    }
}

/// A snapshot slot with `CONTROL_SLOT` set goes to the control block, and a
/// snapshot can return a code of its own.
#[test]
fn control_slots_and_exit_codes() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let v = f.load(Ty::I64, inputs, NONE, 1, 0);
    let w = f.load(Ty::I64, inputs, NONE, 1, 8);
    let zero = f.iconst(0);
    let vals = [
        SnapVal { val: v, slot: 3 },
        SnapVal {
            val: w,
            slot: CONTROL_SLOT,
        },
        SnapVal {
            val: v,
            slot: CONTROL_SLOT | 2,
        },
    ];
    let snap = f.snap_returning(&vals, i64::MIN + 7);
    let mut g = Inst::new(Op::Guard, Ty::I64);
    g.cc = Cc::Eq;
    g.a = v;
    g.b = zero;
    g.c = snap;
    f.push(g);
    f.exit(&[]);
    f.validate().unwrap();
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        let mut inputs = [5u64, 9];
        let mut control = [0u64; 4];
        let mut area = [0u64; 4];
        let r = code.call(&mut inputs, &mut control, &mut area);
        assert_eq!(r, i64::MIN + 7);
        assert_eq!(area, [0, 0, 0, 5]);
        assert_eq!(control, [9, 0, 5, 0]);
    }
}

/// `Select` with float arms or a float compare, for every condition.
#[test]
fn selects_with_float_arms_or_compares_pick_the_right_arm() {
    let float_ccs = [
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
    ];
    let int_ccs = [
        Cc::Eq,
        Cc::Ne,
        Cc::Lt,
        Cc::Le,
        Cc::Gt,
        Cc::Ge,
        Cc::Ult,
        Cc::Uge,
    ];
    let floats = [1.0f64, 2.0, -3.5, 0.0, f64::NAN, f64::INFINITY];
    // inputs: [a, b, t, f] as 4 words; arms and compare operands are chosen
    // by type below.
    for (cmp_float, arms_float) in [(true, true), (true, false), (false, true)] {
        let ccs: &[Cc] = if cmp_float { &float_ccs } else { &int_ccs };
        for &cc in ccs {
            let mut f = Func::new();
            let (inputs, _control) = f.entry();
            let cmp_ty = if cmp_float { Ty::F64 } else { Ty::I64 };
            let arm_ty = if arms_float { Ty::F64 } else { Ty::I64 };
            let a = f.load(cmp_ty, inputs, NONE, 1, 0);
            let b = f.load(cmp_ty, inputs, NONE, 1, 8);
            let t = f.load(arm_ty, inputs, NONE, 1, 16);
            let e = f.load(arm_ty, inputs, NONE, 1, 24);
            let r = f.select_any(cc, a, b, t, e);
            f.exit(&[(r, 0)]);
            f.validate().unwrap();
            for target in targets() {
                let (code, _) = compile_on(target, &f);
                for &x in &floats {
                    for &y in &floats {
                        let (ta, tb) = (7.25f64, -9.5f64);
                        let (xa, xb) = if cmp_float {
                            (x.to_bits(), y.to_bits())
                        } else {
                            (x.to_bits() >> 52, y.to_bits() >> 52)
                        };
                        let (av, bv) = if arms_float {
                            (ta.to_bits(), tb.to_bits())
                        } else {
                            (11u64, 22u64)
                        };
                        let mut inputs = [xa, xb, av, bv];
                        let mut control = [0u64; 2];
                        let mut area = [0u64; 1];
                        code.call(&mut inputs, &mut control, &mut area);
                        let holds = if cmp_float {
                            cc.eval_f(x, y)
                        } else {
                            cc.eval_i(xa as i64, xb as i64)
                        };
                        let want = if holds { av } else { bv };
                        assert_eq!(
                            area[0], want,
                            "{cc:?} cmp_float {cmp_float} arms_float {arms_float} on {x} {y} (avx {})",
                            target.avx
                        );
                    }
                }
            }
        }
    }
}

/// `SDiv` and `SRem` keep every live register, rdx included, whatever the
/// register pressure and wherever the operands and the result sit.
#[test]
fn division_preserves_every_live_value() {
    use oximond_mir::ir::Op as O;
    for pressure in [1usize, 4, 8, 12, 13, 14] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let vals: Vec<u32> = (0..14)
            .map(|k| f.load(Ty::I64, inputs, NONE, 1, 8 * k))
            .collect();
        // Divide pairs that are live together with `pressure` other values.
        let q = f.bin(O::SDiv, vals[0], vals[1]);
        let r = f.bin(O::SRem, vals[2], vals[3]);
        let q2 = f.bin(O::SDiv, q, vals[4]);
        let mut sum = f.bin(O::Add, q2, r);
        for v in &vals[..pressure.min(14)] {
            sum = f.bin(O::Add, sum, *v);
        }
        let mut snap: Vec<(u32, u32)> = vals
            .iter()
            .enumerate()
            .map(|(k, v)| (*v, k as u32))
            .collect();
        snap.push((q, 14));
        snap.push((r, 15));
        snap.push((q2, 16));
        snap.push((sum, 17));
        f.exit(&snap);
        f.validate().unwrap();
        for target in targets() {
            let (code, _) = compile_on(target, &f);
            for seed in [1i64, 7, -13, 1 << 40] {
                let mut data: Vec<u64> = (0..14)
                    .map(|k| (seed.wrapping_mul(2 * k + 3).wrapping_add(k * 1_000_003)) as u64)
                    .collect();
                // Divisors: positive, negative, never 0 or -1.
                for (slot, d) in [(1usize, 7i64), (3, -5), (4, 3 + seed.rem_euclid(5))] {
                    data[slot] = d as u64;
                }
                let want = data.clone();
                let mut inputs = data;
                let mut control = [0u64; 2];
                let mut area = [0u64; 18];
                code.call(&mut inputs, &mut control, &mut area);
                let (v0, v1) = (want[0] as i64, want[1] as i64);
                let q = v0 / v1;
                let r = (want[2] as i64) % (want[3] as i64);
                let q2 = q / want[4] as i64;
                let mut sum = q2.wrapping_add(r);
                for v in &want[..pressure.min(14)] {
                    sum = sum.wrapping_add(*v as i64);
                }
                assert_eq!(
                    &area[..14],
                    &want[..],
                    "live values (pressure {pressure}, avx {})",
                    target.avx
                );
                assert_eq!(
                    (
                        area[14] as i64,
                        area[15] as i64,
                        area[16] as i64,
                        area[17] as i64
                    ),
                    (q, r, q2, sum),
                    "pressure {pressure}"
                );
            }
        }
    }
}

/// Integer constants too wide for an immediate are read from the constant
/// pool by the sum, difference and bit operations, and a constant divisor of
/// any size by the division, which takes no immediate. A divisor that sits
/// in a slot is divided by where it is.
#[test]
fn wide_constants_and_constant_divisors_are_read_from_the_pool() {
    const WIDE: [i64; 4] = [0x9e37_79b1, 0xffff_ffff, -0x1_0000_0001, i64::MAX];
    for pressure in [0usize, 13] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let vals: Vec<u32> = (0..14)
            .map(|k| f.load(Ty::I64, inputs, NONE, 1, 8 * k))
            .collect();
        let x = vals[0];
        let wide: Vec<u32> = WIDE.iter().map(|k| f.iconst(*k)).collect();
        let sum = f.bin(Op::Add, x, wide[0]);
        let masked = f.bin(Op::And, sum, wide[1]);
        let less = f.bin(Op::Sub, masked, wide[2]);
        let mixed = f.bin(Op::Xor, less, wide[3]);
        let ored = f.bin(Op::Or, mixed, wide[0]);
        // The constant on the left of a difference is no memory operand.
        let from = f.bin(Op::Sub, wide[1], x);
        let (small, big) = (f.iconst(101), f.iconst(1_000_000_007));
        let r1 = f.bin(Op::SRem, x, small);
        let q1 = f.bin(Op::SDiv, ored, big);
        let r2 = f.bin(Op::URem, x, big);
        // With every register taken the divisor `vals[1]` is in a slot.
        let r3 = f.bin(Op::SRem, x, vals[1]);
        let mut acc = r3;
        for v in &vals[..pressure] {
            acc = f.bin(Op::Add, acc, *v);
        }
        let mut snap: Vec<(u32, u32)> = vals
            .iter()
            .enumerate()
            .map(|(k, v)| (*v, k as u32))
            .collect();
        for (k, v) in [ored, from, r1, q1, r2, r3, acc].into_iter().enumerate() {
            snap.push((v, 14 + k as u32));
        }
        f.exit(&snap);
        f.validate().unwrap();
        for target in targets() {
            let (code, stats) = compile_on(target, &f);
            assert!(stats.pool_operands >= 8, "{stats:?}");
            for x in [0i64, 1, -1, 12345, -98765, 1 << 40, i64::MAX, i64::MIN + 1] {
                let mut data: Vec<u64> = (0..14).map(|k| (x ^ (k * 77)) as u64).collect();
                data[0] = x as u64;
                data[1] = 7u64.wrapping_neg();
                let want = data.clone();
                let mut inputs = data;
                let mut control = [0u64; 2];
                let mut area = [0u64; 21];
                code.call(&mut inputs, &mut control, &mut area);
                let ored =
                    ((x.wrapping_add(WIDE[0]) & WIDE[1]).wrapping_sub(WIDE[2]) ^ WIDE[3]) | WIDE[0];
                let r3 = x % -7;
                let mut acc = r3;
                for v in &want[..pressure] {
                    acc = acc.wrapping_add(*v as i64);
                }
                let got: Vec<i64> = area[14..].iter().map(|v| *v as i64).collect();
                assert_eq!(&area[..14], &want[..], "live values, pressure {pressure}");
                assert_eq!(
                    got,
                    [
                        ored,
                        WIDE[1].wrapping_sub(x),
                        x % 101,
                        ored / 1_000_000_007,
                        ((x as u64) % 1_000_000_007) as i64,
                        r3,
                        acc
                    ],
                    "x {x}, pressure {pressure}, avx {}",
                    target.avx
                );
            }
        }
    }
}

/// `UDiv` and `URem` on values with the top bit set and clear.
#[test]
fn unsigned_division_matches_rust() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let a = f.load(Ty::I64, inputs, NONE, 1, 0);
    let b = f.load(Ty::I64, inputs, NONE, 1, 8);
    let keep = f.load(Ty::I64, inputs, NONE, 1, 16);
    let q = f.bin(Op::UDiv, a, b);
    let r = f.bin(Op::URem, a, b);
    f.exit(&[(q, 0), (r, 1), (keep, 2), (a, 3), (b, 4)]);
    f.validate().unwrap();
    for target in targets() {
        let (code, _) = compile_on(target, &f);
        for a in [0u64, 1, 7, u64::MAX, 1 << 63, (1 << 63) + 12345, 1 << 40] {
            for b in [1u64, 2, 3, 10, u64::MAX, 1 << 63, (1 << 63) + 1] {
                let mut inputs = [a, b, 0xdead_beef];
                let mut control = [0u64; 2];
                let mut area = [0u64; 5];
                code.call(&mut inputs, &mut control, &mut area);
                assert_eq!(area, [a / b, a % b, 0xdead_beef, a, b], "{a} {b}");
            }
        }
    }
}

/// `AddMem` adds a register or an immediate to a word in memory.
#[test]
fn add_to_memory_adds_a_register_or_an_immediate() {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let step = f.load(Ty::I64, inputs, NONE, 1, 0);
    let index = f.load(Ty::I64, inputs, NONE, 1, 8);
    let big = f.iconst(1 << 40);
    let small = f.iconst(5);
    for (op_index, value) in [step, big, small].into_iter().enumerate() {
        let mut inst = Inst::new(Op::AddMem, Ty::I64);
        inst.a = inputs;
        inst.b = index;
        inst.scale = 8;
        inst.imm = 16 + 8 * op_index as i64;
        inst.c = value;
        f.push(inst);
    }
    let mut inst = Inst::new(Op::AddMem, Ty::I64);
    inst.a = inputs;
    inst.imm = 16;
    inst.c = small;
    f.push(inst);
    f.exit(&[]);
    f.validate().unwrap();
    for target in targets() {
        let (code, _) = compile_on(target, &f);
        // Words 2.. are the counters; the index scales by 8 bytes.
        let mut inputs = [7u64, 0, 100, 200, 300, 400];
        let mut control = [0u64; 2];
        let mut area = [0u64; 1];
        code.call(&mut inputs, &mut control, &mut area);
        assert_eq!(inputs, [7, 0, 100 + 7 + 5, 200 + (1 << 40), 300 + 5, 400]);
    }
}

/// A sum or a product with a constant leaves its operand where it was when
/// the operand is read again (`lea` and the three-operand `imul`), and
/// overwrites it when it is not (`add` and `imul` in place); registers high
/// and low, a constant at the edge of 32 bits and a negative one.
#[test]
fn a_sum_or_product_with_a_constant_keeps_or_reuses_its_operand() {
    let consts: [i64; 6] = [1, 24, -7, 0x7fff_ffff, -0x8000_0000, 4096];
    // Enough live values that the operands land in many registers.
    let live = 12usize;
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let xs: Vec<u32> = (0..live)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * k as i32))
        .collect();
    let mut results = Vec::new();
    for c in &consts {
        for x in &xs[..live - 1] {
            let c = f.iconst(*c);
            let sum = f.bin(Op::Add, *x, c);
            let product = f.bin(Op::Mul, *x, c);
            results.push(sum);
            results.push(product);
        }
    }
    // The operands are read again at the end.
    results.extend(&xs);
    let vals: Vec<(u32, u32)> = results
        .iter()
        .enumerate()
        .map(|(k, v)| (*v, k as u32))
        .collect();
    f.exit(&vals);
    f.validate().unwrap();
    let words: Vec<u64> = (0..live)
        .map(|k| 0x1234_5678_9abc_def0u64.rotate_left(k as u32 * 5) ^ k as u64)
        .collect();
    for t in targets() {
        let (code, _) = compile_on(t, &f);
        let mut inputs: Vec<u64> = words.iter().flat_map(|w| [*w, 0]).collect();
        let mut control = [0u64; 4];
        let mut area = vec![0u64; results.len() + 2];
        let r = code.call(&mut inputs, &mut control, &mut area);
        assert_eq!(r, -1);
        let mut at = 0;
        for c in consts {
            for x in &words[..live - 1] {
                let x = *x as i64;
                assert_eq!(area[at] as i64, x.wrapping_add(c), "{x} + {c}");
                assert_eq!(area[at + 1] as i64, x.wrapping_mul(c), "{x} * {c}");
                at += 2;
            }
        }
        for (k, w) in words.iter().enumerate() {
            assert_eq!(area[at + k], *w, "operand {k} kept (avx {})", t.avx);
        }
    }
}

/// A flags-only guard (`scale` 1) after a sum with a constant reads the flags
/// of that sum even when its operand is read again, where a plain `lea` would
/// leave the flags of the `Sub` before it.
#[test]
fn a_flags_only_guard_reads_the_flags_of_a_sum_with_a_constant() {
    for k in [-5i64, 7, 0] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let a = f.load(Ty::I64, inputs, NONE, 1, 0);
        let b = f.load(Ty::I64, inputs, NONE, 1, 8);
        // Flags from a subtraction that disagree with the sum's.
        let big = f.bin(Op::Sub, b, a);
        let c = f.iconst(k);
        let y = f.bin(Op::Add, a, c);
        let zero = f.iconst(0);
        f.guard(Cc::Gt, y, zero, &[(y, 0), (a, 1), (big, 2)]);
        let at = f.insts.iter().rposition(|i| i.op == Op::Guard).unwrap();
        f.insts[at].scale = 1;
        // `a` and `big` are read again after the sum.
        f.exit(&[(y, 0), (a, 1), (big, 2)]);
        f.validate().unwrap();
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            for a in [3i64, 10, 5, 6, 8, -20, 0] {
                let b = 100i64;
                let mut inputs = [a as u64, b as u64];
                let mut control = [0u64; 2];
                let mut area = [0u64; 3];
                let r = code.call(&mut inputs, &mut control, &mut area);
                let sum = a.wrapping_add(k);
                // Guard exit is -1, the final exit -2.
                let want = if sum > 0 { -2 } else { -1 };
                assert_eq!(r, want, "{a} + {k} avx {}", t.avx);
                let got = area.map(|w| w as i64);
                assert_eq!(got, [sum, a, b.wrapping_sub(a)], "{a} + {k}");
            }
        }
    }
}

const SPECIALS: [f64; 14] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -2.5,
    1e300,
    -1e300,
    5e-324,
    -5e-324,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -f64::NAN,
];

/// `inputs[0..3]` are floats; the results are stored by an exit. Each value
/// is also read after the operation, so `dst` shares a register with an
/// operand in some of the encodings.
fn ternary_func(fused: bool) -> Func {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let a = f.load(Ty::F64, inputs, NONE, 1, 0);
    let b = f.load(Ty::F64, inputs, NONE, 1, 16);
    let c = f.load(Ty::F64, inputs, NONE, 1, 32);
    let sign = f.bin(Op::FCopySign, a, b);
    let same = f.bin(Op::FCopySign, a, a);
    let fused = if fused {
        f.fnma(a, b, c)
    } else {
        f.fconst(0.0)
    };
    // The inputs again, so the operations do not own their registers.
    f.exit(&[(sign, 0), (same, 1), (fused, 2), (a, 3), (b, 4), (c, 5)]);
    f
}

/// `FCopySign` and `FNma` give the bits Rust gives, for the values that
/// matter (zeros of both signs, infinities, NaNs, tiny and huge), with both
/// encodings (`FNma` where the CPU has FMA).
#[test]
fn copysign_and_fused_negated_multiply_add_match_rust() {
    // `mul_add` calls the C library's `fma`. Wine's returns 3.3e-24 for this
    // exact zero, so a reference that misses it is not checked against.
    let reference_fused = (-1.0f64).mul_add(-1e300, -1e300) == 0.0;
    for t in targets() {
        let f = ternary_func(t.fma);
        f.validate().unwrap();
        let (code, _) = compile_on(t, &f);
        for a in SPECIALS {
            for b in SPECIALS {
                for c in [0.0, -0.0, 3.0, -1e300, f64::NAN] {
                    let mut inputs = [a.to_bits(), 0, b.to_bits(), 0, c.to_bits(), 0];
                    let mut control = [0u64; 4];
                    let mut area = [0u64; 8];
                    assert_eq!(code.call(&mut inputs, &mut control, &mut area), -1);
                    assert_eq!(
                        area[0],
                        a.copysign(b).to_bits(),
                        "copysign {a} {b} (avx {})",
                        t.avx
                    );
                    assert_eq!(area[1], a.copysign(a).to_bits(), "copysign {a} {a}");
                    if t.fma && reference_fused {
                        let want = (-a).mul_add(b, c);
                        // The sign of a NaN is the hardware's to choose.
                        let got = f64::from_bits(area[2]);
                        if want.is_nan() {
                            assert!(got.is_nan(), "fnma {a} {b} {c}");
                        } else {
                            assert_eq!(got.to_bits(), want.to_bits(), "fnma {a} {b} {c}");
                        }
                    }
                    assert_eq!(
                        (area[3], area[4], area[5]),
                        (a.to_bits(), b.to_bits(), c.to_bits())
                    );
                }
            }
        }
    }
}

/// A conditional call on a float compare is made exactly when the compare
/// holds, NaN included, whatever the condition.
#[test]
fn a_conditional_call_on_floats_follows_the_compare() {
    extern "C" fn made(_: i64) -> i64 {
        1
    }
    for cc in [
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
    ] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let a = f.load(Ty::F64, inputs, NONE, 1, 0);
        let b = f.load(Ty::F64, inputs, NONE, 1, 16);
        let one = f.iconst(7);
        f.call_if_storing(
            cc,
            (a, b),
            made as *const () as usize,
            false,
            &[one],
            Some(8),
        );
        let after = f.load(Ty::I64, control, NONE, 1, 8);
        f.exit(&[(after, 0)]);
        f.validate().unwrap();
        for t in targets() {
            let (code, _) = compile_on(t, &f);
            for x in SPECIALS {
                for y in SPECIALS {
                    let mut inputs = [x.to_bits(), 0, y.to_bits(), 0];
                    let mut control = [0u64, 0x55];
                    let mut area = [0u64; 4];
                    code.call(&mut inputs, &mut control, &mut area);
                    let holds = cc.eval_f(x, y);
                    let want = if holds { 1 } else { 0x55 };
                    assert_eq!(area[0], want, "{cc:?} {x} {y} (avx {})", t.avx);
                }
            }
        }
    }
}
