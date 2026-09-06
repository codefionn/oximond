//! The allocator's instruction-saving forms: a back edge with no moves
//! folded into the guard before it, a guard on a byte load compared in
//! memory, and a sum of two live registers computed into a third. Each form
//! matches the evaluator, and each shape that must not take the form is
//! compiled the plain way.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, Inst, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};
use oximond_mir::ra::Stats;

fn check(f: &Func, kinds: &[Input], fuel: i64, data: &[u64], what: &str) -> Stats {
    f.validate().unwrap();
    let mut last = Stats::default();
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        let mut sc = Scenario::new(kinds, fuel, f.exit_area_len(), data.to_vec());
        if let Err(e) = check_native(f, &code, &mut sc) {
            panic!("{what}: fuel={fuel} avx={}: {e}; {stats:?}", t.avx);
        }
        last = stats;
    }
    last
}

fn load8(f: &mut Func, base: Val, index: Val, scale: u8, disp: i32) -> Val {
    let dst = f.val(Ty::I64);
    let mut i = Inst::new(Op::Load8, Ty::I64);
    i.dst = dst;
    i.a = base;
    i.b = index;
    i.scale = scale;
    i.imm = i64::from(disp);
    f.push(i);
    dst
}

/// How the back edge of [`counting`] hands on its values.
#[derive(Clone, Copy, PartialEq)]
enum Edge {
    /// Each parameter gets its own next value: nothing moves.
    Plain,
    /// One value goes to two parameters: a copy into a free register.
    Twin,
    /// A parameter gets a constant, which is no copy of a register.
    Constant,
}

/// `x` climbs by `step` to `limit` and the poll guard closes the iteration.
/// With `stale` the exit of the poll guard stores the `x` and `y` the
/// iteration started with, which the loop reads nowhere else.
fn counting(edge: Edge, stale: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let limit = f.load(Ty::I64, inputs, NONE, 1, 8);
    let step = f.load(Ty::I64, inputs, NONE, 1, 24);
    let x0 = f.load(Ty::I64, inputs, NONE, 1, 40);
    let y0 = f.load(Ty::I64, inputs, NONE, 1, 56);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let h = f.header(&[x0, y0, fuel0]);
    let (x, y, fuel) = (h, h + 1, h + 2);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let x2 = f.bin(Op::Add, x, step);
    let y2 = f.bin(Op::Add, y, x2);
    f.guard(Cc::Le, x2, limit, &[(x2, 0), (y2, 1), (fuel, CONTROL_SLOT)]);
    let fuel2 = f.bin(Op::Sub, fuel, one);
    if stale {
        f.guard(
            Cc::Gt,
            fuel2,
            zero,
            &[(x, 0), (y, 1), (x2, 2), (y2, 3), (fuel2, CONTROL_SLOT)],
        );
    } else {
        f.guard(
            Cc::Gt,
            fuel2,
            zero,
            &[(x2, 0), (y2, 1), (fuel2, CONTROL_SLOT)],
        );
    }
    match edge {
        Edge::Plain => f.back(&[x2, y2, fuel2]),
        Edge::Twin => f.back(&[x2, x2, fuel2]),
        Edge::Constant => f.back(&[x2, one, fuel2]),
    }
    f
}

#[test]
fn a_back_edge_without_moves_is_the_closing_guard() {
    let f = counting(Edge::Plain, false);
    for fuel in [1, 2, 3, 50, 5000] {
        for (limit, step, x, y) in [
            (100, 7, 0, 3),
            (100, 1, 99, 0),
            (5, 100, 0, 1),
            (-3, 2, -90, 9),
        ] {
            let kinds = [
                Input::Int(limit),
                Input::Int(step),
                Input::Int(x),
                Input::Int(y),
            ];
            let st = check(&f, &kinds, fuel, &[0; 4], "rotated loop");
            assert_eq!((st.rotated_backs, st.backedge_moves), (1, 0), "{st:?}");
        }
    }
}

#[test]
fn a_back_edge_that_copies_into_free_registers_copies_before_the_closing_guard() {
    // The copy stands between the instruction that sets the flags and the
    // branch to the header, and the exit of the guard runs after it: with
    // `stale` it stores values the copy must not have overwritten.
    for stale in [false, true] {
        let f = counting(Edge::Twin, stale);
        for fuel in [1, 2, 3, 50, 5000] {
            for (limit, step, x, y) in [(100, 7, 0, 3), (1000, 1, 99, 0), (5, 100, 0, 1)] {
                let kinds = [
                    Input::Int(limit),
                    Input::Int(step),
                    Input::Int(x),
                    Input::Int(y),
                ];
                let st = check(&f, &kinds, fuel, &[0; 8], "twin loop");
                assert_eq!((st.rotated_backs, st.rotated_with_moves), (1, 1), "{st:?}");
                assert!(st.backedge_moves > 0, "{st:?}");
            }
        }
    }
}

#[test]
fn a_back_edge_that_sets_a_constant_keeps_its_jump() {
    for stale in [false, true] {
        let f = counting(Edge::Constant, stale);
        for fuel in [1, 2, 3, 50, 5000] {
            for (limit, step, x, y) in [(100, 7, 0, 3), (1000, 1, 99, 0), (5, 100, 0, 1)] {
                let kinds = [
                    Input::Int(limit),
                    Input::Int(step),
                    Input::Int(x),
                    Input::Int(y),
                ];
                let st = check(&f, &kinds, fuel, &[0; 8], "constant loop");
                assert_eq!(st.rotated_backs, 0, "{st:?}");
                assert!(st.backedge_moves > 0, "{st:?}");
            }
        }
    }
}

#[test]
fn a_guard_compares_with_a_wide_constant_in_the_pool() {
    // `x + 2^53 <= 2^54` unsigned, the test that an integer is exact as a
    // float: neither constant is an immediate.
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let x = f.load(Ty::I64, inputs, NONE, 1, 8);
    let y = f.load(Ty::I64, inputs, NONE, 1, 24);
    let (bias, limit) = (f.iconst(1 << 53), f.iconst(1 << 54));
    let biased = f.bin(Op::Add, x, bias);
    f.guard(Cc::Ule, biased, limit, &[(x, 0), (y, 1)]);
    let wide = f.iconst(-0x1234_5678_9abc);
    f.guard(Cc::Gt, y, wide, &[(biased, 0), (x, 1)]);
    let sum = f.bin(Op::Add, biased, y);
    f.exit(&[(sum, 0), (x, 1), (y, 2)]);
    let edge = 1i64 << 53;
    for x in [
        0,
        7,
        -7,
        edge,
        edge + 1,
        -edge,
        -edge - 1,
        i64::MAX,
        i64::MIN,
    ] {
        for y in [0, -0x1234_5678_9abc, -0x1234_5678_9abb, i64::MIN] {
            let st = check(
                &f,
                &[Input::Int(x), Input::Int(y)],
                5,
                &[0; 4],
                "wide compare",
            );
            assert_eq!(st.pool_operands, 3, "{st:?}");
            assert_eq!(st.remats, 0, "{st:?}");
        }
    }
}

/// `x` climbs by `step`; a block the loop skips while `x` is at most `limit`
/// holds the guards that end it. With `joined` the block hands a value to the
/// code after it, and with `trailing` an instruction stands between the block
/// and the back edge.
fn skipping(joined: bool, trailing: bool) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let limit = f.load(Ty::I64, inputs, NONE, 1, 8);
    let step = f.load(Ty::I64, inputs, NONE, 1, 24);
    let x0 = f.load(Ty::I64, inputs, NONE, 1, 40);
    let y0 = f.load(Ty::I64, inputs, NONE, 1, 56);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let h = f.header(&[x0, y0, fuel0]);
    let (x, y, fuel) = (h, h + 1, h + 2);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let y1 = f.bin(Op::Add, y, x);
    let x2 = f.bin(Op::Add, x, step);
    let fuel2 = f.bin(Op::Sub, fuel, one);
    let open = f.begin_if(Cc::Gt, x2, limit);
    let before = f.bin(Op::Sub, x2, step);
    f.guard(
        Cc::Gt,
        fuel2,
        zero,
        &[(before, 0), (y1, 1), (fuel2, CONTROL_SLOT)],
    );
    f.guard(
        Cc::Le,
        x2,
        limit,
        &[(x2, 0), (y1, 1), (fuel2, CONTROL_SLOT)],
    );
    let y2 = if joined {
        f.end_if(open, &[(before, y1)])
    } else {
        f.end_if(open, &[]);
        y1
    };
    let y2 = if trailing {
        f.bin(Op::Add, y2, one)
    } else {
        y2
    };
    f.back(&[x2, y2, fuel2]);
    f
}

#[test]
fn a_skip_that_ends_at_the_back_edge_branches_to_the_header() {
    for (joined, trailing, rotated) in [(false, false, 1), (true, false, 0), (false, true, 0)] {
        let f = skipping(joined, trailing);
        for fuel in [1, 2, 3, 50, 5000] {
            for (limit, step, x, y) in [
                (100, 7, 0, 3),
                (100, 1, 99, 0),
                (5, 100, 0, 1),
                (-3, 2, -90, 9),
            ] {
                let kinds = [
                    Input::Int(limit),
                    Input::Int(step),
                    Input::Int(x),
                    Input::Int(y),
                ];
                let st = check(&f, &kinds, fuel, &[0; 4], "skipping loop");
                assert_eq!(st.rotated_ifs, rotated, "{joined} {trailing} {st:?}");
            }
        }
    }
}

/// What the loop of [`tags`] does with the byte it loads.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Byte {
    /// Guarded against a small constant and read nowhere else.
    Guarded,
    /// Named by the guard's snapshot.
    InSnapshot,
    /// Read again after the guard.
    ReadAgain,
    /// Guarded against a constant that does not fit a signed byte.
    Wide,
    /// A store sits between the load and the guard.
    StoreBetween,
}

/// Walks `data` in steps of one word while the low byte of each word is the
/// tag (2, or 200 for `Wide`), summing the words.
fn tags(kind: Byte) -> Func {
    let mut f = Func::new();
    let (inputs, control) = f.entry();
    let data = f.load(Ty::I64, inputs, NONE, 1, 8);
    let n = f.load(Ty::I64, inputs, NONE, 1, 24);
    let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
    let (zero, one) = (f.iconst(0), f.iconst(1));
    let h = f.header(&[zero, zero, fuel0]);
    let (i, sum, fuel) = (h, h + 1, h + 2);
    let tag = load8(&mut f, data, i, 8, 0);
    if kind == Byte::StoreBetween {
        f.store(data, i, 8, 0, sum);
    }
    let want = f.iconst(if kind == Byte::Wide { 200 } else { 2 });
    let mut snap = vec![(i, 0), (sum, 1), (fuel, CONTROL_SLOT)];
    if kind == Byte::InSnapshot {
        snap.push((tag, 2));
    }
    f.guard(Cc::Eq, tag, want, &snap);
    let word = f.load(Ty::I64, data, i, 8, 0);
    let mut sum2 = f.bin(Op::Add, sum, word);
    if kind == Byte::ReadAgain {
        sum2 = f.bin(Op::Add, sum2, tag);
    }
    let i2 = f.bin(Op::Add, i, one);
    f.guard(Cc::Lt, i2, n, &[(i2, 0), (sum2, 1), (fuel, CONTROL_SLOT)]);
    let fuel2 = f.bin(Op::Sub, fuel, one);
    f.guard(
        Cc::Gt,
        fuel2,
        zero,
        &[(i2, 0), (sum2, 1), (fuel2, CONTROL_SLOT)],
    );
    f.back(&[i2, sum2, fuel2]);
    f
}

#[test]
fn a_guard_on_a_byte_load_compares_in_memory_only_when_nothing_else_reads_it() {
    for kind in [
        Byte::Guarded,
        Byte::InSnapshot,
        Byte::ReadAgain,
        Byte::Wide,
        Byte::StoreBetween,
    ] {
        let f = tags(kind);
        let tag = if kind == Byte::Wide { 200 } else { 2 };
        // Every word tagged, a word with another tag in the middle, a tag
        // that differs above the low byte only (still a match), and a first
        // word that fails.
        let all: Vec<u64> = (0..12).map(|k| tag + 0x100 * (k + 1)).collect();
        let mut broken = all.clone();
        broken[7] = 3 + 0x900;
        let mut first = all.clone();
        first[0] = 0x8000_0000_0000_0000 + tag + 1;
        for data in [&all, &broken, &first] {
            for fuel in [1, 4, 100] {
                let kinds = [Input::DataPtr, Input::Int(data.len() as i64)];
                let st = check(&f, &kinds, fuel, data, "tag walk");
                let fused = u32::from(kind == Byte::Guarded);
                assert_eq!(st.byte_guards, fused, "{kind:?}: {st:?}");
            }
        }
    }
}

/// `c = a + b` where `a` and `b` are both read after the sum, with enough
/// other live values that the registers run out for the larger `extra`.
#[test]
fn a_sum_of_two_values_that_live_on_keeps_both() {
    for extra in [0usize, 4, 13] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let a0 = f.load(Ty::I64, inputs, NONE, 1, 8);
        let b0 = f.load(Ty::I64, inputs, NONE, 1, 24);
        let more: Vec<Val> = (0..extra)
            .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * (2 + k as i32) + 8))
            .collect();
        let fuel0 = f.load(Ty::I64, control, NONE, 1, 0);
        let h = f.header(&[a0, b0, fuel0]);
        let (a, b, fuel) = (h, h + 1, h + 2);
        let (one, zero) = (f.iconst(1), f.iconst(0));
        let c = f.bin(Op::Add, a, b);
        // The sum of a value with itself, and a sum whose operand dies.
        let twice = f.bin(Op::Add, c, c);
        let mut d = f.bin(Op::Add, twice, a);
        for v in &more {
            d = f.bin(Op::Xor, d, *v);
        }
        let b2 = f.bin(Op::Sub, b, a);
        let fuel2 = f.bin(Op::Sub, fuel, one);
        let mut snap = vec![(a, 0), (b, 1), (c, 2), (d, 3)];
        for (k, v) in more.iter().enumerate() {
            snap.push((*v, 4 + k as u32));
        }
        snap.push((fuel2, CONTROL_SLOT));
        f.guard(Cc::Gt, fuel2, zero, &snap);
        f.back(&[c, b2, fuel2]);
        let mut kinds = vec![Input::Int(0x1234_5678_9abc), Input::Int(-77)];
        for k in 0..extra {
            kinds.push(Input::Int(19 * (k as i64 + 2)));
        }
        for fuel in [1, 2, 9, 300] {
            check(&f, &kinds, fuel, &[0; 4], "live sum");
        }
    }
}

/// A loop over `data[k]` for `k` from `n - 1` down to 0 that reads each word
/// twice, with a load for each read: one for a product, one for a
/// difference. `shape` picks what keeps a load from being read in place.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Between {
    /// Nothing: both loads are read in place.
    Nothing,
    /// A store to the array, which a load may not be moved past.
    Store,
    /// The exit of the loop's guard stores the loaded word.
    Snapshot,
    /// The loaded word is read twice.
    Twice,
}

fn float_sum(shape: Between) -> Func {
    let mut f = Func::new();
    let (inputs, _control) = f.entry();
    let data = f.load(Ty::I64, inputs, NONE, 1, 8);
    let n = f.load(Ty::I64, inputs, NONE, 1, 24);
    let acc0 = f.load(Ty::F64, inputs, NONE, 1, 40);
    let h = f.header(&[n, acc0]);
    let (k, acc) = (h, h + 1);
    let (one, zero) = (f.iconst(1), f.iconst(0));
    let next = f.bin(Op::Sub, k, one);
    let word = f.load(Ty::F64, data, next, 8, 0);
    if shape == Between::Store {
        // Over the word just loaded: read after it, the product would take
        // the stored value.
        f.store(data, next, 8, 0, acc);
    }
    // The loaded word is the first operand of a product: it is commuted.
    let product = f.bin(Op::FMul, word, acc);
    let below = f.load(Ty::F64, data, next, 8, 0);
    // The second operand of a difference.
    let sum = f.bin(Op::FSub, acc, below);
    let sum = f.bin(Op::FAdd, sum, product);
    let sum = if shape == Between::Twice {
        f.bin(Op::FDiv, sum, below)
    } else {
        sum
    };
    if shape == Between::Snapshot {
        f.guard(
            Cc::Gt,
            next,
            zero,
            &[(next, 0), (sum, 1), (word, 2), (below, 3)],
        );
    } else {
        f.guard(Cc::Gt, next, zero, &[(next, 0), (sum, 1)]);
    }
    f.back(&[next, sum]);
    f
}

#[test]
fn a_float_load_with_one_reader_is_read_in_place() {
    let data: Vec<u64> = [1.5f64, -2.25, 4.0, 0.125, 9.0, -7.5, 3.0, 2.0, 1.0, 0.5]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    for shape in [
        Between::Nothing,
        Between::Store,
        Between::Snapshot,
        Between::Twice,
    ] {
        let f = float_sum(shape);
        for n in [1, 2, 5, 7] {
            let kinds = [Input::DataPtr, Input::Int(n), Input::Float(0.75)];
            let st = check(&f, &kinds, 50, &data, "fused load");
            let fused = match shape {
                Between::Nothing => 2,
                // The load before the store stays where it is.
                Between::Store => 1,
                // What an exit stores is loaded, and so is what two
                // operations read.
                Between::Snapshot => 0,
                Between::Twice => 1,
            };
            assert_eq!(st.fused_loads, fused, "{shape:?}: {st:?}");
        }
    }
}
