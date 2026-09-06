//! The forward skip (`If` and `EndIf`): values live across a block that may
//! be jumped over keep their places on both paths, join values take the
//! block's value or the skipped side's, blocks nest and hold calls and
//! conditional calls, and a block in a loop works under pressure. The native
//! code matches the evaluator on every scenario, with both encodings.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};
use oximond_mir::ra::Stats;

extern "C" fn answer(a: i64, b: i64) -> i64 {
    a.wrapping_mul(1000).wrapping_add(b)
}

extern "C" fn mix(a: i64, b: i64) -> i64 {
    a.wrapping_mul(31).wrapping_add(b) ^ (a >> 3)
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
    load_inputs_from(f, inputs, 0, ints, floats)
}

/// The integers and floats from input slot `first` on.
fn load_inputs_from(
    f: &mut Func,
    inputs: Val,
    first: usize,
    ints: usize,
    floats: usize,
) -> (Vec<Val>, Vec<Val>) {
    let is = (0..ints)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * (first + k) as i32 + 8))
        .collect();
    let fs = (0..floats)
        .map(|k| f.load(Ty::F64, inputs, NONE, 1, 16 * (first + ints + k) as i32 + 8))
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

/// Runs `f` on both encodings against the evaluator, once per scenario the
/// caller's `tweak` makes of the inputs. Returns the stats of the last compile.
fn check(f: &Func, kinds: &[Input], what: &str, tweaks: &[&dyn Fn(&mut Scenario)]) -> Stats {
    f.validate().unwrap();
    let mut last = Stats::default();
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        last = stats;
        for (n, tweak) in tweaks.iter().enumerate() {
            let mut sc = Scenario::new(kinds, 10, f.exit_area_len(), vec![0; 8]);
            sc.control = vec![10, answer as *const () as u64, 0x77, 0];
            tweak(&mut sc);
            if let Err(e) = check_native(f, &code, &mut sc) {
                panic!("{what}, scenario {n} (avx {}): {e}; {stats:?}", t.avx);
            }
        }
    }
    last
}

type Tweak = Box<dyn Fn(&mut Scenario)>;

/// The block runs when input 0 is below input 1, and not otherwise.
fn taken(sc: &mut Scenario) {
    sc.inputs[1] = 3;
    sc.inputs[3] = 5000;
}

fn skipped(sc: &mut Scenario) {
    sc.inputs[1] = 5000;
    sc.inputs[3] = 3;
}

/// `ints` integers and `floats` floats loaded first and read after a block
/// that holds `temps` products live at once: the block takes registers from
/// the values that live across it.
fn live_across(ints: usize, floats: usize, temps: usize) -> Func {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, ints, floats);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    let mut prods = Vec::new();
    let mut fprods = Vec::new();
    for k in 0..temps {
        let c = f.iconst(k as i64 + 3);
        prods.push(f.bin(Op::Mul, is[k % ints], c));
        let x = fs[k % floats];
        fprods.push(f.bin(Op::FMul, x, x));
    }
    let (mut sum, mut fsum) = (prods[0], fprods[0]);
    for k in 1..temps {
        sum = f.bin(Op::Add, sum, prods[k]);
        fsum = f.bin(Op::FAdd, fsum, fprods[k]);
    }
    let joined = f.end_if(open, &[(sum, is[0]), (fsum, fs[0])]);
    let mut total = joined;
    for v in &is {
        total = f.bin(Op::Add, total, *v);
    }
    let mut ftotal = joined + 1;
    for v in &fs {
        ftotal = f.bin(Op::FAdd, ftotal, *v);
    }
    let mut all = vec![total, ftotal, joined, joined + 1];
    all.extend(&is);
    all.extend(&fs);
    exit_all(&mut f, &all);
    f
}

/// Integers and floats live across a block, at every amount of pressure in the
/// block: a register the block takes is given back on the skipped path.
#[test]
fn values_live_across_a_skipped_block_survive() {
    let mut stubs = 0;
    for (ints, floats, temps) in [
        (2, 1, 1),
        (2, 2, 3),
        (6, 4, 6),
        (6, 4, 14),
        (10, 9, 14),
        (13, 13, 16),
        (13, 13, 3),
    ] {
        let f = live_across(ints, floats, temps);
        let mut kinds = inputs_of(ints, floats);
        kinds.truncate(ints + floats);
        let st = check(
            &f,
            &kinds,
            &format!("{ints}+{floats} with {temps} temporaries"),
            &[&taken, &skipped],
        );
        assert_eq!(st.ifs, 1);
        stubs += st.if_stubs;
    }
    assert!(stubs > 0, "some skip path needed a stub");
}

/// A block that takes no register a live value holds gives the skip branch
/// the join as its target, with no move.
#[test]
fn a_block_that_leaves_every_register_alone_is_skipped_by_one_branch() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let ptr = f.load(Ty::I64, inputs, NONE, 1, 8);
    let (is, fs) = load_inputs_from(&mut f, inputs, 1, 5, 3);
    let open = f.begin_if(Cc::Lt, is[1], is[2]);
    f.store(ptr, NONE, 1, 8, is[3]);
    f.end_if(open, &[]);
    let mut total = is[0];
    for v in &is {
        total = f.bin(Op::Add, total, *v);
    }
    let mut ftotal = fs[0];
    for v in &fs {
        ftotal = f.bin(Op::FAdd, ftotal, *v);
    }
    let mut all = vec![total, ftotal];
    all.extend(&is);
    exit_all(&mut f, &all);
    let mut kinds = vec![Input::DataPtr];
    kinds.extend(inputs_of(5, 3));
    let st = check(&f, &kinds, "store in a block", &[&|_| {}]);
    assert_eq!((st.ifs, st.if_stubs), (1, 0), "{st:?}");
}

/// Join values: a block's value or a value from before it, a constant on
/// either side, integers and floats, more of them than there are registers
/// (the rest live in slots), all read after the join.
#[test]
fn join_values_take_the_blocks_value_or_the_skipped_one() {
    for (n, fl) in [(1usize, 0usize), (3, 2), (8, 4), (15, 15), (20, 4)] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, 4, 2);
        let mut skipped_side = Vec::new();
        for k in 0..n {
            skipped_side.push(match k % 3 {
                0 => f.iconst(-5 - k as i64),
                1 => is[k % 4],
                _ => f.bin(Op::Sub, is[3], is[2]),
            });
        }
        let mut fskipped = Vec::new();
        for k in 0..fl {
            fskipped.push(if k % 2 == 0 {
                fs[1]
            } else {
                f.fconst(1.5 + k as f64)
            });
        }
        let open = f.begin_if(Cc::Lt, is[0], is[1]);
        let mut joins = Vec::new();
        for (k, &e) in skipped_side.iter().enumerate() {
            let one = f.iconst(7 + k as i64);
            let t = match k % 4 {
                0 => f.bin(Op::Add, is[k % 4], one),
                1 => f.bin(Op::Xor, is[2], one),
                2 => is[3],
                _ => one,
            };
            joins.push((t, e));
        }
        for (k, &e) in fskipped.iter().enumerate() {
            let x = f.fconst(1.5 + k as f64);
            let t = f.bin(Op::FMul, fs[k % 2], x);
            joins.push((t, e));
        }
        let first = f.end_if(open, &joins);
        let mut all: Vec<Val> = Vec::new();
        let (mut sum, mut fsum) = (is[0], fs[0]);
        for k in 0..joins.len() {
            let v = first + k as u32;
            if f.val_ty[v as usize] == Ty::I64 {
                sum = f.bin(Op::Add, sum, v);
            } else {
                fsum = f.bin(Op::FAdd, fsum, v);
            }
            all.push(v);
        }
        all.insert(0, fsum);
        all.insert(0, sum);
        exit_all(&mut f, &all);
        let kinds = inputs_of(4, 2);
        check(
            &f,
            &kinds,
            &format!("{n} integer and {fl} float join values"),
            &[&taken, &skipped],
        );
    }
}

/// A join value whose two sources are not read after the join is built in
/// its own register on both paths: the skip branch needs no stub.
#[test]
fn a_join_value_built_in_its_own_register_needs_no_stub() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 3, 0);
    let cell = f.new_cell();
    f.cell_store(cell, is[2]);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    // A cell load names no register but its result's.
    let t = f.cell_load(Ty::I64, cell);
    let j = f.end_if(open, &[(t, is[1])]);
    let r = f.bin(Op::Mul, j, j);
    exit_all(&mut f, &[r, j]);
    let st = check(&f, &inputs_of(3, 0), "coalesced join", &[&taken, &skipped]);
    assert_eq!((st.ifs, st.if_stubs), (1, 0), "{st:?}");
}

/// A comparison of floats (with NaN, which every ordered condition fails)
/// chooses between the paths as a guard on it does.
#[test]
fn a_skipped_block_on_a_float_comparison_follows_the_guards_rules() {
    for cc in [
        Cc::Lt,
        Cc::Le,
        Cc::Gt,
        Cc::Ge,
        Cc::Eq,
        Cc::Ne,
        Cc::Nlt,
        Cc::Nge,
    ] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, 2, 2);
        let open = f.begin_if(cc, fs[0], fs[1]);
        let t = f.bin(Op::FAdd, fs[0], fs[1]);
        let ti = f.bin(Op::Add, is[0], is[1]);
        let j = f.end_if(open, &[(t, fs[0]), (ti, is[1])]);
        exit_all(&mut f, &[j, j + 1, is[0]]);
        let set = |a: f64, b: f64| {
            move |sc: &mut Scenario| {
                sc.inputs[2 * 2 + 1] = a.to_bits();
                sc.inputs[2 * 3 + 1] = b.to_bits();
            }
        };
        let (lt, eq, gt, nan) = (
            set(1.0, 2.0),
            set(2.0, 2.0),
            set(3.0, 2.0),
            set(f64::NAN, 1.0),
        );
        check(
            &f,
            &inputs_of(2, 2),
            &format!("float {cc:?}"),
            &[&lt, &eq, &gt, &nan],
        );
    }
}

/// A block that holds a call, a conditional call whose result goes to the
/// control block, and a skipped block of its own, with values of both classes
/// live across all of it.
#[test]
fn a_skipped_block_holds_calls_and_nested_skips() {
    for (ints, floats) in [(2usize, 1usize), (6, 4), (11, 9), (13, 13)] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, ints, floats);
        let outer = f.begin_if(Cc::Lt, is[0], is[1]);
        let m = f.call(mix as *const () as usize, &[is[1], is[0]], Some(Ty::I64));
        f.call_if_storing(
            Cc::Lt,
            (is[0], m),
            answer as *const () as usize,
            false,
            &[is[0], is[1]],
            Some(16),
        );
        let stored = f.load(Ty::I64, control, NONE, 1, 16);
        let inner = f.begin_if(Cc::Gt, stored, is[1]);
        let deep = f.bin(Op::Xor, stored, m);
        f.call_if(
            Cc::Ne,
            (deep, is[1]),
            mix as *const () as usize,
            false,
            &[deep, is[0]],
        );
        let after_inner = f.end_if(inner, &[(deep, m)]);
        let boosted = f.bin(Op::FAdd, fs[0], fs[floats - 1]);
        let j = f.end_if(outer, &[(after_inner, is[1]), (boosted, fs[0])]);
        let mut total = j;
        for v in &is {
            total = f.bin(Op::Add, total, *v);
        }
        let mut ftotal = j + 1;
        for v in &fs {
            ftotal = f.bin(Op::FAdd, ftotal, *v);
        }
        let mut all = vec![total, ftotal];
        all.extend(&is);
        all.extend(&fs);
        exit_all(&mut f, &all);
        let kinds = inputs_of(ints, floats);
        // Both conditions each way: the stored answer is 1000 * a + b.
        let big = |sc: &mut Scenario| sc.inputs[1] = 4_000_000;
        let st = check(
            &f,
            &kinds,
            &format!("{ints}+{floats}"),
            &[&taken, &skipped, &big, &|sc| {
                taken(sc);
                sc.inputs[5] = (-9i64) as u64;
            }],
        );
        assert_eq!(st.ifs, 2);
    }
}

/// Skips nested three deep, each with join values read by the next level out.
#[test]
fn skips_nest_and_pass_their_join_values_outward() {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 6, 0);
    let a = f.begin_if(Cc::Lt, is[0], is[1]);
    let x = f.bin(Op::Add, is[2], is[3]);
    let b = f.begin_if(Cc::Ge, is[4], is[5]);
    let y = f.bin(Op::Mul, x, is[5]);
    let c = f.begin_if(Cc::Ne, y, is[0]);
    let z = f.bin(Op::Sub, y, is[4]);
    let jc = f.end_if(c, &[(z, y), (is[3], x)]);
    let w = f.bin(Op::Xor, jc, jc + 1);
    let jb = f.end_if(b, &[(w, x), (jc, is[2])]);
    let v = f.bin(Op::Add, jb, jb + 1);
    let ja = f.end_if(a, &[(v, is[0])]);
    let mut all = vec![ja];
    all.extend(&is);
    exit_all(&mut f, &all);
    let set = |v: [i64; 6]| {
        move |sc: &mut Scenario| {
            for (k, x) in v.iter().enumerate() {
                sc.inputs[2 * k + 1] = *x as u64;
            }
        }
    };
    let cases = [
        set([1, 2, 3, 4, 9, 5]),
        set([1, 2, 3, 4, 5, 9]),
        set([5, 2, 3, 4, 9, 5]),
        set([1, 2, 3, 4, 9, 0]),
        set([1, 2, 0, 0, 9, 5]),
        set([-4, 100, 7, 8, 9, 1]),
    ];
    let tweaks: Vec<&dyn Fn(&mut Scenario)> =
        cases.iter().map(|c| c as &dyn Fn(&mut Scenario)).collect();
    let st = check(&f, &inputs_of(6, 0), "three deep", &tweaks);
    assert_eq!(st.ifs, 3);
}

/// A block in the body of a loop: the counter and an accumulator go round the
/// loop through the join value, and invariants of every amount live across the
/// block and the back edge.
#[test]
fn a_skipped_block_in_a_loop_keeps_the_carried_and_invariant_values() {
    let (mut rebound, mut entry_moves) = (0, 0);
    for (ints, floats, temps) in [
        (2usize, 1usize, 1usize),
        (7, 5, 6),
        (11, 9, 14),
        (13, 13, 14),
    ] {
        let mut f = Func::new();
        let (inputs, _) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, ints, floats);
        let counter = is[0];
        let acc0 = f.iconst(0);
        let facc0 = f.fconst(0.0);
        let h = f.header(&[counter, acc0, facc0]);
        let (c, acc, facc) = (h, h + 1, h + 2);
        let zero = f.iconst(0);
        let mut snap: Vec<(Val, u32)> = vec![(c, 0), (acc, 1), (facc, 2)];
        for (k, v) in is.iter().chain(&fs).enumerate() {
            snap.push((*v, 3 + k as u32));
        }
        f.guard(Cc::Gt, c, zero, &snap);
        let one = f.iconst(1);
        let odd = f.bin(Op::And, c, one);
        let open = f.begin_if(Cc::Eq, odd, zero);
        let mut prods = Vec::new();
        let mut fprods = Vec::new();
        for k in 0..temps {
            let m = f.iconst(k as i64 + 3);
            prods.push(f.bin(Op::Mul, is[1 + k % (ints - 1)], m));
            let x = fs[k % floats];
            fprods.push(f.bin(Op::FMul, x, facc));
        }
        let (mut sum, mut fsum) = (acc, facc);
        for k in 0..temps {
            sum = f.bin(Op::Add, sum, prods[k]);
            fsum = f.bin(Op::FAdd, fsum, fprods[k]);
        }
        let j = f.end_if(open, &[(sum, acc), (fsum, facc)]);
        let step = f.bin(Op::Add, j, is[1]);
        let fstep = f.bin(Op::FAdd, j + 1, fs[0]);
        let next = f.bin(Op::Sub, c, one);
        f.back(&[next, step, fstep]);
        let kinds = inputs_of(ints, floats);
        let counts: Vec<Tweak> = [0i64, 1, 2, 5, 8]
            .into_iter()
            .map(|n| Box::new(move |sc: &mut Scenario| sc.inputs[1] = n as u64) as Box<_>)
            .collect();
        let tweaks: Vec<&dyn Fn(&mut Scenario)> = counts.iter().map(|b| &**b).collect();
        let st = check(
            &f,
            &kinds,
            &format!("{ints}+{floats} with {temps} temporaries"),
            &tweaks,
        );
        assert_eq!(st.ifs, 1);
        rebound += st.if_rebound;
        entry_moves += st.if_entry_moves;
    }
    assert!(
        rebound > 0 && entry_moves > 0,
        "a block that took the registers of values live across it left them in place above the \
         skip: {rebound} rebound, {entry_moves} moves at the start of the block"
    );
}

/// One step of the chain of [`a_chain_of_skips_ends_in_a_conditional_call`]:
/// a skip that runs while `state` is zero, which makes an answer from the
/// `k`th value (zero when it has none) and goes on to the next step inside
/// its own block. Returns the state after all of them.
fn chain_step(f: &mut Func, is: &[Val], (k, depth): (usize, usize), state: Val) -> Val {
    if k == depth {
        return state;
    }
    let zero = f.iconst(0);
    let open = f.begin_if(Cc::Eq, state, zero);
    let by = f.iconst(k as i64 + 3);
    let m = f.bin(Op::Mul, is[k % is.len()], by);
    let mask = f.iconst(7);
    let low = f.bin(Op::And, m, mask);
    let one = f.iconst(1);
    let made = f.bin(Op::Or, m, one);
    let found = f.select(Cc::Eq, low, zero, made, zero);
    let after = chain_step(f, is, (k + 1, depth), found);
    f.end_if(open, &[(after, state)])
}

/// The shape of the string-index probe: a chain of skips, each nested in the
/// one before and run only while no step has answered, the state passed out
/// as a join value, and behind the chain a skip that holds a conditional call
/// for when none did. Integers and floats live across all of it, in a loop and
/// out of one.
#[test]
fn a_chain_of_skips_ends_in_a_conditional_call() {
    for (depth, ints, floats, looped) in [
        (1usize, 3usize, 1usize, false),
        (4, 6, 4, false),
        (9, 9, 6, false),
        (4, 6, 4, true),
        (9, 11, 9, true),
        (12, 13, 13, true),
    ] {
        let mut f = Func::new();
        let (inputs, control) = f.entry();
        let (is, fs) = load_inputs(&mut f, inputs, ints, floats);
        let zero = f.iconst(0);
        let facc0 = f.fconst(0.0);
        let counter = is[0];
        let (h, facc) = if looped {
            let h = f.header(&[counter, zero, facc0]);
            let mut snap: Vec<(Val, u32)> = vec![(h, 0), (h + 1, 1), (h + 2, 2)];
            for (k, v) in is.iter().chain(&fs).enumerate() {
                snap.push((*v, 3 + k as u32));
            }
            f.guard(Cc::Gt, h, zero, &snap);
            (h, h + 2)
        } else {
            (counter, facc0)
        };
        let seed = f.bin(Op::Add, h, is[1]);
        let start = seed_of(&mut f, seed, zero);
        let state = chain_step(&mut f, &is, (0, depth), start);
        let open = f.begin_if(Cc::Eq, state, zero);
        f.call_if_storing(
            Cc::Eq,
            (state, zero),
            answer as *const () as usize,
            false,
            &[seed, is[1]],
            Some(16),
        );
        let late = f.load(Ty::I64, control, NONE, 1, 16);
        let state = f.end_if(open, &[(late, state)]);
        let boosted = f.bin(Op::FAdd, facc, fs[0]);
        let mut total = state;
        for v in &is {
            total = f.bin(Op::Add, total, *v);
        }
        let mut ftotal = boosted;
        for v in &fs {
            ftotal = f.bin(Op::FAdd, ftotal, *v);
        }
        if looped {
            let one = f.iconst(1);
            let next = f.bin(Op::Sub, h, one);
            f.back(&[next, total, ftotal]);
        } else {
            let mut all = vec![total, ftotal, state];
            all.extend(&is);
            exit_all(&mut f, &all);
        }
        let kinds = inputs_of(ints, floats);
        // The first steps answer for some inputs and none does for others.
        let counts: Vec<Tweak> = [0i64, 1, 2, 3, 6]
            .into_iter()
            .map(|n| {
                Box::new(move |sc: &mut Scenario| {
                    sc.inputs[1] = n as u64;
                    sc.inputs[3] = (n * 5 + 1) as u64;
                }) as Tweak
            })
            .collect();
        let tweaks: Vec<&dyn Fn(&mut Scenario)> = counts.iter().map(|b| &**b).collect();
        let st = check(
            &f,
            &kinds,
            &format!("depth {depth}, {ints}+{floats}, loop {looped}"),
            &tweaks,
        );
        assert_eq!(st.ifs as usize, depth + 1);
    }
}

/// The state a chain starts from: zero when the seed has no low bits.
fn seed_of(f: &mut Func, seed: Val, zero: Val) -> Val {
    let mask = f.iconst(3);
    let low = f.bin(Op::And, seed, mask);
    f.select(Cc::Eq, low, zero, zero, seed)
}

/// The structure is checked: what a block defines is out of scope after it,
/// a block cannot hold a loop or end the region, the skipped side's values
/// come from before the block.
#[test]
fn the_validator_rejects_a_misplaced_skip() {
    // A value of the block used after it.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 2, 0);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    let t = f.bin(Op::Add, is[0], is[1]);
    f.end_if(open, &[]);
    exit_all(&mut f, &[t]);
    assert!(f.validate().unwrap_err().contains("undefined value"));

    // The skipped side names a value the block defines.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 2, 0);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    let t = f.bin(Op::Add, is[0], is[1]);
    f.end_if(open, &[(t, t)]);
    exit_all(&mut f, &[is[0]]);
    assert!(f.validate().unwrap_err().contains("undefined value"));

    // A block that leaves.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 2, 0);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    exit_all(&mut f, &[is[0]]);
    f.end_if(open, &[]);
    exit_all(&mut f, &[is[0]]);
    assert!(f.validate().unwrap_err().contains("inside an if"));

    // A loop in a block.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 2, 0);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    f.header(&[is[0]]);
    f.end_if(open, &[]);
    exit_all(&mut f, &[is[0]]);
    assert!(f.validate().unwrap_err().contains("inside an if"));

    // Mixed types in a join.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, fs) = load_inputs(&mut f, inputs, 2, 1);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    f.end_if(open, &[(is[0], fs[0])]);
    exit_all(&mut f, &[is[0]]);
    assert!(f.validate().unwrap_err().contains("type mismatch"));

    // A guard that reuses the flags right after a join has none to reuse.
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let (is, _) = load_inputs(&mut f, inputs, 2, 0);
    let open = f.begin_if(Cc::Lt, is[0], is[1]);
    let sum = f.bin(Op::Add, is[0], is[1]);
    let j = f.end_if(open, &[(sum, is[0])]);
    let zero = f.iconst(0);
    f.guard(Cc::Ne, j, zero, &[(j, 0)]);
    exit_all(&mut f, &[j]);
    assert!(f.validate().is_ok());
    let last = f.insts.iter().position(|i| i.op == Op::Guard).unwrap();
    f.insts[last].scale = 1;
    assert!(f.validate().unwrap_err().contains("flags of a skip"));

    // A well-formed one passes.
    assert!(live_across(3, 2, 2).validate().is_ok());
}
