//! `ExitArea`: code that stores an exit's state in line before a call, as
//! the trace backend does for a call of a host function, which reads that
//! state from the exit area. The callee must see every word the stores put
//! there (values in registers, in slots, constants, a cold value), in the
//! root, in a loop and in a side, and the run must match the evaluator.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]
#![allow(unsafe_code)]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{COLD, Cc, Func, Inst, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};
use std::cell::RefCell;

thread_local! {
    /// The words each call of [`record`] found in the exit area.
    static SEEN: RefCell<Vec<Vec<u64>>> = const { RefCell::new(Vec::new()) };
}

/// Copies the first `n` words of `area` aside and returns `n`: its result
/// does not depend on what it reads, so the evaluator, whose stores go to
/// copies of the buffers, gives the same one.
extern "C" fn record(area: i64, n: i64) -> i64 {
    let words = (0..n as usize)
        // SAFETY: the test sized the exit area for `n` words.
        .map(|k| unsafe { *(area as *const u64).add(k) })
        .collect();
    SEEN.with(|seen| seen.borrow_mut().push(words));
    n
}

fn record_at() -> usize {
    record as *const () as usize
}

fn inputs_of(ints: usize, floats: usize) -> Vec<Input> {
    let mut v = vec![Input::Int(3)];
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

/// The count input, then `ints` integers and `floats` floats.
fn load_inputs(f: &mut Func, inputs: Val, ints: usize, floats: usize) -> (Val, Vec<Val>) {
    let count = f.load(Ty::I64, inputs, NONE, 1, 8);
    let mut vals: Vec<Val> = (0..ints)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * (1 + k) as i32 + 8))
        .collect();
    vals.extend(
        (0..floats).map(|k| f.load(Ty::F64, inputs, NONE, 1, 16 * (1 + ints + k) as i32 + 8)),
    );
    (count, vals)
}

/// A float sum only snapshots and the in-line stores read (`COLD`).
fn cold_sum(f: &mut Func, a: Val, b: Val) -> Val {
    let dst = f.val(Ty::F64);
    let mut inst = Inst::new(Op::FAdd, Ty::F64);
    inst.dst = dst;
    inst.a = a;
    inst.b = b;
    inst.scale = COLD;
    f.push(inst);
    dst
}

/// Stores `vals` at slots `0..` of the exit area and calls [`record`] with
/// it, as a host call's spill does. Returns the area and the call's result.
fn spill_and_call(f: &mut Func, vals: &[Val]) -> (Val, Val) {
    let area = f.exit_area();
    for (k, v) in vals.iter().enumerate() {
        f.store(area, NONE, 1, 8 * k as i32, *v);
    }
    let n = f.iconst(vals.len() as i64);
    let r = f.call(record_at(), &[area, n], Some(Ty::I64));
    (area, r)
}

fn snapshot(vals: &[Val]) -> Vec<(Val, u32)> {
    vals.iter()
        .enumerate()
        .map(|(k, v)| (*v, k as u32))
        .collect()
}

/// The bits of the words the scenario's inputs give `vals` (the count left
/// out), in order.
fn input_words(sc: &Scenario, n: usize) -> Vec<u64> {
    (0..n).map(|k| sc.inputs[2 * (1 + k) + 1]).collect()
}

fn check(f: &Func, kinds: &[Input], what: &str, verify: &dyn Fn(&Scenario, &[Vec<u64>])) {
    f.validate().unwrap();
    for t in targets() {
        let (code, stats) = compile_on(t, f);
        let mut sc = Scenario::new(kinds, 10, f.exit_area_len().max(64), vec![0; 8]);
        SEEN.with(|seen| seen.borrow_mut().clear());
        let _ = sc.eval(f);
        let calls = SEEN.with(|seen| seen.borrow().len());
        if let Err(e) = check_native(f, &code, &mut sc) {
            panic!("{what} (avx {}): {e}; {stats:?}", t.avx);
        }
        // `check_native` ran the evaluator once more, then the native code:
        // the native calls are the last ones.
        let seen = SEEN.with(|seen| seen.borrow().clone());
        assert_eq!(seen.len(), 3 * calls, "{what}: as many calls natively");
        verify(&sc, &seen[2 * calls..]);
    }
}

/// Values of both classes, at every amount of register pressure, a
/// constant and a cold float stored before a call: the callee reads each
/// where the exit's stub would write it, and the values live after the call.
#[test]
fn the_state_stored_before_a_call_is_what_the_callee_reads() {
    for (ints, floats) in [(2, 1), (6, 4), (13, 13), (14, 15)] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let (_, mut vals) = load_inputs(&mut f, inputs, ints, floats);
        let loaded = vals.len();
        let constant = f.iconst(0x1234_5678_9abc);
        let small = f.iconst(-5);
        let cold = cold_sum(&mut f, vals[ints], vals[ints]);
        vals.extend([constant, small, cold]);
        let (area, r) = spill_and_call(&mut f, &vals);
        let mut after = vals.clone();
        after.push(r);
        after.push(area);
        f.exit(&snapshot(&after));
        check(
            &f,
            &inputs_of(ints, floats),
            &format!("pressure {ints}+{floats}"),
            &|sc, seen| {
                let words = &seen[0];
                assert_eq!(words[..loaded], input_words(sc, loaded)[..]);
                assert_eq!(words[loaded], 0x1234_5678_9abc);
                assert_eq!(words[loaded + 1] as i64, -5);
                let x = f64::from_bits(sc.inputs[2 * (1 + ints) + 1]);
                assert_eq!(words[loaded + 2], (x + x).to_bits());
                assert_eq!(sc.area[after.len() - 1], sc.area.as_ptr() as u64);
            },
        );
    }
}

/// The stores in a loop, before a call whose guard leaves through the same
/// state, with a loop-carried value among them; and in a side that a guard
/// before them enters, which stores its own state and calls too.
#[test]
fn the_state_is_stored_in_a_loop_and_in_a_side() {
    for (ints, floats) in [(3, 2), (12, 12)] {
        let mut f = Func::new();
        let (inputs, _control) = f.entry();
        let (count, vals) = load_inputs(&mut f, inputs, ints, floats);
        let zero = f.iconst(0);
        let first = f.header(&[count, zero]);
        let (left, acc) = (first, first + 1);
        let four = f.iconst(4);
        // A side for the first round of a long count.
        f.guard_side(Cc::Lt, left, four, 0, &[left, acc, vals[0], vals[ints]]);
        let mut state = vals.clone();
        state.push(left);
        state.push(acc);
        let (_, r) = spill_and_call(&mut f, &state);
        // The call's guard leaves through the state stored before it.
        f.guard(Cc::Ne, r, zero, &snapshot(&state));
        let acc2 = f.bin(Op::Add, acc, r);
        let acc3 = f.bin(Op::Add, acc2, left);
        let one = f.iconst(1);
        let left2 = f.bin(Op::Sub, left, one);
        f.guard(Cc::Gt, left2, zero, &snapshot(&[left2, acc3]));
        f.back(&[left2, acc3]);
        let (_, p) = f.begin_side(&[Ty::I64, Ty::I64, Ty::I64, Ty::F64]);
        let k = f.iconst(1000);
        let bumped = f.bin(Op::Add, p + 1, k);
        let side_state = [p, bumped, p + 2, p + 3];
        let (area, r) = spill_and_call(&mut f, &side_state);
        f.exit(&snapshot(&[p, bumped, p + 2, p + 3, r, area]));
        for start in [3, 9] {
            let mut kinds = inputs_of(ints, floats);
            kinds[0] = Input::Int(start);
            let n = ints + floats;
            check(
                &f,
                &kinds,
                &format!("loop {ints}+{floats} from {start}"),
                &|sc, seen| {
                    let words = input_words(sc, n);
                    if start < 4 {
                        // The loop runs `start` rounds, each call reads the round's
                        // state.
                        assert_eq!(seen.len(), start as usize);
                        let mut acc = 0u64;
                        for (round, call) in seen.iter().enumerate() {
                            assert_eq!(call[..n], words[..]);
                            let left = start as u64 - round as u64;
                            assert_eq!(call[n], left);
                            assert_eq!(call[n + 1], acc);
                            acc = acc.wrapping_add(call.len() as u64).wrapping_add(left);
                        }
                    } else {
                        // The side runs at once and calls once.
                        assert_eq!(seen.len(), 1);
                        assert_eq!(seen[0], vec![start as u64, 1000, words[0], words[ints]]);
                    }
                },
            );
        }
    }
}
