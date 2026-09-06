//! Exits that store the same values from the same places share the stores of
//! one stub: the second sets its own exit code and jumps there. The native
//! code matches the evaluator on every exit, with both encodings, for values
//! in registers, in spill slots and constants of every width.
#![cfg(all(any(target_os = "linux", windows), target_arch = "x86_64"))]

mod common;

use common::{compile_on, targets};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, NONE, Op, Ty, Val};
use oximond_mir::kernels::{Input, Scenario, check_native};
use oximond_mir::ra::Stats;

/// A loop over `live` integers and two floats, all carried around the back
/// edge, with two guards in a row whose snapshots store the same values: the
/// counter reaching zero, and the counter reaching 7. With `control`, the
/// snapshot also stores the counter and a wide constant to the control block.
fn two_guards_one_snapshot(live: usize, control: bool) -> Func {
    let mut f = Func::new();
    let (inputs, _) = f.entry();
    let mut init: Vec<Val> = (0..=live)
        .map(|k| f.load(Ty::I64, inputs, NONE, 1, 16 * k as i32 + 8))
        .collect();
    for k in 0..2 {
        let at = 16 * (1 + live + k) as i32 + 8;
        init.push(f.load(Ty::F64, inputs, NONE, 1, at));
    }
    let first = f.header(&init);
    let carried: Vec<Val> = (0..init.len() as u32).map(|k| first + k).collect();
    let counter = carried[0];
    let wide = f.iconst(0x1234_5678_9abc_def0);
    let narrow = f.iconst(-5);
    let half = f.fconst(0.5);
    let mut snap: Vec<(Val, u32)> = carried
        .iter()
        .enumerate()
        .map(|(k, &v)| (v, k as u32))
        .collect();
    let n = snap.len() as u32;
    snap.extend([(wide, n), (narrow, n + 1), (half, n + 2)]);
    if control {
        snap.extend([(counter, CONTROL_SLOT | 1), (wide, CONTROL_SLOT | 2)]);
    }
    let zero = f.iconst(0);
    f.guard(Cc::Gt, counter, zero, &snap);
    let seven = f.iconst(7);
    f.guard(Cc::Ne, counter, seven, &snap);
    let one = f.iconst(1);
    let mut next = vec![f.bin(Op::Sub, counter, one)];
    for &v in &carried[1..carried.len() - 2] {
        next.push(f.bin(Op::Add, v, counter));
    }
    let x = carried[carried.len() - 2];
    let y = carried[carried.len() - 1];
    next.push(f.bin(Op::FAdd, x, y));
    next.push(f.bin(Op::FMul, y, half));
    f.back(&next);
    f
}

fn check(live: usize, control: bool) -> Stats {
    let f = two_guards_one_snapshot(live, control);
    f.validate().unwrap();
    let mut kinds = vec![Input::Int(3)];
    kinds.extend((0..live).map(|k| Input::Int(k as i64 * 3 - 4)));
    kinds.extend([Input::Float(1.25), Input::Float(-3.5)]);
    let mut last = Stats::default();
    for t in targets() {
        let (code, stats) = compile_on(t, &f);
        last = stats;
        let mut results = Vec::new();
        // The first guard exits for a count of 3, the second for 10.
        for count in [3, 10] {
            let mut sc = Scenario::new(&kinds, 100, f.exit_area_len(), vec![0; 8]);
            sc.inputs[1] = count;
            sc.control.resize(4, 0);
            match check_native(&f, &code, &mut sc) {
                Ok(code) => results.push(code),
                Err(e) => panic!("live {live}, count {count} (avx {}): {e}; {stats:?}", t.avx),
            }
        }
        assert_ne!(results[0], results[1], "the exits return their own codes");
    }
    last
}

#[test]
fn a_second_exit_with_the_same_stores_jumps_to_the_first() {
    let st = check(4, false);
    assert_eq!(st.shared_stubs, 1, "{st:?}");
}

#[test]
fn shared_stores_write_the_control_block() {
    let st = check(4, true);
    assert_eq!(st.shared_stubs, 1, "{st:?}");
}

#[test]
fn shared_stores_copy_spilled_values() {
    let st = check(24, false);
    assert_eq!(st.shared_stubs, 1, "{st:?}");
    assert!(st.slots > 0, "nothing spilled: {st:?}");
}
