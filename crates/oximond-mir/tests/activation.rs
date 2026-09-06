//! Functions that return and call themselves, as a function trace's
//! activation does: `Entry` with more than three arguments, `Ret` with two
//! results, `Call` with `CALL_REL` to its own start and to code placed
//! before the caller, exits that return a value (`RETURN_SLOT`) and copy
//! words through the control block (`INDIRECT_SLOT`). Native code must match
//! the evaluator in its results and in every word it writes.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(unsafe_code)]

mod common;

use common::{compile_on, targets};
use oximond_mir::eval::{Mem, run_function};
use oximond_mir::ir::{CONTROL_SLOT, Cc, Func, INDIRECT_SLOT, NONE, Op, RETURN_SLOT, Ty, Val};
use oximond_mir::ra::Compiler;
use oximond_mir::x64::X64;

/// Words per activation in the exit area; the last holds the unwind mark.
const STRIDE: u32 = 4;
/// Byte of the control block holding the inputs buffer's address.
const INDIRECT: i32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    status: i64,
    bits: u64,
}

/// What SysV returns in rax and xmm0.
#[repr(C)]
#[derive(Clone, Copy)]
struct PairF {
    status: i64,
    value: f64,
}

/// `fib(left, control, area, fuel, n) -> (status, fib n)`, a recursive
/// activation. Each activation spends one fuel (the poll exit stores what is
/// left and `left`), returns `n` below 2 from a side, exits when no depth is
/// left (storing `n`, and copying input word 1 when `indirect`), and after a
/// call whose status is not positive marks its own area and returns that
/// status. `float` makes the result an `F64`.
fn fib(float: bool, indirect: bool) -> Func {
    let mut f = Func::new();
    f.indirect = INDIRECT;
    let a = f.entry_args(&[Ty::I64; 5]);
    let (left, control, area, fuel, n) = (a, a + 1, a + 2, a + 3, a + 4);
    let zero = f.iconst(0);
    let one = f.iconst(1);
    let two = f.iconst(2);
    let fuel1 = f.bin(Op::Sub, fuel, one);
    f.guard(
        Cc::Gt,
        fuel1,
        zero,
        &[(fuel1, CONTROL_SLOT), (left, CONTROL_SLOT | 1), (n, 0)],
    );
    f.guard_side(Cc::Ge, n, two, 0, &[fuel1, n]);
    let mut depth = vec![(left, CONTROL_SLOT | 1), (fuel1, CONTROL_SLOT), (n, 0)];
    if indirect {
        depth.push((16 + 8, INDIRECT_SLOT | 1));
    }
    f.guard(Cc::Gt, left, zero, &depth);
    let left1 = f.bin(Op::Sub, left, one);
    let step = f.iconst(8 * i64::from(STRIDE));
    let area1 = f.bin(Op::Add, area, step);
    let n1 = f.bin(Op::Sub, n, one);
    let ty = if float { Ty::F64 } else { Ty::I64 };
    let (s1, r1) = f.call_rel(0, &[left1, control, area1, fuel1, n1], Some(ty));
    let mark1 = f.iconst(101);
    f.guard(Cc::Gt, s1, zero, &[(mark1, STRIDE - 1), (s1, RETURN_SLOT)]);
    let n2 = f.bin(Op::Sub, n, two);
    let (s2, r2) = f.call_rel(0, &[left1, control, area1, s1, n2], Some(ty));
    let mark2 = f.iconst(102);
    f.guard(Cc::Gt, s2, zero, &[(mark2, STRIDE - 1), (s2, RETURN_SLOT)]);
    let sum = f.bin(if float { Op::FAdd } else { Op::Add }, r1, r2);
    f.ret(s2, sum);
    let (_, p) = f.begin_side(&[Ty::I64, Ty::I64]);
    let v = if float { f.i2f(p + 1) } else { p + 1 };
    f.ret(p, v);
    f.validate().expect("valid");
    f
}

fn fib_ref(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        fib_ref(n - 1) + fib_ref(n - 2)
    }
}

struct Bufs {
    inputs: Vec<u64>,
    control: Vec<u64>,
    area: Vec<u64>,
}

impl Bufs {
    fn new(depth: u32) -> Bufs {
        let inputs: Vec<u64> = (0..8).map(|k| 0x5000 + k).collect();
        let mut control = vec![0u64; 4];
        control[(INDIRECT / 8) as usize] = inputs.as_ptr() as u64;
        Bufs {
            inputs,
            control,
            area: vec![0u64; (STRIDE * (depth + 2)) as usize],
        }
    }

    fn mem(&self) -> Mem {
        let mut mem = Mem::default();
        for r in [&self.inputs, &self.control, &self.area] {
            mem.add_words(r.as_ptr() as u64, r);
        }
        mem
    }

    fn check(&self, mem: &Mem, what: &str) {
        assert_eq!(self.inputs, mem.words(0), "{what}: inputs");
        assert_eq!(self.control, mem.words(1), "{what}: control");
        assert_eq!(self.area, mem.words(2), "{what}: exit area");
    }
}

type Act = extern "C" fn(i64, *mut u64, *mut u64, i64, i64) -> Pair;
type ActF = extern "C" fn(i64, *mut u64, *mut u64, i64, i64) -> PairF;
type Wrap = extern "C" fn(*mut u64, *mut u64, *mut u64) -> i64;

fn code(t: X64, f: &Func) -> Vec<u8> {
    let mut c = Compiler::new(t);
    c.compile(f);
    let mut img = Vec::new();
    c.copy_code(&mut img);
    img
}

/// Whether the function's `Ret` returns a float second result.
fn float_result(f: &Func) -> bool {
    f.insts
        .iter()
        .any(|i| i.op == Op::Ret && i.b != NONE && f.val_ty[i.b as usize] == Ty::F64)
}

/// Runs the activation natively and in the evaluator from depth `left` with
/// `fuel`, compares, and returns the status and result bits.
fn run_act(t: X64, f: &Func, left: i64, fuel: i64, n: i64) -> Pair {
    let (exec, _) = compile_on(t, f);
    let mut b = Bufs::new(left as u32);
    let mut mem = b.mem();
    let args = [
        left as u64,
        b.control.as_ptr() as u64,
        b.area.as_ptr() as u64,
        fuel as u64,
        n as u64,
    ];
    let (status, bits) = run_function(f, None, &mut mem, &args, 100_000_000);
    // SAFETY: the code was compiled from `f`, whose entry takes five
    // integers and returns an integer and a second result; the buffers live
    // across the call and are large enough for `left` activations.
    let got = if float_result(f) {
        let act: ActF = unsafe { std::mem::transmute::<usize, ActF>(exec.address()) };
        let r = act(left, b.control.as_mut_ptr(), b.area.as_mut_ptr(), fuel, n);
        Pair {
            status: r.status,
            bits: r.value.to_bits(),
        }
    } else {
        let act: Act = unsafe { std::mem::transmute::<usize, Act>(exec.address()) };
        act(left, b.control.as_mut_ptr(), b.area.as_mut_ptr(), fuel, n)
    };
    let want = Pair {
        status: status as i64,
        bits,
    };
    let what = format!("left {left} fuel {fuel} n {n}");
    if got.status > 0 {
        assert_eq!(got, want, "{what}");
    } else {
        // The second result of an exit is whatever the register held.
        assert_eq!(got.status, want.status, "{what}");
    }
    b.check(&mem, &what);
    got
}

#[test]
fn recursion_returns_integer_and_float() {
    for t in targets() {
        let int = fib(false, false);
        let flt = fib(true, false);
        for n in 0..=16 {
            let got = run_act(t, &int, 40, 1 << 40, n);
            assert_eq!(got.bits as i64, fib_ref(n));
            let got = run_act(t, &flt, 40, 1 << 40, n);
            assert_eq!(f64::from_bits(got.bits), fib_ref(n) as f64);
        }
    }
}

#[test]
fn fuel_is_spent_once_per_activation() {
    // fib(n) makes 2 fib(n+1) - 1 activations.
    for t in targets() {
        let f = fib(false, false);
        let got = run_act(t, &f, 40, 1000, 10);
        assert_eq!(got.status, 1000 - (2 * 89 - 1));
    }
}

#[test]
fn depth_exit_unwinds_with_marks() {
    for t in targets() {
        for indirect in [false, true] {
            let f = fib(false, indirect);
            for left in 0..6 {
                let got = run_act(t, &f, left, 1 << 40, 12);
                assert_eq!(got.status, -3, "the depth guard is snapshot 2");
            }
        }
    }
}

#[test]
fn poll_exit_inside_nested_activations() {
    for t in targets() {
        let f = fib(true, true);
        for fuel in 1..40 {
            let got = run_act(t, &f, 30, fuel, 9);
            assert_eq!(got.status, -1, "the poll guard is snapshot 0");
        }
    }
}

/// A wrapper with the three-argument entry that calls the activation placed
/// `before` bytes before it: it reads `n` and the depth cap from the inputs,
/// returns `i64::MIN` with the result at `area[0]` and the fuel at
/// `control[0]`, or `-1 - status` from a side when the activation exits.
fn wrapper(before: i64, float: bool) -> Func {
    let mut f = Func::new();
    f.indirect = INDIRECT;
    let a = f.entry_args(&[Ty::I64; 3]);
    let (inputs, control, area) = (a, a + 1, a + 2);
    let n = f.load(Ty::I64, inputs, NONE, 1, 0);
    let cap = f.load(Ty::I64, inputs, NONE, 1, 8);
    let fuel = f.load(Ty::I64, control, NONE, 1, 0);
    let ty = if float { Ty::F64 } else { Ty::I64 };
    let (status, r) = f.call_rel(-before, &[cap, control, area, fuel, n], Some(ty));
    let zero = f.iconst(0);
    f.guard_side(Cc::Gt, status, zero, 0, &[status]);
    f.store(control, NONE, 1, 0, status);
    f.store(area, NONE, 1, 0, r);
    let done = f.iconst(i64::MIN);
    f.ret(done, NONE);
    let (_, p) = f.begin_side(&[Ty::I64]);
    let m1 = f.iconst(-1);
    let code = f.bin(Op::Sub, m1, p);
    f.ret(code, NONE);
    f.validate().expect("valid");
    f
}

#[test]
fn wrapper_calls_code_before_it() {
    for t in targets() {
        for float in [false, true] {
            let act = fib(float, true);
            let mut img = code(t, &act);
            let before = img.len().next_multiple_of(16);
            img.resize(before, 0xCC);
            let wrap = wrapper(before as i64, float);
            img.extend_from_slice(&code(t, &wrap));
            let exec = oximond_mir::exec::ExecMem::new(&[&img]);
            for (n, cap, fuel) in [(12, 40, 1 << 30), (12, 3, 1 << 30), (12, 40, 50), (1, 0, 5)] {
                let mut b = Bufs::new(cap as u32);
                b.inputs[0] = n as u64;
                b.inputs[1] = cap as u64;
                b.control[0] = fuel as u64;
                let mut mem = b.mem();
                let args = [
                    b.inputs.as_ptr() as u64,
                    b.control.as_ptr() as u64,
                    b.area.as_ptr() as u64,
                ];
                let (want, _) = run_function(&wrap, Some(&act), &mut mem, &args, 100_000_000);
                // SAFETY: the wrapper takes the three buffers and calls the
                // activation compiled before it, inside the same mapping.
                let entry: Wrap =
                    unsafe { std::mem::transmute::<usize, Wrap>(exec.address() + before) };
                let got = entry(
                    b.inputs.as_mut_ptr(),
                    b.control.as_mut_ptr(),
                    b.area.as_mut_ptr(),
                );
                let what = format!("float {float} n {n} cap {cap} fuel {fuel}");
                assert_eq!(got, want as i64, "{what}");
                b.check(&mem, &what);
                if got == i64::MIN {
                    let v = b.area[0];
                    let v = if float {
                        f64::from_bits(v) as i64
                    } else {
                        v as i64
                    };
                    assert_eq!(v, fib_ref(n), "{what}");
                }
            }
        }
    }
}

/// A second result of either class next to a first result, when both
/// return registers are also where the caller wants other values: the
/// results are swapped and combined after the call.
#[test]
fn second_result_moves_with_the_first() {
    for t in targets() {
        for float in [false, true] {
            let ty = if float { Ty::F64 } else { Ty::I64 };
            let mut f = Func::new();
            let a = f.entry_args(&[Ty::I64; 5]);
            let (left, x) = (a, a + 4);
            let zero = f.iconst(0);
            f.guard_side(Cc::Gt, left, zero, 0, &[x]);
            let one = f.iconst(1);
            let left1 = f.bin(Op::Sub, left, one);
            let x1 = f.bin(Op::Add, x, one);
            let (s, r) = f.call_rel(0, &[left1, a + 1, a + 2, a + 3, x1], Some(ty));
            // The status goes where the second result was and back.
            let (first, second): (Val, Val) = if float {
                let k = f.i2f(s);
                let sum = f.bin(Op::FAdd, r, k);
                (x, sum)
            } else {
                let sum = f.bin(Op::Add, r, s);
                (r, sum)
            };
            f.ret(first, second);
            let (_, p) = f.begin_side(&[Ty::I64]);
            let v = if float { f.i2f(p) } else { p };
            f.ret(p, v);
            f.validate().expect("valid");
            for left in 0..5 {
                run_act(t, &f, left, 7, 3);
            }
        }
    }
}
