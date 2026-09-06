//! Reference evaluator for MIR. Memory is a set of byte regions at fixed
//! addresses, so native code and the evaluator can run on copies of the same
//! buffers and be compared byte for byte.

use crate::ir::{CALL_REL, CONTROL_SLOT, Func, INDIRECT_SLOT, NONE, Op, RETURN_SLOT, Ty};

pub struct Region {
    pub base: u64,
    pub data: Vec<u8>,
}

#[derive(Default)]
pub struct Mem {
    pub regions: Vec<Region>,
}

impl Mem {
    /// Adds a copy of `words` living at address `base`.
    pub fn add_words(&mut self, base: u64, words: &[u64]) {
        let mut data = Vec::with_capacity(words.len() * 8);
        for w in words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        self.regions.push(Region { base, data });
    }

    #[must_use]
    pub fn words(&self, region: usize) -> Vec<u64> {
        let (chunks, _) = self.regions[region].data.as_chunks::<8>();
        chunks.iter().map(|c| u64::from_le_bytes(*c)).collect()
    }

    fn find(&self, addr: u64) -> (usize, usize) {
        for (i, r) in self.regions.iter().enumerate() {
            if addr >= r.base && addr + 8 <= r.base + r.data.len() as u64 {
                return (i, (addr - r.base) as usize);
            }
        }
        panic!("evaluator: access to {addr:#x} outside every region");
    }

    fn read(&self, addr: u64) -> u64 {
        let (i, o) = self.find(addr);
        u64::from_le_bytes(self.regions[i].data[o..o + 8].try_into().unwrap())
    }

    fn write(&mut self, addr: u64, v: u64) {
        let (i, o) = self.find(addr);
        self.regions[i].data[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }
}

/// Runs the tree like the native entry would: `args` are `inputs`,
/// `control` and `exit_area`. Returns the entry's result, `-1 - exit index`.
///
/// # Panics
///
/// Panics on an out-of-region access or after `step_limit` instructions.
pub fn run(f: &Func, mem: &mut Mem, args: [u64; 3], step_limit: u64) -> i64 {
    let mut steps = 0u64;
    let image = Image {
        func: f,
        before: None,
    };
    image.call(f, mem, &args, &mut steps, step_limit).0 as i64
}

/// Runs a function that ends in `Ret` (or exits) with the arguments of its
/// `Entry`, `args` (at least three: the second is the control block and the
/// third the exit area). `before` is the function a `CALL_REL` call with a
/// negative offset reaches (placed before `f`); one with offset 0 calls the
/// calling function itself. Returns both results: an exit's code (or its
/// `RETURN_SLOT` value) and 0.
///
/// # Panics
///
/// As [`run`].
pub fn run_function(
    f: &Func,
    before: Option<&Func>,
    mem: &mut Mem,
    args: &[u64],
    step_limit: u64,
) -> (u64, u64) {
    let mut steps = 0u64;
    let image = Image { func: f, before };
    image.call(f, mem, args, &mut steps, step_limit)
}

struct Image<'a> {
    func: &'a Func,
    before: Option<&'a Func>,
}

impl Image<'_> {
    fn target<'b>(&'b self, caller: &'b Func, at: i64) -> &'b Func {
        match at {
            0 => caller,
            _ if std::ptr::eq(caller, self.func) => self
                .before
                .expect("a call before the image names a function"),
            _ => panic!("evaluator: only the outer function calls code before it"),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn call(
        &self,
        f: &Func,
        mem: &mut Mem,
        args: &[u64],
        steps: &mut u64,
        step_limit: u64,
    ) -> (u64, u64) {
        let mut vals = vec![0u64; f.num_vals()];
        let (mut pc, mut end) = f.region(0);
        let mut header = usize::MAX;
        let mut cells = vec![0u64; f.cells as usize];
        loop {
            assert!(pc < end, "evaluator ran off a region");
            *steps += 1;
            assert!(*steps <= step_limit, "evaluator step limit");
            let i = f.insts[pc];
            let fl = |v: &Vec<u64>, x: u32| f64::from_bits(v[x as usize]);
            match i.op {
                Op::Entry => {
                    for k in 0..i.b {
                        vals[(i.a + k) as usize] = args[k as usize];
                    }
                }
                Op::SideEntry => {}
                Op::Const => vals[i.dst as usize] = i.imm as u64,
                Op::Load => {
                    let idx = if i.b == NONE {
                        0
                    } else {
                        vals[i.b as usize].wrapping_mul(u64::from(i.scale))
                    };
                    let addr = vals[i.a as usize]
                        .wrapping_add(idx)
                        .wrapping_add(i.imm as u64);
                    vals[i.dst as usize] = mem.read(addr);
                }
                Op::Store => {
                    let idx = if i.b == NONE {
                        0
                    } else {
                        vals[i.b as usize].wrapping_mul(u64::from(i.scale))
                    };
                    let addr = vals[i.a as usize]
                        .wrapping_add(idx)
                        .wrapping_add(i.imm as u64);
                    mem.write(addr, vals[i.c as usize]);
                }
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::And
                | Op::Or
                | Op::Xor
                | Op::Shl
                | Op::Shr
                | Op::Sar => {
                    let (a, b) = (vals[i.a as usize], vals[i.b as usize]);
                    vals[i.dst as usize] = match i.op {
                        Op::Add => a.wrapping_add(b),
                        Op::Sub => a.wrapping_sub(b),
                        Op::Mul => a.wrapping_mul(b),
                        Op::And => a & b,
                        Op::Or => a | b,
                        Op::Xor => a ^ b,
                        Op::Shl => a << (b & 63),
                        Op::Shr => a >> (b & 63),
                        _ => ((a as i64) >> (b & 63)) as u64,
                    };
                }
                Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
                    let (a, b) = (fl(&vals, i.a), fl(&vals, i.b));
                    let r = match i.op {
                        Op::FAdd => a + b,
                        Op::FSub => a - b,
                        Op::FMul => a * b,
                        _ => a / b,
                    };
                    vals[i.dst as usize] = r.to_bits();
                }
                Op::FSqrt => vals[i.dst as usize] = fl(&vals, i.a).sqrt().to_bits(),
                Op::SDiv | Op::SRem => {
                    let (a, b) = (vals[i.a as usize] as i64, vals[i.b as usize] as i64);
                    assert!(b != 0 && b != -1, "the divisor of SDiv or SRem is 0 or -1");
                    vals[i.dst as usize] = if i.op == Op::SDiv { a / b } else { a % b } as u64;
                }
                Op::UDiv | Op::URem => {
                    let (a, b) = (vals[i.a as usize], vals[i.b as usize]);
                    assert!(b != 0, "the divisor of UDiv or URem is 0");
                    vals[i.dst as usize] = if i.op == Op::UDiv { a / b } else { a % b };
                }
                Op::FNeg => vals[i.dst as usize] = (-fl(&vals, i.a)).to_bits(),
                Op::FAbs => vals[i.dst as usize] = fl(&vals, i.a).abs().to_bits(),
                Op::FFloor => vals[i.dst as usize] = fl(&vals, i.a).floor().to_bits(),
                Op::FCeil => vals[i.dst as usize] = fl(&vals, i.a).ceil().to_bits(),
                Op::FTrunc => vals[i.dst as usize] = fl(&vals, i.a).trunc().to_bits(),
                Op::FCopySign => {
                    let r = fl(&vals, i.a).copysign(fl(&vals, i.b));
                    vals[i.dst as usize] = r.to_bits();
                }
                Op::FNma => {
                    let (a, b, c) = (fl(&vals, i.a), fl(&vals, i.b), fl(&vals, i.c));
                    // The instruction passes on the first NaN operand, quieted and
                    // with its own sign (the negation is of the product), where a
                    // negated `a` would flip it.
                    let r = match [a, b, c].into_iter().find(|x| x.is_nan()) {
                        Some(nan) => f64::from_bits(nan.to_bits() | (1 << 51)),
                        None => (-a).mul_add(b, c),
                    };
                    vals[i.dst as usize] = r.to_bits();
                }
                Op::F2I => {
                    let x = fl(&vals, i.a);
                    vals[i.dst as usize] = if x.is_nan() || x.abs() >= i64::MAX as f64 {
                        i64::MIN as u64
                    } else {
                        x as i64 as u64
                    };
                }
                Op::Bits | Op::FromBits => vals[i.dst as usize] = vals[i.a as usize],
                Op::Load8 | Op::Load32 => {
                    let idx = if i.b == NONE {
                        0
                    } else {
                        vals[i.b as usize].wrapping_mul(u64::from(i.scale))
                    };
                    let addr = vals[i.a as usize]
                        .wrapping_add(idx)
                        .wrapping_add(i.imm as u64);
                    let mask = if i.op == Op::Load8 { 0xff } else { 0xffff_ffff };
                    vals[i.dst as usize] = mem.read(addr) & mask;
                }
                Op::AddMem => {
                    let idx = if i.b == NONE {
                        0
                    } else {
                        vals[i.b as usize].wrapping_mul(u64::from(i.scale))
                    };
                    let addr = vals[i.a as usize]
                        .wrapping_add(idx)
                        .wrapping_add(i.imm as u64);
                    let old = mem.read(addr);
                    mem.write(addr, old.wrapping_add(vals[i.c as usize]));
                }
                Op::Store8 => {
                    let idx = if i.b == NONE {
                        0
                    } else {
                        vals[i.b as usize].wrapping_mul(u64::from(i.scale))
                    };
                    let addr = vals[i.a as usize]
                        .wrapping_add(idx)
                        .wrapping_add(i.imm as u64);
                    let old = mem.read(addr);
                    mem.write(addr, (old & !0xff) | (vals[i.c as usize] & 0xff));
                }
                Op::I2F => vals[i.dst as usize] = (vals[i.a as usize] as i64 as f64).to_bits(),
                Op::Select => {
                    let c = if f.val_ty[i.a as usize] == Ty::F64 {
                        i.cc.eval_f(fl(&vals, i.a), fl(&vals, i.b))
                    } else {
                        i.cc.eval_i(vals[i.a as usize] as i64, vals[i.b as usize] as i64)
                    };
                    vals[i.dst as usize] = if c {
                        vals[i.c as usize]
                    } else {
                        vals[i.d as usize]
                    };
                }
                Op::Guard => {
                    let ok = match i.ty {
                        Ty::I64 => {
                            i.cc.eval_i(vals[i.a as usize] as i64, vals[i.b as usize] as i64)
                        }
                        Ty::F64 => i.cc.eval_f(fl(&vals, i.a), fl(&vals, i.b)),
                    };
                    if !ok {
                        if i.d == NONE {
                            return (exit(f, mem, &vals, args, i.c), 0);
                        }
                        let (ss, se) = f.region(1 + i.d as usize);
                        let side = f.insts[ss];
                        let snap: Vec<u64> = f
                            .snap_of(i.c)
                            .iter()
                            .map(|s| vals[s.val as usize])
                            .collect();
                        for (k, v) in snap.into_iter().enumerate() {
                            vals[side.a as usize + k] = v;
                        }
                        pc = ss;
                        end = se;
                    }
                }
                Op::Header | Op::Join => {
                    for k in 0..i.b {
                        vals[(i.a + k) as usize] = vals[f.lists[(i.c + k) as usize] as usize];
                    }
                    if i.op == Op::Header {
                        header = pc;
                    }
                }
                Op::Rejoin => {
                    let args: Vec<u64> =
                        f.list(i.a, i.b).iter().map(|&v| vals[v as usize]).collect();
                    let (js, je) = f.region(i.d as usize);
                    let nth = if i.e == NONE { 0 } else { i.e as usize };
                    let join = (js..je)
                        .filter(|&k| f.insts[k].op == Op::Join)
                        .nth(nth)
                        .expect("a rejoin names a region with a join");
                    let j = f.insts[join];
                    for (k, v) in args.into_iter().enumerate() {
                        vals[j.a as usize + k] = v;
                    }
                    pc = join;
                    end = je;
                }
                Op::Goto => {
                    let (ss, se) = f.region(1 + i.d as usize);
                    let side = f.insts[ss];
                    let snap: Vec<u64> = f
                        .snap_of(i.c)
                        .iter()
                        .map(|s| vals[s.val as usize])
                        .collect();
                    for (k, v) in snap.into_iter().enumerate() {
                        vals[side.a as usize + k] = v;
                    }
                    pc = ss;
                    end = se;
                    continue;
                }
                Op::CallIf => {
                    let hold = match i.ty {
                        Ty::I64 => {
                            i.cc.eval_i(vals[i.a as usize] as i64, vals[i.b as usize] as i64)
                        }
                        Ty::F64 => i.cc.eval_f(fl(&vals, i.a), fl(&vals, i.b)),
                    };
                    if hold {
                        let target = if i.scale == 1 {
                            mem.read(args[1].wrapping_add(i.imm as u64)) as usize
                        } else {
                            i.imm as usize
                        };
                        let result = call_list(f, &vals, f.list(i.c, i.d), target, Ty::I64);
                        if i.e != NONE {
                            mem.write(args[1].wrapping_add(u64::from(i.e)), result);
                        }
                    }
                }
                Op::If => {
                    let ok = match i.ty {
                        Ty::I64 => {
                            i.cc.eval_i(vals[i.a as usize] as i64, vals[i.b as usize] as i64)
                        }
                        Ty::F64 => i.cc.eval_f(fl(&vals, i.a), fl(&vals, i.b)),
                    };
                    if !ok {
                        // Skipped: the join values take the values from before
                        // the block, and the end is passed.
                        let end = f.insts[i.e as usize];
                        for k in 0..end.b {
                            let s = f.lists[(end.c + end.b + k) as usize];
                            vals[(end.a + k) as usize] = vals[s as usize];
                        }
                        pc = i.e as usize;
                    }
                }
                Op::EndIf => {
                    for k in 0..i.b {
                        let t = f.lists[(i.c + k) as usize];
                        vals[(i.a + k) as usize] = vals[t as usize];
                    }
                }
                Op::CellStore => cells[i.imm as usize] = vals[i.c as usize],
                Op::CellLoad => vals[i.dst as usize] = cells[i.imm as usize],
                Op::ExitArea => vals[i.dst as usize] = args[2],
                Op::Back => {
                    let args: Vec<u64> =
                        f.list(i.a, i.b).iter().map(|&v| vals[v as usize]).collect();
                    let h = f.insts[header];
                    for (k, v) in args.into_iter().enumerate() {
                        vals[h.a as usize + k] = v;
                    }
                    pc = header;
                }
                Op::Exit => return (exit(f, mem, &vals, args, i.c), 0),
                Op::Ret => {
                    let b = if i.b == NONE { 0 } else { vals[i.b as usize] };
                    return (vals[i.a as usize], b);
                }
                Op::Call if i.scale == CALL_REL => {
                    let callee = self.target(f, i.imm);
                    let list: Vec<u64> =
                        f.list(i.a, i.b).iter().map(|&v| vals[v as usize]).collect();
                    let (r0, r1) = self.call(callee, mem, &list, steps, step_limit);
                    if i.dst != NONE {
                        vals[i.dst as usize] = r0;
                    }
                    if i.c != NONE {
                        vals[i.c as usize] = r1;
                    }
                }
                Op::Call | Op::CallCtl => {
                    let target = if i.op == Op::CallCtl {
                        mem.read(args[1].wrapping_add(i.imm as u64)) as usize
                    } else {
                        i.imm as usize
                    };
                    let bits = call_list(f, &vals, f.list(i.a, i.b), target, i.ty);
                    if i.dst != NONE {
                        vals[i.dst as usize] = bits;
                    }
                }
            }
            pc += 1;
        }
    }
}

/// Writes snapshot `snap` and returns what the exit returns: the snapshot's
/// code, or the value of its `RETURN_SLOT` entry.
fn exit(f: &Func, mem: &mut Mem, vals: &[u64], args: &[u64], snap: u32) -> u64 {
    let mut code = f.snaps[snap as usize].code as u64;
    for s in f.snap_of(snap) {
        if s.slot & RETURN_SLOT != 0 {
            code = vals[s.val as usize];
            continue;
        }
        if s.slot & INDIRECT_SLOT != 0 {
            let buf = mem.read(args[1].wrapping_add(f.indirect as u64));
            let word = mem.read(buf.wrapping_add(u64::from(s.val)));
            mem.write(args[2] + 8 * u64::from(s.slot & !INDIRECT_SLOT), word);
            continue;
        }
        let (base, slot) = if s.slot & CONTROL_SLOT != 0 {
            (args[1], s.slot & !CONTROL_SLOT)
        } else {
            (args[2], s.slot)
        };
        mem.write(base + 8 * u64::from(slot), vals[s.val as usize]);
    }
    code
}

/// Calls the C function at `target` with the values `list` names as its
/// arguments, and returns the bits of its result of type `ret`.
#[cfg(not(windows))]
fn call_list(f: &Func, vals: &[u64], list: &[u32], target: usize, ret: Ty) -> u64 {
    let mut ints = [0u64; 16];
    let mut floats = [0f64; 8];
    let (mut ni, mut nf) = (0, 0);
    for &a in list {
        match f.val_ty[a as usize] {
            Ty::I64 => {
                ints[ni] = vals[a as usize];
                ni += 1;
            }
            Ty::F64 => {
                floats[nf] = f64::from_bits(vals[a as usize]);
                nf += 1;
            }
        }
    }
    call_native(target, ret, ints, floats)
}

/// Win64 gives every argument one slot by position, so the arguments are
/// passed as their bits, in order, through [`oximond_mir_call_slots`].
#[cfg(windows)]
#[allow(unsafe_code)]
fn call_list(_f: &Func, vals: &[u64], list: &[u32], target: usize, ret: Ty) -> u64 {
    let mut slots = [0u64; 24];
    for (k, &a) in list.iter().enumerate() {
        slots[k] = vals[a as usize];
    }
    let mut out = [0u64; 2];
    // SAFETY: as for `call_native`; the shim reads `list.len()` slots and
    // writes the two result registers to `out`.
    unsafe { oximond_mir_call_slots(target, slots.as_ptr(), list.len(), out.as_mut_ptr()) };
    match ret {
        Ty::I64 => out[0],
        Ty::F64 => out[1],
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod win64 {
    // Calls rcx with the rdx[0..r8] slots as its arguments: the first four in
    // both their integer and their float register, so the callee finds each
    // where its type wants it, the rest on the stack above the shadow space.
    // Stores rax at r9[0] and xmm0 at r9[1].
    std::arch::global_asm!(
        ".globl oximond_mir_call_slots",
        "oximond_mir_call_slots:",
        "push rbp",
        "mov rbp, rsp",
        "push rsi",
        "push rdi",
        "push r12",
        "push r13",
        "mov r12, rcx",
        "mov rsi, rdx",
        "mov rdi, r8",
        "mov r13, r9",
        "lea rax, [r8 * 8 + 47]",
        "and rax, -16",
        "sub rsp, rax",
        "xor ecx, ecx",
        "2:",
        "cmp rcx, rdi",
        "jae 3f",
        "mov rax, [rsi + rcx * 8]",
        "mov [rsp + rcx * 8], rax",
        "inc rcx",
        "jmp 2b",
        "3:",
        "mov rcx, [rsp]",
        "mov rdx, [rsp + 8]",
        "mov r8, [rsp + 16]",
        "mov r9, [rsp + 24]",
        "movq xmm0, qword ptr [rsp]",
        "movq xmm1, qword ptr [rsp + 8]",
        "movq xmm2, qword ptr [rsp + 16]",
        "movq xmm3, qword ptr [rsp + 24]",
        "call r12",
        "mov [r13], rax",
        "movq qword ptr [r13 + 8], xmm0",
        "lea rsp, [rbp - 32]",
        "pop r13",
        "pop r12",
        "pop rdi",
        "pop rsi",
        "pop rbp",
        "ret",
    );

    unsafe extern "C" {
        pub(super) fn oximond_mir_call_slots(
            target: usize,
            slots: *const u64,
            count: usize,
            out: *mut u64,
        );
    }
}
#[cfg(windows)]
use win64::oximond_mir_call_slots;

#[cfg(not(windows))]
type CallInt = extern "C" fn(
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
) -> u64;
#[cfg(not(windows))]
type CallFloat = extern "C" fn(
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
) -> f64;

/// Calls the C function at `addr` with sixteen integer and eight float
/// arguments (the C ABI puts each class in its own registers, and the integers
/// past the sixth on the stack in order, so a callee that takes fewer ignores
/// the rest) and returns the bits of its result of type `ret`.
#[cfg(not(windows))]
#[allow(unsafe_code)]
fn call_native(addr: usize, ret: Ty, i: [u64; 16], f: [f64; 8]) -> u64 {
    // SAFETY: the test or lowering that built the function gave the address
    // of a C function taking only integer and float arguments, at most
    // sixteen and eight, and returning a value of type `ret` (or nothing,
    // whose result is not read).
    unsafe {
        match ret {
            Ty::I64 => {
                let c: CallInt = std::mem::transmute::<usize, CallInt>(addr);
                c(
                    i[0], i[1], i[2], i[3], i[4], i[5], f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                    f[7], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15],
                )
            }
            Ty::F64 => {
                let c: CallFloat = std::mem::transmute::<usize, CallFloat>(addr);
                c(
                    i[0], i[1], i[2], i[3], i[4], i[5], f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                    f[7], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14], i[15],
                )
                .to_bits()
            }
        }
    }
}
