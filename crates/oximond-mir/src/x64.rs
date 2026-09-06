//! x86-64: encoder for the forms the prototype uses, and the [`Target`]
//! implementation, for the SysV call ABI or, on Windows, Win64. Each emit
//! function encodes forward into a small buffer and hands the block to the
//! sink, which places it (down for the body, up for the stubs). Jumps compute
//! their displacement from the sink's coordinates.

use crate::ir::{CONTROL_SLOT, Cc, INDIRECT_SLOT, Op, RETURN_SLOT, Ty};
use crate::target::{Addr, Class, Loc, NOREG, Reg, Sink, Src, Target, Tgt, class_of_reg};

pub const RAX: Reg = 0;
pub const RCX: Reg = 1;
pub const RDX: Reg = 2;
pub const RBX: Reg = 3;
pub const RSP: Reg = 4;
pub const RBP: Reg = 5;
pub const RSI: Reg = 6;
pub const RDI: Reg = 7;
pub const R11: Reg = 11;
pub const XMM15: Reg = 16 + 15;

#[derive(Clone, Copy)]
struct Enc {
    b: [u8; 24],
    n: usize,
}

#[derive(Clone, Copy)]
struct Mem {
    base: Reg,
    index: Reg,
    scale: u8,
    disp: i32,
}

/// Marks a rip-relative operand; the displacement is patched by `put_pool`.
const RIP: Reg = 253;

fn pool_mem() -> Mem {
    Mem {
        base: RIP,
        index: NOREG,
        scale: 1,
        disp: 0,
    }
}

fn slot_mem(slot: u32) -> Mem {
    Mem {
        base: RSP,
        index: NOREG,
        scale: 1,
        disp: 8 * slot as i32,
    }
}

fn addr_mem(a: Addr) -> Mem {
    Mem {
        base: a.base,
        index: a.index,
        scale: a.scale,
        disp: a.disp,
    }
}

impl Enc {
    fn new() -> Enc {
        Enc { b: [0; 24], n: 0 }
    }
    fn u8(&mut self, x: u8) {
        self.b[self.n] = x;
        self.n += 1;
    }
    fn i32(&mut self, x: i32) {
        self.b[self.n..self.n + 4].copy_from_slice(&x.to_le_bytes());
        self.n += 4;
    }
    fn i64(&mut self, x: i64) {
        self.b[self.n..self.n + 8].copy_from_slice(&x.to_le_bytes());
        self.n += 8;
    }
    fn bytes(&self) -> &[u8] {
        &self.b[..self.n]
    }
    fn rex(&mut self, w: bool, r: u8, x: u8, b: u8) {
        let v = 0x40
            | (u8::from(w) << 3)
            | (((r >> 3) & 1) << 2)
            | (((x >> 3) & 1) << 1)
            | ((b >> 3) & 1);
        if v != 0x40 {
            self.u8(v);
        }
    }
    /// A REX prefix even when no bit is set, so byte registers 4 to 7 mean
    /// `spl`, `bpl`, `sil` and `dil` and not `ah` to `bh`.
    fn rex_force(&mut self, w: bool, r: u8, x: u8, b: u8) {
        self.u8(0x40
            | (u8::from(w) << 3)
            | (((r >> 3) & 1) << 2)
            | (((x >> 3) & 1) << 1)
            | ((b >> 3) & 1));
    }
    fn modrm_rr(&mut self, reg: u8, rm: u8) {
        self.u8(0xC0 | ((reg & 7) << 3) | (rm & 7));
    }
    fn modrm_ext(&mut self, ext: u8, rm: u8) {
        self.modrm_rr(ext, rm);
    }
    fn modrm_mem(&mut self, reg: u8, m: Mem) {
        let reg3 = (reg & 7) << 3;
        if m.base == RIP {
            self.u8(reg3 | 5);
            self.i32(0);
            return;
        }
        let base = m.base & 15;
        let has_index = m.index != NOREG;
        let (md, dsz) = if m.disp == 0 && (base & 7) != 5 {
            (0u8, 0)
        } else if i8::try_from(m.disp).is_ok() {
            (1, 1)
        } else {
            (2, 4)
        };
        if has_index || (base & 7) == 4 {
            self.u8((md << 6) | reg3 | 4);
            let ss = match m.scale {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                s => panic!("bad scale {s}"),
            };
            let idx = if has_index { m.index & 7 } else { 4 };
            self.u8((ss << 6) | (idx << 3) | (base & 7));
        } else {
            self.u8((md << 6) | reg3 | (base & 7));
        }
        match dsz {
            1 => self.u8(m.disp as i8 as u8),
            4 => self.i32(m.disp),
            _ => {}
        }
    }
    /// `[prefix] [rex] opcode modrm(reg, rm)`
    fn op_rr(&mut self, prefix: u8, w: bool, opc: &[u8], reg: Reg, rm: Reg) {
        if prefix != 0 {
            self.u8(prefix);
        }
        self.rex(w, reg & 15, 0, rm & 15);
        for &o in opc {
            self.u8(o);
        }
        self.modrm_rr(reg & 15, rm & 15);
    }
    fn op_rm(&mut self, prefix: u8, w: bool, opc: &[u8], reg: Reg, m: Mem) {
        if prefix != 0 {
            self.u8(prefix);
        }
        let x = if m.index == NOREG { 0 } else { m.index & 15 };
        let b = if m.base == RIP { 0 } else { m.base & 15 };
        self.rex(w, reg & 15, x, b);
        for &o in opc {
            self.u8(o);
        }
        self.modrm_mem(reg & 15, m);
    }
    /// VEX prefix + opcode + modrm for a register `rm`. `pp`: 0 none, 1 = 66,
    /// 2 = F3, 3 = F2. `v` is the extra source (use 0 when unused).
    fn vex_rr(&mut self, pp: u8, w: bool, opc: u8, reg: Reg, v: Reg, rm: Reg) {
        self.vex_prefix(pp, w, reg & 15, v & 15, 0, rm & 15);
        self.u8(opc);
        self.modrm_rr(reg & 15, rm & 15);
    }
    fn vex_rm(&mut self, pp: u8, w: bool, opc: u8, reg: Reg, v: Reg, m: Mem) {
        let x = if m.index == NOREG { 0 } else { m.index & 15 };
        let b = if m.base == RIP { 0 } else { m.base & 15 };
        self.vex_prefix(pp, w, reg & 15, v & 15, x, b);
        self.u8(opc);
        self.modrm_mem(reg & 15, m);
    }
    fn vex_prefix(&mut self, pp: u8, w: bool, r: u8, v: u8, x: u8, b: u8) {
        self.vex_prefix_map(1, pp, w, r, v, x, b);
    }
    /// The VEX prefix for opcode map `map` (1 is `0F`, 2 is `0F38`).
    fn vex_prefix_map(&mut self, map: u8, pp: u8, w: bool, r: u8, v: u8, x: u8, b: u8) {
        if x < 8 && b < 8 && !w && map == 1 {
            self.u8(0xC5);
            self.u8((u8::from(r < 8) << 7) | ((!v & 15) << 3) | pp);
        } else {
            self.u8(0xC4);
            self.u8((u8::from(r < 8) << 7) | (u8::from(x < 8) << 6) | (u8::from(b < 8) << 5) | map);
            self.u8((u8::from(w) << 7) | ((!v & 15) << 3) | pp);
        }
    }
    fn mov_ri(&mut self, d: Reg, imm: i64) {
        if (0..=i64::from(u32::MAX)).contains(&imm) {
            self.rex(false, 0, 0, d);
            self.u8(0xB8 + (d & 7));
            self.i32(imm as u32 as i32);
        } else if i32::try_from(imm).is_ok() {
            self.rex(true, 0, 0, d);
            self.u8(0xC7);
            self.modrm_ext(0, d);
            self.i32(imm as i32);
        } else {
            self.rex(true, 0, 0, d);
            self.u8(0xB8 + (d & 7));
            self.i64(imm);
        }
    }
    fn alu_ri(&mut self, ext: u8, d: Reg, imm: i32) {
        self.rex(true, 0, 0, d);
        if let Ok(i) = i8::try_from(imm) {
            self.u8(0x83);
            self.modrm_ext(ext, d);
            self.u8(i as u8);
        } else {
            self.u8(0x81);
            self.modrm_ext(ext, d);
            self.i32(imm);
        }
    }
}

fn put<S: Sink>(s: &mut S, e: &Enc) {
    s.put(e.bytes());
}

/// Places an instruction whose last four bytes are a rip-relative reference to
/// pool entry `k`, filling in the displacement when the pool address is known
/// (it is not until the stub region is sized, so the sink usually records it).
fn put_pool<S: Sink>(s: &mut S, e: &mut Enc, k: u32) {
    let end = if S::BACKWARD {
        s.mark()
    } else {
        s.mark() + e.n as u32
    };
    if let Some(r) = s.rel(Tgt::Pool(k), end) {
        e.b[e.n - 4..e.n].copy_from_slice(&r.to_le_bytes());
    }
    put(s, e);
}

fn cc_code(cc: Cc) -> u8 {
    match cc {
        Cc::Eq => 0x4,
        Cc::Ne => 0x5,
        Cc::Lt => 0xC,
        Cc::Le => 0xE,
        Cc::Gt => 0xF,
        Cc::Ge => 0xD,
        Cc::Ult => 0x2,
        Cc::Ule => 0x6,
        Cc::Ugt => 0x7,
        Cc::Uge => 0x3,
        // Float-only conditions never reach an integer compare.
        Cc::Nlt | Cc::Nle | Cc::Ngt | Cc::Nge => panic!("float condition on an integer compare"),
    }
}

#[derive(Clone, Copy)]
enum Part {
    Bytes(Enc),
    /// An instruction that ends in the displacement of a pool entry.
    Pool(Enc, u32),
    Jcc(u8, Tgt),
    /// `jcc` over the instruction emitted right after it in program order.
    Skip(u8),
}

/// Emits parts given in program order. Backward sinks take them last first.
fn emit_parts<S: Sink>(s: &mut S, parts: &[Part]) {
    let n = parts.len();
    let mut last = 0u32;
    for k in 0..n {
        let idx = if S::BACKWARD { n - 1 - k } else { k };
        let before = s.mark();
        match parts[idx] {
            Part::Bytes(e) => put(s, &e),
            Part::Pool(mut e, k) => put_pool(s, &mut e, k),
            Part::Jcc(cc, tgt) => jcc(s, cc, tgt),
            Part::Skip(cc) => {
                let d = if S::BACKWARD { last } else { 6 };
                s.put(&[0x70 | cc, d as u8]);
            }
        }
        last = s.mark().abs_diff(before);
    }
}

fn jcc<S: Sink>(s: &mut S, cc: u8, tgt: Tgt) {
    if S::BACKWARD {
        let end = s.mark();
        match s.rel(tgt, end) {
            Some(r) if i8::try_from(r).is_ok() => s.put(&[0x70 | cc, r as i8 as u8]),
            other => {
                let r = other.unwrap_or(0);
                let mut e = Enc::new();
                e.u8(0x0F);
                e.u8(0x80 | cc);
                e.i32(r);
                put(s, &e);
            }
        }
    } else {
        let end = s.mark() + 6;
        let r = s.rel(tgt, end).unwrap_or(0);
        let mut e = Enc::new();
        e.u8(0x0F);
        e.u8(0x80 | cc);
        e.i32(r);
        put(s, &e);
    }
}

fn jmp<S: Sink>(s: &mut S, tgt: Tgt) {
    if S::BACKWARD {
        let end = s.mark();
        match s.rel(tgt, end) {
            Some(r) if i8::try_from(r).is_ok() => s.put(&[0xEB, r as i8 as u8]),
            other => {
                let r = other.unwrap_or(0);
                let mut e = Enc::new();
                e.u8(0xE9);
                e.i32(r);
                put(s, &e);
            }
        }
    } else {
        let end = s.mark() + 5;
        let r = s.rel(tgt, end).unwrap_or(0);
        let mut e = Enc::new();
        e.u8(0xE9);
        e.i32(r);
        put(s, &e);
    }
}

/// Caller-saved integer registers first, so a function that fits in them
/// saves nothing in its prologue.
const GPR_ALLOC: u32 = (1 << RCX)
    | (1 << RDX)
    | (1 << RBX)
    | (1 << RBP)
    | (1 << RSI)
    | (1 << RDI)
    | (1 << 8)
    | (1 << 9)
    | (1 << 10)
    | (1 << 12)
    | (1 << 13)
    | (1 << 14)
    | (1 << 15);
const FPR_ALLOC: u32 = 0x7FFF << 16;
#[cfg(not(windows))]
const GPR_CALLER: u32 =
    (1 << RCX) | (1 << RDX) | (1 << RSI) | (1 << RDI) | (1 << 8) | (1 << 9) | (1 << 10);
/// Win64 keeps rsi, rdi and xmm6 to xmm15 across a call.
#[cfg(windows)]
const GPR_CALLER: u32 = (1 << RCX) | (1 << RDX) | (1 << 8) | (1 << 9) | (1 << 10);
#[cfg(not(windows))]
const FPR_CALLER: u32 = FPR_ALLOC;
#[cfg(windows)]
const FPR_CALLER: u32 = 0x3F << 16;

/// x86-64 with the host's call ABI: SysV, or Win64 on Windows. With `avx` the
/// float operations use the three-operand VEX forms, so they need no copy
/// before an operation that overwrites a live value.
#[derive(Clone, Copy)]
pub struct X64 {
    pub avx: bool,
    /// The fused multiply-add of FMA3 (`FNma`); needs the VEX encodings.
    pub fma: bool,
    /// The shifts by a register of BMI2 (`shlx`, `shrx`, `sarx`), which take
    /// the count in any register and set no flags (`Shl`, `Shr` and `Sar`
    /// with a count that is not a constant).
    pub bmi2: bool,
}

impl X64 {
    /// Uses AVX if the CPU has it.
    #[must_use]
    pub fn detect() -> X64 {
        let avx = std::arch::is_x86_feature_detected!("avx");
        X64 {
            avx,
            fma: avx && std::arch::is_x86_feature_detected!("fma"),
            bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        }
    }

    #[must_use]
    pub fn sse2() -> X64 {
        X64 {
            avx: false,
            fma: false,
            bmi2: false,
        }
    }
}

impl Default for X64 {
    fn default() -> Self {
        X64::detect()
    }
}

fn alu_opcodes(op: Op) -> (u8, u8, u8) {
    // (r/m <- r opcode, r <- r/m opcode, /ext for the immediate form)
    match op {
        Op::Add => (0x01, 0x03, 0),
        Op::Or => (0x09, 0x0B, 1),
        Op::And => (0x21, 0x23, 4),
        Op::Sub => (0x29, 0x2B, 5),
        Op::Xor => (0x31, 0x33, 6),
        _ => (0x39, 0x3B, 7), // cmp
    }
}

fn sse_opcode(op: Op) -> u8 {
    match op {
        Op::FAdd => 0x58,
        Op::FMul => 0x59,
        Op::FSub => 0x5C,
        Op::FDiv => 0x5E,
        Op::FSqrt => 0x51,
        _ => unreachable!(),
    }
}

impl X64 {
    /// Float memory-or-register operand through the legacy SSE encoding.
    fn sse_rm(e: &mut Enc, prefix: u8, opc: u8, reg: Reg, b: Src) -> Option<u32> {
        match b {
            Src::Reg(r) => e.op_rr(prefix, false, &[0x0F, opc], reg, r),
            Src::Slot(k) => e.op_rm(prefix, false, &[0x0F, opc], reg, slot_mem(k)),
            Src::Mem(m) => e.op_rm(prefix, false, &[0x0F, opc], reg, addr_mem(m)),
            Src::Pool(k) => {
                e.op_rm(prefix, false, &[0x0F, opc], reg, pool_mem());
                return Some(k);
            }
            Src::Imm(_) => panic!("float immediate"),
        }
        None
    }

    /// The same through VEX; `v` is the first source (`NOREG` for none).
    fn vex_rm(e: &mut Enc, pp: u8, opc: u8, reg: Reg, v: Reg, b: Src) -> Option<u32> {
        let v = if v == NOREG { 0 } else { v };
        match b {
            Src::Reg(r) => e.vex_rr(pp, false, opc, reg, v, r),
            Src::Slot(k) => e.vex_rm(pp, false, opc, reg, v, slot_mem(k)),
            Src::Mem(m) => e.vex_rm(pp, false, opc, reg, v, addr_mem(m)),
            Src::Pool(k) => {
                e.vex_rm(pp, false, opc, reg, v, pool_mem());
                return Some(k);
            }
            Src::Imm(_) => panic!("float immediate"),
        }
        None
    }
}

fn place<S: Sink>(s: &mut S, mut e: Enc, pool: Option<u32>) {
    match pool {
        Some(k) => put_pool(s, &mut e, k),
        None => put(s, &e),
    }
}

impl Target for X64 {
    fn name(&self) -> &'static str {
        match (cfg!(windows), self.avx) {
            (false, true) => "x86-64 SysV + AVX",
            (false, false) => "x86-64 SysV (SSE2)",
            (true, true) => "x86-64 Win64 + AVX",
            (true, false) => "x86-64 Win64 (SSE2)",
        }
    }
    fn alloc_regs(&self) -> u32 {
        GPR_ALLOC | FPR_ALLOC
    }
    fn caller_saved(&self) -> u32 {
        GPR_CALLER | FPR_CALLER
    }
    fn division_regs(&self) -> u32 {
        1 << RDX
    }
    #[cfg(not(windows))]
    fn arg_regs(&self, class: Class) -> &'static [Reg] {
        match class {
            Class::Gpr => &[RDI, RSI, RDX, RCX, 8, 9],
            Class::Fpr => &[16, 17, 18, 19, 20, 21, 22, 23],
        }
    }
    #[cfg(windows)]
    fn arg_regs(&self, class: Class) -> &'static [Reg] {
        match class {
            Class::Gpr => &[RCX, RDX, 8, 9],
            Class::Fpr => &[16, 17, 18, 19],
        }
    }
    fn positional_args(&self) -> bool {
        cfg!(windows)
    }
    fn shadow_slots(&self) -> u32 {
        if cfg!(windows) { 4 } else { 0 }
    }
    fn always_saved(&self) -> u32 {
        // The float scratch is xmm15, which Win64 preserves.
        if cfg!(windows) { 1 << XMM15 } else { 0 }
    }
    fn save_area(&self, callee: u32) -> u32 {
        16 * (callee >> 16).count_ones()
    }
    fn ret_reg(&self, class: Class) -> Reg {
        match class {
            Class::Gpr => RAX,
            Class::Fpr => 16,
        }
    }
    fn ret2_reg(&self, class: Class) -> Reg {
        // What SysV returns a pair in: rax and rdx, or rax and xmm0. Both
        // are caller-saved under Win64 as well.
        match class {
            Class::Gpr => RDX,
            Class::Fpr => 16,
        }
    }
    fn call_frame(&self, frame: u32, callee: u32) -> u32 {
        // The return address and the pushed registers must leave rsp a
        // multiple of 16 once the frame is reserved.
        if (callee & 0xFFFF).count_ones() % 2 == 0 {
            frame + 8
        } else {
            frame
        }
    }
    fn scratch_gpr(&self) -> Reg {
        RAX
    }
    fn scratch_fpr(&self) -> Reg {
        XMM15
    }
    fn tmp_gpr(&self) -> Reg {
        R11
    }
    fn entry_regs(&self) -> [Reg; 3] {
        if cfg!(windows) {
            [RCX, RDX, 8]
        } else {
            [RDI, RSI, RDX]
        }
    }
    fn two_address(&self, _op: Op, ty: Ty) -> bool {
        ty == Ty::I64 || !self.avx
    }
    fn imm_ok(&self, op: Op, ty: Ty, imm: i64) -> bool {
        match op {
            Op::Shl | Op::Shr | Op::Sar => (0..64).contains(&imm),
            Op::Add
            | Op::Sub
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Mul
            | Op::Guard
            | Op::Store
            | Op::AddMem => ty == Ty::I64 && i32::try_from(imm).is_ok(),
            _ => false,
        }
    }
    fn int_pool_ok(&self, op: Op) -> bool {
        matches!(
            op,
            Op::Add | Op::Sub | Op::And | Op::Or | Op::Xor | Op::Guard
        )
    }
    fn mem_src_ok(&self, op: Op, _ty: Ty) -> bool {
        !matches!(
            op,
            Op::Store | Op::AddMem | Op::Shl | Op::Shr | Op::Sar | Op::I2F | Op::F2I
        ) && !matches!(
            op,
            Op::FNeg | Op::FAbs | Op::FFloor | Op::FCeil | Op::FTrunc | Op::FCopySign
        )
    }
    fn fcmp_swapped(&self, cc: Cc) -> bool {
        matches!(cc, Cc::Lt | Cc::Le | Cc::Nlt | Cc::Nle)
    }

    fn mov<S: Sink>(&self, s: &mut S, dst: Loc, src: Loc) {
        let mut e = Enc::new();
        let mut pool = None;
        let avx = self.avx;
        match (dst, src) {
            (Loc::Reg(d), Loc::Reg(r)) => match (class_of_reg(d), class_of_reg(r)) {
                (Class::Gpr, Class::Gpr) => e.op_rr(0, true, &[0x89], r, d),
                (Class::Fpr, Class::Fpr) if avx => e.vex_rr(0, false, 0x28, d, 0, r),
                (Class::Fpr, Class::Fpr) => e.op_rr(0, false, &[0x0F, 0x28], d, r),
                (Class::Fpr, Class::Gpr) if avx => e.vex_rr(1, true, 0x6E, d, 0, r),
                (Class::Fpr, Class::Gpr) => e.op_rr(0x66, true, &[0x0F, 0x6E], d, r),
                (Class::Gpr, Class::Fpr) if avx => e.vex_rr(1, true, 0x7E, r, 0, d),
                (Class::Gpr, Class::Fpr) => e.op_rr(0x66, true, &[0x0F, 0x7E], r, d),
            },
            (Loc::Reg(d), Loc::Slot(k)) => match class_of_reg(d) {
                Class::Gpr => e.op_rm(0, true, &[0x8B], d, slot_mem(k)),
                Class::Fpr if avx => e.vex_rm(3, false, 0x10, d, 0, slot_mem(k)),
                Class::Fpr => e.op_rm(0xF2, false, &[0x0F, 0x10], d, slot_mem(k)),
            },
            (Loc::Reg(d), Loc::Imm(v)) => match class_of_reg(d) {
                Class::Gpr => e.mov_ri(d, v),
                Class::Fpr if v == 0 && avx => e.vex_rr(0, false, 0x57, d, d, d),
                Class::Fpr if v == 0 => e.op_rr(0, false, &[0x0F, 0x57], d, d),
                Class::Fpr => {
                    // movsd xmm, [rip + pool entry]
                    let k = s.pool(v as u64);
                    if avx {
                        e.vex_rm(3, false, 0x10, d, 0, pool_mem());
                    } else {
                        e.op_rm(0xF2, false, &[0x0F, 0x10], d, pool_mem());
                    }
                    pool = Some(k);
                }
            },
            (Loc::Slot(k), Loc::Reg(r)) => match class_of_reg(r) {
                Class::Gpr => e.op_rm(0, true, &[0x89], r, slot_mem(k)),
                Class::Fpr if avx => e.vex_rm(3, false, 0x11, r, 0, slot_mem(k)),
                Class::Fpr => e.op_rm(0xF2, false, &[0x0F, 0x11], r, slot_mem(k)),
            },
            (Loc::Slot(k), Loc::Imm(v)) => store_imm(&mut e, slot_mem(k), v),
            _ => panic!("unsupported move {dst:?} <- {src:?}"),
        }
        place(s, e, pool);
    }

    fn swap<S: Sink>(&self, s: &mut S, a: Reg, b: Reg) {
        let mut e = Enc::new();
        e.op_rr(0, true, &[0x87], a, b);
        put(s, &e);
    }

    fn alu<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, a: Reg, b: Src) {
        if op == Op::FCopySign {
            let Src::Reg(b) = b else {
                panic!("copysign takes registers")
            };
            self.copysign(s, dst, a, b);
            return;
        }
        let mut e = Enc::new();
        let mut pool = None;
        match op {
            Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
                if self.avx {
                    pool = Self::vex_rm(&mut e, 3, sse_opcode(op), dst, a, b);
                } else {
                    assert_eq!(dst, a, "SSE2 arithmetic is two-address");
                    pool = Self::sse_rm(&mut e, 0xF2, sse_opcode(op), dst, b);
                }
            }
            Op::Mul => match b {
                Src::Reg(r) => {
                    assert_eq!(dst, a, "x86 integer ALU is two-address");
                    e.op_rr(0, true, &[0x0F, 0xAF], dst, r);
                }
                Src::Slot(k) => {
                    assert_eq!(dst, a, "x86 integer ALU is two-address");
                    e.op_rm(0, true, &[0x0F, 0xAF], dst, slot_mem(k));
                }
                // `imul r, r/m, imm32` has a separate source.
                Src::Imm(v) => {
                    e.op_rr(0, true, &[0x69], dst, a);
                    e.i32(v as i32);
                }
                Src::Pool(_) | Src::Mem(_) => panic!("integer pool or memory operand"),
            },
            Op::Shl | Op::Shr | Op::Sar => {
                if let Src::Reg(count) = b {
                    assert!(self.bmi2, "a shift by a register needs BMI2");
                    // shlx/shrx/sarx dst, a, count: VEX.LZ.0F38.W1 F7, with
                    // the 66, F2 and F3 prefixes.
                    let pp = match op {
                        Op::Shl => 1,
                        Op::Shr => 3,
                        _ => 2,
                    };
                    e.vex_prefix_map(2, pp, true, dst & 15, count & 15, 0, a & 15);
                    e.u8(0xF7);
                    e.modrm_rr(dst & 15, a & 15);
                    place(s, e, pool);
                    return;
                }
                assert_eq!(dst, a);
                let Src::Imm(c) = b else {
                    panic!("shift count must be immediate")
                };
                let ext = match op {
                    Op::Shl => 4,
                    Op::Shr => 5,
                    _ => 7,
                };
                e.rex(true, 0, 0, dst);
                e.u8(0xC1);
                e.modrm_ext(ext, dst);
                e.u8(c as u8);
            }
            // So is a sum of two registers that lands in a third.
            Op::Add if dst != a && matches!(b, Src::Reg(r) if r != dst) => {
                let Src::Reg(r) = b else { unreachable!() };
                let m = Mem {
                    base: a,
                    index: r,
                    scale: 1,
                    disp: 0,
                };
                e.op_rm(0, true, &[0x8D], dst, m);
            }
            // A sum with a constant that lands in another register is `lea`.
            Op::Add if dst != a && matches!(b, Src::Imm(_)) => {
                let Src::Imm(v) = b else { unreachable!() };
                let m = Mem {
                    base: a,
                    index: NOREG,
                    scale: 1,
                    disp: v as i32,
                };
                e.op_rm(0, true, &[0x8D], dst, m);
            }
            _ => {
                assert_eq!(dst, a, "x86 integer ALU is two-address");
                let (rm_r, r_rm, ext) = alu_opcodes(op);
                match b {
                    Src::Reg(r) => e.op_rr(0, true, &[rm_r], r, dst),
                    Src::Slot(k) => e.op_rm(0, true, &[r_rm], dst, slot_mem(k)),
                    Src::Imm(v) => e.alu_ri(ext, dst, v as i32),
                    Src::Pool(k) => {
                        e.op_rm(0, true, &[r_rm], dst, pool_mem());
                        pool = Some(k);
                    }
                    Src::Mem(_) => panic!("integer memory operand"),
                }
            }
        }
        place(s, e, pool);
    }

    fn unary<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, src: Src) {
        let mut e = Enc::new();
        let mut pool = None;
        match (op, src) {
            (Op::FSqrt, _) if self.avx => {
                // The upper lanes come from the first source: the operand
                // itself when it is a register, so there is no false dependency.
                let v = if let Src::Reg(r) = src { r } else { dst };
                pool = Self::vex_rm(&mut e, 3, 0x51, dst, v, src);
            }
            (Op::FSqrt, _) => pool = Self::sse_rm(&mut e, 0xF2, 0x51, dst, src),
            (Op::I2F, Src::Reg(r)) if self.avx => e.vex_rr(3, true, 0x2A, dst, dst, r),
            (Op::I2F, Src::Reg(r)) => e.op_rr(0xF2, true, &[0x0F, 0x2A], dst, r),
            // cvttsd2si r64, xmm
            (Op::F2I, Src::Reg(r)) if self.avx => e.vex_rr(3, true, 0x2C, dst, 0, r),
            (Op::F2I, Src::Reg(r)) => e.op_rr(0xF2, true, &[0x0F, 0x2C], dst, r),
            // Without AVX the result is built in `dst`, so a result that only
            // has a slot, which is computed in the scratch register, cannot
            // use that register for the mask: flip the sign bit in rax.
            (Op::FNeg | Op::FAbs, Src::Reg(r)) if !self.avx && dst == XMM15 => {
                e.op_rr(0x66, true, &[0x0F, 0x7E], r, RAX);
                e.rex(true, 0, 0, RAX);
                e.u8(0x0F);
                e.u8(0xBA);
                // btc rax, 63 or btr rax, 63
                e.modrm_ext(if op == Op::FNeg { 7 } else { 6 }, RAX);
                e.u8(63);
                e.op_rr(0x66, true, &[0x0F, 0x6E], dst, RAX);
            }
            // With AVX the mask is a memory operand in the constant pool
            // (`vxorpd` or `vandpd dst, r, [mask]`): one instruction. The
            // operand is 16 bytes wide, so it also reads the word after the
            // mask, which the image always has (`Compiler::code_len`).
            (Op::FNeg | Op::FAbs, Src::Reg(r)) if self.avx => {
                let mask: i64 = if op == Op::FNeg { i64::MIN } else { i64::MAX };
                let k = s.pool(mask as u64);
                let opc = if op == Op::FNeg { 0x57 } else { 0x54 };
                pool = Self::vex_rm(&mut e, 1, opc, dst, r, Src::Pool(k));
            }
            (Op::FNeg | Op::FAbs, Src::Reg(r)) => {
                // The mask goes through rax and the scratch xmm register,
                // which no allocated value uses.
                let mask: i64 = if op == Op::FNeg { i64::MIN } else { i64::MAX };
                e.mov_ri(RAX, mask);
                if self.avx {
                    e.vex_rr(1, true, 0x6E, XMM15, 0, RAX);
                    let opc = if op == Op::FNeg { 0x57 } else { 0x54 };
                    e.vex_rr(1, false, opc, dst, r, XMM15);
                } else {
                    e.op_rr(0x66, true, &[0x0F, 0x6E], XMM15, RAX);
                    if dst != r {
                        e.op_rr(0, false, &[0x0F, 0x28], dst, r);
                    }
                    let opc = if op == Op::FNeg { 0x57 } else { 0x54 };
                    e.op_rr(0x66, false, &[0x0F, opc], dst, XMM15);
                }
            }
            (Op::FFloor | Op::FCeil | Op::FTrunc, Src::Reg(r)) => {
                // roundsd with the "round toward -inf, +inf or zero, suppress
                // precision exception" immediate (9, 10 and 11).
                let imm = match op {
                    Op::FFloor => 9,
                    Op::FCeil => 10,
                    _ => 11,
                };
                if self.avx {
                    // vroundsd dst, r, r, imm
                    e.u8(0xC4);
                    let rr = dst & 15;
                    let bb = r & 15;
                    e.u8((u8::from(rr < 8) << 7) | (1 << 6) | (u8::from(bb < 8) << 5) | 3);
                    e.u8(((!(r & 15) & 15) << 3) | 1);
                    e.u8(0x0B);
                    e.modrm_rr(rr, bb);
                    e.u8(imm);
                } else {
                    e.u8(0x66);
                    e.rex(false, dst & 15, 0, r & 15);
                    e.u8(0x0F);
                    e.u8(0x3A);
                    e.u8(0x0B);
                    e.modrm_rr(dst & 15, r & 15);
                    e.u8(imm);
                }
            }
            _ => panic!("unsupported unary {op:?}"),
        }
        place(s, e, pool);
    }

    fn fused(&self) -> bool {
        self.fma
    }

    fn register_shifts(&self) -> bool {
        self.bmi2
    }

    fn fnma<S: Sink>(&self, s: &mut S, dst: Reg, b: Reg, c: Src) {
        assert!(self.fma, "FNma needs FMA3");
        // vfnmadd213sd dst, b, c: dst = -(b * dst) + c.
        let mut e = Enc::new();
        let mut pool = None;
        match c {
            Src::Reg(r) => {
                e.vex_prefix_map(2, 1, true, dst & 15, b & 15, 0, r & 15);
                e.u8(0xAD);
                e.modrm_rr(dst & 15, r & 15);
            }
            Src::Slot(k) | Src::Pool(k) => {
                let m = if matches!(c, Src::Slot(_)) {
                    slot_mem(k)
                } else {
                    pool = Some(k);
                    pool_mem()
                };
                let x = if m.index == NOREG { 0 } else { m.index & 15 };
                let base = if m.base == RIP { 0 } else { m.base & 15 };
                e.vex_prefix_map(2, 1, true, dst & 15, b & 15, x, base);
                e.u8(0xAD);
                e.modrm_mem(dst & 15, m);
            }
            Src::Imm(_) | Src::Mem(_) => panic!("float immediate or loaded word"),
        }
        place(s, e, pool);
    }

    fn copysign<S: Sink>(&self, s: &mut S, dst: Reg, a: Reg, b: Reg) {
        // The sign of `b` goes to the scratch register first, so `dst` may be
        // either operand; `a` loses its sign bit by a shift left and back.
        let part = |f: &dyn Fn(&mut Enc)| {
            let mut e = Enc::new();
            f(&mut e);
            Part::Bytes(e)
        };
        if self.avx {
            // The bits where the signs differ, in the scratch register, flip
            // the sign of `a`: `(a ^ b) & sign`, then `a ^` that. Three
            // instructions with the mask in the constant pool (a 16 byte
            // operand, see `unary`), and `dst` may be either operand or the
            // scratch register itself.
            let k = s.pool(i64::MIN as u64);
            let mut differ = Enc::new();
            differ.vex_rr(1, false, 0x57, XMM15, a, b);
            let mut sign = Enc::new();
            let pool = Self::vex_rm(&mut sign, 1, 0x54, XMM15, XMM15, Src::Pool(k));
            let mut flip = Enc::new();
            flip.vex_rr(1, false, 0x57, dst, a, XMM15);
            if S::BACKWARD {
                put(s, &flip);
                place(s, sign, pool);
                put(s, &differ);
            } else {
                put(s, &differ);
                place(s, sign, pool);
                put(s, &flip);
            }
            return;
        }
        if dst == XMM15 {
            // A result that only has a slot is computed into the scratch
            // register, which the sign of `b` would need: work in rax and r11.
            let gpr_shift = |ext: u8, d: Reg, count: u8| {
                part(&|e| {
                    e.rex(true, 0, 0, d);
                    e.u8(0xC1);
                    e.modrm_ext(ext, d);
                    e.u8(count);
                })
            };
            emit_parts(
                s,
                &[
                    // rax = the sign bit of b; r11 = the magnitude of a.
                    part(&|e| e.op_rr(0x66, true, &[0x0F, 0x7E], b, RAX)),
                    gpr_shift(5, RAX, 63),
                    gpr_shift(4, RAX, 63),
                    part(&|e| e.op_rr(0x66, true, &[0x0F, 0x7E], a, R11)),
                    gpr_shift(4, R11, 1),
                    gpr_shift(5, R11, 1),
                    part(&|e| e.op_rr(0, true, &[0x0B], RAX, R11)),
                    part(&|e| e.op_rr(0x66, true, &[0x0F, 0x6E], dst, RAX)),
                ],
            );
        } else if self.avx {
            // vpsrlq/vpsllq xmm, xmm, imm8 are `66 0F 73 /2` and `/6`.
            let shift = |ext: u8, d: Reg, src: Reg, count: u8| {
                part(&|e| {
                    e.vex_rr(1, false, 0x73, ext, d, src);
                    e.u8(count);
                })
            };
            emit_parts(
                s,
                &[
                    shift(2, XMM15, b, 63),
                    shift(6, XMM15, XMM15, 63),
                    shift(6, dst, a, 1),
                    shift(2, dst, dst, 1),
                    // vorpd dst, dst, xmm15
                    part(&|e| e.vex_rr(1, false, 0x56, dst, dst, XMM15)),
                ],
            );
        } else {
            let shift = |ext: u8, d: Reg, count: u8| {
                part(&|e| {
                    e.op_rr(0x66, false, &[0x0F, 0x73], ext, d);
                    e.u8(count);
                })
            };
            let copy = |d: Reg, src: Reg| part(&|e| e.op_rr(0, false, &[0x0F, 0x28], d, src));
            let mut parts = [copy(XMM15, b); 7];
            let mut n = 1;
            for p in [shift(2, XMM15, 63), shift(6, XMM15, 63)] {
                parts[n] = p;
                n += 1;
            }
            if dst != a {
                parts[n] = copy(dst, a);
                n += 1;
            }
            for p in [
                shift(6, dst, 1),
                shift(2, dst, 1),
                // orpd dst, xmm15
                part(&|e| e.op_rr(0x66, false, &[0x0F, 0x56], dst, XMM15)),
            ] {
                parts[n] = p;
                n += 1;
            }
            emit_parts(s, &parts[..n]);
        }
    }

    fn load<S: Sink>(&self, s: &mut S, ty: Ty, dst: Reg, addr: Addr) {
        let mut e = Enc::new();
        match ty {
            Ty::I64 => e.op_rm(0, true, &[0x8B], dst, addr_mem(addr)),
            Ty::F64 if self.avx => e.vex_rm(3, false, 0x10, dst, 0, addr_mem(addr)),
            Ty::F64 => e.op_rm(0xF2, false, &[0x0F, 0x10], dst, addr_mem(addr)),
        }
        put(s, &e);
    }

    fn store<S: Sink>(&self, s: &mut S, ty: Ty, addr: Addr, src: Src) {
        let mut e = Enc::new();
        match (ty, src) {
            (Ty::I64, Src::Reg(r)) => e.op_rm(0, true, &[0x89], r, addr_mem(addr)),
            (Ty::I64, Src::Imm(v)) => {
                e.op_rm(0, true, &[0xC7], 0, addr_mem(addr));
                e.i32(v as i32);
            }
            (Ty::F64, Src::Reg(r)) if self.avx => e.vex_rm(3, false, 0x11, r, 0, addr_mem(addr)),
            (Ty::F64, Src::Reg(r)) => e.op_rm(0xF2, false, &[0x0F, 0x11], r, addr_mem(addr)),
            _ => panic!("unsupported store form"),
        }
        put(s, &e);
    }

    fn divide<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, a: Reg, b: Src) {
        // `idiv` takes its dividend in rdx:rax and leaves the quotient in rax
        // and the remainder in rdx. A function with a division does not
        // allocate rdx (`division_regs`), so rax and rdx are both free here
        // and a divisor in a register is neither of them.
        let mut parts = [Part::Bytes(Enc::new()); 4];
        let mut n = 0;
        let mut push = |part: Part| {
            parts[n] = part;
            n += 1;
        };
        debug_assert!(a != RDX && !matches!(b, Src::Reg(RAX | RDX)));
        if a != RAX {
            let mut e = Enc::new();
            e.op_rr(0, true, &[0x89], a, RAX);
            push(Part::Bytes(e));
        }
        let signed = matches!(op, Op::SDiv | Op::SRem);
        let mut e = Enc::new();
        if signed {
            e.u8(0x48);
            e.u8(0x99); // cqo
        } else {
            e.u8(0x31);
            e.u8(0xD2); // xor edx, edx
        }
        push(Part::Bytes(e));
        // idiv or div b
        let ext = if signed { 7 } else { 6 };
        let mut e = Enc::new();
        match b {
            Src::Reg(b) => {
                e.rex(true, 0, 0, b);
                e.u8(0xF7);
                e.modrm_ext(ext, b);
                push(Part::Bytes(e));
            }
            Src::Slot(k) => {
                e.op_rm(0, true, &[0xF7], ext, slot_mem(k));
                push(Part::Bytes(e));
            }
            Src::Pool(k) => {
                e.op_rm(0, true, &[0xF7], ext, pool_mem());
                push(Part::Pool(e, k));
            }
            Src::Imm(_) | Src::Mem(_) => panic!("division by an immediate or a loaded word"),
        }
        let result = if matches!(op, Op::SDiv | Op::UDiv) {
            RAX
        } else {
            RDX
        };
        if dst != result {
            let mut e = Enc::new();
            e.op_rr(0, true, &[0x89], result, dst);
            push(Part::Bytes(e));
        }
        emit_parts(s, &parts[..n]);
    }

    fn add_mem<S: Sink>(&self, s: &mut S, addr: Addr, src: Src) {
        let m = addr_mem(addr);
        let mut e = Enc::new();
        match src {
            Src::Reg(r) => e.op_rm(0, true, &[0x01], r, m),
            Src::Imm(v) if i8::try_from(v).is_ok() => {
                e.op_rm(0, true, &[0x83], 0, m);
                e.u8(v as u8);
            }
            Src::Imm(v) => {
                e.op_rm(0, true, &[0x81], 0, m);
                e.i32(v as i32);
            }
            _ => panic!("unsupported add to memory form"),
        }
        put(s, &e);
    }

    fn load8<S: Sink>(&self, s: &mut S, dst: Reg, addr: Addr) {
        // movzx r64, byte [mem]
        let mut e = Enc::new();
        e.op_rm(0, true, &[0x0F, 0xB6], dst, addr_mem(addr));
        put(s, &e);
    }

    fn load32<S: Sink>(&self, s: &mut S, dst: Reg, addr: Addr) {
        // mov r32, [mem] zero-extends into the whole register
        let mut e = Enc::new();
        e.op_rm(0, false, &[0x8B], dst, addr_mem(addr));
        put(s, &e);
    }

    fn store8<S: Sink>(&self, s: &mut S, addr: Addr, src: Src) {
        let m = addr_mem(addr);
        let x = if m.index == NOREG { 0 } else { m.index & 15 };
        let mut e = Enc::new();
        match src {
            Src::Reg(r) => {
                e.rex_force(false, r & 15, x, m.base & 15);
                e.u8(0x88);
                e.modrm_mem(r & 15, m);
            }
            Src::Imm(v) => {
                e.rex(false, 0, x, m.base & 15);
                e.u8(0xC6);
                e.modrm_mem(0, m);
                e.u8(v as u8);
            }
            _ => panic!("unsupported byte store form"),
        }
        put(s, &e);
    }

    fn select<S: Sink>(&self, s: &mut S, cc: Cc, dst: Reg, a: Reg, b: Src, t: Src) {
        let mut cmp = Enc::new();
        let (rm_r, r_rm, ext) = alu_opcodes(Op::Guard);
        match b {
            Src::Reg(r) => cmp.op_rr(0, true, &[rm_r], r, a),
            Src::Slot(k) => cmp.op_rm(0, true, &[r_rm], a, slot_mem(k)),
            Src::Imm(v) => cmp.alu_ri(ext, a, v as i32),
            Src::Pool(_) | Src::Mem(_) => panic!("integer pool or memory operand"),
        }
        let mut cm = Enc::new();
        let opc = [0x0F, 0x40 | cc_code(cc)];
        match t {
            Src::Reg(r) => cm.op_rr(0, true, &opc, dst, r),
            Src::Slot(k) => cm.op_rm(0, true, &opc, dst, slot_mem(k)),
            Src::Imm(_) | Src::Pool(_) | Src::Mem(_) => {
                panic!("cmov takes a register or slot")
            }
        }
        emit_parts(s, &[Part::Bytes(cmp), Part::Bytes(cm)]);
    }

    fn select_any<S: Sink>(
        &self,
        s: &mut S,
        ty: Ty,
        cmp_ty: Ty,
        cc: Cc,
        dst: Reg,
        a: Reg,
        b: Src,
        t: Reg,
        f: Reg,
    ) {
        const NE: u8 = 0x5;
        const P: u8 = 0xA;
        // `x` is what the result keeps, `y` what replaces it when the
        // condition holds. An ordered float `==` holds when ZF is set and PF
        // is not: its replacement runs on the opposite flags, so the roles
        // of the two values swap.
        let float = cmp_ty == Ty::F64;
        if self.avx && float && ty == Ty::F64 {
            // Float arms on a float compare: a compare into a mask and a
            // blend. `vcmpsd` predicates, with the operands already swapped
            // where `fcmp_swapped` says (so `Lt` is a `>` of the swapped).
            let Src::Reg(b) = b else {
                panic!("a float select compares registers")
            };
            let predicate: u8 = match cc {
                Cc::Eq => 0,
                Cc::Ne => 4,
                Cc::Gt | Cc::Lt => 30,
                Cc::Ge | Cc::Le => 29,
                Cc::Ngt | Cc::Nlt => 26,
                Cc::Nge | Cc::Nle => 25,
                _ => panic!("unsigned condition on a float select"),
            };
            let mut compare = Enc::new();
            compare.vex_rr(3, false, 0xC2, XMM15, a, b);
            compare.u8(predicate);
            // vblendvpd dst, f, t, xmm15: where the mask is set, `t`.
            let mut blend = Enc::new();
            blend.u8(0xC4);
            blend.u8((u8::from(dst & 15 < 8) << 7) | (1 << 6) | (u8::from(t & 15 < 8) << 5) | 3);
            blend.u8(((!f & 15) << 3) | 1);
            blend.u8(0x4B);
            blend.modrm_rr(dst & 15, t & 15);
            blend.u8((XMM15 & 15) << 4);
            emit_parts(s, &[Part::Bytes(compare), Part::Bytes(blend)]);
            return;
        }
        let (x, y) = if float && cc == Cc::Eq {
            (t, f)
        } else {
            (f, t)
        };
        let to_scratch = |src: Reg, scratch: Reg| {
            let mut e = Enc::new();
            if class_of_reg(src) == Class::Gpr {
                e.op_rr(0, true, &[0x89], src, scratch);
            } else {
                e.op_rr(0x66, true, &[0x0F, 0x7E], src, scratch);
            }
            e
        };
        let mut compare = Enc::new();
        if float {
            let Src::Reg(r) = b else {
                panic!("a float select compares registers")
            };
            if self.avx {
                Self::vex_rm(&mut compare, 1, 0x2E, a, NOREG, Src::Reg(r));
            } else {
                Self::sse_rm(&mut compare, 0x66, 0x2E, a, Src::Reg(r));
            }
        } else {
            let (rm_r, r_rm, ext) = alu_opcodes(Op::Guard);
            match b {
                Src::Reg(r) => compare.op_rr(0, true, &[rm_r], r, a),
                Src::Slot(k) => compare.op_rm(0, true, &[r_rm], a, slot_mem(k)),
                Src::Imm(v) => compare.alu_ri(ext, a, v as i32),
                Src::Pool(_) | Src::Mem(_) => panic!("integer pool or memory operand"),
            }
        }
        let cmov = |code: u8| {
            let mut e = Enc::new();
            e.op_rr(0, true, &[0x0F, 0x40 | code], RAX, R11);
            e
        };
        let mut parts = [Part::Bytes(Enc::new()); 8];
        let mut n = 0;
        let mut push = |e: Enc| {
            parts[n] = Part::Bytes(e);
            n += 1;
        };
        push(to_scratch(x, RAX));
        push(to_scratch(y, R11));
        push(compare);
        if float {
            match cc {
                Cc::Lt | Cc::Gt => push(cmov(0x7)),
                Cc::Le | Cc::Ge => push(cmov(0x3)),
                Cc::Nlt | Cc::Ngt => push(cmov(0x6)),
                Cc::Nle | Cc::Nge => push(cmov(0x2)),
                Cc::Eq | Cc::Ne => {
                    push(cmov(NE));
                    push(cmov(P));
                }
                _ => panic!("unsigned condition on a float select"),
            }
        } else {
            push(cmov(cc_code(cc)));
        }
        if dst != RAX {
            let mut e = Enc::new();
            if class_of_reg(dst) == Class::Gpr {
                e.op_rr(0, true, &[0x89], RAX, dst);
            } else if self.avx {
                e.vex_rr(1, true, 0x6E, dst, 0, RAX);
            } else {
                e.op_rr(0x66, true, &[0x0F, 0x6E], dst, RAX);
            }
            push(e);
        }
        let _ = ty;
        emit_parts(s, &parts[..n]);
    }

    fn guard<S: Sink>(&self, s: &mut S, ty: Ty, cc: Cc, a: Reg, b: Src, tgt: Tgt) {
        if ty == Ty::I64 {
            let mut cmp = Enc::new();
            let (rm_r, r_rm, ext) = alu_opcodes(Op::Guard);
            let mut pool = None;
            match b {
                Src::Reg(r) => cmp.op_rr(0, true, &[rm_r], r, a),
                Src::Slot(k) => cmp.op_rm(0, true, &[r_rm], a, slot_mem(k)),
                Src::Imm(v) => cmp.alu_ri(ext, a, v as i32),
                Src::Pool(k) => {
                    cmp.op_rm(0, true, &[r_rm], a, pool_mem());
                    pool = Some(k);
                }
                Src::Mem(_) => panic!("integer memory operand"),
            }
            let compare = pool.map_or(Part::Bytes(cmp), |k| Part::Pool(cmp, k));
            emit_parts(s, &[compare, Part::Jcc(cc_code(cc) ^ 1, tgt)]);
            return;
        }
        let mut u = Enc::new();
        let pool = if self.avx {
            Self::vex_rm(&mut u, 1, 0x2E, a, NOREG, b)
        } else {
            Self::sse_rm(&mut u, 0x66, 0x2E, a, b)
        };
        const B: u8 = 0x2;
        const BE: u8 = 0x6;
        const A: u8 = 0x7;
        const AE: u8 = 0x3;
        const NE: u8 = 0x5;
        const E: u8 = 0x4;
        const P: u8 = 0xA;
        // The compare may reference the pool, so it needs the sink to place
        // it: emit the compare and the branches in the sink's order.
        let tail: &[Part] = match cc {
            Cc::Lt | Cc::Gt | Cc::Ult | Cc::Ugt => &[Part::Jcc(BE, tgt)],
            Cc::Le | Cc::Ge | Cc::Ule | Cc::Uge => &[Part::Jcc(B, tgt)],
            Cc::Eq => &[Part::Jcc(NE, tgt), Part::Jcc(P, tgt)],
            Cc::Ne => &[Part::Skip(P), Part::Jcc(E, tgt)],
            // The negations are true on NaN, which sets CF, so they exit only
            // when the ordered form holds: `ja` and `jae` are false then.
            Cc::Nlt | Cc::Ngt => &[Part::Jcc(A, tgt)],
            Cc::Nle | Cc::Nge => &[Part::Jcc(AE, tgt)],
        };
        if S::BACKWARD {
            emit_parts(s, tail);
            place(s, u, pool);
        } else {
            place(s, u, pool);
            emit_parts(s, tail);
        }
    }

    fn three_address_sum(&self) -> bool {
        true
    }

    fn guard_byte<S: Sink>(&self, s: &mut S, cc: Cc, addr: Addr, imm: u8, tgt: Tgt) {
        // cmp byte [mem], imm8
        let mut cmp = Enc::new();
        cmp.op_rm(0, false, &[0x80], 7, addr_mem(addr));
        cmp.u8(imm);
        emit_parts(s, &[Part::Bytes(cmp), Part::Jcc(cc_code(cc) ^ 1, tgt)]);
    }

    fn guard_flags<S: Sink>(&self, s: &mut S, cc: Cc, tgt: Tgt) {
        emit_parts(s, &[Part::Jcc(cc_code(cc) ^ 1, tgt)]);
    }

    fn jump<S: Sink>(&self, s: &mut S, tgt: Tgt) {
        jmp(s, tgt);
    }

    fn exit_stub<S: Sink>(
        &self,
        s: &mut S,
        vals: &[(Loc, u32)],
        code: i64,
        area_slot: u32,
        indirect: i32,
    ) -> u32 {
        assert!(!S::BACKWARD, "stubs are emitted forward");
        let mut e = Enc::new();
        e.mov_ri(RAX, code);
        put(s, &e);
        let stores = s.mark();
        // The exit-area pointer is in frame slot 0 and the control pointer in
        // frame slot 1; r11 holds whichever base the stores in hand use. The
        // stores are encoded here directly (every one is a store through
        // r11), and none touches rax, which holds the code: a slot is copied
        // with `push` and `pop`.
        let mut q = Seq::new();
        // The words read through the control block: r11 is the buffer's
        // address for the push, then the exit area's for the pop (read with
        // rsp 8 lower than the frame has it).
        for &(loc, slot) in vals {
            if slot & INDIRECT_SLOT == 0 {
                continue;
            }
            let Loc::Imm(at) = loc else {
                panic!("an indirect word names a byte offset");
            };
            if q.n > Seq::CAP - 40 {
                s.put(q.bytes());
                q.n = 0;
            }
            // mov r11, [rsp + 8 * (area_slot + 1)]; mov r11, [r11 + indirect]
            q.raw(&[0x4C, 0x8B]);
            q.rsp_disp(R11 & 7, 8 * (area_slot + 1) as i32);
            q.raw(&[0x4D, 0x8B]);
            q.r11_disp(R11 & 7, indirect);
            // push qword [r11 + at]
            q.raw(&[0x41, 0xFF]);
            q.r11_disp(6, at as i32);
            // mov r11, [rsp + 8 + 8 * area_slot]; pop qword [r11 + disp]
            q.raw(&[0x4C, 0x8B]);
            q.rsp_disp(R11 & 7, 8 * (area_slot + 1) as i32);
            q.raw(&[0x41, 0x8F]);
            q.r11_disp(0, 8 * (slot & !INDIRECT_SLOT) as i32);
        }
        for control in [false, true] {
            let mut loaded = false;
            for &(loc, slot) in vals {
                if slot & (INDIRECT_SLOT | RETURN_SLOT) != 0
                    || (slot & CONTROL_SLOT != 0) != control
                {
                    continue;
                }
                if q.n > Seq::CAP - 32 {
                    s.put(q.bytes());
                    q.n = 0;
                }
                if !loaded {
                    // mov r11, [rsp + 8 * slot]
                    q.raw(&[0x4C, 0x8B]);
                    q.rsp_disp(R11 & 7, 8 * (area_slot + u32::from(control)) as i32);
                    loaded = true;
                }
                let disp = 8 * (slot & !CONTROL_SLOT) as i32;
                match loc {
                    Loc::Reg(r) => match class_of_reg(r) {
                        // mov [r11 + disp], r
                        Class::Gpr => {
                            q.raw(&[0x49 | (((r >> 3) & 1) << 2), 0x89]);
                            q.r11_disp(r & 7, disp);
                        }
                        // movsd [r11 + disp], xmm
                        Class::Fpr => {
                            let x = r & 15;
                            q.raw(&[0xF2, 0x41 | (((x >> 3) & 1) << 2), 0x0F, 0x11]);
                            q.r11_disp(x & 7, disp);
                        }
                    },
                    // push qword [rsp + 8k]; pop qword [r11 + disp]
                    Loc::Slot(k) => {
                        q.raw(&[0xFF]);
                        q.rsp_disp(6, 8 * k as i32);
                        q.raw(&[0x41, 0x8F]);
                        q.r11_disp(0, disp);
                    }
                    Loc::Imm(v) => {
                        if let Ok(i) = i32::try_from(v) {
                            // mov qword [r11 + disp], imm32
                            q.raw(&[0x49, 0xC7]);
                            q.r11_disp(0, disp);
                            q.raw(&i.to_le_bytes());
                        } else {
                            // mov dword [r11 + disp], low; mov dword [r11 + disp + 4], high
                            for (half, bits) in [(0, v as u32), (4, (v >> 32) as u32)] {
                                q.raw(&[0x41, 0xC7]);
                                q.r11_disp(0, disp + half);
                                q.raw(&bits.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        if q.n > 0 {
            s.put(q.bytes());
        }
        // A returned value replaces the code, after the stores, so a stub
        // that shares them returns it as well.
        for &(loc, slot) in vals {
            if slot & RETURN_SLOT != 0 {
                let mut e = Enc::new();
                match loc {
                    Loc::Reg(r) => {
                        assert_eq!(class_of_reg(r), Class::Gpr, "an integer is returned");
                        e.op_rr(0, true, &[0x89], r, RAX);
                    }
                    Loc::Slot(k) => e.op_rm(0, true, &[0x8B], RAX, slot_mem(k)),
                    Loc::Imm(v) => e.mov_ri(RAX, v),
                }
                put(s, &e);
            }
        }
        jmp(s, Tgt::Epilogue);
        stores
    }

    fn exit_code_stub<S: Sink>(&self, s: &mut S, code: i64, stores: u32) {
        let mut e = Enc::new();
        e.mov_ri(RAX, code);
        put(s, &e);
        jmp(s, Tgt::Stub(stores));
    }

    fn call<S: Sink>(&self, s: &mut S, addr: u64) {
        let mut e = Enc::new();
        e.mov_ri(RAX, addr as i64);
        e.u8(0xFF);
        e.u8(0xD0);
        put(s, &e);
    }

    fn call_rel<S: Sink>(&self, s: &mut S, tgt: Tgt) {
        // call rel32
        let end = if S::BACKWARD { s.mark() } else { s.mark() + 5 };
        let r = s.rel(tgt, end).unwrap_or(0);
        let mut e = Enc::new();
        e.u8(0xE8);
        e.i32(r);
        put(s, &e);
    }

    fn ret<S: Sink>(&self, s: &mut S, a: Loc, b: Option<(Loc, Class)>) {
        // In program order: the first result, then the second (whose source
        // is not the return register, which holds no value), then the jump.
        // A value already where it is returned stays.
        let second = b.and_then(|(b, class)| {
            let to = Loc::Reg(self.ret2_reg(class));
            (b != to).then_some((to, b))
        });
        let first = (a != Loc::Reg(RAX)).then_some(a);
        if S::BACKWARD {
            jmp(s, Tgt::Epilogue);
            if let Some((to, b)) = second {
                self.mov(s, to, b);
            }
            if let Some(a) = first {
                self.mov(s, Loc::Reg(RAX), a);
            }
        } else {
            if let Some(a) = first {
                self.mov(s, Loc::Reg(RAX), a);
            }
            if let Some((to, b)) = second {
                self.mov(s, to, b);
            }
            jmp(s, Tgt::Epilogue);
        }
    }

    fn store_result_ctl<S: Sink>(&self, s: &mut S, control_slot: u32, off: i32) {
        // mov r11, [rsp + slot]; mov [r11 + off], rax
        let mut e = Enc::new();
        e.op_rm(0, true, &[0x8B], R11, slot_mem(control_slot));
        e.op_rm(
            0,
            true,
            &[0x89],
            RAX,
            Mem {
                base: R11,
                index: NOREG,
                scale: 1,
                disp: off,
            },
        );
        put(s, &e);
    }

    fn call_ctl<S: Sink>(&self, s: &mut S, control_slot: u32, off: i32) {
        let mut e = Enc::new();
        e.op_rm(0, true, &[0x8B], RAX, slot_mem(control_slot));
        // call qword ptr [rax + off]
        e.op_rm(
            0,
            false,
            &[0xFF],
            2,
            Mem {
                base: RAX,
                index: NOREG,
                scale: 1,
                disp: off,
            },
        );
        put(s, &e);
    }

    fn epilogue<S: Sink>(&self, s: &mut S, frame: u32, callee: u32) {
        let mut q = Seq::new();
        for (r, at) in float_saves(frame, callee) {
            // movups xmm, [rsp + at]
            let mut e = Enc::new();
            e.op_rm(0, false, &[0x0F, 0x10], r, rsp_mem(at));
            q.push(&e);
        }
        let mut e = Enc::new();
        if frame > 0 {
            e.alu_ri(0, RSP, frame as i32);
        }
        for r in (0..16u8).rev() {
            if callee & (1 << r) != 0 {
                e.rex(false, 0, 0, r);
                e.u8(0x58 + (r & 7));
            }
        }
        e.u8(0xC3);
        q.push(&e);
        s.put(q.bytes());
    }

    fn prologue<S: Sink>(&self, s: &mut S, frame: u32, callee: u32) {
        // One block, so the body sink, which grows down, keeps the order.
        let mut q = Seq::new();
        let mut e = Enc::new();
        for r in 0..16u8 {
            if callee & (1 << r) != 0 {
                e.rex(false, 0, 0, r);
                e.u8(0x50 + (r & 7));
            }
        }
        q.push(&e);
        if cfg!(windows) && frame >= 4096 {
            // Windows commits the stack one guard page at a time, so a frame
            // past a page touches each page on the way down first:
            //     mov rax, rsp
            //     lea r11, [rsp - frame]
            // 1:  sub rax, 4096
            //     cmp rax, r11
            //     jbe 2f
            //     test [rax], eax
            //     jmp 1b
            // 2:  test [r11], eax
            q.raw(&[0x48, 0x89, 0xE0, 0x4C, 0x8D, 0x9C, 0x24]);
            q.raw(&(-(frame as i32)).to_le_bytes());
            q.raw(&[0x48, 0x2D, 0x00, 0x10, 0x00, 0x00]);
            q.raw(&[0x4C, 0x39, 0xD8, 0x76, 0x04, 0x85, 0x00, 0xEB, 0xF1]);
            q.raw(&[0x41, 0x85, 0x03]);
        }
        if frame > 0 {
            let mut e = Enc::new();
            e.alu_ri(5, RSP, frame as i32);
            q.push(&e);
        }
        for (r, at) in float_saves(frame, callee) {
            // movups [rsp + at], xmm
            let mut e = Enc::new();
            e.op_rm(0, false, &[0x0F, 0x11], r, rsp_mem(at));
            q.push(&e);
        }
        if q.n > 0 {
            s.put(q.bytes());
        }
    }
}

/// A run of instructions longer than one [`Enc`], placed as one block.
struct Seq {
    b: [u8; Seq::CAP],
    n: usize,
}

impl Seq {
    const CAP: usize = 256;

    fn new() -> Seq {
        Seq {
            b: [0; Seq::CAP],
            n: 0,
        }
    }

    /// ModRM, and the displacement, of `[r11 + disp]` with `reg` (its low
    /// three bits) in the reg field.
    fn r11_disp(&mut self, reg: u8, disp: i32) {
        self.mem_disp(reg, R11 & 7, disp, false);
    }

    /// ModRM, SIB and displacement of `[rsp + disp]` with `reg` (its low
    /// three bits) in the reg field.
    fn rsp_disp(&mut self, reg: u8, disp: i32) {
        self.mem_disp(reg, 4, disp, true);
    }

    fn mem_disp(&mut self, reg: u8, rm: u8, disp: i32, sib: bool) {
        let md = if disp == 0 {
            0
        } else if i8::try_from(disp).is_ok() {
            1
        } else {
            2
        };
        self.b[self.n] = (md << 6) | ((reg & 7) << 3) | rm;
        self.n += 1;
        if sib {
            self.b[self.n] = 0x24;
            self.n += 1;
        }
        match md {
            1 => self.raw(&[disp as i8 as u8]),
            2 => self.raw(&disp.to_le_bytes()),
            _ => {}
        }
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.b[self.n..self.n + bytes.len()].copy_from_slice(bytes);
        self.n += bytes.len();
    }

    fn push(&mut self, e: &Enc) {
        self.raw(e.bytes());
    }

    fn bytes(&self) -> &[u8] {
        &self.b[..self.n]
    }
}

fn rsp_mem(disp: i32) -> Mem {
    Mem {
        base: RSP,
        index: NOREG,
        scale: 1,
        disp,
    }
}

/// The float registers in `callee` (none on SysV, where every one is
/// caller-saved) and where in the top of a `frame` the prologue keeps each.
fn float_saves(frame: u32, callee: u32) -> impl Iterator<Item = (Reg, i32)> {
    (16..32u8)
        .filter(move |&r| callee & (1 << r) != 0)
        .enumerate()
        .map(move |(k, r)| (r, frame as i32 - 16 * (k as i32 + 1)))
}

fn store_imm(e: &mut Enc, m: Mem, v: i64) {
    if let Ok(i) = i32::try_from(v) {
        e.op_rm(0, true, &[0xC7], 0, m);
        e.i32(i);
    } else {
        e.mov_ri(RAX, v);
        e.op_rm(0, true, &[0x89], RAX, m);
    }
}
