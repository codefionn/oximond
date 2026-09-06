//! Target description and the parts of lowering that do not depend on one:
//! locations, parallel-move planning, and the byte sinks the emitters write to.
//!
//! Everything target-specific the allocator needs is behind [`Target`]: the
//! register sets, which operand forms exist (the capability table) and the
//! emission primitives. aarch64 and riscv64 implement the same trait; see
//! `docs/impl/native-backend/mir/prototype.md` for how they fit.

use crate::ir::{Cc, Op, Ty};

/// Register id. Class `Gpr` uses 0..16 and `Fpr` 16..32, whatever the
/// target's own numbering is.
pub type Reg = u8;
pub const NOREG: Reg = 255;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Gpr,
    Fpr,
}

impl Class {
    #[must_use]
    pub fn of(ty: Ty) -> Class {
        match ty {
            Ty::I64 => Class::Gpr,
            Ty::F64 => Class::Fpr,
        }
    }
}

#[must_use]
pub fn class_of_reg(r: Reg) -> Class {
    if r < 16 { Class::Gpr } else { Class::Fpr }
}

/// Where a value is, or a source that needs no location.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Loc {
    Reg(Reg),
    /// Spill slot; slot 0 holds the exit-area pointer.
    Slot(u32),
    /// Constant bits, rematerialized.
    Imm(i64),
}

/// Second source operand of an ALU, compare or store: a register, an
/// immediate the capability table accepted, or a memory (spill slot) operand.
#[derive(Clone, Copy, Debug)]
pub enum Src {
    Reg(Reg),
    Imm(i64),
    Slot(u32),
    /// A float constant in the sink's own constant pool.
    Pool(u32),
    /// The word a load would read: the second operand of a float sum,
    /// difference, product or quotient that is the load's only reader.
    Mem(Addr),
}

#[derive(Clone, Copy, Debug)]
pub struct Addr {
    pub base: Reg,
    pub index: Reg,
    pub scale: u8,
    pub disp: i32,
}

/// A branch target in the three byte regions of one compile.
#[derive(Clone, Copy, Debug)]
pub enum Tgt {
    /// Code in the body, at this distance from the body's top.
    Body(u32),
    /// An exit stub, at this offset in the stub region that follows the body.
    Stub(u32),
    /// The shared epilogue, placed last in the stub region.
    Epilogue,
    /// The loop header, not emitted yet (the body grows down).
    Label,
    /// A join (`Op::Join`), by its index among the function's joins, not
    /// emitted yet.
    Outer(u32),
    /// Entry `k` of the sink's own constant pool, placed after the stubs.
    Pool(u32),
    /// The code at this byte offset from the first byte of the image (the
    /// body's top, which a backward sink knows only at the end).
    Start(i32),
}

/// Where emitted bytes go. The body sink grows down (backward emission), the
/// stub sink grows up. Coordinates: backward sinks count the distance from
/// the top, forward sinks count the length.
pub trait Sink {
    const BACKWARD: bool;
    fn mark(&self) -> u32;
    fn put(&mut self, bytes: &[u8]);
    /// Displacement for a branch whose end is at `end` (this sink's
    /// coordinate), or `None` when the target is not placed yet; the sink then
    /// remembers the field to patch.
    fn rel(&mut self, tgt: Tgt, end: u32) -> Option<i32>;
    /// Index of `bits` in this sink's constant pool, adding it if new.
    fn pool(&mut self, bits: u64) -> u32;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveOp {
    Mov(Loc, Loc),
    Swap(Reg, Reg),
}

/// Turns a set of simultaneous moves `(dst, src)` into a sequence in program
/// order. Each destination appears once. Cycles use `Swap` for two registers
/// of one class, and otherwise a scratch register. Returns the number of
/// cycles broken. `tmp_gpr` carries memory-to-memory copies and must differ
/// from `scratch_gpr`, which holds a cycle's saved value meanwhile.
pub fn plan_moves(
    moves: &mut Vec<(Loc, Loc)>,
    out: &mut Vec<MoveOp>,
    scratch_gpr: Reg,
    scratch_fpr: Reg,
    tmp_gpr: Reg,
) -> u32 {
    out.clear();
    moves.retain(|(d, s)| d != s);
    let mut cycles = 0;
    // Non-immediate moves first; immediates last because they read nothing.
    loop {
        let mut progressed = false;
        let mut i = 0;
        while i < moves.len() {
            let (d, s) = moves[i];
            if matches!(s, Loc::Imm(_)) {
                i += 1;
                continue;
            }
            let blocked = moves
                .iter()
                .enumerate()
                .any(|(j, &(_, s2))| j != i && s2 == d);
            if blocked {
                i += 1;
                continue;
            }
            push_mov(out, d, s, tmp_gpr);
            moves.swap_remove(i);
            progressed = true;
        }
        let pending = moves.iter().any(|(_, s)| !matches!(s, Loc::Imm(_)));
        if !pending {
            break;
        }
        if progressed {
            continue;
        }
        // Every remaining non-immediate move is in a cycle.
        let k = moves
            .iter()
            .position(|(_, s)| !matches!(s, Loc::Imm(_)))
            .unwrap();
        let (d, s) = moves[k];
        cycles += 1;
        if let (Loc::Reg(rd), Loc::Reg(rs)) = (d, s)
            && class_of_reg(rd) == class_of_reg(rs)
            && class_of_reg(rd) == Class::Gpr
        {
            out.push(MoveOp::Swap(rd, rs));
            for m in moves.iter_mut() {
                if m.1 == d {
                    m.1 = s;
                } else if m.1 == s {
                    m.1 = d;
                }
            }
            moves.swap_remove(k);
            // The swap that closes a cycle leaves its last move in place.
            moves.retain(|(d, s)| d != s);
        } else {
            // Save the destination's old value in scratch, redirect readers.
            let scr = match d {
                Loc::Reg(r) if class_of_reg(r) == Class::Fpr => scratch_fpr,
                _ => scratch_gpr,
            };
            out.push(MoveOp::Mov(Loc::Reg(scr), d));
            for m in moves.iter_mut() {
                if m.1 == d {
                    m.1 = Loc::Reg(scr);
                }
            }
        }
    }
    for &(d, s) in moves.iter() {
        push_mov(out, d, s, tmp_gpr);
    }
    moves.clear();
    cycles
}

fn push_mov(out: &mut Vec<MoveOp>, d: Loc, s: Loc, tmp_gpr: Reg) {
    // The emitter does not copy memory to memory itself.
    if let (Loc::Slot(_), Loc::Slot(_)) = (d, s) {
        out.push(MoveOp::Mov(Loc::Reg(tmp_gpr), s));
        out.push(MoveOp::Mov(d, Loc::Reg(tmp_gpr)));
    } else {
        out.push(MoveOp::Mov(d, s));
    }
}

pub trait Target {
    fn name(&self) -> &'static str;
    /// Registers the allocator may hand out, as a bit set over the unified ids.
    fn alloc_regs(&self) -> u32;
    /// The subset of `alloc_regs` the native ABI lets a function clobber.
    fn caller_saved(&self) -> u32;
    /// Registers a division clobbers that are otherwise allocatable. A
    /// function with a division does not allocate them.
    fn division_regs(&self) -> u32 {
        0
    }
    /// Integer and float argument registers of the native call ABI, in order.
    fn arg_regs(&self, class: Class) -> &'static [Reg];
    /// Whether an argument takes the register of its position among all the
    /// arguments (Win64), not of its position among those of its class (SysV).
    fn positional_args(&self) -> bool {
        false
    }
    /// Slots at the bottom of the frame a call leaves to the callee, below
    /// the stack arguments (Win64's 32 bytes of shadow space).
    fn shadow_slots(&self) -> u32 {
        0
    }
    /// Callee-saved registers the prologue saves whether or not the
    /// allocator hands them out (a scratch register the ABI preserves).
    fn always_saved(&self) -> u32 {
        0
    }
    /// Bytes the prologue needs above the slots to save the float registers
    /// in `callee`, a multiple of 16.
    fn save_area(&self, _callee: u32) -> u32 {
        0
    }
    /// The register a call returns an integer or a float in.
    fn ret_reg(&self, class: Class) -> Reg;
    /// Where a function made here returns its second result (`Op::Ret`) and
    /// a `CALL_REL` call finds it, after an integer first result: a
    /// caller-saved register other than `ret_reg(Gpr)` and the argument
    /// registers' order does not matter, as both ends are code made here.
    fn ret2_reg(&self, class: Class) -> Reg;
    /// The frame size a function that calls needs, given the `frame` its
    /// slots and save area need (a multiple of 16) and the callee-saved
    /// registers of its prologue, so the stack is aligned as the ABI wants at
    /// a call.
    fn call_frame(&self, frame: u32, _callee: u32) -> u32 {
        frame
    }
    fn scratch_gpr(&self) -> Reg;
    fn scratch_fpr(&self) -> Reg;
    /// Second scratch: memory-to-memory copies and exit stubs.
    fn tmp_gpr(&self) -> Reg;
    /// Registers carrying (inputs, control, exit_area) on entry.
    fn entry_regs(&self) -> [Reg; 3];
    /// Capability table. Two-address targets overwrite the first source.
    fn two_address(&self, op: Op, ty: Ty) -> bool;
    /// Can `imm` be the second operand of `op` (`Guard` stands for compare)?
    fn imm_ok(&self, op: Op, ty: Ty, imm: i64) -> bool;
    /// Can the second operand of `op` be a memory operand?
    fn mem_src_ok(&self, op: Op, ty: Ty) -> bool;
    /// Does a float compare need its operands in swapped order for `cc`?
    fn fcmp_swapped(&self, cc: Cc) -> bool;

    fn mov<S: Sink>(&self, s: &mut S, dst: Loc, src: Loc);
    fn swap<S: Sink>(&self, s: &mut S, a: Reg, b: Reg);
    /// `dst = a op b`. Two-address targets are called with `dst == a`.
    fn alu<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, a: Reg, b: Src);
    /// `FSqrt` (register or memory source), `I2F`, `F2I`, `FNeg`, `FAbs`, `FFloor`, `FCeil`, `FTrunc`.
    fn unary<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, src: Src);
    /// `dst = a / b` or `a % b` (`op` is `SDiv`, `SRem`, `UDiv` or `URem`).
    /// The divisor is a register, a slot or a constant in the pool, never an
    /// immediate. The division clobbers the scratch registers and
    /// `division_regs`, none of which holds a value then; `dst`, `a` and a
    /// divisor in a register are other registers.
    fn divide<S: Sink>(&self, s: &mut S, op: Op, dst: Reg, a: Reg, b: Src);
    /// Can an integer constant that is no immediate of `op` be read from the
    /// constant pool as its second operand?
    fn int_pool_ok(&self, op: Op) -> bool;
    /// Whether the target has the fused multiply-add `FNma` needs.
    fn fused(&self) -> bool;
    /// Whether `Shl`, `Shr` and `Sar` take a count that is not a constant (in
    /// a register, reduced modulo 64 as the evaluator does).
    fn register_shifts(&self) -> bool;
    /// `dst = c - dst * b` with one rounding (`FNma`); `dst` holds `a`.
    fn fnma<S: Sink>(&self, s: &mut S, dst: Reg, b: Reg, c: Src);
    /// `dst = |a|` with the sign bit of `b` (`FCopySign`), registers only.
    /// `dst` may be `a` or `b`; the scratch float register is free.
    fn copysign<S: Sink>(&self, s: &mut S, dst: Reg, a: Reg, b: Reg);
    fn load<S: Sink>(&self, s: &mut S, ty: Ty, dst: Reg, addr: Addr);
    fn store<S: Sink>(&self, s: &mut S, ty: Ty, addr: Addr, src: Src);
    /// `[addr] += src`, integers.
    fn add_mem<S: Sink>(&self, s: &mut S, addr: Addr, src: Src);
    /// Zero-extending byte load and byte store.
    fn load8<S: Sink>(&self, s: &mut S, dst: Reg, addr: Addr);
    /// Zero-extending 32-bit load.
    fn load32<S: Sink>(&self, s: &mut S, dst: Reg, addr: Addr);
    fn store8<S: Sink>(&self, s: &mut S, addr: Addr, src: Src);
    /// `dst = (a cc b) ? t : dst`, integers; `dst` already holds the false value.
    fn select<S: Sink>(&self, s: &mut S, cc: Cc, dst: Reg, a: Reg, b: Src, t: Src);
    /// `dst = (a cc b) ? t : f` for any mix of integer and float compare
    /// (`cmp_ty`) and result (`ty`). Everything is in registers but `b`.
    /// The result is built in the scratch registers, so `dst` may be any of
    /// the others. A float compare has its operands swapped already where
    /// `fcmp_swapped` says.
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
    );
    /// Compare `a` with `b` and branch to `tgt` when `a cc b` is false
    /// (float: false includes unordered, except `Ne`).
    fn guard<S: Sink>(&self, s: &mut S, ty: Ty, cc: Cc, a: Reg, b: Src, tgt: Tgt);
    /// Whether `alu` takes an integer `Add` of two registers whose
    /// destination is a third, where `two_address` otherwise holds.
    fn three_address_sum(&self) -> bool;
    /// Compare the byte at `addr`, zero-extended, with `imm` and branch to
    /// `tgt` when `byte cc imm` is false. `cc` is `Eq` or `Ne`.
    fn guard_byte<S: Sink>(&self, s: &mut S, cc: Cc, addr: Addr, imm: u8, tgt: Tgt);
    /// Branch to `tgt` when `cc` does not hold of the flags the previous
    /// instruction set (a compare with zero of its own result).
    fn guard_flags<S: Sink>(&self, s: &mut S, cc: Cc, tgt: Tgt);
    fn jump<S: Sink>(&self, s: &mut S, tgt: Tgt);
    /// Calls the function at `addr`. The arguments are in place already; the
    /// scratch registers may be used to hold the address.
    fn call<S: Sink>(&self, s: &mut S, addr: u64);
    /// Calls the code at `tgt` (a `Tgt::Start`) by a relative call.
    fn call_rel<S: Sink>(&self, s: &mut S, tgt: Tgt);
    /// Jumps to the epilogue with `a` in the return register and `b` (when
    /// not `None`) in `ret2_reg` of its class, as `Op::Ret` does. Neither
    /// source is the return register, which holds no value.
    fn ret<S: Sink>(&self, s: &mut S, a: Loc, b: Option<(Loc, Class)>);
    /// Calls the function whose pointer is at byte `off` of the control block,
    /// which the frame keeps in slot `control_slot`. The scratch registers may
    /// be used.
    fn call_ctl<S: Sink>(&self, s: &mut S, control_slot: u32, off: i32);
    /// Stores the result register of the call just made (rax) at byte `off` of
    /// the control block, which the frame keeps in slot `control_slot`. The
    /// scratch registers may be used.
    fn store_result_ctl<S: Sink>(&self, s: &mut S, control_slot: u32, off: i32);
    /// Writes each `(loc, slot)` to `exit_area[slot]` (or, for a slot with
    /// `CONTROL_SLOT` set, to the control block), returns `code` through the
    /// epilogue. The frame keeps the exit-area pointer in slot `area_slot` and
    /// the control pointer in slot `area_slot + 1`. The code goes to the
    /// return register first and the stores leave it alone, so the stub
    /// returns the offset in `s` where its stores start: another exit that
    /// writes the same values jumps there ([`Self::exit_code_stub`]).
    ///
    /// A slot with `RETURN_SLOT` set returns its value in place of `code`
    /// (moved last, so a stub that shares the stores returns it too). A slot
    /// with `INDIRECT_SLOT` set has a `Loc::Imm` byte offset: it copies the
    /// word there in the buffer whose address is at byte `indirect` of the
    /// control block.
    fn exit_stub<S: Sink>(
        &self,
        s: &mut S,
        vals: &[(Loc, u32)],
        code: i64,
        area_slot: u32,
        indirect: i32,
    ) -> u32;
    /// The stub of an exit that writes what the stub whose stores start at
    /// `stores` writes: it sets its own `code` and jumps there.
    fn exit_code_stub<S: Sink>(&self, s: &mut S, code: i64, stores: u32);
    /// Epilogue the stubs jump to: frees `frame` bytes, restores `callee`, returns.
    fn epilogue<S: Sink>(&self, s: &mut S, frame: u32, callee: u32);
    /// Prologue: saves `callee`, reserves `frame` bytes. The float registers
    /// in `callee` go to the top `save_area(callee)` bytes of the frame.
    fn prologue<S: Sink>(&self, s: &mut S, frame: u32, callee: u32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::collections::HashMap;

    fn apply(state: &mut HashMap<Loc, i64>, ops: &[MoveOp]) {
        for op in ops {
            match *op {
                MoveOp::Mov(d, s) => {
                    let v = match s {
                        Loc::Imm(v) => v,
                        other => *state
                            .get(&other)
                            .unwrap_or_else(|| panic!("read of undefined {other:?}")),
                    };
                    state.insert(d, v);
                }
                MoveOp::Swap(a, b) => {
                    let (x, y) = (state[&Loc::Reg(a)], state[&Loc::Reg(b)]);
                    state.insert(Loc::Reg(a), y);
                    state.insert(Loc::Reg(b), x);
                }
            }
        }
    }

    /// A cycle of `n` registers of one class is `n - 1` swaps: the last swap
    /// puts two values in place, and no move of a register to itself follows.
    #[test]
    fn a_register_cycle_is_one_swap_short_of_its_length() {
        for n in 2..8u8 {
            let mut moves: Vec<(Loc, Loc)> = (0..n)
                .map(|k| (Loc::Reg(1 + k), Loc::Reg(1 + (k + 1) % n)))
                .collect();
            let mut state = HashMap::new();
            for k in 0..n {
                state.insert(Loc::Reg(1 + k), 100 + i64::from(k));
            }
            let mut out = Vec::new();
            assert_eq!(
                plan_moves(&mut moves, &mut out, 0, 31, 11),
                u32::from(n) - 1
            );
            assert_eq!(out.len(), usize::from(n) - 1, "{out:?}");
            apply(&mut state, &out);
            for k in 0..n {
                assert_eq!(state[&Loc::Reg(1 + k)], 100 + i64::from((k + 1) % n));
            }
        }
    }

    #[test]
    fn random_parallel_moves_keep_parallel_semantics() {
        let mut rng = Rng::new(7);
        let (mut cycles, mut slot_moves) = (0, 0);
        for _ in 0..5000 {
            // Locations: gprs 1..7, fprs 17..21, slots 1..5.
            let mut locs = Vec::new();
            locs.extend((1..7).map(Loc::Reg));
            locs.extend((17..21).map(Loc::Reg));
            locs.extend((1..5).map(Loc::Slot));
            let mut state = HashMap::new();
            for (k, &l) in locs.iter().enumerate() {
                state.insert(l, 100 + k as i64);
            }
            let mut moves: Vec<(Loc, Loc)> = Vec::new();
            let mut used = Vec::new();
            for _ in 0..rng.below(10) {
                let d = locs[rng.below(locs.len() as u64) as usize];
                if used.contains(&d) {
                    continue;
                }
                let s = if rng.chance(15) {
                    Loc::Imm(rng.below(50) as i64)
                } else {
                    locs[rng.below(locs.len() as u64) as usize]
                };
                // A float register only copies from float registers or memory.
                let class = |l: Loc| match l {
                    Loc::Reg(r) => Some(class_of_reg(r)),
                    _ => None,
                };
                if let (Some(cd), Some(cs)) = (class(d), class(s))
                    && cd != cs
                {
                    continue;
                }
                used.push(d);
                moves.push((d, s));
            }
            let expect: Vec<(Loc, i64)> = moves
                .iter()
                .map(|&(d, s)| (d, if let Loc::Imm(v) = s { v } else { state[&s] }))
                .collect();
            slot_moves += moves
                .iter()
                .filter(|(d, s)| matches!((d, s), (Loc::Slot(_), Loc::Slot(_))))
                .count();
            let mut out = Vec::new();
            // rax = 0, xmm15 = 31, r11 = 11 are the scratch registers.
            for r in [0u8, 31, 11] {
                state.insert(Loc::Reg(r), -1);
            }
            cycles += plan_moves(&mut moves, &mut out, 0, 31, 11);
            apply(&mut state, &out);
            for (d, v) in expect {
                assert_eq!(state[&d], v, "{d:?} after {out:?}");
            }
        }
        assert!(
            cycles > 100 && slot_moves > 50,
            "coverage: {cycles} cycles, {slot_moves} memory copies"
        );
    }
}
