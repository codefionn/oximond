//! Exit stubs that compute cold values (see [`crate::ir::COLD`]).
//!
//! A cold value is read by snapshots only, so the loop does not compute it:
//! a stub that exits through a snapshot that holds it computes it from its
//! operands, which the snapshot reads (weakly) as it reads any value. The
//! stub leaves the function, so every register the snapshot does not read
//! is free for the computation: a float one for a float result, a general
//! one for the integer of a shift.

use super::{Compiler, F_COLD, Func, Inst, NOREG, Op, Reg, SNAP_WARM, Src, Target, Ty, is_binary};
use crate::ir::names_value;
use crate::target::{Class, Loc, Sink, class_of_reg};

/// The most operations one stub computes. A longer chain (a long run of
/// statements on one variable, which every exit after it would repeat) is
/// computed where it stands instead.
const MAX_COLD_OPS: usize = 64;

/// A source in a stub: where a hot value is, or the result of cold operation
/// `k` of the stub.
#[derive(Clone, Copy, Debug)]
enum Cv {
    Loc(Loc),
    Tmp(usize),
}

#[derive(Clone, Copy, Debug)]
struct ColdOp {
    inst: Inst,
    a: Cv,
    /// Unused by a unary operation.
    b: Cv,
}

#[derive(Default)]
pub(super) struct ColdScratch {
    ops: Vec<ColdOp>,
    /// Per value: `epoch` when the pass in progress has met it, and then the
    /// operation that computes it (`op_of`, set by the building pass only).
    stamp: Vec<u32>,
    op_of: Vec<u32>,
    epoch: u32,
    snap: Vec<(Cv, u32)>,
    /// The register each operation's result is in.
    temp: Vec<Reg>,
    /// The last operation that reads each result, or `usize::MAX` for a
    /// result the snapshot stores.
    last: Vec<usize>,
    /// The stores of the stubs emitted so far that compute nothing, for
    /// another exit that stores the same to jump to.
    shared: SharedStores,
}

/// The stores of the exit stubs of one function, by what they store. Many
/// exits of a trace tree store the same values from the same places (the
/// guards of one stretch of a member, where nothing changes between them):
/// the second one sets its exit code and jumps to the first one's stores.
#[derive(Default)]
struct SharedStores {
    /// Open addressing, a power of two long: an index into `stubs` plus one,
    /// or 0 for an empty entry.
    table: Vec<u32>,
    /// Hash, first entry in `entries`, entry count, and the offset of the
    /// stores in the stub region.
    stubs: Vec<(u64, u32, u32, u32)>,
    entries: Vec<(Loc, u32)>,
}

impl SharedStores {
    fn clear(&mut self) {
        self.table.clear();
        self.stubs.clear();
        self.entries.clear();
    }

    fn hash(entries: &[(Loc, u32)]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &(loc, slot) in entries {
            let (kind, bits) = match loc {
                Loc::Reg(r) => (1, u64::from(r)),
                Loc::Slot(k) => (2, u64::from(k)),
                Loc::Imm(v) => (3, v as u64),
            };
            for word in [kind | (u64::from(slot) << 2), bits] {
                h = (h ^ word).wrapping_mul(0x100_0000_01b3).rotate_left(29);
            }
        }
        h
    }

    /// The table position of `entries`, which hash to `hash`: where they
    /// are, or the empty entry they would take.
    fn find(&self, hash: u64, entries: &[(Loc, u32)]) -> usize {
        let mask = self.table.len() - 1;
        let mut at = (hash as usize) & mask;
        loop {
            let k = self.table[at];
            if k == 0 {
                return at;
            }
            let (h, first, n, _) = self.stubs[k as usize - 1];
            if h == hash && self.entries[first as usize..(first + n) as usize] == *entries {
                return at;
            }
            at = (at + 1) & mask;
        }
    }

    /// The offset of the stores of a stub that stores `entries`, if one was
    /// emitted.
    fn get(&self, hash: u64, entries: &[(Loc, u32)]) -> Option<u32> {
        if self.table.is_empty() {
            return None;
        }
        let k = self.table[self.find(hash, entries)];
        (k != 0).then(|| self.stubs[k as usize - 1].3)
    }

    /// Records that the stub whose stores start at `stores` stores
    /// `entries`.
    fn insert(&mut self, hash: u64, entries: &[(Loc, u32)], stores: u32) {
        if 2 * (self.stubs.len() + 1) > self.table.len() {
            let len = (2 * self.table.len()).max(64);
            self.table.clear();
            self.table.resize(len, 0);
            for k in 0..self.stubs.len() {
                let (h, first, n, _) = self.stubs[k];
                let at = self.find(h, &self.entries[first as usize..(first + n) as usize]);
                self.table[at] = k as u32 + 1;
            }
        }
        let at = self.find(hash, entries);
        if self.table[at] != 0 {
            return;
        }
        let first = self.entries.len() as u32;
        self.entries.extend_from_slice(entries);
        self.stubs.push((hash, first, entries.len() as u32, stores));
        self.table[at] = self.stubs.len() as u32;
    }
}

impl ColdScratch {
    pub(super) fn reset(&mut self, vals: usize) {
        self.stamp.clear();
        self.stamp.resize(vals, 0);
        self.op_of.clear();
        self.op_of.resize(vals, 0);
        self.epoch = 0;
        self.shared.clear();
    }
}

impl<T: Target> Compiler<T> {
    /// Emits the stub of an exit through snapshot `snap` and returns its
    /// offset in the stub region. `end` is set for the exit that ends a
    /// region, which may take a free register for a value with no location.
    pub(super) fn exit_stub_for(&mut self, f: &Func, snap: u32, end: bool) -> u32 {
        let mut hot = false;
        loop {
            self.cold.ops.clear();
            self.cold.snap.clear();
            // A chain that is too long is found before any operand of it is
            // given a register or a slot.
            // A snapshot of no cold value fits: the prescan saw it.
            if !hot && self.snap_cold[snap as usize] != SNAP_WARM && !self.cold_fits(f, snap) {
                self.stats.cold_hot += 1;
                hot = true;
            }
            self.cold.epoch += 1;
            for sv in f.snap_of(snap) {
                let cv = if names_value(sv) {
                    self.cold_cv(f, sv.val, end, hot)
                } else {
                    Cv::Loc(Loc::Imm(i64::from(sv.val)))
                };
                self.cold.snap.push((cv, sv.slot));
            }
            if self.cold.ops.is_empty()
                || (self.cold.ops.len() <= MAX_COLD_OPS && self.assign_temps())
            {
                break;
            }
            // Too few registers for the temporaries: the values are computed
            // where they stand, as any value a snapshot reads.
            self.stats.cold_hot += 1;
            hot = true;
        }
        let at = self.stubs.mark();
        if !self.cold.ops.is_empty() {
            self.stats.cold_stubs += 1;
            self.emit_cold_ops();
        }
        self.entries.clear();
        for k in 0..self.cold.snap.len() {
            let (cv, slot) = self.cold.snap[k];
            let loc = self.cv_loc(cv);
            self.entries.push((loc, slot));
        }
        let code = f.snaps[snap as usize].code;
        if self.cold.ops.is_empty() {
            let hash = SharedStores::hash(&self.entries);
            if let Some(stores) = self.cold.shared.get(hash, &self.entries) {
                self.stats.shared_stubs += 1;
                self.t.exit_code_stub(&mut self.stubs, code, stores);
                return at;
            }
            let stores =
                self.t
                    .exit_stub(&mut self.stubs, &self.entries, code, self.base, f.indirect);
            self.cold.shared.insert(hash, &self.entries, stores);
        } else {
            self.t
                .exit_stub(&mut self.stubs, &self.entries, code, self.base, f.indirect);
        }
        at
    }

    /// Whether the cold values snapshot `snap` makes the stub compute are at
    /// most `MAX_COLD_OPS`. Counts each once and stops at the first one over.
    fn cold_fits(&mut self, f: &Func, snap: u32) -> bool {
        self.cold.epoch += 1;
        let mut n = 0;
        f.snap_of(snap)
            .iter()
            .all(|sv| !names_value(sv) || self.count_cold(f, sv.val, &mut n))
    }

    fn count_cold(&mut self, f: &Func, v: u32, n: &mut usize) -> bool {
        if self.flags[v as usize] & F_COLD == 0 || self.live(v) {
            return true;
        }
        if self.cold.stamp[v as usize] == self.cold.epoch {
            return true;
        }
        self.cold.stamp[v as usize] = self.cold.epoch;
        *n += 1;
        if *n > MAX_COLD_OPS {
            return false;
        }
        let inst = f.insts[self.def_at[v as usize] as usize];
        self.count_cold(f, inst.a, n) && (!is_binary(inst.op) || self.count_cold(f, inst.b, n))
    }

    /// The source for `v` in the stub: a cold value nothing keeps in a
    /// register or a slot is computed there (with the operands it needs, one
    /// operation after the other), anything else is read where it is.
    fn cold_cv(&mut self, f: &Func, v: u32, end: bool, hot: bool) -> Cv {
        if !hot && self.flags[v as usize] & F_COLD != 0 && !self.live(v) {
            if self.cold.stamp[v as usize] == self.cold.epoch {
                return Cv::Tmp(self.cold.op_of[v as usize] as usize);
            }
            let inst = f.insts[self.def_at[v as usize] as usize];
            let a = self.cold_cv(f, inst.a, end, hot);
            let b = if is_binary(inst.op) {
                self.cold_cv(f, inst.b, end, hot)
            } else {
                Cv::Loc(Loc::Imm(0))
            };
            self.cold.ops.push(ColdOp { inst, a, b });
            let k = self.cold.ops.len() - 1;
            self.cold.stamp[v as usize] = self.cold.epoch;
            self.cold.op_of[v as usize] = k as u32;
            return Cv::Tmp(k);
        }
        Cv::Loc(if end {
            self.exit_loc(v)
        } else {
            self.weak_loc(v)
        })
    }

    fn cv_loc(&self, cv: Cv) -> Loc {
        match cv {
            Cv::Loc(l) => l,
            Cv::Tmp(k) => Loc::Reg(self.cold.temp[k]),
        }
    }

    /// Picks a register for each operation's result among the registers of
    /// its class no source of the stub is in. A result takes the register of
    /// its first operand when that operand dies there and is of its class.
    /// Fails when the registers run out.
    fn assign_temps(&mut self) -> bool {
        let n = self.cold.ops.len();
        self.cold.last.clear();
        self.cold.last.resize(n, 0);
        let mut used = 0u32;
        let note = |cv: Cv, used: &mut u32| {
            if let Cv::Loc(Loc::Reg(r)) = cv {
                *used |= 1 << r;
            }
        };
        for &(cv, _) in &self.cold.snap {
            note(cv, &mut used);
            if let Cv::Tmp(k) = cv {
                self.cold.last[k] = usize::MAX;
            }
        }
        for j in 0..n {
            let op = self.cold.ops[j];
            let operands = [Some(op.a), is_binary(op.inst.op).then_some(op.b)];
            for cv in operands.into_iter().flatten() {
                note(cv, &mut used);
                if let Cv::Tmp(k) = cv
                    && self.cold.last[k] != usize::MAX
                {
                    self.cold.last[k] = j;
                }
            }
        }
        let mut avail = (self.gpr_mask | self.fpr_mask) & !used;
        self.cold.temp.clear();
        self.cold.temp.resize(n, NOREG);
        for j in 0..n {
            let op = self.cold.ops[j];
            let class = if op.inst.ty == Ty::I64 {
                Class::Gpr
            } else {
                Class::Fpr
            };
            let reuse = match op.a {
                Cv::Tmp(k)
                    if self.cold.last[k] == j && class_of_reg(self.cold.temp[k]) == class =>
                {
                    Some(self.cold.temp[k])
                }
                _ => None,
            };
            let t = if let Some(t) = reuse {
                t
            } else {
                let of_class = avail & self.class_mask(class);
                if of_class == 0 {
                    return false;
                }
                let t = of_class.trailing_zeros() as Reg;
                avail &= !(1 << t);
                t
            };
            self.cold.temp[j] = t;
            // What dies here is free for the next result.
            let operands = [Some(op.a), is_binary(op.inst.op).then_some(op.b)];
            for cv in operands.into_iter().flatten() {
                if let Cv::Tmp(k) = cv
                    && self.cold.last[k] == j
                    && self.cold.temp[k] != t
                {
                    avail |= 1 << self.cold.temp[k];
                }
            }
        }
        for j in 0..n {
            self.used_callee |= (1 << self.cold.temp[j]) & self.callee_mask;
        }
        true
    }

    fn emit_cold_ops(&mut self) {
        for j in 0..self.cold.ops.len() {
            let ColdOp { inst, a, b } = self.cold.ops[j];
            let t = self.cold.temp[j];
            let al = self.cv_loc(a);
            self.stats.cold_ops += 1;
            if inst.op == Op::Sar {
                // An integer in a general register, shifted in place by the
                // constant count.
                debug_assert_eq!(class_of_reg(t), Class::Gpr);
                if al != Loc::Reg(t) {
                    self.t.mov(&mut self.stubs, Loc::Reg(t), al);
                }
                let count = self.kbits[inst.b as usize] & 63;
                self.t.alu(&mut self.stubs, Op::Sar, t, t, Src::Imm(count));
                continue;
            }
            debug_assert_eq!(class_of_reg(t), Class::Fpr);
            if inst.op == Op::I2F {
                // The operand is an integer: a register converts as it is, a
                // slot or a constant goes through the integer scratch.
                let src = if let Loc::Reg(r) = al {
                    r
                } else {
                    let s = self.t.scratch_gpr();
                    self.t.mov(&mut self.stubs, Loc::Reg(s), al);
                    s
                };
                self.t.unary(&mut self.stubs, inst.op, t, Src::Reg(src));
                continue;
            }
            if !is_binary(inst.op) {
                let src = if let Loc::Reg(r) = al {
                    Src::Reg(r)
                } else {
                    self.t.mov(&mut self.stubs, Loc::Reg(t), al);
                    Src::Reg(t)
                };
                self.t.unary(&mut self.stubs, inst.op, t, src);
                continue;
            }
            if al != Loc::Reg(t) {
                self.t.mov(&mut self.stubs, Loc::Reg(t), al);
            }
            let src = match self.cv_loc(b) {
                Loc::Reg(r) => Src::Reg(r),
                Loc::Slot(k) if self.t.mem_src_ok(inst.op, Ty::F64) => Src::Slot(k),
                other => {
                    let s = self.t.scratch_fpr();
                    self.t.mov(&mut self.stubs, Loc::Reg(s), other);
                    Src::Reg(s)
                }
            };
            self.t.alu(&mut self.stubs, inst.op, t, t, src);
        }
    }
}
