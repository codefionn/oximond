//! Backward register allocation fused with emission.
//!
//! One pass per region, last instruction to first, writing machine code into a
//! buffer that grows down. At an instruction the allocator already knows every
//! later use of every value, so registers are final when the instruction is
//! encoded. What it implements (names as in `research.md`):
//!
//! - first backward use of a value gives it a register (`use_reg`), its
//!   definition frees the register (`def`);
//! - eviction spills to a slot that is written once, at the definition, and
//!   reloads right after the evicting instruction;
//! - snapshot uses are weak: they never take or evict a register, they read
//!   whatever location the value has, or force a slot written at its definition;
//! - constants are rematerialized at each use, as an immediate when the
//!   capability table allows. A float constant, an integer too wide for an
//!   immediate of a sum, a bit operation or a guard's compare, and a constant
//!   divisor are memory operands in the constant pool instead;
//! - a float load whose only reader is a float sum, difference, product or
//!   quotient in the same stretch of code is that operation's memory operand
//!   (`fusable_load`);
//! - loop: the back edge is processed first, so live-through values get their
//!   "incoming" locations there. Parameters and the back-edge arguments are
//!   hinted to those locations; one parallel move at the back edge and one at
//!   the header fix the rest, with cycle breaking. The guard that closes the
//!   iteration branches to the header itself when the edge moves nothing, or
//!   only copies registers into free ones, which it then does before the
//!   branch (`moves_precede`);
//! - sides are compiled before the root; their entry locations hint the values
//!   the parent's guards pass;
//! - join: a region that sides rejoin (the root, or a side with a loop of its
//!   own) has an outer header, `Join`, whose parameter locations are fixed
//!   before any side is compiled, so a side's `Rejoin` is one parallel move
//!   into them, whichever region it names. A `Goto` is the same move into the
//!   entry locations of a side, as a guard that has a side makes it;
//! - cells: a word of the frame that no allocation touches (`CellStore`,
//!   `CellLoad`), where a region keeps what the code after its `Join` reads
//!   and the sides that rejoin cannot hand back. A store into a cell no kept
//!   load reads is dropped. The frame is laid out as the stack arguments of
//!   calls, the cells, the slots a `CallIf` saves registers in, then the spill
//!   slots;
//! - call: every caller-saved register is dead across it. The prescan marks the
//!   values live across a call, and integers among them prefer callee-saved
//!   registers. Whatever still sits in a caller-saved register at the call is
//!   evicted like any value (stored once at its definition, reloaded after the
//!   call), the arguments are one parallel move into the ABI registers, and
//!   the frame is padded so the stack is aligned at the call. Arguments past
//!   the sixth integer or the eighth float (SysV), or past the fourth
//!   argument (Win64), go to the stack, in order, at the bottom of the frame,
//!   above Win64's shadow slots. `CallCtl` calls the pointer at a byte offset
//!   of the control block;
//! - cold values: a float operation or a shift right by a constant marked
//!   `COLD` (nothing but snapshots read it) is not emitted where it stands.
//!   The exit stub that holds its value computes it from its operands, in
//!   registers the stub's sources do not use (`ra/cold.rs`); a value
//!   something else reads after all is emitted in place;
//! - conditional call: `CallIf` branches to a stub that saves the caller-saved
//!   registers holding values, places the arguments, calls, restores and
//!   jumps back, so the straight line pays one compare and a branch and its
//!   registers stay where they are;
//! - a function of its own: `Entry` takes its arguments in the ABI's argument
//!   registers, `Ret` moves its results to the return registers and jumps to
//!   the epilogue, and a `Call` with `CALL_REL` calls code at an offset from
//!   the image's first byte (itself, at 0), with a second result.

mod cold;

use crate::buf::{CodeBuf, Stubs};
use crate::ir::{CALL_REL, COLD, Cc, Func, Inst, NONE, Op, Ty, is_coldable, names_value};
use crate::target::{Addr, Class, Loc, MoveOp, NOREG, Reg, Sink, Src, Target, Tgt, plan_moves};

const NOSLOT: u32 = u32::MAX;
/// Added to the eviction score of a value the loop defines (see `alloc_reg`).
const LOOP_STORE_PENALTY: u64 = 8;
const LIN_SLOT: Reg = 254;
const DEAD: Loc = Loc::Slot(u32::MAX);

/// Registers of a class that must stay free after a snapshot-only value takes
/// one (see `weak_loc`).
const WEAK_SPARE_REGS: u32 = 3;

const F_CONST: u8 = 1;
const F_WEAK: u8 = 2;
const F_INV: u8 = 4;
const F_FLOAT: u8 = 8;
/// The value is live across a call.
const F_CROSS: u8 = 16;
/// A guard reads the flags this value's defining instruction left (`Op::Guard`
/// with `scale` 1), so a sum must not become `lea`, which sets none.
const F_FLAGS: u8 = 32;
/// The value is cold (`ir::COLD`).
const F_COLD: u8 = 64;

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub insts: u32,
    /// Slots handed out (not counting the exit-area slot).
    pub slots: u32,
    /// Stores to a spill slot emitted at a definition.
    pub def_spill_stores: u32,
    /// Reloads emitted by evictions.
    pub reloads: u32,
    pub evictions: u32,
    /// Constants rematerialized into a register at a use.
    pub remats: u32,
    /// Float constants used as a memory operand of the constant pool.
    pub pool_operands: u32,
    /// Snapshot reads of a value, and how many had to create a slot.
    pub weak_uses: u32,
    pub weak_slots: u32,
    /// Snapshot-only values that took a register because the file had room.
    pub weak_in_regs: u32,
    /// Times a hint was given, and times the hinted register was free.
    pub hints: u32,
    pub hint_hits: u32,
    /// Moves in the back-edge, header and side-trampoline parallel moves.
    pub backedge_moves: u32,
    pub header_moves: u32,
    pub trampoline_moves: u32,
    pub cycles_broken: u32,
    /// Guards that jump straight into a side, and ones that need a trampoline.
    pub side_direct: u32,
    pub trampolines: u32,
    pub pinned_invariants: u32,
    pub dead_skipped: u32,
    /// `If` regions, those whose skip branch needed a stub, and the moves in
    /// those stubs.
    pub ifs: u32,
    pub if_stubs: u32,
    pub if_moves: u32,
    /// Values a block evicted that the `If` binds again to the register the
    /// join expects, so the skip path does not move them.
    pub if_rebound: u32,
    /// The moves at the start of blocks that make room for them.
    pub if_entry_moves: u32,
    /// Exit stubs that compute cold values, the operations in them, and the
    /// stubs that had too few registers and read the values where they stand.
    pub cold_stubs: u32,
    pub cold_ops: u32,
    pub cold_hot: u32,
    /// Exit stubs that set their exit code and jump to the stores of an
    /// earlier stub that stores the same values from the same places.
    pub shared_stubs: u32,
    /// `CellLoad`s of a value that stays in its cell: no load, no store.
    pub cell_homes: u32,
    /// Guards of a byte against a constant that compare the byte in memory,
    /// with no load of their own.
    pub byte_guards: u32,
    /// Float loads read in place by the one operation that reads them.
    pub fused_loads: u32,
    /// Loops whose closing guard branches back to the header itself.
    pub rotated_backs: u32,
    /// Those of them whose back edge moves values, before the branch.
    pub rotated_with_moves: u32,
    /// Skips that end right before a back edge with nothing to move and
    /// branch to the loop header themselves.
    pub rotated_ifs: u32,
}

/// An open `If` while the pass is inside its block: where the join is, and
/// the parts of `if_saved` and `if_dst` that belong to it.
#[derive(Clone, Copy)]
struct IfFrame {
    join: u32,
    saved: u32,
    dst: u32,
}

struct Cx {
    locked: u32,
    temps: u32,
    pre: [(Loc, Loc); 8],
    npre: usize,
}

impl Cx {
    fn new() -> Cx {
        Cx {
            locked: 0,
            temps: 0,
            pre: [(DEAD, DEAD); 8],
            npre: 0,
        }
    }
    fn pre(&mut self, d: Loc, s: Loc) {
        self.pre[self.npre] = (d, s);
        self.npre += 1;
    }
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent per-pass flags of the allocator, read one at a time"
)]
pub struct Compiler<T: Target> {
    t: T,
    /// What the target allows, and what this function may use (less the
    /// registers a division clobbers, when it has one).
    base_alloc: u32,
    alloc_mask: u32,
    caller_mask: u32,
    callee_mask: u32,
    gpr_mask: u32,
    fpr_mask: u32,
    reg: Vec<Reg>,
    slot: Vec<u32>,
    uses: Vec<u32>,
    hint: Vec<Reg>,
    def_at: Vec<u32>,
    flags: Vec<u8>,
    kbits: Vec<i64>,
    lin: Vec<Reg>,
    /// Where the parameters of the `Join` arrive; apart from `lin` because
    /// a join parameter can be an invariant of the inner loop too.
    jlin: Vec<Reg>,
    reg_val: [u32; 32],
    free: u32,
    nslots: u32,
    max_slots: u32,
    used_callee: u32,
    has_calls: bool,
    last_call: usize,
    loop_call: bool,
    prefer_callee: bool,
    /// The region being allocated is the root and the pass is inside its loop.
    in_loop: bool,
    /// The loop header of the region being allocated.
    hdr_idx: usize,
    /// Per region: its loop header, or `usize::MAX`.
    hdr_of: Vec<usize>,
    /// Every `Join` of the function, in program order, and per region the
    /// index in `joins` of its first one (one more entry than there are
    /// regions, so a region's joins are `join_start[r]..join_start[r + 1]`).
    /// The label of a join is its index in `joins`.
    joins: Vec<usize>,
    join_start: Vec<u32>,
    /// Per region: the part of `invariants` its loop has.
    inv_ranges: Vec<(usize, usize)>,
    inv_lo: usize,
    inv_hi: usize,
    cur_region: usize,
    /// The index of the instruction the pass is at.
    cur_idx: usize,
    /// The guard that closes the loop of the region the pass is in, when it
    /// branches back to the header itself (see `do_back`).
    rotated: usize,
    /// The moves of the back edge that the rotated guard makes before its
    /// branch (see `moves_precede`). Empty when the edge moves nothing.
    back_plan: Vec<MoveOp>,
    /// The `EndIf` that stands right before the back edge of the loop being
    /// compiled, when the back edge moves nothing: its `If` branches to the
    /// header instead of to the jump that follows the join.
    rotated_if: usize,
    /// Frame words below the spill slots: the stack arguments of calls, then
    /// the cells. Slot numbers are word offsets from the stack pointer.
    out_slots: u32,
    cells: u32,
    /// Per cell: some `CellLoad` of it was kept. The allocator meets a cell's
    /// loads before its stores (it goes backwards, and a region's sides come
    /// before it), so a store of a cell nobody reads is dropped.
    cell_read: Vec<bool>,
    /// Per cell: the `CellStore` was met and dropped, see `do_cell_store`.
    cell_dropped: Vec<bool>,
    /// Per cell: whether a value loaded from it may live in it, see
    /// `cell_home`. Set by the scan at the start of `compile`.
    cell_homed: Vec<bool>,
    /// Per value: the cell a `CellLoad` defined it from, or the cell every
    /// entry hands a side's parameter from (`find_side_cells`), or `NONE`.
    cell_of: Vec<u32>,
    /// Scratch of `find_side_cells`.
    side_cells: Vec<u32>,
    /// Slot of the exit-area pointer (the control pointer follows it).
    base: u32,
    /// The entry's control-block and exit-area values (`NONE` for none),
    /// whose slots are the frame's own words for them: spilling either
    /// stores nothing more.
    entry_ctl: u32,
    entry_area: u32,
    /// First of the 32 slots a `CallIf` saves registers in, when the function
    /// has one.
    save0: u32,
    /// Frame slots reserved, in every region, for the parameters of the
    /// `Join`s that did not get a register.
    join_slots: u32,
    cb: CodeBuf,
    stubs: Stubs,
    moves: Vec<(Loc, Loc)>,
    plan: Vec<MoveOp>,
    entries: Vec<(Loc, u32)>,
    side_locs: Vec<Loc>,
    side_loc_start: Vec<u32>,
    side_top: Vec<u32>,
    invariants: Vec<u32>,
    /// Per value defined before a loop header: its reads by the loop's
    /// instructions, two for one the loop always runs and one for one inside
    /// an `If` (see `pin_invariants`).
    loop_uses: Vec<u32>,
    /// The `If` regions open at the instruction the prescan is at.
    if_depth: u32,
    cand: Vec<u32>,
    /// Open `If` regions, innermost last, with the values live after each
    /// join and their registers there, and the register of each join value.
    if_frames: Vec<IfFrame>,
    if_saved: Vec<(u32, Reg)>,
    if_dst: Vec<Reg>,
    cold: cold::ColdScratch,
    /// Per value: the instruction (plus one) whose snapshot read it last
    /// while the prescan followed a chain of cold values.
    cold_seen: Vec<u32>,
    /// Per snapshot: whether the prescan saw it read a cold value
    /// ([`SNAP_COLD`]), saw it read none ([`SNAP_WARM`]), or never saw it.
    snap_cold: Vec<u8>,
    stats: Stats,
}

/// A snapshot the prescan never saw read: its stub checks its values for
/// cold ones.
const SNAP_UNSEEN: u8 = 0;
/// A snapshot that reads no cold value: its stub computes nothing.
const SNAP_WARM: u8 = 1;
/// A snapshot that reads a cold value.
const SNAP_COLD: u8 = 2;

fn class_of(ty: Ty) -> Class {
    Class::of(ty)
}

fn is_binary(op: Op) -> bool {
    matches!(op, Op::FAdd | Op::FSub | Op::FMul | Op::FDiv)
}

impl<T: Target> Compiler<T> {
    pub fn new(t: T) -> Self {
        let alloc_mask = t.alloc_regs();
        let caller_mask = t.caller_saved();
        Compiler {
            base_alloc: alloc_mask,
            alloc_mask,
            caller_mask,
            callee_mask: alloc_mask & !caller_mask,
            gpr_mask: alloc_mask & 0xFFFF,
            fpr_mask: alloc_mask & 0xFFFF_0000,
            t,
            reg: Vec::new(),
            slot: Vec::new(),
            uses: Vec::new(),
            hint: Vec::new(),
            def_at: Vec::new(),
            flags: Vec::new(),
            kbits: Vec::new(),
            lin: Vec::new(),
            jlin: Vec::new(),
            reg_val: [NONE; 32],
            free: 0,
            nslots: 2,
            max_slots: 2,
            used_callee: 0,
            has_calls: false,
            last_call: usize::MAX,
            loop_call: false,
            prefer_callee: false,
            in_loop: false,
            hdr_idx: usize::MAX,
            hdr_of: Vec::new(),
            joins: Vec::new(),
            join_start: Vec::new(),
            inv_ranges: Vec::new(),
            inv_lo: 0,
            inv_hi: 0,
            cur_region: 0,
            cur_idx: 0,
            rotated: usize::MAX,
            rotated_if: usize::MAX,
            out_slots: 0,
            cells: 0,
            cell_read: Vec::new(),
            cell_dropped: Vec::new(),
            cell_homed: Vec::new(),
            cell_of: Vec::new(),
            side_cells: Vec::new(),
            base: 0,
            entry_ctl: NONE,
            entry_area: NONE,
            save0: 0,
            join_slots: 0,
            cb: CodeBuf::default(),
            stubs: Stubs::default(),
            moves: Vec::new(),
            plan: Vec::new(),
            back_plan: Vec::new(),
            entries: Vec::new(),
            side_locs: Vec::new(),
            side_loc_start: Vec::new(),
            side_top: Vec::new(),
            invariants: Vec::new(),
            loop_uses: Vec::new(),
            if_depth: 0,
            cand: Vec::new(),
            if_frames: Vec::new(),
            if_saved: Vec::new(),
            if_dst: Vec::new(),
            cold: cold::ColdScratch::default(),
            cold_seen: Vec::new(),
            snap_cold: Vec::new(),
            stats: Stats::default(),
        }
    }

    #[must_use]
    pub fn target(&self) -> &T {
        &self.t
    }

    /// Machine code of the last compile: the body, then the stubs. Entry is
    /// the first byte.
    #[must_use]
    pub fn code_parts(&self) -> (&[u8], &[u8]) {
        (self.cb.body(), &self.stubs.bytes)
    }

    /// Constant pools that follow the stubs: the body's, then the stubs'.
    #[must_use]
    pub fn pools(&self) -> (&[u64], &[u64]) {
        (&self.cb.pool, &self.stubs.pool)
    }

    /// Writes the image [`Self::copy_code`] makes into `out`, which must be
    /// at least [`Self::code_len`] bytes. No allocation.
    pub fn write_code(&self, out: &mut [u8]) {
        let mut at = 0;
        for part in [self.cb.body(), &self.stubs.bytes] {
            out[at..at + part.len()].copy_from_slice(part);
            at += part.len();
        }
        for w in self.cb.pool.iter().chain(self.stubs.pool.iter()) {
            out[at..at + 8].copy_from_slice(&w.to_le_bytes());
            at += 8;
        }
        out[at..at + 8].fill(0);
    }

    /// The pools end in a word of padding: a packed operand that names the
    /// last constant (a sign mask, say) reads 16 bytes.
    #[must_use]
    pub fn code_len(&self) -> usize {
        self.cb.len()
            + self.stubs.bytes.len()
            + 8 * (self.cb.pool.len() + self.stubs.pool.len() + 1)
    }

    /// The whole image: body, stubs, constant pools. Entry is the first byte.
    pub fn copy_code(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(self.cb.body());
        out.extend_from_slice(&self.stubs.bytes);
        for w in self.cb.pool.iter().chain(self.stubs.pool.iter()) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&[0; 8]);
    }

    // ---- small state helpers -------------------------------------------

    fn is_const(&self, v: u32) -> bool {
        self.flags[v as usize] & F_CONST != 0
    }

    fn class_mask(&self, c: Class) -> u32 {
        match c {
            Class::Gpr => self.gpr_mask,
            Class::Fpr => self.fpr_mask,
        }
    }

    fn take(&mut self, r: Reg) {
        self.free &= !(1 << r);
        self.used_callee |= (1 << r) & self.callee_mask;
    }

    fn bind(&mut self, v: u32, r: Reg) {
        self.reg[v as usize] = r;
        self.reg_val[r as usize] = v;
    }

    fn unbind(&mut self, v: u32) {
        let r = self.reg[v as usize];
        self.reg_val[r as usize] = NONE;
        self.reg[v as usize] = NOREG;
        self.free |= 1 << r;
    }

    fn slot_of(&mut self, v: u32) -> u32 {
        if self.slot[v as usize] == NOSLOT {
            if let Some(cell) = self.cell_home(v) {
                // The value is read from the cell from here on, so the store
                // of the cell stays even when no load of it does (a side's
                // parameter that lives in the cell has none of its own).
                self.cell_read[self.cell_of[v as usize] as usize] = true;
                self.slot[v as usize] = cell;
                return cell;
            }
            if let Some(home) = self.entry_home(v) {
                self.slot[v as usize] = home;
                return home;
            }
            self.slot[v as usize] = self.nslots;
            self.nslots += 1;
            self.stats.slots += 1;
        }
        self.slot[v as usize]
    }

    /// The frame word the entry stores `v` in, if `v` is the entry's
    /// control-block or exit-area pointer (see `do_entry`).
    fn entry_home(&self, v: u32) -> Option<u32> {
        if v == self.entry_ctl {
            Some(self.base + 1)
        } else if v == self.entry_area {
            Some(self.base)
        } else {
            None
        }
    }

    /// The slot of the cell `v` is loaded from, when `v` may live there:
    /// the function stores that cell once, before every load of it, outside
    /// any loop or join, so the cell holds the value as long as `v` is read,
    /// and a value that has no register needs no copy of it in a slot of its
    /// own. A loop that a side rejoins would otherwise copy each such value
    /// from its cell to its slot on every link. `find_cell_homes` decides.
    /// A side's parameter that every entry hands such a value of one cell
    /// lives there as well (`find_side_cells`).
    fn cell_home(&self, v: u32) -> Option<u32> {
        let cell = self.cell_of[v as usize];
        (cell != NONE && self.cell_homed[cell as usize]).then(|| self.out_slots + cell)
    }

    /// Per cell, whether a loaded value may live in it: one `CellStore`, met
    /// before any `CellLoad` of the cell in program order, and not between a
    /// `Header` or `Join` and the end of its region (that store runs again
    /// while a loaded value may still be read, or a load above it reads the
    /// store of the iteration before).
    fn find_cell_homes(&mut self, f: &Func) {
        // 0: no store met, 1: one store, 2: no home (more stores, a load
        // first, or a store under a loop).
        let mut state = vec![0u8; f.cells as usize];
        for r in 0..=f.num_sides() {
            let (start, end) = f.region(r);
            let mut looping = false;
            for i in &f.insts[start..end] {
                let cell = i.imm as usize;
                match i.op {
                    Op::Header | Op::Join => looping = true,
                    Op::CellStore => {
                        state[cell] = if state[cell] == 0 && !looping { 1 } else { 2 };
                    }
                    Op::CellLoad if state[cell] == 0 => state[cell] = 2,
                    _ => {}
                }
            }
        }
        self.cell_homed.clear();
        self.cell_homed.extend(state.iter().map(|&s| s == 1));
    }

    fn dec_use(&mut self, v: u32) {
        let u = &mut self.uses[v as usize];
        *u = u.saturating_sub(1);
    }

    fn clear_bindings(&mut self) {
        for r in 0..32 {
            let v = self.reg_val[r];
            if v != NONE {
                self.reg[v as usize] = NOREG;
                self.reg_val[r] = NONE;
            }
        }
        self.free = self.alloc_mask;
    }

    fn alloc_opt(&mut self, class: Class, hint: Reg, locked: u32) -> Option<Reg> {
        let avail = self.free & self.class_mask(class) & !locked;
        if avail == 0 {
            return None;
        }
        let r = if hint != NOREG && avail & (1 << hint) != 0 {
            self.stats.hints += 1;
            self.stats.hint_hits += 1;
            hint
        } else {
            if hint != NOREG {
                self.stats.hints += 1;
            }
            let pref = if self.prefer_callee && class == Class::Gpr && avail & self.callee_mask != 0
            {
                avail & self.callee_mask
            } else {
                avail & self.caller_mask
            };
            (if pref != 0 { pref } else { avail }).trailing_zeros() as Reg
        };
        self.take(r);
        Some(r)
    }

    fn alloc_reg(&mut self, class: Class, hint: Reg, locked: u32) -> Reg {
        if let Some(r) = self.alloc_opt(class, hint, locked) {
            return r;
        }
        // Evict the bound value with the fewest earlier uses; a value that
        // already has a slot is cheaper (only a reload, no new store).
        let mut occ = self.class_mask(class) & !self.free & !locked;
        let mut best = NOREG;
        let mut best_score = u64::MAX;
        while occ != 0 {
            let r = occ.trailing_zeros() as Reg;
            occ &= occ - 1;
            let v = self.reg_val[r as usize];
            if v == NONE {
                continue;
            }
            let mut score = (u64::from(self.uses[v as usize]) << 1)
                | u64::from(self.slot[v as usize] == NOSLOT);
            // A value the loop defines (its parameters too: they are defined
            // at the header) and that has no slot yet would be stored on every
            // iteration: a loop invariant costs one store before the loop, so
            // it goes first. For a parameter the store and the reload are also
            // on the path the next iteration waits for (an accumulator spilled
            // around the products of the body adds the store-forwarding
            // latency to every iteration).
            if self.in_loop
                && self.slot[v as usize] == NOSLOT
                && self.def_at[v as usize] as usize >= self.hdr_idx
            {
                score += LOOP_STORE_PENALTY;
            }
            if score < best_score {
                best_score = score;
                best = r;
            }
        }
        assert!(best != NOREG, "no register to evict");
        let r = best;
        let v = self.reg_val[r as usize];
        let s = self.slot_of(v);
        self.cb.ensure();
        self.t.mov(&mut self.cb, Loc::Reg(r), Loc::Slot(s));
        self.stats.reloads += 1;
        self.stats.evictions += 1;
        self.reg[v as usize] = NOREG;
        self.reg_val[r as usize] = NONE;
        self.used_callee |= (1 << r) & self.callee_mask;
        r
    }

    /// Frees the value's register at its definition and returns the register
    /// the instruction writes. A value nobody holds in a register is computed
    /// into the scratch register (it only needs its slot, if it has one).
    fn def(&mut self, w: u32, class: Class) -> Reg {
        let r = if self.reg[w as usize] != NOREG {
            let r = self.reg[w as usize];
            self.unbind(w);
            r
        } else if class == Class::Gpr {
            self.t.scratch_gpr()
        } else {
            self.t.scratch_fpr()
        };
        let s = self.slot[w as usize];
        if s != NOSLOT && self.cell_home(w) != Some(s) {
            self.cb.ensure();
            self.t.mov(&mut self.cb, Loc::Slot(s), Loc::Reg(r));
            self.stats.def_spill_stores += 1;
        }
        r
    }

    fn use_reg(&mut self, v: u32, cx: &mut Cx) -> Reg {
        self.use_reg_w(v, NOREG, cx)
    }

    /// Like `use_reg`; `want` is a register to prefer for a value that is not
    /// bound yet (the destination, when the operation may overwrite a source).
    fn use_reg_w(&mut self, v: u32, want: Reg, cx: &mut Cx) -> Reg {
        self.dec_use(v);
        let class = self.val_class(v);
        if self.is_const(v) {
            let r = self.alloc_reg(class, NOREG, cx.locked);
            cx.locked |= 1 << r;
            cx.temps |= 1 << r;
            cx.pre(Loc::Reg(r), Loc::Imm(self.kbits[v as usize]));
            self.stats.remats += 1;
            return r;
        }
        let r = self.reg[v as usize];
        if r != NOREG {
            cx.locked |= 1 << r;
            return r;
        }
        let h = if want != NOREG && self.free & (1 << want) != 0 {
            want
        } else {
            self.hint[v as usize]
        };
        self.prefer_callee = self.flags[v as usize] & F_CROSS != 0;
        let r = self.alloc_reg(class, h, cx.locked);
        self.prefer_callee = false;
        self.bind(v, r);
        cx.locked |= 1 << r;
        r
    }

    fn use_src(&mut self, v: u32, op: Op, ty: Ty, cx: &mut Cx) -> Src {
        if self.is_const(v) {
            let bits = self.kbits[v as usize];
            if self.t.imm_ok(op, ty, bits) {
                self.dec_use(v);
                return Src::Imm(bits);
            }
            if ty == Ty::F64 && self.t.mem_src_ok(op, ty) {
                // A float constant is a memory operand in the constant pool.
                self.dec_use(v);
                self.stats.pool_operands += 1;
                return Src::Pool(self.cb.pool(bits as u64));
            }
            if ty == Ty::I64 && op != Op::Guard && self.t.int_pool_ok(op) {
                // So is an integer too wide for an immediate, where the
                // instruction reads memory: the register and the `mov` that
                // would fill it at every use are saved.
                self.dec_use(v);
                self.stats.pool_operands += 1;
                return Src::Pool(self.cb.pool(bits as u64));
            }
            return Src::Reg(self.use_reg(v, cx));
        }
        let r = self.reg[v as usize];
        if r != NOREG {
            self.dec_use(v);
            cx.locked |= 1 << r;
            return Src::Reg(r);
        }
        if self.slot[v as usize] != NOSLOT && self.t.mem_src_ok(op, ty) {
            self.dec_use(v);
            return Src::Slot(self.slot[v as usize]);
        }
        Src::Reg(self.use_reg(v, cx))
    }

    /// First operand of a two-address instruction: it must sit in `dst`.
    fn tied(&mut self, a: u32, dst: Reg, cx: &mut Cx) -> Reg {
        if self.is_const(a) {
            self.dec_use(a);
            cx.pre(Loc::Reg(dst), Loc::Imm(self.kbits[a as usize]));
            self.stats.remats += 1;
            return dst;
        }
        let r = self.reg[a as usize];
        if r != NOREG {
            // The copy reads `r` before the instruction: nothing may take it.
            self.dec_use(a);
            cx.locked |= 1 << r;
            cx.pre(Loc::Reg(dst), Loc::Reg(r));
            return dst;
        }
        if self.alloc_mask & (1 << dst) != 0 {
            // Last use: `a` lives in the destination register until here.
            self.dec_use(a);
            self.take(dst);
            self.bind(a, dst);
            self.stats.hints += 1;
            self.stats.hint_hits += 1;
            return dst;
        }
        // The destination is the scratch register: bring `a` in by a copy.
        let r = self.use_reg(a, cx);
        cx.pre(Loc::Reg(dst), Loc::Reg(r));
        dst
    }

    fn weak_loc(&mut self, v: u32) -> Loc {
        self.stats.weak_uses += 1;
        if self.is_const(v) {
            return Loc::Imm(self.kbits[v as usize]);
        }
        let r = self.reg[v as usize];
        if r != NOREG {
            return Loc::Reg(r);
        }
        if self.slot[v as usize] == NOSLOT && self.cell_home(v).is_some() {
            // The value is in its cell already: a register would cost a load.
            return Loc::Slot(self.slot_of(v));
        }
        if self.slot[v as usize] == NOSLOT {
            // A value only a snapshot reads goes to a register when the file
            // has room to spare, so the loop does not store it on every pass.
            // The register is the first to go when something earlier needs
            // one: the value then gets a slot and a reload, as an evicted
            // value does. A value live across a call takes only a register
            // that survives calls: in one that does not, the call would
            // evict it, and the code after the call would reload it for the
            // stub alone. Without such a register the stub reads the slot.
            let class = self.val_class(v);
            let regs = self.free & self.class_mask(class);
            let cross = self.flags[v as usize] & F_CROSS != 0;
            if regs.count_ones() > WEAK_SPARE_REGS && (!cross || regs & self.callee_mask != 0) {
                let locked = if cross { !self.callee_mask } else { 0 };
                if let Some(r) = self.alloc_opt(class, self.hint[v as usize], locked) {
                    self.bind(v, r);
                    self.stats.weak_in_regs += 1;
                    return Loc::Reg(r);
                }
            }
            self.stats.weak_slots += 1;
        }
        Loc::Slot(self.slot_of(v))
    }

    /// Location for an exit that ends a region: a free register if there is
    /// one (the value then needs no spill), otherwise a slot.
    fn exit_loc(&mut self, v: u32) -> Loc {
        if self.is_const(v) || self.reg[v as usize] != NOREG {
            return self.weak_loc(v);
        }
        let class = self.val_class(v);
        if let Some(r) = self.alloc_opt(class, self.hint[v as usize], 0) {
            self.bind(v, r);
            self.stats.weak_uses += 1;
            return Loc::Reg(r);
        }
        self.weak_loc(v)
    }

    fn flush(&mut self, cx: &Cx) {
        for i in (0..cx.npre).rev() {
            let (d, s) = cx.pre[i];
            self.cb.ensure();
            self.t.mov(&mut self.cb, d, s);
        }
        self.free |= cx.temps;
    }

    fn emit_plan(&mut self) -> u32 {
        let (a, b, c) = (self.t.scratch_gpr(), self.t.scratch_fpr(), self.t.tmp_gpr());
        self.stats.cycles_broken += plan_moves(&mut self.moves, &mut self.plan, a, b, c);
        for i in (0..self.plan.len()).rev() {
            self.cb.ensure();
            match self.plan[i] {
                MoveOp::Mov(d, s) => self.t.mov(&mut self.cb, d, s),
                MoveOp::Swap(x, y) => self.t.swap(&mut self.cb, x, y),
            }
        }
        self.plan.len() as u32
    }

    fn emit_plan_stub(&mut self) -> u32 {
        let (a, b, c) = (self.t.scratch_gpr(), self.t.scratch_fpr(), self.t.tmp_gpr());
        self.stats.cycles_broken += plan_moves(&mut self.moves, &mut self.plan, a, b, c);
        for i in 0..self.plan.len() {
            match self.plan[i] {
                MoveOp::Mov(d, s) => self.t.mov(&mut self.stubs, d, s),
                MoveOp::Swap(x, y) => self.t.swap(&mut self.stubs, x, y),
            }
        }
        self.plan.len() as u32
    }

    // ---- compile driver ------------------------------------------------

    /// Compiles every region of `f`. The result is in [`Self::code_parts`].
    pub fn compile(&mut self, f: &Func) -> Stats {
        let divides = f
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::SDiv | Op::SRem | Op::UDiv | Op::URem));
        self.has_calls = f
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::Call | Op::CallCtl | Op::CallIf));
        let has_cold = f.insts.iter().any(|i| i.op == Op::CallIf);
        // Stack arguments of calls come first in the frame, then the cells.
        let mut out = 0;
        for i in &f.insts {
            if matches!(i.op, Op::Call | Op::CallCtl | Op::CallIf) {
                let (start, len) = if i.op == Op::CallIf {
                    (i.c, i.d)
                } else {
                    (i.a, i.b)
                };
                let mut counts = [0usize; 2];
                let mut stack = 0u32;
                for (pos, &a) in f.list(start, len).iter().enumerate() {
                    let class = Class::of(f.val_ty[a as usize]);
                    self.arg_dst(class, pos, &mut counts, &mut stack);
                }
                out = out.max(self.t.shadow_slots() + stack);
            }
        }
        self.out_slots = out;
        self.cells = f.cells;
        self.cell_read.clear();
        self.cell_read.resize(f.cells as usize, false);
        self.cell_dropped.clear();
        self.cell_dropped.resize(f.cells as usize, false);
        self.find_cell_homes(f);
        self.save0 = out + f.cells;
        self.base = self.save0 + if has_cold { 32 } else { 0 };
        let reserved = if divides { self.t.division_regs() } else { 0 };
        self.alloc_mask = self.base_alloc & !reserved;
        self.gpr_mask = self.alloc_mask & 0xFFFF;
        self.fpr_mask = self.alloc_mask & 0xFFFF_0000;
        self.callee_mask = self.alloc_mask & !self.caller_mask;
        self.prescan(f);
        self.find_side_cells(f);
        (self.entry_ctl, self.entry_area) = match f.insts.first() {
            Some(e) if e.op == Op::Entry => (e.a + 1, if e.b >= 3 { e.a + 2 } else { NONE }),
            _ => (NONE, NONE),
        };
        self.stats = Stats {
            insts: f.insts.len() as u32,
            ..Stats::default()
        };
        self.cb.reset(f.insts.len() * 64 + 2048);
        self.stubs.reset();
        self.used_callee = 0;
        self.max_slots = self.base + 2;
        let nsides = f.num_sides();
        self.side_locs.clear();
        self.side_loc_start.clear();
        self.side_top.clear();
        self.side_top.resize(nsides, 0);
        for k in 0..nsides {
            let (s, _) = f.region(1 + k);
            self.side_loc_start.push(self.side_locs.len() as u32);
            let n = f.insts[s].b;
            self.side_locs.extend(std::iter::repeat_n(DEAD, n as usize));
        }
        self.join_slots = 0;
        for k in 0..self.joins.len() {
            self.plan_join(f, self.joins[k]);
        }
        // A header parameter that starts as a join parameter is hinted to the
        // register the join receives it in, so the join's move into the
        // header is empty when the back edge allows it: the side's jump to the
        // join is then the only move a link makes.
        for r in 0..=nsides {
            let h = self.hdr_of[r];
            if h == usize::MAX {
                continue;
            }
            let h = f.insts[h];
            for k in 0..h.b {
                let start = f.lists[(h.c + k) as usize];
                let reg = self.jlin[start as usize];
                if reg != NOREG && reg != LIN_SLOT {
                    self.hint[(h.a + k) as usize] = reg;
                }
            }
        }
        for r in (1..=nsides).rev() {
            self.region(f, r);
        }
        self.region(f, 0);
        self.cb.patch_start();
        let s_len = self.stubs.bytes.len() as u32;
        self.cb.patch_pool(s_len);
        self.stubs.patch_pool(self.cb.pool.len() as u32);
        self.stats
    }

    fn prescan(&mut self, f: &Func) {
        let n = f.num_vals();
        macro_rules! fill {
            ($v:expr, $x:expr) => {{
                $v.clear();
                $v.resize(n, $x);
            }};
        }
        fill!(self.reg, NOREG);
        fill!(self.slot, NOSLOT);
        fill!(self.uses, 0);
        fill!(self.hint, NOREG);
        fill!(self.def_at, 0);
        fill!(self.flags, 0);
        fill!(self.cold_seen, 0);
        self.snap_cold.clear();
        self.snap_cold.resize(f.snaps.len(), SNAP_UNSEEN);
        self.cold.reset(n);
        fill!(self.kbits, 0);
        fill!(self.cell_of, NONE);
        fill!(self.lin, NOREG);
        fill!(self.loop_uses, 0);
        fill!(self.jlin, NOREG);
        for (v, t) in f.val_ty.iter().enumerate() {
            if *t == Ty::F64 {
                self.flags[v] |= F_FLOAT;
            }
        }
        self.reg_val = [NONE; 32];
        self.invariants.clear();
        self.hdr_idx = usize::MAX;
        self.hdr_of.clear();
        self.joins.clear();
        self.join_start.clear();
        self.inv_ranges.clear();
        for r in 0..=f.num_sides() {
            let (s, e) = f.region(r);
            let mut hdr = usize::MAX;
            self.join_start.push(self.joins.len() as u32);
            let inv_start = self.invariants.len();
            self.last_call = usize::MAX;
            self.loop_call = false;
            self.if_depth = 0;
            for idx in s..e {
                let i = f.insts[idx];
                if i.scale == COLD && is_coldable(i.op) {
                    // Read where a stub computes it, or where it stands when
                    // something else reads it after all: the operands are
                    // weak uses either way (see `weak_snap`).
                    self.def_at[i.dst as usize] = idx as u32;
                    self.flags[i.dst as usize] |= F_COLD;
                    assert!(
                        i.op != Op::Sar || self.flags[i.b as usize] & F_CONST != 0,
                        "a cold shift has a constant count"
                    );
                    self.weak(i.a, idx, hdr);
                    if is_binary(i.op) {
                        self.weak(i.b, idx, hdr);
                    }
                    continue;
                }
                match i.op {
                    Op::Entry | Op::SideEntry => {
                        for v in i.a..i.a + i.b {
                            self.def_at[v as usize] = idx as u32;
                        }
                    }
                    Op::Const => {
                        self.def_at[i.dst as usize] = idx as u32;
                        self.flags[i.dst as usize] |= F_CONST;
                        self.kbits[i.dst as usize] = i.imm;
                    }
                    Op::Load | Op::Load8 | Op::Load32 => {
                        self.def_at[i.dst as usize] = idx as u32;
                        self.strong(i.a, idx, hdr);
                        if i.b != NONE {
                            self.strong(i.b, idx, hdr);
                        }
                    }
                    Op::Store | Op::Store8 | Op::AddMem => {
                        self.strong(i.a, idx, hdr);
                        if i.b != NONE {
                            self.strong(i.b, idx, hdr);
                        }
                        self.strong(i.c, idx, hdr);
                    }
                    Op::FSqrt
                    | Op::I2F
                    | Op::F2I
                    | Op::Bits
                    | Op::FromBits
                    | Op::FNeg
                    | Op::FAbs
                    | Op::FFloor
                    | Op::FCeil
                    | Op::FTrunc => {
                        self.def_at[i.dst as usize] = idx as u32;
                        self.strong(i.a, idx, hdr);
                    }
                    Op::FNma => {
                        self.def_at[i.dst as usize] = idx as u32;
                        for v in [i.a, i.b, i.c] {
                            self.strong(v, idx, hdr);
                        }
                    }
                    Op::Select => {
                        self.def_at[i.dst as usize] = idx as u32;
                        for v in [i.a, i.b, i.c, i.d] {
                            self.strong(v, idx, hdr);
                        }
                    }
                    Op::Guard => {
                        if i.scale == 1 {
                            self.flags[i.a as usize] |= F_FLAGS;
                        }
                        self.strong(i.a, idx, hdr);
                        self.strong(i.b, idx, hdr);
                        self.note_snap(f, i.c);
                        for sv in f.snap_of(i.c) {
                            if !names_value(sv) {
                                continue;
                            }
                            if i.d == NONE {
                                self.weak_snap(f, sv.val, idx, hdr);
                            } else {
                                self.weak(sv.val, idx, hdr);
                            }
                        }
                    }
                    Op::Join => {
                        self.joins.push(idx);
                        for k in 0..i.b {
                            self.def_at[(i.a + k) as usize] = idx as u32;
                            self.strong(f.lists[(i.c + k) as usize], idx, usize::MAX);
                        }
                    }
                    Op::Rejoin => {
                        for &v in f.list(i.a, i.b) {
                            self.strong(v, idx, hdr);
                        }
                    }
                    Op::Header => {
                        hdr = idx;
                        for k in 0..i.b {
                            self.def_at[(i.a + k) as usize] = idx as u32;
                            self.strong(f.lists[(i.c + k) as usize], idx, usize::MAX);
                        }
                    }
                    Op::Back => {
                        for &v in f.list(i.a, i.b) {
                            self.strong(v, idx, hdr);
                        }
                        if self.loop_call {
                            for k in inv_start..self.invariants.len() {
                                let v = self.invariants[k] as usize;
                                self.flags[v] |= F_CROSS;
                            }
                        }
                    }
                    Op::Call | Op::CallCtl => {
                        for &v in f.list(i.a, i.b) {
                            self.strong(v, idx, hdr);
                        }
                        if i.dst != NONE {
                            self.def_at[i.dst as usize] = idx as u32;
                        }
                        if i.c != NONE {
                            self.def_at[i.c as usize] = idx as u32;
                        }
                        self.last_call = idx;
                        self.loop_call |= hdr != usize::MAX;
                    }
                    Op::CallIf => {
                        self.strong(i.a, idx, hdr);
                        self.strong(i.b, idx, hdr);
                        for &v in f.list(i.c, i.d) {
                            self.weak(v, idx, hdr);
                        }
                    }
                    Op::If => {
                        self.strong(i.a, idx, hdr);
                        self.strong(i.b, idx, hdr);
                        self.if_depth += 1;
                        // The skipped side's join values are read at the
                        // branch, like a snapshot.
                        let end = f.insts[i.e as usize];
                        for &v in f.list(end.c + end.b, end.b) {
                            self.weak(v, idx, hdr);
                        }
                    }
                    Op::EndIf => {
                        for k in 0..i.b {
                            self.def_at[(i.a + k) as usize] = idx as u32;
                            self.strong(f.lists[(i.c + k) as usize], idx, hdr);
                        }
                        self.if_depth = self.if_depth.saturating_sub(1);
                    }
                    Op::CellStore => self.strong(i.c, idx, hdr),
                    Op::CellLoad => {
                        self.def_at[i.dst as usize] = idx as u32;
                        self.cell_of[i.dst as usize] = i.imm as u32;
                    }
                    Op::ExitArea => self.def_at[i.dst as usize] = idx as u32,
                    Op::Exit => {
                        self.note_snap(f, i.c);
                        for sv in f.snap_of(i.c) {
                            if names_value(sv) {
                                self.weak_snap(f, sv.val, idx, hdr);
                            }
                        }
                    }
                    Op::Ret => {
                        self.strong(i.a, idx, hdr);
                        if i.b != NONE {
                            self.strong(i.b, idx, hdr);
                        }
                    }
                    Op::Goto => {
                        for sv in f.snap_of(i.c) {
                            self.weak(sv.val, idx, hdr);
                        }
                    }
                    _ => {
                        self.def_at[i.dst as usize] = idx as u32;
                        self.strong(i.a, idx, hdr);
                        self.strong(i.b, idx, hdr);
                    }
                }
            }
            self.hdr_of.push(hdr);
            self.inv_ranges.push((inv_start, self.invariants.len()));
        }
        self.join_start.push(self.joins.len() as u32);
    }

    fn note_inv(&mut self, v: u32, idx: usize, hdr: usize) {
        if hdr != usize::MAX
            && idx > hdr
            && (self.def_at[v as usize] as usize) < hdr
            && self.flags[v as usize] & (F_CONST | F_INV) == 0
        {
            self.flags[v as usize] |= F_INV;
            self.invariants.push(v);
        }
    }

    fn strong(&mut self, v: u32, idx: usize, hdr: usize) {
        self.uses[v as usize] += 1;
        if hdr != usize::MAX && idx > hdr && (self.def_at[v as usize] as usize) < hdr {
            self.loop_uses[v as usize] += if self.if_depth == 0 { 2 } else { 1 };
        }
        self.note_inv(v, idx, hdr);
        self.note_cross(v);
    }

    fn weak(&mut self, v: u32, idx: usize, hdr: usize) {
        self.flags[v as usize] |= F_WEAK;
        self.note_inv(v, idx, hdr);
        self.note_cross(v);
    }

    /// Notes whether snapshot `snap` reads a cold value. The values it reads
    /// are defined before the read, so their flags are final here.
    fn note_snap(&mut self, f: &Func, snap: u32) {
        let cold = f
            .snap_of(snap)
            .iter()
            .any(|sv| names_value(sv) && self.flags[sv.val as usize] & F_COLD != 0);
        let state = &mut self.snap_cold[snap as usize];
        *state = (*state).max(if cold { SNAP_COLD } else { SNAP_WARM });
    }

    /// A snapshot read of `v` by an exit stub: a cold value is computed in the
    /// stub, from operands that are read there, so those are the weak uses.
    fn weak_snap(&mut self, f: &Func, v: u32, idx: usize, hdr: usize) {
        // A cold value defined before the loop and read inside it is an
        // invariant like any other: computed once, before the loop, and read
        // where it stands when no stub can compute it.
        if self.flags[v as usize] & F_COLD == 0
            || (hdr != usize::MAX && idx > hdr && (self.def_at[v as usize] as usize) < hdr)
        {
            self.weak(v, idx, hdr);
            return;
        }
        // A cold value many others share is visited once for each read
        // site, not once for each path to it.
        if self.cold_seen[v as usize] == idx as u32 + 1 {
            return;
        }
        self.cold_seen[v as usize] = idx as u32 + 1;
        let d = f.insts[self.def_at[v as usize] as usize];
        self.weak_snap(f, d.a, idx, hdr);
        if is_binary(d.op) {
            self.weak_snap(f, d.b, idx, hdr);
        }
    }

    /// A use after a call of a value defined before it: live across the call.
    fn note_cross(&mut self, v: u32) {
        if self.last_call != usize::MAX && (self.def_at[v as usize] as usize) < self.last_call {
            self.flags[v as usize] |= F_CROSS;
        }
    }

    fn region(&mut self, f: &Func, r: usize) {
        let (start, end) = f.region(r);
        self.clear_bindings();
        self.in_loop = false;
        self.rotated = usize::MAX;
        self.rotated_if = usize::MAX;
        self.cur_region = r;
        self.hdr_idx = self.hdr_of[r];
        (self.inv_lo, self.inv_hi) = self.inv_ranges[r];
        // The first slot after the call and cell area keeps the exit-area
        // pointer and the next the control pointer; the `Join` slots follow.
        self.nslots = self.base + 2 + self.join_slots;
        self.if_frames.clear();
        self.if_saved.clear();
        self.if_dst.clear();
        if r == 0 {
            // Each argument is hinted to the register it arrives in.
            let entry = f.insts[0];
            let mut counts = [0usize; 2];
            let mut stack = 0u32;
            for k in 0..entry.b {
                let v = entry.a + k;
                let at = self.arg_dst(self.val_class(v), k as usize, &mut counts, &mut stack);
                if let Loc::Reg(r) = at {
                    self.hint[v as usize] = r;
                }
            }
        }
        self.side_hints(f, start, end);
        let mut i = end;
        while i > start {
            i -= 1;
            let inst = f.insts[i];
            self.cur_idx = i;
            self.cb.ensure();
            match inst.op {
                Op::Entry => self.do_entry(f, &inst),
                Op::SideEntry => self.do_side_entry(&inst, r - 1),
                Op::Const => {}
                Op::Load | Op::Load8 | Op::Load32 => self.do_load(&inst),
                Op::Bits | Op::FromBits => self.do_bits(&inst),
                Op::Store | Op::Store8 | Op::AddMem => self.do_store(&inst),
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::And
                | Op::Or
                | Op::Xor
                | Op::Shl
                | Op::Shr
                | Op::Sar
                | Op::FAdd
                | Op::FSub
                | Op::FMul
                | Op::FDiv
                | Op::FCopySign => self.do_alu(f, &inst),
                Op::FSqrt
                | Op::I2F
                | Op::F2I
                | Op::FNeg
                | Op::FAbs
                | Op::FFloor
                | Op::FCeil
                | Op::FTrunc => self.do_unary(&inst),
                Op::FNma => self.do_fnma(&inst),
                Op::SDiv | Op::SRem | Op::UDiv | Op::URem => self.do_div(&inst),
                Op::Select => self.do_select(&inst),
                Op::Guard => self.do_guard(f, &inst),
                Op::Header => self.do_header(f, &inst, None),
                Op::Join => {
                    let label = self
                        .joins
                        .binary_search(&i)
                        .expect("the prescan met the join");
                    self.do_header(f, &inst, Some(label as u32));
                }
                Op::Rejoin => self.do_rejoin(f, &inst),
                Op::Goto => self.do_goto(f, &inst),
                Op::CallIf => self.do_call_if(f, &inst),
                Op::If => self.do_if(f, &inst),
                Op::EndIf => self.do_end_if(f, &inst),
                Op::CellStore => self.do_cell_store(&inst),
                Op::CellLoad => self.do_cell_load(&inst),
                Op::ExitArea => self.do_exit_area(&inst),
                Op::Back => self.do_back(f, &inst),
                Op::Exit => self.do_exit(f, &inst),
                Op::Ret => self.do_ret(&inst),
                Op::Call | Op::CallCtl => self.do_call(f, &inst),
            }
        }
        self.max_slots = self.max_slots.max(self.nslots);
    }

    /// A parent's guard passes values to a side whose entry registers are
    /// already known: hint each value toward its register.
    fn side_hints(&mut self, f: &Func, start: usize, end: usize) {
        for idx in start..end {
            let i = f.insts[idx];
            if !matches!(i.op, Op::Guard | Op::Goto) || i.d == NONE {
                continue;
            }
            let base = self.side_loc_start[i.d as usize] as usize;
            for (k, sv) in f.snap_of(i.c).iter().enumerate() {
                if let Loc::Reg(r) = self.side_locs[base + k]
                    && !self.is_const(sv.val)
                    && self.hint[sv.val as usize] == NOREG
                {
                    self.hint[sv.val as usize] = r;
                }
            }
        }
    }

    // ---- instructions --------------------------------------------------

    fn live(&self, w: u32) -> bool {
        self.reg[w as usize] != NOREG || self.slot[w as usize] != NOSLOT
    }

    fn addr(
        &mut self,
        base: u32,
        index: u32,
        scale: u8,
        disp: i64,
        want: Reg,
        cx: &mut Cx,
    ) -> Addr {
        let b = self.use_reg_w(base, want, cx);
        let i = if index == NONE {
            NOREG
        } else {
            self.use_reg(index, cx)
        };
        Addr {
            base: b,
            index: i,
            scale: if i == NOREG { 1 } else { scale },
            disp: disp as i32,
        }
    }

    fn do_load(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let dst = self.def(i.dst, class_of(i.ty));
        let a = self.addr(i.a, i.b, i.scale, i.imm, dst, &mut cx);
        self.cb.ensure();
        if i.op == Op::Load8 {
            self.t.load8(&mut self.cb, dst, a);
        } else if i.op == Op::Load32 {
            self.t.load32(&mut self.cb, dst, a);
        } else {
            self.t.load(&mut self.cb, i.ty, dst, a);
        }
        self.flush(&cx);
    }

    /// A move between the register files.
    fn do_bits(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let dst = self.def(i.dst, class_of(i.ty));
        let src = self.use_reg(i.a, &mut cx);
        self.cb.ensure();
        self.t.mov(&mut self.cb, Loc::Reg(dst), Loc::Reg(src));
        self.flush(&cx);
    }

    fn do_store(&mut self, i: &Inst) {
        let mut cx = Cx::new();
        let src = if self.is_const(i.c) && i.ty == Ty::I64 {
            self.use_src(i.c, Op::Store, i.ty, &mut cx)
        } else {
            Src::Reg(self.use_reg(i.c, &mut cx))
        };
        let a = self.addr(i.a, i.b, i.scale, i.imm, NOREG, &mut cx);
        self.cb.ensure();
        if i.op == Op::AddMem {
            self.t.add_mem(&mut self.cb, a, src);
        } else if i.op == Op::Store8 {
            self.t.store8(&mut self.cb, a, src);
        } else {
            self.t.store(&mut self.cb, i.ty, a, src);
        }
        self.flush(&cx);
    }

    /// The float `Load` that defines `v` when the operation at the pass's
    /// place can read the word in memory instead: the operation is the
    /// load's only reader (no snapshot reads the value either), the load
    /// is in the same stretch of code and nothing between the two writes
    /// memory, calls or branches into another block. The address values are
    /// then read at the operation, where they hold what they held at the
    /// load, and the load is dropped as dead.
    fn fusable_load(&self, f: &Func, v: u32) -> Option<Inst> {
        const FAR: usize = 24;
        if self.is_const(v)
            || self.uses[v as usize] != 1
            || self.live(v)
            || self.flags[v as usize] & (F_WEAK | F_COLD | F_INV) != 0
        {
            return None;
        }
        let at = self.def_at[v as usize] as usize;
        let load = f.insts[at];
        if load.op != Op::Load
            || load.ty != Ty::F64
            || load.dst != v
            || at >= self.cur_idx
            || self.cur_idx - at > FAR
        {
            return None;
        }
        let pure = f.insts[at + 1..self.cur_idx].iter().all(|between| {
            matches!(
                between.op,
                Op::Const
                    | Op::Load
                    | Op::Load8
                    | Op::Load32
                    | Op::Add
                    | Op::Sub
                    | Op::Mul
                    | Op::And
                    | Op::Or
                    | Op::Xor
                    | Op::Shl
                    | Op::Shr
                    | Op::Sar
                    | Op::FAdd
                    | Op::FSub
                    | Op::FMul
                    | Op::FDiv
                    | Op::FSqrt
                    | Op::FNeg
                    | Op::FAbs
                    | Op::FFloor
                    | Op::FCeil
                    | Op::FTrunc
                    | Op::FCopySign
                    | Op::FNma
                    | Op::I2F
                    | Op::F2I
                    | Op::Bits
                    | Op::FromBits
                    | Op::Select
                    | Op::Guard
            )
        });
        pure.then_some(load)
    }

    fn do_alu(&mut self, f: &Func, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let class = class_of(i.ty);
        let mut cx = Cx::new();
        let dst = self.def(i.dst, class);
        let mut two = self.t.two_address(i.op, i.ty);
        let (mut a, mut b) = (i.a, i.b);
        // A shift by a register is three-address (BMI2): the destination is
        // free of the value shifted and of the count.
        let register_shift = matches!(i.op, Op::Shl | Op::Shr | Op::Sar) && !self.is_const(i.b);
        let shift = matches!(i.op, Op::Shl | Op::Shr | Op::Sar) && !register_shift;
        if register_shift {
            two = false;
        }
        let commutative = matches!(
            i.op,
            Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor | Op::FAdd | Op::FMul
        );
        // A float operand that a load made for this operation alone is read
        // in memory.
        let mut fused = None;
        if matches!(i.op, Op::FAdd | Op::FSub | Op::FMul | Op::FDiv)
            && a != b
            && self.t.mem_src_ok(i.op, Ty::F64)
        {
            fused = self.fusable_load(f, b);
            if fused.is_none() && commutative {
                fused = self.fusable_load(f, a);
                if fused.is_some() {
                    std::mem::swap(&mut a, &mut b);
                }
            }
        }
        if commutative && a != b && fused.is_none() {
            let a_last = !self.is_const(a) && self.reg[a as usize] == NOREG;
            let b_last = !self.is_const(b) && self.reg[b as usize] == NOREG;
            if (!a_last && b_last) || (self.is_const(a) && !self.is_const(b)) {
                std::mem::swap(&mut a, &mut b);
            }
        }
        // A product with a constant is `imul r, r/m, imm`, and a sum with one
        // is `lea`: each reads one register and writes another. A sum that
        // a flags-only guard reads stays an `add`, `lea` sets no flags.
        if two
            && self.flags[i.dst as usize] & F_FLAGS == 0
            && matches!(i.op, Op::Mul | Op::Add)
            && i.ty == Ty::I64
            && !self.is_const(a)
            && self.is_const(b)
            && self.t.imm_ok(i.op, i.ty, self.kbits[b as usize])
        {
            two = false;
        }
        // A sum of two values that both live on is `lea` too: a two-address
        // `add` would first copy one of them into the destination.
        let sum_of_live = two
            && i.op == Op::Add
            && i.ty == Ty::I64
            && self.flags[i.dst as usize] & F_FLAGS == 0
            && !self.is_const(a)
            && !self.is_const(b)
            && self.reg[a as usize] != NOREG
            && self.reg[a as usize] != dst
            && self.reg[b as usize] != NOREG
            && self.reg[b as usize] != dst
            && self.t.three_address_sum();
        if sum_of_live {
            two = false;
        }
        if two {
            cx.locked |= 1 << dst;
        }
        let ra = if two {
            self.tied(a, dst, &mut cx)
        } else {
            self.use_reg_w(a, dst, &mut cx)
        };
        let sb = if let Some(load) = fused {
            self.dec_use(b);
            self.stats.fused_loads += 1;
            Src::Mem(self.addr(load.a, load.b, load.scale, load.imm, NOREG, &mut cx))
        } else if sum_of_live {
            Src::Reg(self.use_reg(b, &mut cx))
        } else if register_shift {
            assert!(
                self.t.register_shifts(),
                "a shift by a register needs a target that has it"
            );
            Src::Reg(self.use_reg(b, &mut cx))
        } else if shift {
            assert!(self.is_const(b), "a shift count must be a constant");
            self.dec_use(b);
            Src::Imm(self.kbits[b as usize] & 63)
        } else {
            self.use_src(b, i.op, i.ty, &mut cx)
        };
        self.cb.ensure();
        self.t.alu(&mut self.cb, i.op, dst, ra, sb);
        self.flush(&cx);
    }

    /// `c - a * b`: the destination holds a factor, as `vfnmadd213sd` has it.
    fn do_fnma(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let dst = self.def(i.dst, Class::Fpr);
        cx.locked |= 1 << dst;
        // The product is commutative: the factor the allocator already holds
        // in a register is read, and the other is the destination's.
        let (mut a, mut b) = (i.a, i.b);
        let a_last = !self.is_const(a) && self.reg[a as usize] == NOREG;
        let b_last = !self.is_const(b) && self.reg[b as usize] == NOREG;
        if !a_last && b_last {
            std::mem::swap(&mut a, &mut b);
        }
        let ra = self.tied(a, dst, &mut cx);
        debug_assert_eq!(ra, dst);
        let rb = self.use_reg(b, &mut cx);
        let sc = self.use_src(i.c, Op::FNma, Ty::F64, &mut cx);
        self.cb.ensure();
        self.t.fnma(&mut self.cb, dst, rb, sc);
        self.flush(&cx);
    }

    fn do_div(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let dst = self.def(i.dst, Class::Gpr);
        let a = self.use_reg(i.a, &mut cx);
        // A constant divisor is read from the pool and a spilled one from its
        // slot: `idiv` takes memory, and no immediate.
        let b = if self.is_const(i.b) {
            self.dec_use(i.b);
            self.stats.pool_operands += 1;
            Src::Pool(self.cb.pool(self.kbits[i.b as usize] as u64))
        } else if self.reg[i.b as usize] == NOREG && self.slot[i.b as usize] != NOSLOT {
            self.dec_use(i.b);
            Src::Slot(self.slot[i.b as usize])
        } else {
            Src::Reg(self.use_reg(i.b, &mut cx))
        };
        self.cb.ensure();
        self.t.divide(&mut self.cb, i.op, dst, a, b);
        self.flush(&cx);
    }

    fn do_unary(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let class = if i.op == Op::F2I {
            Class::Gpr
        } else {
            Class::Fpr
        };
        let dst = self.def(i.dst, class);
        let src = if i.op == Op::FSqrt {
            self.use_src(i.a, Op::FSqrt, Ty::F64, &mut cx)
        } else {
            Src::Reg(self.use_reg(i.a, &mut cx))
        };
        self.cb.ensure();
        self.t.unary(&mut self.cb, i.op, dst, src);
        self.flush(&cx);
    }

    fn do_select(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let float_cmp = self.val_class(i.a) == Class::Fpr;
        let class = class_of(i.ty);
        let mut cx = Cx::new();
        if class == Class::Gpr && !float_cmp {
            let dst = self.def(i.dst, Class::Gpr);
            cx.locked |= 1 << dst;
            let (mut a, mut b, mut cc) = (i.a, i.b, i.cc);
            if self.is_const(a) && !self.is_const(b) {
                std::mem::swap(&mut a, &mut b);
                cc = cc.swap();
            }
            // The false value goes in the destination, the true one is cmov'ed in.
            let _ = self.tied(i.d, dst, &mut cx);
            let ra = self.use_reg(a, &mut cx);
            let sb = self.use_src(b, Op::Guard, Ty::I64, &mut cx);
            let st = if self.is_const(i.c) {
                Src::Reg(self.use_reg(i.c, &mut cx))
            } else {
                self.use_src(i.c, Op::Select, Ty::I64, &mut cx)
            };
            self.cb.ensure();
            self.t.select(&mut self.cb, cc, dst, ra, sb, st);
            self.flush(&cx);
            return;
        }
        // Float arms or a float compare: everything in registers, the result
        // built in the scratch registers.
        let dst = self.def(i.dst, class);
        let (mut a, mut b, mut cc) = (i.a, i.b, i.cc);
        let cmp_ty = if float_cmp { Ty::F64 } else { Ty::I64 };
        if float_cmp {
            if self.t.fcmp_swapped(cc) {
                std::mem::swap(&mut a, &mut b);
            }
        } else if self.is_const(a) && !self.is_const(b) {
            std::mem::swap(&mut a, &mut b);
            cc = cc.swap();
        }
        let ra = self.use_reg(a, &mut cx);
        let sb = if float_cmp {
            Src::Reg(self.use_reg(b, &mut cx))
        } else {
            self.use_src(b, Op::Guard, Ty::I64, &mut cx)
        };
        let rt = self.use_reg(i.c, &mut cx);
        let rf = self.use_reg(i.d, &mut cx);
        self.cb.ensure();
        self.t
            .select_any(&mut self.cb, i.ty, cmp_ty, cc, dst, ra, sb, rt, rf);
        self.flush(&cx);
    }

    fn do_guard(&mut self, f: &Func, i: &Inst) {
        let mut cx = Cx::new();
        // The registers the back edge's moves write before this guard's
        // branch (see `moves_precede`): no value takes one here.
        let mut reserved = 0u32;
        if self.rotated == self.cur_idx {
            for step in &self.back_plan {
                if let MoveOp::Mov(Loc::Reg(d), _) = step {
                    reserved |= 1 << d;
                }
            }
            debug_assert_eq!(self.free & reserved, reserved);
            self.free &= !reserved;
        }
        let (mut a, mut b, mut cc) = (i.a, i.b, i.cc);
        if i.ty == Ty::F64 {
            if self.t.fcmp_swapped(cc) {
                std::mem::swap(&mut a, &mut b);
            }
        } else if self.is_const(a) && !self.is_const(b) {
            std::mem::swap(&mut a, &mut b);
            cc = cc.swap();
        } else if !self.is_const(a)
            && !self.is_const(b)
            && self.reg[a as usize] == NOREG
            && self.slot[a as usize] != NOSLOT
            && self.reg[b as usize] != NOREG
            && self.t.mem_src_ok(Op::Guard, i.ty)
        {
            // The first operand is in its slot and the second in a register:
            // compared the other way round, the slot is the memory operand
            // and nothing is reloaded.
            std::mem::swap(&mut a, &mut b);
            cc = cc.swap();
        }
        // A guard of a value against zero that follows the instruction which
        // defined it needs no compare (see `Op::Guard`).
        let flags_only = i.scale == 1
            && i.ty == Ty::I64
            && self.is_const(i.b)
            && self.kbits[i.b as usize] == 0
            && !self.is_const(i.a);
        let byte = if flags_only {
            None
        } else {
            self.byte_guard(f, i.ty, Some(i.c), a, b, cc, &mut cx)
        };
        let (ra, sb) = if flags_only || byte.is_some() {
            self.dec_use(i.a);
            self.dec_use(i.b);
            (NOREG, Src::Imm(0))
        } else {
            let ra = self.use_reg(a, &mut cx);
            let wide = i.ty == Ty::I64
                && self.is_const(b)
                && !self.t.imm_ok(Op::Guard, i.ty, self.kbits[b as usize])
                && self.t.int_pool_ok(Op::Guard);
            let sb = if wide {
                // A constant too wide for an immediate is compared with in
                // the constant pool. Only a guard does so: a select and a
                // skip compare with a register.
                self.dec_use(b);
                self.stats.pool_operands += 1;
                Src::Pool(self.cb.pool(self.kbits[b as usize] as u64))
            } else {
                self.use_src(b, Op::Guard, i.ty, &mut cx)
            };
            (ra, sb)
        };
        // Snapshot reads come last: they see the final state at the branch.
        let tgt = if i.d == NONE {
            Tgt::Stub(self.exit_stub_for(f, i.c, false))
        } else {
            let base = self.side_loc_start[i.d as usize] as usize;
            self.moves.clear();
            for (k, sv) in f.snap_of(i.c).iter().enumerate() {
                let dst = self.side_locs[base + k];
                if dst != DEAD {
                    let src = self.weak_loc(sv.val);
                    self.moves.push((dst, src));
                }
            }
            self.moves.retain(|(d, s)| d != s);
            if self.moves.is_empty() {
                self.stats.side_direct += 1;
                Tgt::Body(self.side_top[i.d as usize])
            } else {
                self.stats.trampolines += 1;
                let s = self.stubs.mark();
                let n = self.emit_plan_stub();
                self.stats.trampoline_moves += n;
                self.t
                    .jump(&mut self.stubs, Tgt::Body(self.side_top[i.d as usize]));
                Tgt::Stub(s)
            }
        };
        self.cb.ensure();
        // The guard that closes a loop with no moves on its back edge: the
        // branch goes to the header when the guard holds, the exit follows.
        let (cc, tgt) = if self.rotated == self.cur_idx {
            self.t.jump(&mut self.cb, tgt);
            self.cb.ensure();
            (cc.negate(false), Tgt::Label)
        } else {
            (cc, tgt)
        };
        if flags_only {
            self.t.guard_flags(&mut self.cb, cc, tgt);
        } else if let Some((addr, imm)) = byte {
            self.t.guard_byte(&mut self.cb, cc, addr, imm, tgt);
        } else {
            self.t.guard(&mut self.cb, i.ty, cc, ra, sb, tgt);
        }
        if reserved != 0 {
            debug_assert_eq!(cx.npre, 0, "the rotated guard loads nothing");
            for k in (0..self.back_plan.len()).rev() {
                self.cb.ensure();
                if let MoveOp::Mov(d, s) = self.back_plan[k] {
                    self.t.mov(&mut self.cb, d, s);
                }
            }
            self.free |= reserved;
        }
        self.flush(&cx);
    }

    /// The address and the constant of a guard that tests a byte of memory
    /// for equality with a constant, when the guard can compare the byte
    /// where it is: the guard is the only reader of a `Load8` that stands
    /// right before it (constants apart), so the registers of the address
    /// hold at the guard what they held at the load, and the load itself is
    /// dropped as dead. The loaded byte is zero-extended, so a constant
    /// outside `0..=127` is left to the full compare.
    ///
    /// An `If` that skips its block on such a test compares the byte the
    /// same way (`snap` is `None`: it has no snapshot).
    fn byte_guard(
        &mut self,
        f: &Func,
        ty: Ty,
        snap: Option<u32>,
        a: u32,
        b: u32,
        cc: Cc,
        cx: &mut Cx,
    ) -> Option<(Addr, u8)> {
        let load = self.byte_load(f, ty, snap, a, b, cc)?;
        let addr = self.addr(load.a, load.b, load.scale, load.imm, NOREG, cx);
        self.stats.byte_guards += 1;
        Some((addr, self.kbits[b as usize] as u8))
    }

    /// The `Load8` a test of `a` against the constant `b` can read in place
    /// (see [`Self::byte_guard`]), if there is one.
    fn byte_load(
        &self,
        f: &Func,
        ty: Ty,
        snap: Option<u32>,
        a: u32,
        b: u32,
        cc: Cc,
    ) -> Option<Inst> {
        if ty != Ty::I64
            || !matches!(cc, Cc::Eq | Cc::Ne)
            || self.is_const(a)
            || !self.is_const(b)
            || !(0..=127).contains(&self.kbits[b as usize])
            || self.uses[a as usize] != 1
            || self.live(a)
            || self.flags[a as usize] & (F_WEAK | F_COLD | F_INV) != 0
        {
            return None;
        }
        let at = self.def_at[a as usize] as usize;
        let load = f.insts[at];
        if load.op != Op::Load8
            || load.dst != a
            || at >= self.cur_idx
            || f.insts[at + 1..self.cur_idx]
                .iter()
                .any(|between| between.op != Op::Const)
            || snap.is_some_and(|snap| {
                f.snap_of(snap)
                    .iter()
                    .any(|sv| names_value(sv) && sv.val == a)
            })
        {
            return None;
        }
        Some(load)
    }

    fn do_exit(&mut self, f: &Func, i: &Inst) {
        let s = self.exit_stub_for(f, i.c, true);
        self.cb.ensure();
        self.t.jump(&mut self.cb, Tgt::Stub(s));
    }

    /// Where a result of `Ret` is read: nothing is bound at the end of a
    /// region, so a value takes a free register (`want` when it is free:
    /// the register it is returned in) and is computed there.
    fn ret_loc(&mut self, v: u32, want: Reg) -> Loc {
        self.dec_use(v);
        if self.is_const(v) {
            return Loc::Imm(self.kbits[v as usize]);
        }
        let r = self.reg[v as usize];
        if r != NOREG {
            return Loc::Reg(r);
        }
        let class = self.val_class(v);
        let hint = if want != NOREG {
            want
        } else {
            self.hint[v as usize]
        };
        if let Some(r) = self.alloc_opt(class, hint, 0) {
            self.bind(v, r);
            return Loc::Reg(r);
        }
        Loc::Slot(self.slot_of(v))
    }

    fn do_ret(&mut self, i: &Inst) {
        // The second result first, so it can have the register it goes to.
        let second = (i.b != NONE).then(|| {
            let class = self.val_class(i.b);
            let want = self.t.ret2_reg(class);
            let want = if self.alloc_mask & (1 << want) != 0 {
                want
            } else {
                NOREG
            };
            (self.ret_loc(i.b, want), class)
        });
        let first = self.ret_loc(i.a, NOREG);
        self.cb.ensure();
        self.t.ret(&mut self.cb, first, second);
    }

    fn do_call(&mut self, f: &Func, i: &Inst) {
        let mut result = NOREG;
        if i.dst != NONE && self.live(i.dst) {
            result = self.def(i.dst, class_of(i.ty));
        }
        // A second result is moved from its return register with the first,
        // in one parallel move: either may be where the other goes.
        self.moves.clear();
        if i.c != NONE && self.live(i.c) {
            let class = self.val_class(i.c);
            let src = Loc::Reg(self.t.ret2_reg(class));
            let r = self.reg[i.c as usize];
            let s = self.slot[i.c as usize];
            if r != NOREG {
                self.unbind(i.c);
                if s != NOSLOT && self.cell_home(i.c) != Some(s) {
                    self.cb.ensure();
                    self.t.mov(&mut self.cb, Loc::Slot(s), Loc::Reg(r));
                    self.stats.def_spill_stores += 1;
                }
                self.moves.push((Loc::Reg(r), src));
            } else {
                self.moves.push((Loc::Slot(s), src));
            }
        }
        // Everything the callee may clobber is dead across the call: what is
        // still bound there goes to its slot, and is read back after the call.
        let mut occ = 0u32;
        for r in 0..32u8 {
            if self.reg_val[r as usize] != NONE && self.caller_mask & (1 << r) != 0 {
                occ |= 1 << r;
            }
        }
        while occ != 0 {
            let r = occ.trailing_zeros() as Reg;
            occ &= occ - 1;
            let v = self.reg_val[r as usize];
            let s = self.slot_of(v);
            self.cb.ensure();
            self.t.mov(&mut self.cb, Loc::Reg(r), Loc::Slot(s));
            self.stats.reloads += 1;
            self.stats.evictions += 1;
            self.unbind(v);
        }
        let ret = self.t.ret_reg(class_of(i.ty));
        if result != NOREG && result != ret {
            self.moves.push((Loc::Reg(result), Loc::Reg(ret)));
        }
        if !self.moves.is_empty() {
            self.emit_plan();
        }
        self.cb.ensure();
        if i.op == Op::CallCtl {
            self.t.call_ctl(&mut self.cb, self.base + 1, i.imm as i32);
        } else if i.scale == CALL_REL {
            self.t.call_rel(
                &mut self.cb,
                Tgt::Start(i32::try_from(i.imm).expect("a near call")),
            );
        } else {
            self.t.call(&mut self.cb, i.imm as u64);
        }
        self.moves.clear();
        let mut counts = [0usize; 2];
        let mut stack = 0u32;
        for (pos, &a) in f.list(i.a, i.b).iter().enumerate() {
            let class = self.val_class(a);
            let dst = self.arg_dst(class, pos, &mut counts, &mut stack);
            self.dec_use(a);
            let src = if self.is_const(a) {
                Loc::Imm(self.kbits[a as usize])
            } else if self.reg[a as usize] != NOREG {
                Loc::Reg(self.reg[a as usize])
            } else {
                self.prefer_callee = self.flags[a as usize] & F_CROSS != 0;
                let hint = if let Loc::Reg(r) = dst { r } else { NOREG };
                let got = self.alloc_opt(class, hint, 0);
                self.prefer_callee = false;
                if let Some(r) = got {
                    self.bind(a, r);
                    Loc::Reg(r)
                } else {
                    Loc::Slot(self.slot_of(a))
                }
            };
            self.moves.push((dst, src));
        }
        self.emit_plan();
    }

    /// Where the call argument at `pos` goes: the ABI register for its
    /// position (counted over every argument, or over those of its class,
    /// as the target says), else the next stack argument at the bottom of the
    /// frame, above the slots the callee may use as its own. `counts` and
    /// `stack` carry the count over one call's arguments.
    fn arg_dst(&self, class: Class, pos: usize, counts: &mut [usize; 2], stack: &mut u32) -> Loc {
        let k = usize::from(class == Class::Fpr);
        let nth = if self.t.positional_args() {
            pos
        } else {
            counts[k]
        };
        counts[k] += 1;
        let regs = self.t.arg_regs(class);
        if nth < regs.len() {
            Loc::Reg(regs[nth])
        } else {
            *stack += 1;
            Loc::Slot(self.t.shadow_slots() + *stack - 1)
        }
    }

    // ---- region boundaries ---------------------------------------------

    /// Whether a function whose `Entry` takes arguments of the types `tys`
    /// finds each in a register of the target's ABI (see [`Op::Entry`]).
    #[must_use]
    pub fn entry_fits(&self, tys: &[Ty]) -> bool {
        let mut counts = [0usize; 2];
        let mut stack = 0u32;
        tys.iter().enumerate().all(|(k, &ty)| {
            matches!(
                self.arg_dst(class_of(ty), k, &mut counts, &mut stack),
                Loc::Reg(_)
            )
        })
    }

    fn do_entry(&mut self, _f: &Func, i: &Inst) {
        let er = self.t.entry_regs();
        self.moves.clear();
        let mut counts = [0usize; 2];
        let mut stack = 0u32;
        for k in 0..i.b {
            let v = i.a + k;
            let Loc::Reg(arg) =
                self.arg_dst(self.val_class(v), k as usize, &mut counts, &mut stack)
            else {
                panic!("entry argument {k} is on the stack");
            };
            debug_assert!(k >= 3 || arg == er[k as usize]);
            if self.reg[v as usize] != NOREG {
                self.moves
                    .push((Loc::Reg(self.reg[v as usize]), Loc::Reg(arg)));
            }
            let s = self.slot[v as usize];
            // The control and exit-area pointers' slots are stored below.
            if s != NOSLOT && self.entry_home(v) != Some(s) {
                self.moves.push((Loc::Slot(s), Loc::Reg(arg)));
                self.stats.def_spill_stores += 1;
            }
        }
        // Slot 0 keeps the exit-area pointer for the stubs, slot 1 the
        // control block's.
        self.moves.push((Loc::Slot(self.base), Loc::Reg(er[2])));
        self.moves.push((Loc::Slot(self.base + 1), Loc::Reg(er[1])));
        self.emit_plan();
        self.max_slots = self.max_slots.max(self.nslots);
        let callee = self.used_callee | self.t.always_saved();
        let mut frame = (8 * self.max_slots).next_multiple_of(16) + self.t.save_area(callee);
        if self.has_calls {
            frame = self.t.call_frame(frame, callee);
        }
        self.cb.ensure();
        self.t.prologue(&mut self.cb, frame, callee);
        let s_epi = self.stubs.mark();
        self.t.epilogue(&mut self.stubs, frame, callee);
        self.stubs.patch_epilogue(s_epi);
        self.cb.patch_epilogue(s_epi);
    }

    fn do_side_entry(&mut self, i: &Inst, side: usize) {
        let base = self.side_loc_start[side] as usize;
        for k in 0..i.b {
            let v = i.a + k;
            let r = self.reg[v as usize];
            let s = self.slot[v as usize];
            // A parameter that lives in the cell every entry hands it from
            // (`find_side_cells`) is there already: no entry moves it, and
            // nothing stores it.
            let homed = s != NOSLOT && self.cell_home(v) == Some(s);
            self.side_locs[base + k as usize] = if r != NOREG {
                Loc::Reg(r)
            } else if s != NOSLOT && !homed {
                Loc::Slot(s)
            } else {
                DEAD
            };
            if r != NOREG && s != NOSLOT && !homed {
                self.cb.ensure();
                self.t.mov(&mut self.cb, Loc::Slot(s), Loc::Reg(r));
                self.stats.def_spill_stores += 1;
            }
        }
        self.side_top[side] = self.cb.mark();
    }

    fn lin_loc(&self, v: u32, join: bool) -> Option<Loc> {
        if join {
            return match self.jlin[v as usize] {
                NOREG => None,
                LIN_SLOT => Some(Loc::Slot(self.slot[v as usize])),
                r => Some(Loc::Reg(r)),
            };
        }
        match self.lin[v as usize] {
            NOREG => None,
            LIN_SLOT => Some(Loc::Slot(self.slot[v as usize])),
            r => Some(Loc::Reg(r)),
        }
    }

    /// First thing a looping region does: choose where live-through values
    /// arrive on the back edge, and move the arguments there.
    fn do_back(&mut self, f: &Func, i: &Inst) {
        let h = f.insts[self.hdr_idx];
        self.in_loop = true;
        let mut nparams = [0usize; 2];
        // The registers the live parameters are hinted to (those a `Join`
        // hands them in): an invariant pinned in one would put the parameter
        // elsewhere, and every link would move it there.
        let mut hinted = 0u32;
        for k in 0..h.b {
            let p = h.a + k;
            if self.uses[p as usize] > 0 || self.flags[p as usize] & F_WEAK != 0 {
                nparams[usize::from(self.val_class(p) == Class::Fpr)] += 1;
                if self.hint[p as usize] != NOREG {
                    hinted |= 1 << self.hint[p as usize];
                }
            }
        }
        self.pin_invariants(nparams, hinted);
        let mut taken = 0u32;
        for k in 0..h.b {
            let p = h.a + k;
            if self.uses[p as usize] == 0 && self.flags[p as usize] & F_WEAK == 0 {
                continue;
            }
            let class = class_of(f.val_ty[p as usize]);
            let avail = self.free & self.class_mask(class) & !taken;
            if avail != 0 {
                let pref = avail & self.caller_mask;
                let hint = self.hint[p as usize];
                let r = if hint != NOREG && avail & (1 << hint) != 0 {
                    self.stats.hints += 1;
                    self.stats.hint_hits += 1;
                    hint
                } else {
                    (if pref != 0 { pref } else { avail }).trailing_zeros() as Reg
                };
                taken |= 1 << r;
                self.used_callee |= (1 << r) & self.callee_mask;
                self.lin[p as usize] = r;
                self.hint[p as usize] = r;
            } else {
                self.slot_of(p);
                self.lin[p as usize] = LIN_SLOT;
            }
        }
        self.edge_moves(f, i.a, h.b, h.a, false);
        let (a, b, c) = (self.t.scratch_gpr(), self.t.scratch_fpr(), self.t.tmp_gpr());
        self.stats.cycles_broken += plan_moves(&mut self.moves, &mut self.plan, a, b, c);
        self.stats.backedge_moves += self.plan.len() as u32;
        // The guard that closes the iteration (the poll guard of a trace)
        // branches to the header when it holds and the jump goes to its
        // exit: one instruction an iteration, not two. With nothing to move
        // that is always possible, and with moves when they can be made
        // before the branch.
        let closing = self.cur_idx.wrapping_sub(1);
        self.back_plan.clear();
        if self.cur_idx > self.hdr_idx + 1
            && f.insts[closing].op == Op::Guard
            && f.insts[closing].ty == Ty::I64
            && f.insts[closing].d == NONE
            && (self.plan.is_empty() || self.moves_precede(&f.insts[closing]))
        {
            if !self.plan.is_empty() {
                self.back_plan.extend_from_slice(&self.plan);
                self.stats.rotated_with_moves += 1;
            }
            self.rotated = closing;
            self.stats.rotated_backs += 1;
            return;
        }
        // A skip that ends here and joins nothing: the path that skips the
        // block finds the registers as the header wants them, so its branch
        // can go to the header. The jump below stays for the block's end.
        if self.plan.is_empty()
            && self.cur_idx > self.hdr_idx + 1
            && f.insts[closing].op == Op::EndIf
            && f.insts[closing].b == 0
        {
            self.rotated_if = closing;
        }
        self.cb.ensure();
        self.t.jump(&mut self.cb, Tgt::Label);
        for k in (0..self.plan.len()).rev() {
            self.cb.ensure();
            match self.plan[k] {
                MoveOp::Mov(d, s) => self.t.mov(&mut self.cb, d, s),
                MoveOp::Swap(x, y) => self.t.swap(&mut self.cb, x, y),
            }
        }
    }

    /// Whether the moves of the back edge (`self.plan`) can be made before
    /// the branch of the guard `g` that closes the loop, so the branch goes
    /// to the header. The exit of the guard then runs after the moves.
    ///
    /// Every move copies a register into a register that holds nothing here,
    /// at the end of the body. Such a copy changes no flags, so it may stand
    /// between a flags-only guard and the instruction that set the flags.
    /// What the guard and its exit read is a constant, a slot or a register
    /// that holds a value here, so none of it is a destination, as long as
    /// the guard gives no value a register of its own: its operands are in
    /// registers already (or an immediate), and `do_guard` keeps the
    /// destinations from the values only its snapshot reads.
    fn moves_precede(&self, g: &Inst) -> bool {
        let copies = self.plan.iter().all(|step| {
            matches!(step, MoveOp::Mov(Loc::Reg(d), Loc::Reg(_)) if self.free & (1 << d) != 0)
        });
        if !copies {
            return false;
        }
        let flags_only = g.scale == 1
            && self.is_const(g.b)
            && self.kbits[g.b as usize] == 0
            && !self.is_const(g.a);
        if flags_only {
            return true;
        }
        let (mut a, mut b) = (g.a, g.b);
        if self.is_const(a) && !self.is_const(b) {
            std::mem::swap(&mut a, &mut b);
        }
        let held = |v: u32| !self.is_const(v) && self.reg[v as usize] != NOREG;
        held(a)
            && (held(b)
                || (self.is_const(b) && self.t.imm_ok(Op::Guard, g.ty, self.kbits[b as usize])))
    }

    fn pin_invariants(&mut self, nparams: [usize; 2], hinted: u32) {
        for class in [Class::Gpr, Class::Fpr] {
            self.cand.clear();
            for k in self.inv_lo..self.inv_hi {
                let v = self.invariants[k];
                if self.val_class(v) == class {
                    self.cand.push(v);
                }
            }
            let total = self.class_mask(class).count_ones() as usize;
            let cap = total.saturating_sub(4 + nparams[usize::from(class == Class::Fpr)]);
            if self.cand.len() > cap {
                // The registers go to the invariants the loop reads most: a
                // read before the loop costs nothing here, one in a block
                // the loop may skip half of one it always runs, and a
                // snapshot's nothing.
                let (uses, loop_uses) = (&self.uses, &self.loop_uses);
                self.cand
                    .sort_by_key(|&v| std::cmp::Reverse((loop_uses[v as usize], uses[v as usize])));
            }
            for k in 0..self.cand.len() {
                let v = self.cand[k];
                if k < cap {
                    let hint = self.hint[v as usize];
                    let r = self
                        .alloc_opt(class, hint, hinted)
                        .or_else(|| self.alloc_opt(class, hint, 0))
                        .expect("pin has a register");
                    self.bind(v, r);
                    self.lin[v as usize] = r;
                    self.hint[v as usize] = r;
                    self.stats.pinned_invariants += 1;
                } else {
                    self.slot_of(v);
                    self.lin[v as usize] = LIN_SLOT;
                }
            }
        }
    }

    /// Moves `args` (a list in `f.lists`) into the incoming locations of the
    /// parameters starting at `first`, as one parallel move.
    fn edge(&mut self, f: &Func, list: u32, n: u32, first: u32, join: bool) {
        self.edge_moves(f, list, n, first, join);
        self.emit_plan();
    }

    /// The parallel move of [`Self::edge`] in `self.moves`, with a place
    /// given to each argument, and nothing emitted.
    fn edge_moves(&mut self, f: &Func, list: u32, n: u32, first: u32, join: bool) {
        self.moves.clear();
        for k in 0..n {
            let p = first + k;
            let Some(dst) = self.lin_loc(p, join) else {
                continue;
            };
            let a = f.lists[(list + k) as usize];
            self.dec_use(a);
            let src = if self.is_const(a) {
                Loc::Imm(self.kbits[a as usize])
            } else if self.reg[a as usize] != NOREG {
                Loc::Reg(self.reg[a as usize])
            } else {
                let class = self.val_class(a);
                let hint = if let Loc::Reg(r) = dst {
                    r
                } else {
                    self.hint[a as usize]
                };
                if let Some(r) = self.alloc_opt(class, hint, 0) {
                    self.bind(a, r);
                    Loc::Reg(r)
                } else {
                    Loc::Slot(self.slot_of(a))
                }
            };
            self.moves.push((dst, src));
        }
    }

    /// Chooses where each parameter of the `Join` arrives, before the sides
    /// that jump to it are compiled: the argument registers of the ABI, in
    /// order, so the entry registers need no move.
    fn plan_join(&mut self, f: &Func, join_idx: usize) {
        let j = f.insts[join_idx];
        let mut taken = 0u32;
        for k in 0..j.b {
            let p = j.a + k;
            let class = class_of(f.val_ty[p as usize]);
            let mut pick = NOREG;
            for &r in self.t.arg_regs(class) {
                if self.alloc_mask & (1 << r) != 0 && taken & (1 << r) == 0 {
                    pick = r;
                    break;
                }
            }
            if pick == NOREG {
                let avail = self.class_mask(class) & !taken;
                if avail == 0 {
                    // No register left: the parameter arrives in a slot of
                    // the root's frame, reserved here.
                    self.slot[p as usize] = self.base + 2 + self.join_slots;
                    self.join_slots += 1;
                    self.jlin[p as usize] = LIN_SLOT;
                    continue;
                }
                pick = avail.trailing_zeros() as Reg;
            }
            taken |= 1 << pick;
            self.jlin[p as usize] = pick;
            self.hint[p as usize] = pick;
        }
    }

    /// A side's jump to the `Join`: one parallel move into the parameters.
    fn do_rejoin(&mut self, f: &Func, i: &Inst) {
        let nth = if i.e == NONE { 0 } else { i.e };
        let label = self.join_start[i.d as usize] + nth;
        let j = f.insts[self.joins[label as usize]];
        self.t.jump(&mut self.cb, Tgt::Outer(label));
        self.edge(f, i.a, j.b, j.a, true);
    }

    /// The end of a region that enters a side: the parameters of the side go
    /// to the locations it expects them in, as for a guard that has a side.
    fn do_goto(&mut self, f: &Func, i: &Inst) {
        self.t
            .jump(&mut self.cb, Tgt::Body(self.side_top[i.d as usize]));
        let base = self.side_loc_start[i.d as usize] as usize;
        self.moves.clear();
        for (k, sv) in f.snap_of(i.c).iter().enumerate() {
            let dst = self.side_locs[base + k];
            if dst != DEAD {
                let src = self.weak_loc(sv.val);
                self.moves.push((dst, src));
            }
        }
        self.stats.trampoline_moves += self.emit_plan();
    }

    /// A call made only when the condition holds, out of line. The stub saves
    /// every caller-saved register that holds a value, moves the arguments
    /// where the ABI wants them, calls, restores the registers and jumps back
    /// to the code after the instruction.
    fn do_call_if(&mut self, f: &Func, i: &Inst) {
        let mut cx = Cx::new();
        let (mut a, mut b, mut cc) = (i.a, i.b, i.cc);
        if i.ty == Ty::I64 && self.is_const(a) && !self.is_const(b) {
            std::mem::swap(&mut a, &mut b);
            cc = cc.swap();
        }
        // The guard branches when its condition is false: this one jumps to
        // the stub when the call's condition is true. A float compare has
        // its operands swapped where the target says (see `do_guard`).
        let go = cc.negate(i.ty == Ty::F64);
        if i.ty == Ty::F64 && self.t.fcmp_swapped(go) {
            std::mem::swap(&mut a, &mut b);
        }
        let ra = self.use_reg(a, &mut cx);
        let sb = self.use_src(b, Op::Guard, i.ty, &mut cx);
        // Where each argument is when the condition holds.
        self.moves.clear();
        let mut counts = [0usize; 2];
        let mut stack = 0u32;
        for (pos, &v) in f.list(i.c, i.d).iter().enumerate() {
            let class = self.val_class(v);
            let dst = self.arg_dst(class, pos, &mut counts, &mut stack);
            let src = self.weak_loc(v);
            self.moves.push((dst, src));
        }
        let cont = self.cb.mark();
        let s = self.stubs.mark();
        let mut saved = 0u32;
        for r in 0..32u8 {
            if self.reg_val[r as usize] != NONE && self.caller_mask & (1 << r) != 0 {
                saved |= 1 << r;
                self.t.mov(
                    &mut self.stubs,
                    Loc::Slot(self.save0 + u32::from(r)),
                    Loc::Reg(r),
                );
            }
        }
        self.stats.trampoline_moves += self.emit_plan_stub();
        if i.scale == 1 {
            self.t
                .call_ctl(&mut self.stubs, self.base + 1, i.imm as i32);
        } else {
            self.t.call(&mut self.stubs, i.imm as u64);
        }
        if i.e != NONE {
            self.t
                .store_result_ctl(&mut self.stubs, self.base + 1, i.e as i32);
        }
        let mut rest = saved;
        while rest != 0 {
            let r = rest.trailing_zeros() as Reg;
            rest &= rest - 1;
            self.t.mov(
                &mut self.stubs,
                Loc::Reg(r),
                Loc::Slot(self.save0 + u32::from(r)),
            );
        }
        self.t.jump(&mut self.stubs, Tgt::Body(cont));
        self.cb.ensure();
        self.t.guard(&mut self.cb, i.ty, go, ra, sb, Tgt::Stub(s));
        self.flush(&cx);
    }

    /// The end of a skipped block, which the pass meets first. It notes the
    /// join point and the registers the values live after it are expected in,
    /// then ends the block: the block's values move into the join values.
    fn do_end_if(&mut self, f: &Func, i: &Inst) {
        let join = self.cb.mark();
        let dst = self.if_dst.len() as u32;
        for k in 0..i.b {
            let j = i.a + k;
            let r = self.reg[j as usize];
            self.if_dst.push(r);
            if r != NOREG {
                self.unbind(j);
            }
        }
        let saved = self.if_saved.len() as u32;
        for r in 0..32u8 {
            let v = self.reg_val[r as usize];
            if v != NONE {
                self.if_saved.push((v, r));
            }
        }
        self.if_frames.push(IfFrame { join, saved, dst });
        self.stats.ifs += 1;
        // The join values are in a register, a slot or both, wherever the
        // code after the join reads them.
        self.moves.clear();
        for k in 0..i.b {
            let j = i.a + k;
            let t = f.lists[(i.c + k) as usize];
            self.dec_use(t);
            let jr = self.if_dst[(dst + k) as usize];
            let js = self.slot[j as usize];
            if jr == NOREG && js == NOSLOT {
                continue;
            }
            let src = if self.is_const(t) {
                Loc::Imm(self.kbits[t as usize])
            } else if self.reg[t as usize] != NOREG {
                Loc::Reg(self.reg[t as usize])
            } else {
                let hint = if jr != NOREG {
                    jr
                } else {
                    self.hint[t as usize]
                };
                if let Some(r) = self.alloc_opt(self.val_class(t), hint, 0) {
                    self.bind(t, r);
                    Loc::Reg(r)
                } else {
                    Loc::Slot(self.slot_of(t))
                }
            };
            if jr != NOREG {
                self.moves.push((Loc::Reg(jr), src));
            }
            if js != NOSLOT {
                self.moves.push((Loc::Slot(js), src));
            }
        }
        self.emit_plan();
    }

    /// The start of a skipped block: the branch over it. The skipped path
    /// arrives at the join with the registers as they are here, so it moves
    /// what differs from the state the join expects, and passes the values the
    /// join takes from this side.
    fn do_if(&mut self, f: &Func, i: &Inst) {
        let frame = self.if_frames.pop().expect("an if has its end");
        let end = f.insts[i.e as usize];
        let mut cx = Cx::new();
        let (mut a, mut b, mut cc) = (i.a, i.b, i.cc);
        if i.ty == Ty::F64 {
            if self.t.fcmp_swapped(cc) {
                std::mem::swap(&mut a, &mut b);
            }
        } else if self.is_const(a) && !self.is_const(b) {
            std::mem::swap(&mut a, &mut b);
            cc = cc.swap();
        }
        // A skipped value that nothing holds yet is best made in the register
        // of its join value, so the skip path has nothing to move. (The
        // compare below may be its first use.)
        for k in 0..end.b {
            let e = f.lists[(end.c + end.b + k) as usize];
            let jr = self.if_dst[(frame.dst + k) as usize];
            if jr != NOREG && !self.is_const(e) && self.hint[e as usize] == NOREG {
                self.hint[e as usize] = jr;
            }
        }
        // An operand of the compare that nothing holds yet stays out of the
        // registers the join expects its values in, when another is free:
        // the compare locks its registers, so a value the block evicted
        // could not go back to the one the join reads it from
        // (`shuffle_entry`), and the skipped path would reload it.
        // The same for the registers of the join values, which the skipped
        // values are made in below.
        // A test of a byte that only this `If` reads compares it in memory,
        // unless the skipped path hands the byte to the join.
        let in_place = (0..end.b).all(|k| f.lists[(end.c + end.b + k) as usize] != a)
            && self.byte_load(f, i.ty, None, a, b, cc).is_some();
        let mut expected = 0u32;
        for saved in &self.if_saved[frame.saved as usize..] {
            expected |= 1 << saved.1;
        }
        for k in 0..end.b {
            let jr = self.if_dst[(frame.dst + k) as usize];
            if jr != NOREG {
                expected |= 1 << jr;
            }
        }
        for v in [a, b] {
            // (The second operand is read from its slot when it has one.)
            if self.is_const(v)
                || self.reg[v as usize] != NOREG
                || (v != a && self.slot[v as usize] != NOSLOT)
                || in_place
            {
                continue;
            }
            // An operand that is the skipped value of a join belongs in that
            // join's register (the hint above).
            let mut avoid = expected;
            for k in 0..end.b {
                let jr = self.if_dst[(frame.dst + k) as usize];
                if jr != NOREG && f.lists[(end.c + end.b + k) as usize] == v {
                    avoid &= !(1 << jr);
                }
            }
            if let Some(r) = self.alloc_opt(self.val_class(v), self.hint[v as usize], avoid) {
                self.bind(v, r);
            }
        }
        let byte = if in_place {
            self.byte_guard(f, i.ty, None, a, b, cc, &mut cx)
        } else {
            None
        };
        let (ra, sb) = if byte.is_some() {
            self.dec_use(a);
            self.dec_use(b);
            (NOREG, Src::Imm(0))
        } else {
            let ra = self.use_reg(a, &mut cx);
            (ra, self.use_src(b, Op::Guard, i.ty, &mut cx))
        };
        let joined = self.shuffle_entry(&frame, cx.locked);
        self.moves.clear();
        for k in frame.saved as usize..self.if_saved.len() {
            let (v, r) = self.if_saved[k];
            let src = if self.reg[v as usize] != NOREG {
                Loc::Reg(self.reg[v as usize])
            } else {
                Loc::Slot(self.slot_of(v))
            };
            self.moves.push((Loc::Reg(r), src));
        }
        for k in 0..end.b {
            let j = end.a + k;
            let e = f.lists[(end.c + end.b + k) as usize];
            let jr = self.if_dst[(frame.dst + k) as usize];
            let js = self.slot[j as usize];
            if jr == NOREG && js == NOSLOT {
                continue;
            }
            // The join value is wanted in a register, so a skipped value
            // nothing holds goes to one (its join value's, if that is free).
            if jr != NOREG
                && !self.is_const(e)
                && self.reg[e as usize] == NOREG
                && self.slot[e as usize] == NOSLOT
                && let Some(r) = self.alloc_opt(self.val_class(e), jr, joined | cx.locked)
            {
                self.bind(e, r);
            }
            let src = self.weak_loc(e);
            if jr != NOREG {
                self.moves.push((Loc::Reg(jr), src));
            }
            if js != NOSLOT {
                self.moves.push((Loc::Slot(js), src));
            }
        }
        self.if_saved.truncate(frame.saved as usize);
        self.if_dst.truncate(frame.dst as usize);
        self.moves.retain(|(d, s)| d != s);
        let tgt = if self.moves.is_empty() {
            if self.rotated_if == i.e as usize {
                self.stats.rotated_ifs += 1;
                Tgt::Label
            } else {
                Tgt::Body(frame.join)
            }
        } else {
            self.stats.if_stubs += 1;
            let s = self.stubs.mark();
            self.stats.if_moves += self.emit_plan_stub();
            self.t.jump(&mut self.stubs, Tgt::Body(frame.join));
            Tgt::Stub(s)
        };
        self.cb.ensure();
        if let Some((addr, imm)) = byte {
            self.t.guard_byte(&mut self.cb, cc, addr, imm, tgt);
        } else {
            self.t.guard(&mut self.cb, i.ty, cc, ra, sb, tgt);
        }
        self.flush(&cx);
    }

    /// The registers the code above an `If` holds its values in, chosen so
    /// that the skipped path has as little to move as possible. The block was
    /// compiled first, and it may have evicted a value the join expects in a
    /// register (it is in its slot at the top of the block) or taken that
    /// register for one of its own. The skipped path would load the first
    /// back and the block path pays nothing, which is the wrong way round
    /// when the block is the rare one. So the value is bound to the register
    /// the join expects, here above the `If`, and what sat in that register
    /// at the top of the block moves to a free one, with a move at the start
    /// of the block, where only the block pays for it. The block reloads the
    /// value from the slot it already has where it needs it. `locked` are the
    /// registers the compare reads. Returns the registers the join expects.
    fn shuffle_entry(&mut self, frame: &IfFrame, locked: u32) -> u32 {
        self.moves.clear();
        let saved = frame.saved as usize..self.if_saved.len();
        let mut joined = 0u32;
        for k in saved.clone() {
            joined |= 1 << self.if_saved[k].1;
        }
        for k in saved.clone() {
            let (v, r) = self.if_saved[k];
            let cur = self.reg[v as usize];
            if cur == r
                || locked & (1 << r) != 0
                || (cur == NOREG && self.slot[v as usize] == NOSLOT)
                || (cur != NOREG && locked & (1 << cur) != 0)
            {
                continue;
            }
            let w = self.reg_val[r as usize];
            if w != NONE {
                if self.if_saved[saved.clone()].iter().any(|&(u, _)| u == w) {
                    continue;
                }
                let Some(to) = self.alloc_opt(self.val_class(w), NOREG, joined | locked) else {
                    continue;
                };
                self.moves.push((Loc::Reg(r), Loc::Reg(to)));
                self.unbind(w);
                self.bind(w, to);
            } else if self.free & (1 << r) == 0 {
                continue;
            }
            self.take(r);
            if cur != NOREG {
                self.moves.push((Loc::Reg(cur), Loc::Reg(r)));
                self.unbind(v);
            }
            self.bind(v, r);
            self.stats.if_rebound += 1;
        }
        if !self.moves.is_empty() {
            self.stats.if_entry_moves += self.emit_plan();
            self.moves.clear();
        }
        joined
    }

    /// Gives each parameter of a side the cell it may live in: every entry
    /// into the side hands it a value that lives in that one cell (a load of
    /// it, or a parameter of the entering side that lives there). A homed
    /// cell is stored once before anything reads it and keeps its value for
    /// as long as a value loaded from it is read, so the parameter is the
    /// cell's word for the side's whole run, and an entry has nothing to
    /// move for it. Without this each entry copies the value from the cell
    /// to a slot of the side, for the side's exits alone.
    ///
    /// A side's parameters are settled where its region starts, from the
    /// entries met before it, so an entry met later (none is made today)
    /// leaves every side as it was.
    fn find_side_cells(&mut self, f: &Func) {
        const MIXED: u32 = NONE - 1;
        let sides = f.num_sides();
        let mut handed = std::mem::take(&mut self.side_cells);
        handed.clear();
        handed.resize(f.val_ty.len(), NONE);
        let mut late = false;
        'regions: for r in 0..=sides {
            let (start, end) = f.region(r);
            if r > 0 {
                let entry = f.insts[start];
                for p in entry.a..entry.a + entry.b {
                    let cell = handed[p as usize];
                    if cell != NONE && cell != MIXED {
                        self.cell_of[p as usize] = cell;
                    }
                }
            }
            for idx in start..end {
                let i = f.insts[idx];
                let to = match i.op {
                    Op::Guard if i.d != NONE => i.d as usize,
                    Op::Goto => i.d as usize,
                    _ => continue,
                };
                let (target, _) = f.region(1 + to);
                if target <= idx {
                    late = true;
                    break 'regions;
                }
                let entry = f.insts[target];
                for (k, sv) in f.snap_of(i.c).iter().enumerate().take(entry.b as usize) {
                    let p = (entry.a + k as u32) as usize;
                    let cell = if names_value(sv) && self.cell_home(sv.val).is_some() {
                        self.cell_of[sv.val as usize]
                    } else {
                        MIXED
                    };
                    handed[p] = if handed[p] == NONE || handed[p] == cell {
                        cell
                    } else {
                        MIXED
                    };
                }
            }
        }
        if late {
            for r in 1..=sides {
                let entry = f.insts[f.region(r).0];
                for p in entry.a..entry.a + entry.b {
                    self.cell_of[p as usize] = NONE;
                }
            }
        }
        self.side_cells = handed;
    }

    fn do_cell_store(&mut self, i: &Inst) {
        if !self.cell_read[i.imm as usize] {
            self.cell_dropped[i.imm as usize] = true;
            self.stats.dead_skipped += 1;
            return;
        }
        let mut cx = Cx::new();
        let src = if self.is_const(i.c) && i.ty == Ty::I64 {
            match self.use_src(i.c, Op::Store, i.ty, &mut cx) {
                Src::Imm(v) => Loc::Imm(v),
                Src::Reg(r) => Loc::Reg(r),
                _ => unreachable!("an integer constant is an immediate or a register"),
            }
        } else {
            Loc::Reg(self.use_reg(i.c, &mut cx))
        };
        self.cb.ensure();
        self.t
            .mov(&mut self.cb, Loc::Slot(self.out_slots + i.imm as u32), src);
        self.flush(&cx);
    }

    fn do_cell_load(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        debug_assert!(
            !self.cell_dropped[i.imm as usize],
            "a kept load of cell {} comes before a store of it that was dropped: \
             a load above a store needs a kept load below it",
            i.imm
        );
        self.cell_read[i.imm as usize] = true;
        if self.reg[i.dst as usize] == NOREG
            && self.cell_home(i.dst) == Some(self.slot[i.dst as usize])
        {
            // Every reader takes the value from the cell.
            self.stats.cell_homes += 1;
            return;
        }
        let dst = self.def(i.dst, class_of(i.ty));
        self.cb.ensure();
        self.t.mov(
            &mut self.cb,
            Loc::Reg(dst),
            Loc::Slot(self.out_slots + i.imm as u32),
        );
    }

    /// The exit-area pointer, from the frame slot that keeps it for the
    /// stubs (see `do_entry`).
    fn do_exit_area(&mut self, i: &Inst) {
        if !self.live(i.dst) {
            self.stats.dead_skipped += 1;
            return;
        }
        let dst = self.def(i.dst, Class::Gpr);
        self.cb.ensure();
        self.t
            .mov(&mut self.cb, Loc::Reg(dst), Loc::Slot(self.base));
    }

    fn do_header(&mut self, f: &Func, i: &Inst, label: Option<u32>) {
        let join = label.is_some();
        self.moves.clear();
        for k in 0..i.b {
            let p = i.a + k;
            let Some(lin) = self.lin_loc(p, join) else {
                continue;
            };
            if self.reg[p as usize] != NOREG {
                let a = Loc::Reg(self.reg[p as usize]);
                if a != lin {
                    self.moves.push((a, lin));
                }
            }
            if self.slot[p as usize] != NOSLOT {
                let a = Loc::Slot(self.slot[p as usize]);
                if a != lin {
                    self.moves.push((a, lin));
                    self.stats.def_spill_stores += 1;
                }
            }
        }
        if !join {
            for k in self.inv_lo..self.inv_hi {
                let v = self.invariants[k];
                if self.reg[v as usize] != NOREG {
                    let l = self
                        .lin_loc(v, false)
                        .expect("invariant has an incoming location");
                    let a = Loc::Reg(self.reg[v as usize]);
                    if a != l {
                        self.moves.push((a, l));
                    }
                }
            }
        }
        self.stats.header_moves += self.emit_plan();
        if let Some(label) = label {
            self.cb.bind_outer(label);
        } else {
            self.cb.bind_label();
            self.in_loop = false;
        }
        self.clear_bindings();
        if !join {
            for k in self.inv_lo..self.inv_hi {
                let v = self.invariants[k];
                let l = self.lin[v as usize];
                if l != LIN_SLOT && l != NOREG {
                    self.take(l);
                    self.bind(v, l);
                }
            }
        }
        self.edge(f, i.c, i.b, i.a, join);
    }
}

impl<T: Target> Compiler<T> {
    fn val_class(&self, v: u32) -> Class {
        if self.flags[v as usize] & F_FLOAT != 0 {
            Class::Fpr
        } else {
            Class::Gpr
        }
    }
}
