//! MIR types: dense `u32`-indexed arrays, no per-instruction heap allocation.
//!
//! A [`Func`] is one trace tree. Region 0 is the root: `Entry`, a prologue,
//! an optional `Header` ... `Back` loop, and it ends in `Back` or `Exit`.
//! Regions 1.. are sides: `SideEntry` (defines the side's parameters), a body
//! and a closing `Exit`. A guard either exits through its snapshot or, when it
//! names a side, passes its snapshot values to that side's parameters.
//!
//! A function that is called as a function of its own (an activation of a
//! function trace, which calls itself) ends its regions in `Ret` as well,
//! takes more arguments than the entry contract's three (`Entry` with more
//! values), and calls itself, or code placed next to it, by `Call` with
//! [`CALL_REL`].
//!
//! Values are SSA ids (`Val`), defined once. Constants are values defined by
//! `Const` and are never given registers: each use is an immediate or a
//! rematerialization.

pub const NONE: u32 = u32::MAX;
pub type Val = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Ty {
    I64,
    F64,
}

/// Condition. Integers use all of them (`U*` are unsigned); floats use
/// `Eq Ne Lt Le Gt Ge` with ordered semantics, `Ne` true on NaN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Cc {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Ult,
    Ule,
    Ugt,
    Uge,
    /// Float only: the negation of the ordered `Lt`, `Le`, `Gt`, `Ge`, so
    /// true on NaN. (`Ne` is the negation of `Eq` already.)
    Nlt,
    Nle,
    Ngt,
    Nge,
}

impl Cc {
    /// The condition that holds exactly when `self` does not. Floats keep
    /// the NaN rule: the negation of an ordered compare is true on NaN.
    #[must_use]
    pub fn negate(self, float: bool) -> Cc {
        match (self, float) {
            (Cc::Eq, _) => Cc::Ne,
            (Cc::Ne, _) => Cc::Eq,
            (Cc::Lt, false) => Cc::Ge,
            (Cc::Le, false) => Cc::Gt,
            (Cc::Gt, false) => Cc::Le,
            (Cc::Ge, false) => Cc::Lt,
            (Cc::Lt, true) => Cc::Nlt,
            (Cc::Le, true) => Cc::Nle,
            (Cc::Gt, true) => Cc::Ngt,
            (Cc::Ge, true) => Cc::Nge,
            (Cc::Nlt, _) => Cc::Lt,
            (Cc::Nle, _) => Cc::Le,
            (Cc::Ngt, _) => Cc::Gt,
            (Cc::Nge, _) => Cc::Ge,
            (Cc::Ult, _) => Cc::Uge,
            (Cc::Ule, _) => Cc::Ugt,
            (Cc::Ugt, _) => Cc::Ule,
            (Cc::Uge, _) => Cc::Ult,
        }
    }

    /// The condition after swapping the two compared operands.
    #[must_use]
    pub fn swap(self) -> Cc {
        match self {
            Cc::Eq => Cc::Eq,
            Cc::Ne => Cc::Ne,
            Cc::Lt => Cc::Gt,
            Cc::Le => Cc::Ge,
            Cc::Gt => Cc::Lt,
            Cc::Ge => Cc::Le,
            Cc::Ult => Cc::Ugt,
            Cc::Ule => Cc::Uge,
            Cc::Ugt => Cc::Ult,
            Cc::Uge => Cc::Ule,
            Cc::Nlt => Cc::Ngt,
            Cc::Nle => Cc::Nge,
            Cc::Ngt => Cc::Nlt,
            Cc::Nge => Cc::Nle,
        }
    }

    /// Integer comparison.
    #[must_use]
    pub fn eval_i(self, a: i64, b: i64) -> bool {
        let (ua, ub) = (a as u64, b as u64);
        match self {
            Cc::Eq => a == b,
            Cc::Ne => a != b,
            Cc::Lt => a < b,
            Cc::Le => a <= b,
            Cc::Gt => a > b,
            Cc::Ge => a >= b,
            Cc::Ult => ua < ub,
            Cc::Ule => ua <= ub,
            Cc::Ugt => ua > ub,
            Cc::Uge => ua >= ub,
            Cc::Nlt => a >= b,
            Cc::Nle => a > b,
            Cc::Ngt => a <= b,
            Cc::Nge => a < b,
        }
    }

    /// Float comparison; false on NaN except `Ne` and the negations.
    #[must_use]
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "the negations are true on NaN"
    )]
    pub fn eval_f(self, a: f64, b: f64) -> bool {
        match self {
            Cc::Eq => a == b,
            Cc::Ne => a != b,
            Cc::Lt | Cc::Ult => a < b,
            Cc::Le | Cc::Ule => a <= b,
            Cc::Gt | Cc::Ugt => a > b,
            Cc::Ge | Cc::Uge => a >= b,
            Cc::Nlt => !(a < b),
            Cc::Nle => !(a <= b),
            Cc::Ngt => !(a > b),
            Cc::Nge => !(a >= b),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Op {
    /// Defines `b` values from `a`: value `a + k` is argument `k` of the
    /// native call. Argument 1 is the control pointer and argument 2 the
    /// exit-area pointer, which the prologue keeps in their frame slots
    /// whatever `b` is: with `b` 2 the entry contract's `inputs` and
    /// `control`, with `b` 3 or more the exit-area pointer and the arguments
    /// after it too, in the argument registers of the target's ABI (no
    /// argument may go to the stack, see `Compiler::entry_fits`).
    Entry,
    /// Defines `b` values from `a`: the side's parameters.
    SideEntry,
    /// `dst = imm` (f64 bits for `F64`).
    Const,
    /// `dst = [a + b*scale + imm]`, `b` may be `NONE`.
    Load,
    /// `[a + b*scale + imm] = c`.
    Store,
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Shifts take a constant count in `b`.
    Shl,
    Shr,
    Sar,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FSqrt,
    /// Signed division and remainder, truncating. The divisor must not be 0
    /// or -1 (the lowering replaces those before it divides), so the machine
    /// instruction never traps.
    SDiv,
    SRem,
    /// Unsigned division and remainder. The divisor must not be 0.
    UDiv,
    URem,
    /// `dst:f64 = -a`, `|a|`, `floor(a)`, `ceil(a)`, `trunc(a)` (toward zero).
    FNeg,
    FAbs,
    FFloor,
    FCeil,
    FTrunc,
    /// `dst:f64 = |a|` with the sign bit of `b`.
    FCopySign,
    /// `dst:f64 = c - a * b`, the product not rounded before the subtraction
    /// (a fused multiply-add of `-a`, `b`, `c`). Only a target that has the
    /// instruction ([`crate::target::Target::fused`]) compiles it. A NaN
    /// operand is passed on as it is, quieted (the first of `a`, `b`, `c`), so
    /// its sign survives to a `FCopySign` or a store.
    FNma,
    /// `dst:f64 = a:i64 as f64`.
    I2F,
    /// `dst:i64 = trunc(a:f64)`, for a value known to be in range (the
    /// out-of-range result is the machine's, `i64::MIN`).
    F2I,
    /// `dst:i64 = zero-extended byte at [a + b*scale + imm]`.
    Load8,
    /// `dst:i64 = zero-extended 32-bit word at [a + b*scale + imm]`.
    Load32,
    /// `dst:i64 = the bits of a:f64`.
    Bits,
    /// `dst:f64 = the float whose bits are a:i64`.
    FromBits,
    /// `byte [a + b*scale + imm] = low byte of c`.
    Store8,
    /// `[a + b*scale + imm] += c`, integers: a read-modify-write in memory.
    AddMem,
    /// `dst = (a cc b) ? c : d`, integers.
    Select,
    /// Continue when `a cc b`, otherwise exit through snapshot `c`, or go to
    /// side `d` (when not `NONE`) passing the snapshot's values.
    ///
    /// An integer guard against the constant zero with `scale` 1 relies on the
    /// flags of the machine instruction before it: the `Add` or `Sub` that
    /// defined `a`, with nothing that makes code in between (constants make
    /// none, an `If` or `EndIf` does). It saves the compare. The two differ
    /// only when the `Add` or `Sub` overflows.
    Guard,
    /// Loop header. Parameters are `a..a+b`, initial values are `lists[c..c+b]`.
    Header,
    /// Back edge with arguments `lists[a..a+b]`, one per header parameter.
    Back,
    /// Leave through snapshot `c`.
    Exit,
    /// Return from the function: `a` (an integer) in the first return
    /// register, and `b` (`NONE` for none) in the second: the integer one
    /// after the first for an integer, the first float one for a float (see
    /// `Target::ret2_reg`). Ends a region.
    Ret,
    /// A second, outer loop header in a region (the root or a side that has a
    /// loop; a region may have several, in sequence): parameters are `a..a+b`, initial values are `lists[c..c+b]`.
    /// Nothing in its own region jumps back to it; sides do, with `Rejoin`. It
    /// comes before `Header`, anywhere after the region's entry (a root puts
    /// it below its preheader when a link does not run that again), and what
    /// is defined before it and used after it is only its parameters, the
    /// constants (made again where they are read) and what the region keeps
    /// in frame cells, see `CellStore`.
    Join,
    /// Jump to a `Join` of region `d` (the root when 0) with arguments
    /// `lists[a..a+b]`, one per parameter: the first of the region, or the
    /// `e`th when `e` is not `NONE`. Ends a side.
    Rejoin,
    /// Calls the C function at address `imm` with the arguments
    /// `lists[a..a+b]` (integers in the integer argument registers in order,
    /// floats in the float ones) and the native ABI of the target. `dst`
    /// (`NONE` for none) is the result, of type `ty`. With `scale`
    /// [`CALL_REL`] the function is the code at byte `imm` from the first
    /// byte of this function's image (0 is its own entry), and `c` (`NONE`
    /// for none) is a second result, where `Ret` leaves its `b`. Every register that
    /// ABI lets the callee clobber is dead across the call: a value live
    /// after it is kept in a register the ABI preserves, or is stored once at
    /// its definition and reloaded after the call. Arguments past the sixth
    /// integer and the eighth float go to the stack in order, as the ABI says.
    Call,
    /// As `Call`, but the function is the pointer stored at byte `imm` of the
    /// control block (the second entry argument).
    CallCtl,
    /// Calls like `Call` (or like `CallCtl` when `scale` is 1) only when
    /// `a cc b` holds (integers or floats, as for `Guard`), with the arguments `lists[c..c+d]`, and
    /// goes on either way. The call is out of line, off the path that does
    /// not take it: the code saves the registers the callee may clobber
    /// around it, so the values of the loop stay where they are. The call has
    /// no result value: when `e` is not `NONE` the result is stored at that
    /// byte offset of the control block, where the code after the instruction
    /// reads it (it holds what it held before when the call is not made).
    CallIf,
    /// `cell[imm] = c`: a word of the frame no allocation touches. A region
    /// that has a `Join` keeps there what the code below the `Join` reads of
    /// what is above it, since only the `Join`'s parameters cross it: the
    /// root's `Join` (the sides that rejoin it), and the `Join` of a loop
    /// side that its nested sides rejoin.
    ///
    /// The allocator goes backwards and drops a store unless a kept `CellLoad`
    /// of the cell follows it in program order (`cell_read` in `ra.rs`). So a
    /// cell is stored above the loads that read it. A load above a store, in a
    /// loop, reads the store of the iteration before and needs another kept
    /// load of the cell below the store; the allocator asserts in debug builds
    /// that no kept load precedes a store it dropped. A value loaded from a
    /// cell that has one store, outside any loop or `Join`, and no load above
    /// it may live in the cell instead of a register or a slot of its own.
    CellStore,
    /// `dst = cell[imm]`.
    CellLoad,
    /// `dst:i64 = the exit-area pointer` the entry was given (its third
    /// argument), which the frame keeps for the exit stubs. Code that writes
    /// an exit's state in line, before a call that reads it, stores through
    /// it: the same words the exit's stub writes, at the same places.
    ExitArea,
    /// Ends a region by jumping into side `d` with the values of snapshot `c`
    /// as its parameters, as a failed guard that has a side does.
    Goto,
    /// Opens a forward skip: the instructions up to the matching `EndIf`
    /// (instruction `e`) run when `a cc b` holds (integers or floats, as for
    /// `Guard`) and are jumped over otherwise. Nothing the block defines can
    /// be used after `EndIf` except through its join values. The block may
    /// hold any instruction that does not end a region or open a loop,
    /// including guards, calls, conditional calls and further `If` regions.
    If,
    /// Closes an `If`. Defines `b` values from `a`: join value `k` is
    /// `lists[c + k]` when the block ran (a value the block may define) and
    /// `lists[c + b + k]` when it was skipped (a value defined before the
    /// `If`).
    EndIf,
}

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub op: Op,
    pub ty: Ty,
    pub cc: Cc,
    pub scale: u8,
    pub dst: Val,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    /// `CallIf`: the byte offset in the control block where the call's result
    /// is stored when it is made, or `NONE`. `If`: the index of its `EndIf`.
    pub e: u32,
    pub imm: i64,
}

impl Inst {
    #[must_use]
    pub fn new(op: Op, ty: Ty) -> Inst {
        Inst {
            op,
            ty,
            cc: Cc::Eq,
            scale: 0,
            dst: NONE,
            a: NONE,
            b: NONE,
            c: NONE,
            d: NONE,
            e: NONE,
            imm: 0,
        }
    }
}

/// The `scale` of an operation that is cold (a float operation, or an
/// arithmetic shift right by a constant, see [`Func::sar_cold`]): nothing
/// but snapshots (and other cold operations) read its value, so the
/// allocator does not compute it where it stands. Each exit stub that reads it computes it from its
/// operands instead, as the exit leaves the function anyway, and the loop
/// pays nothing for it. A value that something else reads after all (a
/// guard that has a side, a join, a back edge) is computed where it stands,
/// as without the mark. The operands are read as a snapshot reads a value.
pub const COLD: u8 = 1;

/// The `scale` of a `Call` whose target is relative to the start of the
/// function's own image: `imm` bytes from its first byte. A function calls
/// itself with `imm` 0, and code placed before it with a negative `imm`.
pub const CALL_REL: u8 = 1;

/// Whether `op` is an operation a stub can compute (see [`COLD`]): the float
/// operations, and `Sar` by a constant count. `I2F` is one: its operand is an
/// integer, which the stub reads where the snapshot would.
#[must_use]
pub fn is_coldable(op: Op) -> bool {
    matches!(
        op,
        Op::I2F
            | Op::Sar
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FNeg
            | Op::FAbs
            | Op::FFloor
            | Op::FCeil
            | Op::FTrunc
    )
}

/// A snapshot: `snap_vals[start..start+len]`.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    pub start: u32,
    pub len: u32,
    /// What the entry returns for this exit. `snap_code` of the builder
    /// gives `-1 - index`; a mismatch exit may return another code.
    pub code: i64,
}

/// A `SnapVal::slot` with this bit set names word `slot & !CONTROL_SLOT` of
/// the control block (the second entry argument) instead of the exit area.
pub const CONTROL_SLOT: u32 = 0x8000_0000;

/// A `SnapVal::slot` with this bit set names no word: the exit returns the
/// value instead of the snapshot's code.
pub const RETURN_SLOT: u32 = 0x4000_0000;

/// A `SnapVal::slot` with this bit set copies a word that no value holds into
/// exit-area word `slot & !INDIRECT_SLOT`: the word at byte `val` (a byte
/// offset, not a value) of the buffer whose address is at byte
/// [`Func::indirect`] of the control block. Only exits take it, not the
/// guards that enter sides.
pub const INDIRECT_SLOT: u32 = 0x2000_0000;

/// Whether a snapshot entry names a value: every one but an
/// [`INDIRECT_SLOT`] copy.
#[must_use]
pub const fn names_value(sv: &SnapVal) -> bool {
    sv.slot & INDIRECT_SLOT == 0
}

/// Value `val` is written to `exit_area[slot]` when the exit is taken.
#[derive(Clone, Copy, Debug)]
pub struct SnapVal {
    pub val: Val,
    pub slot: u32,
}

/// Reusable arena for one tree. `clear` keeps every allocation.
#[derive(Default)]
pub struct Func {
    pub insts: Vec<Inst>,
    pub val_ty: Vec<Ty>,
    pub lists: Vec<u32>,
    pub snaps: Vec<Snap>,
    pub snap_vals: Vec<SnapVal>,
    /// First instruction of each side; the root ends at the first of these.
    pub side_starts: Vec<u32>,
    /// Frame cells the function uses (`CellStore`, `CellLoad`).
    pub cells: u32,
    /// The float operations the builders make from here on are cold, see
    /// [`COLD`].
    pub cold: bool,
    /// The byte in the control block that holds the address the
    /// [`INDIRECT_SLOT`] entries of snapshots read through.
    pub indirect: i32,
}

impl Func {
    #[must_use]
    pub fn new() -> Func {
        Func::default()
    }

    pub fn clear(&mut self) {
        self.insts.clear();
        self.val_ty.clear();
        self.lists.clear();
        self.snaps.clear();
        self.snap_vals.clear();
        self.side_starts.clear();
        self.cells = 0;
        self.cold = false;
        self.indirect = 0;
    }

    #[must_use]
    pub fn num_vals(&self) -> usize {
        self.val_ty.len()
    }

    #[must_use]
    pub fn num_sides(&self) -> usize {
        self.side_starts.len()
    }

    /// Instruction range of region `r` (0 is the root, `1 + k` is side `k`).
    #[must_use]
    pub fn region(&self, r: usize) -> (usize, usize) {
        let start = if r == 0 {
            0
        } else {
            self.side_starts[r - 1] as usize
        };
        let end = if r < self.side_starts.len() {
            self.side_starts[r] as usize
        } else {
            self.insts.len()
        };
        (start, end)
    }

    pub fn val(&mut self, ty: Ty) -> Val {
        self.val_ty.push(ty);
        (self.val_ty.len() - 1) as Val
    }

    pub fn push(&mut self, inst: Inst) {
        self.insts.push(inst);
    }

    /// Appends `n` empty snapshots, so the exit index of each is its
    /// position. Fill them with [`Func::set_snap`].
    pub fn reserve_snaps(&mut self, n: usize) {
        for _ in 0..n {
            let code = -1 - self.snaps.len() as i64;
            self.snaps.push(Snap {
                start: 0,
                len: 0,
                code,
            });
        }
    }

    /// Fills snapshot `index` with `vals`.
    pub fn set_snap(&mut self, index: u32, vals: &[SnapVal]) {
        let start = self.snap_vals.len() as u32;
        self.snap_vals.extend_from_slice(vals);
        let s = &mut self.snaps[index as usize];
        s.start = start;
        s.len = vals.len() as u32;
    }

    /// A new snapshot that returns `code` instead of `-1 - index`.
    pub fn snap_returning(&mut self, vals: &[SnapVal], code: i64) -> u32 {
        let start = self.snap_vals.len() as u32;
        self.snap_vals.extend_from_slice(vals);
        self.snaps.push(Snap {
            start,
            len: vals.len() as u32,
            code,
        });
        (self.snaps.len() - 1) as u32
    }

    /// Starts a side without writing its entry instruction: the lowering
    /// that owns the parameter values calls this, then pushes `SideEntry`.
    pub fn mark_side(&mut self) -> u32 {
        self.side_starts.push(self.insts.len() as u32);
        (self.side_starts.len() - 1) as u32
    }

    fn snap(&mut self, vals: &[(Val, u32)]) -> u32 {
        let start = self.snap_vals.len() as u32;
        for &(val, slot) in vals {
            self.snap_vals.push(SnapVal { val, slot });
        }
        let code = -1 - self.snaps.len() as i64;
        self.snaps.push(Snap {
            start,
            len: vals.len() as u32,
            code,
        });
        (self.snaps.len() - 1) as u32
    }

    /// Starts the root; returns the `inputs` and `control` pointer values.
    pub fn entry(&mut self) -> (Val, Val) {
        let a = self.val(Ty::I64);
        let b = self.val(Ty::I64);
        let mut i = Inst::new(Op::Entry, Ty::I64);
        i.a = a;
        i.b = 2;
        self.push(i);
        (a, b)
    }

    /// Starts the root of a function called with arguments of the types
    /// `tys` (see [`Op::Entry`]): value `first + k` is argument `k`. Returns
    /// `first`.
    pub fn entry_args(&mut self, tys: &[Ty]) -> Val {
        let first = self.val_ty.len() as Val;
        for &t in tys {
            self.val(t);
        }
        let mut i = Inst::new(Op::Entry, Ty::I64);
        i.a = first;
        i.b = tys.len() as u32;
        self.push(i);
        first
    }

    /// Ends the region by returning `a`, and `b` unless it is `NONE` (see
    /// [`Op::Ret`]).
    pub fn ret(&mut self, a: Val, b: Val) {
        let mut i = Inst::new(Op::Ret, Ty::I64);
        i.a = a;
        i.b = b;
        self.push(i);
    }

    /// A call of the code at byte `at` from the start of this function's
    /// image (see [`CALL_REL`]), with an integer result and a second one of
    /// type `ret2`. Returns the two results, `NONE` for the second when
    /// `ret2` is `None`.
    pub fn call_rel(&mut self, at: i64, args: &[Val], ret2: Option<Ty>) -> (Val, Val) {
        let dst = self.call(0, args, Some(Ty::I64));
        let second = match ret2 {
            Some(ty) => self.val(ty),
            None => NONE,
        };
        let i = self.insts.last_mut().expect("the call was pushed");
        i.imm = at;
        i.scale = CALL_REL;
        i.c = second;
        (dst, second)
    }

    pub fn iconst(&mut self, v: i64) -> Val {
        let dst = self.val(Ty::I64);
        let mut i = Inst::new(Op::Const, Ty::I64);
        i.dst = dst;
        i.imm = v;
        self.push(i);
        dst
    }

    pub fn fconst(&mut self, v: f64) -> Val {
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(Op::Const, Ty::F64);
        i.dst = dst;
        i.imm = v.to_bits() as i64;
        self.push(i);
        dst
    }

    pub fn load(&mut self, ty: Ty, base: Val, index: Val, scale: u8, disp: i32) -> Val {
        let dst = self.val(ty);
        let mut i = Inst::new(Op::Load, ty);
        i.dst = dst;
        i.a = base;
        i.b = index;
        i.scale = scale;
        i.imm = i64::from(disp);
        self.push(i);
        dst
    }

    pub fn store(&mut self, base: Val, index: Val, scale: u8, disp: i32, value: Val) {
        let ty = self.val_ty[value as usize];
        let mut i = Inst::new(Op::Store, ty);
        i.a = base;
        i.b = index;
        i.c = value;
        i.scale = scale;
        i.imm = i64::from(disp);
        self.push(i);
    }

    pub fn bin(&mut self, op: Op, a: Val, b: Val) -> Val {
        let ty = if matches!(
            op,
            Op::FAdd | Op::FSub | Op::FMul | Op::FDiv | Op::FCopySign
        ) {
            Ty::F64
        } else {
            Ty::I64
        };
        let dst = self.val(ty);
        let mut i = Inst::new(op, ty);
        i.dst = dst;
        i.a = a;
        i.b = b;
        // A cold shift is made by `sar_cold`, which knows its count.
        if self.cold && is_coldable(op) && op != Op::Sar {
            i.scale = COLD;
        }
        self.push(i);
        dst
    }

    /// `a >> count` (arithmetic) by the constant `count`, cold when the
    /// builder is: an integer a stub makes from another, as the index of a
    /// loop from the scaled copy the loop counts in.
    pub fn sar_cold(&mut self, a: Val, count: u8) -> Val {
        let b = self.iconst(i64::from(count));
        let dst = self.val(Ty::I64);
        let mut i = Inst::new(Op::Sar, Ty::I64);
        i.dst = dst;
        i.a = a;
        i.b = b;
        if self.cold {
            i.scale = COLD;
        }
        self.push(i);
        dst
    }

    pub fn fsqrt(&mut self, a: Val) -> Val {
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(Op::FSqrt, Ty::F64);
        i.dst = dst;
        i.a = a;
        self.push(i);
        dst
    }

    /// `-a`, `|a|`, `floor(a)`, `ceil(a)` or `trunc(a)` (`op` is `FNeg`, `FAbs`,
    /// `FFloor`, `FCeil` or `FTrunc`).
    pub fn funary(&mut self, op: Op, a: Val) -> Val {
        debug_assert!(is_coldable(op) && !matches!(op, Op::FAdd | Op::FSub | Op::FMul | Op::FDiv));
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(op, Ty::F64);
        i.dst = dst;
        i.a = a;
        if self.cold {
            i.scale = COLD;
        }
        self.push(i);
        dst
    }

    /// `c - a * b` with one rounding (see [`Op::FNma`]).
    pub fn fnma(&mut self, a: Val, b: Val, c: Val) -> Val {
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(Op::FNma, Ty::F64);
        i.dst = dst;
        i.a = a;
        i.b = b;
        i.c = c;
        self.push(i);
        dst
    }

    /// The bits of a float, as an integer.
    pub fn bits(&mut self, a: Val) -> Val {
        let dst = self.val(Ty::I64);
        let mut i = Inst::new(Op::Bits, Ty::I64);
        i.dst = dst;
        i.a = a;
        self.push(i);
        dst
    }

    /// The float with the bits of an integer.
    pub fn from_bits(&mut self, a: Val) -> Val {
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(Op::FromBits, Ty::F64);
        i.dst = dst;
        i.a = a;
        self.push(i);
        dst
    }

    pub fn i2f(&mut self, a: Val) -> Val {
        let dst = self.val(Ty::F64);
        let mut i = Inst::new(Op::I2F, Ty::F64);
        i.dst = dst;
        i.a = a;
        if self.cold {
            i.scale = COLD;
        }
        self.push(i);
        dst
    }

    pub fn select(&mut self, cc: Cc, a: Val, b: Val, t: Val, f: Val) -> Val {
        let dst = self.val(Ty::I64);
        let mut i = Inst::new(Op::Select, Ty::I64);
        i.dst = dst;
        i.cc = cc;
        i.a = a;
        i.b = b;
        i.c = t;
        i.d = f;
        self.push(i);
        dst
    }

    /// `(a cc b) ? t : f` with any mix of integer or float compare and arms.
    pub fn select_any(&mut self, cc: Cc, a: Val, b: Val, t: Val, f: Val) -> Val {
        let ty = self.val_ty[t as usize];
        let dst = self.val(ty);
        let mut i = Inst::new(Op::Select, ty);
        i.dst = dst;
        i.cc = cc;
        i.a = a;
        i.b = b;
        i.c = t;
        i.d = f;
        self.push(i);
        dst
    }

    /// A guard that exits through a new snapshot. Returns the exit index.
    pub fn guard(&mut self, cc: Cc, a: Val, b: Val, snap: &[(Val, u32)]) -> u32 {
        let s = self.snap(snap);
        let mut i = Inst::new(Op::Guard, self.val_ty[a as usize]);
        i.cc = cc;
        i.a = a;
        i.b = b;
        i.c = s;
        self.push(i);
        s
    }

    /// A guard that passes `args` to side `side` (numbered from 0, in
    /// `begin_side` order) when the condition fails.
    pub fn guard_side(&mut self, cc: Cc, a: Val, b: Val, side: u32, args: &[Val]) -> u32 {
        let pairs: Vec<(Val, u32)> = args.iter().map(|&v| (v, 0)).collect();
        let s = self.snap(&pairs);
        let mut i = Inst::new(Op::Guard, self.val_ty[a as usize]);
        i.cc = cc;
        i.a = a;
        i.b = b;
        i.c = s;
        i.d = side;
        self.push(i);
        s
    }

    /// Opens a loop. Returns the first parameter; parameter `k` is
    /// `first + k` and has the type of `init[k]`.
    pub fn header(&mut self, init: &[Val]) -> Val {
        let first = self.val_ty.len() as Val;
        for &v in init {
            let ty = self.val_ty[v as usize];
            self.val(ty);
        }
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(init);
        let mut i = Inst::new(Op::Header, Ty::I64);
        i.a = first;
        i.b = init.len() as u32;
        i.c = start;
        self.push(i);
        first
    }

    /// Opens the outer header of a root that sides rejoin. Returns the first
    /// parameter, as [`Func::header`] does.
    pub fn join(&mut self, init: &[Val]) -> Val {
        let first = self.val_ty.len() as Val;
        for &v in init {
            let ty = self.val_ty[v as usize];
            self.val(ty);
        }
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(init);
        let mut i = Inst::new(Op::Join, Ty::I64);
        i.a = first;
        i.b = init.len() as u32;
        i.c = start;
        self.push(i);
        first
    }

    /// Ends a side by jumping to the root's `Join` with `args`.
    pub fn rejoin(&mut self, args: &[Val]) {
        self.rejoin_to(0, args);
    }

    /// Ends a side by jumping to the `Join` of region `region` with `args`.
    pub fn rejoin_to(&mut self, region: u32, args: &[Val]) {
        self.rejoin_nth(region, 0, args);
    }

    /// Ends a side by jumping to the `nth` `Join` of region `region` (a region
    /// may have several, one above the other), with `args`.
    pub fn rejoin_nth(&mut self, region: u32, nth: u32, args: &[Val]) {
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(args);
        let mut i = Inst::new(Op::Rejoin, Ty::I64);
        i.a = start;
        i.b = args.len() as u32;
        i.d = region;
        i.e = if nth == 0 { NONE } else { nth };
        self.push(i);
    }

    /// Ends a region by entering side `side` with `args` as its parameters.
    pub fn goto_side(&mut self, side: u32, args: &[Val]) {
        let pairs: Vec<(Val, u32)> = args.iter().map(|&v| (v, 0)).collect();
        let s = self.snap(&pairs);
        let mut i = Inst::new(Op::Goto, Ty::I64);
        i.c = s;
        i.d = side;
        self.push(i);
    }

    /// Opens a forward skip (see [`Op::If`]): the block that follows runs when
    /// `a cc b` holds. Returns the token [`Func::end_if`] takes.
    pub fn begin_if(&mut self, cc: Cc, a: Val, b: Val) -> u32 {
        let mut i = Inst::new(Op::If, self.val_ty[a as usize]);
        i.cc = cc;
        i.a = a;
        i.b = b;
        self.push(i);
        (self.insts.len() - 1) as u32
    }

    /// Closes the skip `open` made (see [`Op::EndIf`]). `joins` holds a
    /// `(block value, skipped value)` pair for each join value. Returns the
    /// first join value, the next ones follow it.
    pub fn end_if(&mut self, open: u32, joins: &[(Val, Val)]) -> Val {
        let first = self.val_ty.len() as Val;
        for &(t, _) in joins {
            let ty = self.val_ty[t as usize];
            self.val(ty);
        }
        let start = self.lists.len() as u32;
        self.lists.extend(joins.iter().map(|&(t, _)| t));
        self.lists.extend(joins.iter().map(|&(_, e)| e));
        let mut i = Inst::new(Op::EndIf, Ty::I64);
        i.a = first;
        i.b = joins.len() as u32;
        i.c = start;
        self.insts[open as usize].e = self.insts.len() as u32;
        self.push(i);
        first
    }

    /// A new frame cell.
    pub fn new_cell(&mut self) -> u32 {
        self.cells += 1;
        self.cells - 1
    }

    pub fn cell_store(&mut self, cell: u32, value: Val) {
        let mut i = Inst::new(Op::CellStore, self.val_ty[value as usize]);
        i.c = value;
        i.imm = i64::from(cell);
        self.push(i);
    }

    pub fn cell_load(&mut self, ty: Ty, cell: u32) -> Val {
        let dst = self.val(ty);
        let mut i = Inst::new(Op::CellLoad, ty);
        i.dst = dst;
        i.imm = i64::from(cell);
        self.push(i);
        dst
    }

    /// `if a cc b { call }` out of line (see [`Op::CallIf`]); the function is
    /// the one at address `target`, or the pointer at byte `target` of the
    /// control block when `ctl`.
    pub fn call_if(&mut self, cc: Cc, (a, b): (Val, Val), target: usize, ctl: bool, args: &[Val]) {
        self.call_if_storing(cc, (a, b), target, ctl, args, None);
    }

    /// [`Func::call_if`] that stores the call's result at byte `result_at` of
    /// the control block when the call is made.
    pub fn call_if_storing(
        &mut self,
        cc: Cc,
        (a, b): (Val, Val),
        target: usize,
        ctl: bool,
        args: &[Val],
        result_at: Option<u32>,
    ) {
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(args);
        let mut i = Inst::new(Op::CallIf, self.val_ty[a as usize]);
        i.cc = cc;
        i.a = a;
        i.b = b;
        i.c = start;
        i.d = args.len() as u32;
        i.imm = target as i64;
        i.scale = u8::from(ctl);
        i.e = result_at.unwrap_or(NONE);
        self.push(i);
    }

    /// A call of the function whose pointer is at byte `off` of the control
    /// block (see [`Op::CallCtl`]).
    pub fn call_ctl(&mut self, off: i32, args: &[Val], ret: Option<Ty>) -> Val {
        let dst = match ret {
            Some(ty) => self.val(ty),
            None => NONE,
        };
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(args);
        let mut i = Inst::new(Op::CallCtl, ret.unwrap_or(Ty::I64));
        i.dst = dst;
        i.a = start;
        i.b = args.len() as u32;
        i.imm = i64::from(off);
        self.push(i);
        dst
    }

    pub fn back(&mut self, args: &[Val]) {
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(args);
        let mut i = Inst::new(Op::Back, Ty::I64);
        i.a = start;
        i.b = args.len() as u32;
        self.push(i);
    }

    pub fn exit(&mut self, snap: &[(Val, u32)]) -> u32 {
        let s = self.snap(snap);
        let mut i = Inst::new(Op::Exit, Ty::I64);
        i.c = s;
        self.push(i);
        s
    }

    /// The exit-area pointer (see [`Op::ExitArea`]).
    pub fn exit_area(&mut self) -> Val {
        let dst = self.val(Ty::I64);
        let mut i = Inst::new(Op::ExitArea, Ty::I64);
        i.dst = dst;
        self.push(i);
        dst
    }

    /// A call of the C function at `target` (see [`Op::Call`]). Returns the
    /// result value of type `ret`, or `NONE` when `ret` is `None`.
    pub fn call(&mut self, target: usize, args: &[Val], ret: Option<Ty>) -> Val {
        let dst = match ret {
            Some(ty) => self.val(ty),
            None => NONE,
        };
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(args);
        let mut i = Inst::new(Op::Call, ret.unwrap_or(Ty::I64));
        i.dst = dst;
        i.a = start;
        i.b = args.len() as u32;
        i.imm = target as i64;
        self.push(i);
        dst
    }

    /// Starts a side with parameters of the given types. Returns its index
    /// and first parameter value.
    pub fn begin_side(&mut self, tys: &[Ty]) -> (u32, Val) {
        let idx = self.side_starts.len() as u32;
        self.side_starts.push(self.insts.len() as u32);
        let first = self.val_ty.len() as Val;
        for &t in tys {
            self.val(t);
        }
        let mut i = Inst::new(Op::SideEntry, Ty::I64);
        i.a = first;
        i.b = tys.len() as u32;
        self.push(i);
        (idx, first)
    }

    #[must_use]
    pub fn list(&self, start: u32, len: u32) -> &[u32] {
        &self.lists[start as usize..(start + len) as usize]
    }

    #[must_use]
    pub fn snap_of(&self, s: u32) -> &[SnapVal] {
        let s = self.snaps[s as usize];
        &self.snap_vals[s.start as usize..(s.start + s.len) as usize]
    }

    /// Exit-area length (in 8-byte words) the snapshots need.
    #[must_use]
    pub fn exit_area_len(&self) -> usize {
        self.snap_vals
            .iter()
            .filter(|s| s.slot & (CONTROL_SLOT | RETURN_SLOT) == 0)
            .map(|s| (s.slot & !INDIRECT_SLOT) as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Checks structure and types. For tests and debugging.
    ///
    /// # Errors
    ///
    /// Returns a description of the first problem found.
    pub fn validate(&self) -> Result<(), String> {
        let mut region_of = vec![u32::MAX; self.num_vals()];
        for r in 0..=self.num_sides() {
            let rid = r as u32;
            let (s, e) = self.region(r);
            if s >= e {
                return Err(format!("region {r} is empty"));
            }
            let first = self.insts[s].op;
            if (r == 0 && first != Op::Entry) || (r > 0 && first != Op::SideEntry) {
                return Err(format!("region {r} starts with {first:?}"));
            }
            let last = self.insts[e - 1].op;
            if !matches!(last, Op::Back | Op::Exit | Op::Rejoin | Op::Goto | Op::Ret) {
                return Err(format!("region {r} ends with {last:?}"));
            }
            let mut open: Vec<u32> = Vec::new();
            for idx in s..e {
                let i = self.insts[idx];
                if !open.is_empty()
                    && matches!(
                        i.op,
                        Op::Entry
                            | Op::SideEntry
                            | Op::Header
                            | Op::Join
                            | Op::Back
                            | Op::Exit
                            | Op::Ret
                            | Op::Rejoin
                            | Op::Goto
                    )
                {
                    return Err(format!("inst {idx} {:?} inside an if", i.op));
                }
                let chk = |v: u32, d: &Vec<u32>| -> Result<(), String> {
                    if v == NONE || d.get(v as usize).copied() != Some(rid) {
                        Err(format!("inst {idx} {:?} uses undefined value {v}", i.op))
                    } else {
                        Ok(())
                    }
                };
                match i.op {
                    Op::Entry | Op::SideEntry => {
                        if i.op == Op::Entry && i.b < 2 {
                            return Err(format!("entry {idx} takes {} arguments", i.b));
                        }
                        for v in i.a..i.a + i.b {
                            region_of[v as usize] = rid;
                        }
                    }
                    Op::Ret => {
                        chk(i.a, &region_of)?;
                        if self.val_ty[i.a as usize] != Ty::I64 {
                            return Err(format!("ret {idx} returns a float first"));
                        }
                        if i.b != NONE {
                            chk(i.b, &region_of)?;
                        }
                    }
                    Op::Const => region_of[i.dst as usize] = rid,
                    Op::Load => {
                        chk(i.a, &region_of)?;
                        if i.b != NONE {
                            chk(i.b, &region_of)?;
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::Store => {
                        chk(i.a, &region_of)?;
                        if i.b != NONE {
                            chk(i.b, &region_of)?;
                        }
                        chk(i.c, &region_of)?;
                    }
                    Op::Shl | Op::Shr | Op::Sar => {
                        chk(i.a, &region_of)?;
                        chk(i.b, &region_of)?;
                        region_of[i.dst as usize] = rid;
                    }
                    Op::FNma => {
                        for v in [i.a, i.b, i.c] {
                            chk(v, &region_of)?;
                        }
                        region_of[i.dst as usize] = rid;
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
                        chk(i.a, &region_of)?;
                        if i.scale == COLD && !is_coldable(i.op) {
                            return Err(format!("{:?} {idx} cannot be cold", i.op));
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::Load8 | Op::Load32 => {
                        chk(i.a, &region_of)?;
                        if i.b != NONE {
                            chk(i.b, &region_of)?;
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::Store8 | Op::AddMem => {
                        chk(i.a, &region_of)?;
                        if i.b != NONE {
                            chk(i.b, &region_of)?;
                        }
                        chk(i.c, &region_of)?;
                    }
                    Op::Select => {
                        for v in [i.a, i.b, i.c, i.d] {
                            chk(v, &region_of)?;
                        }
                        if self.val_ty[i.a as usize] != self.val_ty[i.b as usize]
                            || self.val_ty[i.c as usize] != self.val_ty[i.d as usize]
                            || self.val_ty[i.c as usize] != self.val_ty[i.dst as usize]
                        {
                            return Err(format!("select {idx} mixes types"));
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::Guard => {
                        chk(i.a, &region_of)?;
                        chk(i.b, &region_of)?;
                        if self.val_ty[i.a as usize] != self.val_ty[i.b as usize] {
                            return Err(format!("guard {idx} compares mixed types"));
                        }
                        if i.scale == 1 {
                            // The flags of the instruction before it (constants
                            // make no code): not those of a compare or a join.
                            let mut before = idx;
                            while before > s && self.insts[before - 1].op == Op::Const {
                                before -= 1;
                            }
                            if before > s && matches!(self.insts[before - 1].op, Op::If | Op::EndIf)
                            {
                                return Err(format!("guard {idx} reads the flags of a skip"));
                            }
                        }
                        for sv in self.snap_of(i.c) {
                            if i.d != NONE && sv.slot & (INDIRECT_SLOT | RETURN_SLOT) != 0 {
                                return Err(format!("guard {idx} passes a word to a side"));
                            }
                            if names_value(sv) {
                                chk(sv.val, &region_of)?;
                            }
                        }
                        if i.d != NONE {
                            let (ss, _) = self.region(1 + i.d as usize);
                            let side = self.insts[ss];
                            let args = self.snap_of(i.c);
                            if side.b as usize != args.len() {
                                return Err(format!(
                                    "guard {idx} passes {} values to a side of {}",
                                    args.len(),
                                    side.b
                                ));
                            }
                            for (k, sv) in args.iter().enumerate() {
                                if self.val_ty[sv.val as usize] != self.val_ty[side.a as usize + k]
                                {
                                    return Err(format!("guard {idx} side arg {k} type mismatch"));
                                }
                            }
                        }
                    }
                    Op::Header | Op::Join => {
                        for k in 0..i.b {
                            chk(self.lists[(i.c + k) as usize], &region_of)?;
                            region_of[(i.a + k) as usize] = rid;
                        }
                    }
                    Op::Goto => {
                        for sv in self.snap_of(i.c) {
                            chk(sv.val, &region_of)?;
                        }
                        if i.d as usize >= self.num_sides() {
                            return Err(format!("goto {idx} names no side"));
                        }
                        let (ss, _) = self.region(1 + i.d as usize);
                        let side = self.insts[ss];
                        let args = self.snap_of(i.c);
                        if side.b as usize != args.len() {
                            return Err(format!(
                                "goto {idx} passes {} values to a side of {}",
                                args.len(),
                                side.b
                            ));
                        }
                        for (k, sv) in args.iter().enumerate() {
                            if self.val_ty[sv.val as usize] != self.val_ty[side.a as usize + k] {
                                return Err(format!("goto {idx} side arg {k} type mismatch"));
                            }
                        }
                    }
                    Op::CellStore => {
                        chk(i.c, &region_of)?;
                        if i.imm as u32 >= self.cells {
                            return Err(format!("cell store {idx} names no cell"));
                        }
                    }
                    Op::CellLoad => {
                        if i.imm as u32 >= self.cells {
                            return Err(format!("cell load {idx} names no cell"));
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::ExitArea => {
                        if i.ty != Ty::I64 || self.val_ty[i.dst as usize] != Ty::I64 {
                            return Err(format!("exit area {idx} is not an integer"));
                        }
                        region_of[i.dst as usize] = rid;
                    }
                    Op::Rejoin => {
                        if i.d as usize > self.num_sides() {
                            return Err(format!("rejoin {idx} names no region"));
                        }
                        let (js, je) = self.region(i.d as usize);
                        let nth = if i.e == NONE { 0 } else { i.e as usize };
                        let Some(join) = self.insts[js..je]
                            .iter()
                            .filter(|j| j.op == Op::Join)
                            .nth(nth)
                        else {
                            return Err(format!("rejoin {idx} without a join"));
                        };
                        if join.b != i.b {
                            return Err(format!(
                                "rejoin {idx} passes {} values to {}",
                                i.b, join.b
                            ));
                        }
                        for (k, &v) in self.list(i.a, i.b).iter().enumerate() {
                            chk(v, &region_of)?;
                            if self.val_ty[v as usize] != self.val_ty[join.a as usize + k] {
                                return Err(format!("rejoin {idx} argument {k} type mismatch"));
                            }
                        }
                    }
                    Op::Back => {
                        for &v in self.list(i.a, i.b) {
                            chk(v, &region_of)?;
                        }
                    }
                    Op::Exit => {
                        for sv in self.snap_of(i.c) {
                            if names_value(sv) {
                                chk(sv.val, &region_of)?;
                            }
                        }
                    }
                    Op::CallIf => {
                        chk(i.a, &region_of)?;
                        chk(i.b, &region_of)?;
                        if self.val_ty[i.a as usize] != self.val_ty[i.b as usize] {
                            return Err(format!("call-if {idx} compares mixed types"));
                        }
                        for &v in self.list(i.c, i.d) {
                            chk(v, &region_of)?;
                        }
                    }
                    Op::If => {
                        chk(i.a, &region_of)?;
                        chk(i.b, &region_of)?;
                        if self.val_ty[i.a as usize] != self.val_ty[i.b as usize] {
                            return Err(format!("if {idx} compares mixed types"));
                        }
                        let end = i.e as usize;
                        if i.e == NONE || end <= idx || end >= e || self.insts[end].op != Op::EndIf
                        {
                            return Err(format!("if {idx} names no end"));
                        }
                        let end = self.insts[end];
                        for &v in self.list(end.c + end.b, end.b) {
                            chk(v, &region_of)?;
                        }
                        open.push(idx as u32);
                    }
                    Op::EndIf => {
                        let Some(start) = open.pop() else {
                            return Err(format!("end-if {idx} without an if"));
                        };
                        if self.insts[start as usize].e as usize != idx {
                            return Err(format!("end-if {idx} closes another if"));
                        }
                        for k in 0..i.b {
                            let t = self.lists[(i.c + k) as usize];
                            let s = self.lists[(i.c + i.b + k) as usize];
                            chk(t, &region_of)?;
                            if self.val_ty[t as usize] != self.val_ty[s as usize]
                                || self.val_ty[t as usize] != self.val_ty[(i.a + k) as usize]
                            {
                                return Err(format!("end-if {idx} join {k} type mismatch"));
                            }
                        }
                        // What the block defined is out of scope past its end.
                        for inner in &self.insts[start as usize + 1..idx] {
                            if inner.op == Op::EndIf {
                                for v in inner.a..inner.a + inner.b {
                                    region_of[v as usize] = u32::MAX;
                                }
                            } else if inner.dst != NONE {
                                region_of[inner.dst as usize] = u32::MAX;
                            }
                        }
                        for v in i.a..i.a + i.b {
                            region_of[v as usize] = rid;
                        }
                    }
                    Op::Call | Op::CallCtl => {
                        let (mut ints, mut floats) = (0, 0);
                        for &v in self.list(i.a, i.b) {
                            chk(v, &region_of)?;
                            match self.val_ty[v as usize] {
                                Ty::I64 => ints += 1,
                                Ty::F64 => floats += 1,
                            }
                        }
                        if ints + floats > 32 {
                            return Err(format!(
                                "call {idx} has {ints} integer and {floats} float arguments"
                            ));
                        }
                        if i.dst != NONE {
                            if self.val_ty[i.dst as usize] != i.ty {
                                return Err(format!("call {idx} result type"));
                            }
                            region_of[i.dst as usize] = rid;
                        }
                        if i.c != NONE {
                            if i.op != Op::Call || i.scale != CALL_REL || i.ty != Ty::I64 {
                                return Err(format!("call {idx} has a second result"));
                            }
                            region_of[i.c as usize] = rid;
                        }
                    }
                    _ => {
                        chk(i.a, &region_of)?;
                        chk(i.b, &region_of)?;
                        region_of[i.dst as usize] = rid;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Func {
    /// A readable listing, for debugging.
    #[must_use]
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for r in 0..=self.num_sides() {
            let (s, e) = self.region(r);
            let _ = writeln!(out, "region {r}:");
            for idx in s..e {
                let i = self.insts[idx];
                let _ = write!(out, "  {idx:3}: ");
                let t = |v: u32| {
                    if v == NONE {
                        "-".to_string()
                    } else {
                        format!("v{v}")
                    }
                };
                let _ = match i.op {
                    Op::Entry | Op::SideEntry => {
                        writeln!(out, "{:?} -> v{}..v{}", i.op, i.a, i.a + i.b)
                    }
                    Op::Const => writeln!(out, "v{} = {:?} const {:#x}", i.dst, i.ty, i.imm),
                    Op::Load => writeln!(
                        out,
                        "v{} = load {:?} [{} + {}*{} + {}]",
                        i.dst,
                        i.ty,
                        t(i.a),
                        t(i.b),
                        i.scale,
                        i.imm
                    ),
                    Op::Store => writeln!(
                        out,
                        "store {:?} [{} + {}*{} + {}] = {}",
                        i.ty,
                        t(i.a),
                        t(i.b),
                        i.scale,
                        i.imm,
                        t(i.c)
                    ),
                    Op::Select => writeln!(
                        out,
                        "v{} = select {:?} {} {} ? {} : {}",
                        i.dst,
                        i.cc,
                        t(i.a),
                        t(i.b),
                        t(i.c),
                        t(i.d)
                    ),
                    Op::Guard => writeln!(
                        out,
                        "guard {:?} {:?} {} {} -> {} {:?}",
                        i.ty,
                        i.cc,
                        t(i.a),
                        t(i.b),
                        if i.d == NONE {
                            format!("exit {}", i.c)
                        } else {
                            format!("side {}", i.d)
                        },
                        self.snap_of(i.c)
                            .iter()
                            .map(|s| (s.val, s.slot))
                            .collect::<Vec<_>>()
                    ),
                    Op::Header => writeln!(
                        out,
                        "header v{}..v{} = {:?}",
                        i.a,
                        i.a + i.b,
                        self.list(i.c, i.b)
                    ),
                    Op::Back => writeln!(out, "back {:?}", self.list(i.a, i.b)),
                    Op::Join => writeln!(
                        out,
                        "join v{}..v{} = {:?}",
                        i.a,
                        i.a + i.b,
                        self.list(i.c, i.b)
                    ),
                    Op::Rejoin => writeln!(
                        out,
                        "rejoin {}.{} {:?}",
                        i.d,
                        if i.e == NONE { 0 } else { i.e },
                        self.list(i.a, i.b)
                    ),
                    Op::Goto => writeln!(
                        out,
                        "goto side {} {:?}",
                        i.d,
                        self.snap_of(i.c).iter().map(|s| s.val).collect::<Vec<_>>()
                    ),
                    Op::CallIf => writeln!(
                        out,
                        "if {} {:?} {} call {:#x} {:?}",
                        t(i.a),
                        i.cc,
                        t(i.b),
                        i.imm,
                        self.list(i.c, i.d)
                    ),
                    Op::If => writeln!(out, "if {} {:?} {} until {}", t(i.a), i.cc, t(i.b), i.e),
                    Op::EndIf => writeln!(
                        out,
                        "end-if v{}..v{} = {:?} else {:?}",
                        i.a,
                        i.a + i.b,
                        self.list(i.c, i.b),
                        self.list(i.c + i.b, i.b)
                    ),
                    Op::CellStore => writeln!(out, "cell[{}] = {}", i.imm, t(i.c)),
                    Op::CellLoad => writeln!(out, "{} = cell[{}]", t(i.dst), i.imm),
                    Op::ExitArea => writeln!(out, "{} = exit area", t(i.dst)),
                    Op::CallCtl => writeln!(
                        out,
                        "{} = call [control + {}] {:?}",
                        t(i.dst),
                        i.imm,
                        self.list(i.a, i.b)
                    ),
                    Op::Call if i.scale == CALL_REL => writeln!(
                        out,
                        "{}, {} = call image{:+} {:?}",
                        t(i.dst),
                        t(i.c),
                        i.imm,
                        self.list(i.a, i.b)
                    ),
                    Op::Call => writeln!(
                        out,
                        "{} = call {:#x} {:?}",
                        t(i.dst),
                        i.imm,
                        self.list(i.a, i.b)
                    ),
                    Op::Ret => writeln!(out, "ret {} {}", t(i.a), t(i.b)),
                    Op::Exit => writeln!(
                        out,
                        "exit {} {:?}",
                        i.c,
                        self.snap_of(i.c)
                            .iter()
                            .map(|s| (s.val, s.slot))
                            .collect::<Vec<_>>()
                    ),
                    Op::FNma => writeln!(out, "v{} = fnma {} {} {}", i.dst, t(i.a), t(i.b), t(i.c)),
                    _ => writeln!(
                        out,
                        "v{} = {:?} {} {}{}",
                        i.dst,
                        i.op,
                        t(i.a),
                        t(i.b),
                        if i.scale == COLD && is_coldable(i.op) {
                            " (cold)"
                        } else {
                            ""
                        }
                    ),
                };
            }
        }
        out
    }
}
