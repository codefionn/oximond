//! The operations that call into the VM: helpers (`CallHelper`,
//! `HelperResult`), host calls, hash lookups and table stores.
//!
//! Every call here is a plain call of the function Cranelift's code reaches
//! on its slow path, with the arguments it builds (`call_helper_call`,
//! `host_call_in_vm`, `hash_call`, `array_set` in `emit.rs`). Where Cranelift
//! answers some cases in line first (an in-place store) this backend always
//! calls: the VM's function does the same work, so the observable result is
//! the same, only slower. The length of a lane string is in line, as in
//! Cranelift (`string_length`). The string-index probe and the stores through
//! it are the other exception, in `probe.rs`, behind
//! `OXIMOND_JIT_MIR_PROBE`: with it set to `0` a tree that probes is
//! declined. A value live across the call is kept in a callee-saved register
//! or spilled once by the allocator, which is why the register-preserving
//! stubs of `SharedStubs` are not called: they exist for a call on a cold
//! path, and none of these calls has a fast path to be cold beside.
//!
//! A call of a host C function (`HelperId::vm_call`) reads the state of its
//! guard's exit from the exit area: the code stores that state there in line
//! just before the call (`spill_exit`, through `Op::ExitArea`), the words the
//! exit's stub writes, as Cranelift's code does.
//!
//! Anything a call may move is read again afterwards: the words of the
//! array and length bindings are loaded when first used after the call
//! (`words_of`), not once at entry, in a member that has such a call.

use super::super::super::emit::RELEASE_WEIGHT;
use super::super::super::exit::TRACE_PENDING_STRINGS;
#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;
use crate::jit::trace::helpers::helper_address;
use crate::jit_ir::trace::{
    HelperEffects, HelperId, HelperType, HostFunction, MAX_HELPER_RESULTS, TABLE_ADDRESS_KEY,
};

use super::super::super::emit::{
    ClosureCellUse, closure_cell_use, string_equal_by_words, string_equal_constant,
    trace_ensure_row, trace_hash_get,
};
use super::super::super::exit::{
    CLOSURE_ROW_CELLS, CLOSURE_ROW_TABLE_WORDS, ExitSlot, TRACE_TAG_BORROWED_STRING,
    TRACE_TAG_FLOAT, TRACE_TAG_INTEGER, TRACE_TAG_STRING,
};

/// What an operand of [`Lowerer::string_equal_by_words`] that the code cannot
/// read is read as: a value whose payload word is zero, and zero words where
/// that payload's storage would be (zero plus the storage offset is inside
/// it, for the select that stands it in reads from its own address). Never
/// written.
static NO_STRING: [u64; 8] = [0; 8];

/// Words of the control block, as byte offsets.
fn control_at(field: usize) -> i64 {
    offset(field)
}

/// What an in-line closure access answers.
enum CellResult {
    /// A value.
    Word(Val),
    /// The comparison `a != b`, held until something reads it.
    NonZero(Val, Val),
}

impl Lowerer<'_, '_> {
    /// Whether Cranelift's lowering of the same tree probes the string index
    /// of the run (`Lowerer::native_probe` in `emit.rs`): some member reads
    /// by it and none moves hash storage.
    pub(super) fn native_probe(&self) -> bool {
        self.members.iter().any(|m| m.ir.uses_string_index())
            && !self.members.iter().any(|m| m.ir.moves_hash_storage())
    }

    /// The reason to leave a helper call to Cranelift, which answers it in
    /// line (`call_helper` in `emit.rs`): a hash store or delete through the
    /// probe while `OXIMOND_JIT_MIR_PROBE=0` (see `probe.rs`). A conservative
    /// test of the condition Cranelift checks. The fill of a field in a made
    /// table (`made.rs`) and the native step of a boxed `next` loop
    /// (`steps.rs`) are in line now.
    fn inline_in_cranelift(&self, helper: HelperId) -> Option<&'static str> {
        if matches!(
            helper,
            HelperId::SET_PRESENT | HelperId::SET_REVIVE | HelperId::DELETE_STRING
        ) && self.native_probe()
            && !crate::jit::debug_env::mir_probe()
        {
            return Some("hash store or delete without the inline probe");
        }
        None
    }

    /// How Cranelift does the helper call without calling it, when the tree's
    /// closure rows are laid out for it (`closure_cell_use` in `emit.rs`).
    fn closure_use(&self, helper: HelperId, args: &[ValueId]) -> Option<ClosureCellUse> {
        if self.closure_words == 0 {
            return None;
        }
        closure_cell_use(
            self.ir,
            helper,
            args,
            self.closure_words == CLOSURE_ROW_TABLE_WORDS,
        )
    }

    /// Whether the upvalue cells reached before `op` are the same after it,
    /// in address and type: `op` runs no other code and writes no cell, or it
    /// is an access of a cell in line, which keeps the cell's type.
    pub(super) fn keeps_cells(&self, op: &TraceOp) -> bool {
        match op {
            TraceOp::CallHelper { helper, args, .. } => matches!(
                self.closure_use(*helper, args),
                Some(
                    ClosureCellUse::Proto | ClosureCellUse::Get { .. } | ClosureCellUse::Set { .. }
                )
            ),
            _ => keeps_probed_keys(op),
        }
    }

    /// Whether `op` reads a row of the run's row storage: a field, a
    /// metatable word or a closure's prototype or cell.
    pub(super) fn reads_rows(op: &TraceOp, ir: &TraceIr, closure_words: usize) -> bool {
        match op {
            TraceOp::TableFieldLoad { .. }
            | TraceOp::TableFieldStore { .. }
            | TraceOp::LaneArrayLoad { .. }
            | TraceOp::LaneArrayStore { .. }
            | TraceOp::MetaGuard { .. } => true,
            TraceOp::CallHelper { helper, args, .. } => {
                closure_words > 0
                    && closure_cell_use(ir, *helper, args, closure_words == CLOSURE_ROW_TABLE_WORDS)
                        .is_some()
            }
            _ => false,
        }
    }

    /// A word of the row of the closure the trace pinned, which is a
    /// constant lane: one load from the rows pointer, with no multiplication.
    fn pinned_row_word(&mut self, pinned: ValueId, word: usize) -> Res<Val> {
        let Some(TraceScalar::Closure(lane)) = self.s.konst[pinned.index()] else {
            let lane = self.get(pinned)?;
            let row = self.row_address(lane)?;
            return Ok(self.load_at(Ty::I64, row, Off::K(0), offset(word * 8)));
        };
        if self.rows == NONE {
            return self.reject("row access without the rows pointer");
        }
        let at = i64::try_from(lane).expect("a lane") * self.row_bytes + offset(word * 8);
        Ok(self.load_at(Ty::I64, self.rows, Off::K(0), at))
    }

    /// The prototype check or cell access of `cell`, in line as in
    /// `closure_cell` of `emit.rs`. The guard leaves where the helper would
    /// miss: the closure has no such cell, or the cell holds a value of
    /// another type. Returns the check's result, if it has one.
    fn closure_cell(
        &mut self,
        cell: ClosureCellUse,
        args: &[ValueId],
        guard: GuardId,
    ) -> Res<Option<CellResult>> {
        let lane = self.get(args[0])?;
        let word = |this: &mut Self, lane: Val, word: usize| -> Res<Val> {
            let row = this.row_address(lane)?;
            Ok(this.load_at(
                Ty::I64,
                row,
                Off::K(0),
                offset(word * mem::size_of::<usize>()),
            ))
        };
        let (index, ty) = match cell {
            ClosureCellUse::Proto => {
                let live = word(self, lane, 0)?;
                let known = self.pinned_row_word(args[1], 0)?;
                // The prototype is the pinned one and is not zero (a row
                // that was never filled): the live word where it equals the
                // pinned one, zero elsewhere, which holds when it is not
                // zero. The answer is that comparison, which a guard on it
                // takes as one.
                let zero = self.iconst(0);
                let same = self.f.select(Cc::Eq, live, known, live, zero);
                return Ok(Some(CellResult::NonZero(same, zero)));
            }
            ClosureCellUse::GetTable { index } => {
                return self
                    .closure_table_cell(lane, index, guard)
                    .map(|lane| Some(CellResult::Word(lane)));
            }
            ClosureCellUse::Get { index, ty } | ClosureCellUse::Set { index, ty } => (index, ty),
        };
        let Some(wanted) = tag_of(ty) else {
            return self.reject("closure cell of an unsupported type");
        };
        // A cell this stretch of the trace reached for the same type: its
        // address and its tag are what they were (see `State::cells`).
        let reached = self
            .s
            .cells
            .iter()
            .position(|(of, at, _, held, _)| *of == lane && *at == index && *held == ty);
        let at = if let Some(at) = reached {
            at
        } else {
            let slot = word(self, lane, 1 + index)?;
            let zero = self.iconst(0);
            self.guard_cc(Cc::Ne, slot, zero, guard)?;
            let tag = self.load8(slot, Off::K(0), 0);
            let wanted = self.iconst(wanted);
            self.guard_cc(Cc::Eq, tag, wanted, guard)?;
            self.s.cells.push((lane, index, slot, ty, NONE));
            self.s.cells.len() - 1
        };
        let (slot, held) = (self.s.cells[at].2, self.s.cells[at].4);
        // Only a number is held as the cell holds it.
        let number = matches!(ty, TraceType::Float | TraceType::Integer);
        if let ClosureCellUse::Get { .. } = cell {
            if held != NONE {
                return Ok(Some(CellResult::Word(held)));
            }
            let value = self.payload_access(ty, slot, Off::K(0), None)?;
            if number {
                self.s.cells[at].4 = value;
            }
            Ok(Some(CellResult::Word(value)))
        } else {
            let stored = self.get(args[2])?;
            self.payload_access(ty, slot, Off::K(0), Some(stored))?;
            // Another closure may share the cell under another index.
            for other in &mut self.s.cells {
                other.4 = NONE;
            }
            if number {
                self.s.cells[at].4 = stored;
            }
            Ok(None)
        }
    }

    /// A read of upvalue `index` that holds a table, with no call, as in
    /// `closure_table_cell` of `emit.rs`: the lane the closure's row names
    /// for the cell, after a check that the cell still holds the table the
    /// lane's bank entry does. The guards leave for a closure with no such
    /// cell, a cell that holds no table, a cell the row has no lane for, or a
    /// cell that changed since the row was filled.
    fn closure_table_cell(&mut self, lane: Val, index: usize, guard: GuardId) -> Res<Val> {
        let word = |this: &mut Self, word: usize| -> Res<Val> {
            let row = this.row_address(lane)?;
            Ok(this.load_at(
                Ty::I64,
                row,
                Off::K(0),
                offset(word * mem::size_of::<usize>()),
            ))
        };
        let slot = word(self, 1 + index)?;
        let named = word(self, 1 + CLOSURE_ROW_CELLS + index)?;
        let bank = self.control_word(mem::offset_of!(TraceControl, bank));
        let zero = self.iconst(0);
        self.guard_cc(Cc::Ne, slot, zero, guard)?;
        self.guard_cc(Cc::Ne, named, zero, guard)?;
        self.guard_cc(Cc::Ne, bank, zero, guard)?;
        let tag = self.load8(slot, Off::K(0), 0);
        let table = self.iconst(i64::from(NATIVE_VALUE_TAG_TABLE));
        self.guard_cc(Cc::Eq, tag, table, guard)?;
        let minus_one = self.iconst(-1);
        let lane = self.f.bin(Op::Add, named, minus_one);
        let four = self.iconst(4);
        let at = self.f.bin(Op::Shl, lane, four);
        let entry = self.f.bin(Op::Add, bank, at);
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        let held = self.load_at(Ty::I64, slot, Off::K(0), payload);
        let banked = self.load_at(Ty::I64, entry, Off::K(0), payload);
        self.guard_cc(Cc::Eq, held, banked, guard)?;
        Ok(lane)
    }

    /// Whether `op` is one this member reads its binding words again after.
    pub(super) fn moves_words(op: &TraceOp) -> bool {
        match op {
            // A self call runs a whole activation, which may store into
            // any table.
            TraceOp::ArraySet { .. } | TraceOp::SelfCall { .. } => true,
            TraceOp::CallHelper { helper, .. } => {
                let effects = helper.effects();
                effects.moves_storage() || effects.takes_words()
            }
            _ => false,
        }
    }

    /// The binding words, loaded now if the member reads them lazily and
    /// has not since the last call that moved them.
    pub(super) fn words_of(&mut self, binding: u32) -> Res<Words> {
        let known = self.s.words[binding as usize];
        if !self.lazy || known.base != NONE || self.s.used[binding as usize] == 0 {
            return Ok(known);
        }
        let words = self.load_binding(binding as usize)?;
        self.s.words[binding as usize] = words;
        Ok(words)
    }

    /// Forgets the binding words a call may have moved, and what was read
    /// through them.
    pub(super) fn forget_words(&mut self) {
        if self.lazy {
            self.s.words.fill(NO_WORDS);
        }
        self.s.slots.clear();
        self.s.fields.clear();
        self.s.rows.clear();
        self.s.lane_words.clear();
        self.s.keys.clear();
        self.forget_bytes();
    }

    pub(super) fn control_word(&mut self, field: usize) -> Val {
        self.load_at(Ty::I64, self.control, Off::K(0), control_at(field))
    }

    pub(super) fn control_address(&mut self, field: usize) -> Val {
        let at = self.iconst(control_at(field));
        self.f.bin(Op::Add, self.control, at)
    }

    /// The address of the first binding slot, which follows the entry inputs.
    pub(super) fn binding_slots(&mut self) -> Val {
        let first = self.members[0].ir.entry.inputs.len();
        let at = self.iconst(offset(first * mem::size_of::<TraceInputSlot>()));
        self.f.bin(Op::Add, self.inputs, at)
    }

    /// A scalar as the entry tag and bits a `TraceInputSlot` holds, as
    /// registers.
    pub(super) fn scalar_words(&mut self, ty: TraceType, value: Val) -> Res<(Val, Val)> {
        if ty == TraceType::StringRef {
            let tag = self.iconst(TRACE_TAG_BORROWED_STRING);
            return Ok((tag, value));
        }
        let Some(tag) = entry_tag(ty) else {
            return self.reject("argument without an entry tag");
        };
        let tag = self.iconst(tag);
        let bits = match ty {
            TraceType::Float => self.f.bits(value),
            TraceType::Nil => self.iconst(0),
            _ => value,
        };
        Ok((tag, bits))
    }

    /// The call of a helper, as `call_helper_call` makes it. Returns nothing:
    /// the first result is `result`, the others are read by `HelperResult`.
    pub(super) fn call_helper(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
    ) -> Res<()> {
        if let Some(reason) = self.inline_in_cranelift(helper) {
            return self.reject(reason);
        }
        self.note_made_table_written(helper, args);
        if let Some(binding) = self.native_store_binding(helper, args) {
            // `t[k] = nil` and the `t[k] = v` after it probe once, as
            // Cranelift's `hash_overwrite` does. The delete waits for its
            // store, which has the same conditions, and the store does both.
            if self.closure_words == 0 {
                if helper == HelperId::DELETE_STRING
                    && let Some((store, _)) = self.store_after_delete(self.at)
                    && self
                        .native_store_binding(HelperId::SET_REVIVE, store)
                        .is_some()
                {
                    return Ok(());
                }
                if helper == HelperId::SET_REVIVE
                    && let Some(delete) = self.delete_before_store(self.at, guard)
                    && self
                        .native_store_binding(HelperId::DELETE_STRING, delete.0)
                        .is_some()
                {
                    return self.probed_overwrite(delete, (args, guard), binding);
                }
            }
            return self.probed_store(helper, args, binding, guard);
        }
        if let Some((borrowed, words)) = string_equal_constant(self.ir, helper, args) {
            return self.string_equal_inline(borrowed, words, result);
        }
        if string_equal_by_words(self.ir, helper, args) {
            return self.string_equal_by_words(helper, args, result, guard);
        }
        if let Some(cell) = self.closure_use(helper, args) {
            let word = self.closure_cell(cell, args, guard)?;
            if let (Some(result), Some(word)) = (result, word) {
                match word {
                    CellResult::Word(val) => self.s.vals[result.index()] = val,
                    CellResult::NonZero(a, b) => {
                        self.s.vals[result.index()] = NONE;
                        self.s.cmps[result.index()] = Cmp {
                            a,
                            b,
                            cc: Cc::Ne,
                            float: false,
                            live: true,
                        };
                    }
                }
            }
            return Ok(());
        }
        if helper == HelperId::TABLE_INIT {
            if self.table_init_in_place(args, guard)? {
                return Ok(());
            }
            self.note_table_init_by_helper(args);
        }
        if let Some(step) = self.box_step_of(helper, args) {
            return self.call_helper_stepping(&step, (helper, args), result, guard);
        }
        self.call_helper_in_vm(helper, args, result, guard)
    }

    /// `borrowed == constant` for a short root-string constant whose storage
    /// words are `words`, as `string_equal_inline` in `emit.rs` makes it:
    /// the words of the borrowed string's storage against those.
    fn string_equal_inline(
        &mut self,
        borrowed: ValueId,
        words: [u64; crate::runtime::NATIVE_STRING_INLINE_WORDS],
        result: Option<ValueId>,
    ) -> Res<()> {
        let value_at = self.get(borrowed)?;
        let payload = self.load_at(
            Ty::I64,
            value_at,
            Off::K(0),
            i64::from(crate::runtime::NATIVE_VALUE_PAYLOAD_OFFSET),
        );
        let mut different = None;
        for (word, constant) in words.into_iter().enumerate() {
            let at = i64::try_from(crate::runtime::NATIVE_STRING_STORAGE_OFFSET + 8 * word)
                .expect("small");
            let held = self.load_at(Ty::I64, payload, Off::K(0), at);
            let constant = self.iconst(constant.cast_signed());
            let apart = self.f.bin(Op::Xor, held, constant);
            different = Some(match different {
                None => apart,
                Some(before) => self.f.bin(Op::Or, before, apart),
            });
        }
        let different = different.expect("an inline string has words");
        if let Some(result) = result {
            let zero = self.iconst(0);
            self.s.vals[result.index()] = NONE;
            self.s.cmps[result.index()] = Cmp {
                a: different,
                b: zero,
                cc: Cc::Eq,
                float: false,
                live: true,
            };
        }
        Ok(())
    }

    /// `==` of two strings by the words of their storage, as
    /// `string_equal_by_words` in `emit.rs` makes it, with the helper called
    /// in a skip when the words cannot tell. MIR has no branch out of the
    /// middle, so an operand the code cannot read is given the address of
    /// [`NO_STRING`] in its place, which every load here can read, and the
    /// helper answers.
    fn string_equal_by_words(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
    ) -> Res<()> {
        use crate::jit_ir::trace::ROOT_STRING_LANE_BASE;
        use crate::runtime::{
            NATIVE_STRING_INLINE_WORDS, NATIVE_STRING_STORAGE_OFFSET, NATIVE_VALUE_PAYLOAD_OFFSET,
            NATIVE_VALUE_TAG_STRING,
        };
        self.prepare_helpers(&[(helper, args, guard)])?;
        let zero = self.iconst(0);
        let one = self.iconst(1);
        let nowhere = self.iconst(std::ptr::addr_of!(NO_STRING) as usize as i64);
        let mut unreadable = zero;
        let mut storages = [zero; 2];
        for (position, &arg) in args.iter().enumerate() {
            let value = self.get(arg)?;
            let value_at = if self.ty(arg) == TraceType::StringRef {
                value
            } else {
                let lane_base = self.iconst(ROOT_STRING_LANE_BASE.cast_signed());
                let roots = self.control_word(mem::offset_of!(TraceControl, roots));
                let roots_len = self.control_word(mem::offset_of!(TraceControl, roots_len));
                let bank = self.control_word(mem::offset_of!(TraceControl, bank));
                let base = self.f.select(Cc::Uge, value, lane_base, roots, bank);
                let first = self.f.select(Cc::Uge, value, lane_base, lane_base, zero);
                let index = self.f.bin(Op::Sub, value, first);
                let unbounded = self.iconst(-1);
                let limit = self
                    .f
                    .select(Cc::Uge, value, lane_base, roots_len, unbounded);
                let outside = self.f.select(Cc::Ult, index, limit, zero, one);
                let missing = self.f.select(Cc::Eq, base, zero, one, zero);
                let bad = self.f.bin(Op::Or, outside, missing);
                let four = self.iconst(4);
                let scaled = self.f.bin(Op::Shl, index, four);
                let at = self.f.bin(Op::Add, base, scaled);
                let at = self.f.select(Cc::Eq, bad, zero, at, nowhere);
                let tag = self.load8(at, Off::K(0), 0);
                let string = self.iconst(i64::from(NATIVE_VALUE_TAG_STRING));
                let wrong = self.f.select(Cc::Eq, tag, string, zero, one);
                let bad = self.f.bin(Op::Or, bad, wrong);
                unreadable = self.f.bin(Op::Or, unreadable, bad);
                at
            };
            storages[position] = self.load_at(
                Ty::I64,
                value_at,
                Off::K(0),
                i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
            );
        }
        // An unreadable operand reads the words of `NO_STRING` instead.
        for storage in &mut storages {
            *storage = self.f.select(Cc::Eq, unreadable, zero, *storage, nowhere);
        }
        let mut different = zero;
        let mut firsts = [zero; 2];
        for word in 0..NATIVE_STRING_INLINE_WORDS {
            let at = i64::try_from(NATIVE_STRING_STORAGE_OFFSET + 8 * word).expect("small");
            let ours = self.load_at(Ty::I64, storages[0], Off::K(0), at);
            let theirs = self.load_at(Ty::I64, storages[1], Off::K(0), at);
            if word == 0 {
                firsts = [ours, theirs];
            }
            let apart = self.f.bin(Op::Xor, ours, theirs);
            different = self.f.bin(Op::Or, different, apart);
        }
        let equal = self.f.select(Cc::Eq, different, zero, one, zero);
        let forms = self.f.bin(Op::And, firsts[0], firsts[1]);
        let low = self.iconst(0xff);
        let heap = self.f.bin(Op::And, forms, low);
        let both_heap = self.f.select(Cc::Ne, heap, zero, one, zero);
        let apart = self.f.select(Cc::Ne, storages[0], storages[1], one, zero);
        let heaps_apart = self.f.bin(Op::And, both_heap, apart);
        let undecided = self.f.bin(Op::Or, unreadable, heaps_apart);
        self.store_by_helpers(&[(helper, args, guard)], undecided)?;
        let out_at = offset(mem::offset_of!(TraceControl, helper_out));
        let answered = self.load8(self.control, Off::K(0), out_at);
        if let Some(result) = result {
            self.s.vals[result.index()] = self.f.select(Cc::Eq, undecided, zero, equal, answered);
        }
        Ok(())
    }

    /// The call of a helper in the VM, with the guard on its status and the
    /// row fills of its results (the end of `call_helper_call` in `emit.rs`).
    pub(super) fn call_helper_in_vm(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
        guard: GuardId,
    ) -> Res<()> {
        let Some(info) = helper.info() else {
            return self.reject("helper without a description");
        };
        let Some(address) = helper_address(helper) else {
            return self.reject("helper nobody installed");
        };
        let makes_slots = self.made_table_call(helper, args, result);
        // A host function call writes the state of its guard's exit back to
        // the frame before it calls: the code stores that state where the
        // exit's stub would, and passes the area and the exit's index after
        // the binding slots, as `call_helper_call` in `emit.rs` does.
        let spill = if helper.is_vm_call() {
            let exit = self.exit_index(guard);
            let area = self.spill_exit(exit)?;
            let index = self.iconst(i64::try_from(exit).expect("exit index fits"));
            Some((area, index))
        } else {
            None
        };
        let mut list = mem::take(&mut self.s.list);
        list.clear();
        let built = self.helper_arguments(helper, info, args, &mut list);
        if let Err(error) = built {
            self.s.list = list;
            return Err(error);
        }
        if let Some((area, index)) = spill {
            let at = usize::from(info.context) + 1 + usize::from(info.effects.takes_words());
            list.insert(at, area);
            list.insert(at + 1, index);
        }
        if makes_slots {
            // The second result word is the address of the new table's value
            // slots, which only a helper that has them writes.
            self.clear_slots_word();
        }
        let status = self.f.call(address, &list, Some(Ty::I64));
        self.s.list = list;
        if info.effects.contains(HelperEffects::MAY_EXIT) {
            let zero = self.iconst(0);
            self.guard_cc(Cc::Eq, status, zero, guard)?;
        }
        if info.effects.moves_storage() || info.effects.takes_words() {
            self.forget_words();
        }
        // The results are loaded right away: the next call overwrites them.
        let words = self.load_helper_results(info, result);
        let first = words[0];
        let fills_table = info.effects.contains(HelperEffects::REWRITES_ARRAYS)
            || matches!(
                helper,
                HelperId::INDEX_HANDLER | HelperId::NEWINDEX_HANDLER | HelperId::CELL_GET_TABLE
            )
            || self.reads_table_addresses()
            // A table a lookup call gives in a loop tree with row fields:
            // the tree may read its fields (`starts[hash].starts`), and its
            // row holds nothing or another run's words until it is filled,
            // as rows are not cleared between runs. Only the lookup: a table
            // a constructor or a cell read gives keeps the row it had. The
            // same condition as `call_helper_call` in `emit.rs`.
            || (helper == HelperId::TABLE_GET_TABLE
                && self.members[0].ir.function.is_none()
                && !self.members[0].ir.fields.is_empty());
        // A closure a call gives has a row of its own to fill, or the
        // prototype word of a stale row answers the check after it.
        let fills_closure = self.closure_words > 0;
        // Every result that is a table or a closure, not only the first: the
        // value of a `next` step is the second result, and a loop over a
        // table of tables reads its elements through the value's row
        // (`box[5]` in `for _, box in pairs (fixed)`).
        for (kind, word) in info.results.iter().zip(words) {
            let fills = match kind {
                HelperType::Table => fills_table,
                HelperType::Closure => fills_closure,
                _ => false,
            };
            let Some(lane) = word.filter(|_| fills) else {
                continue;
            };
            let dispatch = self.control_word(mem::offset_of!(TraceControl, tables));
            self.f.call(
                trace_ensure_row as *const () as usize,
                &[dispatch, lane],
                None,
            );
        }
        if info.effects.contains(HelperEffects::WRITES_METATABLE)
            && self.uses_meta_rows()
            && let Some(table) = args.first()
        {
            let lane = self.get(*table)?;
            let minus_one = self.iconst(-1);
            let complement = self.f.bin(Op::Xor, lane, minus_one);
            let dispatch = self.control_word(mem::offset_of!(TraceControl, tables));
            self.f.call(
                trace_ensure_row as *const () as usize,
                &[dispatch, complement],
                None,
            );
        }
        if makes_slots {
            self.note_made_table(args, result);
        }
        if let (Some(result), Some(word)) = (result, first) {
            self.s.vals[result.index()] = word;
        }
        Ok(())
    }

    /// Writes the state of exit `index` to the exit area in line, as
    /// `spill_exit` in `emit.rs` does: each slot of the exit's descriptor
    /// that holds a value, at the place its stub writes it. The fuel the stub
    /// also writes is left out, as Cranelift leaves it out. Returns the exit
    /// area.
    fn spill_exit(&mut self, index: usize) -> Res<Val> {
        let plan = self.plan;
        // An activation of a function trace has its own area, which its
        // exit stubs write too; the root's code has the entry's.
        let area = match self.function {
            Some(activation) => activation.area,
            None => self.f.exit_area(),
        };
        for ExitSlot { value, ty, slot } in &plan.descriptors[index].slots {
            if matches!(ty, TraceType::Nil | TraceType::Any) {
                continue;
            }
            let val = self.get(*value)?;
            self.store_at(
                area,
                Off::K(0),
                offset(*slot as usize * mem::size_of::<i64>()),
                val,
            );
        }
        Ok(area)
    }

    /// Loads the words a helper left in the control block as its results
    /// (typed as `info` says), and notes the ones after the first for the
    /// `HelperResult`s of `result`.
    pub(super) fn load_helper_results(
        &mut self,
        info: &'static crate::jit_ir::trace::HelperInfo,
        result: Option<ValueId>,
    ) -> [Option<Val>; MAX_HELPER_RESULTS] {
        let out_at = mem::offset_of!(TraceControl, helper_out);
        let mut words = [None; MAX_HELPER_RESULTS];
        for (position, kind) in info.results.iter().enumerate() {
            let at = offset(out_at + position * mem::size_of::<i64>());
            let word = match kind {
                HelperType::Float => self.load_at(Ty::F64, self.control, Off::K(0), at),
                HelperType::Boolean => self.load8(self.control, Off::K(0), at),
                _ => self.load_at(Ty::I64, self.control, Off::K(0), at),
            };
            words[position] = Some(word);
            if position > 0
                && let Some(result) = result
            {
                self.s
                    .hres
                    .push((result.index() as u32, position as u8, word));
            }
        }
        if let (Some(result), Some(first)) = (result, words[0]) {
            self.s.vals[result.index()] = first;
        }
        words
    }

    /// The arguments `call_helper_call` builds: the context, the result
    /// words, the binding slots, the count of a variadic call, then each
    /// parameter (a value is two words, a tag and its bits).
    pub(super) fn helper_arguments(
        &mut self,
        helper: HelperId,
        info: &'static crate::jit_ir::trace::HelperInfo,
        args: &[ValueId],
        list: &mut Vec<Val>,
    ) -> Res<()> {
        if info.context {
            let context = self.control_word(mem::offset_of!(TraceControl, host_context));
            list.push(context);
        }
        let out = self.control_address(mem::offset_of!(TraceControl, helper_out));
        list.push(out);
        if info.effects.takes_words() {
            let words = self.binding_slots();
            list.push(words);
        }
        if info.required < info.params.len() {
            let count = self.iconst(self.argument_count(helper, args.len()));
            list.push(count);
        }
        let mut nil_word = NONE;
        for (position, param) in info.params.iter().enumerate() {
            let Some(arg) = args.get(position) else {
                // A parameter the call leaves out is nil: two zero words.
                if nil_word == NONE {
                    nil_word = self.iconst(0);
                }
                list.push(nil_word);
                list.push(nil_word);
                continue;
            };
            let value = self.get(*arg)?;
            match param {
                HelperType::Value => {
                    let ty = self.ty(*arg);
                    let (tag, bits) = self.scalar_words(ty, value)?;
                    list.push(tag);
                    list.push(bits);
                }
                _ => list.push(value),
            }
        }
        Ok(())
    }

    /// Whether the trace lists the address of a table in a row: not in a
    /// function trace, as in `emit.rs`.
    fn reads_table_addresses(&self) -> bool {
        self.members[0].ir.function.is_none()
            && self.members[0]
                .ir
                .fields
                .iter()
                .any(|field| &*field.key == TABLE_ADDRESS_KEY)
    }

    fn uses_meta_rows(&self) -> bool {
        self.members[0]
            .ir
            .fields
            .iter()
            .any(|field| field.key.as_ref() == META_FIELD_KEY)
    }

    /// `HelperResult`: a word the call loaded.
    pub(super) fn helper_result(
        &mut self,
        call: ValueId,
        index: u8,
        result: Option<ValueId>,
    ) -> Res<()> {
        let found = self
            .s
            .hres
            .iter()
            .find(|(held, at, _)| *held == call.index() as u32 && *at == index)
            .map(|(_, _, word)| *word);
        match found {
            Some(word) => {
                self.set(result, word);
                Ok(())
            }
            None => self.reject("helper result the call did not load"),
        }
    }

    /// A host call, as `host_call` of `emit.rs` makes it: the maximum or
    /// minimum of two integers as a select, the length of a string held by
    /// lane in line (`string_length`), any other call in the VM.
    pub(super) fn host_call(
        &mut self,
        function: HostFunction,
        args: &[ValueId],
        result: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        if matches!(function, HostFunction::MathMax | HostFunction::MathMin) {
            let (left, right) = (self.get(args[0])?, self.get(args[1])?);
            let cc = if function == HostFunction::MathMax {
                Cc::Gt
            } else {
                Cc::Lt
            };
            return Ok(self.f.select(cc, left, right, left, right));
        }
        // A string constant is a root lane, which the VM measures.
        if function == HostFunction::StringLength
            && self.ty(args[0]) == TraceType::String
            && !matches!(self.s.konst[args[0].index()], Some(TraceScalar::String(_)))
        {
            return self.string_length(function, args, result, guard);
        }
        self.host_call_in_vm(function, args, result, guard)
    }

    /// The length of a string held by lane, as Cranelift reads it: a word of
    /// the run's length table for every lane the run made. A lane from the
    /// root base on (a loop-carried constant) is not in that table and goes
    /// to the VM, in a skip of its own so that the table is not read past its
    /// end. The two paths join in the length.
    fn string_length(
        &mut self,
        function: HostFunction,
        args: &[ValueId],
        result: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let lane = self.get(args[0])?;
        let base = self.iconst(ROOT_STRING_LANE_BASE.cast_signed());
        // What the skip yields when it is jumped over, defined before it.
        let other = if result == TraceType::Float {
            self.f.fconst(0.0)
        } else {
            self.iconst(0)
        };
        let open = self.f.begin_if(Cc::Ult, lane, base);
        let table = self.control_word(mem::offset_of!(TraceControl, lengths));
        let mut load = Inst::new(Op::Load32, Ty::I64);
        let length = self.f.val(Ty::I64);
        load.dst = length;
        load.a = table;
        load.b = lane;
        load.scale = mem::size_of::<u32>() as u8;
        self.f.push(load);
        // Loaded zero-extended, so the signed conversion is exact.
        let length = if result == TraceType::Float {
            self.un(Op::I2F, Ty::F64, length)
        } else {
            length
        };
        let known = self.f.end_if(open, &[(length, other)]);
        let open = self.f.begin_if(Cc::Uge, lane, base);
        let by_vm = self.host_call_in_vm(function, args, result, guard)?;
        Ok(self.f.end_if(open, &[(by_vm, known)]))
    }

    /// A host call the VM answers, as `host_call_in_vm` makes it: the
    /// arguments as tagged words in the control block, the call through its
    /// entry, a guard on the tag of the answer.
    fn host_call_in_vm(
        &mut self,
        function: HostFunction,
        args: &[ValueId],
        result: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let slot_size = mem::size_of::<TraceInputSlot>();
        let args_at = mem::offset_of!(TraceControl, host_args);
        for (position, arg) in args.iter().enumerate() {
            let ty = self.ty(*arg);
            let value = self.get(*arg)?;
            let at = args_at + position * slot_size;
            let tag_at = offset(at + mem::offset_of!(TraceInputSlot, tag));
            let bits_at = offset(at + mem::offset_of!(TraceInputSlot, bits));
            let tag = if ty == TraceType::StringRef {
                TRACE_TAG_BORROWED_STRING
            } else {
                let Some(tag) = entry_tag(ty) else {
                    return self.reject("host call argument without an entry tag");
                };
                tag
            };
            let tag = self.iconst(tag);
            self.store_at(self.control, Off::K(0), tag_at, tag);
            match ty {
                TraceType::Nil => {
                    let zero = self.iconst(0);
                    self.store_at(self.control, Off::K(0), bits_at, zero);
                }
                _ => self.store_at(self.control, Off::K(0), bits_at, value),
            }
        }
        let context = self.control_word(mem::offset_of!(TraceControl, host_context));
        let args_address = self.control_address(args_at);
        let out = self.control_address(mem::offset_of!(TraceControl, host_result));
        let function_word = self.iconst(function.code());
        let count = self.iconst(i64::try_from(args.len()).expect("few arguments"));
        self.f.call_ctl(
            offset(mem::offset_of!(TraceControl, host_entry)) as i32,
            &[context, args_address, out, function_word, count],
            None,
        );
        let result_at = mem::offset_of!(TraceControl, host_result);
        let tag = self.load_at(
            Ty::I64,
            self.control,
            Off::K(0),
            offset(result_at + mem::offset_of!(TraceInputSlot, tag)),
        );
        let Some(wanted) = entry_tag(result) else {
            return self.reject("host call result without an entry tag");
        };
        let wanted = self.iconst(wanted);
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        let bits_at = offset(result_at + mem::offset_of!(TraceInputSlot, bits));
        Ok(match result {
            TraceType::Nil => self.iconst(0),
            TraceType::Float => self.load_at(Ty::F64, self.control, Off::K(0), bits_at),
            TraceType::Boolean => self.load8(self.control, Off::K(0), bits_at),
            _ => self.load_at(Ty::I64, self.control, Off::K(0), bits_at),
        })
    }

    /// `ReleaseStrings`: when the strings made since the last release weigh
    /// enough, the VM drops every string lane the run made but `keep`. The
    /// lanes to keep are stored where the host call reads them whether the call
    /// is made or not, and the call itself is out of line.
    pub(super) fn release_strings(&mut self, keep: &[ValueId]) -> Res<()> {
        let slot_size = mem::size_of::<TraceInputSlot>();
        let args_at = mem::offset_of!(TraceControl, host_args);
        for (position, lane) in keep.iter().enumerate() {
            let lane = self.get(*lane)?;
            let at = args_at + position * slot_size;
            let tag = self.iconst(TRACE_TAG_STRING);
            self.store_at(
                self.control,
                Off::K(0),
                offset(at + mem::offset_of!(TraceInputSlot, tag)),
                tag,
            );
            self.store_at(
                self.control,
                Off::K(0),
                offset(at + mem::offset_of!(TraceInputSlot, bits)),
                lane,
            );
        }
        let table = self.control_word(mem::offset_of!(TraceControl, lengths));
        let pending = self.f.val(Ty::I64);
        let mut load = Inst::new(Op::Load32, Ty::I64);
        load.dst = pending;
        load.a = table;
        load.imm = offset(TRACE_PENDING_STRINGS * mem::size_of::<u32>());
        self.f.push(load);
        let weight = self.iconst(RELEASE_WEIGHT);
        let context = self.control_word(mem::offset_of!(TraceControl, host_context));
        let args_address = self.control_address(args_at);
        let out = self.control_address(mem::offset_of!(TraceControl, host_result));
        let function = self.iconst(HostFunction::ReleaseStrings.code());
        let count = self.iconst(i64::try_from(keep.len()).expect("few lanes"));
        self.f.call_if(
            Cc::Ugt,
            (pending, weight),
            mem::offset_of!(TraceControl, host_entry),
            true,
            &[context, args_address, out, function, count],
        );
        Ok(())
    }

    /// `HashLoad`: the VM's lookup (`trace_hash_get`), then a guard on the
    /// type of what it found. A string key the run probes in line goes to
    /// `probed_load`.
    pub(super) fn hash_load(
        &mut self,
        binding: u32,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
    ) -> Res<Val> {
        let key_ty = self.ty(key);
        if self.cranelift_probes_load(key, expected) {
            if !crate::jit::debug_env::mir_probe() {
                return self.reject("HashLoad without the inline probe");
            }
            return self.probed_load(binding, key, expected, guard);
        }
        let key_tag = match key_ty {
            TraceType::Integer => TRACE_TAG_INTEGER,
            TraceType::Float => TRACE_TAG_FLOAT,
            TraceType::String => TRACE_TAG_STRING,
            TraceType::StringRef => TRACE_TAG_BORROWED_STRING,
            _ => return self.reject("hash lookup of another key"),
        };
        let key_tag = self.iconst(key_tag);
        let key_val = self.get(key)?;
        let key_bits = if key_ty == TraceType::Float {
            self.f.bits(key_val)
        } else {
            key_val
        };
        let binding_value = self.iconst(i64::from(binding));
        let dispatch = self.control_word(mem::offset_of!(TraceControl, tables));
        let out = self.control_address(mem::offset_of!(TraceControl, hash_result));
        self.f.call(
            trace_hash_get as *const () as usize,
            &[dispatch, out, binding_value, key_tag, key_bits],
            None,
        );
        let result_at = mem::offset_of!(TraceControl, hash_result);
        let tag = self.load_at(
            Ty::I64,
            self.control,
            Off::K(0),
            offset(result_at + mem::offset_of!(TraceInputSlot, tag)),
        );
        let Some(wanted) = entry_tag(expected) else {
            return self.reject("hash lookup of a type without an entry tag");
        };
        let wanted = self.iconst(wanted);
        self.guard_cc(Cc::Eq, tag, wanted, guard)?;
        let bits_at = offset(result_at + mem::offset_of!(TraceInputSlot, bits));
        Ok(match expected {
            TraceType::Nil => self.iconst(0),
            TraceType::Float => self.load_at(Ty::F64, self.control, Off::K(0), bits_at),
            TraceType::Boolean => self.load8(self.control, Off::K(0), bits_at),
            _ => self.load_at(Ty::I64, self.control, Off::K(0), bits_at),
        })
    }

    /// A zero-extending 32-bit load at `[base + disp]`.
    pub(super) fn load32(&mut self, base: Val, disp: i64) -> Val {
        let dst = self.f.val(Ty::I64);
        let mut inst = Inst::new(Op::Load32, Ty::I64);
        inst.dst = dst;
        inst.a = base;
        inst.imm = disp;
        self.f.push(inst);
        dst
    }

    /// The lane of the table (or closure, or thread) in the array slot at
    /// `slot`, as `admit_table_lane` of `emit.rs` finds it: the first place of
    /// the run's lane index when it names the table and says its row is filled
    /// (a hash of the table's handle word, a few loads and compares, no
    /// branch), the dispatcher's admission otherwise (a conditional call that
    /// leaves its answer in the control block's `admit_lane`).
    pub(super) fn admit_lane(&mut self, slot: Val, is_table: bool) -> Val {
        let dispatch = self.control_word(mem::offset_of!(TraceControl, tables));
        if !(self.lane_probe && is_table) {
            return self.f.call(
                trace_admit_table as *const () as usize,
                &[dispatch, slot],
                Some(Ty::I64),
            );
        }
        let zero = self.iconst(0);
        let one = self.iconst(1);
        let index = self.control_word(mem::offset_of!(TraceControl, lane_index));
        // The table's handle word: its slot, then its generation.
        let handle = self.load_at(Ty::I64, slot, Off::K(0), 8);
        let shift = self.iconst(32);
        let high = self.f.bin(Op::Shr, handle, shift);
        let mixed = self.f.bin(Op::Xor, handle, high);
        let multiplier = self.iconst(0x9E37_79B9_7F4A_7C15_u64.cast_signed());
        let product = self.f.bin(Op::Mul, mixed, multiplier);
        let hashed = self.f.bin(Op::Shr, product, shift);
        let places = self.iconst(
            i64::try_from(super::super::super::exit::TRACE_LANE_INDEX_PLACES - 1)
                .expect("places fit"),
        );
        let place = self.f.bin(Op::And, hashed, places);
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, place, four);
        let candidate = self.f.bin(Op::Add, index, scaled);
        // Without an index the entry is read from a word the control block
        // owns, and the answer is a miss.
        let scratch = self.control_address(mem::offset_of!(TraceControl, hash_result));
        let entry = self.f.select(Cc::Ne, index, zero, candidate, scratch);
        let identity = self.load_at(Ty::I64, entry, Off::K(0), 0);
        let ready = self.load32(entry, 12);
        let wanted = self.control_word(mem::offset_of!(TraceControl, lane_ready));
        let fast = self.load32(entry, 8);
        let have = self.f.select(Cc::Ne, index, zero, one, zero);
        let same = self.f.select(Cc::Eq, identity, handle, one, zero);
        let current = self.f.select(Cc::Eq, ready, wanted, one, zero);
        let both = self.f.bin(Op::And, same, current);
        let hit = self.f.bin(Op::And, both, have);
        self.f.call_if_storing(
            Cc::Eq,
            (hit, zero),
            trace_admit_table as *const () as usize,
            false,
            &[dispatch, slot],
            Some(u32::try_from(mem::offset_of!(TraceControl, admit_lane)).expect("small")),
        );
        let slow_lane = self.control_word(mem::offset_of!(TraceControl, admit_lane));
        self.f.select(Cc::Ne, hit, zero, fast, slow_lane)
    }
}
