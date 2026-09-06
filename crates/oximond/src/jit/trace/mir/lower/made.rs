//! The tables a `TABLE_NEW` call makes and the `TABLE_INIT` calls that fill
//! them, in line, as Cranelift does (`table_init_in_place` in `emit.rs`).
//!
//! When some `TABLE_INIT` of the member stores under a root string key of a
//! table the call made, or the member fills a short array of it (see
//! [`crate::jit::trace::made_table`], which both backends ask), the helper
//! leaves the address of the table's first value slot in its second result
//! word. Each fill then writes its slot where it is: a tag and a payload, and
//! one more field counted in the control block's `initialized`, which the run
//! charges to the collector. A number or a boolean needs no write barrier. A
//! table (which a function trace stores, the callee having made it) is
//! stored in place while no major cycle runs and the run has a bank to copy
//! its lane value from, as Cranelift's `table_init_in_place` does, and by the
//! helper otherwise. The helper also fills a table the shape has no slots for.
//!
//! The fill is two skips. The first stores in place when `ready` is not
//! zero (the slot address itself for a number or a boolean). The second
//! calls the helper when it is zero, with the same call and the guard on its
//! status as `store_by_helpers` makes (`probe.rs`); the call's arguments are
//! made inside it, so the store in place does not compute them. A helper call that
//! may have moved the slots (any other store into the table, or a fill the
//! helper answered with a key the shape lacks or a nil) marks the table
//! `moved`, and later fills of it go to the helper.

use super::super::super::emit::constant_integer;
#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;
use crate::jit::trace::helpers::helper_address;
use crate::jit::trace::made_table::{
    MadeShape, fills_in_place, init_by_helper_moves, init_field, made_table_shape, prefills_array,
};
use crate::jit_ir::trace::{HelperEffects, HelperId, TABLE_NEW_PREFILL};

/// A table a `TABLE_NEW` call made, as the code after it can fill it.
pub(super) struct MadeTable {
    /// The value of the table (its result).
    table: usize,
    /// The address of the table's first value slot, which the helper left in
    /// the control block: zero when the table has no slots to name.
    slots: Val,
    made: MadeShape,
    /// The slots filled in place so far.
    filled: Vec<usize>,
    /// Whether a store by helper may have moved the slots.
    moved: bool,
}

impl Lowerer<'_, '_> {
    /// What a `TABLE_NEW` call with arguments `args` and result `result`
    /// needs before it is made: whether the code fills the table, and the
    /// array part prefilled. Sets [`Lowerer::prefill_call`] for the argument
    /// count.
    pub(super) fn made_table_call(
        &mut self,
        helper: HelperId,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> bool {
        self.prefill_call = false;
        if helper != HelperId::TABLE_NEW {
            return false;
        }
        let Some(result) = result else {
            return false;
        };
        self.prefill_call = self.prefills_array_of(result, args);
        self.prefill_call || fills_in_place(self.ir, result)
    }

    fn prefills_array_of(&self, made: ValueId, args: &[ValueId]) -> bool {
        prefills_array(self.ir, made, args, constant_integer)
    }

    /// The argument count a call of `helper` passes: the arguments, and for a
    /// `TABLE_NEW` that the code fills the flag for the array part.
    pub(super) fn argument_count(&self, helper: HelperId, args: usize) -> i64 {
        let count = i64::try_from(args).expect("few arguments");
        if helper == HelperId::TABLE_NEW && self.prefill_call {
            count | TABLE_NEW_PREFILL
        } else {
            count
        }
    }

    /// Zeroes the second result word before the call, which only a helper
    /// that has slots to name writes.
    pub(super) fn clear_slots_word(&mut self) {
        let zero = self.iconst(0);
        let at = mem::offset_of!(TraceControl, helper_out) + mem::size_of::<i64>();
        self.store_at(self.control, Off::K(0), offset(at), zero);
    }

    /// Records the table the call just made, whose slots address is in the
    /// second result word.
    pub(super) fn note_made_table(&mut self, args: &[ValueId], result: Option<ValueId>) {
        let Some(result) = result else {
            return;
        };
        let at = mem::offset_of!(TraceControl, helper_out) + mem::size_of::<i64>();
        let slots = self.load_at(Ty::I64, self.control, Off::K(0), offset(at));
        let array = match self.s.konst[args[0].index()] {
            Some(TraceScalar::Integer(length)) => length,
            _ => 0,
        };
        let shape = made_table_shape(self.ir, args, &self.s.konst);
        self.s.made.push(MadeTable {
            table: result.index(),
            slots,
            made: MadeShape {
                shape,
                prefilled: self.prefill_call,
                array,
            },
            filled: Vec::new(),
            moved: false,
        });
    }

    /// Any other store by helper may grow the table a made table's slots
    /// belong to, which later in-place fills must not trust.
    pub(super) fn note_made_table_written(&mut self, helper: HelperId, args: &[ValueId]) {
        if helper == HelperId::TABLE_INIT
            || !helper
                .info()
                .is_some_and(|info| info.effects.contains(HelperEffects::WRITES_HEAP))
        {
            return;
        }
        if let Some(table) = args.first()
            && let Some(made) = self.s.made.iter_mut().find(|m| m.table == table.index())
        {
            made.moved = true;
        }
    }

    /// Notes that the helper stores under the key of a `TABLE_INIT` call that
    /// was not lowered in line.
    pub(super) fn note_table_init_by_helper(&mut self, args: &[ValueId]) {
        let key = self.s.konst[args[1].index()];
        let ir = self.ir;
        if let Some(made) = self.s.made.iter_mut().find(|m| m.table == args[0].index())
            && init_by_helper_moves(ir, &made.made, key, args[2])
        {
            made.moved = true;
        }
    }

    /// A `TABLE_INIT` call done in line when it fills a field of a table made
    /// in this member with a number, a boolean or a table (the conditions of
    /// `table_init_in_place`). Returns whether the call was lowered; anything
    /// else (a second store under the key, a string, a key the shape lacks, a
    /// table with no slots) is the helper's.
    pub(super) fn table_init_in_place(&mut self, args: &[ValueId], guard: GuardId) -> Res<bool> {
        let ir = self.ir;
        let key = self.s.konst[args[1].index()];
        let Some(at) = self.s.made.iter().position(|m| m.table == args[0].index()) else {
            return Ok(false);
        };
        let made = &self.s.made[at];
        if made.moved {
            return Ok(false);
        }
        let Some(field) = init_field(ir, &made.made, key) else {
            return Ok(false);
        };
        let ty = ir
            .value_type(args[2])
            .ok_or(Reject::new("table fill of a value without a type"))?;
        if !matches!(
            ty,
            TraceType::Float | TraceType::Integer | TraceType::Boolean | TraceType::Table
        ) || made.filled.contains(&field)
        {
            return Ok(false);
        }
        let slots = made.slots;
        self.s.made[at].filled.push(field);
        let helper = HelperId::TABLE_INIT;
        let Some(info) = helper.info() else {
            return self.reject("helper without a description");
        };
        let Some(address) = helper_address(helper) else {
            return self.reject("helper nobody installed");
        };
        // A value either skip reads, and one the lowering keeps for later
        // uses, is made before the first opens.
        for arg in args {
            self.get(*arg)?;
        }
        let value = self.get(args[2])?;
        if info.effects.contains(HelperEffects::MAY_EXIT) {
            self.guard_target(guard)?;
        }
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let mut ready = slots;
        // A table (a function trace's, which stores what its callee made)
        // goes in place only while no major cycle needs the write barrier
        // and the run has a bank to copy the table's lane value from.
        let bank = if ty == TraceType::Table {
            let idle = self.control_word(mem::offset_of!(TraceControl, barrier_idle));
            let bank = self.control_word(mem::offset_of!(TraceControl, bank));
            // The slot address while both words are not zero, else zero.
            let idle_ready = self.f.select(Cc::Ne, idle, zero, slots, zero);
            ready = self.f.select(Cc::Ne, bank, zero, idle_ready, zero);
            bank
        } else {
            NONE
        };
        let field_at = i64::try_from(field * mem::size_of::<crate::runtime::Value>())
            .map_err(|_| Reject::new("table field out of reach"))?;
        let open = self.f.begin_if(Cc::Ne, ready, zero);
        if bank == NONE {
            let tag = match ty {
                TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
                TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
                _ => NATIVE_VALUE_TAG_BOOLEAN,
            };
            let tag = self.iconst(i64::from(tag));
            self.store8(slots, Off::K(0), field_at, tag);
            self.payload_access(ty, slots, Off::K(field_at), Some(value))?;
        } else {
            // The table is the value of its lane: both words of it move.
            let four = self.iconst(4);
            let scaled = self.f.bin(Op::Shl, value, four);
            let from = self.f.bin(Op::Add, bank, scaled);
            for word in [0, 8] {
                let copied = self.load_at(Ty::I64, from, Off::K(0), word);
                self.store_at(slots, Off::K(0), field_at + word, copied);
            }
        }
        self.add_at(
            self.control,
            Off::K(0),
            offset(mem::offset_of!(TraceControl, initialized)),
            one,
        );
        self.f.end_if(open, &[]);
        // The helper makes the store the code did not.
        let open = self.f.begin_if(Cc::Eq, ready, zero);
        let set_at = mem::offset_of!(TraceControl, set_status);
        self.store_at(self.control, Off::K(0), offset(set_at), zero);
        let mut list = mem::take(&mut self.s.list);
        list.clear();
        let built = self.helper_arguments(helper, info, args, &mut list);
        if let Err(error) = built {
            self.s.list = list;
            return Err(error);
        }
        self.f.call_if_storing(
            Cc::Eq,
            (ready, zero),
            address,
            false,
            &list,
            Some(u32::try_from(set_at).expect("small")),
        );
        self.s.list = list;
        if info.effects.contains(HelperEffects::MAY_EXIT) {
            let status = self.control_word(set_at);
            self.guard_cc(Cc::Eq, status, zero, guard)?;
        }
        self.f.end_if(open, &[]);
        Ok(true)
    }
}
