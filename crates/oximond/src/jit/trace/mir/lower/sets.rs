//! `ArraySet`: a raw set of a key in the array part of a bound table, and
//! the nil read past the end of an array.
//!
//! The VM's store entry does any set. A table whose register has a header
//! binding ([`BindingKind::Header`]) lets native code do part of it in place,
//! and this lowering does what `set_in_place` and `set_in_holes` of `emit.rs`
//! do:
//!
//! - any table takes a scalar over a slot inside the array that holds a
//!   value of the scalar's type: the payload alone is written;
//! - a dense list (flags [`NativeTableHeader::DENSE`]) takes the key one past
//!   its last slot as an append (the array grows through
//!   [`native_table_reserve`] when it is full, the length, the border and the
//!   count of appended slots follow), a nil there as no store at all, the last
//!   slot with a nil as a pop, and any slot inside the array a scalar or a
//!   string over a slot that holds no reference;
//! - a solid table that is not a dense list (flags [`NativeTableHeader::HOLEY`])
//!   takes a scalar over a slot inside the array that holds no reference,
//!   except the slot after the border, and a boolean a few slots past the end,
//!   with the nil holes between (the sieve).
//!
//! Everything that is not done in place falls to the general path
//! (`Lowerer::rest_when`): under a strict header the guard leaves, so the VM
//! is never called, and otherwise the VM's store entry does the set.
//!
//! The code is a tree of forward skips (`Op::If`) on where the key lies and
//! what its slot holds, ordered so that the overwrite of a slot of the
//! value's type runs the fewest instructions: it reads the table's address,
//! its length and array, and the slot's tag. Each case that can fail ends in
//! its own general path, so the cases that cannot fail join nothing and the
//! skips they jump over cost a compare and a branch. Nothing a skip defines
//! is read after it, and a guard's exit is made before the first skip opens
//! ([`Lowerer::guard_target`]).

#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;
use crate::runtime::{
    NATIVE_BOOLEAN_GROWTH_REACH, NATIVE_VALUE_SIZE, NativeTableHeader, native_table_reserve,
};

/// Writes the nil tag of the `count` slots after the first `length` of the
/// array at `array`: the holes a boolean store a few slots past the end of
/// an array leaves (the loop of `set_in_holes` in `emit.rs`). It takes what
/// the code holds in registers already, so the call costs the path that does
/// not make it nothing.
#[allow(
    unsafe_code,
    reason = "native code calls it with the array's own slots"
)]
extern "C" fn fill_nil_slots(array: usize, length: usize, count: usize) {
    for hole in length..length + count {
        // SAFETY: the caller checked that the array has room for `count`
        // slots after its first `length`, each `NATIVE_VALUE_SIZE` bytes.
        unsafe {
            ((array + hole * NATIVE_VALUE_SIZE) as *mut u8).write(NATIVE_VALUE_TAG_NIL);
        }
    }
}

/// What a store in place needs from the table: the words of its header
/// binding, and the layout of its fields.
#[derive(Clone, Copy)]
struct Table {
    /// The index of the header binding.
    header: usize,
    /// The table's address.
    address: Val,
    /// The header's flags word when the member holds it already, `NONE` when
    /// the case that needs it loads it (`Lowerer::flags_of`).
    flags: Val,
    layout: NativeTableLayout,
}

/// The value a set stores, and the key it stores it under as an integer.
#[derive(Clone, Copy)]
struct Entry {
    index: Val,
    ty: TraceType,
    value: Val,
}

/// What the general path of a set takes: the array binding, the key and the
/// value as the trace holds them, the guard that leaves when nothing can be
/// stored, and whether the header is strict.
#[derive(Clone, Copy)]
struct Rest {
    binding: u32,
    key: (TraceType, Val),
    value: (TraceType, Val),
    guard: GuardId,
    strict: bool,
}

impl Lowerer<'_, '_> {
    /// The nil read of an [`TraceOp::ArrayLoad`] with `past_end`
    /// (`array_nil_read` in `emit.rs`): nil when `key` names a slot inside the
    /// array that holds nil, and nil for any other key when the table's header
    /// says it is solid, since its hash part is empty and holds no such key.
    /// The guard leaves otherwise. A table with no header binding has the
    /// first case only, which is a load of a nil slot.
    ///
    /// No branch: the slot's address is the slot or a word the control block
    /// owns, by whether the key is inside, and one condition says whether the
    /// read holds.
    pub(super) fn array_nil_read(
        &mut self,
        binding: u32,
        key: ValueId,
        guard: GuardId,
    ) -> Res<Val> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let Some((header, _)) = self.header_binding(binding) else {
            return self.array_element_ref(binding, key, TraceType::Nil, guard);
        };
        let words = self.words_of(binding)?;
        let flags = self
            .words_of(u32::try_from(header).expect("few bindings"))?
            .len;
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let key_val = self.get(key)?;
        let (index, inside) = if self.ty(key) == TraceType::Float {
            if words.lenf == NONE {
                return self.reject("float key without the length as a float");
            }
            let at_least = self.f.fconst(1.0);
            let high = self.f.select_any(Cc::Ge, key_val, at_least, one, zero);
            let low = self.f.select_any(Cc::Le, key_val, words.lenf, one, zero);
            let magic = self.f.fconst(FLOAT_INTEGER_MAGIC);
            let shifted = self.f.bin(Op::FAdd, key_val, magic);
            let restored = self.f.bin(Op::FSub, shifted, magic);
            let exact = self.f.select_any(Cc::Eq, restored, key_val, one, zero);
            let range = self.f.bin(Op::And, high, low);
            let inside = self.f.bin(Op::And, range, exact);
            let whole = self.un(Op::F2I, Ty::I64, key_val);
            let minus = self.iconst(-1);
            (self.f.bin(Op::Add, whole, minus), inside)
        } else {
            let minus = self.iconst(-1);
            let index = self.f.bin(Op::Add, key_val, minus);
            let inside = self.f.select(Cc::Ult, index, words.len, one, zero);
            (index, inside)
        };
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, index, four);
        let slot = self.f.bin(Op::Add, words.base, scaled);
        let scratch = self.control_address(mem::offset_of!(TraceControl, hash_result));
        let address = self.f.select(Cc::Ne, inside, zero, slot, scratch);
        let tag = self.load8(address, Off::K(0), 0);
        let hole = self.f.select(Cc::Eq, tag, zero, one, zero);
        let held = self.f.bin(Op::And, inside, hole);
        let solid = self.f.select(Cc::Ne, flags, zero, one, zero);
        let outside = self.f.select(Cc::Eq, inside, zero, solid, zero);
        let good = self.f.bin(Op::Or, held, outside);
        self.guard_cc(Cc::Ne, good, zero, guard)?;
        Ok(zero)
    }

    /// `ArraySet`: a table with no header goes to the general path. One with
    /// a header is done in place where the module's comment says, and each
    /// case that is not falls to [`Self::rest_when`]. The array words of the
    /// bindings, which the VM's store or an append rewrote, are read again
    /// after.
    pub(super) fn array_set(
        &mut self,
        binding: u32,
        key: ValueId,
        value: ValueId,
        guard: GuardId,
    ) -> Res<()> {
        let (key_ty, value_ty) = (self.ty(key), self.ty(value));
        let header = self.header_binding(binding);
        if header.is_some() {
            if !matches!(key_ty, TraceType::Float | TraceType::Integer) {
                return self.reject("ArraySet under a header with a key that is not a number");
            }
            if !matches!(
                value_ty,
                TraceType::Float
                    | TraceType::Integer
                    | TraceType::Boolean
                    | TraceType::Nil
                    | TraceType::String
                    | TraceType::Table
            ) {
                return self.reject("ArraySet under a header of a value the store does not take");
            }
        }
        // What the skips below read is made before the first opens: a value
        // that is a pending comparison is made where it is first read, and
        // the guard's exit keeps the snapshot every later guard of it sees.
        if let Some(off) = self.proven_off(binding, key, guard)? {
            return self.set_proven(binding, off, (key, value), guard);
        }
        let (key_val, value_val) = (self.get(key)?, self.get(value)?);
        self.guard_target(guard)?;
        let Some((header, strict)) = header else {
            return self.general_set(binding, (key, key_val), (value, value_val), guard);
        };
        let Some(layout) = NativeTableLayout::get() else {
            return self.reject("table layout of this build is not known");
        };
        // The address is all the store over a slot of the value's type reads
        // of the header, so the flags are loaded by the case that needs them
        // unless the member holds the words already.
        let known = self.s.words[header];
        let (address, flags) = if known.base == NONE {
            let at = self.header_word_at(header, mem::offset_of!(TraceInputSlot, bits));
            (self.load_at(Ty::I64, self.inputs, Off::K(0), at), NONE)
        } else {
            (known.base, known.len)
        };
        let table = Table {
            header,
            address,
            flags,
            layout,
        };
        let rest = Rest {
            binding,
            key: (key_ty, key_val),
            value: (value_ty, value_val),
            guard,
            strict,
        };
        // A float key names a slot when it is a whole number of at least 1.
        // Under a strict header the guard leaves for a key whose truncation
        // is below it, which a key of at least 0 that is not whole has; a
        // negative key names no slot in any case below. Otherwise the key
        // less its truncation, a float that is zero for a whole key alone,
        // goes into the slot's index as bits (see below).
        let (index, fraction) = if key_ty == TraceType::Float {
            let converted = self.un(Op::F2I, Ty::I64, key_val);
            let back = self.un(Op::I2F, Ty::F64, converted);
            if strict {
                self.guard_cc(Cc::Ge, back, key_val, guard)?;
                (converted, NONE)
            } else {
                let fraction = self.f.bin(Op::FSub, key_val, back);
                (converted, self.f.bits(fraction))
            }
        } else {
            (key_val, NONE)
        };
        let entry = Entry {
            index,
            ty: value_ty,
            value: value_val,
        };
        let length = self.load_at(Ty::I64, address, Off::K(0), i64::from(layout.array_length));
        // A sum with a constant keeps the key in its register.
        let minus = self.iconst(-1);
        let mut slot_index = self.f.bin(Op::Add, index, minus);
        if fraction != NONE {
            // The bits of a fraction that is not zero put the index past
            // any array: those of a key of 1 or more have an exponent (at
            // least 2^61 as an integer), those of a negative key the sign,
            // and a key between 0 and 1 truncates to 0, whose index is all
            // ones already. So a key that is not whole is neither inside
            // the array nor near its end, and takes the general path with
            // every other key the cases below refuse. They compare the
            // index, not the key, wherever the key may not be whole.
            slot_index = self.f.bin(Op::Or, slot_index, fraction);
        }
        let open = self.f.begin_if(Cc::Ult, slot_index, length);
        self.set_inside(table, entry, rest, (length, slot_index))?;
        self.f.end_if(open, &[]);
        let open = self.f.begin_if(Cc::Uge, slot_index, length);
        self.set_outside(table, entry, rest, (length, slot_index))?;
        self.f.end_if(open, &[]);
        self.forget_words();
        Ok(())
    }

    /// `ArraySet` in the fast version of the root's loop, with a key the
    /// range plan proved: the slot at `off` is inside the array, whose words
    /// the loop holds. A slot of the value's type takes the payload. Any other
    /// slot that holds no reference takes the tag and the payload when the
    /// table is solid and the slot is not the one after the border, as
    /// [`Self::set_inside`] stores it. Neither moves the array, its length or
    /// its border, so the proof stands.
    ///
    /// Everything else (a slot that holds a reference, a table with a hash
    /// part, the slot after the border, whose fill moves the border) is the
    /// VM's store, as in the guarded loop ([`Self::rest_when`]): a loop that
    /// fills the nil slots after a border would otherwise leave the trace at
    /// every iteration. The store is inside the array, so the VM is expected
    /// to move nothing the loop holds; the words of every binding the loop
    /// holds are compared with the ones the VM wrote back, and a difference
    /// leaves through the guard, where the interpreter stores the same value
    /// under the same key again. Under a strict header the guard leaves
    /// before anything is stored, and the loop that is entered again checks
    /// its plan again.
    fn set_proven(
        &mut self,
        binding: u32,
        off: Off,
        (key, value): (ValueId, ValueId),
        guard: GuardId,
    ) -> Res<()> {
        let ty = self.ty(value);
        let Some(wanted) = tag_of(ty) else {
            return self.reject("proven ArraySet of a value that is no number or boolean");
        };
        let Some((header, strict)) = self.header_binding(binding) else {
            return self.reject("proven ArraySet of a table without a header");
        };
        let Some(layout) = NativeTableLayout::get() else {
            return self.reject("table layout of this build is not known");
        };
        let Off::VK(scaled, disp) = off else {
            return self.reject("proven ArraySet without a scaled index");
        };
        let stored = self.get(value)?;
        let rest = Rest {
            binding,
            // Made from the slot's offset where the VM stores.
            key: (self.ty(key), NONE),
            value: (ty, stored),
            guard,
            strict,
        };
        let words = self.words_of(binding)?;
        let address = self
            .words_of(u32::try_from(header).expect("few bindings"))?
            .base;
        if words.base == NONE || address == NONE {
            return self.reject("proven ArraySet without the words of its table");
        }
        self.guard_target(guard)?;
        let zero = self.iconst(0);
        // The tag is read again where the slot holds another type, so the
        // compare the loop always runs reads it from memory.
        let tag = self.load8(words.base, off, 0);
        let wanted = self.iconst(wanted);
        let open = self.f.begin_if(Cc::Ne, tag, wanted);
        let tag = self.load8(words.base, off, 0);
        let string = self.iconst(i64::from(NATIVE_VALUE_TAG_STRING));
        // The flags as the VM last wrote them, not as the loop found them.
        let at = self.header_word_at(header, mem::offset_of!(TraceInputSlot, tag));
        let flags = self.load_at(Ty::I64, self.inputs, Off::K(0), at);
        // The slot after the border is the one whose zero-based index is the
        // border.
        let border = self.load_at(Ty::I64, address, Off::K(0), i64::from(layout.border));
        let four = self.iconst(4);
        let border = self.f.bin(Op::Shl, border, four);
        let slot = if disp == 0 {
            scaled
        } else {
            let disp = self.iconst(disp);
            self.f.bin(Op::Add, scaled, disp)
        };
        // Three compares the fill passes without a branch taken. After a
        // store of the VM the ones that follow see the flags and the border
        // of before, and at worst hand it the same store again.
        self.rest_checked(rest, Cc::Uge, tag, string, Some(slot))?;
        self.rest_checked(rest, Cc::Eq, flags, zero, Some(slot))?;
        self.rest_checked(rest, Cc::Eq, slot, border, Some(slot))?;
        // After the VM's store the slot holds the tag and the value already,
        // and takes the same tag and payload again.
        self.store8(words.base, off, 0, wanted);
        self.f.end_if(open, &[]);
        self.payload_access(ty, words.base, off, Some(stored))?;
        // Other slots of the array changed their type.
        self.forget_words();
        Ok(())
    }

    /// After a store of the VM in the fast version of the root's loop: the
    /// guard leaves unless every array, length and header word the loop
    /// holds is the one the VM wrote back to the run's binding slots (it
    /// rewrites them after each of its stores). The loop read them once,
    /// where it started, and goes on with them.
    fn held_words_stand(&mut self, guard: GuardId) -> Res<()> {
        let root = self.members[0].ir;
        let first = root.entry.inputs.len();
        let slot_size = mem::size_of::<TraceInputSlot>();
        for index in 0..root.bindings.len() {
            let held = self.s.words[index];
            if held.base == NONE {
                continue;
            }
            let bits_at =
                offset((first + index) * slot_size + mem::offset_of!(TraceInputSlot, bits));
            let tag_at = offset((first + index) * slot_size + mem::offset_of!(TraceInputSlot, tag));
            match &root.bindings[index].kind {
                BindingKind::Array { .. } | BindingKind::Header { .. } => {
                    let base = self.load_at(Ty::I64, self.inputs, Off::K(0), bits_at);
                    self.guard_cc(Cc::Eq, base, held.base, guard)?;
                    if held.len != NONE {
                        let aux = self.load_at(Ty::I64, self.inputs, Off::K(0), tag_at);
                        self.guard_cc(Cc::Eq, aux, held.len, guard)?;
                    }
                }
                BindingKind::Length { .. } => {
                    let ty = self.f.val_ty[held.base as usize];
                    let now = self.load_at(ty, self.inputs, Off::K(0), bits_at);
                    self.guard_cc(Cc::Eq, now, held.base, guard)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The offset in the run's inputs of the word at `field` of the slot of
    /// binding `header`.
    fn header_word_at(&self, header: usize, field: usize) -> i64 {
        let first = self.members[0].ir.entry.inputs.len();
        offset((first + header) * mem::size_of::<TraceInputSlot>() + field)
    }

    /// The flags of the table's header: the word the member holds, or a load
    /// of it where the case that asks stands.
    fn flags_of(&mut self, table: Table) -> Val {
        if table.flags != NONE {
            return table.flags;
        }
        let at = self.header_word_at(table.header, mem::offset_of!(TraceInputSlot, tag));
        self.load_at(Ty::I64, self.inputs, Off::K(0), at)
    }

    /// The general path for the set when `a cc b` holds, which says that
    /// nothing was stored in place. Under a strict header, which promises
    /// that the set moves no storage for the VM to rewrite, the guard leaves
    /// instead of calling the VM. Otherwise the VM's store entry does the
    /// set, out of line, and the guard leaves when it refuses.
    fn rest_when(&mut self, rest: Rest, cc: Cc, a: Val, b: Val) -> Res<()> {
        self.rest_checked(rest, cc, a, b, None)
    }

    /// [`Self::rest_when`]; with `proven`, in the fast version of the root's
    /// loop, the key is made here from the offset of its slot (the loop may
    /// carry no number of it), and the guard also leaves after the VM's
    /// store unless the words the loop holds are still the table's
    /// ([`Self::held_words_stand`]).
    fn rest_checked(&mut self, rest: Rest, cc: Cc, a: Val, b: Val, proven: Option<Val>) -> Res<()> {
        if rest.strict {
            let float = self.f.val_ty[a as usize] == Ty::F64;
            return self.guard_cc(cc.negate(float), a, b, rest.guard);
        }
        let open = self.f.begin_if(cc, a, b);
        let key = match proven {
            Some(slot) => {
                let four = self.iconst(4);
                let index = self.f.bin(Op::Shr, slot, four);
                let one = self.iconst(1);
                let key = self.f.bin(Op::Add, index, one);
                if rest.key.0 == TraceType::Float {
                    self.un(Op::I2F, Ty::F64, key)
                } else {
                    key
                }
            }
            None => rest.key.1,
        };
        let (key_tag, key_bits) = self.scalar_words(rest.key.0, key)?;
        let (value_tag, value_bits) = self.scalar_words(rest.value.0, rest.value.1)?;
        let context = self.control_word(mem::offset_of!(TraceControl, host_context));
        let words = self.binding_slots();
        let binding = self.iconst(i64::from(rest.binding));
        let zero = self.iconst(0);
        let set_at = mem::offset_of!(TraceControl, set_status);
        let set_entry = mem::offset_of!(TraceControl, set_entry);
        // A conditional call keeps the values of the loop in their
        // registers, where a plain call has each stored at its definition.
        self.store_at(self.control, Off::K(0), offset(set_at), zero);
        self.f.call_if_storing(
            cc,
            (a, b),
            set_entry,
            true,
            &[
                context, words, binding, key_tag, key_bits, value_tag, value_bits,
            ],
            Some(u32::try_from(set_at).expect("small")),
        );
        let status = self.control_word(set_at);
        self.guard_cc(Cc::Eq, status, zero, rest.guard)?;
        if proven.is_some() {
            self.held_words_stand(rest.guard)?;
        }
        self.f.end_if(open, &[]);
        Ok(())
    }

    /// Whether the general path overwrites a slot of the value's type in
    /// place before it calls the VM: a number or boolean under a number key.
    fn overwrites_in_place(key_ty: TraceType, value_ty: TraceType) -> bool {
        matches!(
            value_ty,
            TraceType::Float | TraceType::Integer | TraceType::Boolean
        ) && matches!(key_ty, TraceType::Float | TraceType::Integer)
    }

    /// The set of a table with no header: the VM's store entry with the key
    /// and value as tagged words, after an in-place overwrite of a slot
    /// inside the array that holds a value of the value's type when the key
    /// and the value are numbers and booleans. Everything is computed without
    /// a branch: an address that may be outside the array is replaced by a
    /// word the control block owns before anything reads or writes it, and
    /// the VM's store (a conditional call) takes what is left.
    fn general_set(
        &mut self,
        binding: u32,
        (key, key_val): (ValueId, Val),
        (value, value_val): (ValueId, Val),
        guard: GuardId,
    ) -> Res<()> {
        let (key_ty, value_ty) = (self.ty(key), self.ty(value));
        let (key_tag, key_bits) = self.scalar_words(key_ty, key_val)?;
        let (value_tag, value_bits) = self.scalar_words(value_ty, value_val)?;
        let context = self.control_word(mem::offset_of!(TraceControl, host_context));
        let words = self.binding_slots();
        let binding_value = self.iconst(i64::from(binding));
        let zero = self.iconst(0);
        let args = [
            context,
            words,
            binding_value,
            key_tag,
            key_bits,
            value_tag,
            value_bits,
        ];
        let set_at = mem::offset_of!(TraceControl, set_status);
        let set_entry = mem::offset_of!(TraceControl, set_entry);
        if !Self::overwrites_in_place(key_ty, value_ty) {
            let status = self
                .f
                .call_ctl(offset(set_entry) as i32, &args, Some(Ty::I64));
            self.guard_cc(Cc::Eq, status, zero, guard)?;
            self.forget_words();
            return Ok(());
        }
        let words_of = self.words_of(binding)?;
        let one = self.iconst(1);
        let (index, whole) = if key_ty == TraceType::Integer {
            (key_val, one)
        } else {
            let truncated = self.un(Op::F2I, Ty::I64, key_val);
            let back = self.un(Op::I2F, Ty::F64, truncated);
            let whole = self.f.select_any(Cc::Eq, back, key_val, one, zero);
            (truncated, whole)
        };
        let from_zero = self.f.bin(Op::Sub, index, one);
        let inside = self.f.select(Cc::Ult, from_zero, words_of.len, one, zero);
        let inside = self.f.bin(Op::And, inside, whole);
        let four = self.iconst(4);
        let offset_bytes = self.f.bin(Op::Shl, from_zero, four);
        let slot = self.f.bin(Op::Add, words_of.base, offset_bytes);
        let scratch = self.control_address(mem::offset_of!(TraceControl, hash_result));
        let address = self.f.select(Cc::Ne, inside, zero, slot, scratch);
        let tag = self.load8(address, Off::K(0), 0);
        let wanted = match value_ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let wanted = self.iconst(i64::from(wanted));
        let tagged = self.f.select(Cc::Eq, tag, wanted, one, zero);
        let stored = self.f.bin(Op::And, inside, tagged);
        let target = self.f.select(Cc::Ne, stored, zero, slot, scratch);
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        match value_ty {
            TraceType::Boolean => self.store8(target, Off::K(0), payload, value_val),
            _ => self.store_at(target, Off::K(0), payload, value_val),
        }
        // The status of the store the VM does, if it does one.
        self.store_at(self.control, Off::K(0), offset(set_at), zero);
        self.f.call_if_storing(
            Cc::Eq,
            (stored, zero),
            set_entry,
            true,
            &args,
            Some(u32::try_from(set_at).expect("small")),
        );
        let status = self.control_word(set_at);
        self.guard_cc(Cc::Eq, status, zero, guard)?;
        self.forget_words();
        Ok(())
    }

    /// A key inside the array part.
    ///
    /// A scalar over a slot of its own type writes the payload, whatever the
    /// table is: the slot keeps its tag, so the border, the count of keys and
    /// every GC edge stay as they are, and no `__newindex` could see it. Over
    /// any other slot that holds no reference a solid table takes the tag and
    /// the payload, except in the slot after the border, whose fill moves the
    /// border over the slots that follow (the VM's job). A dense list has no
    /// such slot inside: its border is its length.
    ///
    /// A string or table goes into a dense list through the VM's slot store,
    /// which releases what the slot held, and a nil takes the last slot of
    /// one off.
    fn set_inside(
        &mut self,
        table: Table,
        entry: Entry,
        rest: Rest,
        (length, slot_index): (Val, Val),
    ) -> Res<()> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let layout = table.layout;
        let pointer = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.array_pointer),
        );
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, slot_index, four);
        let at = Off::V(scaled);
        if let Some(wanted) = tag_of(entry.ty) {
            let tag = self.load8(pointer, at, 0);
            let wanted = self.iconst(wanted);
            let open = self.f.begin_if(Cc::Eq, tag, wanted);
            self.payload_access(entry.ty, pointer, at, Some(entry.value))?;
            self.f.end_if(open, &[]);
            let open = self.f.begin_if(Cc::Ne, tag, wanted);
            let string = self.iconst(i64::from(NATIVE_VALUE_TAG_STRING));
            let open_plain = self.f.begin_if(Cc::Ult, tag, string);
            let flags = self.flags_of(table);
            let open_solid = self.f.begin_if(Cc::Ne, flags, zero);
            let border = self.load_at(Ty::I64, table.address, Off::K(0), i64::from(layout.border));
            let successor = self.f.bin(Op::Add, border, one);
            let open_fill = self.f.begin_if(Cc::Ne, entry.index, successor);
            self.store8(pointer, at, 0, wanted);
            self.payload_access(entry.ty, pointer, at, Some(entry.value))?;
            let filled = self.f.end_if(open_fill, &[(one, zero)]);
            let filled = self.f.end_if(open_solid, &[(filled, zero)]);
            let filled = self.f.end_if(open_plain, &[(filled, zero)]);
            self.rest_when(rest, Cc::Eq, filled, zero)?;
            self.f.end_if(open, &[]);
            return Ok(());
        }
        let flags = self.flags_of(table);
        let dense = self.iconst(NativeTableHeader::DENSE);
        let slot = self.f.bin(Op::Add, pointer, scaled);
        if entry.ty != TraceType::Nil {
            let open = self.f.begin_if(Cc::Eq, flags, dense);
            let status = self.store_reference(table.header, slot, entry, true)?;
            let status = self.f.end_if(open, &[(status, one)]);
            return self.rest_when(rest, Cc::Ne, status, zero);
        }
        // A nil: only the last slot of a dense list, which leaves the array.
        let open = self.f.begin_if(Cc::Eq, flags, dense);
        let open_last = self.f.begin_if(Cc::Eq, entry.index, length);
        let tag = self.load8(slot, Off::K(0), 0);
        let string = self.iconst(i64::from(NATIVE_VALUE_TAG_STRING));
        let open_release = self.f.begin_if(Cc::Uge, tag, string);
        let status = self.store_reference(table.header, slot, entry, true)?;
        let status = self.f.end_if(open_release, &[(status, zero)]);
        let open_shrink = self.f.begin_if(Cc::Eq, status, zero);
        // The slot leaves the array, the border follows, and the key is one
        // of the deleted keys the table has not recorded.
        self.store_at(
            table.address,
            Off::K(0),
            i64::from(layout.array_length),
            slot_index,
        );
        self.store_at(
            table.address,
            Off::K(0),
            i64::from(layout.border),
            slot_index,
        );
        let pending = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.pending_deletions),
        );
        let raised = self.f.select(Cc::Ult, pending, length, length, pending);
        self.store_at(
            table.address,
            Off::K(0),
            i64::from(layout.pending_deletions),
            raised,
        );
        self.store_array_words(table.header, slot_index, true, None);
        let popped = self.f.end_if(open_shrink, &[(one, zero)]);
        let popped = self.f.end_if(open_last, &[(popped, zero)]);
        let popped = self.f.end_if(open, &[(popped, zero)]);
        self.rest_when(rest, Cc::Eq, popped, zero)
    }

    /// A key that is not inside the array part: a solid table that is not a
    /// dense list takes a boolean a few slots past the end, with the nil
    /// holes between (the sieve), and a dense list takes the key after its
    /// last slot as an append.
    fn set_outside(
        &mut self,
        table: Table,
        entry: Entry,
        rest: Rest,
        (length, slot_index): (Val, Val),
    ) -> Res<()> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let flags = self.flags_of(table);
        let mut done = zero;
        if entry.ty == TraceType::Boolean {
            let holey = self.iconst(NativeTableHeader::HOLEY);
            let open = self.f.begin_if(Cc::Eq, flags, holey);
            let grown = self.grow_with_holes(table, entry, (length, slot_index))?;
            done = self.f.end_if(open, &[(grown, zero)]);
        }
        // The two cases are exclusive, so what the first answered is the
        // answer of the skip that does not run.
        let dense = self.iconst(NativeTableHeader::DENSE);
        let open = self.f.begin_if(Cc::Eq, flags, dense);
        let open_last = self.f.begin_if(Cc::Eq, slot_index, length);
        let last = self.f.bin(Op::Add, length, one);
        let appended = self.append(table, entry, (length, last))?;
        let appended = self.f.end_if(open_last, &[(appended, zero)]);
        let done = self.f.end_if(open, &[(appended, done)]);
        self.rest_when(rest, Cc::Eq, done, zero)
    }

    /// The key one past the last slot of a dense list: nothing for a nil,
    /// otherwise the value in a new slot, with room made first. Returns a
    /// word that is 1 when the set is complete and 0 when nothing was stored.
    fn append(&mut self, table: Table, entry: Entry, (length, last): (Val, Val)) -> Res<Val> {
        let one = self.iconst(1);
        if entry.ty == TraceType::Nil {
            // The key is absent from the array and, as for an append, from the
            // hash part, so the set is complete and a strict header stays.
            return Ok(one);
        }
        let layout = table.layout;
        let capacity = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.array_capacity),
        );
        self.f.call_if(
            Cc::Uge,
            (length, capacity),
            native_table_reserve as *const () as usize,
            false,
            &[table.address],
        );
        let pointer = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.array_pointer),
        );
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, length, four);
        let slot = self.f.bin(Op::Add, pointer, scaled);
        let zero = self.iconst(0);
        let grown = |this: &mut Self| {
            this.store_at(
                table.address,
                Off::K(0),
                i64::from(layout.array_length),
                last,
            );
            this.store_at(table.address, Off::K(0), i64::from(layout.border), last);
            let one = this.iconst(1);
            let at = offset(mem::offset_of!(TraceControl, appended));
            this.add_at(this.control, Off::K(0), at, one);
            this.store_array_words(table.header, last, true, Some(pointer));
        };
        if tag_of(entry.ty).is_some() {
            self.store_slot(slot, entry.ty, entry.value)?;
            grown(self);
            return Ok(one);
        }
        let status = self.store_reference(table.header, slot, entry, false)?;
        let open = self.f.begin_if(Cc::Eq, status, zero);
        grown(self);
        Ok(self.f.end_if(open, &[(one, zero)]))
    }

    /// A boolean at most `NATIVE_BOOLEAN_GROWTH_REACH` slots past the end of a
    /// non-empty array part, which the runtime stores by growing the array
    /// part with nil holes for the slots between (`append_dense_scalar`): done
    /// when the new slots fit the capacity and no deletion is pending past the
    /// end. The border stays, since the first new slot is a hole. The key after
    /// the last slot is an append, which this does not do. Returns a word that
    /// is 1 when the boolean was stored and 0 when nothing was.
    fn grow_with_holes(
        &mut self,
        table: Table,
        entry: Entry,
        (length, slot_index): (Val, Val),
    ) -> Res<Val> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let layout = table.layout;
        // The holes the store leaves: at least one, since the key after the
        // last slot is the append's, and fewer than the reach. A key below 1
        // is far past both as an unsigned count.
        let holes = self.f.bin(Op::Sub, slot_index, length);
        let minus = self.iconst(-1);
        let spare = self.f.bin(Op::Add, holes, minus);
        let reach = self.iconst(NATIVE_BOOLEAN_GROWTH_REACH as i64 - 1);
        let open_near = self.f.begin_if(Cc::Ult, spare, reach);
        let open_started = self.f.begin_if(Cc::Ne, length, zero);
        let capacity = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.array_capacity),
        );
        let open_fits = self.f.begin_if(Cc::Ule, entry.index, capacity);
        let pending = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.pending_deletions),
        );
        let open_settled = self.f.begin_if(Cc::Ule, pending, length);
        let pointer = self.load_at(
            Ty::I64,
            table.address,
            Off::K(0),
            i64::from(layout.array_pointer),
        );
        let four = self.iconst(4);
        let from = self.f.bin(Op::Shl, length, four);
        // The first hole is written here, and the others, when a store
        // leaves more than one, out of line.
        let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
        self.store8(pointer, Off::V(from), 0, nil);
        self.f.call_if(
            Cc::Ugt,
            (holes, one),
            fill_nil_slots as *const () as usize,
            false,
            &[pointer, length, holes],
        );
        let scaled = self.f.bin(Op::Shl, slot_index, four);
        let at = Off::V(scaled);
        let boolean = self.iconst(i64::from(NATIVE_VALUE_TAG_BOOLEAN));
        self.store8(pointer, at, 0, boolean);
        self.payload_access(TraceType::Boolean, pointer, at, Some(entry.value))?;
        self.store_at(
            table.address,
            Off::K(0),
            i64::from(layout.array_length),
            entry.index,
        );
        let added = self.f.bin(Op::Add, holes, one);
        let at = offset(mem::offset_of!(TraceControl, appended));
        self.add_at(self.control, Off::K(0), at, added);
        // The array stays where it is, so the bindings keep its address.
        self.store_array_words(table.header, entry.index, false, None);
        let grown = self.f.end_if(open_settled, &[(one, zero)]);
        let grown = self.f.end_if(open_fits, &[(grown, zero)]);
        let grown = self.f.end_if(open_started, &[(grown, zero)]);
        Ok(self.f.end_if(open_near, &[(grown, zero)]))
    }

    /// Writes a scalar into the array slot at `slot`: its tag and its payload.
    fn store_slot(&mut self, slot: Val, ty: TraceType, value: Val) -> Res<()> {
        let tag = match ty {
            TraceType::Float => NATIVE_VALUE_TAG_NUMBER,
            TraceType::Integer => NATIVE_VALUE_TAG_INTEGER,
            _ => NATIVE_VALUE_TAG_BOOLEAN,
        };
        let tag = self.iconst(i64::from(tag));
        self.store8(slot, Off::K(0), 0, tag);
        self.payload_store(ty, slot, value)
    }

    /// Has the VM store the string, table or nil of `entry` into the array
    /// slot at `slot` (see [`crate::jit::HostEntry::store`]); `replace` says
    /// the slot holds a value. Returns the status, nonzero when the VM stored
    /// nothing.
    fn store_reference(
        &mut self,
        header: usize,
        slot: Val,
        entry: Entry,
        replace: bool,
    ) -> Res<Val> {
        let (tag, bits) = self.scalar_words(entry.ty, entry.value)?;
        let context = self.control_word(mem::offset_of!(TraceControl, host_context));
        let binding = self.iconst(i64::try_from(header).expect("a binding index fits i64"));
        let replace = self.iconst(i64::from(replace));
        let store_entry = mem::offset_of!(TraceControl, store_entry);
        Ok(self.f.call_ctl(
            offset(store_entry) as i32,
            &[context, binding, tag, bits, slot, replace],
            Some(Ty::I64),
        ))
    }

    /// The array and length words of the bindings of the table of `header` in
    /// the run's binding slots, set to a table of `length` slots at `pointer`
    /// (`None` leaves the address): an append, a pop or a growth did it in
    /// place, so the VM has not rewritten them (`store_array_words` of
    /// `emit.rs`). The length bindings follow only if the border moved with the
    /// length (`moves_border`): a growth with holes leaves it.
    fn store_array_words(
        &mut self,
        header: usize,
        length: Val,
        moves_border: bool,
        pointer: Option<Val>,
    ) {
        let bindings = &self.members[0].ir.bindings;
        let register = bindings[header].register;
        let first = self.members[0].ir.entry.inputs.len();
        let slot = mem::size_of::<TraceInputSlot>();
        let mine: Vec<(usize, BindingKind)> = bindings
            .iter()
            .enumerate()
            .filter(|(_, other)| other.register == register)
            .map(|(index, other)| (index, other.kind.clone()))
            .collect();
        for (index, kind) in mine {
            let at = offset((first + index) * slot);
            let tag_at = at + offset(mem::offset_of!(TraceInputSlot, tag));
            let bits_at = at + offset(mem::offset_of!(TraceInputSlot, bits));
            match kind {
                BindingKind::Array { .. } => {
                    self.store_at(self.inputs, Off::K(0), tag_at, length);
                    if let Some(pointer) = pointer {
                        self.store_at(self.inputs, Off::K(0), bits_at, pointer);
                    }
                }
                BindingKind::Length { ty, .. } if moves_border => {
                    let border = if ty == TraceType::Float {
                        self.un(Op::I2F, Ty::F64, length)
                    } else {
                        length
                    };
                    self.store_at(self.inputs, Off::K(0), bits_at, border);
                }
                _ => {}
            }
        }
    }
}
