//! `ArrayNext`: the step along the array part of a bound table (`array_next`
//! of `emit.rs`).
//!
//! The step gives the one-based index of the first slot that holds a value
//! after the slot the control key names (the first slot of the array for the
//! start of a traversal). The guard leaves, having done nothing, when the key
//! is not an integral number naming a slot that holds a value now, and when
//! no slot with a value follows.
//!
//! Cranelift finds the slot with a loop over the nil slots after a look at the
//! slot right after the key, which holds a value in nearly every step. MIR has
//! no loop inside a region, so the look is in line and the loop is a call of
//! [`scan_value_slots`] that a skip makes only when that slot is a hole. The
//! slot the step found is kept as a slot fact for the key it gave, so an
//! `ArrayLoad` of that key reads the slot with no bounds check
//! (`next_slot` of `emit.rs`).
//!
//! A loop whose back edge passes the result of the step for the key, and
//! whose sides link back with such a result, is entered with a key that names
//! a slot holding a value, and every step after finds the slot it gives
//! (`trusted_array_key` of `emit.rs`): the entry checks the key it starts with
//! ([`Lowerer::check_trusted_key`]) and declines the run if it does not hold,
//! and the steps take their key as it is.
//!
//! The step of a boxed `next` loop, in line, as Cranelift takes it
//! (`call_helper_stepping` in `emit.rs`).
//!
//! The control value of such a loop lives in a lane of the run's bank. Before
//! the helper is called, the code tries to make the step itself: along the
//! array part of the table (the lane holds nil or a whole number inside the
//! array, and the slot after it holds a value), then along the snapshot of the
//! hash part (`NativeCursor`). An attempt that does not fit changes nothing
//! and the helper makes the step, so the results and the lane afterwards are
//! the same either way.
//!
//! Cranelift scans over the holes of the array and the deleted entries of the
//! snapshot. A skip cannot loop, so this backend looks at the next slot or
//! entry only and leaves a hole to the helper, which skips it. That is the
//! same answer, and it costs one call per hole, where the array of a loop
//! that reaches this step has few.
//!
//! The attempt is a chain of skips, one for each check that must pass before
//! the next load, since a load past the end of the array or a snapshot must
//! not run. Each skip yields the flag "the step was taken" and the results,
//! which the skip after it joins with the helper's.

use super::super::super::emit::BoxStep;
#[allow(
    clippy::wildcard_imports,
    reason = "a child module of the lowering, which shares its names"
)]
use super::*;
use crate::jit_ir::trace::{BoxedValue, HelperEffects, HelperId, HelperType, NextKind};
use crate::runtime::{
    NATIVE_CURSOR_SLOT_KEY_BITS, NATIVE_CURSOR_SLOT_KEY_TAG, NATIVE_CURSOR_SLOT_VALUE,
    NATIVE_VALUE_SIZE, NativeCursor,
};

/// The index of the first slot from `from` on, of the `len` slots at address
/// `base`, that holds a value, or `len` when there is none.
#[allow(
    unsafe_code,
    reason = "native code calls it with the array's own slots"
)]
extern "C" fn scan_value_slots(base: usize, len: usize, from: usize) -> usize {
    let mut index = from;
    while index < len {
        // SAFETY: the caller passes the address and the length of the array
        // part, which has `len` slots of `NATIVE_VALUE_SIZE` bytes.
        let tag = unsafe { ((base + index * NATIVE_VALUE_SIZE) as *const u8).read() };
        if tag != NATIVE_VALUE_TAG_NIL {
            break;
        }
        index += 1;
    }
    index
}

/// The control lane of a step and what the attempts read of it.
struct StepLane {
    /// The address of the lane in the bank.
    at: Val,
    /// The tag of the lane.
    tag: Val,
    /// The description of the hash part's snapshot, when the run keeps one
    /// (the address the lane two after the control lane holds).
    view: Option<Val>,
}

impl Lowerer<'_, '_> {
    /// An [`TraceOp::ArrayNext`] on the array `binding`: the key of the slot
    /// the step found, as a number of type `ty`.
    pub(super) fn array_next(
        &mut self,
        binding: u32,
        key: ValueId,
        (start, ty): (bool, TraceType),
        result: Option<ValueId>,
        guard: GuardId,
    ) -> Res<Val> {
        if !self.members[0].ir.bindings[binding as usize].is_array() {
            return self.reject("array access through another binding");
        }
        let words = self.words_of(binding)?;
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let four = self.iconst(4);
        let trusted = !start
            && self.member == 0
            && self
                .trusted
                .is_some_and(|trusted| trusted.key == key && trusted.binding == binding);
        let from = if start {
            zero
        } else if trusted {
            // The key names a slot that holds a value, so the slot after it
            // is the key as a zero-based index.
            let key_val = self.get(key)?;
            if self.ty(key) == TraceType::Float {
                self.un(Op::F2I, Ty::I64, key_val)
            } else {
                key_val
            }
        } else {
            // The key is a slot inside the array, and the slot holds a value.
            let index = match self.index_within(words, key, guard, 0)? {
                Off::K(slot) => self.iconst(slot),
                Off::V(index) => index,
                Off::VK(..) => unreachable!("an index of shift 0 has no constant part"),
            };
            let scaled = self.f.bin(Op::Shl, index, four);
            let tag = self.load8(words.base, Off::V(scaled), 0);
            let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
            self.guard_cc(Cc::Ne, tag, nil, guard)?;
            self.f.bin(Op::Add, index, one)
        };
        self.guard_cc(Cc::Ult, from, words.len, guard)?;
        let scaled = self.f.bin(Op::Shl, from, four);
        let first = self.load8(words.base, Off::V(scaled), 0);
        let open = self.f.begin_if(Cc::Eq, first, zero);
        let scanned = self.f.call(
            scan_value_slots as *const () as usize,
            &[words.base, words.len, from],
            Some(Ty::I64),
        );
        let found = self.f.end_if(open, &[(scanned, from)]);
        self.guard_cc(Cc::Ult, found, words.len, guard)?;
        let scaled = self.f.bin(Op::Shl, found, four);
        let step = self.f.bin(Op::Add, found, one);
        let number = if ty == TraceType::Float {
            self.un(Op::I2F, Ty::F64, step)
        } else {
            step
        };
        if let Some(result) = result {
            self.s
                .slots
                .retain(|fact| !(fact.binding == binding && fact.key == result.index() as u32));
            self.s.slots.push(SlotFact {
                binding,
                key: result.index() as u32,
                off: Off::V(scaled),
                ty: TraceType::Any,
            });
        }
        Ok(number)
    }

    /// At entry: declines the run (as an input of another type does) unless
    /// the trusted key names a slot of its array that holds a value
    /// (`check_trusted_key` of `emit.rs`).
    pub(super) fn check_trusted_key(&mut self) -> Res<()> {
        let Some(trusted) = self.trusted else {
            return Ok(());
        };
        let words = self.words_of(trusted.binding)?;
        let key = self.get(trusted.init)?;
        let decline = self
            .f
            .snap_returning(&[], -i64::try_from(trusted.input).expect("few inputs") - 1);
        let guard = |this: &mut Self, cc: Cc, a: Val, b: Val| {
            let mut inst = Inst::new(Op::Guard, this.f.val_ty[a as usize]);
            inst.cc = cc;
            inst.a = a;
            inst.b = b;
            inst.c = decline;
            this.f.push(inst);
        };
        let minus = self.iconst(-1);
        let index = if self.ty(trusted.init) == TraceType::Float {
            if words.lenf == NONE {
                return self.reject("float key without the length as a float");
            }
            let one = self.f.fconst(1.0);
            guard(self, Cc::Ge, key, one);
            guard(self, Cc::Le, key, words.lenf);
            let magic = self.f.fconst(FLOAT_INTEGER_MAGIC);
            let shifted = self.f.bin(Op::FAdd, key, magic);
            let restored = self.f.bin(Op::FSub, shifted, magic);
            guard(self, Cc::Eq, restored, key);
            let whole = self.un(Op::F2I, Ty::I64, key);
            self.f.bin(Op::Add, whole, minus)
        } else {
            let index = self.f.bin(Op::Add, key, minus);
            guard(self, Cc::Ult, index, words.len);
            index
        };
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, index, four);
        let tag = self.load8(words.base, Off::V(scaled), 0);
        let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
        guard(self, Cc::Ne, tag, nil);
        Ok(())
    }

    /// What a native step does for the call of `helper`, if the trace gives it
    /// one ([`TraceIr::boxed_steps`]) and the call is one it can take
    /// (`box_step_of` in `emit.rs`): a key of the type the profile's keys of
    /// the array part have, and a value that is a number or a boolean (a
    /// string or a table needs a lane). The hash part's snapshot is the
    /// storage as it stands: no operation of the tree may move it.
    pub(super) fn box_step_of(&self, helper: HelperId, args: &[ValueId]) -> Option<BoxStep> {
        let steps = self.members[0].ir.boxed_steps?;
        let (key, value) = helper.next_box_shape()?;
        let wanted = if steps.integer {
            NextKind::Integer
        } else {
            NextKind::Float
        };
        if args.len() != 2
            || key.is_some_and(|key| key != wanted)
            || matches!(value, BoxedValue::Kind(NextKind::String | NextKind::Table))
        {
            return None;
        }
        let moves_hash = self.members.iter().any(|m| m.ir.moves_hash_storage());
        let hash = steps.cursor && !moves_hash;
        if steps.array.is_none() && !hash {
            return None;
        }
        Some(BoxStep {
            array: steps.array,
            hash,
            cursor_lane: steps.cursor,
            integer: steps.integer,
            key,
            value,
        })
    }

    /// The call of a boxed `next` helper with the native attempt before it.
    /// The helper is called in a skip of its own when the attempt did not take
    /// the step, and the results of the two ways are joined.
    pub(super) fn call_helper_stepping(
        &mut self,
        step: &BoxStep,
        (helper, args): (HelperId, &[ValueId]),
        result: Option<ValueId>,
        guard: GuardId,
    ) -> Res<()> {
        let Some(info) = helper.info() else {
            return self.reject("helper without a description");
        };
        let words = match step.array {
            Some(binding) => {
                let words = self.words_of(binding)?;
                if words.base == NONE {
                    return self.reject("native next step without the array words");
                }
                Some(words)
            }
            None => None,
        };
        // What the skips read is made before the first opens.
        let lane = self.get(args[1])?;
        self.get(args[0])?;
        if info.effects.contains(HelperEffects::MAY_EXIT) {
            self.guard_target(guard)?;
        }
        let zero = self.iconst(0);
        let mut skipped = vec![zero];
        for kind in info.results {
            skipped.push(match kind {
                HelperType::Float => self.f.fconst(0.0),
                _ => zero,
            });
        }
        let bank = self.control_word(mem::offset_of!(TraceControl, bank));
        let taken = self.skip_if(Cc::Ne, (bank, zero), &skipped, |this| {
            this.step_in_bank(step, (lane, bank), words, &skipped)
        })?;
        // The helper makes the step the code did not.
        let hres_start = self.s.hres.len();
        let open = self.f.begin_if(Cc::Eq, taken[0], zero);
        self.call_helper_in_vm(helper, args, result, guard)?;
        let Some(result) = result.filter(|_| !info.results.is_empty()) else {
            self.f.end_if(open, &[]);
            return Ok(());
        };
        let mut by_helper = vec![self.s.vals[result.index()]];
        by_helper.resize(info.results.len(), NONE);
        for &(value, position, word) in &self.s.hres[hres_start..] {
            if value as usize == result.index() {
                by_helper[usize::from(position)] = word;
            }
        }
        self.s.hres.truncate(hres_start);
        let pairs: Vec<(Val, Val)> = by_helper
            .iter()
            .zip(&taken[1..])
            .map(|(&helper, &native)| (helper, native))
            .collect();
        let first = self.f.end_if(open, &pairs);
        self.s.vals[result.index()] = first;
        for position in 1..pairs.len() {
            let joined = first + Val::try_from(position).expect("few results");
            self.s
                .hres
                .push((result.index() as u32, position as u8, joined));
        }
        Ok(())
    }

    /// `if a cc b { body }`, yielding what the body yields when it ran and
    /// `skipped` when it did not.
    fn skip_if(
        &mut self,
        cc: Cc,
        (a, b): (Val, Val),
        skipped: &[Val],
        body: impl FnOnce(&mut Self) -> Res<Vec<Val>>,
    ) -> Res<Vec<Val>> {
        let open = self.f.begin_if(cc, a, b);
        let inner = body(self)?;
        debug_assert_eq!(inner.len(), skipped.len());
        let pairs: Vec<(Val, Val)> = inner.into_iter().zip(skipped.iter().copied()).collect();
        let first = self.f.end_if(open, &pairs);
        Ok((0..pairs.len())
            .map(|k| first + Val::try_from(k).expect("few results"))
            .collect())
    }

    /// The attempts, in a run that has a bank. Yields the flag and the results
    /// in the order of the helper's, as [`Self::skip_if`] does.
    fn step_in_bank(
        &mut self,
        step: &BoxStep,
        (lane, bank): (Val, Val),
        words: Option<Words>,
        skipped: &[Val],
    ) -> Res<Vec<Val>> {
        let four = self.iconst(4);
        let scaled = self.f.bin(Op::Shl, lane, four);
        let at = self.f.bin(Op::Add, bank, scaled);
        let tag = self.load8(at, Off::K(0), 0);
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        // The description of the hash part's snapshot: the address the lane two
        // after the control lane holds.
        let view = if step.cursor_lane {
            Some(self.load_at(Ty::I64, at, Off::K(0), 2 * 16 + payload))
        } else {
            None
        };
        let lane = StepLane { at, tag, view };
        let by_array = match words {
            Some(words) => Some(self.array_attempt(step, &lane, words, skipped)?),
            None => None,
        };
        let Some(view) = lane.view.filter(|_| step.hash) else {
            return Ok(by_array.unwrap_or_else(|| skipped.to_vec()));
        };
        // A tree that steps along no array part goes straight to the hash
        // part: the flag of the array attempt would be the constant 0.
        let Some(by_array) = by_array else {
            return self.hash_attempt(step, &lane, view, skipped);
        };
        // A step the array part did not take goes to the hash part.
        let zero = self.iconst(0);
        let taken = by_array[0];
        let kept = by_array.clone();
        self.skip_if(Cc::Eq, (taken, zero), &kept, |this| {
            this.hash_attempt(step, &lane, view, skipped)
        })
    }

    /// The step along the array part: from the start (the lane holds nil) or
    /// from a whole number inside the array whose slot holds a value, to the
    /// slot after it, when that holds a value of the kind the call wants.
    #[allow(clippy::too_many_lines)]
    fn array_attempt(
        &mut self,
        step: &BoxStep,
        lane: &StepLane,
        words: Words,
        skipped: &[Val],
    ) -> Res<Vec<Val>> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
        let and = |this: &mut Self, a: Val, b: Val| this.f.bin(Op::And, a, b);
        // A lane whose key the hash part's description holds is in the hash
        // part.
        let clear = match lane.view {
            Some(view) => {
                let pending = self.load_at(
                    Ty::I64,
                    view,
                    Off::K(0),
                    offset(mem::offset_of!(NativeCursor, pending)),
                );
                self.f.select(Cc::Eq, pending, zero, one, zero)
            }
            None => one,
        };
        let number_tag = if step.integer {
            NATIVE_VALUE_TAG_INTEGER
        } else {
            NATIVE_VALUE_TAG_NUMBER
        };
        let number_tag = self.iconst(i64::from(number_tag));
        let is_start = self.f.select(Cc::Eq, lane.tag, nil, one, zero);
        let is_number = self.f.select(Cc::Eq, lane.tag, number_tag, one, zero);
        let key_ty = if step.integer { Ty::I64 } else { Ty::F64 };
        let key = self.load_at(
            key_ty,
            lane.at,
            Off::K(0),
            i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
        );
        // The key as an integer, and whether it was one (a float key is
        // converted back and compared).
        let (key, whole) = if step.integer {
            (key, one)
        } else {
            let converted = self.un(Op::F2I, Ty::I64, key);
            let back = self.un(Op::I2F, Ty::F64, converted);
            let whole = self.f.select_any(Cc::Eq, back, key, one, zero);
            (converted, whole)
        };
        let minus_one = self.iconst(-1);
        let current = self.f.bin(Op::Add, key, minus_one);
        let inside = self.f.select(Cc::Ult, current, words.len, one, zero);
        let numbered = and(self, is_number, whole);
        let numbered = and(self, numbered, inside);
        let numbered = and(self, numbered, clear);
        // The slot of the key holds a value.
        let held = self.skip_if(Cc::Ne, (numbered, zero), &[zero], |this| {
            let four = this.iconst(4);
            let scaled = this.f.bin(Op::Shl, current, four);
            let slot_tag = this.load8(words.base, Off::V(scaled), 0);
            Ok(vec![this.f.select(Cc::Ne, slot_tag, nil, one, zero)])
        })?[0];
        let from_number = and(self, numbered, held);
        let right_key = self.f.bin(Op::Or, is_start, from_number);
        let ready = and(self, clear, right_key);
        // The slot after the key (the first one at the start) is inside.
        let next = self.f.select(Cc::Ne, is_start, zero, zero, key);
        let room = self.f.select(Cc::Ult, next, words.len, one, zero);
        let go = and(self, ready, room);
        self.skip_if(Cc::Ne, (go, zero), skipped, |this| {
            let four = this.iconst(4);
            let scaled = this.f.bin(Op::Shl, next, four);
            let slot = this.f.bin(Op::Add, words.base, scaled);
            let element = this.load8(slot, Off::K(0), 0);
            let pass = this.step_passes(step, lane, element, None)?;
            this.skip_if(pass.0, (pass.1, pass.2), skipped, |this| {
                // Every check passed: the stores, and the results.
                let number = this.f.bin(Op::Add, next, one);
                let number = if step.integer {
                    number
                } else {
                    this.un(Op::I2F, Ty::F64, number)
                };
                this.store_key(step, lane, number);
                if let Some(view) = lane.view {
                    // The lane no longer holds a key of the hash part.
                    let at = offset(mem::offset_of!(NativeCursor, in_sync));
                    this.store_at(view, Off::K(0), at, zero);
                }
                this.copy_value(step, lane, slot, element);
                Ok(this.step_results(step, number, slot))
            })
        })
    }

    /// The step along the hash part's snapshot: the entry after the position,
    /// when its value is not deleted and has the kind the call wants, and its
    /// key is an integer when the call wants the key.
    #[allow(clippy::too_many_lines)]
    fn hash_attempt(
        &mut self,
        step: &BoxStep,
        lane: &StepLane,
        view: Val,
        skipped: &[Val],
    ) -> Res<Vec<Val>> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let field = |at: usize| offset(at);
        // The lane holds the key at the snapshot's position.
        let in_sync = self.load_at(
            Ty::I64,
            view,
            Off::K(0),
            field(mem::offset_of!(NativeCursor, in_sync)),
        );
        self.skip_if(Cc::Ne, (in_sync, zero), skipped, |this| {
            let slots = this.load_at(
                Ty::I64,
                view,
                Off::K(0),
                field(mem::offset_of!(NativeCursor, slots)),
            );
            let len = this.load_at(
                Ty::I64,
                view,
                Off::K(0),
                field(mem::offset_of!(NativeCursor, len)),
            );
            let position_at = this.load_at(
                Ty::I64,
                view,
                Off::K(0),
                field(mem::offset_of!(NativeCursor, position)),
            );
            let position = this.load_at(Ty::I64, position_at, Off::K(0), 0);
            let first = this.f.bin(Op::Add, position, one);
            this.skip_if(Cc::Ult, (first, len), skipped, |this| {
                let five = this.iconst(5);
                let scaled = this.f.bin(Op::Shl, first, five);
                let entry = this.f.bin(Op::Add, slots, scaled);
                let value_at =
                    this.load_at(Ty::I64, entry, Off::K(0), field(NATIVE_CURSOR_SLOT_VALUE));
                let element = this.load8(value_at, Off::K(0), 0);
                // The key is an integer, as the call wants it.
                let is_integer = if step.key.is_some() {
                    let key_tag =
                        this.load_at(Ty::I64, entry, Off::K(0), field(NATIVE_CURSOR_SLOT_KEY_TAG));
                    let integer = this.iconst(i64::from(NATIVE_VALUE_TAG_INTEGER));
                    Some(this.f.select(Cc::Eq, key_tag, integer, one, zero))
                } else {
                    None
                };
                let pass = this.step_passes(step, lane, element, is_integer)?;
                this.skip_if(pass.0, (pass.1, pass.2), skipped, |this| {
                    // Every check passed. The snapshot's position moves to the
                    // entry, and the control lane does not hold its key until
                    // the run needs the lane (the VM commits `pending` then).
                    this.store_at(position_at, Off::K(0), 0, first);
                    this.store_at(
                        view,
                        Off::K(0),
                        field(mem::offset_of!(NativeCursor, pending)),
                        entry,
                    );
                    let number = if step.key.is_some() {
                        let bits = this.load_at(
                            Ty::I64,
                            entry,
                            Off::K(0),
                            field(NATIVE_CURSOR_SLOT_KEY_BITS),
                        );
                        if step.integer {
                            bits
                        } else {
                            this.un(Op::I2F, Ty::F64, bits)
                        }
                    } else {
                        zero
                    };
                    this.copy_value(step, lane, value_at, element);
                    Ok(this.step_results(step, number, value_at))
                })
            })
        })
    }

    /// The condition under which an attempt takes the step: the slot holds a
    /// value (its tag is `element`) of the kind the call wants, and `extra`,
    /// when given, is 1. A call that wants a number or a boolean needs the one
    /// compare of the tag, which a nil does not pass either.
    fn step_passes(
        &mut self,
        step: &BoxStep,
        lane: &StepLane,
        element: Val,
        extra: Option<Val>,
    ) -> Res<(Cc, Val, Val)> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let tag = match step.value {
            BoxedValue::Kind(NextKind::Integer) => Some(NATIVE_VALUE_TAG_INTEGER),
            BoxedValue::Kind(NextKind::Float) => Some(NATIVE_VALUE_TAG_NUMBER),
            BoxedValue::Kind(_) => Some(NATIVE_VALUE_TAG_BOOLEAN),
            _ => None,
        };
        if let (Some(tag), None) = (tag, extra) {
            let tag = self.iconst(i64::from(tag));
            return Ok((Cc::Eq, element, tag));
        }
        let nil = self.iconst(i64::from(NATIVE_VALUE_TAG_NIL));
        let holds = self.f.select(Cc::Ne, element, nil, one, zero);
        let kind = self.step_value_check(step, lane, element)?;
        let mut pass = self.f.bin(Op::And, holds, kind);
        if let Some(extra) = extra {
            pass = self.f.bin(Op::And, pass, extra);
        }
        Ok((Cc::Ne, pass, zero))
    }

    /// Whether the value of the entry the attempt found (its tag is `element`)
    /// is of the kind the call wants, and the lane that keeps it, for a loop
    /// variable the body never reads, can take a copy: 1 or 0.
    fn step_value_check(&mut self, step: &BoxStep, lane: &StepLane, element: Val) -> Res<Val> {
        let (zero, one) = (self.iconst(0), self.iconst(1));
        let is = |this: &mut Self, tag: u8| {
            let tag = this.iconst(i64::from(tag));
            this.f.select(Cc::Eq, element, tag, one, zero)
        };
        Ok(match step.value {
            BoxedValue::Kind(NextKind::Integer) => is(self, NATIVE_VALUE_TAG_INTEGER),
            BoxedValue::Kind(NextKind::Float) => is(self, NATIVE_VALUE_TAG_NUMBER),
            BoxedValue::Kind(_) => is(self, NATIVE_VALUE_TAG_BOOLEAN),
            BoxedValue::Skipped => one,
            BoxedValue::Lane => {
                // The lane after the control lane takes a copy of a number or
                // a boolean, over a value that holds no count either.
                let number = self.iconst(i64::from(NATIVE_VALUE_TAG_NUMBER));
                let scalar = self.f.select(Cc::Ule, element, number, one, zero);
                let old = self.load8(lane.at, Off::K(0), 16);
                let free = self.f.select(Cc::Ule, old, number, one, zero);
                self.f.bin(Op::And, scalar, free)
            }
        })
    }

    /// Stores `number` as the key the control lane holds.
    fn store_key(&mut self, step: &BoxStep, lane: &StepLane, number: Val) {
        let tag = if step.integer {
            NATIVE_VALUE_TAG_INTEGER
        } else {
            NATIVE_VALUE_TAG_NUMBER
        };
        let tag = self.iconst(i64::from(tag));
        self.store8(lane.at, Off::K(0), 0, tag);
        self.store_at(
            lane.at,
            Off::K(0),
            i64::from(NATIVE_VALUE_PAYLOAD_OFFSET),
            number,
        );
    }

    /// For a loop variable the body never reads, copies the value at `slot`
    /// (its tag is `element`) into the lane after the control lane.
    fn copy_value(&mut self, step: &BoxStep, lane: &StepLane, slot: Val, element: Val) {
        if !matches!(step.value, BoxedValue::Lane) {
            return;
        }
        let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
        let bits = self.load_at(Ty::I64, slot, Off::K(0), payload);
        self.store8(lane.at, Off::K(0), 16, element);
        self.store_at(lane.at, Off::K(0), 16 + payload, bits);
    }

    /// The flag and the results of a step that passed its checks: the key
    /// (`number`) and the value read from the slot at `slot`, as the call
    /// asks.
    fn step_results(&mut self, step: &BoxStep, number: Val, slot: Val) -> Vec<Val> {
        let mut gave = vec![self.iconst(1)];
        if step.key.is_some() {
            gave.push(number);
        }
        if let BoxedValue::Kind(kind) = step.value {
            let payload = i64::from(NATIVE_VALUE_PAYLOAD_OFFSET);
            gave.push(match kind {
                NextKind::Integer => self.load_at(Ty::I64, slot, Off::K(0), payload),
                NextKind::Float => self.load_at(Ty::F64, slot, Off::K(0), payload),
                _ => self.load8(slot, Off::K(0), payload),
            });
        }
        gave
    }
}
