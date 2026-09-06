//! `next`, `pairs` and `ipairs`, and with them the loops
//! `for k, v in pairs (t)` and `for k, v in next, t` run in a trace.
//!
//! `pairs (t)` and `ipairs (t)` give the iterator, the table and the start of
//! the traversal as constants and moves: the callee is pinned, so the guard on
//! it is all the trace keeps of the call. `pairs` also checks, where the
//! profile honours `__pairs`, that the table's metatable has no handler.
//!
//! A step of `next (t, key)` has three forms.
//!
//! The array form moves along the array part of a table the trace binds, with
//! the two operations of [`Recorder::record_array_next`]: a scan that finds
//! the first slot holding a value after the control value, whatever the
//! length of the run of nil slots between ([`oximond::jit_ir::trace::TraceOp::ArrayNext`]),
//! and the read of the element `ipairs` makes. The key it gives has the type
//! the profile's `next` gives, which a float control value on Lua 5.3 and
//! later does not have. It also starts a traversal. It runs whether or not
//! loops enclose the traversal, for a table that has no hash part: a `for`
//! loop takes it when the key of the iteration it records is a whole number
//! with a slot after it in the array part. The end of the array part leaves
//! through the guard of the scan, and a side trace records what follows (the
//! end of the traversal).
//!
//! The keyed form calls a helper for the step, which is the interpreter's own
//! (`Runtime::table_next_fast`, then `Runtime::table_next`): the table's
//! traversal cursor answers it with no lookup of the key. It passes the key as
//! an argument and gives the next key and the value. Its helper is chosen by
//! the types of the key and the value the recording saw (see
//! [`oximond::jit_ir::trace::HelperId::next`]). A step that gives another
//! shape, the end of the table, or a continuation key the cursor cannot
//! settle (the interpreter raises an error for an invalid one) misses, leaves
//! through the `Before` snapshot of the call, and the interpreter makes the
//! step. The end of the traversal has its own helper
//! ([`oximond::jit_ir::trace::HelperId::NEXT_END`]), so a side trace takes the
//! path past the loop (see `side_exits_of`). A call of `next` outside a loop
//! and an array whose elements are strings or tables take this form.
//!
//! The lane form is for a `for` loop over a table that has a hash part, whose
//! keys have several types (the array part's integers among them), so a
//! register typed by the key of one iteration would not take the next. The
//! control register is an entry input of type string whose lane holds the
//! key itself, whatever its type (see `TraceIr::boxed_control`), and the step
//! helper reads the key from the lane and stores the one it finds
//! ([`oximond::jit_ir::trace::HelperId::next_box`]). An exit restores the
//! control register from the lane, so the interpreter continues the
//! traversal from the key it would hold. A write to the control register
//! outside a step (`pairs (t)` giving nil, the start of the next traversal
//! in an outer loop) stores into the lane
//! ([`oximond::jit_ir::trace::HelperId::BOX_SET`]).
//!
//! One version of the lane form traverses the whole table. The code of its
//! step takes the steps along the array part and along the hash part itself
//! where it can, and calls the helper for the others (see
//! [`oximond::jit_ir::trace::BoxedSteps`]): the helper is the step at a part's
//! end, the first step into the hash part, a key or a value of another kind
//! than the recording saw, and every step of a trace with an operation that
//! may move hash storage. The array part's steps need the table's array
//! binding, which the recorder adds for a table that has an array part.
//!
//! The loop variables the body neither reads nor writes are not produced as
//! values of the trace: a loop that sums the values never makes a key, a lane
//! or a string. An exit still gives them what the interpreter holds, since
//! `debug.getlocal` shows them to a function the exit calls and to an error
//! handler. The first variable holds the key the control lane holds, so a
//! snapshot names that lane for it. The second gets a lane of its own, after
//! the control lane, which the step stores the value in
//! ([`oximond::jit_ir::trace::BoxedValue::Lane`]). A later variable is nil, as
//! the entry left it.
//!
//! In the array form and the keyed form the control value of the loop is the
//! key, as the interpreter holds it, so every exit restores the registers the
//! interpreter would have.

use super::{
    Flow, HookPoint, Live, Number, Operand, RecKind, Recorder, SnapshotPhase, TraceAbortReason,
    TraceScalar, TraceType,
};
use crate::jit_ir::trace::{
    BindingKind, BoxedSteps, BoxedValue, HelperId, HelperType, NextKind, PINNED_BUILTIN_IPAIRS,
    PINNED_BUILTIN_IPAIRS_ITERATOR, PINNED_BUILTIN_NEXT, PINNED_BUILTIN_PAIRS,
    pinned_builtin_identity,
};
use crate::runtime::Value;

/// The kind a key or a value of `next` has in a helper's result.
fn next_kind(value: &Value) -> Option<NextKind> {
    match value {
        Value::Integer(_) => Some(NextKind::Integer),
        Value::Number(_) => Some(NextKind::Float),
        Value::Boolean(_) => Some(NextKind::Boolean),
        Value::String(_) => Some(NextKind::String),
        Value::Table(_) => Some(NextKind::Table),
        _ => None,
    }
}

impl Recorder {
    /// A call of `pairs (t)`: the iterator `next`, the table and nil.
    pub(super) fn record_pairs(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let Some(&(table, TraceType::Table)) = arguments.first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let Value::Table(handle) = at.registers[first_argument] else {
            return Err(TraceAbortReason::Divergence);
        };
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_PAIRS))?;
        let iterator = self.pin(pinned_builtin_identity(PINNED_BUILTIN_NEXT))?;
        // The profiles of the Lua 5.1 family never look for `__pairs`.
        if self.honours_pairs {
            let runtime = at
                .surroundings
                .ok_or(TraceAbortReason::UnsupportedOperand)?
                .runtime;
            if !crate::vm::jit::trace::helpers::pairs_is_plain(runtime, handle) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        }
        self.guard_callee(pc, callee, lane);
        if self.honours_pairs {
            self.emit_helper_call(pc, HelperId::PAIRS_PLAIN, &arguments[..1]);
        }
        let iterator = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(iterator))),
            Some(TraceType::Closure),
        );
        let state = self.emit(pc, RecKind::Move(table), Some(TraceType::Table));
        let start = self.scalar(pc, TraceScalar::Nil);
        Ok(self.native_results(pc, &[iterator, state, start], (base, results)))
    }

    /// A call of `ipairs (t)`: the iterator, the table and zero.
    pub(super) fn record_ipairs(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let Some(&(table, TraceType::Table)) = arguments.first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_IPAIRS))?;
        let iterator = self.pin(pinned_builtin_identity(PINNED_BUILTIN_IPAIRS_ITERATOR))?;
        self.guard_callee(pc, callee, lane);
        let iterator = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(iterator))),
            Some(TraceType::Closure),
        );
        let state = self.emit(pc, RecKind::Move(table), Some(TraceType::Table));
        let start = self.number_scalar(pc, 0);
        Ok(self.native_results(pc, &[iterator, state, start], (base, results)))
    }

    /// The step of a `next` loop whose control value is a lane of the run's
    /// bank (see [`Recorder::boxes_loop`]), recorded as the call of the
    /// loop's `ForInCall`. The helper reads the key from the lane and stores
    /// the key it finds there, so a step makes no key to look up and none to
    /// publish unless the body reads the variable.
    ///
    /// The loop variables the body neither reads nor writes are not results
    /// of the step: an exit restores the first from the control lane, which
    /// holds the key, and the second from a lane of its own that the step
    /// stores the value in (see the module documentation). The first loop
    /// variable the body does read keeps its type from one iteration to the
    /// next, as a loop parameter does.
    pub(super) fn record_boxed_next(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let control = base - 1;
        let (table, TraceType::Table) = self.read(at.registers, control - 1)? else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let Value::Table(handle) = at.registers[control - 1] else {
            return Err(TraceAbortReason::Divergence);
        };
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        // Run the step, to see how the interpreter's ends. An invalid
        // continuation key raises an error there, which no trace repeats.
        let (found_key, found_value) = runtime
            .table_next(handle, &at.registers[control])
            .map_err(|_| TraceAbortReason::UnsupportedOperand)?;
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_NEXT))?;
        let arguments = [
            (table, TraceType::Table),
            (Operand::Carried(control), TraceType::String),
        ];
        if matches!(found_key, Value::Nil) {
            self.guard_callee(pc, callee, lane);
            self.emit_helper_call(pc, HelperId::NEXT_BOX_END, &arguments);
            self.boxed_found = Some(false);
            self.write_nils(pc, base..base + results);
            return Ok(Flow::Next(pc + 1));
        }
        // What the body reads of the entry. The loop writes it back as the
        // parameter of its type, so a side trace that finds the other type
        // cannot join the loop: it would only be recorded and rejected.
        let kind = |wanted: bool, value: &Value| {
            if wanted {
                next_kind(value)
                    .map(Some)
                    .ok_or(TraceAbortReason::UnsupportedOperand)
            } else {
                Ok(None)
            }
        };
        let key_kind = kind(!self.unread.contains(&base), &found_key)?;
        // A second variable the body never reads is kept in its lane, for an
        // exit to restore; one it reads is a result of the step.
        let value_kind = kind(
            results > 1 && !self.unread.contains(&(base + 1)),
            &found_value,
        )?;
        let value = match value_kind {
            Some(kind) => BoxedValue::Kind(kind),
            None if self.boxed_value && results > 1 => BoxedValue::Lane,
            None => BoxedValue::Skipped,
        };
        // The steps along the table's parts that native code takes before the
        // helper (see [`BoxedSteps`]): the array part, when the table has
        // one, and the snapshot of the hash part. Both give a key of the
        // type the array part's keys have and a number or a boolean, so a
        // loop that reads a string key or a value that is a string or a table
        // has no use for them.
        let native = key_kind.is_none_or(|kind| {
            kind == if self.integers {
                NextKind::Integer
            } else {
                NextKind::Float
            }
        }) && !matches!(value, BoxedValue::Kind(NextKind::String | NextKind::Table));
        if native && self.boxed_steps.is_none() && self.side.is_none() {
            let array = runtime
                .table_array_len(handle)
                .is_ok_and(|len| len > 0)
                .then(|| self.table_receiver(at, control - 1))
                .and_then(Result::ok)
                .and_then(|(receiver, _, _)| Self::bound_window(receiver).ok())
                .map(|window| self.binding(window, BindingKind::Array { first: None }));
            self.boxed_steps = Some(BoxedSteps {
                array,
                cursor: true,
                integer: self.integers,
            });
        }
        if let Some(seed) = self.side.as_ref().filter(|seed| seed.start_pc == pc) {
            for (register, kind) in [(base, key_kind), (base + 1, value_kind)] {
                let parameter = seed
                    .params
                    .iter()
                    .find(|(known, _)| usize::from(*known) == register);
                if let (Some(kind), Some((_, ty))) = (kind, parameter)
                    && kind.helper_type().trace_type() != *ty
                {
                    return Err(TraceAbortReason::TypeUnstable);
                }
            }
        }
        self.guard_callee(pc, callee, lane);
        let call = self.emit_helper_call(pc, HelperId::next_box(key_kind, value), &arguments);
        let mut produced: Vec<(usize, u32)> = Vec::new();
        let mut index = 0;
        for (register, kind) in [(base, key_kind), (base + 1, value_kind)] {
            let Some(kind) = kind else { continue };
            let value = if index == 0 {
                call
            } else {
                self.emit(
                    pc,
                    RecKind::HelperResult { call, index },
                    Some(kind.helper_type().trace_type()),
                )
            };
            produced.push((register, value));
            index += 1;
        }
        for (register, value) in produced {
            self.write(register, value);
        }
        // A variable past the second that the body reads is nil.
        self.write_nils(pc, base + 2..base + results);
        self.boxed_found = Some(true);
        Ok(Flow::Next(pc + 1))
    }

    /// Writes nil to the loop variables in `range` that the body reads.
    fn write_nils(&mut self, pc: usize, range: std::ops::Range<usize>) {
        let mut nil = None;
        for register in range {
            if !self.unread.contains(&register) {
                let value = *nil.get_or_insert_with(|| self.scalar(pc, TraceScalar::Nil));
                self.write(register, value);
            }
        }
    }

    /// A call of `next (t [, key])`: the entry after `key`, or nil at the end
    /// of the traversal.
    pub(super) fn record_next_step(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let Some(&(table, TraceType::Table)) = arguments.first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        if arguments.len() > 2 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let Value::Table(handle) = at.registers[first_argument] else {
            return Err(TraceAbortReason::Divergence);
        };
        let key = match arguments.get(1) {
            Some(&(operand, ty)) if HelperType::Value.accepts(ty) => (operand, ty),
            Some(_) => return Err(TraceAbortReason::UnsupportedOperand),
            None => {
                let nil = self.scalar(pc, TraceScalar::Nil);
                (Operand::Op(nil), TraceType::Nil)
            }
        };
        let live = if arguments.len() > 1 {
            at.registers[first_argument + 1].clone()
        } else {
            Value::Nil
        };
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        // Run the step, to see how the interpreter's ends. An invalid
        // continuation key raises an error there, which no trace repeats.
        let (found_key, found_value) = runtime
            .table_next(handle, &live)
            .map_err(|_| TraceAbortReason::UnsupportedOperand)?;
        let ended = matches!(found_key, Value::Nil);
        // A side that starts at this step takes the path the loop's trace
        // missed. An entry whose key has another type than the loop's would
        // link back to a loop typed for the first one, so only the end of the
        // traversal, or an entry of the same key type (the next element
        // after a hole), is a path to record.
        if !ended
            && self.crossed.is_empty()
            && self.side.as_ref().is_some_and(|seed| seed.start_pc == pc)
            && !matches!(key.1, TraceType::Nil)
            && NextKind::of(key.1).map(NextKind::helper_type)
                != next_kind(&found_key).map(NextKind::helper_type)
        {
            return Err(TraceAbortReason::TypeUnstable);
        }
        if !ended
            && let Some(flow) =
                self.record_array_next(at, callee, (arguments, first_argument), (base, results))?
        {
            return Ok(flow);
        }
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_NEXT))?;
        let arguments = [(table, TraceType::Table), key];
        if ended {
            self.guard_callee(pc, callee, lane);
            self.emit_helper_call(pc, HelperId::NEXT_END, &arguments);
            // One nil, as the interpreter's `next` ends; fixed results
            // are nil padded.
            let first = self.scalar(pc, TraceScalar::Nil);
            return Ok(self.native_results(pc, &[first], (base, results)));
        }
        let key_kind = next_kind(&found_key).ok_or(TraceAbortReason::UnsupportedOperand)?;
        // A call that wants the key only takes no notice of the value.
        let value_kind = if results == 1 {
            None
        } else {
            Some(next_kind(&found_value).ok_or(TraceAbortReason::UnsupportedOperand)?)
        };
        self.guard_callee(pc, callee, lane);
        let call = self.emit_helper_call(pc, HelperId::next(key_kind, value_kind), &arguments);
        let mut values = vec![call];
        if let Some(kind) = value_kind {
            let value = self.emit(
                pc,
                RecKind::HelperResult { call, index: 1 },
                Some(kind.helper_type().trace_type()),
            );
            values.push(value);
        }
        Ok(self.native_results(pc, &values, (base, results)))
    }
}

impl Recorder {
    /// A step of `next` that moves to the next non-nil slot of the array part
    /// of a bound table, recorded as [`oximond::jit_ir::trace::TraceOp::ArrayNext`]
    /// and the read of the element `ipairs` makes. The first op scans the nil
    /// slots between the control value and the slot with a value in a native
    /// loop, as the interpreter's traversal does, so the step is the
    /// interpreter's own however long the run of holes is: the first entry
    /// past a key of the array part is the first non-nil slot after it,
    /// whether or not the slot of the key itself still holds a value. The
    /// start of a traversal (the key is nil, or `next (t)`) is the same step
    /// from slot zero.
    ///
    /// A step that finds the end of the array part, a key that is no slot
    /// holding a value, or an element of another type leaves before the call,
    /// and the interpreter (or a side trace) makes it. The key it gives has
    /// the type the profile's `next` gives for a key of the array part,
    /// whatever the control value's type is (a float key gives an integer on
    /// the profiles that have integers). `None` where the step is not that,
    /// which takes the helper.
    pub(super) fn record_array_next(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        if arguments.len() > 2 {
            return Ok(None);
        }
        let Ok((receiver, handle, runtime)) = self.table_receiver(at, first_argument) else {
            return Ok(None);
        };
        let Ok(window) = Self::bound_window(receiver) else {
            return Ok(None);
        };
        let Ok(len) = runtime.table_array_len(handle) else {
            return Ok(None);
        };
        let Some((control, from)) = self.array_next_control(at, arguments, first_argument, len)
        else {
            return Ok(None);
        };
        // The slot of the control value holds a value now: the interpreter
        // continues from a key of the array part only then (a deleted key
        // is the helper's to settle).
        if from >= 1
            && runtime
                .table_raw_array_range(handle, from, from)
                .ok()
                .flatten()
                .and_then(<[Value]>::first)
                .is_none_or(|live| matches!(live, Value::Nil))
        {
            return Ok(None);
        }
        // The first slot with a value after the key, and the element in it.
        let Some(slots) = runtime
            .table_raw_array_range(handle, from + 1, len)
            .ok()
            .flatten()
        else {
            return Ok(None);
        };
        let Some(skipped) = slots.iter().position(|slot| !matches!(slot, Value::Nil)) else {
            return Ok(None);
        };
        // The element is a value of a type with a slot: `array_read` would
        // refuse another value, and abort the recording. A string is
        // borrowed from its slot, which a loop cannot carry round (its
        // variable is typed as a string of the bank), and a table needs a
        // lane, so the helper steps over those.
        if self.array_slot_type(&slots[skipped]).is_none_or(|ty| {
            matches!(
                ty,
                TraceType::StringRef | TraceType::Table | TraceType::Closure
            )
        }) {
            return Ok(None);
        }
        let found = from + 1 + skipped;
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_NEXT))?;
        self.guard_callee(pc, callee, lane);
        // The key the profile's `next` gives.
        let ty = if self.integers {
            TraceType::Integer
        } else {
            TraceType::Float
        };
        let key = control.map_or_else(
            || {
                let zero = if self.integers {
                    TraceScalar::Integer(0)
                } else {
                    TraceScalar::Float(0.0)
                };
                Operand::Op(self.emit(pc, RecKind::Constant(zero), Some(ty)))
            },
            |control| control.operand,
        );
        if runtime.table_has_hash_part(handle).unwrap_or(true) {
            self.array_steps.0 += 1;
        } else {
            self.array_steps.1 += 1;
        }
        let first = self.first_slot_type(runtime, handle);
        let binding = self.binding(window, BindingKind::Array { first });
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        let next = self.emit(
            pc,
            RecKind::ArrayNext {
                binding,
                table: Operand::Carried(window),
                key,
                start: control.is_none(),
                ty,
                snapshot,
            },
            Some(ty),
        );
        let live = if self.integers {
            Live::Integer(i64::try_from(found).unwrap_or(i64::MAX))
        } else {
            #[allow(clippy::cast_precision_loss)]
            Live::Float(found as f64)
        };
        let (element, _) = self.array_read(
            pc,
            (Operand::Carried(window), window),
            (handle, runtime),
            Number {
                operand: Operand::Op(next),
                live,
            },
            false,
        )?;
        Ok(Some(self.native_results(
            pc,
            &[next, element],
            (base, results),
        )))
    }

    /// The control value of an array step and the index it names: a whole
    /// number from 1 with the slot after it inside the array part of `len`
    /// slots, or none when the traversal starts. `None` where the step is
    /// not that, which takes the helper.
    fn array_next_control(
        &mut self,
        at: &HookPoint<'_>,
        arguments: &[(Operand, TraceType)],
        first_argument: usize,
        len: usize,
    ) -> Option<(Option<Number>, usize)> {
        if matches!(arguments.get(1), None | Some(&(_, TraceType::Nil))) {
            return Some((None, 0));
        }
        let control = self.read_number(at.registers, first_argument + 1).ok()?;
        let from = match control.live {
            Live::Integer(index) if index >= 1 => usize::try_from(index).ok(),
            Live::Float(index) if index >= 1.0 && index.fract() == 0.0 =>
            {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(index as usize)
            }
            _ => None,
        }?;
        (from <= len).then_some((Some(control), from))
    }
}
