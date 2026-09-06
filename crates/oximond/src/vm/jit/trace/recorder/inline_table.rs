//! `rawget`, `rawset`, `rawlen`, `getmetatable`, `setmetatable` and `unpack`
//! as the table operations of a trace.
//!
//! Each one is recorded behind the guard on the callee's identity, and any
//! guard that fails leaves through the `Before` snapshot of the call, so the
//! interpreter makes the call itself and raises whatever error it raises.
//!
//! `rawget`, `rawset`, `rawlen` and `unpack` reuse the operations of a table
//! access (`t[k]`): an array load, a field load, an in-place store, a bound
//! length. They differ from the access only where the metatable matters. A
//! raw read finds a key the table holds itself and never an `__index` result,
//! so the field a read finds through `__index` is no answer. A raw store
//! never runs `__newindex`, which is what the in-place stores already do for
//! a key that is present.
//!
//! `getmetatable` and `setmetatable` call a helper (see
//! [`oximond::jit_ir::trace::HelperId::GETMETATABLE`]): native code never
//! holds a table's header, only its lane in the run's bank, so the VM reads
//! and writes the metatable word. `setmetatable` leaves the trace for a
//! protected metatable, a metatable with `__gc` or `__mode`, and a read-only
//! table, which the interpreter treats in ways a helper does not. A trace that
//! changes a metatable may not also hold a fact about one (see
//! [`oximond::jit_ir::trace::TraceIr::stale_metatable_fact`]).

use super::{
    BindingKind, Flow, HookPoint, Live, Number, Operand, RecKind, Recorder, Sym, TraceAbortReason,
    TraceScalar, TraceType, VirtualMeta, is_number,
};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::{ArithmeticOp, HelperId, SnapshotPhase, pinned_native_identity};
use crate::runtime::{
    DirectNativeKind, Metamethod, NativeFunctionHandle, Runtime, TableHandle, Value,
};

/// The most values one `unpack` may give a trace.
const MAX_UNPACK_VALUES: i64 = 8;

impl Recorder {
    /// Whether `kind` is a function of this module that takes its arguments
    /// as plain operands (`rawget` and `rawset` read their key and value from
    /// registers instead).
    pub(super) const fn is_table_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::RawLen
                | DirectNativeKind::GetMetatable
                | DirectNativeKind::SetMetatable
                | DirectNativeKind::Unpack
                | DirectNativeKind::TableUnpack
        )
    }

    /// Whether `kind` reads a key from a register and so takes its arguments
    /// as symbols, like `select` (a string constant is one).
    pub(super) const fn is_raw_access_kind(kind: DirectNativeKind) -> bool {
        matches!(kind, DirectNativeKind::RawGet | DirectNativeKind::RawSet)
    }

    /// A call of `rawget` or `rawset` with fixed arguments.
    pub(super) fn record_raw_access(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let setting = kind == DirectNativeKind::RawSet;
        if fixed != if setting { 3 } else { 2 } {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if setting {
            // A store is committed: a function trace has none, and a call
            // that starts over would repeat it.
            if self.function.is_some() {
                return Err(TraceAbortReason::SelfCall);
            }
            if self.reexec.is_some() {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        }
        let (table, table_ty) = self.read(at.registers, first_argument)?;
        if table_ty != TraceType::Table {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let receiver = self.table_receiver(at, first_argument)?;
        let key = self.table_key(at, first_argument + 1)?;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        if setting {
            self.record_set_keyed_as(at, first_argument + 2, key, receiver, true)?;
            let value = self.emit(pc, RecKind::Move(table), Some(TraceType::Table));
            return Ok(self.native_result(pc, value, (base, results)));
        }
        let value = self.read_keyed(at, key, receiver, true)?;
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// A call of `rawlen`, `getmetatable`, `setmetatable` or `unpack`. The
    /// arguments are the read argument registers, the first in register
    /// `first_argument`. A call that takes open results (`hoisted`) has no
    /// snapshot to leave through, so none of these can be one.
    #[allow(clippy::too_many_lines)]
    pub(super) fn record_table_native(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        if hoisted || arguments.iter().any(|(_, ty)| ty.is_wide()) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let pc = at.pc;
        match kind {
            DirectNativeKind::RawLen => {
                let &[(operand, ty)] = arguments else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                let number = self.number_type();
                let value = match ty {
                    TraceType::Table => {
                        let (receiver, handle, runtime) =
                            self.table_receiver(at, first_argument)?;
                        let window = Self::bound_window(receiver)?;
                        runtime
                            .table_length(handle)
                            .map_err(|_| TraceAbortReason::Divergence)?;
                        let lane = self.pin(pinned_native_identity(function.index()))?;
                        self.guard_callee(pc, callee, lane);
                        // The border whatever `__len` says.
                        let binding = self.binding(
                            window,
                            BindingKind::Length {
                                ty: number,
                                raw: true,
                            },
                        );
                        self.emit(
                            pc,
                            RecKind::TableLength {
                                binding,
                                table: Operand::Carried(window),
                            },
                            Some(number),
                        )
                    }
                    TraceType::String | TraceType::StringRef => {
                        let lane = self.pin(pinned_native_identity(function.index()))?;
                        self.guard_callee(pc, callee, lane);
                        // A string constant has a constant length.
                        if let (Sym::Root(_), Value::String(text)) =
                            (self.sym(first_argument), &at.registers[first_argument])
                        {
                            let length = i64::try_from(text.len())
                                .map_err(|_| TraceAbortReason::Divergence)?;
                            let value = self.number_scalar(pc, length);
                            return Ok(self.native_result(pc, value, (base, results)));
                        }
                        match self.bound_string(operand) {
                            Ok(window) if ty == TraceType::String => {
                                let binding =
                                    self.binding(window, BindingKind::Bytes { ty: number });
                                self.emit(
                                    pc,
                                    RecKind::StringLength {
                                        binding,
                                        string: Operand::Carried(window),
                                    },
                                    Some(number),
                                )
                            }
                            _ => self.emit_string_length(pc, operand, ty),
                        }
                    }
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                };
                Ok(self.native_result(pc, value, (base, results)))
            }
            DirectNativeKind::GetMetatable => {
                let &[(operand, ty)] = arguments else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                if !matches!(
                    ty,
                    TraceType::Nil
                        | TraceType::Boolean
                        | TraceType::Integer
                        | TraceType::Float
                        | TraceType::String
                        | TraceType::StringRef
                        | TraceType::Table
                ) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let runtime = at
                    .surroundings
                    .ok_or(TraceAbortReason::UnsupportedOperand)?
                    .runtime;
                let (helper, answer) = match runtime
                    .raw_metatable(&at.registers[first_argument])
                    .map_err(|_| TraceAbortReason::Divergence)?
                {
                    None => (HelperId::GETMETATABLE_ABSENT, TraceType::Nil),
                    // A `__metatable` field is the answer, and the helper
                    // does not give it.
                    Some(metatable) => {
                        if !runtime.metatable_lacks(metatable, Metamethod::Metatable) {
                            return Err(TraceAbortReason::UnsupportedOperand);
                        }
                        (HelperId::GETMETATABLE, TraceType::Table)
                    }
                };
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                self.note_type_read(operand);
                let value = self.emit_library_helper(pc, helper, &[(operand, ty)], answer);
                Ok(self.native_result(pc, value, (base, results)))
            }
            DirectNativeKind::SetMetatable => {
                let &[(table, TraceType::Table), (metatable, meta_ty)] = arguments else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                if !matches!(meta_ty, TraceType::Table | TraceType::Nil) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // A function trace stores nothing, and a call that starts
                // over would store twice.
                if self.function.is_some() {
                    return Err(TraceAbortReason::SelfCall);
                }
                if self.reexec.is_some() {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                Self::check_set_metatable(at, first_argument)?;
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                self.note_type_read(table);
                self.note_type_read(metatable);
                self.stores += 1;
                self.emit_helper_call(
                    pc,
                    HelperId::SETMETATABLE,
                    &[(table, TraceType::Table), (metatable, meta_ty)],
                );
                let value = self.emit(pc, RecKind::Move(table), Some(TraceType::Table));
                Ok(self.native_result(pc, value, (base, results)))
            }
            // Lua 5.1, 5.2 and `LuaJIT` take the border; 5.3 and later ask `__len`.
            DirectNativeKind::Unpack | DirectNativeKind::TableUnpack => self.record_unpack(
                at,
                (callee, function),
                (arguments, first_argument),
                (base, results),
                kind == DirectNativeKind::Unpack,
            ),
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// A call of `setmetatable (t, mt)` where `t` is a virtual table: the
    /// table stays virtual and its recipe carries `mt`, which the exit sets
    /// after it stores the contents. A helper checks `mt` each time the call
    /// runs (no `__gc`, no `__mode`, and none of the metamethods the
    /// recording did without), so the table behaves as a plain one for the
    /// accesses the recorder resolved. Anything else the table meets with a
    /// metatable that has `__index`, `__newindex` or `__len` makes it escape.
    pub(super) fn record_virtual_set_metatable(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        first_argument: usize,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let id = self
            .virtual_in(first_argument)
            .ok_or(TraceAbortReason::Divergence)?;
        // The call's result is the table, which a register holds.
        if results == 0
            || self.virtuals[id as usize].metatable.is_some()
            || !self.may_hold_virtual(base)
        {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let (metatable, ty) = self.read(at.registers, first_argument + 1)?;
        let Value::Table(live) = at.registers[first_argument + 1] else {
            return Err(TraceAbortReason::VirtualEscape);
        };
        if ty != TraceType::Table {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        if !runtime.metatable_lacks(live, Metamethod::Gc)
            || !runtime.metatable_lacks(live, Metamethod::Mode)
        {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let meta = VirtualMeta {
            table: metatable,
            no_index: runtime.metatable_lacks(live, Metamethod::Index),
            no_newindex: runtime.metatable_lacks(live, Metamethod::NewIndex),
            no_len: runtime.metatable_lacks(live, Metamethod::Len),
            relied: 0,
        };
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        self.note_type_read(metatable);
        // The mask is only a placeholder: the accesses that follow decide
        // which absences the call must check ([`VirtualMeta::mask`]).
        let mask = self.scalar(pc, TraceScalar::Integer(0));
        self.mask_ops.push((mask, id));
        let mask = Operand::Op(mask);
        self.emit_helper_call(
            pc,
            HelperId::PLAIN_METATABLE,
            &[(metatable, TraceType::Table), (mask, TraceType::Integer)],
        );
        self.virtuals[id as usize].metatable = Some(meta);
        self.write(base, Sym::Virtual(id));
        self.fill_nil(pc, base + 1..base + results);
        Ok(Flow::Next(pc + 1))
    }

    /// Refuses a `setmetatable` call the helper would leave to the
    /// interpreter now, so the recording is not one that always exits: a
    /// read-only table, a protected metatable, and a new metatable with a
    /// finalizer or a weak mode.
    fn check_set_metatable(
        at: &HookPoint<'_>,
        first_argument: usize,
    ) -> Result<(), TraceAbortReason> {
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let Value::Table(table) = at.registers[first_argument] else {
            return Err(TraceAbortReason::Divergence);
        };
        if runtime.table_is_readonly(table).unwrap_or(true) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if let Ok(Some(current)) = runtime.raw_metatable(&Value::Table(table))
            && !runtime.metatable_lacks(current, Metamethod::Metatable)
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        match at.registers.get(first_argument + 1) {
            Some(Value::Nil) => {}
            Some(Value::Table(metatable)) => {
                if !runtime.metatable_lacks(*metatable, Metamethod::Gc)
                    || !runtime.metatable_lacks(*metatable, Metamethod::Mode)
                {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            }
            _ => return Err(TraceAbortReason::Divergence),
        }
        Ok(())
    }

    /// Guards that the number `operand` equals `value`, unless it is that
    /// constant already.
    fn guard_number_is(&mut self, pc: usize, operand: Operand, value: i64) {
        let constant = matches!(
            self.origin(operand),
            Operand::Op(op) if matches!(self.ops[op as usize].kind, RecKind::Constant(_))
        );
        if constant {
            return;
        }
        let expected = Operand::Op(self.number_scalar(pc, value));
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, operand, expected),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(same), true);
    }

    /// A call of `unpack` or `table.unpack` whose result count the recording
    /// fixes: the range `i..j`, with `i` defaulting to 1 and `j` to the
    /// table's length. The length (or the two bounds) is guarded against the
    /// one seen, and each element is an array read, so a table that has
    /// since changed length leaves through the `Before` snapshot of the
    /// call. Elements are the raw values at those keys, as the library reads
    /// them (a table with `__index` is read raw where the slot holds a
    /// value).
    fn record_unpack(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        raw: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if arguments.is_empty() || arguments.len() > 3 || arguments[0].1 != TraceType::Table {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let (receiver, handle, runtime) = self.table_receiver(at, first_argument)?;
        let window = Self::bound_window(receiver)?;
        let number = self.number_type();
        let border = if raw {
            runtime.table_length(handle).ok()
        } else {
            runtime
                .table_length_without_len_metamethod(handle)
                .ok()
                .flatten()
        }
        .ok_or(TraceAbortReason::UnsupportedOperand)?;
        #[allow(clippy::cast_possible_truncation, reason = "a border is a small count")]
        let border = border as i64;
        let bound = |index: usize| -> Result<Option<(Operand, i64)>, TraceAbortReason> {
            match arguments.get(index) {
                None => Ok(None),
                Some((operand, ty)) if is_number(*ty) => {
                    let live = Self::live_integer(&at.registers[first_argument + index])
                        .ok_or(TraceAbortReason::UnsupportedOperand)?;
                    Ok(Some((*operand, live)))
                }
                Some(_) => Err(TraceAbortReason::UnsupportedOperand),
            }
        };
        let first = bound(1)?;
        let last = bound(2)?;
        let start = first.map_or(1, |(_, live)| live);
        let end = last.map_or(border, |(_, live)| live);
        let count = (end - start + 1).max(0);
        if count > MAX_UNPACK_VALUES || start < -(1 << 40) || end > 1 << 40 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        if let Some((operand, live)) = first {
            self.guard_number_is(pc, operand, live);
        }
        if let Some((operand, live)) = last {
            self.guard_number_is(pc, operand, live);
        } else {
            let length_binding = self.binding(window, BindingKind::Length { ty: number, raw });
            let length = self.emit(
                pc,
                RecKind::TableLength {
                    binding: length_binding,
                    table: Operand::Carried(window),
                },
                Some(number),
            );
            self.guard_number_is(pc, Operand::Op(length), border);
        }
        let mut values = Vec::new();
        for key in start..start + count {
            let operand = Operand::Op(self.number_scalar(pc, key));
            #[allow(clippy::cast_precision_loss, reason = "the keys are small")]
            let live = if self.integers {
                Live::Integer(key)
            } else {
                Live::Float(key as f64)
            };
            let (value, _) = self.array_read(
                pc,
                (Operand::Carried(window), window),
                (handle, runtime),
                Number { operand, live },
                true,
            )?;
            values.push(value);
        }
        Ok(self.native_results(pc, &values, (base, results)))
    }
}

/// The place a number argument names, as the helpers read it (see
/// `position_of` in the VM's helpers): the integer, or the float truncated.
fn live_position(value: &Value) -> Option<i64> {
    match value {
        Value::Integer(place) => Some(*place),
        #[allow(clippy::cast_possible_truncation, reason = "bounded just before")]
        Value::Number(place) if place.is_finite() && place.abs() < 9.0e18 => {
            Some(place.trunc() as i64)
        }
        _ => None,
    }
}

impl Recorder {
    /// Whether `kind` writes the array part of a table: `table.insert` and
    /// `table.remove`, in either form.
    pub(super) const fn is_table_write_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::TableInsert
                | DirectNativeKind::TableInsertRaw
                | DirectNativeKind::TableRemove
                | DirectNativeKind::TableRemoveRaw
        )
    }

    /// A call of `table.insert` or `table.remove` on a table that is a dense
    /// list, with the elements after the place moving up or down in the
    /// helper (see [`HelperId::TABLE_INSERT`]).
    ///
    /// The recording sees the table as the interpreter is about to leave it,
    /// so the position, the element a remove takes out and its type are
    /// known. A later run that finds something else (another type, a table
    /// that gained a hash part, a hole or a metamethod the profile honours,
    /// a place out of range) misses in the helper, having changed nothing,
    /// and the interpreter makes the call.
    #[allow(clippy::too_many_lines)]
    pub(super) fn record_table_write(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        if hoisted || arguments.iter().any(|(_, ty)| ty.is_wide()) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A store is committed: a function trace has none, and a call that
        // starts over would repeat it.
        if self.function.is_some() {
            return Err(TraceAbortReason::SelfCall);
        }
        // The guards of a call that takes open results leave before the call
        // that made them, which nothing has stored since (see
        // `Recorder::open_region_for`): its own store comes after them all.
        if self.reexec.is_some() && !self.open_region {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let pc = at.pc;
        let inserting = matches!(
            kind,
            DirectNativeKind::TableInsert | DirectNativeKind::TableInsertRaw
        );
        let honours = matches!(
            kind,
            DirectNativeKind::TableInsert | DirectNativeKind::TableRemove
        );
        let Some((&(_, TraceType::Table), rest)) = arguments.split_first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let Value::Table(handle) = at.registers[first_argument] else {
            return Err(TraceAbortReason::Divergence);
        };
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        if let Some(flow) = self.record_table_end(
            at,
            inserting,
            (callee, function),
            (arguments, first_argument),
            (base, results),
        )? {
            return Ok(flow);
        }
        let live = |index: usize| &at.registers[first_argument + index];
        let numeric = |ty: TraceType| matches!(ty, TraceType::Integer | TraceType::Float);
        let (callee_lane, helper, result) = if inserting {
            let (position, value) = match rest {
                [value] => (None, value),
                [position, value] if numeric(position.1) => (Some(position), value),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            };
            if !matches!(
                value.1,
                TraceType::Integer
                    | TraceType::Float
                    | TraceType::Boolean
                    | TraceType::String
                    | TraceType::StringRef
                    | TraceType::Table
            ) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            // The call must succeed now, or the trace would only leave.
            let place = if position.is_some() {
                let place = live_position(live(1)).ok_or(TraceAbortReason::UnsupportedOperand)?;
                Some(usize::try_from(place).map_err(|_| TraceAbortReason::UnsupportedOperand)?)
            } else {
                None
            };
            if !runtime.table_trace_insertable(handle, place, honours) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            (
                self.pin(pinned_native_identity(function.index()))?,
                HelperId::TABLE_INSERT,
                None,
            )
        } else {
            let position = match rest {
                [] => None,
                [position] if numeric(position.1) => Some(position),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            };
            let place = match position {
                None => None,
                Some(_) => {
                    Some(live_position(live(1)).ok_or(TraceAbortReason::UnsupportedOperand)?)
                }
            };
            let nothing = runtime.table_trace_remove_nothing(handle, place, honours);
            let (helper, result) = if nothing {
                (HelperId::TABLE_REMOVE_NOTHING, TraceType::Nil)
            } else {
                let length = runtime
                    .table_trace_list_length(handle, honours)
                    .ok_or(TraceAbortReason::UnsupportedOperand)?;
                let length = i64::try_from(length).map_err(|_| TraceAbortReason::Divergence)?;
                let place = usize::try_from(place.unwrap_or(length))
                    .map_err(|_| TraceAbortReason::Divergence)?;
                let element = runtime
                    .table_raw_array_range(handle, place, place)
                    .ok()
                    .flatten()
                    .and_then(<[Value]>::first)
                    .ok_or(TraceAbortReason::Divergence)?;
                match element {
                    Value::Integer(_) => (HelperId::TABLE_REMOVE_INTEGER, TraceType::Integer),
                    Value::Number(_) => (HelperId::TABLE_REMOVE_FLOAT, TraceType::Float),
                    Value::Boolean(_) => (HelperId::TABLE_REMOVE_BOOLEAN, TraceType::Boolean),
                    Value::Table(_) => (HelperId::TABLE_REMOVE_TABLE, TraceType::Table),
                    Value::String(_) => (HelperId::TABLE_REMOVE_STRING, TraceType::String),
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                }
            };
            (
                self.pin(pinned_native_identity(function.index()))?,
                helper,
                Some(result),
            )
        };
        self.guard_callee(pc, callee, callee_lane);
        for (operand, _) in arguments {
            self.note_type_read(*operand);
        }
        if helper != HelperId::TABLE_REMOVE_NOTHING {
            self.stores += 1;
        }
        let call = self.emit_helper_call(pc, helper, arguments);
        if inserting {
            return Ok(self.no_values(pc, (base, results)));
        }
        // A remove that finds nothing gives nil.
        let value = if result == Some(TraceType::Nil) {
            self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            )
        } else {
            call
        };
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// `table.insert (t, value)` and `table.remove (t)` on a table that is a
    /// dense list native code can append to and pop from in place (see
    /// `oximond::runtime::NativeTableHeader`), recorded as the set they stand
    /// for: an insert stores `value` under the key `#t + 1`, a remove reads
    /// the last slot and sets it to nil. Nothing but the table's own array
    /// changes, so the sets run in place, with no call into the VM, while
    /// the table stays a dense list; when it stops being one (a hash part, a
    /// metamethod, a hole) or the call would do anything else, the guard
    /// leaves and the interpreter makes the call.
    ///
    /// `None` leaves the call to the helpers: a table that is not bound to a
    /// register of the trace, one that is not appendable, a position, a value
    /// that is not a number, boolean or string, an element that is not a
    /// number or boolean, or an empty table.
    fn record_table_end(
        &mut self,
        at: &HookPoint<'_>,
        inserting: bool,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let value = match (inserting, arguments) {
            (true, [_, value])
                if matches!(
                    value.1,
                    TraceType::Integer
                        | TraceType::Float
                        | TraceType::Boolean
                        | TraceType::String
                        | TraceType::Table
                ) =>
            {
                Some(value.0)
            }
            (false, [_]) => None,
            _ => return Ok(None),
        };
        let Ok((receiver, handle, runtime)) = self.table_receiver(at, first_argument) else {
            return Ok(None);
        };
        let Ok(window) = Self::bound_window(receiver) else {
            return Ok(None);
        };
        if self.table_is_aliased(at.registers, window, handle)
            || !runtime
                .table_native_header(handle)
                .is_ok_and(|header| header.appendable)
        {
            return Ok(None);
        }
        let length = runtime
            .table_array_len(handle)
            .map_err(|_| TraceAbortReason::Divergence)?;
        let element = if inserting {
            None
        } else {
            let Some(element) = length.checked_sub(1).and_then(|last| {
                runtime
                    .table_raw_array_range(handle, last + 1, last + 1)
                    .ok()
                    .flatten()
                    .and_then(<[Value]>::first)
            }) else {
                return Ok(None);
            };
            if !matches!(
                element,
                Value::Integer(_) | Value::Number(_) | Value::Boolean(_)
            ) {
                return Ok(None);
            }
            Some(element)
        };
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        for (operand, _) in arguments {
            self.note_type_read(*operand);
        }
        self.stores += 1;
        let number = self.number_type();
        let length_binding = self.binding(
            window,
            BindingKind::Length {
                ty: number,
                raw: false,
            },
        );
        let border = self.emit(
            pc,
            RecKind::TableLength {
                binding: length_binding,
                table: Operand::Carried(window),
            },
            Some(number),
        );
        // An append fills a table whatever it holds: its first slot says
        // nothing about the version to run.
        let first = if value.is_some() {
            None
        } else {
            self.first_slot_type(runtime, handle)
        };
        let array = self.binding(window, BindingKind::Array { first });
        self.binding(window, BindingKind::Header { strict: true });
        let ends = (array, Operand::Carried(window), border);
        if let Some(value) = value {
            return Ok(Some(self.record_append(pc, ends, value, (base, results))));
        }
        let element = element.expect("a remove found its element");
        let at_the_end = ((window, handle, runtime), length, element);
        self.record_pop(pc, ends, at_the_end, (base, results))
            .map(Some)
    }

    /// The set under the key `#t + 1` that `table.insert (t, value)` stands
    /// for. `border` is the operation that read `#t`.
    fn record_append(
        &mut self,
        pc: usize,
        (array, table, border): (u32, Operand, u32),
        value: Operand,
        (base, results): (usize, usize),
    ) -> Flow {
        let number = self.number_type();
        let one = self.number_scalar(pc, 1);
        let key = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, Operand::Op(border), Operand::Op(one)),
            Some(number),
        );
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::ArrayStore {
                binding: array,
                table,
                key: Operand::Op(key),
                value,
                snapshot,
                set: true,
            },
            None,
        );
        self.no_values(pc, (base, results))
    }

    /// The read of the last slot and the set of nil under its key that
    /// `table.remove (t)` stands for, of a list `length` slots long whose
    /// last slot holds `element`.
    fn record_pop(
        &mut self,
        pc: usize,
        (array, table, border): (u32, Operand, u32),
        ((window, handle, runtime), length, element): (
            (usize, TableHandle, &Runtime),
            usize,
            &Value,
        ),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        #[allow(clippy::cast_possible_wrap, reason = "a length is a small count")]
        let last = length as i64;
        #[allow(clippy::cast_precision_loss, reason = "a length is a small count")]
        let live = if self.integers {
            Live::Integer(last)
        } else {
            Live::Float(last as f64)
        };
        let (taken, ty) = self.array_read(
            pc,
            (table, window),
            (handle, runtime),
            Number {
                operand: Operand::Op(border),
                live,
            },
            false,
        )?;
        debug_assert_eq!(self.slot_type(element), Some(ty));
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::ArrayStore {
                binding: array,
                table,
                key: Operand::Op(border),
                value: Operand::Op(nil),
                snapshot,
                set: true,
            },
            None,
        );
        Ok(self.native_result(pc, taken, (base, results)))
    }

    /// The end of a call that gives no value (`table.insert`): the open
    /// results of the call are none, and a register it is to fill gets nil.
    fn no_values(&mut self, pc: usize, (base, results): (usize, usize)) -> Flow {
        if results == 0 {
            self.open = Some(Vec::new());
            return Flow::Next(pc + 1);
        }
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        for register in base..base + results {
            self.write(register, nil);
        }
        Flow::Next(pc + 1)
    }
}
