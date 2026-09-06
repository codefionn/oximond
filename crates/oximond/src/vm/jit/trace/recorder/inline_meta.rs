//! Function metamethods of table accesses (`__index`, `__newindex`, `__len`)
//! as inlined Lua calls.
//!
//! A read of a key the table lacks, a write to one, and `#t` run the handler
//! the metatable names. When that handler is a Lua closure the interpreter
//! pushes a frame for it, and the recorder follows it into the closure like
//! any other inlined call (see `record_call_inner`): the same pinned lane,
//! the same frame depth limits, the same rebuilt frames at an exit. What
//! differs is how the frame is entered and left.
//!
//! - A helper stands where a call has its callee guard
//!   ([`HelperId::INDEX_HANDLER`], [`HelperId::NEWINDEX_HANDLER`],
//!   [`HelperId::LEN_HANDLER`]). It makes the interpreter's walk of the
//!   metatables at run time: the key is still absent from every table of the
//!   chain, and the raw handler at its end is the pinned closure. The check is
//!   about the class, not the object, so one trace serves every object that
//!   shares a metatable. Any other table, key present, metatable or handler
//!   leaves through the `Before` snapshot of the instruction, and the
//!   interpreter makes the access (and a side trace may grow there).
//! - A handler of the receiver's own metatable (`__index` or `__newindex`
//!   named by it, with no table in between) needs no walk at run time. The
//!   trace pins the metatable as a class ([`BindingKind::MetaClass`]): entry
//!   checks that the metatable is alive and that its raw `__index` or
//!   `__newindex` is the pinned closure, and refuses the run otherwise. Each
//!   access then holds the receiver to that metatable with
//!   [`TraceOp::MetaGuard`], one load of the metatable word of the receiver's
//!   row and one compare, and to a key it lacks with a native lookup that
//!   finds nil. A store under a key that names an event misses in the store
//!   helpers, and a `setmetatable` makes the row again, so nothing in a run
//!   changes what the guard stands for. The closure gets the receiver itself.
//!   The helper stays for a handler behind a table, a receiver the entry does
//!   not bind, a key that is no string and a function trace.
//! - The closure gets the arguments the interpreter passes: the table the
//!   walk ended at (which is the receiver unless `__index` or `__newindex`
//!   went through tables first) and the key, and for a write the value. `#t`
//!   passes the table twice. The helper gives the lane of the table the walk
//!   ended at.
//! - A read returns the first value of the closure (nil if it returns none)
//!   into the register the instruction writes, `#t` too. A write discards the
//!   values. A rebuilt frame returns to the same place (see
//!   [`oximond::jit_ir::trace::Entry`]).
//!
//! The profile decides which accesses call a handler: Lua 5.1 and `LuaJIT`
//! ignore `__len` for a table. A handler that is not a Lua closure (a table
//! is walked, anything else is the interpreter's) leaves the access to the
//! code that records accesses without one.
//!
//! A global the environment lacks is such a read (`GetTabUp` of `_ENV`,
//! `GetGlobal` in Lua 5.1 and `LuaJIT`), and a new global such a write, with
//! the environment as the table. A handler that is a native or builtin
//! function, or a metamethod the recorder cannot follow otherwise, aborts the
//! recording with the function's name, or the metamethod's, among the native
//! aborts of the statistics.
//!
//! A store under a string key into a table the trace does not bind is the
//! helper [`HelperId::TABLE_SET_STRING`]. It walks `__newindex` the way a
//! write does, and stores in the table it ends at, so a handler that fills a
//! backing table (`backing[key] = value`) and a table-valued `__newindex` are
//! both one call.

use super::{
    BindingKind, Entry, Flow, HelperId, HookPoint, Inlined, MAX_TRACE_INLINED_CALLS, Operand,
    Pending, PrevPc, RecKind, Receiver, Recorder, Sym, TableKey, TraceAbortReason, TraceFrameInfo,
    TraceScalar, TraceType,
};
use crate::bytecode::{Constant, Instruction, OpCode};
use crate::jit_ir::trace::{
    InlinedFrame, MAX_TRACE_FRAME_DEPTH, META_FIELD_KEY, META_PIN_ABSENT, MetaPin, Protected,
    SnapshotPhase,
};
use crate::runtime::{ClosureHandle, Runtime, TableHandle, Value};
use crate::vm::arith::{metamethod_key, newindex_metamethod_key};
use crate::vm::jit::trace::helpers::{Walk, walk_handlers};
use std::sync::Arc;

/// What a metamethod frame needs beyond the register it writes.
#[derive(Debug, Default)]
pub(super) struct Extra {
    pub(super) after: Pending,
    /// The results the frame keeps, when not one (or none for a write).
    pub(super) results: Option<usize>,
    /// The instruction the handler runs for, when the recorder is at the
    /// return of an earlier handler of it.
    pub(super) pc: Option<usize>,
}

impl Recorder {
    /// Records the instruction at `at.pc` as a call of a function metamethod,
    /// if it runs one: the flow of the entered callee. `None` leaves the
    /// instruction to the code that records accesses the interpreter makes
    /// without a handler.
    pub(super) fn record_metamethod(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        match instruction.opcode() {
            Ok(OpCode::GetTable | OpCode::GetField | OpCode::SelfField) => {
                self.record_index_handler(at, instruction)
            }
            Ok(OpCode::SetTable | OpCode::SetField) => {
                self.record_newindex_handler(at, instruction)
            }
            Ok(OpCode::Length) => self.record_len_handler(at, instruction),
            Ok(opcode @ (OpCode::GetTabUp | OpCode::GetGlobal)) => {
                self.record_global_index_handler(at, instruction, opcode == OpCode::GetTabUp)
            }
            Ok(opcode @ (OpCode::SetTabUp | OpCode::SetGlobal)) => {
                self.record_global_newindex_handler(at, instruction, opcode == OpCode::SetTabUp)
            }
            _ => self.record_operator_metamethod(at, instruction),
        }
    }

    /// The table and the Lua closure a walk of metatables ended at. A walk
    /// that ends at a native or builtin function names it for the statistics
    /// of what aborts the recording: nothing here records such a handler,
    /// and the access that follows cannot be recorded without one.
    fn closure_handler(
        &mut self,
        walk: Walk,
        start: crate::runtime::TableHandle,
        event: &'static str,
    ) -> Option<(crate::runtime::TableHandle, ClosureHandle)> {
        match walk {
            Walk::Closure(holder, handler) => {
                // Named in case the call cannot be entered.
                self.abort_metamethod = Some(event);
                Some((holder, handler))
            }
            Walk::Native(function) => {
                self.note_callable(&function);
                None
            }
            Walk::Other => {
                self.abort_metamethod = Some(event);
                None
            }
            // A chain of tables that ends without a handler is the access
            // of the tables, which the code for them records or refuses.
            Walk::Raw(end) => {
                if end != start {
                    self.abort_metamethod = Some(event);
                }
                None
            }
            Walk::Holds => None,
        }
    }

    /// The key the access at `at.pc` indexes with, as the interpreter holds
    /// it: the constant a field instruction names or the register `register`.
    fn live_key(at: &HookPoint<'_>, opcode: OpCode, operand: u8) -> Option<Value> {
        if matches!(opcode, OpCode::GetTable | OpCode::SetTable) {
            return at.registers.get(usize::from(operand)).cloned();
        }
        match at.chunk.constants.get(usize::from(operand)) {
            Some(Constant::String(text)) => Some(Value::String(text.clone())),
            _ => None,
        }
    }

    /// The operand and type of the key of an access, as a helper takes it.
    fn handler_key(
        &mut self,
        pc: usize,
        key: TableKey,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        Ok(match key {
            TableKey::Field(text) => {
                let slot = self.root_slot(text.as_bytes())?;
                self.root_operand(pc, slot)
            }
            TableKey::Hash(operand, _) | TableKey::Object(operand, _) => {
                (operand, self.operand_type(operand))
            }
            TableKey::Index(number) => (number.operand, number.ty()),
        })
    }

    /// A read (`GetTable`, `GetField`, `SelfField`) that finds the key absent
    /// and runs a Lua closure as `__index`.
    fn record_index_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let opcode = instruction
            .opcode()
            .map_err(|_| TraceAbortReason::UnsupportedOpcode)?;
        let (a, b, c) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            instruction.c(),
        );
        let Some(&Value::Table(handle)) = at.registers.get(b) else {
            return Ok(None);
        };
        if self.virtual_in(b).is_some() {
            return Ok(None);
        }
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(None);
        };
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(Some(_))) {
            return Ok(None);
        }
        let Some(live) = Self::live_key(at, opcode, c) else {
            return Ok(None);
        };
        let walk = walk_handlers(runtime, handle, &live, runtime.index_metamethod_key());
        let Some((holder, handler)) = self.closure_handler(walk, handle, "__index") else {
            return Ok(None);
        };
        self.check_register(a)?;
        let key = if opcode == OpCode::GetTable {
            self.table_key(at, usize::from(c))?
        } else {
            Self::field_key(at, c)?
        };
        if opcode == OpCode::SelfField {
            // The receiver is copied before the lookup, so a handler that
            // leaves through an exit finds the copy in place.
            self.record_move(at.registers, pc, a + 1, b)?;
        }
        let (table, table_ty) = self.read(at.registers, b)?;
        if table_ty != TraceType::Table {
            return Err(TraceAbortReason::Divergence);
        }
        let (key, key_ty) = self.handler_key(pc, key)?;
        let identity = handler.native_identity();
        // A handler of the receiver's own metatable is checked by the class
        // guard. The handler gets the receiver, whose lane the trace has.
        let first = if holder == handle
            && self.class_access(at, (table, handle), (key, key_ty), ("__index", handler))?
        {
            table
        } else {
            let expected = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
            let walked = self.emit_helper_call(
                pc,
                HelperId::INDEX_HANDLER,
                &[
                    (table, TraceType::Table),
                    (key, key_ty),
                    (Operand::Op(expected), TraceType::Integer),
                ],
            );
            // The table the walk ended at, whatever the chain a run-time
            // object takes to the pinned closure: the helper gives its lane.
            Operand::Op(walked)
        };
        self.enter_metamethod(
            at,
            handler,
            &[(first, TraceType::Table), (key, key_ty)],
            (Entry::Index, a),
        )
        .map(Some)
    }

    /// A write (`SetTable`, `SetField`) that finds the key absent and runs a
    /// Lua closure as `__newindex`.
    fn record_newindex_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        // A function trace stores nothing, and a region that starts the call
        // over would repeat the write: those are the plain write's refusal.
        if self.function.is_some() || self.reexec.is_some() {
            return Ok(None);
        }
        let opcode = instruction
            .opcode()
            .map_err(|_| TraceAbortReason::UnsupportedOpcode)?;
        let (a, b, c) = (
            usize::from(instruction.a()),
            instruction.b(),
            usize::from(instruction.c()),
        );
        let Some(&Value::Table(handle)) = at.registers.get(a) else {
            return Ok(None);
        };
        if self.virtual_in(a).is_some() {
            return Ok(None);
        }
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(None);
        };
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(Some(_))) {
            return Ok(None);
        }
        let Some(live) = Self::live_key(at, opcode, b) else {
            return Ok(None);
        };
        let walk = walk_handlers(runtime, handle, &live, newindex_metamethod_key());
        let Some((holder, handler)) = self.closure_handler(walk, handle, "__newindex") else {
            return Ok(None);
        };
        let key = if opcode == OpCode::SetTable {
            self.table_key(at, usize::from(b))?
        } else {
            Self::field_key(at, b)?
        };
        let (table, table_ty) = self.read(at.registers, a)?;
        if table_ty != TraceType::Table {
            return Err(TraceAbortReason::Divergence);
        }
        let (value, value_ty) = self.host_operand(pc, at.registers, c)?;
        let (key, key_ty) = self.handler_key(pc, key)?;
        let identity = handler.native_identity();
        let first = if holder == handle
            && self.class_access(at, (table, handle), (key, key_ty), ("__newindex", handler))?
        {
            table
        } else {
            let expected = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
            let walked = self.emit_helper_call(
                pc,
                HelperId::NEWINDEX_HANDLER,
                &[
                    (table, TraceType::Table),
                    (key, key_ty),
                    (Operand::Op(expected), TraceType::Integer),
                ],
            );
            // The table the walk ended at, whatever the chain a run-time
            // object takes to the pinned closure: the helper gives its lane.
            Operand::Op(walked)
        };
        self.enter_metamethod(
            at,
            handler,
            &[(first, TraceType::Table), (key, key_ty), (value, value_ty)],
            (Entry::NewIndex, 0),
        )
        .map(Some)
    }

    /// The table and the name a global access names: for `GetTabUp` and
    /// `SetTabUp` the table in the upvalue and the constant `constant`, for
    /// `GetGlobal` and `SetGlobal` the environment of the root frame and the
    /// constant `constant`. `None` where the access is not one the recorder
    /// follows this way (a callee has an environment of its own).
    fn global_table(
        &self,
        at: &HookPoint<'_>,
        upvalue: Option<u8>,
        constant: u32,
    ) -> Option<(crate::runtime::TableHandle, crate::string::LuaString)> {
        let surroundings = at.surroundings?;
        let name = self.global_name(at, constant).ok()?;
        let handle = match upvalue {
            Some(upvalue) => {
                let (cell, runtime) = Self::upvalue_cell(at, u16::from(upvalue)).ok()?;
                match runtime.cell_get(cell) {
                    Ok(Value::Table(handle)) => handle,
                    _ => return None,
                }
            }
            None if self.in_home() => surroundings.environment,
            None => return None,
        };
        Some((handle, name))
    }

    /// A read of a global the environment lacks, which its metatable's
    /// function `__index` answers (a strict-mode `_ENV`). The same as
    /// [`Self::record_index_handler`] with the environment, or the table in
    /// an upvalue, as the table and the name as the key.
    fn record_global_index_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        tab_up: bool,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        // A side's callee has the cells of the root's to name.
        if self.side.is_some() && !self.frames.is_empty() {
            return Ok(None);
        }
        let a = usize::from(instruction.a());
        let (upvalue, constant) = if tab_up {
            (Some(instruction.b()), u32::from(instruction.c()))
        } else {
            (None, u32::from(instruction.d()))
        };
        let Some((handle, name)) = self.global_table(at, upvalue, constant) else {
            return Ok(None);
        };
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(None);
        };
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(Some(_))) {
            return Ok(None);
        }
        let walk = walk_handlers(
            runtime,
            handle,
            &Value::String(name.clone()),
            runtime.index_metamethod_key(),
        );
        let Some((_, handler)) = self.closure_handler(walk, handle, "__index") else {
            return Ok(None);
        };
        self.check_register(a)?;
        let (receiver, _, _) = match upvalue {
            Some(upvalue) => self.tab_up_receiver(at, u16::from(upvalue))?,
            None => self.environment_receiver(at)?,
        };
        let table = Self::receiver_operand(receiver);
        let slot = self.root_slot(name.as_bytes())?;
        let (key, key_ty) = self.root_operand(pc, slot);
        let identity = handler.native_identity();
        let expected = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
        let walked = self.emit_helper_call(
            pc,
            HelperId::INDEX_HANDLER,
            &[
                (table, TraceType::Table),
                (key, key_ty),
                (Operand::Op(expected), TraceType::Integer),
            ],
        );
        // The table the walk ended at, whatever the chain a run-time object
        // takes to the pinned closure: the helper gives its lane.
        let first = Operand::Op(walked);
        self.enter_metamethod(
            at,
            handler,
            &[(first, TraceType::Table), (key, key_ty)],
            (Entry::Index, a),
        )
        .map(Some)
    }

    /// A write of a global the environment lacks, which its metatable's
    /// function `__newindex` takes.
    fn record_global_newindex_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        tab_up: bool,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        if self.function.is_some()
            || self.reexec.is_some()
            || (self.side.is_some() && !self.frames.is_empty())
        {
            return Ok(None);
        }
        let (upvalue, constant, source) = if tab_up {
            (
                Some(instruction.a()),
                u32::from(instruction.b()),
                usize::from(instruction.c()),
            )
        } else {
            (
                None,
                u32::from(instruction.d()),
                usize::from(instruction.a()),
            )
        };
        let Some((handle, name)) = self.global_table(at, upvalue, constant) else {
            return Ok(None);
        };
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(None);
        };
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(Some(_))) {
            return Ok(None);
        }
        let walk = walk_handlers(
            runtime,
            handle,
            &Value::String(name.clone()),
            newindex_metamethod_key(),
        );
        let Some((_, handler)) = self.closure_handler(walk, handle, "__newindex") else {
            return Ok(None);
        };
        // A slot holds the value the tree read at entry, which this store
        // would leave stale, as for a store `__newindex` does not take.
        if upvalue.is_none() && self.holds_global_slot(at, &name) {
            return Ok(None);
        }
        let (receiver, _, _) = match upvalue {
            Some(upvalue) => self.tab_up_receiver(at, u16::from(upvalue))?,
            None => self.environment_receiver(at)?,
        };
        let table = Self::receiver_operand(receiver);
        let (value, value_ty) = self.host_operand(pc, at.registers, source)?;
        let slot = self.root_slot(name.as_bytes())?;
        let (key, key_ty) = self.root_operand(pc, slot);
        let identity = handler.native_identity();
        let expected = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
        let walked = self.emit_helper_call(
            pc,
            HelperId::NEWINDEX_HANDLER,
            &[
                (table, TraceType::Table),
                (key, key_ty),
                (Operand::Op(expected), TraceType::Integer),
            ],
        );
        // The table the walk ended at, whatever the chain a run-time object
        // takes to the pinned closure: the helper gives its lane.
        let first = Operand::Op(walked);
        self.enter_metamethod(
            at,
            handler,
            &[(first, TraceType::Table), (key, key_ty), (value, value_ty)],
            (Entry::NewIndex, 0),
        )
        .map(Some)
    }

    /// `#t` of a table with a Lua closure as `__len`, in a profile that asks
    /// for it.
    fn record_len_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        if !self.honours_len {
            return Ok(None);
        }
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        let Some(&Value::Table(handle)) = at.registers.get(b) else {
            return Ok(None);
        };
        if self.virtual_in(b).is_some() {
            return Ok(None);
        }
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(None);
        };
        let Ok(Some(metatable)) = runtime.raw_metatable(&Value::Table(handle)) else {
            return Ok(None);
        };
        let handler = runtime.table_get_string_ref(
            metatable,
            metamethod_key("__len").expect("__len is an operator metamethod"),
        );
        let Some(&Value::Closure(handler)) = handler else {
            // A handler the recorder does not follow: the interpreter's.
            match handler {
                Some(function @ (Value::NativeFunction(_) | Value::BuiltinFunction(_))) => {
                    self.note_callable(function);
                }
                Some(_) => self.abort_metamethod = Some("__len"),
                None => {}
            }
            return Ok(None);
        };
        self.abort_metamethod = Some("__len");
        self.check_register(a)?;
        let (table, table_ty) = self.read(at.registers, b)?;
        if table_ty != TraceType::Table {
            return Err(TraceAbortReason::Divergence);
        }
        let identity = handler.native_identity();
        let expected = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
        self.emit_helper_call(
            pc,
            HelperId::LEN_HANDLER,
            &[
                (table, TraceType::Table),
                (Operand::Op(expected), TraceType::Integer),
            ],
        );
        // The interpreter passes the operand as both arguments.
        self.enter_metamethod(
            at,
            handler,
            &[(table, TraceType::Table), (table, TraceType::Table)],
            (Entry::Len, a),
        )
        .map(Some)
    }

    /// Enters `handler`, the Lua closure the instruction at `at.pc` runs as a
    /// metamethod, with `arguments`. A read or `#t` writes the first value
    /// the closure returns into register `base`; a write drops them.
    ///
    /// The frame is built as for a call of the closure (see
    /// `record_call_inner`). The check that the handler is the right one was
    /// emitted before, and leaves through the instruction's `Before`
    /// snapshot.
    pub(super) fn enter_metamethod(
        &mut self,
        at: &HookPoint<'_>,
        handler: ClosureHandle,
        arguments: &[(Operand, TraceType)],
        (entry, base): (Entry, usize),
    ) -> Result<Flow, TraceAbortReason> {
        self.enter_metamethod_with(at, handler, arguments, (entry, base), Extra::default())
    }

    /// [`Self::enter_metamethod`] for a frame that returns more than a value
    /// into a register: the number of results it keeps, what its return does
    /// afterwards, and whether the instruction starts over when anything in
    /// the call leaves the trace.
    #[allow(clippy::too_many_lines)]
    pub(super) fn enter_metamethod_with(
        &mut self,
        at: &HookPoint<'_>,
        handler: ClosureHandle,
        arguments: &[(Operand, TraceType)],
        (entry, base): (Entry, usize),
        extra: Extra,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = extra.pc.unwrap_or(at.pc);
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let chunk = runtime
            .closure_call_parts(handler)
            .map(|parts| Arc::clone(parts.chunk))
            .ok_or(TraceAbortReason::Divergence)?;
        if chunk.arg_register.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A handler already on the stack of inlined calls is recursion, which
        // the trace has no way to bound.
        if Arc::ptr_eq(&chunk, &self.chunk)
            || self
                .activation
                .as_ref()
                .is_some_and(|activation| Arc::ptr_eq(&activation.chunk, &chunk))
            || self
                .frames
                .iter()
                .any(|frame| Arc::ptr_eq(&frame.chunk, &chunk))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // `MAX_TRACE_FRAME_DEPTH` counts the root frame.
        let outer = self.frames.len();
        if outer + 2 > MAX_TRACE_FRAME_DEPTH || self.inlined.len() >= MAX_TRACE_INLINED_CALLS {
            return Err(super::limit("inlined calls/depth"));
        }
        let lane = self.pin(handler.native_identity())?;
        // A handler with varargs starts a region that re-executes the access.
        let opens_region = chunk.is_vararg && self.reexec.is_none();
        if self.frames.is_empty() || opens_region {
            let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            if self.frames.is_empty() {
                self.call_site = Some((before, self.stores));
                self.call_ops = self.ops.len();
            }
            if opens_region {
                self.reexec = Some(before);
            }
        }
        let register_count = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(register_count);
        let passed = parameters.min(arguments.len());
        let mut regs = Vec::with_capacity(register_count);
        for (source, ty) in &arguments[..passed] {
            let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
            regs.push(Sym::Op(value));
        }
        let mut varargs = Vec::new();
        if chunk.is_vararg {
            for (source, ty) in arguments.iter().skip(parameters) {
                let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
                varargs.push(Sym::Op(value));
            }
        }
        if passed < register_count {
            // A missing argument or an ordinary local starts as nil.
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            regs.resize(register_count, Sym::Op(nil));
        }
        // The interpreter marks the access as the caller's last instruction.
        self.set_prev(PrevPc::At(pc));
        let parent = super::FrameId(self.frame_id());
        let results = extra
            .results
            .unwrap_or_else(|| usize::from(entry != Entry::NewIndex));
        self.inlined.push(TraceFrameInfo {
            register_count: u16::try_from(register_count).expect("register count fits u16"),
            code_len: chunk.code.len(),
            inlined: Some(InlinedFrame {
                closure: lane,
                parent,
                call_pc: pc,
                result_base: u8::try_from(base).expect("register index fits u8"),
                results: u8::try_from(results).expect("result count fits u8"),
                reexec: opens_region,
                tail_calls: 0,
                tail_site: None,
                protected: Protected::No,
                entry,
                dynamic: false,
            }),
            activation: None,
        });
        self.frames.push(Inlined {
            chunk,
            closure: lane,
            frame: u32::try_from(self.inlined.len()).expect("frame count fits u32"),
            regs,
            prev: PrevPc::Unset,
            call_pc: pc,
            result_base: base,
            results,
            varargs,
            owns_reexec: opens_region,
            tail_calls: 0,
            protected: Protected::No,
            handler: false,
            entry,
            after: extra.after,
            dynamic: None,
        });
        // The call is entered: nothing aborts on the metamethod.
        self.abort_metamethod = None;
        Ok(Flow::Enter)
    }

    /// The pin of `event` in `metatable`: the Lua closure it holds, or the
    /// absence of a value. `None` for a field of another type (a native
    /// function, a table), which a class cannot pin.
    pub(super) fn event_pin(
        runtime: &Runtime,
        metatable: TableHandle,
        event: &str,
    ) -> Option<MetaPin> {
        let key = crate::vm::arith::metamethod_key(event)?;
        let handler = match runtime.table_get_string_ref(metatable, key) {
            None | Some(Value::Nil) => META_PIN_ABSENT,
            Some(Value::Closure(handler)) => handler.native_identity(),
            Some(_) => return None,
        };
        Some(MetaPin {
            event: event.as_bytes().into(),
            handler,
        })
    }

    /// The window register a metatable class binds through: a table entry
    /// input the iteration does not write, since a binding needs one to be
    /// part of a run's entry. Prefers `bound`, the register of the table the
    /// access is on, then a register another binding of the tree names, then
    /// (in a root recording) a register that held a table with `metatable`
    /// when the recording began. The class is the pinned metatable and not
    /// that table's, so the choice only has to be a table at entry.
    pub(super) fn class_window(
        &self,
        bound: Option<usize>,
        metatable: TableHandle,
        runtime: &Runtime,
    ) -> Option<usize> {
        let usable = |window: usize| self.class_register_usable(window);
        if let Some(window) = bound.filter(|window| usable(*window)) {
            return Some(window);
        }
        if let Some(binding) = self.bindings.iter().find(|binding| {
            !matches!(binding.kind, BindingKind::Bytes { .. })
                && usable(usize::from(binding.register))
        }) {
            return Some(usize::from(binding.register));
        }
        if self.side.is_some() {
            return None;
        }
        self.start_tables
            .iter()
            .find(|(window, table)| {
                usable(*window)
                    && matches!(
                        runtime.raw_metatable(&Value::Table(*table)),
                        Ok(Some(found)) if found == metatable
                    )
            })
            .map(|(window, _)| *window)
    }

    /// Whether a metatable class or its receiver's key-absence lookup may
    /// bind the table entry input in register `window`: nothing in the
    /// iteration so far or in the loops around writes it. A side recording
    /// sees only its own writes, not the root's, and the tree refuses a side
    /// whose binding names a register some member writes, so a side binds only
    /// through a register a binding of the tree already names (one the tree
    /// checked nobody writes). A side that cannot keeps the helper.
    fn class_register_usable(&self, window: usize) -> bool {
        window < self.register_count
            && !self.written.get(window).copied().unwrap_or(true)
            && !self.changes_around(window)
            && (self.side.is_none()
                || self.bindings[..self.seeded_bindings.min(self.bindings.len())]
                    .iter()
                    .any(|binding| {
                        usize::from(binding.register) == window
                            && !matches!(binding.kind, BindingKind::Bytes { .. })
                    }))
    }

    /// The binding of the class `pins` describe for `metatable`, through the
    /// table entry input in `window`.
    pub(super) fn class_binding(
        &mut self,
        window: usize,
        metatable: TableHandle,
        pins: Vec<MetaPin>,
        bare: u8,
    ) -> u32 {
        // The register must be an entry input for the binding to resolve.
        self.condition_reads.push(window);
        self.binding(
            window,
            BindingKind::MetaClass {
                metatable: metatable.native_identity(),
                pins: pins.into(),
                bare,
            },
        )
    }

    /// Whether the trace can list the metatable word of a row, which every
    /// [`TraceOp::MetaGuard`] reads. Called before a class records anything,
    /// so a trace with no room for the field keeps the helper instead of
    /// abandoning the recording.
    pub(super) fn class_guard_fits(&mut self) -> bool {
        self.field(
            &crate::string::LuaString::from(META_FIELD_KEY),
            TraceType::Integer,
        )
        .is_ok()
    }

    /// Weighs a class-checked metamethod like the helper call it replaces.
    /// The trace saves the interpreter's metatable walk however cheaply it
    /// checks the class, so the loop's work for the compile policy (see
    /// `TracePolicy::runs_too_short`) must not drop with the helper: a short
    /// top-level loop around a handler would otherwise read as too little
    /// work to repay a compile and never trace.
    pub(super) fn charge_class_dispatch(&mut self) {
        self.host_work = self.host_work.saturating_add(super::HOST_CALL_GUEST_WORK);
    }

    /// Emits the guard that the metatable of `table` is the class `binding`
    /// pins, leaving through the `Before` snapshot of the instruction at
    /// `pc`.
    pub(super) fn guard_class(
        &mut self,
        pc: usize,
        table: Operand,
        binding: u32,
    ) -> Result<(), TraceAbortReason> {
        // The guard reads the metatable word of the table's row.
        self.field(
            &crate::string::LuaString::from(META_FIELD_KEY),
            TraceType::Integer,
        )?;
        let snapshot = self.read_snapshot(pc);
        self.emit(
            pc,
            RecKind::MetaGuard {
                binding,
                table,
                snapshot,
            },
            None,
        );
        Ok(())
    }

    /// The access of a table whose metatable names `handler` as `event`
    /// (`__index` or `__newindex`), checked by a class instead of a walk of
    /// the metatables at run time: the metatable is the pinned one, which
    /// entry fixed to hold `handler` under `event`, and the table itself lacks
    /// the key (a lookup that finds nil). Applies to a string key of a table
    /// the entry binds, in a loop trace. `false` records nothing and leaves
    /// the access to the helper.
    pub(super) fn class_access(
        &mut self,
        at: &HookPoint<'_>,
        (table, handle): (Operand, TableHandle),
        (key, key_ty): (Operand, TraceType),
        (event, handler): (&str, ClosureHandle),
    ) -> Result<bool, TraceAbortReason> {
        let pc = at.pc;
        if self.function.is_some() || !matches!(key_ty, TraceType::String | TraceType::StringRef) {
            return Ok(false);
        }
        let Some(runtime) = at.surroundings.map(|around| around.runtime) else {
            return Ok(false);
        };
        let Ok(Some(metatable)) = runtime.raw_metatable(&Value::Table(handle)) else {
            return Ok(false);
        };
        let Some(pin) = Self::event_pin(runtime, metatable, event) else {
            return Ok(false);
        };
        if pin.handler != handler.native_identity() {
            return Ok(false);
        }
        let Ok((Receiver::Bound(receiver), ..)) = self.receiver_of(at, table, handle) else {
            return Ok(false);
        };
        // The key-absence lookup binds the receiver's own window, so a
        // receiver without a usable one (a global or an upvalue) is left to
        // the helper; only the class may bind through another table.
        if !self.class_register_usable(receiver) {
            return Ok(false);
        }
        let Some(window) = self.class_window(Some(receiver), metatable, runtime) else {
            return Ok(false);
        };
        if !self.class_guard_fits() {
            return Ok(false);
        }
        self.charge_class_dispatch();
        let class = self.class_binding(window, metatable, vec![pin], 0);
        self.guard_class(pc, table, class)?;
        // The table itself lacks the key: a lookup that finds nil, with the
        // class guard standing for the metatable that runs the handler.
        let hash = self.binding(receiver, BindingKind::Hash);
        let snapshot = self.read_snapshot(pc);
        self.emit(
            pc,
            RecKind::HashLoad {
                binding: hash,
                table: Operand::Carried(receiver),
                key,
                expected: TraceType::Nil,
                snapshot,
            },
            Some(TraceType::Nil),
        );
        Ok(true)
    }

    /// The table `receiver` names, as an operand a helper takes.
    pub(super) fn receiver_operand(receiver: Receiver) -> Operand {
        match receiver {
            Receiver::Bound(window) | Receiver::Around(window) => Operand::Carried(window),
            Receiver::Loaded(table) => table,
        }
    }

    /// The store of `value` under the string `key` into the table of
    /// `receiver`, by [`HelperId::TABLE_SET_STRING`]: a write that runs no
    /// function (the walk of `__newindex` ends at a table that takes it), or
    /// with `raw` `rawset`'s.
    pub(super) fn record_string_store(
        &mut self,
        pc: usize,
        receiver: Receiver,
        (key, key_ty): (Operand, TraceType),
        (value, value_ty): (Operand, TraceType),
        raw: bool,
    ) {
        let table = Self::receiver_operand(receiver);
        let mode = self.scalar(pc, TraceScalar::Integer(i64::from(raw)));
        self.emit_helper_call(
            pc,
            HelperId::TABLE_SET_STRING,
            &[
                (table, TraceType::Table),
                (key, key_ty),
                (value, value_ty),
                (Operand::Op(mode), TraceType::Integer),
            ],
        );
    }
}
