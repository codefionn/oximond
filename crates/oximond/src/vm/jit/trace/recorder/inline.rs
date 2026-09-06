//! Library functions the recorder turns into operations of the trace, with no
//! call: `type`, `rawequal`, `select`, `assert` and `math.floor`, `ceil`,
//! `abs`, `tointeger` and `type`.
//!
//! Each one is recorded behind the guard on the callee's identity, like any
//! native call. The types of the arguments are already fixed by the entry
//! contract or by the operation that made the value, so a different type
//! never reaches the code: the entry refuses it or an earlier guard leaves.
//! What depends on a value (the sign of a float, whether it fits an integer,
//! a constant index) is guarded, and any guard that fails leaves through the
//! `Before` snapshot of the call, so the interpreter makes the call itself and
//! raises whatever error it raises. Nothing here raises an error.
//!
//! `type` and `math.type` give a string constant of the trace's roots (a
//! [`Sym::Root`]), which only rides to a snapshot or a comparison with
//! another constant, so `type(x) == "number"` folds to a fixed branch.

use super::{
    ArithmeticOp, BindingKind, ExternalSlot, Flow, HookPoint, Kind, Open, Operand, RecKind,
    Recorder, SnapshotPhase, Sym, TraceAbortReason, TraceScalar, TraceType, is_number,
};
use crate::bytecode::{Constant, Instruction, OpCode};
use crate::jit_ir::Comparison;
use crate::jit_ir::trace::ROOT_STRING_LANE_BASE;
use crate::jit_ir::trace::{HelperId, NumericOp, pinned_native_identity};
use crate::runtime::{DirectNativeKind, NativeFunctionHandle, Value};

/// The most bytes one `string.byte` call may give a trace.
const MAX_BYTE_RESULTS: i64 = 16;

/// An argument of a call that takes symbols: the symbol and, unless it is a
/// string constant, its operand and type.
type CallSymbol = (Sym, Option<(Operand, TraceType)>);

/// 2^63 as a float: the first float above the range of an integer.
const TWO_63: f64 = 9_223_372_036_854_775_808.0;

impl Recorder {
    /// Whether the recorder records `kind` as operations (see this module).
    pub(super) const fn is_inline_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::Type
                | DirectNativeKind::RawEqual
                | DirectNativeKind::MathFloor
                | DirectNativeKind::MathCeil
                | DirectNativeKind::MathAbs
                | DirectNativeKind::MathToInteger
                | DirectNativeKind::MathType
        )
    }

    /// Whether `kind` reads its arguments as symbolic registers (a string
    /// constant is one), so the call must not read them first.
    pub(super) const fn is_symbolic_kind(kind: DirectNativeKind) -> bool {
        matches!(kind, DirectNativeKind::Select | DirectNativeKind::Assert)
            || Self::is_raw_access_kind(kind)
    }

    /// The native-function kind of `function`, if the VM knows one.
    pub(super) fn native_kind(
        at: &HookPoint<'_>,
        function: NativeFunctionHandle,
    ) -> Option<DirectNativeKind> {
        at.surroundings
            .and_then(|around| around.runtime.direct_native_kind(function))
    }

    /// Records a call of `type`, `rawequal` or one of the `math` functions of
    /// [`Self::is_inline_kind`]. `arguments` are the read argument registers,
    /// the first in register `first_argument`.
    ///
    /// A call that takes the open results of the call before it (`hoisted`)
    /// may not emit a guard: it has no snapshot to leave through, since the
    /// interpreter holds the values in the frame's open-results slot. Its
    /// callee was guarded before the call that made them. A recording that
    /// needs a guard aborts.
    pub(super) fn record_inline_native(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let first_op = self.ops.len();
        let flow = self.record_inline_call(
            at,
            kind,
            (callee, function),
            (arguments, first_argument),
            (base, results),
            hoisted,
        )?;
        if hoisted
            && self.ops[first_op..]
                .iter()
                .any(|op| matches!(op.kind, RecKind::GuardBranch { .. }))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        Ok(flow)
    }

    fn record_inline_call(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let Some(&(argument, ty)) = arguments.first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        if ty.is_wide() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        if hoisted {
            if !self.take_hoisted(callee) {
                return Err(TraceAbortReason::Divergence);
            }
        } else {
            self.guard_callee(pc, callee, lane);
        }
        let live = &at.registers[first_argument];
        match kind {
            DirectNativeKind::Type => {
                let name = match ty {
                    TraceType::Nil => "nil",
                    TraceType::Boolean => "boolean",
                    TraceType::Integer | TraceType::Float => "number",
                    TraceType::Table => "table",
                    TraceType::Closure => "function",
                    TraceType::Thread => "thread",
                    TraceType::Userdata => "userdata",
                    TraceType::String | TraceType::StringRef => "string",
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                };
                self.note_type_read(argument);
                self.root_result(pc, name, (base, results))
            }
            DirectNativeKind::MathType => {
                self.note_type_read(argument);
                match ty {
                    TraceType::Integer => self.root_result(pc, "integer", (base, results)),
                    TraceType::Float => self.root_result(pc, "float", (base, results)),
                    _ => Ok(self.nil_result(pc, (base, results))),
                }
            }
            DirectNativeKind::RawEqual => {
                let Some(&(other, other_ty)) = arguments.get(1) else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                if other_ty.is_wide() {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // rawequal never runs `__eq`, so two tables compare by
                // identity with no guard.
                let value = if ty == TraceType::Table && other_ty == TraceType::Table {
                    self.compare_tables(pc, argument, other)
                } else {
                    self.record_comparison(
                        pc,
                        Comparison::Equal,
                        (argument, ty),
                        (other, other_ty),
                        Some((live, &at.registers[first_argument + 1])),
                    )?
                };
                Ok(self.native_result(pc, value, (base, results)))
            }
            DirectNativeKind::MathFloor | DirectNativeKind::MathCeil => {
                let ceil = kind == DirectNativeKind::MathCeil;
                let value = self.record_round(pc, ceil, (argument, ty), live)?;
                Ok(self.native_result(pc, value, (base, results)))
            }
            DirectNativeKind::MathAbs => {
                if !is_number(ty) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let value = self.emit(pc, RecKind::Numeric(NumericOp::Abs, argument), Some(ty));
                Ok(self.native_result(pc, value, (base, results)))
            }
            DirectNativeKind::MathToInteger => {
                self.record_to_integer(pc, (argument, ty), live, (base, results))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The bytes of the string constant register `register` holds: one the
    /// interpreter loaded, or a copy of one that moved through an inlined
    /// call.
    pub(super) fn constant_text<'a>(
        &'a self,
        registers: &'a [Value],
        register: usize,
    ) -> Option<&'a [u8]> {
        match self.sym(register) {
            Sym::Root(_) => match registers.get(register) {
                Some(Value::String(text)) => Some(text.as_bytes()),
                _ => None,
            },
            Sym::Op(index) => self.root_constant_bytes(Operand::Op(index)),
            _ => None,
        }
    }

    /// The bytes of the root string `operand` is, through moves.
    pub(super) fn root_constant_bytes(&self, operand: Operand) -> Option<&[u8]> {
        let Operand::Op(index) = self.origin(operand) else {
            return None;
        };
        match self.ops[index as usize].kind {
            RecKind::Constant(TraceScalar::String(lane)) => {
                let slot = usize::try_from(lane.checked_sub(ROOT_STRING_LANE_BASE)?).ok()?;
                self.root_constants.get(slot).map(|text| &**text)
            }
            _ => None,
        }
    }

    /// Whether the string constants in registers `left` and `right` are
    /// equal, when both registers hold one.
    pub(super) fn root_strings_equal(
        &self,
        registers: &[Value],
        left: usize,
        right: usize,
    ) -> Option<bool> {
        Some(self.constant_text(registers, left)? == self.constant_text(registers, right)?)
    }

    /// The outcome of the equality test `instruction` when both its operands
    /// are string constants: registers that hold one, or a register and a
    /// constant of the chunk.
    pub(super) fn string_constants_equal(
        &self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Option<bool> {
        let a = usize::from(instruction.a());
        if opcode.is_test_constant() {
            let Some(Constant::String(constant)) =
                at.chunk.constants.get(usize::from(instruction.b()))
            else {
                return None;
            };
            return Some(self.constant_text(at.registers, a)? == constant.as_bytes());
        }
        self.root_strings_equal(at.registers, a, usize::from(instruction.b()))
    }

    /// Notes that the entry type of a carried operand decides what the trace
    /// does, so the entry contract must keep it.
    pub(super) fn note_type_read(&mut self, operand: Operand) {
        if let Operand::Carried(register) = operand {
            self.condition_reads.push(register);
        }
    }

    /// Makes the constant operation holding `scalar`.
    pub(super) fn scalar(&mut self, pc: usize, scalar: TraceScalar) -> u32 {
        self.emit(pc, RecKind::Constant(scalar), Some(scalar.ty()))
    }

    /// The constant operation of the integer `value` as the profile's number.
    pub(super) fn number_scalar(&mut self, pc: usize, value: i64) -> u32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "a profile without integers has floats only"
        )]
        let scalar = if self.integers {
            TraceScalar::Integer(value)
        } else {
            TraceScalar::Float(value as f64)
        };
        self.scalar(pc, scalar)
    }

    /// Emits the guard that leaves through the `Before` snapshot of the
    /// instruction at `pc` unless the boolean `condition` is `expected`.
    pub(super) fn guard_flag(&mut self, pc: usize, condition: Operand, expected: bool) {
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition,
                expected,
                snapshot,
            },
            None,
        );
    }

    /// Guards the two range comparisons of the float `value` against the
    /// outcome the recording saw: `-2^63 <= value` and `value < 2^63`. Both
    /// true means it lies in the range of an integer. NaN fails both, a huge
    /// positive value only the second, a huge negative one only the first.
    pub(super) fn guard_range_outcome(
        &mut self,
        pc: usize,
        value: Operand,
        (above, below): (bool, bool),
    ) {
        let low = Operand::Op(self.scalar(pc, TraceScalar::Float(-TWO_63)));
        let high = Operand::Op(self.scalar(pc, TraceScalar::Float(TWO_63)));
        let at_least = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, low, value),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(at_least), above);
        let under = self.emit(
            pc,
            RecKind::Compare(Comparison::LessThan, value, high),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(under), below);
    }

    /// Guards that the float `value` lies in the range of an integer. NaN
    /// fails both comparisons.
    pub(super) fn guard_integer_range(&mut self, pc: usize, value: Operand) {
        self.guard_range_outcome(pc, value, (true, true));
    }

    /// `math.floor` or `math.ceil` of one number. An integer is its own
    /// result. A float rounds to a float, and where the profile has integers
    /// the result is an integer when the interpreter gives one, which the
    /// recorded outcome and a range guard decide.
    fn record_round(
        &mut self,
        pc: usize,
        ceil: bool,
        (argument, ty): (Operand, TraceType),
        live: &Value,
    ) -> Result<u32, TraceAbortReason> {
        match (ty, live) {
            (TraceType::Integer, Value::Integer(_)) => {
                Ok(self.emit(pc, RecKind::Move(argument), Some(ty)))
            }
            (TraceType::Float, Value::Number(number)) => {
                let op = if ceil {
                    NumericOp::Ceil
                } else {
                    NumericOp::Floor
                };
                let rounded = self.emit(pc, RecKind::Numeric(op, argument), Some(TraceType::Float));
                if !self.integers {
                    return Ok(rounded);
                }
                let live = if ceil { number.ceil() } else { number.floor() };
                // The interpreter gives an integer only when the rounded
                // float fits one. Otherwise (NaN, an infinity, 2^63 and up)
                // the result stays the float. The guards keep whichever
                // outcome the recording saw.
                let above = -TWO_63 <= live;
                let below = live < TWO_63;
                self.guard_range_outcome(pc, Operand::Op(rounded), (above, below));
                if !(above && below) {
                    return Ok(rounded);
                }
                Ok(self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::FloatToInteger, Operand::Op(rounded)),
                    Some(TraceType::Integer),
                ))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// `math.tointeger` of one value: an integer is its own result, a float
    /// gives an integer when it is integral and in range and nil otherwise,
    /// and any other value the interpreter does not convert gives nil.
    fn record_to_integer(
        &mut self,
        pc: usize,
        (argument, ty): (Operand, TraceType),
        live: &Value,
        results: (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        match (ty, live) {
            (TraceType::Integer, Value::Integer(_)) => {
                let value = self.emit(pc, RecKind::Move(argument), Some(ty));
                Ok(self.native_result(pc, value, results))
            }
            (TraceType::Float, Value::Number(number)) => {
                let floor = self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::Floor, argument),
                    Some(TraceType::Float),
                );
                let integral = self.emit(
                    pc,
                    RecKind::Compare(Comparison::Equal, Operand::Op(floor), argument),
                    Some(TraceType::Boolean),
                );
                if number.is_finite() && number.fract() == 0.0 && (-TWO_63..TWO_63).contains(number)
                {
                    self.guard_flag(pc, Operand::Op(integral), true);
                    self.guard_integer_range(pc, argument);
                    let value = self.emit(
                        pc,
                        RecKind::Numeric(NumericOp::FloatToInteger, argument),
                        Some(TraceType::Integer),
                    );
                    Ok(self.native_result(pc, value, results))
                } else if number.is_nan() || (number.is_finite() && number.fract() != 0.0) {
                    // NaN is not equal to its floor either.
                    self.guard_flag(pc, Operand::Op(integral), false);
                    Ok(self.nil_result(pc, results))
                } else {
                    // An integral float out of range, or an infinity: the
                    // guard for that case would take a second comparison.
                    Err(TraceAbortReason::UnsupportedOperand)
                }
            }
            (TraceType::Nil | TraceType::Boolean | TraceType::Table | TraceType::Closure, _) => {
                self.note_type_read(argument);
                Ok(self.nil_result(pc, results))
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The call's one result is the string constant `text`.
    fn root_result(
        &mut self,
        pc: usize,
        text: &str,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let slot = self.root_slot(text.as_bytes())?;
        // As an open result the constant waits for the instruction that
        // takes it.
        if results == 0 {
            self.open = Some(vec![Open::Root(slot)]);
            return Ok(Flow::Next(pc + 1));
        }
        self.write(base, Sym::Root(slot));
        self.fill_nil(pc, base + 1..base + results);
        Ok(Flow::Next(pc + 1))
    }

    /// The call's one result is nil.
    pub(super) fn nil_result(&mut self, pc: usize, results: (usize, usize)) -> Flow {
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        self.native_result(pc, nil, results)
    }

    /// Writes nil to each register of `range`.
    pub(super) fn fill_nil(&mut self, pc: usize, range: std::ops::Range<usize>) {
        if range.is_empty() {
            return;
        }
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        for register in range {
            self.write(register, nil);
        }
    }

    /// Gives a call its results: the registers from `base` (nil past the
    /// values), or, for an open count, the open results of the next
    /// instruction.
    pub(super) fn native_results(
        &mut self,
        pc: usize,
        values: &[u32],
        (base, results): (usize, usize),
    ) -> Flow {
        if results == 0 {
            self.open = Some(values.iter().map(|value| Open::Op(*value)).collect());
            return Flow::Next(pc + 1);
        }
        for (register, value) in (base..base + results).zip(values) {
            self.write(register, *value);
        }
        self.fill_nil(pc, base + values.len().min(results)..base + results);
        Flow::Next(pc + 1)
    }

    /// The symbol of argument register `register` and, unless it is a string
    /// constant, its operand and type. A value the iteration did not make
    /// gets a copy, so the symbol is always one of the trace's operations or
    /// a constant.
    pub(super) fn argument_symbol(
        &mut self,
        pc: usize,
        registers: &[Value],
        register: usize,
    ) -> Result<(Sym, Option<(Operand, TraceType)>), TraceAbortReason> {
        self.check_register(register)?;
        if let Sym::Root(slot) = self.sym(register) {
            if !matches!(registers[register], Value::String(_)) {
                return Err(TraceAbortReason::Divergence);
            }
            return Ok((Sym::Root(slot), None));
        }
        let (operand, ty) = self.read(registers, register)?;
        let symbol = match operand {
            Operand::Op(index) => Sym::Op(index),
            Operand::Carried(_) => Sym::Op(self.emit(pc, RecKind::Move(operand), Some(ty))),
        };
        Ok((symbol, Some((operand, ty))))
    }

    /// The values `...` holds in the running frame. An inlined callee's are
    /// fixed by the path. The root frame's are entry inputs: the trace guards
    /// their count against the one the recording saw, then reads each from a
    /// window slot ([`ExternalSlot::Vararg`]).
    pub(super) fn current_varargs(
        &mut self,
        at: &HookPoint<'_>,
    ) -> Result<Vec<Sym>, TraceAbortReason> {
        if let Some(frame) = self.frames.last() {
            return Ok(frame.varargs.clone());
        }
        if let Some(known) = &self.root_varargs {
            return Ok(known.clone());
        }
        if self.function.is_some() || self.activation.is_some() || self.loop_tail.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A named vararg table (`...t`, Lua 5.5) expands `...` from the table,
        // which the loop can change, so the frame's raw varargs are not it.
        if at.chunk.arg_register.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let live = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .varargs;
        let pc = at.pc;
        let count = i64::try_from(live.len()).map_err(|_| TraceAbortReason::LimitExceeded)?;
        let register = self.special_register(ExternalSlot::VarargCount, Kind::Integer)?;
        let (held, _) = self.read_window(register)?;
        let expected = Operand::Op(self.scalar(pc, TraceScalar::Integer(count)));
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, held, expected),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(same), true);
        let mut values = Vec::with_capacity(live.len());
        for (index, value) in live.iter().enumerate() {
            let kind = Kind::of(value, self.integers);
            if kind == Kind::Other {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let slot = ExternalSlot::Vararg(
                u32::try_from(index).map_err(|_| TraceAbortReason::LimitExceeded)?,
            );
            let register = self.special_register(slot, kind)?;
            let (operand, ty) = self.read_window(register)?;
            values.push(Sym::Op(self.emit(pc, RecKind::Move(operand), Some(ty))));
        }
        self.root_varargs = Some(values.clone());
        Ok(values)
    }

    /// The symbols a call of `select` or `assert` takes: the `fixed` argument
    /// registers from `first_argument`, then the running frame's varargs for
    /// a call that forwards them.
    fn call_symbols(
        &mut self,
        at: &HookPoint<'_>,
        (first_argument, fixed): (usize, usize),
        with_varargs: bool,
    ) -> Result<Vec<CallSymbol>, TraceAbortReason> {
        let mut symbols = Vec::with_capacity(fixed);
        for index in 0..fixed {
            symbols.push(self.argument_symbol(at.pc, at.registers, first_argument + index)?);
        }
        if with_varargs {
            for value in self.current_varargs(at)? {
                let Sym::Op(index) = value else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                let ty = self.ops[index as usize]
                    .ty
                    .expect("an operand operation defines a value");
                symbols.push((value, Some((Operand::Op(index), ty))));
            }
        }
        Ok(symbols)
    }

    /// Gives a call the symbols it returns: the registers from `base`, or the
    /// open results of the next instruction, which only operations can be.
    fn symbol_results(
        &mut self,
        pc: usize,
        symbols: &[Sym],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        if results == 0 {
            let mut open = Vec::with_capacity(symbols.len());
            for symbol in symbols {
                open.push(match symbol {
                    Sym::Op(index) => Open::Op(*index),
                    Sym::Root(slot) => Open::Root(*slot),
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                });
            }
            self.open = Some(open);
            return Ok(Flow::Next(pc + 1));
        }
        for (register, symbol) in (base..base + results).zip(symbols) {
            self.write(register, *symbol);
        }
        self.fill_nil(pc, base + symbols.len().min(results)..base + results);
        Ok(Flow::Next(pc + 1))
    }

    /// A call of `rawequal` on two strings. Two string constants fold to a
    /// boolean. Any other pair is a helper call that compares the bytes:
    /// two lanes (or a lane and a table slot) may name equal strings, and
    /// `__eq` never runs for strings.
    pub(super) fn record_rawequal_strings(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        first_argument: usize,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let value = if let Some(equal) =
            self.root_strings_equal(at.registers, first_argument, first_argument + 1)
        {
            self.emit(
                pc,
                RecKind::Constant(TraceScalar::Boolean(equal)),
                Some(TraceType::Boolean),
            )
        } else {
            let left = self.host_operand(pc, at.registers, first_argument)?;
            let right = self.host_operand(pc, at.registers, first_argument + 1)?;
            for (operand, _) in [left, right] {
                self.note_type_read(operand);
            }
            self.emit_helper_call(pc, HelperId::STRING_EQUAL, &[left, right])
        };
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// A call of `assert` with at least one argument. A false or nil first
    /// argument makes the interpreter raise, so the trace leaves before the
    /// call unless that argument is true. The call returns its arguments.
    pub(super) fn record_assert(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
        with_varargs: bool,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        // A falsy condition raises the error a protected call of the trace
        // catches. Its message is read there, so a table the trace made may
        // be one: it goes no further.
        if self.catches_assert(at, function, (first_argument, fixed), with_varargs) {
            let (_, condition) = self.argument_symbol(pc, at.registers, first_argument)?;
            let (operand, ty) = match condition {
                Some((operand, ty)) => (Some(operand), ty),
                None => return Err(TraceAbortReason::Divergence),
            };
            return self.record_assert_failure(
                at,
                (callee, function),
                (first_argument, fixed),
                (operand, ty),
            );
        }
        let symbols = self.call_symbols(at, (first_argument, fixed), with_varargs)?;
        let Some((_, condition)) = symbols.first() else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        match *condition {
            // A string constant is true.
            None => {}
            Some((_, TraceType::Nil)) => return Err(TraceAbortReason::UnsupportedOperand),
            Some((operand, TraceType::Boolean)) => self.guard_flag(pc, operand, true),
            Some((operand, _)) => self.note_type_read(operand),
        }
        let symbols: Vec<Sym> = symbols.into_iter().map(|(symbol, _)| symbol).collect();
        self.symbol_results(pc, &symbols, (base, results))
    }

    /// A call of `select` with fixed arguments, then the running frame's
    /// varargs when `with_varargs` or the open results of the call before
    /// when `hoisted`. `select('#', ...)` is the count, and `select(k, ...)`
    /// the values from the k-th, for a k the recording fixes: a constant, or
    /// one a guard pins. The count of values is fixed by the path the trace
    /// follows (the callee guard and the call site).
    ///
    /// With open results pending no guard may leave, so the callee was
    /// guarded before the call that made them and the index has to be a
    /// constant.
    pub(super) fn record_select(
        &mut self,
        at: &HookPoint<'_>,
        (callee, function): (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
        (with_varargs, hoisted): (bool, bool),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if fixed == 0 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let mut symbols = self.call_symbols(at, (first_argument, fixed), with_varargs)?;
        let (_, selector) = symbols.remove(0);
        let mut values: Vec<Sym> = symbols.into_iter().map(|(symbol, _)| symbol).collect();
        if hoisted {
            for value in self.open.take().ok_or(TraceAbortReason::Divergence)? {
                values.push(match value {
                    Open::Op(index) => Sym::Op(index),
                    Open::Root(slot) => Sym::Root(slot),
                    Open::Char(_) => return Err(TraceAbortReason::UnsupportedOperand),
                });
            }
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        if hoisted {
            if !self.take_hoisted(callee) {
                return Err(TraceAbortReason::Divergence);
            }
        } else {
            self.guard_callee(pc, callee, lane);
        }
        let count = i64::try_from(values.len()).map_err(|_| TraceAbortReason::LimitExceeded)?;
        let (index, operand, scalar) = match (&at.registers[first_argument], selector) {
            // The string constant `'#'`.
            (Value::String(text), None) if text.as_bytes() == b"#" => {
                let value = self.number_scalar(pc, count);
                return Ok(self.native_result(pc, value, (base, results)));
            }
            (Value::Integer(index), Some((operand, TraceType::Integer))) => {
                (*index, operand, TraceScalar::Integer(*index))
            }
            (Value::Number(index), Some((operand, TraceType::Float)))
                if index.is_finite() && index.fract() == 0.0 && index.abs() < 9.0e15 =>
            {
                #[allow(clippy::cast_possible_truncation, reason = "checked to be small")]
                (*index as i64, operand, TraceScalar::Float(*index))
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        // The interpreter's own bounds: nonzero, and not before the first
        // value. A positive index past the last value gives no values.
        let start = if index > 0 { index } else { count + index + 1 };
        if index == 0 || start < 1 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let start = start.min(count + 1);
        // A constant index needs no guard.
        let constant = matches!(
            self.origin(operand),
            Operand::Op(op) if matches!(self.ops[op as usize].kind, RecKind::Constant(_))
        );
        if !constant {
            if hoisted {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let expected = Operand::Op(self.scalar(pc, scalar));
            let same = self.emit(
                pc,
                RecKind::Compare(Comparison::Equal, operand, expected),
                Some(TraceType::Boolean),
            );
            self.guard_flag(pc, Operand::Op(same), true);
        }
        let start = usize::try_from(start - 1).map_err(|_| TraceAbortReason::Divergence)?;
        self.symbol_results(pc, &values[start..], (base, results))
    }

    /// A call of `string.byte(string, first, last)` for two numbers of the
    /// profile. The trace takes a run of bytes whose length the recording
    /// fixes: `last` must stay `count - 1` past `first`, and every byte is
    /// read with the bounds check of [`RecKind::StringByte`].
    pub(super) fn record_byte_run(
        &mut self,
        at: &HookPoint<'_>,
        (string, first, last): (Operand, (Operand, TraceType), (Operand, TraceType)),
        live: (&Value, &Value, &Value),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let number = self.number_type();
        if first.1 != number || last.1 != number {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let (Value::String(text), Some(start), Some(end)) = (
            live.0,
            Self::live_integer(live.1),
            Self::live_integer(live.2),
        ) else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let length = i64::try_from(text.len()).map_err(|_| TraceAbortReason::Divergence)?;
        // Only a run inside the string. The interpreter makes every call
        // that is empty, negative or clipped.
        if start < 1 || end < start || end > length || end - start >= MAX_BYTE_RESULTS {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let window = self.bound_string(string)?;
        let binding = self.binding(window, BindingKind::Bytes { ty: number });
        // `last` is `count - 1` past `first`, so the run keeps its length.
        let span = Operand::Op(self.number_scalar(pc, end - start));
        let target = if end == start {
            first.0
        } else {
            Operand::Op(self.emit(
                pc,
                RecKind::Arithmetic(ArithmeticOp::Add, first.0, span),
                Some(number),
            ))
        };
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, last.0, target),
            Some(TraceType::Boolean),
        );
        self.guard_flag(pc, Operand::Op(same), true);
        let mut bytes = Vec::new();
        for offset in 0..=(end - start) {
            let key = if offset == 0 {
                first.0
            } else {
                let step = Operand::Op(self.number_scalar(pc, offset));
                Operand::Op(self.emit(
                    pc,
                    RecKind::Arithmetic(ArithmeticOp::Add, first.0, step),
                    Some(number),
                ))
            };
            let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            bytes.push(self.emit(
                pc,
                RecKind::StringByte {
                    binding,
                    string: Operand::Carried(window),
                    key,
                    snapshot,
                },
                Some(number),
            ));
        }
        Ok(self.native_results(pc, &bytes, (base, results)))
    }

    /// The integer a live number holds, if it is an integral one.
    pub(super) fn live_integer(value: &Value) -> Option<i64> {
        match value {
            Value::Integer(value) => Some(*value),
            Value::Number(number) if number.is_finite() && number.fract() == 0.0 =>
            {
                #[allow(clippy::cast_possible_truncation, reason = "checked to be small")]
                (number.abs() < 9.0e15).then_some(*number as i64)
            }
            _ => None,
        }
    }
}
