//! Numbers whose subtype is known only at run time.
//!
//! A loop that reads `values[i]` from an array holding integers and floats
//! sees a different subtype from one iteration to the next. Recorded for the
//! subtype of one iteration, the load's type guard fails for the others and
//! each of those leaves through a side trace. Here the load is a
//! [`RecKind::NumberLoad`]: a float with the number's value (exact, because
//! the load leaves for an integer beyond 2^53 in magnitude) and a tag, the
//! boolean [`RecKind::NumberTag`], that says whether the slot held an
//! integer. A register that holds one is a [`Sym::Op`] of the float whose
//! tag is in [`Recorder::number_tags`].
//!
//! The instructions that take a number of either subtype are the arithmetic
//! of `+`, `-`, `*`, `/` and `%`, the comparisons and `Move`. They compute on
//! the floats. An operation of two numbers whose result may be an integer
//! (`+`, `-`, `*`, `%`) has the tag `a and b`, and checks that an integer
//! result stays inside 2^53 (`+`, `-`, `*`: the float is then exact and the
//! integer wrap of Lua never applies) or has a nonzero divisor (`%`: an
//! integer zero divisor raises an error, which the interpreter does by
//! running the instruction). Floats and integer-valued floats below 2^53 take
//! the same floor modulo on integers and floats, so one operation serves
//! both. The result of an operation with a float operand, or of `/`, is a
//! float.
//!
//! Every other instruction first narrows the registers it reads to the
//! subtype they have now, behind a guard on the tag. A snapshot restores a
//! register of this kind from its float and its tag
//! ([`oximond::jit_ir::trace::NumberRegister`]). At the end of the trace
//! the registers still live are narrowed too, since a loop parameter or a
//! link has one type.

use super::{
    ArithmeticOp, Comparison, FloatModulo, HookPoint, Instruction, Live, Number, OpCode, Operand,
    RecKind, Recorder, SnapshotPhase, Sym, TraceAbortReason, TraceScalar, TraceType,
};
use crate::jit_ir::trace::NumericOp;
use crate::jit_ir::{observable_registers, register_set_contains};
use crate::runtime::{Runtime, TableHandle, Value};

/// The magnitude up to which an integer converts to a float and back exactly.
const EXACT_LIMIT: i64 = 1 << 53;

/// The most array slots the recorder looks at to tell whether a table holds
/// numbers of both subtypes.
const MIXED_SCAN: usize = 4096;

/// The instructions after a table read that are searched for the first one
/// to read the result.
const READER_SCAN: usize = 24;

/// An operand of arithmetic or a comparison that may be a number pair.
#[derive(Clone, Copy)]
enum NumOperand {
    /// A number of run-time subtype: its float and its integer tag.
    Pair { value: Operand, tag: Operand },
    /// A float.
    Float(Operand),
    /// An integer, with its value when it is a constant.
    Integer {
        operand: Operand,
        constant: Option<i64>,
    },
}

impl Recorder {
    /// The float and the tag operations of the number `register` holds, if
    /// its subtype is known only at run time.
    pub(super) fn number_pair(&self, register: usize) -> Option<(u32, u32)> {
        if self.number_tags.is_empty() || register >= self.regs.len() {
            return None;
        }
        match self.sym(register) {
            Sym::Op(index) => self.number_tags.get(&index).map(|tag| (index, *tag)),
            _ => None,
        }
    }

    /// Whether `opcode` takes a number of run-time subtype as it is.
    pub(super) const fn takes_number_pair(opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::Add
                | OpCode::Subtract
                | OpCode::Multiply
                | OpCode::Divide
                | OpCode::Modulo
                | OpCode::AddK
                | OpCode::SubtractK
                | OpCode::MultiplyK
                | OpCode::DivideK
                | OpCode::ModuloK
                | OpCode::TestEqual
                | OpCode::TestLessThan
                | OpCode::TestLessEqual
                | OpCode::TestEqualK
                | OpCode::TestLessThanK
                | OpCode::TestLessEqualK
                | OpCode::Move
        )
    }

    /// Narrows every number of run-time subtype that `instruction` reads,
    /// unless it takes one as it is.
    pub(super) fn narrow_numbers_read(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<(), TraceAbortReason> {
        if self.number_tags.is_empty() || !self.frames.is_empty() {
            return Ok(());
        }
        if instruction.opcode().is_ok_and(Self::takes_number_pair) {
            return Ok(());
        }
        let reads = at.chunk.instruction_reads(at.pc);
        for register in 0..self.regs.len().min(256) {
            if self.number_pair(register).is_some()
                && u8::try_from(register).is_ok_and(|register| reads.contains(register))
            {
                self.narrow_number(at, register)?;
            }
        }
        Ok(())
    }

    /// Narrows every number of run-time subtype that the code at `at.pc`
    /// reads again (a loop parameter, a link, the code after an exit).
    pub(super) fn narrow_live_numbers(
        &mut self,
        at: &HookPoint<'_>,
    ) -> Result<(), TraceAbortReason> {
        if self.number_tags.is_empty() || !self.frames.is_empty() {
            return Ok(());
        }
        let live = self
            .liveness
            .get_or_init(|| observable_registers(&self.chunk))
            .get(at.pc)
            .copied();
        for register in 0..self.regs.len().min(256) {
            if self.number_pair(register).is_none() {
                continue;
            }
            let observed = match (live, u8::try_from(register)) {
                (Some(set), Ok(register)) => register_set_contains(&set, register),
                _ => true,
            };
            if observed || register >= self.register_count {
                self.narrow_number(at, register)?;
            }
        }
        Ok(())
    }

    /// Replaces the number of run-time subtype in `register` with one of the
    /// subtype it has now, behind a guard on the tag.
    fn narrow_number(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<(), TraceAbortReason> {
        let Some((value, tag)) = self.number_pair(register) else {
            return Ok(());
        };
        let pc = at.pc;
        let integer = match at.registers.get(register) {
            Some(Value::Integer(_)) => true,
            Some(Value::Number(_)) => false,
            _ => return Err(TraceAbortReason::Divergence),
        };
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(tag),
                expected: integer,
                snapshot: before,
            },
            None,
        );
        let narrowed = if integer {
            self.emit(
                pc,
                RecKind::Numeric(NumericOp::FloatToInteger, Operand::Op(value)),
                Some(TraceType::Integer),
            )
        } else {
            self.emit(
                pc,
                RecKind::Move(Operand::Op(value)),
                Some(TraceType::Float),
            )
        };
        self.write(register, Sym::Op(narrowed));
        Ok(())
    }

    /// Whether the read of the table slot into register `a` at `at.pc` should
    /// give a number of either subtype: the first instruction after it that
    /// reads `a` takes such a number.
    pub(super) fn first_reader_takes_numbers(at: &HookPoint<'_>, a: usize) -> bool {
        let Ok(register) = u8::try_from(a) else {
            return false;
        };
        for pc in at.pc + 1..(at.pc + 1 + READER_SCAN).min(at.chunk.code.len()) {
            let Some(instruction) = at.chunk.code.get(pc) else {
                return false;
            };
            let Ok(opcode) = instruction.opcode() else {
                return false;
            };
            if at.chunk.instruction_reads(pc).contains(register) {
                // The sum, difference and product of two numbers of
                // run-time subtype carry a range check for the integer
                // result, which costs about what the side traces it saves
                // do. A modulo, a division and a comparison carry none.
                return matches!(
                    opcode,
                    OpCode::Modulo
                        | OpCode::ModuloK
                        | OpCode::Divide
                        | OpCode::DivideK
                        | OpCode::TestEqual
                        | OpCode::TestLessThan
                        | OpCode::TestLessEqual
                        | OpCode::TestEqualK
                        | OpCode::TestLessThanK
                        | OpCode::TestLessEqualK
                );
            }
            // A call or a loop ends the straight line the search follows. A
            // forward jump does not: the instructions it skips are searched
            // too, and the join after them is where the number is read.
            if matches!(
                opcode,
                OpCode::Call
                    | OpCode::CallOne
                    | OpCode::CallVararg
                    | OpCode::CallOpen
                    | OpCode::Return
                    | OpCode::ReturnOpen
                    | OpCode::ReturnVararg
                    | OpCode::NumericForLoop
                    | OpCode::ForInLoop
            ) {
                return false;
            }
        }
        false
    }

    /// Whether the array part of `handle` holds both integers and floats, so
    /// a read of it gives numbers of either subtype.
    pub(super) fn array_mixes_numbers(runtime: &Runtime, handle: TableHandle) -> bool {
        let Ok(len) = runtime.table_array_len(handle) else {
            return false;
        };
        let Ok(Some(slots)) = runtime.table_raw_array_range(handle, 1, len.min(MIXED_SCAN)) else {
            return false;
        };
        let (mut integers, mut floats) = (false, false);
        for slot in slots {
            match slot {
                Value::Integer(value) if value.unsigned_abs() <= EXACT_LIMIT.unsigned_abs() => {
                    integers = true;
                }
                Value::Number(_) => floats = true,
                _ => {}
            }
            if integers && floats {
                return true;
            }
        }
        false
    }

    /// The operands of the comparison of `register`: a number of run-time
    /// subtype compares as its float, which is exact.
    pub(super) fn read_for_comparison(
        &self,
        registers: &[Value],
        register: usize,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        match self.number_pair(register) {
            Some((value, _)) => Ok((Operand::Op(value), TraceType::Float)),
            None => self.read(registers, register),
        }
    }

    /// Records a number of run-time subtype loaded from the array slot `key`
    /// names, with its tag, and returns the float's operation.
    pub(super) fn emit_number_load(
        &mut self,
        pc: usize,
        binding: u32,
        (table, key): (Operand, Operand),
        snapshot: u32,
    ) -> u32 {
        let value = self.emit(
            pc,
            RecKind::NumberLoad {
                binding,
                table,
                key,
                snapshot,
            },
            Some(TraceType::Float),
        );
        let tag = self.emit(
            pc,
            RecKind::NumberTag(Operand::Op(value)),
            Some(TraceType::Boolean),
        );
        self.number_tags.insert(value, tag);
        value
    }

    /// Whether a read of the array slot `live` of `handle`, for the
    /// destination register the recorder was told of, gives a number of
    /// either subtype.
    pub(super) fn wants_number_load(
        &mut self,
        (handle, runtime): (TableHandle, &Runtime),
        live: &Value,
    ) -> bool {
        if !std::mem::take(&mut self.number_wanted)
            || self.function.is_some()
            || !self.frames.is_empty()
            || !self.integers
        {
            return false;
        }
        let exact = match live {
            Value::Integer(value) => value.unsigned_abs() <= EXACT_LIMIT.unsigned_abs(),
            Value::Number(_) => true,
            _ => false,
        };
        exact && Self::array_mixes_numbers(runtime, handle)
    }

    fn classify(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<NumOperand, TraceAbortReason> {
        if let Some((value, tag)) = self.number_pair(register) {
            return Ok(NumOperand::Pair {
                value: Operand::Op(value),
                tag: Operand::Op(tag),
            });
        }
        let number = self.read_arithmetic(at, register, None)?;
        Ok(Self::classify_number(number))
    }

    const fn classify_number(number: Number) -> NumOperand {
        match number.live {
            Live::Float(_) => NumOperand::Float(number.operand),
            Live::Integer(_) => NumOperand::Integer {
                operand: number.operand,
                constant: None,
            },
        }
    }

    fn constant_operand(operand: Operand, scalar: TraceScalar) -> Option<NumOperand> {
        match scalar {
            TraceScalar::Float(_) => Some(NumOperand::Float(operand)),
            TraceScalar::Integer(value) => Some(NumOperand::Integer {
                operand,
                constant: Some(value),
            }),
            _ => None,
        }
    }

    /// Records `opcode` on `b` and `c` (or the constant) when one of them is
    /// a number of run-time subtype. Returns the operation that holds the
    /// result, or `None` when neither is one and the caller records the
    /// instruction as it always did.
    pub(super) fn record_number_arithmetic(
        &mut self,
        at: &HookPoint<'_>,
        opcode: OpCode,
        (b, c): (usize, usize),
        constant: Option<(Operand, TraceScalar)>,
    ) -> Result<Option<u32>, TraceAbortReason> {
        let pair_b = self.number_pair(b).is_some();
        let pair_c = constant.is_none() && self.number_pair(c).is_some();
        if !pair_b && !pair_c {
            return Ok(None);
        }
        let pc = at.pc;
        let op = match opcode {
            OpCode::Add => ArithmeticOp::Add,
            OpCode::Subtract => ArithmeticOp::Subtract,
            OpCode::Multiply => ArithmeticOp::Multiply,
            OpCode::Divide => ArithmeticOp::Divide,
            OpCode::Modulo => ArithmeticOp::Modulo,
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let left = self.classify(at, b)?;
        let right = match constant {
            Some((operand, scalar)) => Self::constant_operand(operand, scalar)
                .ok_or(TraceAbortReason::UnsupportedOperand)?,
            None => self.classify(at, c)?,
        };
        let float_op = match op {
            ArithmeticOp::Modulo if self.float_arith.modulo != FloatModulo::Floor => {
                ArithmeticOp::ModuloWith(self.float_arith.modulo)
            }
            op => op,
        };
        // An operation with a float operand, and a division, give a float.
        let is_float = |operand: &NumOperand| matches!(operand, NumOperand::Float(_));
        if op == ArithmeticOp::Divide || is_float(&left) || is_float(&right) {
            let left = self.float_of(pc, left)?;
            let right = self.float_of(pc, right)?;
            let result = self.emit(
                pc,
                RecKind::Arithmetic(float_op, left, right),
                Some(TraceType::Float),
            );
            return Ok(Some(result));
        }
        // Numbers that are integers or floats: the sum, difference, product
        // or modulo of the floats, with the tag `left and right`.
        let (left_value, left_tag) = self.pair_of(pc, left)?;
        let (right_value, right_tag) = self.pair_of(pc, right)?;
        let tag = match (left_tag, right_tag) {
            (None, None) => return Err(TraceAbortReason::Divergence),
            (Some(tag), None) | (None, Some(tag)) => tag,
            (Some(left), Some(right)) => {
                Operand::Op(self.emit(pc, RecKind::BoolAnd(left, right), Some(TraceType::Boolean)))
            }
        };
        let result = if op == ArithmeticOp::Modulo {
            // An integer zero divisor raises an error, which the interpreter
            // does by running the instruction. A float zero divisor leaves
            // too: the guard is the one a comparison with zero before the
            // modulo has made already, and a float zero is rare.
            let zero = self.float_constant(pc, 0.0);
            let is_zero = self.emit(
                pc,
                RecKind::Compare(Comparison::Equal, right_value, zero),
                Some(TraceType::Boolean),
            );
            self.guard_false(pc, is_zero);
            self.emit(
                pc,
                RecKind::NumberModulo(float_op, left_value, right_value, tag),
                Some(TraceType::Float),
            )
        } else {
            self.emit(
                pc,
                RecKind::Arithmetic(float_op, left_value, right_value),
                Some(TraceType::Float),
            )
        };
        if op != ArithmeticOp::Modulo {
            // An integer result stays exact below 2^53 in magnitude. A float
            // result may be anything.
            let magnitude = self.emit(
                pc,
                RecKind::Numeric(NumericOp::Abs, Operand::Op(result)),
                Some(TraceType::Float),
            );
            let limit = self.float_constant(pc, 9_007_199_254_740_992.0);
            let outside = self.emit(
                pc,
                RecKind::Compare(Comparison::LessEqual, limit, Operand::Op(magnitude)),
                Some(TraceType::Boolean),
            );
            let bad = self.emit(
                pc,
                RecKind::BoolAnd(tag, Operand::Op(outside)),
                Some(TraceType::Boolean),
            );
            self.guard_false(pc, bad);
        }
        self.number_tags.insert(
            result,
            match tag {
                Operand::Op(tag) => tag,
                Operand::Carried(_) => return Err(TraceAbortReason::Divergence),
            },
        );
        Ok(Some(result))
    }

    /// A guard that leaves before the instruction when `condition` is true.
    fn guard_false(&mut self, pc: usize, condition: u32) {
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(condition),
                expected: false,
                snapshot: before,
            },
            None,
        );
    }

    fn float_constant(&mut self, pc: usize, value: f64) -> Operand {
        Operand::Op(self.emit(
            pc,
            RecKind::Constant(TraceScalar::Float(value)),
            Some(TraceType::Float),
        ))
    }

    /// `operand` as a float operand.
    fn float_of(&mut self, pc: usize, operand: NumOperand) -> Result<Operand, TraceAbortReason> {
        Ok(match operand {
            NumOperand::Pair { value, .. } => value,
            NumOperand::Float(operand) => operand,
            NumOperand::Integer { operand, .. } => {
                Operand::Op(self.emit(pc, RecKind::IntToFloat(operand), Some(TraceType::Float)))
            }
        })
    }

    /// `operand` as a float and, when its subtype is not known to be
    /// integer, the tag. An integer operand is a float that is exact.
    fn pair_of(
        &mut self,
        pc: usize,
        operand: NumOperand,
    ) -> Result<(Operand, Option<Operand>), TraceAbortReason> {
        match operand {
            NumOperand::Pair { value, tag } => Ok((value, Some(tag))),
            NumOperand::Float(_) => Err(TraceAbortReason::Divergence),
            NumOperand::Integer {
                operand,
                constant: Some(value),
            } => {
                if value.unsigned_abs() > EXACT_LIMIT.unsigned_abs() {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                Ok((
                    self.float_of(
                        pc,
                        NumOperand::Integer {
                            operand,
                            constant: Some(value),
                        },
                    )?,
                    None,
                ))
            }
            NumOperand::Integer { operand, .. } => {
                let value = self.float_of(
                    pc,
                    NumOperand::Integer {
                        operand,
                        constant: None,
                    },
                )?;
                // An integer beyond 2^53 does not convert exactly.
                let magnitude = self.emit(
                    pc,
                    RecKind::Numeric(NumericOp::Abs, value),
                    Some(TraceType::Float),
                );
                let limit = self.float_constant(pc, 9_007_199_254_740_992.0);
                let outside = self.emit(
                    pc,
                    RecKind::Compare(Comparison::LessEqual, limit, Operand::Op(magnitude)),
                    Some(TraceType::Boolean),
                );
                self.guard_false(pc, outside);
                Ok((value, None))
            }
        }
    }
}
