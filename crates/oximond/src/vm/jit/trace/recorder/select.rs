//! Value selects for pure two-arm branches.
//!
//! `ssum + (walkable_p (cid) and 1.0 or 0.0) * s` branches on data: the
//! boolean goes one way on one voxel and the other way on the next. A branch
//! guard there fails on every other iteration, and each failure is an exit
//! (or a side trace holding the rest of the iteration). Both arms of that
//! branch only load a constant into the same register and meet again two
//! instructions later, so the recorder records both and picks the value with
//! a [`RecKind::Select`] ([`oximond::jit_ir::trace::TraceOp::Select`])
//! instead of guarding the direction.
//!
//! At a branch on a boolean (`JumpIfFalse` of a boolean register, or a
//! compare-and-branch test), the recorder decodes both successors from the
//! chunk's code, the arm the interpreter will not run included. An arm may
//! hold only:
//!
//! - `LoadConstant` of a number, a boolean or nil, and `LoadNil`;
//! - `Move` and `Not`;
//! - `Negate`, `+`, `-`, `*` and `/` (and their constant forms) on integers
//!   and floats, which are total on those types: integers wrap, and an
//!   integer `/` is a float division. `%` and `//` are not, since an integer
//!   zero divisor raises an error;
//! - `JumpIfFalse` on a value whose truthiness the arm knows (the branch's
//!   own condition, a constant, a nil, or a value of a type that is always
//!   true), and `Jump`. Both only forward.
//!
//! Both successors of the branch lie after it (a backward one is a loop's
//! edge, and its arm would walk the loop body). The arms are walked in step,
//! the one at the lower pc first. Since every jump is forward, the first pc
//! both reach is where they join. The
//! `p and 1.0 or 0.0` shape is `JumpIfFalse d; LoadConstant d 1.0;
//! JumpIfFalse d; Jump end; LoadConstant d 0.0`: the true arm's second
//! `JumpIfFalse` tests a float (always true), the false arm's tests the
//! condition again (false there), and both reach `end` after one load. An
//! arm that meets any other instruction before the join, or takes more than
//! [`MAX_ARM_INSTRUCTIONS`], declines the select and the branch keeps its
//! guard: nothing the decode emitted stays.
//!
//! Every register either arm writes then holds, after the join, the value of
//! the true arm or the false arm (a register only one arm writes keeps its
//! value from before the branch on the other). Two arms that agree need no
//! select; two values of one integer, float or boolean type take one. Any
//! other pair (a nil against a number, two tables) declines the select.
//!
//! This is exact: each arm's operations are pure and total on the types the
//! recording already guards, so running the arm the interpreter skipped
//! changes nothing it could observe, and the select gives every register
//! the value the interpreter's arm gives it. There is no guard and no
//! snapshot, so nothing can leave between the branch and the join.
//!
//! The types the recording guards include those of carried registers the
//! arms decide on without an op reading them: the truthiness an arm's
//! `JumpIfFalse` takes from a carried nil or number, and the type two
//! different carried values join on. A select that succeeds puts those
//! registers in `condition_reads`, so the entry contract guards their types.
//!
//! While the interpreter runs the arm it takes, the recorder follows its
//! pcs without recording them again (see [`SelectSkip`]); the next hook
//! that differs from the decoded path is a divergence, as anywhere else.
//! Count hooks need the count of the arm that ran, which a select does not
//! know, so a recording that counts keeps its guards. So does the region in
//! the root frame's caller.

use super::{
    ArithmeticOp, HookPoint, Instruction, OpCode, Operand, PrevPc, RecKind, Recorder, Sym,
    TraceScalar, TraceType, short_offset,
};

/// The most instructions an arm of a value select may run before the join.
pub(super) const MAX_ARM_INSTRUCTIONS: usize = 6;

/// The rest of the arm the interpreter runs after a value select: the pcs
/// its hooks land on, the join last. The recorder recorded them already.
#[derive(Clone, Debug)]
pub(super) struct SelectSkip {
    pub(super) path: Vec<usize>,
    pub(super) next: usize,
}

/// A register's value inside an arm, and its truthiness when the arm knows
/// it.
#[derive(Clone, Copy, Debug)]
struct ArmValue {
    operand: Operand,
    ty: TraceType,
    truth: Option<bool>,
}

/// One arm being decoded: the pc it is at, the pcs it ran, and the registers
/// it wrote (the last write of each wins).
struct Arm {
    pc: usize,
    path: Vec<usize>,
    writes: Vec<(usize, ArmValue)>,
    /// Carried registers whose entry type decided a `JumpIfFalse` of the
    /// arm.
    type_reads: Vec<usize>,
    /// What the branch condition is in this arm.
    truth: bool,
}

impl Arm {
    fn new(pc: usize, truth: bool) -> Self {
        Self {
            pc,
            path: Vec::new(),
            writes: Vec::new(),
            type_reads: Vec::new(),
            truth,
        }
    }

    fn written(&self, register: usize) -> Option<ArmValue> {
        self.writes
            .iter()
            .rev()
            .find(|(written, _)| *written == register)
            .map(|(_, value)| *value)
    }

    fn set(&mut self, register: usize, value: ArmValue) {
        self.writes.push((register, value));
    }
}

impl Recorder {
    /// Records the branch at `pc` on the boolean `condition` as a value
    /// select when both arms are pure and join (see the module docs).
    /// `on_true` and `on_false` are the successors for each value of the
    /// condition, and `truthy` the value it has now. Returns the pc the next
    /// hook lands on, or `None` when the branch needs its guard; then nothing
    /// was recorded.
    pub(super) fn value_select(
        &mut self,
        at: &HookPoint<'_>,
        condition: Operand,
        (on_true, on_false): (usize, usize),
        truthy: bool,
    ) -> Option<usize> {
        // A backward successor is a loop's: the arms would walk its body.
        if !self.value_selects
            || self.counting
            || self.running_caller()
            || self.open.is_some()
            || on_true == on_false
            || on_true <= at.pc
            || on_false <= at.pc
        {
            return None;
        }
        let mark = self.ops.len();
        let selected = self.decode_select(at, condition, on_true, on_false, truthy);
        if selected.is_none() {
            self.ops.truncate(mark);
        }
        selected
    }

    fn decode_select(
        &mut self,
        at: &HookPoint<'_>,
        condition: Operand,
        on_true: usize,
        on_false: usize,
        truthy: bool,
    ) -> Option<usize> {
        let mut arms = [Arm::new(on_true, true), Arm::new(on_false, false)];
        // Every jump goes forward, so the arm behind can still reach the
        // other's pc and the one ahead cannot come back: the first pc both
        // stand on is the first both reach. The arms join there when their
        // registers merge; otherwise (the `and` of `p and 1.0 or 0.0` meets
        // its `or` with a boolean in one arm and a float in the other) both
        // go on, each its own way.
        let joined = loop {
            let behind = if arms[0].pc == arms[1].pc {
                if let Some(joined) = self.merge_arms(at, condition, &arms) {
                    break joined;
                }
                0
            } else {
                usize::from(arms[1].pc < arms[0].pc)
            };
            if arms[behind].path.len() >= MAX_ARM_INSTRUCTIONS {
                return None;
            }
            self.arm_step(at, condition, &mut arms[behind])?;
        };
        let (joined, merge_reads) = joined;
        let join = arms[0].pc;
        // The arms stay inside the loop the recording follows.
        if self.in_home() && !self.outer_side {
            let inside = |pc: &usize| (self.header..=self.latch).contains(pc);
            if !inside(&join) || !arms.iter().all(|arm| arm.path.iter().all(inside)) {
                return None;
            }
        }
        // The interpreter's previous pc at the join is the last instruction
        // of the arm it ran, which a snapshot there cannot know: it names the
        // recorded arm's. The interpreter replaces it before it runs the join,
        // and until then only the debug information reads it, so both arms
        // must end on one line with the same locals live.
        let last = |arm: &Arm| arm.path.last().copied().unwrap_or(at.pc);
        let (true_last, false_last) = (last(&arms[0]), last(&arms[1]));
        let debug = &at.chunk.debug;
        if debug.line_at(true_last) != debug.line_at(false_last)
            || !debug
                .locals_at(true_last)
                .map(std::ptr::from_ref)
                .eq(debug.locals_at(false_last).map(std::ptr::from_ref))
        {
            return None;
        }
        // A carried register whose type decided a direction or a join is a
        // read the trace keeps: the entry contract guards its type.
        for arm in &arms {
            self.condition_reads.extend_from_slice(&arm.type_reads);
        }
        self.condition_reads.extend(merge_reads);
        for (register, if_true, if_false) in joined {
            let operand = if if_true.operand == if_false.operand || if_true.ty == TraceType::Nil {
                if_true.operand
            } else {
                Operand::Op(self.emit(
                    at.pc,
                    RecKind::Select(condition, if_true.operand, if_false.operand),
                    Some(if_true.ty),
                ))
            };
            match operand {
                // Unchanged.
                Operand::Carried(source) if source == register => {}
                Operand::Carried(_) => {
                    let value = self.emit(at.pc, RecKind::Move(operand), Some(if_true.ty));
                    self.write(register, value);
                }
                Operand::Op(value) => self.write(register, Sym::Op(value)),
            }
        }
        let taken = &arms[usize::from(!truthy)];
        let Some((&first, rest)) = taken.path.split_first() else {
            return Some(join);
        };
        let mut path = rest.to_vec();
        path.push(join);
        self.select_skip = Some(SelectSkip { path, next: 0 });
        Some(first)
    }

    /// The value of every register either arm wrote, in the arm where the
    /// condition is true and in the other, when each pair can join: equal,
    /// both nil, or two values of one integer, float or boolean type.
    #[allow(clippy::type_complexity)]
    fn merge_arms(
        &self,
        at: &HookPoint<'_>,
        condition: Operand,
        arms: &[Arm; 2],
    ) -> Option<(Vec<(usize, ArmValue, ArmValue)>, Vec<usize>)> {
        let mut registers: Vec<usize> = arms
            .iter()
            .flat_map(|arm| arm.writes.iter().map(|(register, _)| *register))
            .collect();
        registers.sort_unstable();
        registers.dedup();
        let value = |arm: &Arm, register: usize| {
            arm.written(register)
                .or_else(|| self.base_value(at, condition, arm.truth, register))
        };
        let mut type_reads = Vec::new();
        let joined = registers
            .into_iter()
            .map(|register| {
                let if_true = value(&arms[0], register)?;
                let if_false = value(&arms[1], register)?;
                let joins = if_true.ty == if_false.ty
                    && (if_true.operand == if_false.operand
                        || matches!(
                            if_true.ty,
                            TraceType::Nil
                                | TraceType::Integer
                                | TraceType::Float
                                | TraceType::Boolean
                        ));
                // Two different values join on their types; a carried one
                // has its type from the entry contract only if it is an
                // input there. One value in both arms needs no type.
                if joins && if_true.operand != if_false.operand {
                    for arm_value in [if_true, if_false] {
                        if let Operand::Carried(carried) = arm_value.operand {
                            type_reads.push(carried);
                        }
                    }
                }
                joins.then_some((register, if_true, if_false))
            })
            .collect::<Option<Vec<_>>>()?;
        Some((joined, type_reads))
    }

    /// Follows a hook of the arm a value select recorded already: the
    /// interpreter runs it, the trace has its values. Returns the pc the
    /// next hook lands on.
    pub(super) fn skip_selected(&mut self, pc: usize) -> Option<usize> {
        let skip = self.select_skip.as_mut()?;
        let next = skip.path[skip.next];
        skip.next += 1;
        if skip.next == skip.path.len() {
            self.select_skip = None;
        }
        self.set_prev(PrevPc::At(pc));
        Some(next)
    }

    /// A register as the state before the branch holds it, seen from an arm
    /// where the condition is `truth`.
    fn base_value(
        &self,
        at: &HookPoint<'_>,
        condition: Operand,
        truth: bool,
        register: usize,
    ) -> Option<ArmValue> {
        // A run-time subtype, a string constant or a virtual table: not a
        // value an arm can hold.
        if !matches!(self.sym(register), Sym::Carried | Sym::Op(_)) {
            return None;
        }
        let (operand, ty) = self.read(at.registers, register).ok()?;
        let known = if operand == condition {
            Some(truth)
        } else {
            self.truth_of(operand, ty)
        };
        Some(ArmValue {
            operand,
            ty,
            truth: known,
        })
    }

    /// The truthiness of a value whenever the trace runs, if its type or
    /// its constant decides it.
    fn truth_of(&self, operand: Operand, ty: TraceType) -> Option<bool> {
        match ty {
            TraceType::Nil => Some(false),
            TraceType::Boolean => match operand {
                Operand::Op(index) => match self.ops[index as usize].kind {
                    RecKind::Constant(TraceScalar::Boolean(flag)) => Some(flag),
                    _ => None,
                },
                Operand::Carried(_) => None,
            },
            TraceType::Any => None,
            _ => Some(true),
        }
    }

    fn arm_read(
        &self,
        at: &HookPoint<'_>,
        condition: Operand,
        arm: &Arm,
        register: usize,
    ) -> Option<ArmValue> {
        self.check_register(register).ok()?;
        arm.written(register)
            .or_else(|| self.base_value(at, condition, arm.truth, register))
    }

    /// Decodes the instruction at the arm's pc, emitting its operations, and
    /// moves the arm on. `None` for an instruction an arm may not hold.
    fn arm_step(&mut self, at: &HookPoint<'_>, condition: Operand, arm: &mut Arm) -> Option<()> {
        let pc = arm.pc;
        let instruction = *at.chunk.code.get(pc)?;
        let opcode = instruction.opcode().ok()?;
        let a = usize::from(instruction.a());
        let b = usize::from(instruction.b());
        // The boxed control register's writes are helper calls.
        let writes = |this: &Self| {
            this.check_register(a).is_ok() && !(this.frames.is_empty() && this.boxed == Some(a))
        };
        let next = match opcode {
            OpCode::LoadConstant | OpCode::LoadNil => {
                if !writes(self) {
                    return None;
                }
                let scalar = if opcode == OpCode::LoadNil {
                    TraceScalar::Nil
                } else {
                    self.constant(at.chunk, instruction).ok()?
                };
                if !matches!(
                    scalar,
                    TraceScalar::Nil
                        | TraceScalar::Boolean(_)
                        | TraceScalar::Integer(_)
                        | TraceScalar::Float(_)
                ) {
                    return None;
                }
                let value = self.emit(pc, RecKind::Constant(scalar), Some(scalar.ty()));
                arm.set(
                    a,
                    ArmValue {
                        operand: Operand::Op(value),
                        ty: scalar.ty(),
                        truth: Some(scalar.is_truthy()),
                    },
                );
                pc + 1
            }
            OpCode::Move => {
                if !writes(self) {
                    return None;
                }
                let source = self.arm_read(at, condition, arm, b)?;
                arm.set(a, source);
                pc + 1
            }
            OpCode::Not => {
                if !writes(self) {
                    return None;
                }
                let source = self.arm_read(at, condition, arm, b)?;
                let value = self.emit(pc, RecKind::Not(source.operand), Some(TraceType::Boolean));
                arm.set(
                    a,
                    ArmValue {
                        operand: Operand::Op(value),
                        ty: TraceType::Boolean,
                        truth: source.truth.map(|truth| !truth),
                    },
                );
                pc + 1
            }
            OpCode::Negate => {
                if !writes(self) {
                    return None;
                }
                let source = self.arm_read(at, condition, arm, b)?;
                if !matches!(source.ty, TraceType::Integer | TraceType::Float) {
                    return None;
                }
                let value = self.emit(pc, RecKind::Negate(source.operand), Some(source.ty));
                arm.set(a, number(value, source.ty));
                pc + 1
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK => {
                if !writes(self) {
                    return None;
                }
                self.arm_arithmetic(at, condition, arm, instruction, opcode)?;
                pc + 1
            }
            OpCode::JumpIfFalse => {
                let tested = self.arm_read(at, condition, arm, a)?;
                let truth = tested.truth?;
                // A carried value's truthiness comes from its entry type
                // (the branch condition's from the branch).
                if let Operand::Carried(carried) = tested.operand
                    && tested.operand != condition
                {
                    arm.type_reads.push(carried);
                }
                if truth {
                    pc + 1
                } else {
                    self.target(pc, short_offset(instruction)).ok()?
                }
            }
            OpCode::Jump => self.target(pc, instruction.jump_offset()).ok()?,
            _ => return None,
        };
        if next <= pc {
            return None;
        }
        arm.path.push(pc);
        arm.pc = next;
        Some(())
    }

    /// `+`, `-`, `*` and `/` in an arm, on integers and floats.
    fn arm_arithmetic(
        &mut self,
        at: &HookPoint<'_>,
        condition: Operand,
        arm: &mut Arm,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Option<()> {
        let pc = arm.pc;
        let a = usize::from(instruction.a());
        let left = self.arm_read(at, condition, arm, usize::from(instruction.b()))?;
        let (opcode, right) = match opcode.register_form() {
            Some(register_opcode) => {
                let scalar = self
                    .constant_at(at.chunk, usize::from(instruction.c()))
                    .ok()?;
                if !matches!(scalar, TraceScalar::Integer(_) | TraceScalar::Float(_)) {
                    return None;
                }
                let constant = self.emit(pc, RecKind::Constant(scalar), Some(scalar.ty()));
                (register_opcode, number(constant, scalar.ty()))
            }
            None => (
                opcode,
                self.arm_read(at, condition, arm, usize::from(instruction.c()))?,
            ),
        };
        let numbers = |value: &ArmValue| matches!(value.ty, TraceType::Integer | TraceType::Float);
        if !numbers(&left) || !numbers(&right) {
            return None;
        }
        let op = match opcode {
            OpCode::Add => ArithmeticOp::Add,
            OpCode::Subtract => ArithmeticOp::Subtract,
            OpCode::Multiply => ArithmeticOp::Multiply,
            OpCode::Divide => ArithmeticOp::Divide,
            _ => return None,
        };
        let integers = left.ty == TraceType::Integer && right.ty == TraceType::Integer;
        let value = if integers && op != ArithmeticOp::Divide {
            self.emit(
                pc,
                RecKind::Arithmetic(op, left.operand, right.operand),
                Some(TraceType::Integer),
            )
        } else {
            // A float operand, or a division: both sides as floats.
            let mut float = |value: ArmValue| {
                if value.ty == TraceType::Float {
                    value.operand
                } else {
                    Operand::Op(self.emit(
                        pc,
                        RecKind::IntToFloat(value.operand),
                        Some(TraceType::Float),
                    ))
                }
            };
            let (left, right) = (float(left), float(right));
            self.emit(
                pc,
                RecKind::Arithmetic(op, left, right),
                Some(TraceType::Float),
            )
        };
        let ty = if integers && op != ArithmeticOp::Divide {
            TraceType::Integer
        } else {
            TraceType::Float
        };
        arm.set(a, number(value, ty));
        Some(())
    }
}

/// A number an arm computed: always true.
const fn number(value: u32, ty: TraceType) -> ArmValue {
    ArmValue {
        operand: Operand::Op(value),
        ty,
        truth: Some(true),
    }
}

#[cfg(test)]
mod value_select_tests {
    use super::super::{HookPoint, HookSource, Recorder, Step};
    use crate::bytecode::{Chunk, Constant, Instruction, OpCode};
    use crate::compat::CompatibilityProfile;
    use crate::jit_ir::trace::{TraceId, TraceIr, TraceOp};
    use crate::runtime::Value;
    use crate::vm::jit::trace::TraceAbortReason;
    use std::sync::Arc;

    fn hook<'a>(chunk: &'a Arc<Chunk>, pc: usize, registers: &'a [Value]) -> HookPoint<'a> {
        HookPoint {
            chunk,
            depth: 1,
            top_level: true,
            pc,
            registers,
            open: &[],
            coroutine: None,
            hooks_active: false,
            source: HookSource::General,
            surroundings: None,
        }
    }

    fn jump(offset: i32) -> Instruction {
        Instruction::aj(OpCode::Jump, offset).expect("a short jump fits")
    }

    /// `r2 = r2 + (r0 and 1.0 or 0.0)` in a loop closed by a jump back to
    /// pc 0, as the compiler emits the `and`/`or`:
    ///
    /// ```text
    /// 0 Move r1 r0
    /// 1 JumpIfFalse r1 -> 3
    /// 2 LoadConstant r1 1.0
    /// 3 JumpIfFalse r1 -> 5
    /// 4 Jump -> 6
    /// 5 LoadConstant r1 0.0
    /// 6 Add r2 r2 r1
    /// 7 Jump -> 0
    /// ```
    fn and_or(true_arm: Instruction) -> Arc<Chunk> {
        Arc::new(Chunk::new(
            4,
            vec![Constant::Number(1.0), Constant::Number(0.0)],
            vec![
                Instruction::move_(1, 0),
                Instruction::jump_if_false(1, 1),
                true_arm,
                Instruction::jump_if_false(1, 1),
                jump(1),
                Instruction::load_constant(1, 1),
                Instruction::binary(OpCode::Add, 2, 2, 1),
                jump(-8),
            ],
        ))
    }

    fn recorder(chunk: &Arc<Chunk>) -> Recorder {
        Recorder::new(
            Arc::clone(chunk),
            1,
            0,
            7,
            None,
            CompatibilityProfile::Lua51,
        )
    }

    /// Drives the recorder along `path`, the pcs the interpreter runs, with
    /// the registers it holds before each, and returns the closed trace.
    fn record(chunk: &Arc<Chunk>, path: &[(usize, Vec<Value>)]) -> Result<TraceIr, Step> {
        let mut recorder = recorder(chunk);
        for (pc, registers) in path {
            match recorder.step(&hook(chunk, *pc, registers), TraceId(0)) {
                Step::Recorded => {}
                Step::Closed(closed) => return Ok(closed.ir),
                other => return Err(other),
            }
        }
        panic!("the loop did not close");
    }

    fn registers(condition: bool, r1: Value, sum: f64) -> Vec<Value> {
        vec![
            Value::Boolean(condition),
            r1,
            Value::Number(sum),
            Value::Nil,
        ]
    }

    /// The pcs and registers of one iteration that takes the true arm.
    fn true_path() -> Vec<(usize, Vec<Value>)> {
        vec![
            (0, registers(true, Value::Nil, 0.0)),
            (1, registers(true, Value::Boolean(true), 0.0)),
            (2, registers(true, Value::Boolean(true), 0.0)),
            (3, registers(true, Value::Number(1.0), 0.0)),
            (4, registers(true, Value::Number(1.0), 0.0)),
            (6, registers(true, Value::Number(1.0), 0.0)),
            (7, registers(true, Value::Number(1.0), 1.0)),
            (0, registers(true, Value::Number(1.0), 1.0)),
        ]
    }

    fn count(ir: &TraceIr, wanted: fn(&TraceOp) -> bool) -> usize {
        ir.body
            .iter()
            .filter(|instruction| wanted(&instruction.op))
            .count()
    }

    #[test]
    fn a_value_select_decodes_the_and_or_arm_the_interpreter_skips() {
        let chunk = and_or(Instruction::load_constant(1, 0));
        let ir = record(&chunk, &true_path()).unwrap_or_else(|_| panic!("closes"));
        assert_eq!(
            count(&ir, |op| matches!(op, TraceOp::Select { .. })),
            1,
            "{:?}",
            ir.body
        );
        assert_eq!(
            count(&ir, |op| matches!(op, TraceOp::GuardBranch { .. })),
            0,
            "{:?}",
            ir.body
        );
    }

    #[test]
    fn a_value_select_follows_the_false_arm_too() {
        let chunk = and_or(Instruction::load_constant(1, 0));
        let path = vec![
            (0, registers(false, Value::Nil, 0.0)),
            (1, registers(false, Value::Boolean(false), 0.0)),
            (3, registers(false, Value::Boolean(false), 0.0)),
            (5, registers(false, Value::Boolean(false), 0.0)),
            (6, registers(false, Value::Number(0.0), 0.0)),
            (7, registers(false, Value::Number(0.0), 0.0)),
            (0, registers(false, Value::Number(0.0), 0.0)),
        ];
        let ir = record(&chunk, &path).unwrap_or_else(|_| panic!("closes"));
        assert_eq!(count(&ir, |op| matches!(op, TraceOp::Select { .. })), 1);
        assert_eq!(
            count(&ir, |op| matches!(op, TraceOp::GuardBranch { .. })),
            0
        );
    }

    #[test]
    fn a_value_select_arm_that_reads_a_table_keeps_the_guard() {
        // The true arm reads `r3[r3]`, which may run `__index`.
        let chunk = and_or(Instruction::get_table(1, 3, 3));
        let mut recorder = recorder(&chunk);
        let id = TraceId(0);
        assert!(matches!(
            recorder.step(&hook(&chunk, 0, &registers(true, Value::Nil, 0.0)), id),
            Step::Recorded
        ));
        assert!(matches!(
            recorder.step(
                &hook(&chunk, 1, &registers(true, Value::Boolean(true), 0.0)),
                id
            ),
            Step::Recorded
        ));
        assert!(recorder.select_skip.is_none());
        assert!(
            recorder
                .ops
                .iter()
                .any(|op| matches!(op.kind, super::RecKind::GuardBranch { .. })),
            "{:?}",
            recorder.ops
        );
        assert!(
            !recorder
                .ops
                .iter()
                .any(|op| matches!(op.kind, super::RecKind::Select(..)))
        );
    }

    #[test]
    fn a_hook_off_the_decoded_arm_is_a_divergence() {
        let chunk = and_or(Instruction::load_constant(1, 0));
        let mut path = true_path();
        // The interpreter lands on the false arm's load instead of pc 2.
        path[2].0 = 5;
        assert!(matches!(
            record(&chunk, &path),
            Err(Step::Abort(TraceAbortReason::Divergence))
        ));
    }

    #[test]
    fn a_value_select_needs_both_arms_of_one_type() {
        // `if r0 then r1 = true else r1 = 0.0 end`: the arms join at pc 4 with
        // a boolean in one and a float in the other, and the arm that goes on
        // meets the loop's backward jump.
        let chunk = Arc::new(Chunk::new(
            4,
            vec![Constant::Boolean(true), Constant::Number(0.0)],
            vec![
                Instruction::jump_if_false(0, 2),
                Instruction::load_constant(1, 0),
                jump(1),
                Instruction::load_constant(1, 1),
                jump(-5),
            ],
        ));
        let mut recorder = Recorder::new(
            Arc::clone(&chunk),
            1,
            0,
            4,
            None,
            CompatibilityProfile::Lua51,
        );
        assert!(matches!(
            recorder.step(
                &hook(&chunk, 0, &registers(true, Value::Nil, 0.0)),
                TraceId(0)
            ),
            Step::Recorded
        ));
        assert!(recorder.select_skip.is_none());
        assert!(
            recorder
                .ops
                .iter()
                .any(|op| matches!(op.kind, super::RecKind::GuardBranch { .. }))
        );
        assert!(
            !recorder
                .ops
                .iter()
                .any(|op| matches!(op.kind, super::RecKind::Select(..)))
        );
    }

    #[test]
    fn an_arm_that_rewrites_the_condition_with_its_own_value_is_a_select() {
        // `r1 = r0 and true or 0.0` meets its `or` with `true` in one arm and
        // the condition, `false`, in the other: both booleans, so the arms
        // join there with a select, and the `or` keeps its guard.
        let chunk = Arc::new(Chunk::new(
            4,
            vec![Constant::Boolean(true), Constant::Number(0.0)],
            and_or(Instruction::load_constant(1, 0)).code.clone(),
        ));
        let mut recorder = recorder(&chunk);
        let id = TraceId(0);
        recorder.step(&hook(&chunk, 0, &registers(true, Value::Nil, 0.0)), id);
        recorder.step(
            &hook(&chunk, 1, &registers(true, Value::Boolean(true), 0.0)),
            id,
        );
        assert!(
            recorder
                .ops
                .iter()
                .any(|op| matches!(op.kind, super::RecKind::Select(..)))
        );
        assert!(recorder.select_skip.is_some());
    }
}
