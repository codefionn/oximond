//! Operator metamethods as inlined Lua calls: arithmetic and bitwise
//! operators, `-x` and `~x`, `..`, `==`, `<` and `<=`, `tostring` and
//! `pairs`.
//!
//! When an operand is a table whose metatable names a Lua closure for the
//! event, the interpreter calls it, and the recorder follows it into the
//! closure like any other inlined call (see `enter_metamethod`). A helper
//! stands where a call has its callee guard ([`HelperId::OPERATOR_HANDLER`]).
//! It makes the interpreter's choice of handler at run time, with the rule the
//! profile has for the event, and checks that it is the pinned closure. The
//! check is about the class, not the objects, so one trace serves every
//! operand whose metatable names that closure, on either side of the
//! operator, mixed with numbers and strings.
//!
//! - When every operand is a table with a metatable, a number, or a string
//!   whose metatable lacks the event, the helper is replaced by metatable
//!   classes (see `Recorder::guard_operator_class`): each table operand is
//!   held to the metatable it has now with a [`TraceOp::MetaGuard`], and
//!   entry checks that the metatable holds the pinned closure (or nothing)
//!   for the event, so the rule picks the handler from the classes alone.
//!   Anything else keeps the helper.
//! - The closure gets the operands as the interpreter passes them: left, then
//!   right (`-x` passes the operand twice, `~x` once, as the interpreter
//!   does). `__le` answered by the
//!   `__lt` of the swapped operands (every profile but Lua 5.5) passes them
//!   swapped, and negates the answer. A guard before it checks that no `__le`
//!   answers.
//! - An operator returns the closure's first value (nil if it returns none)
//!   into the register the instruction writes. A comparison takes the truth
//!   of that value (negated for the `__lt` fallback), as a boolean in a
//!   register or as the branch of a compare-and-branch test, which guards its
//!   direction like any other test. `__eq` runs only for two tables that are
//!   not the same table, and a guard checks that.
//! - A `..` chain folds from the right as the interpreter does: runs of
//!   strings and numbers are one concatenation, and each pair with a table is
//!   one call of the handler, whose first value is the next right operand.
//!   A snapshot inside the call rebuilds the interpreter's range
//!   continuation: the operands left of the pair are the registers of the
//!   caller (the instruction leaves them alone), and the return supplies the
//!   accumulator. A step that follows a handler leaves through the snapshot
//!   before the instruction, which would run a handler that stored again, so
//!   a chain goes on after a handler only while nothing has committed an
//!   effect since it began; otherwise the recording is refused.
//! - `tostring (t)` with a `__tostring` closure is a call of it whose return
//!   is the string the call gives: a string in the profiles that check, any
//!   value in the Lua 5.1 family. A number, which the interpreter formats, or
//!   anything else, which it raises for, leaves the loop to the interpreter.
//! - `pairs (t)` with a `__pairs` closure (every profile but the Lua 5.1
//!   family) is a call of that closure with the table alone, whose results are
//!   the `pairs` call's. The loop that follows runs the iterator it returns.
//!
//! A handler that is not a Lua closure (a native or builtin function, a
//! callable table) leaves the operation to the code that records it without
//! one, and the abort names the function or the metamethod among the native
//! aborts of the statistics.

use super::inline_meta::Extra;
use super::{
    Entry, Flow, HelperId, HookPoint, Open, Operand, Pending, RecKind, Receiver, Recorder, Sym,
    TraceAbortReason, TraceScalar, TraceType,
};
use crate::bytecode::{Instruction, OpCode};
use crate::jit_ir::trace::pinned_builtin_identity;
use crate::jit_ir::trace::{META_BARE_NUMBERS, META_BARE_STRINGS, META_PIN_ABSENT};
use crate::jit_ir::trace::{PINNED_BUILTIN_PAIRS, PINNED_BUILTIN_TOSTRING, SnapshotPhase};
use crate::runtime::{ClosureHandle, Runtime, Value};
use crate::string::LuaString;
use crate::vm::arith::{metamethod_key, operator_event, scalar_metamethod};
use crate::vm::jit::trace::helpers::{OPERATOR_NO_HANDLER, operator_handler};

/// Handler rules of [`HelperId::OPERATOR_HANDLER`].
mod rule {
    /// The left operand's handler, unless it passes the operation on.
    pub(super) const ARITHMETIC: i64 = 0;
    /// The left operand's handler, else the right one's.
    pub(super) const FIRST: i64 = 1;
    /// The handler both operands share.
    pub(super) const SHARED: i64 = 2;
    /// The handler both share, for operands of one type.
    pub(super) const SAME_TYPE: i64 = 3;
}

/// One operand of an operator: the live value and the trace's symbol for it.
#[derive(Clone, Debug)]
struct Side {
    live: Value,
    operand: Operand,
    ty: TraceType,
}

impl Side {
    fn pair(&self) -> (Operand, TraceType) {
        (self.operand, self.ty)
    }
}

/// An operand of a `..` chain, folded from the right.
#[derive(Clone, Debug)]
pub(super) struct Piece {
    side: Side,
}

/// A `..` that ran a handler: what is left to fold when the handler returns.
#[derive(Debug)]
pub(super) struct ConcatChain {
    /// The `Concat` instruction.
    pc: usize,
    /// The register the chain's result goes to.
    destination: usize,
    /// The registers of the first and last operand.
    first: usize,
    last: usize,
    /// The operands left of the pair the handler is running for, in order.
    rest: Vec<Piece>,
    /// The table stores and operations the recording held when the chain
    /// began, to tell whether a handler has committed an effect since.
    start: (usize, usize),
    /// Whether a handler of the chain has returned: the chain goes on from
    /// the callee's return rather than from the instruction.
    returned: bool,
}

/// What a value is to `..`: a string or a number, which the VM concatenates.
fn text_like(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::Integer(_) | Value::Number(_)
    )
}

/// What a value may be as an operand of an operator the recorder follows to
/// a handler.
fn operand_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Table(_) | Value::String(_) | Value::Integer(_) | Value::Number(_)
    )
}

impl Recorder {
    /// Records the operator at `at.pc` as a call of a Lua closure handler, if
    /// one runs: the flow of the entered callee. `None` leaves the
    /// instruction to the code that records it without a handler.
    pub(super) fn record_operator_metamethod(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let Ok(opcode) = instruction.opcode() else {
            return Ok(None);
        };
        match opcode {
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
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => self.record_binary_handler(at, instruction, opcode),
            OpCode::Negate | OpCode::BitNot => self.record_unary_handler(at, instruction, opcode),
            OpCode::Concat => self.record_concat_handler(at, instruction),
            OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK => self.record_comparison_handler(at, instruction, opcode),
            _ => Ok(None),
        }
    }

    /// The runtime, if the surroundings are known.
    fn operator_runtime<'a>(at: &HookPoint<'a>) -> Option<&'a Runtime> {
        at.surroundings.map(|around| around.runtime)
    }

    /// The operand in `register` as a side of an operator, or `None` for a
    /// value no handler is looked up for here.
    fn operator_side(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<Option<Side>, TraceAbortReason> {
        let Some(live) = at.registers.get(register) else {
            return Ok(None);
        };
        if !operand_like(live) {
            return Ok(None);
        }
        let (operand, ty) = self.host_operand(at.pc, at.registers, register)?;
        if ty.is_wide() || ty.is_borrowed() {
            return Ok(None);
        }
        Ok(Some(Side {
            live: live.clone(),
            operand,
            ty,
        }))
    }

    /// The Lua closure `picked` is, naming it for the statistics of what
    /// aborts the recording when it is another function or value.
    fn handler_closure(
        &mut self,
        picked: Option<&Value>,
        event: &'static str,
    ) -> Option<ClosureHandle> {
        match picked {
            Some(&Value::Closure(handler)) => Some(handler),
            Some(function @ (Value::NativeFunction(_) | Value::BuiltinFunction(_))) => {
                self.note_callable(function);
                None
            }
            Some(_) => {
                self.abort_metamethod = Some(event);
                None
            }
            None => None,
        }
    }

    /// The check of [`Self::guard_operator`] as metatable classes: each table
    /// operand is held to the metatable it has now, which entry fixed to name
    /// `handler`'s closure (or nothing) for `event`, and the numbers and
    /// strings among the operands to have none. What `rule` picks is then a
    /// function of the classes alone, and they give what the recording saw.
    /// `false` records nothing, for operands the classes cannot describe
    /// (a table without a metatable, a handler that is no closure, a string
    /// the string metatable answers for, a table the trace has no lane for).
    fn guard_operator_class(
        &mut self,
        at: &HookPoint<'_>,
        pc: usize,
        event: &'static str,
        (left, right): (&Side, &Side),
        handler: Option<ClosureHandle>,
    ) -> Result<bool, TraceAbortReason> {
        // `pc` is the instruction the guard leaves before. It is not
        // `at.pc` when a handler's return folds on into the rest of a chain:
        // `at` is then that return, in the handler's chunk.
        if self.function.is_some() {
            return Ok(false);
        }
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(false);
        };
        let key = metamethod_key(event).expect("an operator event is a metamethod");
        let mut bare = 0u8;
        // The tables, each with its metatable and what the event holds there.
        let mut tables: Vec<(
            &Side,
            crate::runtime::TableHandle,
            crate::jit_ir::trace::MetaPin,
        )> = Vec::new();
        for side in [left, right] {
            match &side.live {
                Value::Table(handle) => {
                    let Ok(Some(metatable)) = runtime.raw_metatable(&side.live) else {
                        return Ok(false);
                    };
                    let Some(pin) = Self::event_pin(runtime, metatable, event) else {
                        return Ok(false);
                    };
                    if self.receiver_of(at, side.operand, *handle).is_err() {
                        return Ok(false);
                    }
                    tables.push((side, metatable, pin));
                }
                Value::Integer(_) | Value::Number(_) => bare |= META_BARE_NUMBERS,
                Value::String(_) => {
                    // The string metatable has no handler for the event.
                    if runtime
                        .raw_metatable(&side.live)
                        .ok()
                        .flatten()
                        .is_some_and(|strings| {
                            !matches!(
                                runtime.table_get_string_ref(strings, key),
                                None | Some(Value::Nil)
                            )
                        })
                    {
                        return Ok(false);
                    }
                    bare |= META_BARE_STRINGS;
                }
                _ => return Ok(false),
            }
        }
        // The classes must name the handler the recording picked.
        let named = tables
            .iter()
            .any(|(_, _, pin)| pin.handler != META_PIN_ABSENT);
        if named != handler.is_some() {
            return Ok(false);
        }
        if let Some(handler) = handler
            && !tables
                .iter()
                .any(|(_, _, pin)| pin.handler == handler.native_identity())
        {
            return Ok(false);
        }
        // A window for each distinct metatable.
        let mut classes: Vec<(crate::runtime::TableHandle, usize)> = Vec::new();
        for (side, metatable, _) in &tables {
            if classes.iter().any(|(known, _)| known == metatable) {
                continue;
            }
            let Value::Table(handle) = side.live else {
                return Ok(false);
            };
            let bound = match self.receiver_of(at, side.operand, handle) {
                Ok((Receiver::Bound(window), ..)) => Some(window),
                _ => None,
            };
            let Some(window) = self.class_window(bound, *metatable, runtime) else {
                return Ok(false);
            };
            classes.push((*metatable, window));
        }
        if !self.class_guard_fits() {
            return Ok(false);
        }
        self.charge_class_dispatch();
        // Bare operands are checked by the first binding, against its event.
        let mut bare_left = bare;
        for (side, metatable, pin) in &tables {
            let window = classes
                .iter()
                .find(|(known, _)| known == metatable)
                .map(|(_, window)| *window)
                .expect("a window was found for every metatable");
            let class = self.class_binding(window, *metatable, vec![pin.clone()], bare_left);
            bare_left = 0;
            self.guard_class(pc, side.operand, class)?;
        }
        Ok(true)
    }

    /// Emits the check that the handler `rule` picks for the two operands of
    /// `event` is `handler`, or none for `None`.
    fn guard_operator(
        &mut self,
        at: Option<&HookPoint<'_>>,
        pc: usize,
        (event, rule): (&'static str, i64),
        (left, right): (&Side, &Side),
        handler: Option<ClosureHandle>,
    ) -> Result<(), TraceAbortReason> {
        if let Some(at) = at
            && self.guard_operator_class(at, pc, event, (left, right), handler)?
        {
            // The handler is pinned like a call's callee.
            if let Some(handler) = handler {
                self.pin(handler.native_identity())?;
            }
            return Ok(());
        }
        let event = operator_event(event).expect("an operator metamethod is numbered");
        let identity = match handler {
            Some(handler) => {
                self.pin(handler.native_identity())?;
                handler.native_identity().cast_signed()
            }
            None => OPERATOR_NO_HANDLER,
        };
        let selector = self.scalar(pc, TraceScalar::Integer(i64::from(event) | (rule << 8)));
        let expected = self.scalar(pc, TraceScalar::Integer(identity));
        self.emit_helper_call(
            pc,
            HelperId::OPERATOR_HANDLER,
            &[
                left.pair(),
                right.pair(),
                (Operand::Op(selector), TraceType::Integer),
                (Operand::Op(expected), TraceType::Integer),
            ],
        );
        Ok(())
    }

    /// `a = b op c` of an arithmetic or bitwise operator with a table among
    /// the operands.
    fn record_binary_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let (a, b, c) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            usize::from(instruction.c()),
        );
        let register_opcode = opcode.register_form().unwrap_or(opcode);
        let event = scalar_metamethod(register_opcode);
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(None);
        };
        // A constant operand is the right one, a number.
        let constant = match opcode.register_form() {
            Some(_) => match self.constant_at(at.chunk, c) {
                Ok(scalar @ TraceScalar::Integer(value)) => Some((scalar, Value::Integer(value))),
                Ok(scalar @ TraceScalar::Float(value)) => Some((scalar, Value::Number(value))),
                _ => return Ok(None),
            },
            None => None,
        };
        let right_live = match &constant {
            Some((_, live)) => Some(live),
            None => at.registers.get(c),
        };
        let (Some(left_live), Some(right_live)) = (at.registers.get(b), right_live) else {
            return Ok(None);
        };
        if !matches!(left_live, Value::Table(_)) && !matches!(right_live, Value::Table(_)) {
            return Ok(None);
        }
        if !operand_like(left_live) || !operand_like(right_live) {
            return Ok(None);
        }
        let key = metamethod_key(event).expect("an arithmetic event is an operator metamethod");
        let picked = operator_handler(runtime, (left_live, right_live), key, rule::ARITHMETIC);
        let Some(handler) = self.handler_closure(picked, event) else {
            return Ok(None);
        };
        self.check_register(a)?;
        let Some(left) = self.operator_side(at, b)? else {
            return Ok(None);
        };
        let right = match constant {
            Some((scalar, live)) => Side {
                live,
                operand: Operand::Op(self.scalar(pc, scalar)),
                ty: scalar.ty(),
            },
            None => match self.operator_side(at, c)? {
                Some(side) => side,
                None => return Ok(None),
            },
        };
        self.abort_metamethod = Some(event);
        self.guard_operator(
            Some(at),
            pc,
            (event, rule::ARITHMETIC),
            (&left, &right),
            Some(handler),
        )?;
        self.enter_metamethod(
            at,
            handler,
            &[left.pair(), right.pair()],
            (Entry::Operator, a),
        )
        .map(Some)
    }

    /// `a = -b` or `a = ~b` of a table.
    fn record_unary_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        let event = if opcode == OpCode::Negate {
            "__unm"
        } else {
            "__bnot"
        };
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(None);
        };
        let Some(live @ Value::Table(_)) = at.registers.get(b) else {
            return Ok(None);
        };
        let key = metamethod_key(event).expect("a unary event is an operator metamethod");
        let picked = operator_handler(runtime, (live, live), key, rule::FIRST);
        let Some(handler) = self.handler_closure(picked, event) else {
            return Ok(None);
        };
        self.check_register(a)?;
        let Some(operand) = self.operator_side(at, b)? else {
            return Ok(None);
        };
        self.abort_metamethod = Some(event);
        self.guard_operator(
            Some(at),
            pc,
            (event, rule::FIRST),
            (&operand, &operand),
            Some(handler),
        )?;
        // The interpreter passes `-x` the operand as both arguments, and
        // `~x` the operand alone.
        let arguments = [operand.pair(), operand.pair()];
        let passed = if opcode == OpCode::Negate { 2 } else { 1 };
        self.enter_metamethod(at, handler, &arguments[..passed], (Entry::Operator, a))
            .map(Some)
    }

    /// The operands of a comparison in the order it compares them, and where
    /// its answer goes: `Ok(None)` for a form this does not follow.
    #[allow(clippy::type_complexity)]
    fn comparison_sides(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<Option<(Side, Side)>, TraceAbortReason> {
        let pc = at.pc;
        if opcode.is_test_constant() {
            let Ok(scalar) = self.constant_at(at.chunk, usize::from(instruction.b())) else {
                return Ok(None);
            };
            let live = match scalar {
                TraceScalar::Integer(value) => Value::Integer(value),
                TraceScalar::Float(value) => Value::Number(value),
                _ => return Ok(None),
            };
            let Some(register) = self.operator_side(at, usize::from(instruction.a()))? else {
                return Ok(None);
            };
            let constant = Side {
                live,
                operand: Operand::Op(self.scalar(pc, scalar)),
                ty: scalar.ty(),
            };
            return Ok(Some(if instruction.test_constant_left() {
                (constant, register)
            } else {
                (register, constant)
            }));
        }
        let (left, right) = if opcode.is_test() {
            (usize::from(instruction.a()), usize::from(instruction.b()))
        } else {
            (usize::from(instruction.b()), usize::from(instruction.c()))
        };
        let (Some(left), Some(right)) = (at.registers.get(left), at.registers.get(right)) else {
            return Ok(None);
        };
        if !operand_like(left) || !operand_like(right) {
            return Ok(None);
        }
        let (left, right) = if opcode.is_test() {
            (usize::from(instruction.a()), usize::from(instruction.b()))
        } else {
            (usize::from(instruction.b()), usize::from(instruction.c()))
        };
        let (Some(left), Some(right)) = (
            self.operator_side(at, left)?,
            self.operator_side(at, right)?,
        ) else {
            return Ok(None);
        };
        Ok(Some((left, right)))
    }

    /// `==`, `<` and `<=` of operands one of which is a table, in the form
    /// that stores a boolean and in the compare-and-branch forms.
    #[allow(clippy::too_many_lines)]
    fn record_comparison_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let test = opcode.is_test();
        let form = opcode.comparison_form().unwrap_or(opcode);
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(None);
        };
        // The operands are looked at before anything is recorded.
        let live_sides = {
            let (left, right) = if opcode.is_test_constant() {
                let register = at.registers.get(usize::from(instruction.a()));
                let constant = match self.constant_at(at.chunk, usize::from(instruction.b())) {
                    Ok(TraceScalar::Integer(value)) => Value::Integer(value),
                    Ok(TraceScalar::Float(value)) => Value::Number(value),
                    _ => return Ok(None),
                };
                match register {
                    Some(register) if instruction.test_constant_left() => {
                        (constant, register.clone())
                    }
                    Some(register) => (register.clone(), constant),
                    None => return Ok(None),
                }
            } else {
                let (left, right) = if test {
                    (usize::from(instruction.a()), usize::from(instruction.b()))
                } else {
                    (usize::from(instruction.b()), usize::from(instruction.c()))
                };
                match (at.registers.get(left), at.registers.get(right)) {
                    (Some(left), Some(right)) => (left.clone(), right.clone()),
                    _ => return Ok(None),
                }
            };
            if !operand_like(&left) || !operand_like(&right) {
                return Ok(None);
            }
            (left, right)
        };
        let (left_live, right_live) = (&live_sides.0, &live_sides.1);
        let tables = (
            matches!(left_live, Value::Table(_)),
            matches!(right_live, Value::Table(_)),
        );
        match form {
            // Only two tables ask for `__eq`; the same table is equal.
            OpCode::Equal if !(tables.0 && tables.1) => return Ok(None),
            OpCode::Equal if Runtime::raw_equal(left_live, right_live) => return Ok(None),
            OpCode::Equal => {}
            _ if !(tables.0 || tables.1) => return Ok(None),
            _ => {}
        }
        let (event, rule, fallback) = match form {
            OpCode::Equal => (
                "__eq",
                if self.integers {
                    rule::FIRST
                } else {
                    rule::SHARED
                },
                false,
            ),
            OpCode::LessThan => (
                "__lt",
                if self.lua51_family {
                    rule::SAME_TYPE
                } else {
                    rule::FIRST
                },
                false,
            ),
            _ => (
                "__le",
                if self.lua51_family {
                    rule::SAME_TYPE
                } else {
                    rule::FIRST
                },
                true,
            ),
        };
        let key = metamethod_key(event).expect("a comparison event is an operator metamethod");
        let picked = operator_handler(runtime, (left_live, right_live), key, rule);
        // The handler, and whether the answer is the `__lt` of the swapped
        // operands, negated.
        let mut swapped = false;
        let mut handler_event = event;
        let handler = match picked {
            Some(_) => {
                let Some(handler) = self.handler_closure(picked, event) else {
                    return Ok(None);
                };
                Some(handler)
            }
            None if form == OpCode::Equal => None,
            None if fallback && self.le_falls_back => {
                let lt = metamethod_key("__lt").expect("__lt is an operator metamethod");
                let picked = operator_handler(runtime, (right_live, left_live), lt, rule);
                let Some(handler) = self.handler_closure(picked, "__lt") else {
                    return Ok(None);
                };
                swapped = true;
                handler_event = "__lt";
                Some(handler)
            }
            None => return Ok(None),
        };
        let Some((left, right)) = self.comparison_sides(at, instruction, opcode)? else {
            return Ok(None);
        };
        let destination = if test {
            0
        } else {
            usize::from(instruction.a())
        };
        if !test {
            self.check_register(destination)?;
        }
        self.abort_metamethod = Some(handler_event);
        if form == OpCode::Equal {
            // The same table is equal without the handler, so the trace
            // leaves for it.
            let same = self.compare_tables(pc, left.operand, right.operand);
            let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            self.emit(
                pc,
                RecKind::GuardBranch {
                    condition: Operand::Op(same),
                    expected: false,
                    snapshot: before,
                },
                None,
            );
        }
        if swapped {
            // No `__le` answers, and the `__lt` of the swapped operands does.
            self.guard_operator(Some(at), pc, (event, rule), (&left, &right), None)?;
            self.guard_operator(
                Some(at),
                pc,
                (handler_event, rule),
                (&right, &left),
                handler,
            )?;
        } else {
            self.guard_operator(Some(at), pc, (event, rule), (&left, &right), handler)?;
        }
        let Some(handler) = handler else {
            // Two tables no handler answers for are not equal.
            self.abort_metamethod = None;
            return self.finish_comparison_without_handler(at, instruction, test);
        };
        let arguments = if swapped {
            [right.pair(), left.pair()]
        } else {
            [left.pair(), right.pair()]
        };
        let entry = if test {
            Entry::Branch {
                expected: instruction.test_expected(),
                invert: swapped,
            }
        } else {
            Entry::Compare { invert: swapped }
        };
        self.enter_metamethod(at, handler, &arguments, (entry, destination))
            .map(Some)
    }

    /// The end of an `==` of two tables no handler answers for: false.
    fn finish_comparison_without_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        test: bool,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        if !test {
            let value = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Boolean(false)),
                Some(TraceType::Boolean),
            );
            self.write(usize::from(instruction.a()), value);
            return Ok(Some(Flow::Next(pc + 1)));
        }
        let word = *at
            .chunk
            .code
            .get(pc + 1)
            .ok_or(TraceAbortReason::Divergence)?;
        if word.opcode() != Ok(OpCode::Jump) {
            return Err(TraceAbortReason::Divergence);
        }
        let target = self.target(pc + 1, word.jump_offset())?;
        if instruction.test_expected() {
            Ok(Some(Flow::Next(pc + 2)))
        } else {
            self.check_branch(pc + 1, target)?;
            Ok(Some(Flow::Next(target)))
        }
    }

    /// `a = b .. ... .. c` with a table among the operands and strings or
    /// numbers for the rest.
    fn record_concat_handler(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let (a, first, last) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            usize::from(instruction.c()),
        );
        if last <= first || Self::operator_runtime(at).is_none() {
            return Ok(None);
        }
        let Some(live) = at.registers.get(first..=last) else {
            return Ok(None);
        };
        if !live.iter().any(|value| matches!(value, Value::Table(_)))
            || !live
                .iter()
                .all(|value| text_like(value) || matches!(value, Value::Table(_)))
        {
            return Ok(None);
        }
        self.check_register(a)?;
        let mut rest = Vec::with_capacity(last - first);
        for register in first..=last {
            let Some(side) = self.operator_side(at, register)? else {
                return Ok(None);
            };
            rest.push(Piece { side });
        }
        let accumulator = rest.pop().expect("a chain has operands");
        let chain = ConcatChain {
            pc,
            destination: a,
            first,
            last,
            rest,
            start: (self.stores, self.ops.len()),
            returned: false,
        };
        self.fold_concat(at, chain, accumulator).map(Some)
    }

    /// Folds the operands of `chain` into `accumulator` from the right, as
    /// the interpreter does, until a table needs its handler: the flow of
    /// that call, or of the end of the chain.
    fn fold_concat(
        &mut self,
        at: &HookPoint<'_>,
        mut chain: ConcatChain,
        mut accumulator: Piece,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = chain.pc;
        let runtime = Self::operator_runtime(at).ok_or(TraceAbortReason::UnsupportedOperand)?;
        let from_return = chain.returned;
        loop {
            let Some(left) = chain.rest.pop() else {
                // The chain is folded: the result is the accumulator.
                let Operand::Op(value) = accumulator.side.operand else {
                    return Err(TraceAbortReason::Divergence);
                };
                self.write(chain.destination, Sym::Op(value));
                return Ok(if from_return {
                    Flow::Leave { resume: pc + 1 }
                } else {
                    Flow::Next(pc + 1)
                });
            };
            // A step after a handler that committed an effect would leave
            // through the snapshot before the instruction, which would run
            // that handler again.
            if from_return
                && (self.stores != chain.start.0
                    || self.ops[chain.start.1..]
                        .iter()
                        .any(|op| op.kind.commits_effect()))
            {
                self.abort_metamethod = Some("__concat");
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            if text_like(&left.side.live) && text_like(&accumulator.side.live) {
                // A run of strings and numbers is one concatenation.
                let mut run = vec![left];
                while chain
                    .rest
                    .last()
                    .is_some_and(|next| text_like(&next.side.live))
                {
                    run.push(chain.rest.pop().expect("a piece was seen"));
                }
                run.reverse();
                let mut operands: Vec<(Operand, TraceType)> =
                    run.iter().map(|piece| piece.side.pair()).collect();
                operands.push(accumulator.side.pair());
                if operands.len() > super::MAX_HOST_ARGS {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let value =
                    self.emit_library_helper(pc, HelperId::CONCAT, &operands, TraceType::String);
                accumulator = Piece {
                    side: Side {
                        live: Value::String(LuaString::from("")),
                        operand: Operand::Op(value),
                        ty: TraceType::String,
                    },
                };
                continue;
            }
            let key = metamethod_key("__concat").expect("__concat is an operator metamethod");
            let picked = operator_handler(
                runtime,
                (&left.side.live, &accumulator.side.live),
                key,
                rule::ARITHMETIC,
            );
            let Some(handler) = self.handler_closure(picked, "__concat") else {
                return Err(TraceAbortReason::UnsupportedOperand);
            };
            self.abort_metamethod = Some("__concat");
            if !operand_like(&accumulator.side.live) || accumulator.side.ty.is_borrowed() {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            self.guard_operator(
                Some(at),
                pc,
                ("__concat", rule::ARITHMETIC),
                (&left.side, &accumulator.side),
                Some(handler),
            )?;
            let arguments = [left.side.pair(), accumulator.side.pair()];
            let destination = chain.destination;
            let entry = Entry::Concat {
                first: u8::try_from(chain.first).map_err(|_| TraceAbortReason::LimitExceeded)?,
                last: u8::try_from(chain.last).map_err(|_| TraceAbortReason::LimitExceeded)?,
                pending: u8::try_from(chain.rest.len())
                    .map_err(|_| TraceAbortReason::LimitExceeded)?,
            };
            chain.returned = true;
            let extra = Extra {
                after: Pending::Concat(Box::new(chain)),
                results: Some(1),
                pc: Some(pc),
            };
            let flow =
                self.enter_metamethod_with(at, handler, &arguments, (entry, destination), extra)?;
            // After a return the handler's frame replaces the one that left.
            return Ok(if from_return { Flow::Replace } else { flow });
        }
    }

    /// The `Return` of a frame whose instruction goes on after it: a
    /// comparison turns the first value into its answer, a `..` folds it into
    /// the rest of the chain.
    #[allow(clippy::too_many_lines)]
    pub(super) fn return_to_operator(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        count: usize,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let (live, value) = self.first_returned(at, instruction, count)?;
        let first = match (live, value) {
            (Some(live), Some(Sym::Op(operand))) => {
                let ty = self.ops[operand as usize]
                    .ty
                    .expect("an operand operation defines a value");
                Some(Side {
                    live,
                    operand: Operand::Op(operand),
                    ty,
                })
            }
            (Some(live), Some(Sym::Root(slot))) => {
                let (operand, ty) = self.root_operand(pc, slot);
                Some(Side { live, operand, ty })
            }
            (None, None) => None,
            _ => return Err(TraceAbortReason::Divergence),
        };
        let mut callee = self.frames.pop().expect("an inlined frame returns");
        if callee.owns_reexec {
            self.reexec = None;
        }
        if self.frames.is_empty() {
            self.call_site = None;
        }
        if let Pending::Concat(chain) = std::mem::take(&mut callee.after) {
            let accumulator = first.unwrap_or_else(|| Side {
                live: Value::Nil,
                operand: Operand::Op(self.scalar(pc, TraceScalar::Nil)),
                ty: TraceType::Nil,
            });
            return self.fold_concat(at, *chain, Piece { side: accumulator });
        }
        let (invert, branch) = match callee.entry {
            Entry::Compare { invert } => (invert, None),
            Entry::Branch { expected, invert } => (invert, Some(expected)),
            _ => return Err(TraceAbortReason::Divergence),
        };
        // The truth of the first value, negated for the swapped `__lt`.
        let live = first.as_ref().is_some_and(|side| side.live.is_truthy());
        let outcome = live != invert;
        let (condition, dynamic) = match first {
            Some(Side {
                operand: Operand::Op(value),
                ty: TraceType::Boolean,
                ..
            }) => {
                let condition = if invert {
                    self.emit(
                        pc,
                        RecKind::Not(Operand::Op(value)),
                        Some(TraceType::Boolean),
                    )
                } else {
                    value
                };
                (condition, true)
            }
            _ => (
                self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Boolean(outcome)),
                    Some(TraceType::Boolean),
                ),
                false,
            ),
        };
        let call_pc = callee.call_pc;
        let Some(expected) = branch else {
            self.write(callee.result_base, Sym::Op(condition));
            return Ok(Flow::Leave {
                resume: call_pc + 1,
            });
        };
        // The branch of the test, followed like any test: the jump word is
        // not hooked on its own.
        let word = *self
            .running_chunk()
            .code
            .get(call_pc + 1)
            .ok_or(TraceAbortReason::Divergence)?;
        if word.opcode() != Ok(OpCode::Jump) {
            return Err(TraceAbortReason::Divergence);
        }
        let target = self.target(call_pc + 1, word.jump_offset())?;
        let taken = outcome == expected;
        let next = if taken {
            self.check_branch(call_pc + 1, target)?;
            target
        } else {
            call_pc + 2
        };
        if dynamic {
            let alternative = if taken { call_pc + 2 } else { target };
            let snapshot = self.branch_snapshot(call_pc, alternative);
            self.emit(
                call_pc,
                RecKind::GuardBranch {
                    condition: Operand::Op(condition),
                    expected: outcome,
                    snapshot,
                },
                None,
            );
        }
        Ok(Flow::Leave { resume: next })
    }

    /// The first value a handler returns, live and as a symbol, and whether
    /// it returns one. A `ReturnOpen` returns the values the call before it
    /// made after its own fixed ones.
    fn first_returned(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        count: usize,
    ) -> Result<(Option<Value>, Option<Sym>), TraceAbortReason> {
        let a = usize::from(instruction.a());
        let open = if instruction.opcode() == Ok(OpCode::ReturnOpen) {
            Some(self.open.take().ok_or(TraceAbortReason::Divergence)?)
        } else {
            None
        };
        if count > 0 {
            self.check_register(a)?;
            let sym = match self.sym(a) {
                sym @ Sym::Root(_) => sym,
                _ => match self.read(at.registers, a)?.0 {
                    Operand::Op(op) => Sym::Op(op),
                    Operand::Carried(_) => return Err(TraceAbortReason::Divergence),
                },
            };
            return Ok((at.registers.get(a).cloned(), Some(sym)));
        }
        let Some(&first) = open.as_deref().and_then(<[Open]>::first) else {
            return Ok((None, None));
        };
        let sym = match first {
            Open::Op(op) => Sym::Op(op),
            Open::Root(slot) => Sym::Root(slot),
            Open::Char(_) => return Err(TraceAbortReason::UnsupportedOperand),
        };
        Ok((at.open.first().cloned(), Some(sym)))
    }

    /// The `Return` of a `__tostring` handler: `tostring` gives the first
    /// value as its one result, which the interpreter accepts as a string. In
    /// the profiles that check, a number (which it formats) or anything else
    /// (which it raises for) leaves the loop to the interpreter; Lua 5.1 and
    /// `LuaJIT` return whatever the handler returned.
    pub(super) fn return_from_tostring(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        count: usize,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let (live, value) = self.first_returned(at, instruction, count)?;
        if !self.lua51_family && !matches!(live, Some(Value::String(_))) {
            self.abort_metamethod = Some("__tostring");
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let value = match value {
            Some(value) => value,
            None => Sym::Op(self.scalar(pc, TraceScalar::Nil)),
        };
        let callee = self.frames.pop().expect("an inlined frame returns");
        if callee.owns_reexec {
            self.reexec = None;
        }
        if self.frames.is_empty() {
            self.call_site = None;
        }
        let (base, results) = (callee.result_base, callee.results);
        if results == 0 {
            // The call's one result is the open values of the instruction
            // that takes them.
            self.open = Some(vec![match value {
                Sym::Op(op) => Open::Op(op),
                Sym::Root(slot) => Open::Root(slot),
                _ => return Err(TraceAbortReason::Divergence),
            }]);
        } else {
            self.check_register(base + results - 1)?;
            self.write(base, value);
            if results > 1 {
                let nil = self.scalar(pc, TraceScalar::Nil);
                for register in base + 1..base + results {
                    self.write(register, nil);
                }
            }
        }
        Ok(Flow::Leave {
            resume: callee.call_pc + 1,
        })
    }

    /// A call of `tostring (t)` for a table with a Lua closure as
    /// `__tostring`. `None` leaves the call to the code that records it
    /// without a handler.
    pub(super) fn record_tostring_handler(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        argument: (Operand, TraceType),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let instruction = *at.chunk.code.get(pc).ok_or(TraceAbortReason::Divergence)?;
        let register = match instruction.opcode() {
            Ok(OpCode::CallOne) => usize::from(instruction.c()),
            Ok(OpCode::Call) => usize::from(instruction.a()) + 1,
            _ => return Ok(None),
        };
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(None);
        };
        let Some(live @ Value::Table(_)) = at.registers.get(register) else {
            return Ok(None);
        };
        if argument.1 != TraceType::Table {
            return Ok(None);
        }
        let key = metamethod_key("__tostring").expect("__tostring is an operator metamethod");
        let picked = operator_handler(runtime, (live, live), key, rule::FIRST);
        let Some(handler) = self.handler_closure(picked, "__tostring") else {
            // The call of `tostring` is what aborts; its handler is named too.
            if picked.is_some() {
                self.abort_metamethod = Some("__tostring");
            }
            return Ok(None);
        };
        self.abort_metamethod = Some("__tostring");
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_TOSTRING))?;
        self.guard_callee(pc, callee, lane);
        let side = Side {
            live: live.clone(),
            operand: argument.0,
            ty: argument.1,
        };
        self.guard_operator(
            Some(at),
            pc,
            ("__tostring", rule::FIRST),
            (&side, &side),
            Some(handler),
        )?;
        let extra = Extra {
            after: Pending::ToString,
            results: Some(results),
            pc: None,
        };
        self.enter_metamethod_with(at, handler, &[argument], (Entry::ToString, base), extra)
            .map(Some)
    }

    /// `tostring (t)` of a table whose metatable has no `__tostring`: the VM
    /// spells it `table: N` (no profile reads `__name` here). The guards are
    /// the `tostring` callee and the absent handler, so one trace serves
    /// every object of a class. `None` when the table has a handler.
    pub(super) fn record_tostring_plain(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        argument: (Operand, TraceType),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let instruction = *at.chunk.code.get(pc).ok_or(TraceAbortReason::Divergence)?;
        let register = match instruction.opcode() {
            Ok(OpCode::CallOne) => usize::from(instruction.c()),
            Ok(OpCode::Call) => usize::from(instruction.a()) + 1,
            _ => return Ok(None),
        };
        let Some(runtime) = Self::operator_runtime(at) else {
            return Ok(None);
        };
        let Some(live @ Value::Table(_)) = at.registers.get(register) else {
            return Ok(None);
        };
        if argument.1 != TraceType::Table {
            return Ok(None);
        }
        let key = metamethod_key("__tostring").expect("__tostring is an operator metamethod");
        if operator_handler(runtime, (live, live), key, rule::FIRST).is_some() {
            return Ok(None);
        }
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_TOSTRING))?;
        self.guard_callee(pc, callee, lane);
        let side = Side {
            live: live.clone(),
            operand: argument.0,
            ty: argument.1,
        };
        self.guard_operator(
            Some(at),
            pc,
            ("__tostring", rule::FIRST),
            (&side, &side),
            None,
        )?;
        let value =
            self.emit_library_helper(pc, HelperId::TOSTRING, &[argument], TraceType::String);
        Ok(Some(self.native_result(pc, value, (base, results))))
    }

    /// The Lua closure `pairs (t)` runs as the table's `__pairs`, if the call
    /// is one the recorder follows into it: the profile honours `__pairs`,
    /// and the callee is `pairs` called with a table that has one.
    pub(super) fn pairs_handler(
        &mut self,
        at: &HookPoint<'_>,
        (callee, first_argument, fixed): (usize, usize, usize),
    ) -> Option<ClosureHandle> {
        if !self.honours_pairs || fixed == 0 {
            return None;
        }
        let runtime = Self::operator_runtime(at)?;
        if !matches!(
            at.registers.get(callee),
            Some(Value::BuiltinFunction(
                crate::runtime::BuiltinFunction::Pairs
            ))
        ) {
            return None;
        }
        let live @ Value::Table(_) = at.registers.get(first_argument)? else {
            return None;
        };
        let key = metamethod_key("__pairs").expect("__pairs is an operator metamethod");
        let picked = operator_handler(runtime, (live, live), key, rule::FIRST);
        let handler = self.handler_closure(picked, "__pairs")?;
        Some(handler)
    }

    /// The guards of a `pairs (t)` that runs `handler`: the callee is
    /// `pairs`, and the table's `__pairs` is the closure. The closure then
    /// runs as the callee, with the table alone.
    pub(super) fn guard_pairs_handler(
        &mut self,
        pc: usize,
        callee: Operand,
        table: (Operand, TraceType),
        live: &Value,
        handler: ClosureHandle,
    ) -> Result<(), TraceAbortReason> {
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_PAIRS))?;
        self.guard_callee(pc, callee, lane);
        let side = Side {
            live: live.clone(),
            operand: table.0,
            ty: table.1,
        };
        self.guard_operator(
            None,
            pc,
            ("__pairs", rule::FIRST),
            (&side, &side),
            Some(handler),
        )
    }
}
