//! If-conversion of small forward-branching numeric callees into inline plans.
//!
//! A callee such as `if x % 2 == 0 then return x + 1 else return x + 2 end`
//! has no loop and no effect, so running both arms and selecting the result
//! is exact. The converter walks the bytecode in PC order, which is a
//! topological order because every jump goes forward. Each PC receives the
//! paths that reach it, each with the predicate under which it is taken.
//! Paths into one PC are mutually exclusive, so a register that differs
//! between them becomes a chain of `Select`s, built only when it is read.

use crate::bytecode::{Chunk, Constant, OpCode};
use crate::jit_ir::inline_plans::{
    InlineCallPlan, InlineNumericOperation, inline_numeric_binary, inline_numeric_constant,
};
use crate::jit_ir::operation::Comparison;

/// Every arm runs, so the plan costs the callee's whole body per call.
const MAX_OPERATIONS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Number,
    Condition,
}

/// A callee register on entry to a PC.
#[derive(Clone, Debug)]
enum Slot {
    Empty,
    Plain(u8, Kind),
    /// One value per incoming path, keyed by that path's predicate.
    Merge(Kind, Vec<(Option<u8>, Slot)>),
}

impl Slot {
    fn kind(&self) -> Option<Kind> {
        match self {
            Self::Empty => None,
            Self::Plain(_, kind) | Self::Merge(kind, _) => Some(*kind),
        }
    }
}

#[derive(Clone, Debug)]
struct Path {
    /// `None` means the path is taken on every call.
    predicate: Option<u8>,
    slots: Vec<Slot>,
}

#[derive(Default)]
struct Builder {
    operations: Vec<InlineNumericOperation>,
    next_register: usize,
    constants: [Option<u8>; 2],
}

impl Builder {
    fn fresh(&mut self) -> Option<u8> {
        let register = u8::try_from(self.next_register)
            .ok()
            .filter(|&r| r < u8::MAX)?;
        self.next_register += 1;
        Some(register)
    }

    fn push(&mut self, operation: InlineNumericOperation) -> Option<()> {
        if self.operations.len() >= MAX_OPERATIONS {
            return None;
        }
        self.operations.push(operation);
        Some(())
    }

    fn constant(&mut self, value: bool) -> Option<u8> {
        let index = usize::from(value);
        if let Some(register) = self.constants[index] {
            return Some(register);
        }
        let destination = self.fresh()?;
        self.push(InlineNumericOperation::LoadNumber {
            destination,
            value: f64::from(u8::from(value)),
        })?;
        self.constants[index] = Some(destination);
        Some(destination)
    }

    fn select(&mut self, condition: u8, if_true: u8, if_false: u8) -> Option<u8> {
        if if_true == if_false {
            return Some(if_true);
        }
        let destination = self.fresh()?;
        self.push(InlineNumericOperation::Select {
            destination,
            condition,
            if_true,
            if_false,
        })?;
        Some(destination)
    }

    fn not(&mut self, source: u8) -> Option<u8> {
        let destination = self.fresh()?;
        self.push(InlineNumericOperation::Not {
            destination,
            source,
        })?;
        Some(destination)
    }

    fn and(&mut self, predicate: Option<u8>, condition: u8) -> Option<u8> {
        match predicate {
            None => Some(condition),
            Some(predicate) => {
                let zero = self.constant(false)?;
                self.select(predicate, condition, zero)
            }
        }
    }

    /// Materializes a slot, replacing a merge with its `Select` chain.
    fn read(&mut self, slot: &mut Slot) -> Option<(u8, Kind)> {
        let value = match slot {
            Slot::Empty => return None,
            Slot::Plain(register, kind) => return Some((*register, *kind)),
            Slot::Merge(kind, arms) => {
                let kind = *kind;
                let mut values = Vec::with_capacity(arms.len());
                for (predicate, arm) in arms.iter_mut() {
                    let (register, arm_kind) = self.read(arm)?;
                    if arm_kind != kind {
                        return None;
                    }
                    values.push((*predicate, register));
                }
                (self.choose(&values)?, kind)
            }
        };
        *slot = Slot::Plain(value.0, value.1);
        Some(value)
    }

    /// Picks the value of whichever exclusive predicate holds.
    fn choose(&mut self, values: &[(Option<u8>, u8)]) -> Option<u8> {
        if let Some(&(_, register)) = values.iter().find(|(predicate, _)| predicate.is_none()) {
            return Some(register);
        }
        let (&(_, last), rest) = values.split_last()?;
        let mut chosen = last;
        for &(predicate, register) in rest.iter().rev() {
            chosen = self.select(predicate?, register, chosen)?;
        }
        Some(chosen)
    }

    fn merge(&mut self, mut paths: Vec<Path>) -> Option<Path> {
        if paths.len() == 1 {
            return paths.pop();
        }
        if let Some(index) = paths.iter().position(|path| path.predicate.is_none()) {
            return Some(paths.swap_remove(index));
        }
        let one = self.constant(true)?;
        let predicates: Vec<_> = paths.iter().map(|path| (path.predicate, one)).collect();
        // `p1 or p2 or ...` as `select(p1, 1, select(p2, 1, ...))`.
        let mut predicate = paths.last()?.predicate?;
        for &(arm, _) in predicates[..predicates.len() - 1].iter().rev() {
            predicate = self.select(arm?, one, predicate)?;
        }
        let register_count = paths[0].slots.len();
        let slots = (0..register_count)
            .map(|register| {
                let first = &paths[0].slots[register];
                if let Slot::Plain(value, kind) = first
                    && paths.iter().all(|path| {
                        matches!(path.slots[register], Slot::Plain(other, other_kind)
                            if other == *value && other_kind == *kind)
                    })
                {
                    return first.clone();
                }
                let Some(kind) = first.kind() else {
                    return Slot::Empty;
                };
                if paths
                    .iter()
                    .any(|path| path.slots[register].kind() != Some(kind))
                {
                    return Slot::Empty;
                }
                Slot::Merge(
                    kind,
                    paths
                        .iter()
                        .map(|path| (path.predicate, path.slots[register].clone()))
                        .collect(),
                )
            })
            .collect();
        Some(Path {
            predicate: Some(predicate),
            slots,
        })
    }
}

fn forward_target(pc: usize, offset: i32, length: usize) -> Option<usize> {
    let target = (pc + 1).checked_add_signed(isize::try_from(offset).ok()?)?;
    (target > pc && target < length).then_some(target)
}

fn number(builder: &mut Builder, slots: &mut [Slot], register: u8) -> Option<u8> {
    let (value, kind) = builder.read(slots.get_mut(usize::from(register))?)?;
    (kind == Kind::Number).then_some(value)
}

/// If-converts a forward-branching numeric callee. `None` refuses it.
///
/// Accepts the straight-line subset plus numeric comparisons, `not`, boolean
/// constants, and forward jumps. Captured cells may be read but not written,
/// because a write under a predicate would need its own merge. Every path
/// must end in a single numeric return.
#[allow(
    clippy::too_many_lines,
    reason = "the accepted bytecode grammar is intentionally visible in one match"
)]
pub(crate) fn validate_branching(chunk: &Chunk) -> Option<InlineCallPlan> {
    if chunk.upvalues.len() > 1 || !chunk.prototypes.is_empty() {
        return None;
    }
    let length = chunk.code.len();
    let register_count = usize::from(chunk.register_count);
    let parameter_count = chunk.parameter_count;
    let mut builder = Builder {
        next_register: usize::from(parameter_count),
        ..Builder::default()
    };
    let mut arrivals: Vec<Vec<Path>> = vec![Vec::new(); length];
    *arrivals.first_mut()? = vec![Path {
        predicate: None,
        slots: (0..register_count)
            .map(|register| {
                if register < usize::from(parameter_count) {
                    Slot::Plain(u8::try_from(register).ok().unwrap_or(0), Kind::Number)
                } else {
                    Slot::Empty
                }
            })
            .collect(),
    }];
    let mut returns: Vec<(Option<u8>, Slot)> = Vec::new();
    let mut accessed_upvalue = false;

    for pc in 0..length {
        let paths = std::mem::take(&mut arrivals[pc]);
        if paths.is_empty() {
            continue;
        }
        let mut path = builder.merge(paths)?;
        let instruction = chunk.code[pc];
        let opcode = instruction.opcode().ok()?;
        let (a, b, c) = (instruction.a(), instruction.b(), instruction.c());
        let destination = usize::from(a);
        if destination >= register_count && !matches!(opcode, OpCode::Jump | OpCode::Halt) {
            return None;
        }
        let mut next = Some(pc + 1);
        match opcode {
            OpCode::LoadConstant | OpCode::LoadConstantWide => {
                let index = if opcode == OpCode::LoadConstant {
                    u32::from(instruction.d())
                } else {
                    next = Some(pc + 2);
                    chunk.code.get(pc + 1)?.extra_argument_index()
                };
                let constant = chunk.constants.get(usize::try_from(index).ok()?)?;
                path.slots[destination] = if let Constant::Boolean(value) = constant {
                    Slot::Plain(builder.constant(*value)?, Kind::Condition)
                } else {
                    let value = inline_numeric_constant(constant)?;
                    let register = builder.fresh()?;
                    builder.push(InlineNumericOperation::LoadNumber {
                        destination: register,
                        value,
                    })?;
                    Slot::Plain(register, Kind::Number)
                };
            }
            OpCode::Move => {
                let source = path.slots.get(usize::from(b))?.clone();
                source.kind()?;
                path.slots[destination] = source;
            }
            OpCode::ClearRange => {
                let end = destination.checked_add(usize::from(b))?;
                for slot in path.slots.get_mut(destination..end)? {
                    *slot = Slot::Empty;
                }
            }
            OpCode::Negate => {
                let source = number(&mut builder, &mut path.slots, b)?;
                let register = builder.fresh()?;
                builder.push(InlineNumericOperation::Negate {
                    destination: register,
                    source,
                })?;
                path.slots[destination] = Slot::Plain(register, Kind::Number);
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power => {
                let left = number(&mut builder, &mut path.slots, b)?;
                let right = number(&mut builder, &mut path.slots, c)?;
                let register = builder.fresh()?;
                builder.push(inline_numeric_binary(opcode, register, left, right))?;
                path.slots[destination] = Slot::Plain(register, Kind::Number);
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                let left = number(&mut builder, &mut path.slots, b)?;
                let value = inline_numeric_constant(chunk.constants.get(usize::from(c))?)?;
                let right = builder.fresh()?;
                builder.push(InlineNumericOperation::LoadNumber {
                    destination: right,
                    value,
                })?;
                let register = builder.fresh()?;
                builder.push(inline_numeric_binary(
                    opcode.register_form()?,
                    register,
                    left,
                    right,
                ))?;
                path.slots[destination] = Slot::Plain(register, Kind::Number);
            }
            OpCode::Equal | OpCode::LessThan | OpCode::LessEqual => {
                let left = number(&mut builder, &mut path.slots, b)?;
                let right = number(&mut builder, &mut path.slots, c)?;
                let register = builder.fresh()?;
                builder.push(InlineNumericOperation::Compare {
                    destination: register,
                    left,
                    right,
                    comparison: match opcode {
                        OpCode::Equal => Comparison::Equal,
                        OpCode::LessThan => Comparison::LessThan,
                        _ => Comparison::LessEqual,
                    },
                })?;
                path.slots[destination] = Slot::Plain(register, Kind::Condition);
            }
            OpCode::Not => {
                let (source, kind) = builder.read(path.slots.get_mut(usize::from(b))?)?;
                // Numbers are always truthy.
                let register = match kind {
                    Kind::Number => builder.constant(false)?,
                    Kind::Condition => builder.not(source)?,
                };
                path.slots[destination] = Slot::Plain(register, Kind::Condition);
            }
            OpCode::GetUpvalue => {
                if instruction.d() != 0 || chunk.upvalues.is_empty() {
                    return None;
                }
                let register = builder.fresh()?;
                builder.push(InlineNumericOperation::LoadUpvalue {
                    destination: register,
                    upvalue: 0,
                })?;
                accessed_upvalue = true;
                path.slots[destination] = Slot::Plain(register, Kind::Number);
            }
            OpCode::Vararg if b > 0 => {
                for offset in 0..b {
                    let slot = destination.checked_add(usize::from(offset))?;
                    if slot >= register_count {
                        return None;
                    }
                    let register = builder.fresh()?;
                    builder.push(InlineNumericOperation::LoadVararg {
                        destination: register,
                        index: offset,
                    })?;
                    path.slots[slot] = Slot::Plain(register, Kind::Number);
                }
            }
            OpCode::Jump => {
                let target = forward_target(pc, instruction.jump_offset(), length)?;
                arrivals[target].push(path.clone());
                next = None;
            }
            // A comparison and the `Jump` word after it. The jump is taken
            // when the result equals `expected`, so the two paths mirror the
            // `JumpIfFalse` case: the jump's path takes the predicate that
            // `expected` selects, and execution continues after the word.
            OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK => {
                let word = *chunk.code.get(pc + 1)?;
                if word.opcode().ok()? != OpCode::Jump {
                    return None;
                }
                let target = forward_target(pc + 1, word.jump_offset(), length)?;
                let (left, right) = if opcode.is_test_constant() {
                    let value = inline_numeric_constant(chunk.constants.get(usize::from(b))?)?;
                    let constant = builder.fresh()?;
                    builder.push(InlineNumericOperation::LoadNumber {
                        destination: constant,
                        value,
                    })?;
                    let register = number(&mut builder, &mut path.slots, a)?;
                    if instruction.test_constant_left() {
                        (constant, register)
                    } else {
                        (register, constant)
                    }
                } else {
                    let left = number(&mut builder, &mut path.slots, a)?;
                    let right = number(&mut builder, &mut path.slots, b)?;
                    (left, right)
                };
                let condition = builder.fresh()?;
                builder.push(InlineNumericOperation::Compare {
                    destination: condition,
                    left,
                    right,
                    comparison: match opcode.comparison_form()? {
                        OpCode::Equal => Comparison::Equal,
                        OpCode::LessThan => Comparison::LessThan,
                        _ => Comparison::LessEqual,
                    },
                })?;
                let negated = builder.not(condition)?;
                let (jumps, stays) = if instruction.test_expected() {
                    (condition, negated)
                } else {
                    (negated, condition)
                };
                let taken = builder.and(path.predicate, jumps)?;
                let stays = builder.and(path.predicate, stays)?;
                arrivals[target].push(Path {
                    predicate: Some(taken),
                    slots: path.slots.clone(),
                });
                path.predicate = Some(stays);
                next = Some(pc + 2);
            }
            OpCode::JumpIfFalse => {
                let target = forward_target(pc, i32::from(instruction.d().cast_signed()), length)?;
                let (condition, kind) = builder.read(path.slots.get_mut(destination)?)?;
                if kind == Kind::Condition {
                    let negated = builder.not(condition)?;
                    let taken = builder.and(path.predicate, negated)?;
                    let stays = builder.and(path.predicate, condition)?;
                    arrivals[target].push(Path {
                        predicate: Some(taken),
                        slots: path.slots.clone(),
                    });
                    path.predicate = Some(stays);
                }
            }
            OpCode::Return => {
                if b != 1 || path.slots[destination].kind() != Some(Kind::Number) {
                    return None;
                }
                returns.push((path.predicate, path.slots[destination].clone()));
                next = None;
            }
            _ => return None,
        }
        if let Some(next) = next {
            // Falling off the end, or into the trailing `Halt`, returns no value.
            if next >= length || chunk.code[next].opcode().ok()? == OpCode::Halt {
                return None;
            }
            arrivals[next].push(path);
        }
    }
    if !chunk.upvalues.is_empty() && !accessed_upvalue {
        return None;
    }
    let mut values = Vec::with_capacity(returns.len());
    for (predicate, mut slot) in returns {
        let (register, kind) = builder.read(&mut slot)?;
        if kind != Kind::Number {
            return None;
        }
        values.push((predicate, register));
    }
    let result_register = builder.choose(&values)?;
    let register_count = u8::try_from(builder.next_register).ok()?;
    Some(InlineCallPlan::from_parts(
        chunk,
        register_count,
        result_register,
        builder.operations,
    ))
}
