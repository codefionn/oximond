//! Which captured locals a function can still write after a closure takes
//! them.
//!
//! The VM used to keep a captured local in its register and mirror it into a
//! shared cell, and this pass found the locals that needed no write back. Open
//! upvalues replaced the mirror: a captured local lives in its register and the
//! cell reads it there, so the VM no longer asks. The analysis stays for
//! callers that want to know which captured locals the chunk can still write.

use crate::bytecode::{Chunk, Instruction, OpCode, UpvalueSource};
use std::sync::OnceLock;

/// A set of the 256 registers of a frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegisterSet(pub(crate) [u64; 4]);

impl RegisterSet {
    /// The set with all 256 registers.
    pub const ALL: Self = Self([u64::MAX; 4]);

    /// The set with no registers.
    pub const EMPTY: Self = Self([0; 4]);

    /// The union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// The registers in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    /// Whether `register` is in the set.
    #[must_use]
    pub const fn contains(&self, register: u8) -> bool {
        self.0[(register >> 6) as usize] & (1 << (register & 63)) != 0
    }

    /// Adds `register` to the set.
    pub const fn insert(&mut self, register: u8) {
        self.0[(register >> 6) as usize] |= 1 << (register & 63);
    }

    fn remove(&mut self, register: u8) {
        self.0[(register >> 6) as usize] &= !(1 << (register & 63));
    }
}

/// The lazily computed analysis stored in a [`Chunk`].
///
/// It is a cache: clones start empty and two chunks compare equal whatever
/// they have computed.
#[derive(Default)]
pub struct CaptureVolatility {
    volatile: OnceLock<RegisterSet>,
}

impl Clone for CaptureVolatility {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for CaptureVolatility {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl core::fmt::Debug for CaptureVolatility {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("CaptureVolatility")
    }
}

impl Chunk {
    /// The registers whose captured value may be newer in the frame than in
    /// its cell: every register except the ones this analysis proves no
    /// instruction writes once a closure has captured them.
    ///
    /// A register is proved when every instruction that can write it comes
    /// before the first closure capturing it, and no backward jump leads from
    /// code after that capture to code before it. A parameter's initial value
    /// is written before the first instruction.
    #[must_use]
    pub fn volatile_captures(&self) -> &RegisterSet {
        self.captures.volatile.get_or_init(|| analyze(self))
    }
}

/// The registers one instruction may write.
pub(crate) enum Written {
    None,
    /// `start..start + count`.
    Range(u8, usize),
    /// An instruction that does not decode: any register.
    Any,
}

pub(crate) fn written_by(instruction: Instruction) -> Written {
    let Ok(opcode) = instruction.opcode() else {
        return Written::Any;
    };
    let (a, b) = (instruction.a(), instruction.b());
    match opcode {
        OpCode::LoadNil
        | OpCode::LoadConstant
        | OpCode::LoadConstantWide
        | OpCode::GetGlobal
        | OpCode::GetGlobalWide
        | OpCode::GetUpvalue
        | OpCode::GetTabUp
        | OpCode::NewTable
        | OpCode::Move
        | OpCode::Negate
        | OpCode::Not
        | OpCode::Length
        | OpCode::BitNot
        | OpCode::AddK
        | OpCode::SubtractK
        | OpCode::MultiplyK
        | OpCode::DivideK
        | OpCode::ModuloK
        | OpCode::Add
        | OpCode::Subtract
        | OpCode::Multiply
        | OpCode::Divide
        | OpCode::Modulo
        | OpCode::Power
        | OpCode::FloorDivide
        | OpCode::BitAnd
        | OpCode::BitOr
        | OpCode::BitXor
        | OpCode::ShiftLeft
        | OpCode::ShiftRight
        | OpCode::Equal
        | OpCode::LessThan
        | OpCode::LessEqual
        | OpCode::GetTable
        | OpCode::GetField
        | OpCode::CallOne
        | OpCode::Closure
        | OpCode::Concat
        | OpCode::TableVararg
        | OpCode::TableOpen => Written::Range(a, 1),
        // The method and the receiver copied after it.
        OpCode::SelfField => Written::Range(a, 2),
        OpCode::ClearRange => Written::Range(a, usize::from(b)),
        OpCode::NumericForPrep | OpCode::NumericForLoop => Written::Range(a, 4),
        // The call window and the results, as `Call` below leaves them.
        OpCode::ForInCall => Written::Range(
            a.saturating_add(3),
            256 - usize::from(a.saturating_add(3)),
        ),
        OpCode::ForInLoop => Written::Range(a.saturating_add(2), 1),
        // The results of a call, or the varargs, land from `a` up.
        OpCode::Call | OpCode::CallVararg | OpCode::CallOpen | OpCode::Vararg => {
            Written::Range(a, 256 - usize::from(a))
        }
        OpCode::SetTable
        | OpCode::SetField
        | OpCode::SetTabUp
        | OpCode::DeclareTable
        | OpCode::SetGlobal
        | OpCode::SetGlobalWide
        | OpCode::DeclareGlobal
        | OpCode::DeclareGlobalWide
        | OpCode::SetUpvalue
        | OpCode::ExtraArgument
        | OpCode::Jump
        | OpCode::JumpIfFalse
        | OpCode::TestEqual
        | OpCode::TestLessThan
        | OpCode::TestLessEqual
        | OpCode::TestEqualK
        | OpCode::TestLessThanK
        | OpCode::TestLessEqualK
        | OpCode::MarkClose
        | OpCode::CloseLocal
        | OpCode::CloseUpvalue
        // These leave the frame, so nothing after them runs in it.
        | OpCode::Return
        | OpCode::ReturnVararg
        | OpCode::ReturnOpen
        | OpCode::TailCall
        | OpCode::TailCallVararg
        | OpCode::TailCallOpen
        | OpCode::Halt => Written::None,
    }
}

/// The pc a backward jump at `pc` lands on.
fn backward_target(pc: usize, instruction: Instruction) -> Option<usize> {
    let offset = match instruction.opcode() {
        Ok(OpCode::Jump) => instruction.jump_offset(),
        Ok(OpCode::JumpIfFalse | OpCode::NumericForLoop | OpCode::ForInLoop) => {
            i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
        }
        _ => return None,
    };
    let target = usize::try_from(i64::try_from(pc).ok()? + 1 + i64::from(offset)).ok()?;
    (target <= pc).then_some(target)
}

fn analyze(chunk: &Chunk) -> RegisterSet {
    let mut first_capture: [Option<usize>; 256] = [None; 256];
    // `None` is "never written by an instruction".
    let mut last_write: [Option<usize>; 256] = [None; 256];
    let mut backward: Vec<(usize, usize)> = Vec::new();
    for (pc, &instruction) in chunk.code.iter().enumerate() {
        match written_by(instruction) {
            Written::None => {}
            Written::Range(start, count) => {
                let start = usize::from(start);
                for slot in &mut last_write[start..(start + count).min(256)] {
                    *slot = Some(pc);
                }
            }
            Written::Any => last_write = [Some(pc); 256],
        }
        if let Some(target) = backward_target(pc, instruction) {
            backward.push((pc, target));
        }
        if instruction.opcode() == Ok(OpCode::Closure)
            && let Some(prototype) = chunk.prototypes.get(usize::from(instruction.d()))
        {
            for descriptor in &prototype.upvalues {
                if let UpvalueSource::ParentRegister(register) = descriptor.source {
                    first_capture[usize::from(register)].get_or_insert(pc);
                }
            }
        }
    }
    let mut volatile = RegisterSet::ALL;
    for register in 0..=255u8 {
        let Some(mut reach) = first_capture[usize::from(register)] else {
            continue;
        };
        // Code after the capture can run again from any loop header it
        // jumps back to.
        loop {
            let before = reach;
            for &(latch, header) in &backward {
                if latch >= reach && header < reach {
                    reach = header;
                }
            }
            if reach == before {
                break;
            }
        }
        if last_write[usize::from(register)].is_none_or(|pc| pc < reach) {
            volatile.remove(register);
        }
    }
    volatile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::UpvalueDescriptor;
    use crate::string::LuaString;

    fn child(register: u8) -> Chunk {
        Chunk::with_parts(
            1,
            0,
            Vec::new(),
            vec![Instruction::abc(OpCode::Return, 0, 1, 0)],
            Vec::new(),
            vec![UpvalueDescriptor {
                name: LuaString::from("x"),
                source: UpvalueSource::ParentRegister(register),
            }],
        )
    }

    fn parent(code: Vec<Instruction>) -> Chunk {
        Chunk::with_parts(8, 0, Vec::new(), code, vec![child(1)], Vec::new())
    }

    #[test]
    fn a_register_written_only_before_its_capture_is_stable() {
        let chunk = parent(vec![
            Instruction::abc(OpCode::LoadNil, 1, 0, 0),
            Instruction::closure(2, 0),
            Instruction::call(2, 0, 1),
        ]);
        assert!(!chunk.volatile_captures().contains(1));
        // Registers no closure captures stay volatile: nothing is proved.
        assert!(chunk.volatile_captures().contains(2));
    }

    #[test]
    fn a_write_after_the_capture_keeps_the_register_volatile() {
        let chunk = parent(vec![
            Instruction::abc(OpCode::LoadNil, 1, 0, 0),
            Instruction::closure(2, 0),
            Instruction::abc(OpCode::LoadNil, 1, 0, 0),
        ]);
        assert!(chunk.volatile_captures().contains(1));
    }

    #[test]
    fn a_call_at_or_below_the_register_keeps_it_volatile() {
        let chunk = parent(vec![
            Instruction::abc(OpCode::LoadNil, 1, 0, 0),
            Instruction::closure(2, 0),
            Instruction::call(1, 0, 1),
        ]);
        assert!(chunk.volatile_captures().contains(1));
    }

    #[test]
    fn a_loop_around_the_capture_makes_earlier_writes_count() {
        let chunk = parent(vec![
            Instruction::abc(OpCode::LoadNil, 1, 0, 0),
            Instruction::closure(2, 0),
            Instruction::aj(OpCode::Jump, -3).unwrap(),
        ]);
        assert!(chunk.volatile_captures().contains(1));
    }

    #[test]
    fn a_closure_that_captures_its_own_register_is_volatile() {
        let chunk = parent(vec![Instruction::closure(1, 0)]);
        assert!(chunk.volatile_captures().contains(1));
    }

    #[test]
    fn a_parameter_written_by_nothing_else_is_stable() {
        let mut chunk = parent(vec![Instruction::closure(2, 0)]);
        chunk.parameter_count = 2;
        assert!(!chunk.volatile_captures().contains(1));
    }

    #[test]
    fn clones_recompute_and_compare_equal() {
        let chunk = parent(vec![Instruction::closure(2, 0)]);
        let _ = chunk.volatile_captures();
        assert_eq!(chunk, chunk.clone());
    }
}
