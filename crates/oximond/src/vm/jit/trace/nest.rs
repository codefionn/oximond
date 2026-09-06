//! The loop structure of a chunk, read from its bytecode.
//!
//! A loop is what its backward branch shows: the header the branch lands on
//! and the pc of the branch (its latch). Nothing here looks at run time
//! values. The trace policy uses it to tell, before anything is recorded,
//! whether a trace tree could ever run the loops around a hot loop (see
//! [`super`], "Nested loops"), so that it does not wait for one that cannot.

use crate::bytecode::{Chunk, Constant, Instruction, OpCode};

/// A loop: the pc of its first instruction and of the backward branch that
/// closes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoopSpan {
    pub(crate) header: usize,
    pub(crate) latch: usize,
}

impl LoopSpan {
    pub(crate) const fn contains(self, pc: usize) -> bool {
        pc >= self.header && pc <= self.latch
    }
}

/// The pc `instruction` at `pc` jumps to, for the opcodes that jump.
pub(super) fn jump_target(pc: usize, instruction: Instruction) -> Option<usize> {
    let offset = match instruction.opcode() {
        Ok(OpCode::Jump) => instruction.jump_offset(),
        Ok(OpCode::JumpIfFalse | OpCode::NumericForLoop | OpCode::ForInLoop) => {
            i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
        }
        _ => return None,
    };
    let target = i64::try_from(pc).ok()? + 1 + i64::from(offset);
    usize::try_from(target).ok()
}

/// The loops whose backward branch is at a pc in `from..=to`, one per
/// header (the outermost latch when several branches land on one header).
// `pc` is the program counter: an index used as a value (`jump_target`,
// `LoopSpan::latch`), not just to slice `chunk.code`.
#[allow(clippy::needless_range_loop)]
fn backward_branches(chunk: &Chunk, from: usize, to: usize) -> Vec<LoopSpan> {
    let mut found: Vec<LoopSpan> = Vec::new();
    for pc in from..=to.min(chunk.code.len().saturating_sub(1)) {
        let Some(target) = jump_target(pc, chunk.code[pc]) else {
            continue;
        };
        if target > pc {
            continue;
        }
        match found.iter_mut().find(|span| span.header == target) {
            Some(span) => span.latch = span.latch.max(pc),
            None => found.push(LoopSpan {
                header: target,
                latch: pc,
            }),
        }
    }
    found
}

/// Every loop of `chunk`, one per header.
pub(crate) fn loops_of(chunk: &Chunk) -> Vec<LoopSpan> {
    backward_branches(chunk, 0, chunk.code.len().saturating_sub(1))
}

/// The loops that enclose `root`: backward branches after its latch that
/// land before its header. Innermost first.
pub(crate) fn enclosing_loops(chunk: &Chunk, root: LoopSpan) -> Vec<LoopSpan> {
    let mut found: Vec<LoopSpan> = backward_branches(chunk, root.latch + 1, chunk.code.len())
        .into_iter()
        .filter(|span| span.header < root.header)
        .collect();
    found.sort_by_key(|span| std::cmp::Reverse(span.header));
    found
}

/// The loops inside `outer`: backward branches in its span other than its
/// own closing branch.
pub(crate) fn inner_loops(chunk: &Chunk, outer: LoopSpan) -> Vec<LoopSpan> {
    backward_branches(chunk, outer.header, outer.latch)
        .into_iter()
        .filter(|span| span.header != outer.header && span.latch != outer.latch)
        .collect()
}

/// Whether two of `spans` lie side by side, neither inside the other.
pub(crate) fn has_siblings(spans: &[LoopSpan]) -> bool {
    spans.iter().enumerate().any(|(index, first)| {
        spans[index + 1..]
            .iter()
            .any(|second| first.latch < second.header || second.latch < first.header)
    })
}

/// The guest registers one instruction may write.
enum Written {
    None,
    /// `start..start + count`.
    Range(u8, usize),
    /// An instruction that does not decode: any register.
    Any,
}

fn written_by(instruction: Instruction) -> Written {
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
        | OpCode::GetTabUp
        | OpCode::CallOne => Written::Range(a, 1),
        // The method and the receiver copied after it.
        OpCode::SelfField => Written::Range(a, 2),
        OpCode::Closure | OpCode::Concat | OpCode::TableVararg | OpCode::TableOpen => {
            Written::Range(a, 1)
        }
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

/// The guest registers the code of the loops around `root` may write outside
/// `root` itself. A table a trace of `root` finds in one of them can be a
/// different table at every entry and, in a trace tree that runs through
/// the outer loops, at every iteration, so the trace cannot bind its storage
/// once at entry.
// `pc` is the program counter: an index used as a value (`root.contains`),
// not just to slice `chunk.code`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn written_around(chunk: &Chunk, root: LoopSpan) -> Vec<bool> {
    let mut written = vec![false; 256];
    for outer in enclosing_loops(chunk, root) {
        let last = outer.latch.min(chunk.code.len().saturating_sub(1));
        for pc in outer.header..=last {
            if root.contains(pc) {
                continue;
            }
            match written_by(chunk.code[pc]) {
                Written::None => {}
                Written::Range(start, count) => {
                    let start = usize::from(start);
                    for flag in written
                        .iter_mut()
                        .take((start + count).min(256))
                        .skip(start)
                    {
                        *flag = true;
                    }
                }
                Written::Any => return vec![true; 256],
            }
        }
    }
    written
}

/// Whether the code of `root` (its header to its latch) may write the guest
/// register `register`.
pub(crate) fn written_within(chunk: &Chunk, root: LoopSpan, register: usize) -> bool {
    let last = root.latch.min(chunk.code.len().saturating_sub(1));
    (root.header..=last).any(|pc| match written_by(chunk.code[pc]) {
        Written::None => false,
        Written::Range(start, count) => {
            let start = usize::from(start);
            (start..start + count).contains(&register)
        }
        Written::Any => true,
    })
}

/// Whether some call in `chunk` takes an argument that a table read filled:
/// the nearest instruction before the call that writes the argument register
/// is a `GetField` or `GetTable` (or a `Move` of such a register). The code
/// order stands for the control flow, so a register that branches fill
/// differently can make this say yes when it need not, which only costs
/// speed (see `FunctionSeed::lanes`).
pub(crate) fn passes_loaded_tables(chunk: &Chunk) -> bool {
    let holds_read = |pc: usize, register: u8| {
        let (mut pc, mut register) = (pc, register);
        // A copy of a copy is as far as the compiler's temporaries go.
        for _ in 0..4 {
            let writer = (0..pc)
                .rev()
                .find(|&earlier| match written_by(chunk.code[earlier]) {
                    Written::None => false,
                    Written::Range(start, count) => {
                        register >= start && usize::from(register) < usize::from(start) + count
                    }
                    Written::Any => true,
                });
            let Some(writer) = writer else {
                return false;
            };
            let instruction = chunk.code[writer];
            match instruction.opcode() {
                Ok(OpCode::GetField | OpCode::GetTable) => return true,
                Ok(OpCode::Move) => {
                    pc = writer;
                    register = instruction.b();
                }
                _ => return false,
            }
        }
        false
    };
    chunk
        .code
        .iter()
        .enumerate()
        .any(|(pc, instruction)| match instruction.opcode() {
            Ok(OpCode::CallOne) => holds_read(pc, instruction.c()),
            Ok(OpCode::Call | OpCode::CallOpen | OpCode::TailCall | OpCode::TailCallOpen) => {
                let base = usize::from(instruction.a());
                (base + 1..=base + usize::from(instruction.b()))
                    .filter_map(|register| u8::try_from(register).ok())
                    .any(|register| holds_read(pc, register))
            }
            _ => false,
        })
}

/// Whether the register `register` holds a string constant at `pc` of the
/// span that starts at `from`: the nearest instruction before `pc` that
/// writes it is a `LoadConstant` of a string, or a `Move` of a register that
/// holds one. A register the span has not written yet is unknown.
fn holds_string_constant(chunk: &Chunk, from: usize, pc: usize, register: u8) -> bool {
    let (mut pc, mut register) = (pc, register);
    // A copy of a copy is as far as the compiler's temporaries go.
    for _ in 0..4 {
        let writer = (from..pc)
            .rev()
            .find(|&earlier| match written_by(chunk.code[earlier]) {
                Written::None => false,
                Written::Range(start, count) => {
                    register >= start && usize::from(register) < usize::from(start) + count
                }
                Written::Any => true,
            });
        let Some(writer) = writer else {
            return false;
        };
        let instruction = chunk.code[writer];
        match instruction.opcode() {
            Ok(OpCode::LoadConstant) => {
                return matches!(
                    chunk.constants.get(usize::from(instruction.d())),
                    Some(Constant::String(_))
                );
            }
            Ok(OpCode::Move) => (pc, register) = (writer, instruction.b()),
            _ => return false,
        }
    }
    false
}

/// Whether the recorder follows an instruction with this opcode in a side
/// trace. A call is on the list: a side inlines the callee when it is a Lua
/// closure with no upvalue of its own, or a pure numeric native. Another
/// callee aborts the recording, which the guard's attempts then end.
const fn records_in_side(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::Move
            | OpCode::Call
            | OpCode::CallOne
            | OpCode::LoadConstant
            | OpCode::LoadNil
            | OpCode::ClearRange
            | OpCode::Not
            | OpCode::Negate
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
            | OpCode::GetUpvalue
            | OpCode::SetUpvalue
            | OpCode::GetGlobal
            | OpCode::GetTable
            | OpCode::SetTable
            | OpCode::GetField
            | OpCode::SelfField
            | OpCode::SetField
            | OpCode::GetTabUp
            | OpCode::SetTabUp
            | OpCode::Length
            | OpCode::Equal
            | OpCode::LessThan
            | OpCode::LessEqual
            | OpCode::Jump
            | OpCode::JumpIfFalse
            | OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK
            | OpCode::NumericForPrep
            | OpCode::NumericForLoop
            | OpCode::ForInCall
            | OpCode::ForInLoop
    )
}

/// Whether a side trace could run the part of `outer`'s iteration that lies
/// outside its inner loops, as far as the opcodes say. An instruction that
/// runs in every iteration and that the recorder never follows (a call, say)
/// makes every recording of that part abort. One that only a branch reaches
/// does not: the side records the iteration along the other arm, and the arm
/// with the call stays a guard exit. A jump out of the loop (a `break`, the
/// exit test of a `while`) skips nothing in the steady state, so it does not
/// make the instructions it jumps over conditional.
// `pc` is the program counter: an index used as a value (`jump_target`,
// `skipped`), not just to slice `chunk.code`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn outer_part_recordable(chunk: &Chunk, outer: LoopSpan) -> bool {
    let inner = inner_loops(chunk, outer);
    let mut skipped: Vec<(usize, usize)> = Vec::new();
    for pc in outer.header..=outer.latch.min(chunk.code.len().saturating_sub(1)) {
        let opcode = chunk.code[pc].opcode();
        if matches!(opcode, Ok(OpCode::Jump | OpCode::JumpIfFalse))
            && let Some(target) = jump_target(pc, chunk.code[pc])
            && target > pc
            && target <= outer.latch
        {
            skipped.push((pc + 1, target));
        }
    }
    let last = outer.latch.min(chunk.code.len().saturating_sub(1));
    let in_inner = |pc: usize| inner.iter().any(|span| span.contains(pc));
    let conditional = |pc: usize| skipped.iter().any(|(from, to)| pc >= *from && pc < *to);
    // A side meets a table read out of a table (`local row = m[i]`) by lane.
    // Its named fields are row reads, and with lane array views
    // (`TraceOp::LaneArrayLoad`, which needs the build's
    // `NativeTableLayout`) an index with a number reads the table's array
    // words inline; a length is a `TABLE_LENGTH` call and a store with a
    // number key a `TABLE_SET_INDEX` call. None of these makes a recording
    // abort, so with the layout known no use of such a table refuses the
    // outer part. Without it, an unconditional `GetTable` whose result is
    // indexed by number, measured or stored into by number later in the
    // loop (`local row = m[i]; row[j]`) still keeps the outer part from
    // being covered, as it did before lane views: each of those accesses
    // would be a lookup call per trip.
    let lane_views = crate::runtime::NativeTableLayout::get().is_some();
    let reads_table = |pc: usize| {
        if lane_views {
            return false;
        }
        let instruction = chunk.code[pc];
        if !matches!(
            instruction.opcode(),
            Ok(OpCode::GetTable | OpCode::GetField | OpCode::GetTabUp)
        ) {
            return false;
        }
        // The row may be copied to another local first (`local q = row`), so
        // the registers a `Move` copies it into count too. A register reused
        // for something else stays in the set, which can only refuse a loop
        // that was coverable, never accept one that is not.
        let mut holders = vec![instruction.a()];
        (pc + 1..=last).any(|later| {
            let user = chunk.code[later];
            let named = |register: u8| holds_string_constant(chunk, outer.header, later, register);
            match user.opcode() {
                Ok(OpCode::GetTable) => holders.contains(&user.b()) && !named(user.c()),
                Ok(OpCode::Length) => holders.contains(&user.b()),
                Ok(OpCode::SetTable) => holders.contains(&user.a()) && !named(user.b()),
                Ok(OpCode::Move) => {
                    if holders.contains(&user.b()) {
                        holders.push(user.a());
                    }
                    false
                }
                _ => false,
            }
        })
    };
    (outer.header..=last).all(|pc| {
        if in_inner(pc) {
            return true;
        }
        if conditional(pc) {
            return true;
        }
        chunk.code[pc].opcode().is_ok_and(records_in_side) && !reads_table(pc)
    })
}

/// Whether a trace tree could ever run the loop `outer`, given that it is
/// rooted at an inner loop: no two inner loops lie side by side (a recording
/// of one aborts at the other's backward branch), and what an outer side
/// would record of the iteration can be recorded.
pub(crate) fn outer_can_be_covered(chunk: &Chunk, outer: LoopSpan) -> bool {
    outer_can_be_covered_from(chunk, outer, None)
}

/// [`outer_can_be_covered`] for a tree rooted at `root`, a loop inside
/// `outer`, when that root ends in front of the loops inside its own (see
/// "Outer roots" in the module docs of `trace`): those loops are loop sides
/// of the tree, which its sides reach and leave like any other code, so they
/// count as no siblings. Two loops side by side elsewhere in `outer` still
/// do: an outer side would abort at the other one's backward branch.
pub(crate) fn outer_can_be_covered_from(
    chunk: &Chunk,
    outer: LoopSpan,
    root: Option<LoopSpan>,
) -> bool {
    let inner: Vec<LoopSpan> = inner_loops(chunk, outer)
        .into_iter()
        .filter(|span| {
            root.is_none_or(|root| !(span.header > root.header && span.latch < root.latch))
        })
        .collect();
    !has_siblings(&inner) && outer_part_recordable(chunk, outer)
}

/// Whether a trace tree rooted at `root` could ever run the loops around it:
/// a loop encloses it and [`outer_can_be_covered`] holds for the innermost one,
/// whose iteration every outer side starts by recording the rest of. When
/// this is false no outer side can succeed, so nothing waits for one.
pub(crate) fn can_cover(chunk: &Chunk, root: LoopSpan) -> bool {
    enclosing_loops(chunk, root)
        .first()
        .is_some_and(|outer| outer_can_be_covered(chunk, *outer))
}

/// [`can_cover`] for a root that ends in front of the loops inside its own
/// (see [`outer_can_be_covered_from`]): the loops inside it are no siblings.
pub(crate) fn can_cover_from_outer_root(chunk: &Chunk, root: LoopSpan) -> bool {
    enclosing_loops(chunk, root)
        .first()
        .is_some_and(|outer| outer_can_be_covered_from(chunk, *outer, Some(root)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loops of `source`, compiled as a top-level chunk: the one other
    /// loops enclose, and the chunk.
    fn innermost(source: &str) -> (Chunk, LoopSpan) {
        let chunk = crate::compiler::compile(source).expect("the source compiles");
        let loops = loops_of(&chunk);
        let inner = *loops
            .iter()
            .find(|span| {
                !enclosing_loops(&chunk, **span).is_empty()
                    && inner_loops(&chunk, **span).is_empty()
            })
            .expect("a loop inside another");
        (chunk, inner)
    }

    #[test]
    fn an_outer_part_that_indexes_and_measures_a_table_read_out_of_a_table_is_coverable() {
        // `row` is read out of `m`, then indexed by number, measured and
        // stored into by number in the outer part. A side meets it by lane:
        // the reads are lane array views, the length and the store helper
        // calls, so nothing there aborts a recording.
        let (chunk, inner) = innermost(
            "local m = { {1, 2}, {3, 4, 5} }
             local s, i = 0, 1
             while i <= #m do
               local row = m[i]
               local j = 1
               while j <= 2 do s = s + j; j = j + 1 end
               s = s + #row + row[1]
               row[2] = s
               i = i + 1
             end
             return s",
        );
        assert!(crate::runtime::NativeTableLayout::get().is_some());
        assert!(can_cover(&chunk, inner));
    }

    #[test]
    fn an_outer_part_with_an_instruction_no_side_records_is_not_coverable() {
        // A concatenation in every outer iteration still refuses the nest.
        let (chunk, inner) = innermost(
            "local s, i, last = 0, 1, nil
             while i <= 10 do
               local j = 1
               while j <= 2 do s = s + j; j = j + 1 end
               last = 'k' .. i
               i = i + 1
             end
             return s, last",
        );
        assert!(!can_cover(&chunk, inner));
    }

    /// The loop of `source` that encloses another and is enclosed by one.
    fn middle(source: &str) -> (Chunk, LoopSpan) {
        let chunk = crate::compiler::compile(source).expect("the source compiles");
        let middle = *loops_of(&chunk)
            .iter()
            .find(|span| {
                !enclosing_loops(&chunk, **span).is_empty()
                    && !inner_loops(&chunk, **span).is_empty()
            })
            .expect("a loop between two others");
        (chunk, middle)
    }

    #[test]
    fn loops_inside_an_outer_root_are_no_siblings_of_it() {
        // The middle loop holds two loops side by side. Rooted at an inner
        // loop the outermost loop cannot be covered, but rooted at the middle
        // loop, which runs both as loop sides, it can.
        let (chunk, root) = middle(
            "local s, i = 0, 1
             while i <= 10 do
               local k = 1
               while k <= 3 do
                 local j = 1
                 while j <= 2 do s = s + j; j = j + 1 end
                 local l = 1
                 while l <= 2 do s = s + l; l = l + 1 end
                 k = k + 1
               end
               i = i + 1
             end
             return s",
        );
        assert!(!can_cover(&chunk, root));
        assert!(can_cover_from_outer_root(&chunk, root));
    }

    #[test]
    fn a_loop_beside_an_outer_root_is_still_a_sibling() {
        // The loop after the middle one lies beside it in the outermost loop:
        // an outer side of the middle loop's tree would abort there.
        let (chunk, root) = middle(
            "local s, i = 0, 1
             while i <= 10 do
               local k = 1
               while k <= 3 do
                 local j = 1
                 while j <= 2 do s = s + j; j = j + 1 end
                 k = k + 1
               end
               local l = 1
               while l <= 2 do s = s + l; l = l + 1 end
               i = i + 1
             end
             return s",
        );
        assert!(!can_cover_from_outer_root(&chunk, root));
    }
}
