//! Inline call plans for tiny numeric closures and their validation.

use crate::bytecode::{Chunk, Constant, Instruction, OpCode};
use crate::compat::{CompatibilityProfile, FloatArith};
use crate::jit_ir::digests::ChunkDigest;
use crate::jit_ir::operation::Comparison;
use core::fmt;

/// One numeric operation in an immutable fixed-arity closure inline plan.
///
/// Registers here belong to the callee, not to the caller. A plan is
/// straight-line and begins with its parameters in the low registers. A
/// callee with forward branches is if-converted: every operation runs, and
/// `Select` merges the values that reach a join or a return. Conditions are
/// `1.0` (true) or `0.0` (false) and never leave the plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InlineNumericOperation {
    /// Reads one of the callee's incoming varargs into a local register.
    LoadVararg {
        destination: u8,
        index: u8,
    },
    /// Reads the sole captured numeric cell into a callee-local register.
    LoadUpvalue {
        destination: u8,
        upvalue: u16,
    },
    LoadNumber {
        destination: u8,
        value: f64,
    },
    Move {
        destination: u8,
        source: u8,
    },
    Negate {
        destination: u8,
        source: u8,
    },
    Add {
        destination: u8,
        left: u8,
        right: u8,
    },
    Subtract {
        destination: u8,
        left: u8,
        right: u8,
    },
    Multiply {
        destination: u8,
        left: u8,
        right: u8,
    },
    Divide {
        destination: u8,
        left: u8,
        right: u8,
    },
    Modulo {
        destination: u8,
        left: u8,
        right: u8,
    },
    Power {
        destination: u8,
        left: u8,
        right: u8,
    },
    /// Writes a callee-local numeric register back to the captured cell.
    StoreUpvalue {
        source: u8,
        upvalue: u16,
    },
    /// Compares two numbers into a condition.
    Compare {
        destination: u8,
        left: u8,
        right: u8,
        comparison: Comparison,
    },
    /// Inverts a condition.
    Not {
        destination: u8,
        source: u8,
    },
    /// Picks `if_true` when `condition` holds, else `if_false`.
    Select {
        destination: u8,
        condition: u8,
        if_true: u8,
        if_false: u8,
    },
}

impl InlineNumericOperation {
    #[must_use]
    pub const fn destination(self) -> u8 {
        match self {
            Self::LoadVararg { destination, .. }
            | Self::LoadUpvalue { destination, .. }
            | Self::LoadNumber { destination, .. }
            | Self::Move { destination, .. }
            | Self::Negate { destination, .. }
            | Self::Add { destination, .. }
            | Self::Subtract { destination, .. }
            | Self::Multiply { destination, .. }
            | Self::Divide { destination, .. }
            | Self::Modulo { destination, .. }
            | Self::Power { destination, .. }
            | Self::Compare { destination, .. }
            | Self::Not { destination, .. }
            | Self::Select { destination, .. } => destination,
            Self::StoreUpvalue { source, .. } => source,
        }
    }
}

/// Immutable, allocation-free-at-execution representation of a tiny numeric
/// Lua closure. It deliberately contains no runtime handle, heap pointer, or
/// borrowed VM state, so optimized compilation can safely happen off-thread.
#[derive(Clone, Debug, PartialEq)]
pub struct InlineCallPlan {
    callee_digest: ChunkDigest,
    register_count: u8,
    parameter_count: u8,
    result_register: u8,
    upvalue_count: u16,
    /// Cached [`Self::minimum_argument_count`]; the plan is immutable.
    minimum_argument_count: u8,
    /// Whether a forward scan of `operations` proved that every register an
    /// operation reads (and the result register) was written by a parameter
    /// or an earlier operation, with all indices below `register_count`. Such
    /// a plan needs no zeroed scratch.
    writes_before_reads: bool,
    /// Cached [`Self::writes_upvalue`]; the plan is immutable.
    writes_upvalue: bool,
    /// True when the callee loads an integer constant. Plans evaluate in
    /// `f64`, so under a profile with an integer subtype the real call can
    /// return an integer the plan cannot represent.
    loads_integer_constant: bool,
    /// Instructions a count hook counts over one call of the callee, when
    /// every call runs the same ones (straight-line bodies). `None` when the
    /// count depends on the path or is not known.
    counted_instructions: Option<u32>,
    /// How the plan computes `%` and `^`. Defaults to `LuaJIT`'s; the VM sets
    /// the profile's with [`Self::with_float_arith`].
    float_arith: FloatArith,
    operations: Vec<InlineNumericOperation>,
}

impl InlineCallPlan {
    /// Semantic identity of the closure body this plan was validated from.
    #[must_use]
    pub const fn callee_digest(&self) -> ChunkDigest {
        self.callee_digest
    }

    /// Number of callee-local registers used by this straight-line plan.
    #[must_use]
    pub const fn register_count(&self) -> u8 {
        self.register_count
    }

    /// Number of positional parameters initialized by the caller.
    #[must_use]
    pub const fn parameter_count(&self) -> u8 {
        self.parameter_count
    }

    /// Smallest fixed call window that can evaluate this numeric plan.
    #[must_use]
    pub const fn minimum_argument_count(&self) -> u8 {
        self.minimum_argument_count
    }

    /// The plan with the profile's float `%` and `^`.
    #[must_use]
    pub const fn with_float_arith(mut self, float_arith: FloatArith) -> Self {
        self.float_arith = float_arith;
        self
    }

    /// Whether the callee loads an integer constant. Integer profiles must not
    /// apply such a plan, because its `f64` result loses the integer subtype.
    #[must_use]
    pub const fn loads_integer_constant(&self) -> bool {
        self.loads_integer_constant
    }

    /// Instructions a count hook counts over one call of the callee, or
    /// `None` when that depends on the path taken. A direct evaluation under
    /// a count hook charges the hook this many.
    #[must_use]
    pub const fn counted_instructions(&self) -> Option<u32> {
        self.counted_instructions
    }

    /// Callee-local register returned to the caller.
    #[must_use]
    pub const fn result_register(&self) -> u8 {
        self.result_register
    }

    /// Number of captured cells the plan reads or writes.
    #[must_use]
    pub const fn upvalue_count(&self) -> u16 {
        self.upvalue_count
    }

    /// The immutable numeric operations evaluated before the return.
    #[must_use]
    pub fn operations(&self) -> &[InlineNumericOperation] {
        &self.operations
    }

    /// Whether evaluating this plan mutates its captured cell.
    #[must_use]
    pub const fn writes_upvalue(&self) -> bool {
        self.writes_upvalue
    }

    /// Evaluates the plan for a caller-proven numeric argument.
    ///
    /// The validator proves every read is initialized, so this needs neither
    /// Lua coercion nor a metamethod path. Keeping the scratch registers on
    /// the stack avoids allocating a temporary frame for interpreter calls.
    #[must_use]
    pub fn evaluate(&self, argument: f64) -> f64 {
        debug_assert_eq!(self.upvalue_count, 0);
        self.evaluate_arguments_with_upvalues(&[argument], &mut [])
    }

    /// Evaluates the plan with the caller-provided numeric captured cells.
    ///
    /// The validator accepts at most one captured cell and orders every read
    /// before its write. Callers therefore can validate the live cell before
    /// any mutation, then write the resulting value back without a Lua frame.
    #[must_use]
    pub fn evaluate_with_upvalues(&self, argument: f64, upvalues: &mut [f64]) -> f64 {
        self.evaluate_arguments_with_upvalues(&[argument], upvalues)
    }

    /// Whether a forward scan proved that every register the plan reads is
    /// written first by a parameter or an earlier operation. Such a plan
    /// evaluates correctly on scratch registers that start uninitialised,
    /// provided the caller passes at least [`Self::parameter_count`] arguments.
    #[must_use]
    pub const fn writes_before_reads(&self) -> bool {
        self.writes_before_reads
    }

    /// Evaluates a fixed numeric argument window and numeric captured cells.
    #[must_use]
    pub fn evaluate_arguments_with_upvalues(&self, arguments: &[f64], upvalues: &mut [f64]) -> f64 {
        debug_assert_eq!(upvalues.len(), usize::from(self.upvalue_count));
        // Zeroing scratch costs per call, so size it to the plan: most plans
        // use a handful of registers.
        let count = usize::from(self.register_count);
        if count <= 8 {
            return self.evaluate_in(arguments, upvalues, &mut [0.0_f64; 8]);
        }
        if count <= 16 {
            return self.evaluate_in(arguments, upvalues, &mut [0.0_f64; 16]);
        }
        if count <= 64 {
            return self.evaluate_in(arguments, upvalues, &mut [0.0_f64; 64]);
        }
        self.evaluate_in(arguments, upvalues, &mut Box::new([0.0_f64; 256]))
    }

    /// Runs the plan over caller-provided scratch registers.
    ///
    /// The validator proves every register index is below `register_count`,
    /// and `S` has a power-of-two number of slots at least that large (this
    /// panics otherwise). Masking each index with the slot count minus one is
    /// then the identity, and it lets the compiler drop the bounds check on
    /// every access. The scratch need not be zeroed when
    /// [`Self::writes_before_reads`] holds and `arguments` covers every
    /// parameter, because no register is read before it is written.
    ///
    /// # Panics
    ///
    /// Panics when `S` has fewer slots than the plan's register count or a
    /// slot count that is not a power of two, and when `arguments` omits a
    /// vararg a `LoadVararg` operation reads.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn evaluate_in<S: RegisterFile>(
        &self,
        arguments: &[f64],
        upvalues: &mut [f64],
        registers: &mut S,
    ) -> f64 {
        assert!(
            S::SLOTS.is_power_of_two() && S::SLOTS >= usize::from(self.register_count),
            "scratch registers cover the plan"
        );
        for (register, argument) in arguments
            .iter()
            .take(usize::from(self.parameter_count))
            .enumerate()
        {
            registers.write(register, *argument);
        }
        for operation in &self.operations {
            let (destination, value) = match *operation {
                InlineNumericOperation::LoadVararg { destination, index } => (
                    destination,
                    arguments
                        .get(usize::from(self.parameter_count) + usize::from(index))
                        .copied()
                        .expect("inline caller supplied every required vararg"),
                ),
                InlineNumericOperation::LoadUpvalue {
                    destination,
                    upvalue,
                } => (destination, upvalues[usize::from(upvalue)]),
                InlineNumericOperation::LoadNumber { destination, value } => (destination, value),
                InlineNumericOperation::Move {
                    destination,
                    source,
                } => (destination, registers.read(usize::from(source))),
                InlineNumericOperation::Negate {
                    destination,
                    source,
                } => (destination, -registers.read(usize::from(source))),
                InlineNumericOperation::Add {
                    destination,
                    left,
                    right,
                } => (
                    destination,
                    registers.read(usize::from(left)) + registers.read(usize::from(right)),
                ),
                InlineNumericOperation::Subtract {
                    destination,
                    left,
                    right,
                } => (
                    destination,
                    registers.read(usize::from(left)) - registers.read(usize::from(right)),
                ),
                InlineNumericOperation::Multiply {
                    destination,
                    left,
                    right,
                } => (
                    destination,
                    registers.read(usize::from(left)) * registers.read(usize::from(right)),
                ),
                InlineNumericOperation::Divide {
                    destination,
                    left,
                    right,
                } => (
                    destination,
                    registers.read(usize::from(left)) / registers.read(usize::from(right)),
                ),
                InlineNumericOperation::Modulo {
                    destination,
                    left,
                    right,
                } => {
                    let left = registers.read(usize::from(left));
                    let right = registers.read(usize::from(right));
                    (destination, self.float_arith.modulo(left, right))
                }
                InlineNumericOperation::Power {
                    destination,
                    left,
                    right,
                } => (
                    destination,
                    self.float_arith.power(
                        registers.read(usize::from(left)),
                        registers.read(usize::from(right)),
                    ),
                ),
                InlineNumericOperation::StoreUpvalue { source, upvalue } => {
                    let value = registers.read(usize::from(source));
                    upvalues[usize::from(upvalue)] = value;
                    (source, value)
                }
                InlineNumericOperation::Compare {
                    destination,
                    left,
                    right,
                    comparison,
                } => {
                    let left = registers.read(usize::from(left));
                    let right = registers.read(usize::from(right));
                    #[allow(clippy::float_cmp, reason = "Lua compares numbers exactly")]
                    let holds = match comparison {
                        Comparison::Equal => left == right,
                        Comparison::LessThan => left < right,
                        Comparison::LessEqual => left <= right,
                    };
                    (destination, f64::from(u8::from(holds)))
                }
                InlineNumericOperation::Not {
                    destination,
                    source,
                } => (destination, 1.0 - registers.read(usize::from(source))),
                InlineNumericOperation::Select {
                    destination,
                    condition,
                    if_true,
                    if_false,
                } => (
                    destination,
                    if registers.read(usize::from(condition)) == 0.0 {
                        registers.read(usize::from(if_false))
                    } else {
                        registers.read(usize::from(if_true))
                    },
                ),
            };
            registers.write(usize::from(destination), value);
        }
        registers.read(usize::from(self.result_register))
    }
}

/// Scratch registers for [`InlineCallPlan::evaluate_in`]: `SLOTS` (a power of
/// two) numbers, addressed modulo `SLOTS`.
pub trait RegisterFile {
    /// Number of slots; a power of two.
    const SLOTS: usize;
    /// Reads slot `index % SLOTS`.
    fn read(&self, index: usize) -> f64;
    /// Writes slot `index % SLOTS`.
    fn write(&mut self, index: usize, value: f64);
}

impl<const N: usize> RegisterFile for [f64; N] {
    const SLOTS: usize = N;

    #[inline(always)]
    fn read(&self, index: usize) -> f64 {
        self[index & (N - 1)]
    }

    #[inline(always)]
    fn write(&mut self, index: usize, value: f64) {
        self[index & (N - 1)] = value;
    }
}

impl RegisterFile for Box<[f64; 256]> {
    const SLOTS: usize = 256;

    #[inline(always)]
    fn read(&self, index: usize) -> f64 {
        self[index & 255]
    }

    #[inline(always)]
    fn write(&mut self, index: usize, value: f64) {
        self[index & 255] = value;
    }
}

/// Why a closure cannot become an inline numeric plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InlinePlanError {
    InvalidChunk,
    /// Retained for callers that distinguish historical one-argument refusals.
    /// New plans accept all fixed arities.
    WrongParameterCount {
        actual: u8,
    },
    /// Retained for callers that distinguish historical vararg refusals.
    /// Fixed vararg loads are now part of the numeric subset.
    Vararg,
    Captures,
    TooManyUpvalues {
        actual: usize,
    },
    InvalidUpvalue {
        pc: usize,
        index: u16,
    },
    NestedClosures,
    NonNumericConstant {
        pc: usize,
        index: u32,
        kind: ConstantKind,
    },
    UnsupportedOpcode {
        pc: usize,
        opcode: OpCode,
    },
    UninitializedRegister {
        pc: usize,
        register: u8,
    },
    WrongReturnCount {
        pc: usize,
        count: u8,
    },
    MissingReturn,
    InstructionAfterReturn {
        pc: usize,
        opcode: OpCode,
    },
}

impl fmt::Display for InlinePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChunk => formatter.write_str("inline candidate has invalid bytecode"),
            Self::WrongParameterCount { actual } => {
                write!(formatter, "inline candidate has {actual} fixed parameters")
            }
            Self::Vararg => formatter.write_str("inline candidate is vararg"),
            Self::Captures => formatter.write_str("inline candidate captures upvalues"),
            Self::TooManyUpvalues { actual } => write!(
                formatter,
                "inline candidate captures {actual} upvalues, expected at most one"
            ),
            Self::InvalidUpvalue { pc, index } => write!(
                formatter,
                "inline candidate reads unsupported upvalue {index} at PC {pc}"
            ),
            Self::NestedClosures => formatter.write_str("inline candidate creates nested closures"),
            Self::NonNumericConstant { pc, index, kind } => write!(
                formatter,
                "inline candidate reads non-numeric constant {index:?} ({kind:?}) at PC {pc}"
            ),
            Self::UnsupportedOpcode { pc, opcode } => {
                write!(formatter, "inline candidate uses {opcode:?} at PC {pc}")
            }
            Self::UninitializedRegister { pc, register } => write!(
                formatter,
                "inline candidate reads uninitialized register {register} at PC {pc}"
            ),
            Self::WrongReturnCount { pc, count } => {
                write!(
                    formatter,
                    "inline candidate returns {count} values at PC {pc}"
                )
            }
            Self::MissingReturn => formatter.write_str("inline candidate has no scalar return"),
            Self::InstructionAfterReturn { pc, opcode } => write!(
                formatter,
                "inline candidate has {opcode:?} after its return at PC {pc}"
            ),
        }
    }
}

impl std::error::Error for InlinePlanError {}

/// Validates a closure body for the deliberately small numeric inline subset.
///
/// The returned plan contains only owned bytecode-derived data. In particular,
/// it does not carry a closure handle, environment, captured cell, or VM
/// address. Callers must still verify that the live callee has this semantic
/// digest and no captured cells immediately before native entry.
///
/// # Errors
/// Returns a precise refusal reason whenever the bytecode can observe Lua
/// state, needs a non-numeric value, loops, or does not return one
/// initialized numeric register. Forward branches are if-converted.
#[allow(
    clippy::too_many_lines,
    reason = "the accepted bytecode grammar is intentionally visible in one exhaustive match"
)]
pub fn validate_inline_call_plan(chunk: &Chunk) -> Result<InlineCallPlan, InlinePlanError> {
    chunk
        .validate()
        .map_err(|_| InlinePlanError::InvalidChunk)?;
    validate_inline_call_plan_after_validation(chunk)
}

/// Validates an inline plan for a nested closure whose enclosing root chunk
/// has already been validated.
///
/// A nested closure may legitimately refer to a parent register or upvalue,
/// but root validation rejects those descriptors. The VM calls this only for
/// a live nested closure from a chunk tree it validated before execution.
/// Other callers should use [`validate_inline_call_plan`].
///
/// # Errors
/// Returns a precise refusal reason whenever the body is outside the small
/// numeric inline subset.
///
/// # Panics
/// Panics if `chunk` has not already passed [`Chunk::validate`].
pub fn validate_inline_call_plan_after_validation(
    chunk: &Chunk,
) -> Result<InlineCallPlan, InlinePlanError> {
    match validate_straight_line(chunk) {
        Err(
            error @ (InlinePlanError::UnsupportedOpcode { .. }
            | InlinePlanError::InstructionAfterReturn { .. }),
        ) => crate::jit_ir::inline_branches::validate_branching(chunk).ok_or(error),
        result => result,
    }
}

/// Whether every register read by `operations`, and the result, is written
/// first by a parameter or an earlier operation, all within `register_count`.
fn writes_before_reads(
    parameter_count: u8,
    register_count: u8,
    result_register: u8,
    operations: &[InlineNumericOperation],
) -> bool {
    use InlineNumericOperation as Op;
    let count = usize::from(register_count);
    if usize::from(parameter_count) > count {
        return false;
    }
    let mut written = vec![false; count];
    for slot in written.iter_mut().take(usize::from(parameter_count)) {
        *slot = true;
    }
    let read = |written: &[bool], register: u8| written.get(usize::from(register)) == Some(&true);
    for operation in operations {
        let reads_hold = match *operation {
            Op::LoadVararg { .. } | Op::LoadUpvalue { .. } | Op::LoadNumber { .. } => true,
            Op::Move { source, .. } | Op::Negate { source, .. } | Op::Not { source, .. } => {
                read(&written, source)
            }
            Op::StoreUpvalue { source, .. } => read(&written, source),
            Op::Add { left, right, .. }
            | Op::Subtract { left, right, .. }
            | Op::Multiply { left, right, .. }
            | Op::Divide { left, right, .. }
            | Op::Modulo { left, right, .. }
            | Op::Power { left, right, .. }
            | Op::Compare { left, right, .. } => read(&written, left) && read(&written, right),
            Op::Select {
                condition,
                if_true,
                if_false,
                ..
            } => read(&written, condition) && read(&written, if_true) && read(&written, if_false),
        };
        let Some(slot) = written.get_mut(usize::from(operation.destination())) else {
            return false;
        };
        if !reads_hold {
            return false;
        }
        *slot = true;
    }
    read(&written, result_register)
}

fn stores_upvalue(operations: &[InlineNumericOperation]) -> bool {
    operations
        .iter()
        .any(|operation| matches!(operation, InlineNumericOperation::StoreUpvalue { .. }))
}

/// Smallest fixed call window that covers every parameter and vararg read.
fn minimum_argument_count(parameter_count: u8, operations: &[InlineNumericOperation]) -> u8 {
    operations
        .iter()
        .fold(parameter_count, |count, operation| match operation {
            InlineNumericOperation::LoadVararg { index, .. } => {
                count.max(parameter_count.saturating_add(*index).saturating_add(1))
            }
            _ => count,
        })
}

/// The instructions a count hook counts when a straight-line body runs to its
/// return: every instruction up to and including the `Return`, minus the free
/// ones. `None` for a body with a two-word instruction, which this does not
/// model, or without a plain `Return`.
fn straight_line_counted_instructions(chunk: &Chunk) -> Option<u32> {
    let mut counted = 0_u32;
    for instruction in &chunk.code {
        let opcode = instruction.opcode().ok()?;
        if matches!(opcode, OpCode::LoadConstantWide | OpCode::ExtraArgument) {
            return None;
        }
        counted += u32::from(opcode.counts_for_hooks());
        if opcode == OpCode::Return {
            return Some(counted);
        }
    }
    None
}

/// Whether any constant load in `chunk` reads an integer constant.
fn chunk_loads_integer_constant(chunk: &Chunk) -> bool {
    let mut pc = 0;
    while let Some(instruction) = chunk.code.get(pc) {
        let index = match instruction.opcode() {
            Ok(OpCode::LoadConstant) => Some(u32::from(instruction.d())),
            Ok(
                OpCode::AddK
                | OpCode::SubtractK
                | OpCode::MultiplyK
                | OpCode::DivideK
                | OpCode::ModuloK,
            ) => Some(u32::from(instruction.c())),
            Ok(OpCode::TestEqualK | OpCode::TestLessThanK | OpCode::TestLessEqualK) => {
                Some(u32::from(instruction.b()))
            }
            Ok(OpCode::LoadConstantWide) => {
                pc += 1;
                chunk.code.get(pc).map(|extra| extra.extra_argument_index())
            }
            _ => None,
        };
        if let Some(index) = index
            && matches!(
                usize::try_from(index)
                    .ok()
                    .and_then(|i| chunk.constants.get(i)),
                Some(Constant::Integer(_))
            )
        {
            return true;
        }
        pc += 1;
    }
    false
}

impl InlineCallPlan {
    /// Builds a plan the branch if-converter already checked.
    pub(crate) fn from_parts(
        chunk: &Chunk,
        register_count: u8,
        result_register: u8,
        operations: Vec<InlineNumericOperation>,
    ) -> Self {
        Self {
            callee_digest: ChunkDigest::of(chunk),
            register_count,
            parameter_count: chunk.parameter_count,
            result_register,
            upvalue_count: u16::try_from(chunk.upvalues.len())
                .expect("validated bytecode upvalue count fits u16"),
            minimum_argument_count: minimum_argument_count(chunk.parameter_count, &operations),
            writes_before_reads: writes_before_reads(
                chunk.parameter_count,
                register_count,
                result_register,
                &operations,
            ),
            writes_upvalue: stores_upvalue(&operations),
            loads_integer_constant: chunk_loads_integer_constant(chunk),
            counted_instructions: None,
            float_arith: CompatibilityProfile::LuaJit21.float_arith(),
            operations,
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the accepted bytecode grammar is intentionally visible in one exhaustive match"
)]
fn validate_straight_line(chunk: &Chunk) -> Result<InlineCallPlan, InlinePlanError> {
    if chunk.upvalues.len() > 1 {
        return Err(InlinePlanError::TooManyUpvalues {
            actual: chunk.upvalues.len(),
        });
    }
    if !chunk.prototypes.is_empty() {
        return Err(InlinePlanError::NestedClosures);
    }

    let mut initialized = vec![false; usize::from(chunk.register_count)];
    for initialized in initialized
        .iter_mut()
        .take(usize::from(chunk.parameter_count))
    {
        *initialized = true;
    }
    let mut operations = Vec::new();
    let mut result_register = None;
    let mut accessed_upvalue = false;
    let mut uses_scratch = false;
    let mut pc = 0;
    while pc < chunk.code.len() {
        let instruction = chunk.code[pc];
        let opcode = instruction
            .opcode()
            .map_err(|_| InlinePlanError::InvalidChunk)?;
        if result_register.is_some() {
            if opcode == OpCode::Halt && pc + 1 == chunk.code.len() {
                break;
            }
            return Err(InlinePlanError::InstructionAfterReturn { pc, opcode });
        }
        match opcode {
            OpCode::Vararg if instruction.b() > 0 => {
                for offset in 0..instruction.b() {
                    let destination = instruction.a() + offset;
                    initialized[usize::from(destination)] = true;
                    operations.push(InlineNumericOperation::LoadVararg {
                        destination,
                        index: offset,
                    });
                }
                pc += 1;
            }
            OpCode::GetUpvalue => {
                let upvalue = instruction.d();
                if upvalue != 0 || chunk.upvalues.is_empty() {
                    return Err(InlinePlanError::InvalidUpvalue { pc, index: upvalue });
                }
                let destination = instruction.a();
                initialized[usize::from(destination)] = true;
                accessed_upvalue = true;
                operations.push(InlineNumericOperation::LoadUpvalue {
                    destination,
                    upvalue,
                });
                pc += 1;
            }
            OpCode::SetUpvalue => {
                let upvalue = instruction.d();
                if upvalue != 0 || chunk.upvalues.is_empty() {
                    return Err(InlinePlanError::InvalidUpvalue { pc, index: upvalue });
                }
                let source = instruction.a();
                inline_initialized(&initialized, pc, source)?;
                accessed_upvalue = true;
                operations.push(InlineNumericOperation::StoreUpvalue { source, upvalue });
                pc += 1;
            }
            OpCode::LoadConstant | OpCode::LoadConstantWide => {
                let index = if opcode == OpCode::LoadConstant {
                    u32::from(instruction.d())
                } else {
                    chunk.code[pc + 1].extra_argument_index()
                };
                let constant = usize::try_from(index)
                    .ok()
                    .and_then(|index| chunk.constants.get(index))
                    .ok_or(InlinePlanError::InvalidChunk)?;
                let value = inline_numeric_constant(constant).ok_or(
                    InlinePlanError::NonNumericConstant {
                        pc,
                        index,
                        kind: constant.into(),
                    },
                )?;
                let destination = instruction.a();
                initialized[usize::from(destination)] = true;
                operations.push(InlineNumericOperation::LoadNumber { destination, value });
                pc += if opcode == OpCode::LoadConstantWide {
                    2
                } else {
                    1
                };
            }
            OpCode::Move => {
                let destination = instruction.a();
                let source = instruction.b();
                inline_initialized(&initialized, pc, source)?;
                initialized[usize::from(destination)] = true;
                operations.push(InlineNumericOperation::Move {
                    destination,
                    source,
                });
                pc += 1;
            }
            OpCode::ClearRange => {
                let first = instruction.a();
                let end = first + instruction.b();
                for register in first..end {
                    initialized[usize::from(register)] = false;
                }
                pc += 1;
            }
            OpCode::Negate => {
                let destination = instruction.a();
                let source = instruction.b();
                inline_initialized(&initialized, pc, source)?;
                initialized[usize::from(destination)] = true;
                operations.push(InlineNumericOperation::Negate {
                    destination,
                    source,
                });
                pc += 1;
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::Power => {
                let destination = instruction.a();
                let left = instruction.b();
                let right = instruction.c();
                inline_initialized(&initialized, pc, left)?;
                inline_initialized(&initialized, pc, right)?;
                initialized[usize::from(destination)] = true;
                operations.push(inline_numeric_binary(opcode, destination, left, right));
                pc += 1;
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                // The constant loads into a scratch register above the
                // chunk's own, then the register form runs, as the compiler's
                // `LoadConstant` and arithmetic pair did.
                let index = u32::from(instruction.c());
                let constant = chunk
                    .constants
                    .get(usize::from(instruction.c()))
                    .ok_or(InlinePlanError::InvalidChunk)?;
                let value = inline_numeric_constant(constant).ok_or(
                    InlinePlanError::NonNumericConstant {
                        pc,
                        index,
                        kind: constant.into(),
                    },
                )?;
                let scratch = chunk.register_count;
                if scratch == u8::MAX {
                    return Err(InlinePlanError::InvalidChunk);
                }
                let destination = instruction.a();
                let left = instruction.b();
                inline_initialized(&initialized, pc, left)?;
                initialized[usize::from(destination)] = true;
                uses_scratch = true;
                operations.push(InlineNumericOperation::LoadNumber {
                    destination: scratch,
                    value,
                });
                operations.push(inline_numeric_binary(
                    opcode.register_form().expect("constant arithmetic opcode"),
                    destination,
                    left,
                    scratch,
                ));
                pc += 1;
            }
            OpCode::Return => {
                if instruction.b() != 1 {
                    return Err(InlinePlanError::WrongReturnCount {
                        pc,
                        count: instruction.b(),
                    });
                }
                inline_initialized(&initialized, pc, instruction.a())?;
                result_register = Some(instruction.a());
                pc += 1;
            }
            unsupported => {
                return Err(InlinePlanError::UnsupportedOpcode {
                    pc,
                    opcode: unsupported,
                });
            }
        }
    }
    let result_register = result_register.ok_or(InlinePlanError::MissingReturn)?;
    if !chunk.upvalues.is_empty() && !accessed_upvalue {
        return Err(InlinePlanError::Captures);
    }
    let minimum_argument_count = minimum_argument_count(chunk.parameter_count, &operations);
    Ok(InlineCallPlan {
        callee_digest: ChunkDigest::of(chunk),
        register_count: chunk.register_count + u8::from(uses_scratch),
        parameter_count: chunk.parameter_count,
        result_register,
        upvalue_count: u16::try_from(chunk.upvalues.len())
            .expect("validated bytecode upvalue count fits u16"),
        minimum_argument_count,
        writes_before_reads: writes_before_reads(
            chunk.parameter_count,
            chunk.register_count + u8::from(uses_scratch),
            result_register,
            &operations,
        ),
        writes_upvalue: stores_upvalue(&operations),
        loads_integer_constant: chunk_loads_integer_constant(chunk),
        counted_instructions: straight_line_counted_instructions(chunk),
        float_arith: CompatibilityProfile::LuaJit21.float_arith(),
        operations,
    })
}

/// Recognizes an already-validated nested closure that only forwards every
/// incoming vararg to its sole captured target.
///
/// The compiler emits this exact sequence for `return target(...)`:
/// `GetUpvalue`, zero or more moves that carry that closure to the tail-call
/// base, `TailCallVararg` with no fixed arguments, `ReturnOpen`, and `Halt`.
/// A fixed-parameter wrapper is deliberately refused because its vararg range
/// starts after those parameters and therefore does not map to a caller's
/// fixed numeric argument window.
///
/// `target` must already be a capture-free numeric plan for the captured live
/// closure. On success this returns a clone of `target`, including its digest,
/// parameter count, result register, and operations. The VM guards the
/// wrapper's digest and captured target separately. Keeping the target digest
/// in the result means a cached plan is invalidated when the target body
/// changes.
///
/// This intentionally does not call [`Chunk::validate`]. A forwarding wrapper
/// is normally nested and may have a `ParentRegister` or `ParentUpvalue`
/// descriptor that cannot validate as a root. Call it only after validating
/// the enclosing chunk tree.
#[must_use]
pub fn compose_vararg_forwarding_inline_call_plan_after_validation(
    wrapper: &Chunk,
    target: &InlineCallPlan,
) -> Option<InlineCallPlan> {
    if !wrapper.is_vararg
        || wrapper.parameter_count != 0
        || wrapper.upvalues.len() != 1
        || !wrapper.constants.is_empty()
        || !wrapper.prototypes.is_empty()
        || target.upvalue_count() != 0
    {
        return None;
    }

    let mut code = wrapper.code.iter().copied();
    let get_upvalue = code.next()?;
    if get_upvalue.opcode().ok()? != OpCode::GetUpvalue || get_upvalue.d() != 0 {
        return None;
    }
    let mut target_register = get_upvalue.a();

    loop {
        let instruction = code.next()?;
        match instruction.opcode().ok()? {
            OpCode::Move if instruction.b() == target_register && instruction.c() == 0 => {
                target_register = instruction.a();
            }
            OpCode::TailCallVararg
                if instruction.a() == target_register
                    && instruction.b() == 0
                    && instruction.c() == 0 =>
            {
                break;
            }
            _ => return None,
        }
    }

    let return_open = code.next()?;
    if return_open.opcode().ok()? != OpCode::ReturnOpen
        || return_open.a() != target_register
        || return_open.b() != 0
        || return_open.c() != 0
    {
        return None;
    }
    let halt = code.next()?;
    if halt.opcode().ok()? != OpCode::Halt || halt.raw() != Instruction::halt().raw() {
        return None;
    }
    code.next().is_none().then(|| InlineCallPlan {
        counted_instructions: None,
        ..target.clone()
    })
}

pub(crate) fn inline_initialized(
    initialized: &[bool],
    pc: usize,
    register: u8,
) -> Result<(), InlinePlanError> {
    if initialized.get(usize::from(register)) == Some(&true) {
        Ok(())
    } else {
        Err(InlinePlanError::UninitializedRegister { pc, register })
    }
}

pub(crate) fn inline_numeric_constant(constant: &Constant) -> Option<f64> {
    match constant {
        Constant::Integer(value) => Some(integer_to_number(*value)),
        Constant::Number(value) => Some(*value),
        Constant::Nil
        | Constant::Boolean(_)
        | Constant::String(_)
        | Constant::Int64(_)
        | Constant::UInt64(_) => None,
    }
}

pub(crate) fn integer_to_number(value: i64) -> f64 {
    value as f64
}

pub(crate) fn inline_numeric_binary(
    opcode: OpCode,
    destination: u8,
    left: u8,
    right: u8,
) -> InlineNumericOperation {
    match opcode {
        OpCode::Add => InlineNumericOperation::Add {
            destination,
            left,
            right,
        },
        OpCode::Subtract => InlineNumericOperation::Subtract {
            destination,
            left,
            right,
        },
        OpCode::Multiply => InlineNumericOperation::Multiply {
            destination,
            left,
            right,
        },
        OpCode::Divide => InlineNumericOperation::Divide {
            destination,
            left,
            right,
        },
        OpCode::Modulo => InlineNumericOperation::Modulo {
            destination,
            left,
            right,
        },
        OpCode::Power => InlineNumericOperation::Power {
            destination,
            left,
            right,
        },
        _ => unreachable!("inline numeric binary opcode"),
    }
}

/// The kind of a constant an inline plan cannot load as a number.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstantKind {
    Nil,
    Boolean,
    Integer,
    Number,
    String,
    Cdata,
}

impl From<&Constant> for ConstantKind {
    fn from(value: &Constant) -> Self {
        match value {
            Constant::Nil => Self::Nil,
            Constant::Boolean(_) => Self::Boolean,
            Constant::Integer(_) => Self::Integer,
            Constant::Number(_) => Self::Number,
            Constant::String(_) => Self::String,
            Constant::Int64(_) | Constant::UInt64(_) => Self::Cdata,
        }
    }
}

#[cfg(test)]
mod writes_before_reads_tests {
    use super::{InlineNumericOperation as Op, writes_before_reads};

    #[test]
    fn accepts_parameters_and_earlier_destinations() {
        let operations = [
            Op::Add {
                destination: 1,
                left: 0,
                right: 0,
            },
            Op::Multiply {
                destination: 2,
                left: 1,
                right: 0,
            },
        ];
        assert!(writes_before_reads(1, 3, 2, &operations));
    }

    #[test]
    fn rejects_a_read_before_any_write() {
        let operations = [Op::Add {
            destination: 1,
            left: 0,
            right: 2,
        }];
        assert!(!writes_before_reads(1, 3, 1, &operations));
    }

    #[test]
    fn rejects_an_unwritten_result_and_out_of_range_indices() {
        assert!(!writes_before_reads(1, 3, 2, &[]));
        let operations = [Op::Move {
            destination: 9,
            source: 0,
        }];
        assert!(!writes_before_reads(1, 3, 9, &operations));
        assert!(!writes_before_reads(4, 3, 0, &[]));
    }
}
