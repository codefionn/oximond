#![forbid(unsafe_code)]

//! The immutable, portable bytecode format shared by Oximond's compiler and VM.
//!
//! Instructions are `u32` words. The opcode occupies the low byte. `ABC` words
//! use the remaining bytes for registers or small counts, `AD` words use a
//! register and an unsigned 16-bit payload, and `AJ` words use a signed 24-bit
//! relative jump offset. Jumps are relative to the instruction after the jump.

use crate::string::LuaString;
use core::fmt;
use std::sync::Arc;

pub mod binary;
mod captures;
mod live;
pub mod load;
mod shapes;
mod sites;

pub use captures::{CaptureVolatility, RegisterSet};
pub use live::{ENTRY_VARARG_BIT, LiveRegisters};
pub use shapes::{ConstructorShape, ConstructorShapes};
pub use sites::{FieldSite, FieldSites, SiteKind};

/// A constant owned by a bytecode chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Nil,
    Boolean(bool),
    Integer(i64),
    Number(f64),
    String(LuaString),
    /// A `LuaJIT` `LL` literal: a signed 64-bit integer cdata.
    Int64(i64),
    /// A `LuaJIT` `ULL` literal: an unsigned 64-bit integer cdata.
    UInt64(u64),
}

/// Operations understood by the baseline register VM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OpCode {
    LoadNil = 0,
    LoadConstant = 1,
    Move = 2,
    Negate = 3,
    Not = 4,
    Add = 5,
    Subtract = 6,
    Multiply = 7,
    Divide = 8,
    Equal = 9,
    LessThan = 10,
    LessEqual = 11,
    Jump = 12,
    JumpIfFalse = 13,
    GetGlobal = 14,
    SetGlobal = 15,
    NewTable = 16,
    GetTable = 17,
    SetTable = 18,
    Call = 19,
    Return = 20,
    Halt = 21,
    Closure = 22,
    Modulo = 23,
    NumericForPrep = 24,
    NumericForLoop = 25,
    Length = 26,
    Concat = 27,
    GetUpvalue = 28,
    SetUpvalue = 29,
    Power = 30,
    Vararg = 31,
    CallVararg = 32,
    ReturnVararg = 33,
    TableVararg = 34,
    CallOpen = 35,
    ReturnOpen = 36,
    TableOpen = 37,
    /// Closes the active captured cell for one register, if that register is
    /// currently captured. This makes a loop iteration's locals independent
    /// from the register values used by the following iteration.
    CloseUpvalue = 38,
    /// Payload word for a wide constant or global reference. It is valid only
    /// immediately after one of the wide reference opcodes.
    ExtraArgument = 39,
    /// Loads a constant whose index is carried by the next `ExtraArgument`.
    LoadConstantWide = 40,
    /// Gets a global whose string-constant index is carried by the next
    /// `ExtraArgument`.
    GetGlobalWide = 41,
    /// Sets a global whose string-constant index is carried by the next
    /// `ExtraArgument`.
    SetGlobalWide = 42,
    /// Calls a function then forwards all of its results through the current
    /// frame's caller. The VM may replace the current Lua frame in place.
    TailCall = 43,
    /// Tail-call form which appends the current frame's varargs.
    TailCallVararg = 44,
    /// Tail-call form which appends pending open call results.
    TailCallOpen = 45,
    /// Calls one argument from independent registers and stores one result.
    ///
    /// `a` is the result destination, `b` is the callee, and `c` is the
    /// argument. Unlike [`Self::Call`], these registers need not be adjacent.
    CallOne = 46,
    /// Clears `b` consecutive registers beginning at `a`.
    ///
    /// `c` is reserved and encoded as zero. This is the compact form of
    /// adjacent [`Self::LoadNil`] instructions emitted when a statement drops
    /// several dead temporary registers at once.
    ClearRange = 47,
    /// Lua floor division (`//`).
    FloorDivide = 48,
    /// Lua integer bitwise AND (`&`).
    BitAnd = 49,
    /// Lua integer bitwise OR (`|`).
    BitOr = 50,
    /// Lua integer bitwise exclusive OR (`~`).
    BitXor = 51,
    /// Lua integer left shift (`<<`).
    ShiftLeft = 52,
    /// Lua integer right shift (`>>`).
    ShiftRight = 53,
    /// Lua integer bitwise NOT (`~`).
    BitNot = 54,
    /// Marks register `a` as a Lua 5.4 to-be-closed local.
    ///
    /// The VM owns metamethod lookup and exceptional unwinding. Compiled
    /// lexical exits pair this with [`Self::CloseLocal`].
    MarkClose = 55,
    /// Closes the to-be-closed local in register `a`, if it is still active.
    CloseLocal = 56,
    /// Initializes a global only when its current value is nil.
    DeclareGlobal = 57,
    /// Wide-name form of [`Self::DeclareGlobal`].
    DeclareGlobalWide = 58,
    /// Initializes `table[key]` only when its current value is nil.
    ///
    /// `a` is the table, `b` is the key, and `c` is the initializer value.
    DeclareTable = 59,
    /// Reads the field named by a string constant: `R[a] = R[b][K[c]]`.
    ///
    /// The same operation as [`Self::GetTable`] with the key in a register,
    /// without loading the name into one first.
    GetField = 60,
    /// Writes the field named by a string constant: `R[a][K[b]] = R[c]`.
    SetField = 61,
    /// `R[a] = R[b] + K[c]`, where `K[c]` is a numeric constant.
    ///
    /// Equivalent to loading the constant into a temporary and running
    /// [`Self::Add`] with it on the right.
    AddK = 64,
    /// `R[a] = R[b] - K[c]`. See [`Self::AddK`].
    SubtractK = 65,
    /// `R[a] = R[b] * K[c]`. See [`Self::AddK`].
    MultiplyK = 66,
    /// `R[a] = R[b] / K[c]`. See [`Self::AddK`].
    DivideK = 67,
    /// `R[a] = R[b] % K[c]`. See [`Self::AddK`].
    ModuloK = 68,
    /// Compares `R[a]` with `R[b]` for equality and branches.
    ///
    /// Always followed by a [`Self::Jump`] word. Bit 0 of `c` is `expected`:
    /// when the comparison result equals it, the jump is taken, and otherwise
    /// execution continues after the jump word. Behaves exactly as
    /// [`Self::Equal`] into a temporary followed by [`Self::JumpIfFalse`] (or
    /// a negation of it when `expected` is false).
    TestEqual = 69,
    /// `R[a] < R[b]` and branch. See [`Self::TestEqual`].
    TestLessThan = 70,
    /// `R[a] <= R[b]` and branch. See [`Self::TestEqual`].
    TestLessEqual = 71,
    /// Compares `R[a]` with the constant `K[b]` for equality and branches.
    ///
    /// `K[b]` is a nil, boolean, integer, number or string constant. See
    /// [`Self::TestEqual`].
    TestEqualK = 72,
    /// `R[a] < K[b]` and branch, where `K[b]` is a numeric constant.
    ///
    /// Bit 1 of `c` (`constant_left`) compares `K[b] < R[a]` instead. See
    /// [`Self::TestEqual`].
    TestLessThanK = 73,
    /// `R[a] <= K[b]` and branch. See [`Self::TestLessThanK`].
    TestLessEqualK = 74,
    /// Calls the iterator of a generic `for`. `a` is the base of the loop's
    /// control registers (`R[a]` iterator, `R[a+1]` state, `R[a+2]` control)
    /// and `b` the number of loop variables, 1 to 255. `c` is zero.
    ///
    /// Behaves exactly as `Move a+3, a`; `Move a+4, a+1`; `Move a+5, a+2`
    /// followed by `Call a+3` with two arguments and `b` results, so the
    /// results land in `R[a+3 ..= a+2+b]`, padded with nil, and the call site
    /// entry at this pc names a `for iterator`.
    ForInCall = 75,
    /// Ends or continues a generic `for` after [`Self::ForInCall`]. `a` is
    /// the base. When `R[a+3]` is not nil, sets `R[a+2] = R[a+3]` and jumps
    /// by the signed 16-bit offset in `d` (a loop backedge to the body).
    /// Otherwise falls through. Only nil ends the loop.
    ForInLoop = 76,
    /// Reads the field named by a string constant from the table in an
    /// upvalue: `R[a] = Up[b][K[c]]`. Stands for [`Self::GetUpvalue`] into
    /// `R[a]` followed by [`Self::GetField`] from `R[a]`, without the
    /// intermediate write. Lua 5.2+ globals read `_ENV` this way.
    GetTabUp = 77,
    /// Writes the field named by a string constant into the table in an
    /// upvalue: `Up[a][K[b]] = R[c]`. See [`Self::GetTabUp`].
    SetTabUp = 78,
    /// A method lookup, like Lua's SELF: `R[a+1] = R[b]; R[a] = R[b][K[c]]`.
    /// Stands for [`Self::Move`] into `R[a+1]` followed by [`Self::GetField`]
    /// into `R[a]`, in one dispatch. The receiver is copied first, as Lua's
    /// SELF does, so a metamethod that runs for the lookup sees the argument
    /// already in place. `b` may be `a` or `a + 1`.
    SelfField = 79,
}

impl OpCode {
    /// The register-operand opcode that a constant-operand arithmetic opcode
    /// behaves as, with the constant `K[c]` loaded into a register on the
    /// right.
    #[must_use]
    pub const fn register_form(self) -> Option<Self> {
        match self {
            Self::AddK => Some(Self::Add),
            Self::SubtractK => Some(Self::Subtract),
            Self::MultiplyK => Some(Self::Multiply),
            Self::DivideK => Some(Self::Divide),
            Self::ModuloK => Some(Self::Modulo),
            _ => None,
        }
    }
}

impl OpCode {
    /// The opcodes a count hook does not count. They are register
    /// maintenance that Oximond's fixed-frame compiler emits; Lua 5.1 does
    /// the equivalent stack work inside the surrounding opcodes, so they do
    /// not advance its count hook. The interpreter's prologue, the fast loop
    /// and the trace recorder all decide by this one list.
    const COUNT_FREE: [Self; 3] = [Self::Move, Self::LoadNil, Self::ClearRange];

    /// Whether an instruction with this opcode advances a count hook.
    #[must_use]
    #[inline]
    pub const fn counts_for_hooks(self) -> bool {
        Self::byte_counts_for_hooks(self as u8)
    }

    /// [`Self::counts_for_hooks`] for a raw opcode byte, for the fast loop's
    /// dispatch.
    #[must_use]
    #[inline]
    pub const fn byte_counts_for_hooks(byte: u8) -> bool {
        let mut index = 0;
        while index < Self::COUNT_FREE.len() {
            if Self::COUNT_FREE[index] as u8 == byte {
                return false;
            }
            index += 1;
        }
        true
    }

    /// Whether this opcode is a compare-and-branch test, which the
    /// following [`Self::Jump`] word completes.
    #[must_use]
    pub const fn is_test(self) -> bool {
        matches!(
            self,
            Self::TestEqual
                | Self::TestLessThan
                | Self::TestLessEqual
                | Self::TestEqualK
                | Self::TestLessThanK
                | Self::TestLessEqualK
        )
    }

    /// The register comparison opcode a test opcode behaves as.
    #[must_use]
    pub const fn comparison_form(self) -> Option<Self> {
        match self {
            Self::TestEqual | Self::TestEqualK => Some(Self::Equal),
            Self::TestLessThan | Self::TestLessThanK => Some(Self::LessThan),
            Self::TestLessEqual | Self::TestLessEqualK => Some(Self::LessEqual),
            _ => None,
        }
    }

    /// Whether this test opcode compares against a constant `K[b]`.
    #[must_use]
    pub const fn is_test_constant(self) -> bool {
        matches!(
            self,
            Self::TestEqualK | Self::TestLessThanK | Self::TestLessEqualK
        )
    }

    /// Decodes an opcode byte through a table. The opcode numbers have gaps,
    /// which a `match` turns into several range checks on the hot decode.
    const fn decode(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::LoadNil),
            1 => Some(Self::LoadConstant),
            2 => Some(Self::Move),
            3 => Some(Self::Negate),
            4 => Some(Self::Not),
            5 => Some(Self::Add),
            6 => Some(Self::Subtract),
            7 => Some(Self::Multiply),
            8 => Some(Self::Divide),
            9 => Some(Self::Equal),
            10 => Some(Self::LessThan),
            11 => Some(Self::LessEqual),
            12 => Some(Self::Jump),
            13 => Some(Self::JumpIfFalse),
            14 => Some(Self::GetGlobal),
            15 => Some(Self::SetGlobal),
            16 => Some(Self::NewTable),
            17 => Some(Self::GetTable),
            18 => Some(Self::SetTable),
            19 => Some(Self::Call),
            20 => Some(Self::Return),
            21 => Some(Self::Halt),
            22 => Some(Self::Closure),
            23 => Some(Self::Modulo),
            24 => Some(Self::NumericForPrep),
            25 => Some(Self::NumericForLoop),
            26 => Some(Self::Length),
            27 => Some(Self::Concat),
            28 => Some(Self::GetUpvalue),
            29 => Some(Self::SetUpvalue),
            30 => Some(Self::Power),
            31 => Some(Self::Vararg),
            32 => Some(Self::CallVararg),
            33 => Some(Self::ReturnVararg),
            34 => Some(Self::TableVararg),
            35 => Some(Self::CallOpen),
            36 => Some(Self::ReturnOpen),
            37 => Some(Self::TableOpen),
            38 => Some(Self::CloseUpvalue),
            39 => Some(Self::ExtraArgument),
            40 => Some(Self::LoadConstantWide),
            41 => Some(Self::GetGlobalWide),
            42 => Some(Self::SetGlobalWide),
            43 => Some(Self::TailCall),
            44 => Some(Self::TailCallVararg),
            45 => Some(Self::TailCallOpen),
            46 => Some(Self::CallOne),
            47 => Some(Self::ClearRange),
            48 => Some(Self::FloorDivide),
            49 => Some(Self::BitAnd),
            50 => Some(Self::BitOr),
            51 => Some(Self::BitXor),
            52 => Some(Self::ShiftLeft),
            53 => Some(Self::ShiftRight),
            54 => Some(Self::BitNot),
            55 => Some(Self::MarkClose),
            56 => Some(Self::CloseLocal),
            57 => Some(Self::DeclareGlobal),
            58 => Some(Self::DeclareGlobalWide),
            59 => Some(Self::DeclareTable),
            60 => Some(Self::GetField),
            61 => Some(Self::SetField),
            64 => Some(Self::AddK),
            65 => Some(Self::SubtractK),
            66 => Some(Self::MultiplyK),
            67 => Some(Self::DivideK),
            68 => Some(Self::ModuloK),
            69 => Some(Self::TestEqual),
            70 => Some(Self::TestLessThan),
            71 => Some(Self::TestLessEqual),
            72 => Some(Self::TestEqualK),
            73 => Some(Self::TestLessThanK),
            74 => Some(Self::TestLessEqualK),
            75 => Some(Self::ForInCall),
            76 => Some(Self::ForInLoop),
            77 => Some(Self::GetTabUp),
            78 => Some(Self::SetTabUp),
            79 => Some(Self::SelfField),
            _ => None,
        }
    }
}

const OPCODE_TABLE: [Option<OpCode>; 256] = {
    let mut table = [None; 256];
    let mut byte = 0;
    while byte < 256 {
        table[byte] = OpCode::decode(byte as u8);
        byte += 1;
    }
    table
};

impl TryFrom<u8> for OpCode {
    type Error = InvalidOpCode;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match OPCODE_TABLE[usize::from(value)] {
            Some(opcode) => Ok(opcode),
            None => Err(InvalidOpCode(value)),
        }
    }
}

/// An opcode byte which does not name an Oximond operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidOpCode(pub u8);

impl fmt::Display for InvalidOpCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown opcode {}", self.0)
    }
}

impl std::error::Error for InvalidOpCode {}

/// A compact, decoded-on-demand instruction word.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Instruction(u32);

impl Instruction {
    /// Builds an `ABC` instruction. `a`, `b`, and `c` are opcode-specific.
    #[must_use]
    pub const fn abc(opcode: OpCode, a: u8, b: u8, c: u8) -> Self {
        Self((opcode as u32) | ((a as u32) << 8) | ((b as u32) << 16) | ((c as u32) << 24))
    }

    /// Builds an `AD` instruction. `d` is an opcode-specific unsigned payload.
    #[must_use]
    pub const fn ad(opcode: OpCode, a: u8, d: u16) -> Self {
        Self((opcode as u32) | ((a as u32) << 8) | ((d as u32) << 16))
    }

    /// Builds an `AJ` instruction with a signed 24-bit offset.
    ///
    /// The offset is relative to the next instruction. Values outside the
    /// encodable range are rejected instead of silently truncating.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::JumpOffsetOutOfRange`] when `offset` does not
    /// fit in the signed 24-bit jump field.
    pub fn aj(opcode: OpCode, offset: i32) -> Result<Self, EncodeError> {
        if !(-8_388_608..=8_388_607).contains(&offset) {
            return Err(EncodeError::JumpOffsetOutOfRange(offset));
        }
        Ok(Self(
            (opcode as u32) | ((offset.cast_unsigned() & 0x00ff_ffff) << 8),
        ))
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Decode the instruction's opcode.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidOpCode`] when the word's low byte is unassigned.
    pub fn opcode(self) -> Result<OpCode, InvalidOpCode> {
        OpCode::try_from(self.0.to_le_bytes()[0])
    }

    #[must_use]
    pub const fn a(self) -> u8 {
        self.0.to_le_bytes()[1]
    }

    #[must_use]
    pub const fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }

    #[must_use]
    pub const fn c(self) -> u8 {
        self.0.to_le_bytes()[3]
    }

    #[must_use]
    pub const fn d(self) -> u16 {
        u16::from_le_bytes([self.0.to_le_bytes()[2], self.0.to_le_bytes()[3]])
    }

    /// Builds a compare-and-branch test. The next word must be a jump.
    ///
    /// `expected` is bit 0 of `c`, and `constant_left` bit 1.
    #[must_use]
    pub const fn test(opcode: OpCode, a: u8, b: u8, expected: bool, constant_left: bool) -> Self {
        Self::abc(
            opcode,
            a,
            b,
            (expected as u8) | ((constant_left as u8) << 1),
        )
    }

    /// Whether a test instruction jumps when its comparison is true.
    #[must_use]
    pub const fn test_expected(self) -> bool {
        self.c() & 1 != 0
    }

    /// Whether a constant test compares `K[b] op R[a]`.
    #[must_use]
    pub const fn test_constant_left(self) -> bool {
        self.c() & 2 != 0
    }

    #[must_use]
    pub const fn jump_offset(self) -> i32 {
        ((self.0 >> 8) << 8).cast_signed() >> 8
    }

    #[must_use]
    pub const fn load_nil(destination: u8) -> Self {
        Self::abc(OpCode::LoadNil, destination, 0, 0)
    }
    /// Clears `count` consecutive registers beginning at `start`.
    #[must_use]
    pub const fn clear_range(start: u8, count: u8) -> Self {
        Self::abc(OpCode::ClearRange, start, count, 0)
    }
    #[must_use]
    pub const fn load_constant(destination: u8, constant: u16) -> Self {
        Self::ad(OpCode::LoadConstant, destination, constant)
    }
    /// Begins a two-word constant load. The following instruction must be
    /// [`Self::extra_argument`] carrying the absolute 24-bit pool index.
    #[must_use]
    pub const fn load_constant_wide(destination: u8) -> Self {
        Self::abc(OpCode::LoadConstantWide, destination, 0, 0)
    }
    #[must_use]
    pub const fn move_(destination: u8, source: u8) -> Self {
        Self::abc(OpCode::Move, destination, source, 0)
    }
    #[must_use]
    pub const fn unary(opcode: OpCode, destination: u8, source: u8) -> Self {
        Self::abc(opcode, destination, source, 0)
    }
    /// Gets the length of a table value.
    #[must_use]
    pub const fn length(destination: u8, source: u8) -> Self {
        Self::unary(OpCode::Length, destination, source)
    }
    #[must_use]
    pub const fn binary(opcode: OpCode, destination: u8, left: u8, right: u8) -> Self {
        Self::abc(opcode, destination, left, right)
    }
    /// Concatenates the inclusive register range `first..=last` into
    /// `destination`.
    #[must_use]
    pub const fn concat(destination: u8, first: u8, last: u8) -> Self {
        Self::binary(OpCode::Concat, destination, first, last)
    }
    /// Raises a numeric left operand to a numeric right operand.
    #[must_use]
    pub const fn power(destination: u8, left: u8, right: u8) -> Self {
        Self::binary(OpCode::Power, destination, left, right)
    }
    /// Loads an upvalue into a register.
    #[must_use]
    pub const fn get_upvalue(destination: u8, upvalue: u16) -> Self {
        Self::ad(OpCode::GetUpvalue, destination, upvalue)
    }
    /// Stores a register into an upvalue.
    #[must_use]
    pub const fn set_upvalue(source: u8, upvalue: u16) -> Self {
        Self::ad(OpCode::SetUpvalue, source, upvalue)
    }
    /// Copies exactly `count` incoming varargs to registers beginning at `destination`.
    #[must_use]
    pub const fn vararg(destination: u8, count: u8) -> Self {
        Self::abc(OpCode::Vararg, destination, count, 0)
    }
    #[must_use]
    pub const fn get_global(destination: u8, name: u16) -> Self {
        Self::ad(OpCode::GetGlobal, destination, name)
    }
    /// Begins a two-word global load. The following instruction must be
    /// [`Self::extra_argument`] carrying the absolute 24-bit name index.
    #[must_use]
    pub const fn get_global_wide(destination: u8) -> Self {
        Self::abc(OpCode::GetGlobalWide, destination, 0, 0)
    }
    #[must_use]
    pub const fn set_global(source: u8, name: u16) -> Self {
        Self::ad(OpCode::SetGlobal, source, name)
    }
    /// Begins a two-word global store. The following instruction must be
    /// [`Self::extra_argument`] carrying the absolute 24-bit name index.
    #[must_use]
    pub const fn set_global_wide(source: u8) -> Self {
        Self::abc(OpCode::SetGlobalWide, source, 0, 0)
    }
    /// Initializes a global whose current value must be nil.
    #[must_use]
    pub const fn declare_global(source: u8, name: u16) -> Self {
        Self::ad(OpCode::DeclareGlobal, source, name)
    }
    /// Begins a two-word checked global initialization.
    #[must_use]
    pub const fn declare_global_wide(source: u8) -> Self {
        Self::abc(OpCode::DeclareGlobalWide, source, 0, 0)
    }
    /// Initializes an entry in a lexical environment table whose current value must be nil.
    #[must_use]
    pub const fn declare_table(table: u8, key: u8, value: u8) -> Self {
        Self::abc(OpCode::DeclareTable, table, key, value)
    }
    /// Encodes the payload for a preceding wide reference.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::WideIndexOutOfRange`] when `index` does not fit
    /// in the 24-bit payload field.
    pub fn extra_argument(index: u32) -> Result<Self, EncodeError> {
        if index > Self::MAX_WIDE_INDEX {
            return Err(EncodeError::WideIndexOutOfRange(index));
        }
        Ok(Self((OpCode::ExtraArgument as u32) | (index << 8)))
    }
    /// Largest constant/global index accepted by a wide reference.
    pub const MAX_WIDE_INDEX: u32 = 0x00ff_ffff;
    /// Decodes a validated `ExtraArgument` payload.
    #[must_use]
    pub const fn extra_argument_index(self) -> u32 {
        self.0 >> 8
    }
    #[must_use]
    pub const fn new_table(destination: u8, array_hint: u8, hash_hint: u8) -> Self {
        Self::abc(OpCode::NewTable, destination, array_hint, hash_hint)
    }
    #[must_use]
    pub const fn get_table(destination: u8, table: u8, key: u8) -> Self {
        Self::abc(OpCode::GetTable, destination, table, key)
    }
    #[must_use]
    pub const fn set_table(table: u8, key: u8, value: u8) -> Self {
        Self::abc(OpCode::SetTable, table, key, value)
    }
    /// Reads `table[name]`, where constant `name` is a string.
    #[must_use]
    pub const fn get_field(destination: u8, table: u8, name: u8) -> Self {
        Self::abc(OpCode::GetField, destination, table, name)
    }
    /// Writes `table[name] = value`, where constant `name` is a string.
    #[must_use]
    pub const fn set_field(table: u8, name: u8, value: u8) -> Self {
        Self::abc(OpCode::SetField, table, name, value)
    }
    /// Reads `Up[upvalue][name]` into `destination`, where constant `name` is
    /// a string.
    #[must_use]
    pub const fn get_tab_up(destination: u8, upvalue: u8, name: u8) -> Self {
        Self::abc(OpCode::GetTabUp, destination, upvalue, name)
    }
    /// Writes `Up[upvalue][name] = value`, where constant `name` is a string.
    #[must_use]
    pub const fn set_tab_up(upvalue: u8, name: u8, value: u8) -> Self {
        Self::abc(OpCode::SetTabUp, upvalue, name, value)
    }
    /// Copies `table` to `method + 1` and reads `table[name]` into `method`,
    /// where constant `name` is a string.
    #[must_use]
    pub const fn self_field(method: u8, table: u8, name: u8) -> Self {
        Self::abc(OpCode::SelfField, method, table, name)
    }
    /// Calls the function in `base`; arguments occupy `base + 1..=base + arguments`.
    /// Results start at `base`, and `results == 0` requests an open result range.
    #[must_use]
    pub const fn call(base: u8, arguments: u8, results: u8) -> Self {
        Self::abc(OpCode::Call, base, arguments, results)
    }
    /// Calls `callee` with `argument` and writes its first result to
    /// `destination`.
    #[must_use]
    pub const fn call_one(destination: u8, callee: u8, argument: u8) -> Self {
        Self::abc(OpCode::CallOne, destination, callee, argument)
    }
    /// Calls the function in `base` with fixed arguments followed by every vararg.
    #[must_use]
    pub const fn call_vararg(base: u8, fixed_arguments: u8, results: u8) -> Self {
        Self::abc(OpCode::CallVararg, base, fixed_arguments, results)
    }
    /// Calls the function in `base` with fixed arguments followed by the
    /// frame's pending open call results.
    #[must_use]
    pub const fn call_open(base: u8, fixed_arguments: u8, results: u8) -> Self {
        Self::abc(OpCode::CallOpen, base, fixed_arguments, results)
    }
    /// Starts a proper tail call. The next instruction must be
    /// [`Self::return_open`] with no fixed results, which forwards results for
    /// native and VM-coordinated callees that cannot replace the frame.
    #[must_use]
    pub const fn tail_call(base: u8, arguments: u8) -> Self {
        Self::abc(OpCode::TailCall, base, arguments, 0)
    }
    /// Tail-calls with fixed arguments followed by every incoming vararg.
    #[must_use]
    pub const fn tail_call_vararg(base: u8, fixed_arguments: u8) -> Self {
        Self::abc(OpCode::TailCallVararg, base, fixed_arguments, 0)
    }
    /// Tail-calls with fixed arguments followed by pending open results.
    #[must_use]
    pub const fn tail_call_open(base: u8, fixed_arguments: u8) -> Self {
        Self::abc(OpCode::TailCallOpen, base, fixed_arguments, 0)
    }
    #[must_use]
    pub const fn return_(source: u8) -> Self {
        Self::return_values(source, 1)
    }
    /// Returns `values` registers starting at `base`. A value count of zero
    /// represents an open result range.
    #[must_use]
    pub const fn return_values(base: u8, values: u8) -> Self {
        Self::abc(OpCode::Return, base, values, 0)
    }
    /// Returns fixed register values followed by every incoming vararg.
    #[must_use]
    pub const fn return_vararg(base: u8, fixed_values: u8) -> Self {
        Self::abc(OpCode::ReturnVararg, base, fixed_values, 0)
    }
    /// Returns fixed register values followed by the pending open call results.
    #[must_use]
    pub const fn return_open(base: u8, fixed_values: u8) -> Self {
        Self::abc(OpCode::ReturnOpen, base, fixed_values, 0)
    }
    /// Appends every incoming vararg to table `table` beginning at one-based `start`.
    #[must_use]
    pub const fn table_vararg(table: u8, start: u16) -> Self {
        Self::ad(OpCode::TableVararg, table, start)
    }
    /// Appends pending open call results to `table` beginning at one-based `start`.
    #[must_use]
    pub const fn table_open(table: u8, start: u16) -> Self {
        Self::ad(OpCode::TableOpen, table, start)
    }
    #[must_use]
    pub const fn halt() -> Self {
        Self::abc(OpCode::Halt, 0, 0, 0)
    }

    /// Builds a conditional jump with a signed 16-bit relative offset.
    #[must_use]
    pub const fn jump_if_false(condition: u8, offset: i16) -> Self {
        Self::ad(OpCode::JumpIfFalse, condition, offset.cast_unsigned())
    }
    /// Creates a closure from a nested function prototype.
    #[must_use]
    pub const fn closure(destination: u8, prototype: u16) -> Self {
        Self::ad(OpCode::Closure, destination, prototype)
    }
    /// Initializes the four-register numeric-for layout and jumps to its loop instruction.
    #[must_use]
    pub const fn numeric_for_prep(base: u8, offset: i16) -> Self {
        Self::ad(OpCode::NumericForPrep, base, offset.cast_unsigned())
    }
    /// Advances and checks the four-register numeric-for layout, then jumps to its body.
    #[must_use]
    pub const fn numeric_for_loop(base: u8, offset: i16) -> Self {
        Self::ad(OpCode::NumericForLoop, base, offset.cast_unsigned())
    }
    /// Calls a generic-for iterator held in the control registers at `base`
    /// with `variables` results. See [`OpCode::ForInCall`].
    #[must_use]
    pub const fn for_in_call(base: u8, variables: u8) -> Self {
        Self::abc(OpCode::ForInCall, base, variables, 0)
    }
    /// Continues a generic `for` at `base` when `R[base+3]` is not nil. See
    /// [`OpCode::ForInLoop`].
    #[must_use]
    pub const fn for_in_loop(base: u8, offset: i16) -> Self {
        Self::ad(OpCode::ForInLoop, base, offset.cast_unsigned())
    }
    /// Closes the live captured cell associated with `register`.
    #[must_use]
    pub const fn close_upvalue(register: u8) -> Self {
        Self::abc(OpCode::CloseUpvalue, register, 0, 0)
    }
    /// Marks a Lua 5.4 to-be-closed local after its initializer has run.
    #[must_use]
    pub const fn mark_close(register: u8) -> Self {
        Self::abc(OpCode::MarkClose, register, 0, 0)
    }
    /// Closes one active Lua 5.4 to-be-closed local.
    #[must_use]
    pub const fn close_local(register: u8) -> Self {
        Self::abc(OpCode::CloseLocal, register, 0, 0)
    }
}

/// A failure to encode an instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodeError {
    JumpOffsetOutOfRange(i32),
    WideIndexOutOfRange(u32),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JumpOffsetOutOfRange(offset) => {
                write!(formatter, "jump offset {offset} does not fit in 24 bits")
            }
            Self::WideIndexOutOfRange(index) => {
                write!(formatter, "wide index {index} does not fit in 24 bits")
            }
        }
    }
}
impl std::error::Error for EncodeError {}

/// A complete function body. The compiler emits chunks, which are immutable to
/// consumers after validation; execution-specific caches belong to the VM.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub register_count: u8,
    pub parameter_count: u8,
    pub constants: Vec<Constant>,
    pub code: Vec<Instruction>,
    /// Nested function prototypes. Shared so closures created from one prototype
    /// keep a single chunk identity, which trace and region caches key on.
    pub prototypes: Vec<Arc<Chunk>>,
    pub upvalues: Vec<UpvalueDescriptor>,
    pub is_vararg: bool,
    pub arg_register: Option<u8>,
    /// Source and call-site data used by diagnostics and stack traces.
    pub debug: DebugMetadata,
    /// Lazily computed; see [`Chunk::volatile_captures`].
    pub captures: CaptureVolatility,
    /// Lazily computed; see [`Chunk::constructor_shape`].
    pub constructors: ConstructorShapes,
    /// Lazily computed; see [`Chunk::live_register_counts`].
    pub live: LiveRegisters,
    /// Inline caches of string-keyed table accesses; see [`FieldSite`].
    pub field_sites: FieldSites,
}

/// Immutable source-level data attached to one bytecode chunk.
///
/// The source name is shared by nested prototypes cheaply because [`LuaString`]
/// is reference counted. Each call instruction has one entry. Dynamic calls use
/// an unnamed entry so a consumer can still distinguish them from missing data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DebugMetadata {
    source_name: LuaString,
    /// The source line where this function was declared. Zero means that the
    /// producer stripped function-definition data.
    definition_line: u32,
    /// The source line containing this function's closing `end`. Zero means
    /// that the producer stripped function-definition data.
    last_definition_line: u32,
    call_sites: Vec<CallSite>,
    /// Whether `call_sites` is ascending by `pc`. The compiler emits sites in
    /// instruction order, so this is true for every produced chunk and the
    /// runtime lookup can binary-search instead of scanning.
    call_sites_sorted: bool,
    diagnostic_sites: Vec<DiagnosticSite>,
    /// Lexical local bindings, expressed as half-open instruction ranges.
    ///
    /// A binding is live at program counter `pc` when
    /// `start_pc <= pc && pc < end_pc`. Keeping a range rather than a snapshot
    /// per instruction makes debug information small while still preserving
    /// register moves performed by the compiler's temporary reclamation pass.
    local_variables: Vec<LocalVariable>,
    /// One 1-based source line per instruction. An empty vector means the
    /// chunk was stripped; zero entries are unknown source positions.
    line_info: Vec<u32>,
}

impl DebugMetadata {
    /// Builds debug metadata for one chunk.
    #[must_use]
    pub fn new(source_name: impl Into<LuaString>, call_sites: Vec<CallSite>) -> Self {
        let call_sites_sorted = call_sites
            .windows(2)
            .all(|window| window[0].pc() <= window[1].pc());
        Self {
            source_name: source_name.into(),
            definition_line: 0,
            last_definition_line: 0,
            call_sites,
            call_sites_sorted,
            diagnostic_sites: Vec::new(),
            local_variables: Vec::new(),
            line_info: Vec::new(),
        }
    }

    /// Finds the call site recorded for one instruction, if any.
    ///
    /// Every `Call`-family instruction carries one site, so hot lookup paths
    /// (each Lua-to-Lua call reads its caller's site) search instead of scan.
    #[must_use]
    #[inline]
    pub fn call_site_at(&self, pc: usize) -> Option<&CallSite> {
        if self.call_sites_sorted {
            let index = self
                .call_sites
                .binary_search_by_key(&pc, CallSite::pc)
                .ok()?;
            Some(&self.call_sites[index])
        } else {
            self.call_sites.iter().find(|site| site.pc() == pc)
        }
    }

    /// Returns the name of the source that produced this chunk.
    #[must_use]
    pub fn source_name(&self) -> &LuaString {
        &self.source_name
    }

    /// Attaches Lua-style function definition bounds.
    #[must_use]
    pub const fn with_function_lines(
        mut self,
        definition_line: u32,
        last_definition_line: u32,
    ) -> Self {
        self.definition_line = definition_line;
        self.last_definition_line = last_definition_line;
        self
    }

    /// Returns the line containing the function declaration, if known.
    #[must_use]
    pub const fn definition_line(&self) -> u32 {
        self.definition_line
    }

    /// Returns the line containing the function's closing `end`, if known.
    #[must_use]
    pub const fn last_definition_line(&self) -> u32 {
        self.last_definition_line
    }

    /// Returns one entry for each emitted call instruction.
    #[must_use]
    pub fn call_sites(&self) -> &[CallSite] {
        &self.call_sites
    }

    /// Attaches source names for operands whose runtime type errors can name
    /// the original Lua expression.
    #[must_use]
    pub fn with_diagnostic_sites(mut self, diagnostic_sites: Vec<DiagnosticSite>) -> Self {
        self.diagnostic_sites = diagnostic_sites;
        self
    }

    /// Returns operand names keyed by instruction and source register.
    #[must_use]
    pub fn diagnostic_sites(&self) -> &[DiagnosticSite] {
        &self.diagnostic_sites
    }

    /// Attaches lexical local bindings for source-level inspection.
    #[must_use]
    pub fn with_local_variables(mut self, local_variables: Vec<LocalVariable>) -> Self {
        self.local_variables = local_variables;
        self
    }

    /// Returns each lexical local binding in source declaration order.
    #[must_use]
    pub fn local_variables(&self) -> &[LocalVariable] {
        &self.local_variables
    }

    /// Returns local bindings live at `pc`, in source declaration order.
    pub fn locals_at(&self, pc: usize) -> impl Iterator<Item = &LocalVariable> {
        self.local_variables
            .iter()
            .filter(move |local| local.live_at(pc))
    }

    /// Attaches instruction source lines. Supplying no entries represents
    /// stripped line data.
    #[must_use]
    pub fn with_line_info(mut self, line_info: Vec<u32>) -> Self {
        self.line_info = line_info;
        self
    }

    /// Returns the instruction-to-source-line mapping.
    #[must_use]
    pub fn line_info(&self) -> &[u32] {
        &self.line_info
    }

    /// Returns an instruction's 1-based source line, when present.
    #[must_use]
    pub fn line_at(&self, pc: usize) -> Option<u32> {
        self.line_info.get(pc).copied().filter(|line| *line != 0)
    }

    /// Returns each executable source line once, in source order.
    ///
    /// Lua's debug library exposes this set as `activelines` for a Lua
    /// function. The mapping remains the single serialized representation so
    /// dumped chunks cannot disagree about instruction locations and active
    /// lines.
    #[must_use]
    pub fn active_lines(&self) -> Vec<u32> {
        let mut lines: Vec<u32> = self
            .line_info
            .iter()
            .copied()
            .filter(|line| *line != 0)
            .collect();
        lines.sort_unstable();
        lines.dedup();
        lines
    }
}

impl Default for DebugMetadata {
    fn default() -> Self {
        // Every chunk starts with this metadata and the compiler replaces it,
        // so the name is shared rather than built per chunk.
        static UNKNOWN: std::sync::LazyLock<LuaString> =
            std::sync::LazyLock::new(|| LuaString::from("=?"));
        Self::new(UNKNOWN.clone(), Vec::new())
    }
}

/// The statically known origin of a call expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallKind {
    Global,
    Local,
    Upvalue,
    Field,
    Method,
    /// A string constant, which Lua 5.4 names in operand errors.
    Constant,
    /// The call of a generic `for` loop's iterator function.
    ForIterator,
}

/// Debug data for one call instruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallSite {
    pc: usize,
    name: Option<LuaString>,
    kind: Option<CallKind>,
    iterator: bool,
}

/// A statically named operand used by one bytecode instruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticSite {
    pc: usize,
    register: u8,
    name: LuaString,
    kind: CallKind,
}

/// A lexical local variable stored in a frame register for an instruction range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalVariable {
    name: LuaString,
    register: u8,
    start_pc: usize,
    end_pc: usize,
}

impl LocalVariable {
    /// Builds a local binding. The program-counter range is half-open.
    #[must_use]
    pub fn new(name: impl Into<LuaString>, register: u8, start_pc: usize, end_pc: usize) -> Self {
        Self {
            name: name.into(),
            register,
            start_pc,
            end_pc,
        }
    }

    /// Returns the source-level local name.
    #[must_use]
    pub fn name(&self) -> &LuaString {
        &self.name
    }

    /// Returns the frame register which contains the local in this range.
    #[must_use]
    pub const fn register(&self) -> u8 {
        self.register
    }

    /// Returns the first instruction at which the binding is live.
    #[must_use]
    pub const fn start_pc(&self) -> usize {
        self.start_pc
    }

    /// Returns the first instruction after the binding ceases to be live.
    #[must_use]
    pub const fn end_pc(&self) -> usize {
        self.end_pc
    }

    /// Returns whether the binding is live at `pc`.
    #[must_use]
    pub const fn live_at(&self, pc: usize) -> bool {
        self.start_pc <= pc && pc < self.end_pc
    }
}

impl DiagnosticSite {
    /// Builds an operand diagnostic entry.
    #[must_use]
    pub fn new(pc: usize, register: u8, name: impl Into<LuaString>, kind: CallKind) -> Self {
        Self {
            pc,
            register,
            name: name.into(),
            kind,
        }
    }

    #[must_use]
    pub const fn pc(&self) -> usize {
        self.pc
    }

    #[must_use]
    pub const fn register(&self) -> u8 {
        self.register
    }

    #[must_use]
    pub fn name(&self) -> &LuaString {
        &self.name
    }

    #[must_use]
    pub const fn kind(&self) -> CallKind {
        self.kind
    }
}

impl CallSite {
    /// Builds a call-site entry for a statically named call.
    #[must_use]
    pub fn named(pc: usize, name: impl Into<LuaString>, kind: CallKind) -> Self {
        Self {
            pc,
            name: Some(name.into()),
            kind: Some(kind),
            iterator: false,
        }
    }

    /// Builds a call-site entry whose callee has no static name.
    #[must_use]
    pub const fn anonymous(pc: usize) -> Self {
        Self {
            pc,
            name: None,
            kind: None,
            iterator: false,
        }
    }

    /// Builds the entry for a generic `for` loop's iterator call. Its name,
    /// `for iterator`, follows from the kind, so the entry stores none and
    /// compiling a loop allocates no string for it.
    #[must_use]
    pub const fn for_iterator(pc: usize) -> Self {
        Self {
            pc,
            name: None,
            kind: Some(CallKind::ForIterator),
            iterator: true,
        }
    }

    /// Marks this call as the iterator call of a generic `for` loop.
    #[must_use]
    pub const fn with_iterator(mut self, iterator: bool) -> Self {
        self.iterator = iterator;
        self
    }

    /// True for the iterator call of a generic `for` loop.
    #[must_use]
    pub const fn is_iterator(&self) -> bool {
        self.iterator
    }

    /// Returns the program counter of the call instruction.
    #[must_use]
    pub const fn pc(&self) -> usize {
        self.pc
    }

    /// Returns the static callee name, when the expression has one. A
    /// [`CallKind::ForIterator`] site has none: its kind names it.
    #[must_use]
    pub fn name(&self) -> Option<&LuaString> {
        self.name.as_ref()
    }

    /// Returns how the static callee name was resolved, when known.
    #[must_use]
    pub const fn kind(&self) -> Option<CallKind> {
        self.kind
    }
}

/// The source for one future closure capture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpvalueDescriptor {
    /// The lexical identifier captured by this slot, preserved as Lua bytes.
    pub name: LuaString,
    pub source: UpvalueSource,
}

/// An explicit capture source for a nested function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpvalueSource {
    ParentRegister(u8),
    ParentUpvalue(u16),
    /// A root-level capture restored from a serialized chunk.
    Detached,
}

impl Chunk {
    #[must_use]
    pub fn new(register_count: u8, constants: Vec<Constant>, code: Vec<Instruction>) -> Self {
        let code_len = code.len();
        Self {
            register_count,
            parameter_count: 0,
            constants,
            code,
            prototypes: Vec::new(),
            upvalues: Vec::new(),
            is_vararg: false,
            arg_register: None,
            debug: DebugMetadata::default(),
            captures: CaptureVolatility::default(),
            constructors: ConstructorShapes::default(),
            live: LiveRegisters::default(),
            field_sites: FieldSites::for_code_len(code_len),
        }
    }

    /// Builds a chunk with nested functions and an explicit capture layout.
    #[must_use]
    pub fn with_parts(
        register_count: u8,
        parameter_count: u8,
        constants: Vec<Constant>,
        code: Vec<Instruction>,
        prototypes: Vec<Chunk>,
        upvalues: Vec<UpvalueDescriptor>,
    ) -> Self {
        let code_len = code.len();
        Self {
            register_count,
            parameter_count,
            constants,
            code,
            prototypes: prototypes.into_iter().map(Arc::new).collect(),
            upvalues,
            is_vararg: false,
            arg_register: None,
            debug: DebugMetadata::default(),
            captures: CaptureVolatility::default(),
            constructors: ConstructorShapes::default(),
            live: LiveRegisters::default(),
            field_sites: FieldSites::for_code_len(code_len),
        }
    }

    /// Builds a vararg chunk with an optional Lua 5.1 compatibility `arg` table register.
    #[must_use]
    pub fn with_varargs(
        register_count: u8,
        parameter_count: u8,
        constants: Vec<Constant>,
        code: Vec<Instruction>,
        prototypes: Vec<Chunk>,
        upvalues: Vec<UpvalueDescriptor>,
        arg_register: Option<u8>,
    ) -> Self {
        let code_len = code.len();
        Self {
            register_count,
            parameter_count,
            constants,
            code,
            prototypes: prototypes.into_iter().map(Arc::new).collect(),
            upvalues,
            is_vararg: true,
            arg_register,
            debug: DebugMetadata::default(),
            captures: CaptureVolatility::default(),
            constructors: ConstructorShapes::default(),
            live: LiveRegisters::default(),
            field_sites: FieldSites::for_code_len(code_len),
        }
    }

    /// Attaches immutable source-level metadata to this chunk.
    #[must_use]
    ///
    /// The live-register analysis reads the debug locals, so a result
    /// computed before is dropped.
    pub fn with_debug_metadata(mut self, debug: DebugMetadata) -> Self {
        self.debug = debug;
        self.live.reset();
        self
    }

    /// Checks all references the VM would otherwise trust.
    ///
    /// # Errors
    ///
    /// Returns a validation error for malformed instructions, references, or
    /// control-flow targets.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_with_parent(None)
    }

    /// Checks a chunk that is instantiated by itself, as a top-level chunk or
    /// as a prototype handed to a runtime apart from its parent. In the second
    /// case captures from the parent are accepted for any parent register or
    /// upvalue, since the parent is not known here.
    ///
    /// # Errors
    ///
    /// Returns a validation error for malformed instructions, references, or
    /// control-flow targets.
    pub fn validate_detached_prototype(&self) -> Result<(), ValidationError> {
        self.validate_with_parent(None)
            .or_else(|_| self.validate_with_parent(Some((u8::MAX, usize::MAX))))
    }

    #[allow(clippy::too_many_lines)]
    fn validate_with_parent(&self, parent: Option<(u8, usize)>) -> Result<(), ValidationError> {
        // The VM packs program counters into 32 bits.
        if u32::try_from(self.code.len()).is_err() {
            return Err(ValidationError::CodeTooLong {
                length: self.code.len(),
            });
        }
        if self.parameter_count > self.register_count {
            return Err(ValidationError::ParameterOutOfBounds {
                parameter_count: self.parameter_count,
                register_count: self.register_count,
            });
        }
        if let Some(register) = self.arg_register {
            if !self.is_vararg {
                return Err(ValidationError::ArgRegisterInFixedChunk { register });
            }
            if register >= self.register_count {
                return Err(ValidationError::ArgRegisterOutOfBounds {
                    register,
                    register_count: self.register_count,
                });
            }
            if register < self.parameter_count {
                return Err(ValidationError::ArgRegisterOverlapsParameter {
                    register,
                    parameter_count: self.parameter_count,
                });
            }
        }
        for (index, upvalue) in self.upvalues.iter().enumerate() {
            match (parent, &upvalue.source) {
                (None, UpvalueSource::Detached) => {}
                (None, _) => return Err(ValidationError::RootUpvalue { index }),
                (Some(_), UpvalueSource::Detached) => {
                    return Err(ValidationError::DetachedUpvalueInNestedChunk { index });
                }
                (Some((parent_registers, _)), UpvalueSource::ParentRegister(register))
                    if *register < parent_registers => {}
                (Some((parent_registers, _)), UpvalueSource::ParentRegister(register)) => {
                    return Err(ValidationError::UpvalueRegisterOutOfBounds {
                        index,
                        register: *register,
                        parent_register_count: parent_registers,
                    });
                }
                (Some((_, parent_upvalues)), UpvalueSource::ParentUpvalue(source))
                    if usize::from(*source) < parent_upvalues => {}
                (Some((_, parent_upvalues)), UpvalueSource::ParentUpvalue(source)) => {
                    return Err(ValidationError::UpvalueOutOfBounds {
                        index,
                        source: *source,
                        parent_upvalue_count: parent_upvalues,
                    });
                }
            }
        }
        if self.code.is_empty() {
            return Err(ValidationError::EmptyCode);
        }
        for (pc, instruction) in self.code.iter().copied().enumerate() {
            let opcode = instruction
                .opcode()
                .map_err(|error| ValidationError::InvalidOpcode { pc, error })?;
            match opcode {
                OpCode::ExtraArgument => {
                    let previous = pc.checked_sub(1).and_then(|pc| self.code.get(pc)).copied();
                    let is_payload = previous
                        .and_then(|instruction| instruction.opcode().ok())
                        .is_some_and(is_wide_reference);
                    if !is_payload {
                        return Err(ValidationError::OrphanExtraArgument { pc });
                    }
                }
                opcode if is_wide_reference(opcode) => {
                    if !self
                        .code
                        .get(pc + 1)
                        .and_then(|instruction| instruction.opcode().ok())
                        .is_some_and(|opcode| opcode == OpCode::ExtraArgument)
                    {
                        return Err(ValidationError::MissingExtraArgument { pc, opcode });
                    }
                }
                OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen
                    if !self.code.get(pc + 1).is_some_and(|next| {
                        next.opcode() == Ok(OpCode::ReturnOpen) && next.b() == 0
                    }) =>
                {
                    return Err(ValidationError::TailCallMissingForwarder { pc });
                }
                _ => {}
            }
            self.validate_instruction(pc, instruction, opcode)?;
        }
        self.validate_debug_metadata()?;
        for prototype in &self.prototypes {
            prototype.validate_with_parent(Some((self.register_count, self.upvalues.len())))?;
        }
        match self
            .code
            .last()
            .and_then(|instruction| instruction.opcode().ok())
        {
            Some(OpCode::Return | OpCode::ReturnVararg | OpCode::ReturnOpen | OpCode::Halt) => {
                Ok(())
            }
            _ => Err(ValidationError::MissingTerminator),
        }
    }

    fn validate_debug_metadata(&self) -> Result<(), ValidationError> {
        let line_info = self.debug.line_info();
        if !line_info.is_empty() && line_info.len() != self.code.len() {
            return Err(ValidationError::DebugLineInfoLength {
                line_count: line_info.len(),
                code_len: self.code.len(),
            });
        }
        // Producers emit sites in pc order, so a duplicate is next to its
        // twin; only an unordered list needs a set.
        let call_sites = self.debug.call_sites();
        let mut seen_pcs = (!call_sites
            .windows(2)
            .all(|pair| pair[0].pc() <= pair[1].pc()))
        .then(std::collections::BTreeSet::new);
        let mut previous_pc = None;
        for call_site in call_sites {
            let pc = call_site.pc();
            let duplicate = match &mut seen_pcs {
                Some(seen) => !seen.insert(pc),
                None => previous_pc.replace(pc) == Some(pc),
            };
            if duplicate {
                return Err(ValidationError::DuplicateDebugCallSite { pc });
            }
            let Some(instruction) = self.code.get(pc) else {
                return Err(ValidationError::DebugCallSiteOutOfBounds {
                    pc,
                    code_len: self.code.len(),
                });
            };
            let opcode = instruction
                .opcode()
                .map_err(|error| ValidationError::InvalidOpcode { pc, error })?;
            if !matches!(
                opcode,
                OpCode::Call
                    | OpCode::ForInCall
                    | OpCode::CallOne
                    | OpCode::CallVararg
                    | OpCode::CallOpen
                    | OpCode::TailCall
                    | OpCode::TailCallVararg
                    | OpCode::TailCallOpen
            ) {
                return Err(ValidationError::DebugCallSiteNotCall { pc, opcode });
            }
        }
        let sites = self.debug.diagnostic_sites();
        let mut seen_operands = (!sites.windows(2).all(|pair| pair[0].pc() <= pair[1].pc()))
            .then(std::collections::BTreeSet::new);
        // The registers of the sites at the current pc, for an ordered list.
        let mut current_pc = None;
        let mut registers_at_pc = [0_u64; 4];
        for site in sites {
            if site.pc() >= self.code.len() {
                return Err(ValidationError::DebugDiagnosticSiteOutOfBounds {
                    pc: site.pc(),
                    code_len: self.code.len(),
                });
            }
            self.check_register(site.pc(), site.register())?;
            let duplicate = if let Some(seen) = &mut seen_operands {
                !seen.insert((site.pc(), site.register()))
            } else {
                if current_pc.replace(site.pc()) != Some(site.pc()) {
                    registers_at_pc = [0; 4];
                }
                let register = usize::from(site.register());
                let bit = 1 << (register % 64);
                let seen = registers_at_pc[register / 64] & bit != 0;
                registers_at_pc[register / 64] |= bit;
                seen
            };
            if duplicate {
                return Err(ValidationError::DuplicateDebugDiagnosticSite {
                    pc: site.pc(),
                    register: site.register(),
                });
            }
        }
        for local in self.debug.local_variables() {
            if local.register() >= self.register_count {
                return Err(ValidationError::DebugLocalRegisterOutOfBounds {
                    register: local.register(),
                    register_count: self.register_count,
                });
            }
            if local.start_pc() > local.end_pc() || local.end_pc() > self.code.len() {
                return Err(ValidationError::DebugLocalRangeOutOfBounds {
                    start_pc: local.start_pc(),
                    end_pc: local.end_pc(),
                    code_len: self.code.len(),
                });
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    // `NumericForPrep`/`NumericForLoop` and `ForInLoop` are distinct opcode
    // groups whose validation happens to coincide.
    #[allow(clippy::match_same_arms)]
    fn validate_instruction(
        &self,
        pc: usize,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<(), ValidationError> {
        let register = |register| self.check_register(pc, register);
        let constant = |index| self.check_constant(pc, index);
        if matches!(
            opcode,
            OpCode::Vararg
                | OpCode::CallVararg
                | OpCode::TailCallVararg
                | OpCode::ReturnVararg
                | OpCode::TableVararg
        ) && !self.is_vararg
        {
            return Err(ValidationError::VarargOpcodeInFixedChunk { pc, opcode });
        }
        match opcode {
            OpCode::LoadNil
            | OpCode::NewTable
            | OpCode::CloseUpvalue
            | OpCode::MarkClose
            | OpCode::CloseLocal => register(instruction.a()),
            OpCode::ClearRange => self.check_register_range(pc, instruction.a(), instruction.b()),
            OpCode::Closure => {
                register(instruction.a())?;
                if usize::from(instruction.d()) < self.prototypes.len() {
                    Ok(())
                } else {
                    Err(ValidationError::PrototypeOutOfBounds {
                        pc,
                        index: instruction.d(),
                        prototype_count: self.prototypes.len(),
                    })
                }
            }
            OpCode::LoadConstant => {
                register(instruction.a())?;
                constant(u32::from(instruction.d()))
            }
            OpCode::LoadConstantWide => {
                register(instruction.a())?;
                constant(self.wide_index(pc)?)
            }
            OpCode::GetGlobal | OpCode::SetGlobal | OpCode::DeclareGlobal => {
                register(instruction.a())?;
                self.check_global_name(pc, u32::from(instruction.d()))
            }
            OpCode::GetGlobalWide | OpCode::SetGlobalWide | OpCode::DeclareGlobalWide => {
                register(instruction.a())?;
                self.check_global_name(pc, self.wide_index(pc)?)
            }
            OpCode::GetUpvalue | OpCode::SetUpvalue => {
                register(instruction.a())?;
                self.check_instruction_upvalue(pc, instruction.d())
            }
            OpCode::Move | OpCode::Negate | OpCode::Not | OpCode::Length | OpCode::BitNot => {
                register(instruction.a())?;
                register(instruction.b())
            }
            OpCode::Add
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
            | OpCode::SetTable
            | OpCode::DeclareTable
            | OpCode::CallOne => {
                register(instruction.a())?;
                register(instruction.b())?;
                register(instruction.c())
            }
            OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK => {
                register(instruction.a())?;
                register(instruction.b())?;
                self.check_number_constant(pc, instruction.c())
            }
            OpCode::TestEqual | OpCode::TestLessThan | OpCode::TestLessEqual => {
                self.check_test_jump(pc)?;
                Self::check_test_flags(pc, instruction, 1)?;
                register(instruction.a())?;
                register(instruction.b())
            }
            OpCode::TestEqualK => {
                self.check_test_jump(pc)?;
                Self::check_test_flags(pc, instruction, 1)?;
                register(instruction.a())?;
                constant(u32::from(instruction.b()))?;
                if matches!(
                    self.constants[usize::from(instruction.b())],
                    Constant::Nil
                        | Constant::Boolean(_)
                        | Constant::Integer(_)
                        | Constant::Number(_)
                        | Constant::String(_)
                ) {
                    Ok(())
                } else {
                    Err(ValidationError::TestConstantInvalid {
                        pc,
                        index: instruction.b(),
                    })
                }
            }
            OpCode::TestLessThanK | OpCode::TestLessEqualK => {
                self.check_test_jump(pc)?;
                Self::check_test_flags(pc, instruction, 3)?;
                register(instruction.a())?;
                self.check_number_constant(pc, instruction.b())
            }
            OpCode::GetField => {
                register(instruction.a())?;
                register(instruction.b())?;
                self.check_field_name(pc, instruction.c())
            }
            OpCode::SelfField => {
                register(instruction.a())?;
                self.check_register(pc, instruction.a().wrapping_add(1))?;
                register(instruction.b())?;
                self.check_field_name(pc, instruction.c())
            }
            OpCode::SetField => {
                register(instruction.a())?;
                self.check_field_name(pc, instruction.b())?;
                register(instruction.c())
            }
            OpCode::GetTabUp => {
                register(instruction.a())?;
                self.check_instruction_upvalue(pc, u16::from(instruction.b()))?;
                self.check_field_name(pc, instruction.c())
            }
            OpCode::SetTabUp => {
                self.check_instruction_upvalue(pc, u16::from(instruction.a()))?;
                self.check_field_name(pc, instruction.b())?;
                register(instruction.c())
            }
            OpCode::Concat => {
                register(instruction.a())?;
                if instruction.b() > instruction.c() {
                    return Err(ValidationError::RegisterOutOfBounds {
                        pc,
                        register: usize::from(instruction.b()),
                        register_count: self.register_count,
                    });
                }
                for register in instruction.b()..=instruction.c() {
                    self.check_register(pc, register)?;
                }
                Ok(())
            }
            OpCode::Jump => self.check_jump(pc, instruction.jump_offset()),
            OpCode::JumpIfFalse => {
                register(instruction.a())?;
                self.check_jump(
                    pc,
                    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
                )
            }
            OpCode::NumericForPrep | OpCode::NumericForLoop => {
                self.check_register_range(pc, instruction.a(), 4)?;
                self.check_jump(
                    pc,
                    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
                )
            }
            OpCode::ForInCall => {
                if instruction.b() == 0 || instruction.c() != 0 {
                    return Err(ValidationError::ForInCallInvalid {
                        pc,
                        variables: instruction.b(),
                    });
                }
                // The call window `a+3 ..= a+5` and the results.
                let window = instruction.b().max(3);
                self.check_register_range(pc, instruction.a(), 3)?;
                let Some(first) = instruction.a().checked_add(3) else {
                    return Err(ValidationError::RegisterOutOfBounds {
                        pc,
                        register: usize::from(instruction.a()) + 3,
                        register_count: self.register_count,
                    });
                };
                self.check_register_range(pc, first, window)
            }
            OpCode::ForInLoop => {
                self.check_register_range(pc, instruction.a(), 4)?;
                self.check_jump(
                    pc,
                    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes())),
                )
            }
            OpCode::Call
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen => {
                let argument_last =
                    usize::from(instruction.a()).saturating_add(usize::from(instruction.b()));
                let result_last = usize::from(instruction.a())
                    .saturating_add(usize::from(instruction.c().saturating_sub(1)));
                let last = argument_last.max(result_last);
                if last >= usize::from(self.register_count) {
                    return Err(ValidationError::RegisterOutOfBounds {
                        pc,
                        register: last,
                        register_count: self.register_count,
                    });
                }
                Ok(())
            }
            OpCode::Return | OpCode::ReturnVararg | OpCode::ReturnOpen | OpCode::Vararg => {
                self.check_register_range(pc, instruction.a(), instruction.b())
            }
            OpCode::TableVararg | OpCode::TableOpen => {
                self.check_register(pc, instruction.a())?;
                if instruction.d() == 0 {
                    Err(ValidationError::TableVarargStartIsZero { pc })
                } else {
                    Ok(())
                }
            }
            OpCode::ExtraArgument | OpCode::Halt => Ok(()),
        }
    }

    fn check_register(&self, pc: usize, register: u8) -> Result<(), ValidationError> {
        if register < self.register_count {
            Ok(())
        } else {
            Err(ValidationError::RegisterOutOfBounds {
                pc,
                register: usize::from(register),
                register_count: self.register_count,
            })
        }
    }

    fn check_register_range(&self, pc: usize, base: u8, count: u8) -> Result<(), ValidationError> {
        let last = usize::from(base).saturating_add(usize::from(count.saturating_sub(1)));
        if last < usize::from(self.register_count) {
            Ok(())
        } else {
            Err(ValidationError::RegisterOutOfBounds {
                pc,
                register: last,
                register_count: self.register_count,
            })
        }
    }
    fn wide_index(&self, pc: usize) -> Result<u32, ValidationError> {
        self.code
            .get(pc + 1)
            .copied()
            .filter(|instruction| instruction.opcode().ok() == Some(OpCode::ExtraArgument))
            .map(Instruction::extra_argument_index)
            .ok_or_else(|| {
                let opcode = self.code[pc]
                    .opcode()
                    .expect("wide index is requested only for a decoded opcode");
                ValidationError::MissingExtraArgument { pc, opcode }
            })
    }
    fn check_constant(&self, pc: usize, index: u32) -> Result<(), ValidationError> {
        if usize::try_from(index).is_ok_and(|index| index < self.constants.len()) {
            Ok(())
        } else {
            Err(ValidationError::ConstantOutOfBounds {
                pc,
                index,
                constant_count: self.constants.len(),
            })
        }
    }
    fn check_global_name(&self, pc: usize, index: u32) -> Result<(), ValidationError> {
        self.check_constant(pc, index)?;
        if matches!(
            self.constants[usize::try_from(index).expect("validated constant index")],
            Constant::String(_)
        ) {
            Ok(())
        } else {
            Err(ValidationError::GlobalNameNotString { pc, index })
        }
    }
    fn check_field_name(&self, pc: usize, index: u8) -> Result<(), ValidationError> {
        self.check_constant(pc, u32::from(index))?;
        if matches!(self.constants[usize::from(index)], Constant::String(_)) {
            Ok(())
        } else {
            Err(ValidationError::FieldNameNotString { pc, index })
        }
    }
    fn check_test_jump(&self, pc: usize) -> Result<(), ValidationError> {
        if self
            .code
            .get(pc + 1)
            .is_some_and(|next| next.opcode() == Ok(OpCode::Jump))
        {
            Ok(())
        } else {
            Err(ValidationError::TestMissingJump { pc })
        }
    }
    fn check_test_flags(
        pc: usize,
        instruction: Instruction,
        mask: u8,
    ) -> Result<(), ValidationError> {
        if instruction.c() & !mask == 0 {
            Ok(())
        } else {
            Err(ValidationError::TestFlagsInvalid {
                pc,
                flags: instruction.c(),
            })
        }
    }
    fn check_number_constant(&self, pc: usize, index: u8) -> Result<(), ValidationError> {
        self.check_constant(pc, u32::from(index))?;
        if matches!(
            self.constants[usize::from(index)],
            Constant::Integer(_) | Constant::Number(_)
        ) {
            Ok(())
        } else {
            Err(ValidationError::ArithmeticConstantNotNumber { pc, index })
        }
    }
    fn check_instruction_upvalue(&self, pc: usize, index: u16) -> Result<(), ValidationError> {
        if usize::from(index) < self.upvalues.len() {
            Ok(())
        } else {
            Err(ValidationError::InstructionUpvalueOutOfBounds {
                pc,
                index,
                upvalue_count: self.upvalues.len(),
            })
        }
    }
    fn check_jump(&self, pc: usize, offset: i32) -> Result<(), ValidationError> {
        let target = i64::try_from(pc)
            .ok()
            .and_then(|pc| pc.checked_add(1))
            .and_then(|next| next.checked_add(i64::from(offset)));
        let code_len = i64::try_from(self.code.len()).ok();
        if target
            .zip(code_len)
            .is_some_and(|(target, len)| (0..len).contains(&target))
        {
            let target = usize::try_from(target.expect("target was checked above"))
                .expect("validated non-negative jump target fits usize");
            if self.code[target].opcode().ok() == Some(OpCode::ExtraArgument) {
                Err(ValidationError::JumpIntoExtraArgument { pc, offset })
            } else {
                Ok(())
            }
        } else {
            Err(ValidationError::JumpOutOfBounds {
                pc,
                offset,
                code_len: self.code.len(),
            })
        }
    }
}

/// A rejected malformed chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    EmptyCode,
    InvalidOpcode {
        pc: usize,
        error: InvalidOpCode,
    },
    RegisterOutOfBounds {
        pc: usize,
        register: usize,
        register_count: u8,
    },
    ConstantOutOfBounds {
        pc: usize,
        index: u32,
        constant_count: usize,
    },
    GlobalNameNotString {
        pc: usize,
        index: u32,
    },
    FieldNameNotString {
        pc: usize,
        index: u8,
    },
    ArithmeticConstantNotNumber {
        pc: usize,
        index: u8,
    },
    ForInCallInvalid {
        pc: usize,
        variables: u8,
    },
    TestMissingJump {
        pc: usize,
    },
    TestFlagsInvalid {
        pc: usize,
        flags: u8,
    },
    TestConstantInvalid {
        pc: usize,
        index: u8,
    },
    OrphanExtraArgument {
        pc: usize,
    },
    MissingExtraArgument {
        pc: usize,
        opcode: OpCode,
    },
    JumpIntoExtraArgument {
        pc: usize,
        offset: i32,
    },
    JumpOutOfBounds {
        pc: usize,
        offset: i32,
        code_len: usize,
    },
    MissingTerminator,
    CodeTooLong {
        length: usize,
    },
    TailCallMissingForwarder {
        pc: usize,
    },
    ParameterOutOfBounds {
        parameter_count: u8,
        register_count: u8,
    },
    PrototypeOutOfBounds {
        pc: usize,
        index: u16,
        prototype_count: usize,
    },
    RootUpvalue {
        index: usize,
    },
    DetachedUpvalueInNestedChunk {
        index: usize,
    },
    UpvalueRegisterOutOfBounds {
        index: usize,
        register: u8,
        parent_register_count: u8,
    },
    UpvalueOutOfBounds {
        index: usize,
        source: u16,
        parent_upvalue_count: usize,
    },
    InstructionUpvalueOutOfBounds {
        pc: usize,
        index: u16,
        upvalue_count: usize,
    },
    ArgRegisterInFixedChunk {
        register: u8,
    },
    ArgRegisterOutOfBounds {
        register: u8,
        register_count: u8,
    },
    ArgRegisterOverlapsParameter {
        register: u8,
        parameter_count: u8,
    },
    VarargOpcodeInFixedChunk {
        pc: usize,
        opcode: OpCode,
    },
    TableVarargStartIsZero {
        pc: usize,
    },
    DuplicateDebugCallSite {
        pc: usize,
    },
    DebugCallSiteOutOfBounds {
        pc: usize,
        code_len: usize,
    },
    DebugCallSiteNotCall {
        pc: usize,
        opcode: OpCode,
    },
    DuplicateDebugDiagnosticSite {
        pc: usize,
        register: u8,
    },
    DebugDiagnosticSiteOutOfBounds {
        pc: usize,
        code_len: usize,
    },
    DebugLineInfoLength {
        line_count: usize,
        code_len: usize,
    },
    DebugLocalRegisterOutOfBounds {
        register: u8,
        register_count: u8,
    },
    DebugLocalRangeOutOfBounds {
        start_pc: usize,
        end_pc: usize,
        code_len: usize,
    },
}

const fn is_wide_reference(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::LoadConstantWide
            | OpCode::GetGlobalWide
            | OpCode::SetGlobalWide
            | OpCode::DeclareGlobalWide
    )
}

impl fmt::Display for ValidationError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCode => write!(formatter, "chunk has no instructions"),
            Self::InvalidOpcode { pc, error } => write!(formatter, "instruction {pc}: {error}"),
            Self::RegisterOutOfBounds {
                pc,
                register,
                register_count,
            } => write!(
                formatter,
                "instruction {pc}: register {register} is outside frame of {register_count} registers"
            ),
            Self::ConstantOutOfBounds {
                pc,
                index,
                constant_count,
            } => write!(
                formatter,
                "instruction {pc}: constant {index} is outside pool of {constant_count} constants"
            ),
            Self::GlobalNameNotString { pc, index } => {
                write!(
                    formatter,
                    "instruction {pc}: global name constant {index} is not a string"
                )
            }
            Self::FieldNameNotString { pc, index } => {
                write!(
                    formatter,
                    "instruction {pc}: field name constant {index} is not a string"
                )
            }
            Self::ArithmeticConstantNotNumber { pc, index } => {
                write!(
                    formatter,
                    "instruction {pc}: arithmetic constant {index} is not a number"
                )
            }
            Self::ForInCallInvalid { pc, variables } => write!(
                formatter,
                "instruction {pc}: a generic-for call needs 1 to 255 variables and a zero c, not {variables}"
            ),
            Self::TestMissingJump { pc } => write!(
                formatter,
                "instruction {pc}: a test instruction must be followed by Jump"
            ),
            Self::TestFlagsInvalid { pc, flags } => write!(
                formatter,
                "instruction {pc}: test flags {flags:#x} are not valid for this opcode"
            ),
            Self::TestConstantInvalid { pc, index } => write!(
                formatter,
                "instruction {pc}: test constant {index} has an unsupported type"
            ),
            Self::OrphanExtraArgument { pc } => {
                write!(
                    formatter,
                    "instruction {pc}: ExtraArgument has no wide reference"
                )
            }
            Self::MissingExtraArgument { pc, opcode } => write!(
                formatter,
                "instruction {pc}: {opcode:?} must be followed by ExtraArgument"
            ),
            Self::JumpIntoExtraArgument { pc, offset } => write!(
                formatter,
                "instruction {pc}: jump offset {offset} lands on an ExtraArgument payload"
            ),
            Self::JumpOutOfBounds {
                pc,
                offset,
                code_len,
            } => write!(
                formatter,
                "instruction {pc}: jump offset {offset} leaves code of {code_len} instructions"
            ),
            Self::MissingTerminator => write!(formatter, "chunk does not end in Return or Halt"),
            Self::CodeTooLong { length } => {
                write!(
                    formatter,
                    "chunk has {length} instructions, more than 2^32 - 1"
                )
            }
            Self::TailCallMissingForwarder { pc } => write!(
                formatter,
                "instruction {pc}: tail call must be followed by ReturnOpen with no fixed values"
            ),
            Self::ParameterOutOfBounds {
                parameter_count,
                register_count,
            } => write!(
                formatter,
                "{parameter_count} parameters do not fit in {register_count} registers"
            ),
            Self::PrototypeOutOfBounds {
                pc,
                index,
                prototype_count,
            } => write!(
                formatter,
                "instruction {pc}: prototype {index} is outside {prototype_count} prototypes"
            ),
            Self::RootUpvalue { index } => {
                write!(formatter, "root chunk upvalue {index} has no parent")
            }
            Self::DetachedUpvalueInNestedChunk { index } => {
                write!(formatter, "nested chunk upvalue {index} is detached")
            }
            Self::UpvalueRegisterOutOfBounds {
                index,
                register,
                parent_register_count,
            } => write!(
                formatter,
                "upvalue {index}: parent register {register} is outside frame of {parent_register_count} registers"
            ),
            Self::UpvalueOutOfBounds {
                index,
                source,
                parent_upvalue_count,
            } => write!(
                formatter,
                "upvalue {index}: parent upvalue {source} is outside {parent_upvalue_count} upvalues"
            ),
            Self::InstructionUpvalueOutOfBounds {
                pc,
                index,
                upvalue_count,
            } => write!(
                formatter,
                "instruction {pc}: upvalue {index} is outside {upvalue_count} captures"
            ),
            Self::ArgRegisterInFixedChunk { register } => {
                write!(
                    formatter,
                    "arg register {register} belongs to a fixed-arity chunk"
                )
            }
            Self::ArgRegisterOutOfBounds {
                register,
                register_count,
            } => write!(
                formatter,
                "arg register {register} is outside frame of {register_count} registers"
            ),
            Self::ArgRegisterOverlapsParameter {
                register,
                parameter_count,
            } => write!(
                formatter,
                "arg register {register} overlaps one of {parameter_count} fixed parameters"
            ),
            Self::VarargOpcodeInFixedChunk { pc, opcode } => write!(
                formatter,
                "instruction {pc}: {opcode:?} requires a vararg chunk"
            ),
            Self::TableVarargStartIsZero { pc } => {
                write!(
                    formatter,
                    "instruction {pc}: TableVararg starts at index zero"
                )
            }
            Self::DuplicateDebugCallSite { pc } => {
                write!(
                    formatter,
                    "debug metadata has multiple call sites at instruction {pc}"
                )
            }
            Self::DebugCallSiteOutOfBounds { pc, code_len } => write!(
                formatter,
                "debug call site {pc} is outside code of {code_len} instructions"
            ),
            Self::DebugCallSiteNotCall { pc, opcode } => write!(
                formatter,
                "debug call site {pc} points to {opcode:?}, not a call instruction"
            ),
            Self::DuplicateDebugDiagnosticSite { pc, register } => write!(
                formatter,
                "debug metadata names register {register} more than once at instruction {pc}"
            ),
            Self::DebugDiagnosticSiteOutOfBounds { pc, code_len } => write!(
                formatter,
                "debug diagnostic site {pc} is outside code of {code_len} instructions"
            ),
            Self::DebugLineInfoLength {
                line_count,
                code_len,
            } => write!(
                formatter,
                "debug line data has {line_count} entries for {code_len} instructions"
            ),
            Self::DebugLocalRegisterOutOfBounds {
                register,
                register_count,
            } => write!(
                formatter,
                "debug local register {register} is outside frame of {register_count} registers"
            ),
            Self::DebugLocalRangeOutOfBounds {
                start_pc,
                end_pc,
                code_len,
            } => write!(
                formatter,
                "debug local range {start_pc}..{end_pc} is outside code of {code_len} instructions"
            ),
        }
    }
}
impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_formats_round_trip() {
        let abc = Instruction::abc(OpCode::Add, 3, 4, 5);
        assert_eq!(abc.opcode(), Ok(OpCode::Add));
        assert_eq!((abc.a(), abc.b(), abc.c()), (3, 4, 5));
        let ad = Instruction::ad(OpCode::LoadConstant, 7, u16::MAX);
        assert_eq!((ad.a(), ad.d()), (7, u16::MAX));
        let jump = Instruction::aj(OpCode::Jump, -23).unwrap();
        assert_eq!(jump.jump_offset(), -23);
    }

    #[test]
    fn lua53_integer_operators_encode_and_validate() {
        for opcode in [
            OpCode::FloorDivide,
            OpCode::BitAnd,
            OpCode::BitOr,
            OpCode::BitXor,
            OpCode::ShiftLeft,
            OpCode::ShiftRight,
        ] {
            let instruction = Instruction::binary(opcode, 0, 1, 2);
            assert_eq!(instruction.opcode(), Ok(opcode));
            assert_eq!(
                Chunk::new(3, Vec::new(), vec![instruction, Instruction::halt()]).validate(),
                Ok(())
            );
        }
        let bit_not = Instruction::unary(OpCode::BitNot, 0, 1);
        assert_eq!(bit_not.opcode(), Ok(OpCode::BitNot));
        assert_eq!(
            Chunk::new(2, Vec::new(), vec![bit_not, Instruction::halt()]).validate(),
            Ok(())
        );
    }

    #[test]
    fn clear_range_encodes_and_validates_its_register_span() {
        let instruction = Instruction::clear_range(2, 3);
        assert_eq!(instruction.opcode(), Ok(OpCode::ClearRange));
        assert_eq!(
            (instruction.a(), instruction.b(), instruction.c()),
            (2, 3, 0)
        );

        let valid = Chunk::new(5, Vec::new(), vec![instruction, Instruction::halt()]);
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            4,
            Vec::new(),
            vec![Instruction::clear_range(2, 3), Instruction::halt()],
        );
        assert_eq!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds {
                pc: 0,
                register: 4,
                register_count: 4,
            })
        );
    }

    #[test]
    fn validates_wide_constant_references_and_rejects_payload_misuse() {
        let wide_index = u32::from(u16::MAX) + 1;
        let mut constants = vec![Constant::Nil; usize::try_from(wide_index + 1).unwrap()];
        constants[usize::try_from(wide_index).unwrap()] = Constant::Number(42.0);
        let valid = Chunk::new(
            1,
            constants,
            vec![
                Instruction::load_constant_wide(0),
                Instruction::extra_argument(wide_index).unwrap(),
                Instruction::halt(),
            ],
        );
        assert_eq!(valid.validate(), Ok(()));

        let missing = Chunk::new(
            1,
            vec![],
            vec![Instruction::load_constant_wide(0), Instruction::halt()],
        );
        assert_eq!(
            missing.validate(),
            Err(ValidationError::MissingExtraArgument {
                pc: 0,
                opcode: OpCode::LoadConstantWide,
            })
        );

        let orphan = Chunk::new(
            1,
            vec![],
            vec![Instruction::extra_argument(0).unwrap(), Instruction::halt()],
        );
        assert_eq!(
            orphan.validate(),
            Err(ValidationError::OrphanExtraArgument { pc: 0 })
        );

        let jumped_payload = Chunk::new(
            1,
            vec![Constant::Nil; usize::try_from(wide_index + 1).unwrap()],
            vec![
                Instruction::aj(OpCode::Jump, 1).unwrap(),
                Instruction::load_constant_wide(0),
                Instruction::extra_argument(wide_index).unwrap(),
                Instruction::halt(),
            ],
        );
        assert_eq!(
            jumped_payload.validate(),
            Err(ValidationError::JumpIntoExtraArgument { pc: 0, offset: 1 })
        );
    }

    #[test]
    fn verifies_a_small_program() {
        let chunk = Chunk::new(
            3,
            vec![Constant::Integer(42), Constant::String("answer".into())],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::set_global(0, 1),
                Instruction::return_(0),
            ],
        );
        assert_eq!(chunk.validate(), Ok(()));
    }

    #[test]
    fn chunks_default_to_an_unknown_source_without_call_sites() {
        let chunk = Chunk::new(1, vec![], vec![Instruction::halt()]);
        assert_eq!(chunk.debug.source_name().as_bytes(), b"=?");
        assert_eq!(chunk.debug.call_sites(), [] as [CallSite; 0]);
        assert_eq!(chunk.validate(), Ok(()));
    }

    #[test]
    fn local_debug_metadata_filters_half_open_ranges_in_declaration_order() {
        let metadata = DebugMetadata::new("locals.lua", Vec::new()).with_local_variables(vec![
            LocalVariable::new("first", 0, 0, 3),
            LocalVariable::new("second", 1, 1, 2),
            LocalVariable::new("third", 2, 2, 3),
        ]);
        let names = |pc| {
            metadata
                .locals_at(pc)
                .map(|local| local.name().as_bytes().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(0), vec![b"first".to_vec()]);
        assert_eq!(names(1), vec![b"first".to_vec(), b"second".to_vec()]);
        assert_eq!(names(2), vec![b"first".to_vec(), b"third".to_vec()]);
        assert_eq!(names(3), [] as [Vec<u8>; 0]);

        let chunk = Chunk::new(
            3,
            vec![],
            vec![
                Instruction::load_nil(0),
                Instruction::load_nil(1),
                Instruction::halt(),
            ],
        )
        .with_debug_metadata(metadata);
        assert_eq!(chunk.validate(), Ok(()));
    }

    #[test]
    fn rejects_invalid_local_debug_metadata() {
        let invalid_register = Chunk::new(1, vec![], vec![Instruction::halt()])
            .with_debug_metadata(
                DebugMetadata::new("locals.lua", Vec::new())
                    .with_local_variables(vec![LocalVariable::new("value", 1, 0, 1)]),
            );
        assert!(matches!(
            invalid_register.validate(),
            Err(ValidationError::DebugLocalRegisterOutOfBounds { .. })
        ));

        let invalid_range = Chunk::new(1, vec![], vec![Instruction::halt()]).with_debug_metadata(
            DebugMetadata::new("locals.lua", Vec::new())
                .with_local_variables(vec![LocalVariable::new("value", 0, 1, 0)]),
        );
        assert!(matches!(
            invalid_range.validate(),
            Err(ValidationError::DebugLocalRangeOutOfBounds { .. })
        ));
    }

    #[test]
    fn validates_debug_call_site_program_counters() {
        let valid = || {
            Chunk::new(
                3,
                vec![],
                vec![Instruction::call_one(0, 1, 2), Instruction::halt()],
            )
        };
        let named = valid().with_debug_metadata(DebugMetadata::new(
            "calls.lua",
            vec![CallSite::named(0, "print", CallKind::Global)],
        ));
        assert_eq!(named.validate(), Ok(()));

        let duplicate = valid().with_debug_metadata(DebugMetadata::new(
            "calls.lua",
            vec![CallSite::anonymous(0), CallSite::anonymous(0)],
        ));
        assert!(matches!(
            duplicate.validate(),
            Err(ValidationError::DuplicateDebugCallSite { pc: 0 })
        ));

        let out_of_bounds = valid().with_debug_metadata(DebugMetadata::new(
            "calls.lua",
            vec![CallSite::anonymous(2)],
        ));
        assert!(matches!(
            out_of_bounds.validate(),
            Err(ValidationError::DebugCallSiteOutOfBounds { pc: 2, .. })
        ));

        let not_a_call = valid().with_debug_metadata(DebugMetadata::new(
            "calls.lua",
            vec![CallSite::anonymous(1)],
        ));
        assert!(matches!(
            not_a_call.validate(),
            Err(ValidationError::DebugCallSiteNotCall {
                pc: 1,
                opcode: OpCode::Halt,
            })
        ));
    }

    #[test]
    fn validates_every_call_one_register() {
        let valid = Chunk::new(
            3,
            vec![],
            vec![Instruction::call_one(0, 1, 2), Instruction::halt()],
        );
        assert_eq!(valid.validate(), Ok(()));
        for instruction in [
            Instruction::call_one(3, 1, 2),
            Instruction::call_one(0, 3, 2),
            Instruction::call_one(0, 1, 3),
        ] {
            let chunk = Chunk::new(3, vec![], vec![instruction, Instruction::halt()]);
            assert!(matches!(
                chunk.validate(),
                Err(ValidationError::RegisterOutOfBounds { .. })
            ));
        }
    }

    #[test]
    fn validates_complete_or_stripped_debug_line_data() {
        let chunk = Chunk::new(
            1,
            vec![],
            vec![Instruction::load_nil(0), Instruction::halt()],
        );
        assert_eq!(chunk.validate(), Ok(()));

        let complete = chunk.clone().with_debug_metadata(
            DebugMetadata::new("lines.lua", Vec::new()).with_line_info(vec![0, 4]),
        );
        assert_eq!(complete.debug.line_at(0), None);
        assert_eq!(complete.debug.active_lines(), vec![4]);
        assert_eq!(complete.validate(), Ok(()));

        let incomplete = chunk.with_debug_metadata(
            DebugMetadata::new("lines.lua", Vec::new()).with_line_info(vec![4]),
        );
        assert!(matches!(
            incomplete.validate(),
            Err(ValidationError::DebugLineInfoLength { .. })
        ));
    }

    #[test]
    fn validates_length_as_a_unary_register_operation() {
        let valid = Chunk::new(
            2,
            vec![],
            vec![Instruction::length(0, 1), Instruction::return_(0)],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            1,
            vec![],
            vec![Instruction::length(0, 1), Instruction::halt()],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_concat_as_a_binary_register_operation() {
        let valid = Chunk::new(
            3,
            vec![],
            vec![Instruction::concat(0, 1, 2), Instruction::return_(0)],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            2,
            vec![],
            vec![Instruction::concat(0, 1, 2), Instruction::halt()],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_get_and_set_upvalue_indexes() {
        let name = LuaString::from(vec![b'x', 0xff]);
        let child = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
            vec![],
            vec![UpvalueDescriptor {
                name: name.clone(),
                source: UpvalueSource::ParentRegister(0),
            }],
        );
        let parent = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::closure(0, 0), Instruction::return_(0)],
            vec![child],
            vec![],
        );
        assert_eq!(parent.validate(), Ok(()));
        assert_eq!(parent.prototypes[0].upvalues[0].name, name);

        let invalid = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::get_upvalue(0, 0), Instruction::halt()],
            vec![],
            vec![],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::InstructionUpvalueOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_close_upvalue_registers() {
        let valid = Chunk::new(
            1,
            vec![],
            vec![Instruction::close_upvalue(0), Instruction::halt()],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            1,
            vec![],
            vec![Instruction::close_upvalue(1), Instruction::halt()],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_to_be_closed_local_registers() {
        let valid = Chunk::new(
            1,
            vec![],
            vec![
                Instruction::mark_close(0),
                Instruction::close_local(0),
                Instruction::halt(),
            ],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            1,
            vec![],
            vec![
                Instruction::mark_close(1),
                Instruction::close_local(0),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_checked_global_initialization_forms() {
        let compact = Chunk::new(
            1,
            vec![Constant::String("answer".into())],
            vec![Instruction::declare_global(0, 0), Instruction::halt()],
        );
        assert_eq!(compact.validate(), Ok(()));

        let lexical = Chunk::new(
            3,
            vec![],
            vec![Instruction::declare_table(0, 1, 2), Instruction::halt()],
        );
        assert_eq!(lexical.validate(), Ok(()));

        let wide = Chunk::new(
            1,
            vec![Constant::String("answer".into())],
            vec![
                Instruction::declare_global_wide(0),
                Instruction::extra_argument(0).unwrap(),
                Instruction::halt(),
            ],
        );
        assert_eq!(wide.validate(), Ok(()));
    }

    #[test]
    fn rejects_untrusted_references() {
        let bad_register = Chunk::new(
            1,
            vec![],
            vec![Instruction::move_(0, 1), Instruction::halt()],
        );
        assert!(matches!(
            bad_register.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        let bad_constant = Chunk::new(
            1,
            vec![],
            vec![Instruction::load_constant(0, 0), Instruction::halt()],
        );
        assert!(matches!(
            bad_constant.validate(),
            Err(ValidationError::ConstantOutOfBounds { pc: 0, .. })
        ));
        let bad_jump = Chunk::new(
            1,
            vec![],
            vec![
                Instruction::aj(OpCode::Jump, 1).unwrap(),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            bad_jump.validate(),
            Err(ValidationError::JumpOutOfBounds { pc: 0, .. })
        ));
        let bad_global_name = Chunk::new(
            1,
            vec![Constant::Integer(1)],
            vec![Instruction::get_global(0, 0), Instruction::halt()],
        );
        assert!(matches!(
            bad_global_name.validate(),
            Err(ValidationError::GlobalNameNotString { pc: 0, .. })
        ));
        let bad_field_name = Chunk::new(
            2,
            vec![Constant::Integer(1)],
            vec![Instruction::get_field(0, 1, 0), Instruction::halt()],
        );
        assert!(matches!(
            bad_field_name.validate(),
            Err(ValidationError::FieldNameNotString { pc: 0, index: 0 })
        ));
        // A method lookup writes `a` and `a + 1`, and reads a string field.
        let method = || vec![Constant::String(LuaString::from("m"))];
        assert_eq!(
            Chunk::new(
                3,
                method(),
                vec![Instruction::self_field(0, 2, 0), Instruction::halt()]
            )
            .validate(),
            Ok(())
        );
        assert!(matches!(
            Chunk::new(
                2,
                method(),
                vec![Instruction::self_field(1, 0, 0), Instruction::halt()]
            )
            .validate(),
            Err(ValidationError::RegisterOutOfBounds {
                pc: 0,
                register: 2,
                ..
            })
        ));
        assert!(matches!(
            Chunk::new(
                3,
                vec![Constant::Integer(1)],
                vec![Instruction::self_field(0, 2, 0), Instruction::halt()]
            )
            .validate(),
            Err(ValidationError::FieldNameNotString { pc: 0, index: 0 })
        ));
        let env = || UpvalueDescriptor {
            name: LuaString::from("_ENV"),
            source: UpvalueSource::Detached,
        };
        let tab_up_chunk = |code: Vec<Instruction>, constants: Vec<Constant>| {
            let mut chunk = Chunk::new(2, constants, code);
            chunk.upvalues = vec![env()];
            chunk
        };
        let name = || vec![Constant::String(LuaString::from("x"))];
        assert_eq!(
            tab_up_chunk(
                vec![
                    Instruction::get_tab_up(0, 0, 0),
                    Instruction::set_tab_up(0, 0, 1),
                    Instruction::halt(),
                ],
                name()
            )
            .validate(),
            Ok(())
        );
        assert!(matches!(
            tab_up_chunk(
                vec![Instruction::get_tab_up(0, 1, 0), Instruction::halt()],
                name()
            )
            .validate(),
            Err(ValidationError::InstructionUpvalueOutOfBounds {
                pc: 0,
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            tab_up_chunk(
                vec![Instruction::set_tab_up(3, 0, 0), Instruction::halt()],
                name()
            )
            .validate(),
            Err(ValidationError::InstructionUpvalueOutOfBounds {
                pc: 0,
                index: 3,
                ..
            })
        ));
        assert!(matches!(
            tab_up_chunk(
                vec![Instruction::get_tab_up(0, 0, 0), Instruction::halt()],
                vec![Constant::Integer(1)]
            )
            .validate(),
            Err(ValidationError::FieldNameNotString { pc: 0, index: 0 })
        ));
        assert!(matches!(
            tab_up_chunk(
                vec![Instruction::set_tab_up(0, 0, 0), Instruction::halt()],
                vec![Constant::Integer(1)]
            )
            .validate(),
            Err(ValidationError::FieldNameNotString { pc: 0, index: 0 })
        ));
        assert!(matches!(
            tab_up_chunk(
                vec![Instruction::get_tab_up(2, 0, 0), Instruction::halt()],
                name()
            )
            .validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        let bad_arithmetic_constant = Chunk::new(
            2,
            vec![Constant::Number(1.5), Constant::Nil],
            vec![
                Instruction::abc(OpCode::AddK, 0, 1, 0),
                Instruction::abc(OpCode::ModuloK, 0, 1, 1),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            bad_arithmetic_constant.validate(),
            Err(ValidationError::ArithmeticConstantNotNumber { pc: 1, index: 1 })
        ));
        let missing_arithmetic_constant = Chunk::new(
            2,
            vec![Constant::Integer(1)],
            vec![
                Instruction::abc(OpCode::DivideK, 0, 1, 1),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            missing_arithmetic_constant.validate(),
            Err(ValidationError::ConstantOutOfBounds { pc: 0, .. })
        ));
        let bad_arithmetic_register = Chunk::new(
            2,
            vec![Constant::Integer(1)],
            vec![
                Instruction::abc(OpCode::SubtractK, 0, 2, 0),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            bad_arithmetic_register.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        let good_arithmetic = Chunk::new(
            2,
            vec![Constant::Integer(1), Constant::Number(2.0)],
            vec![
                Instruction::abc(OpCode::MultiplyK, 0, 1, 1),
                Instruction::abc(OpCode::AddK, 1, 0, 0),
                Instruction::halt(),
            ],
        );
        assert_eq!(good_arithmetic.validate(), Ok(()));
        // Compare-and-branch tests: a following jump, flags and constants.
        let jump = Instruction::aj(OpCode::Jump, 0).unwrap();
        let test_chunk =
            |constants: Vec<Constant>, code: Vec<Instruction>| Chunk::new(2, constants, code);
        let accepted = [
            Instruction::test(OpCode::TestEqual, 0, 1, true, false),
            Instruction::test(OpCode::TestLessThan, 1, 0, false, false),
            Instruction::test(OpCode::TestLessEqual, 1, 1, true, false),
        ];
        for test in accepted {
            let chunk = test_chunk(vec![], vec![test, jump, Instruction::halt()]);
            assert_eq!(chunk.validate(), Ok(()), "{test:?}");
        }
        for (constant, opcode, flags) in [
            (Constant::Nil, OpCode::TestEqualK, false),
            (Constant::Boolean(true), OpCode::TestEqualK, false),
            (Constant::String("s".into()), OpCode::TestEqualK, false),
            (Constant::Integer(1), OpCode::TestEqualK, true),
            (Constant::Number(1.5), OpCode::TestLessThanK, true),
            (Constant::Integer(1), OpCode::TestLessEqualK, true),
        ] {
            let chunk = test_chunk(
                vec![constant.clone()],
                vec![
                    Instruction::test(opcode, 0, 0, flags, opcode != OpCode::TestEqualK),
                    jump,
                    Instruction::halt(),
                ],
            );
            assert_eq!(chunk.validate(), Ok(()), "{opcode:?} {constant:?}");
        }
        let no_jump = test_chunk(
            vec![],
            vec![
                Instruction::test(OpCode::TestEqual, 0, 1, false, false),
                Instruction::halt(),
            ],
        );
        assert_eq!(
            no_jump.validate(),
            Err(ValidationError::TestMissingJump { pc: 0 })
        );
        let last_word = test_chunk(
            vec![],
            vec![
                Instruction::halt(),
                Instruction::test(OpCode::TestEqual, 0, 1, false, false),
            ],
        );
        assert!(last_word.validate().is_err());
        // Only bit 0 of `c` is a flag on a register test, and bit 1 only on
        // the ordered constant tests.
        for (opcode, flags) in [
            (OpCode::TestEqual, 2),
            (OpCode::TestLessThan, 4),
            (OpCode::TestEqualK, 2),
            (OpCode::TestLessThanK, 4),
        ] {
            let chunk = test_chunk(
                vec![Constant::Integer(1)],
                vec![
                    Instruction::abc(opcode, 0, 0, flags),
                    jump,
                    Instruction::halt(),
                ],
            );
            assert_eq!(
                chunk.validate(),
                Err(ValidationError::TestFlagsInvalid { pc: 0, flags }),
                "{opcode:?}"
            );
        }
        let bad_register = test_chunk(
            vec![],
            vec![
                Instruction::test(OpCode::TestLessEqual, 0, 2, false, false),
                jump,
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            bad_register.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        for (opcode, constant) in [
            (OpCode::TestLessThanK, Constant::Nil),
            (OpCode::TestLessEqualK, Constant::String("s".into())),
        ] {
            let chunk = test_chunk(
                vec![constant],
                vec![
                    Instruction::test(opcode, 0, 0, false, false),
                    jump,
                    Instruction::halt(),
                ],
            );
            assert!(
                matches!(
                    chunk.validate(),
                    Err(ValidationError::ArithmeticConstantNotNumber { pc: 0, index: 0 })
                ),
                "{opcode:?}"
            );
        }
        let bad_equality_constant = test_chunk(
            vec![Constant::Int64(1)],
            vec![
                Instruction::test(OpCode::TestEqualK, 0, 0, false, false),
                jump,
                Instruction::halt(),
            ],
        );
        assert_eq!(
            bad_equality_constant.validate(),
            Err(ValidationError::TestConstantInvalid { pc: 0, index: 0 })
        );
        let missing_constant = test_chunk(
            vec![],
            vec![
                Instruction::test(OpCode::TestEqualK, 0, 0, false, false),
                jump,
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            missing_constant.validate(),
            Err(ValidationError::ConstantOutOfBounds { pc: 0, .. })
        ));
        // Generic-for opcodes: the control registers, the call window of at
        // least three registers, and the loop's backward jump.
        let for_in = |variables: u8, registers: u8| {
            Chunk::new(
                registers,
                vec![],
                vec![
                    Instruction::for_in_call(0, variables),
                    Instruction::for_in_loop(0, -2),
                    Instruction::halt(),
                ],
            )
        };
        assert_eq!(for_in(1, 6).validate(), Ok(()));
        assert_eq!(for_in(3, 6).validate(), Ok(()));
        assert_eq!(for_in(4, 7).validate(), Ok(()));
        // Fewer variables still need the three window registers.
        assert!(matches!(
            for_in(1, 5).validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        assert!(matches!(
            for_in(4, 6).validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        assert_eq!(
            for_in(0, 6).validate(),
            Err(ValidationError::ForInCallInvalid {
                pc: 0,
                variables: 0
            })
        );
        let nonzero_c = Chunk::new(
            6,
            vec![],
            vec![
                Instruction::abc(OpCode::ForInCall, 0, 1, 1),
                Instruction::halt(),
            ],
        );
        assert!(matches!(
            nonzero_c.validate(),
            Err(ValidationError::ForInCallInvalid { pc: 0, .. })
        ));
        // A loop test reads `a + 3`, and its jump stays inside the code.
        let short_loop = Chunk::new(
            3,
            vec![],
            vec![Instruction::for_in_loop(0, 0), Instruction::halt()],
        );
        assert!(matches!(
            short_loop.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        let wild_loop = Chunk::new(
            6,
            vec![],
            vec![Instruction::for_in_loop(0, 40), Instruction::halt()],
        );
        assert!(wild_loop.validate().is_err());
        let near_the_top = Chunk::new(
            255,
            vec![],
            vec![Instruction::for_in_call(253, 1), Instruction::halt()],
        );
        assert!(near_the_top.validate().is_err());
        let missing_field_name = Chunk::new(
            2,
            vec![],
            vec![Instruction::set_field(0, 0, 1), Instruction::halt()],
        );
        assert!(matches!(
            missing_field_name.validate(),
            Err(ValidationError::ConstantOutOfBounds { pc: 0, .. })
        ));
        let bad_call_results = Chunk::new(
            1,
            vec![],
            vec![Instruction::call(0, 0, 2), Instruction::halt()],
        );
        assert!(matches!(
            bad_call_results.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_zero_argument_calls_and_signed_jumps() {
        let call = Chunk::new(
            1,
            vec![],
            vec![Instruction::call(0, 0, 1), Instruction::halt()],
        );
        assert_eq!(call.validate(), Ok(()));

        let backwards = Chunk::new(
            1,
            vec![],
            vec![
                Instruction::aj(OpCode::Jump, 0).unwrap(),
                Instruction::aj(OpCode::Jump, -2).unwrap(),
                Instruction::halt(),
            ],
        );
        assert_eq!(backwards.validate(), Ok(()));
        assert_eq!(backwards.code[1].jump_offset(), -2);
    }

    #[test]
    fn validates_tail_call_forwarders() {
        let valid = Chunk::new(
            2,
            vec![],
            vec![Instruction::tail_call(0, 1), Instruction::return_open(0, 0)],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            2,
            vec![],
            vec![Instruction::tail_call(0, 1), Instruction::return_(0)],
        );
        assert_eq!(
            invalid.validate(),
            Err(ValidationError::TailCallMissingForwarder { pc: 0 })
        );
    }

    #[test]
    fn validates_nested_prototypes_and_capture_descriptors() {
        let prototype =
            Chunk::with_parts(1, 1, vec![], vec![Instruction::return_(0)], vec![], vec![]);
        let parent = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::closure(0, 0), Instruction::halt()],
            vec![prototype],
            vec![],
        );
        assert_eq!(parent.validate(), Ok(()));

        let missing_prototype = Chunk::new(
            1,
            vec![],
            vec![Instruction::closure(0, 0), Instruction::halt()],
        );
        assert!(matches!(
            missing_prototype.validate(),
            Err(ValidationError::PrototypeOutOfBounds { .. })
        ));
        let root_capture = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::halt()],
            vec![],
            vec![UpvalueDescriptor {
                name: "root".into(),
                source: UpvalueSource::ParentRegister(0),
            }],
        );
        assert!(matches!(
            root_capture.validate(),
            Err(ValidationError::RootUpvalue { .. })
        ));
    }

    #[test]
    fn validates_numeric_for_register_layout_and_jumps() {
        let valid = Chunk::new(
            4,
            vec![],
            vec![
                Instruction::numeric_for_prep(0, 0),
                Instruction::numeric_for_loop(0, -1),
                Instruction::halt(),
            ],
        );
        assert_eq!(valid.validate(), Ok(()));
        let short_frame = Chunk::new(
            4,
            vec![],
            vec![Instruction::numeric_for_prep(1, 0), Instruction::halt()],
        );
        assert!(matches!(
            short_frame.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
        let bad_jump = Chunk::new(
            4,
            vec![],
            vec![Instruction::numeric_for_loop(0, 1), Instruction::halt()],
        );
        assert!(matches!(
            bad_jump.validate(),
            Err(ValidationError::JumpOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_power_as_a_binary_register_operation() {
        let valid = Chunk::new(
            3,
            vec![],
            vec![Instruction::power(2, 0, 1), Instruction::return_(2)],
        );
        assert_eq!(valid.validate(), Ok(()));

        let invalid = Chunk::new(
            2,
            vec![],
            vec![Instruction::power(2, 0, 1), Instruction::halt()],
        );
        assert!(matches!(
            invalid.validate(),
            Err(ValidationError::RegisterOutOfBounds { pc: 0, .. })
        ));
    }

    #[test]
    fn validates_vararg_metadata_and_instruction_forms() {
        let valid = Chunk::with_varargs(
            4,
            1,
            vec![],
            vec![
                Instruction::vararg(1, 2),
                Instruction::call_vararg(0, 1, 1),
                Instruction::table_vararg(3, 1),
                Instruction::return_vararg(1, 0),
            ],
            vec![],
            vec![],
            Some(3),
        );
        assert_eq!(valid.validate(), Ok(()));

        let fixed = Chunk::new(
            1,
            vec![],
            vec![Instruction::vararg(0, 1), Instruction::halt()],
        );
        assert!(matches!(
            fixed.validate(),
            Err(ValidationError::VarargOpcodeInFixedChunk {
                opcode: OpCode::Vararg,
                ..
            })
        ));

        let bad_start = Chunk::with_varargs(
            1,
            0,
            vec![],
            vec![Instruction::table_vararg(0, 0), Instruction::halt()],
            vec![],
            vec![],
            None,
        );
        assert!(matches!(
            bad_start.validate(),
            Err(ValidationError::TableVarargStartIsZero { .. })
        ));

        let bad_arg_register = Chunk::with_varargs(
            2,
            1,
            vec![],
            vec![Instruction::halt()],
            vec![],
            vec![],
            Some(0),
        );
        assert!(matches!(
            bad_arg_register.validate(),
            Err(ValidationError::ArgRegisterOverlapsParameter { .. })
        ));
    }

    #[test]
    fn validates_open_result_consumers_and_return_terminator() {
        let valid = Chunk::new(
            3,
            vec![],
            vec![
                Instruction::call(0, 0, 0),
                Instruction::call_open(1, 1, 0),
                Instruction::table_open(2, 1),
                Instruction::return_open(0, 0),
            ],
        );
        assert_eq!(valid.validate(), Ok(()));

        let bad_table_start = Chunk::new(
            1,
            vec![],
            vec![Instruction::table_open(0, 0), Instruction::halt()],
        );
        assert!(matches!(
            bad_table_start.validate(),
            Err(ValidationError::TableVarargStartIsZero { .. })
        ));

        let bad_call_range = Chunk::new(
            1,
            vec![],
            vec![Instruction::call_open(0, 1, 1), Instruction::halt()],
        );
        assert!(matches!(
            bad_call_range.validate(),
            Err(ValidationError::RegisterOutOfBounds { .. })
        ));
    }
}
