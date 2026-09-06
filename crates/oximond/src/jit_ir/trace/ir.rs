//! Trace ids, types, operations and the owned [`TraceIr`].

use super::helper::{HelperEffects, HelperId};
use super::register_map::RegisterMap;
use super::snapshot::{RegisterEntry, Snapshot};
use super::verify::{TraceVerifyError, verify_trace};
use crate::compat::FloatModulo;
use crate::jit_ir::Comparison;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Most logical frames one snapshot may describe.
pub const MAX_TRACE_FRAME_DEPTH: usize = 8;

/// Most `__index` hops a field binding may follow (see
/// [`BindingKind::Field`]).
pub const MAX_INDEX_HOPS: u8 = 4;

macro_rules! trace_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        pub struct $name(pub u32);

        impl $name {
            /// The id as a table index.
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

trace_id!(
    /// Identifies one trace in the cache.
    TraceId
);
trace_id!(
    /// Identifies one SSA value inside a trace.
    ValueId
);
trace_id!(
    /// Identifies one guard inside a trace, in execution order: preheader
    /// guards first, then body guards.
    GuardId
);
trace_id!(
    /// Identifies one snapshot inside a trace.
    SnapshotId
);
trace_id!(
    /// Identifies one logical guest frame a trace touches. The root frame is
    /// `FrameId(0)`.
    FrameId
);
trace_id!(
    /// Identifies one committed effect, numbered in body order from zero.
    EffectId
);

/// The static type of a trace value. Integer and float are distinct, and nil
/// and boolean are not numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum TraceType {
    Nil,
    Boolean,
    Integer,
    Float,
    /// `LuaJIT`'s `int64_t` cdata, as its two's-complement bits. It is a number
    /// in arithmetic and comparisons, and a cdata everywhere else (never
    /// equal to a non-number).
    Int64,
    /// `LuaJIT`'s `uint64_t` cdata. Like [`Self::Int64`] with unsigned
    /// division, remainder, ordering and conversion to a float.
    UInt64,
    /// A table, known only by identity. Native code carries it as a lane id
    /// the VM resolves through its per-run root bank, never as pointer bits.
    /// Nothing can be done with it except move it, test it for identity and
    /// truthiness, restore it into a register or cell, read and overwrite its
    /// fields (see [`TableField`]), and, if it is an entry input the loop
    /// never writes, bind its storage for table operations (see
    /// [`TableBinding`]).
    Table,
    /// A Lua closure, known only by identity, carried like [`Self::Table`].
    Closure,
    /// A string, known only by identity, carried like [`Self::Table`]. Its
    /// bytes are readable once the entry binds the register that holds it
    /// ([`BindingKind::Bytes`]), by [`TraceOp::StringLength`] and
    /// [`TraceOp::StringByte`]. Two lanes are one string only when they are
    /// the same object, so a trace never compares strings.
    String,
    /// A string read out of a table slot and borrowed: no refcount and no
    /// lane in the root bank. Its bits are the address of the slot that holds
    /// the string (tag [`TraceType::String`], raw bits), so the owner table
    /// must stay rooted and its slot untouched for the borrow's lifetime.
    /// Only [`TraceOp::ArrayLoad`] makes one. It can be moved, tested for
    /// truthiness (always true) and restored into a register, where the
    /// restore clones the string out of the slot. It is never an entry
    /// input, a constant, a loop parameter or a comparison operand. Any
    /// operation for which [`TraceOp::ends_borrows`] holds ends the lifetime
    /// of every borrow: no borrow defined before it may be used after it
    /// (see the verifier).
    StringRef,
    /// A dynamically typed value, the result of an effect. A type guard
    /// narrows it. It cannot appear in a snapshot, which has no way to
    /// rebuild it.
    Any,
    /// A coroutine, known only by identity, carried like [`Self::Table`]: a
    /// lane of the run's root bank, which keeps the thread alive. Nothing can
    /// be done with it except move it, test it for truthiness (always true),
    /// restore it into a register or cell, and pass it to a call that
    /// resumes it ([`HelperEffects::RESUMES`](super::HelperEffects::RESUMES)).
    /// Only an array read or an entry input makes one.
    Thread,
    /// A userdata, known only by identity, carried like [`Self::Thread`]: a
    /// lane of the run's root bank, which keeps it alive. Nothing can be done
    /// with it except move it, test it for identity and truthiness (always
    /// true) and restore it into a register or cell. It is never an entry
    /// input or a constant: only a table read by a lookup call makes one.
    Userdata,
}

impl TraceType {
    /// Whether values of this type are 64-bit integer cdata.
    #[must_use]
    pub const fn is_wide(self) -> bool {
        matches!(self, Self::Int64 | Self::UInt64)
    }

    /// Whether values of this type are opaque identities in a root bank.
    #[must_use]
    pub const fn is_opaque(self) -> bool {
        matches!(
            self,
            Self::Table | Self::Closure | Self::Thread | Self::Userdata | Self::String
        )
    }

    /// Whether values of this type borrow from a table slot
    /// ([`Self::StringRef`]).
    #[must_use]
    pub const fn is_borrowed(self) -> bool {
        matches!(self, Self::StringRef)
    }
}

/// A scalar guest value: a trace constant, or a value read from an exit area.
/// Float equality compares bits, so NaN payloads and signed zero survive.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TraceScalar {
    Nil,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    /// An `int64_t` cdata.
    Int64(i64),
    /// A `uint64_t` cdata.
    UInt64(u64),
    /// A table, as the lane id of its slot in the VM's root bank.
    Table(u64),
    /// A closure, as the lane id of its slot in the VM's root bank.
    Closure(u64),
    /// A string, as the lane id of its slot in the VM's root bank.
    String(u64),
    /// A borrowed string, as the address of the table slot that holds it.
    /// Never a constant or an entry value: only a read of a table produces
    /// one.
    StringRef(u64),
    /// A coroutine, as the lane id of its slot in the VM's root bank.
    Thread(u64),
    /// A userdata, as the lane id of its slot in the VM's root bank.
    Userdata(u64),
}

impl TraceScalar {
    /// The static type of this scalar.
    #[must_use]
    pub const fn ty(self) -> TraceType {
        match self {
            Self::Nil => TraceType::Nil,
            Self::Boolean(_) => TraceType::Boolean,
            Self::Integer(_) => TraceType::Integer,
            Self::Float(_) => TraceType::Float,
            Self::Int64(_) => TraceType::Int64,
            Self::UInt64(_) => TraceType::UInt64,
            Self::Table(_) => TraceType::Table,
            Self::Closure(_) => TraceType::Closure,
            Self::Thread(_) => TraceType::Thread,
            Self::Userdata(_) => TraceType::Userdata,
            Self::String(_) => TraceType::String,
            Self::StringRef(_) => TraceType::StringRef,
        }
    }

    /// Lua truthiness: only nil and false are falsy.
    #[must_use]
    pub const fn is_truthy(self) -> bool {
        !matches!(self, Self::Nil | Self::Boolean(false))
    }
}

impl PartialEq for TraceScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::UInt64(a), Self::UInt64(b)) => a == b,
            (Self::Table(a), Self::Table(b))
            | (Self::Closure(a), Self::Closure(b))
            | (Self::Thread(a), Self::Thread(b))
            | (Self::Userdata(a), Self::Userdata(b))
            | (Self::String(a), Self::String(b))
            | (Self::StringRef(a), Self::StringRef(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for TraceScalar {}

/// A guest location: a frame and a program counter inside its chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TraceSite {
    pub frame: FrameId,
    pub pc: usize,
}

/// How the trace entered a logical frame it inlined: the call that the
/// interpreter would have made, which a snapshot inside the frame must
/// rebuild.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InlinedFrame {
    /// The closure the frame runs, as an index into
    /// [`TraceIr::pinned_closures`]. Its chunk, cells and environment come
    /// from that closure, which never changes them.
    pub closure: u32,
    /// The logical frame that made the call.
    pub parent: FrameId,
    /// The pc of the call instruction in the parent. A snapshot that lists the
    /// parent and this frame gives the parent `last_interpreted_pc` equal to
    /// it and resumes it at the next instruction when the call returns.
    pub call_pc: usize,
    /// The first parent register the call's results land in.
    pub result_base: u8,
    /// How many results the call keeps. The callee's missing values are nil.
    /// Zero is an open result count: the interpreter keeps every value the
    /// callee returns for the call that follows, and the rebuilt frame
    /// returns them the same way.
    pub results: u8,
    /// Whether the call starts over in the interpreter when anything in it
    /// leaves the trace. Every guard of this frame, and of every frame it
    /// calls, leaves through a `Before` snapshot of the call instruction in
    /// `parent`, so no snapshot has to describe this frame. The interpreter
    /// then makes the call itself, with whatever frame state (varargs, a
    /// tail call count) that call builds. That is exact only while nothing
    /// observable (a table store) happens inside the call, so the recorder
    /// refuses a store there.
    pub reexec: bool,
    /// How many tail calls replaced the frame the call made. The interpreter
    /// counts them on the frame (tracebacks show them), so a rebuilt frame
    /// gets the same count. Such a frame's `closure` is the last callee.
    pub tail_calls: u32,
    /// For a frame a tail call made: the pinned closure whose frame made the
    /// last tail call (a lane of [`TraceIr::pinned_closures`]) and the pc of
    /// its call instruction. The interpreter names the new frame after that
    /// call site, which a rebuilt frame takes from the closure's chunk.
    pub tail_site: Option<(u32, usize)>,
    /// Whether `pcall` or `xpcall` made the call. A rebuilt frame returns to
    /// the protected call of the caller, not to a plain call: a later error
    /// the interpreter raises inside it is caught by that call, and its
    /// return delivers `true` before the callee's values.
    #[serde(default)]
    pub protected: Protected,
    /// What made the call: an instruction that runs a metamethod enters the
    /// frame with a return target of its own, which a rebuilt frame needs
    /// for tracebacks, for what the return does with the callee's values,
    /// and for a later error. For an [`Entry::NewIndex`] frame `results` is
    /// zero and the return discards the callee's values.
    #[serde(default)]
    pub entry: Entry,
    /// Whether the frame runs whichever closure of the pinned closure's
    /// prototype the call was made with. [`Self::closure`] is then only the
    /// prototype's name: a call-site guard checks the live callee has the
    /// same chunk as the pinned closure
    /// ([`HelperId::CLOSURE_SAME_PROTO`](super::helper::HelperId::CLOSURE_SAME_PROTO)),
    /// and the callee's cells are read and written through the live closure
    /// ([`HelperId::CELL_GET_INTEGER`](super::helper::HelperId::CELL_GET_INTEGER)
    /// and its siblings), never as window slots. A snapshot frame of such a
    /// call has one register more than the chunk: the last one holds the
    /// closure of this run, which a rebuilt frame runs.
    #[serde(default)]
    pub dynamic: bool,
}

/// What entered an inlined frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum Entry {
    /// A call instruction, or `pcall` and `xpcall` (see
    /// [`InlinedFrame::protected`]).
    #[default]
    Call,
    /// A table read that found the key absent and ran a function `__index`.
    /// The frame's `result_base` is the register the read writes, and its
    /// return writes the callee's first value there.
    Index,
    /// A table write that found the key absent and ran a function
    /// `__newindex`. The return discards the callee's values.
    NewIndex,
    /// `#t` that ran a function `__len`. The frame's `result_base` is the
    /// register the length goes to.
    Len,
    /// An arithmetic, bitwise or unary operator instruction that ran a
    /// function metamethod (`__add`, `__bnot`, ...). The frame's
    /// `result_base` is the register the operator writes, and the return
    /// writes the callee's first value there.
    Operator,
    /// A `==`, `<` or `<=` instruction that ran a function metamethod. The
    /// return writes the truth of the callee's first value into `result_base`,
    /// negated when `invert` (`__le` answered by `__lt` of the swapped
    /// operands).
    Compare { invert: bool },
    /// A compare-and-branch test that ran a function metamethod. The return
    /// takes the truth of the callee's first value (negated when `invert`)
    /// and branches on it against `expected`, as the interpreter does at the
    /// jump word that follows the test.
    Branch { expected: bool, invert: bool },
    /// A `..` that ran a function `__concat` for one pair of operands. The
    /// chain's operands sit in the registers `first..=last` of the parent,
    /// `pending` of them (those from `first` on) not yet folded in. The
    /// frame's `result_base` is the register the chain writes. The return
    /// folds the callee's first value into the pending operands, as the
    /// interpreter's range continuation does.
    Concat { first: u8, last: u8, pending: u8 },
    /// A call of `tostring` that ran a function `__tostring`. The frame's
    /// `result_base` and `results` are the call's, and the return takes the
    /// value as `tostring` does: a string, or an error.
    ToString,
    /// Not a call: the frame is the *caller* of the root frame, which the root
    /// frame returned into (see "Linking through a return" in
    /// `oximond::vm::jit::trace`). The interpreter already owns this frame, so
    /// an exit builds no frame for it: it writes the registers the snapshot
    /// names into the interpreter's frame, resumes it where the snapshot's
    /// phase says, and pops the root frame as the `Return` would have. Frames
    /// above it in the same snapshot (callees the caller's code inlined) are
    /// pushed on the caller as usual. The frame is the first inlined frame of
    /// every snapshot in the caller's code, and [`InlinedFrame::call_pc`] is
    /// the root frame's `Return`, which the root frame stands at in those
    /// snapshots. The caller's registers the code never wrote are
    /// `Unchanged`: the interpreter's frame holds them, or a window slot
    /// ([`ExternalSlot::CallerRegister`]) carries them through the loop.
    ///
    /// `call_pc` is the pc of the call that made the root frame in the
    /// caller's chunk and `result_base` and `results` its result window
    /// (`results` is never zero). A tree with such a frame is entered only by
    /// a root frame that the caller's frame called exactly so.
    Caller {
        call_pc: u32,
        result_base: u8,
        results: u8,
    },
}

/// How a protected call (`pcall`, `xpcall`) made an inlined frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum Protected {
    /// An ordinary call.
    #[default]
    No,
    /// `pcall(f, ...)`.
    Pcall,
    /// `xpcall(f, handler, ...)`, with the handler pinned as a lane of
    /// [`TraceIr::pinned_closures`]. The frame's `result_base` and `results`
    /// are the `xpcall` call's.
    Xpcall { handler: u32 },
}

/// A tail call the interpreter names a frame after: the instruction at `pc`
/// in the chunk of the closure whose frame made the call.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TailSite {
    /// That closure, as a lane of [`TraceIr::pinned_closures`], or `None` for
    /// the closure the root frame ran when the trace was entered (its chunk is
    /// the trace's root chunk).
    pub closure: Option<u32>,
    pub pc: usize,
}

/// A later activation of the root frame: a tail call in the root frame
/// replaced the frame's closure, as the interpreter's in-place tail call
/// does, and the frame now runs `closure` from its pc 0.
///
/// Such a frame is not inlined: it is the interpreter's root frame with
/// another chunk. Its window is the root frame's (the same guest registers
/// and the same slots after them), and a snapshot that names it as its first
/// frame restores the root frame running `closure`. The registers past the
/// closure's own register count are not part of the frame and restore
/// nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootActivation {
    /// The closure the frame runs, as a lane of [`TraceIr::pinned_closures`].
    pub closure: u32,
    /// The tail call that made this activation. The interpreter counts it on
    /// the frame (see [`ExternalSlot::TailCalls`]) and names the frame after
    /// it.
    pub site: TailSite,
}

/// Static facts about one logical frame a trace touches.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TraceFrameInfo {
    /// Registers in the frame's window. Snapshot register maps are this long.
    pub register_count: u16,
    /// Instructions in the frame's chunk. Snapshot pcs are below this.
    pub code_len: usize,
    /// `None` for the root frame, which the interpreter already owns. Other
    /// frames exist only inside the trace until an exit rebuilds them.
    pub inlined: Option<InlinedFrame>,
    /// For a later activation of the root frame (see [`RootActivation`]):
    /// what replaced it. `None` for the root frame itself and inlined frames.
    pub activation: Option<RootActivation>,
}

impl TraceFrameInfo {
    /// The root frame of a trace: a real interpreter frame.
    #[must_use]
    pub const fn root(register_count: u16, code_len: usize) -> Self {
        Self {
            register_count,
            code_len,
            inlined: None,
            activation: None,
        }
    }

    /// Whether the frame is the interpreter's root frame: the one the trace
    /// was entered on, or a later activation of it.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.inlined.is_none()
    }
}

// Keep serde support local to the IR crate instead of requiring it of the
// compatibility crate's interpreter helpers.
#[derive(Serialize, Deserialize)]
#[serde(remote = "FloatModulo")]
enum PersistedFloatModulo {
    Floor,
    Fmod53,
    Fmod54,
}

/// Binary arithmetic on two operands of one numeric type.
///
/// `Divide` and `Power` take floats: the recorder converts integer operands
/// with [`TraceOp::IntToFloat`] first, as the interpreter does. `Modulo` and
/// `FloorDivide` take either type. Float modulo is `a - floor(a / b) * b`,
/// float floor division is `floor(a / b)`. Integer modulo and floor division
/// round toward negative infinity and wrap `MIN / -1`. They are only
/// meaningful for a non-zero divisor: the recorder guards the divisor before
/// the operation so the interpreter raises the error, and every consumer
/// defines a zero divisor as a zero result so that an operation the optimizer
/// moved ahead of its guard still cannot trap. The bitwise operations and
/// shifts take integers only, with Lua's shift semantics (a distance of 64
/// or more clears the value, a negative distance shifts the other way, and
/// right shifts are logical).
///
/// On [`TraceType::Int64`] and [`TraceType::UInt64`] operands (both of one
/// type) the operations are `LuaJIT`'s: add, subtract and multiply wrap,
/// `Divide` and `Modulo` truncate toward zero and give `INT64_MIN` (signed) or
/// `2^63` (unsigned) for a zero divisor, and `MIN / -1` wraps. The
/// bitwise operations work on the bits. The shifts and rotates are the `bit`
/// library's: they use the low six bits of the count, `ShiftRight` is logical
/// whatever the type, and `ArithmeticShiftRight` copies the sign bit. A
/// float or integer operand never takes `FloorDivide`, `Power`,
/// `ArithmeticShiftRight`, `RotateLeft` or `RotateRight`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    /// Float modulo as one profile spells it. `Modulo` is the `LuaJIT`
    /// form, `FloatModulo::Floor`; integer operands take `Modulo`.
    ModuloWith(#[serde(with = "PersistedFloatModulo")] FloatModulo),
    /// Float power with Lua 5.4's `a ^ 2` as `a * a`. Float operands only.
    PowerSquare,
    FloorDivide,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    /// 64-bit integer cdata only.
    ArithmeticShiftRight,
    /// 64-bit integer cdata only.
    RotateLeft,
    /// 64-bit integer cdata only.
    RotateRight,
}

impl ArithmeticOp {
    /// Whether integer operands are legal. Integer add, subtract and multiply
    /// wrap.
    #[must_use]
    pub const fn allows_integers(self) -> bool {
        !matches!(
            self,
            Self::Divide
                | Self::Power
                | Self::ModuloWith(_)
                | Self::PowerSquare
                | Self::ArithmeticShiftRight
                | Self::RotateLeft
                | Self::RotateRight
        )
    }

    /// Whether float operands are legal.
    #[must_use]
    pub const fn allows_floats(self) -> bool {
        !matches!(
            self,
            Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::ShiftLeft
                | Self::ShiftRight
                | Self::ArithmeticShiftRight
                | Self::RotateLeft
                | Self::RotateRight
        )
    }

    /// Whether 64-bit integer cdata operands are legal.
    #[must_use]
    pub const fn allows_wide(self) -> bool {
        !matches!(
            self,
            Self::Power | Self::PowerSquare | Self::ModuloWith(_) | Self::FloorDivide
        )
    }
}

/// A pure operation of one number that the inline library functions
/// (`math.floor`, `math.abs`, ...) compile to. None of them can fail or trap.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum NumericOp {
    /// The largest integral float not above the float.
    Floor,
    /// The smallest integral float not below the float.
    Ceil,
    /// The absolute value of an integer (`MIN` wraps to itself) or a float
    /// (the sign bit cleared, so NaN payloads survive).
    Abs,
    /// A float as an integer, truncating toward zero and saturating: NaN is
    /// zero. The recorder guards that the float is integral and in range
    /// first, so the saturation is not observable.
    FloatToInteger,
    /// A float as the low 32 bits `LuaJIT`'s `bit` library reads, as an
    /// integer in `0..2^32`: add `1.5 * 2^52` so the float rounds to an
    /// integer, ties to even, in the low mantissa bits, and keep the low word
    /// of the sum. Defined for every float (NaN and the infinities give 0),
    /// with the same result in every consumer. For a float of magnitude
    /// below `2^51` it is the float rounded to an integer, modulo `2^32`.
    TrickBits,
}

impl NumericOp {
    /// The type the operation gives for an operand of type `source`, or
    /// `None` when it takes no such operand.
    #[must_use]
    pub const fn result_type(self, source: TraceType) -> Option<TraceType> {
        match (self, source) {
            (Self::Floor | Self::Ceil, TraceType::Float) => Some(TraceType::Float),
            (Self::Abs, TraceType::Float) => Some(TraceType::Float),
            (Self::Abs, TraceType::Integer) => Some(TraceType::Integer),
            (Self::FloatToInteger | Self::TrickBits, TraceType::Float) => Some(TraceType::Integer),
            _ => None,
        }
    }
}

/// How [`TraceOp::ToWide`] converts a float.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WideConversion {
    /// C's conversion to the target type: truncation, with `INT64_MIN` bits
    /// for NaN and values out of range (unsigned targets wrap values from
    /// `2^63` first). A 64-bit integer converts to the other signedness by
    /// keeping its bits.
    Truncate,
    /// `bit` library operand conversion: round to nearest, ties to even,
    /// keep the low 32 bits (zero for a non-finite float). The result is an
    /// `Int64` below `2^32`.
    Bits32,
}

/// A pure native function of one float, such as `math.sin`. It must give the
/// same result as the Lua function it stands for when that function gets one
/// float, never touch the runtime and never fail. Equality and hashing use
/// the address.
///
/// Serialization writes no address. Deserialization installs a placeholder
/// returning NaN, which the VM must replace using its native relocation data
/// before compiling a loaded trace.
#[derive(Clone, Copy)]
pub struct NativeUnary(pub extern "C" fn(f64) -> f64);

/// A pure native function of two floats, such as `math.max`. The same rules
/// as [`NativeUnary`] apply.
#[derive(Clone, Copy)]
pub struct NativeBinary(pub extern "C" fn(f64, f64) -> f64);

macro_rules! native_function {
    ($name:ident) => {
        impl $name {
            /// The address of the function, which generated code calls.
            #[must_use]
            pub fn address(self) -> usize {
                self.0 as usize
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($name), "({:#x})"), self.address())
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.address() == other.address()
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.address().hash(state);
            }
        }
    };
}

native_function!(NativeUnary);
native_function!(NativeBinary);

extern "C" fn unresolved_unary(_value: f64) -> f64 {
    f64::NAN
}

extern "C" fn unresolved_binary(_left: f64, _right: f64) -> f64 {
    f64::NAN
}

macro_rules! native_serialization {
    ($name:ident, $placeholder:ident) => {
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_unit()
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                <()>::deserialize(deserializer)?;
                Ok(Self($placeholder))
            }
        }
    };
}

native_serialization!(NativeUnary, unresolved_unary);
native_serialization!(NativeBinary, unresolved_binary);

/// The identity a trace pins for the native function with registry index
/// `index`, in [`TraceIr::pinned_closures`]. A closure's identity has its
/// arena slot in the low 32 bits, and the runtime never hands out slot
/// `u32::MAX`, so that slot marks a native function and the generation field
/// holds its index.
#[must_use]
pub const fn pinned_native_identity(index: u32) -> u64 {
    ((index as u64) << 32) | 0xffff_ffff
}

/// The builtin a trace may pin: the iterator of `ipairs`.
pub const PINNED_BUILTIN_IPAIRS_ITERATOR: u32 = 0;
/// The builtin `tostring`, which a trace calls through a helper
/// ([`HelperId::TOSTRING`](super::HelperId::TOSTRING)).
pub const PINNED_BUILTIN_TOSTRING: u32 = 1;
/// The builtin `string.format` of the modern profiles, which a trace calls
/// through a helper
/// ([`HelperId::STRING_FORMAT`](super::HelperId::STRING_FORMAT)).
pub const PINNED_BUILTIN_STRING_FORMAT: u32 = 2;
/// The builtin `pcall`, which a trace inlines as a protected frame.
pub const PINNED_BUILTIN_PCALL: u32 = 3;
/// The builtin `xpcall`, which a trace inlines as a protected frame.
pub const PINNED_BUILTIN_XPCALL: u32 = 4;
/// The builtin `debug.traceback`, which a trace only carries as an `xpcall`
/// handler: an error raised under it exits and the interpreter runs it.
pub const PINNED_BUILTIN_DEBUG_TRACEBACK: u32 = 5;
/// The builtin `string.gsub`, which a trace calls through a helper
/// ([`HelperId::STRING_GSUB`](super::HelperId::STRING_GSUB)).
pub const PINNED_BUILTIN_STRING_GSUB: u32 = 6;
/// The builtin `next`, which a trace calls through a helper
/// ([`HelperId::next`](super::HelperId::next)).
pub const PINNED_BUILTIN_NEXT: u32 = 7;
/// The builtin `pairs`, which a trace records as the `next` loop it starts.
pub const PINNED_BUILTIN_PAIRS: u32 = 8;
/// The builtin `ipairs`, which a trace records as the loop it starts.
pub const PINNED_BUILTIN_IPAIRS: u32 = 9;
/// The builtin `coroutine.resume`, which a trace calls through a helper
/// ([`HelperId::coroutine_resume`](super::HelperId::coroutine_resume)).
pub const PINNED_BUILTIN_COROUTINE_RESUME: u32 = 10;
/// The builtin `coroutine.yield`, which a trace never calls: a loop that
/// reaches it is left to the interpreter, but reading it as a callee keeps
/// the recording on its way to that call.
pub const PINNED_BUILTIN_COROUTINE_YIELD: u32 = 11;

/// The identity a trace pins for a builtin function (see
/// [`PINNED_BUILTIN_IPAIRS_ITERATOR`]), in [`TraceIr::pinned_closures`]. Like
/// a native function's it uses an arena slot the runtime never hands out,
/// `u32::MAX - 1`, and the generation field holds the builtin.
#[must_use]
pub const fn pinned_builtin_identity(builtin: u32) -> u64 {
    ((builtin as u64) << 32) | 0xffff_fffe
}

/// The builtin behind a pinned identity, or `None` for anything else.
#[must_use]
pub const fn pinned_builtin_kind(identity: u64) -> Option<u32> {
    if identity & 0xffff_ffff == 0xffff_fffe {
        Some((identity >> 32) as u32)
    } else {
        None
    }
}

/// The native function index behind a pinned identity, or `None` for a
/// closure.
#[must_use]
pub const fn pinned_native_index(identity: u64) -> Option<u32> {
    if identity & 0xffff_ffff == 0xffff_ffff {
        Some((identity >> 32) as u32)
    } else {
        None
    }
}

/// What the VM resolves about one table when it enters a trace.
///
/// Native code never walks a table. At entry the VM looks at the table in
/// the entry register and hands the code one word pair per binding: the
/// array part's address and length, the address of a field's slot, or the
/// table's length. The table must be an entry input that no iteration
/// writes, so every iteration sees the same table. Entry refuses a frame
/// whose table has no such storage or whose field has another type than the
/// binding says, which counts as an entry mismatch.
///
/// The addresses stay valid for the whole run because nothing in a run
/// mutates table storage except overwriting a payload with a value of the
/// same type (see [`TraceOp::ArrayStore`]). The exceptions are
/// [`TraceOp::ArraySet`] and the helpers that write tables, which may resize
/// a table: after one, native code reads the words of every array and length
/// binding again (the VM rewrites them), and a tree that holds one that can
/// also move a hash part binds no storage that a rehash could move (no field
/// or lane access, see [`TraceIr::moves_hash_storage`]).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TableBinding {
    /// The root window register holding the table, an entry input of type
    /// `Table`.
    pub register: u16,
    pub kind: BindingKind,
}

/// What a [`TableBinding`] resolves.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BindingKind {
    /// The array part: address of slot 1 and slot count.
    ///
    /// `first` is the type slot 1 held when the trace was recorded, a hint
    /// for choosing between versions: arrays are not homogeneous in general,
    /// so each access still guards its own slot, but a loop is usually run
    /// over arrays of one element type (integers, then floats), and a trace
    /// that guards the other type at every access is worse than none. Entry
    /// refuses an array whose slot 1 has another native type, or is nil, or
    /// does not exist, so another version (or the interpreter) takes it. It
    /// does not take part in telling bindings apart: one table register has
    /// one array binding.
    Array { first: Option<TraceType> },
    /// The slot of the field named `key`, a string the recorder copied from
    /// the chunk constant. Entry requires a present value of type `ty`
    /// (`Integer`, `Float` or `Boolean`). The key is looked up by content, so
    /// a table whose key string was allocated elsewhere still matches.
    ///
    /// `depth` counts the table-valued `__index` hops from the bound table to
    /// the one that holds the field (0 when the table holds it). Entry
    /// follows exactly that many hops, through tables that have no value for
    /// the key, and refuses anything else. Only a read can go through a hop,
    /// since a store to an absent key would land in the bound table.
    Field {
        key: Arc<[u8]>,
        ty: TraceType,
        depth: u8,
    },
    /// The table's length, as a value of type `ty` (`Integer` or `Float`).
    /// Entry refuses a table with a `__len` metamethod, unless `raw` says the
    /// trace wants the border whatever `__len` does (`rawlen`).
    Length { ty: TraceType, raw: bool },
    /// The field named `key` holds exactly the callable that pinned lane
    /// `lane` stands for (a Lua closure or a native function). It resolves
    /// no storage: entry checks the field and refuses a table whose field
    /// holds anything else. Nothing a trace does can rewrite such a field, so
    /// the recorder reads it as that lane's constant. `depth` counts `__index`
    /// hops as for [`Self::Field`].
    Callee {
        key: Arc<[u8]>,
        lane: u32,
        depth: u8,
    },
    /// The field named `key` holds a string with exactly the bytes `text`.
    /// Like [`Self::Callee`] it resolves no storage: entry checks the field
    /// and refuses a table whose field holds anything else, and the recorder
    /// reads the field as the string constant (`utf8.charpattern`).
    StringField {
        key: Arc<[u8]>,
        text: Arc<[u8]>,
        depth: u8,
    },
    /// The metatable class a [`TraceOp::MetaGuard`] compares tables with.
    /// Entry requires the table with identity `metatable` to be alive and each
    /// pin to hold in it: the raw value under the event is the Lua closure of
    /// the pinned identity, or is absent when the identity is
    /// [`META_PIN_ABSENT`]. `bare` lists the operand types the trace reads as
    /// having no handler (see [`META_BARE_NUMBERS`] and its siblings), each of
    /// which entry checks against the runtime's metatable for the type. Entry
    /// refuses any other state, so the interpreter runs. The binding resolves
    /// no storage and names the table of its register only to have an entry
    /// input: the class is the pinned metatable, not that table's.
    ///
    /// Nothing a trace does can change what entry checked: a store under a
    /// key that starts with `__` misses in the run's store helpers
    /// (`PinnedKeys` of the VM). A call that changes a table's metatable
    /// ([`HelperEffects::WRITES_METATABLE`]) makes that table's row again, so
    /// a [`TraceOp::MetaGuard`] after it sees the new metatable, and no guard
    /// before it answers for one after it (guard elimination forgets).
    MetaClass {
        metatable: u64,
        pins: Arc<[MetaPin]>,
        bare: u8,
    },
    /// The bytes of a string: address of byte 0 and byte count. The binding's
    /// register holds a `String` entry input, which no iteration writes. The
    /// string is immutable and the run's root bank keeps it alive, so the
    /// address stays valid for the run. Unlike the table kinds this one does
    /// not belong to a table. `ty` (`Integer` or `Float`) is the number type
    /// of the profile: the type of a byte and of a length.
    Bytes { ty: TraceType },
    /// The table has no metatable. It resolves no storage: entry refuses a
    /// table with a metatable. A trace that reads nil out of a hole of the
    /// table's array part ([`TraceOp::ArrayLoad`] expecting `Nil`) needs it,
    /// because only then is nil the value of the access: no `__index` can
    /// run. Nothing a trace does can give a table a metatable.
    NoMetatable,
    /// A table whose hash part a [`TraceOp::HashLoad`] reads for a key the
    /// recording found present. It resolves no storage and entry checks
    /// nothing: the table may have a metatable, since the lookup's guard
    /// leaves unless the value found has the expected non-nil type, so a
    /// missing key (which would run `__index`) never reads as nil.
    Hash,
    /// The table's header: the words are whether the table is a dense list
    /// native code may append to and pop from in place, and the table's
    /// address (see `oximond::runtime::NativeTableHeader`). The recorder binds
    /// it beside the array binding of every table a trace sets, which lets an
    /// [`TraceOp::ArraySet`] whose key is the slot after the last, or the
    /// last, do the append or the pop without calling the VM. Entry refuses a
    /// run in which another binding of a different register names the same
    /// table, since the set updates the words of the bindings of its own
    /// register only. The VM rewrites the first word after any operation it
    /// does that may change the answer.
    ///
    /// With `strict`, the sets of the table never call the VM: the recording
    /// saw the table as a list that takes them in place (`table.insert` and
    /// `table.remove` do), and a set that finds anything else leaves through
    /// its guard. Such a set moves no hash storage
    /// ([`TraceIr::moves_hash_storage`]).
    Header { strict: bool },
}

impl TableBinding {
    /// Whether this binding resolves an array part.
    #[must_use]
    pub const fn is_array(&self) -> bool {
        matches!(self.kind, BindingKind::Array { .. })
    }
}

/// The identity [`MetaPin::handler`] has for an event the metatable lacks. No
/// closure has it (the identity of a handle is an index and a generation, and
/// the first closure of a runtime has identity zero).
pub const META_PIN_ABSENT: u64 = u64::MAX;

/// The word a table's row holds for the metatable with `identity`
/// ([`META_FIELD_KEY`]): one more than the identity, so that the zero of a
/// table with no metatable (or a row that cannot say) is never a metatable's
/// word, whichever identity it has.
#[must_use]
pub const fn meta_word(identity: u64) -> u64 {
    identity.wrapping_add(1)
}

/// [`BindingKind::MetaClass::bare`] bit: numbers have no metatable.
pub const META_BARE_NUMBERS: u8 = 1;
/// [`BindingKind::MetaClass::bare`] bit: nil and booleans have no metatable.
pub const META_BARE_SIMPLE: u8 = 2;
/// [`BindingKind::MetaClass::bare`] bit: the string metatable lacks every
/// event of the binding's pins.
pub const META_BARE_STRINGS: u8 = 4;

/// The name of the field of a table row that holds the word of the table's
/// metatable ([`meta_word`]; zero for none, and for a table the row cannot
/// serve), which [`TraceOp::MetaGuard`] compares. The key starts with a NUL byte, so
/// no guest field read can share it by accident, and it is listed among
/// [`TraceIr::fields`] like any field, with the type `Integer`.
pub const META_FIELD_KEY: &[u8] = b"\0metatable";

/// The name of the field of a table row that holds the table's address, the
/// word [`TraceOp::LaneArrayLoad`] reads the array part's words from (see
/// `oximond::runtime::NativeTableLayout`). Zero for a table the row cannot
/// serve. Like [`META_FIELD_KEY`] it starts with a NUL byte, so no guest field
/// shares it, and it is listed among [`TraceIr::fields`] with the type
/// `Integer`. The word names the table, not a slot, so a move of the table's
/// hash or array storage leaves it valid: a table's address is stable for a
/// run (see `oximond::runtime::NativeTableHeader::address`).
pub const TABLE_ADDRESS_KEY: &[u8] = b"\0table";

/// One event of a [`BindingKind::MetaClass`]: the key of the metatable's
/// field (`__index`) and what it holds.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetaPin {
    pub event: Arc<[u8]>,
    /// The identity of the Lua closure the field holds, or
    /// [`META_PIN_ABSENT`] for a field that is nil or missing.
    pub handler: u64,
}

/// A field a trace reads or writes on tables it only meets while running,
/// such as one it loaded out of an array: the string key (copied from the
/// chunk constant) and the type of value the access needs.
///
/// A trace lists its fields once, and an access names one by position (see
/// [`TraceOp::TableFieldLoad`]). The VM keeps a row for every table of a run
/// with one slot address per listed field: the address of the present value
/// of that field if it has the field's type, and nothing otherwise. A tree
/// shares one list, like its table bindings.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TableField {
    pub key: Arc<[u8]>,
    /// `Integer`, `Float` or `Boolean`. A load (never a store) also takes
    /// `Table`, whose present value is read as a lane of the run's bank, and
    /// `Nil`, whose row holds a marker address when the table has no such
    /// key and no metatable, so the read is nil.
    pub ty: TraceType,
}

/// The lane base of a string constant of the trace's root bank. A
/// `Constant(String(lane))` with `lane >= ROOT_STRING_LANE_BASE` names root
/// slot `lane - ROOT_STRING_LANE_BASE` ([`TraceIr::root_constants`]). Only a
/// [`TraceOp::HostCall`] or a [`TraceOp::CallHelper`] reads one; the VM resolves it from the trace's roots,
/// not from the run's bank, so a snapshot never names it.
pub const ROOT_STRING_LANE_BASE: u64 = 1 << 32;

/// The root slot a string constant's `lane` names, or `None` for a lane of
/// the run's bank.
#[must_use]
pub fn root_string_slot(lane: u64) -> Option<usize> {
    usize::try_from(lane.checked_sub(ROOT_STRING_LANE_BASE)?).ok()
}

/// Whether a call of `helper` with `arg_count` arguments, whose pattern is
/// the constant `pattern`, runs a prepared pattern. A `string.find` with a
/// plain-flag argument is the native's. The plain-text find helpers go through
/// the matcher only for a pattern with a magic byte.
#[must_use]
pub fn find_uses_prepared_pattern(helper: HelperId, pattern: &[u8], arg_count: usize) -> bool {
    if !helper.is_find_call() {
        return true;
    }
    arg_count <= 3
        && (!matches!(helper, HelperId::STRING_FIND | HelperId::STRING_FIND_ABSENT)
            || pattern.iter().any(|byte| b"^$*+?.([%-".contains(byte)))
}

/// The root-string lane that every `string.gmatch` call of `irs` (the root of
/// a tree and its sides) names as its pattern, or `None` when there is none,
/// a pattern is not a root-string constant, or the calls name different ones.
/// A step of an iterator that entered a trace is guessed to come from it.
#[must_use]
pub fn gmatch_constant_lane<'a>(irs: impl IntoIterator<Item = &'a TraceIr>) -> Option<u64> {
    let mut found = None;
    for ir in irs {
        let mut lanes = std::collections::HashMap::new();
        for instruction in ir.preheader.iter().chain(&ir.body) {
            if let (Some(result), TraceOp::Constant(TraceScalar::String(lane))) =
                (instruction.result, &instruction.op)
            {
                lanes.insert(result, *lane);
            }
        }
        for instruction in ir.preheader.iter().chain(&ir.body) {
            let TraceOp::CallHelper { helper, args, .. } = &instruction.op else {
                continue;
            };
            if *helper != HelperId::GMATCH_NEW {
                continue;
            }
            let lane = *lanes.get(args.get(1)?)?;
            root_string_slot(lane)?;
            if found.is_some_and(|known| known != lane) {
                return None;
            }
            found = Some(lane);
        }
    }
    found
}

/// The most arguments a [`TraceOp::HostCall`] passes.
pub const MAX_HOST_ARGS: usize = 5;

/// A library function a trace calls through the VM instead of leaving the
/// trace ([`TraceOp::HostCall`]). Every one of them is pure: it reads its
/// arguments, allocates at most strings, and changes no guest state, so a
/// call that cannot be answered natively may leave the trace before the
/// instruction and let the interpreter make the call, which then raises
/// whatever error it raises.
///
/// A host call marshals its arguments as tagged words and dispatches on
/// [`Self::code`] inside the VM. It is for what needs the VM. Everything that
/// needs only the run's bank is a [`TraceOp::CallHelper`] (see
/// [`super::helper`]), which calls an `extern "C"` function with typed
/// register arguments: `string.len`, `string.char`, `tostring`,
/// `string.format`, `..`, `string.find` and `table.concat`. What remains a
/// host call, and why:
///
/// - [`Self::Native`] runs the registered callback of any native function.
///   It needs the whole runtime (the callback takes `&mut Runtime`) and may
///   redispatch to the VM, which no helper may. The recorder emits none today
///   (every library call it records has a helper), but a native without a
///   helper would take this form, and traces profiled by an earlier build
///   carry it.
/// - [`Self::StringLength`] of a string the trace holds by lane is no call at
///   all: the backend reads the length from the run's length table. A root or
///   borrowed string is a helper.
/// - [`Self::MathMax`] and [`Self::MathMin`] lower to a compare, not a call.
/// - [`Self::ConcatLength`] is `#(a .. b)` of a concatenation nothing else
///   reads: the sum of the operands' byte counts, with no string made.
/// - [`Self::ReleaseStrings`] is not a call of the guest's: it tells the VM
///   which string lanes to keep, which needs the run's bank and its
///   bookkeeping of what the run's helpers made.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum HostFunction {
    /// The native function with this registry index, run through the
    /// callback the interpreter runs.
    Native(u32),
    /// The byte count of a string, as a number of the profile.
    StringLength,
    /// `math.max` and `math.min` of two integers.
    MathMax,
    MathMin,
    /// Not a call: the code [`TraceOp::ReleaseStrings`] passes to the VM.
    ReleaseStrings,
    /// The byte count of the concatenation `..` of its two to five
    /// arguments (strings and numbers), as a number of the profile, which
    /// the optimizer takes in place of the string where nothing else reads
    /// it. It leaves where the concatenation would (an operand it cannot
    /// size, a result too long for a string).
    ConcatLength,
}

impl HostFunction {
    /// The function as the word native code passes to the VM.
    #[must_use]
    pub const fn code(self) -> i64 {
        match self {
            Self::Native(index) => index as i64,
            Self::StringLength => -4,
            Self::MathMax => -5,
            Self::MathMin => -6,
            Self::ReleaseStrings => -8,
            Self::ConcatLength => -9,
        }
    }

    /// The function a word names.
    #[must_use]
    pub const fn from_code(code: i64) -> Option<Self> {
        match code {
            0..=0xffff_ffff => Some(Self::Native(code as u32)),
            -4 => Some(Self::StringLength),
            -5 => Some(Self::MathMax),
            -6 => Some(Self::MathMin),
            -8 => Some(Self::ReleaseStrings),
            -9 => Some(Self::ConcatLength),
            _ => None,
        }
    }
}

/// One trace operation. Operands are earlier values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TraceOp {
    /// Defines a constant.
    Constant(TraceScalar),
    /// Copies a value. Copy elimination removes it.
    Move { source: ValueId },
    /// Negates a number.
    Negate { source: ValueId },
    /// Converts an integer or a 64-bit integer cdata to the nearest float, as
    /// `i as f64`.
    IntToFloat { source: ValueId },
    /// Converts a float, or a 64-bit integer of the other signedness, to
    /// `target` (`Int64` or `UInt64`) as `conversion` says.
    ToWide {
        source: ValueId,
        target: TraceType,
        conversion: WideConversion,
    },
    /// A pure operation of one number (see [`NumericOp`]).
    Numeric { op: NumericOp, source: ValueId },
    /// Calls a pure native function on a float. The result is a float.
    NativeUnary {
        function: NativeUnary,
        source: ValueId,
    },
    /// Calls a pure native function on two floats. The result is a float.
    NativeBinary {
        function: NativeBinary,
        left: ValueId,
        right: ValueId,
    },
    /// Arithmetic on two numbers of the same type.
    Arithmetic {
        op: ArithmeticOp,
        left: ValueId,
        right: ValueId,
    },
    /// Compares two values of the same type, or an integer with a float.
    /// Ordering needs numbers. An integer and a float compare by their exact
    /// mathematical values, with no lossy conversion. Two `Int64` or two
    /// `UInt64` values compare as signed or unsigned numbers. The result is
    /// a boolean.
    Compare {
        comparison: Comparison,
        left: ValueId,
        right: ValueId,
    },
    /// `if_true` when the boolean `condition` is true, else `if_false`: the
    /// join of a pure two-arm branch whose arms the recorder both recorded
    /// (a value select). Both arms have the same type, which is the result's:
    /// an integer, a float or a boolean. It has no guard and never leaves; the
    /// arms' operations run whichever way the condition goes, so each of them
    /// is total on the types the trace guards.
    Select {
        condition: ValueId,
        if_true: ValueId,
        if_false: ValueId,
    },
    /// Lua truthiness of a value, as a boolean.
    Truthy { source: ValueId },
    /// Negated Lua truthiness, as a boolean.
    Not { source: ValueId },
    /// Leaves through the guard's snapshot unless `source` has dynamic type
    /// `expected`. Defines a new value narrowed to `expected`.
    GuardType {
        source: ValueId,
        expected: TraceType,
        guard: GuardId,
    },
    /// Leaves through the guard's snapshot unless the boolean `condition`
    /// equals `expected`.
    GuardBranch {
        condition: ValueId,
        expected: bool,
        guard: GuardId,
    },
    /// Reads element `key` of the array part of the table `table`, whose
    /// storage the entry bound as binding `binding` ([`BindingKind::Array`]).
    /// Leaves through the guard's snapshot unless `key` is an integral number
    /// inside the array and the slot holds a non-nil value of type
    /// `expected`. The snapshot is a `Before` one, so the interpreter runs
    /// the instruction itself: it handles a nil slot, a key in the hash
    /// part, a fractional key and `__index`. The result has type `expected`,
    /// which may be `Table`: the VM then gives the loaded table a lane of the
    /// run's root bank, and the load leaves too when it has no lane to give.
    /// It is a read: it orders against stores and is never merged.
    ///
    /// With `expected` `Nil` the slot must lie inside the array and hold
    /// nil, and the table must have a [`BindingKind::NoMetatable`] binding,
    /// so the nil is what the interpreter would read.
    ///
    /// With `past_end` too (only with `expected` `Nil`) the key may be any
    /// number outside the array part (past its end, below 1, fractional or
    /// NaN) when the table's [`BindingKind::Header`] binding says the table
    /// is solid (see `oximond::runtime::NativeTableHeader`): its hash part is
    /// empty, so no such key is present. The load leaves through its guard
    /// when the table is not solid.
    ///
    /// With `expected` [`TraceType::StringRef`] the slot must hold a string
    /// and the result borrows it: the address of the slot, with no lane and
    /// no refcount.
    ArrayLoad {
        binding: u32,
        table: ValueId,
        key: ValueId,
        expected: TraceType,
        past_end: bool,
        guard: GuardId,
    },
    /// Reads element `key` of the array part of `table` like an
    /// [`Self::ArrayLoad`] of a number, whichever subtype the slot holds: the
    /// result is a float with the value of the number, and
    /// [`Self::NumberTag`] of it says whether the slot held an integer. The
    /// guard leaves unless the slot is inside the array and holds a float or
    /// an integer of magnitude at most 2^53 (so the float is exact and
    /// converts back to the same integer). The snapshot is a `Before` one.
    NumberLoad {
        binding: u32,
        table: ValueId,
        key: ValueId,
        guard: GuardId,
    },
    /// [`Self::ArrayLoad`] on a table met by lane (`table` is a lane of the
    /// run's bank, of any origin), with no entry binding: native code reads
    /// the table's address from field `field` of the lane's row (the field
    /// [`TABLE_ADDRESS_KEY`]) and the array part's address and length from
    /// the table itself, the way `LuaJIT`'s `FLOAD tab.array` does. Those words
    /// are read again after anything that may move an array
    /// ([`Self::ArraySet`], a helper that rewrites arrays), as an array
    /// binding's are.
    ///
    /// Leaves through the guard's `Before` snapshot unless the row names the
    /// table, `key` is an integral number inside the array and the slot holds
    /// a non-nil value of type `expected` (`Integer`, `Float`, `Boolean` or
    /// `Table`), so no `__index` can run and the interpreter does every other
    /// access. A `Table` result is the loaded table's lane, as for
    /// [`Self::ArrayLoad`]. It is a read: it orders against stores and is
    /// never merged with a store.
    LaneArrayLoad {
        table: ValueId,
        field: u32,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
    },
    /// The store of [`Self::LaneArrayLoad`]: overwrites element `key` of
    /// the array part of `table`, a table met by lane, with `value`, an
    /// integer, a float or a boolean. Native code finds the array the way
    /// the load does, through the address word `field` of the lane's row
    /// (zero for a read-only table). The guard leaves through its `Before`
    /// snapshot unless the row names the table, `key` names a slot inside
    /// the array and the slot holds a non-nil value of `value`'s type, so the
    /// store keeps the slot's type, the table's border and its GC
    /// reachability (a number or a boolean needs no write barrier), and
    /// `__newindex` could not have run: Lua runs it only for an absent key.
    /// Nothing moves, so the rows and the array words the run holds stay
    /// right. When the guard passes the store is committed, once, and never
    /// replayed. The store of `region.solids[idx] = bor (region.solids[idx],
    /// mask)` in Mineclonia's `mcl_util/shape.lua`, whose table the trace
    /// meets through the field `solids`.
    LaneArrayStore {
        table: ValueId,
        field: u32,
        key: ValueId,
        value: ValueId,
        effect: EffectId,
        guard: GuardId,
    },
    /// Whether the number loaded by the [`Self::NumberLoad`] `load` is an
    /// integer. A boolean.
    NumberTag { load: ValueId },
    /// The conjunction of two booleans.
    BoolAnd { left: ValueId, right: ValueId },
    /// The modulo of two numbers given as floats (`left` and `right`). When
    /// `integer` is true both are integers of magnitude at most 2^53 and the
    /// right one is not zero: the result is the integer floor modulo, as a
    /// float. Otherwise it is `float`, the float modulo of the profile
    /// ([`ArithmeticOp::Modulo`] or [`ArithmeticOp::ModuloWith`]).
    NumberModulo {
        float: ArithmeticOp,
        left: ValueId,
        right: ValueId,
        integer: ValueId,
    },
    /// Overwrites element `key` of the array part of `table` with `value`.
    /// The guard leaves through its `Before` snapshot unless `key` names a
    /// slot inside the array that holds a non-nil value of `value`'s type, so
    /// the store keeps the slot's type, the table's border and its GC
    /// reachability, and `__newindex` could not have run. The guard does not
    /// look at the table's read-only flag: entry refuses a read-only table
    /// for a binding any member of the tree stores into, so the interpreter
    /// raises the error. When the guard passes the store is committed, once,
    /// and never replayed.
    ArrayStore {
        binding: u32,
        table: ValueId,
        key: ValueId,
        value: ValueId,
        effect: EffectId,
        guard: GuardId,
    },
    /// [`Self::ArrayStore`] with no guard: an earlier access in the same
    /// segment (an [`Self::ArrayLoad`] that found a value of `value`'s type,
    /// or an [`Self::ArrayStore`] or overwrite of such a value) reached the
    /// same slot of the same binding, and nothing between them may move,
    /// grow, shrink or retype an array ([`Self::keeps_array_slots`]). So the
    /// slot is still inside the array and holds a non-nil value of `value`'s
    /// type, and the store cannot fail: it overwrites the payload, as the
    /// guarded form does, and takes no snapshot. The verifier checks the
    /// proof (see `verify::overwrite_proven`). `__newindex` never runs for a
    /// key whose raw value is not nil, so no metatable matters.
    ///
    /// The recorder emits it for the in-place stores of Mineclonia's ULL
    /// helpers (`a[1] = tonumber (band (value, 0xffffffffull))` after `a[1]`
    /// was read), whose snapshots filled `maxsnap` (see the slot-fact notes
    /// in the recorder).
    ArrayOverwrite {
        binding: u32,
        table: ValueId,
        key: ValueId,
        value: ValueId,
        effect: EffectId,
    },
    /// Sets key `key` (any number) of the bound table `table` to `value` (a
    /// number, a boolean, a string, a table or nil) as a raw store, the way
    /// the interpreter stores into a table that has no `__newindex` for the
    /// key. A slot inside the array that holds a value of `value`'s type is
    /// overwritten in place, as [`Self::ArrayStore`] does (entry has refused a
    /// read-only table, as for that op). Every other store (a nil hole, the
    /// slot past the end, a key of the hash part, a nil value) is done by the
    /// VM, which may grow or rehash the table, so native code reads every
    /// array binding's words again after it. The guard leaves through its
    /// `Before` snapshot, having stored nothing, when the VM cannot store
    /// raw: the key is NaN, the table is read-only, or the key is absent and
    /// the metatable of the table has `__newindex`. The interpreter then runs
    /// the instruction. When the guard passes the store is committed, once,
    /// and never replayed.
    ///
    /// A table with a header binding ([`BindingKind::Header`]) that is a dense
    /// list takes three stores without the VM: the key one past the last slot
    /// appends the value (growing the array part in place when it is full),
    /// nil under the last key pops it, and a key inside the array overwrites
    /// its slot. The code writes the array part, the border and the words of
    /// the bindings of the table itself. A string or table is given by the
    /// VM's store entry, which owns the reference count and the collector's
    /// write barrier. A pop leaves the key as a deleted key the table has not
    /// recorded yet (see `oximond::runtime::NativeTableLayout`).
    ArraySet {
        binding: u32,
        table: ValueId,
        key: ValueId,
        value: ValueId,
        effect: EffectId,
        guard: GuardId,
    },
    /// Looks up the string `key` in the hash part of `table`, a table the
    /// entry bound with a [`BindingKind::NoMetatable`] binding `binding`, so
    /// no `__index` can run and a missing key reads as nil. `key` is a lane
    /// string ([`TraceType::String`]) or a borrow ([`TraceType::StringRef`]);
    /// strings are not interned, so the lookup compares bytes.
    ///
    /// The result has type `expected` (`Integer`, `Float`, `Boolean` or
    /// `Nil`; strings, tables and other values are out of scope). The guard
    /// leaves through its `Before` snapshot, having changed nothing, unless
    /// the value found has that type, so the interpreter runs the
    /// instruction itself. A missing key and a stored nil both read as nil.
    ///
    /// The lookup is a call into the VM that is read-only, nonallocating,
    /// noncollecting and non-reentrant, so it ends no borrow
    /// ([`Self::ends_borrows`]). It is a read: it orders against stores and
    /// is never merged or moved. A dead one stays: its type guard carries
    /// the speculation that later constant folding rests on.
    HashLoad {
        binding: u32,
        table: ValueId,
        key: ValueId,
        expected: TraceType,
        guard: GuardId,
    },
    /// Reads the field the entry bound as binding `binding`
    /// ([`BindingKind::Field`]) of `table`. The slot's type was checked at
    /// entry and nothing in a run changes it, so the read cannot fail. The
    /// result has the binding's type. A read: it orders against stores.
    FieldLoad { binding: u32, table: ValueId },
    /// Overwrites the bound field with `value`, which has the binding's type.
    /// Cannot fail. A committed effect.
    FieldStore {
        binding: u32,
        table: ValueId,
        value: ValueId,
        effect: EffectId,
    },
    /// Reads field `field` ([`TraceIr::fields`]) of `table`, a table of any
    /// origin: an entry input, a loop parameter or a table an array read
    /// produced. Leaves through the guard's snapshot unless the table has the
    /// field as a present value of the field's type, so the `Before` snapshot
    /// makes the interpreter run the instruction (an absent key may run
    /// `__index`). The result has the field's type. A read: it orders against
    /// stores and is never merged.
    ///
    /// A field of type `Table` gives the table's lane: the load adds the
    /// table to the run's bank, and leaves through the guard when the run has
    /// no room. A field of type `Nil` is an absent key of a table without a
    /// metatable, and only appears in function traces, which store nothing.
    /// A loop trace may also read a `Table` field; it holds no store that
    /// moves hash storage then ([`TraceIr::storage_a_set_would_move`]), so a
    /// row never goes stale during a run.
    TableFieldLoad {
        field: u32,
        table: ValueId,
        guard: GuardId,
    },
    /// Overwrites field `field` of `table` with `value`, which has the
    /// field's type. The guard leaves through its `Before` snapshot unless the
    /// table has the field as a present value of that type, so the store
    /// keeps the slot's type and the table's keys, and `__newindex` could not
    /// have run. When the guard passes the store is committed, once, and
    /// never replayed.
    TableFieldStore {
        field: u32,
        table: ValueId,
        value: ValueId,
        effect: EffectId,
        guard: GuardId,
    },
    /// Leaves through the guard's snapshot unless the metatable of `table`,
    /// a table of any origin, is the one binding `binding`
    /// ([`BindingKind::MetaClass`]) pins. Reads the metatable's identity from
    /// the table's row (the field [`META_FIELD_KEY`]), so a table the row
    /// cannot serve (a read-only one) leaves too. The entry fixed what the
    /// pinned metatable holds for every event of the binding, which is why
    /// this one compare stands for a walk of the metatable at run time. A
    /// trace with this operation changes no metatable, so the answer is the
    /// same for the whole run.
    MetaGuard {
        table: ValueId,
        binding: u32,
        guard: GuardId,
    },
    /// The byte count of the string `string`, whose bytes the entry bound as
    /// binding `binding` ([`BindingKind::Bytes`]). Constant for the run. The
    /// result is an integer or a float, as the recorder's profile has `#s`.
    StringLength { binding: u32, string: ValueId },
    /// Byte `key` (counting from 1) of the string `string`, bound as binding
    /// `binding`, as a number: an integer or a float, as the profile's
    /// `string.byte` gives. Leaves through the guard's `Before` snapshot
    /// unless `key` is an integral number in `1..=#string`, so the
    /// interpreter runs the call itself for any other index. A read of
    /// immutable bytes, but never merged: it carries a guard.
    StringByte {
        binding: u32,
        string: ValueId,
        key: ValueId,
        guard: GuardId,
    },
    /// The length (border) of `table`, which the entry read as binding
    /// `binding` ([`BindingKind::Length`]). Only a store that may resize the
    /// table changes a border ([`TraceIr::sets_tables`]), and then the VM
    /// rewrites the binding's word, which native code reads again at every
    /// use. In a trace with no such store it is constant for the run. The
    /// result has the binding's type.
    TableLength { binding: u32, table: ValueId },
    /// One table constructor the trace ran without allocating its table (see
    /// [`super::snapshot::VirtualTable`]). It charges the collector the
    /// phantom debt of a table of `entries` entries, as
    /// `Runtime::charge_elided_table` does. Native code only counts: it adds
    /// one table and the footprint's bytes to the run's totals, and the VM
    /// charges the sums when the run leaves, at an exit or a poll. It touches
    /// no guest state and no heap, but it is a count the run owes, so a dead
    /// one stays and it never moves out of the loop body.
    PhantomTable { entries: u32 },
    /// Calls the trace's own function again: a direct self call of a function
    /// trace ([`TraceIr::function`]). `args` are the callee's parameters, one
    /// per [`FunctionTrace::parameters`] entry and of that type (a missing
    /// argument is a nil constant). Native code runs the callee as a nested
    /// activation of the same code, with the window slots beyond the guest
    /// registers (cells, globals) and the table bindings of the caller, which
    /// no function trace writes. The result is the value the callee returned,
    /// of type [`FunctionTrace::result`]; the operation has no result when the
    /// function returns no value.
    ///
    /// The guard leaves through its `Before` snapshot of the call, having run
    /// nothing, when the run is as deep as it may go: the interpreter makes
    /// the call itself. The same snapshot, resumed after the call with the
    /// callee's frame above it, is what this activation leaves behind when
    /// the callee leaves the trace (an unwind): the frame stands at the call,
    /// waiting for the callee's results.
    ///
    /// It commits no effect of its own (a function trace stores nothing), but
    /// it runs a whole activation, so it orders against every heap access, is
    /// never merged, moved or removed, and ends borrows.
    SelfCall { args: Vec<ValueId>, guard: GuardId },
    /// Calls `function` on `args` through the VM, which answers with one
    /// value of type `result`. The call is pure (see [`HostFunction`]): the
    /// guard leaves through its `Before` snapshot, having changed nothing,
    /// when the VM cannot answer (an error, a metamethod, a pattern, a
    /// result of another type than `result`, a run out of room for strings),
    /// so the interpreter makes the call itself.
    ///
    /// A string result is a lane of the run's root bank, which keeps it
    /// alive until the run ends. The call reads table storage, so it orders
    /// against stores, and it is never merged, moved or removed: a dead call
    /// could still have raised an error. It moves no table storage, so it
    /// leaves a borrowed string valid, and reads one as an argument.
    HostCall {
        function: HostFunction,
        args: Vec<ValueId>,
        result: TraceType,
        guard: GuardId,
    },
    /// Calls the helper `helper` ([`super::helper`]) on `args` with typed
    /// register arguments: a direct call of an `extern "C"` function, with no
    /// dispatch on a function code. The instruction's result is the helper's
    /// first result; a helper with more results has them read by
    /// [`Self::HelperResult`]. A helper with no result defines no value.
    ///
    /// A helper that can miss ([`HelperEffects::MAY_EXIT`]) reports it, and
    /// the guard leaves through its `Before` snapshot, having changed
    /// nothing, so the interpreter replays the call and raises whatever it
    /// raises. The call is never merged, moved or removed.
    ///
    /// The helper's effect flags say what it does to the heap. A helper that
    /// writes ([`HelperEffects::WRITES_HEAP`]) commits `effect`, numbered in
    /// body order like a store's, and invalidates every cached table read; a
    /// helper that only advances private state
    /// ([`HelperEffects::PRIVATE_STATE`]) commits one too and leaves cached
    /// reads valid; any other helper has none. A writing helper
    /// checks every precondition before its first write, so the guard still
    /// leaves before any change.
    CallHelper {
        helper: HelperId,
        args: Vec<ValueId>,
        effect: Option<EffectId>,
        guard: GuardId,
    },
    /// Result `index` (at least 1) of the call `call`, the value of an
    /// earlier [`Self::CallHelper`]. Reads the word the backend loaded when
    /// it made the call, so it commutes with everything and is removed when
    /// unused.
    HelperResult { call: ValueId, index: u8 },
    /// Ends the loop body of a trace that made strings through
    /// [`Self::HostCall`]: the VM drops every string lane the run made, except
    /// `keep` (the strings the next iteration, and an exit after the
    /// loop-back, can still name), and reuses the lanes. Without it a run
    /// holds every string it made until it ends. It is the last operation of
    /// the body of a loop trace, changes nothing a guest can see, and is
    /// never moved or removed.
    ReleaseStrings { keep: Vec<ValueId> },
    /// A symbolic heap read, such as a table load. It orders against effects
    /// but commits nothing, so a `Before` snapshot may replay it. The result
    /// is `Any` until a type guard narrows it.
    Read {
        label: u32,
        operands: Vec<ValueId>,
        result: TraceType,
    },
    /// A symbolic committed effect, such as a table store. Later packages
    /// replace it with real operations. A guard after it must resume after
    /// it, so the effect is never replayed. `label` names it in tests.
    Effect {
        effect: EffectId,
        label: u32,
        operands: Vec<ValueId>,
        /// The type of the value the effect produces, if any.
        result: Option<TraceType>,
    },
    /// The step of `next` along the array part of `table`, whose storage the
    /// entry bound as binding `binding` ([`BindingKind::Array`]): the
    /// one-based index of the first slot that holds a value after the slot
    /// `key` names, found by a scan over the nil slots between, as the
    /// interpreter's traversal does. With `start` the traversal begins and
    /// `key` is ignored (a constant zero): the first slot of the array part
    /// that holds a value. The result is a number of type `ty` (`Integer` or
    /// `Float`, the type the profile's `next` gives for a key of the array
    /// part), whatever the type of `key`.
    ///
    /// Leaves through the guard's `Before` snapshot, having changed nothing,
    /// unless a slot with a value follows inside the array part. Without
    /// `start` the key must also be an integral number from 1 to the length
    /// of the array part whose slot holds a value now: the interpreter
    /// continues from a key of the array part only then (a deleted key, a
    /// key of the hash part and a key that is no slot are its job). The
    /// element is read by an [`Self::ArrayLoad`] of the result.
    ///
    /// It is a read: it orders against stores and is never merged or moved.
    ArrayNext {
        binding: u32,
        table: ValueId,
        key: ValueId,
        start: bool,
        ty: TraceType,
        guard: GuardId,
    },
}

impl TraceOp {
    /// Every value operand, in order.
    pub fn operands(&self, mut visit: impl FnMut(ValueId)) {
        match self {
            Self::Constant(_) | Self::PhantomTable { .. } => {}
            Self::Move { source }
            | Self::Negate { source }
            | Self::Truthy { source }
            | Self::Not { source }
            | Self::IntToFloat { source }
            | Self::Numeric { source, .. }
            | Self::ToWide { source, .. }
            | Self::NativeUnary { source, .. }
            | Self::GuardType { source, .. }
            | Self::NumberTag { load: source }
            | Self::HelperResult { call: source, .. } => visit(*source),
            Self::Arithmetic { left, right, .. }
            | Self::Compare { left, right, .. }
            | Self::BoolAnd { left, right }
            | Self::NativeBinary { left, right, .. } => {
                visit(*left);
                visit(*right);
            }
            Self::NumberModulo {
                left,
                right,
                integer,
                ..
            } => {
                visit(*left);
                visit(*right);
                visit(*integer);
            }
            Self::Select {
                condition,
                if_true,
                if_false,
            } => {
                visit(*condition);
                visit(*if_true);
                visit(*if_false);
            }
            Self::GuardBranch { condition, .. } => visit(*condition),
            Self::Read { operands, .. }
            | Self::Effect { operands, .. }
            | Self::HostCall { args: operands, .. }
            | Self::CallHelper { args: operands, .. }
            | Self::ReleaseStrings { keep: operands }
            | Self::SelfCall { args: operands, .. } => {
                operands.iter().copied().for_each(visit);
            }
            Self::ArrayLoad { table, key, .. }
            | Self::LaneArrayLoad { table, key, .. }
            | Self::NumberLoad { table, key, .. }
            | Self::HashLoad { table, key, .. }
            | Self::ArrayNext { table, key, .. } => {
                visit(*table);
                visit(*key);
            }
            Self::ArrayStore {
                table, key, value, ..
            }
            | Self::ArrayOverwrite {
                table, key, value, ..
            }
            | Self::LaneArrayStore {
                table, key, value, ..
            }
            | Self::ArraySet {
                table, key, value, ..
            } => {
                visit(*table);
                visit(*key);
                visit(*value);
            }
            Self::FieldLoad { table, .. }
            | Self::TableLength { table, .. }
            | Self::MetaGuard { table, .. }
            | Self::TableFieldLoad { table, .. } => visit(*table),
            Self::StringLength { string, .. } => visit(*string),
            Self::StringByte { string, key, .. } => {
                visit(*string);
                visit(*key);
            }
            Self::FieldStore { table, value, .. } | Self::TableFieldStore { table, value, .. } => {
                visit(*table);
                visit(*value);
            }
        }
    }

    /// Every value operand, mutably.
    pub fn operands_mut(&mut self, mut visit: impl FnMut(&mut ValueId)) {
        match self {
            Self::Constant(_) | Self::PhantomTable { .. } => {}
            Self::Move { source }
            | Self::Negate { source }
            | Self::Truthy { source }
            | Self::Not { source }
            | Self::IntToFloat { source }
            | Self::Numeric { source, .. }
            | Self::ToWide { source, .. }
            | Self::NativeUnary { source, .. }
            | Self::GuardType { source, .. }
            | Self::NumberTag { load: source }
            | Self::HelperResult { call: source, .. } => visit(source),
            Self::Arithmetic { left, right, .. }
            | Self::Compare { left, right, .. }
            | Self::BoolAnd { left, right }
            | Self::NativeBinary { left, right, .. } => {
                visit(left);
                visit(right);
            }
            Self::NumberModulo {
                left,
                right,
                integer,
                ..
            } => {
                visit(left);
                visit(right);
                visit(integer);
            }
            Self::Select {
                condition,
                if_true,
                if_false,
            } => {
                visit(condition);
                visit(if_true);
                visit(if_false);
            }
            Self::GuardBranch { condition, .. } => visit(condition),
            Self::Read { operands, .. }
            | Self::Effect { operands, .. }
            | Self::HostCall { args: operands, .. }
            | Self::CallHelper { args: operands, .. }
            | Self::ReleaseStrings { keep: operands }
            | Self::SelfCall { args: operands, .. } => {
                operands.iter_mut().for_each(visit);
            }
            Self::ArrayLoad { table, key, .. }
            | Self::LaneArrayLoad { table, key, .. }
            | Self::NumberLoad { table, key, .. }
            | Self::HashLoad { table, key, .. }
            | Self::ArrayNext { table, key, .. } => {
                visit(table);
                visit(key);
            }
            Self::ArrayStore {
                table, key, value, ..
            }
            | Self::ArrayOverwrite {
                table, key, value, ..
            }
            | Self::LaneArrayStore {
                table, key, value, ..
            }
            | Self::ArraySet {
                table, key, value, ..
            } => {
                visit(table);
                visit(key);
                visit(value);
            }
            Self::FieldLoad { table, .. }
            | Self::TableLength { table, .. }
            | Self::MetaGuard { table, .. }
            | Self::TableFieldLoad { table, .. } => visit(table),
            Self::StringLength { string, .. } => visit(string),
            Self::StringByte { string, key, .. } => {
                visit(string);
                visit(key);
            }
            Self::FieldStore { table, value, .. } | Self::TableFieldStore { table, value, .. } => {
                visit(table);
                visit(value);
            }
        }
    }

    /// The guard this operation carries, if it is one.
    #[must_use]
    pub const fn guard(&self) -> Option<GuardId> {
        match self {
            Self::GuardType { guard, .. }
            | Self::GuardBranch { guard, .. }
            | Self::ArrayLoad { guard, .. }
            | Self::LaneArrayLoad { guard, .. }
            | Self::NumberLoad { guard, .. }
            | Self::ArrayNext { guard, .. }
            | Self::HashLoad { guard, .. }
            | Self::StringByte { guard, .. }
            | Self::ArrayStore { guard, .. }
            | Self::ArraySet { guard, .. }
            | Self::TableFieldLoad { guard, .. }
            | Self::TableFieldStore { guard, .. }
            | Self::LaneArrayStore { guard, .. }
            | Self::MetaGuard { guard, .. }
            | Self::HostCall { guard, .. }
            | Self::CallHelper { guard, .. }
            | Self::SelfCall { guard, .. } => Some(*guard),
            _ => None,
        }
    }

    /// The effect this operation commits, if it is a store.
    #[must_use]
    pub const fn committed_effect(&self) -> Option<EffectId> {
        match self {
            Self::Effect { effect, .. }
            | Self::ArrayStore { effect, .. }
            | Self::ArrayOverwrite { effect, .. }
            | Self::ArraySet { effect, .. }
            | Self::FieldStore { effect, .. }
            | Self::TableFieldStore { effect, .. }
            | Self::LaneArrayStore { effect, .. } => Some(*effect),
            Self::CallHelper { effect, .. } => *effect,
            _ => None,
        }
    }

    /// Whether the operation may end the lifetime of every borrowed value
    /// ([`TraceType::StringRef`]): it mutates, allocates, collects, re-enters
    /// the VM, or has an effect the IR cannot describe. A borrow points into
    /// a table slot, so a table store that may resize or replace the slot
    /// ([`Self::ArraySet`], which calls the VM) and every opaque
    /// [`Self::Read`] or [`Self::Effect`] end it.
    ///
    /// The ones that do not are exactly the nonallocating, noncollecting and
    /// non-reentrant operations: arithmetic and comparisons on scalars,
    /// guards, reads of immutable or in-place storage, and payload
    /// overwrites that keep a slot's type ([`Self::ArrayStore`],
    /// [`Self::FieldStore`], [`Self::TableFieldStore`]: their guards refuse a
    /// slot that is not a number or boolean, so a string slot is never
    /// touched). [`Self::ArrayLoad`] of a table adds a lane to the run's
    /// bank, which moves no table storage.
    ///
    /// The match has no wildcard on purpose: a new operation must say.
    #[must_use]
    pub fn ends_borrows(&self) -> bool {
        match self {
            Self::ArraySet { .. }
            | Self::Read { .. }
            | Self::Effect { .. }
            | Self::SelfCall { .. } => true,
            // A helper that writes may resize or replace a slot; one that
            // only reads or makes strings of the bank moves no storage.
            Self::CallHelper { helper, .. } => helper.effects().ends_borrows(),
            Self::HelperResult { .. } => false,
            Self::Constant(_)
            | Self::PhantomTable { .. }
            | Self::Move { .. }
            | Self::Negate { .. }
            | Self::IntToFloat { .. }
            | Self::Numeric { .. }
            | Self::ToWide { .. }
            | Self::NativeUnary { .. }
            | Self::NativeBinary { .. }
            | Self::Arithmetic { .. }
            | Self::Compare { .. }
            | Self::Truthy { .. }
            | Self::Not { .. }
            | Self::BoolAnd { .. }
            | Self::Select { .. }
            | Self::NumberModulo { .. }
            | Self::NumberTag { .. }
            | Self::GuardType { .. }
            | Self::GuardBranch { .. }
            | Self::ArrayLoad { .. }
            | Self::LaneArrayLoad { .. }
            | Self::NumberLoad { .. }
            | Self::ArrayNext { .. }
            | Self::HashLoad { .. }
            | Self::ArrayStore { .. }
            | Self::ArrayOverwrite { .. }
            | Self::FieldLoad { .. }
            | Self::FieldStore { .. }
            | Self::TableFieldLoad { .. }
            | Self::TableFieldStore { .. }
            | Self::LaneArrayStore { .. }
            | Self::MetaGuard { .. }
            | Self::TableLength { .. }
            | Self::StringLength { .. }
            | Self::StringByte { .. }
            | Self::ReleaseStrings { .. }
            | Self::HostCall { .. } => false,
        }
    }

    /// The table binding the operation names, if it names one.
    #[must_use]
    pub const fn binding(&self) -> Option<u32> {
        match self {
            Self::ArrayLoad { binding, .. }
            | Self::NumberLoad { binding, .. }
            | Self::ArrayStore { binding, .. }
            | Self::ArrayOverwrite { binding, .. }
            | Self::ArraySet { binding, .. }
            | Self::HashLoad { binding, .. }
            | Self::FieldLoad { binding, .. }
            | Self::FieldStore { binding, .. }
            | Self::MetaGuard { binding, .. }
            | Self::StringLength { binding, .. }
            | Self::StringByte { binding, .. }
            | Self::TableLength { binding, .. }
            | Self::ArrayNext { binding, .. } => Some(*binding),
            _ => None,
        }
    }

    /// The table binding the operation names, if it names one, to renumber.
    pub fn binding_mut(&mut self) -> Option<&mut u32> {
        match self {
            Self::ArrayLoad { binding, .. }
            | Self::NumberLoad { binding, .. }
            | Self::ArrayStore { binding, .. }
            | Self::ArrayOverwrite { binding, .. }
            | Self::ArraySet { binding, .. }
            | Self::HashLoad { binding, .. }
            | Self::FieldLoad { binding, .. }
            | Self::FieldStore { binding, .. }
            | Self::MetaGuard { binding, .. }
            | Self::StringLength { binding, .. }
            | Self::StringByte { binding, .. }
            | Self::TableLength { binding, .. }
            | Self::ArrayNext { binding, .. } => Some(binding),
            _ => None,
        }
    }

    /// Whether the operation reads or writes the heap, so it orders against
    /// stores and is never merged. Only the table reads can sit in a preheader.
    #[must_use]
    pub const fn touches_heap(&self) -> bool {
        matches!(
            self,
            Self::Read { .. }
                | Self::Effect { .. }
                | Self::ArrayLoad { .. }
                | Self::LaneArrayLoad { .. }
                | Self::NumberLoad { .. }
                | Self::ArrayNext { .. }
                | Self::HashLoad { .. }
                | Self::StringByte { .. }
                | Self::StringLength { .. }
                | Self::ArrayStore { .. }
                | Self::ArrayOverwrite { .. }
                | Self::ArraySet { .. }
                | Self::FieldLoad { .. }
                | Self::FieldStore { .. }
                | Self::TableFieldLoad { .. }
                | Self::TableFieldStore { .. }
                | Self::LaneArrayStore { .. }
                | Self::MetaGuard { .. }
                | Self::TableLength { .. }
                | Self::HostCall { .. }
                | Self::CallHelper { .. }
                | Self::ReleaseStrings { .. }
                | Self::SelfCall { .. }
        )
    }

    /// Whether every slot of every array part stays where it is, inside its
    /// array and with its type, across the operation: it does not set a key
    /// ([`Self::ArraySet`]), call the VM or a self call, run an opaque read
    /// or effect, or call a helper that writes the heap or runs a coroutine.
    /// An in-place overwrite keeps its slot's type, so the stores of
    /// [`Self::ArrayStore`], [`Self::ArrayOverwrite`], [`Self::FieldStore`],
    /// [`Self::TableFieldStore`] and [`Self::LaneArrayStore`] keep every
    /// slot. What
    /// [`Self::ArrayOverwrite`] rests on.
    #[must_use]
    pub fn keeps_array_slots(&self) -> bool {
        match self {
            Self::ArraySet { .. }
            | Self::Read { .. }
            | Self::Effect { .. }
            | Self::HostCall { .. }
            | Self::SelfCall { .. } => false,
            Self::CallHelper { helper, .. } => {
                let effects = helper.effects();
                !effects.contains(HelperEffects::WRITES_HEAP)
                    && !effects.contains(HelperEffects::RESUMES)
            }
            _ => true,
        }
    }
}

/// One instruction of the trace body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceInstruction {
    pub site: TraceSite,
    /// The value this instruction defines, if any.
    pub result: Option<ValueId>,
    pub op: TraceOp,
}

/// A guard and the snapshot it restores on failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GuardInfo {
    pub snapshot: SnapshotId,
    pub site: TraceSite,
    /// Names the guard across optimization, which renumbers guards. The
    /// recorder numbers guards in body order, so a recorded trace has
    /// `origin == id`. The optimizer keeps each surviving guard's origin, and
    /// a side trace names the guard it was recorded from by it (see
    /// [`TraceSide`]).
    pub origin: u32,
}

/// A root-frame register the trace reads on entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EntryInput {
    pub register: u16,
    /// The type the register must have. `Any` is not allowed.
    pub ty: TraceType,
    /// The value that carries the register's contents.
    pub value: ValueId,
}

/// What a trace requires of the interpreter state at entry. A mismatch exits
/// before any trace operation runs.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EntryContract {
    pub inputs: Vec<EntryInput>,
}

/// A register of an inlined frame that a side trace starts with: the side
/// continues a guard of its parent that failed inside inlined calls, and its
/// recording rebuilds those calls as frames `1..=n` of its own (see
/// [`TraceIr::seed_frames`]). The value is the one the parent's snapshot names
/// for that register, handed over like an entry input: its definition is
/// [`ValueDef::FrameInput`]. Frames above the root do not exist in the
/// interpreter, so there is no `Unchanged` to fall back on: every register of a
/// seeded frame the side reads or restores is one of these.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrameInput {
    /// The side's frame, which is also the position of the frame in the
    /// parent snapshot's frame list.
    pub frame: FrameId,
    pub register: u16,
    /// The type the parent snapshot's entry has. `Any` is not allowed.
    pub ty: TraceType,
    pub value: ValueId,
}

/// Where a loop parameter's first-iteration value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LoopParamInit {
    /// An entry input value.
    Input(ValueId),
    Const(TraceScalar),
}

/// A value that changes from one iteration to the next.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopParam {
    pub value: ValueId,
    pub ty: TraceType,
    pub init: LoopParamInit,
    /// The window register the parameter carries, when the recorder knows it.
    /// A side trace that links back to this loop hands over one value per
    /// parameter, found by register. [`TraceBuilder::loop_param`] fills it
    /// from the initial entry input.
    pub register: Option<u16>,
}

/// How a trace ends.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TraceTerminator {
    /// Starts the next iteration. `args` give each loop parameter's next
    /// value, all evaluated before any is assigned, so swaps are exact.
    LoopBack { args: Vec<ValueId> },
    /// Leaves through a snapshot after the body, with every effect committed.
    /// A side that ends in front of a loop of its own ends this way too, and
    /// a loop side ([`TraceSide`]) continues the exit in place.
    Exit { snapshot: SnapshotId },
    /// Ends a side trace by rejoining the loop it was recorded from. The
    /// snapshot is the state the loop header would see: the closing
    /// instruction committed, resumed at the header (the same shape as a
    /// loop trace's poll snapshot). Native code jumps to the root's loop
    /// header with one value per root loop parameter, taken from the
    /// snapshot by register, and spends one unit of fuel as a loop-back does.
    /// When the fuel runs out it leaves through the snapshot instead.
    Link { snapshot: SnapshotId },
    /// Ends a side trace by rejoining the loop of the loop side `side` (an
    /// index into [`TraceIr::sides`]), an ancestor of the linking side, at
    /// that loop's header. The snapshot has the shape of a [`Self::Link`]
    /// snapshot for the loop side: the loop's closing branch committed,
    /// resumed at its header. Native code jumps to the loop side's header
    /// with one value per loop parameter of the loop side, taken from the
    /// snapshot by register, and spends one unit of fuel as a loop-back does;
    /// when the fuel runs out it leaves through the snapshot instead. The
    /// linking side may write no register the loop side reads at its header
    /// other than its loop parameters (the plan checks it), so the loop
    /// side's invariants still hold.
    LinkLoop { side: usize, snapshot: SnapshotId },
    /// Ends an activation of a function trace ([`TraceIr::function`]) with
    /// the guest `Return` that closed it: the function returns `value` (of
    /// type [`FunctionTrace::result`], `None` for a function that returns no
    /// value) to its caller. Native code hands it to the calling activation,
    /// or, for the activation the VM entered, to the VM, which returns from
    /// the frame as the interpreter does.
    Return { value: Option<ValueId> },
    /// Ends an activation of a function trace with a tail call of its own
    /// function (`return f(args)`, the guest `TailCall` at `pc`): as the
    /// interpreter's in-place tail call does, the frame starts over at pc 0
    /// with `args` as its parameters (one per [`FunctionTrace::parameters`]
    /// entry, of that type), and counts one more tail call. Native code jumps
    /// back to the activation's start, in the same activation. The window
    /// has an [`ExternalSlot::TailCalls`] slot, which holds the activation's
    /// count of tail calls: every snapshot names it, so an exit restores the
    /// count, and a frame whose count the run changed is named after the
    /// tail call, as the interpreter names it.
    TailCall { args: Vec<ValueId>, pc: usize },
}

/// How a value is defined.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ValueDef {
    /// The index-th entry input.
    Input(usize),
    /// The index-th [`TraceIr::frame_inputs`] entry: a register of a seeded
    /// frame of a side trace.
    FrameInput(usize),
    /// The index-th loop parameter.
    LoopParam(usize),
    /// The index-th body instruction.
    Instruction(usize),
    /// The index-th preheader instruction.
    Preheader(usize),
}

/// The type and definition of one value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    pub ty: TraceType,
    pub def: ValueDef,
}

/// What a window slot beyond the root frame's registers stands for.
///
/// A trace sees the running closure's upvalue cells and the globals it reads
/// as extra registers appended to the root frame's window. The recorder
/// treats them like registers, so loop formation, snapshots and the optimizer
/// need no special case. The VM reads them at entry and writes the upvalue
/// ones back at every exit.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ExternalSlot {
    /// The cell of the running closure's upvalue with this index.
    Upvalue(u16),
    /// The cell of upvalue `index` of pinned closure `closure`, which an
    /// inlined callee reads and writes. The closure's cells never change, so
    /// the cell is the same one whenever the trace runs.
    ClosureUpvalue { closure: u32, index: u16 },
    /// Like [`Self::Global`], read through pinned closure `closure`: the name
    /// is a constant of that closure's chunk and the table is that closure's
    /// environment.
    ClosureGlobal { closure: u32, constant: u32 },
    /// The global whose name is this constant of the root chunk. The trace
    /// only reads it, and only while the environment holds a present raw
    /// value for it (so no `__index` runs). It can never be written.
    ///
    /// The value is read once at entry and held for the whole run. That is
    /// exact only while the trace commits no effect (a call or a store could
    /// rebind the global), so the verifier rejects a trace that has both,
    /// until a site guard or dependency epoch exists.
    Global(u32),
    /// The table-valued field named by this constant of the root chunk, of
    /// the table in upvalue `upvalue` of the running closure (`_ENV.math` in
    /// the profiles that read globals through `_ENV`). Like
    /// [`Self::Global`] it is read once at entry, only while the table holds
    /// a present raw table for the name, and never written.
    UpvalueField { upvalue: u16, constant: u32 },
    /// Like [`Self::UpvalueField`], read through pinned closure `closure`:
    /// its upvalue `upvalue` and a constant of its chunk.
    ClosureUpvalueField {
        closure: u32,
        upvalue: u16,
        constant: u32,
    },
    /// Register `index` of the frame that called the root frame (the frame
    /// below it on the call stack). The window slot holds the register's value
    /// at entry and carries the caller's state through the iterations of a
    /// loop that returns into the caller and calls the root frame's function
    /// again (see [`Entry::Caller`]). Every exit writes the slot back, so the
    /// interpreter's caller frame is right wherever the trace leaves.
    CallerRegister(u16),
    /// Like [`Self::Global`], for a constant of the chunk of the frame that
    /// called the root frame, read from that frame's environment. It is read
    /// once at entry, which checks the caller (see [`Entry::Caller`]), and
    /// never written.
    CallerGlobal(u32),
    /// The root frame's tail call count (the interpreter's
    /// `Frame::tail_calls`), an integer. A trace whose loop runs through tail
    /// calls of the root frame (see [`TraceIr::loop_tail`]) reads it at entry,
    /// adds one at each such call and writes it back at every exit, where a
    /// count that changed also renames the frame after the last tail call.
    /// In a function trace with tail calls of its own function
    /// ([`TraceTerminator::TailCall`]) it is the count of the activation
    /// running: the frame's at entry, zero for a nested activation, one more
    /// after each tail call.
    TailCalls,
    /// The closure the root frame runs at entry. Only read: a loop that tail
    /// calls back into the root chunk checks that the callee is this closure,
    /// so the next iteration runs with the cells the entry read.
    RootClosure,
    /// The table the root frame's globals live in (its environment, in the
    /// profiles that read globals by name rather than through `_ENV`). Only
    /// read: a trace binds fields of it, which is how it reads and writes the
    /// globals it stores. The entry takes the frame's environment, so a
    /// function environment that differs between runs is another input.
    Environment,
    /// The number of extra arguments of the root frame (the values `...`
    /// holds), an integer. Only read: a trace that reads `...` guards it
    /// against the count the recording saw.
    VarargCount,
    /// Extra argument `index` (from 0) of the root frame, nil past the last.
    /// Only read.
    Vararg(u32),
}

/// A way to name one variable's cell: an upvalue of a closure the trace
/// runs, or a local of the root frame that an inner closure captured.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum CellSource {
    /// Upvalue `index` of the root frame's closure.
    Upvalue(u16),
    /// Upvalue `index` of pinned closure `closure`.
    ClosureUpvalue { closure: u32, index: u16 },
    /// The cell of the root frame's captured local in this guest register.
    /// The trace holds that variable in the register, and the VM writes the
    /// register back to its cell at every exit.
    Captured(u16),
}

/// A promise that two names refer to one cell, which the trace relies on to
/// keep one value for the variable: an inlined callee that shares a variable
/// with the root frame (or with another callee) reads and writes the root
/// window register of the other name. The VM checks every promise on entry
/// and refuses a frame that breaks one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CellAlias {
    /// The name the trace gave no window register of its own.
    pub cell: CellSource,
    /// The name whose register or window slot stands for both.
    pub is: CellSource,
}

/// What makes a trace a function-entry trace: it is entered where a self call
/// pushed a frame of its function (pc 0), runs one activation, and ends at
/// the function's `Return` ([`TraceTerminator::Return`]). A direct self call
/// in the function ([`TraceOp::SelfCall`]) runs the trace again as a nested
/// activation. Every member of a tree (the root and its sides) has the same
/// one, apart from `callee_registers`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FunctionTrace {
    /// The type of each fixed parameter at entry, which is the type of the
    /// guest register of the same number: the root's entry contract checks
    /// them, and every self call passes arguments of exactly these types, so
    /// a nested activation never mismatches.
    pub parameters: Vec<TraceType>,
    /// The type of the value every `Return` gives, or `None` when the function
    /// returns no value.
    pub result: Option<TraceType>,
    /// The window registers (upvalue cells or globals) a self call reads its
    /// callee from. The VM enters only a frame whose closure every one of
    /// them holds, and nothing in a run writes them, so a self call needs no
    /// identity guard of its own.
    pub callee_registers: Vec<u16>,
}

/// The [`TraceSide::guard`] of a side that continues its parent's terminal
/// exit ([`TraceTerminator::Exit`]) instead of one of its guards. The parent
/// ends where the interpreter would enter a loop of its own (a numeric for
/// loop beside the root's), and the side is that loop.
pub const TERMINAL_GUARD: u32 = u32::MAX;

/// A side trace: the rest of an iteration, recorded from the state a guard of
/// another trace of the same tree restores, up to the point where it rejoins
/// the root loop.
///
/// It is not entered from the interpreter. The backend compiles the root and
/// its sides as one native function, and a failing guard that has a side
/// jumps into it with the guard's snapshot values as the side's entry inputs
/// (an [`EntryInput`] names a register, and its value is the parent
/// snapshot's entry for that register). Sides therefore carry no entry type
/// checks of their own: their types were checked when the tree was assembled.
///
/// A side that ends in [`TraceTerminator::LoopBack`] is a *loop side*: a loop
/// of its own inside the tree. It has loop parameters and a poll snapshot like
/// a root, its entry inputs bind to the parent's state as any side's do, and
/// the guards of its body may take sides in turn. It continues the parent's
/// [`TERMINAL_GUARD`], and the loop ends when its closing guard fails.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSide {
    /// The side whose guard this one continues, as an index into
    /// [`TraceIr::sides`] (always an earlier one), or `None` for the root.
    pub parent: Option<usize>,
    /// The [`GuardInfo::origin`] of the parent's guard, or [`TERMINAL_GUARD`].
    pub guard: u32,
    pub ir: Box<TraceIr>,
}

/// An owned, verifiable trace.
///
/// Serde preserves the IR data, but does not produce an executable portable
/// profile by itself. The VM must relocate pinned closure identities and native
/// calls, reject process-local borrowed string addresses, and verify the trace
/// before compiling it. Use a binary serializer to preserve non-finite floats
/// and NaN payloads.
// Deliberate shape: the bools are independent serialized trace facts, one
// per optimization decision; the data layout is the IR's public format.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceIr {
    pub id: TraceId,
    /// Logical frames the trace touches. Index 0 is the root frame. The
    /// others are calls the trace inlined, one entry per call instance.
    pub frames: Vec<TraceFrameInfo>,
    /// The closures the trace specialised on, by generation-aware identity
    /// (`ClosureHandle::native_identity`). Lane `k` of every run's root bank
    /// holds closure `k` of this list, so `Constant(Closure(k))` is that
    /// closure and a call-site guard compares against it. A closure that no
    /// longer exists matches nothing, and the VM refuses to enter then.
    pub pinned_closures: Vec<u64>,
    /// Cells that two names share, see [`CellAlias`].
    pub cell_aliases: Vec<CellAlias>,
    /// Slots in the trace root bank that snapshots may name.
    pub root_slot_count: u32,
    /// For a loop that closes by a tail call back into the root closure
    /// instead of a backward branch: that tail call. The root frame then runs
    /// other closures between iterations ([`RootActivation`]), and the loop
    /// header is pc 0 of the root chunk, entered with no previous pc (the poll
    /// snapshot is a `Before` one whose root frame has no
    /// `last_interpreted_pc`). A snapshot of the root frame itself restores
    /// this site as the frame's name whenever the run made a tail call (see
    /// [`ExternalSlot::TailCalls`]).
    pub loop_tail: Option<TailSite>,
    /// For a function-entry trace: its parameters, result and callee slots
    /// (see [`FunctionTrace`]). Such a trace has no loop: no loop parameters,
    /// and its poll snapshot is the entry state (`Before` pc 0 with no previous
    /// pc), which an activation leaves through when the run's fuel is spent.
    /// `None` for a loop trace.
    pub function: Option<FunctionTrace>,
    /// The string behind each root slot, as bytes. The recorder copies it from
    /// the constant of whichever chunk (the root's or an inlined callee's)
    /// loaded it, so a slot never depends on which chunk's constant table
    /// the VM would consult. The VM turns them into the values a `Root` entry
    /// restores. Slots beyond this list have no string and cannot be
    /// restored.
    pub root_constants: Vec<Arc<[u8]>>,
    /// The last `external.len()` registers of the root frame's window, which
    /// are not guest registers. `register_count` counts them.
    pub external: Vec<ExternalSlot>,
    pub entry: EntryContract,
    /// For a side trace that starts inside inlined calls of its parent: the
    /// registers of those calls' frames it takes from the parent's snapshot
    /// (see [`FrameInput`]). Empty for every other trace.
    #[serde(default)]
    pub frame_inputs: Vec<FrameInput>,
    /// For a side trace: how many inlined frames above the root the parent's
    /// snapshot at the guard has, which the side's own frames `1..=n` stand
    /// for (the calls the guard failed in). The side starts in the innermost
    /// of them. Zero for a side that starts in the root frame, and for a root.
    #[serde(default)]
    pub seed_frames: u8,
    /// The table storage the entry resolves for [`TraceOp::ArrayLoad`] and
    /// its siblings, one word pair per binding after the entry inputs.
    ///
    /// A side trace starts its list with the root's and may add bindings the
    /// root does not have: the tree binds them all at entry and an operation
    /// names a binding by its position in that shared list (see
    /// [`super::plan_tree`]). Every member of a planned tree has an entry
    /// input for the table of each binding, because no member writes a bound
    /// table's register.
    pub bindings: Vec<TableBinding>,
    /// The fields of tables met at run time that [`TraceOp::TableFieldLoad`]
    /// and [`TraceOp::TableFieldStore`] name by position. A side trace starts
    /// its list with the root's and may add fields (see [`super::plan_tree`]).
    pub fields: Vec<TableField>,
    pub loop_params: Vec<LoopParam>,
    pub values: Vec<ValueInfo>,
    /// Straight-line code that runs once after entry, before the first
    /// iteration. The optimizer fills it with loop-invariant work: operations
    /// whose operands are constants or entry inputs, which no iteration can
    /// change, and guards on such values. It holds no effects and no reads.
    /// A guard here leaves through a snapshot that restores the entry state,
    /// so the interpreter simply starts the iteration itself.
    pub preheader: Vec<TraceInstruction>,
    pub body: Vec<TraceInstruction>,
    pub guards: Vec<GuardInfo>,
    pub snapshots: Vec<Snapshot>,
    /// Every guest register the trace writes anywhere in its body, as the
    /// recorder reports it. In a loop a write anywhere counts, because the
    /// next iteration sees it. A snapshot may not map such a register to
    /// `Unchanged`.
    pub written_registers: Vec<(FrameId, u16)>,
    /// The state a loop-back publishes when the VM asks the trace to stop
    /// between iterations (fuel, GC, control). It is the iteration's final
    /// state with the closing branch committed, resumed at the loop header:
    /// `After { pc: last body pc, successor: header }`. A trace that loops
    /// needs one to be pollable. `None` for traces that never loop back.
    pub poll_snapshot: Option<SnapshotId>,
    pub terminator: TraceTerminator,
    /// The side traces that continue this trace's guards, parents first. Only
    /// a root trace has them: a side's own list is empty.
    pub sides: Vec<TraceSide>,
    /// The guest registers live at the loop header, from the chunk's
    /// liveness. Set once a trace has sides: they decide which registers the
    /// whole tree must carry through the loop. Empty otherwise.
    pub header_live: Vec<u16>,
    /// Whether the trace counts guest instructions for a count hook: every
    /// snapshot's [`Snapshot::counted`] is the exact number of counted
    /// instructions executed when an exit restores it, and the backend
    /// carries the hook's budget through the loop. Only a loop trace can be
    /// counted. A side of a counted tree is counted too.
    #[serde(default)]
    pub counted: bool,
    /// For a loop over `next` whose control value the trace keeps in a lane
    /// of the run's bank instead of a typed register (see
    /// [`HelperId::NEXT_BOX_BASE`](super::HelperId::NEXT_BOX_BASE)): the
    /// control register. Its entry input has type [`TraceType::String`], but
    /// the VM makes the lane at entry and puts whatever value the register
    /// holds in it: a key of any type, or nil at the start of a traversal.
    /// The lane is read by every exit that restores the register.
    #[serde(default)]
    pub boxed_control: Option<u16>,
    /// For a boxed loop whose body never reads its first variable (the
    /// register after the control register): the variable always holds the
    /// control value, so an exit restores it from the control lane.
    #[serde(default)]
    pub boxed_key: bool,
    /// For a boxed loop whose body never reads its second variable (two
    /// registers after the control register): the register is an entry input
    /// of type [`TraceType::String`] whose lane is the one after the control
    /// lane, and each step stores the entry's value there. An exit restores
    /// the variable from it.
    #[serde(default)]
    pub boxed_value: bool,
    /// For a boxed loop: the parts of the table's traversal that native code
    /// steps along itself, before the step helper is called (see
    /// [`BoxedSteps`]).
    #[serde(default)]
    pub boxed_steps: Option<BoxedSteps>,
    /// Whether every array step of a `next` loop ([`TraceOp::ArrayNext`]) ran
    /// over a table with no hash part when it was recorded. The exit at the
    /// end of such a loop's array part is the end of the traversal, which
    /// takes no side trace: like the exit of an `ipairs` loop, it happens
    /// once per run of the loop, and a side that carries the loop's
    /// enclosing loop into the tree makes the registers of the hot loop
    /// compete with the enclosing loop's.
    #[serde(default)]
    pub array_steps_plain: bool,
    /// Whether the optimizer keeps branch guards on invariant conditions in
    /// the loop body instead of hoisting them into the preheader. Set once
    /// the root of a `while` or generic-for loop that other loops enclose
    /// turns out to be a stepping stone (it runs a few trips per entry): its
    /// recorded trip may hold still what the next run of the loop changes (a
    /// `while lz ~= lzm or rz ~= rzm` recorded on a trip that steps only
    /// `rz`), and the preheader runs again at every link of an outer side, so
    /// a hoisted guard would leave there, where no side can start, on nearly
    /// every pass the tree's outer sides make. A long loop keeps hoisting.
    #[serde(default)]
    pub keep_branch_guards: bool,
}

/// The steps of a boxed `next` loop that native code takes without calling
/// the helper.
///
/// A step of the lane-keeping helpers ([`HelperId::next_box`]) is the entry
/// after the key the control lane holds. Native code takes it when it can,
/// and calls the helper when it cannot, in both cases leaving the lane as the
/// helper would, so an exit restores the registers from it.
///
/// With `array`, a step along the array part: the lane holds nil (the start
/// of a traversal) or a whole number inside the array part whose slot holds
/// a value, and a slot with a value follows it. Native code finds that slot
/// by a scan over the nil slots, stores its index in the lane, and gives the
/// key and the value.
///
/// With `cursor`, a step along the hash part: the table keeps a snapshot of
/// its hash part in traversal order, and the lane holds the key the
/// snapshot's position names. The run keeps that fact in the lane two after
/// the control lane, which holds the address of the snapshot's description
/// (`oximond::runtime::NativeCursor`); native code steps the position over the
/// deleted entries to the next one and stores its key in the lane. It needs
/// a trace with nothing that can move hash storage, since the snapshot is the
/// storage as it stands.
///
/// Either step gives up, and the helper makes it, at the end of a part, for
/// a key or a value of a kind the call does not expect, and when the lane
/// does not hold what the step needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoxedSteps {
    /// The binding of the table's array part, when the table has one.
    pub array: Option<u32>,
    /// Whether the run makes the lane for the hash part's snapshot.
    pub cursor: bool,
    /// Whether the profile's keys of the array part are integers (they are
    /// floats otherwise): the type the lane holds after a native step.
    pub integer: bool,
}

impl TraceIr {
    /// Checks every invariant the compiler relies on.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn verify(&self) -> Result<(), TraceVerifyError> {
        verify_trace(self)
    }

    /// [`Self::verify`] for a trace the optimizer rewrote, without the two
    /// checks that read guest steps off instruction adjacency (see
    /// [`super::verify::verify_optimized_trace`]).
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn verify_optimized(&self) -> Result<(), TraceVerifyError> {
        super::verify::verify_optimized_trace(self)
    }

    /// A byte estimate of the trace's owned data, for cache admission. It
    /// counts the heap vectors at their element size and ignores allocator
    /// overhead, so it is a lower bound that grows with the trace.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        use super::snapshot::{RegisterEntry, SnapshotFrame};
        use std::mem::size_of;
        let snapshots: usize = self
            .snapshots
            .iter()
            .map(|snapshot| {
                size_of::<Snapshot>()
                    + snapshot
                        .frames
                        .iter()
                        .map(|frame| {
                            size_of::<SnapshotFrame>()
                                + frame.registers.len() * size_of::<RegisterEntry>()
                        })
                        .sum::<usize>()
                    + snapshot
                        .virtuals
                        .iter()
                        .map(|table| {
                            size_of::<super::snapshot::VirtualTable>()
                                + (table.array.len() + table.fields.len())
                                    * (size_of::<RegisterEntry>()
                                        + size_of::<super::snapshot::VirtualKey>())
                        })
                        .sum::<usize>()
            })
            .sum();
        size_of::<Self>()
            + self.frames.len() * size_of::<TraceFrameInfo>()
            + self.pinned_closures.len() * size_of::<u64>()
            + self.cell_aliases.len() * size_of::<CellAlias>()
            + self
                .root_constants
                .iter()
                .map(|text| text.len() + size_of::<Arc<[u8]>>())
                .sum::<usize>()
            + self.external.len() * size_of::<ExternalSlot>()
            + self.entry.inputs.len() * size_of::<EntryInput>()
            + self.bindings.len() * size_of::<TableBinding>()
            + self.fields.len() * size_of::<TableField>()
            + self.loop_params.len() * size_of::<LoopParam>()
            + self.values.len() * size_of::<ValueInfo>()
            + (self.preheader.len() + self.body.len()) * size_of::<TraceInstruction>()
            + self.guards.len() * size_of::<GuardInfo>()
            + self.written_registers.len() * size_of::<(FrameId, u16)>()
            + self.header_live.len() * size_of::<u16>()
            + self
                .sides
                .iter()
                .map(|side| size_of::<TraceSide>() + side.ir.estimated_bytes())
                .sum::<usize>()
            + snapshots
    }

    /// Whether `other` is the same trace apart from its id. Two recordings of
    /// the same loop under the same types produce the same content, so the
    /// compile broker treats them as one request.
    #[must_use]
    pub fn same_content(&self, other: &Self) -> bool {
        self.frames == other.frames
            && self.pinned_closures == other.pinned_closures
            && self.cell_aliases == other.cell_aliases
            && self.root_slot_count == other.root_slot_count
            && self.loop_tail == other.loop_tail
            && self.function == other.function
            && self.root_constants == other.root_constants
            && self.external == other.external
            && self.entry == other.entry
            && self.bindings == other.bindings
            && self.fields == other.fields
            && self.loop_params == other.loop_params
            && self.values == other.values
            && self.preheader == other.preheader
            && self.body == other.body
            && self.guards == other.guards
            && self.snapshots == other.snapshots
            && self.written_registers == other.written_registers
            && self.poll_snapshot == other.poll_snapshot
            && self.terminator == other.terminator
            && self.header_live == other.header_live
            && self.counted == other.counted
            && self.sides.len() == other.sides.len()
            && self.sides.iter().zip(&other.sides).all(|(a, b)| {
                a.parent == b.parent && a.guard == b.guard && a.ir.same_content(&b.ir)
            })
    }

    /// The type of a value, if it exists.
    #[must_use]
    pub fn value_type(&self, value: ValueId) -> Option<TraceType> {
        self.values.get(value.index()).map(|info| info.ty)
    }

    /// Marks in `cold` the values of the body that only exits, sides and
    /// links name, directly or through other such values, and whose
    /// instruction is float arithmetic (`Negate`, or an `Arithmetic` with two
    /// float operands whose operator `deferrable` accepts). The values the
    /// back edge, a tail call or the return take stay hot: exits, links and
    /// snapshots are not uses. A backend skips the marked values in its loop
    /// and computes each where an exit reads it; their operands are defined
    /// before them. `hot` is scratch.
    pub fn mark_cold_values(
        &self,
        deferrable: impl Fn(&ArithmeticOp) -> bool,
        cold: &mut Vec<bool>,
        hot: &mut Vec<bool>,
    ) {
        self.mark_cold_values_except(deferrable, |_| None, cold, hot);
    }

    /// [`Self::mark_cold_values`] for a lowering that does not read one
    /// operand of some operations: `unread` names that operand (the key of an
    /// array access whose slot the backend addresses another way, say), which
    /// then counts as a use no more than a snapshot's does.
    pub fn mark_cold_values_except(
        &self,
        deferrable: impl Fn(&ArithmeticOp) -> bool,
        unread: impl Fn(&TraceOp) -> Option<ValueId>,
        cold: &mut Vec<bool>,
        hot: &mut Vec<bool>,
    ) {
        self.mark_cold_values_seeded(deferrable, unread, |_| true, &[], cold, hot);
    }

    /// [`Self::mark_cold_values_except`] for a lowering that also drops some
    /// loop parameters from its loop: `carried` says whether the loop-back
    /// passes on the argument of parameter `index` (one it does not pass is
    /// no use of the argument), and `read` names values something reads
    /// whatever the walk finds (what a guard hands to a side lowered with
    /// the tree, say).
    pub fn mark_cold_values_seeded(
        &self,
        deferrable: impl Fn(&ArithmeticOp) -> bool,
        unread: impl Fn(&TraceOp) -> Option<ValueId>,
        carried: impl Fn(usize) -> bool,
        read: &[ValueId],
        cold: &mut Vec<bool>,
        hot: &mut Vec<bool>,
    ) {
        cold.clear();
        cold.resize(self.values.len(), false);
        hot.clear();
        hot.resize(self.values.len(), false);
        for value in read {
            hot[value.index()] = true;
        }
        match &self.terminator {
            TraceTerminator::LoopBack { args } => {
                for (index, arg) in args.iter().enumerate() {
                    if carried(index) {
                        hot[arg.index()] = true;
                    }
                }
            }
            TraceTerminator::TailCall { args, .. } => {
                for arg in args {
                    hot[arg.index()] = true;
                }
            }
            TraceTerminator::Return { value } => {
                if let Some(value) = value {
                    hot[value.index()] = true;
                }
            }
            TraceTerminator::Exit { .. }
            | TraceTerminator::Link { .. }
            | TraceTerminator::LinkLoop { .. } => {}
        }
        let float = |value: ValueId| self.value_type(value) == Some(TraceType::Float);
        for instruction in self.body.iter().rev() {
            let deferred = instruction.result.filter(|result| {
                !hot[result.index()]
                    && match &instruction.op {
                        TraceOp::Arithmetic { op, left, right } => {
                            float(*left) && float(*right) && deferrable(op)
                        }
                        TraceOp::Negate { source } => float(*source),
                        _ => false,
                    }
            });
            if let Some(result) = deferred {
                cold[result.index()] = true;
            } else {
                let skipped = unread(&instruction.op);
                // An operation that names the value twice reads it.
                let mut seen = 0;
                instruction.op.operands(|value| {
                    if Some(value) == skipped {
                        seen += 1;
                    }
                });
                instruction.op.operands(|value| {
                    if Some(value) != skipped || seen != 1 {
                        hot[value.index()] = true;
                    }
                });
            }
        }
    }

    /// Whether the guard with origin `origin` is the type guard of an entry
    /// input of either type (nil or a number, see `Recorder::with_lazy`).
    /// Such a guard fails when the run holds the other type, and a side
    /// continues it with that type.
    #[must_use]
    pub fn guards_untyped_input(&self, origin: u32) -> bool {
        let Some(index) = self.guards.iter().position(|info| info.origin == origin) else {
            return false;
        };
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(instruction.op, TraceOp::GuardType { source, guard, .. }
            if guard.index() == index
                && self.value_type(source) == Some(TraceType::Any)
                && matches!(
                    self.values.get(source.index()).map(|info| info.def),
                    Some(ValueDef::Input(_))
                ))
        })
    }

    /// Whether this trace or one of its sides holds a [`TraceOp::ArraySet`]
    /// or a helper that writes tables, which may resize an array part in the
    /// middle of a run. Native code then reads the words of every array and
    /// length binding again after each such operation.
    #[must_use]
    pub fn sets_tables(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .any(|instruction| match &instruction.op {
                TraceOp::ArraySet { .. } => true,
                // A helper that writes may grow any table.
                TraceOp::CallHelper { helper, .. } => helper.effects().moves_storage(),
                _ => false,
            })
            || self.sides.iter().any(|side| side.ir.sets_tables())
    }

    /// Whether this trace or one of its sides holds an operation that may
    /// clear a slot of an array part: an [`TraceOp::ArraySet`] (a nil store
    /// goes through the VM), a helper that writes the heap, a call of a
    /// whole activation, or an effect the trace knows nothing about. A trace
    /// with none leaves every slot that holds a value holding one for the
    /// whole run.
    #[must_use]
    pub fn may_clear_array_slots(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .any(|instruction| match &instruction.op {
                TraceOp::ArraySet { .. } | TraceOp::SelfCall { .. } | TraceOp::Effect { .. } => {
                    true
                }
                TraceOp::CallHelper { helper, .. } => {
                    helper.effects().contains(HelperEffects::WRITES_HEAP)
                }
                _ => false,
            })
            || self
                .sides
                .iter()
                .any(|side| side.ir.may_clear_array_slots())
    }

    /// Whether this trace or one of its sides holds an operation that may
    /// move the hash part of a table, not only its array part: an
    /// [`TraceOp::ArraySet`] (a key in the hash part, a rehash) or a helper
    /// that writes without promising to leave hash storage alone.
    #[must_use]
    pub fn moves_hash_storage(&self) -> bool {
        self.moves_hash_storage_in(true)
    }

    /// [`Self::moves_hash_storage`] without the host function calls
    /// ([`HelperId::vm_call`]). The run makes the rows of tables met by lane
    /// again after such a call, and the recording keeps field bindings out of
    /// a trace that makes one, so a call moves no storage that a trace
    /// relies on. Native code still must not probe a string index it loaded
    /// at entry beside the call, which is why [`Self::moves_hash_storage`]
    /// counts it.
    #[must_use]
    pub fn moves_hash_storage_outside_vm_calls(&self) -> bool {
        self.moves_hash_storage_in(false)
    }

    fn moves_hash_storage_in(&self, count_vm_calls: bool) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .any(|instruction| match &instruction.op {
                TraceOp::ArraySet { binding, .. } => !self.set_stays_in_place(*binding),
                TraceOp::CallHelper { helper, .. } => {
                    helper.effects().moves_hash_storage()
                        && (count_vm_calls || !helper.is_vm_call())
                }
                _ => false,
            })
            || self
                .sides
                .iter()
                .any(|side| side.ir.moves_hash_storage_in(count_vm_calls))
    }

    /// Whether this trace or one of its sides holds a helper that deletes a
    /// key where it stands ([`HelperEffects::DELETES_KEYS`]). The table's
    /// storage stays put, but a field the trace bound or met by lane may now
    /// hold nil, so [`Self::storage_a_set_would_move`] applies to it too.
    #[must_use]
    pub fn deletes_keys(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. } if helper.effects().deletes_keys()
            )
        }) || self.sides.iter().any(|side| side.ir.deletes_keys())
    }

    /// Whether this trace or one of its sides holds an arithmetic operation
    /// that satisfies `wanted`.
    #[must_use]
    pub fn has_arithmetic(&self, wanted: &impl Fn(ArithmeticOp) -> bool) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                instruction.op,
                TraceOp::Arithmetic { op, .. } | TraceOp::NumberModulo { float: op, .. }
                    if wanted(op)
            )
        }) || self.sides.iter().any(|side| side.ir.has_arithmetic(wanted))
    }

    /// Whether this trace or one of its sides holds a [`TraceOp::HashLoad`].
    #[must_use]
    pub fn reads_hash(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .any(|instruction| matches!(instruction.op, TraceOp::HashLoad { .. }))
            || self.sides.iter().any(|side| side.ir.reads_hash())
    }

    /// Whether this trace or one of its sides holds a [`TraceOp::HashLoad`]
    /// with a number key (a raw lookup over both parts of the table).
    #[must_use]
    pub fn reads_number_hash(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .any(|instruction| match instruction.op {
                TraceOp::HashLoad { key, .. } => matches!(
                    self.value_type(key),
                    Some(TraceType::Integer | TraceType::Float)
                ),
                _ => false,
            })
            || self.sides.iter().any(|side| side.ir.reads_number_hash())
    }

    /// Whether this trace or one of its sides reads or stores through the
    /// string index of a table's hash part: a [`TraceOp::HashLoad`], or a call
    /// of one of the three helpers that store in place, which native code
    /// makes itself ([`HelperId::SET_PRESENT`], [`HelperId::DELETE_STRING`]
    /// and [`HelperId::SET_REVIVE`]).
    #[must_use]
    pub fn uses_string_index(&self) -> bool {
        self.reads_hash()
            || self.preheader.iter().chain(&self.body).any(|instruction| {
                matches!(
                    &instruction.op,
                    TraceOp::CallHelper { helper, .. }
                        if matches!(
                            *helper,
                            HelperId::SET_PRESENT | HelperId::DELETE_STRING | HelperId::SET_REVIVE
                        )
                )
            })
            || self.sides.iter().any(|side| side.ir.uses_string_index())
    }

    /// The helper of every [`TraceOp::CallHelper`] in this trace and its
    /// sides, in body order.
    #[must_use]
    pub fn helper_calls(&self) -> Vec<HelperId> {
        let mut found: Vec<HelperId> = self
            .preheader
            .iter()
            .chain(&self.body)
            .filter_map(|instruction| match instruction.op {
                TraceOp::CallHelper { helper, .. } => Some(helper),
                _ => None,
            })
            .collect();
        for side in &self.sides {
            found.extend(side.ir.helper_calls());
        }
        found
    }

    /// The events of the metatable classes this trace and its sides check
    /// with a [`TraceOp::MetaGuard`] (`__index`, `__add`), one per guard. A
    /// class guard stands for the handler walk a helper such as
    /// [`HelperId::OPERATOR_HANDLER`] makes otherwise. A diagnostic for
    /// tests.
    #[must_use]
    pub fn class_guard_events(&self) -> Vec<Arc<[u8]>> {
        let mut found: Vec<Arc<[u8]>> = self
            .preheader
            .iter()
            .chain(&self.body)
            .filter_map(|instruction| match instruction.op {
                TraceOp::MetaGuard { binding, .. } => {
                    match &self.bindings.get(binding as usize)?.kind {
                        BindingKind::MetaClass { pins, .. } => Some(pins),
                        _ => None,
                    }
                }
                _ => None,
            })
            .flat_map(|pins| pins.iter().map(|pin| Arc::clone(&pin.event)))
            .collect();
        for side in &self.sides {
            found.extend(side.ir.class_guard_events());
        }
        found
    }

    /// How many operations of this trace and its sides call out of the
    /// generated code for a library function: helper calls, host calls and
    /// pure native functions. A diagnostic for tests.
    #[must_use]
    pub fn external_calls(&self) -> usize {
        let own = self
            .preheader
            .iter()
            .chain(&self.body)
            .filter(|instruction| {
                matches!(
                    instruction.op,
                    TraceOp::HostCall { .. }
                        | TraceOp::CallHelper { .. }
                        | TraceOp::NativeUnary { .. }
                        | TraceOp::NativeBinary { .. }
                )
            })
            .count();
        own + self
            .sides
            .iter()
            .map(|side| side.ir.external_calls())
            .sum::<usize>()
    }

    /// Whether this trace or one of its sides calls a helper that runs a
    /// coroutine or a host function ([`HelperEffects::RESUMES`]).
    #[must_use]
    pub fn resumes(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. } if helper.effects().resumes()
            )
        }) || self.sides.iter().any(|side| side.ir.resumes())
    }

    /// Whether every helper of this trace and its sides that runs other
    /// code ([`HelperEffects::RESUMES`]) is a host function call
    /// ([`HelperId::vm_call`]), none a coroutine.
    #[must_use]
    pub fn resumes_only_in_vm_calls(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .all(|instruction| match &instruction.op {
                TraceOp::CallHelper { helper, .. } => {
                    !helper.effects().resumes() || helper.is_vm_call()
                }
                _ => true,
            })
            && self
                .sides
                .iter()
                .all(|side| side.ir.resumes_only_in_vm_calls())
    }

    /// Whether this trace or one of its sides calls a host C function
    /// itself ([`HelperId::vm_call`]).
    #[must_use]
    pub fn makes_vm_calls(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. } if helper.is_vm_call()
            )
        }) || self.sides.iter().any(|side| side.ir.makes_vm_calls())
    }

    /// Why this trace and its sides cannot call a helper that runs a
    /// coroutine ([`HelperEffects::RESUMES`]), if they cannot. The coroutine
    /// runs arbitrary Lua code while the trace waits in the middle of its
    /// loop, so the trace may hold nothing the code could invalidate and the
    /// run cannot repair: a cell or a global it writes (native code keeps it
    /// in a register and stores it at an exit), a table field or row, a
    /// string index or a header (they address storage a write moves), a
    /// table set or another call that writes the heap, which would have to
    /// read its bindings again before and after the thread, and a table the
    /// trace made. The words of array and length bindings are
    /// the exception: the helper reads them again from the tables afterwards
    /// (and checks that a table still has no `__len`), and so are the
    /// callables and strings a binding fixes (`assert` read out of the
    /// environment table) and the cells, globals and captured locals the
    /// trace only reads, which it checks again. A function trace never resumes.
    ///
    /// A trace whose only such helpers call host functions
    /// ([`HelperId::vm_call`]) may also read fields, guard metatables, look
    /// up hash keys, set table keys and read and write the array words of
    /// tables met by lane: the run sends the lookups of string
    /// indexes to the VM and makes the rows of the tables it met by lane
    /// again after the call. It may also write cells, globals and captured
    /// locals, which the call's restore stores and the run compares with
    /// memory afterwards.
    #[must_use]
    pub fn resume_conflict(&self) -> Option<&'static str> {
        if !self.resumes() {
            return None;
        }
        if self.function.is_some() || self.loop_tail.is_some() {
            return Some("a resume in a function trace or a loop through tail calls");
        }
        // A cell or a global the trace only reads is read again after the
        // resume and compared with what the run started with (the VM's
        // `RunTables::externals_hold`). One the trace writes lives in a
        // native value until an exit stores it, where the thread could not
        // see it. A host function call sees it: the call's restore stores
        // it first, and the run reads it again afterwards and leaves if the
        // function changed it.
        let calls_only = self.resumes_only_in_vm_calls();
        let first_external = self.frames.first().map_or(usize::MAX, |root| {
            usize::from(root.register_count).saturating_sub(self.external.len())
        });
        if !calls_only
            && self.written_registers.iter().any(|&(frame, register)| {
                frame == FrameId(0) && usize::from(register) >= first_external
            })
        {
            return Some("a resume beside a cell or global the trace writes");
        }
        // A host function call reads these again after the call: what the
        // metatable of a class holds, that a
        // table has no metatable, and the string index of a hash part, which
        // the lookups then leave to the VM (`RunTables::refresh_bindings`),
        // and every row of a table met by lane (`RunTables::refill_rows`).
        if self.bindings.iter().any(|binding| {
            let read_again = matches!(
                binding.kind,
                BindingKind::Array { .. }
                    | BindingKind::Bytes { .. }
                    | BindingKind::Length { .. }
                    | BindingKind::Callee { .. }
                    | BindingKind::StringField { .. }
            );
            let checked_after_call = calls_only
                && matches!(
                    binding.kind,
                    BindingKind::MetaClass { .. } | BindingKind::Hash | BindingKind::NoMetatable
                );
            !read_again && !checked_after_call
        }) {
            return Some("a resume beside a binding a coroutine could invalidate");
        }
        self.resume_op_conflict(calls_only)
    }

    /// Whether the only code this trace resumes is host function calls and
    /// it holds field or header bindings, whose slot addresses native code
    /// keeps across the call.
    #[must_use]
    pub fn vm_call_beside_field_bindings(&self) -> bool {
        self.resumes()
            && self.resumes_only_in_vm_calls()
            && self.bindings.iter().any(|binding| {
                matches!(
                    binding.kind,
                    BindingKind::Field { .. } | BindingKind::Header { .. }
                )
            })
    }

    /// [`Self::resume_conflict`] for the operations of this trace and its
    /// sides.
    fn resume_op_conflict(&self, calls_only: bool) -> Option<&'static str> {
        let conflict =
            self.preheader
                .iter()
                .chain(&self.body)
                .any(|instruction| match &instruction.op {
                    // The row of a table is made again after a host call, so
                    // a field read or written through it, and the metatable
                    // word a guard reads, are the table's as it stands.
                    TraceOp::TableFieldLoad { .. }
                    | TraceOp::TableFieldStore { .. }
                    | TraceOp::MetaGuard { .. }
                    | TraceOp::HashLoad { .. }
                    // A table set and the array words of a table met by lane
                    // are written to memory when they run, and read again
                    // from the tables after the call.
                    | TraceOp::ArraySet { .. }
                    | TraceOp::LaneArrayLoad { .. }
                    | TraceOp::LaneArrayStore { .. }
                        if calls_only =>
                    {
                        false
                    }
                    TraceOp::ArraySet { .. }
                    | TraceOp::TableFieldLoad { .. }
                    | TraceOp::TableFieldStore { .. }
                    | TraceOp::MetaGuard { .. }
                    | TraceOp::HashLoad { .. }
                    | TraceOp::SelfCall { .. }
                    | TraceOp::Effect { .. }
                    | TraceOp::Read { .. } => true,
                    // A table met by lane: a thread could run code that
                    // stores into it where the trace holds its row, or
                    // resize the array whose words a lane read holds.
                    TraceOp::ArrayLoad { expected, .. } => {
                        *expected == TraceType::Table && !calls_only
                    }
                    TraceOp::LaneArrayLoad { .. } | TraceOp::LaneArrayStore { .. } => true,
                    TraceOp::CallHelper { helper, .. } => {
                        let effects = helper.effects();
                        !effects.resumes()
                            && (effects.commits_effect()
                                || effects.contains(HelperEffects::FRESH_TABLE))
                    }
                    _ => false,
                });
        if conflict {
            return Some(
                "a resume beside a table set, a call that writes, a table of the trace's own or a field read",
            );
        }
        self.sides
            .iter()
            .find_map(|side| side.ir.resume_op_conflict(calls_only))
    }

    /// Whether this trace or one of its sides holds a [`TraceOp::HostCall`]
    /// or a [`TraceOp::CallHelper`].
    #[must_use]
    pub fn calls_host(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                instruction.op,
                TraceOp::HostCall { .. }
                    | TraceOp::CallHelper { .. }
                    | TraceOp::ReleaseStrings { .. }
            )
        }) || self.sides.iter().any(|side| side.ir.calls_host())
    }

    /// Whether the trace, or a side of it, makes a table
    /// ([`HelperId::TABLE_NEW`]).
    #[must_use]
    pub fn makes_tables(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. } if *helper == HelperId::TABLE_NEW
            )
        }) || self.sides.iter().any(|side| side.ir.makes_tables())
    }

    /// Whether a [`TraceOp::ArraySet`] on the array binding `binding` never
    /// calls the VM: its table's header binding is strict.
    fn set_stays_in_place(&self, binding: u32) -> bool {
        let Some(register) = self
            .bindings
            .get(binding as usize)
            .map(|bound| bound.register)
        else {
            return false;
        };
        self.bindings.iter().any(|other| {
            other.register == register && other.kind == BindingKind::Header { strict: true }
        })
    }

    /// Why this trace and its sides cannot hold an operation that moves hash
    /// storage ([`Self::moves_hash_storage`]) or deletes keys
    /// ([`Self::deletes_keys`]), if they cannot: a rehash moves
    /// the storage that a field binding, or the row of a table met by lane,
    /// addresses for the whole run. Array bindings and length bindings are
    /// read again after every operation that may move an array.
    #[must_use]
    pub fn storage_a_set_would_move(&self) -> Option<&'static str> {
        self.storage_a_set_would_move_in(self.only_array_sets_move_storage())
    }

    /// [`Self::storage_a_set_would_move`], with `array_sets_only` taken from
    /// the whole tree.
    fn storage_a_set_would_move_in(&self, array_sets_only: bool) -> Option<&'static str> {
        // An [`TraceOp::ArraySet`] moves the hash part of its own table
        // only. Entry refuses a run whose set binding and field binding name
        // one table, so the tables a field binding addresses stay put.
        if !array_sets_only
            && self
                .bindings
                .iter()
                .any(|binding| matches!(binding.kind, BindingKind::Field { .. }))
        {
            return Some("a table set beside a field binding");
        }
        let lanes =
            self.preheader
                .iter()
                .chain(&self.body)
                .any(|instruction| match instruction.op {
                    // The operations that read a lane's row. A table met by
                    // lane that only helpers look into (they look the key up
                    // on each call) holds no address a rehash could move.
                    TraceOp::TableFieldLoad { .. } | TraceOp::TableFieldStore { .. } => true,
                    _ => false,
                });
        if lanes {
            return Some("a table set beside a table met by lane");
        }
        self.sides
            .iter()
            .find_map(|side| side.ir.storage_a_set_would_move_in(array_sets_only))
    }

    /// Whether no helper of this trace or its sides moves hash storage or
    /// deletes a key: the only operations that do are
    /// [`TraceOp::ArraySet`]s, each of which touches the table of its own
    /// binding.
    #[must_use]
    pub fn only_array_sets_move_storage(&self) -> bool {
        self.preheader
            .iter()
            .chain(&self.body)
            .all(|instruction| match &instruction.op {
                TraceOp::CallHelper { helper, .. } => {
                    let effects = helper.effects();
                    // A host function call moves whatever it likes. The run
                    // makes the rows of tables met by lane again afterwards,
                    // and the recording keeps field bindings out of a trace
                    // that makes one (`vm_call_beside_field_bindings`).
                    (!effects.moves_hash_storage() || helper.is_vm_call())
                        && !effects.deletes_keys()
                }
                _ => true,
            })
            && self
                .sides
                .iter()
                .all(|side| side.ir.only_array_sets_move_storage())
    }

    /// Whether the set binding `binding` may move the hash part of its
    /// table: its header is not strict ([`Self::set_stays_in_place`]).
    #[must_use]
    pub fn set_moves_storage(&self, binding: u32) -> bool {
        !self.set_stays_in_place(binding)
    }

    /// Why this trace and its sides cannot both change a metatable and rely
    /// on one, if they cannot. Entry establishes what a binding says about a
    /// metatable once (no metatable, no `__len`, a field found through
    /// `__index`) and nothing in a run changes it, unless a call replaces a
    /// table's metatable ([`HelperEffects::WRITES_METATABLE`]).
    #[must_use]
    pub fn stale_metatable_fact(&self) -> Option<&'static str> {
        if !self.writes_metatables() {
            return None;
        }
        self.relies_on_metatables()
            .then_some("a metatable change beside a binding that assumes one")
    }

    /// Whether this trace or one of its sides changes a table's metatable.
    fn writes_metatables(&self) -> bool {
        self.preheader.iter().chain(&self.body).any(|instruction| {
            matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. }
                    if helper.effects().contains(HelperEffects::WRITES_METATABLE)
            )
        }) || self.sides.iter().any(|side| side.ir.writes_metatables())
    }

    /// Whether a binding of this trace or one of its sides depends on what a
    /// metatable holds.
    fn relies_on_metatables(&self) -> bool {
        self.bindings.iter().any(|binding| match &binding.kind {
            BindingKind::NoMetatable
            | BindingKind::Header { .. }
            | BindingKind::Length { raw: false, .. } => true,
            // The class guard reads the row of each table it guards, which a
            // change of metatable makes again, and what the metatable holds
            // is no table's own: a change of metatable cannot stale it.
            BindingKind::MetaClass { .. } => false,
            BindingKind::Field { depth, .. }
            | BindingKind::Callee { depth, .. }
            | BindingKind::StringField { depth, .. } => *depth != 0,
            BindingKind::Array { .. }
            | BindingKind::Bytes { .. }
            | BindingKind::Hash
            | BindingKind::Length { raw: true, .. } => false,
        }) || self.sides.iter().any(|side| side.ir.relies_on_metatables())
    }

    /// Number of effects in the body.
    #[must_use]
    pub fn effect_count(&self) -> u32 {
        let count = self
            .body
            .iter()
            .filter(|instruction| instruction.op.committed_effect().is_some())
            .count();
        u32::try_from(count).expect("effect count fits u32")
    }
}

/// Builds a [`TraceIr`] with dense ids. Building does not verify; call
/// [`TraceIr::verify`] on the result.
#[derive(Debug)]
pub struct TraceBuilder {
    ir: TraceIr,
    effects: u32,
}

impl TraceBuilder {
    /// Starts a trace whose root frame has the given shape.
    #[must_use]
    pub fn new(id: TraceId, root: TraceFrameInfo) -> Self {
        Self {
            ir: TraceIr {
                id,
                frames: vec![root],
                pinned_closures: Vec::new(),
                cell_aliases: Vec::new(),
                root_slot_count: 0,
                loop_tail: None,
                function: None,
                root_constants: Vec::new(),
                external: Vec::new(),
                entry: EntryContract::default(),
                frame_inputs: Vec::new(),
                seed_frames: 0,
                bindings: Vec::new(),
                fields: Vec::new(),
                loop_params: Vec::new(),
                values: Vec::new(),
                preheader: Vec::new(),
                body: Vec::new(),
                guards: Vec::new(),
                snapshots: Vec::new(),
                written_registers: Vec::new(),
                poll_snapshot: None,
                terminator: TraceTerminator::LoopBack { args: Vec::new() },
                sides: Vec::new(),
                header_live: Vec::new(),
                counted: false,
                boxed_control: None,
                boxed_key: false,
                boxed_value: false,
                boxed_steps: None,
                array_steps_plain: false,
                keep_branch_guards: false,
            },
            effects: 0,
        }
    }

    /// Records that the trace writes guest register `register` of `frame`.
    /// Repeats are ignored.
    pub fn mark_written(&mut self, frame: FrameId, register: u16) {
        if !self.ir.written_registers.contains(&(frame, register)) {
            self.ir.written_registers.push((frame, register));
        }
    }

    /// Marks the trace as counting guest instructions for a count hook (see
    /// [`TraceIr::counted`]).
    pub fn set_counted(&mut self, counted: bool) {
        self.ir.counted = counted;
    }

    /// Marks the trace as keeping the control value of a `next` loop in the
    /// lane of register `register` (see [`TraceIr::boxed_control`]).
    pub fn set_boxed_control(&mut self, register: Option<u16>) {
        self.ir.boxed_control = register;
    }

    /// Declares the steps a boxed loop takes natively (see
    /// [`TraceIr::boxed_steps`]).
    pub fn set_boxed_steps(&mut self, steps: Option<BoxedSteps>) {
        self.ir.boxed_steps = steps;
    }

    /// Declares that the trace's array steps ran over tables with no hash
    /// part (see [`TraceIr::array_steps_plain`]).
    pub fn set_array_steps_plain(&mut self, plain: bool) {
        self.ir.array_steps_plain = plain;
    }

    /// Keeps the branch guards of the loop body in it (see
    /// [`TraceIr::keep_branch_guards`]).
    pub fn set_keep_branch_guards(&mut self, keep: bool) {
        self.ir.keep_branch_guards = keep;
    }

    /// Marks which variables of the boxed loop the trace restores from lanes
    /// (see [`TraceIr::boxed_key`] and [`TraceIr::boxed_value`]).
    pub fn set_boxed_variables(&mut self, key: bool, value: bool) {
        self.ir.boxed_key = key;
        self.ir.boxed_value = value;
    }

    /// Names the snapshot a loop-back publishes at a poll.
    pub fn set_poll_snapshot(&mut self, snapshot: SnapshotId) {
        self.ir.poll_snapshot = Some(snapshot);
    }

    /// Declares the size of the root bank.
    pub fn set_root_slot_count(&mut self, count: u32) {
        self.ir.root_slot_count = count;
    }

    /// Names the string behind each root slot. See
    /// [`TraceIr::root_constants`].
    pub fn set_root_constants(&mut self, constants: Vec<Arc<[u8]>>) {
        self.ir.root_slot_count = u32::try_from(constants.len()).expect("root slots fit u32");
        self.ir.root_constants = constants;
    }

    /// Declares the tail call that closes the loop. See [`TraceIr::loop_tail`].
    pub fn set_loop_tail(&mut self, site: TailSite) {
        self.ir.loop_tail = Some(site);
    }

    /// Makes the trace a function-entry trace. See [`TraceIr::function`].
    pub fn set_function(&mut self, function: FunctionTrace) {
        self.ir.function = Some(function);
    }

    /// Declares the closures the trace pins. See [`TraceIr::pinned_closures`].
    pub fn set_pinned_closures(&mut self, closures: Vec<u64>) {
        self.ir.pinned_closures = closures;
    }

    /// Declares the cells two names share. See [`TraceIr::cell_aliases`].
    pub fn set_cell_aliases(&mut self, aliases: Vec<CellAlias>) {
        self.ir.cell_aliases = aliases;
    }

    /// Declares the table bindings the entry resolves. See
    /// [`TraceIr::bindings`].
    pub fn set_bindings(&mut self, bindings: Vec<TableBinding>) {
        self.ir.bindings = bindings;
    }

    /// Declares the fields of run-time tables. See [`TraceIr::fields`].
    pub fn set_fields(&mut self, fields: Vec<TableField>) {
        self.ir.fields = fields;
    }

    /// Declares the window slots beyond the root frame's registers. The root
    /// frame's `register_count` must already include them.
    pub fn set_external(&mut self, external: Vec<ExternalSlot>) {
        self.ir.external = external;
    }

    /// Adds a logical frame and returns its id.
    pub fn add_frame(&mut self, info: TraceFrameInfo) -> FrameId {
        self.ir.frames.push(info);
        FrameId(u32::try_from(self.ir.frames.len() - 1).expect("frame count fits u32"))
    }

    fn define(&mut self, ty: TraceType, def: ValueDef) -> ValueId {
        self.ir.values.push(ValueInfo { ty, def });
        ValueId(u32::try_from(self.ir.values.len() - 1).expect("value count fits u32"))
    }

    /// The id the next value defined will have.
    #[must_use]
    pub fn next_value_id(&self) -> ValueId {
        ValueId(u32::try_from(self.ir.values.len()).expect("value count fits u32"))
    }

    /// Declares a root-frame register the trace reads on entry.
    pub fn input(&mut self, register: u16, ty: TraceType) -> ValueId {
        let index = self.ir.entry.inputs.len();
        let value = self.define(ty, ValueDef::Input(index));
        self.ir.entry.inputs.push(EntryInput {
            register,
            ty,
            value,
        });
        value
    }

    /// Declares a register of seeded frame `frame` the side reads on entry
    /// (see [`FrameInput`]).
    pub fn frame_input(&mut self, frame: FrameId, register: u16, ty: TraceType) -> ValueId {
        let index = self.ir.frame_inputs.len();
        let value = self.define(ty, ValueDef::FrameInput(index));
        self.ir.frame_inputs.push(FrameInput {
            frame,
            register,
            ty,
            value,
        });
        value
    }

    /// Sets how many inlined frames of the parent's snapshot the side's
    /// frames `1..=n` stand for (see [`TraceIr::seed_frames`]).
    pub fn set_seed_frames(&mut self, frames: u8) {
        self.ir.seed_frames = frames;
    }

    /// Declares a loop parameter.
    pub fn loop_param(&mut self, ty: TraceType, init: LoopParamInit) -> ValueId {
        let index = self.ir.loop_params.len();
        let value = self.define(ty, ValueDef::LoopParam(index));
        let register = match init {
            LoopParamInit::Input(input) => self
                .ir
                .entry
                .inputs
                .iter()
                .find(|entry| entry.value == input)
                .map(|entry| entry.register),
            LoopParamInit::Const(_) => None,
        };
        self.ir.loop_params.push(LoopParam {
            value,
            ty,
            init,
            register,
        });
        value
    }

    /// The register map `entries` make for the frame at `position` (0 is the
    /// root frame) of the snapshot added next, sharing with the same frame
    /// of the snapshot added last every run that holds the same entries
    /// there. Snapshots in a row differ in a few registers, so most runs are
    /// shared.
    #[must_use]
    pub fn register_map(&self, position: usize, entries: &[RegisterEntry]) -> RegisterMap {
        match self
            .ir
            .snapshots
            .last()
            .and_then(|last| last.frames.get(position))
        {
            Some(frame) => RegisterMap::sharing(&frame.registers, entries),
            None => RegisterMap::from(entries),
        }
    }

    /// Adds a snapshot, assigning its id.
    pub fn snapshot(&mut self, mut snapshot: Snapshot) -> SnapshotId {
        let id = SnapshotId(u32::try_from(self.ir.snapshots.len()).expect("snapshots fit u32"));
        snapshot.id = id;
        self.ir.snapshots.push(snapshot);
        id
    }

    /// The type of an emitted value. Tests build values from earlier ones.
    #[cfg(test)]
    pub(crate) fn peek_type(&self, value: ValueId) -> TraceType {
        self.ir.values[value.index()].ty
    }

    /// The type of a value defined so far, if it exists.
    #[must_use]
    pub fn value_type(&self, value: ValueId) -> Option<TraceType> {
        self.ir.values.get(value.index()).map(|info| info.ty)
    }

    /// Number of effects emitted so far, which is the effect position a
    /// snapshot taken now must record.
    #[must_use]
    pub const fn effect_position(&self) -> u32 {
        self.effects
    }

    /// Appends an operation and returns the value it defines.
    ///
    /// `result` is the type of the defined value, or `None` for operations
    /// that define none. Guards take their `GuardId` from [`Self::guard`].
    pub fn emit(
        &mut self,
        site: TraceSite,
        op: TraceOp,
        result: Option<TraceType>,
    ) -> Option<ValueId> {
        let index = self.ir.body.len();
        let value = result.map(|ty| self.define(ty, ValueDef::Instruction(index)));
        self.ir.body.push(TraceInstruction {
            site,
            result: value,
            op,
        });
        value
    }

    /// Allocates the next effect id.
    pub fn next_effect(&mut self) -> EffectId {
        let id = EffectId(self.effects);
        self.effects += 1;
        id
    }

    /// Allocates a guard id that restores `snapshot`.
    pub fn guard(&mut self, site: TraceSite, snapshot: SnapshotId) -> GuardId {
        let origin = u32::try_from(self.ir.guards.len()).expect("guard count fits u32");
        self.ir.guards.push(GuardInfo {
            snapshot,
            site,
            origin,
        });
        GuardId(origin)
    }

    /// Finishes the trace with a terminator.
    #[must_use]
    pub fn finish(mut self, terminator: TraceTerminator) -> TraceIr {
        self.ir.terminator = terminator;
        self.ir
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::{NativeBinary, NativeUnary};
    use serde::Deserialize;
    use serde::de::value::{Error, U64Deserializer, UnitDeserializer};

    #[test]
    fn native_deserialization_installs_safe_placeholders() {
        let unary = NativeUnary::deserialize(UnitDeserializer::<Error>::new()).unwrap();
        let binary = NativeBinary::deserialize(UnitDeserializer::<Error>::new()).unwrap();
        assert!((unary.0)(1.0).is_nan());
        assert!((binary.0)(1.0, 2.0).is_nan());
    }

    #[test]
    fn native_deserialization_rejects_serialized_addresses() {
        assert!(NativeUnary::deserialize(U64Deserializer::<Error>::new(123)).is_err());
        assert!(NativeBinary::deserialize(U64Deserializer::<Error>::new(123)).is_err());
    }
}
