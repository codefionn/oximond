//! Lua values, runtime handle newtypes, builtins, and native callback types.

use std::sync::Arc;

use crate::runtime::arena::ArenaKey;
use crate::runtime::error::RuntimeResult;
use crate::runtime::runtime::Runtime;
use crate::string::LuaString;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TableHandle(pub(crate) ArenaKey);

impl TableHandle {
    /// Restores an opaque native token. Runtime operations still validate its
    /// slot and generation, just as they validate any other table handle.
    #[must_use]
    pub const fn from_native_identity(identity: u64) -> Self {
        Self(ArenaKey {
            slot: (identity & 0xffff_ffff) as u32,
            generation: (identity >> 32) as u32,
        })
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }

    /// Returns a generation-aware identity for short-lived native guards.
    #[must_use]
    pub const fn native_identity(self) -> u64 {
        ((self.0.generation as u64) << 32) | self.0.slot as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeFunctionHandle(pub(crate) u32);

/// Standard-library callbacks the VM and JIT recognize by handle.
///
/// Only the exact installed handle matches, so a Lua program that replaces
/// the library field takes ordinary call dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectNativeKind {
    /// `string.find`, eligible for guarded plain-find calls from native code.
    Find,
    /// `table.concat`, eligible for guarded calls from native code.
    TableConcat,
    /// The `table` functions that honour `__index`, `__newindex` and `__len`
    /// from Lua 5.3 on. Their natives decline a table with such a metamethod
    /// (see `RuntimeError::Redispatch`) and the VM runs the call itself.
    TableInsert,
    TableRemove,
    TableUnpack,
    TableMove,
    /// LuaJIT's `string.format`, which declines a `%s` argument with `__tostring`.
    StringFormatLuaJit,
    /// LuaJIT's `string.format` without the `Redispatch` check.
    StringFormatLuaJitRaw,
    /// `print`, which declines an argument with a `__tostring` metamethod.
    Print,
    /// The raw formatter behind the VM's modern `string.format` builtin.
    Format,
    /// The string metatable's arithmetic metamethods from Lua 5.4. Like
    /// lstrlib's `trymt`, the VM passes an operation on a string and an
    /// operand with a metamethod of its own to that operand's metamethod.
    StringAdd,
    StringSub,
    StringMul,
    StringDiv,
    StringMod,
    StringPow,
    StringUnm,
    StringFloorDiv,
    /// `string.byte`, which a trace reads from a bound string.
    StringByte,
    /// `string.sub`, which a trace reads one byte of.
    StringSlice,
    /// `tonumber`, which a trace calls natively for a number or a 64-bit
    /// integer.
    Tonumber,
    /// The `bit` library functions a trace calls natively for numbers and
    /// 64-bit integers.
    BitTobit,
    BitBnot,
    BitBand,
    BitBor,
    BitBxor,
    BitLshift,
    BitRshift,
    BitArshift,
    BitRol,
    BitRor,
    /// `string.char`, which a trace calls through the VM for integers.
    StringChar,
    /// `string.len`, which a trace calls through the VM for a string.
    StringLen,
    /// `math.max` and `math.min`, which a trace calls through the VM for two
    /// integers.
    MathMax,
    MathMin,
    /// `type`, `rawequal`, `select` and `assert`, which a trace records as
    /// operations of its own with no call.
    Type,
    RawEqual,
    Select,
    Assert,
    /// `error`, which a trace inside a protected call records as a caught
    /// exit.
    Error,
    /// `math.floor`, `math.ceil`, `math.abs`, `math.tointeger` and
    /// `math.type`, recorded the same way.
    MathFloor,
    MathCeil,
    MathAbs,
    MathToInteger,
    MathType,
    /// `rawget`, `rawset`, `rawlen`, `getmetatable` and `setmetatable`, which
    /// a trace records as table operations.
    RawGet,
    RawSet,
    RawLen,
    GetMetatable,
    SetMetatable,
    /// `unpack` of Lua 5.1, LuaJIT and Lua 5.2's `table.unpack`, which read
    /// the table raw. (Lua 5.3 and later have [`Self::TableUnpack`].)
    Unpack,
    /// The `bit32` library functions, which a trace records as integer
    /// operations on 32-bit words.
    Bit32Band,
    Bit32Bor,
    Bit32Bxor,
    Bit32Bnot,
    Bit32Lshift,
    Bit32Rshift,
    Bit32Arshift,
    Bit32Lrotate,
    Bit32Rrotate,
    Bit32Btest,
    Bit32Extract,
    Bit32Replace,
    /// `table.insert` and `table.remove` of the profiles before Lua 5.3,
    /// which read and write the elements raw. (Lua 5.3 and later have
    /// [`Self::TableInsert`] and [`Self::TableRemove`].)
    TableInsertRaw,
    TableRemoveRaw,
    /// `string.rep`, `string.upper`, `string.lower` and `string.reverse`,
    /// which a trace calls through helpers for strings.
    StringRep,
    StringUpper,
    StringLower,
    StringReverse,
    /// `math.random`, whose generator a trace advances through a helper.
    MathRandom,
    /// `math.modf`, which a trace calls through a helper for a number.
    MathModf,
    /// `math.log` and `math.pow`, which a trace computes on floats. The
    /// logarithm of Lua 5.2 and later takes a base.
    MathLog,
    MathLogBase,
    MathPow,
    /// `string.match`, which a trace calls through helpers.
    Match,
    /// `string.gmatch`, whose iterator a trace makes and steps through
    /// helpers.
    Gmatch,
    /// The `utf8` functions a trace calls through helpers. The step of
    /// `utf8.codes` has a native per mode: the strict one, Lua 5.3's and the
    /// lax one.
    Utf8Char,
    Utf8Codepoint,
    Utf8Len,
    Utf8Offset,
    Utf8Codes,
    Utf8Next,
    Utf8Next53,
    Utf8NextLax,
    /// `string.pack`, `string.unpack` and `string.packsize`, which a trace
    /// calls through helpers.
    StringPack,
    StringUnpack,
    StringPacksize,
}

impl DirectNativeKind {
    pub(crate) const COUNT: usize = 87;

    /// Every kind, in declaration order.
    pub(crate) const ALL: [Self; Self::COUNT] = [
        Self::Find,
        Self::TableConcat,
        Self::TableInsert,
        Self::TableRemove,
        Self::TableUnpack,
        Self::TableMove,
        Self::StringFormatLuaJit,
        Self::StringFormatLuaJitRaw,
        Self::Print,
        Self::Format,
        Self::StringAdd,
        Self::StringSub,
        Self::StringMul,
        Self::StringDiv,
        Self::StringMod,
        Self::StringPow,
        Self::StringUnm,
        Self::StringFloorDiv,
        Self::StringByte,
        Self::StringSlice,
        Self::Tonumber,
        Self::BitTobit,
        Self::BitBnot,
        Self::BitBand,
        Self::BitBor,
        Self::BitBxor,
        Self::BitLshift,
        Self::BitRshift,
        Self::BitArshift,
        Self::BitRol,
        Self::BitRor,
        Self::StringChar,
        Self::StringLen,
        Self::MathMax,
        Self::MathMin,
        Self::Type,
        Self::RawEqual,
        Self::Select,
        Self::Assert,
        Self::Error,
        Self::MathFloor,
        Self::MathCeil,
        Self::MathAbs,
        Self::MathToInteger,
        Self::MathType,
        Self::RawGet,
        Self::RawSet,
        Self::RawLen,
        Self::GetMetatable,
        Self::SetMetatable,
        Self::Unpack,
        Self::Bit32Band,
        Self::Bit32Bor,
        Self::Bit32Bxor,
        Self::Bit32Bnot,
        Self::Bit32Lshift,
        Self::Bit32Rshift,
        Self::Bit32Arshift,
        Self::Bit32Lrotate,
        Self::Bit32Rrotate,
        Self::Bit32Btest,
        Self::Bit32Extract,
        Self::Bit32Replace,
        Self::TableInsertRaw,
        Self::TableRemoveRaw,
        Self::StringRep,
        Self::StringUpper,
        Self::StringLower,
        Self::StringReverse,
        Self::MathRandom,
        Self::MathModf,
        Self::MathLog,
        Self::MathLogBase,
        Self::MathPow,
        Self::Match,
        Self::Gmatch,
        Self::Utf8Char,
        Self::Utf8Codepoint,
        Self::Utf8Len,
        Self::Utf8Offset,
        Self::Utf8Codes,
        Self::Utf8Next,
        Self::Utf8Next53,
        Self::Utf8NextLax,
        Self::StringPack,
        Self::StringUnpack,
        Self::StringPacksize,
    ];

    /// Whether this is one of the string metatable's arithmetic metamethods.
    #[must_use]
    pub const fn is_string_arithmetic(self) -> bool {
        matches!(
            self,
            Self::StringAdd
                | Self::StringSub
                | Self::StringMul
                | Self::StringDiv
                | Self::StringMod
                | Self::StringPow
                | Self::StringUnm
                | Self::StringFloorDiv
        )
    }
}

impl NativeFunctionHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// The handle of registry entry `index`, for native code that kept the
    /// index. Runtime operations check it against the registry.
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// A stable identity for a Lua closure body owned by the runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClosureHandle(pub(crate) ArenaKey);

impl ClosureHandle {
    /// Restores an opaque native token. Closure operations validate its slot
    /// and generation before accessing runtime storage.
    #[must_use]
    pub const fn from_native_identity(identity: u64) -> Self {
        Self(ArenaKey {
            slot: (identity & 0xffff_ffff) as u32,
            generation: (identity >> 32) as u32,
        })
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }

    /// Returns a generation-aware identity for short-lived native guards.
    #[must_use]
    pub const fn native_identity(self) -> u64 {
        ((self.0.generation as u64) << 32) | self.0.slot as u64
    }
}

/// An opaque identity for a captured Lua value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellHandle(pub(crate) ArenaKey);

impl Default for CellHandle {
    /// A handle that names no cell: the arena never hands out its slot.
    /// For filling arrays that are overwritten before they are read.
    fn default() -> Self {
        Self(ArenaKey {
            slot: u32::MAX,
            generation: 0,
        })
    }
}

impl CellHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }

    /// Restores a handle from its [`Self::native_identity`]. Runtime
    /// operations still validate its slot and generation.
    #[must_use]
    pub const fn from_native_identity(identity: u64) -> Self {
        Self(ArenaKey {
            slot: (identity & 0xffff_ffff) as u32,
            generation: (identity >> 32) as u32,
        })
    }

    /// The handle's slot and generation in one word.
    #[must_use]
    pub const fn native_identity(self) -> u64 {
        ((self.0.generation as u64) << 32) | self.0.slot as u64
    }
}

/// An opaque identity for a runtime-owned file object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FileHandle(pub(crate) ArenaKey);

impl FileHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }
}

/// An opaque identity for a Lua userdata value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserdataHandle(pub(crate) ArenaKey);

impl UserdataHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }
}

/// An opaque identity for the state behind a `string.gmatch` iterator.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GmatchIteratorHandle(pub(crate) ArenaKey);

impl GmatchIteratorHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }
}

/// A stable identity for a Lua coroutine owned by the VM.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ThreadHandle(pub(crate) ArenaKey);

impl ThreadHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.slot
    }

    /// Returns a generation-aware identity for short-lived native guards.
    #[must_use]
    pub const fn native_identity(self) -> u64 {
        ((self.0.generation as u64) << 32) | self.0.slot as u64
    }
}

/// Builtins whose control flow is implemented by the VM rather than a native
/// Rust callback.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BuiltinFunction {
    /// Calls a function and converts a raised runtime error into results.
    ProtectedCall,
    /// Calls a function and sends a raised value to an error handler.
    ExtendedProtectedCall,
    /// Advances a table traversal.
    Next,
    /// Produces the generic-for triple backed by [`Self::Next`].
    Pairs,
    /// Produces the array-style generic-for triple.
    IPairs,
    /// Advances one array-style generic-for traversal.
    IPairsIterator,
    /// Compiles source through a loader configured on the VM.
    Load,
    /// Lua 5.1's `loadstring`: [`Self::Load`] that takes only a string.
    LoadString,
    /// Reads and compiles a file through the VM-owned source loader.
    LoadFile,
    /// Loads and executes a file through the VM-owned source loader.
    DoFile,
    /// Sorts a table through a VM continuation so Lua comparators can yield results.
    TableSort,
    /// Traverses every table entry through a VM continuation for callbacks.
    TableForeach,
    /// Traverses a table's contiguous list entries through a VM continuation for callbacks.
    TableForeachi,
    /// Inspects VM-owned call and function metadata.
    DebugGetInfo,
    /// Formats the active Lua call stack for the debug library.
    DebugTraceback,
    /// Serializes a VM-owned Lua closure to bytecode bytes.
    StringDump,
    /// Replaces pattern matches, resuming through the VM for Lua callbacks.
    StringGsub,
    /// Converts values through `__tostring`, resuming through the VM for Lua callbacks.
    ToString,
    /// Formats strings while allowing `%s` to invoke Lua `__tostring` callbacks.
    StringFormat,
    /// Replaces a VM-owned closure upvalue by descriptor index.
    DebugSetupValue,
    /// Installs or clears the execution hook owned by a Lua thread.
    DebugSetHook,
    /// Returns the execution hook owned by a Lua thread.
    DebugGetHook,
    /// Reads a local value from an active Lua frame.
    DebugGetLocal,
    /// Replaces a local value in an active Lua frame.
    DebugSetLocal,
    /// Reads a VM-owned closure upvalue by descriptor index.
    DebugGetUpvalue,
    /// Returns an opaque identity for a VM-owned closure upvalue.
    DebugUpvalueId,
    /// Makes one VM-owned closure upvalue share another closure's cell.
    DebugUpvalueJoin,
    /// Returns the runtime's Lua registry table.
    DebugGetRegistry,
    /// Reads a userdata's uservalue slot.
    DebugGetUserValue,
    /// Replaces a userdata's uservalue slot.
    DebugSetUserValue,
    /// Runs the interactive debug console.
    DebugDebug,
    /// Searches a module path using the host's source reader.
    PackageSearchPath,
    /// Finds and compiles a source module for package searchers.
    PackageSourceSearcher,
    /// Reads the environment associated with a function or stack level.
    GetFenv,
    /// Replaces the environment associated with a function or stack level.
    SetFenv,
    /// Runs the VM-coordinated garbage collector.
    CollectGarbage,
    /// Writes bytes to a runtime-owned temporary file.
    FileWrite,
    /// Moves a runtime-owned temporary file cursor.
    FileSeek,
    /// Reads bytes from a runtime-owned temporary file.
    FileRead,
    /// Closes a runtime-owned temporary file.
    FileClose,
    /// Flushes a runtime-owned file.
    FileFlush,
    /// Changes a runtime-owned file's buffering mode.
    FileSetVbuf,
    /// Creates an iterator over a runtime-owned file's lines.
    FileLines,
    /// Loads and executes a package through the VM-owned source loader.
    Require,
    /// Creates or reopens a Lua 5.1 module and changes the caller environment.
    Module,
    /// Creates a suspended Lua coroutine.
    CoroutineCreate,
    /// Resumes a suspended Lua coroutine.
    CoroutineResume,
    /// Suspends the currently running coroutine.
    CoroutineYield,
    /// Reports a coroutine's execution state.
    CoroutineStatus,
    /// Closes a suspended Lua coroutine and releases its VM frames.
    CoroutineClose,
    /// Creates an error-propagating callable coroutine wrapper.
    CoroutineWrap,
    /// Returns the currently running coroutine, or nil on the main thread.
    CoroutineRunning,
    /// Reports whether a thread can yield.
    CoroutineIsYieldable,
}

impl BuiltinFunction {
    /// The library-qualified name (`pcall`, `string.gsub`) the builtin is
    /// installed under, for the statistics and messages that tell builtins
    /// apart.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ProtectedCall => "pcall",
            Self::ExtendedProtectedCall => "xpcall",
            Self::Next => "next",
            Self::Pairs => "pairs",
            Self::IPairs => "ipairs",
            Self::IPairsIterator => "ipairs.iterator",
            Self::Load => "load",
            Self::LoadString => "loadstring",
            Self::LoadFile => "loadfile",
            Self::DoFile => "dofile",
            Self::TableSort => "table.sort",
            Self::TableForeach => "table.foreach",
            Self::TableForeachi => "table.foreachi",
            Self::DebugGetInfo => "debug.getinfo",
            Self::DebugTraceback => "debug.traceback",
            Self::StringDump => "string.dump",
            Self::StringGsub => "string.gsub",
            Self::ToString => "tostring",
            Self::StringFormat => "string.format",
            Self::DebugSetupValue => "debug.setupvalue",
            Self::DebugSetHook => "debug.sethook",
            Self::DebugGetHook => "debug.gethook",
            Self::DebugGetLocal => "debug.getlocal",
            Self::DebugSetLocal => "debug.setlocal",
            Self::DebugGetUpvalue => "debug.getupvalue",
            Self::DebugUpvalueId => "debug.upvalueid",
            Self::DebugUpvalueJoin => "debug.upvaluejoin",
            Self::DebugGetRegistry => "debug.getregistry",
            Self::DebugGetUserValue => "debug.getuservalue",
            Self::DebugSetUserValue => "debug.setuservalue",
            Self::DebugDebug => "debug.debug",
            Self::PackageSearchPath => "package.searchpath",
            Self::PackageSourceSearcher => "package.searchers.source",
            Self::GetFenv => "getfenv",
            Self::SetFenv => "setfenv",
            Self::CollectGarbage => "collectgarbage",
            Self::FileWrite => "file:write",
            Self::FileSeek => "file:seek",
            Self::FileRead => "file:read",
            Self::FileClose => "file:close",
            Self::FileFlush => "file:flush",
            Self::FileSetVbuf => "file:setvbuf",
            Self::FileLines => "file:lines",
            Self::Require => "require",
            Self::Module => "module",
            Self::CoroutineCreate => "coroutine.create",
            Self::CoroutineResume => "coroutine.resume",
            Self::CoroutineYield => "coroutine.yield",
            Self::CoroutineStatus => "coroutine.status",
            Self::CoroutineClose => "coroutine.close",
            Self::CoroutineWrap => "coroutine.wrap",
            Self::CoroutineRunning => "coroutine.running",
            Self::CoroutineIsYieldable => "coroutine.isyieldable",
        }
    }
}

/// The explicit tagged representation used by the initial interpreter.
///
/// `repr(C, u8)` places every payload at offset 8. With mixed payload
/// offsets, copying a value compiled to a byte-blob copy through the stack
/// whose wide reloads cannot forward from the narrow stores that built it;
/// that stall dominated register moves in the interpreter.
#[derive(Clone, Debug, PartialEq)]
#[repr(C, u8)]
pub enum Value {
    Nil,
    Boolean(bool),
    Integer(i64),
    Number(f64),
    String(LuaString),
    Table(TableHandle),
    File(FileHandle),
    Userdata(UserdataHandle),
    GmatchIterator(GmatchIteratorHandle),
    Thread(ThreadHandle),
    CoroutineWrapper(ThreadHandle),
    Closure(ClosureHandle),
    NativeFunction(NativeFunctionHandle),
    BuiltinFunction(BuiltinFunction),
    // Appended last: native views and the JIT depend on the earlier tags.
    /// LuaJIT's `int64_t` cdata, immediate. See [`oximond::runtime::cint64`].
    Int64(i64),
    /// LuaJIT's `uint64_t` cdata, immediate. See [`oximond::runtime::cint64`].
    UInt64(u64),
}

impl Value {
    /// The table handle of a value that holds a table, loaded as the one
    /// word it is. Matching on the value loads the handle's slot and its
    /// generation as two four-byte halves, and a half of a word stored whole
    /// (a register written by a move or an argument copy) does not forward
    /// from the store buffer.
    #[must_use]
    #[inline(always)]
    #[allow(unsafe_code)]
    pub fn table_handle_word(&self) -> Option<TableHandle> {
        if !matches!(self, Self::Table(_)) {
            return None;
        }
        // SAFETY: the value holds a table, whose payload is the handle, in
        // the word at offset 8 of this `repr(C, u8)` enum: the slot in the
        // low half and the generation in the high half, as
        // `TableHandle::native_identity` builds the word (the test
        // `table_handle_is_the_payload_word` checks it).
        let identity = unsafe {
            std::ptr::from_ref(self)
                .cast::<u8>()
                .add(8)
                .cast::<u64>()
                .read()
        };
        Some(TableHandle::from_native_identity(identity))
    }

    #[must_use]
    pub const fn is_truthy(&self) -> bool {
        !matches!(self, Self::Nil | Self::Boolean(false))
    }

    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Nil => "nil",
            Self::Boolean(_) => "boolean",
            Self::Integer(_) | Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Table(_) => "table",
            Self::File(_) | Self::Userdata(_) => "userdata",
            Self::Thread(_) => "thread",
            Self::Int64(_) | Self::UInt64(_) => "cdata",
            Self::Closure(_)
            | Self::NativeFunction(_)
            | Self::BuiltinFunction(_)
            | Self::GmatchIterator(_)
            | Self::CoroutineWrapper(_) => "function",
        }
    }

    #[must_use]
    pub fn string(value: impl AsRef<str>) -> Self {
        Self::String(LuaString::from(value.as_ref()))
    }

    /// Creates a Lua string from arbitrary bytes.
    #[must_use]
    pub fn bytes(value: impl Into<LuaString>) -> Self {
        Self::String(value.into())
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}
impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}
impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::string(value)
    }
}
impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::string(value)
    }
}
impl From<LuaString> for Value {
    fn from(value: LuaString) -> Self {
        Self::String(value)
    }
}
impl From<Vec<u8>> for Value {
    fn from(value: Vec<u8>) -> Self {
        Self::bytes(value)
    }
}
impl From<&[u8]> for Value {
    fn from(value: &[u8]) -> Self {
        Self::bytes(value)
    }
}

pub type NativeCallback = fn(&mut Runtime, &[Value]) -> RuntimeResult<Vec<Value>>;
/// A native function that returns exactly one Lua value without a result vector.
pub type NativeScalarCallback = fn(&mut Runtime, &[Value]) -> RuntimeResult<Value>;
/// Up to two result values of a native, returned by value so that no
/// vector is allocated.
pub struct NativePair {
    count: usize,
    values: [Value; 2],
}

impl NativePair {
    /// One value.
    #[must_use]
    pub const fn one(value: Value) -> Self {
        Self {
            count: 1,
            values: [value, Value::Nil],
        }
    }

    /// Two values.
    #[must_use]
    pub const fn two(first: Value, second: Value) -> Self {
        Self {
            count: 2,
            values: [first, second],
        }
    }

    /// The values returned.
    #[must_use]
    pub fn as_slice(&self) -> &[Value] {
        &self.values[..self.count]
    }

    /// The number of values returned.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether no value is returned.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Takes the value at `position` out, leaving nil, or nil past the end.
    pub fn take(&mut self, position: usize) -> Value {
        if position < self.count {
            std::mem::replace(&mut self.values[position], Value::Nil)
        } else {
            Value::Nil
        }
    }
}

/// The fast form of a multi-result native: `Ok(None)` declines a call that
/// needs the general callback, which then runs in its place; otherwise the
/// result is what the callback would return, in a fixed buffer.
pub type NativePairCallback = fn(&mut Runtime, &[Value]) -> RuntimeResult<Option<NativePair>>;
/// A pure native numeric operation callable directly by generated code.
///
/// This ABI only accepts and returns `f64`; argument conversion, Lua errors,
/// and normal callback dispatch remain the responsibility of its scalar
/// fallback callback.
pub type NativeNumericUnary = extern "C" fn(f64) -> f64;
/// What a pure native answers for arguments, without running it: the value
/// its callback returns, or `None` when the call raises an error or is not
/// one the probe predicts. A trace recorder uses it to learn the type of a
/// result before the interpreter makes the call.
pub type NativeProbe = fn(&Runtime, &[Value]) -> Option<Value>;
/// [`NativeProbe`] for a native with any number of results: all of them, or
/// `None` when the call raises an error. The probe and the callback are one
/// piece of code, so a trace that runs the probe on its arguments knows the
/// exact answer of the call.
pub type NativeResultsProbe = fn(&Runtime, &[Value]) -> Option<Vec<Value>>;
/// The most results a [`NativeResults`] holds.
pub const NATIVE_RESULTS_CAPACITY: usize = 6;

/// A few results of a native, kept in place instead of in a `Vec`, so that a
/// trace that asks a native for its answer on every pass allocates nothing
/// for the list. A native with more results than fit has no fixed answer.
pub struct NativeResults {
    values: [Value; NATIVE_RESULTS_CAPACITY],
    len: usize,
}

impl NativeResults {
    /// No results.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: [const { Value::Nil }; NATIVE_RESULTS_CAPACITY],
            len: 0,
        }
    }

    /// Adds a result. Gives it back when the list is full.
    ///
    /// # Errors
    ///
    /// Returns `value` when [`NATIVE_RESULTS_CAPACITY`] results are in.
    #[inline(always)]
    pub fn push(&mut self, value: Value) -> Result<(), Value> {
        match self.values.get_mut(self.len) {
            Some(slot) => {
                *slot = value;
                self.len += 1;
                Ok(())
            }
            None => Err(value),
        }
    }

    /// The results, in order.
    #[must_use]
    pub fn as_slice(&self) -> &[Value] {
        &self.values[..self.len]
    }
}

impl Default for NativeResults {
    fn default() -> Self {
        Self::new()
    }
}

/// [`NativeResultsProbe`] that adds the results to an empty [`NativeResults`]
/// the caller holds: the same answer, with no allocation and no list to move.
/// `false` when the call raises an error or has more results than a
/// [`NativeResults`] holds.
pub type NativeFixedResultsProbe = fn(&Runtime, &[Value], &mut NativeResults) -> bool;
/// A pure numeric callback for exactly two `Value::Number` arguments.
/// The scalar fallback handles every other argument shape and Lua errors.
pub type NativeNumericBinary = extern "C" fn(f64, f64) -> f64;
/// The pure form of a two-argument native for argument pairs other than two
/// numbers, such as the `bit` operations on 64-bit integers. It allocates
/// nothing and raises nothing: `None` leaves the call to the scalar callback.
pub(crate) type NativeWideBinary = fn(&Value, &Value) -> Option<Value>;

/// How the interpreter's native call path runs a registered native. The
/// single-result forms carry the pure numeric function when the native has
/// one: a caller holding exactly the right number of `Value::Number`
/// arguments may call it instead of the callback.
#[derive(Clone, Copy)]
pub enum NativePlan {
    /// Returns exactly one value.
    Scalar(NativeScalarCallback),
    /// Returns one value, and has a pure form on one `Value::Number`.
    Unary(NativeScalarCallback, NativeNumericUnary),
    /// Returns one value, and has a pure form on two `Value::Number`s.
    Binary(NativeScalarCallback, NativeNumericBinary),
    /// Returns any number of values, and has a form that returns up to two
    /// without a vector.
    Paired(NativeCallback, NativePairCallback),
    /// Returns any number of values.
    Multiple(NativeCallback),
    /// Needs the VM, or is not a plain callback.
    Other,
}

/// How a registered native runs when only the runtime is at hand.
#[derive(Clone, Copy)]
pub enum NativeCallKind {
    /// Returns exactly one value.
    Scalar(NativeScalarCallback),
    /// Returns any number of values.
    Multiple(NativeCallback),
    /// Needs the VM, or is not a plain callback.
    Other,
}

/// The single-result variants come first: telling them from the rest is
/// one comparison on the discriminant in the interpreter's native call path.
#[derive(Clone)]
pub(crate) enum NativeEntry {
    Scalar(NativeScalarCallback),
    NumericUnary {
        callback: NativeScalarCallback,
        unary: NativeNumericUnary,
    },
    NumericBinary {
        callback: NativeScalarCallback,
        binary: NativeNumericBinary,
        /// The pure form for other argument pairs, when the native has one.
        wide: Option<NativeWideBinary>,
    },
    Multiple(NativeCallback),
    /// A multi-result native with a fixed-buffer fast form.
    Paired {
        callback: NativeCallback,
        paired: NativePairCallback,
    },
    /// Dispatched by the owning VM. The runtime cannot invoke this entry alone.
    Host(u64),
    // The callback signature is the public shape of a host closure; keeping
    // it inline names its arguments at every construction site.
    #[allow(clippy::type_complexity)]
    Closure(Arc<dyn Fn(&mut Runtime, &[Value]) -> RuntimeResult<Vec<Value>>>),
    /// A single-result callback that only the VM's general call path may
    /// run, because the VM can run another function in its place.
    Dispatched(NativeScalarCallback),
}
