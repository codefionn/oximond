//! Runtime error type and result alias.

use crate::runtime::number_format::NumberFormat;
use crate::runtime::value::{
    CellHandle, ClosureHandle, FileHandle, GmatchIteratorHandle, NativeFunctionHandle, TableHandle,
    UserdataHandle, Value,
};
use std::error::Error;
use std::fmt;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeError {
    /// A native allocator failed. Preserve Lua's memory-error status.
    Memory,
    /// Carries hook-error provenance across a reentrant native boundary.
    #[doc(hidden)]
    Hook(Box<RuntimeError>),
    InvalidTable(TableHandle),
    InvalidClosure(ClosureHandle),
    InvalidCell(CellHandle),
    InvalidFile(FileHandle),
    InvalidUserdata(UserdataHandle),
    InvalidGmatchIterator(GmatchIteratorHandle),
    InvalidNativeFunction(NativeFunctionHandle),
    InvalidTableKey(&'static str),
    InvalidTableContinuationKey,
    TypeError {
        operation: &'static str,
        expected: &'static str,
        found: &'static str,
    },
    /// Argument `position`, counted from 1, of a library function is not
    /// of the `expected` type. `found` is a type name, or `"no value"` for
    /// a missing argument. The VM names the function after its call; alone
    /// the error reads as Lua writes it for a function it cannot name.
    BadArgument {
        position: usize,
        expected: &'static str,
        found: &'static str,
    },
    /// Argument `position` of a library function is invalid for the reason
    /// in `message`, such as `"value expected"`. The VM names the function
    /// as for [`Self::BadArgument`].
    ArgumentError {
        position: usize,
        message: String,
    },
    /// Lua's `error(nil)`. Protected calls must preserve this as a nil error
    /// object instead of converting it to the diagnostic string `"nil"`.
    RaisedNil,
    /// A Lua error object which must survive protected calls unchanged.
    Raised(Value),
    /// A Lua error object with the stack level requested by `error`.
    RaisedAtLevel {
        value: Value,
        level: usize,
    },
    /// An operating-system file operation failed. Lua exposes these as a
    /// `nil, message, errno` result tuple instead of raising an error.
    FileIo {
        message: String,
        code: i32,
    },
    /// A configured host standard stream rejected a write or flush. This is
    /// kept distinct from Lua file errors so an embedding CLI can report its
    /// own output failure accurately.
    HostOutputIo {
        message: String,
        code: i32,
    },
    Message(String),
    /// An error a Lua 5.2 or 5.3 finalizer raised, already wrapped as
    /// `error in __gc metamethod (message)`. Stock Lua marks it with
    /// `LUA_ERRGCMM`, so a later `GCTM` passes it on without wrapping again.
    FinalizerWrapped(String),
    /// A library native that needs VM services (metamethod dispatch, calls
    /// into Lua) declines the call before it has any effect. The VM's call
    /// paths then run it again through [`DirectNativeKind`](oximond::runtime::DirectNativeKind)
    /// dispatch. Direct runtime callers see this as an error.
    Redispatch,
    /// Lua's `os.exit` asked the program to end. No protected call, coroutine,
    /// or finalizer catches it, and unwinding skips `__close` handlers unless
    /// the request closes the state. The host decides what exiting means: the
    /// `oximond` command exits the process with the request's status.
    Exit(ExitRequest),
}

/// The status and close flag Lua passed to `os.exit`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitRequest {
    /// The C `int` status, after Lua maps `true` and `false` to
    /// `EXIT_SUCCESS` and `EXIT_FAILURE`. A process keeps its low 8 bits.
    pub status: i32,
    /// Whether Lua asked to close the state first (Lua 5.2 and later, and
    /// LuaJIT). Unwinding the main thread then runs its pending `__close`
    /// handlers, and the host should run the remaining finalizers, as
    /// `Vm::run_close_finalizers` does, before it exits.
    pub close: bool,
}

impl RuntimeError {
    #[must_use]
    pub const fn memory_error() -> Self {
        Self::Memory
    }

    #[must_use]
    pub const fn is_memory_error(&self) -> bool {
        match self {
            Self::Memory => true,
            Self::Hook(error) => error.is_memory_error(),
            _ => false,
        }
    }

    #[must_use]
    pub const fn is_hook_error(&self) -> bool {
        matches!(self, Self::Hook(_))
    }

    #[must_use]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Attributes a type error to argument `position` of a library function.
    /// Other errors are unchanged.
    #[must_use]
    pub fn at_argument(self, position: usize) -> Self {
        match self {
            Self::TypeError {
                expected, found, ..
            }
            | Self::BadArgument {
                expected, found, ..
            } => Self::BadArgument {
                position,
                expected,
                found,
            },
            error => error,
        }
    }

    /// Moves an argument error `offset` positions right, as for a method
    /// whose receiver the failing check did not count.
    #[must_use]
    pub fn shift_argument(self, offset: usize) -> Self {
        match self {
            Self::BadArgument {
                position,
                expected,
                found,
            } => Self::BadArgument {
                position: position + offset,
                expected,
                found,
            },
            Self::ArgumentError { position, message } => Self::ArgumentError {
                position: position + offset,
                message,
            },
            error => error,
        }
    }

    /// A [`Self::ArgumentError`] at argument `position`.
    #[must_use]
    pub fn argument_error(position: usize, message: impl Into<String>) -> Self {
        Self::ArgumentError {
            position,
            message: message.into(),
        }
    }

    /// The argument position and the parenthesized detail of an argument
    /// error, as `luaL_argerror` spells them.
    #[must_use]
    pub fn argument_detail(&self) -> Option<(usize, String)> {
        match self {
            Self::BadArgument {
                position,
                expected,
                found,
            } => Some((*position, format!("{expected} expected, got {found}"))),
            Self::ArgumentError { position, message } => Some((*position, message.clone())),
            _ => None,
        }
    }

    /// A [`Self::BadArgument`] for `argument`, which is `None` when the
    /// caller passed fewer arguments.
    #[must_use]
    pub fn bad_argument(position: usize, expected: &'static str, argument: Option<&Value>) -> Self {
        Self::BadArgument {
            position,
            expected,
            found: argument.map_or("no value", Value::type_name),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => formatter.write_str("not enough memory"),
            Self::Redispatch => formatter.write_str("native function needs the VM"),
            Self::Hook(error) => error.fmt(formatter),
            Self::InvalidTable(handle) => {
                write!(formatter, "invalid table handle {}", handle.index())
            }
            Self::InvalidClosure(handle) => {
                write!(formatter, "invalid closure handle {}", handle.index())
            }
            Self::InvalidCell(handle) => {
                write!(formatter, "invalid cell handle {}", handle.index())
            }
            Self::InvalidFile(handle) => {
                write!(formatter, "invalid file handle {}", handle.index())
            }
            Self::InvalidUserdata(handle) => {
                write!(formatter, "invalid userdata handle {}", handle.index())
            }
            Self::InvalidGmatchIterator(handle) => {
                write!(
                    formatter,
                    "invalid gmatch iterator handle {}",
                    handle.index()
                )
            }
            Self::InvalidNativeFunction(handle) => write!(
                formatter,
                "invalid native function handle {}",
                handle.index()
            ),
            Self::InvalidTableKey("number") => formatter.write_str("table index is NaN"),
            Self::InvalidTableKey(kind) => write!(formatter, "table index is {kind}"),
            Self::InvalidTableContinuationKey => formatter.write_str("invalid key to 'next'"),
            Self::TypeError {
                operation,
                expected,
                found,
            } => write!(
                formatter,
                "attempt to {operation} a {found} value (expected {expected})"
            ),
            Self::BadArgument {
                position,
                expected,
                found,
            } => write!(
                formatter,
                "bad argument #{position} to '?' ({expected} expected, got {found})"
            ),
            Self::ArgumentError { position, message } => {
                write!(formatter, "bad argument #{position} to '?' ({message})")
            }
            Self::RaisedNil | Self::Raised(Value::Nil) => formatter.write_str("nil"),
            Self::Raised(Value::String(value)) => formatter.write_str(&value.to_lossy()),
            Self::Raised(Value::Integer(value)) => write!(formatter, "{value}"),
            Self::Raised(Value::Number(value)) => write_number(formatter, *value),
            Self::Raised(Value::Boolean(value)) => write!(formatter, "{value}"),
            Self::Raised(value) => write!(formatter, "{}", value.type_name()),
            Self::RaisedAtLevel { value, .. } => match value {
                Value::String(value) => formatter.write_str(&value.to_lossy()),
                Value::Integer(value) => write!(formatter, "{value}"),
                Value::Number(value) => write_number(formatter, *value),
                Value::Boolean(value) => write!(formatter, "{value}"),
                Value::Nil => formatter.write_str("nil"),
                value => write!(formatter, "{}", value.type_name()),
            },
            Self::FileIo { message, .. }
            | Self::HostOutputIo { message, .. }
            | Self::FinalizerWrapped(message)
            | Self::Message(message) => formatter.write_str(message),
            Self::Exit(request) => write!(formatter, "os.exit({}) called", request.status),
        }
    }
}
impl Error for RuntimeError {}

/// Uncaught error values carry no profile, so they use the `"%.14g"`
/// spelling every stock Lua before 5.5 shares.
fn write_number(formatter: &mut fmt::Formatter<'_>, value: f64) -> fmt::Result {
    formatter.write_str(&String::from_utf8_lossy(
        NumberFormat::default().format(value).as_bytes(),
    ))
}
