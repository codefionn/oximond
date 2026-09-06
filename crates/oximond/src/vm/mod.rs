#![deny(unsafe_code)]

//! The baseline match-based register bytecode interpreter.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::bytecode::load::{ChunkLoad, ChunkLoader};
use crate::bytecode::{CallKind, Chunk, OpCode, ValidationError};
use crate::compat::CompatibilityProfile;
use crate::compat::ExecutionMode;
use crate::compat::{TracingOptFlags, TracingOptLevel};
use crate::pattern::{Match as PatternMatch, Pattern};
use crate::runtime::{
    CaptureCells, ClosureHandle, ExitRequest, GcRoots, NativeFunctionHandle, Runtime, RuntimeError,
    TableHandle, ThreadHandle, Value,
};
use crate::string::LuaString;

pub type VmResult<T> = Result<T, VmError>;

/// Lua 5.5's compact collector parameter encoding, decoded as an exact value.
/// The C API and collectgarbage report the truncated integer part.
#[must_use]
pub fn quantize_lua55_gc_parameter(parameter: u32) -> f64 {
    let encoded = if parameter >= 396_800 {
        255
    } else {
        let scaled = (parameter * 128).div_ceil(100);
        if scaled < 16 {
            scaled
        } else {
            let log = scaled.ilog2() - 4;
            ((scaled >> log) - 16) | ((log + 1) << 4)
        }
    };
    let exponent = encoded >> 4;
    let mantissa = if exponent == 0 {
        encoded & 15
    } else {
        16 + (encoded & 15)
    };
    f64::from(mantissa << exponent.saturating_sub(1)) * (100.0 / 128.0)
}

fn initialize_gc_profile(runtime: &mut Runtime, compatibility: CompatibilityProfile) {
    if compatibility == CompatibilityProfile::Lua55 {
        // Lua 5.5's 64-bit Table is 48 bytes; its default step is 200 tables.
        for (index, value) in [20, 50, 70, 250, 200, 9_600].into_iter().enumerate() {
            runtime.set_gc_parameter(index, quantize_lua55_gc_parameter(value));
        }
    } else if compatibility == CompatibilityProfile::Lua54 {
        runtime.set_gc_parameter(0, 20.0);
        runtime.set_gc_parameter(2, 100.0);
    }
}

/// A host callback with access to the owning VM. The token identifies the
/// closure instance and can be used to look up host-owned state.
pub type HostCallback = fn(&mut Vm, u64, &[Value]) -> VmResult<Vec<Value>>;

/// A second entry of a host callback, for the calls that know where the
/// results go: `callback(vm, token, arguments, count, results, wanted)`
/// reads `count` arguments at `arguments` before it runs anything, and after
/// its last call into the VM stores exactly `wanted` values at `results`,
/// the first of what the callback returned and nil past them, each through
/// an assignment that drops the value it replaces. It does everything else
/// the [`HostCallback`] registered for the same function does.
///
/// The two ranges may overlap, which is why they are addresses: a fixed call
/// writes its results over the callee and its arguments. The entry spares
/// such a call the vector of results and the copy out of it.
pub type HostDirectCallback =
    unsafe fn(&mut Vm, u64, *const Value, usize, *mut Value, usize) -> VmResult<()>;

/// What the VM keeps of a host function besides its dispatch entry.
struct HostFunction {
    captures: Vec<Value>,
    _owner: Option<std::rc::Rc<dyn std::any::Any>>,
}

/// Debug information for one frame visible to a native host callback.
#[derive(Clone, Debug)]
pub struct VmStackFrame {
    pub function: Option<Value>,
    pub source: LuaString,
    pub short_source: LuaString,
    pub current_line: i32,
    pub what: &'static str,
    pub definition_line: i32,
    pub last_definition_line: i32,
    pub upvalue_count: usize,
    pub parameter_count: u8,
    pub is_vararg: bool,
    pub is_tail_call: bool,
    /// One-based first argument or result during a call or return hook.
    pub first_transfer: usize,
    /// Number of arguments or results visible to that hook.
    pub transfer_count: usize,
    pub name: Option<LuaString>,
    pub name_what: &'static str,
}

// Allocation between two automatic steps of an incremental major cycle. Each
// step does this many kilobytes of structural work scaled by the step
// multiplier: 256 units at the default. Steps between the cycle start and the
// atomic step need no root snapshot. Lua's explicit `step` remains exact.
const AUTOMATIC_GC_STEP_KILOBYTES: usize = 8;
/// Backward jumps between the interpreter's collector polls.
const BACKEDGE_GC_POLL_INTERVAL: u32 = 64;
const MAX_LUA_CALL_FRAMES: usize = 4_096;
/// Coroutines a resume may nest inside, like Lua's `LUAI_MAXCCALLS`. Each
/// resume runs the interpreter on the native stack.
const MAX_NESTED_RESUMES: usize = 200;

/// Closure call classifications stored in the runtime's per-closure hint.
/// Not yet classified: the general call path decides on first use.
const CALL_HINT_UNKNOWN: u8 = 0;
/// A fixed-parameter Lua body: the interpreter's fast loop pushes an
/// activation for a call, whatever plan flags below apply. The dispatch loop
/// and tracing mode may still evaluate its plan.
const CALL_HINT_PLAIN: u8 = 1;
/// An interpreter numeric inline plan exists for the closure.
const CALL_HINT_NUMERIC_PLAN: u8 = 2;
/// An interpreter table inline plan exists for the closure.
const CALL_HINT_TABLE_PLAN: u8 = 4;
/// Activations of this closure are built by the general call path (vararg
/// bodies that bind a named-vararg table). Alone, it also marks plans whose
/// refusal may be retried, so every call stays general.
const CALL_HINT_GENERAL: u8 = 8;
/// A vararg Lua body without a named-vararg table: the interpreter's fast
/// loop pushes its activation too, with the arguments past the parameters
/// stored on the value stack below the new window. Alone (no plan flag, no
/// `CALL_HINT_GENERAL`), it is the whole classification.
const CALL_HINT_VARARG: u8 = 16;
/// No closure: `ClosureHandle::native_identity` never has every bit set,
/// because the last two arena slots are never handed out.
const NO_PLAN_CALLEE: u32 = u32::MAX;

/// A pointer to the host's interrupt flag, or to a flag that never is set.
/// The VM keeps the `Arc` that owns the flag alive for as long as it holds
/// the pointer.
#[derive(Clone, Copy)]
struct InterruptPoll(std::ptr::NonNull<AtomicBool>);

// SAFETY: the pointer is only read, through an atomic, and the flag it
// points to is an `Arc<AtomicBool>` shared across threads or a static.
#[allow(unsafe_code)]
unsafe impl Send for InterruptPoll {}
// SAFETY: see the `Send` impl.
#[allow(unsafe_code)]
unsafe impl Sync for InterruptPoll {}

static NEVER_INTERRUPTED: AtomicBool = AtomicBool::new(false);

impl InterruptPoll {
    const NEVER: Self = Self(std::ptr::NonNull::from_ref(&NEVER_INTERRUPTED));

    fn of(flag: &Arc<AtomicBool>) -> Self {
        Self(std::ptr::NonNull::from_ref(&**flag))
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn pending(self) -> bool {
        // SAFETY: the pointer addresses a live `AtomicBool` (see the type).
        unsafe { self.0.as_ref() }.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug)]
pub enum VmError {
    /// A host callback completed by yielding its values to the resumer.
    Yield(Vec<Value>),
    #[doc(hidden)]
    CoroutineClosed,
    /// A debug hook raised this error. Message handlers run with hooks disabled.
    HookError(Box<VmError>),
    InvalidChunk(ValidationError),
    Runtime(RuntimeError),
    LocatedRuntime {
        source: LuaString,
        line: u32,
        error: RuntimeError,
    },
    InvalidGlobalName(u32),
    InvalidDispatch(OpCode),
    MissingPrototype(u16),
    MissingClosureBody(ClosureHandle),
    UnsupportedUpvalues,
    RegisterRange {
        start: u8,
        count: u8,
    },
    ProgramCounterOutOfBounds(usize),
}

impl fmt::Display for VmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yield(_) => write!(formatter, "coroutine yielded"),
            Self::CoroutineClosed => write!(formatter, "coroutine closed"),
            Self::HookError(error) => error.fmt(formatter),
            Self::InvalidChunk(error) => write!(formatter, "invalid bytecode: {error}"),
            Self::Runtime(error) => error.fmt(formatter),
            Self::LocatedRuntime {
                source,
                line,
                error,
            } => write!(
                formatter,
                "{}:{line}: {error}",
                display_chunk_source(source)
            ),
            Self::InvalidGlobalName(index) => {
                write!(formatter, "global name constant {index} is not a string")
            }
            Self::InvalidDispatch(opcode) => {
                write!(
                    formatter,
                    "opcode {opcode:?} reached an incompatible VM dispatch path"
                )
            }
            Self::MissingPrototype(index) => write!(formatter, "missing closure prototype {index}"),
            Self::MissingClosureBody(handle) => {
                write!(formatter, "closure {} has no VM body", handle.index())
            }
            Self::UnsupportedUpvalues => {
                formatter.write_str("closures with captured upvalues are not implemented")
            }
            Self::RegisterRange { start, count } => write!(
                formatter,
                "register range {start}..{} is outside the frame",
                usize::from(*start) + usize::from(*count)
            ),
            Self::ProgramCounterOutOfBounds(pc) => {
                write!(formatter, "program counter {pc} is outside bytecode")
            }
        }
    }
}

/// Observable counters for the VM's native execution policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JitStatistics {
    /// Fast-loop backedges that left for the dispatch loop's native lookup, which the fast
    /// loop could not settle itself.
    pub fast_backedge_lookups: u64,
    /// Fast-loop backedges that left the fast loop for the in-place settling of
    /// `resume_fast_backedge` (counting a cold header, re-arming the backedge skip). Those it
    /// cannot settle also count in `fast_backedge_lookups`.
    pub fast_backedge_resumes: u64,
    /// `CallOne` instructions evaluated directly from a validated numeric plan.
    pub interpreter_inline_calls: u64,
    /// Ordinary `Call` instructions evaluated by the staged raw-table plan.
    pub interpreter_inline_table_calls: u64,
    /// Fixed two-result `ipairs` iterator calls completed from registers.
    pub interpreter_ipairs_steps: u64,
    /// `coroutine.resume` calls the fast loop completed itself, with no
    /// dispatch through the general call path and no argument vector, and
    /// calls of functions `coroutine.wrap` made that it completed the same
    /// way.
    pub coroutine_direct_resumes: u64,
    /// `coroutine.yield` calls the fast loop completed itself.
    pub coroutine_direct_yields: u64,
    /// Resumes the fast loop made by running the thread in its own place:
    /// the stacks are exchanged and the loop goes on in the thread's frame
    /// (`Vm::resume_in_place`). Each is also a `coroutine_direct_resumes`.
    pub coroutine_in_place_resumes: u64,
    /// Resumes in place that the fast loop left before the thread yielded
    /// back to it, and that went on as an ordinary nested resume.
    pub coroutine_in_place_exits: u64,
    /// Suspended stacks whose runtime roots were rebuilt for a collection.
    /// A yield only marks its stack stale, so a ping-pong loop that never
    /// collects leaves this at zero.
    pub coroutine_root_snapshots: u64,
    /// Coroutines the collector's refresh of suspended stacks looked at. Only
    /// a stack that yielded since the last refresh is listed, so this does not
    /// grow with the number of clean suspended coroutines.
    pub coroutine_root_visits: u64,
    /// `coroutine.resume` calls a native trace made itself, through the same
    /// switch as `coroutine_direct_resumes`, and kept running after.
    pub coroutine_trace_resumes: u64,
    /// Calls the interpreter replayed from what a trace's resume already
    /// gave, because the trace could not take the answer (an error, a dead
    /// coroutine, another number or type of values than it recorded).
    /// Also the calls of a wrapper whose thread failed under the fast
    /// loop's resume, which the general path reports.
    pub coroutine_trace_replays: u64,
    /// Host C function calls a native trace made itself, without leaving
    /// (`TraceOp::CallHelper` of `HelperId::vm_call`).
    pub host_trace_calls: u64,
    /// Resumes whose thread ran only the short run of its few instructions
    /// to a yield (`Vm::run_micro`), without the fast loop.
    pub coroutine_short_runs: u64,
    /// Resumes a native trace took without reading again what the thread
    /// could have written: the thread ran only a short run, which writes no
    /// table, cell or global.
    pub coroutine_trace_unchecked: u64,
    /// Trace engine counters: recording, and later native execution.
    pub trace: TraceStatistics,
}

/// Counters describing interpreter hot paths.
///
/// Each counter is a single interior-mutable word so `&self` helpers such
/// as [`Vm::closure_body`] can record observations without changing their
/// signatures. Recording is gated on [`Vm::set_interpreter_profiling`]
/// (off by default) so hot loops skip the accounting unless a probe asked
/// for it. The tail-call benchmark probe (`tailcall_probe` example)
/// renders these counters as an instruction mix and frame-replacement
/// breakdown.
#[derive(Clone, Debug)]
pub struct InterpreterCounters {
    /// Total `step` dispatches.
    pub steps: Cell<u64>,
    /// `step` dispatches indexed by `OpCode as usize`.
    pub steps_by_opcode: [Cell<u64>; 64],
    /// Fixed tail-call instructions attempted.
    pub tail_fixed_attempts: Cell<u64>,
    /// Fixed tail calls completed by the in-place frame replacement.
    pub tail_fixed_fast: Cell<u64>,
    /// Fixed tail calls refused because hooks, captures, or close slots
    /// required the generic path.
    pub tail_fixed_miss_observed: Cell<u64>,
    /// Fixed tail calls refused because the callee was not a Lua closure
    /// in adjacent registers.
    pub tail_fixed_miss_callee: Cell<u64>,
    /// Fixed tail calls refused because the callee chunk is vararg.
    pub tail_fixed_miss_vararg: Cell<u64>,
    /// Tail calls executed by the generic close-and-rebuild path, including
    /// every vararg/open form.
    pub tail_generic_entries: Cell<u64>,
    /// Environment (`GetGlobal`) lookups executed.
    pub get_global_lookups: Cell<u64>,
    /// Upvalue (`GetUpvalue`) reads executed.
    pub get_upvalue_reads: Cell<u64>,
    /// Closure body snapshots taken (one `Arc` clone plus cell-handle
    /// clone per call).
    pub closure_body_calls: Cell<u64>,
    /// Scalar arithmetic/comparison instructions completed without
    /// metamethod or coercion continuations.
    pub scalar_fast: Cell<u64>,
    /// Scalar instructions that needed the general continuation path.
    pub scalar_slow: Cell<u64>,
}

impl InterpreterCounters {
    /// Copies the per-opcode histogram so callers can render it without
    /// touching the interior-mutable cells.
    #[must_use]
    pub fn opcode_steps(&self) -> [u64; 64] {
        std::array::from_fn(|index| self.steps_by_opcode[index].get())
    }
}

impl Default for InterpreterCounters {
    fn default() -> Self {
        Self {
            steps: Cell::new(0),
            steps_by_opcode: std::array::from_fn(|_| Cell::new(0)),
            tail_fixed_attempts: Cell::new(0),
            tail_fixed_fast: Cell::new(0),
            tail_fixed_miss_observed: Cell::new(0),
            tail_fixed_miss_callee: Cell::new(0),
            tail_fixed_miss_vararg: Cell::new(0),
            tail_generic_entries: Cell::new(0),
            get_global_lookups: Cell::new(0),
            get_upvalue_reads: Cell::new(0),
            closure_body_calls: Cell::new(0),
            scalar_fast: Cell::new(0),
            scalar_slow: Cell::new(0),
        }
    }
}

fn bump(counter: &Cell<u64>) {
    counter.set(counter.get().wrapping_add(1));
}

impl Error for VmError {}
impl From<RuntimeError> for VmError {
    fn from(error: RuntimeError) -> Self {
        match error {
            RuntimeError::Hook(error) => Self::HookError(Box::new((*error).into())),
            error => Self::Runtime(error),
        }
    }
}

impl VmError {
    /// Whether this is an error in a finalizer that Lua 5.2 and 5.3 wrapped
    /// as `error in __gc metamethod (...)`, which their C API reports with
    /// status `LUA_ERRGCMM`.
    #[must_use]
    pub const fn is_wrapped_finalizer_error(&self) -> bool {
        match self {
            Self::HookError(error) => error.is_wrapped_finalizer_error(),
            Self::Runtime(error) | Self::LocatedRuntime { error, .. } => {
                matches!(error, RuntimeError::FinalizerWrapped(_))
            }
            _ => false,
        }
    }

    /// Whether this error came from an allocator failure rather than Lua code.
    #[must_use]
    pub const fn is_memory_error(&self) -> bool {
        match self {
            Self::HookError(error) => error.is_memory_error(),
            Self::Runtime(error) | Self::LocatedRuntime { error, .. } => error.is_memory_error(),
            _ => false,
        }
    }

    /// Whether an error originated inside a debug hook.
    #[must_use]
    pub const fn is_hook_error(&self) -> bool {
        matches!(
            self,
            Self::HookError(_)
                | Self::Runtime(RuntimeError::Hook(_))
                | Self::LocatedRuntime {
                    error: RuntimeError::Hook(_),
                    ..
                }
        )
    }

    /// The `os.exit` request this error carries, if Lua asked to exit.
    #[must_use]
    pub const fn exit_request(&self) -> Option<ExitRequest> {
        match self {
            Self::HookError(error) => error.exit_request(),
            Self::Runtime(RuntimeError::Exit(request))
            | Self::LocatedRuntime {
                error: RuntimeError::Exit(request),
                ..
            } => Some(*request),
            _ => None,
        }
    }
}

/// Owns execution state while delegating heap state to [`Runtime`].
/// `Vm::count_fetched` before the counting fast loop fetched anything.
pub(crate) const NOT_FETCHED: u64 = u64::MAX;

/// `Vm::count_fetched` for the instruction at `pc` of a stack `depth` deep.
/// Program counters fit 32 bits (see the fast loop's jump record).
#[inline(always)]
pub(crate) fn fetched_mark(depth: usize, pc: usize) -> u64 {
    ((depth as u64) << 32) | pc as u64
}

/// How many emptied host-entry stacks a VM keeps (see [`Vm::run`]): one per
/// level of host calls nested in Lua called from the host, which seldom goes
/// deeper than a few.
const SPARE_HOST_STACKS: usize = 4;

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent facts about the installed hooks and the fast loop's mode"
)]
pub struct Vm {
    runtime: Runtime,
    host_functions: BTreeMap<NativeFunctionHandle, HostFunction>,
    /// The callback and token of each registered host function, by native
    /// handle index: one indexed load where `host_functions` is a tree walk
    /// on every host call.
    host_dispatch: Vec<Option<(HostCallback, u64)>>,
    /// The [`HostDirectCallback`] of each host function that has one, by
    /// native handle index like `host_dispatch`.
    host_direct: Vec<Option<HostDirectCallback>>,
    /// Roots of the Lua stacks suspended under each active host callback.
    suspended_roots: Vec<GcRoots>,
    /// Emptied roots of finished host callbacks, whose buffers the next
    /// ones reuse: a host call allocates no roots in a steady state.
    spare_roots: Vec<GcRoots>,
    /// The Lua stacks suspended under each active host callback, for
    /// `debug.getinfo`, `debug.getlocal` and [`Vm::host_stack_info`] inside
    /// the callback and for the collector.
    host_snapshots: Vec<host_locals::HostSnapshot>,
    /// Emptied call stacks of finished host entries (see [`Self::run`]),
    /// kept for the next one. A host such as Luanti calls small Lua
    /// functions (`read_vector`) hundreds of thousands of times, and a fresh
    /// stack allocates and nils a register window per call and frees it
    /// again.
    spare_host_stacks: Vec<CallStack>,
    // Successful debugger local writes, including edits beneath host callbacks.
    local_mutation_epoch: u64,
    active_host_function: Option<NativeFunctionHandle>,
    suspended_host_functions: Vec<NativeFunctionHandle>,
    compatibility: CompatibilityProfile,
    execution_mode: ExecutionMode,
    tracing_opt_level: TracingOptLevel,
    /// The main thread's default environment. This becomes per-coroutine
    /// state once coroutines exist.
    thread_environment: TableHandle,
    parked_main_environment: Option<TableHandle>,
    chunk_loader: Option<Box<dyn ChunkLoader>>,
    // The callback type is the embedding API shape; a type alias would only rename it.
    #[allow(clippy::type_complexity)]
    source_reader: Option<Box<dyn Fn(Option<&[u8]>) -> std::io::Result<Vec<u8>>>>,
    // Same callback shape as `source_reader`.
    #[allow(clippy::type_complexity)]
    module_resolver: Option<Box<dyn Fn(&[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>, String>>>,
    /// The private values used to mark modules currently loading, or whose
    /// loader failed. Lua 5.1 deliberately leaves its sentinel in
    /// `package.loaded` after a loader error. Keeping the identity here lets
    /// us distinguish that private table from an ordinary module table.
    require_sentinels: BTreeMap<(TableHandle, LuaString), TableHandle>,
    coroutines: coroutines::CoroutineTable,
    current_coroutine: Option<ThreadHandle>,
    main_thread: Option<ThreadHandle>,
    /// The stacks of the threads that wait on a resume, the innermost last,
    /// left where their owners hold them (see `coroutines::ParkedStack`).
    resumer_stacks: Vec<coroutines::ParkedStack>,
    /// The resumes the fast loop made by running the thread in its own
    /// place (see `coroutines::InPlace`), the innermost last. Each has an
    /// entry in `resumer_stacks` too.
    in_place: Vec<coroutines::InPlace>,
    /// How many entries of `in_place` belong to fast loops beneath the one
    /// that runs: that loop's yields and exits handle only the entries past
    /// them. `usize::MAX` while the running loop makes no resume in place
    /// (a hooked loop, or no fast loop at all).
    in_place_base: usize,
    /// The values of a yield on their way to the resume, with the call that
    /// takes the resume's values. When the flag is set, the values already
    /// start with the `true` of a successful resume (see `Vm::fast_yield`).
    pending_yield: Option<(CallReturn, Vec<Value>, bool)>,
    /// The values the bottom activation of a host entry returned in the fast
    /// loop, on their way out of the run with `FastExit::Returned`.
    host_returned: Vec<Value>,
    /// An empty vector with room kept from an earlier resume or yield. The
    /// values a switch passes travel in it, so a resume and yield pair
    /// allocates nothing once the buffer has grown (see `recycle_values`).
    spare_values: Vec<Value>,
    /// The values of a direct yield that fit the small buffer (see
    /// `coroutines::QuickYield`), which the resume takes in place of
    /// `pending_yield`.
    quick: coroutines::QuickYield,
    pending_host_yield: Option<Vec<Value>>,
    /// What a resume a native trace made gave, when the trace could not take
    /// the answer and left: the interpreter replays the call and takes the
    /// answer from here instead of running the coroutine again (see
    /// `Vm::resume_for_trace`).
    resume_memo: Option<coroutines::ResumeMemo>,
    /// The error a host call a native trace made raised, mapped as the
    /// dispatch loop maps it. The trace's entry returns it as the run's
    /// failure (see `Vm::host_call_for_trace`).
    vmcall_error: Option<VmError>,
    /// The argument vector of the last host call a native trace made, kept
    /// for the next so that a call allocates none. A call made while another
    /// holds it (from inside the host function) takes a fresh one.
    vmcall_arguments: Vec<Value>,
    /// How many times `debug.setlocal` or `lua_setlocal` wrote a local. A
    /// host call that changes it may have changed a register the trace holds.
    local_writes: u64,
    /// The coroutines whose `roots_stale` went true since the collector last
    /// refreshed them, in the order they yielded. A refresh drains it, so it
    /// visits the stale stacks and not every coroutine. An entry may be
    /// outdated (the roots were rebuilt, or the thread is gone): the flag on
    /// the coroutine decides. Each thread is listed at most once (`listed`).
    stale_coroutines: Vec<ThreadHandle>,
    /// While a native trace's resume runs a coroutine: the place in
    /// `resumer_stacks` of the stack the trace runs on. Its registers hold
    /// what the trace entered with, not what its pc says is live, so a
    /// collection roots its whole window; every other stack is rooted as
    /// the interpreter would (see `Vm::gc_roots_with_coroutines`).
    trace_resumer: Option<usize>,
    pending_host_resume: Option<HostResume>,
    pending_host_child: Option<HostChild>,
    resumable_host_boundaries: Vec<usize>,
    host_yield_boundaries: Vec<(Option<ThreadHandle>, bool)>,
    /// Hooks are thread-local in Lua. `None` identifies the main thread.
    debug_hooks: BTreeMap<Option<ThreadHandle>, DebugHook>,
    debug_hook_depth: usize,
    /// Set while the fast loop counts instructions for a count-only hook.
    /// The count lives in `count_budget` meanwhile, not in the hook.
    count_lease: Option<CountLease>,
    /// The leased hook's `remaining`, decremented by the counting fast loop.
    count_budget: u32,
    /// Stack depth and pc of the last instruction the counting fast loop
    /// fetched, so its exit can tell whether the instruction it hands the
    /// dispatch loop was already counted.
    count_fetched: u64,
    /// Set when the second call of a fused open-results pair (see
    /// `Vm::open_native_in_window`) raised the error the fast loop reports,
    /// so it names that call's site. Taken by the loop's error arm.
    fused_open_error: bool,
    /// Whether some hook wants calls reported (call, return or line mask).
    /// A count hook alone does not: the shortcuts that run a native call
    /// without reporting it are exact under it, since the call is one counted
    /// instruction either way. Kept up to date by `debug_hooks_changed`.
    hooks_observe_calls: bool,
    /// Whether a debug hook was ever installed. Only a hook that failed
    /// marks a frame as running with hooks off (`suppress_debug_hooks`), so
    /// until one is installed a host call need not look for such a frame.
    /// Set by `debug_hooks_changed`; never cleared, since marked frames
    /// outlive the hook.
    debug_hooks_installed: bool,
    /// What the hooked fast loop observes: see [`FastHookMode::Hooked`].
    fast_hook_count: bool,
    fast_hook_calls: bool,
    /// Event names passed to hooks, built on first use: a call hook would
    /// otherwise allocate its event string on every call.
    hook_event_names: [Option<Value>; 6],
    /// Nested `table_meta` host callbacks, bounded like stock's C calls.
    /// The bound (195) fits a byte, which fits in the struct's padding.
    table_meta_depth: u8,
    /// Host-requested interruption, delivered as a catchable Lua error.
    interrupt_flag: Option<Arc<AtomicBool>>,
    /// Backward jumps left before the interpreter's fast loop polls the
    /// collector. Allocating instructions and native calls poll it themselves,
    /// so the backedge only has to catch up eventually. Starts at 1 so the
    /// first backedge polls.
    backedge_gc_countdown: u32,
    /// The flag [`Self::interrupt_pending`] polls: `interrupt_flag`'s, or one
    /// nobody sets. A call polls it, so it is one load away.
    interrupt_poll: InterruptPoll,
    /// The attribution sampler attached to this VM, if any (see
    /// [`attribution`]). Off by default: a hot interpreter loop should not
    /// pay for observability it is not using.
    sampler: Option<attribution::SamplerHandle>,
    pending_debug_sethook: Option<Vec<Value>>,
    /// Stack depth of the Lua activation that requested a full collection.
    /// Rechecking after that activation returns prevents its released object
    /// graph from leaving an obsolete, oversized automatic-GC threshold.
    pending_gc_recheck_depth: Option<usize>,
    /// Whether a return has more to do than hand its values back: a
    /// collection recheck is pending, a hook is installed on some thread, or
    /// a count lease is open (`Vm::refresh_return_watch` holds it to that).
    /// The fast loop's plain return tests this one flag instead of the
    /// three.
    plain_return_watch: bool,
    /// The native identity of the closure the fast loop last found better
    /// called through the general routine, with its inline plan, than as a
    /// plain activation (`NO_PLAN_CALLEE` for none). A call of it skips the
    /// plain attempt.
    plan_callee: u32,
    /// Allocation observed by the most recent interpreter-side GC slice.
    /// Delaying a slice one instruction keeps dead temporary results out of
    /// the root snapshot while preserving allocation-boundary safety.
    gc_observed_allocation_serial: u64,
    interpreter_counters: InterpreterCounters,
    /// Where constant-string reads found their values in hashed tables.
    string_read_cache: crate::runtime::StringReadCache,
    /// The position of the entry the interpreter's last `next` step
    /// returned: where the next step looks for its key first.
    next_hint: usize,
    /// Enables interpreter profiling counters. Off by default: a hot
    /// interpreter loop should not pay for observability it is not using.
    interpreter_profiling: bool,
    #[cfg(test)]
    table_sort_fast_path_hits: usize,
    /// Calls the tracing fast loop handed to `call_closure::<true>` instead
    /// of pushing inline, for the tests that pin the inline path.
    #[cfg(test)]
    tracing_out_of_line_calls: usize,
    /// Plain `__index` and `__newindex` handlers the fast loop entered, the
    /// index misses that took the general resolver instead, and the misses
    /// that left the fast loop altogether, for the tests that pin the
    /// in-loop entry.
    #[cfg(test)]
    index_handler_entries: usize,
    #[cfg(test)]
    index_miss_resolves: usize,
    /// Plain operator handlers the fast loop entered directly (see
    /// `Vm::operator_miss_hot`), for the tests that pin that path.
    #[cfg(test)]
    operator_handler_entries: usize,
    #[cfg(test)]
    index_miss_exits: usize,
    /// `pcall` activations, `__call` handler activations, protected returns
    /// and dispatch-loop fallbacks of the fast loop, for the tests that pin
    /// the `pcall` and `__call` paths.
    #[cfg(test)]
    protected_fast_pushes: usize,
    #[cfg(test)]
    protected_fast_returns: usize,
    #[cfg(test)]
    call_handler_pushes: usize,
    #[cfg(test)]
    fast_fallback_exits: usize,
    /// Native-tier and plan-cache state. Interpreter-only VMs keep it empty.
    jit: JitState,
    /// Native calls can re-enter this VM through host callbacks. Cache flushes
    /// wait until the outermost compiled call returns.
    active_native_runs: usize,
}

impl Default for Vm {
    fn default() -> Self {
        let mut runtime = Runtime::default();
        runtime.configure_jit(false);
        let thread_environment = runtime.global_environment();
        let mut vm = Self {
            runtime,
            host_functions: BTreeMap::new(),
            host_dispatch: Vec::new(),
            host_direct: Vec::new(),
            suspended_roots: Vec::new(),
            spare_roots: Vec::new(),
            host_snapshots: Vec::new(),
            spare_host_stacks: Vec::new(),
            local_mutation_epoch: 0,
            active_host_function: None,
            suspended_host_functions: Vec::new(),
            compatibility: CompatibilityProfile::default(),
            execution_mode: ExecutionMode::default(),
            tracing_opt_level: TracingOptLevel::default(),
            thread_environment,
            chunk_loader: None,
            source_reader: None,
            module_resolver: None,
            require_sentinels: BTreeMap::new(),
            coroutines: coroutines::CoroutineTable::default(),
            current_coroutine: None,
            main_thread: None,
            resumer_stacks: Vec::new(),
            in_place: Vec::new(),
            in_place_base: usize::MAX,
            pending_yield: None,
            host_returned: Vec::new(),
            spare_values: Vec::new(),
            quick: coroutines::QuickYield::default(),
            parked_main_environment: None,
            pending_host_yield: None,
            resume_memo: None,
            vmcall_error: None,
            vmcall_arguments: Vec::new(),
            local_writes: 0,
            stale_coroutines: Vec::new(),
            trace_resumer: None,
            pending_host_resume: None,
            pending_host_child: None,
            resumable_host_boundaries: Vec::new(),
            host_yield_boundaries: Vec::new(),
            debug_hooks: BTreeMap::new(),
            debug_hook_depth: 0,
            count_lease: None,
            count_budget: 0,
            count_fetched: NOT_FETCHED,
            fused_open_error: false,
            hooks_observe_calls: false,
            debug_hooks_installed: false,
            fast_hook_count: false,
            fast_hook_calls: false,
            hook_event_names: Default::default(),
            table_meta_depth: 0,
            interrupt_flag: None,
            backedge_gc_countdown: 1,
            interrupt_poll: InterruptPoll::NEVER,
            sampler: None,
            pending_debug_sethook: None,
            pending_gc_recheck_depth: None,
            plain_return_watch: false,
            plan_callee: NO_PLAN_CALLEE,
            gc_observed_allocation_serial: 0,
            jit: JitState::default(),
            active_native_runs: 0,
            interpreter_counters: InterpreterCounters::default(),
            string_read_cache: crate::runtime::StringReadCache::default(),
            next_hint: 0,
            interpreter_profiling: false,
            #[cfg(test)]
            table_sort_fast_path_hits: 0,
            #[cfg(test)]
            tracing_out_of_line_calls: 0,
            #[cfg(test)]
            index_handler_entries: 0,
            #[cfg(test)]
            index_miss_resolves: 0,
            #[cfg(test)]
            index_miss_exits: 0,
            #[cfg(test)]
            protected_fast_pushes: 0,
            #[cfg(test)]
            protected_fast_returns: 0,
            #[cfg(test)]
            call_handler_pushes: 0,
            #[cfg(test)]
            fast_fallback_exits: 0,
            #[cfg(test)]
            operator_handler_entries: 0,
        };
        vm.attach_global_attribution();
        vm
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        if let Some(sampler) = self.sampler.take() {
            sampler.finish(&self.jit.statistics.trace);
        }
        // Stacks that are still parked close their captured locals before
        // they go, so that nothing leaks (see `CallStack`'s `Drop`).
        let mut coroutines = std::mem::take(&mut self.coroutines);
        for coroutine in coroutines.values_mut() {
            self.close_coroutine_upvalues(coroutine);
        }
    }
}

impl Vm {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a VM with the default compatibility profile and a caller-selected
    /// execution policy.
    #[must_use]
    pub fn with_execution_mode(execution_mode: ExecutionMode) -> Self {
        Self::with_profile_and_execution_mode(CompatibilityProfile::default(), execution_mode)
    }
    /// Creates a VM with a caller-selected compatibility identity.
    #[must_use]
    pub fn with_profile(compatibility: CompatibilityProfile) -> Self {
        Self::with_profile_and_execution_mode(compatibility, ExecutionMode::default())
    }
    /// Creates a VM with caller-selected compatibility and execution policies.
    #[must_use]
    pub fn with_profile_and_execution_mode(
        compatibility: CompatibilityProfile,
        execution_mode: ExecutionMode,
    ) -> Self {
        Self::with_runtime_profile_and_execution_mode(
            Runtime::default(),
            compatibility,
            execution_mode,
        )
    }
    #[must_use]
    pub fn with_runtime(runtime: Runtime) -> Self {
        Self::with_runtime_and_profile(runtime, CompatibilityProfile::default())
    }
    /// Creates a VM around a caller-provided runtime and compatibility identity.
    #[must_use]
    pub fn with_runtime_and_profile(runtime: Runtime, compatibility: CompatibilityProfile) -> Self {
        Self::with_runtime_profile_and_execution_mode(
            runtime,
            compatibility,
            ExecutionMode::default(),
        )
    }
    /// Creates a VM around a caller-provided runtime with independent
    /// compatibility and execution policies.
    #[must_use]
    pub fn with_runtime_profile_and_execution_mode(
        mut runtime: Runtime,
        compatibility: CompatibilityProfile,
        execution_mode: ExecutionMode,
    ) -> Self {
        // WebAssembly cannot allocate executable pages for the native tier.
        #[cfg(target_arch = "wasm32")]
        let execution_mode = {
            let _ = execution_mode;
            ExecutionMode::Interpreter
        };
        runtime.configure_jit(execution_mode != ExecutionMode::Interpreter);
        initialize_gc_profile(&mut runtime, compatibility);
        let thread_environment = runtime.global_environment();
        let mut vm = Self {
            runtime,
            host_functions: BTreeMap::new(),
            host_dispatch: Vec::new(),
            host_direct: Vec::new(),
            suspended_roots: Vec::new(),
            spare_roots: Vec::new(),
            host_snapshots: Vec::new(),
            spare_host_stacks: Vec::new(),
            local_mutation_epoch: 0,
            active_host_function: None,
            suspended_host_functions: Vec::new(),
            compatibility,
            execution_mode,
            tracing_opt_level: TracingOptLevel::default(),
            thread_environment,
            chunk_loader: None,
            source_reader: None,
            module_resolver: None,
            require_sentinels: BTreeMap::new(),
            coroutines: coroutines::CoroutineTable::default(),
            current_coroutine: None,
            main_thread: None,
            resumer_stacks: Vec::new(),
            in_place: Vec::new(),
            in_place_base: usize::MAX,
            pending_yield: None,
            host_returned: Vec::new(),
            spare_values: Vec::new(),
            quick: coroutines::QuickYield::default(),
            parked_main_environment: None,
            pending_host_yield: None,
            resume_memo: None,
            vmcall_error: None,
            vmcall_arguments: Vec::new(),
            local_writes: 0,
            stale_coroutines: Vec::new(),
            trace_resumer: None,
            pending_host_resume: None,
            pending_host_child: None,
            resumable_host_boundaries: Vec::new(),
            host_yield_boundaries: Vec::new(),
            debug_hooks: BTreeMap::new(),
            debug_hook_depth: 0,
            count_lease: None,
            count_budget: 0,
            count_fetched: NOT_FETCHED,
            fused_open_error: false,
            hooks_observe_calls: false,
            debug_hooks_installed: false,
            fast_hook_count: false,
            fast_hook_calls: false,
            hook_event_names: Default::default(),
            table_meta_depth: 0,
            interrupt_flag: None,
            backedge_gc_countdown: 1,
            interrupt_poll: InterruptPoll::NEVER,
            sampler: None,
            pending_debug_sethook: None,
            pending_gc_recheck_depth: None,
            plain_return_watch: false,
            plan_callee: NO_PLAN_CALLEE,
            gc_observed_allocation_serial: 0,
            jit: JitState::new(execution_mode),
            active_native_runs: 0,
            interpreter_counters: InterpreterCounters::default(),
            string_read_cache: crate::runtime::StringReadCache::default(),
            next_hint: 0,
            interpreter_profiling: false,
            #[cfg(test)]
            table_sort_fast_path_hits: 0,
            #[cfg(test)]
            tracing_out_of_line_calls: 0,
            #[cfg(test)]
            index_handler_entries: 0,
            #[cfg(test)]
            index_miss_resolves: 0,
            #[cfg(test)]
            index_miss_exits: 0,
            #[cfg(test)]
            protected_fast_pushes: 0,
            #[cfg(test)]
            protected_fast_returns: 0,
            #[cfg(test)]
            call_handler_pushes: 0,
            #[cfg(test)]
            fast_fallback_exits: 0,
            #[cfg(test)]
            operator_handler_entries: 0,
        };
        vm.attach_global_attribution();
        vm
    }
    /// Sets a host flag that requests a catchable "interrupted!" error.
    /// The VM clears the flag when it delivers the error. Hosts can set it
    /// again to interrupt a later call, including a resumed coroutine.
    pub fn set_interrupt_flag(&mut self, flag: Arc<AtomicBool>) {
        // An attached sampler keeps the poll: its ticks bring the dispatch
        // loop to `check_interrupt`, which reads this flag.
        if self.sampler.is_none() {
            self.interrupt_poll = InterruptPoll::of(&flag);
        }
        self.interrupt_flag = Some(flag);
    }

    #[inline(always)]
    fn interrupt_pending(&self) -> bool {
        self.interrupt_poll.pending()
    }

    fn check_interrupt(&self) -> VmResult<()> {
        if self
            .interrupt_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed) && flag.swap(false, Ordering::Relaxed))
        {
            return Err(RuntimeError::message("interrupted!").into());
        }
        Ok(())
    }

    /// The compatibility profile selected for this VM.
    #[must_use]
    pub const fn compatibility_profile(&self) -> CompatibilityProfile {
        self.compatibility
    }
    /// The execution policy selected for this VM.
    #[must_use]
    pub const fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }
    /// The tracing engine's optimization level. See [`TracingOptLevel`].
    #[must_use]
    pub const fn tracing_opt_level(&self) -> TracingOptLevel {
        self.tracing_opt_level
    }
    /// Selects the tracing engine's optimization level and its flags for
    /// traces compiled from now on. Every level runs the level 0
    /// optimizations today; levels 1 and 2 are reserved for advanced
    /// optimizations of hot traces.
    pub const fn set_tracing_opt_level(&mut self, level: TracingOptLevel) {
        self.tracing_opt_level = level;
        self.jit.trace.set_opt_flags(level.flags());
    }
    /// The optimizations traces compiled from now on may get.
    #[must_use]
    pub const fn tracing_opt_flags(&self) -> TracingOptFlags {
        self.jit.trace.opt_flags()
    }
    /// Overrides the optimizations of the current level for traces compiled
    /// from now on, for example to switch one pass off:
    /// `vm.set_tracing_opt_flags(vm.tracing_opt_flags().without(TracingOptFlags::MEMORY))`.
    pub const fn set_tracing_opt_flags(&mut self, flags: TracingOptFlags) {
        self.jit.trace.set_opt_flags(flags);
    }
    /// Reports whether this VM currently enters native code.
    #[must_use]
    pub const fn jit_enabled(&self) -> bool {
        self.runtime.jit_enabled()
    }
    /// Enables or disables native entry when the execution mode supports it.
    pub const fn set_jit_enabled(&mut self, enabled: bool) {
        #[cfg(target_arch = "wasm32")]
        let enabled = {
            let _ = enabled;
            false
        };
        self.runtime.set_jit_enabled(enabled);
    }
    /// Native compilation and execution counters for this VM.
    #[must_use]
    pub const fn jit_statistics(&self) -> JitStatistics {
        self.jit.statistics
    }
    /// The names of the helpers the installed traces call (`string.len`), one
    /// per call site. A diagnostic for tests and tools: it shows which
    /// library calls compiled to a direct call.
    #[must_use]
    pub fn trace_helper_calls(&self) -> Vec<&'static str> {
        self.jit.trace.installed_helper_names()
    }
    /// The metamethod events (`__index`, `__add`) whose handler the
    /// installed traces pick by a metatable class guard instead of a helper,
    /// one per guard. A diagnostic for tests: such an access calls none of
    /// the handler helpers [`Self::trace_helper_calls`] lists.
    #[must_use]
    pub fn trace_class_guard_events(&self) -> Vec<String> {
        self.jit.trace.installed_class_guard_events()
    }
    /// How many steps of `next` the trace helpers made on this thread: a
    /// diagnostic for tests, which compare it with the entries a loop
    /// traversed to see that the array part was read in place.
    #[must_use]
    pub fn trace_next_helper_steps(&self) -> u64 {
        jit::trace::next_helper_steps()
    }
    /// How many times a specialized pattern helper ran on this thread: a
    /// diagnostic for tests. A trace compiled with
    /// `TracingOptFlags::STRING_CONSTANTS` calls them for a constant pattern
    /// of `string.find` and `string.match`.
    #[must_use]
    pub fn trace_prepared_helper_calls(&self) -> u64 {
        jit::trace::prepared_helper_calls()
    }
    /// How many single-run patterns (`%d+`, `(%a+)`) of `string.find` and
    /// `string.match` the installed traces scan in their own code instead of
    /// calling a pattern helper: a diagnostic for tests.
    #[must_use]
    pub fn trace_inline_runs(&self) -> usize {
        self.jit.trace.installed_inline_runs()
    }
    /// How many operations of the installed traces call out of the generated
    /// code for a library function: helpers, host calls and pure native
    /// functions. A diagnostic for tests: a library function the trace
    /// records as operations adds none.
    #[must_use]
    pub fn trace_external_calls(&self) -> usize {
        self.jit.trace.installed_external_calls()
    }
    /// [`Self::trace_external_calls`] for the optimized traces that run,
    /// which can have dropped a call the recording made (a `string.sub`
    /// whose result only a recovery path reads).
    #[must_use]
    pub fn trace_compiled_external_calls(&self) -> usize {
        self.jit.trace.installed_compiled_external_calls()
    }
    /// The cache behind constant-string global and field reads.
    #[must_use]
    pub fn string_read_cache(&self) -> &crate::runtime::StringReadCache {
        &self.string_read_cache
    }

    /// Interpreter hot-path counters for this VM.
    ///
    /// Recording is enabled with [`Vm::set_interpreter_profiling`]; all
    /// counters read zero unless a probe turned it on.
    #[must_use]
    pub fn interpreter_counters(&self) -> &InterpreterCounters {
        &self.interpreter_counters
    }
    /// Enables or disables interpreter profiling counters for this VM.
    ///
    /// When enabled, every `step` records its opcode in the counters
    /// histogram and hot-path events (tail calls, global/upvalue reads,
    /// snapshots, scalar outcomes) increment their counters. Disabled by
    /// default so hot loops skip that accounting.
    pub fn set_interpreter_profiling(&mut self, enabled: bool) {
        self.interpreter_profiling = enabled;
    }
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }
    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }
    #[must_use]
    pub fn into_runtime(mut self) -> Runtime {
        // `Vm` has a destructor, so the runtime cannot move out of it.
        std::mem::replace(&mut self.runtime, Runtime::new())
    }

    /// Configures the move-only compiler session used by Lua's `load` builtin.
    pub fn set_chunk_loader(&mut self, loader: Box<dyn ChunkLoader>) {
        self.chunk_loader = Some(loader);
    }

    /// Replaces host file reads used by `loadfile`, `dofile`, and Lua module
    /// search. The callback receives raw path bytes, or `None` for `dofile()`
    /// reading standard input. Once set, failed reads never fall back to the
    /// host filesystem or standard input.
    // The callback type is the embedding API shape; a type alias would only rename it.
    #[allow(clippy::type_complexity)]
    pub fn set_source_reader(
        &mut self,
        reader: Box<dyn Fn(Option<&[u8]>) -> std::io::Result<Vec<u8>>>,
    ) {
        self.source_reader = Some(reader);
    }

    /// Consults a host module resolver after `package.preload` and before
    /// searching `package.path`. `None` continues the normal path search.
    /// The returned pair is `(source, source_name)` in raw bytes.
    // The callback type is the embedding API shape; a type alias would only rename it.
    #[allow(clippy::type_complexity)]
    pub fn set_module_resolver(
        &mut self,
        resolver: Box<dyn Fn(&[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>, String>>,
    ) {
        self.module_resolver = Some(resolver);
    }

    /// Executes a validated top-level chunk.
    ///
    /// # Errors
    ///
    /// Returns bytecode validation errors, Lua runtime errors, or a checked
    /// execution error for malformed register ranges.
    pub fn execute(&mut self, chunk: &Chunk) -> VmResult<Vec<Value>> {
        self.execute_with_arguments(chunk, Vec::new())
    }

    /// Executes a validated top-level chunk with Lua top-level varargs.
    ///
    /// # Errors
    ///
    /// Returns bytecode validation or execution errors.
    pub fn execute_with_arguments(
        &mut self,
        chunk: &Chunk,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        chunk.validate().map_err(VmError::InvalidChunk)?;
        let upvalues = self.initial_chunk_upvalues(chunk, self.thread_environment)?;
        let frame = self.make_frame(
            Arc::new(chunk.clone()),
            arguments,
            upvalues,
            None,
            None,
            self.thread_environment,
            None,
        )?;
        self.run(frame)
    }

    /// Executes a chunk with a protected error handler. The handler can inspect
    /// the failed stack through the debug library before its error is returned.
    ///
    /// # Errors
    /// Returns validation errors or the error object produced by the handler.
    pub fn execute_with_message_handler(
        &mut self,
        chunk: &Chunk,
        arguments: Vec<Value>,
        handler: Value,
    ) -> VmResult<Vec<Value>> {
        chunk.validate().map_err(VmError::InvalidChunk)?;
        let mut caller = self.make_frame(
            host::builtin_call_chunk(0)?,
            Vec::new(),
            Vec::new(),
            None,
            None,
            self.thread_environment,
            None,
        )?;
        caller.pc = 1;
        let mut frames = CallStack::with_frame(caller);
        let upvalues = self.initial_chunk_upvalues(chunk, self.thread_environment)?;
        let target = self.make_frame(
            Arc::new(chunk.clone()),
            arguments,
            upvalues,
            Some(ReturnTarget::ExtendedProtectedTarget(Box::new(
                ExtendedProtectedCall {
                    target: CallReturn {
                        base: 0,
                        results: 0,
                    },
                    handler,
                    suppress_hooks: false,
                    handler_calls: 0,
                    pending_unwind: None,
                },
            ))),
            None,
            self.thread_environment,
            None,
        )?;
        frames.push(target);
        let mut values = self.run_stack(frames)?;
        if matches!(values.first(), Some(Value::Boolean(true))) {
            values.remove(0);
            Ok(values)
        } else {
            Err(errors::raised_value(
                values.get(1).cloned().unwrap_or(Value::Nil),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_frame(
        &mut self,
        chunk: impl Into<frame::FrameChunk>,
        arguments: impl IntoIterator<Item = Value>,
        upvalues: impl Into<CaptureCells>,
        caller: Option<ReturnTarget>,
        call_site: Option<ActiveCallSite>,
        environment: TableHandle,
        closure: Option<ClosureHandle>,
    ) -> VmResult<OwnedFrame> {
        let chunk: frame::FrameChunk = chunk.into();
        let registers = Vec::with_capacity(usize::from(chunk.register_count));
        let mut frame = Frame::new_with_registers(
            chunk,
            arguments,
            upvalues.into(),
            caller,
            call_site,
            environment,
            closure,
            registers,
        );
        if let Some(register) = frame.chunk.arg_register {
            let table = self.runtime.create_table()?;
            for (offset, value) in frame.varargs.iter().enumerate() {
                if !matches!(value, Value::Nil) {
                    let index = vararg_index(1, offset)?;
                    self.runtime
                        .table_set(table, &Value::Number(index), value.clone())?;
                }
            }
            let vararg_count = frame.varargs.len();
            self.runtime.table_set(
                table,
                &Value::from("n"),
                if matches!(
                    self.compatibility,
                    CompatibilityProfile::Lua53
                        | CompatibilityProfile::Lua54
                        | CompatibilityProfile::Lua55
                ) {
                    Value::Integer(i64::try_from(vararg_count).unwrap_or(i64::MAX))
                } else {
                    Value::Number(usize_to_number(vararg_count)?)
                },
            )?;
            frame.registers[usize::from(register)] = Value::Table(table);
        }
        Ok(frame)
    }

    /// The stack [`Self::execute`] starts `chunk` on, for tests that step it.
    #[cfg(test)]
    pub(crate) fn stack_for(&mut self, chunk: &Chunk) -> VmResult<CallStack> {
        chunk.validate().map_err(VmError::InvalidChunk)?;
        let upvalues = self.initial_chunk_upvalues(chunk, self.thread_environment)?;
        let frame = self.make_frame(
            Arc::new(chunk.clone()),
            Vec::new(),
            upvalues,
            None,
            None,
            self.thread_environment,
            None,
        )?;
        Ok(CallStack::with_frame(frame))
    }

    /// Runs a host entry: `initial` on a stack of its own. The stack comes
    /// from [`Self::spare_host_stacks`] and goes back there once its
    /// activations have all returned, so a host that calls a small function
    /// again and again reuses one register window instead of allocating,
    /// niling and freeing one per call. The dead slots a returned stack keeps
    /// are the ones the fast loop already leaves behind between two calls on
    /// one stack (see [`CallStack`]): never read, rooted or exposed.
    fn run(&mut self, initial: OwnedFrame) -> VmResult<Vec<Value>> {
        let mut frames = self.spare_host_stacks.pop().unwrap_or_default();
        frames.push(initial);
        self.run_pushed(frames)
    }

    /// Runs the host entry whose bottom activation `frames` holds, then
    /// keeps the stack for the next entry.
    fn run_pushed(&mut self, mut frames: CallStack) -> VmResult<Vec<Value>> {
        if self.sampler.is_some() {
            self.sample_host_entry();
        }
        frames.set_host_entry();
        let result = self.run_stack_inner(&mut frames);
        if self.spare_host_stacks.len() < SPARE_HOST_STACKS
            && let Some(frames) = frames.recycled()
        {
            self.spare_host_stacks.push(frames);
        }
        result
    }

    fn run_stack(&mut self, mut frames: CallStack) -> VmResult<Vec<Value>> {
        self.run_stack_inner(&mut frames)
    }

    fn run_stack_inner(&mut self, frames: &mut CallStack) -> VmResult<Vec<Value>> {
        loop {
            match self.interpret(frames, false) {
                Ok(Some(values)) => return Ok(values),
                Ok(None) => {}
                Err(error) => {
                    if frames.is_empty() {
                        return Err(error);
                    }
                    let error = locate_runtime_error(frames, self.compatibility, error);
                    if let Some(action) = self.resume_unwind_handler_error(frames, error.clone())? {
                        if let UnwindAction::Return(error) = action {
                            return Err(error);
                        }
                    } else if let UnwindAction::Return(error) =
                        self.begin_error_unwind(frames, error)?
                    {
                        return Err(error);
                    }
                }
            }
        }
    }
}

mod arith;
pub mod attribution;
mod builtins;
mod calls;
mod continuations;
mod coroutines;
mod debug_info;
mod errors;
mod fast_path;
mod frame;
mod host;
mod host_continuations;
mod host_locals;
mod instructions;
mod interp;
mod jit;

/// Installs the addresses of the trace helpers the VM implements, which a
/// compile outside a VM (`jit::compile_dump`) needs.
pub(crate) fn install_trace_helpers() {
    jit::trace::native::install_helpers();
}
#[cfg(test)]
pub(crate) use jit::trace::native::restore_table_helpers;
pub use jit::trace::{
    NativeAborts, TraceAbortReason, TraceExitKind, TraceGuardKind, TraceStatistics,
};

use crate::vm::jit::state::JitState;
mod libraries;
mod load;
#[cfg(feature = "op-stats")]
mod op_stats;
mod protected;
mod registers;
mod stack;
mod table_meta;
mod unwind;
mod upvalues;

use crate::vm::arith::usize_to_number;
use crate::vm::errors::{display_chunk_source, locate_runtime_error};
use crate::vm::registers::vararg_index;
use crate::vm::stack::{CallStack, OwnedFrame};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CallReturn {
    base: u8,
    results: u8,
}

#[derive(Clone)]
pub(crate) struct ExtendedProtectedCall {
    target: CallReturn,
    handler: Value,
    suppress_hooks: bool,
    handler_calls: usize,
    // Handler retries share one pending unwind. Only the final handler result
    // consumes it, then __close callbacks receive that transformed error.
    pending_unwind: Option<Arc<std::sync::Mutex<Option<UnwindClose>>>>,
}

pub(crate) struct Coroutine {
    environment: TableHandle,
    state: CoroutineState,
    traceback: Option<Traceback>,
    failed: Option<(CallStack, VmError)>,
    /// The suspended stack changed after the thread's runtime roots were
    /// built, so the next collection rebuilds them first.
    roots_stale: bool,
    /// The thread sits in `Vm::stale_coroutines`. Only that list sets and
    /// clears it, so the list holds each thread at most once and stays no
    /// longer than the number of coroutines, however often the flag above
    /// is reset without a collection.
    listed: bool,
    /// A resume found that the thread does not run a few instructions to a
    /// yield (`Vm::resume_direct`): it has more than one frame, or the
    /// short run left before a yield. The fast loop then resumes it in
    /// place (`Vm::resume_in_place`) and spares it the short run. Only a
    /// hint: either way runs any thread.
    long_runs: bool,
    /// The resumes `Vm::resume_direct` made past the short run since the
    /// hint above was set or last tried. One in `LONG_RUN_RETRY` tries the
    /// short run again, so a thread that left it once (a call at its
    /// start, a read that missed) does not pay the switch for the rest of
    /// its life.
    long_resumes: u8,
}

/// Execution status of a host-created coroutine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostCoroutineStatus {
    New,
    Running,
    Suspended,
    Dead,
}

enum CoroutineState {
    New(Option<Value>),
    Suspended {
        /// Boxed: a switch moves the stack in and out of the state, and
        /// the box makes that a pointer copy.
        frames: Box<CallStack>,
        yield_return: CallReturn,
        host_resume: Option<HostResume>,
    },
    Running,
    Dead,
}

pub(crate) struct HostResume {
    function: NativeFunctionHandle,
    child: Option<Box<HostChild>>,
    close_handler: Option<NativeFunctionHandle>,
}

pub(crate) struct HostChild {
    frames: CallStack,
    yield_return: CallReturn,
    host_resume: Option<HostResume>,
    protected: bool,
}

pub(crate) enum CollectGarbageAction {
    Immediate(Vec<Value>),
    Collected(Vec<Value>),
}

#[derive(Clone)]
pub(crate) enum FinalizerCompletion {
    Automatic,
    Call {
        target: CallReturn,
        values: Vec<Value>,
    },
    Protected {
        target: CallReturn,
        values: Vec<Value>,
    },
}

const MAX_INDEX_HOPS: u8 = 100;

enum IndexAction {
    Get { destination: u8 },
    Set { value: Value },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailCallArguments {
    Fixed,
    Vararg,
    Open,
}

pub(crate) enum IndexCompletion {
    Frame,
    /// An `ipairs` step in Lua 5.3 and later: the call `instruction` gets
    /// `index, value`, or `nil` when the element is absent.
    IPairs {
        instruction: crate::bytecode::Instruction,
        index: Value,
    },
    StringGsub {
        continuation: Box<StringGsubContinuation>,
        found: PatternMatch,
    },
}

pub(crate) struct IndexContinuation {
    receiver: Value,
    key: Value,
    action: IndexAction,
    completion: IndexCompletion,
    hops: u8,
    diagnostic: Option<ActiveCallSite>,
}

pub(crate) struct ToStringContinuation {
    completion: ToStringCompletion,
    value: Value,
}

#[derive(Clone, Copy)]
pub(crate) enum ToStringCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

pub(crate) struct StringFormatContinuation {
    completion: StringFormatCompletion,
    arguments: Vec<Value>,
    format_index: usize,
    argument_index: usize,
    converting: usize,
}

#[derive(Clone, Copy)]
enum StringFormatCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum ScalarKind {
    Value,
    Comparison,
}

pub(crate) struct ScalarContinuation {
    destination: u8,
    kind: ScalarKind,
    left: Value,
    right: Option<Value>,
    invert: bool,
    /// Set for a comparison run by a compare-and-branch test: the value of
    /// `expected`. The result then picks the branch instead of being stored.
    test: Option<bool>,
}

/// How the first result of an operator metamethod is stored in the caller:
/// the plain-data form of a [`ScalarContinuation`] whose operands the
/// handler's own arguments and the caller's registers already hold.
#[derive(Clone, Copy)]
pub(crate) struct OperatorReturn {
    destination: u8,
    mode: OperatorMode,
}

#[derive(Clone, Copy)]
enum OperatorMode {
    /// The result itself.
    Value,
    /// The result's truth, negated when `invert`.
    Compare { invert: bool },
    /// The result's truth, negated when `invert`, picks the branch of the
    /// test instruction whose `expected` flag is given.
    Branch { expected: bool, invert: bool },
}

pub(crate) struct ConcatRangeContinuation {
    destination: u8,
    /// Registers of the first and last operands, which name an operand
    /// in a type error.
    first: u8,
    last: u8,
    /// Operands to the left of `accumulator`, retained in source order. The
    /// reduction pops from the end, matching Lua's right-associated `..`.
    remaining: Vec<Value>,
    accumulator: Value,
}

/// Large continuations are boxed: every frame embeds a return target, so
/// its size is copied on each Lua call and return.
pub(crate) enum ReturnTarget {
    Direct(CallReturn),
    Protected(CallReturn),
    ExtendedProtectedTarget(Box<ExtendedProtectedCall>),
    ExtendedProtectedHandler(Box<ExtendedProtectedCall>),
    TableSort(Box<SortContinuation>),
    ProtectedSort(Box<SortContinuation>),
    TableForeach(Box<ForeachContinuation>),
    LoadReader(Box<LoadReaderContinuation>),
    Dofile(DofileCompletion),
    Require(Box<RequireContinuation>),
    RequireSearch(Box<RequireSearchContinuation>),
    Module(Box<ModuleContinuation>),
    StringGsub(Box<StringGsubContinuation>),
    Index(Box<IndexContinuation>),
    ProtectedIndex(Box<IndexContinuation>),
    /// A `__newindex` Lua handler returned into an ordinary frame. Its results
    /// are discarded, so the continuation is not boxed. Same behaviour as
    /// `Index` with a `Set` action and `IndexCompletion::Frame`.
    SetMetamethod,
    /// A plain Lua `__index` handler returned into an ordinary frame: the
    /// first result goes to the register named. Same behaviour as `Index`
    /// with a `Get` action at that destination and `IndexCompletion::Frame`,
    /// without the box: the operands are the handler's own arguments, and
    /// the caller's registers keep what the instruction read.
    IndexResult(u8),
    ToString(Box<ToStringContinuation>),
    StringFormat(Box<StringFormatContinuation>),
    Scalar(Box<ScalarContinuation>),
    /// A plain Lua operator handler returned into an ordinary frame. Same
    /// behaviour as `Scalar`, without the box: see [`OperatorReturn`].
    Operator(OperatorReturn),
    ConcatRange(Box<ConcatRangeContinuation>),
    UserdataFinalizer(Box<FinalizerCompletion>),
    CloseLocal,
    UnwindClose(Box<UnwindClose>),
    DebugHook(Box<DebugHookReturn>),
}

/// Where a debug hook callback returns to. Boxed so the common return
/// targets stay two words wide in every frame record.
pub(crate) struct DebugHookReturn {
    thread: Option<ThreadHandle>,
    remaining_tail_returns: usize,
}

pub(crate) enum UnwindAction {
    Continue,
    Return(VmError),
}

pub(crate) struct UnwindClose {
    error: VmError,
    frames: CallStack,
    completion: UnwindCompletion,
}

enum UnwindCompletion {
    Unprotected,
    Protected {
        target: ReturnTarget,
        traceback: Traceback,
    },
    Resume(Box<UnwindClose>),
    ExtendedHandled {
        continuation: Box<ExtendedProtectedCall>,
        needs_handler: bool,
    },
}

pub(crate) struct StringGsubContinuation {
    source: LuaString,
    pattern: std::rc::Rc<Pattern>,
    replacement: Value,
    maximum: usize,
    output: Vec<u8>,
    cursor: usize,
    search: usize,
    /// End of the previous match, which 5.3 and later never match empty at.
    last_match: Option<usize>,
    count: usize,
    pending: Option<PatternMatch>,
    completion: GsubCompletion,
}

#[derive(Clone, Copy)]
pub(crate) enum GsubCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

pub(crate) struct LoadReaderContinuation {
    reader: Value,
    session: LoadPipeline,
    completion: LoadCompletion,
}

const BINARY_MAGIC: &[u8] = b"\x1bOXIMOND";

/// VM-owned adapter that chooses binary decoding only after seeing its whole
/// magic prefix. Text is delegated to the host's move-only loading session.
pub(crate) enum LoadPipeline {
    Undecided {
        source_name: LuaString,
        bytes: Vec<u8>,
    },
    Text(Box<dyn ChunkLoad>),
    Binary(Vec<u8>),
}

// Boxing the `Loaded(Chunk)` variant would add an allocation to every load
// that completes, for a poll result that is never stored in bulk.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PipelinePoll {
    NeedInput(LoadPipeline),
    Loaded(Chunk),
    Failed(LuaString),
}

#[derive(Clone, Copy)]
pub(crate) enum LoadCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone, Copy)]
pub(crate) enum DofileCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone, Copy)]
pub(crate) enum RequireCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone)]
pub(crate) struct RequireContinuation {
    name: LuaString,
    loader_data: Value,
    loaded: TableHandle,
    sentinel: TableHandle,
    completion: RequireCompletion,
}

#[derive(Clone)]
pub(crate) struct RequireSearchContinuation {
    name: LuaString,
    loaded: TableHandle,
    loaders: TableHandle,
    next_index: usize,
    diagnostics: String,
    completion: RequireCompletion,
}

#[derive(Clone)]
pub(crate) struct ModuleContinuation {
    module: TableHandle,
    options: Vec<Value>,
    next_option: usize,
    completion: ModuleCompletion,
}

#[derive(Clone, Copy)]
pub(crate) enum ModuleCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone)]
pub(crate) struct ForeachContinuation {
    table: TableHandle,
    callback: Value,
    mode: ForeachMode,
    previous: Value,
    index: usize,
    completion: ForeachCompletion,
}

#[derive(Clone, Copy)]
pub(crate) enum ForeachMode {
    All,
    Array,
}

#[derive(Clone, Copy)]
pub(crate) enum ForeachCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone)]
pub(crate) struct SortContinuation {
    /// The table being sorted. For a Lua 5.3+ table with `__index`,
    /// `__newindex` or `__len`, a raw copy of its elements.
    table: TableHandle,
    /// The original table when `table` is such a copy: the sorted elements
    /// are stored back into it through `__newindex` at the end.
    proxy: Option<Value>,
    comparator: Option<Value>,
    length: usize,
    phase: SortPhase,
    comparison: Option<SortComparison>,
    completion: SortCompletion,
}

#[derive(Clone, Copy)]
enum SortPhase {
    Build {
        next_root: usize,
    },
    Extract {
        next_end: usize,
    },
    Sift {
        root: usize,
        end: usize,
        after: SortAfter,
    },
    CompareRoot {
        root: usize,
        end: usize,
        child: usize,
        after: SortAfter,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum SortAfter {
    Build { next_root: usize },
    Extract { next_end: usize },
}

#[derive(Clone, Copy)]
pub(crate) enum SortComparison {
    ChooseLargerChild {
        root: usize,
        end: usize,
        child: usize,
        after: SortAfter,
    },
    SwapRootIfLess {
        root: usize,
        end: usize,
        child: usize,
        after: SortAfter,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum SortCompletion {
    Direct(CallReturn),
    Protected(CallReturn),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ActiveCallSite {
    name: LuaString,
    kind: CallKind,
}

/// What names an activation: the call that made it, when its caller was
/// replaced by a tail call.
pub(crate) enum FrameSite {
    Named(ActiveCallSite),
    /// The activation of `origin` made a tail call at the instruction before
    /// `pc`. The name is looked up in `chunk`, the chunk of `origin`, when a
    /// traceback or `debug.getinfo` asks, so a tail call stores three words
    /// and takes no reference count. `origin` is rooted with the frame
    /// (`Vm::root_frame`), which keeps `chunk` alive.
    Tail {
        origin: ClosureHandle,
        chunk: frame::FrameChunk,
        pc: u32,
    },
}

impl FrameSite {
    fn boxed(site: ActiveCallSite) -> Box<Self> {
        Box::new(Self::Named(site))
    }

    /// The named call, if the site has a name.
    fn resolve(&self) -> Option<ActiveCallSite> {
        match self {
            Self::Named(site) => Some(site.clone()),
            Self::Tail { chunk, pc, .. } => debug_info::call_site_at(chunk, *pc as usize),
        }
    }
}

impl ActiveCallSite {
    /// The name `profile`'s stock Lua reports: Lua 5.1 and `LuaJIT` call the
    /// generic-`for` iterator `(for generator)`.
    fn display_name(&self, profile: CompatibilityProfile) -> String {
        if self.kind == CallKind::ForIterator && profile.is_lua51_family() {
            "(for generator)".to_owned()
        } else {
            self.name.to_lossy().into_owned()
        }
    }
}

#[derive(Clone)]
pub(crate) struct Traceback {
    debug_frames: Vec<DebugInfo>,
    frames: Vec<TraceFrame>,
}

#[derive(Clone)]
struct TraceFrame {
    source: LuaString,
    line: Option<u32>,
    call_site: Option<ActiveCallSite>,
    is_main: bool,
    /// What a stock traceback needs beyond the position: the function, its
    /// kind, and how it was entered.
    extra: TraceExtra,
}

/// What kind of stack level a [`TraceFrame`] stands for.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum TraceKind {
    /// A Lua function (or the main chunk).
    #[default]
    Lua,
    /// A native function that called Lua, or that raised the error.
    Native,
    /// A level that tail calls replaced (Lua 5.1 lists each one).
    Tail,
    /// The host's call into the main thread, below the main chunk.
    Base,
}

#[derive(Clone, Default)]
struct TraceExtra {
    kind: TraceKind,
    /// The running function, which Lua 5.2 and later look up by value in
    /// `package.loaded`.
    function: Option<Value>,
    /// Line where the function was defined, for `function <src:line>`.
    definition_line: Option<u32>,
    /// The metamethod event (`index`, `add`, ...) that called the function.
    metamethod: Option<&'static str>,
    /// A native's qualified name (`table.sort`) when no function value is
    /// at hand.
    native_name: Option<&'static str>,
    /// Whether tail calls replaced the function's caller.
    tail_called: bool,
    /// The call that entered the frame, which `LuaJIT` keeps naming the frame
    /// after tail calls replaced its function.
    caller_site: Option<ActiveCallSite>,
    /// Whether the frame is in the middle of a `return f(...)` of a native
    /// `f`. `LuaJIT`'s tail call reuses such a frame, so its tracebacks omit it.
    in_tail_call: bool,
    /// How `LuaJIT` names a native that the frame below tail-called. `None`
    /// for any other level.
    luajit_tail: Option<LuajitTailName>,
}

/// The name `LuaJIT`'s traceback gives a tail-called native.
#[derive(Clone)]
enum LuajitTailName {
    /// A tail call replaced the caller's record, so nothing names the native.
    Unnamed,
    /// The metamethod event that ran the tail-calling handler.
    Event(&'static str),
    /// The call that reached the tail-calling function.
    Site(ActiveCallSite),
}

#[derive(Clone)]
pub(crate) struct DebugInfo {
    function: Option<Value>,
    source: LuaString,
    current_line: Option<u32>,
    call_site: Option<ActiveCallSite>,
    what: &'static str,
    definition_line: i32,
    last_definition_line: i32,
    upvalue_count: usize,
    hook_transfer: (usize, usize),
    /// The Lua function's chunk. `debug.getinfo` derives `activelines`
    /// from it only when option `L` asks for them.
    chunk: Option<Arc<Chunk>>,
}

#[derive(Clone)]
pub(crate) struct DebugHook {
    function: Value,
    mask: DebugHookMask,
    count: u32,
    remaining: u32,
    /// Hooks are disabled while their callback runs. This is deliberately
    /// stored on the hook rather than globally, so a suspended coroutine does
    /// not accidentally silence the thread that resumed it.
    running: bool,
}

/// How the fast loop may run under the current thread's debug hook.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastHookMode {
    /// No hook event can fire.
    Plain,
    /// Count, call, or return events can fire. The fast loop counts
    /// instructions for `count` and leaves calls and returns to the
    /// dispatch loop for `calls`.
    Hooked { count: bool, calls: bool },
    /// Line events can fire: the dispatch loop only.
    Off,
}

/// The thread whose count hook the fast loop is counting for. Installing or
/// removing any hook invalidates the lease: the loop then stops at its next
/// counted instruction and leaves the hook's new count alone.
#[derive(Clone, Copy)]
struct CountLease {
    thread: Option<ThreadHandle>,
    invalidated: bool,
    /// The leased hook reports no call or return, so the fast loop may
    /// evaluate a call without a frame and charge its instructions.
    count_only: bool,
}

#[derive(Clone, Copy, Default)]
struct DebugHookMask {
    call: bool,
    return_: bool,
    line: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum DebugHookEvent {
    Call,
    TailCall,
    Return,
    TailReturn,
    Line(u32),
    Count,
}

#[derive(Clone, Copy)]
pub(crate) enum InlineCallArguments<'a> {
    Values(&'a [Value]),
    Registers { start: usize, count: usize },
}

/// Interior frame state that most activations never allocate.
///
/// Varargs, open results, close slots, native-resume
/// bookkeeping, traceback snapshots, and debug-hook scratch state only exist
/// on frames that use them. Boxing them behind one lazily created allocation
/// keeps the hot Lua-to-Lua call path moving and dropping a small frame
/// instead of a wide struct with a dozen mostly-empty collections.
#[derive(Default)]
pub(crate) struct FrameExtras {
    close_slots: Vec<u8>,
    host_close_handler: Option<NativeFunctionHandle>,
    /// Stack at the point an `xpcall` target failed. Lua invokes error
    /// handlers while that target is still inspectable. The failed frames are
    /// parked in the continuation, so the handler carries this debug snapshot.
    traceback: Option<Traceback>,
    last_hook_line: Option<u32>,
    last_hook_pc: Option<usize>,
    skip_hook_pc_once: Option<usize>,
    debug_hook_subject: Option<Value>,
    suppress_debug_hooks: bool,
    debug_hook_transfer: (usize, usize),
    suppress_return_hook: bool,
}

/// The field order keeps the offsets the dispatch loop's code was tuned for
/// (its speed shifts with them), and the frame at 128 bytes, two cache lines.
/// The call site is boxed and the open results sit in the room that left.
#[repr(C)]
pub(crate) struct Frame {
    last_interpreted_pc: frame::LastPc,
    upvalues: CaptureCells,
    call_site: Option<Box<FrameSite>>,
    /// The values of the last call with an open result count, until the
    /// instruction that consumes them runs. Kept here, not in the extras, so
    /// a frame that uses open results keeps its fast return.
    open: Option<Box<OpenResults>>,
    caller: Option<ReturnTarget>,
    chunk: frame::FrameChunk,
    environment: TableHandle,
    pc: usize,
    extras: Option<Box<FrameExtras>>,
    closure: Option<ClosureHandle>,
    /// First value-stack slot of this activation's registers.
    base: u32,
    /// One past the last value-stack slot of this activation's registers.
    top: u32,
    tail_calls: u32,
    /// The arguments a vararg function received past its parameters. They sit
    /// on the value stack in the slots just below `base`, so a call of a
    /// vararg function allocates nothing. The caller's window ends where they
    /// start. The count is in the low 30 bits (`Frame::vararg_count`); bit 31
    /// is [`frame::VARARG_TRUNCATED`]: the caller was cut short to make room
    /// for them, and whatever removes this activation gives it its window
    /// back.
    ///
    /// Bit 30 is the plain mark, [`frame::VARARG_PLAIN`], and a marked record
    /// has no other bit set here: the word equals the mark. It says that this
    /// record is a plain activation the fast loop's call pushed, and that
    /// nothing has touched it since in a way the plain call or the plain
    /// return would have to look at. One compare of the word then stands
    /// for every test of the record the two would make:
    ///
    /// - `caller` is a [`ReturnTarget::Direct`], pushed over a live Lua
    ///   caller in the record just below;
    /// - `extras` is `None`, `chunk` and `upvalues` are borrowed from
    ///   `closure`, and `call_site` holds no [`FrameSite::Named`];
    /// - `tail_calls` is zero, and there are no varargs and no cut caller.
    ///
    /// Only [`stack::CallStack`]'s plain push sets the mark. Whatever else
    /// writes one of the fields named above, or moves the record, clears it
    /// ([`Frame::clear_plain_mark`]); a write of the vararg count clears it
    /// by itself. The mark survives the return, so the next call at this
    /// depth finds a retired record it can trust.
    ///
    /// Named apart from the accessor so that nothing reads the raw word
    /// where it wants the count.
    vararg_word: u32,
    /// The [`oximond::runtime::ClosureCallParts::shape`] of `closure` that
    /// `environment` and `upvalues` were taken at, or
    /// [`oximond::runtime::NO_SHAPE`]. A call that finds the same closure
    /// here at the same shape leaves both as they are.
    shape: u32,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Frame>() == 128);

/// The values of a call with an open result count. The box stays with the
/// frame record, so its buffer is reused by the next open call there.
#[derive(Default)]
struct OpenResults {
    /// Whether `values` hold open results awaiting their consumer.
    active: bool,
    values: Vec<Value>,
}

enum LogicalFrame<'frame> {
    Frame(&'frame Frame),
    TailCall,
    OutOfRange,
}

#[cfg(test)]
mod tests;
