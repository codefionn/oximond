//! Coroutine creation, resume, yield, status, and runtime root bookkeeping.

use crate::bytecode::{Instruction, OpCode};
use crate::compat::CompatibilityProfile;
use crate::compat::ExecutionMode;
use crate::runtime::{GcRootBuilder, RuntimeError, TableHandle, ThreadHandle, Value};
use crate::vm::debug_info::{capture_error_traceback, capture_traceback};
use crate::vm::errors::{locate_runtime_error, vm_error_object};
use crate::vm::fast_path::{FastExit, Micro, Stop};
use crate::vm::frame::LastPc;
use crate::vm::stack::CallStack;
use crate::vm::{
    CallReturn, Coroutine, CoroutineState, HostCoroutineStatus, IndexCompletion,
    MAX_NESTED_RESUMES, ReturnTarget, UnwindAction, Vm, VmError, VmResult,
};

/// The most resume arguments a switch from the fast loop keeps on the Rust
/// stack.
const SMALL_ARGUMENTS: usize = 4;

/// Of the resumes of a thread that is taken to run long
/// (`Coroutine::long_runs`), one in this many tries the short run again.
const LONG_RUN_RETRY: u8 = 64;

/// The most values a direct yield passes in the VM's small buffer.
pub(crate) const QUICK_VALUES: usize = 4;

/// The values of a direct `coroutine.yield` from the fast loop, in a buffer
/// of the VM that a resume takes them from, so that a switch builds no
/// vector. The slots from `len` on are nil.
pub(crate) struct QuickYield {
    /// A yield parked its values here and nothing has taken them.
    pub(crate) pending: bool,
    /// The run that parked them ran nothing that could write a table, a
    /// cell or a global: only the short run of `Vm::run_micro`.
    pure: bool,
    pub(crate) len: u8,
    /// The call of the yielding thread that takes the resume's values.
    pub(crate) target: CallReturn,
    pub(crate) values: [Value; QUICK_VALUES],
}

impl Default for QuickYield {
    fn default() -> Self {
        Self {
            pending: false,
            pure: false,
            len: 0,
            target: CallReturn::default(),
            values: [const { Value::Nil }; QUICK_VALUES],
        }
    }
}

impl QuickYield {
    /// The parked values.
    pub(crate) fn values(&self) -> &[Value] {
        &self.values[..usize::from(self.len)]
    }

    /// Whether the run that parked the values changed nothing a caller
    /// could have read before it: no table, cell or global was written, and
    /// no code but the thread's own few instructions ran.
    pub(crate) const fn pure(&self) -> bool {
        self.pure
    }

    /// Drops the parked values.
    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        for value in &mut self.values[..usize::from(self.len)] {
            *value = Value::Nil;
        }
        self.len = 0;
        // The flag describes parked values; with none parked a later reader
        // must not inherit it.
        self.pure = false;
    }

    /// The parked values behind the `true` of a resume that succeeded, in a
    /// vector.
    #[cold]
    fn take_with_status(&mut self) -> Vec<Value> {
        let mut values = Vec::with_capacity(usize::from(self.len) + 1);
        values.push(Value::Boolean(true));
        for value in &mut self.values[..usize::from(self.len)] {
            values.push(std::mem::replace(value, Value::Nil));
        }
        self.len = 0;
        self.pure = false;
        values
    }
}

/// How a resume that ran a thread ended.
pub(crate) enum Resumed {
    /// The thread yielded values that fit the VM's quick buffer
    /// ([`QuickYield`]), which the caller takes and clears.
    Yielded,
    /// Any other end, as the vector of a resume's results. The answer is
    /// boxed: an error makes the bare result 56 bytes, and the usual end,
    /// `Yielded`, would come back through memory that the caller reads in
    /// wider loads than the callee wrote it in (a store-forwarding stall on
    /// every switch). Boxed, the whole `Option<Resumed>` is two registers.
    Done(Box<VmResult<Vec<Value>>>),
}

impl Resumed {
    #[cold]
    fn done(answer: VmResult<Vec<Value>>) -> Self {
        Self::Done(Box::new(answer))
    }
}

/// How [`Vm::resume_direct`] went.
pub(crate) enum Direct<R> {
    /// The thread ran its few instructions to a yield, and the caller's
    /// closure took the values out of its registers.
    Yielded(R),
    /// The thread ran, and its run ended some other way.
    Ran(Resumed),
    /// Not a resume for this path; nothing changed.
    Declined,
}

/// How a direct resume ([`Vm::resume_direct`]) passes values between the
/// resumer and the thread. A trait and not two closures, so that both
/// halves are inlined into the resume: a closure passed down is a call, and
/// each one costs as much as the copy it makes.
pub(crate) trait Exchange<R> {
    /// Fills `slots`, the registers the thread's yield gives its results,
    /// from the resumer.
    fn give(&mut self, slots: &mut [Value]);
    /// Takes `yielded`, the values of the thread's next yield, out of the
    /// thread's registers.
    fn take(self, yielded: &[Value]) -> R;
    /// Whether the resumer is a native trace in the middle of its run (see
    /// [`Vm::resume_for_trace`]).
    const FROM_TRACE: bool;
}

/// The exchange of a resume from the fast loop: the arguments are `count`
/// registers of the caller's window from `first`, and the results go to
/// `wanted` registers from `base`, behind a `true` unless the call is a
/// wrapper's. The caller checked both ranges against the window.
///
/// The window and its frame are held by address, found once: the resumer's
/// stack is passed along beside the exchange for a resume that turns into
/// a nested one, and nothing touches it between the two halves.
struct WindowExchange {
    frame: std::ptr::NonNull<crate::vm::Frame>,
    registers: std::ptr::NonNull<[Value]>,
    first: usize,
    count: usize,
    base: usize,
    wanted: usize,
    wrapped: bool,
}

impl Exchange<()> for WindowExchange {
    const FROM_TRACE: bool = false;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn give(&mut self, slots: &mut [Value]) {
        // SAFETY: the registers of the resumer's top frame, which stays
        // live and unmoved while its call of the resume is in progress.
        // `slots` are registers of the thread's stack, another allocation.
        let registers = unsafe { self.registers.as_ref() };
        let mut given = registers[self.first..self.first + self.count].iter();
        for slot in slots {
            match given.next() {
                Some(value) => crate::vm::registers::store_clone_scalar(slot, value),
                None => crate::vm::registers::set_register(slot, Value::Nil),
            }
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn take(mut self, yielded: &[Value]) {
        // SAFETY: as in `give`; the short run between the two ran on the
        // thread's stack and left the resumer's alone, and no reference
        // into the resumer's stack is in use.
        let (frame, registers) = unsafe { (self.frame.as_mut(), self.registers.as_mut()) };
        frame.clear_open_results();
        let mut slots = registers[self.base..self.base + self.wanted].iter_mut();
        if !self.wrapped
            && let Some(slot) = slots.next()
        {
            crate::vm::registers::set_register(slot, Value::Boolean(true));
        }
        let mut yielded = yielded.iter();
        for slot in slots {
            match yielded.next() {
                Some(value) => crate::vm::registers::store_clone_scalar(slot, value),
                None => crate::vm::registers::set_register(slot, Value::Nil),
            }
        }
    }
}

/// The exchange of a resume from a native trace: the arguments are moved
/// out of a slice the helper built, and `take` makes the helper's answer of
/// the values the thread yields.
struct TraceExchange<'a, F> {
    arguments: &'a mut [Value],
    take: F,
}

impl<R, F: FnOnce(&[Value]) -> R> Exchange<R> for TraceExchange<'_, F> {
    const FROM_TRACE: bool = true;

    #[inline(always)]
    fn give(&mut self, slots: &mut [Value]) {
        let mut given = self.arguments.iter_mut();
        for slot in slots {
            match given.next() {
                Some(value) => crate::vm::registers::move_value(slot, value),
                None => crate::vm::registers::set_register(slot, Value::Nil),
            }
        }
        // Values the call does not take are dropped, so that nothing is
        // left in the slice.
        for rest in given {
            *rest = Value::Nil;
        }
    }

    #[inline(always)]
    fn take(self, yielded: &[Value]) -> R {
        (self.take)(yielded)
    }
}

/// How [`Vm::resume_plain`] went.
enum Plain {
    /// The resume ran, whatever came of it.
    Done(VmResult<Vec<Value>>),
    /// The thread yielded values that wait in the quick buffer.
    Yielded,
    /// The handle is not a coroutine; nothing changed.
    Unknown,
    /// The resume is not a plain one; nothing changed.
    General,
}

/// The coroutines of a VM, by handle. A thread handle names a slot of the
/// runtime's arena and a generation of it, so a table indexed by slot finds
/// a coroutine with one comparison. The slots go in handle order, which is
/// the order a map keyed by handle would give.
#[derive(Default)]
pub(crate) struct CoroutineTable {
    slots: Vec<Option<(ThreadHandle, Coroutine)>>,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
impl CoroutineTable {
    #[inline]
    pub(crate) fn get(&self, thread: &ThreadHandle) -> Option<&Coroutine> {
        match self.slots.get(thread.index() as usize) {
            Some(Some((held, coroutine))) if held == thread => Some(coroutine),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, thread: &ThreadHandle) -> Option<&mut Coroutine> {
        match self.slots.get_mut(thread.index() as usize) {
            Some(Some((held, coroutine))) if held == thread => Some(coroutine),
            _ => None,
        }
    }

    pub(crate) fn contains_key(&self, thread: &ThreadHandle) -> bool {
        self.get(thread).is_some()
    }

    /// Adds `coroutine` under `thread`, returning the one it replaces.
    pub(crate) fn insert(
        &mut self,
        thread: ThreadHandle,
        coroutine: Coroutine,
    ) -> Option<Coroutine> {
        let index = thread.index() as usize;
        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || None);
        }
        self.slots[index]
            .replace((thread, coroutine))
            .map(|(_, replaced)| replaced)
    }

    pub(crate) fn remove(&mut self, thread: &ThreadHandle) -> Option<Coroutine> {
        let slot = self.slots.get_mut(thread.index() as usize)?;
        match slot {
            Some((held, _)) if held == thread => slot.take().map(|(_, coroutine)| coroutine),
            _ => None,
        }
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &ThreadHandle> {
        self.slots.iter().flatten().map(|(thread, _)| thread)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&ThreadHandle, &Coroutine)> {
        self.slots
            .iter()
            .flatten()
            .map(|(thread, coroutine)| (thread, coroutine))
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut Coroutine> {
        self.slots
            .iter_mut()
            .flatten()
            .map(|(_, coroutine)| coroutine)
    }
}

impl std::ops::Index<&ThreadHandle> for CoroutineTable {
    type Output = Coroutine;

    fn index(&self, thread: &ThreadHandle) -> &Coroutine {
        self.get(thread).expect("coroutine exists")
    }
}

/// A stack that waits beneath a resume, for the collector to root and the
/// debug library to patch: the resumer's own, left where its owner has it.
///
/// The resumer passes its stack to the resume as a `&mut`, and its owner does
/// nothing with it until the resume returns. [`Vm::enter_thread`] records it
/// when the thread starts and [`Vm::leave_thread`] drops the record before the
/// resume returns, on every path out, so a record always points at a stack
/// that is alive and that nothing else touches.
#[derive(Clone, Copy)]
pub(crate) struct ParkedStack(std::ptr::NonNull<CallStack>);

impl ParkedStack {
    fn new(stack: &mut CallStack) -> Self {
        Self(std::ptr::NonNull::from(stack))
    }

    #[allow(unsafe_code)]
    pub(crate) fn get(&self) -> &CallStack {
        // SAFETY: as the type says, the stack is alive and no other
        // reference to it is in use while the record exists.
        unsafe { self.0.as_ref() }
    }

    #[allow(unsafe_code)]
    pub(crate) fn get_mut(&mut self) -> &mut CallStack {
        // SAFETY: as for `get`.
        unsafe { self.0.as_mut() }
    }
}

/// What a resume saves of the thread that made it, to restore on the way
/// back (see `Vm::enter_thread`).
#[derive(Clone, Copy)]
struct Switch {
    previous_coroutine: Option<ThreadHandle>,
    previous_environment: TableHandle,
}

/// A resume the fast loop made in place: the thread's stack and the
/// resumer's changed places, and the loop went on in the thread's top frame
/// without leaving ([`Vm::resume_in_place`]). The loop holds one `&mut
/// CallStack` for its whole run, so the thread runs where the resumer's
/// stack was, and the resumer's frames wait in the allocation the thread's
/// came from.
///
/// The record lasts until the thread yields back to the loop
/// ([`Vm::yield_in_place`]), which exchanges the stacks again, or until the
/// loop ends some other way with the thread still running, which turns the
/// resume into the nested one it would otherwise have been
/// ([`Vm::leave_in_place`]). While it lasts the VM is in every other respect
/// what a nested resume makes it: the thread is the running one, its state
/// is `Running`, and `Vm::resumer_stacks` has the resumer's frames, by the
/// address below.
pub(crate) struct InPlace {
    /// The resumer's frames, in the box of the thread's stack. Owned by the
    /// record: leaking the box made it and `Box::from_raw` ends it. Held as
    /// a pointer because `Vm::resumer_stacks` names the same allocation.
    parked: std::ptr::NonNull<CallStack>,
    thread: ThreadHandle,
    switch: Switch,
    /// The resumer's last interpreted pc before the call, for a call the
    /// general path has to make again.
    previous: LastPc,
    /// The call's result registers in the resumer's top frame.
    base: u8,
    results: u8,
    /// The call is one of a function `coroutine.wrap` made.
    wrapped: bool,
}

/// What a resume made from a native trace gave, held for the interpreter's
/// replay of the call. A trace that runs a coroutine cannot take every answer
/// (an error, a dead coroutine, values of other types than the recording
/// saw), and it cannot run the coroutine again. It leaves before the call, as
/// every trace does when a library call misses, and the interpreter executes
/// the call: [`Vm::resume_coroutine_mode`] hands it this instead of running
/// the thread.
pub(crate) struct ResumeMemo {
    thread: ThreadHandle,
    result: VmResult<Vec<Value>>,
}

impl ResumeMemo {
    pub(crate) const fn new(thread: ThreadHandle, result: VmResult<Vec<Value>>) -> Self {
        Self { thread, result }
    }

    /// Roots what the resume gave while the memo waits for its replay: the
    /// values of a success, and of a failure the values it holds (the values
    /// of a host yield, the object a raised error carries), which the
    /// interpreter's replay hands to its caller as they are.
    pub(crate) fn root(&self, roots: &mut GcRootBuilder) {
        match &self.result {
            Ok(values) => roots.values(values.iter()),
            Err(error) => root_error_values(error, roots),
        }
    }
}

/// Roots the values a [`VmError`] holds.
fn root_error_values(error: &VmError, roots: &mut GcRootBuilder) {
    match error {
        VmError::Yield(values) => roots.values(values.iter()),
        VmError::HookError(inner) => root_error_values(inner, roots),
        _ => {
            if let Ok(value) = crate::vm::errors::vm_error_object(error) {
                roots.value(&value);
            }
        }
    }
}

impl Vm {
    /// Allocates a thread that will receive its entry function on first resume.
    ///
    /// # Errors
    ///
    /// Returns an error when the thread cannot be allocated or its GC roots
    /// cannot be stored.
    pub fn create_host_coroutine(&mut self) -> VmResult<ThreadHandle> {
        let handle = self.runtime.create_thread()?;
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(self.thread_environment));
        self.runtime.set_thread_roots(handle, roots.finish())?;
        let displaced = self.coroutines.insert(
            handle,
            Coroutine {
                environment: self.thread_environment,
                state: CoroutineState::New(None),
                traceback: None,
                failed: None,
                roots_stale: false,
                listed: false,
                long_runs: false,
                long_resumes: 0,
            },
        );
        // The arena gave the slot of a thread the collector freed before the
        // table dropped its coroutine (`Vm::prune_collected_coroutines`).
        if let Some(mut displaced) = displaced {
            self.close_coroutine_upvalues(&mut displaced);
        }
        Ok(handle)
    }

    /// Installs the entry of a new host thread. Results have coroutine.resume's success prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine, when it has
    /// already been started, when its GC roots cannot be stored, or whatever
    /// the first resume raises.
    // The by-value `Value` argument is part of the public embedding API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn start_host_coroutine(
        &mut self,
        thread: ThreadHandle,
        function: Value,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        let coroutine = self
            .coroutines
            .get_mut(&thread)
            .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
        if !matches!(coroutine.state, CoroutineState::New(_)) {
            return Err(RuntimeError::message("coroutine has already been started").into());
        }
        coroutine.state = CoroutineState::New(Some(function.clone()));
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(coroutine.environment));
        roots.value(&function);
        self.runtime.set_thread_roots(thread, roots.finish())?;
        self.resume_host_coroutine(thread, arguments)
    }

    /// Returns the initial function still retained by an unstarted coroutine.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine.
    pub fn host_coroutine_entry(&self, thread: ThreadHandle) -> VmResult<Option<Value>> {
        let coroutine = self
            .coroutines
            .get(&thread)
            .filter(|_| self.runtime.thread_exists(thread))
            .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
        Ok(match &coroutine.state {
            CoroutineState::New(entry) => entry.clone(),
            _ => None,
        })
    }

    /// Resumes a coroutine without retaining the embedding application's stack.
    /// Allocator failures return a typed error; other outcomes retain Lua's
    /// coroutine.resume success prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine or on an
    /// allocator failure; a coroutine that raises reports its failure through
    /// the success prefix in the results.
    pub fn resume_host_coroutine(
        &mut self,
        thread: ThreadHandle,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        self.resume_coroutine_mode(&mut CallStack::new(), thread, arguments, true)
    }

    /// The lifecycle state of a host thread.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine.
    pub fn host_coroutine_status(&self, thread: ThreadHandle) -> VmResult<HostCoroutineStatus> {
        let coroutine = self
            .coroutines
            .get(&thread)
            .filter(|_| self.runtime.thread_exists(thread))
            .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
        Ok(match coroutine.state {
            CoroutineState::New(_) => HostCoroutineStatus::New,
            CoroutineState::Running => HostCoroutineStatus::Running,
            CoroutineState::Suspended { .. } => HostCoroutineStatus::Suspended,
            CoroutineState::Dead => HostCoroutineStatus::Dead,
        })
    }

    /// Distinguishes a coroutine that failed from one that returned normally.
    ///
    /// # Errors
    ///
    /// Returns an error when `thread` is not a live coroutine.
    pub fn host_coroutine_failed(&self, thread: ThreadHandle) -> VmResult<bool> {
        let coroutine = self
            .coroutines
            .get(&thread)
            .filter(|_| self.runtime.thread_exists(thread))
            .ok_or_else(|| RuntimeError::message("invalid coroutine"))?;
        Ok(matches!(coroutine.state, CoroutineState::Dead) && coroutine.traceback.is_some())
    }

    /// Checks Lua 5.1's C-call and metamethod yield restrictions before a C unwind.
    ///
    /// # Errors
    ///
    /// Returns an error when the yield happens outside a coroutine, or when a
    /// suspended frame above the base does not return through a resumable
    /// target for the active compatibility profile.
    pub fn validate_host_yield(&self) -> VmResult<()> {
        let Some(thread) = self.current_coroutine else {
            return Err(RuntimeError::message("attempt to yield from outside a coroutine").into());
        };
        let Some((callback_thread, allowed)) = self.host_yield_boundaries.last() else {
            return Err(self.yield_boundary_error().into());
        };
        if !allowed
            || *callback_thread != Some(thread)
            || self
                .host_yield_boundaries
                .iter()
                .enumerate()
                .take(self.host_yield_boundaries.len() - 1)
                .any(|(index, (owner, _))| {
                    *owner == Some(thread) && !self.host_boundary_allows_reentrant_yield(index)
                })
        {
            return Err(self.yield_boundary_error().into());
        }
        Ok(())
    }

    /// Returns the typed completion signal a yielding host callback must return.
    pub fn request_host_yield(&self, values: Vec<Value>) -> VmError {
        match self.validate_host_yield() {
            Ok(()) => VmError::Yield(values),
            Err(error) => error,
        }
    }

    pub(crate) fn create_coroutine(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let entry = match arguments.first() {
            Some(entry @ Value::Closure(_)) => entry.clone(),
            Some(entry @ (Value::NativeFunction(_) | Value::BuiltinFunction(_)))
                if self.compatibility != CompatibilityProfile::Lua51 =>
            {
                entry.clone()
            }
            Some(Value::NativeFunction(_) | Value::BuiltinFunction(_)) => {
                return Err(RuntimeError::argument_error(1, "Lua function expected").into());
            }
            _ if self.compatibility == CompatibilityProfile::Lua51 => {
                return Err(RuntimeError::argument_error(1, "Lua function expected").into());
            }
            argument => return Err(RuntimeError::bad_argument(1, "function", argument).into()),
        };
        let handle = self.create_host_coroutine()?;
        let coroutine = self
            .coroutines
            .get_mut(&handle)
            .expect("created coroutine exists");
        coroutine.state = CoroutineState::New(Some(entry.clone()));
        let mut roots = GcRootBuilder::new();
        roots.value(&Value::Table(self.thread_environment));
        roots.value(&entry);
        self.runtime.set_thread_roots(handle, roots.finish())?;
        Ok(vec![Value::Thread(handle)])
    }

    pub(crate) fn coroutine_status(&self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let thread = coroutine_argument(arguments, self.compatibility)?;
        let Some(coroutine) = self.coroutines.get(&thread) else {
            return Err(RuntimeError::message("invalid coroutine").into());
        };
        let status = match coroutine.state {
            CoroutineState::New(_) | CoroutineState::Suspended { .. } => "suspended",
            CoroutineState::Running if self.current_coroutine == Some(thread) => "running",
            CoroutineState::Running => "normal",
            CoroutineState::Dead => "dead",
        };
        Ok(vec![Value::from(status)])
    }

    pub(crate) fn coroutine_running(&mut self) -> VmResult<Vec<Value>> {
        let modern = matches!(
            self.compatibility,
            CompatibilityProfile::Lua52
                | CompatibilityProfile::Lua53
                | CompatibilityProfile::Lua54
                | CompatibilityProfile::Lua55
        );
        if let Some(thread) = self.current_coroutine {
            let mut results = vec![Value::Thread(thread)];
            if modern {
                results.push(Value::Boolean(false));
            }
            return Ok(results);
        }
        if !modern {
            return Ok(vec![Value::Nil]);
        }
        let thread = if let Some(thread) = self.main_thread {
            thread
        } else {
            let thread = self.runtime.create_thread()?;
            self.main_thread = Some(thread);
            thread
        };
        Ok(vec![Value::Thread(thread), Value::Boolean(true)])
    }

    /// `coroutine.isyieldable`: false on the main thread, true inside a
    /// coroutine. Lua 5.4 and later accept a thread to ask about; Lua 5.3 and
    /// `LuaJIT` ignore their arguments.
    pub(crate) fn coroutine_isyieldable(
        &self,
        frames: &CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let thread = if arguments.is_empty()
            || matches!(
                self.compatibility,
                CompatibilityProfile::Lua53 | CompatibilityProfile::LuaJit21
            ) {
            self.current_coroutine
        } else {
            let thread = coroutine_argument(arguments, self.compatibility)?;
            (self.main_thread != Some(thread)).then_some(thread)
        };
        let yieldable = thread.is_some_and(|thread| {
            self.current_coroutine != Some(thread) || !self.yield_blocked(frames)
        });
        Ok(vec![Value::Boolean(yieldable)])
    }

    /// The error for a yield across a boundary the profile cannot resume.
    /// Stock raises it from the C function `yield`, so Lua 5.1 to 5.5 carry
    /// no position. `LuaJIT` reports the calling Lua line.
    pub(crate) fn yield_boundary_error(&self) -> RuntimeError {
        match self.compatibility {
            CompatibilityProfile::Lua51 => RuntimeError::Raised(Value::from(
                "attempt to yield across metamethod/C-call boundary",
            )),
            CompatibilityProfile::LuaJit21 => {
                RuntimeError::message("attempt to yield across C-call boundary")
            }
            _ => RuntimeError::Raised(Value::from("attempt to yield across a C-call boundary")),
        }
    }

    /// Runs `body`, which calls Lua metamethods from VM code on behalf of a
    /// library function, as a C call: a yield inside it is refused the way
    /// stock Lua refuses a yield across a C function without a continuation.
    pub(crate) fn in_c_call<T>(
        &mut self,
        body: impl FnOnce(&mut Self) -> VmResult<T>,
    ) -> VmResult<T> {
        self.host_yield_boundaries
            .push((self.current_coroutine, false));
        let result = body(self);
        self.host_yield_boundaries.pop();
        result
    }

    /// True when a yield from the current coroutine would cross a host call,
    /// metamethod or other boundary that cannot be resumed.
    fn yield_blocked(&self, frames: &CallStack) -> bool {
        self.host_yield_boundaries
            .iter()
            .enumerate()
            .any(|(index, (thread, _))| {
                *thread == self.current_coroutine
                    && !self.host_boundary_allows_reentrant_yield(index)
            })
            || !self.frames_allow_yield(frames)
    }

    /// Whether `continuation`, pushed by the arithmetic instruction at the
    /// top of `caller`, is the right operand's metamethod reached through
    /// the string library's `__add` and its relatives. Lua 5.4 and 5.5 give
    /// strings arithmetic metamethods written in C, which call the other
    /// operand's handler with `lua_call`, so a yield there crosses a C
    /// call. Only a left string operand reaches that code, and only the
    /// arithmetic operators have such metamethods. Runs when a yield asks.
    fn is_string_arithmetic_callback(
        &self,
        continuation: &crate::vm::ScalarContinuation,
        caller: &crate::vm::Frame,
    ) -> bool {
        if !matches!(
            self.compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) || continuation.kind != crate::vm::ScalarKind::Value
            || !matches!(continuation.left, Value::String(_))
            || continuation.right.is_none()
        {
            return false;
        }
        let Some(pc) = caller
            .last_interpreted_pc
            .get()
            .or_else(|| caller.pc.checked_sub(1))
        else {
            return false;
        };
        caller
            .chunk
            .code
            .get(pc)
            .and_then(|instruction| instruction.opcode().ok())
            .is_some_and(|opcode| {
                matches!(
                    opcode,
                    OpCode::AddK
                        | OpCode::SubtractK
                        | OpCode::MultiplyK
                        | OpCode::DivideK
                        | OpCode::ModuloK
                        | OpCode::Add
                        | OpCode::Subtract
                        | OpCode::Multiply
                        | OpCode::Divide
                        | OpCode::FloorDivide
                        | OpCode::Modulo
                        | OpCode::Power
                )
            })
    }

    /// The one rule for which suspended Lua frames a yield may cross. Every
    /// frame above the base must return through a resumable target: a plain
    /// call, or, outside Lua 5.1, a pcall or xpcall frame, a metamethod
    /// continuation or a `__close` handler.
    // Two groups of return targets share the `resumable_ok` answer at separate
    // positions in the match; merging their arms would move a group away from
    // the arms it is documented with.
    #[allow(clippy::match_same_arms)]
    pub(crate) fn frames_allow_yield(&self, frames: &CallStack) -> bool {
        // A thread's bottom frame returns to the resumer; there is nothing
        // above it to cross.
        if frames.len() <= 1 {
            return true;
        }
        let lua51 = self.compatibility == CompatibilityProfile::Lua51;
        let resumable_ok = !lua51;
        frames
            .iter()
            .enumerate()
            .skip(1)
            .all(|(index, frame)| match &frame.caller {
                // Lua 5.1 runs a generic-for iterator through a C call.
                Some(ReturnTarget::Direct(_)) => {
                    !lua51
                        || !frames[index - 1]
                            .pc
                            .checked_sub(1)
                            .and_then(|pc| frames[index - 1].chunk.debug.call_site_at(pc))
                            .is_some_and(crate::bytecode::CallSite::is_iterator)
                }
                // An operator metamethod, except one that Lua 5.4's string
                // arithmetic calls from C.
                Some(ReturnTarget::Scalar(continuation)) => {
                    resumable_ok
                        && !self.is_string_arithmetic_callback(continuation, &frames[index - 1])
                }
                // pcall and xpcall frames, and the metamethod frames whose
                // continuation lives in the frame record: a resumed yield returns
                // through them like any other return. Lua 5.1 refuses all of them.
                Some(
                    ReturnTarget::Protected(_)
                    | ReturnTarget::ExtendedProtectedTarget(_)
                    | ReturnTarget::ConcatRange(_)
                    | ReturnTarget::CloseLocal
                    | ReturnTarget::UnwindClose(_),
                ) => resumable_ok,
                // string.gsub reads a table replacement through lua_gettable from
                // C, so its __index frame stays a C-call boundary.
                Some(ReturnTarget::Index(continuation)) => {
                    resumable_ok && matches!(continuation.completion, IndexCompletion::Frame)
                }
                Some(
                    ReturnTarget::SetMetamethod
                    | ReturnTarget::IndexResult(_)
                    | ReturnTarget::Operator(_),
                ) => resumable_ok,
                // LuaJIT's tostring runs __tostring as an ordinary Lua call.
                // Every other profile calls it from C, and print and
                // string.format refuse in LuaJIT too (they never push this
                // target).
                Some(ReturnTarget::ToString(_)) => {
                    self.compatibility == CompatibilityProfile::LuaJit21
                }
                _ => false,
            })
    }

    pub(crate) fn close_coroutine(
        &mut self,
        caller: &mut CallStack,
        arguments: &[Value],
    ) -> VmResult<Vec<Value>> {
        let thread = coroutine_argument(arguments, self.compatibility)?;
        let mut roots = GcRootBuilder::new();
        roots.extend(&Self::gc_roots(self.thread_environment, caller, arguments));
        self.suspended_roots.push(roots.finish());
        let result = self.close_host_coroutine(thread);
        self.suspended_roots.pop();
        result
    }

    /// Lua 5.4's wrapper closes a failed thread before raising its error.
    pub(crate) fn resume_wrapped_coroutine_mode(
        &mut self,
        caller: &mut CallStack,
        thread: ThreadHandle,
        arguments: Vec<Value>,
        preserve_memory_error: bool,
    ) -> VmResult<Vec<Value>> {
        let result =
            self.resume_coroutine_mode(caller, thread, arguments, preserve_memory_error)?;
        if matches!(
            self.compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) && matches!(result.first(), Some(Value::Boolean(false)))
        {
            let closed = self.close_coroutine(caller, &[Value::Thread(thread)])?;
            if matches!(closed.first(), Some(Value::Boolean(false))) {
                return Ok(closed);
            }
        }
        Ok(result)
    }

    pub(crate) fn resume_coroutine(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: Vec<Value>,
    ) -> VmResult<Vec<Value>> {
        self.resume_coroutine_mode(resumer_frames, thread, arguments, false)
    }

    pub(crate) fn resume_coroutine_mode(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        mut arguments: Vec<Value>,
        preserve_memory_error: bool,
    ) -> VmResult<Vec<Value>> {
        let result = self.resume_mode_known(
            resumer_frames,
            thread,
            &mut arguments,
            preserve_memory_error,
        );
        self.recycle_values(arguments);
        result.unwrap_or_else(|| Err(RuntimeError::message("invalid coroutine").into()))
    }

    /// [`Self::resume_coroutine_mode`] for a handle that may not be a
    /// coroutine: `None` for one that is not, with nothing changed. The
    /// resume takes the values out of `arguments`, which the caller keeps.
    pub(crate) fn resume_mode_known(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: &mut [Value],
        preserve_memory_error: bool,
    ) -> Option<VmResult<Vec<Value>>> {
        match self.resume_mode_quick(resumer_frames, thread, arguments, preserve_memory_error)? {
            Resumed::Yielded => Some(Ok(self.quick.take_with_status())),
            Resumed::Done(answer) => Some(*answer),
        }
    }

    /// [`Self::resume_mode_known`] that leaves the values of a plain yield
    /// in the quick buffer.
    pub(crate) fn resume_mode_quick(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: &mut [Value],
        preserve_memory_error: bool,
    ) -> Option<Resumed> {
        // The call a native trace already made and left before it could take
        // the answer (see `ResumeMemo`).
        if self.resume_memo.is_some()
            && let Some(result) = self.replay_resume(thread)
        {
            return Some(Resumed::done(result));
        }
        self.resume_quick(resumer_frames, thread, arguments, preserve_memory_error)
    }

    /// The answer a native trace's resume already got from `thread`, if the
    /// memo is for it, which the interpreter's replay of the call takes in
    /// place of running the thread.
    #[cold]
    #[inline(never)]
    fn replay_resume(&mut self, thread: ThreadHandle) -> Option<VmResult<Vec<Value>>> {
        match self.resume_memo.take() {
            Some(memo) if memo.thread == thread => {
                self.jit.statistics.coroutine_trace_replays += 1;
                Some(memo.result)
            }
            _ => None,
        }
    }

    /// The values of `arguments` in a vector of their own, for the general
    /// resume, which hands them to the thread's frames.
    #[cold]
    fn owned_arguments(arguments: &mut [Value]) -> Vec<Value> {
        arguments
            .iter_mut()
            .map(|value| std::mem::replace(value, Value::Nil))
            .collect()
    }

    /// The usual resume: `thread` waits in a plain `coroutine.yield` that
    /// wants a fixed number of results, which the registers of its top frame
    /// hold room for. Nothing is built that the switch does not need: the
    /// values move into the registers, the thread runs on the fast loop to
    /// its next plain yield, and its stack parks again. Any other state
    /// of the thread, or any other end of the run, goes the general way
    /// (`General` before anything has changed, or through
    /// [`Self::drive_thread`] after).
    #[inline]
    #[allow(unsafe_code)]
    fn resume_plain(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: &mut [Value],
        preserve_memory_error: bool,
    ) -> Plain {
        let nested_full = self.resumer_stacks.len() >= MAX_NESTED_RESUMES;
        let Some(coroutine) = self.coroutines.get_mut(&thread) else {
            return Plain::Unknown;
        };
        let CoroutineState::Suspended {
            frames,
            yield_return,
            host_resume: None,
        } = &coroutine.state
        else {
            return Plain::General;
        };
        let target = *yield_return;
        let room = usize::from(target.base) + usize::from(target.results);
        // The results must fit the registers of the thread's top frame.
        if nested_full
            || target.results == 0
            || frames
                .last_frame()
                .is_none_or(|frame| ((frame.top - frame.base) as usize) < room)
        {
            return Plain::General;
        }
        let environment = coroutine.environment;
        coroutine.traceback = None;
        let CoroutineState::Suspended { mut frames, .. } =
            std::mem::replace(&mut coroutine.state, CoroutineState::Running)
        else {
            unreachable!("the state was matched as suspended")
        };
        let switch = self.enter_thread(resumer_frames, thread, environment);
        let outcome = match Self::write_plain_resume(&mut frames, target, arguments) {
            Ok(()) => self.run_resumed(&mut frames),
            Err(error) => Err(error),
        };
        self.park_or_drive(thread, switch, frames, outcome, preserve_memory_error)
    }

    /// The end of a run of the thread entered with [`Self::enter_thread`]:
    /// a plain yield whose values wait in the quick buffer parks the stack
    /// and leaves the thread, and every other end goes through
    /// [`Self::drive_thread`].
    fn park_or_drive(
        &mut self,
        thread: ThreadHandle,
        switch: Switch,
        frames: Box<CallStack>,
        outcome: VmResult<Option<Vec<Value>>>,
        preserve_memory_error: bool,
    ) -> Plain {
        if matches!(outcome, Ok(None)) && self.pending_host_resume.is_none() && self.quick.pending {
            // `Ok(None)` owns nothing: forgetting it spares the usual switch
            // the call that drops a result.
            std::mem::forget(outcome);
            self.quick.pending = false;
            self.park_plain(thread, switch, frames, self.quick.target);
            return Plain::Yielded;
        }
        Plain::Done(self.drive_thread(thread, switch, frames, Some(outcome), preserve_memory_error))
    }

    /// Parks the stack of `thread`, which stopped in a plain
    /// `coroutine.yield` whose call `target` takes the next resume's
    /// values, and hands control back to the resumer. The stack's roots are
    /// rebuilt when a collection starts, not here.
    #[inline(always)]
    #[allow(unsafe_code)]
    fn park_plain(
        &mut self,
        thread: ThreadHandle,
        switch: Switch,
        frames: Box<CallStack>,
        target: CallReturn,
    ) {
        let coroutine = self.coroutines.get_mut(&thread).expect("coroutine exists");
        coroutine.environment = self.thread_environment;
        let suspended = CoroutineState::Suspended {
            frames,
            yield_return: target,
            host_resume: None,
        };
        if matches!(coroutine.state, CoroutineState::Running) {
            // SAFETY: `Running` owns nothing, so overwriting it
            // without a drop leaks nothing. Assigning would read the
            // old state back with wide loads over the narrow store
            // that set `Running` a moment ago, and wait for it.
            unsafe { std::ptr::write(&raw mut coroutine.state, suspended) };
        } else {
            coroutine.state = suspended;
        }
        coroutine.roots_stale = true;
        if !std::mem::replace(&mut coroutine.listed, true) {
            self.stale_coroutines.push(thread);
        }
        self.leave_thread(switch);
    }

    /// Whether a resumed thread may run without the dispatch loop's
    /// prologue: no hook, recording, profile, landing, flush or interrupt
    /// waits for it. This is what [`Self::run_resumed`] and
    /// [`Self::run_micro`] check before the short run.
    #[inline(always)]
    fn short_run_allowed(&self) -> bool {
        self.debug_hooks.is_empty()
            && !self.jit.trace.recording()
            && !self.interpreter_profiling
            && !self.jit.trace.has_loaded_profile()
            && !self.jit.tail_landing
            && !self.jit.call_landing
            && !self.runtime.jit_flush_requested()
            && !self.interrupt_pending()
            && !self
                .interrupt_flag
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The resume of a thread of one frame that waits in a plain
    /// `coroutine.yield` and runs a few instructions to its next one, with
    /// the values passed in place. The exchange fills the registers the
    /// yield's call gives its results, the short run ([`Self::short_run`])
    /// executes the thread's instructions, and the exchange reads what the
    /// thread yields straight from its registers, before the stack parks
    /// again. No value goes through a buffer of the VM, and nothing is
    /// checked twice.
    ///
    /// `Declined` is a resume that is not of this shape, with nothing
    /// changed and the exchange unused. A short run that stops before a
    /// yield goes on as [`Self::resume_plain`] would have (`Ran`): the
    /// exchange has given and has not taken.
    #[inline(always)]
    fn resume_direct<R, X: Exchange<R>>(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        preserve_memory_error: bool,
        mut exchange: X,
    ) -> Direct<R> {
        if self.resumer_stacks.len() >= MAX_NESTED_RESUMES || !self.short_run_allowed() {
            return Direct::Declined;
        }
        let Some(coroutine) = self.coroutines.get_mut(&thread) else {
            return Direct::Declined;
        };
        let CoroutineState::Suspended {
            frames,
            yield_return,
            host_resume: None,
        } = &mut coroutine.state
        else {
            return Direct::Declined;
        };
        let target = *yield_return;
        let (taken, taking) = (usize::from(target.base), usize::from(target.results));
        if frames.len() != 1 {
            coroutine.long_runs = true;
            return Direct::Declined;
        }
        // SAFETY: the stack has its one frame live. The frame and its
        // window are used until the short run ends and not after, and
        // nothing else touches the thread's stack meanwhile (see below).
        #[allow(unsafe_code)]
        let (frame, registers) = unsafe {
            let (frame, window) = frames.running_view();
            (&mut *frame, &mut *window)
        };
        let width = (frame.top.saturating_sub(frame.base) as usize).min(registers.len());
        // The frame takes the values in its registers and goes on at an
        // instruction of its chunk, after the one it last ran.
        if taking == 0
            || width < taken + taking
            || frame.pc >= frame.chunk.code.len()
            || frame
                .last_interpreted_pc
                .get()
                .is_some_and(|last| frame.pc <= last)
        {
            return Direct::Declined;
        }
        frame.clear_open_results();
        exchange.give(&mut registers[taken..taken + taking]);
        if coroutine.traceback.is_some() {
            coroutine.traceback = None;
        }
        // A thread known to leave the short run before its yield starts in
        // the fast loop: the short run's entry would be paid for nothing.
        // One resume in `LONG_RUN_RETRY` tries the short run all the same:
        // what made the thread leave it may have been its first steps.
        if coroutine.long_runs {
            coroutine.long_resumes = coroutine.long_resumes.wrapping_add(1);
            if coroutine.long_resumes % LONG_RUN_RETRY != 0 {
                return Direct::Ran(self.finish_long_run(
                    resumer_frames,
                    thread,
                    None,
                    preserve_memory_error,
                    X::FROM_TRACE,
                ));
            }
        }
        // The short run happens with the thread still parked. Its
        // instructions read and write the thread's registers, its upvalues
        // and arrays, and nothing else: none of them asks which thread
        // runs, what its status or environment is, or who resumed it, and
        // none reaches a collection, a hook or an error. So nothing can
        // tell that the VM has not switched, and a thread that yields
        // again needs no switch at all, only the new target of its yield.
        let mut held = std::ptr::NonNull::from(coroutine);
        // A trace's resume keeps native execution off for the thread, so
        // its backward jumps count for nothing.
        let tracing = !X::FROM_TRACE
            && self.execution_mode == ExecutionMode::Tracing
            && self.runtime.jit_enabled();
        let left = match self.micro_steps(frame, registers, tracing, true) {
            Stop::Yielded(site) => {
                let (first, count) = (usize::from(site.first), usize::from(site.count));
                // SAFETY: `held` is the thread's record in the VM's table
                // of coroutines. The short run creates, resumes and frees
                // no coroutine, so the table did not move, and it reached
                // neither the record nor the thread's boxed stack through
                // the VM.
                #[allow(unsafe_code)]
                let coroutine = unsafe { held.as_mut() };
                // The short run checked the range against the frame.
                let yielded = &registers[first..first + count];
                let answer = exchange.take(yielded);
                if let CoroutineState::Suspended { yield_return, .. } = &mut coroutine.state {
                    *yield_return = site.target;
                }
                coroutine.roots_stale = true;
                if !std::mem::replace(&mut coroutine.listed, true) {
                    self.stale_coroutines.push(thread);
                }
                // A short run that reached its yield: the thread runs short
                // again, whatever an earlier resume found.
                if coroutine.long_runs {
                    coroutine.long_runs = false;
                }
                self.jit.statistics.coroutine_short_runs += 1;
                return Direct::Yielded(answer);
            }
            Stop::Before => None,
            Stop::Backedge(exit) => Some(exit),
        };
        Direct::Ran(self.finish_long_run(
            resumer_frames,
            thread,
            left,
            preserve_memory_error,
            X::FROM_TRACE,
        ))
    }

    /// The rest of a resume whose short run stopped before a yield, with
    /// `left` as the exit a backedge asked for. Only now does the thread
    /// become the running one, and it goes on in the fast loop or the
    /// dispatch loop, as [`Self::run_resumed`] sends it. For a trace's
    /// resume (`from_trace`) the run happens as [`Self::resume_for_trace`]
    /// says one must.
    #[cold]
    #[allow(clippy::needless_pass_by_value)]
    #[inline(never)]
    fn finish_long_run(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        left: Option<FastExit>,
        preserve_memory_error: bool,
        from_trace: bool,
    ) -> Resumed {
        let coroutine = self.coroutines.get_mut(&thread).expect("coroutine exists");
        // A backward jump the trace engine wanted to see says nothing about
        // the thread: only an instruction the short run does not know does.
        if left.is_none() {
            coroutine.long_runs = true;
        }
        let environment = coroutine.environment;
        let CoroutineState::Suspended { mut frames, .. } =
            std::mem::replace(&mut coroutine.state, CoroutineState::Running)
        else {
            unreachable!("the short run left the thread parked")
        };
        let trace_run = from_trace.then(|| {
            let held = self.runtime.suspend_jit();
            self.active_native_runs += 1;
            (held, self.trace_resumer.replace(self.resumer_stacks.len()))
        });
        let switch = self.enter_thread(resumer_frames, thread, environment);
        let outcome = match left {
            Some(FastExit::Safepoint) => self.safepoint_then_interpret(&mut frames),
            Some(_) => self.interpret(&mut frames, false),
            None => self.run_resumed_in_fast_loop(&mut frames),
        };
        self.quick.pure = false;
        let resumed =
            match self.park_or_drive(thread, switch, frames, outcome, preserve_memory_error) {
                Plain::Yielded => Resumed::Yielded,
                Plain::Done(answer) => Resumed::done(answer),
                Plain::Unknown | Plain::General => {
                    unreachable!("a thread that ran yielded or ended")
                }
            };
        if let Some((held, outer)) = trace_run {
            self.trace_resumer = outer;
            self.active_native_runs -= 1;
            self.runtime.restore_jit(held);
        }
        resumed
    }

    /// The coroutine `thread` names, taken out of its state to run, or
    /// `None` (nothing changed, `arguments` untouched) for a handle that is
    /// not a coroutine, which the main thread's handle is. A resume that
    /// finds the state refuses it answers at once, with the state left
    /// alone, in the `Err` case.
    #[allow(clippy::type_complexity)]
    fn take_coroutine_state(
        &mut self,
        thread: ThreadHandle,
    ) -> Option<Result<(CoroutineState, TableHandle), Vec<Value>>> {
        let nested_full = self.resumer_stacks.len() >= MAX_NESTED_RESUMES;
        let coroutine = self.coroutines.get_mut(&thread)?;
        if nested_full
            && matches!(
                coroutine.state,
                CoroutineState::New(_) | CoroutineState::Suspended { .. }
            )
        {
            return Some(Err(vec![
                Value::Boolean(false),
                Value::from("C stack overflow"),
            ]));
        }
        let state = std::mem::replace(&mut coroutine.state, CoroutineState::Running);
        if matches!(
            state,
            CoroutineState::New(_) | CoroutineState::Suspended { .. }
        ) {
            coroutine.traceback = None;
        }
        Some(Ok((state, coroutine.environment)))
    }

    /// [`Self::resume_coroutine_mode`] without the replay, for a handle that
    /// may not be a coroutine: `None` for one that is not, with nothing
    /// changed. The values of a plain yield stay in the quick buffer.
    fn resume_quick(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: &mut [Value],
        preserve_memory_error: bool,
    ) -> Option<Resumed> {
        match self.resume_plain(resumer_frames, thread, arguments, preserve_memory_error) {
            Plain::Done(answer) => return Some(Resumed::done(answer)),
            Plain::Yielded => return Some(Resumed::Yielded),
            Plain::Unknown => return None,
            Plain::General => {}
        }
        let arguments = Self::owned_arguments(arguments);
        let (state, coroutine_environment) = match self.take_coroutine_state(thread)? {
            Ok(taken) => taken,
            Err(answer) => return Some(Resumed::done(Ok(answer))),
        };
        Some(Resumed::done(self.resume_state(
            resumer_frames,
            thread,
            arguments,
            preserve_memory_error,
            state,
            coroutine_environment,
        )))
    }

    #[allow(clippy::too_many_lines)]
    fn resume_state(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: Vec<Value>,
        preserve_memory_error: bool,
        state: CoroutineState,
        coroutine_environment: TableHandle,
    ) -> VmResult<Vec<Value>> {
        let mut suspended_call = None;
        let mut frames = match state {
            CoroutineState::New(entry) => {
                let entry = entry.unwrap_or(Value::Nil);
                let prepared = (|| -> VmResult<_> {
                    let frame = if let Value::Closure(entry) = entry {
                        let body = self.closure_body(entry)?;
                        self.make_frame(
                            body.chunk(),
                            arguments,
                            body.shared_cells(),
                            None,
                            None,
                            body.environment(),
                            Some(entry),
                        )?
                    } else {
                        // A native entry has a bytecode return continuation. After a yield,
                        // resume arguments become its return values with no native frames kept.
                        let chunk = crate::bytecode::Chunk::with_varargs(
                            1,
                            1,
                            Vec::new(),
                            vec![
                                Instruction::call_vararg(0, 0, 0),
                                Instruction::return_open(0, 0),
                            ],
                            Vec::new(),
                            Vec::new(),
                            None,
                        )
                        .with_debug_metadata(
                            crate::bytecode::DebugMetadata::new(
                                crate::vm::host::C_SOURCE,
                                Vec::new(),
                            ),
                        );
                        let mut arguments = arguments;
                        arguments.insert(0, entry);
                        self.make_frame(
                            chunk,
                            arguments,
                            Vec::<crate::runtime::CellHandle>::new(),
                            None,
                            None,
                            coroutine_environment,
                            None,
                        )?
                    };
                    Ok(Box::new(CallStack::with_frame(frame)))
                })();
                match prepared {
                    Ok(frames) => frames,
                    Err(error) => {
                        let coroutine = self.coroutines.get_mut(&thread).expect("coroutine exists");
                        coroutine.state = CoroutineState::Dead;
                        coroutine.traceback = Some(capture_traceback(&CallStack::new()));
                        if matches!(
                            self.compatibility,
                            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
                        ) {
                            coroutine.failed = Some((CallStack::new(), error.clone()));
                        }
                        return if preserve_memory_error && error.is_memory_error() {
                            Err(error)
                        } else {
                            vm_error_object(&error).map(|error| vec![Value::Boolean(false), error])
                        };
                    }
                }
            }
            CoroutineState::Suspended {
                frames,
                yield_return,
                host_resume,
            } => {
                suspended_call = Some((yield_return, host_resume, arguments));
                frames
            }
            CoroutineState::Running => {
                return Ok(vec![
                    Value::Boolean(false),
                    Value::from("cannot resume non-suspended coroutine"),
                ]);
            }
            CoroutineState::Dead => {
                self.coroutines
                    .get_mut(&thread)
                    .expect("coroutine exists")
                    .state = CoroutineState::Dead;
                return Ok(vec![
                    Value::Boolean(false),
                    Value::from("cannot resume dead coroutine"),
                ]);
            }
        };

        let switch = self.enter_thread(resumer_frames, thread, coroutine_environment);

        let initial_result = if let Some((target, resume, arguments)) = suspended_call {
            if resume.is_none() {
                // A plain `coroutine.yield` parks no host continuation: the
                // resume values are the call's results.
                self.write_host_resume_results(&mut frames, target, arguments)
                    .err()
                    .map(Err)
            } else {
                match self.resume_host_call(&mut frames, resume, arguments) {
                    Ok(values) => self
                        .write_host_resume_results(&mut frames, target, values)
                        .err()
                        .map(Err),
                    Err(VmError::Yield(values)) => {
                        self.pending_yield = Some((target, values, false));
                        Some(Ok(None))
                    }
                    Err(error) => Some(Err(error)),
                }
            }
        } else {
            None
        };

        self.drive_thread(
            thread,
            switch,
            frames,
            initial_result,
            preserve_memory_error,
        )
    }

    /// Makes `thread` the running one: the resumer's environment and stack
    /// park, and the thread's environment becomes the current one.
    #[inline(always)]
    fn enter_thread(
        &mut self,
        resumer_frames: &mut CallStack,
        thread: ThreadHandle,
        coroutine_environment: TableHandle,
    ) -> Switch {
        let previous_coroutine = self.current_coroutine;
        let previous_environment = self.thread_environment;
        if let Some(previous) = previous_coroutine {
            self.coroutines
                .get_mut(&previous)
                .expect("resumer coroutine exists")
                .environment = previous_environment;
        } else {
            self.parked_main_environment = Some(previous_environment);
        }
        self.current_coroutine = Some(thread);
        self.thread_environment = coroutine_environment;
        self.resumer_stacks.push(ParkedStack::new(resumer_frames));
        Switch {
            previous_coroutine,
            previous_environment,
        }
    }

    /// Hands control back to the resumer, once the thread has switched or
    /// ended: its environment, its stack and the thread it was running.
    #[inline(always)]
    fn leave_thread(&mut self, switch: Switch) {
        self.thread_environment = match switch.previous_coroutine {
            None => self
                .parked_main_environment
                .take()
                .unwrap_or(switch.previous_environment),
            Some(previous) => {
                self.coroutines
                    .get(&previous)
                    .expect("resumer coroutine exists")
                    .environment
            }
        };
        self.current_coroutine = switch.previous_coroutine;
        self.resumer_stacks.pop();
    }

    /// Marks a thread whose run ended without parking as dead, if nothing
    /// has: an internal error skips the assignments of the normal ends and
    /// would leave it running and rooted for good.
    pub(crate) fn kill_if_running(&mut self, thread: ThreadHandle) {
        let coroutine = self.coroutines.get_mut(&thread).expect("coroutine exists");
        if matches!(coroutine.state, CoroutineState::Running) {
            coroutine.state = CoroutineState::Dead;
            // The thread is dead whether or not its roots can be dropped.
            let _ = self.set_coroutine_runtime_roots(thread, &mut CallStack::new());
        }
    }

    /// Runs the thread entered with [`Self::enter_thread`] until it switches
    /// or ends, then leaves it. `first` is the result of the step already
    /// taken, if any.
    #[allow(clippy::too_many_lines)]
    fn drive_thread(
        &mut self,
        thread: ThreadHandle,
        switch: Switch,
        mut frames: Box<CallStack>,
        mut initial_result: Option<VmResult<Option<Vec<Value>>>>,
        preserve_memory_error: bool,
    ) -> VmResult<Vec<Value>> {
        let mut parked = None;
        // An internal error ends the run like any other result, so that the
        // resumer is restored below.
        macro_rules! step {
            ($step:expr) => {
                match $step {
                    Ok(value) => value,
                    Err(error) => break Err(error.into()),
                }
            };
        }
        let result = loop {
            match initial_result
                .take()
                .unwrap_or_else(|| self.run_resumed(&mut frames))
            {
                Ok(Some(mut values)) => {
                    values.insert(0, Value::Boolean(true));
                    self.coroutines
                        .get_mut(&thread)
                        .expect("running coroutine exists")
                        .state = CoroutineState::Dead;
                    step!(self.set_coroutine_runtime_roots(thread, &mut CallStack::new()));
                    break Ok(values);
                }
                Ok(None) => {
                    if let Some((yield_return, mut values, has_status)) = self.take_pending_yield()
                    {
                        if !has_status {
                            values.insert(0, Value::Boolean(true));
                        }
                        // The stack parks below, with the coroutine's
                        // environment.
                        parked = Some((frames, yield_return, self.pending_host_resume.take()));
                        break Ok(values);
                    }
                }
                Err(error) => {
                    let traceback =
                        capture_error_traceback(&frames, self.native_metamethod_handler(&frames));
                    let error = locate_runtime_error(&frames, self.compatibility, error);
                    let action = if let Some(action) =
                        step!(self.resume_unwind_handler_error(&mut frames, error.clone()))
                    {
                        action
                    } else {
                        step!(self.begin_coroutine_error_unwind(&mut frames, error))
                    };
                    if matches!(action, UnwindAction::Continue) {
                        continue;
                    }
                    let UnwindAction::Return(error) = action else {
                        unreachable!("continued unwind has returned")
                    };
                    self.coroutines
                        .get_mut(&thread)
                        .expect("running coroutine exists")
                        .state = CoroutineState::Dead;
                    self.coroutines
                        .get_mut(&thread)
                        .expect("running coroutine exists")
                        .traceback = Some(traceback);
                    if error.exit_request().is_none()
                        && matches!(
                            self.compatibility,
                            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
                        )
                    {
                        let mut roots = GcRootBuilder::new();
                        // A failed stack stopped at its error pc, which says
                        // nothing about what close handlers and debuggers
                        // still read: root it whole and clear nothing.
                        roots.extend(&Self::gc_roots_full_windows(
                            self.thread_environment,
                            &frames,
                            &[],
                        ));
                        if let Ok(value) = vm_error_object(&error) {
                            roots.value(&value);
                        }
                        step!(self.runtime.set_thread_roots(thread, roots.finish()));
                        self.coroutines
                            .get_mut(&thread)
                            .expect("failed coroutine exists")
                            .failed = Some((*frames, error.clone()));
                    } else {
                        step!(self.set_coroutine_runtime_roots(thread, &mut CallStack::new()));
                    }
                    // An `os.exit` request keeps unwinding into the resumer.
                    break if preserve_memory_error && error.is_memory_error() {
                        Err(error)
                    } else {
                        vm_error_object(&error).map(|error| vec![Value::Boolean(false), error])
                    };
                }
            }
        };

        if parked.is_none() {
            self.kill_if_running(thread);
        }
        let coroutine = self.coroutines.get_mut(&thread).expect("coroutine exists");
        coroutine.environment = self.thread_environment;
        if let Some((frames, yield_return, host_resume)) = parked {
            // The stack's roots are rebuilt when a collection starts
            // (`refresh_stale_coroutine_roots`), not on every yield.
            coroutine.state = CoroutineState::Suspended {
                frames,
                yield_return,
                host_resume,
            };
            coroutine.roots_stale = true;
            if !std::mem::replace(&mut coroutine.listed, true) {
                self.stale_coroutines.push(thread);
            }
        }
        self.leave_thread(switch);
        result
    }

    /// Whether a yield waits for the resume to take its values.
    #[inline]
    pub(crate) fn yield_pending(&self) -> bool {
        self.quick.pending || self.pending_yield.is_some()
    }

    /// The values a yield parked, with the call that takes the resume's
    /// values, and whether they start with the `true` of the resume.
    pub(crate) fn take_pending_yield(&mut self) -> Option<(CallReturn, Vec<Value>, bool)> {
        if std::mem::take(&mut self.quick.pending) {
            let target = self.quick.target;
            return Some((target, self.quick.take_with_status(), true));
        }
        self.pending_yield.take()
    }

    /// Whether a yield from the top frame of `frames` crosses a boundary the
    /// profile cannot resume.
    ///
    /// Lua 5.1 runs a generic-for iterator through a C call. A builtin
    /// yield used as the iterator pushes no frame, so this checks the
    /// yielding instruction's own call site.
    fn yield_refused(&self, frames: &CallStack) -> bool {
        let iterator_yield = self.compatibility == CompatibilityProfile::Lua51
            && frames.last().is_some_and(|frame| {
                frame
                    .pc
                    .checked_sub(1)
                    .and_then(|pc| frame.chunk.debug.call_site_at(pc))
                    .is_some_and(crate::bytecode::CallSite::is_iterator)
            });
        iterator_yield || self.yield_blocked(frames)
    }

    /// The values a switch carries go in the VM's spare buffer, emptied.
    pub(crate) fn take_value_buffer(&mut self) -> Vec<Value> {
        let mut buffer = std::mem::take(&mut self.spare_values);
        buffer.clear();
        buffer
    }

    /// Gives the thread parked in a plain `coroutine.yield` the values of the
    /// resume as the results of the call, which is what
    /// [`Self::write_host_resume_results`] does for a yield with a host
    /// continuation. The values move into the registers, and leave nothing
    /// behind in `values`.
    #[inline]
    fn write_plain_resume(
        frames: &mut CallStack,
        target: CallReturn,
        values: &mut [Value],
    ) -> VmResult<()> {
        let index = frames.len() - 1;
        frames[index].clear_open_results();
        let registers = frames.at_mut(index).registers;
        let base = usize::from(target.base);
        let count = usize::from(target.results);
        let Some(slots) = registers.get_mut(base..base + count) else {
            values.fill_with(|| Value::Nil);
            return Err(VmError::RegisterRange {
                start: target.base,
                count: target.results,
            });
        };
        let mut given = values.iter_mut();
        for slot in slots {
            match given.next() {
                Some(value) => crate::vm::registers::move_value(slot, value),
                None => *slot = Value::Nil,
            }
        }
        // Values the call does not take are dropped, so that nothing is left
        // in `values`.
        for rest in given {
            *rest = Value::Nil;
        }
        Ok(())
    }

    /// Hands a buffer a switch has finished with back for the next one.
    pub(crate) fn recycle_values(&mut self, mut values: Vec<Value>) {
        values.clear();
        if values.capacity() >= self.spare_values.capacity() {
            self.spare_values = values;
        }
    }

    /// `coroutine.yield` from the fast loop: parks the yield exactly as
    /// `yield_coroutine` does, with the values copied from the call's
    /// argument registers into the spare buffer, behind the `true` of the
    /// resume that will return them. Returns false, changing nothing, when
    /// the general path has to take the call (a yield that is refused raises
    /// its error there).
    ///
    /// The fast loop makes this call only for a `Call` or `CallOne`, never a
    /// generic `for`'s `ForInCall`, so the iterator check of
    /// [`Self::yield_refused`] has nothing to find here.
    #[inline]
    pub(crate) fn fast_yield(
        &mut self,
        frames: &CallStack,
        call: Instruction,
        first_argument: usize,
        count: usize,
    ) -> bool {
        if self.current_coroutine.is_none()
            || !self.debug_hooks.is_empty()
            || self.jit.trace.recording()
        {
            return false;
        }
        let Some(arguments) = frames
            .last_registers()
            .and_then(|registers| registers.get(first_argument..first_argument + count))
        else {
            return false;
        };
        // A thread of one frame under no host call has nothing to cross.
        if !(frames.len() <= 1 && self.host_yield_boundaries.is_empty())
            && self.yield_blocked(frames)
        {
            return false;
        }
        let target = CallReturn {
            base: call.a(),
            results: call.c(),
        };
        if count <= QUICK_VALUES {
            let quick = &mut self.quick;
            for (slot, value) in quick.values.iter_mut().zip(arguments) {
                *slot = crate::vm::registers::clone_register_value(value);
            }
            quick.len = u8::try_from(count).unwrap_or(0);
            quick.target = target;
            quick.pending = true;
            // Purity is the short run's to claim, after this yield.
            quick.pure = false;
        } else {
            let mut values = self.take_value_buffer();
            values.push(Value::Boolean(true));
            values.extend(
                arguments
                    .iter()
                    .map(crate::vm::registers::clone_register_value),
            );
            self.pending_yield = Some((target, values, true));
        }
        self.jit.statistics.coroutine_direct_yields += 1;
        true
    }

    /// Runs a resumed thread until it switches or ends: the dispatch loop's
    /// first pass, without its prologue, for the usual case of a thread that
    /// goes straight back to the fast loop. Anything else is the dispatch
    /// loop's to run, from the state the fast loop left.
    fn run_resumed(&mut self, frames: &mut CallStack) -> VmResult<Option<Vec<Value>>> {
        // What `interpret_loop` checks before its first pass, for the thread
        // that stands after the call it yielded from.
        let top = &frames[frames.len() - 1];
        let result = if !self.debug_hooks.is_empty()
            || self.interpreter_profiling
            || self.jit.trace.has_loaded_profile()
            || self.jit.tail_landing
            || self.jit.call_landing
            || self.runtime.jit_flush_requested()
            || top
                .last_interpreted_pc
                .get()
                .is_some_and(|last| top.pc <= last)
            || self
                .interrupt_flag
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.interpret(frames, false)
        } else {
            match self.run_micro(frames) {
                Micro::Yielded => {
                    // Nothing but the instructions of the short run ran, and
                    // they write no table, cell or global. This is set last:
                    // a thread that resumed another one left its flag.
                    self.quick.pure = true;
                    self.jit.statistics.coroutine_short_runs += 1;
                    return Ok(None);
                }
                Micro::Left(Some(FastExit::Safepoint)) => self.safepoint_then_interpret(frames),
                Micro::Left(Some(_)) => self.interpret(frames, false),
                Micro::Left(None) => self.run_resumed_in_fast_loop(frames),
            }
        };
        self.quick.pure = false;
        result
    }

    /// The collector's turn after a fast exit, then the dispatch loop.
    fn safepoint_then_interpret(&mut self, frames: &mut CallStack) -> VmResult<Option<Vec<Value>>> {
        self.gc_safepoint(frames)?;
        self.gc_observed_allocation_serial = self.runtime.allocation_serial();
        self.interpret(frames, false)
    }

    /// [`Self::run_resumed`] on the fast loop, which leaves the dispatch loop
    /// what it cannot run.
    fn run_resumed_in_fast_loop(&mut self, frames: &mut CallStack) -> VmResult<Option<Vec<Value>>> {
        match self.interpret_fast(frames) {
            Ok(FastExit::Yielded) => Ok(None),
            Ok(FastExit::Safepoint) => self.safepoint_then_interpret(frames),
            Ok(_) => self.interpret(frames, false),
            Err(error) => {
                self.abort_recording_on_error();
                Err(error)
            }
        }
    }

    /// `coroutine.resume` from the fast loop: runs the thread to its next
    /// switch with the arguments copied from the call's registers into the
    /// spare buffer and the results written straight into the call's result
    /// registers. Returns false, changing nothing, for a call the general
    /// path has to take (a first argument that is not a thread, which it
    /// reports).
    pub(crate) fn fast_resume(
        &mut self,
        frames: &mut CallStack,
        call: Instruction,
        first_argument: usize,
        count: usize,
    ) -> VmResult<bool> {
        let frame_index = frames.len() - 1;
        let Some(&Value::Thread(thread)) = frames.at(frame_index).registers.get(first_argument)
        else {
            return Ok(false);
        };
        self.fast_resume_thread(frames, call, thread, (first_argument + 1, count - 1), false)
    }

    /// A call of a function `coroutine.wrap` made, from the fast loop: the
    /// resume of [`Self::fast_resume`] with every argument the thread's and
    /// results that carry no leading `true`. Returns false for a call the
    /// general path has to take or to finish (see
    /// [`Self::fast_resume_thread`]).
    pub(crate) fn fast_wrapped_resume(
        &mut self,
        frames: &mut CallStack,
        call: Instruction,
        thread: ThreadHandle,
        first_argument: usize,
        count: usize,
    ) -> VmResult<bool> {
        self.fast_resume_thread(frames, call, thread, (first_argument, count), true)
    }

    /// The resume of `thread` both calls above make, with the `count`
    /// registers from `first` as the values the thread takes.
    ///
    /// `wrapped` is a call of the thread's wrapper. What the thread yields
    /// or returns goes to the call's result registers without the `true` of
    /// a resume. A thread that failed, or that cannot be resumed, is the
    /// general path's to report: it raises the error, with the position of
    /// the caller for a string, and Lua 5.4 closes the thread first. The
    /// thread has run by then and cannot run again, so its answer waits in
    /// the VM ([`ResumeMemo`]), this returns false, and the general path's
    /// call of the wrapper takes the answer in place of a resume, as it does
    /// for a native trace's.
    // Inlined into both callers, each of which knows `wrapped`: the resume
    // of a thread then carries no test for it.
    #[inline(always)]
    fn fast_resume_thread(
        &mut self,
        frames: &mut CallStack,
        call: Instruction,
        thread: ThreadHandle,
        (first, count): (usize, usize),
        wrapped: bool,
    ) -> VmResult<bool> {
        if !self.debug_hooks.is_empty() || self.jit.trace.recording() {
            return Ok(false);
        }
        let frame_index = frames.len() - 1;
        // A thread that runs a few instructions to its next yield takes the
        // arguments from the call's registers and leaves what it yields in
        // the call's result registers, with nothing in between.
        let (base, wanted) = (usize::from(call.a()), usize::from(call.c()));
        let direct = if self.resume_memo.is_none()
            && let Some((frame, registers)) = frames.top_mut()
            && first + count <= registers.len()
            && base + wanted <= registers.len()
        {
            let exchange = WindowExchange {
                frame: std::ptr::NonNull::from(frame),
                registers: std::ptr::NonNull::from(registers),
                first,
                count,
                base,
                wanted,
                wrapped,
            };
            self.resume_direct(frames, thread, wrapped, exchange)
        } else {
            Direct::Declined
        };
        let results = match direct {
            Direct::Yielded(()) => {
                self.jit.statistics.coroutine_direct_resumes += 1;
                return Ok(true);
            }
            Direct::Ran(resumed) => Some(resumed),
            Direct::Declined => self.fast_resume_general(frames, thread, (first, count), wrapped),
        };
        let Some(results) = results else {
            return Ok(false);
        };
        self.jit.statistics.coroutine_direct_resumes += 1;
        let results = match results {
            Resumed::Yielded => {
                let frame_index = frames.len() - 1;
                let mut frame = frames.at_mut(frame_index);
                frame.clear_open_results();
                let registers = frame.registers;
                let (base, wanted) = (usize::from(call.a()), usize::from(call.c()));
                let Some(slots) = registers.get_mut(base..base + wanted) else {
                    self.quick.clear();
                    return Err(VmError::RegisterRange {
                        start: call.a(),
                        count: call.c(),
                    });
                };
                let mut given = self.quick.values.iter_mut();
                for (index, slot) in slots.iter_mut().enumerate() {
                    *slot = if index == 0 && !wrapped {
                        Value::Boolean(true)
                    } else {
                        given
                            .next()
                            .map_or(Value::Nil, |value| std::mem::replace(value, Value::Nil))
                    };
                }
                self.quick.clear();
                return Ok(true);
            }
            Resumed::Done(answer) => match *answer {
                Ok(results)
                    if !wrapped || matches!(results.first(), Some(Value::Boolean(true))) =>
                {
                    results
                }
                Err(error) if !wrapped => {
                    return Err(self.builtin_call_error(
                        frames,
                        frame_index,
                        &Value::BuiltinFunction(crate::runtime::BuiltinFunction::CoroutineResume),
                        error,
                    ));
                }
                answer => {
                    self.remember_resume(thread, answer);
                    return Ok(false);
                }
            },
        };
        let frame_index = frames.len() - 1;
        let mut frame = frames.at_mut(frame_index);
        frame.clear_open_results();
        // A wrapper's results follow the `true` of the resume.
        let values = &results[usize::from(wrapped)..];
        crate::vm::registers::write_fixed_results(frame.registers, call, values)?;
        self.recycle_values(results);
        Ok(true)
    }

    /// The resume of [`Self::fast_resume_thread`] for a thread
    /// [`Self::resume_direct`] declined: the arguments are cloned out of
    /// the call's registers into a small array, or a vector when there are
    /// more, and what the thread yields waits in the quick buffer. A
    /// wrapper's resume keeps a memory error an error, as the general call
    /// of a wrapper does.
    #[inline(never)]
    fn fast_resume_general(
        &mut self,
        frames: &mut CallStack,
        thread: ThreadHandle,
        (first, count): (usize, usize),
        wrapped: bool,
    ) -> Option<Resumed> {
        let frame_index = frames.len() - 1;
        let given = &frames.at(frame_index).registers[first..first + count];
        if given.len() <= SMALL_ARGUMENTS {
            let mut small = [const { Value::Nil }; SMALL_ARGUMENTS];
            for (slot, value) in small.iter_mut().zip(given) {
                *slot = crate::vm::registers::clone_register_value(value);
            }
            let used = given.len();
            self.resume_mode_quick(frames, thread, &mut small[..used], wrapped)
        } else {
            let mut arguments = self.take_value_buffer();
            arguments.extend(given.iter().map(crate::vm::registers::clone_register_value));
            let results = self.resume_mode_quick(frames, thread, &mut arguments, wrapped);
            self.recycle_values(arguments);
            results
        }
    }

    /// `coroutine.resume` (or, with `wrapped`, a call of the function
    /// `coroutine.wrap` made) for a native trace: runs the thread to its next
    /// switch exactly as the fast loop's resume does, with `arguments` and
    /// the trace's frames as the resumer. The trace is in the middle of its
    /// run, so native execution stays off for the thread (`suspend_jit`) and
    /// a flush waits (`active_native_runs`), and the collector sees the run
    /// in flight (whole windows, the run's bank).
    ///
    /// Returns `None`, having changed nothing, for a call the interpreter has
    /// to make: a hook is installed, or the thread is unknown. The answer
    /// is the call's own result, which the caller either takes or keeps with
    /// [`Self::remember_resume`] before it leaves. A thread that ran only a
    /// short run to its yield ([`Self::resume_direct`]) hands the values it
    /// yielded to `take`, from its registers, and the answer is what `take`
    /// made of them.
    pub(crate) fn resume_for_trace<R>(
        &mut self,
        frames: &mut CallStack,
        thread: ThreadHandle,
        arguments: &mut [Value],
        wrapped: bool,
        take: impl FnOnce(&[Value]) -> R,
    ) -> Option<Direct<R>> {
        if !self.debug_hooks.is_empty() {
            return None;
        }
        // A memo that nothing replayed is stale.
        if self.resume_memo.is_some() {
            self.resume_memo = None;
        }
        let direct = self.resume_direct(
            frames,
            thread,
            wrapped,
            TraceExchange {
                arguments: &mut *arguments,
                take,
            },
        );
        let result = match direct {
            Direct::Declined => {
                let held = self.runtime.suspend_jit();
                self.active_native_runs += 1;
                let outer = self.trace_resumer.replace(self.resumer_stacks.len());
                let resumed = self.resume_quick(frames, thread, arguments, wrapped);
                self.trace_resumer = outer;
                self.active_native_runs -= 1;
                self.runtime.restore_jit(held);
                resumed.map(Direct::Ran)
            }
            direct => Some(direct),
        };
        if result.is_some() {
            self.jit.statistics.coroutine_trace_resumes += 1;
        }
        result
    }

    /// Whether `thread` is known to run long between its yields (see
    /// `Coroutine::long_runs`), which is when the fast loop resumes it in
    /// place.
    #[inline(always)]
    pub(crate) fn runs_long(&self, thread: ThreadHandle) -> bool {
        self.coroutines
            .get(&thread)
            .is_some_and(|coroutine| coroutine.long_runs)
    }

    /// `coroutine.resume` of `thread` (or, with `wrapped`, a call of its
    /// wrapper) from the fast loop, for a thread that waits in a plain
    /// `coroutine.yield`: the `count` registers from `first` of the running
    /// frame become the results of the yield, the thread's stack takes the
    /// place of `stack`, and the loop goes on in the thread's top frame. The
    /// call `call` gets its results when the thread yields
    /// ([`Self::yield_in_place`]) or when the loop ends
    /// ([`Self::leave_in_place`]).
    ///
    /// Returns false, having changed nothing, for every other resume: a
    /// thread in another state, a hook, a recording, a full resume depth.
    /// Those are [`Self::fast_resume`]'s.
    #[inline(never)]
    #[allow(unsafe_code)]
    pub(crate) fn resume_in_place(
        &mut self,
        stack: &mut CallStack,
        call: Instruction,
        thread: ThreadHandle,
        (first, count): (usize, usize),
        wrapped: bool,
    ) -> bool {
        // A thread that runs a few instructions to its yield is cheaper as
        // a short run ([`Self::resume_direct`]): the caller asked
        // [`Self::runs_long`].
        if self.in_place_base == usize::MAX
            || self.resumer_stacks.len() >= MAX_NESTED_RESUMES
            || !(self.short_run_allowed()
                && self.resume_memo.is_none()
                && self.pending_gc_recheck_depth.is_none())
        {
            return false;
        }
        let Some(coroutine) = self.coroutines.get_mut(&thread) else {
            return false;
        };
        let CoroutineState::Suspended {
            frames,
            yield_return,
            host_resume: None,
        } = &mut coroutine.state
        else {
            return false;
        };
        let target = *yield_return;
        let (taken, taking) = (usize::from(target.base), usize::from(target.results));
        // The thread's top frame takes the values in its registers and goes
        // on at an instruction of its chunk, after the one it last ran.
        let Some(top) = frames.last_frame() else {
            return false;
        };
        if taking == 0
            || ((top.top - top.base) as usize) < taken + taking
            || top.pc >= top.chunk.code.len()
            || top
                .last_interpreted_pc
                .get()
                .is_some_and(|last| top.pc <= last)
        {
            return false;
        }
        let depth = stack.len();
        let resumer = stack.at_mut(depth - 1);
        let (base, wanted) = (usize::from(call.a()), usize::from(call.c()));
        let Some(given) = resumer.registers.get(first..first + count) else {
            return false;
        };
        if base + wanted > resumer.registers.len() {
            return false;
        }
        // Nothing below declines.
        let index = frames.len() - 1;
        let entered = frames.at_mut(index);
        entered.frame.clear_open_results();
        let mut given = given.iter();
        for slot in &mut entered.registers[taken..taken + taking] {
            match given.next() {
                Some(value) => crate::vm::registers::store_clone_scalar(slot, value),
                None => crate::vm::registers::set_register(slot, Value::Nil),
            }
        }
        // The call is behind the resumer while the thread runs, as the fast
        // loop leaves it for every native step.
        let previous = resumer.frame.last_interpreted_pc;
        resumer.frame.last_interpreted_pc = LastPc::some(resumer.frame.pc);
        resumer.frame.pc += 1;
        let environment = coroutine.environment;
        if coroutine.traceback.is_some() {
            coroutine.traceback = None;
        }
        let CoroutineState::Suspended { frames, .. } =
            std::mem::replace(&mut coroutine.state, CoroutineState::Running)
        else {
            unreachable!("the state was matched as suspended")
        };
        let parked = std::ptr::NonNull::from(Box::leak(frames));
        // SAFETY: `parked` is the allocation of the box just leaked, which
        // the record made below owns until it is ended. No reference to it
        // outlives this function: `Vm::resumer_stacks` takes its address.
        let waiting = unsafe { &mut *parked.as_ptr() };
        std::mem::swap(stack, waiting);
        let switch = self.enter_thread(waiting, thread, environment);
        self.in_place.push(InPlace {
            parked,
            thread,
            switch,
            previous,
            base: call.a(),
            results: call.c(),
            wrapped,
        });
        self.jit.statistics.coroutine_direct_resumes += 1;
        self.jit.statistics.coroutine_in_place_resumes += 1;
        true
    }

    /// `coroutine.yield` from the top frame of a thread the fast loop
    /// resumed in place: the `count` registers from `first` go to the
    /// result registers of the resume, behind its `true`, the stacks change
    /// places again and the thread parks exactly as a direct yield parks it
    /// (see [`Self::resume_plain`]), with `call` as the call that takes the
    /// next resume's values. The loop goes on in the resumer's frame, after
    /// its call.
    ///
    /// Returns `None`, having changed nothing, for a yield the general path
    /// has to take or to refuse. The caller has checked that the innermost
    /// resume in place is the running loop's.
    #[inline(never)]
    #[allow(unsafe_code, clippy::option_option)]
    pub(crate) fn yield_in_place(
        &mut self,
        stack: &mut CallStack,
        call: Instruction,
        first: usize,
        count: usize,
    ) -> Option<Option<FastExit>> {
        if !self.debug_hooks.is_empty()
            || self.jit.trace.recording()
            || self.pending_host_resume.is_some()
        {
            return None;
        }
        let depth = stack.len();
        // A thread of one frame under no host call has nothing to cross.
        if !(depth <= 1 && self.host_yield_boundaries.is_empty()) && self.yield_blocked(stack) {
            return None;
        }
        let record = self.in_place.last()?;
        let (parked, thread, switch) = (record.parked, record.thread, record.switch);
        let (base, wanted, wrapped) = (
            usize::from(record.base),
            usize::from(record.results),
            record.wrapped,
        );
        debug_assert_eq!(self.current_coroutine, Some(thread));
        // SAFETY: the record owns the allocation, and nothing else holds a
        // reference to it (`Vm::resumer_stacks` has its address, which
        // nothing reads during this function).
        let waiting = unsafe { &mut *parked.as_ptr() };
        {
            let yielding = stack.at_mut(depth - 1);
            let given = yielding.registers.get(first..first + count)?;
            let index = waiting.len() - 1;
            let resumer = waiting.at_mut(index);
            // The resume checked the range, and a parked frame keeps its
            // registers.
            let slots = resumer.registers.get_mut(base..base + wanted)?;
            resumer.frame.clear_open_results();
            let mut slots = slots.iter_mut();
            if !wrapped && let Some(slot) = slots.next() {
                crate::vm::registers::set_register(slot, Value::Boolean(true));
            }
            let mut given = given.iter();
            for slot in slots {
                match given.next() {
                    Some(value) => crate::vm::registers::store_clone_scalar(slot, value),
                    None => crate::vm::registers::set_register(slot, Value::Nil),
                }
            }
            // The thread stands after its call, as the fast loop leaves it
            // around every direct yield.
            yielding.frame.last_interpreted_pc = LastPc::some(yielding.frame.pc);
            yielding.frame.pc += 1;
        }
        self.in_place.pop();
        std::mem::swap(stack, waiting);
        // SAFETY: the record is gone, and this takes over what it owned.
        // Parking the thread drops the address `Vm::resumer_stacks` has.
        let frames = unsafe { Box::from_raw(parked.as_ptr()) };
        self.park_plain(
            thread,
            switch,
            frames,
            CallReturn {
                base: call.a(),
                results: call.c(),
            },
        );
        self.jit.statistics.coroutine_direct_yields += 1;
        // The resumer stands after a native step that may have allocated.
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Some(Some(FastExit::Safepoint));
        }
        Some(None)
    }

    /// Ends the resumes in place past the first `base`, for a fast loop that
    /// ended with `exit` while their threads ran, the innermost first.
    /// Returns `None` when the loop has to go on in `stack` (a thread
    /// finished its resume and the resumer stands after its call), and
    /// otherwise the exit of the loop for the stack it entered with.
    #[cold]
    #[inline(never)]
    pub(crate) fn leave_in_place_from(
        &mut self,
        stack: &mut CallStack,
        base: usize,
        mut exit: VmResult<FastExit>,
    ) -> Option<VmResult<FastExit>> {
        while self.in_place.len() > base {
            exit = match self.leave_in_place(stack, exit) {
                Ok(None) => return None,
                Ok(Some(exit)) => Ok(exit),
                Err(error) => Err(error),
            };
        }
        Some(exit)
    }

    /// Turns the innermost resume in place into a nested one: the fast loop
    /// ended with `exit` in the thread's frames, which this takes out of
    /// `stack`, giving the resumer its frames back. The thread then runs to
    /// its next switch as [`Self::run_resumed_in_fast_loop`] goes on after
    /// the same exit, and its answer goes to the call's registers as
    /// [`Self::fast_resume_thread`] writes it.
    ///
    /// `None` is a resume that finished: the resumer stands after its call.
    /// An exit is the resumer's: the call the general path has to make again
    /// (it reports a failed wrapper), or a collection that is due.
    #[allow(unsafe_code)]
    fn leave_in_place(
        &mut self,
        stack: &mut CallStack,
        exit: VmResult<FastExit>,
    ) -> VmResult<Option<FastExit>> {
        let record = self
            .in_place
            .pop()
            .expect("the caller found a resume in place");
        // SAFETY: the record owned the allocation (see `InPlace`), and no
        // reference to it is live.
        let mut frames = unsafe {
            std::ptr::swap(std::ptr::from_mut(stack), record.parked.as_ptr());
            Box::from_raw(record.parked.as_ptr())
        };
        // The resumer's frames are where its owner holds them again, as for
        // a nested resume.
        *self
            .resumer_stacks
            .last_mut()
            .expect("the resume parked its resumer") = ParkedStack::new(stack);
        self.jit.statistics.coroutine_in_place_exits += 1;
        let first = match exit {
            // A yield the general way parked its values.
            Ok(FastExit::Yielded) => Ok(None),
            Ok(FastExit::Safepoint) => self.safepoint_then_interpret(&mut frames),
            Ok(_) => self.interpret(&mut frames, false),
            Err(error) => {
                self.abort_recording_on_error();
                Err(error)
            }
        };
        self.quick.pure = false;
        let answer = self.drive_thread(
            record.thread,
            record.switch,
            frames,
            Some(first),
            record.wrapped,
        );
        let frame_index = stack.len() - 1;
        let call = Instruction::call(record.base, 0, record.results);
        match answer {
            Ok(results)
                if !record.wrapped || matches!(results.first(), Some(Value::Boolean(true))) =>
            {
                let frame = stack.at_mut(frame_index);
                frame.frame.clear_open_results();
                // A wrapper's results follow the `true` of the resume.
                let values = &results[usize::from(record.wrapped)..];
                crate::vm::registers::write_fixed_results(frame.registers, call, values)?;
                self.recycle_values(results);
            }
            Err(error) if !record.wrapped => {
                return Err(self.builtin_call_error(
                    stack,
                    frame_index,
                    &Value::BuiltinFunction(crate::runtime::BuiltinFunction::CoroutineResume),
                    error,
                ));
            }
            // A wrapper whose thread failed: the general path's call takes
            // the answer and raises it (see `fast_resume_thread`).
            answer => {
                self.remember_resume(record.thread, answer);
                let frame = &mut stack[frame_index];
                frame.pc -= 1;
                frame.last_interpreted_pc = record.previous;
                return Ok(Some(FastExit::Fallback));
            }
        }
        if self.runtime.allocation_serial() != self.gc_observed_allocation_serial
            && self.runtime.gc_needs_collection()
        {
            return Ok(Some(FastExit::Safepoint));
        }
        Ok(None)
    }

    /// Whether any debug hook is installed on any thread.
    pub(crate) fn hooks_installed(&self) -> bool {
        !self.debug_hooks.is_empty()
    }

    /// Keeps the answer of a resume the trace made for the interpreter's
    /// replay of the call ([`Self::resume_coroutine_mode`] takes it).
    pub(crate) fn remember_resume(&mut self, thread: ThreadHandle, result: VmResult<Vec<Value>>) {
        self.resume_memo = Some(ResumeMemo::new(thread, result));
    }

    /// [`Self::remember_resume`] for the answer of [`Self::resume_for_trace`].
    pub(crate) fn remember_resumed(&mut self, thread: ThreadHandle, resumed: Resumed) {
        let result = match resumed {
            Resumed::Yielded => Ok(self.quick.take_with_status()),
            Resumed::Done(result) => *result,
        };
        self.remember_resume(thread, result);
    }

    /// Forgets a remembered answer that no replay took.
    pub(crate) fn forget_resume(&mut self) {
        self.resume_memo = None;
    }

    pub(crate) fn yield_coroutine(
        &mut self,
        frames: &CallStack,
        instruction: Instruction,
        arguments: Vec<Value>,
    ) -> VmResult<()> {
        if self.current_coroutine.is_none() {
            return Err(RuntimeError::message("attempt to yield from outside a coroutine").into());
        }
        if self.yield_refused(frames) {
            let error = self.yield_boundary_error();
            // Stock LuaJIT adds the line only when the Lua frame calls
            // yield itself. A tail call has replaced that frame.
            if self.compatibility == CompatibilityProfile::LuaJit21
                && matches!(
                    instruction.opcode(),
                    Ok(OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen)
                )
            {
                return Err(RuntimeError::Raised(Value::from(
                    "attempt to yield across C-call boundary",
                ))
                .into());
            }
            return Err(error.into());
        }
        self.pending_yield = Some((
            CallReturn {
                base: instruction.a(),
                results: instruction.c(),
            },
            arguments,
            false,
        ));
        Ok(())
    }

    pub(crate) fn set_coroutine_runtime_roots(
        &mut self,
        thread: ThreadHandle,
        frames: &mut CallStack,
    ) -> VmResult<()> {
        let roots = Self::gc_roots(self.thread_environment, frames, &[]);
        self.runtime.set_thread_roots(thread, roots)?;
        Ok(())
    }

    pub(crate) fn prune_collected_coroutines(&mut self) {
        let runtime = &self.runtime;
        if self
            .coroutines
            .keys()
            .all(|thread| runtime.thread_exists(*thread))
        {
            return;
        }
        let collected: Vec<ThreadHandle> = self
            .coroutines
            .keys()
            .filter(|thread| !runtime.thread_exists(**thread))
            .copied()
            .collect();
        for thread in collected {
            if let Some(mut coroutine) = self.coroutines.remove(&thread) {
                // Closures that outlive the thread keep the variables its
                // frames captured: close them before the stack goes.
                self.close_coroutine_upvalues(&mut coroutine);
            }
        }
    }

    /// Closes every captured local on the stacks a coroutine owns.
    pub(crate) fn close_coroutine_upvalues(&mut self, coroutine: &mut Coroutine) {
        if let CoroutineState::Suspended {
            frames,
            host_resume,
            ..
        } = &mut coroutine.state
        {
            self.close_upvalues(frames, 0);
            let mut resume = host_resume.as_mut();
            while let Some(continuation) = resume {
                let Some(child) = continuation.child.as_mut() else {
                    break;
                };
                self.close_upvalues(&mut child.frames, 0);
                resume = child.host_resume.as_mut();
            }
        }
        if let Some((frames, _)) = &mut coroutine.failed {
            self.close_upvalues(frames, 0);
        }
    }
}

/// The coroutine a `coroutine` function takes first: Lua 5.1 and `LuaJIT`
/// say "coroutine expected", and Lua 5.2 and later name the type found.
pub(crate) fn coroutine_argument(
    arguments: &[Value],
    compatibility: CompatibilityProfile,
) -> VmResult<ThreadHandle> {
    match arguments.first() {
        Some(Value::Thread(thread)) => Ok(*thread),
        _ if compatibility.is_lua51_family() => {
            Err(RuntimeError::argument_error(1, "coroutine expected").into())
        }
        argument => Err(RuntimeError::bad_argument(1, "thread", argument).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Chunk;
    use crate::compat::CompatibilityProfile;
    use crate::vm::DofileCompletion;

    fn stack_with(vm: &mut Vm, callers: Vec<Option<ReturnTarget>>) -> CallStack {
        let mut frames = CallStack::new();
        for caller in callers {
            let chunk = Chunk::new(1, Vec::new(), vec![Instruction::halt()]);
            let frame = vm
                .make_frame(
                    chunk,
                    Vec::<Value>::new(),
                    Vec::<crate::runtime::CellHandle>::new(),
                    caller,
                    None,
                    vm.thread_environment,
                    None,
                )
                .unwrap();
            frames.push(frame);
        }
        frames
    }

    fn protected() -> Option<ReturnTarget> {
        Some(ReturnTarget::Protected(CallReturn {
            base: 0,
            results: 1,
        }))
    }

    fn direct() -> Option<ReturnTarget> {
        Some(ReturnTarget::Direct(CallReturn {
            base: 0,
            results: 1,
        }))
    }

    #[test]
    fn protected_frames_are_yieldable_except_in_lua51() {
        for (profile, allowed) in [
            (CompatibilityProfile::Lua51, false),
            (CompatibilityProfile::Lua52, true),
            (CompatibilityProfile::Lua54, true),
            (CompatibilityProfile::Lua55, true),
            (CompatibilityProfile::LuaJit21, true),
        ] {
            let mut vm = Vm::with_profile(profile);
            let frames = stack_with(&mut vm, vec![None, direct(), protected(), direct()]);
            assert_eq!(vm.frames_allow_yield(&frames), allowed, "{profile:?}");
        }
    }

    #[test]
    fn non_resumable_targets_always_block_a_yield() {
        let handler = || {
            Some(ReturnTarget::ExtendedProtectedHandler(Box::new(
                crate::vm::ExtendedProtectedCall {
                    target: CallReturn {
                        base: 0,
                        results: 1,
                    },
                    handler: Value::Nil,
                    suppress_hooks: false,
                    handler_calls: 0,
                    pending_unwind: None,
                },
            )))
        };
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
            CompatibilityProfile::LuaJit21,
        ] {
            let mut vm = Vm::with_profile(profile);
            let frames = stack_with(
                &mut vm,
                vec![
                    None,
                    Some(ReturnTarget::Dofile(DofileCompletion::Direct(CallReturn {
                        base: 0,
                        results: 1,
                    }))),
                ],
            );
            assert!(!vm.frames_allow_yield(&frames), "{profile:?}");
            let frames = stack_with(&mut vm, vec![None, handler()]);
            assert!(!vm.frames_allow_yield(&frames), "handler {profile:?}");
        }
    }

    #[test]
    fn metamethod_continuations_are_yieldable_except_in_lua51() {
        for (profile, allowed) in [
            (CompatibilityProfile::Lua51, false),
            (CompatibilityProfile::Lua54, true),
            (CompatibilityProfile::LuaJit21, true),
        ] {
            let mut vm = Vm::with_profile(profile);
            {
                let target = ReturnTarget::CloseLocal;
                let frames = stack_with(&mut vm, vec![None, Some(target)]);
                assert_eq!(vm.frames_allow_yield(&frames), allowed, "{profile:?}");
            }
        }
    }

    #[test]
    fn boundary_message_follows_the_profile() {
        for (profile, expected, positioned) in [
            (
                CompatibilityProfile::Lua51,
                "attempt to yield across metamethod/C-call boundary",
                false,
            ),
            (
                CompatibilityProfile::Lua54,
                "attempt to yield across a C-call boundary",
                false,
            ),
            (
                CompatibilityProfile::LuaJit21,
                "attempt to yield across C-call boundary",
                true,
            ),
        ] {
            let error = Vm::with_profile(profile).yield_boundary_error();
            match (error, positioned) {
                (RuntimeError::Raised(Value::String(message)), false) => {
                    assert_eq!(message.to_lossy(), expected);
                }
                (RuntimeError::Message(message), true) => assert_eq!(message, expected),
                (other, _) => panic!("{profile:?}: {other:?}"),
            }
        }
    }
    fn blank_coroutine(vm: &Vm) -> Coroutine {
        Coroutine {
            environment: vm.thread_environment,
            state: CoroutineState::Dead,
            traceback: None,
            failed: None,
            roots_stale: false,
            listed: false,
            long_runs: false,
            long_resumes: 0,
        }
    }

    #[test]
    fn the_coroutine_table_finds_replaces_and_removes_by_handle_in_handle_order() {
        let mut vm = Vm::new();
        let handles: Vec<ThreadHandle> = (0..4)
            .map(|_| vm.runtime_mut().create_thread().unwrap())
            .collect();
        let mut table = CoroutineTable::default();
        for handle in handles.iter().rev() {
            assert!(table.insert(*handle, blank_coroutine(&vm)).is_none());
        }
        assert!(handles.iter().all(|handle| table.contains_key(handle)));
        assert_eq!(table.keys().copied().collect::<Vec<_>>(), handles);
        // Replacing hands back the displaced coroutine.
        table.get_mut(&handles[1]).unwrap().roots_stale = true;
        let displaced = table.insert(handles[1], blank_coroutine(&vm)).unwrap();
        assert!(displaced.roots_stale);
        assert!(!table.get(&handles[1]).unwrap().roots_stale);
        // Removing one leaves the others where they were.
        assert!(table.remove(&handles[2]).is_some());
        assert!(table.remove(&handles[2]).is_none());
        assert!(table.get(&handles[2]).is_none());
        assert_eq!(table.keys().count(), 3);
        assert!(table.get(&handles[3]).is_some());
        assert_eq!(table.values_mut().count(), 3);
    }

    #[test]
    fn a_table_of_coroutines_answers_for_a_freed_slot_only_to_the_new_handle() {
        let mut vm = Vm::new();
        let first = vm.runtime_mut().create_thread().unwrap();
        let mut table = CoroutineTable::default();
        table.insert(first, blank_coroutine(&vm));
        // Collect the thread: nothing roots it, and the arena reuses its
        // slot for the next one with a new generation.
        let roots = GcRootBuilder::new().finish();
        vm.runtime_mut().collect_full(roots);
        let second = vm.runtime_mut().create_thread().unwrap();
        assert_eq!(first.index(), second.index());
        assert_ne!(first, second);
        // The old entry still answers to its own handle until it is pruned,
        // never to the new one's.
        assert!(table.get(&first).is_some());
        assert!(table.get(&second).is_none());
        let displaced = table.insert(second, blank_coroutine(&vm));
        assert!(displaced.is_some(), "the stale entry is handed back");
        assert!(table.get(&first).is_none());
        assert!(table.get(&second).is_some());
    }
}

#[cfg(test)]
mod quick_yield_tests {
    use super::*;

    // The purity of a short run belongs to the values it parked, so taking or
    // clearing them must end it.
    #[test]
    fn purity_ends_with_the_parked_values() {
        let mut quick = QuickYield::default();
        quick.values[0] = Value::Boolean(true);
        quick.len = 1;
        quick.pure = true;
        quick.clear();
        assert!(!quick.pure());

        quick.len = 1;
        quick.pure = true;
        quick.take_with_status();
        assert!(!quick.pure());
    }

    // The answer of a trace's resume comes back through two registers. A
    // wider one is written to memory by the callee and read back in loads
    // that cannot forward from its narrow stores, on every switch.
    #[test]
    fn a_resume_answer_fits_two_registers() {
        assert!(std::mem::size_of::<Option<Resumed>>() <= 16);
    }
}
