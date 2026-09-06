//! Interpreter fast paths that evaluate validated inline numeric call plans.

use crate::bytecode::Instruction;
use crate::jit_ir::{InlineCallPlan, RegisterFile};
use crate::runtime::{ClosureHandle, Value};
use crate::vm::stack::CallStack;
use crate::vm::{InlineCallArguments, Vm, VmError, VmResult};
use std::mem::MaybeUninit;

/// Error type of the inline evaluators: one pointer, so the hot `Ok` outcome
/// travels in registers instead of through a 72-byte `VmResult` slot.
pub(crate) type InlineResult = Result<bool, Box<VmError>>;

#[cold]
#[inline(never)]
pub(crate) fn boxed<E: Into<VmError>>(error: E) -> Box<VmError> {
    Box::new(error.into())
}

/// Copies `arguments` as numbers into the front of `out`; `false` if any
/// is not a number. `out` is at least as long as `arguments`.
#[inline(always)]
fn fill_numbers(arguments: &[Value], out: &mut [f64]) -> bool {
    for (slot, argument) in out.iter_mut().zip(arguments) {
        let Value::Number(number) = argument else {
            return false;
        };
        *slot = *number;
    }
    true
}

/// Scratch registers that start uninitialised.
///
/// Invariant: every slot is written before it is read. `new` is `unsafe`
/// because only the caller can promise that for the plan it evaluates.
struct UninitRegisters<const N: usize>([MaybeUninit<f64>; N]);

impl<const N: usize> UninitRegisters<N> {
    /// # Safety
    ///
    /// Use only to evaluate a plan whose `writes_before_reads` holds, with
    /// at least `parameter_count` arguments. Then the plan writes each
    /// register (a parameter or an earlier operation's destination) before
    /// any operation reads it, and every index is below `register_count`.
    #[allow(unsafe_code)]
    #[inline(always)]
    unsafe fn new() -> Self {
        Self([MaybeUninit::uninit(); N])
    }
}

impl<const N: usize> RegisterFile for UninitRegisters<N> {
    const SLOTS: usize = N;

    #[allow(unsafe_code)]
    #[inline(always)]
    fn read(&self, index: usize) -> f64 {
        // SAFETY: by the contract of `new`, this slot was written first.
        unsafe { self.0[index & (N - 1)].assume_init() }
    }

    #[inline(always)]
    fn write(&mut self, index: usize, value: f64) {
        self.0[index & (N - 1)] = MaybeUninit::new(value);
    }
}

/// Evaluates `plan`, skipping the scratch zeroing when the plan provably
/// writes every register before it reads it.
#[inline(always)]
fn evaluate_plan(plan: &InlineCallPlan, arguments: &[f64], upvalues: &mut [f64]) -> f64 {
    let count = usize::from(plan.register_count());
    if plan.writes_before_reads() && arguments.len() >= usize::from(plan.parameter_count()) {
        // SAFETY: the plan writes before reads and every parameter has an
        // argument, which is what `UninitRegisters::new` requires.
        #[allow(unsafe_code)]
        unsafe {
            if count <= 8 {
                return plan.evaluate_in(arguments, upvalues, &mut UninitRegisters::<8>::new());
            }
            if count <= 16 {
                return plan.evaluate_in(arguments, upvalues, &mut UninitRegisters::<16>::new());
            }
            if count <= 64 {
                return plan.evaluate_in(arguments, upvalues, &mut UninitRegisters::<64>::new());
            }
        }
    }
    plan.evaluate_arguments_with_upvalues(arguments, upvalues)
}

impl Vm {
    pub(crate) fn try_inline_numeric_fixed_call(
        &mut self,
        frames: &mut CallStack,
        instruction: Instruction,
        function: &Value,
    ) -> VmResult<bool> {
        if instruction.c() != 1 {
            return Ok(false);
        }
        self.try_inline_numeric_call(
            frames,
            instruction.a(),
            function,
            InlineCallArguments::Registers {
                start: usize::from(instruction.a()) + 1,
                count: usize::from(instruction.b()),
            },
        )
    }

    pub(crate) fn try_inline_numeric_call(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        function: &Value,
        arguments: InlineCallArguments<'_>,
    ) -> VmResult<bool> {
        // Each instantiation carries only its own hook handling.
        let result = if self.debug_hooks.is_empty() {
            self.inline_numeric_call::<false>(frames, destination, function, arguments)
        } else {
            self.inline_numeric_call::<true>(frames, destination, function, arguments)
        };
        result.map_err(|error| *error)
    }

    #[allow(clippy::inline_always, reason = "one body, instantiated for each case")]
    #[inline(always)]
    fn inline_numeric_call<const HOOKED: bool>(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        function: &Value,
        arguments: InlineCallArguments<'_>,
    ) -> InlineResult {
        // A recording needs the callee's frame to follow the call into it.
        if self.jit.trace.recording() {
            return Ok(false);
        }
        // A direct evaluator elides a Lua frame. Under a hook that observes
        // frames or lines it must disappear; under count hooks alone it
        // stays and charges the instructions the frame would have run (see
        // `Self::inline_count_charge`).
        if HOOKED && !self.count_loop_owns_budget() {
            return Ok(false);
        }
        let Value::Closure(closure) = function else {
            return Ok(false);
        };
        let arguments = match arguments {
            InlineCallArguments::Values(values) => values,
            InlineCallArguments::Registers { start, count } => frames
                .last_registers()
                .expect("an inline call has a caller")
                .get(start..start + count)
                .ok_or_else(|| {
                    boxed(VmError::RegisterRange {
                        start: destination,
                        count: u8::try_from(count).unwrap_or(u8::MAX),
                    })
                })?,
        };
        // A cached refusal is final until the cache entry changes, and it is
        // the common answer for recursive and non-numeric callees.
        if matches!(&self.jit.inline_call_hot_cache, Some((cached, None)) if cached == closure) {
            return Ok(false);
        }
        // Plans only evaluate numbers, so gather the arguments as numbers
        // first: ordinary calls with other arguments pay for nothing else.
        // Each width zeroes only the buffer it uses: up to four arguments
        // (nearly every call), up to sixteen, then a heap buffer.
        let count = arguments.len();
        if count <= 4 {
            let mut buffer = [0.0_f64; 4];
            if !fill_numbers(arguments, &mut buffer) {
                return Ok(false);
            }
            self.inline_with_numbers::<HOOKED>(frames, destination, *closure, &buffer[..count])
        } else if count <= 16 {
            let mut buffer = [0.0_f64; 16];
            if !fill_numbers(arguments, &mut buffer) {
                return Ok(false);
            }
            self.inline_with_numbers::<HOOKED>(frames, destination, *closure, &buffer[..count])
        } else {
            let mut buffer = vec![0.0_f64; count];
            if !fill_numbers(arguments, &mut buffer) {
                return Ok(false);
            }
            self.inline_with_numbers::<HOOKED>(frames, destination, *closure, &buffer)
        }
    }

    #[inline(always)]
    fn inline_with_numbers<const HOOKED: bool>(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        closure: ClosureHandle,
        numeric_arguments: &[f64],
    ) -> InlineResult {
        if let Some(plan) = self.hot_inline_plan(closure) {
            // SAFETY: the pointer is the `Arc` the hot slot owns, and that
            // slot is overwritten only by `remember_hot_inline_plan`, which
            // runs from `inline_call_plan` on a hot-cache miss. Evaluation
            // only reads or writes one closure cell. It never reaches `inline_call_plan`, so the
            // slot, and with it the plan, outlives this call.
            #[allow(unsafe_code)]
            let plan = unsafe { &*plan };
            return self.evaluate_charged::<HOOKED>(
                frames,
                destination,
                closure,
                numeric_arguments,
                plan,
            );
        }
        match self.inline_call_plan(closure) {
            Some(cached) => self.evaluate_charged::<HOOKED>(
                frames,
                destination,
                closure,
                numeric_arguments,
                &cached.plan,
            ),
            None => Ok(false),
        }
    }

    /// Evaluates `plan`, charging a leased count budget under count hooks.
    #[allow(clippy::inline_always, reason = "measured: the call loop's hot path")]
    #[inline(always)]
    fn evaluate_charged<const HOOKED: bool>(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        closure: ClosureHandle,
        numeric_arguments: &[f64],
        plan: &InlineCallPlan,
    ) -> InlineResult {
        if !HOOKED {
            return self.evaluate_inline_call_plan(
                frames,
                destination,
                closure,
                numeric_arguments,
                plan,
            );
        }
        let Some(left) = self.inline_count_charge(plan) else {
            return Ok(false);
        };
        let outcome =
            self.evaluate_inline_call_plan(frames, destination, closure, numeric_arguments, plan);
        if matches!(outcome, Ok(true)) {
            self.count_budget = left;
        }
        outcome
    }

    /// Whether the current thread's count hook is leased to `count_budget`
    /// by a counting loop, reports no call, return or line event, and no
    /// hook callback is running. Then a frame that is never pushed costs only
    /// the instructions it would have run, which the lease counts. A lease
    /// that a hook change invalidated has a zero budget, which
    /// `inline_count_charge` refuses.
    #[allow(clippy::inline_always, reason = "measured: the call loop's hot path")]
    #[inline(always)]
    fn count_loop_owns_budget(&self) -> bool {
        // A lease that hook changes invalidated has a zero budget, which
        // `inline_count_charge` refuses.
        self.debug_hook_depth == 0
            && self
                .count_lease
                .is_some_and(|lease| lease.count_only && lease.thread == self.current_coroutine)
    }

    /// The budget a direct evaluation of `plan` leaves in the leased count.
    ///
    /// `None` refuses the evaluation: the callee's counted instruction total
    /// is unknown, or the count would fire inside it, so the real call runs
    /// and the hook fires at its exact instruction.
    #[allow(clippy::inline_always, reason = "measured: the call loop's hot path")]
    #[inline(always)]
    fn inline_count_charge(&self, plan: &InlineCallPlan) -> Option<u32> {
        // The hook fires at a counted instruction that finds one left.
        self.count_budget
            .checked_sub(plan.counted_instructions()?)
            .filter(|left| *left >= 1)
    }

    #[allow(clippy::inline_always, reason = "measured: the call loop's hot path")]
    #[inline(always)]
    fn evaluate_inline_call_plan(
        &mut self,
        frames: &mut CallStack,
        destination: u8,
        closure: ClosureHandle,
        numeric_arguments: &[f64],
        plan: &InlineCallPlan,
    ) -> InlineResult {
        if numeric_arguments.len() < usize::from(plan.minimum_argument_count()) {
            return Ok(false);
        }
        // Vm::inline_call_plan already refuses these; this is a backstop for
        // plans that reach here another way.
        if plan.loads_integer_constant() && self.compatibility.is_lua53_or_later() {
            return Ok(false);
        }
        let mut captured_numbers = [0.0; 1];
        let captured_cell = if plan.upvalue_count() == 1 {
            Some(
                self.runtime
                    .closure_cell(closure, 0)
                    .map_err(boxed)?
                    .expect("a one-upvalue plan has one live cell"),
            )
        } else {
            None
        };
        if let Some(cell) = captured_cell {
            let Some(value) = self.runtime.cell_get_number(cell).map_err(boxed)? else {
                // Lua permits a later write through debug.setupvalue or an
                // aliased open cell. Let the ordinary call path apply its
                // normal coercion and error behavior for every non-number.
                return Ok(false);
            };
            captured_numbers[0] = value;
        }
        let result = evaluate_plan(
            plan,
            numeric_arguments,
            &mut captured_numbers[..usize::from(plan.upvalue_count())],
        );
        let frame = frames
            .last_mut()
            .expect("an inline call executes with an active caller frame");
        if plan.writes_upvalue() {
            let cell = captured_cell.expect("an upvalue-writing plan has one live cell");
            if !self
                .runtime
                .cell_set_number(cell, captured_numbers[0])
                .map_err(boxed)?
            {
                return Ok(false);
            }
        }
        frame.registers[usize::from(destination)] = Value::Number(result);
        self.jit.statistics.interpreter_inline_calls = self
            .jit
            .statistics
            .interpreter_inline_calls
            .saturating_add(1);
        Ok(true)
    }
}
