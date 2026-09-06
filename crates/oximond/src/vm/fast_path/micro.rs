//! The short run of a resumed thread: from the resume to the next direct
//! `coroutine.yield`, for a thread whose loop is a few instructions around
//! the yield.
//!
//! The fast loop ([`Vm::run_fast_activations`]) is a large function. Entering
//! it, dispatching eight instructions and leaving it through the switch and
//! the yield cost several times what the instructions themselves do. This
//! runs the instructions that such a loop is made of on their own, with the
//! fast loop's own code for each (same operands, same results, same stored
//! state), and stops *before* any instruction it does not know, or whose
//! operands miss the fast case, with the frame exactly where the fast loop
//! would have left it. What follows is the fast loop's.
//!
//! The arithmetic it knows is that of numbers, with a constant or between
//! two registers.
//!
//! A generator is such a loop too: a numeric `for` around a yield of the
//! loop variable or of an array element. So the run knows the latch of a
//! numeric `for` and a read of a table's array part, which is all of
//! `for i = 1, #t do yield(t[i]) end` between two yields.
//!
//! It runs a thread of one frame only, with no hook, no recording, no
//! pending interrupt, no profile: anything else belongs to the loop that
//! observes it.

use super::{
    Constant, FastExit, Value, constant_at, fetch, float_k_op, mixed_floats, mixed_k_floats, op,
    opcode_byte, relative, short_offset,
};
use crate::compat::{ExecutionMode, FloatModulo};
use crate::runtime::BuiltinFunction;
use crate::vm::CallReturn;
use crate::vm::arith::{constant_value, integer_floor_mod_nonzero, integer_to_number};
use crate::vm::coroutines::QUICK_VALUES;
use crate::vm::frame::LastPc;
use crate::vm::registers::{copy_register, set_register, store_clone_scalar};
use crate::vm::stack::{CallStack, REGISTER_WINDOW};
use crate::vm::{Frame, Vm};

/// How [`Vm::run_micro`] ended.
pub(crate) enum Micro {
    /// The thread yielded, and its values wait in the VM as a direct yield
    /// leaves them.
    Yielded,
    /// The thread stands where the fast loop would take it from. The exit is
    /// the one a backedge asks for, when it asked.
    Left(Option<FastExit>),
}

/// The call of `coroutine.yield` a short run stopped at: the registers of
/// the thread's frame that hold the values it yields, and the call's result
/// registers, which take the values of the next resume.
#[derive(Clone, Copy)]
pub(crate) struct YieldSite {
    pub(crate) first: u16,
    pub(crate) count: u8,
    pub(crate) target: CallReturn,
}

/// Where the instructions stopped. The frame's `pc` and last interpreted pc
/// are stored as the fast loop's exits store them.
pub(crate) enum Stop {
    /// The instruction at `pc` is for the fast loop, untouched.
    Before,
    /// The instruction before `pc` is a call of `coroutine.yield`. Its
    /// values are still in the frame's registers.
    Yielded(YieldSite),
    /// A backward jump landed at `pc`, and the trace engine wants to see it.
    Backedge(FastExit),
}

impl Vm {
    /// Runs the top frame of a thread of one frame until it yields, as
    /// the module says. Nothing it does can be told from the fast loop's
    /// run of the same instructions.
    pub(crate) fn run_micro(&mut self, stack: &mut CallStack) -> Micro {
        if stack.len() != 1
            || self.jit.trace.recording()
            || self.interrupt_pending()
            || !self.debug_hooks.is_empty()
        {
            return Micro::Left(None);
        }
        let tracing = self.execution_mode == ExecutionMode::Tracing && self.runtime.jit_enabled();
        match self.short_run(stack, tracing, false) {
            Stop::Before => Micro::Left(None),
            Stop::Backedge(exit) => Micro::Left(Some(exit)),
            Stop::Yielded(site) => {
                // The values wait in the VM's quick buffer, as
                // `Vm::fast_yield` leaves them.
                let first = usize::from(site.first);
                let count = usize::from(site.count);
                let registers = stack.last_registers().unwrap_or(&[]);
                let quick = &mut self.quick;
                for (slot, value) in quick
                    .values
                    .iter_mut()
                    .zip(&registers[first..first + count])
                {
                    store_clone_scalar(slot, value);
                }
                quick.len = site.count;
                quick.target = site.target;
                quick.pending = true;
                Micro::Yielded
            }
        }
    }

    /// The instructions of [`Self::run_micro`] for a stack of one frame,
    /// with nothing that observes single instructions: the caller has
    /// checked that no hook is installed, nothing records and no interrupt
    /// waits. `threaded` says the stack is a coroutine's, whichever thread
    /// the VM holds as the running one: [`Vm::resume_direct`] runs a
    /// thread's instructions before it makes the thread the running one.
    #[inline(always)]
    pub(crate) fn short_run(
        &mut self,
        stack: &mut CallStack,
        tracing: bool,
        threaded: bool,
    ) -> Stop {
        debug_assert_eq!(stack.len(), 1);
        // SAFETY: the stack has its one frame live. The frame and window are
        // used before anything else touches the stack, and not after.
        #[allow(unsafe_code)]
        unsafe {
            let (frame, window) = stack.running_view();
            self.micro_steps(&mut *frame, &mut *window, tracing, threaded)
        }
    }

    /// The instructions of [`Self::run_micro`], from the frame's `pc`. It
    /// leaves the frame at the instruction not run, at the landing of a
    /// jump, or after the yield, with the last instruction it ran recorded.
    #[allow(clippy::too_many_lines)]
    #[inline(never)]
    pub(crate) fn micro_steps(
        &mut self,
        frame: &mut Frame,
        registers: &mut [Value; REGISTER_WINDOW],
        tracing: bool,
        threaded: bool,
    ) -> Stop {
        // No instruction: `LastPc`'s own encoding of none.
        const NONE: usize = usize::MAX;
        let chunk = frame.chunk.as_arc();
        let code = &chunk.code[..];
        let compatibility = self.compatibility;
        let tagged_integers = compatibility.is_lua53_or_later();
        let float_arith = compatibility.float_arith();
        let floor_modulo =
            float_arith.modulo == FloatModulo::Floor && FloatModulo::has_hardware_floor();
        let mut pc = frame.pc;
        // The last instruction run is the one before `pc`, unless a jump
        // brought the run there: `landed` is where the last jump taken
        // landed and `came` the jump. Control reaches a landing a second
        // time only through another jump, so the two say enough, and the
        // straight instructions keep no record. At the start `pc` counts as
        // a landing of no jump.
        let mut landed = pc;
        let mut came = NONE;
        // Stops before the instruction at `pc`.
        macro_rules! before {
            () => {{
                frame.pc = pc;
                let last = if pc == landed { came } else { pc - 1 };
                if last != NONE {
                    frame.last_interpreted_pc = LastPc::some(last);
                }
                return Stop::Before;
            }};
        }
        // Stops after the backward jump at `pc` landed at `$target`.
        macro_rules! backedge {
            ($exit:expr, $target:expr) => {{
                frame.pc = $target;
                frame.last_interpreted_pc = LastPc::some(pc);
                return Stop::Backedge($exit);
            }};
        }
        // The sum, difference, product, quotient or remainder of a register
        // and a constant, both numbers: the fast loop's fast cases.
        macro_rules! arithmetic_k {
            ($instruction:expr, $a:expr, $opcode:expr, $integer:expr) => {{
                let instruction: crate::bytecode::Instruction = $instruction;
                let left = &registers[usize::from(instruction.b())];
                let constant = constant_at(&chunk, instruction.c().into());
                let result =
                    if let (Value::Number(left), Constant::Number(right)) = (left, constant) {
                        Value::Number(float_k_op(
                            $opcode,
                            floor_modulo,
                            float_arith,
                            *left,
                            *right,
                        ))
                    } else if tagged_integers
                        && let (Value::Integer(left), Constant::Integer(right)) = (left, constant)
                        && let Some(result) = $integer(*left, *right)
                    {
                        Value::Integer(result)
                    } else if let Some((left, right)) = mixed_k_floats(left, constant) {
                        Value::Number(float_k_op($opcode, floor_modulo, float_arith, left, right))
                    } else {
                        before!();
                    };
                set_register(&mut registers[$a], result);
            }};
        }
        if pc >= code.len() {
            before!();
        }
        loop {
            // Every instruction this runs has a successor in the chunk, and
            // a jump lands inside it (checked below), as the fast loop's
            // fetch relies on.
            let instruction = fetch(code, pc);
            let next = pc + 1;
            let a = usize::from(instruction.a());
            match opcode_byte(instruction) {
                op::MOVE => copy_register(registers, a, usize::from(instruction.b())),
                op::LOAD_CONSTANT => {
                    let slot = &mut registers[a];
                    match constant_at(&chunk, instruction.d().into()) {
                        Constant::Number(number) => set_register(slot, Value::Number(*number)),
                        Constant::Integer(integer) if tagged_integers => {
                            set_register(slot, Value::Integer(*integer));
                        }
                        Constant::Integer(integer) => {
                            set_register(slot, Value::Number(integer_to_number(*integer)));
                        }
                        Constant::Boolean(flag) => set_register(slot, Value::Boolean(*flag)),
                        Constant::Nil => set_register(slot, Value::Nil),
                        Constant::Int64(value) => set_register(slot, Value::Int64(*value)),
                        Constant::UInt64(value) => set_register(slot, Value::UInt64(*value)),
                        Constant::String(text)
                            if matches!(slot, Value::String(current)
                                if current.identity() == text.identity()) => {}
                        constant @ Constant::String(_) => {
                            *slot = constant_value(constant, compatibility);
                        }
                    }
                }
                op::LOAD_NIL => set_register(&mut registers[a], Value::Nil),
                op::JUMP | op::JUMP_IF_FALSE => {
                    let is_jump = opcode_byte(instruction) == op::JUMP;
                    if !is_jump && registers[a].is_truthy() {
                        pc = next;
                        continue;
                    }
                    let offset = if is_jump {
                        instruction.jump_offset()
                    } else {
                        short_offset(instruction)
                    };
                    let target = relative(next, offset);
                    if target >= code.len() {
                        before!();
                    }
                    if target <= pc {
                        // The fast loop's backward jump: the pending
                        // interrupt leaves before it, and a backedge is
                        // offered to the trace engine after it.
                        if self.interrupt_pending() {
                            before!();
                        }
                        if let Some(exit) = self.fast_backedge(tracing, target, code) {
                            backedge!(exit, target);
                        }
                    }
                    came = pc;
                    landed = target;
                    pc = target;
                    continue;
                }
                // The latch of a numeric `for`, out of line like the other
                // arms a generator needs: inlined, they cost the arms of a
                // plain ping-pong loop a spilled register.
                op::NUMERIC_FOR_LOOP => {
                    let target = relative(next, short_offset(instruction));
                    if target >= code.len() {
                        before!();
                    }
                    match self.micro_for_latch(registers, a, tracing, target, code) {
                        Latch::Unknown => before!(),
                        Latch::Over => {}
                        Latch::Back => {
                            came = pc;
                            landed = target;
                            pc = target;
                            continue;
                        }
                        Latch::Leave(exit) => backedge!(exit, target),
                    }
                }
                op::GET_TABLE => {
                    if !self.micro_array_read(registers, instruction) {
                        before!();
                    }
                }
                op::GET_UPVALUE => {
                    let Some(cell) = frame.upvalues.cell(usize::from(instruction.d())) else {
                        before!();
                    };
                    let slot = &mut registers[a];
                    // SAFETY: the running frame roots its captured cells, so
                    // a collection keeps them alive.
                    #[allow(unsafe_code)]
                    let value = unsafe { self.runtime.cell_ref_live(cell) };
                    store_clone_scalar(slot, value);
                }
                op::ADD_K => {
                    arithmetic_k!(instruction, a, op::ADD_K, |left: i64, right: i64| Some(
                        left.wrapping_add(right)
                    ));
                }
                op::SUBTRACT_K => {
                    arithmetic_k!(
                        instruction,
                        a,
                        op::SUBTRACT_K,
                        |left: i64, right: i64| Some(left.wrapping_sub(right))
                    );
                }
                op::MULTIPLY_K => {
                    arithmetic_k!(
                        instruction,
                        a,
                        op::MULTIPLY_K,
                        |left: i64, right: i64| Some(left.wrapping_mul(right))
                    );
                }
                op::DIVIDE_K => {
                    arithmetic_k!(instruction, a, op::DIVIDE_K, |_: i64, _: i64| None::<i64>);
                }
                op::MODULO_K => {
                    arithmetic_k!(instruction, a, op::MODULO_K, integer_floor_mod_nonzero);
                }
                op::ADD | op::SUBTRACT | op::MULTIPLY => {
                    if !micro_arithmetic(registers, instruction, tagged_integers) {
                        before!();
                    }
                }
                op::CALL_ONE | op::CALL => {
                    let (function, first_argument, count, results) =
                        if opcode_byte(instruction) == op::CALL {
                            (a, a + 1, usize::from(instruction.b()), instruction.c())
                        } else {
                            (
                                usize::from(instruction.b()),
                                usize::from(instruction.c()),
                                1,
                                1,
                            )
                        };
                    // A thread of one frame under no host call has nothing
                    // to cross. The values stay in the registers: whoever
                    // resumed the thread takes them from there.
                    if results != 0
                        && matches!(
                            registers[function],
                            Value::BuiltinFunction(BuiltinFunction::CoroutineYield)
                        )
                        && count <= QUICK_VALUES
                        && first_argument + count <= (frame.top - frame.base) as usize
                        && (threaded || self.current_coroutine.is_some())
                        && self.host_yield_boundaries.is_empty()
                    {
                        self.jit.statistics.coroutine_direct_yields += 1;
                        // As the switch stores them around the yield.
                        frame.last_interpreted_pc = LastPc::some(pc);
                        frame.pc = next;
                        // `first_argument` is at most 256 and `count` at
                        // most `QUICK_VALUES`.
                        #[allow(clippy::cast_possible_truncation)]
                        return Stop::Yielded(YieldSite {
                            first: first_argument as u16,
                            count: count as u8,
                            target: CallReturn {
                                base: instruction.a(),
                                results,
                            },
                        });
                    }
                    before!();
                }
                _ => before!(),
            }
            pc = next;
        }
    }

    /// The latch of a numeric `for` at the frame's `pc`, as the fast loop
    /// steps it: the control and the visible variable change in place, and a
    /// jump back to `target` polls after it, with the control already
    /// stepped. Operands that are not three floats or three integers are
    /// left untouched ([`Latch::Unknown`]).
    #[inline(never)]
    fn micro_for_latch(
        &mut self,
        registers: &mut [Value; REGISTER_WINDOW],
        a: usize,
        tracing: bool,
        target: usize,
        code: &[crate::bytecode::Instruction],
    ) -> Latch {
        let Some(window) = registers
            .get_mut(a..a + 4)
            .and_then(|window| <&mut [Value; 4]>::try_from(window).ok())
        else {
            return Latch::Unknown;
        };
        let [control, limit, step, variable] = window;
        let continues = match (control, &*limit, &*step) {
            (Value::Number(control), Value::Number(limit), Value::Number(step)) => {
                let value = *control + *step;
                *control = value;
                let continues = if *step > 0.0 {
                    value <= *limit
                } else {
                    value >= *limit
                };
                if continues {
                    set_register(variable, Value::Number(value));
                }
                continues
            }
            (Value::Integer(control), Value::Integer(limit), Value::Integer(step)) => {
                // Lua integer loops stop at the edge of the integer domain.
                match control.checked_add(*step) {
                    None => false,
                    Some(value) => {
                        *control = value;
                        let continues = if *step > 0 {
                            value <= *limit
                        } else {
                            value >= *limit
                        };
                        if continues {
                            set_register(variable, Value::Integer(value));
                        }
                        continues
                    }
                }
            }
            _ => return Latch::Unknown,
        };
        if !continues {
            return Latch::Over;
        }
        if self.interrupt_pending() {
            return Latch::Leave(FastExit::Fallback);
        }
        match self.fast_backedge(tracing, target, code) {
            Some(exit) => Latch::Leave(exit),
            None => Latch::Back,
        }
    }

    /// The `GetTable` `instruction` for a value the table's array part
    /// holds, which is the raw value whatever the metatable. Returns false,
    /// with nothing written, for every other read (a miss, a string key, a
    /// holder that is no table): those are the fast loop's, with its caches
    /// and metamethods.
    #[inline(never)]
    fn micro_array_read(
        &self,
        registers: &mut [Value; REGISTER_WINDOW],
        instruction: crate::bytecode::Instruction,
    ) -> bool {
        let Value::Table(table) = registers[usize::from(instruction.b())] else {
            return false;
        };
        let hit = match registers[usize::from(instruction.c())] {
            Value::Integer(index) => usize::try_from(index)
                .ok()
                .and_then(|slot| self.runtime.table_array_hit(table, slot)),
            Value::Number(index) => self.runtime.table_array_hit_number(table, index),
            _ => None,
        };
        let Some(value) = hit else {
            return false;
        };
        store_clone_scalar(&mut registers[usize::from(instruction.a())], value);
        true
    }
}

/// How the latch of a numeric `for` went ([`Vm::micro_for_latch`]).
enum Latch {
    /// The operands are not the fast loop's fast case; nothing changed.
    Unknown,
    /// The loop is over: the next instruction follows.
    Over,
    /// The loop goes on at its body.
    Back,
    /// The loop goes on, and the jump back leaves the run with this exit.
    Leave(FastExit),
}

/// The sum, difference or product `instruction` asks for, of two registers
/// that hold numbers: the fast loop's fast cases. Returns false, with nothing
/// written, for its slow ones (strings that coerce, metamethods).
#[inline(never)]
fn micro_arithmetic(
    registers: &mut [Value; REGISTER_WINDOW],
    instruction: crate::bytecode::Instruction,
    tagged_integers: bool,
) -> bool {
    let opcode = opcode_byte(instruction);
    let (left, right) = (
        &registers[usize::from(instruction.b())],
        &registers[usize::from(instruction.c())],
    );
    let float = |left: f64, right: f64| match opcode {
        op::ADD => left + right,
        op::SUBTRACT => left - right,
        _ => left * right,
    };
    let result = if let (Value::Number(left), Value::Number(right)) = (left, right) {
        Value::Number(float(*left, *right))
    } else if tagged_integers && let (Value::Integer(left), Value::Integer(right)) = (left, right) {
        Value::Integer(match opcode {
            op::ADD => left.wrapping_add(*right),
            op::SUBTRACT => left.wrapping_sub(*right),
            _ => left.wrapping_mul(*right),
        })
    } else if let Some((left, right)) = mixed_floats(left, right) {
        Value::Number(float(left, right))
    } else {
        return false;
    };
    set_register(&mut registers[usize::from(instruction.a())], result);
    true
}
