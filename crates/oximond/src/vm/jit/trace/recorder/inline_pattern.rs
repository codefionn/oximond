//! `string.find`, `string.match`, `string.gmatch` and `string.gsub` with
//! patterns, as calls of helpers.
//!
//! The recorder runs the pattern on the live arguments (the interpreter is
//! about to make the same call) to learn how the call ends: whether it finds
//! a match and what the results are. A match has results of fixed shape for a
//! pattern, which captures are strings and which are positions, so the
//! recording calls the helper of that shape (see
//! [`oximond::jit_ir::trace::HelperId::string_match`]). A call that finds none
//! calls the helper that misses when there is a match. Either way a miss
//! leaves through the `Before` snapshot of the call, the interpreter makes it,
//! and the guard of these helpers takes a side trace (see `side_exits_of` and
//! `continue_guard`): the side records the call as it now ends, with the helper
//! of the outcome it sees.
//!
//! The pattern is an operand like any other: it is not guarded to be the
//! recorded one. A later call with another pattern gives its own result,
//! and a helper that finds the shape of its results to differ misses.
//!
//! `string.gmatch` is two helpers. The call makes the iterator, a lane of the
//! run's bank, and a generic `for` that calls an iterator steps it: the
//! helper reads the iterator's cursor and commits the new one, the way the
//! interpreter's step does (`Vm::call_gmatch_iterator`). The cursor is state no
//! table holds, so a step is a private-state effect that stays in order.

use super::{
    Flow, HookPoint, Operand, RecKind, Recorder, TraceAbortReason, TraceScalar, TraceType,
};
use crate::bytecode::OpCode;
use crate::jit_ir::trace::{
    HelperId, MAX_FIND_CAPTURES, MAX_HELPER_RESULTS, PINNED_BUILTIN_STRING_GSUB,
    pinned_builtin_identity, pinned_native_identity,
};
use crate::runtime::{DirectNativeKind, GmatchIteratorHandle, NativeFunctionHandle, Value};
use crate::vm::jit::trace::host::pattern::{self, Outcome, Replacement, Shape};

/// The guest instructions a pattern call counts for, on top of any call of a
/// helper, when the policy weighs the work of a loop. A match of a short
/// subject costs the interpreter a few hundred host instructions.
const PATTERN_GUEST_WORK: u32 = 100;
/// How many steps ahead a gmatch loop's recording looks to tell how long the
/// loop runs. More than any trace needs per entry.
const GMATCH_LOOKAHEAD: usize = 64;

impl Recorder {
    /// Counts the work of a pattern call.
    fn add_pattern_work(&mut self) {
        self.host_work = self.host_work.saturating_add(PATTERN_GUEST_WORK);
    }

    /// Counts the work of a pattern call that scans a subject of `length`
    /// bytes, as a `string.gsub` does all of it.
    fn add_scan_work(&mut self, length: usize) {
        let bytes = u32::try_from(length / 4).unwrap_or(u32::MAX);
        self.host_work = self.host_work.saturating_add(bytes);
    }

    /// [`Recorder::emit_helper_call`] for a pattern helper.
    fn emit_pattern_call(
        &mut self,
        pc: usize,
        helper: HelperId,
        args: &[(Operand, TraceType)],
    ) -> u32 {
        self.add_pattern_work();
        self.emit_helper_call(pc, helper, args)
    }

    /// Whether `kind` is a function of this module.
    pub(super) const fn is_pattern_kind(kind: DirectNativeKind) -> bool {
        matches!(
            kind,
            DirectNativeKind::Match | DirectNativeKind::Gmatch | DirectNativeKind::Find
        )
    }

    /// The results of the helper call `call`, whose result types are `types`
    /// (an integer is a position, as the profile has numbers), as operations.
    pub(super) fn pattern_results(
        &mut self,
        pc: usize,
        call: u32,
        types: impl IntoIterator<Item = TraceType>,
    ) -> Result<Vec<u32>, TraceAbortReason> {
        let number = self.number_type();
        let mut values = Vec::new();
        for (index, ty) in types.into_iter().enumerate() {
            let result = if index == 0 {
                call
            } else {
                self.emit(
                    pc,
                    RecKind::HelperResult {
                        call,
                        index: u8::try_from(index).map_err(|_| TraceAbortReason::Divergence)?,
                    },
                    Some(ty),
                )
            };
            values.push(if ty == TraceType::Integer && number == TraceType::Float {
                self.emit(pc, RecKind::IntToFloat(Operand::Op(result)), Some(number))
            } else {
                result
            });
        }
        Ok(values)
    }

    /// The result types of a match of `shape`, after `lead` positions.
    fn shape_types(shape: Shape, lead: usize) -> Vec<TraceType> {
        let mut types = vec![TraceType::Integer; lead];
        types.extend((0..shape.count - lead).map(|position| {
            if shape.strings & (1 << position) != 0 {
                TraceType::String
            } else {
                TraceType::Integer
            }
        }));
        types
    }

    /// The shape of the results a helper gives for a match of `found`, in a
    /// call that wants `results` of them (zero: all, as an open call does).
    /// A call that wants fewer than the match gives takes the first ones only,
    /// so a pattern with another number of captures, or other kinds past those,
    /// answers the same helper. `lead` is the number of positions a
    /// `string.find` leads with.
    fn wanted_shape(found: Shape, results: usize, lead: usize) -> Result<Shape, TraceAbortReason> {
        let shape = if results == 0 {
            found
        } else {
            found.wanting(results, lead).unwrap_or(found)
        };
        let most = if lead == 0 {
            MAX_HELPER_RESULTS
        } else {
            MAX_FIND_CAPTURES + lead
        };
        if shape.count > most {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        Ok(shape)
    }

    /// A call of `string.find` the plain-text helper does not take,
    /// `string.match` or `string.gmatch`. `None` for a `string.find` of the
    /// plain-text helper.
    #[allow(clippy::too_many_lines)]
    pub(super) fn record_pattern_native(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let text = |ty: TraceType| matches!(ty, TraceType::String | TraceType::StringRef);
        let position =
            |ty: TraceType| matches!(ty, TraceType::Integer | TraceType::Float | TraceType::Nil);
        let find = kind == DirectNativeKind::Find;
        let wanted = match kind {
            DirectNativeKind::Find => 2..=4,
            _ => 2..=3,
        };
        if !wanted.contains(&arguments.len())
            || !text(arguments[0].1)
            || !text(arguments[1].1)
            || !arguments[2..].iter().enumerate().all(|(place, (_, ty))| {
                position(*ty) || (find && place == 1 && matches!(ty, TraceType::Boolean))
            })
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let live = at
            .registers
            .get(first_argument..first_argument + arguments.len())
            .ok_or(TraceAbortReason::Divergence)?;
        if kind == DirectNativeKind::Gmatch {
            // The initial position is for Lua 5.4 and later, whose native
            // takes it; the others ignore a third argument.
            let (Value::String(_), Value::String(_)) = (&live[0], &live[1]) else {
                return Err(TraceAbortReason::UnsupportedOperand);
            };
            let lane = self.pin(pinned_native_identity(function.index()))?;
            self.guard_callee(pc, callee, lane);
            let iterator = self.emit_pattern_call(pc, HelperId::GMATCH_NEW, arguments);
            return Ok(Some(self.native_result(pc, iterator, (base, results))));
        }
        let prediction =
            pattern::predict(runtime, kind, live).ok_or(TraceAbortReason::UnsupportedOperand)?;
        // The plain-text helper answers a `string.find` of one result that
        // looks for plain text, where it agrees with the pattern helpers
        // (LuaJIT starts a search past the end at the end).
        if find && results == 1 && super::predict_find(live) == Some(prediction != Outcome::Absent)
        {
            return Ok(None);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let Outcome::Found(shape) = prediction else {
            // None found: every result is nil, and the one nil is all the
            // call gives.
            let helper = if find {
                HelperId::STRING_FIND_ABSENT
            } else {
                HelperId::STRING_MATCH_ABSENT
            };
            self.add_pattern_work();
            let value = self.emit_library_helper(pc, helper, arguments, TraceType::Nil);
            return Ok(Some(self.native_result(pc, value, (base, results))));
        };
        let lead = if find { 2 } else { 0 };
        let shape = Self::wanted_shape(shape, results, lead)?;
        let (helper, types) = if find {
            let captures = shape.count - 2;
            let helper = if shape.prefix {
                HelperId::string_find_prefix(captures, shape.strings)
            } else {
                HelperId::string_find_pattern(captures, shape.strings)
            };
            (helper, Self::shape_types(shape, 2))
        } else {
            let helper = if shape.prefix {
                HelperId::string_match_prefix(shape.count, shape.strings)
            } else {
                HelperId::string_match(shape.count, shape.strings)
            };
            (helper, Self::shape_types(shape, 0))
        };
        let call = self.emit_pattern_call(pc, helper, arguments);
        let values = self.pattern_results(pc, call, types)?;
        Ok(Some(self.native_results(pc, &values, (base, results))))
    }

    /// Whether the `ForInCall` at this hook is the step of the loop this
    /// recording closes at: the `ForInLoop` after it branches to the header.
    fn steps_the_loop_of_this_recording(&self, at: &HookPoint<'_>) -> bool {
        at.chunk.code.get(at.pc + 1).is_some_and(|next| {
            next.opcode() == Ok(OpCode::ForInLoop)
                && self
                    .target(at.pc + 1, super::short_offset(*next))
                    .is_ok_and(|target| target == self.header)
        })
    }

    /// The generic `for` at `base` steps a `string.gmatch` iterator and goes
    /// back to the header this recording closes at: the iterations the
    /// loop has left are the matches the iterator has left, which the cursor
    /// tells (counted up to [`GMATCH_LOOKAHEAD`]). The policy turns a loop
    /// that is too short to repay a trace away with it, as it does a numeric
    /// for loop.
    pub(super) fn note_gmatch_trips(&mut self, at: &HookPoint<'_>, base: usize) {
        if self.side.is_some() || !self.frames.is_empty() || self.function.is_some() {
            return;
        }
        let (Some(Value::GmatchIterator(handle)), Some(around)) =
            (at.registers.get(base), at.surroundings)
        else {
            return;
        };
        #[allow(clippy::cast_precision_loss)]
        {
            self.remaining_trips =
                pattern::gmatch_remaining(around.runtime, *handle, GMATCH_LOOKAHEAD)
                    .map(|steps| steps as f64);
        }
    }

    /// A call of `string.gsub` with a string or a table for the replacement.
    pub(super) fn record_gsub(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let text = |ty: TraceType| matches!(ty, TraceType::String | TraceType::StringRef);
        if !(3..=4).contains(&arguments.len())
            || !text(arguments[0].1)
            || !text(arguments[1].1)
            || !(text(arguments[2].1) || arguments[2].1 == TraceType::Table)
            || arguments.get(3).is_some_and(|(_, ty)| {
                !matches!(ty, TraceType::Integer | TraceType::Float | TraceType::Nil)
            })
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let live = at
            .registers
            .get(first_argument..first_argument + arguments.len())
            .ok_or(TraceAbortReason::Divergence)?;
        // Run the call, to see that a helper can answer it.
        let (Value::String(source), Value::String(pattern)) = (&live[0], &live[1]) else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let replacement = match &live[2] {
            Value::String(template) => Replacement::Template(template.as_bytes()),
            Value::Table(table) => Replacement::Table(*table),
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let maximum =
            pattern::live_limit(live.get(3)).ok_or(TraceAbortReason::UnsupportedOperand)?;
        pattern::gsub_compute(
            runtime,
            source.as_bytes(),
            pattern.as_bytes(),
            &replacement,
            maximum,
        )
        .ok_or(TraceAbortReason::UnsupportedOperand)?;
        self.add_scan_work(source.len());
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_STRING_GSUB))?;
        self.guard_callee(pc, callee, lane);
        let call = self.emit_pattern_call(pc, HelperId::STRING_GSUB, arguments);
        let values = self.pattern_results(pc, call, [TraceType::String, TraceType::Integer])?;
        Ok(self.native_results(pc, &values, (base, results)))
    }

    /// The call of a `string.gmatch` iterator `handle` held in the callee
    /// operand: a step that gives the next match, or the one that ends the
    /// iteration.
    pub(super) fn record_gmatch_step(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        handle: GmatchIteratorHandle,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let (step, _) =
            pattern::gmatch_probe(runtime, handle).ok_or(TraceAbortReason::UnsupportedOperand)?;
        let arguments = [(callee, TraceType::Closure)];
        // A side that starts at this step takes the path the loop's trace
        // missed. A match of another shape than the trace's would link back
        // to a loop typed for the first one, so only the end of the
        // iteration is a path to record. (Once the side has run round an
        // enclosing loop, the step is a new iterator's first. A step of a
        // loop the trace unrolls links back to nothing typed by the step.)
        if step.found.is_some()
            && self.crossed.is_empty()
            && self.side.as_ref().is_some_and(|seed| seed.start_pc == pc)
            && self.steps_the_loop_of_this_recording(at)
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let Some(found) = step.found else {
            self.emit_pattern_call(pc, HelperId::GMATCH_END, &arguments);
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            return Ok(self.native_result(pc, nil, (base, results)));
        };
        let shape = Self::wanted_shape(pattern::shape_of(&found), results, 0)?;
        let helper = if shape.prefix {
            HelperId::gmatch_next_prefix(shape.count, shape.strings)
        } else {
            HelperId::gmatch_next(shape.count, shape.strings)
        };
        let call = self.emit_pattern_call(pc, helper, &arguments);
        let values = self.pattern_results(pc, call, Self::shape_types(shape, 0))?;
        Ok(self.native_results(pc, &values, (base, results)))
    }
}
