//! Trace verification. Run it before optimization and after every
//! transforming pass in development builds.

use super::helper::{HelperEffects, HelperType, MAX_HELPER_ARGS, MAX_HELPER_RESULTS};
use super::ir::{ArithmeticOp, WideConversion};
use super::ir::{
    BindingKind, EntryInput, ExternalSlot, FrameId, HostFunction, LoopParamInit, MAX_HOST_ARGS,
    MAX_INDEX_HOPS, MAX_TRACE_FRAME_DEPTH, META_FIELD_KEY, Protected, ROOT_STRING_LANE_BASE,
    TABLE_ADDRESS_KEY, TERMINAL_GUARD, TableField, TraceInstruction, TraceIr, TraceOp, TraceScalar,
    TraceTerminator, TraceType, ValueDef, ValueId,
};
use super::snapshot::{
    MAX_VIRTUAL_SHAPE_KEYS, RegisterEntry, Snapshot, SnapshotPhase, VirtualAlloc, VirtualKey,
    VirtualTable,
};
use crate::jit_ir::Comparison;
use std::fmt;

/// The detail of the [`TraceVerifyError::InvalidEntry`] for a side whose
/// window (cells, globals, pins, root strings, bindings and fields) does not
/// extend its tree's.
const SIDE_WINDOW_DETAIL: &str = "a side trace has another window than its root";

/// A violated trace invariant. `instruction` counts body instructions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceVerifyError {
    /// The trace has no root frame.
    NoRootFrame,
    /// A value id is outside the value table.
    UnknownValue(ValueId),
    /// The value table disagrees with where the value is defined.
    ValueTableMismatch(ValueId),
    /// A value is used before its definition dominates the use.
    UseBeforeDefinition { instruction: usize, value: ValueId },
    /// An instruction's operand or result has the wrong type.
    TypeMismatch {
        instruction: usize,
        detail: &'static str,
    },
    /// An instruction names a frame or pc outside the trace's frames.
    InvalidSite { instruction: usize },
    /// An entry input is malformed or repeats a register.
    InvalidEntry { index: usize, detail: &'static str },
    /// A loop parameter is malformed.
    InvalidLoopParam { index: usize, detail: &'static str },
    /// Guards are not numbered in body order, or the guard table is wrong.
    InvalidGuard {
        instruction: usize,
        detail: &'static str,
    },
    /// Effects are not numbered in body order, or a snapshot records the
    /// wrong committed-effect count.
    EffectOrder {
        instruction: Option<usize>,
        detail: &'static str,
    },
    /// A snapshot is malformed.
    InvalidSnapshot {
        snapshot: usize,
        detail: &'static str,
    },
    /// A snapshot names a value that is not defined where it is used.
    SnapshotValueUndefined { snapshot: usize, value: ValueId },
    /// A snapshot describes more frames than the bound allows.
    FrameDepth { snapshot: usize, depth: usize },
    /// A snapshot would make the interpreter repeat a committed effect.
    EffectReplay {
        snapshot: usize,
        instruction: Option<usize>,
    },
    /// An `After` snapshot would make the interpreter skip an effect that has
    /// not committed yet, so a guest operation would be lost.
    EffectSkipped {
        snapshot: usize,
        instruction: Option<usize>,
    },
    /// A guard's snapshot describes another guest location than the guard.
    GuardLocation { instruction: usize },
    /// A snapshot maps a register the trace writes to `Unchanged`.
    UnchangedWritten {
        snapshot: usize,
        frame: FrameId,
        register: usize,
    },
    /// A loop-back's arguments do not match the loop parameters.
    LoopArity { expected: usize, actual: usize },
    /// A loop-back argument has the wrong type.
    LoopArgumentType { index: usize },
    /// A borrowed value is used after an operation that ends borrows (see
    /// [`TraceOp::ends_borrows`]), by an operand, a guard's snapshot, the
    /// poll or the terminator. `ender` counts flat instructions (preheader,
    /// then body).
    BorrowOutlivesEffect { value: ValueId, ender: usize },
}

impl TraceVerifyError {
    /// Whether the error is a side whose window does not extend its tree's
    /// (see [`verify_side_fit`]): a side recorded against a tree that another
    /// side grew since, in another direction.
    #[must_use]
    pub fn is_side_window(&self) -> bool {
        matches!(self, Self::InvalidEntry { detail, .. } if *detail == SIDE_WINDOW_DETAIL)
    }
}

impl fmt::Display for TraceVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRootFrame => write!(f, "trace has no root frame"),
            Self::UnknownValue(value) => write!(f, "unknown value v{}", value.0),
            Self::ValueTableMismatch(value) => {
                write!(f, "value table disagrees about v{}", value.0)
            }
            Self::UseBeforeDefinition { instruction, value } => {
                write!(
                    f,
                    "instruction {instruction} uses v{} before its definition",
                    value.0
                )
            }
            Self::TypeMismatch {
                instruction,
                detail,
            } => {
                write!(f, "instruction {instruction}: {detail}")
            }
            Self::InvalidSite { instruction } => {
                write!(f, "instruction {instruction} has an invalid site")
            }
            Self::InvalidEntry { index, detail } => write!(f, "entry input {index}: {detail}"),
            Self::InvalidLoopParam { index, detail } => {
                write!(f, "loop parameter {index}: {detail}")
            }
            Self::InvalidGuard {
                instruction,
                detail,
            } => {
                write!(f, "instruction {instruction}: {detail}")
            }
            Self::EffectOrder {
                instruction,
                detail,
            } => match instruction {
                Some(instruction) => write!(f, "instruction {instruction}: {detail}"),
                None => write!(f, "{detail}"),
            },
            Self::InvalidSnapshot { snapshot, detail } => {
                write!(f, "snapshot {snapshot}: {detail}")
            }
            Self::SnapshotValueUndefined { snapshot, value } => {
                write!(f, "snapshot {snapshot} uses undefined v{}", value.0)
            }
            Self::FrameDepth { snapshot, depth } => {
                write!(f, "snapshot {snapshot} has {depth} frames, above the bound")
            }
            Self::EffectReplay {
                snapshot,
                instruction,
            } => match instruction {
                Some(instruction) => write!(
                    f,
                    "instruction {instruction}: snapshot {snapshot} resumes before an instruction whose effect already committed"
                ),
                None => write!(
                    f,
                    "exit snapshot {snapshot} resumes before an instruction whose effect already committed"
                ),
            },
            Self::EffectSkipped {
                snapshot,
                instruction,
            } => match instruction {
                Some(instruction) => write!(
                    f,
                    "instruction {instruction}: snapshot {snapshot} resumes after an instruction whose effect has not committed"
                ),
                None => write!(
                    f,
                    "exit snapshot {snapshot} resumes after an instruction whose effect has not committed"
                ),
            },
            Self::GuardLocation { instruction } => write!(
                f,
                "instruction {instruction}: guard snapshot is for another guest location"
            ),
            Self::UnchangedWritten {
                snapshot,
                frame,
                register,
            } => write!(
                f,
                "snapshot {snapshot} maps written register r{register} of frame {} to Unchanged",
                frame.0
            ),
            Self::LoopArity { expected, actual } => {
                write!(
                    f,
                    "loop-back passes {actual} arguments for {expected} parameters"
                )
            }
            Self::LoopArgumentType { index } => {
                write!(f, "loop-back argument {index} has the wrong type")
            }
            Self::BorrowOutlivesEffect { value, ender } => write!(
                f,
                "borrowed v{} is used after instruction {ender}, which ends borrows",
                value.0
            ),
        }
    }
}

impl std::error::Error for TraceVerifyError {}

struct Defined(Vec<bool>);

impl Defined {
    fn check(&self, ir: &TraceIr, value: ValueId) -> Result<(), TraceVerifyError> {
        match self.0.get(value.index()) {
            None => Err(TraceVerifyError::UnknownValue(value)),
            Some(true) => {
                debug_assert!(value.index() < ir.values.len());
                Ok(())
            }
            Some(false) => Err(TraceVerifyError::UseBeforeDefinition {
                instruction: usize::MAX,
                value,
            }),
        }
    }
}

/// Verifies `ir`.
///
/// # Errors
///
/// Returns the first violated invariant.
///
/// A borrowed string ([`TraceType::StringRef`]) may be an entry input here,
/// because a side trace takes one: its parent hands it the address of the slot
/// it holds in the state at the guard the side continues. A root has no
/// parent, so the compiler refuses a root trace with one.
pub fn verify_trace(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    verify_with(ir, true)
}

/// Verifies `ir` after the optimizer rewrote it: every invariant of
/// [`verify_trace`] except the two that read guest steps off adjacency, a
/// guard's `Before` snapshot not replaying a committed effect and an `After`
/// snapshot not skipping a pending one.
///
/// Those two find a guest step as the run of neighbours at one site. That is
/// exact for a recorded trace, where the instructions of another pc (a
/// `forloop` check, a `Move`) separate two executions of one guest
/// instruction. A pass may remove every one of them: constant folding, value
/// numbering or hoisting take the constants and the loop's own checks out of an
/// unrolled inner loop, and two calls of `math.random` at one pc then touch
/// and read as one step. No pass changes a snapshot's pc, and the check that a
/// snapshot's effect position equals the effects committed before its guard
/// still holds (it is not skipped here), so a pass that moved a guard across an
/// effect is still caught. The recorded trace passed the adjacency checks.
///
/// # Errors
///
/// Returns the first violated invariant.
pub fn verify_optimized_trace(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    verify_with(ir, false)
}

fn verify_with(ir: &TraceIr, steps: bool) -> Result<(), TraceVerifyError> {
    let _icount = crate::jit::icount::scope("verify");
    let Some(root) = ir.frames.first() else {
        return Err(TraceVerifyError::NoRootFrame);
    };
    let mut defined = Defined(vec![false; ir.values.len()]);

    for (index, input) in ir.entry.inputs.iter().enumerate() {
        verify_entry_input(ir, &mut defined, index, input, root.register_count)?;
    }
    for (index, input) in ir.frame_inputs.iter().enumerate() {
        verify_frame_input(ir, &mut defined, index, input)?;
    }
    for (index, param) in ir.loop_params.iter().enumerate() {
        let invalid = |detail| TraceVerifyError::InvalidLoopParam { index, detail };
        let info = ir
            .values
            .get(param.value.index())
            .ok_or(TraceVerifyError::UnknownValue(param.value))?;
        if info.def != ValueDef::LoopParam(index) || info.ty != param.ty {
            return Err(TraceVerifyError::ValueTableMismatch(param.value));
        }
        if param.ty == TraceType::Any {
            return Err(invalid("parameter type is Any"));
        }
        if param.ty.is_borrowed() {
            return Err(invalid("a borrowed value cannot cross the loop-back"));
        }
        match param.init {
            LoopParamInit::Input(value) => {
                // A loop side of a loop in a call carries registers of the
                // call's frame, which start as frame inputs.
                let init_ty = ir
                    .entry
                    .inputs
                    .iter()
                    .find(|input| input.value == value)
                    .map(|input| input.ty)
                    .or_else(|| {
                        ir.frame_inputs
                            .iter()
                            .find(|input| input.value == value)
                            .map(|input| input.ty)
                    })
                    .ok_or_else(|| invalid("initial value is not an entry or frame input"))?;
                if init_ty != param.ty {
                    return Err(invalid("initial value has another type"));
                }
            }
            LoopParamInit::Const(scalar) => {
                if scalar.ty() != param.ty {
                    return Err(invalid("initial constant has another type"));
                }
                if scalar.ty().is_opaque() || scalar.ty().is_borrowed() {
                    return Err(invalid("an opaque value cannot be an initial constant"));
                }
            }
        }
        define(&mut defined, param.value);
    }

    // Counted once: every snapshot is held to it.
    let effect_count = ir.effect_count();
    for (snapshot_index, snapshot) in ir.snapshots.iter().enumerate() {
        let previous = snapshot_index
            .checked_sub(1)
            .and_then(|index| ir.snapshots.get(index));
        verify_snapshot_shape(ir, snapshot_index, snapshot, previous, effect_count)?;
    }

    for &(frame, register) in &ir.written_registers {
        let info = ir
            .frames
            .get(frame.index())
            .ok_or(TraceVerifyError::InvalidSite { instruction: 0 })?;
        if register >= info.register_count {
            return Err(TraceVerifyError::InvalidEntry {
                index: usize::from(register),
                detail: "written register is outside its frame",
            });
        }
    }
    verify_frames(ir)?;
    verify_external(ir, root)?;
    verify_cell_aliases(ir, root)?;
    verify_bindings(ir)?;
    verify_fields(ir)?;
    verify_sides(ir)?;
    verify_function(ir)?;
    if (ir.moves_hash_storage_outside_vm_calls() || ir.deletes_keys())
        && let Some(detail) = ir.storage_a_set_would_move()
    {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    if let Some(detail) = ir.stale_metatable_fact() {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    if let Some(detail) = ir.resume_conflict() {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    let entry_state = entry_state_snapshots(ir)?;
    // Sorted once: every register of every snapshot looks itself up.
    let mut written = ir.written_registers.clone();
    written.sort_unstable();
    for snapshot in &ir.snapshots {
        if !entry_state[snapshot.id.index()] {
            verify_unchanged_entries(ir, &written, snapshot)?;
        }
    }

    let mut guards = 0_usize;
    let mut effects = 0_u32;
    // The snapshot whose values were checked last. Values are only ever
    // added to `defined`, so what held for it holds for the ones after.
    let mut checked: Option<&Snapshot> = None;
    let parts = ir
        .preheader
        .iter()
        .map(|instruction| (true, instruction))
        .chain(ir.body.iter().map(|instruction| (false, instruction)));
    for (flat, (in_preheader, instruction)) in parts.enumerate() {
        // `index` counts within the instruction's own part.
        let index = if in_preheader {
            flat
        } else {
            flat - ir.preheader.len()
        };
        // The body prefix and suffix the effect checks look at. The
        // preheader runs before the body, so it has neither.
        let (before_end, after_start) = if in_preheader {
            (0, 0)
        } else {
            (index, index + 1)
        };
        if in_preheader && matches!(instruction.op, TraceOp::PhantomTable { .. }) {
            return Err(TraceVerifyError::EffectOrder {
                instruction: Some(index),
                detail: "the preheader runs once, so it holds no phantom table charge",
            });
        }
        // A call that makes or fills a table makes a new one each time.
        if in_preheader
            && matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. }
                    if helper.effects().contains(HelperEffects::FRESH_TABLE)
            )
        {
            return Err(TraceVerifyError::EffectOrder {
                instruction: Some(index),
                detail: "the preheader runs once, so it makes no table",
            });
        }
        // The preheader may hold reads of table slots the tree never writes
        // (the optimizer proves that), but no store and no symbolic heap op.
        if in_preheader
            && instruction.op.touches_heap()
            && !matches!(
                instruction.op,
                TraceOp::ArrayLoad { .. }
                    | TraceOp::LaneArrayLoad { .. }
                    | TraceOp::FieldLoad { .. }
                    | TraceOp::TableFieldLoad { .. }
                    | TraceOp::MetaGuard { .. }
                    | TraceOp::TableLength { .. }
                    | TraceOp::HostCall { .. }
            )
            // A helper that does not write is a call like a host call.
            && !matches!(
                &instruction.op,
                TraceOp::CallHelper { helper, .. }
                    if !helper.effects().commits_effect()
            )
        {
            return Err(TraceVerifyError::EffectOrder {
                instruction: Some(index),
                detail: "the preheader holds no stores or effects",
            });
        }
        verify_site(ir, index, instruction.site)?;
        if !in_preheader
            && matches!(instruction.op, TraceOp::ArrayOverwrite { .. })
            && !overwrite_proven(ir, index)
        {
            return Err(TraceVerifyError::EffectOrder {
                instruction: Some(index),
                detail: "an unguarded array overwrite has no earlier access of its slot",
            });
        }
        let mut missing = None;
        instruction.op.operands(|value| {
            if missing.is_none()
                && let Err(error) = defined.check(ir, value)
            {
                missing = Some(match error {
                    TraceVerifyError::UseBeforeDefinition { value, .. } => {
                        TraceVerifyError::UseBeforeDefinition {
                            instruction: index,
                            value,
                        }
                    }
                    other => other,
                });
            }
        });
        if let Some(error) = missing {
            return Err(error);
        }

        let result_type = operation_result_type(ir, index, &instruction.op)?;
        match (instruction.result, result_type) {
            (None, None) => {}
            (Some(value), Some(ty)) => {
                let info = ir
                    .values
                    .get(value.index())
                    .ok_or(TraceVerifyError::UnknownValue(value))?;
                let expected = if in_preheader {
                    ValueDef::Preheader(index)
                } else {
                    ValueDef::Instruction(index)
                };
                if info.def != expected || info.ty != ty {
                    return Err(TraceVerifyError::ValueTableMismatch(value));
                }
                if defined.0[value.index()] {
                    return Err(TraceVerifyError::ValueTableMismatch(value));
                }
            }
            _ => {
                return Err(TraceVerifyError::TypeMismatch {
                    instruction: index,
                    detail: "result presence does not match the operation",
                });
            }
        }

        if let Some(guard) = instruction.op.guard() {
            let invalid = |detail| TraceVerifyError::InvalidGuard {
                instruction: index,
                detail,
            };
            if guard.index() != guards {
                return Err(invalid("guard ids must follow body order"));
            }
            guards += 1;
            let info = ir
                .guards
                .get(guard.index())
                .ok_or_else(|| invalid("guard has no table entry"))?;
            if info.site != instruction.site {
                return Err(invalid("guard table site differs from the instruction"));
            }
            let snapshot = ir
                .snapshots
                .get(info.snapshot.index())
                .ok_or_else(|| invalid("guard names a missing snapshot"))?;
            verify_snapshot_values(ir, &defined, snapshot, checked)?;
            checked = Some(snapshot);
            let innermost = snapshot.frames[snapshot.frames.len() - 1].frame;
            // A guard inside an inlined call may leave before the call
            // instruction, in the frame that made the outermost call around
            // it, so the interpreter makes the call itself. Whether the call
            // may be made twice is the effect check's business. Inside a call
            // that has to start over, it is the only way out.
            let (call_site, must_restart) = outer_call(ir, info.site.frame);
            let at_call = call_site.is_some_and(|(caller, call_pc)| {
                innermost == caller
                    && matches!(snapshot.phase, SnapshotPhase::Before { pc } if pc == call_pc)
            });
            let located = at_call
                || (!must_restart
                    && innermost == info.site.frame
                    && snapshot.phase_pc() == info.site.pc);
            if !located {
                return Err(TraceVerifyError::GuardLocation { instruction: index });
            }
            if in_preheader && !entry_state[snapshot.id.index()] {
                return Err(invalid("preheader guard needs an entry-state snapshot"));
            }
            if steps && replays_committed_effect(ir, before_end, snapshot) {
                return Err(TraceVerifyError::EffectReplay {
                    snapshot: snapshot.id.index(),
                    instruction: Some(index),
                });
            }
            if steps && skips_pending_effect(ir, after_start, snapshot) {
                return Err(TraceVerifyError::EffectSkipped {
                    snapshot: snapshot.id.index(),
                    instruction: Some(index),
                });
            }
            if snapshot.effect_position != effects {
                return Err(TraceVerifyError::EffectOrder {
                    instruction: Some(index),
                    detail: "snapshot effect position differs from the committed effects",
                });
            }
        }
        if let Some(effect) = instruction.op.committed_effect() {
            if effect.0 != effects {
                return Err(TraceVerifyError::EffectOrder {
                    instruction: Some(index),
                    detail: "effects must be numbered in body order",
                });
            }
            effects += 1;
        }
        if let Some(value) = instruction.result {
            define(&mut defined, value);
        }
    }
    verify_borrows(ir)?;
    // Every snapshot gets an exit stub. Value checks run for all of them, and
    // each must be named by a guard or the terminator so that the effect and
    // location checks above have covered it.
    // The snapshots a guard names, which the loop above checked already
    // with only the values defined before their guard.
    let mut guarded = vec![false; ir.snapshots.len()];
    for info in &ir.guards {
        if let Some(slot) = guarded.get_mut(info.snapshot.index()) {
            *slot = true;
        }
    }
    for snapshot in &ir.snapshots {
        let by_guard = guarded.get(snapshot.id.index()).copied().unwrap_or(false);
        if !by_guard {
            verify_snapshot_values(ir, &defined, snapshot, checked)?;
            checked = Some(snapshot);
        }
        let referenced = by_guard
            || ir.poll_snapshot == Some(snapshot.id)
            || matches!(&ir.terminator, TraceTerminator::Exit { snapshot: id } | TraceTerminator::Link { snapshot: id } | TraceTerminator::LinkLoop { snapshot: id, .. } if *id == snapshot.id);
        if !referenced {
            return Err(TraceVerifyError::InvalidSnapshot {
                snapshot: snapshot.id.index(),
                detail: "snapshot is not referenced by a guard or the terminator",
            });
        }
    }
    if guards != ir.guards.len() {
        return Err(TraceVerifyError::InvalidGuard {
            instruction: ir.body.len(),
            detail: "guard table has entries no instruction uses",
        });
    }

    if let Some(poll) = ir.poll_snapshot {
        verify_poll_snapshot(ir, poll, effects)?;
    }

    if ir.function.is_some()
        != matches!(
            ir.terminator,
            TraceTerminator::Return { .. } | TraceTerminator::TailCall { .. }
        )
    {
        return Err(TraceVerifyError::InvalidEntry {
            index: 0,
            detail: "a function trace, and only one, ends at a return",
        });
    }
    match &ir.terminator {
        TraceTerminator::TailCall { args, pc } => {
            let function = ir.function.as_ref().expect("checked above");
            let mismatch = |detail| TraceVerifyError::TypeMismatch {
                instruction: ir.body.len(),
                detail,
            };
            if args.len() != function.parameters.len() {
                return Err(mismatch("a tail call passes another number of parameters"));
            }
            for (arg, ty) in args.iter().zip(&function.parameters) {
                defined.check(ir, *arg).map_err(|error| match error {
                    TraceVerifyError::UseBeforeDefinition { value, .. } => {
                        TraceVerifyError::UseBeforeDefinition {
                            instruction: ir.body.len(),
                            value,
                        }
                    }
                    other => other,
                })?;
                if ir.values[arg.index()].ty != *ty {
                    return Err(mismatch(
                        "a tail call passes a parameter of another type than the entry's",
                    ));
                }
            }
            if *pc >= ir.frames[0].code_len || !ir.external.contains(&ExternalSlot::TailCalls) {
                return Err(TraceVerifyError::InvalidEntry {
                    index: 0,
                    detail: "a tail call of a function trace counts in a tail call slot",
                });
            }
        }
        TraceTerminator::Return { value } => {
            let function = ir.function.as_ref().expect("checked above");
            let ty = match value {
                Some(value) => {
                    defined.check(ir, *value).map_err(|error| match error {
                        TraceVerifyError::UseBeforeDefinition { value, .. } => {
                            TraceVerifyError::UseBeforeDefinition {
                                instruction: ir.body.len(),
                                value,
                            }
                        }
                        other => other,
                    })?;
                    Some(ir.values[value.index()].ty)
                }
                None => None,
            };
            if ty != function.result {
                return Err(TraceVerifyError::TypeMismatch {
                    instruction: ir.body.len(),
                    detail: "a return gives another type than the function's",
                });
            }
            if ty.is_some_and(|ty| ty.is_borrowed() || ty == TraceType::Any) {
                return Err(TraceVerifyError::TypeMismatch {
                    instruction: ir.body.len(),
                    detail: "a function returns no borrowed or untyped value",
                });
            }
        }
        TraceTerminator::LoopBack { args } => {
            if args.len() != ir.loop_params.len() {
                return Err(TraceVerifyError::LoopArity {
                    expected: ir.loop_params.len(),
                    actual: args.len(),
                });
            }
            for (index, (arg, param)) in args.iter().zip(&ir.loop_params).enumerate() {
                defined.check(ir, *arg).map_err(|error| match error {
                    TraceVerifyError::UseBeforeDefinition { value, .. } => {
                        TraceVerifyError::UseBeforeDefinition {
                            instruction: ir.body.len(),
                            value,
                        }
                    }
                    other => other,
                })?;
                if ir.values[arg.index()].ty != param.ty {
                    return Err(TraceVerifyError::LoopArgumentType { index });
                }
            }
        }
        TraceTerminator::Exit { snapshot }
        | TraceTerminator::Link { snapshot }
        | TraceTerminator::LinkLoop { snapshot, .. } => {
            let linking = matches!(
                ir.terminator,
                TraceTerminator::Link { .. } | TraceTerminator::LinkLoop { .. }
            );
            let snapshot =
                ir.snapshots
                    .get(snapshot.index())
                    .ok_or(TraceVerifyError::InvalidSnapshot {
                        snapshot: snapshot.index(),
                        detail: "terminator names a missing snapshot",
                    })?;
            if linking {
                verify_link_snapshot(ir, snapshot)?;
            }
            verify_snapshot_values(ir, &defined, snapshot, checked)?;
            if skips_pending_effect(ir, ir.body.len(), snapshot) {
                return Err(TraceVerifyError::EffectSkipped {
                    snapshot: snapshot.id.index(),
                    instruction: None,
                });
            }
            if replays_committed_effect(ir, ir.body.len(), snapshot) {
                return Err(TraceVerifyError::EffectReplay {
                    snapshot: snapshot.id.index(),
                    instruction: None,
                });
            }
            if snapshot.effect_position != effects {
                return Err(TraceVerifyError::EffectOrder {
                    instruction: None,
                    detail: "exit snapshot must record every committed effect",
                });
            }
        }
    }
    Ok(())
}

/// Whether `side` may start from `snapshot` of `parent`: the snapshot has the
/// root frame and exactly the inlined frames the side seeds, and those are
/// the same calls (closure, call instruction, result window) in the same
/// nesting. A side that seeds nothing starts in the root frame.
fn seeded_from(parent: &TraceIr, snapshot: &Snapshot, side: &TraceIr) -> bool {
    let seeded = usize::from(side.seed_frames);
    if snapshot.frames.len() != 1 + seeded || side.frames.len() <= seeded {
        return false;
    }
    snapshot
        .frames
        .iter()
        .enumerate()
        .skip(1)
        .all(|(position, frame)| {
            let (Some(held), Some(own)) = (
                parent.frames.get(frame.frame.index()),
                side.frames.get(position),
            ) else {
                return false;
            };
            let (Some(call), Some(seed)) = (held.inlined, own.inlined) else {
                return false;
            };
            let same_closure = parent
                .pinned_closures
                .get(call.closure as usize)
                .is_some_and(|identity| {
                    side.pinned_closures.get(seed.closure as usize) == Some(identity)
                });
            same_closure
                && held.register_count == own.register_count
                && frame.registers.len() == usize::from(own.register_count)
                && call.call_pc == seed.call_pc
                && call.result_base == seed.result_base
                && call.results == seed.results
                && seed.parent.index() + 1 == position
        })
}

/// The snapshot a side trace links through: the closing instruction
/// committed, resumed at the loop header, with the root frame only. A side
/// has no loop of its own, so it has no poll snapshot either.
fn verify_link_snapshot(ir: &TraceIr, snapshot: &Snapshot) -> Result<(), TraceVerifyError> {
    let invalid = |detail| TraceVerifyError::InvalidSnapshot {
        snapshot: snapshot.id.index(),
        detail,
    };
    if !matches!(snapshot.phase, SnapshotPhase::After { .. }) {
        return Err(invalid(
            "a link snapshot must resume after the closing instruction",
        ));
    }
    if snapshot.frames.len() != 1 {
        return Err(invalid("a link snapshot must describe the root frame only"));
    }
    if ir.poll_snapshot.is_some() || !ir.loop_params.is_empty() {
        return Err(invalid("a linking trace has no loop of its own"));
    }
    if !ir.preheader.is_empty() {
        return Err(invalid("a linking trace has no preheader"));
    }
    Ok(())
}

/// Checks a root trace's side traces one by one, and that they fit the root:
/// they continue a guard of an earlier trace, share its window, and give the
/// loop back values of the types its parameters have.
fn verify_sides(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    for index in 0..ir.sides.len() {
        verify_side_fit(ir, index, true)?;
    }
    Ok(())
}

/// Verifies a tree that a side was just added to (see `grow_tree` in the
/// native policy), without walking the root again.
///
/// Growing a tree changes only its sides and four banks: `pinned_closures`,
/// `cell_aliases`, `root_constants` and `root_slot_count`. This checks
/// everything that reads them besides the root's body: the root's cell
/// aliases and external slots, and the fit of every side, since a grown pin
/// list can stop an earlier side from being a prefix of it. The side
/// at `added` has its own IR verified too.
///
/// It skips the root's body and snapshot walk and the IR of the earlier
/// sides. Those were verified before the tree grew, and what they read of
/// the banks is `index < len` on lists that only grow (a pinned closure lane
/// below `pinned_closures.len()`, a root slot below `root_slot_count`), so
/// the larger banks keep them true. The tree-wide checks that read the sides
/// (`storage_a_set_would_move`, `stale_metatable_fact`, `resume_conflict`)
/// are not monotonic: a new side can add a hash-storage move, a metatable
/// binding or a resuming helper next to the root's lane reads or writes. They
/// run again here.
///
/// # Errors
///
/// Returns the first violated invariant, as [`verify_trace`] would for the
/// grown tree.
pub fn verify_grown_tree(ir: &TraceIr, added: usize) -> Result<(), TraceVerifyError> {
    let Some(root) = ir.frames.first() else {
        return Err(TraceVerifyError::NoRootFrame);
    };
    if added >= ir.sides.len() {
        return Err(TraceVerifyError::InvalidEntry {
            index: added,
            detail: "a side trace is missing",
        });
    }
    verify_external(ir, root)?;
    verify_cell_aliases(ir, root)?;
    // The recorder verified the side it adds when it finished the side, so
    // release builds check its IR only in debug builds.
    for index in 0..ir.sides.len() {
        verify_side_fit(ir, index, cfg!(debug_assertions) && index == added)?;
    }
    // These read the sides too, so a new side can change their answer.
    if (ir.moves_hash_storage_outside_vm_calls() || ir.deletes_keys())
        && let Some(detail) = ir.storage_a_set_would_move()
    {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    if let Some(detail) = ir.stale_metatable_fact() {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    if let Some(detail) = ir.resume_conflict() {
        return Err(TraceVerifyError::InvalidEntry { index: 0, detail });
    }
    Ok(())
}

/// Checks that side `index` of `ir` fits the tree it joins: its window, its
/// parent and the guard it continues (the rules [`verify_sides`] applies).
/// With `check_ir`, also verifies the side's own IR. A caller that knows the
/// side was verified when it was recorded passes `false`.
///
/// # Errors
///
/// Returns the first violated invariant.
pub fn verify_side_fit(ir: &TraceIr, index: usize, check_ir: bool) -> Result<(), TraceVerifyError> {
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    let Some(side) = ir.sides.get(index) else {
        return Err(invalid(index, "a side trace is missing"));
    };
    if check_ir {
        side.ir.verify()?;
    }
    if !side.ir.sides.is_empty() {
        return Err(invalid(index, "a side trace has sides of its own"));
    }
    let ends = match (&ir.function, &side.ir.function) {
        (None, None) => match side.ir.terminator {
            TraceTerminator::Link { .. } | TraceTerminator::Exit { .. } => {
                side.guard != TERMINAL_GUARD
            }
            // A loop of its own, entered where its parent ends. Only a
            // trace that does not count instructions can hold one: a
            // count hook's budget runs through the root's loop.
            TraceTerminator::LoopBack { .. } => {
                side.guard == TERMINAL_GUARD && !side.ir.counted && !ir.counted
            }
            // The loop it rejoins is a loop side the side descends from.
            TraceTerminator::LinkLoop { side: target, .. } => {
                side.guard != TERMINAL_GUARD
                    && ir.sides.get(target).is_some_and(|looping| {
                        looping.guard == TERMINAL_GUARD
                            && matches!(looping.ir.terminator, TraceTerminator::LoopBack { .. })
                    })
                    && std::iter::successors(side.parent, |at| ir.sides[*at].parent)
                        .any(|ancestor| ancestor == target)
            }
            _ => false,
        },
        // A side of a function trace finishes the activation.
        // A root that only tail calls has seen no result yet: the
        // tree's is its sides' (see `plan_tree`).
        (Some(root), Some(function)) => {
            root.parameters == function.parameters
                && (root.result.is_none()
                    || function.result.is_none()
                    || root.result == function.result)
                && side.ir.poll_snapshot.is_none()
        }
        _ => false,
    };
    if !ends {
        return Err(invalid(index, "a side trace neither links nor exits"));
    }
    let parent = match side.parent {
        None => ir,
        Some(parent) if parent < index => &ir.sides[parent].ir,
        Some(_) => return Err(invalid(index, "a side trace precedes its parent")),
    };
    // The window is the root's, which a side may extend with the cells
    // and globals only it names (see `plan_tree`). Its cell aliases are
    // the tree's, which has taken on the ones a side found.
    let guest = |trace: &TraceIr| {
        usize::from(trace.frames[0].register_count).saturating_sub(trace.external.len())
    };
    let shares = guest(&side.ir) == guest(ir)
        && side.ir.frames[0].code_len == ir.frames[0].code_len
        && side.ir.frames[0].inlined.is_none()
        && side.ir.external.starts_with(&ir.external)
        && (side.ir.pinned_closures.starts_with(&ir.pinned_closures)
            || ir.pinned_closures.starts_with(&side.ir.pinned_closures))
        && side
            .ir
            .cell_aliases
            .iter()
            .all(|alias| ir.cell_aliases.contains(alias))
        && ir.root_constants.starts_with(&side.ir.root_constants)
        && side.ir.root_slot_count as usize == side.ir.root_constants.len()
        && side.ir.bindings.starts_with(&ir.bindings)
        && side.ir.fields.starts_with(&ir.fields);
    if !shares {
        if crate::jit::debug_env::trace() {
            let parts = [
                ("registers", guest(&side.ir) == guest(ir)),
                ("code", side.ir.frames[0].code_len == ir.frames[0].code_len),
                ("frame", side.ir.frames[0].inlined.is_none()),
                ("externals", side.ir.external.starts_with(&ir.external)),
                (
                    "pins",
                    side.ir.pinned_closures.starts_with(&ir.pinned_closures)
                        || ir.pinned_closures.starts_with(&side.ir.pinned_closures),
                ),
                (
                    "aliases",
                    side.ir
                        .cell_aliases
                        .iter()
                        .all(|alias| ir.cell_aliases.contains(alias)),
                ),
                (
                    "root strings",
                    ir.root_constants.starts_with(&side.ir.root_constants)
                        && side.ir.root_slot_count as usize == side.ir.root_constants.len(),
                ),
                ("bindings", side.ir.bindings.starts_with(&ir.bindings)),
                ("fields", side.ir.fields.starts_with(&ir.fields)),
            ];
            let differ: Vec<&str> = parts
                .iter()
                .filter(|(_, same)| !same)
                .map(|(name, _)| *name)
                .collect();
            eprintln!(
                "jit-trace: side trace {index} has another window than its root: {differ:?} (pins {} against {}, bindings {} against {}, fields {} against {})",
                side.ir.pinned_closures.len(),
                ir.pinned_closures.len(),
                side.ir.bindings.len(),
                ir.bindings.len(),
                side.ir.fields.len(),
                ir.fields.len(),
            );
        }
        return Err(invalid(index, SIDE_WINDOW_DETAIL));
    }
    // A loop side continues the terminal exit of its parent, which
    // resumes at the loop's header: in the root frame, or in the frames of
    // the calls the side starts inside (see "Loops in calls" in
    // `vm::jit::trace`), which are the calls the side seeds: the same
    // closures, call sites and windows.
    if side.guard == TERMINAL_GUARD {
        let enters = match parent.terminator {
            TraceTerminator::Exit { snapshot } => parent
                .snapshots
                .get(snapshot.index())
                .is_some_and(|snapshot| {
                    (if side.ir.seed_frames == 0 {
                        snapshot.frames.len() == 1
                    } else {
                        seeded_from(parent, snapshot, &side.ir)
                    }) && snapshot.virtuals.is_empty()
                        && matches!(snapshot.phase, SnapshotPhase::After { .. })
                }),
            _ => false,
        };
        let siblings = ir
            .sides
            .iter()
            .enumerate()
            .filter(|(other, held)| {
                held.guard == TERMINAL_GUARD && held.parent == side.parent && *other != index
            })
            .count();
        if !enters || parent.function.is_some() || siblings > 0 {
            return Err(invalid(
                index,
                "a loop side continues no terminal exit of its parent",
            ));
        }
        return Ok(());
    }
    // The guard it continues must be a branch guard of the parent's body
    // that resumes in the root frame, or inside the calls the side seeds.
    let preheader_guards = parent
        .preheader
        .iter()
        .filter(|instruction| instruction.op.guard().is_some())
        .count();
    let continues = parent.guards.iter().skip(preheader_guards).any(|info| {
        info.origin == side.guard
            && seeded_from(parent, &parent.snapshots[info.snapshot.index()], &side.ir)
    });
    if !continues {
        return Err(invalid(
            index,
            "a side trace continues no guard of its parent",
        ));
    }
    Ok(())
}

/// Checks a function-entry trace ([`TraceIr::function`]): it has no loop, no
/// root activation and commits no effect; a root's entry contract holds every
/// parameter with its declared type and nil in the other guest registers it
/// reads (a fresh activation has nothing else there); every self call leaves
/// from the root frame, before its call; and no snapshot keeps a parameter
/// `Unchanged`, since an exit in a nested activation restores into a frame
/// it pushes, whose registers start out nil.
fn verify_function(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    let Some(function) = &ir.function else {
        return Ok(());
    };
    if !ir.loop_params.is_empty() || ir.loop_tail.is_some() {
        return Err(invalid(0, "a function trace has no loop"));
    }
    if ir.frames.iter().any(|info| info.activation.is_some()) {
        return Err(invalid(0, "a function trace has no root activation"));
    }
    if ir.effect_count() > 0 || !ir.preheader.is_empty() {
        return Err(invalid(
            0,
            "a function trace commits no effect and has no preheader",
        ));
    }
    let guest = usize::from(ir.frames[0].register_count).saturating_sub(ir.external.len());
    let parameters = function.parameters.len();
    if parameters > guest {
        return Err(invalid(
            0,
            "a function trace has more parameters than registers",
        ));
    }
    for &register in &function.callee_registers {
        if usize::from(register) < guest || usize::from(register) >= guest + ir.external.len() {
            return Err(invalid(
                usize::from(register),
                "a self call's callee is no cell or global of the window",
            ));
        }
    }
    // A root is entered from the VM: its poll is the entry state.
    if ir.poll_snapshot.is_some() {
        for (register, ty) in function.parameters.iter().enumerate() {
            let held = ir
                .entry
                .inputs
                .iter()
                .find(|input| usize::from(input.register) == register);
            if held.is_none_or(|input| input.ty != *ty) {
                return Err(invalid(
                    register,
                    "a parameter is no entry input of its type",
                ));
            }
        }
        for input in &ir.entry.inputs {
            let register = usize::from(input.register);
            if register >= parameters && register < guest && input.ty != TraceType::Nil {
                return Err(invalid(
                    register,
                    "a register past the parameters is nil at entry",
                ));
            }
        }
    }
    for (index, instruction) in ir.body.iter().enumerate() {
        let TraceOp::SelfCall { guard, .. } = &instruction.op else {
            continue;
        };
        let snapshot = ir
            .guards
            .get(guard.index())
            .and_then(|info| ir.snapshots.get(info.snapshot.index()));
        let fits = instruction.site.frame == FrameId(0)
            && snapshot.is_some_and(|snapshot| {
                snapshot.frames.len() == 1
                    && snapshot.phase
                        == SnapshotPhase::Before {
                            pc: instruction.site.pc,
                        }
            });
        if !fits {
            return Err(TraceVerifyError::InvalidGuard {
                instruction: index,
                detail: "a self call leaves from the root frame, before the call",
            });
        }
    }
    // The activation's tail call count changes with its tail calls.
    let count = ir
        .external
        .iter()
        .position(|slot| *slot == ExternalSlot::TailCalls)
        .map(|position| guest + position);
    for snapshot in &ir.snapshots {
        let kept = snapshot.frames.iter().any(|frame| {
            frame.frame == FrameId(0)
                && (frame
                    .registers
                    .range(0, parameters)
                    .any(|entry| *entry == RegisterEntry::Unchanged)
                    || count.is_some_and(|count| {
                        frame.registers.get(count) == Some(&RegisterEntry::Unchanged)
                    }))
        });
        if kept {
            return Err(TraceVerifyError::InvalidSnapshot {
                snapshot: snapshot.id.index(),
                detail: "a function trace's snapshot keeps a parameter or its tail call count unchanged",
            });
        }
    }
    Ok(())
}

/// The poll snapshot publishes the state at a loop-back: the closing guest
/// instruction committed (it may record no operation, like a plain `Jump`),
/// and the interpreter resumes at the loop header.
fn verify_poll_snapshot(
    ir: &TraceIr,
    poll: super::ir::SnapshotId,
    effects: u32,
) -> Result<(), TraceVerifyError> {
    let invalid = |detail| TraceVerifyError::InvalidSnapshot {
        snapshot: poll.index(),
        detail,
    };
    let snapshot = ir
        .snapshots
        .get(poll.index())
        .ok_or_else(|| invalid("poll names a missing snapshot"))?;
    if ir.body.is_empty() {
        return Err(invalid("a trace without a body cannot be polled"));
    }
    if ir.function.is_some() {
        // A function trace polls on entry to an activation: nothing has run
        // in it, and the interpreter has run nothing in a fresh frame.
        let fresh = snapshot.phase == SnapshotPhase::Before { pc: 0 }
            && snapshot.frames.len() == 1
            && snapshot.frames[0].frame == FrameId(0)
            && snapshot.frames[0].last_interpreted_pc.is_none()
            && snapshot.effect_position == 0;
        if !fresh {
            return Err(invalid(
                "the poll of a function trace is its entry state, before pc 0",
            ));
        }
        return Ok(());
    }
    match (snapshot.phase, ir.loop_tail) {
        (SnapshotPhase::After { .. }, None) => {}
        // A tail call enters the header: nothing ran in the new activation,
        // and the interpreter keeps no previous pc across a tail call.
        (SnapshotPhase::Before { .. }, Some(_))
            if snapshot
                .frames
                .first()
                .is_some_and(|root| root.last_interpreted_pc.is_none()) => {}
        (_, None) => {
            return Err(invalid(
                "poll snapshot must resume after the closing instruction",
            ));
        }
        (_, Some(_)) => {
            return Err(invalid(
                "the poll of a loop through tail calls resumes before the header with no previous pc",
            ));
        }
    }
    // A loop side that starts inside inlined calls loops in the innermost
    // of them (see "Loops in calls" in `vm::jit::trace`): its poll rebuilds
    // them all.
    let frames = 1 + usize::from(ir.seed_frames);
    if snapshot.frames.len() != frames
        || snapshot
            .frames
            .iter()
            .enumerate()
            .any(|(position, frame)| frame.frame.index() != position)
    {
        return Err(invalid("poll snapshot must describe the root frame only"));
    }
    if snapshot.effect_position != effects {
        return Err(TraceVerifyError::EffectOrder {
            instruction: None,
            detail: "poll snapshot must record every committed effect",
        });
    }
    if skips_pending_effect(ir, ir.body.len(), snapshot) {
        return Err(TraceVerifyError::EffectSkipped {
            snapshot: poll.index(),
            instruction: None,
        });
    }
    // The poll publishes the state the next iteration would start from, so a
    // register that enters through a loop parameter must hold that
    // parameter's loop-back argument, not a value from before the final write.
    if let TraceTerminator::LoopBack { args } = &ir.terminator {
        let root = &snapshot.frames[0];
        for (param, arg) in ir.loop_params.iter().zip(args) {
            let LoopParamInit::Input(input) = param.init else {
                continue;
            };
            let Some(entry) = ir.entry.inputs.iter().find(|entry| entry.value == input) else {
                continue;
            };
            if let Some(RegisterEntry::Value(held)) =
                root.registers.get(usize::from(entry.register))
                && held != arg
            {
                return Err(invalid(
                    "poll snapshot register differs from its loop-back argument",
                ));
            }
        }
    }
    Ok(())
}

/// Whether a `Before` snapshot taken after body instruction `end - 1` would
/// resume at a guest instruction that already committed an effect in the
/// trace. The interpreter would execute it again. Only the contiguous run of
/// instructions at the snapshot's own site counts, since those belong to the
/// guest instruction the snapshot resumes at.
fn replays_committed_effect(ir: &TraceIr, end: usize, snapshot: &Snapshot) -> bool {
    let SnapshotPhase::Before { pc } = snapshot.phase else {
        return false;
    };
    let frame = snapshot.frames[snapshot.frames.len() - 1].frame;
    let run: Vec<&TraceInstruction> = ir.body[..end]
        .iter()
        .rev()
        .take_while(|i| belongs_to_step(ir, i, frame, pc))
        .collect();
    // Inlined callee instructions before the step's first own instruction
    // belong to an earlier guest instruction.
    let own = run.iter().rposition(|i| i.site.frame == frame);
    own.is_some_and(|last| {
        run[..=last]
            .iter()
            .any(|i| i.op.committed_effect().is_some())
    })
}

/// Whether an `After` snapshot taken before body instruction `start` would
/// resume past a guest instruction whose effect is still to come. The
/// contiguous run of instructions from `start` at the snapshot's own site
/// belongs to the guest instruction the snapshot calls committed.
fn skips_pending_effect(ir: &TraceIr, start: usize, snapshot: &Snapshot) -> bool {
    let SnapshotPhase::After { pc, .. } = snapshot.phase else {
        return false;
    };
    let frame = snapshot.frames[snapshot.frames.len() - 1].frame;
    ir.body[start.min(ir.body.len())..]
        .iter()
        .take_while(|i| belongs_to_step(ir, i, frame, pc))
        .any(|i| i.op.committed_effect().is_some())
}

/// Whether `instruction` belongs to the guest instruction at (`frame`, `pc`):
/// it sits at that site, or in a frame the call at that site inlined (the
/// callee's instructions run between the caller's own ones). A frame inlined
/// by a call at another pc of `frame` belongs to that call's step.
fn belongs_to_step(
    ir: &TraceIr,
    instruction: &TraceInstruction,
    frame: FrameId,
    pc: usize,
) -> bool {
    if instruction.site.frame == frame {
        return instruction.site.pc == pc;
    }
    let mut current = instruction.site.frame;
    // Frames are listed callee after caller, so the walk is bounded.
    for _ in 0..ir.frames.len() {
        let Some(call) = ir.frames.get(current.index()).and_then(|info| info.inlined) else {
            return false;
        };
        if call.parent == frame {
            return call.call_pc == pc;
        }
        current = call.parent;
    }
    false
}

/// Marks the snapshots only preheader guards use, after checking they
/// describe the entry state: a `Before` snapshot of the root frame whose every
/// register is still in the interpreter, or is the loop parameter that holds
/// it. Nothing has run when a preheader guard fails, so `Unchanged` is right
/// even for a register the trace writes later. A snapshot a body guard, the poll or the terminator also uses is
/// held to the ordinary rules.
fn entry_state_snapshots(ir: &TraceIr) -> Result<Vec<bool>, TraceVerifyError> {
    let mut marked = vec![false; ir.snapshots.len()];
    let preheader_guards = ir
        .preheader
        .iter()
        .filter(|instruction| instruction.op.guard().is_some())
        .count();
    for info in ir.guards.iter().take(preheader_guards) {
        if let Some(slot) = marked.get_mut(info.snapshot.index()) {
            *slot = true;
        }
    }
    let mut unmark = |id: super::ir::SnapshotId| {
        if let Some(slot) = marked.get_mut(id.index()) {
            *slot = false;
        }
    };
    for info in ir.guards.iter().skip(preheader_guards) {
        unmark(info.snapshot);
    }
    if let Some(poll) = ir.poll_snapshot {
        unmark(poll);
    }
    if let TraceTerminator::Exit { snapshot }
    | TraceTerminator::Link { snapshot }
    | TraceTerminator::LinkLoop { snapshot, .. } = &ir.terminator
    {
        unmark(*snapshot);
    }
    // The entry state is the loop header with the latch as the last
    // interpreted instruction, which is what the poll snapshot resumes at.
    // A loop through tail calls enters its header with no previous pc.
    let header = ir
        .poll_snapshot
        .and_then(|poll| ir.snapshots.get(poll.index()))
        .and_then(|poll| match poll.phase {
            SnapshotPhase::After { pc, successor } => Some((Some(pc), successor)),
            SnapshotPhase::Before { pc } if ir.loop_tail.is_some() => Some((None, pc)),
            SnapshotPhase::Before { .. } => None,
        });
    for (index, snapshot) in ir.snapshots.iter().enumerate() {
        if !marked[index] {
            continue;
        }
        let at_header = header.is_some_and(|(latch, successor)| {
            snapshot.phase_pc() == successor
                && snapshot.frames.first().is_some_and(|frame| {
                    frame.frame == FrameId(0) && frame.last_interpreted_pc == latch
                })
        });
        let plain = at_header
            && matches!(snapshot.phase, SnapshotPhase::Before { .. })
            && snapshot.effect_position == 0
            && snapshot.frames.len() == 1
            && snapshot.frames.iter().all(|frame| {
                frame
                    .registers
                    .iter()
                    .enumerate()
                    .all(|(register, entry)| match entry {
                        RegisterEntry::Unchanged => true,
                        // A loop side's entry state is its parent's native
                        // state: it names the first value of a register's
                        // loop parameter, which may be a constant.
                        RegisterEntry::Const(scalar) => ir.loop_params.iter().any(|param| {
                            param.init == LoopParamInit::Const(*scalar)
                                && param
                                    .register
                                    .is_some_and(|held| usize::from(held) == register)
                        }),
                        // A tree's preheader runs again after a link, so it
                        // may name the loop parameter that holds a register,
                        // and a loop side names its entry inputs (a register
                        // nothing writes holds its input at every entry).
                        RegisterEntry::Value(value) => {
                            ir.entry.inputs.iter().any(|input| {
                                input.value == *value && usize::from(input.register) == register
                            }) || ir.loop_params.iter().any(|param| {
                                param.init == LoopParamInit::Input(*value)
                                    && param
                                        .register
                                        .is_some_and(|held| usize::from(held) == register)
                            }) || ir.loop_params.iter().any(|param| {
                                param.value == *value
                                    && param
                                        .register
                                        .is_some_and(|held| usize::from(held) == register)
                            }) || ir.boxed_control.is_some_and(|control| {
                                // The control register and the variables
                                // restored from lanes name an entry input.
                                let control = usize::from(control);
                                let source = if register == control + 1 && ir.boxed_key {
                                    control
                                } else {
                                    register
                                };
                                (source == control || (source == control + 2 && ir.boxed_value))
                                    && ir.entry.inputs.iter().any(|input| {
                                        input.value == *value
                                            && usize::from(input.register) == source
                                    })
                            })
                        }
                        _ => false,
                    })
            });
        if !plain {
            return Err(TraceVerifyError::InvalidSnapshot {
                snapshot: index,
                detail: "an entry-state snapshot must resume at the loop header from the root frame with unchanged registers",
            });
        }
    }
    Ok(marked)
}

/// Rejects `Unchanged` for any register the trace writes. The interpreter
/// register holds the entry value, which a write anywhere in the trace may
/// have made stale.
/// `written` is [`TraceIr::written_registers`], sorted.
fn verify_unchanged_entries(
    ir: &TraceIr,
    written: &[(FrameId, u16)],
    snapshot: &Snapshot,
) -> Result<(), TraceVerifyError> {
    let guest = usize::from(ir.frames[0].register_count).saturating_sub(ir.external.len());
    for frame in &snapshot.frames {
        // A later activation of the root frame shares the root's window: the
        // trace writes it under the root frame's id.
        let activation = ir
            .frames
            .get(frame.frame.index())
            .is_some_and(|info| info.activation.is_some());
        let window = if activation { FrameId(0) } else { frame.frame };
        // The registers the trace writes in this window, in register order.
        // The registers of a snapshot are walked in the same order, so the
        // walk keeps one position in the list instead of searching it for
        // each of the many `Unchanged` entries.
        let first = written.partition_point(|written| *written < (window, 0));
        let in_window = written[first..]
            .iter()
            .take_while(|(frame, _)| *frame == window);
        let mut in_window = in_window.map(|(_, register)| *register).peekable();
        let untyped = |register16: u16| {
            window == FrameId(0)
                && ir
                    .entry
                    .inputs
                    .iter()
                    .any(|input| input.register == register16 && input.ty == TraceType::Any)
        };
        if frame.frame.index() == 0 && !activation {
            // The root frame's own registers: the only fault is a written
            // one kept `Unchanged`, so only the written registers are
            // looked at, not every register of every snapshot.
            for register16 in in_window {
                let register = usize::from(register16);
                if matches!(
                    frame.registers.get(register),
                    Some(RegisterEntry::Unchanged)
                ) && !untyped(register16)
                {
                    return Err(TraceVerifyError::UnchangedWritten {
                        snapshot: snapshot.id.index(),
                        frame: frame.frame,
                        register,
                    });
                }
            }
            continue;
        }
        for (register, entry) in frame.registers.iter().enumerate() {
            if !matches!(entry, RegisterEntry::Unchanged) {
                continue;
            }
            if frame.frame.index() != 0 && !activation {
                // The frame that called the root frame is the interpreter's
                // own: a register the trace never wrote is as it holds it.
                if ir
                    .frames
                    .get(frame.frame.index())
                    .and_then(|info| info.inlined)
                    .is_some_and(|call| matches!(call.entry, super::ir::Entry::Caller { .. }))
                {
                    continue;
                }
                // Inlined frames do not exist in the interpreter during a
                // trace, so there is no register to keep.
                return Err(TraceVerifyError::InvalidSnapshot {
                    snapshot: snapshot.id.index(),
                    detail: "Unchanged is only meaningful in the root frame",
                });
            }
            if activation && register < guest {
                // The activation's registers are a fresh call's, not the
                // ones the interpreter held at entry.
                return Err(TraceVerifyError::InvalidSnapshot {
                    snapshot: snapshot.id.index(),
                    detail: "Unchanged names a register of a replaced root activation",
                });
            }
            let Ok(register16) = u16::try_from(register) else {
                continue;
            };
            while in_window.next_if(|written| *written < register16).is_some() {}
            // An entry input of type `Any` that a side writes (see
            // `Recorder::with_lazy`) is the run's until the write: the frame
            // holds it, with what a link stored in its slot written back when
            // the run leaves. The recorder names the written value from the
            // write on. The inputs are scanned only for a written register.
            if in_window.peek() == Some(&register16) && !untyped(register16) {
                return Err(TraceVerifyError::UnchangedWritten {
                    snapshot: snapshot.id.index(),
                    frame: frame.frame,
                    register,
                });
            }
        }
    }
    Ok(())
}

/// Checks the logical frames: the root is a real frame and every other one is
/// a call of a pinned closure made by an earlier frame.
fn verify_frames(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    let lane_fits =
        |lane: u32| usize::try_from(lane).is_ok_and(|lane| lane < ir.pinned_closures.len());
    if let Some(site) = ir.loop_tail
        && site.closure.is_some_and(|lane| !lane_fits(lane))
    {
        return Err(invalid(0, "the loop's tail call names an unpinned closure"));
    }
    for (index, info) in ir.frames.iter().enumerate() {
        if info.inlined.is_some() && info.activation.is_some() {
            return Err(invalid(index, "an inlined frame is no root activation"));
        }
        match (index, info.inlined) {
            (0, None) if info.activation.is_some() => {
                return Err(invalid(0, "the root frame is its own first activation"));
            }
            (0, None) => {}
            (0, Some(_)) => return Err(invalid(0, "the root frame is not an inlined call")),
            (_, None) => {
                // A later activation of the root frame: a tail call in it
                // replaced its closure. Only a loop through tail calls has one.
                let Some(activation) = info.activation else {
                    return Err(invalid(index, "a frame after the root is not inlined"));
                };
                if ir.loop_tail.is_none() {
                    return Err(invalid(
                        index,
                        "a root activation in a trace that does not loop through tail calls",
                    ));
                }
                if !lane_fits(activation.closure)
                    || activation.site.closure.is_some_and(|lane| !lane_fits(lane))
                {
                    return Err(invalid(
                        index,
                        "a root activation names an unpinned closure",
                    ));
                }
                if info.register_count != ir.frames[0].register_count {
                    return Err(invalid(index, "a root activation has another window"));
                }
            }
            (_, Some(call)) => {
                // The caller of the root frame runs no closure of the trace's.
                let caller = matches!(call.entry, super::ir::Entry::Caller { .. });
                if !caller
                    && usize::try_from(call.closure)
                        .map_or(true, |closure| closure >= ir.pinned_closures.len())
                {
                    return Err(invalid(index, "an inlined frame runs an unpinned closure"));
                }
                let Some(parent) = ir.frames.get(call.parent.index()) else {
                    return Err(invalid(index, "an inlined frame has no parent"));
                };
                if call.parent.index() >= index {
                    return Err(invalid(index, "a frame is created before its parent"));
                }
                if call.call_pc >= parent.code_len {
                    return Err(invalid(index, "an inlined call pc is outside its parent"));
                }
                // A frame that runs whichever closure of a prototype the call
                // had is a plain call: no region starts over, no tail call
                // renamed it, no protected call or metamethod made it. The
                // register after the chunk's holds its closure.
                if call.dynamic
                    && (call.reexec
                        || call.tail_calls != 0
                        || call.tail_site.is_some()
                        || call.protected != Protected::No
                        || call.entry != super::ir::Entry::Call
                        || info.register_count == 0)
                {
                    return Err(invalid(
                        index,
                        "a frame that runs the closure of its call is more than a plain call",
                    ));
                }
                if let Protected::Xpcall { handler } = call.protected
                    && !lane_fits(handler)
                {
                    return Err(invalid(index, "a protected call names an unpinned handler"));
                }
                if usize::from(call.result_base) + usize::from(call.results)
                    > usize::from(parent.register_count)
                {
                    return Err(invalid(index, "an inlined call writes past its parent"));
                }
                let results = match call.entry {
                    super::ir::Entry::Call | super::ir::Entry::ToString => call.results,
                    super::ir::Entry::Index
                    | super::ir::Entry::Len
                    | super::ir::Entry::Operator
                    | super::ir::Entry::Compare { .. }
                    | super::ir::Entry::Branch { .. }
                    | super::ir::Entry::Concat { .. } => 1,
                    super::ir::Entry::NewIndex => 0,
                    // The caller's result window is not the parent's.
                    super::ir::Entry::Caller { .. } => 0,
                };
                if call.results != results {
                    return Err(invalid(
                        index,
                        "a metamethod frame keeps the wrong number of results",
                    ));
                }
                if call.entry != super::ir::Entry::Call && call.protected != Protected::No {
                    return Err(invalid(index, "a metamethod frame is a protected call"));
                }
            }
        }
    }
    Ok(())
}

/// Checks that every cell alias names real things: pinned closures, guest
/// registers of the root frame, and a representative that is a window slot or
/// a register, not another alias.
///
/// The window slots are the tree's: the root's, and those a side extends it
/// with (see "The window grows" in `tree`), which `plan_tree` gives every
/// member. A side that inlines a callee whose cell is a cell only the side
/// names in its window (a `ClosureUpvalue` slot of a closure the root never
/// called) lists the alias, and the tree takes it on when the side joins: the
/// alias stands for that slot of the grown window. A name that has a slot in
/// any member is no alias, since that member's register and the alias's
/// register would hold one variable twice.
fn verify_cell_aliases(
    ir: &TraceIr,
    root: &super::ir::TraceFrameInfo,
) -> Result<(), TraceVerifyError> {
    use super::ir::CellSource;
    let invalid = |detail| TraceVerifyError::InvalidEntry { index: 0, detail };
    let guest = usize::from(root.register_count).saturating_sub(ir.external.len());
    let fits = |source: CellSource| match source {
        CellSource::Upvalue(_) => true,
        CellSource::ClosureUpvalue { closure, .. } => {
            usize::try_from(closure).is_ok_and(|closure| closure < ir.pinned_closures.len())
        }
        CellSource::Captured(register) => usize::from(register) < guest,
    };
    let named = |slot: ExternalSlot| {
        ir.external.contains(&slot) || ir.sides.iter().any(|side| side.ir.external.contains(&slot))
    };
    let slot_of = |source: CellSource| match source {
        CellSource::Upvalue(index) => named(ExternalSlot::Upvalue(index)),
        CellSource::ClosureUpvalue { closure, index } => {
            named(ExternalSlot::ClosureUpvalue { closure, index })
        }
        CellSource::Captured(_) => true,
    };
    for (position, alias) in ir.cell_aliases.iter().enumerate() {
        if !fits(alias.cell) || !fits(alias.is) {
            return Err(invalid("a cell alias names something outside the trace"));
        }
        if alias.cell == alias.is || matches!(alias.cell, CellSource::Captured(_)) {
            return Err(invalid("a cell alias renames a register or itself"));
        }
        if !slot_of(alias.is) {
            return Err(invalid(
                "a cell alias stands for a name with no window slot",
            ));
        }
        if ir.cell_aliases[..position]
            .iter()
            .any(|earlier| earlier.cell == alias.cell)
        {
            return Err(invalid("a cell alias repeats a name"));
        }
        if slot_of(alias.cell) {
            return Err(invalid(
                "a cell alias renames a name that has a window slot",
            ));
        }
    }
    Ok(())
}

/// What the effects of `ir`'s body mean for a global slot (see
/// [`verify_external`]): whether it holds an unknown effect, and how many
/// committed effects could rebind a scalar global. `first` is the window
/// register of the first external slot.
fn global_slot_effects(ir: &TraceIr, first: usize) -> (bool, usize) {
    let unknown_effect = ir
        .body
        .iter()
        .any(|instruction| matches!(instruction.op, TraceOp::Effect { .. }));
    // A store into a field of the environment table is how the trace
    // writes the globals it stores. The recorder reads every global
    // it stores through the same field, never through a slot, so
    // such a store cannot change a global that a slot holds.
    let environment = ir
        .external
        .iter()
        .position(|slot| *slot == ExternalSlot::Environment)
        .map(|position| u16::try_from(first + position).expect("window fits u16"));
    let foreign_effects = ir
        .body
        .iter()
        .filter(|instruction| match &instruction.op {
            TraceOp::FieldStore { binding, .. } => ir
                .bindings
                .get(*binding as usize)
                .is_none_or(|bound| Some(bound.register) != environment),
            // A read or write of a closure's cell through the live
            // closure (a frame guarded on the prototype) is ordered
            // like an effect, but it touches a cell, never a field
            // of an environment table.
            TraceOp::CallHelper { helper, .. } if helper.is_cell_access() => false,
            // A host function call may change the global, which the
            // run compares with memory afterwards
            // (`RunTables::externals_hold`) and leaves if it did.
            TraceOp::CallHelper { helper, .. } if helper.is_vm_call() => false,
            op => op.committed_effect().is_some(),
        })
        .count();
    (unknown_effect, foreign_effects)
}

/// Checks the window slots beyond the root frame's registers: they fit the
/// window, name distinct things, and a global is never written and never
/// read as nil (a missing global would run `__index`).
fn verify_external(ir: &TraceIr, root: &super::ir::TraceFrameInfo) -> Result<(), TraceVerifyError> {
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    if ir.external.len() > usize::from(root.register_count) {
        return Err(invalid(0, "more external slots than window registers"));
    }
    let first = usize::from(root.register_count) - ir.external.len();
    // Looked up once per slot below; a trace can hold hundreds of slots, and
    // a search of the slots, inputs, writes or body per slot made this
    // check quadratic.
    let mut seen = std::collections::HashSet::with_capacity(ir.external.len());
    let window_slots = ir.external.len();
    let slot_of = |register: u16| usize::from(register).checked_sub(first);
    let mut written = vec![false; window_slots];
    for &(frame, register) in &ir.written_registers {
        if frame == FrameId(0)
            && let Some(slot) = slot_of(register).and_then(|slot| written.get_mut(slot))
        {
            *slot = true;
        }
    }
    // Per slot: the type of its first entry input, and whether any input
    // reads it as nil or as a scalar.
    let mut inputs: Vec<(Option<TraceType>, bool, bool)> = vec![(None, false, false); window_slots];
    for input in &ir.entry.inputs {
        if let Some(entry) = slot_of(input.register).and_then(|slot| inputs.get_mut(slot)) {
            entry.0.get_or_insert(input.ty);
            entry.1 |= input.ty == TraceType::Nil;
            entry.2 |= !input.ty.is_opaque();
        }
    }
    let mut global_effects: Option<(bool, usize)> = None;
    for (position, slot) in ir.external.iter().enumerate() {
        if !seen.insert(*slot) {
            return Err(invalid(first + position, "external slot appears twice"));
        }
        let (input_type, reads_nil, reads_scalar) = inputs[position];
        let written = written[position];
        let register = u16::try_from(first + position).expect("window fits u16");
        if let ExternalSlot::ClosureUpvalue { closure, .. }
        | ExternalSlot::ClosureGlobal { closure, .. }
        | ExternalSlot::ClosureUpvalueField { closure, .. } = slot
            && usize::try_from(*closure).map_or(true, |closure| closure >= ir.pinned_closures.len())
        {
            return Err(invalid(
                first + position,
                "external slot names an unpinned closure",
            ));
        }
        match slot {
            ExternalSlot::TailCalls => {
                if ir.loop_tail.is_none() && ir.function.is_none() {
                    return Err(invalid(
                        usize::from(register),
                        "a tail call count in a trace that does not loop through tail calls",
                    ));
                }
                if input_type.is_some_and(|ty| ty != TraceType::Integer) {
                    return Err(invalid(
                        usize::from(register),
                        "a tail call count is no integer",
                    ));
                }
            }
            ExternalSlot::RootClosure => {
                if written {
                    return Err(invalid(
                        usize::from(register),
                        "the root closure slot is written",
                    ));
                }
                if input_type.is_some_and(|ty| ty != TraceType::Closure) {
                    return Err(invalid(
                        usize::from(register),
                        "the root closure slot is no closure",
                    ));
                }
            }
            ExternalSlot::VarargCount | ExternalSlot::Vararg(_) => {
                if written {
                    return Err(invalid(usize::from(register), "a vararg slot is written"));
                }
                if matches!(slot, ExternalSlot::VarargCount)
                    && input_type.is_some_and(|ty| ty != TraceType::Integer)
                {
                    return Err(invalid(
                        usize::from(register),
                        "the vararg count is no integer",
                    ));
                }
            }
            ExternalSlot::Environment => {
                if written {
                    return Err(invalid(
                        usize::from(register),
                        "the environment slot is written",
                    ));
                }
                if input_type.is_some_and(|ty| ty != TraceType::Table) {
                    return Err(invalid(
                        usize::from(register),
                        "the environment slot is no table",
                    ));
                }
            }
            _ => {}
        }
        if matches!(
            slot,
            ExternalSlot::Global(_)
                | ExternalSlot::ClosureGlobal { .. }
                | ExternalSlot::CallerGlobal(_)
                | ExternalSlot::UpvalueField { .. }
                | ExternalSlot::ClosureUpvalueField { .. }
        ) {
            if written {
                return Err(invalid(usize::from(register), "a global slot is written"));
            }
            if reads_nil {
                return Err(invalid(usize::from(register), "a global slot reads as nil"));
            }
            // Globals are read once at entry. An effect (a call or a store)
            // could rebind one mid-run, and no site guard or epoch exists to
            // notice, so such a trace cannot hold a global. A table store
            // only overwrites a number or boolean in place, so it can change
            // the global of such a type (through the environment table) but
            // never rebinds a table or closure global. A store under a
            // run-time string key, which may, misses for a table a global was
            // read from (`GlobalHolders` in the VM).
            let scalar_global = reads_scalar;
            let (unknown_effect, foreign_effects) =
                *global_effects.get_or_insert_with(|| global_slot_effects(ir, first));
            if unknown_effect || (scalar_global && foreign_effects > 0) {
                return Err(invalid(
                    usize::from(register),
                    "a global slot in a trace with effects",
                ));
            }
        }
    }
    if ir.root_constants.len() > ir.root_slot_count as usize {
        return Err(invalid(0, "more root constants than root slots"));
    }
    // A loop through tail calls keeps the frame's tail call count, which
    // every exit writes back.
    if ir.loop_tail.is_some() && !ir.external.contains(&ExternalSlot::TailCalls) {
        return Err(invalid(
            0,
            "a loop through tail calls keeps no tail call count",
        ));
    }
    Ok(())
}

fn define(defined: &mut Defined, value: ValueId) {
    defined.0[value.index()] = true;
}

fn verify_entry_input(
    ir: &TraceIr,
    defined: &mut Defined,
    index: usize,
    input: &EntryInput,
    register_count: u16,
) -> Result<(), TraceVerifyError> {
    let invalid = |detail| TraceVerifyError::InvalidEntry { index, detail };
    if input.register >= register_count {
        return Err(invalid("register is outside the root frame"));
    }
    // An input of either type (nil or a number, see `Recorder::with_lazy`)
    // is a register of the frame that only type guards read: each narrows
    // it to the type its path takes, and nothing else may see it.
    if input.ty == TraceType::Any {
        let guest = usize::from(register_count).saturating_sub(ir.external.len());
        if usize::from(input.register) >= guest || ir.function.is_some() {
            return Err(invalid("entry type is Any"));
        }
        let mut misused = false;
        for instruction in ir.preheader.iter().chain(&ir.body) {
            let guarded = match &instruction.op {
                TraceOp::GuardType { source, .. } => Some(*source),
                _ => None,
            };
            instruction.op.operands(|value| {
                misused |= value == input.value && guarded != Some(value);
            });
        }
        misused |= match &ir.terminator {
            TraceTerminator::LoopBack { args } | TraceTerminator::TailCall { args, .. } => {
                args.contains(&input.value)
            }
            TraceTerminator::Return { value } => *value == Some(input.value),
            _ => false,
        };
        misused |= ir
            .loop_params
            .iter()
            .any(|param| param.init == LoopParamInit::Input(input.value));
        if misused {
            return Err(invalid(
                "an input of type Any is read other than by a type guard",
            ));
        }
    }
    if ir.entry.inputs[..index]
        .iter()
        .any(|earlier| earlier.register == input.register)
    {
        return Err(invalid("register appears twice"));
    }
    let info = ir
        .values
        .get(input.value.index())
        .ok_or(TraceVerifyError::UnknownValue(input.value))?;
    if info.def != ValueDef::Input(index) || info.ty != input.ty {
        return Err(TraceVerifyError::ValueTableMismatch(input.value));
    }
    define(defined, input.value);
    Ok(())
}

/// A register of a seeded frame (see [`super::ir::FrameInput`]): a frame
/// `1..=seed_frames` that a call of the root frame or of the frame below made,
/// a register inside it, a concrete type, and a value defined as this input.
fn verify_frame_input(
    ir: &TraceIr,
    defined: &mut Defined,
    index: usize,
    input: &super::ir::FrameInput,
) -> Result<(), TraceVerifyError> {
    let invalid = |detail| TraceVerifyError::InvalidEntry { index, detail };
    let frame = input.frame.index();
    if frame == 0 || frame > usize::from(ir.seed_frames) {
        return Err(invalid("a frame input names no seeded frame"));
    }
    let info = ir
        .frames
        .get(frame)
        .ok_or_else(|| invalid("a frame input names no seeded frame"))?;
    let seeded = info.inlined.is_some_and(|call| {
        call.parent.index() + 1 == frame
            && call.protected == super::ir::Protected::No
            && call.entry == super::ir::Entry::Call
            && !call.reexec
            && call.results > 0
    });
    if !seeded {
        return Err(invalid(
            "a seeded frame is no plain call of the frame below",
        ));
    }
    if input.register >= info.register_count {
        return Err(invalid("register is outside its frame"));
    }
    if input.ty == TraceType::Any || input.ty.is_borrowed() {
        return Err(invalid("a frame input has no owned concrete type"));
    }
    if ir.frame_inputs[..index]
        .iter()
        .any(|earlier| earlier.frame == input.frame && earlier.register == input.register)
    {
        return Err(invalid("register appears twice"));
    }
    let value = ir
        .values
        .get(input.value.index())
        .ok_or(TraceVerifyError::UnknownValue(input.value))?;
    if value.def != ValueDef::FrameInput(index) || value.ty != input.ty {
        return Err(TraceVerifyError::ValueTableMismatch(input.value));
    }
    define(defined, input.value);
    Ok(())
}

fn verify_site(
    ir: &TraceIr,
    instruction: usize,
    site: super::ir::TraceSite,
) -> Result<(), TraceVerifyError> {
    match ir.frames.get(site.frame.index()) {
        Some(frame) if site.pc < frame.code_len => Ok(()),
        _ => Err(TraceVerifyError::InvalidSite { instruction }),
    }
}

/// `value` past its moves.
fn past_moves(ir: &TraceIr, mut value: ValueId) -> ValueId {
    loop {
        let op = match ir.values.get(value.index()).map(|info| info.def) {
            Some(ValueDef::Instruction(index)) => ir.body.get(index).map(|i| &i.op),
            Some(ValueDef::Preheader(index)) => ir.preheader.get(index).map(|i| &i.op),
            _ => None,
        };
        match op {
            Some(TraceOp::Move { source }) => value = *source,
            _ => return value,
        }
    }
}

/// The number `value` is when it is a constant (past its moves).
fn constant_number(ir: &TraceIr, value: ValueId) -> Option<f64> {
    let value = past_moves(ir, value);
    let op = match ir.values.get(value.index()).map(|info| info.def) {
        Some(ValueDef::Instruction(index)) => ir.body.get(index).map(|i| &i.op),
        Some(ValueDef::Preheader(index)) => ir.preheader.get(index).map(|i| &i.op),
        _ => None,
    };
    match op {
        #[allow(clippy::cast_precision_loss, reason = "keys compare as Lua numbers")]
        Some(TraceOp::Constant(TraceScalar::Integer(number))) => Some(*number as f64),
        Some(TraceOp::Constant(TraceScalar::Float(number))) => Some(*number),
        _ => None,
    }
}

/// Whether two keys name one slot on every run: one value, or two numeric
/// constants that are equal numbers (`1` and `1.0`).
fn same_key(ir: &TraceIr, left: ValueId, right: ValueId) -> bool {
    if past_moves(ir, left) == past_moves(ir, right) {
        return true;
    }
    #[allow(clippy::float_cmp, reason = "keys compare as Lua numbers")]
    match (constant_number(ir, left), constant_number(ir, right)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// The proof [`TraceOp::ArrayOverwrite`] at body instruction `index` rests
/// on: an earlier body instruction reached the same slot of the same
/// binding and table and found (or stored) a value of the overwrite's type,
/// a number or boolean, and nothing from there on may move or retype an
/// array slot ([`TraceOp::keeps_array_slots`]). The body is straight-line
/// code that runs from its start in every iteration, so the earlier access
/// ran, and passed its guard, whenever the overwrite runs.
fn overwrite_proven(ir: &TraceIr, index: usize) -> bool {
    let Some(TraceOp::ArrayOverwrite {
        binding,
        table,
        key,
        value,
        ..
    }) = ir.body.get(index).map(|instruction| &instruction.op)
    else {
        return false;
    };
    let Some(ty) = ir.value_type(*value) else {
        return false;
    };
    if !matches!(
        ty,
        TraceType::Integer | TraceType::Float | TraceType::Boolean
    ) {
        return false;
    }
    for earlier in ir.body[..index].iter().rev() {
        let reached = match &earlier.op {
            TraceOp::ArrayLoad {
                binding: b,
                table: t,
                key: k,
                expected,
                ..
            } => (*expected == ty).then_some((*b, *t, *k)),
            TraceOp::ArrayStore {
                binding: b,
                table: t,
                key: k,
                value: v,
                ..
            }
            | TraceOp::ArrayOverwrite {
                binding: b,
                table: t,
                key: k,
                value: v,
                ..
            } => (ir.value_type(*v) == Some(ty)).then_some((*b, *t, *k)),
            _ => None,
        };
        if let Some((b, t, k)) = reached
            && b == *binding
            && t == *table
            && same_key(ir, k, *key)
        {
            return true;
        }
        if !earlier.op.keeps_array_slots() {
            return false;
        }
    }
    false
}

fn operand_type(ir: &TraceIr, value: ValueId) -> Result<TraceType, TraceVerifyError> {
    ir.value_type(value)
        .ok_or(TraceVerifyError::UnknownValue(value))
}

const fn is_number(ty: TraceType) -> bool {
    matches!(ty, TraceType::Integer | TraceType::Float)
}

fn operation_result_type(
    ir: &TraceIr,
    index: usize,
    op: &TraceOp,
) -> Result<Option<TraceType>, TraceVerifyError> {
    let mismatch = |detail| TraceVerifyError::TypeMismatch {
        instruction: index,
        detail,
    };
    Ok(match op {
        TraceOp::Constant(scalar) => {
            // A closure constant is a pinned closure, named by its lane in
            // the run's root bank. A table has no such lane.
            match scalar {
                TraceScalar::Closure(lane)
                    if usize::try_from(*lane).is_ok_and(|lane| lane < ir.pinned_closures.len()) => {
                }
                // A string of the root bank, which only a host call reads.
                TraceScalar::String(lane)
                    if lane
                        .checked_sub(ROOT_STRING_LANE_BASE)
                        .and_then(|slot| usize::try_from(slot).ok())
                        .is_some_and(|slot| slot < ir.root_constants.len()) => {}
                _ if scalar.ty().is_opaque() || scalar.ty().is_borrowed() => {
                    return Err(mismatch(
                        "an opaque constant must be a pinned closure of the trace",
                    ));
                }
                _ => {}
            }
            Some(scalar.ty())
        }
        TraceOp::PhantomTable { .. } => None,
        TraceOp::SelfCall { args, .. } => {
            let function = ir
                .function
                .as_ref()
                .ok_or_else(|| mismatch("a self call outside a function trace"))?;
            if args.len() != function.parameters.len() {
                return Err(mismatch("a self call passes another number of parameters"));
            }
            for (arg, ty) in args.iter().zip(&function.parameters) {
                if operand_type(ir, *arg)? != *ty {
                    return Err(mismatch(
                        "a self call passes a parameter of another type than the entry's",
                    ));
                }
            }
            function.result
        }
        TraceOp::Move { source } => Some(operand_type(ir, *source)?),
        TraceOp::Negate { source } => {
            let ty = operand_type(ir, *source)?;
            if !is_number(ty) && !ty.is_wide() {
                return Err(mismatch("negate needs a number"));
            }
            Some(ty)
        }
        TraceOp::Arithmetic { op, left, right } => {
            let left = operand_type(ir, *left)?;
            let right = operand_type(ir, *right)?;
            if left != right {
                return Err(mismatch("arithmetic operands differ in type"));
            }
            match left {
                TraceType::Float if op.allows_floats() => Some(TraceType::Float),
                TraceType::Integer if op.allows_integers() => Some(TraceType::Integer),
                TraceType::Int64 | TraceType::UInt64 if op.allows_wide() => Some(left),
                _ => return Err(mismatch("arithmetic operand type is not allowed")),
            }
        }
        TraceOp::IntToFloat { source } => {
            let ty = operand_type(ir, *source)?;
            if ty != TraceType::Integer && !ty.is_wide() {
                return Err(mismatch("integer-to-float needs an integer"));
            }
            Some(TraceType::Float)
        }
        TraceOp::Numeric { op, source } => {
            let ty = operand_type(ir, *source)?;
            Some(
                op.result_type(ty)
                    .ok_or_else(|| mismatch("a numeric operation needs another operand type"))?,
            )
        }
        TraceOp::ToWide {
            source,
            target,
            conversion,
        } => {
            let source = operand_type(ir, *source)?;
            let from_float = source == TraceType::Float;
            let convertible = match conversion {
                WideConversion::Truncate => from_float || (source.is_wide() && source != *target),
                WideConversion::Bits32 => from_float && *target == TraceType::Int64,
            };
            if !target.is_wide() || !convertible {
                return Err(mismatch(
                    "a wide conversion needs a float or the other cdata",
                ));
            }
            Some(*target)
        }
        TraceOp::NativeUnary { source, .. } => {
            if operand_type(ir, *source)? != TraceType::Float {
                return Err(mismatch("a native call needs a float"));
            }
            Some(TraceType::Float)
        }
        TraceOp::NativeBinary { left, right, .. } => {
            if operand_type(ir, *left)? != TraceType::Float
                || operand_type(ir, *right)? != TraceType::Float
            {
                return Err(mismatch("a native call needs floats"));
            }
            Some(TraceType::Float)
        }
        TraceOp::Compare {
            comparison,
            left,
            right,
        } => {
            let left = operand_type(ir, *left)?;
            let right = operand_type(ir, *right)?;
            // An integer and a float compare by exact value.
            let numbers = is_number(left) && is_number(right);
            if (left != right && !numbers) || left == TraceType::Any {
                return Err(mismatch("comparison needs two values of one concrete type"));
            }
            if left.is_borrowed() {
                return Err(mismatch("a borrowed string takes no comparison"));
            }
            if *comparison != Comparison::Equal && !is_number(left) && !left.is_wide() {
                return Err(mismatch("ordering needs numbers"));
            }
            Some(TraceType::Boolean)
        }
        TraceOp::Select {
            condition,
            if_true,
            if_false,
        } => {
            if operand_type(ir, *condition)? != TraceType::Boolean {
                return Err(mismatch("a select needs a boolean condition"));
            }
            let ty = operand_type(ir, *if_true)?;
            if ty != operand_type(ir, *if_false)? {
                return Err(mismatch("a select needs two arms of one type"));
            }
            if !matches!(
                ty,
                TraceType::Integer | TraceType::Float | TraceType::Boolean
            ) {
                return Err(mismatch("a select picks an integer, a float or a boolean"));
            }
            Some(ty)
        }
        TraceOp::Truthy { source } | TraceOp::Not { source } => {
            operand_type(ir, *source)?;
            Some(TraceType::Boolean)
        }
        TraceOp::GuardType {
            source, expected, ..
        } => {
            let ty = operand_type(ir, *source)?;
            if *expected == TraceType::Any {
                return Err(mismatch("guard expects Any"));
            }
            if ty != TraceType::Any && ty != *expected {
                return Err(mismatch("guard can never pass for a typed value"));
            }
            if expected.is_borrowed() {
                return Err(mismatch("a type guard cannot produce a borrow"));
            }
            Some(*expected)
        }
        TraceOp::GuardBranch { condition, .. } => {
            if operand_type(ir, *condition)? != TraceType::Boolean {
                return Err(mismatch("branch guard needs a boolean"));
            }
            None
        }
        TraceOp::Read {
            result, operands, ..
        } => {
            opaque_call_operands(ir, index, operands, Some(*result))?;
            Some(*result)
        }
        TraceOp::Effect {
            result, operands, ..
        } => {
            opaque_call_operands(ir, index, operands, *result)?;
            *result
        }
        TraceOp::ArrayLoad {
            binding,
            table,
            key,
            expected,
            past_end,
            ..
        } => {
            table_operand(ir, index, *binding, *table, BindingShape::Array)?;
            array_key(ir, index, *key)?;
            if *past_end {
                // The read of a key outside the array needs the header's
                // word, which says whether the hash part is empty.
                let register = ir.bindings[*binding as usize].register;
                let header = ir.bindings.iter().any(|other| {
                    other.register == register && matches!(other.kind, BindingKind::Header { .. })
                });
                if *expected != TraceType::Nil || !header {
                    return Err(mismatch(
                        "a read past the end of an array is nil and needs the table's header",
                    ));
                }
            }
            if *expected == TraceType::Nil {
                // A nil read is the access's value only when no `__index`
                // can run.
                let register = ir.bindings[*binding as usize].register;
                let plain = ir.bindings.iter().any(|other| {
                    other.register == register && other.kind == BindingKind::NoMetatable
                });
                if !plain {
                    return Err(mismatch(
                        "a nil array read needs a table without a metatable",
                    ));
                }
            } else if !is_storable(*expected)
                && *expected != TraceType::Table
                && *expected != TraceType::Closure
                && *expected != TraceType::Thread
                && *expected != TraceType::StringRef
            {
                return Err(mismatch(
                    "an array read needs a number, boolean, nil, table, closure, thread or string type",
                ));
            }
            Some(*expected)
        }
        TraceOp::LaneArrayLoad {
            table,
            field,
            key,
            expected,
            ..
        } => {
            // The words of the array come from the table's address, which
            // the row's address field holds.
            let field = field_operand(ir, index, *field, *table)?;
            if field.key.as_ref() != TABLE_ADDRESS_KEY || field.ty != TraceType::Integer {
                return Err(mismatch(
                    "a lane array read needs the table address row field",
                ));
            }
            array_key(ir, index, *key)?;
            // A string is borrowed from the slot (a loop side's `ipairs`
            // elements, see "Loops in calls" in `vm::jit::trace`).
            if !matches!(
                expected,
                TraceType::Integer
                    | TraceType::Float
                    | TraceType::Boolean
                    | TraceType::Table
                    | TraceType::StringRef
            ) {
                return Err(mismatch(
                    "a lane array read needs a number, boolean, table or borrowed string type",
                ));
            }
            Some(*expected)
        }
        TraceOp::NumberLoad {
            binding,
            table,
            key,
            ..
        } => {
            table_operand(ir, index, *binding, *table, BindingShape::Array)?;
            array_key(ir, index, *key)?;
            Some(TraceType::Float)
        }
        TraceOp::NumberTag { load } => {
            if operand_type(ir, *load)? != TraceType::Float {
                return Err(mismatch("the tag of a number load reads a float"));
            }
            Some(TraceType::Boolean)
        }
        TraceOp::NumberModulo {
            float,
            left,
            right,
            integer,
        } => {
            if operand_type(ir, *left)? != TraceType::Float
                || operand_type(ir, *right)? != TraceType::Float
                || operand_type(ir, *integer)? != TraceType::Boolean
                || !matches!(float, ArithmeticOp::Modulo | ArithmeticOp::ModuloWith(_))
            {
                return Err(mismatch("a number modulo takes two floats and a tag"));
            }
            Some(TraceType::Float)
        }
        TraceOp::BoolAnd { left, right } => {
            if operand_type(ir, *left)? != TraceType::Boolean
                || operand_type(ir, *right)? != TraceType::Boolean
            {
                return Err(mismatch("a conjunction needs booleans"));
            }
            Some(TraceType::Boolean)
        }
        TraceOp::ArrayNext {
            binding,
            table,
            key,
            ty,
            ..
        } => {
            table_operand(ir, index, *binding, *table, BindingShape::Array)?;
            array_key(ir, index, *key)?;
            if !matches!(ty, TraceType::Integer | TraceType::Float) {
                return Err(mismatch("an array step gives an integer or a float"));
            }
            Some(*ty)
        }
        TraceOp::ReleaseStrings { keep } => {
            if keep.len() > MAX_HOST_ARGS
                || keep.iter().any(|value| {
                    !matches!(
                        operand_type(ir, *value),
                        Ok(TraceType::String | TraceType::Closure)
                    )
                })
                || !matches!(ir.terminator, super::ir::TraceTerminator::LoopBack { .. })
                || ir.body.last().map(|last| &last.op) != Some(op)
            {
                return Err(mismatch(
                    "a release ends a loop body and keeps at most five strings or iterators",
                ));
            }
            None
        }
        TraceOp::HostCall {
            function,
            args,
            result,
            ..
        } => {
            if args.is_empty() || args.len() > MAX_HOST_ARGS {
                return Err(mismatch("a host call passes one to five arguments"));
            }
            for arg in args {
                if matches!(
                    operand_type(ir, *arg)?,
                    TraceType::Any | TraceType::Closure | TraceType::Int64 | TraceType::UInt64
                ) {
                    return Err(mismatch("a host call takes no closure, cdata or Any"));
                }
            }
            if !matches!(
                result,
                TraceType::Nil
                    | TraceType::Boolean
                    | TraceType::Integer
                    | TraceType::Float
                    | TraceType::String
            ) {
                return Err(mismatch("a host call yields a scalar or a string"));
            }
            if matches!(function, HostFunction::MathMax | HostFunction::MathMin)
                && (*result != TraceType::Integer
                    || args
                        .iter()
                        .any(|arg| operand_type(ir, *arg) != Ok(TraceType::Integer)))
            {
                return Err(mismatch("max and min take and give integers"));
            }
            let arity_ok = match function {
                HostFunction::Native(_) => true,
                HostFunction::StringLength => args.len() == 1,
                HostFunction::ConcatLength => (2..=MAX_HOST_ARGS).contains(&args.len()),
                HostFunction::MathMax | HostFunction::MathMin => args.len() == 2,
                HostFunction::ReleaseStrings => false,
            };
            if !arity_ok {
                return Err(mismatch("a host call passes the wrong number of arguments"));
            }
            if *function == HostFunction::ConcatLength
                && (!matches!(result, TraceType::Integer | TraceType::Float)
                    || args.iter().any(|arg| {
                        !matches!(
                            operand_type(ir, *arg),
                            Ok(TraceType::String
                                | TraceType::StringRef
                                | TraceType::Integer
                                | TraceType::Float)
                        )
                    }))
            {
                return Err(mismatch(
                    "a concatenation length takes strings and numbers and gives a number",
                ));
            }
            Some(*result)
        }
        TraceOp::CallHelper {
            helper,
            args,
            effect,
            ..
        } => {
            let info = helper
                .info()
                .ok_or_else(|| mismatch("a helper call names a helper the table lacks"))?;
            if args.len() < info.required
                || args.len() > info.params.len()
                || args.len() > MAX_HELPER_ARGS
            {
                return Err(mismatch(
                    "a helper call passes the wrong number of arguments",
                ));
            }
            if info.params[info.required..]
                .iter()
                .any(|param| *param != HelperType::Value)
            {
                return Err(mismatch("an optional helper parameter is not a value"));
            }
            for (arg, param) in args.iter().zip(info.params) {
                if !param.accepts(operand_type(ir, *arg)?) {
                    return Err(mismatch("a helper argument has another type than its slot"));
                }
            }
            if info.results.len() > MAX_HELPER_RESULTS {
                return Err(mismatch("a helper gives too many results"));
            }
            if info.effects.contains(HelperEffects::PURE)
                && (info.effects.contains(HelperEffects::READS_HEAP)
                    || info.effects.contains(HelperEffects::WRITES_HEAP))
            {
                return Err(mismatch("a pure helper touches the heap"));
            }
            if effect.is_some() != info.effects.commits_effect() {
                return Err(mismatch(
                    "a helper call commits an effect exactly when its helper writes or advances private state",
                ));
            }
            info.results.first().map(|result| result.trace_type())
        }
        TraceOp::HelperResult { call, index } => {
            let defined_by = match ir.values.get(call.index()).map(|info| info.def) {
                Some(ValueDef::Instruction(at)) => ir.body.get(at),
                _ => None,
            };
            let Some(TraceOp::CallHelper { helper, .. }) = defined_by.map(|found| &found.op) else {
                return Err(mismatch("a helper result reads the value of a helper call"));
            };
            let results = helper.info().map_or(&[][..], |info| info.results);
            match results.get(usize::from(*index)) {
                Some(result) if *index >= 1 => Some(result.trace_type()),
                _ => return Err(mismatch("a helper result names a result the helper lacks")),
            }
        }
        TraceOp::HashLoad {
            binding,
            table,
            key,
            expected,
            ..
        } => {
            // A nil result is the access's value only without a metatable.
            // With a metatable class pinned, a nil is the guard that the key
            // is absent from the table itself, before the handler runs.
            let class = ir
                .bindings
                .iter()
                .any(|bound| matches!(bound.kind, BindingKind::MetaClass { .. }));
            let shape = if *expected == TraceType::Nil && !class {
                BindingShape::NoMetatable
            } else {
                BindingShape::Hash
            };
            table_operand(ir, index, *binding, *table, shape)?;
            if !matches!(
                operand_type(ir, *key)?,
                TraceType::String | TraceType::StringRef | TraceType::Integer | TraceType::Float
            ) {
                return Err(mismatch("a hash lookup needs a string or number key"));
            }
            if !is_storable(*expected) && *expected != TraceType::Nil {
                return Err(mismatch("a hash lookup yields a number, boolean or nil"));
            }
            Some(*expected)
        }
        TraceOp::ArraySet {
            binding,
            table,
            key,
            value,
            ..
        } => {
            table_operand(ir, index, *binding, *table, BindingShape::Array)?;
            array_key(ir, index, *key)?;
            let ty = operand_type(ir, *value)?;
            if !is_storable(ty)
                && !matches!(ty, TraceType::Nil | TraceType::String | TraceType::Table)
            {
                return Err(mismatch(
                    "a table set needs a number, boolean, nil, string or table value",
                ));
            }
            None
        }
        TraceOp::ArrayStore {
            binding,
            table,
            key,
            value,
            ..
        }
        | TraceOp::ArrayOverwrite {
            binding,
            table,
            key,
            value,
            ..
        } => {
            table_operand(ir, index, *binding, *table, BindingShape::Array)?;
            array_key(ir, index, *key)?;
            if !is_storable(operand_type(ir, *value)?) {
                return Err(mismatch("an array store needs a number or boolean value"));
            }
            None
        }
        TraceOp::FieldLoad { binding, table } => {
            let ty = table_operand(ir, index, *binding, *table, BindingShape::Field)?;
            Some(ty)
        }
        TraceOp::FieldStore {
            binding,
            table,
            value,
            ..
        } => {
            let ty = table_operand(ir, index, *binding, *table, BindingShape::FieldStore)?;
            if operand_type(ir, *value)? != ty {
                return Err(mismatch("a field store changes the slot's type"));
            }
            None
        }
        TraceOp::StringLength { binding, string } => {
            let ty = table_operand(ir, index, *binding, *string, BindingShape::Bytes)?;
            Some(ty)
        }
        TraceOp::StringByte {
            binding,
            string,
            key,
            ..
        } => {
            let ty = table_operand(ir, index, *binding, *string, BindingShape::Bytes)?;
            array_key(ir, index, *key)?;
            Some(ty)
        }
        TraceOp::TableLength { binding, table } => {
            let ty = table_operand(ir, index, *binding, *table, BindingShape::Length)?;
            Some(ty)
        }
        TraceOp::MetaGuard { table, binding, .. } => {
            if operand_type(ir, *table)? != TraceType::Table {
                return Err(mismatch("a metatable guard needs a table"));
            }
            if !matches!(
                ir.bindings.get(*binding as usize).map(|bound| &bound.kind),
                Some(BindingKind::MetaClass { .. })
            ) {
                return Err(mismatch("a metatable guard names no metatable class"));
            }
            // The compare reads the metatable word of the table's row.
            if !ir
                .fields
                .iter()
                .any(|field| field.key.as_ref() == META_FIELD_KEY && field.ty == TraceType::Integer)
            {
                return Err(mismatch("a metatable guard needs the metatable row field"));
            }
            None
        }
        TraceOp::TableFieldLoad { field, table, .. } => {
            let field = field_operand(ir, index, *field, *table)?;
            Some(field.ty)
        }
        TraceOp::TableFieldStore {
            field,
            table,
            value,
            ..
        } => {
            let field = field_operand(ir, index, *field, *table)?;
            if !is_storable(field.ty) || operand_type(ir, *value)? != field.ty {
                return Err(mismatch("a field store changes the slot's type"));
            }
            None
        }
        TraceOp::LaneArrayStore {
            table,
            field,
            key,
            value,
            ..
        } => {
            // The array is found as a lane array read finds it.
            let field = field_operand(ir, index, *field, *table)?;
            if field.key.as_ref() != TABLE_ADDRESS_KEY || field.ty != TraceType::Integer {
                return Err(mismatch(
                    "a lane array store needs the table address row field",
                ));
            }
            array_key(ir, index, *key)?;
            if !is_storable(operand_type(ir, *value)?) {
                return Err(mismatch(
                    "a lane array store needs a number or boolean value",
                ));
            }
            None
        }
    })
}

/// The outermost inlined call around `frame` (or `frame` itself): the frame
/// that made it and the pc of the call instruction there, or `None` for the
/// root frame. Also whether some call on the way has to start over when
/// anything in it leaves the trace ([`InlinedFrame::reexec`]).
fn outer_call(ir: &TraceIr, frame: FrameId) -> (Option<(FrameId, usize)>, bool) {
    let mut outermost = None;
    let mut must_restart = false;
    let mut current = frame;
    // Frames are listed callee after caller, so the walk is bounded.
    for _ in 0..ir.frames.len() {
        let Some(call) = ir.frames.get(current.index()).and_then(|info| info.inlined) else {
            break;
        };
        outermost = Some((call.parent, call.call_pc));
        must_restart |= call.reexec;
        current = call.parent;
    }
    (outermost, must_restart)
}

/// A type a table slot can hold natively: one with an in-place payload.
/// An opaque read or effect takes no borrowed operand and gives none.
fn opaque_call_operands(
    ir: &TraceIr,
    index: usize,
    operands: &[ValueId],
    result: Option<TraceType>,
) -> Result<(), TraceVerifyError> {
    let borrowed = result.is_some_and(TraceType::is_borrowed)
        || operands
            .iter()
            .any(|value| ir.value_type(*value).is_some_and(TraceType::is_borrowed));
    if borrowed {
        return Err(TraceVerifyError::TypeMismatch {
            instruction: index,
            detail: "an opaque operation takes and gives no borrowed value",
        });
    }
    Ok(())
}

const fn is_storable(ty: TraceType) -> bool {
    matches!(
        ty,
        TraceType::Integer | TraceType::Float | TraceType::Boolean
    )
}

/// Checks that `field` names a field of the trace and `table` is a table.
fn field_operand(
    ir: &TraceIr,
    index: usize,
    field: u32,
    table: ValueId,
) -> Result<&TableField, TraceVerifyError> {
    let mismatch = |detail| TraceVerifyError::TypeMismatch {
        instruction: index,
        detail,
    };
    let field = ir
        .fields
        .get(field as usize)
        .ok_or_else(|| mismatch("operation names a missing table field"))?;
    if operand_type(ir, table)? != TraceType::Table {
        return Err(mismatch("a field access needs a table"));
    }
    Ok(field)
}

/// Checks the field list: distinct entries with a payload type.
fn verify_fields(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    for (index, field) in ir.fields.iter().enumerate() {
        if ir.fields[..index].contains(field) {
            return Err(invalid(index, "a table field appears twice"));
        }
        if !is_storable(field.ty) && !matches!(field.ty, TraceType::Table | TraceType::Nil) {
            return Err(invalid(
                index,
                "a table field needs a number, a boolean, a table or nil",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BindingShape {
    Array,
    Field,
    /// A field a store overwrites: it must be the bound table's own.
    FieldStore,
    Length,
    /// The bytes of a string, which the entry input holds instead of a table.
    Bytes,
    /// A table known to have no metatable ([`BindingKind::NoMetatable`]).
    NoMetatable,
    /// A table a present-key hash lookup reads ([`BindingKind::Hash`]).
    Hash,
}

/// Checks that binding `binding` has `shape` and that `table` is the entry
/// input of the binding's register, a table no iteration rewrites. Returns
/// the type a field or length binding yields (`Any` for an array).
fn table_operand(
    ir: &TraceIr,
    index: usize,
    binding: u32,
    table: ValueId,
    shape: BindingShape,
) -> Result<TraceType, TraceVerifyError> {
    let mismatch = |detail| TraceVerifyError::TypeMismatch {
        instruction: index,
        detail,
    };
    let binding = ir
        .bindings
        .get(binding as usize)
        .ok_or_else(|| mismatch("operation names a missing table binding"))?;
    let (kind_ok, ty) = match (&binding.kind, shape) {
        (BindingKind::Array { .. }, BindingShape::Array) => (true, TraceType::Any),
        (BindingKind::Field { ty, .. }, BindingShape::Field)
        | (BindingKind::Length { ty, .. }, BindingShape::Length)
        | (BindingKind::Bytes { ty }, BindingShape::Bytes) => (true, *ty),
        (BindingKind::Field { ty, depth: 0, .. }, BindingShape::FieldStore) => (true, *ty),
        (BindingKind::NoMetatable, BindingShape::NoMetatable)
        | (BindingKind::NoMetatable | BindingKind::Hash, BindingShape::Hash) => {
            (true, TraceType::Any)
        }
        _ => (false, TraceType::Any),
    };
    if !kind_ok {
        return Err(mismatch("operation uses a binding of another kind"));
    }
    let info = ir
        .values
        .get(table.index())
        .ok_or(TraceVerifyError::UnknownValue(table))?;
    let ValueDef::Input(input) = info.def else {
        return Err(mismatch("a table operand must be an entry input"));
    };
    let holder = if shape == BindingShape::Bytes {
        TraceType::String
    } else {
        TraceType::Table
    };
    if info.ty != holder || ir.entry.inputs[input].register != binding.register {
        return Err(mismatch(
            "a table operand is not the table its binding resolves",
        ));
    }
    Ok(ty)
}

/// An array key is an integer or a float, which native code checks against
/// the array.
fn array_key(ir: &TraceIr, index: usize, key: ValueId) -> Result<(), TraceVerifyError> {
    if is_number(operand_type(ir, key)?) {
        Ok(())
    } else {
        Err(TraceVerifyError::TypeMismatch {
            instruction: index,
            detail: "an array key must be a number",
        })
    }
}

/// Checks the table bindings: well formed and over table entry inputs. A side
/// trace lists the root's bindings first (see `verify_sides`), so each
/// member of a tree has an entry input for every table the tree binds.
fn verify_bindings(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    let invalid_array = |detail| TraceVerifyError::InvalidEntry { index: 0, detail };
    if let Some(steps) = ir.boxed_steps {
        let is_array = steps.array.is_none_or(|binding| {
            ir.bindings
                .get(binding as usize)
                .is_some_and(super::ir::TableBinding::is_array)
        });
        if !is_array || ir.boxed_control.is_none() {
            return Err(invalid_array(
                "native steps of a boxed loop need a boxed loop and an array binding",
            ));
        }
    }
    let invalid = |index: usize, detail| TraceVerifyError::InvalidEntry { index, detail };
    // Which bindings an operation (or the native steps of a boxed loop)
    // names.
    let mut named = vec![false; ir.bindings.len()];
    let steps = ir.boxed_steps.and_then(|steps| steps.array);
    for binding in ir
        .preheader
        .iter()
        .chain(&ir.body)
        .filter_map(|instruction| instruction.op.binding())
        .chain(steps)
    {
        if let Some(named) = named.get_mut(binding as usize) {
            *named = true;
        }
    }
    for (index, binding) in ir.bindings.iter().enumerate() {
        if ir.bindings[..index].contains(binding) {
            return Err(invalid(index, "a table binding appears twice"));
        }
        let holder = if matches!(binding.kind, BindingKind::Bytes { .. }) {
            TraceType::String
        } else {
            TraceType::Table
        };
        let held = ir
            .entry
            .inputs
            .iter()
            .any(|input| input.register == binding.register && input.ty == holder);
        let register = binding.register;
        // A side lists the root's bindings first, and may start where such a
        // register holds no table: a temporary of the root's iteration (the
        // argument of a call the root makes every trip) that a later
        // instruction of the iteration overwrites. Some member of the tree
        // then writes the register, so the plan demotes the binding to lane
        // views or refuses the tree (`TreeError::TableWritten`), and only the
        // root's entry resolves bindings. Such a side is sound as long as
        // none of its own operations names the binding.
        let side = ir.function.is_none()
            && matches!(
                ir.terminator,
                TraceTerminator::Link { .. }
                    | TraceTerminator::LinkLoop { .. }
                    | TraceTerminator::Exit { .. }
            );
        if !held && (!side || named[index]) {
            return Err(invalid(index, "a table binding names no table entry input"));
        }
        if ir.written_registers.contains(&(FrameId(0), register)) {
            // A table the trace rewrites is not the same one in every
            // iteration. A loop parameter can still carry it, but its entry
            // value is not what the binding resolved.
            let carried = ir
                .loop_params
                .iter()
                .any(|param| param.register == Some(register));
            if carried {
                return Err(invalid(index, "a bound table is a loop parameter"));
            }
        }
        match &binding.kind {
            BindingKind::Array { first } => {
                if first.is_some_and(|first| !is_storable(first) && first != TraceType::Table) {
                    return Err(invalid(
                        index,
                        "an array binding hints a type with no payload",
                    ));
                }
            }
            BindingKind::Field { ty, depth, .. } => {
                if !is_storable(*ty) {
                    return Err(invalid(index, "a field binding needs a number or boolean"));
                }
                if *depth > MAX_INDEX_HOPS {
                    return Err(invalid(index, "a field binding follows too many hops"));
                }
            }
            BindingKind::Length { ty, .. } => {
                if !is_number(*ty) {
                    return Err(invalid(index, "a length binding needs a number type"));
                }
            }
            BindingKind::Bytes { ty } => {
                if !is_number(*ty) {
                    return Err(invalid(index, "a bytes binding needs a number type"));
                }
            }
            BindingKind::Callee { lane, depth, .. } => {
                if usize::try_from(*lane).map_or(true, |lane| lane >= ir.pinned_closures.len()) {
                    return Err(invalid(index, "a callee binding names no pinned callable"));
                }
                if *depth > MAX_INDEX_HOPS {
                    return Err(invalid(index, "a callee binding follows too many hops"));
                }
            }
            BindingKind::StringField { depth, .. } => {
                if *depth > MAX_INDEX_HOPS {
                    return Err(invalid(index, "a string binding follows too many hops"));
                }
            }
            BindingKind::MetaClass { pins, bare, .. } => {
                if pins.is_empty() && *bare == 0 {
                    return Err(invalid(index, "a metatable class pins nothing"));
                }
                if pins.iter().any(|pin| !pin.event.starts_with(b"__")) {
                    return Err(invalid(
                        index,
                        "a metatable class pins a field that is no event",
                    ));
                }
            }
            BindingKind::NoMetatable | BindingKind::Hash | BindingKind::Header { .. } => {}
        }
    }
    Ok(())
}

/// `previous` is the snapshot before `snapshot` in the trace, which passed
/// this check: the entries of a run the two share at one frame position were
/// checked there, when neither snapshot has a recipe an entry could name.
fn verify_snapshot_shape(
    ir: &TraceIr,
    index: usize,
    snapshot: &Snapshot,
    previous: Option<&Snapshot>,
    effect_count: u32,
) -> Result<(), TraceVerifyError> {
    let result = snapshot_shape(ir, index, snapshot, previous, effect_count);
    // Debug builds check every snapshot whole as well.
    debug_assert!(
        previous.is_none()
            || result.is_ok() == snapshot_shape(ir, index, snapshot, None, effect_count).is_ok(),
        "skipping the runs snapshot {index} shares changed its shape check"
    );
    result
}

fn snapshot_shape(
    ir: &TraceIr,
    index: usize,
    snapshot: &Snapshot,
    previous: Option<&Snapshot>,
    effect_count: u32,
) -> Result<(), TraceVerifyError> {
    let no_recipes =
        |snapshot: &Snapshot| snapshot.virtuals.is_empty() && snapshot.slices.is_empty();
    let previous = previous.filter(|previous| no_recipes(previous) && no_recipes(snapshot));
    let invalid = |detail| TraceVerifyError::InvalidSnapshot {
        snapshot: index,
        detail,
    };
    if snapshot.id.index() != index {
        return Err(invalid("id differs from table position"));
    }
    if snapshot.frames.is_empty() {
        return Err(invalid("no frames"));
    }
    if snapshot.frames.len() > MAX_TRACE_FRAME_DEPTH {
        return Err(TraceVerifyError::FrameDepth {
            snapshot: index,
            depth: snapshot.frames.len(),
        });
    }
    // The root frame, or a later activation of it (see `RootActivation`).
    let first = snapshot.frames[0].frame.index();
    if first != 0
        && !ir
            .frames
            .get(first)
            .is_some_and(|info| info.is_root() && info.activation.is_some())
    {
        return Err(invalid("first frame is not the root frame"));
    }
    if snapshot.effect_position > effect_count {
        return Err(invalid("effect position exceeds the trace's effects"));
    }
    for (position, frame) in snapshot.frames.iter().enumerate() {
        let info = ir
            .frames
            .get(frame.frame.index())
            .ok_or_else(|| invalid("names an unknown frame"))?;
        if snapshot.frames[..position]
            .iter()
            .any(|earlier| earlier.frame == frame.frame)
        {
            return Err(invalid("frame appears twice"));
        }
        if frame.registers.len() != usize::from(info.register_count) {
            return Err(invalid("register map is not complete"));
        }
        // A frame that runs the closure of this run holds it in its last
        // register, as a value the exit restores the frame from.
        if info.inlined.is_some_and(|call| call.dynamic) {
            match frame.registers.last() {
                Some(RegisterEntry::Value(value))
                    if ir.value_type(*value) == Some(TraceType::Closure) => {}
                _ => {
                    return Err(invalid(
                        "a frame that runs the closure of this run lacks it in its last register",
                    ));
                }
            }
        }
        if frame
            .last_interpreted_pc
            .is_some_and(|pc| pc >= info.code_len)
        {
            return Err(invalid("last interpreted pc is out of bounds"));
        }
        // The window beyond the guest registers holds cells and globals.
        let guest =
            frame
                .registers
                .len()
                .saturating_sub(if position == 0 { ir.external.len() } else { 0 });
        // The same position and length give the same guest count, and the
        // other limits are the trace's.
        let before = previous
            .and_then(|previous| previous.frames.get(position))
            .map(|frame| &frame.registers);
        for (first, run) in frame.registers.changed_from(before) {
            for (register, entry) in (first..).zip(run) {
                match entry {
                    RegisterEntry::Root(slot) if *slot >= ir.root_slot_count => {
                        return Err(invalid("root slot is out of range"));
                    }
                    RegisterEntry::Virtual(id) => {
                        if id.index() >= snapshot.virtuals.len() {
                            return Err(invalid("a virtual entry names no recipe"));
                        }
                        if register >= guest {
                            return Err(invalid(
                                "a virtual table cannot sit in a cell or global slot",
                            ));
                        }
                    }
                    RegisterEntry::Slice(id) => {
                        if id.index() >= snapshot.slices.len() {
                            return Err(invalid("a slice entry names no recipe"));
                        }
                        if register >= guest {
                            return Err(invalid("a substring cannot sit in a cell or global slot"));
                        }
                    }
                    RegisterEntry::Const(scalar)
                        if scalar.ty().is_opaque() || scalar.ty().is_borrowed() =>
                    {
                        return Err(invalid("an opaque value cannot be a snapshot constant"));
                    }
                    _ => {}
                }
            }
        }
    }
    for number in &snapshot.numbers {
        let register = number.register as usize;
        let root = &snapshot.frames[0];
        let guest = root.registers.len().saturating_sub(ir.external.len());
        let float = match root.registers.get(register) {
            Some(RegisterEntry::Value(value)) => ir.value_type(*value) == Some(TraceType::Float),
            Some(RegisterEntry::Const(scalar)) => scalar.ty() == TraceType::Float,
            _ => false,
        };
        let boolean = match number.is_integer {
            RegisterEntry::Value(value) => ir.value_type(value) == Some(TraceType::Boolean),
            RegisterEntry::Const(scalar) => scalar.ty() == TraceType::Boolean,
            _ => false,
        };
        if register >= guest || !float || !boolean {
            return Err(invalid(
                "a number register is a guest register that holds a float and a boolean tag",
            ));
        }
        if snapshot
            .numbers
            .iter()
            .filter(|other| other.register == number.register)
            .count()
            > 1
        {
            return Err(invalid("a register has two number tags"));
        }
    }
    for (index, table) in snapshot.virtuals.iter().enumerate() {
        verify_recipe(ir, snapshot, index, table)?;
    }
    for recipe in &snapshot.slices {
        let number = |entry: &RegisterEntry| match entry {
            RegisterEntry::Value(value) => matches!(
                ir.value_type(*value),
                Some(TraceType::Integer | TraceType::Float)
            ),
            RegisterEntry::Const(scalar) => {
                matches!(scalar.ty(), TraceType::Integer | TraceType::Float)
            }
            _ => false,
        };
        if !matches!(
            recipe.string,
            RegisterEntry::Value(value) if ir.value_type(value) == Some(TraceType::String)
        ) {
            return Err(invalid("a substring is made from a string value"));
        }
        if !number(&recipe.first) || !number(&recipe.last) {
            return Err(invalid("a substring takes number indices"));
        }
    }
    // The frames form one chain of calls, each outer frame stopped at the call
    // that made the next one.
    for pair in snapshot.frames.windows(2) {
        let [outer, inner] = pair else { continue };
        let Some(call) = ir.frames[inner.frame.index()].inlined else {
            return Err(invalid("an inner frame of a snapshot is not inlined"));
        };
        if call.parent != outer.frame {
            return Err(invalid(
                "a snapshot frame is not called by the frame before it",
            ));
        }
        if outer.last_interpreted_pc != Some(call.call_pc) {
            return Err(invalid(
                "an outer frame of a snapshot must be stopped at its call",
            ));
        }
    }
    let code_len = ir.frames[snapshot.frames[snapshot.frames.len() - 1].frame.index()].code_len;
    match snapshot.phase {
        SnapshotPhase::Before { pc } => {
            if pc >= code_len {
                return Err(invalid("pc is not an instruction boundary"));
            }
        }
        SnapshotPhase::After { pc, successor } => {
            if pc >= code_len || successor >= code_len {
                return Err(invalid("pc or successor is not an instruction boundary"));
            }
        }
    }
    Ok(())
}

/// Checks the static shape of one virtual table recipe: its allocation, its
/// keys and the kind of entry each value is. Whether the values are defined at
/// the snapshot is [`verify_snapshot_values`]'s concern.
fn verify_recipe(
    ir: &TraceIr,
    snapshot: &Snapshot,
    index: usize,
    table: &VirtualTable,
) -> Result<(), TraceVerifyError> {
    let invalid = |detail| TraceVerifyError::InvalidSnapshot {
        snapshot: snapshot.id.index(),
        detail,
    };
    if let VirtualAlloc::Shaped { keys } = &table.alloc {
        if keys.is_empty() || keys.len() > MAX_VIRTUAL_SHAPE_KEYS {
            return Err(invalid("a shaped virtual table needs one to eight keys"));
        }
        for (position, key) in keys.iter().enumerate() {
            if keys[..position].contains(key) {
                return Err(invalid("a virtual table shape repeats a key"));
            }
        }
    }
    for entry in table.entries() {
        match entry {
            RegisterEntry::Unchanged => {
                return Err(invalid("a virtual table cannot hold an unchanged register"));
            }
            RegisterEntry::Slice(_) => {
                return Err(invalid("a virtual table cannot hold a substring"));
            }
            // A table built inside a table: it names a recipe of the snapshot
            // other than its own.
            RegisterEntry::Virtual(id) => {
                if id.index() >= snapshot.virtuals.len() {
                    return Err(invalid("a virtual entry names no recipe"));
                }
                if id.index() == index {
                    return Err(invalid("a virtual table cannot hold itself"));
                }
            }
            RegisterEntry::Root(slot) if *slot >= ir.root_slot_count => {
                return Err(invalid("root slot is out of range"));
            }
            RegisterEntry::Const(scalar)
                if scalar.ty().is_opaque() || scalar.ty().is_borrowed() =>
            {
                return Err(invalid("an opaque value cannot be a snapshot constant"));
            }
            _ => {}
        }
    }
    if let Some(metatable) = &table.metatable {
        let RegisterEntry::Value(value) = metatable else {
            return Err(invalid("a virtual table's metatable is a value"));
        };
        if ir.value_type(*value) != Some(TraceType::Table) {
            return Err(invalid("a virtual table's metatable is a table"));
        }
    }
    let array_len = table.array.len() as u64;
    for (position, (key, _)) in table.fields.iter().enumerate() {
        match key {
            VirtualKey::Integer(index) => {
                if u64::try_from(*index).is_ok_and(|index| (1..=array_len).contains(&index)) {
                    return Err(invalid("a virtual field key lies in the array part"));
                }
            }
            VirtualKey::Float(number) => {
                if number.is_nan() {
                    return Err(invalid("a virtual field key is NaN"));
                }
                if number.fract() == 0.0
                    && (-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(number)
                {
                    return Err(invalid("an integral float key must be an integer key"));
                }
            }
            VirtualKey::String(_) => {}
        }
        let same = |other: &VirtualKey| match (key, other) {
            (VirtualKey::Float(a), VirtualKey::Float(b)) => a.to_bits() == b.to_bits(),
            (a, b) => a == b,
        };
        if table.fields[..position]
            .iter()
            .any(|(earlier, _)| same(earlier))
        {
            return Err(invalid("a virtual table repeats a field key"));
        }
    }
    Ok(())
}

/// `checked` is a snapshot of `ir` that passed this check with no more
/// values defined than now: a value it names in a run that `snapshot` shares
/// is defined and typed still, so those runs are not walked again.
fn verify_snapshot_values(
    ir: &TraceIr,
    defined: &Defined,
    snapshot: &Snapshot,
    checked: Option<&Snapshot>,
) -> Result<(), TraceVerifyError> {
    let result = snapshot_values(ir, defined, snapshot, checked);
    // Debug builds check every snapshot whole as well.
    debug_assert!(
        checked.is_none() || result.is_ok() == snapshot_values(ir, defined, snapshot, None).is_ok(),
        "skipping the runs snapshot {} shares changed its value check",
        snapshot.id.index()
    );
    result
}

fn snapshot_values(
    ir: &TraceIr,
    defined: &Defined,
    snapshot: &Snapshot,
    checked: Option<&Snapshot>,
) -> Result<(), TraceVerifyError> {
    let mut result = Ok(());
    snapshot.for_each_value_since(checked, |value| {
        if result.is_err() {
            return;
        }
        result = match defined.0.get(value.index()) {
            None => Err(TraceVerifyError::UnknownValue(value)),
            Some(false) => Err(TraceVerifyError::SnapshotValueUndefined {
                snapshot: snapshot.id.index(),
                value,
            }),
            Some(true) if ir.values[value.index()].ty == TraceType::Any => {
                Err(TraceVerifyError::InvalidSnapshot {
                    snapshot: snapshot.id.index(),
                    detail: "an Any value cannot be rebuilt; map it through a root slot",
                })
            }
            Some(true) => Ok(()),
        };
    });
    result
}

/// Checks the lifetime contract of borrowed values: a borrow defined before
/// an operation that ends borrows is not used after it.
///
/// A use is an operand, or a mention in the snapshot of a guard (the guard
/// leaves at its own position), of the poll or of the terminator (both after
/// the last instruction). The ending operation itself may still use a borrow
/// and leave through its own `Before` snapshot: nothing it did is visible
/// then.
fn verify_borrows(ir: &TraceIr) -> Result<(), TraceVerifyError> {
    let flat: Vec<&TraceInstruction> = ir.preheader.iter().chain(&ir.body).collect();
    let end = flat.len();
    // The flat position of each borrow's definition and its last use.
    let mut defined_at: Vec<Option<usize>> = vec![None; ir.values.len()];
    let mut last_use: Vec<usize> = vec![0; ir.values.len()];
    let mut any = false;
    // A side's borrowed entry input is defined before every instruction.
    let mut at_entry = vec![false; ir.values.len()];
    for input in &ir.entry.inputs {
        if input.ty.is_borrowed() {
            at_entry[input.value.index()] = true;
            any = true;
        }
    }
    for (position, instruction) in flat.iter().enumerate() {
        if let Some(result) = instruction.result
            && ir.values[result.index()].ty.is_borrowed()
        {
            defined_at[result.index()] = Some(position);
            any = true;
        }
    }
    if !any {
        return Ok(());
    }
    let mut note = |value: ValueId, position: usize| {
        let slot = &mut last_use[value.index()];
        *slot = (*slot).max(position);
    };
    let snapshot_values =
        |snapshot: &Snapshot, position: usize, note: &mut dyn FnMut(ValueId, usize)| {
            snapshot.for_each_value(|value| note(value, position));
        };
    for (position, instruction) in flat.iter().enumerate() {
        instruction.op.operands(|value| note(value, position));
        if instruction.op.guard().is_some()
            && let Some(guard) = ir
                .guards
                .get(instruction.op.guard().map_or(0, super::ir::GuardId::index))
            && let Some(snapshot) = ir.snapshots.get(guard.snapshot.index())
        {
            snapshot_values(snapshot, position, &mut note);
        }
    }
    let mut at_end = |snapshot: Option<&Snapshot>| {
        if let Some(snapshot) = snapshot {
            snapshot_values(snapshot, end, &mut note);
        }
    };
    at_end(ir.poll_snapshot.and_then(|id| ir.snapshots.get(id.index())));
    match &ir.terminator {
        TraceTerminator::LoopBack { args } => {
            for arg in args {
                note(*arg, end);
            }
        }
        TraceTerminator::Return { value } => {
            if let Some(value) = value {
                note(*value, end);
            }
        }
        TraceTerminator::TailCall { args, .. } => {
            for arg in args {
                note(*arg, end);
            }
        }
        TraceTerminator::Exit { snapshot }
        | TraceTerminator::Link { snapshot }
        | TraceTerminator::LinkLoop { snapshot, .. } => {
            at_end(ir.snapshots.get(snapshot.index()));
        }
    }
    for (ender, instruction) in flat.iter().enumerate() {
        if !instruction.op.ends_borrows() {
            continue;
        }
        for (value, defined) in defined_at.iter().enumerate() {
            if (at_entry[value] || defined.is_some_and(|defined| defined < ender))
                && last_use[value] > ender
            {
                return Err(TraceVerifyError::BorrowOutlivesEffect {
                    value: ValueId(u32::try_from(value).expect("value count fits u32")),
                    ender,
                });
            }
        }
    }
    Ok(())
}
