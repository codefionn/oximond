//! Static counts of optimization opportunities in optimized trace IR.
//!
//! These analyses only count. Nothing here changes a trace, and nothing the
//! backend emits depends on them except diagnostic counters. Every answer is
//! conservative: an operation or entry kind this module does not know is
//! never an opportunity.
//!
//! - Range proofs ([`RangeFacts`]): which bounds-checked accesses index with
//!   a loop parameter stepped by an integral term, and which of those sit in
//!   a loop whose exit guard compares that parameter with a loop-invariant
//!   bound. Those are syntactic candidates for checking the range once, not
//!   proofs: a real hoist still has to show the comparison's direction
//!   matches the step's sign, the initial value is integral and in range,
//!   the step cannot overflow, and the bound is finite. Type checks stay.
//! - Recovery recipes ([`member_opportunities`]): which values only
//!   snapshots keep alive, and which of those an exit could recompute from a
//!   small pure expression over values that are live at that exit anyway.

use super::ir::{
    ArithmeticOp, TraceIr, TraceOp, TraceScalar, TraceTerminator, TraceType, ValueDef, ValueId,
};
use super::snapshot::RegisterEntry;

/// How a bounds-checked access names its index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrayKeyClass {
    /// A constant key.
    Constant,
    /// A loop parameter stepped by an integral term each iteration, plus or
    /// minus an integral term (see [`RangeFacts::of`]). `range_once` when the loop is counted: its exit compares
    /// the parameter with a loop-invariant bound, and nothing in the tree
    /// resizes a table.
    Induction {
        range_once: bool,
    },
    /// Like [`Self::Induction`] for a float key, but a step or offset is a
    /// loop-invariant float that is no constant, such as the step of a Lua
    /// 5.1 numeric `for` loop. Whether it is integral is only known at
    /// entry, so a hoist would check it there (as it must check a float
    /// initial value in any case).
    FloatInvariantInduction {
        range_once: bool,
    },
    Other,
}

/// What a member's loop says about its loop parameters, under the strict
/// rule (float steps and offsets must be constants) and the relaxed one
/// (loop-invariant floats will do).
#[derive(Clone, Debug)]
pub struct RangeFacts {
    strict: LoopFacts,
    relaxed: LoopFacts,
}

#[derive(Clone, Debug)]
struct LoopFacts {
    /// Per value: the loop parameter it is, plus a constant offset, when it
    /// is one.
    affine: Vec<Option<usize>>,
    /// Per loop parameter: the induction it follows, itself or the one whose
    /// next value it shares; `None` if it is no induction.
    root: Vec<Option<usize>>,
    /// Per induction root: the loop exit compares it with an invariant bound.
    counted: Vec<bool>,
    resizes: bool,
}

fn constant_of(ir: &TraceIr, value: ValueId) -> Option<TraceScalar> {
    let info = ir.values.get(value.index())?;
    let instruction = match info.def {
        ValueDef::Instruction(index) => ir.body.get(index)?,
        ValueDef::Preheader(index) => ir.preheader.get(index)?,
        ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::LoopParam(_) => return None,
    };
    match instruction.op {
        TraceOp::Constant(scalar) => Some(scalar),
        _ => None,
    }
}

/// An integral, nonzero step or an integral offset: a float key stays
/// integral only if every term is.
fn integral(scalar: TraceScalar) -> Option<bool> {
    match scalar {
        TraceScalar::Integer(number) => Some(number != 0),
        TraceScalar::Float(number) if number.is_finite() && number.fract() == 0.0 => {
            Some(number != 0.0)
        }
        _ => None,
    }
}

fn instruction_op(ir: &TraceIr, value: ValueId) -> Option<&TraceOp> {
    match ir.values.get(value.index())?.def {
        ValueDef::Instruction(index) => ir.body.get(index).map(|instruction| &instruction.op),
        ValueDef::Preheader(index) => ir.preheader.get(index).map(|instruction| &instruction.op),
        ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::LoopParam(_) => None,
    }
}

fn loop_param_of(ir: &TraceIr, value: ValueId) -> Option<usize> {
    match ir.values.get(value.index())?.def {
        ValueDef::LoopParam(index) => Some(index),
        _ => None,
    }
}

/// Whether `term` may step or offset a value of type `ty` without leaving
/// the integers: an integral constant, or for integers any value the loop
/// never changes. A float key stays integral only if every term is, which
/// only a constant shows, unless `relaxed` accepts any invariant float.
fn integral_term(
    ir: &TraceIr,
    term: ValueId,
    ty: Option<TraceType>,
    args: &[ValueId],
    relaxed: bool,
) -> bool {
    if let Some(scalar) = constant_of(ir, term) {
        return integral(scalar).is_some();
    }
    (ty == Some(TraceType::Integer) || (relaxed && ty == Some(TraceType::Float)))
        && invariant(ir, term, args)
}

/// `value` as loop parameter `p` plus or minus an integral term, if it is
/// one.
fn affine_param(ir: &TraceIr, value: ValueId, args: &[ValueId], relaxed: bool) -> Option<usize> {
    if let Some(param) = loop_param_of(ir, value) {
        return Some(param);
    }
    let TraceOp::Arithmetic { op, left, right } = instruction_op(ir, value)? else {
        return None;
    };
    let ty = ir.value_type(value);
    match op {
        ArithmeticOp::Add => {
            if let Some(param) = loop_param_of(ir, *left)
                && integral_term(ir, *right, ty, args, relaxed)
            {
                return Some(param);
            }
            if let Some(param) = loop_param_of(ir, *right)
                && integral_term(ir, *left, ty, args, relaxed)
            {
                return Some(param);
            }
            None
        }
        ArithmeticOp::Subtract => {
            loop_param_of(ir, *left).filter(|_| integral_term(ir, *right, ty, args, relaxed))
        }
        _ => None,
    }
}

/// Whether `value` is the same in every iteration.
fn invariant(ir: &TraceIr, value: ValueId, args: &[ValueId]) -> bool {
    match ir.values.get(value.index()).map(|info| info.def) {
        Some(ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::Preheader(_)) => true,
        Some(ValueDef::Instruction(_)) => constant_of(ir, value).is_some(),
        // A parameter the loop-back hands itself is a pass-through.
        Some(ValueDef::LoopParam(index)) => args.get(index) == Some(&value),
        None => false,
    }
}

impl RangeFacts {
    /// The facts of `ir`, one member of a tree. `tree_resizes` says whether
    /// any member of the tree holds an [`TraceOp::ArraySet`].
    #[must_use]
    pub fn of(ir: &TraceIr, tree_resizes: bool) -> Self {
        Self {
            strict: LoopFacts::of(ir, tree_resizes, false),
            relaxed: LoopFacts::of(ir, tree_resizes, true),
        }
    }

    /// How `key`, a bounds-checked access's index, is formed.
    #[must_use]
    pub fn class(&self, ir: &TraceIr, key: ValueId) -> ArrayKeyClass {
        match self.strict.class(ir, key) {
            ArrayKeyClass::Other => match self.relaxed.class(ir, key) {
                ArrayKeyClass::Induction { range_once } => {
                    ArrayKeyClass::FloatInvariantInduction { range_once }
                }
                other => other,
            },
            class => class,
        }
    }
}

impl LoopFacts {
    /// The facts of `ir`, one member of a tree. `tree_resizes` says whether
    /// any member of the tree holds an [`TraceOp::ArraySet`], which may move
    /// array storage in the middle of a run.
    ///
    /// A parameter is an induction when its next value is itself plus or
    /// minus an integral term ([`integral_term`]). A parameter whose next
    /// value is an induction's next value is one too, one iteration behind
    /// after the first: a numeric `for` loop carries its control and its
    /// visible variable that way.
    fn of(ir: &TraceIr, tree_resizes: bool, relaxed: bool) -> Self {
        let params = ir.loop_params.len();
        let args: &[ValueId] = match &ir.terminator {
            TraceTerminator::LoopBack { args } => args,
            _ => &[],
        };
        let mut root: Vec<Option<usize>> = vec![None; params];
        for (param, next) in args.iter().enumerate().take(params) {
            let Some(TraceOp::Arithmetic { op, left, right }) = instruction_op(ir, *next) else {
                continue;
            };
            let me = ir.loop_params[param].value;
            let ty = ir.value_type(*next);
            let stepped = match op {
                ArithmeticOp::Add if *left == me => integral_term(ir, *right, ty, args, relaxed),
                ArithmeticOp::Add if *right == me => integral_term(ir, *left, ty, args, relaxed),
                ArithmeticOp::Subtract if *left == me => {
                    integral_term(ir, *right, ty, args, relaxed)
                }
                _ => false,
            };
            if stepped {
                root[param] = Some(param);
            }
        }
        for param in 0..params.min(args.len()) {
            if root[param].is_none() {
                root[param] = (0..params.min(args.len()))
                    .find(|other| root[*other] == Some(*other) && args[*other] == args[param]);
            }
        }
        // An exit: a branch guard on a comparison of an induction (or its
        // next value) with an invariant bound.
        let mut counted = vec![false; params];
        for instruction in &ir.body {
            let TraceOp::GuardBranch { condition, .. } = instruction.op else {
                continue;
            };
            let Some(TraceOp::Compare { left, right, .. }) = instruction_op(ir, condition) else {
                continue;
            };
            for (subject, bound) in [(*left, *right), (*right, *left)] {
                if !invariant(ir, bound, args) {
                    continue;
                }
                for param in 0..params {
                    let named = loop_param_of(ir, subject) == Some(param)
                        || args.get(param) == Some(&subject);
                    if named && let Some(root) = root[param] {
                        counted[root] = true;
                    }
                }
            }
        }
        let affine = (0..ir.values.len())
            .map(|index| {
                affine_param(
                    ir,
                    ValueId(u32::try_from(index).expect("value ids fit u32")),
                    args,
                    relaxed,
                )
            })
            .collect();
        Self {
            affine,
            root,
            counted,
            resizes: tree_resizes,
        }
    }

    fn class(&self, ir: &TraceIr, key: ValueId) -> ArrayKeyClass {
        if constant_of(ir, key).is_some() {
            return ArrayKeyClass::Constant;
        }
        match self.affine.get(key.index()).copied().flatten() {
            Some(param) => match self.root.get(param).copied().flatten() {
                Some(root) => ArrayKeyClass::Induction {
                    range_once: !self.resizes && self.counted[root],
                },
                None => ArrayKeyClass::Other,
            },
            _ => ArrayKeyClass::Other,
        }
    }
}

/// The key of a bounds-checked access.
#[must_use]
pub const fn bounds_checked_key(op: &TraceOp) -> Option<ValueId> {
    match op {
        TraceOp::ArrayLoad { key, .. }
        | TraceOp::LaneArrayLoad { key, .. }
        | TraceOp::NumberLoad { key, .. }
        | TraceOp::ArrayStore { key, .. }
        | TraceOp::LaneArrayStore { key, .. }
        | TraceOp::ArraySet { key, .. }
        | TraceOp::StringByte { key, .. } => Some(*key),
        _ => None,
    }
}

/// Static opportunity counts of one compiled member.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemberOpportunities {
    pub array_sites: u64,
    /// Of `array_sites`, [`TraceOp::StringByte`] reads of string bytes.
    pub byte_sites: u64,
    pub const_key_sites: u64,
    pub float_key_sites: u64,
    pub induction_sites: u64,
    pub range_once_sites: u64,
    /// Float sites that are inductions only under the relaxed rule.
    pub float_invariant_induction_sites: u64,
    pub float_invariant_range_once_sites: u64,
    pub type_guards: u64,
    pub guards: u64,
    pub snapshots: u64,
    pub snapshot_frames: u64,
    pub entries: u64,
    pub value_entries: u64,
    pub const_entries: u64,
    pub unchanged_entries: u64,
    pub root_entries: u64,
    /// Frame registers that name a virtual table.
    pub virtual_entries: u64,
    /// Entries of a kind this analysis does not know, or that may not
    /// appear where they do (an unchanged or virtual entry in a recipe).
    pub other_entries: u64,
    /// Distinct recipes the frames of each snapshot reach, summed over
    /// snapshots.
    pub snapshot_virtuals: u64,
    /// Operands of the reached recipes, by kind.
    pub recipe_value_entries: u64,
    pub recipe_const_entries: u64,
    pub recipe_root_entries: u64,
    /// Of `recipe_value_entries`, those in a snapshot whose values feed a
    /// side trace or a loop header, and those in a recovery-only snapshot.
    pub recipe_values_feeding: u64,
    pub recipe_values_recovery: u64,
    /// Snapshots whose recipe graph cannot be walked: an id out of range,
    /// or an entry a recipe may not hold. Every value such a snapshot names
    /// is neither snapshot-only nor cheap.
    pub snapshot_recipes_unknown: u64,
    pub snapshot_values: u64,
    pub snapshot_only_values: u64,
    pub snapshot_only_instructions: u64,
    pub snapshot_only_cheap: u64,
    pub pinned_closures: u64,
}

/// How deep a recomputed expression may nest.
const RECIPE_DEPTH: u32 = 3;

/// Whether `op` is pure, cannot fail and costs about one instruction.
/// Heap reads are never cheap: storage may change before the exit.
fn cheap_pure(op: &TraceOp, ir: &TraceIr, result: ValueId) -> bool {
    match op {
        TraceOp::Constant(_)
        | TraceOp::Move { .. }
        | TraceOp::Negate { .. }
        | TraceOp::IntToFloat { .. }
        | TraceOp::Numeric { .. }
        | TraceOp::Not { .. }
        | TraceOp::Truthy { .. }
        | TraceOp::Compare { .. }
        | TraceOp::Select { .. } => true,
        TraceOp::Arithmetic { op, .. } => match op {
            ArithmeticOp::Add | ArithmeticOp::Subtract | ArithmeticOp::Multiply => true,
            // Float division cannot trap; there is no integer division.
            ArithmeticOp::Divide => ir.value_type(result) == Some(TraceType::Float),
            ArithmeticOp::BitAnd | ArithmeticOp::BitOr | ArithmeticOp::BitXor => true,
            _ => false,
        },
        _ => false,
    }
}

/// Static opportunity counts of `ir`, one member of a compiled tree.
///
/// `side_guards` lists the guards (by index into [`TraceIr::guards`]) whose
/// failing edge continues into a side trace: their snapshot values feed the
/// side, so they are real uses, not recovery. `tree_resizes` is as for
/// [`RangeFacts::of`].
#[must_use]
#[allow(
    unreachable_patterns,
    reason = "entry kinds added later count as other"
)]
pub fn member_opportunities(
    ir: &TraceIr,
    side_guards: &[usize],
    tree_resizes: bool,
) -> MemberOpportunities {
    let mut counts = MemberOpportunities {
        guards: ir.guards.len() as u64,
        snapshots: ir.snapshots.len() as u64,
        pinned_closures: ir.pinned_closures.len() as u64,
        ..MemberOpportunities::default()
    };
    let facts = RangeFacts::of(ir, tree_resizes);
    for instruction in &ir.body {
        if matches!(instruction.op, TraceOp::GuardType { .. }) {
            counts.type_guards += 1;
        }
        let Some(key) = bounds_checked_key(&instruction.op) else {
            continue;
        };
        counts.array_sites += 1;
        if matches!(instruction.op, TraceOp::StringByte { .. }) {
            counts.byte_sites += 1;
        }
        match facts.class(ir, key) {
            ArrayKeyClass::Constant => counts.const_key_sites += 1,
            ArrayKeyClass::Induction { range_once } => {
                counts.induction_sites += 1;
                counts.range_once_sites += u64::from(range_once);
            }
            ArrayKeyClass::FloatInvariantInduction { range_once } => {
                counts.float_invariant_induction_sites += 1;
                counts.float_invariant_range_once_sites += u64::from(range_once);
            }
            ArrayKeyClass::Other => {}
        }
        if ir.value_type(key) == Some(TraceType::Float)
            && facts.class(ir, key) != ArrayKeyClass::Constant
        {
            counts.float_key_sites += 1;
        }
    }

    // Positions: preheader instructions first, then the body, then the
    // terminator. A value's last real use is the last position that reads it
    // other than through a recovery snapshot.
    let preheader = ir.preheader.len();
    let end = preheader + ir.body.len();
    let instructions = ir.preheader.iter().chain(&ir.body);
    let values = ir.values.len();
    let mut last_use: Vec<Option<usize>> = vec![None; values];
    let used = |value: ValueId, position: usize, last_use: &mut Vec<Option<usize>>| {
        if let Some(slot) = last_use.get_mut(value.index()) {
            *slot = Some(slot.map_or(position, |last| last.max(position)));
        }
    };
    // Snapshot id -> positions of the guards (or terminator) that leave
    // through it for recovery.
    let mut recovery_points: Vec<Vec<usize>> = vec![Vec::new(); ir.snapshots.len()];
    let mut feeding: Vec<bool> = vec![false; ir.snapshots.len()];
    for (position, instruction) in instructions.enumerate() {
        instruction
            .op
            .operands(|operand| used(operand, position, &mut last_use));
        if let Some(guard) = instruction.op.guard()
            && let Some(info) = ir.guards.get(guard.index())
        {
            let snapshot = info.snapshot.index();
            if side_guards.contains(&guard.index()) {
                if let Some(flag) = feeding.get_mut(snapshot) {
                    *flag = true;
                }
            } else if let Some(points) = recovery_points.get_mut(snapshot) {
                points.push(position);
            }
        }
    }
    match &ir.terminator {
        TraceTerminator::LoopBack { args } => {
            for arg in args {
                used(*arg, end, &mut last_use);
            }
        }
        TraceTerminator::Return { value } => {
            if let Some(value) = value {
                used(*value, end, &mut last_use);
            }
        }
        TraceTerminator::TailCall { args, .. } => {
            for arg in args {
                used(*arg, end, &mut last_use);
            }
        }
        // A link hands the snapshot's values to the loop header: real uses.
        TraceTerminator::Link { snapshot } | TraceTerminator::LinkLoop { snapshot, .. } => {
            if let Some(flag) = feeding.get_mut(snapshot.index()) {
                *flag = true;
            }
        }
        TraceTerminator::Exit { snapshot } => {
            if let Some(points) = recovery_points.get_mut(snapshot.index()) {
                points.push(end);
            }
        }
    }
    if let Some(snapshot) = ir.poll_snapshot
        && let Some(points) = recovery_points.get_mut(snapshot.index())
    {
        points.push(end);
    }
    // Loop parameters reach the next iteration through the loop-back, and
    // a side's link through its snapshot, both handled above. Values a
    // feeding snapshot names are real uses at the end.
    let mut in_snapshot = vec![false; values];
    // Values named by a snapshot whose recipes cannot be walked.
    let mut poisoned = vec![false; values];
    let mut recovery_uses: Vec<Vec<usize>> = vec![Vec::new(); values];
    for (index, snapshot) in ir.snapshots.iter().enumerate() {
        counts.snapshot_frames += snapshot.frames.len() as u64;
        let mut unknown = false;
        let mut named: Vec<ValueId> = Vec::new();
        let mut visited = vec![false; snapshot.virtuals.len()];
        let mut pending: Vec<usize> = Vec::new();
        for frame in &snapshot.frames {
            for entry in &frame.registers {
                counts.entries += 1;
                match entry {
                    RegisterEntry::Value(value) => {
                        counts.value_entries += 1;
                        named.push(*value);
                    }
                    RegisterEntry::Const(_) => counts.const_entries += 1,
                    RegisterEntry::Unchanged => counts.unchanged_entries += 1,
                    RegisterEntry::Root(_) => counts.root_entries += 1,
                    RegisterEntry::Virtual(id) => {
                        counts.virtual_entries += 1;
                        match visited.get_mut(id.index()) {
                            Some(seen) => {
                                if !*seen {
                                    *seen = true;
                                    pending.push(id.index());
                                }
                            }
                            None => unknown = true,
                        }
                    }
                    RegisterEntry::Slice(id) => {
                        counts.other_entries += 1;
                        match snapshot.slices.get(id.index()) {
                            Some(recipe) => {
                                for entry in recipe.entries() {
                                    if let RegisterEntry::Value(value) = entry {
                                        named.push(*value);
                                    }
                                }
                            }
                            None => unknown = true,
                        }
                    }
                }
            }
        }
        // Each recipe is walked once, however many registers name it. A
        // recipe holds values, constants, root values and tables of its own.
        let mut recipe_values: Vec<ValueId> = Vec::new();
        while let Some(recipe) = pending.pop() {
            counts.snapshot_virtuals += 1;
            for entry in snapshot.virtuals[recipe].entries() {
                match entry {
                    RegisterEntry::Value(value) => {
                        counts.recipe_value_entries += 1;
                        if feeding[index] {
                            counts.recipe_values_feeding += 1;
                        } else {
                            counts.recipe_values_recovery += 1;
                        }
                        recipe_values.push(*value);
                    }
                    RegisterEntry::Const(_) => counts.recipe_const_entries += 1,
                    RegisterEntry::Root(_) => counts.recipe_root_entries += 1,
                    // A table inside a table: walked once like any recipe.
                    RegisterEntry::Virtual(id) => match visited.get_mut(id.index()) {
                        Some(seen) => {
                            if !*seen {
                                *seen = true;
                                pending.push(id.index());
                            }
                        }
                        None => unknown = true,
                    },
                    RegisterEntry::Unchanged | RegisterEntry::Slice(_) => {
                        counts.other_entries += 1;
                        unknown = true;
                    }
                }
            }
        }
        named.extend(recipe_values);
        for value in &named {
            let Some(seen) = in_snapshot.get_mut(value.index()) else {
                continue;
            };
            *seen = true;
            if feeding[index] {
                used(*value, end, &mut last_use);
            } else {
                recovery_uses[value.index()].extend(&recovery_points[index]);
            }
        }
        if unknown {
            counts.snapshot_recipes_unknown += 1;
            for value in &named {
                if let Some(flag) = poisoned.get_mut(value.index()) {
                    *flag = true;
                }
            }
        }
    }
    // Recovery-only values are those no real use reads.
    let snapshot_only: Vec<bool> = (0..values)
        .map(|index| in_snapshot[index] && last_use[index].is_none() && !poisoned[index])
        .collect();
    for index in 0..values {
        if in_snapshot[index] {
            counts.snapshot_values += 1;
        }
        if !snapshot_only[index] {
            continue;
        }
        counts.snapshot_only_values += 1;
        let value = ValueId(u32::try_from(index).expect("value ids fit u32"));
        if instruction_op(ir, value).is_some() {
            counts.snapshot_only_instructions += 1;
            if recomputable(ir, value, &recovery_uses[index], RECIPE_DEPTH, &last_use) {
                counts.snapshot_only_cheap += 1;
            }
        }
    }
    counts
}

/// Whether `value` is available at `position` without keeping anything
/// extra alive: a constant, or a value some real use reads at or after it.
fn live_at(ir: &TraceIr, last_use: &[Option<usize>], value: ValueId, position: usize) -> bool {
    constant_of(ir, value).is_some()
        || last_use
            .get(value.index())
            .copied()
            .flatten()
            .is_some_and(|last| last >= position)
}

/// Whether an exit at each of `points` could compute `value` from a cheap
/// pure expression, at most `depth` deep, over values live there anyway.
fn recomputable(
    ir: &TraceIr,
    value: ValueId,
    points: &[usize],
    depth: u32,
    last_use: &[Option<usize>],
) -> bool {
    let Some(op) = instruction_op(ir, value) else {
        return false;
    };
    if depth == 0 || !cheap_pure(op, ir, value) {
        return false;
    }
    let mut ok = true;
    op.operands(|operand| {
        ok &= points
            .iter()
            .all(|point| live_at(ir, last_use, operand, *point))
            || recomputable(ir, operand, points, depth - 1, last_use);
    });
    ok
}
#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::jit_ir::Comparison;
    use crate::jit_ir::trace::{
        BindingKind, FrameId, LoopParamInit, Snapshot, SnapshotFrame, SnapshotPhase, TableBinding,
        TraceBuilder, TraceFrameInfo, TraceId, TraceSite,
    };

    fn site(pc: usize) -> TraceSite {
        TraceSite {
            frame: FrameId(0),
            pc,
        }
    }

    fn emit(b: &mut TraceBuilder, pc: usize, op: TraceOp, ty: TraceType) -> ValueId {
        b.emit(site(pc), op, Some(ty)).unwrap()
    }

    fn snapshot(b: &mut TraceBuilder, pc: usize, registers: Vec<RegisterEntry>) -> Snapshot {
        Snapshot::new(
            SnapshotPhase::Before { pc },
            b.effect_position(),
            vec![SnapshotFrame {
                frame: FrameId(0),
                last_interpreted_pc: None,
                registers: registers.into(),
            }],
        )
    }

    fn add(left: ValueId, right: ValueId) -> TraceOp {
        TraceOp::Arithmetic {
            op: ArithmeticOp::Add,
            left,
            right,
        }
    }

    /// `for i = i, n do s = s + a[i] + a[i + 1] end` over integers, with the
    /// exit compare of `i + 1` against `n`, or against `s` when `bound_varies`.
    /// Register 3 holds `i * k`, which only the load guards' snapshots name.
    fn counted(bound_varies: bool, offset_key_multiplied: bool) -> TraceIr {
        use RegisterEntry::{Unchanged, Value};
        let mut b = TraceBuilder::new(
            TraceId(1),
            TraceFrameInfo {
                register_count: 5,
                code_len: 32,
                inlined: None,
                activation: None,
            },
        );
        for register in [0, 1, 3] {
            b.mark_written(FrameId(0), register);
        }
        let i_in = b.input(0, TraceType::Integer);
        let s_in = b.input(1, TraceType::Integer);
        let n = b.input(2, TraceType::Integer);
        let table = b.input(4, TraceType::Table);
        b.set_bindings(vec![TableBinding {
            register: 4,
            kind: BindingKind::Array { first: None },
        }]);
        let i = b.loop_param(TraceType::Integer, LoopParamInit::Input(i_in));
        let s = b.loop_param(TraceType::Integer, LoopParamInit::Input(s_in));
        let one = emit(
            &mut b,
            1,
            TraceOp::Constant(TraceScalar::Integer(1)),
            TraceType::Integer,
        );
        let three = emit(
            &mut b,
            1,
            TraceOp::Constant(TraceScalar::Integer(3)),
            TraceType::Integer,
        );
        let scaled = emit(
            &mut b,
            1,
            TraceOp::Arithmetic {
                op: ArithmeticOp::Multiply,
                left: i,
                right: three,
            },
            TraceType::Integer,
        );
        let s0 = snapshot(
            &mut b,
            2,
            vec![Value(i), Value(s), Unchanged, Value(scaled), Unchanged],
        );
        let s0 = b.snapshot(s0);
        let g0 = b.guard(site(2), s0);
        let x = emit(
            &mut b,
            2,
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key: i,
                expected: TraceType::Integer,
                past_end: false,
                guard: g0,
            },
            TraceType::Integer,
        );
        let second_key = if offset_key_multiplied {
            scaled
        } else {
            emit(&mut b, 3, add(i, one), TraceType::Integer)
        };
        let s1 = snapshot(
            &mut b,
            3,
            vec![Value(i), Value(s), Unchanged, Value(scaled), Unchanged],
        );
        let s1 = b.snapshot(s1);
        let g1 = b.guard(site(3), s1);
        let y = emit(
            &mut b,
            3,
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key: second_key,
                expected: TraceType::Integer,
                past_end: false,
                guard: g1,
            },
            TraceType::Integer,
        );
        let partial = emit(&mut b, 4, add(s, x), TraceType::Integer);
        let s_next = emit(&mut b, 4, add(partial, y), TraceType::Integer);
        let i_next = emit(&mut b, 5, add(i, one), TraceType::Integer);
        let bound = if bound_varies { s_next } else { n };
        let more = emit(
            &mut b,
            5,
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: i_next,
                right: bound,
            },
            TraceType::Boolean,
        );
        let s2 = snapshot(
            &mut b,
            5,
            vec![
                Value(i_next),
                Value(s_next),
                Unchanged,
                Value(scaled),
                Unchanged,
            ],
        );
        let s2 = b.snapshot(s2);
        let g2 = b.guard(site(5), s2);
        b.emit(
            site(5),
            TraceOp::GuardBranch {
                condition: more,
                expected: true,
                guard: g2,
            },
            None,
        );
        b.finish(TraceTerminator::LoopBack {
            args: vec![i_next, s_next],
        })
    }

    /// A numeric `for` loop as the recorder shapes it: the control and the
    /// visible variable are two parameters that both take `control + step`,
    /// with `step` and the limit entry inputs. The body reads `a[v]` and
    /// `a[v + 1]`.
    fn for_loop() -> TraceIr {
        use RegisterEntry::{Unchanged, Value};
        let mut b = TraceBuilder::new(
            TraceId(2),
            TraceFrameInfo {
                register_count: 6,
                code_len: 32,
                inlined: None,
                activation: None,
            },
        );
        for register in [0, 3, 4] {
            b.mark_written(FrameId(0), register);
        }
        let s_in = b.input(0, TraceType::Integer);
        let limit = b.input(1, TraceType::Integer);
        let step = b.input(2, TraceType::Integer);
        let control_in = b.input(3, TraceType::Integer);
        let visible_in = b.input(4, TraceType::Integer);
        let table = b.input(5, TraceType::Table);
        b.set_bindings(vec![TableBinding {
            register: 5,
            kind: BindingKind::Array { first: None },
        }]);
        let s = b.loop_param(TraceType::Integer, LoopParamInit::Input(s_in));
        let control = b.loop_param(TraceType::Integer, LoopParamInit::Input(control_in));
        let visible = b.loop_param(TraceType::Integer, LoopParamInit::Input(visible_in));
        let registers = |s| {
            vec![
                Value(s),
                Unchanged,
                Unchanged,
                Value(control),
                Value(visible),
                Unchanged,
            ]
        };
        let s0 = snapshot(&mut b, 2, registers(s));
        let s0 = b.snapshot(s0);
        let g0 = b.guard(site(2), s0);
        let x = emit(
            &mut b,
            2,
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key: visible,
                expected: TraceType::Integer,
                past_end: false,
                guard: g0,
            },
            TraceType::Integer,
        );
        let one = emit(
            &mut b,
            3,
            TraceOp::Constant(TraceScalar::Integer(1)),
            TraceType::Integer,
        );
        let key = emit(&mut b, 3, add(visible, one), TraceType::Integer);
        let s1 = snapshot(&mut b, 3, registers(s));
        let s1 = b.snapshot(s1);
        let g1 = b.guard(site(3), s1);
        let y = emit(
            &mut b,
            3,
            TraceOp::ArrayLoad {
                binding: 0,
                table,
                key,
                expected: TraceType::Integer,
                past_end: false,
                guard: g1,
            },
            TraceType::Integer,
        );
        let partial = emit(&mut b, 4, add(s, x), TraceType::Integer);
        let s_next = emit(&mut b, 4, add(partial, y), TraceType::Integer);
        let next = emit(&mut b, 5, add(control, step), TraceType::Integer);
        let more = emit(
            &mut b,
            5,
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: next,
                right: limit,
            },
            TraceType::Boolean,
        );
        let s2 = snapshot(&mut b, 5, registers(s_next));
        let s2 = b.snapshot(s2);
        let g2 = b.guard(site(5), s2);
        b.emit(
            site(5),
            TraceOp::GuardBranch {
                condition: more,
                expected: true,
                guard: g2,
            },
            None,
        );
        b.finish(TraceTerminator::LoopBack {
            args: vec![s_next, next, next],
        })
    }

    #[test]
    fn a_for_loops_visible_variable_follows_its_control() {
        let ir = for_loop();
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.array_sites, 2);
        assert_eq!(counts.induction_sites, 2);
        assert_eq!(counts.range_once_sites, 2);
    }

    /// [`for_loop`] with every number a float, as Lua 5.1 records it.
    fn float_for_loop() -> TraceIr {
        let mut ir = for_loop();
        for info in &mut ir.values {
            if info.ty == TraceType::Integer {
                info.ty = TraceType::Float;
            }
        }
        for input in &mut ir.entry.inputs {
            if input.ty == TraceType::Integer {
                input.ty = TraceType::Float;
            }
        }
        for param in &mut ir.loop_params {
            if param.ty == TraceType::Integer {
                param.ty = TraceType::Float;
            }
        }
        for instruction in &mut ir.body {
            if let TraceOp::Constant(TraceScalar::Integer(number)) = instruction.op {
                instruction.op = TraceOp::Constant(TraceScalar::Float(number as f64));
            }
            if let TraceOp::ArrayLoad { expected, .. } = &mut instruction.op {
                *expected = TraceType::Float;
            }
        }
        ir
    }

    #[test]
    fn a_float_loop_with_an_invariant_step_is_only_a_relaxed_induction() {
        let ir = float_for_loop();
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.float_key_sites, 2);
        assert_eq!(counts.induction_sites, 0);
        assert_eq!(counts.float_invariant_induction_sites, 2);
        assert_eq!(counts.float_invariant_range_once_sites, 2);
        // The integer loop never counts as relaxed.
        let counts = member_opportunities(&for_loop(), &[], false);
        assert_eq!(counts.float_invariant_induction_sites, 0);
    }

    #[test]
    fn a_counted_loop_can_check_its_induction_range_once() {
        let ir = counted(false, false);
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.array_sites, 2);
        assert_eq!(counts.const_key_sites, 0);
        assert_eq!(counts.float_key_sites, 0);
        assert_eq!(counts.induction_sites, 2);
        assert_eq!(counts.range_once_sites, 2);
    }

    #[test]
    fn a_resizing_tree_or_a_varying_bound_is_no_range_once() {
        let ir = counted(false, false);
        assert_eq!(member_opportunities(&ir, &[], true).range_once_sites, 0);
        assert_eq!(member_opportunities(&ir, &[], true).induction_sites, 2);
        let varying = counted(true, false);
        varying.verify().unwrap();
        let counts = member_opportunities(&varying, &[], false);
        assert_eq!(counts.induction_sites, 2);
        assert_eq!(counts.range_once_sites, 0);
    }

    #[test]
    fn a_scaled_key_is_not_an_induction() {
        let ir = counted(false, true);
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.array_sites, 2);
        assert_eq!(counts.induction_sites, 1);
        assert_eq!(counts.range_once_sites, 1);
    }

    #[test]
    fn a_snapshot_only_product_is_cheap_only_where_its_operand_is_live() {
        let mut ir = counted(false, false);
        let counts = member_opportunities(&ir, &[], false);
        // `i * 3` is named only by snapshots. The exit guard comes after
        // `i`'s last use, so there it would keep `i` alive: not cheap.
        assert_eq!(counts.snapshot_only_values, 1);
        assert_eq!(counts.snapshot_only_instructions, 1);
        assert_eq!(counts.snapshot_only_cheap, 0);
        assert_eq!(counts.snapshots, 3);
        assert_eq!(counts.entries, 15);
        assert_eq!(counts.unchanged_entries, 6);
        assert_eq!(counts.value_entries, 9);
        // Named only by the load guards, before `i + 1` reads `i`: cheap.
        ir.snapshots[2].frames[0]
            .registers
            .set(3, RegisterEntry::Const(TraceScalar::Integer(0)));
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.snapshot_only_cheap, 1);
        assert_eq!(counts.const_entries, 1);
    }

    #[test]
    fn a_guard_with_a_side_makes_its_snapshot_values_real_uses() {
        let ir = counted(false, false);
        let counts = member_opportunities(&ir, &[0], false);
        assert_eq!(counts.snapshot_only_values, 0);
    }

    #[test]
    fn a_loaded_value_is_never_recomputed() {
        assert!(!cheap_pure(
            &TraceOp::ArrayLoad {
                binding: 0,
                table: ValueId(0),
                key: ValueId(1),
                expected: TraceType::Integer,
                past_end: false,
                guard: crate::jit_ir::trace::GuardId(0),
            },
            &counted(false, false),
            ValueId(0),
        ));
    }
    use crate::jit_ir::trace::{VirtualAlloc, VirtualId, VirtualTable};

    fn plain_recipe(array: Vec<RegisterEntry>) -> VirtualTable {
        VirtualTable {
            alloc: VirtualAlloc::Plain,
            array,
            fields: Vec::new(),
            phantom_entries: None,
            metatable: None,
        }
    }

    /// Inputs `x` and `q` (floats), `v = x + x`, and two guards on `q <= q`.
    /// The first guard's snapshot names `v` as a plain value. The second
    /// guard's snapshot, the one a side trace continues from, names `v` only
    /// inside a recipe. Returns the trace; the side guard is index 1.
    fn recipe_only_in_a_side_snapshot() -> TraceIr {
        use RegisterEntry::{Const, Value, Virtual};
        let mut b = TraceBuilder::new(
            TraceId(7),
            TraceFrameInfo {
                register_count: 2,
                code_len: 16,
                inlined: None,
                activation: None,
            },
        );
        for register in 0..2 {
            b.mark_written(FrameId(0), register);
        }
        let x = b.input(0, TraceType::Float);
        let q = b.input(1, TraceType::Float);
        let v = emit(&mut b, 1, add(x, x), TraceType::Float);
        let cmp = emit(
            &mut b,
            2,
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: q,
                right: q,
            },
            TraceType::Boolean,
        );
        let nil = Const(TraceScalar::Nil);
        let early = snapshot(&mut b, 2, vec![Value(v), nil]);
        let early = b.snapshot(early);
        let first = b.guard(site(2), early);
        b.emit(
            site(2),
            TraceOp::GuardBranch {
                condition: cmp,
                expected: true,
                guard: first,
            },
            None,
        );
        let side = snapshot(&mut b, 3, vec![nil, Virtual(VirtualId(0))])
            .with_virtuals(vec![plain_recipe(vec![Value(v)])]);
        let side = b.snapshot(side);
        let second = b.guard(site(3), side);
        b.emit(
            site(3),
            TraceOp::GuardBranch {
                condition: cmp,
                expected: true,
                guard: second,
            },
            None,
        );
        let last = snapshot(&mut b, 4, vec![nil, nil]);
        let last = b.snapshot(last);
        b.finish(TraceTerminator::Exit { snapshot: last })
    }

    #[test]
    fn a_value_a_feeding_recipe_names_is_a_real_use() {
        // On 750f914 this counted snapshot_only_values == 1: the side
        // guard's recipe was never walked, so `v` looked recovery-only.
        let ir = recipe_only_in_a_side_snapshot();
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[1], false);
        assert_eq!(counts.snapshot_values, 1);
        assert_eq!(counts.snapshot_only_values, 0);
    }

    /// `w = p + p` is named by an early guard's snapshot, where `p` is still
    /// live (`r = p + q` reads it later), and by a recipe of the exit's
    /// snapshot, where `p` is dead.
    fn recipe_at_a_later_exit() -> TraceIr {
        use RegisterEntry::{Const, Value, Virtual};
        let mut b = TraceBuilder::new(
            TraceId(8),
            TraceFrameInfo {
                register_count: 2,
                code_len: 16,
                inlined: None,
                activation: None,
            },
        );
        for register in 0..2 {
            b.mark_written(FrameId(0), register);
        }
        let p = b.input(0, TraceType::Float);
        let q = b.input(1, TraceType::Float);
        let w = emit(&mut b, 1, add(p, p), TraceType::Float);
        let cmp = emit(
            &mut b,
            2,
            TraceOp::Compare {
                comparison: Comparison::LessEqual,
                left: q,
                right: q,
            },
            TraceType::Boolean,
        );
        let nil = Const(TraceScalar::Nil);
        let early = snapshot(&mut b, 2, vec![Value(w), nil]);
        let early = b.snapshot(early);
        let g = b.guard(site(2), early);
        b.emit(
            site(2),
            TraceOp::GuardBranch {
                condition: cmp,
                expected: true,
                guard: g,
            },
            None,
        );
        emit(&mut b, 3, add(p, q), TraceType::Float);
        let last = snapshot(&mut b, 4, vec![nil, Virtual(VirtualId(0))])
            .with_virtuals(vec![plain_recipe(vec![Value(w)])]);
        let last = b.snapshot(last);
        b.finish(TraceTerminator::Exit { snapshot: last })
    }

    #[test]
    fn a_recipe_operand_of_a_later_exit_adds_that_recovery_point() {
        // On 750f914 this counted snapshot_only_cheap == 1: only the early
        // guard was checked, where `p` is live. At the exit `p` is dead.
        let ir = recipe_at_a_later_exit();
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.snapshot_only_values, 1);
        assert_eq!(counts.snapshot_only_instructions, 1);
        assert_eq!(counts.snapshot_only_cheap, 0);
    }
    /// `v = x + x`, left through an exit whose snapshot has the given
    /// registers and recipes. Returns the trace and `v`.
    fn exit_with(
        make: impl FnOnce(ValueId) -> (Vec<RegisterEntry>, Vec<VirtualTable>),
    ) -> (TraceIr, ValueId) {
        let mut b = TraceBuilder::new(
            TraceId(9),
            TraceFrameInfo {
                register_count: 3,
                code_len: 16,
                inlined: None,
                activation: None,
            },
        );
        for register in 0..3 {
            b.mark_written(FrameId(0), register);
        }
        b.set_root_slot_count(5);
        let x = b.input(0, TraceType::Float);
        let v = emit(&mut b, 1, add(x, x), TraceType::Float);
        let (registers, virtuals) = make(v);
        let last = snapshot(&mut b, 2, registers).with_virtuals(virtuals);
        let last = b.snapshot(last);
        (b.finish(TraceTerminator::Exit { snapshot: last }), v)
    }

    #[test]
    fn two_registers_naming_one_virtual_walk_its_recipe_once() {
        use RegisterEntry::{Const, Root, Value, Virtual};
        let (ir, _) = exit_with(|v| {
            (
                vec![
                    Virtual(VirtualId(0)),
                    Virtual(VirtualId(0)),
                    Const(TraceScalar::Nil),
                ],
                vec![
                    plain_recipe(vec![Value(v), Const(TraceScalar::Integer(1)), Root(4)]),
                    // Named by nothing: not reached.
                    plain_recipe(vec![Value(v)]),
                ],
            )
        });
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.entries, 3);
        assert_eq!(counts.virtual_entries, 2);
        assert_eq!(counts.const_entries, 1);
        assert_eq!(counts.other_entries, 0);
        assert_eq!(counts.snapshot_virtuals, 1);
        assert_eq!(counts.recipe_value_entries, 1);
        assert_eq!(counts.recipe_const_entries, 1);
        assert_eq!(counts.recipe_root_entries, 1);
        assert_eq!(counts.recipe_values_recovery, 1);
        assert_eq!(counts.recipe_values_feeding, 0);
        assert_eq!(counts.snapshot_recipes_unknown, 0);
        assert_eq!(counts.snapshot_values, 1);
        assert_eq!(counts.snapshot_only_values, 1);
    }

    #[test]
    fn recipe_values_of_a_side_snapshot_count_as_feeding() {
        let ir = recipe_only_in_a_side_snapshot();
        let counts = member_opportunities(&ir, &[1], false);
        assert_eq!(counts.recipe_values_feeding, 1);
        assert_eq!(counts.recipe_values_recovery, 0);
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.recipe_values_feeding, 0);
        assert_eq!(counts.recipe_values_recovery, 1);
        assert_eq!(counts.virtual_entries, 1);
    }

    /// The verifier refuses a virtual entry inside a recipe, so nesting
    /// cannot be built from verified IR. Unverified IR with one is not
    /// walked: it is counted unknown and its values are not opportunities.
    #[test]
    fn a_recipe_naming_a_virtual_is_walked_and_one_naming_none_is_unknown() {
        use RegisterEntry::{Const, Value, Virtual};
        let (ir, _) = exit_with(|v| {
            (
                vec![
                    Virtual(VirtualId(0)),
                    Const(TraceScalar::Nil),
                    Const(TraceScalar::Nil),
                ],
                vec![
                    plain_recipe(vec![Value(v), Virtual(VirtualId(1))]),
                    plain_recipe(vec![]),
                ],
            )
        });
        ir.verify().unwrap();
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.snapshot_recipes_unknown, 0);
        assert_eq!(counts.snapshot_virtuals, 2);
        assert_eq!(counts.other_entries, 0);
        assert_eq!(counts.snapshot_values, 1);
        // The recipe of a table nothing names is no recipe at all.
        let (ir, _) = exit_with(|v| {
            (
                vec![
                    Virtual(VirtualId(0)),
                    Const(TraceScalar::Nil),
                    Const(TraceScalar::Nil),
                ],
                vec![plain_recipe(vec![Value(v), Virtual(VirtualId(4))])],
            )
        });
        assert!(ir.verify().is_err());
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.snapshot_recipes_unknown, 1);
        assert_eq!(counts.snapshot_values, 1);
        assert_eq!(counts.snapshot_only_values, 0);
        assert_eq!(counts.snapshot_only_cheap, 0);
    }

    #[test]
    fn a_virtual_id_out_of_range_is_unknown_and_poisons_the_frame_values() {
        use RegisterEntry::{Const, Value, Virtual};
        let (ir, _) = exit_with(|v| {
            (
                vec![Value(v), Virtual(VirtualId(5)), Const(TraceScalar::Nil)],
                vec![],
            )
        });
        assert!(ir.verify().is_err());
        let counts = member_opportunities(&ir, &[], false);
        assert_eq!(counts.snapshot_recipes_unknown, 1);
        assert_eq!(counts.virtual_entries, 1);
        assert_eq!(counts.snapshot_values, 1);
        assert_eq!(counts.snapshot_only_values, 0);
        assert_eq!(counts.snapshot_only_cheap, 0);
    }

    #[test]
    fn a_walkable_snapshot_reports_no_unknown_recipes() {
        let counts = member_opportunities(&recipe_at_a_later_exit(), &[], false);
        assert_eq!(counts.snapshot_recipes_unknown, 0);
        assert_eq!(counts.other_entries, 0);
        assert_eq!(counts.recipe_values_recovery, 1);
    }
}
