//! Range proofs for the array accesses of a counted loop.
//!
//! A loop trace that indexes an array with its induction variable checks the
//! key at every access: the key must be an integral number inside the array
//! part. When the loop is counted (its exit compares the induction variable
//! with a bound that does not change) the keys of a whole run of the loop
//! are known when the loop starts: they lie between the variable's value at
//! the header and the bound, shifted by the access's constant offset. This
//! module finds those accesses and says what to check, once, where the loop
//! starts, so that the backend can run a version of the loop without the
//! per-access checks (see `Lowerer::lower_versions` in the backend).
//!
//! Nothing here changes a trace. The plan is a claim about the IR that holds
//! only together with the checks it lists: the backend evaluates them each
//! time control reaches the loop from outside (the entry, and every link of
//! a side trace, which brings new loop values), and runs the loop it always
//! ran when one fails. A failing check therefore never leaves the trace.
//!
//! # What is proven
//!
//! An *induction* is a loop parameter `r` whose loop-back value is
//! `next = r + s` (or `r - s`), with `s` an integral constant or a value the
//! loop does not change, of the parameter's type. A *follower* is another
//! parameter whose loop-back value is the same `next`: a numeric `for` loop
//! carries its control and its visible variable that way. The loop is
//! *counted* when a branch guard of the body tests `next < b` or `next <= b`
//! (in either operand order, expecting either outcome) against an invariant
//! `b` of the same type, so that every iteration after the first starts with
//! `next` on the side of `b` the guard demands: below it for an upward
//! guard, above it for a downward one.
//!
//! An access is proven when its table is an array binding, its key is a
//! loop parameter `x` of a counted induction (the root or a follower), plus
//! or minus an integral constant `c` or a loop-invariant value (a *shift*,
//! checked like a step and grouped into a check of its own, where it moves
//! the bounds the start and the bound are held to), and the tree never
//! resizes an array
//! (no member holds an [`TraceOp::ArraySet`] or a helper that moves storage),
//! so the array's length is the entry's for the whole run.
//!
//! For an upward induction with step `s` the values `x` takes at the header
//! are its own start, then values in `[r0 + s, b]`; for a downward one its
//! start, then values in `[b, r0 + s]`. The backend checks, per binding and
//! induction, with `lo` and `hi` the least and greatest offset of the
//! proven accesses:
//!
//! - the step: `1 <= s <= 2^31` upward, `-2^31 <= s <= -1` downward, and for
//!   a float an integral value; a constant step is checked here,
//! - every involved parameter starts equal to the root, which makes them all
//!   equal at every header (the followers take the root's next value),
//! - the root's start: `1 - lo <= r0` and `r0 <= len - hi`, and for floats
//!   that it is integral,
//! - the bound: `b <= len - hi` upward, `b >= 1 - lo` downward.
//!
//! All comparisons are ordered, so a NaN fails them. With the start and the
//! bound inside the array, every value an iteration starts with lies inside
//! `[1 - lo, len - hi]`, far below 2^53, so a float induction stays exact
//! and integral (a float sum that leaves that range rounds to a value the
//! guard rejects, since rounding is monotonic), and an integer one cannot
//! wrap. Every key `x + c` is then an integral number in `[1, len]`. The
//! guard of the access keeps its other job, the slot's type.
//!
//! # Loops that test first, and stores that may resize
//!
//! [`RangePlan::with`] proves two more shapes for a backend that asks
//! ([`RangeOptions`]).
//!
//! A loop may test the value an iteration starts with instead of the next one
//! (`while i <= n do ... i = i + s end`): its guard compares the parameter
//! `r` with `b` before the body does anything with `r`
//! ([`RangeInduction::tested`]). The values `r` takes are the start and then
//! sums that may pass `b`, but an iteration whose `r` is past `b` leaves at
//! the guard, so every access after the guard sees `r` between the start and
//! `b`, and the checks above cover it. No access keyed on the induction may
//! stand before the guard. The sum past the bound is at most `b + 2^31`,
//! still exact.
//!
//! A raw set ([`TraceOp::ArraySet`]) may resize its array, which is why a
//! tree that holds one gets no plan from [`RangePlan::of`]. With
//! [`RangeOptions::sets`] a set of a number or boolean through a header
//! binding is an access like the others when its key is proven: the key is
//! inside the array, where the backend overwrites a slot of the value's type
//! or fills a hole of a table with no hash part in place, and leaves through
//! the set's guard, before the store, for anything else. Such a set moves no
//! storage, so the array's length stays the entry's. A root with a set whose
//! key is not proven gets no plan.
//!
//! # Loops that end where the array does
//!
//! `for _, v in ipairs(t)` and `while t[i] do i = i + 1 end` have no bound:
//! the load that finds no element ends them. With [`RangeOptions::open`] an
//! induction with a constant step of at least one and no bounding guard is
//! *open* ([`RangeInduction::bound`] is `None`), and proves the loads keyed
//! on it halfway. The start is checked as for a counted loop, so every key is
//! an integral number of at least one; whether it is at most the array's
//! length is still compared at each load, as integers, and a key past the
//! end leaves through the load's guard as it does in the guarded loop. Every
//! iteration that reaches the loop-back passed those compares, so the values
//! stay as far below 2^53 as a counted loop's. Stores keyed on an open
//! induction are not proven (a set then costs the root its plan), and no
//! link of a side goes straight to the fast loop of a plan with an open
//! induction: it is checked again at the outer header.

use super::ir::{
    ArithmeticOp, BindingKind, TraceIr, TraceOp, TraceScalar, TraceTerminator, TraceType, ValueDef,
    ValueId,
};
use super::snapshot::{RegisterEntry, Snapshot, SnapshotFrame};
use crate::jit_ir::Comparison;

/// The largest step or offset magnitude a proof accepts. Small enough that
/// no sum the loop computes from values inside an array can leave the exact
/// range of a float or wrap an integer.
pub const MAX_RANGE_TERM: i64 = 1 << 31;

/// The most IR instructions a tree may hold for its root's loop to be
/// lowered in two versions (see the backends' versioned lowering): each version
/// lowers the whole tree, so compile time and code size double.
pub const MAX_VERSIONED_INSTRUCTIONS: usize = 2000;

/// IR instructions of a tree that one proven bounds check per iteration
/// repays lowering twice (twice for a float key, whose check costs about
/// twice as much). Lowering is most of a compile, and a tree whose loop
/// runs few iterations per entry (an inner loop of five, say) spends more
/// compiling the second version than the version saves.
pub const VERSIONED_INSTRUCTIONS_PER_CHECK: usize = 32;

/// Whether the per-iteration checks `plan` removes pay for lowering a tree
/// of `tree_size` IR instructions twice: one per distinct proven key, which
/// is what the slow version checks once the read and the write of a slot
/// share their check.
#[must_use]
pub fn versioning_pays(plan: &RangePlan, tree_size: usize) -> bool {
    let mut keys: Vec<ValueId> = plan.keys.iter().map(|key| key.key).collect();
    keys.sort_unstable();
    keys.dedup();
    let weight: usize = keys
        .iter()
        .map(|key| {
            let float = plan
                .keys
                .iter()
                .find(|proven| proven.key == *key)
                .is_some_and(|proven| plan.inductions[proven.induction].float);
            if float { 2 } else { 1 }
        })
        .sum();
    weight * VERSIONED_INSTRUCTIONS_PER_CHECK >= tree_size
}

/// `1.5 * 2^52`. Adding it to a float of magnitude at most 2^51 rounds the
/// float to an integer.
pub const FLOAT_ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

/// 2^53: float integers below it in magnitude convert to `i64` exactly, and
/// it is the largest quotient whose truncation is exact.
pub const FLOAT_EXACT_LIMIT: f64 = 9_007_199_254_740_992.0;

/// The step of an induction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RangeStep {
    /// A constant step, already checked: integral, nonzero, of the right
    /// sign and at most [`MAX_RANGE_TERM`] in magnitude. Negative for a
    /// downward induction.
    Constant(i64),
    /// A loop-invariant value, of the induction's type, that the backend
    /// checks. With `negated` the loop subtracts it, so the step is its
    /// negation.
    Invariant { value: ValueId, negated: bool },
}

/// A counted induction of the loop.
#[derive(Clone, Debug, PartialEq)]
pub struct RangeInduction {
    /// The loop parameter stepped by the loop-back.
    pub root: usize,
    pub step: RangeStep,
    /// The invariant the exit guard compares the next value with. `None`
    /// for an open induction (see the module's comment), which no guard
    /// bounds: its step is a positive constant, it is `upward`, and each
    /// load keyed on it compares its key with the array's length.
    pub bound: Option<ValueId>,
    /// Whether the guard keeps the next value at or below the bound (an
    /// upward loop) rather than at or above it.
    pub upward: bool,
    /// Whether the induction is a float, whose starts and step must also be
    /// integral.
    pub float: bool,
    /// For a loop that tests first: the body instruction of the guard, which
    /// compares the root parameter itself with the bound. `None` when the
    /// guard tests the next value.
    pub tested: Option<usize>,
    /// The loop parameters of this induction that proven keys are based on,
    /// and the root, in parameter order. The backend checks that each
    /// starts equal to the root and, for a float, carries one integer copy
    /// of the root that stands for all of them.
    pub params: Vec<usize>,
}

/// A loop-invariant value a key adds to its parameter (or subtracts, when
/// `negated`), such as the `o` of `t[i + o]`. The backend checks that it is
/// integral and at most [`MAX_RANGE_TERM`] in magnitude, and shifts the
/// array's bounds by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeShift {
    pub value: ValueId,
    pub negated: bool,
}

/// The check of one array against one induction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeCheck {
    /// The array binding.
    pub binding: u32,
    /// Index into [`RangePlan::inductions`].
    pub induction: usize,
    /// The least and the greatest constant key offset of the proven
    /// accesses (both zero for a shifted check).
    pub low: i64,
    pub high: i64,
    /// The invariant shift of the keys of this check, if they have one.
    pub shift: Option<RangeShift>,
}

/// One proven key: `param + offset` (or `param + shift`) of an access on
/// `binding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenKey {
    pub binding: u32,
    /// The access's key value.
    pub key: ValueId,
    /// The loop parameter the key is based on.
    pub param: usize,
    pub offset: i64,
    pub shift: Option<RangeShift>,
    /// Index into [`RangePlan::inductions`].
    pub induction: usize,
}

/// What the backend may skip in a version of the loop, and what it must
/// check before it runs that version.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RangePlan {
    pub inductions: Vec<RangeInduction>,
    pub checks: Vec<RangeCheck>,
    pub keys: Vec<ProvenKey>,
}

/// What [`RangePlan::with`] may prove beyond counted loops that test their
/// next value over arrays nothing resizes.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RangeOptions {
    /// Some operation of the tree may resize an array and no proof covers
    /// it: every side's sets and helpers, and what [`resizes_beside_sets`]
    /// finds in the root.
    pub resizes: bool,
    /// The backend stores a proven [`TraceOp::ArraySet`] of the root's body
    /// without resizing its array (see the module's comment).
    pub sets: bool,
    /// The backend takes inductions whose guard tests the value the
    /// iteration starts with ([`RangeInduction::tested`]).
    pub tests_first: bool,
    /// The backend takes open inductions ([`RangeInduction::bound`] is
    /// `None`) and compares the key of each load keyed on one with the
    /// array's length.
    pub open: bool,
}

/// Whether `root` may resize an array other than through an
/// [`TraceOp::ArraySet`] of its body: a helper that moves storage, a set in
/// its preheader. A side may resize: the backend sends the link of a side
/// that does through the root's outer header, which reads the arrays again
/// and checks the plan against them.
#[must_use]
pub fn resizes_beside_sets(root: &TraceIr) -> bool {
    let helper = |instruction: &super::ir::TraceInstruction| matches!(&instruction.op, TraceOp::CallHelper { helper, .. } if helper.effects().moves_storage());
    root.preheader.iter().any(|instruction| {
        helper(instruction) || matches!(instruction.op, TraceOp::ArraySet { .. })
    }) || root.body.iter().any(helper)
}

impl RangePlan {
    /// Whether guard `guard` of `root` is the guard of an access whose key
    /// this plan proves. Such a guard fails in the guarded loop when the key
    /// is outside the array (a loop over `t[0]` to `t[n]` starts there), and
    /// the iterations after it may all be inside: a side it enters links to
    /// the root's outer header, which checks the plan again, and not to the
    /// guarded loop, which would run the rest of the loop with the checks.
    #[must_use]
    pub fn guards_proven_key(&self, root: &TraceIr, guard: usize) -> bool {
        root.body.iter().any(|instruction| match instruction.op {
            TraceOp::ArrayLoad {
                binding,
                key,
                guard: id,
                ..
            }
            | TraceOp::NumberLoad {
                binding,
                key,
                guard: id,
                ..
            }
            | TraceOp::ArrayStore {
                binding,
                key,
                guard: id,
                ..
            }
            | TraceOp::ArraySet {
                binding,
                key,
                guard: id,
                ..
            } => id.index() == guard && self.key(binding, key).is_some(),
            _ => false,
        })
    }

    /// The proof of the key `key` of an access on `binding`, if there is
    /// one.
    #[must_use]
    pub fn key(&self, binding: u32, key: ValueId) -> Option<&ProvenKey> {
        self.keys
            .iter()
            .find(|proven| proven.binding == binding && proven.key == key)
    }

    /// The range plan of `ir`, the root of a tree, or `None` when it proves
    /// no access. `tree_resizes` says whether any member of the tree may
    /// resize an array ([`TraceIr::sets_tables`] of any member).
    #[must_use]
    pub fn of(ir: &TraceIr, tree_resizes: bool) -> Option<Self> {
        Self::with(
            ir,
            RangeOptions {
                resizes: tree_resizes,
                ..RangeOptions::default()
            },
        )
    }

    /// [`Self::of`] with the shapes `options` adds.
    #[must_use]
    pub fn with(ir: &TraceIr, options: RangeOptions) -> Option<Self> {
        if options.resizes || ir.function.is_some() || ir.loop_tail.is_some() {
            return None;
        }
        let TraceTerminator::LoopBack { args } = &ir.terminator else {
            return None;
        };
        if args.len() != ir.loop_params.len() {
            return None;
        }
        let mut plan = Self::default();
        // Per loop parameter: the induction it belongs to.
        let mut family: Vec<Option<usize>> = vec![None; ir.loop_params.len()];
        for root in 0..ir.loop_params.len() {
            if family[root].is_some() {
                continue;
            }
            let Some(induction) = induction_of(ir, args, root, options) else {
                continue;
            };
            let index = plan.inductions.len();
            for (param, arg) in args.iter().enumerate() {
                if *arg == args[root] && ir.loop_params[param].ty == ir.loop_params[root].ty {
                    family[param] = Some(index);
                }
            }
            plan.inductions.push(induction);
        }
        if plan.inductions.is_empty() {
            return None;
        }
        // An induction that tests first proves nothing when an access keyed
        // on it stands before its guard, where its value may be past the
        // bound.
        for (index, induction) in plan.inductions.iter().enumerate() {
            let Some(guard) = induction.tested else {
                continue;
            };
            let early = ir.body[..guard].iter().any(|instruction| {
                let (TraceOp::ArrayLoad { key, .. }
                | TraceOp::NumberLoad { key, .. }
                | TraceOp::ArrayStore { key, .. }
                | TraceOp::ArrayOverwrite { key, .. }
                | TraceOp::ArraySet { key, .. }) = instruction.op
                else {
                    return false;
                };
                affine_key(ir, key, args).is_some_and(|(param, ..)| family[param] == Some(index))
            });
            if early {
                for member in &mut family {
                    if *member == Some(index) {
                        *member = None;
                    }
                }
            }
        }
        // Whether the induction of `param` is open: it proves loads only.
        let open = |plan: &Self, family: &[Option<usize>], param: usize| {
            family[param].is_some_and(|index| plan.inductions[index].bound.is_none())
        };
        for instruction in &ir.body {
            let (binding, key, set, load) = match instruction.op {
                TraceOp::ArrayLoad {
                    binding,
                    key,
                    past_end: false,
                    ..
                }
                | TraceOp::NumberLoad { binding, key, .. } => (binding, key, None, true),
                TraceOp::ArrayStore { binding, key, .. } => (binding, key, None, false),
                TraceOp::ArraySet {
                    binding,
                    key,
                    value,
                    ..
                } if options.sets => (binding, key, Some(value), false),
                _ => continue,
            };
            let array = ir
                .bindings
                .get(binding as usize)
                .is_some_and(|bound| matches!(bound.kind, BindingKind::Array { .. }));
            if let Some(value) = set {
                // A set no proof covers may resize: no plan. The backend
                // stores a number or a boolean in place through the header
                // of the table.
                let register = ir.bindings.get(binding as usize)?.register;
                let stored_in_place = array
                    && matches!(
                        ir.value_type(value),
                        Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
                    )
                    && ir.bindings.iter().any(|other| {
                        other.register == register
                            && matches!(other.kind, BindingKind::Header { .. })
                    });
                let proven = stored_in_place
                    && affine_key(ir, key, args).is_some_and(|(param, ..)| {
                        family[param].is_some() && !open(&plan, &family, param)
                    });
                if !proven {
                    return None;
                }
            }
            if !array || plan.key(binding, key).is_some() {
                continue;
            }
            let Some((param, offset, shift)) = affine_key(ir, key, args) else {
                continue;
            };
            if open(&plan, &family, param) && (!load || shift.is_some()) {
                continue;
            }
            let Some(induction) = family[param] else {
                continue;
            };
            plan.keys.push(ProvenKey {
                binding,
                key,
                param,
                offset,
                shift,
                induction,
            });
            let params = &mut plan.inductions[induction].params;
            if !params.contains(&param) {
                params.push(param);
                params.sort_unstable();
            }
            match plan.checks.iter_mut().find(|check| {
                check.binding == binding && check.induction == induction && check.shift == shift
            }) {
                Some(check) => {
                    check.low = check.low.min(offset);
                    check.high = check.high.max(offset);
                }
                None => plan.checks.push(RangeCheck {
                    binding,
                    induction,
                    low: offset,
                    high: offset,
                    shift,
                }),
            }
        }
        // An induction no proven key uses costs checks for nothing.
        let used: Vec<bool> = (0..plan.inductions.len())
            .map(|index| plan.checks.iter().any(|check| check.induction == index))
            .collect();
        if !used.iter().any(|used| *used) {
            return None;
        }
        let renumber: Vec<usize> = used
            .iter()
            .scan(0, |next, used| {
                let index = *next;
                *next += usize::from(*used);
                Some(index)
            })
            .collect();
        let mut index = 0;
        plan.inductions.retain(|_| {
            index += 1;
            used[index - 1]
        });
        for check in &mut plan.checks {
            check.induction = renumber[check.induction];
        }
        for key in &mut plan.keys {
            key.induction = renumber[key.induction];
        }
        Some(plan)
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

/// `value` as an integral constant of type `ty` of at most
/// [`MAX_RANGE_TERM`] in magnitude.
#[must_use]
pub fn small_integral(ir: &TraceIr, value: ValueId, ty: TraceType) -> Option<i64> {
    let TraceOp::Constant(scalar) = instruction_op(ir, value)? else {
        return None;
    };
    let number = match (*scalar, ty) {
        (TraceScalar::Integer(number), TraceType::Integer) => number,
        (TraceScalar::Float(number), TraceType::Float)
            if number.is_finite() && number.fract() == 0.0 && number.abs() <= 2.0_f64.powi(31) =>
        {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "checked integral and small"
            )]
            {
                number as i64
            }
        }
        _ => return None,
    };
    (number.abs() <= MAX_RANGE_TERM).then_some(number)
}

/// Whether `value` is the same at every header of the loop, wherever the
/// loop was entered from: an entry input, a preheader value, a body
/// constant or a parameter the loop-back passes on unchanged. Every such
/// value is defined where the backend checks the plan.
fn invariant(ir: &TraceIr, value: ValueId, args: &[ValueId]) -> bool {
    match ir.values.get(value.index()).map(|info| info.def) {
        Some(ValueDef::Input(_) | ValueDef::FrameInput(_) | ValueDef::Preheader(_)) => true,
        Some(ValueDef::Instruction(_)) => {
            matches!(instruction_op(ir, value), Some(TraceOp::Constant(_)))
        }
        Some(ValueDef::LoopParam(index)) => args.get(index) == Some(&value),
        None => false,
    }
}

/// The counted induction rooted at loop parameter `root`, if it is one.
fn induction_of(
    ir: &TraceIr,
    args: &[ValueId],
    root: usize,
    options: RangeOptions,
) -> Option<RangeInduction> {
    let param = &ir.loop_params[root];
    let float = match param.ty {
        TraceType::Integer => false,
        TraceType::Float => true,
        _ => return None,
    };
    let me = param.value;
    let next = args[root];
    if ir.value_type(next) != Some(param.ty) {
        return None;
    }
    let TraceOp::Arithmetic { op, left, right } = instruction_op(ir, next)? else {
        return None;
    };
    // `next` must be a body instruction: a value of the iteration.
    if !matches!(ir.values[next.index()].def, ValueDef::Instruction(_)) {
        return None;
    }
    let (term, negated) = match op {
        ArithmeticOp::Add if *left == me => (*right, false),
        ArithmeticOp::Add if *right == me => (*left, false),
        ArithmeticOp::Subtract if *left == me && *right != me => (*right, true),
        _ => return None,
    };
    if ir.value_type(term) != Some(param.ty) {
        return None;
    }
    let constant =
        small_integral(ir, term, param.ty).map(|step| if negated { -step } else { step });
    if constant == Some(0) {
        return None;
    }
    // A constant step that is no small integer never passes the backend's
    // check; anything else must be the same in every iteration.
    if constant.is_none()
        && (matches!(instruction_op(ir, term), Some(TraceOp::Constant(_)))
            || !invariant(ir, term, args))
    {
        return None;
    }
    let step = match constant {
        Some(step) => RangeStep::Constant(step),
        None => RangeStep::Invariant {
            value: term,
            negated,
        },
    };
    let fits = |bound: ValueId, upward: bool| {
        ir.value_type(bound) == Some(param.ty)
            && invariant(ir, bound, args)
            && constant.is_none_or(|step| (step > 0) == upward)
    };
    for (_, bound, upward) in bounding_guards_at(ir, next) {
        if fits(bound, upward) {
            return Some(RangeInduction {
                root,
                step,
                bound: Some(bound),
                upward,
                float,
                tested: None,
                params: vec![root],
            });
        }
    }
    if options.tests_first {
        // The guard on the parameter itself, which holds the value of every
        // iteration that goes on to the body.
        for (at, bound, upward) in bounding_guards_at(ir, me) {
            if fits(bound, upward) {
                return Some(RangeInduction {
                    root,
                    step,
                    bound: Some(bound),
                    upward,
                    float,
                    tested: Some(at),
                    params: vec![root],
                });
            }
        }
    }
    // No guard bounds it: an open induction, when it only climbs.
    (options.open && constant.is_some_and(|step| step > 0)).then(|| RangeInduction {
        root,
        step,
        bound: None,
        upward: true,
        float,
        tested: None,
        params: vec![root],
    })
}

/// The bounds the branch guards of `ir`'s body hold `value` to: each other
/// operand of an ordering compare with `value` that a guard tests, and
/// whether passing the guard keeps `value` at or below it (`true`) or at or
/// above it. With NaN an operand, a guard expecting `false` passes without
/// either; a caller that relies on the answer checks the bound with ordered
/// compares, which refuse NaN.
pub fn bounding_guards(ir: &TraceIr, value: ValueId) -> impl Iterator<Item = (ValueId, bool)> {
    bounding_guards_at(ir, value).map(|(_, bound, upward)| (bound, upward))
}

/// [`bounding_guards`] with the body instruction of each guard.
fn bounding_guards_at(
    ir: &TraceIr,
    value: ValueId,
) -> impl Iterator<Item = (usize, ValueId, bool)> {
    ir.body
        .iter()
        .enumerate()
        .filter_map(move |(at, instruction)| {
            let TraceOp::GuardBranch {
                condition,
                expected,
                ..
            } = instruction.op
            else {
                return None;
            };
            let Some(TraceOp::Compare {
                comparison: Comparison::LessThan | Comparison::LessEqual,
                left,
                right,
            }) = instruction_op(ir, condition)
            else {
                return None;
            };
            if *left == value && *right != value {
                Some((at, *right, expected))
            } else if *right == value && *left != value {
                Some((at, *left, !expected))
            } else {
                None
            }
        })
}

/// `key` as a loop parameter plus a small integral constant, or plus or
/// minus a loop-invariant value of its type.
fn affine_key(
    ir: &TraceIr,
    key: ValueId,
    args: &[ValueId],
) -> Option<(usize, i64, Option<RangeShift>)> {
    if let Some(param) = loop_param_of(ir, key) {
        return Some((param, 0, None));
    }
    let ty = ir.value_type(key)?;
    if !matches!(ty, TraceType::Integer | TraceType::Float) {
        return None;
    }
    let TraceOp::Arithmetic { op, left, right } = instruction_op(ir, key)? else {
        return None;
    };
    let param_of =
        |value: ValueId| loop_param_of(ir, value).filter(|param| ir.loop_params[*param].ty == ty);
    // A term: a small constant offset, or an invariant shift. A constant
    // that is no small integer is neither.
    let term = |value: ValueId, negated: bool| {
        if let Some(offset) = small_integral(ir, value, ty) {
            return Some((if negated { -offset } else { offset }, None));
        }
        (ir.value_type(value) == Some(ty)
            && !matches!(instruction_op(ir, value), Some(TraceOp::Constant(_)))
            && invariant(ir, value, args))
        .then_some((0, Some(RangeShift { value, negated })))
    };
    let (param, (offset, shift)) = match op {
        ArithmeticOp::Add => match (param_of(*left), param_of(*right)) {
            (Some(param), _) if term(*right, false).is_some() => (param, term(*right, false)?),
            (_, Some(param)) => (param, term(*left, false)?),
            _ => return None,
        },
        ArithmeticOp::Subtract => (param_of(*left)?, term(*right, true)?),
        _ => return None,
    };
    Some((param, offset, shift))
}

// ---- links of a side that rejoins a versioned root --------------------------
//
// A side of the root that rejoins its loop brings new values for the loop
// parameters. Where the link goes is the backends' common decision: the
// root's outer header, which checks the plan again, or straight to the header
// of one version of the loop when the checks cannot have changed their
// answer. The predicates below say so; each backend turns them into jumps.

/// The loop parameters the versions of a root lowered in two carry on their
/// headers: those the loop-back changes (`passed` are the others), and, of
/// the parameters it passes on unchanged, those a side's link changes when
/// neither the preheader nor the range plan reads them. A link that changes
/// no other may then go straight to its version's header instead of through
/// the outer one, which runs the preheader and checks the plan again: an
/// accumulator that only a side adds to would otherwise send every iteration
/// that takes the side through the checks. Ascending.
#[must_use]
#[allow(
    clippy::needless_pass_by_value,
    reason = "the iterator is cloned for each candidate parameter"
)]
pub fn carried_for_links<'a>(
    root: &TraceIr,
    sides: impl Iterator<Item = &'a TraceIr> + Clone,
    plan: &RangePlan,
    passed: &[usize],
) -> Vec<usize> {
    let mut read: Vec<ValueId> = Vec::new();
    for instruction in &root.preheader {
        instruction.op.operands(|value| read.push(value));
    }
    for induction in &plan.inductions {
        read.extend(induction.bound);
        if let RangeStep::Invariant { value, .. } = induction.step {
            read.push(value);
        }
    }
    read.extend(
        plan.checks
            .iter()
            .filter_map(|check| check.shift.map(|shift| shift.value)),
    );
    passed
        .iter()
        .copied()
        .filter(|index| {
            let param = &root.loop_params[*index];
            let Some(register) = param.register else {
                return false;
            };
            !read.contains(&param.value)
                && sides.clone().any(|side| {
                    let TraceTerminator::Link { snapshot } = &side.terminator else {
                        return false;
                    };
                    let input = side
                        .entry
                        .inputs
                        .iter()
                        .find(|input| input.register == register)
                        .map(|input| input.value);
                    !matches!(
                        (side.snapshots[snapshot.index()].frames[0]
                            .registers
                            .get(usize::from(register)), input),
                        (Some(RegisterEntry::Value(value)), Some(input)) if *value == input
                    )
                })
        })
        .collect()
}

/// For each entry input of `side`, the root value it starts as when the
/// root's guard with snapshot `snapshot` fails and enters `side`: the value
/// the snapshot names for the input's register, or the root's entry input of
/// the register when the snapshot leaves it unchanged. `None` for an input
/// the snapshot does not name a value for (a number of run-time subtype is
/// not the value of the input, which is one subtype of it).
pub fn side_origins(
    root: &TraceIr,
    side: &TraceIr,
    snapshot: &Snapshot,
    origins: &mut Vec<Option<ValueId>>,
) {
    origins.clear();
    origins.extend(side.entry.inputs.iter().map(|input| {
        match snapshot.frames[0]
            .registers
            .get(usize::from(input.register))
        {
            Some(RegisterEntry::Value(_))
                if snapshot.number_tag(usize::from(input.register)).is_some() =>
            {
                None
            }
            Some(RegisterEntry::Value(value)) => Some(*value),
            Some(RegisterEntry::Unchanged) => root
                .entry
                .inputs
                .iter()
                .find(|held| held.register == input.register)
                .map(|held| held.value),
            _ => None,
        }
    }));
}

/// Whether a link of `side`, a side of the root entered by a guard of the
/// root's fast version, may go straight back to the fast loop with no
/// checks: the side steps every induction as the root does and passes the
/// root's own exit guard, so its link brings the next values of the run the
/// checks already covered. That needs, for every induction of the plan, the
/// link to give the root and the parameters the plan reads one value `v`, `v`
/// to be the root's value at the guard stepped by the root's step, and a
/// guard of the side to hold `v` to the root's bound on the same side; and
/// every parameter the fast loop does not carry (`changing` lists those it
/// does) to come back unchanged. `origins` is [`side_origins`] of the side
/// for the guard it was entered by. When false, the link checks the plan
/// again at the root's header.
#[must_use]
pub fn fast_link(
    plan: &RangePlan,
    root: &TraceIr,
    side: &TraceIr,
    origins: &[Option<ValueId>],
    changing: &[usize],
    link: &SnapshotFrame,
) -> bool {
    let origin = |value: ValueId| match side.values.get(value.index())?.def {
        ValueDef::Input(index) => origins.get(index).copied().flatten(),
        _ => None,
    };
    let entry = |param: usize| {
        let register = root.loop_params[param].register?;
        match link.registers.get(usize::from(register))? {
            RegisterEntry::Value(value) => Some(*value),
            _ => None,
        }
    };
    let proven = || -> Option<()> {
        for (index, param) in root.loop_params.iter().enumerate() {
            if !changing.contains(&index) && origin(entry(index)?) != Some(param.value) {
                return None;
            }
        }
        let side_op = |value: ValueId| match side.values.get(value.index())?.def {
            ValueDef::Instruction(index) => side.body.get(index).map(|instruction| &instruction.op),
            _ => None,
        };
        let root_constant = |value: ValueId| {
            let instruction = match root.values.get(value.index())?.def {
                ValueDef::Instruction(index) => root.body.get(index)?,
                ValueDef::Preheader(index) => root.preheader.get(index)?,
                _ => return None,
            };
            match instruction.op {
                TraceOp::Constant(scalar) => Some(scalar),
                _ => None,
            }
        };
        for induction in &plan.inductions {
            // An open induction has no guard a side could pass to show that
            // its next value is one the checks covered.
            let bound = induction.bound?;
            let ty = root.loop_params[induction.root].ty;
            let next = entry(induction.root)?;
            if side.value_type(next) != Some(ty)
                || induction
                    .params
                    .iter()
                    .any(|param| entry(*param) != Some(next))
            {
                return None;
            }
            let TraceOp::Arithmetic { op, left, right } = side_op(next)? else {
                return None;
            };
            let me = root.loop_params[induction.root].value;
            let is_root = |value: ValueId| origin(value) == Some(me);
            let is_step = |value: ValueId, negated: bool| match induction.step {
                RangeStep::Constant(step) => {
                    small_integral(side, value, ty) == Some(if negated { -step } else { step })
                }
                RangeStep::Invariant {
                    value: step,
                    negated: wanted,
                } => wanted == negated && origin(value) == Some(step),
            };
            let stepped = match op {
                ArithmeticOp::Add => {
                    (is_root(*left) && is_step(*right, false))
                        || (is_root(*right) && is_step(*left, false))
                }
                ArithmeticOp::Subtract => is_root(*left) && is_step(*right, true),
                _ => false,
            };
            if !stepped {
                return None;
            }
            let bound_constant = root_constant(bound);
            let bounded = bounding_guards(side, next).any(|(held, upward)| {
                upward == induction.upward
                    && side.value_type(held) == Some(ty)
                    && (origin(held) == Some(bound)
                        || bound_constant.is_some_and(|scalar| {
                            side_op(held) == Some(&TraceOp::Constant(scalar))
                        }))
            });
            if !bounded {
                return None;
            }
        }
        Some(())
    };
    proven().is_some()
}

/// Whether a link of `side`, entered from the guarded (slow) version of the
/// root's loop, may go straight to that version's header, which carries the
/// loop parameters `changing`: every other parameter must come back as the
/// root had it. Such a link stays in the guarded version as its own
/// loop-back would: the checks only choose a version, and a side taken every
/// few iterations would otherwise pay for them each time.
#[must_use]
pub fn slow_link(root: &TraceIr, side: &TraceIr, changing: &[usize], link: &SnapshotFrame) -> bool {
    root.loop_params.iter().enumerate().all(|(index, param)| {
        if changing.contains(&index) {
            return true;
        }
        let Some(register) = param.register else {
            return false;
        };
        // The side's entry input of the register is the root's value at the
        // guard, which for a parameter the root passes on is the parameter.
        let input = side
            .entry
            .inputs
            .iter()
            .find(|input| input.register == register)
            .map(|input| input.value);
        matches!(
            (link.registers.get(usize::from(register)), input),
            (Some(RegisterEntry::Value(value)), Some(input)) if *value == input
        )
    })
}

#[cfg(test)]
#[path = "range_tests.rs"]
pub(crate) mod tests;
